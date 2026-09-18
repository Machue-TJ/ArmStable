"""D435i RGB-D simulation and color-based recognition (no USB required).

Set MUJOCO_GL before importing mujoco on headless machines, e.g. EGL.
Depth is optical-axis Z in metres, not Euclidean range. Detection positions
include visible surfaces and fitted marker centers, never precomputed scene labels.
"""
from dataclasses import dataclass
from copy import deepcopy

import cv2
import mujoco
import numpy as np

from .qt_fonts import configure_qt_fonts

# OpenCV sets its Qt font directory at import time. Correct it before imshow
# creates QApplication; the conda piper wheel does not ship that directory.
configure_qt_fonts()

from config.settings import ROOT, load_settings_section


def load_vision_config(value=None):
    """Load D435i imaging/detection settings from the unified project JSON."""
    config = load_settings_section("vision", value)
    validate_vision_config(config)
    return config


def validate_vision_config(config):
    # This simulator implements these full-FOV USB3 profiles. Other profiles
    # require their own calibrated intrinsics/crop and are rejected explicitly.
    if "fovy_deg" in config:
        raise ValueError("Use episode.markers.marker_fovy for placement; D435i FOV comes from calibration")
    if (config["width"], config["height"]) not in ((1280, 720), (1920, 1080)):
        raise ValueError("Supported RGB profiles: 1280x720 or 1920x1080")
    if (config["depth_width"], config["depth_height"]) not in ((848, 480), (1280, 720)):
        raise ValueError("Supported depth profiles: 848x480 or 1280x720")
    if config["fps"] not in (6, 15, 30):
        raise ValueError("Paired color/depth FPS must be 6, 15 or 30")
    min_z = 0.168 if config["depth_width"] == 848 else 0.28
    if not min_z <= config["min_depth_m"] < config["max_depth_m"]:
        raise ValueError(f"Depth window must start at >= {min_z} m for this profile")
    for name in ("disparity_noise_std_px", "disparity_step_px", "stereo_tolerance_m"):
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    marker = config.get("marker_detection")
    if marker:
        for name in ("diameter_m", "min_area_px", "max_center_uncertainty_m"):
            if not np.isfinite(marker[name]) or marker[name] <= 0:
                raise ValueError(f"marker_detection.{name} must be finite and positive")
        if not 0 < marker["min_circularity"] <= 1 or marker["min_depth_points"] < 1:
            raise ValueError("Invalid marker shape/depth thresholds")
    tracking = config.get("marker_tracking")
    if tracking:
        if not 0 < tracking["smoothing_alpha"] <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        for name in ("max_age_s", "max_pixel_distance", "max_world_distance_m"):
            if not np.isfinite(tracking[name]) or tracking[name] <= 0:
                raise ValueError(f"marker_tracking.{name} must be finite and positive")


def camera_matrix(spec, width, height, fov):
    if spec is None:
        fx, fy = np.array([width, height]) / (2 * np.tan(np.deg2rad(fov) / 2))
        return np.array([[fx, 0, (width-1)/2], [0, fy, (height-1)/2], [0, 0, 1.]])
    if (spec["width"], spec["height"]) != (width, height):
        raise ValueError("Calibration resolution must exactly match requested stream")
    k = np.array([[spec["fx"], 0, spec["ppx"]], [0, spec["fy"], spec["ppy"]], [0, 0, 1.]])
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid calibrated intrinsics")
    return k


def distortion(spec):
    if spec is None:
        return np.zeros(5)
    coefficients = np.asarray(spec.get("coeffs", [0]*5), dtype=float)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError("Expected five finite distortion coefficients")
    if spec.get("model", "none").split(".")[-1] not in ("none", "brown_conrady", "modified_brown_conrady", "inverse_brown_conrady"):
        raise ValueError("Only none/Brown-Conrady distortion is supported")
    return coefficients


def deproject(depth, k):
    y, x = np.indices(depth.shape)
    return np.stack(((x-k[0, 2])*depth/k[0, 0], (y-k[1, 2])*depth/k[1, 1], depth), axis=-1)


def _distort(xy, coeffs, model):
    """Brown variants follow librealsense rs2_project_point_to_pixel."""
    x, y = xy[..., 0], xy[..., 1]
    r2 = x*x+y*y
    k1, k2, p1, p2, k3 = coeffs
    radial = 1+r2*(k1+r2*(k2+r2*k3))
    rx, ry = x*radial, y*radial
    if model in ("modified_brown_conrady", "inverse_brown_conrady"):
        x, y = rx, ry
    return np.stack((rx+2*p1*x*y+p2*(r2+2*x*x), ry+2*p2*x*y+p1*(r2+2*y*y)), axis=-1)


def pixel_rays(pixels, k, coeffs, model="brown_conrady"):
    target = (np.asarray(pixels, dtype=float).reshape(-1, 2)-k[:2, 2])/np.diag(k)[:2]
    result = target.copy()
    if model != "none" and np.any(coeffs):
        # Numerically invert the forward model (also supports modified Brown,
        # which the SDK's direct deprojection API does not accept).
        for _ in range(30):
            error = target-_distort(result, coeffs, model)
            result += error
            if np.max(np.abs(error)) < 1e-10:
                break
        if not np.isfinite(result).all() or np.max(np.abs(target-_distort(result, coeffs, model))) > 1e-6:
            raise ValueError("Distortion inversion did not converge for this calibration")
    return result


def project(points, k, coeffs=None, model="brown_conrady"):
    points = np.asarray(points).reshape(-1, 3).astype(float)
    xy = points[:, :2]/points[:, 2, None]
    if coeffs is not None and model != "none" and np.any(coeffs):
        xy = _distort(xy, coeffs, model)
    return xy*np.diag(k)[:2]+k[:2, 2]


def align_depth_to_color(depth, depth_k, color_k, color_from_depth, color_shape, coeffs=None, distortion_model="brown_conrady"):
    """SDK-style pixel footprint splatting, nearest source depth wins.

    Returns SDK-style source Z and color-axis Z separately. SDK alignment
    copies original depth values; only color Z is valid for color deprojection
    when the extrinsic rotation/translation changes the optical-axis distance.
    """
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[y, x]
    h, w = color_shape
    aligned = np.full(h*w, np.inf, dtype=np.float32)
    rgb_z = np.full(h*w, np.nan, dtype=np.float32)
    if not len(z):
        return np.full((h, w), np.nan, np.float32), rgb_z.reshape(h, w)
    r, t = color_from_depth[:3, :3], color_from_depth[:3, 3]
    corners = []
    for dx, dy in ((-.5, -.5), (.5, -.5), (-.5, .5), (.5, .5)):
        xyz = np.column_stack(((x+dx-depth_k[0, 2])*z/depth_k[0, 0],
                               (y+dy-depth_k[1, 2])*z/depth_k[1, 1], z)) @ r.T + t
        corners.append(project(xyz, color_k, coeffs, distortion_model))
    center = np.column_stack(((x-depth_k[0, 2])*z/depth_k[0, 0],
                              (y-depth_k[1, 2])*z/depth_k[1, 1], z)) @ r.T + t
    bounds = np.asarray(corners)
    lo = np.floor(bounds.min(axis=0)+.5).astype(int)
    hi = np.floor(bounds.max(axis=0)+.5).astype(int)
    valid = (center[:, 2] > 0) & (lo[:, 0] >= 0) & (lo[:, 1] >= 0) & (hi[:, 0] < w) & (hi[:, 1] < h)
    lo, hi, z, color_z = lo[valid], hi[valid], z[valid], center[valid, 2]
    if not len(z):
        return np.full((h, w), np.nan, np.float32), rgb_z.reshape(h, w)
    # Sort far-to-near, then select the closest candidate for duplicate pixels.
    # The footprint is normally 2-4 pixels wide for the supported profiles.
    widths = hi-lo+1
    if widths.max() > 32:
        raise ValueError("Calibration produces an excessive alignment footprint")
    for dy in range(widths[:, 1].max()):
        for dx in range(widths[:, 0].max()):
            mask = (dx < widths[:, 0]) & (dy < widths[:, 1])
            indices = (lo[mask, 1]+dy)*w + lo[mask, 0]+dx
            values, cz = z[mask], color_z[mask]
            order = np.argsort(values, kind="stable")
            indices, values, cz = indices[order], values[order], cz[order]
            unique, first = np.unique(indices, return_index=True)
            better = values[first] < aligned[unique]
            dest, src = unique[better], first[better]
            aligned[dest], rgb_z[dest] = values[src], cz[src]
    aligned[~np.isfinite(aligned)] = np.nan
    return aligned.reshape(h, w), rgb_z.reshape(h, w)


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    aligned_depth_m: np.ndarray
    native_depth_m: np.ndarray
    rgb_intrinsics: np.ndarray
    depth_intrinsics: np.ndarray
    world_from_optical: np.ndarray
    sim_time: float
    native_depth_z16: np.ndarray
    aligned_rgb_z_m: np.ndarray
    rgb_distortion: np.ndarray
    world_from_depth_optical: np.ndarray
    depth_scale_m: float
    calibration_source: str
    rgb_distortion_model: str = "none"


class D435iCamera:
    """Nominal/calibrated D435i geometry, stereo visibility and depth error model.

    Global-shutter depth snapshots; RGB is instantaneous (rolling shutter and
    exposure integration remain unmodeled). This is not an IR matching ASIC.
    capture() holds the previous frame until the configured video period passes.
    """
    def __init__(self, model, config=None):
        from config.d435i import load_calibration, apply_calibration
        self.model = model
        self.config = config if config is not None else load_vision_config()
        validate_vision_config(self.config)
        self.width, self.height = self.config["width"], self.config["height"]
        self.depth_width, self.depth_height = self.config["depth_width"], self.config["depth_height"]
        self.calibration = load_calibration(self.config.get("calibration_path"))
        apply_calibration(model, self.calibration)
        self.rgb_id = model.camera(self.config["rgb_camera"]).id
        self.depth_id = model.camera(self.config["depth_camera"]).id
        self.right_id = model.camera("d435i_ir_right").id
        self._intrinsics = {}
        for cid, key, size, fov in ((self.rgb_id, "color", (self.width, self.height), (69, 42)),
                                    (self.depth_id, "depth", (self.depth_width, self.depth_height), (87, 58)),
                                    (self.right_id, "right", (self.depth_width, self.depth_height), (87, 58))):
            spec = self.calibration.get(key+"_intrinsics")
            k = camera_matrix(spec, *size, fov)
            if key != "color" and np.any(distortion(spec)):
                raise ValueError("Depth/right profiles must be rectified with zero distortion")
            self._intrinsics[cid] = k
            w, h = size
            model.cam_sensorsize[cid] = [w, h]
            model.cam_resolution[cid] = [w, h]
            model.cam_intrinsic[cid] = [k[0, 0], k[1, 1], (w-1)/2-k[0, 2], k[1, 2]-(h-1)/2]
        color_spec = self.calibration.get("color_intrinsics")
        self.rgb_distortion = distortion(color_spec)
        self.rgb_distortion_model = (color_spec or {}).get("model", "none").split(".")[-1]
        self.renderer = self.depth_renderer = None
        self._distortion_map = None
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[3] = 0
        self.scene_option.geomgroup[4] = 0  # Virtual plane is a viewer overlay, never a depth target.
        self.scene_option.geomgroup[5] = 1  # Episode marker spheres.
        if self.config.get("hide_camera_housing"):
            # Optical origins are inside the simplified solid housing. Keep it
            # visible in the external viewer, but omit it from its own images.
            for name in ("d435i_housing", "d435i_front"):
                model.geom(name).group[:] = 1
            self.scene_option.geomgroup[1] = 0
        self.scene_option.sitegroup[:] = 0
        model.vis.global_.offwidth = max(self.width, self.depth_width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(self.height, self.depth_height, model.vis.global_.offheight)
        from .markers import MarkerTracker
        self.marker_tracker = (MarkerTracker(self.config["marker_tracking"])
                               if self.config.get("marker_detection") and self.config.get("marker_tracking") else None)
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(self.config["seed"] if seed is None else seed)
        self._frame = None
        self._detection_frame = self._detections = None
        self._last_request = None
        self._start = self._next_time = None
        if self.marker_tracker is not None:
            self.marker_tracker.reset()

    def detect(self, frame):
        """Detect current image markers and retain IDs between camera frames."""
        if frame is not self._detection_frame:
            detections = detect_targets(frame, self.config)
            self._detections = (self.marker_tracker.update(detections, frame)
                                if self.marker_tracker is not None else detections)
            self._detection_frame = frame
        return deepcopy(self._detections)

    def intrinsics(self, camera_id):
        return self._intrinsics[camera_id].copy()

    def _pose(self, data, cid):
        result = np.eye(4)
        result[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        result[:3, 3] = data.cam_xpos[cid]
        return result

    def _depth(self, data, camera_id):
        self.depth_renderer.enable_depth_rendering()
        self.depth_renderer.update_scene(data, camera=camera_id, scene_option=self.scene_option)
        return self.depth_renderer.render().copy()

    def capture(self, data):
        now = float(data.time)
        if self._last_request is not None and now < self._last_request:
            self.reset()
        self._last_request = now
        if self._frame is not None and now < self._next_time-1e-9:
            return self._frame
        if self._start is None:
            self._start = now
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            self.depth_renderer = mujoco.Renderer(self.model, height=self.depth_height, width=self.depth_width)
        mujoco.mj_forward(self.model, data)
        self.renderer.update_scene(data, camera=self.rgb_id, scene_option=self.scene_option)
        rgb = self.renderer.render().copy()
        if np.any(self.rgb_distortion):
            if self._distortion_map is None:
                y, x = np.indices((self.height, self.width), dtype=np.float32)
                pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
                k = self.intrinsics(self.rgb_id)
                rays = pixel_rays(pixels, k, self.rgb_distortion, self.rgb_distortion_model)
                self._distortion_map = (rays*np.diag(k)[:2]+k[:2, 2]).reshape(self.height, self.width, 2).astype(np.float32)
            rgb = cv2.remap(rgb, self._distortion_map[..., 0], self._distortion_map[..., 1], cv2.INTER_LINEAR)
        left, right = self._depth(data, self.depth_id), self._depth(data, self.right_id)
        world_depth = self._pose(data, self.depth_id)
        right_from_depth = np.linalg.inv(self._pose(data, self.right_id)) @ world_depth
        # Pixels outside the sensor's depth window cannot produce measurements.
        # Avoid deprojecting/projecting the entire background for stereo checks.
        in_range = (np.isfinite(left) & (left >= self.config["min_depth_m"])
                    & (left <= self.config["max_depth_m"]))
        ys, xs = np.nonzero(in_range)
        k = self._intrinsics[self.depth_id]
        depths = left[ys, xs]
        points = np.column_stack(((xs-k[0, 2])*depths/k[0, 0],
                                  (ys-k[1, 2])*depths/k[1, 1], depths))
        right_points = points @ right_from_depth[:3, :3].T + right_from_depth[:3, 3]
        uv = np.floor(project(right_points, self.intrinsics(self.right_id))+.5).astype(int)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < self.depth_width) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < self.depth_height) & (right_points[:, 2] > 0))
        visible = np.zeros(len(points), dtype=bool)
        visible[inside] = np.abs(right[uv[inside, 1], uv[inside, 0]] - right_points[inside, 2]) <= self.config["stereo_tolerance_m"]
        baseline = np.linalg.norm(right_from_depth[:3, 3])
        fb = self.intrinsics(self.depth_id)[0, 0] * baseline
        disparity = fb / depths[visible] + self.rng.normal(0, self.config["disparity_noise_std_px"], visible.sum())
        step = self.config["disparity_step_px"]
        if step:
            disparity = np.round(disparity/step)*step
        depth_scale = self.calibration["depth_scale_m"]
        depths = np.divide(fb, disparity, out=np.full_like(disparity, np.nan), where=disparity > 0)
        counts = np.round(depths/depth_scale)
        good = np.isfinite(counts) & (counts > 0) & (counts <= 65535) & (depths >= self.config["min_depth_m"]) & (depths <= self.config["max_depth_m"])
        z16 = np.zeros(left.shape, dtype=np.uint16)
        z16[ys[visible][good], xs[visible][good]] = counts[good].astype(np.uint16)
        native = z16.astype(np.float32)*depth_scale
        native[z16 == 0] = np.nan
        world_color = self._pose(data, self.rgb_id)
        aligned, rgb_z = align_depth_to_color(native, self.intrinsics(self.depth_id), self.intrinsics(self.rgb_id),
                                               np.linalg.inv(world_color) @ world_depth,
                                               (self.height, self.width), self.rgb_distortion, self.rgb_distortion_model)
        self._frame = RGBDFrame(rgb, aligned, native, self.intrinsics(self.rgb_id), self.intrinsics(self.depth_id),
                                world_color, now, z16, rgb_z, self.rgb_distortion.copy(), world_depth,
                                depth_scale, self.calibration["source"], self.rgb_distortion_model)
        self._next_time = self._start + (np.floor((now-self._start)*self.config["fps"]+1e-9)+1)/self.config["fps"]
        return self._frame

    def close(self):
        for renderer in (self.renderer, self.depth_renderer):
            if renderer is not None:
                renderer.close()
        self.renderer = self.depth_renderer = None
        self._frame = None
        self._detection_frame = self._detections = None
        self._distortion_map = None


def detect_targets(frame, config=None):
    """Identify configured colors with HSV masks and connected components.

    Labels assume the configured demo props have unique colors. This is not
    a general semantic/shape detector. No MuJoCo body positions or geom IDs
    are used in recognition. Invalid depth keeps the 2-D detection but yields
    null 3-D coordinates.
    """
    config = config if config is not None else load_vision_config()
    if config.get("marker_detection"):
        from .markers import detect_markers
        return detect_markers(frame, config)
    hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
    detections = []
    kernel = np.ones((3, 3), np.uint8)
    for target in config["targets"]:
        mask = cv2.inRange(hsv, np.array(target["hsv_lower"], np.uint8),
                           np.array(target["hsv_upper"], np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            x, y, w, h, area = map(int, stats[index])
            if area < config["min_area_px"]:
                continue
            u, v = centers[index]
            # Use an actual interior pixel near the centroid, never background.
            inner = cv2.erode((labels == index).astype(np.uint8), kernel).astype(bool)
            valid = inner & np.isfinite(frame.aligned_depth_m)
            valid &= (frame.aligned_depth_m >= config["min_depth_m"])
            valid &= (frame.aligned_depth_m <= config["max_depth_m"])
            ys, xs = np.nonzero(valid)
            point_camera = point_world = depth_m = depth_pixel = None
            if len(xs):
                nearest = np.argmin((xs - u) ** 2 + (ys - v) ** 2)
                pu, pv = int(xs[nearest]), int(ys[nearest])
                depth_m = float(frame.aligned_rgb_z_m[pv, pu])
                k = frame.rgb_intrinsics
                ray = pixel_rays([[pu, pv]], k, frame.rgb_distortion, frame.rgb_distortion_model)[0]
                point = np.r_[ray, 1] * depth_m
                point_camera = point.tolist()
                point_world = (frame.world_from_optical @ np.r_[point, 1])[:3].tolist()
                depth_pixel = [pu, pv]
            detections.append({"label": target["label"], "bbox_xywh": [x, y, w, h],
                               "center_uv": [float(u), float(v)], "area_px": area,
                               "depth_pixel_uv": depth_pixel, "depth_m": depth_m,
                               "surface_point_optical_m": point_camera,
                               "surface_point_world_m": point_world})
    return detections


def annotate(frame, detections):
    bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    for detection in detections:
        x, y, w, h = detection["bbox_xywh"]
        depth = detection["depth_m"]
        center_depth = detection.get("center_depth_m", depth)
        text = detection["label"] + (f" #{detection['track_id']}" if "track_id" in detection else "")
        text += (f" {center_depth:.3f} m" if center_depth is not None else " depth invalid")
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(bgr, text, (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 255, 255), 1, cv2.LINE_AA)
    return bgr


def depth_preview(depth, config):
    scaled = np.nan_to_num((depth - config["min_depth_m"]) /
                           (config["max_depth_m"] - config["min_depth_m"]), nan=0)
    image = cv2.applyColorMap((np.clip(scaled, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~np.isfinite(depth)] = 0
    return image
