"""Image-only reflective sphere detection, measured center estimation and tracking.

No MuJoCo model/data, CSV coordinates or marker IDs are available to this module.
The known physical sphere diameter is a calibration input, not ground truth pose.
"""
from copy import deepcopy

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def native_points_in_color(frame):
    """Return valid native-depth points in RGB optical axes and RGB pixel positions."""
    from .piper_vision import pixel_rays, project
    ys, xs = np.nonzero(np.isfinite(frame.native_depth_m) & (frame.native_depth_m > 0))
    z = frame.native_depth_m[ys, xs]
    rays = pixel_rays(np.column_stack((xs, ys)), frame.depth_intrinsics, np.zeros(5), "none")
    points = np.column_stack((rays * z[:, None], z))
    color_from_depth = np.linalg.inv(frame.world_from_optical) @ frame.world_from_depth_optical
    points = points @ color_from_depth[:3, :3].T + color_from_depth[:3, 3]
    points = points[points[:, 2] > 0]
    pixels = project(points, frame.rgb_intrinsics, frame.rgb_distortion, frame.rgb_distortion_model)
    return points, pixels


def estimate_sphere_center(points, center_ray, radius_m, min_points=4):
    """Fit center along an observed image ray using known radius and robust depth.

    Each front surface sample P gives t = P.u + sqrt(r²-|P-(P.u)u|²),
    where center C=t*u. Return (C, uncertainty_m, sample_count), or None.
    """
    if len(points) < min_points:
        return None
    unit = np.asarray(center_ray, float)
    unit = unit / np.linalg.norm(unit)
    parallel = points @ unit
    perpendicular2 = np.sum(points * points, axis=1) - parallel * parallel
    valid = (perpendicular2 >= -1e-10) & (perpendicular2 < (0.85 * radius_m)**2)
    estimates = parallel[valid] + np.sqrt(np.maximum(radius_m**2 - perpendicular2[valid], 0))
    if len(estimates) < min_points:
        return None
    median = np.median(estimates)
    sigma = 1.4826 * np.median(np.abs(estimates - median))
    estimates = estimates[np.abs(estimates - median) <= max(3 * sigma, 0.0015)]
    if len(estimates) < min_points:
        return None
    distance = float(np.median(estimates))
    # A stochastic uncertainty indicator, not an absolute accuracy guarantee.
    uncertainty = max(0.0005, float(1.4826 * np.median(np.abs(estimates - distance)))) / np.sqrt(len(estimates))
    return unit * distance, uncertainty, len(estimates)


def detect_markers(frame, config):
    """Return color/shape gated spheres with subpixel centers and native-depth fits."""
    from .piper_vision import pixel_rays
    settings = config["marker_detection"]
    hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
    lower, upper = settings["hsv_lower"], settings["hsv_upper"]
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    # Close small specular holes; fill remaining interior highlights per contour.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points, pixels = native_points_in_color(frame)
    radius = settings["diameter_m"] / 2
    detections = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < settings["min_area_px"] or not perimeter:
            continue
        circularity = float(4 * np.pi * area / perimeter**2)
        solidity = float(area / max(cv2.contourArea(cv2.convexHull(contour)), 1e-9))
        if circularity < settings["min_circularity"] or solidity < .8 or not .65 < w / h < 1.55:
            continue
        moments = cv2.moments(contour)
        center = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
        if len(contour) >= 5:
            ellipse_center, axes, _ = cv2.fitEllipse(contour)
            if min(axes) > 0 and max(axes) / min(axes) < 1.55:
                center = np.array(ellipse_center)
        roi = np.zeros((h, w), np.uint8)
        cv2.drawContours(roi, [contour - [x, y]], -1, 1, cv2.FILLED)
        interior = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        interior = interior >= max(1, .35 * interior.max())
        # Select each native pixel once, avoiding nearest-Z splat bias and
        # duplicated aligned depth pixels when estimating a tiny sphere center.
        inside = ((pixels[:, 0] >= x) & (pixels[:, 0] < x+w-.5) &
                  (pixels[:, 1] >= y) & (pixels[:, 1] < y+h-.5))
        uv = np.rint(pixels[inside] - [x, y]).astype(int)
        samples = points[inside][interior[uv[:, 1], uv[:, 0]]]
        samples = samples[(samples[:, 2] >= config["min_depth_m"]) & (samples[:, 2] <= config["max_depth_m"])]
        ray = np.r_[pixel_rays([center], frame.rgb_intrinsics, frame.rgb_distortion,
                               frame.rgb_distortion_model)[0], 1]
        fit = estimate_sphere_center(samples, ray, radius, settings["min_depth_points"])
        center_camera = center_world = surface_camera = surface_world = depth = center_depth = uncertainty = None
        count = 0
        status = "insufficient_depth"
        if fit is not None:
            estimated, uncertainty, count = fit
            expected_diameter = 2 * radius * np.sqrt(frame.rgb_intrinsics[0, 0] * frame.rgb_intrinsics[1, 1]) / estimated[2]
            observed_diameter = 2 * np.sqrt(area / np.pi)
            if not .5 <= observed_diameter / expected_diameter <= 1.6:
                continue
            if uncertainty <= settings["max_center_uncertainty_m"]:
                center_camera = estimated.tolist()
                center_world = (frame.world_from_optical[:3, :3] @ estimated + frame.world_from_optical[:3, 3]).tolist()
                surface = estimated - radius * ray / np.linalg.norm(ray)
                surface_camera = surface.tolist()
                surface_world = (frame.world_from_optical[:3, :3] @ surface + frame.world_from_optical[:3, 3]).tolist()
                depth, center_depth = float(surface[2]), float(estimated[2])
                status = "measured"
            else:
                status = "uncertain_depth"
        detections.append({"label": "marker", "bbox_xywh": [x, y, w, h],
                           "center_uv": center.tolist(), "area_px": int(np.count_nonzero(roi)),
                           "circularity": circularity, "depth_pixel_uv": None,
                           "depth_m": depth, "center_depth_m": center_depth,
                           "surface_point_optical_m": surface_camera, "surface_point_world_m": surface_world,
                           "center_position_optical_m": center_camera, "center_position_world_m": center_world,
                           "center_uncertainty_m": uncertainty, "depth_sample_count": count,
                           "diameter_m": 2 * radius, "measurement_status": status})
    return sorted(detections, key=lambda item: (item["center_uv"][0], item["center_uv"][1]))


class MarkerTracker:
    """Assign observed track IDs and smooth measured world centers; never invent detections."""

    def __init__(self, config):
        self.config = dict(config)
        self.reset()

    def reset(self):
        """Clear IDs/history/cache on an episode reset."""
        self.tracks = {}
        self.next_id = 0
        self.last_time = None
        self.cached = []

    def update(self, detections, frame):
        """Associate via projected positions and world distances; return current detections only."""
        from .piper_vision import project
        now = frame.sim_time
        if self.last_time is not None and now < self.last_time:
            self.reset()
        if self.last_time == now:
            return deepcopy(self.cached)
        self.tracks = {key: value for key, value in self.tracks.items()
                       if now - value["time"] <= self.config["max_age_s"]}
        result = deepcopy(detections)
        keys = list(self.tracks)
        costs = np.full((len(keys), len(result)), 1e6)
        for i, key in enumerate(keys):
            track = self.tracks[key]
            expected_uv = np.asarray(track["uv"])
            if track["world"] is not None:
                local = frame.world_from_optical[:3, :3].T @ (track["world"] - frame.world_from_optical[:3, 3])
                if local[2] <= 0:
                    continue
                expected_uv = project([local], frame.rgb_intrinsics, frame.rgb_distortion, frame.rgb_distortion_model)[0]
            for j, item in enumerate(result):
                pixel_distance = np.linalg.norm(expected_uv - item["center_uv"])
                if pixel_distance > self.config["max_pixel_distance"]:
                    continue
                distance = 0
                if track["world"] is not None and item["center_position_world_m"] is not None:
                    distance = np.linalg.norm(track["world"] - item["center_position_world_m"])
                    if distance > self.config["max_world_distance_m"]:
                        continue
                costs[i, j] = pixel_distance / self.config["max_pixel_distance"] + distance / self.config["max_world_distance_m"]
        assignment = {}
        if costs.size:
            rows, columns = linear_sum_assignment(costs)
            assignment = {j: keys[i] for i, j in zip(rows, columns) if costs[i, j] < 1e6}
        for j, item in enumerate(result):
            key = assignment.get(j)
            if key is None:
                key, self.next_id = self.next_id, self.next_id + 1
            previous = self.tracks.get(key)
            world = item["center_position_world_m"]
            filtered = None
            if world is not None:
                filtered = np.asarray(world, float)
                if previous is not None and previous["world"] is not None:
                    alpha = self.config["smoothing_alpha"]
                    filtered = alpha * filtered + (1-alpha) * previous["world"]
            item["track_id"] = key
            item["filtered_center_world_m"] = None if filtered is None else filtered.tolist()
            self.tracks[key] = {"time": now, "uv": item["center_uv"],
                                "world": filtered if filtered is not None else (previous["world"] if previous else None)}
        self.last_time, self.cached = now, deepcopy(result)
        return result
