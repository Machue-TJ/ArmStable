"""Ideal D435i-style RGB-D rendering and color-based recognition (no hardware).

Set MUJOCO_GL before importing mujoco on headless machines, e.g. EGL.
Depth is optical-axis Z in metres, not Euclidean range. Detection positions
refer to visible surfaces, not object centers or precomputed scene labels.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    with open(path or Path(__file__).with_name("vision_config.json"), encoding="utf-8") as stream:
        config = json.load(stream)
    if config["width"] <= 0 or config["height"] <= 0 or config["fps"] <= 0:
        raise ValueError("Camera dimensions and fps must be positive")
    if not 0 < config["min_depth_m"] < config["max_depth_m"]:
        raise ValueError("Require 0 < min_depth_m < max_depth_m")
    return config


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    aligned_depth_m: np.ndarray
    native_depth_m: np.ndarray
    rgb_intrinsics: np.ndarray
    depth_intrinsics: np.ndarray
    world_from_optical: np.ndarray
    sim_time: float


class D435iCamera:
    """Lazy, reusable renderer for the fixed cameras on link6.

    Native depth uses its own FOV. Aligned depth is a second depth render at
    the RGB pinhole, so RGB pixels must only be paired with aligned_depth_m.
    No stereo baseline, distortion, IR, IMU or sensor noise is synthesized.
    """

    def __init__(self, model, config=None):
        self.model = model
        self.config = config if config is not None else load_config()
        self.width = int(self.config["width"])
        self.height = int(self.config["height"])
        self.rgb_id = self._camera_id(self.config["rgb_camera"])
        self.depth_id = self._camera_id(self.config["depth_camera"])
        self.renderer = None
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[3] = 0  # Hide collision approximations.
        self.scene_option.sitegroup[:] = 0  # Reference sites are not objects.
        self.model.vis.global_.offwidth = max(self.width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(self.height, self.model.vis.global_.offheight)

    def _camera_id(self, name):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            raise ValueError(f"Camera {name!r} missing; load xml/agilex/scene.xml")
        return camera_id

    def intrinsics(self, camera_id):
        # XML uses centered fovy pinholes, with square pixels.
        f = 0.5 * self.height / np.tan(np.deg2rad(self.model.cam_fovy[camera_id]) / 2)
        return np.array([[f, 0, (self.width - 1) / 2],
                         [0, f, (self.height - 1) / 2], [0, 0, 1]])

    def _depth(self, data, camera_id):
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=camera_id, scene_option=self.scene_option)
        depth = self.renderer.render().copy()
        valid = np.isfinite(depth) & (depth >= self.config["min_depth_m"]) & (
            depth <= self.config["max_depth_m"])
        depth[~valid] = np.nan
        return depth

    def capture(self, data):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        # Ensure transforms correspond to current qpos, including after mj_step.
        mujoco.mj_forward(self.model, data)
        try:
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(data, camera=self.rgb_id, scene_option=self.scene_option)
            rgb = self.renderer.render().copy()
            aligned = self._depth(data, self.rgb_id)
            native = self._depth(data, self.depth_id)
        finally:
            self.renderer.disable_depth_rendering()
        transform = np.eye(4)
        # MuJoCo: X right, Y up, -Z forward. CV: X right, Y down, Z forward.
        transform[:3, :3] = data.cam_xmat[self.rgb_id].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = data.cam_xpos[self.rgb_id]
        return RGBDFrame(rgb, aligned, native, self.intrinsics(self.rgb_id),
                         self.intrinsics(self.depth_id), transform, float(data.time))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def detect_targets(frame, config=None):
    """Identify configured colors with HSV masks and connected components.

    Labels assume the configured demo props have unique colors. This is not
    a general semantic/shape detector. No MuJoCo body positions or geom IDs
    are used in recognition. Invalid depth keeps the 2-D detection but yields
    null 3-D coordinates.
    """
    config = config if config is not None else load_config()
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
                depth_m = float(frame.aligned_depth_m[pv, pu])
                k = frame.rgb_intrinsics
                point = np.array([(pu - k[0, 2]) * depth_m / k[0, 0],
                                  (pv - k[1, 2]) * depth_m / k[1, 1], depth_m])
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
        text = detection["label"] + (f" {depth:.3f} m" if depth is not None else " depth invalid")
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
