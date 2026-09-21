"""Integration checks using actual EGL-rendered RGB/depth, no USB camera."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")
import sys
from pathlib import Path
import unittest
import json
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from config.vision.piper_vision import D435iCamera, ROOT, detect_targets, align_depth_to_color, load_vision_config, pixel_rays, project
from config.d435i import nominal_calibration, apply_calibration, transform
from config.episode import EpisodeInitializer, make_model, load_episode_config
from config.flobase.piper_base import FloatingBase


class VisionIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode_config = load_episode_config({"markers": {
            "count": 2, "plane_depth_m": 0.7, "generator": None}})
        cls.model = make_model(cls.episode_config)
        cls.data = mujoco.MjData(cls.model)
        # Retain legacy D435i profile/calibration coverage, with visible fixture
        # markers generated for the current CSV-compatible camera bracket.
        cls.vision_config = load_vision_config()
        cls.vision_config.update(hide_camera_housing=True, min_area_px=8, marker_detection=None, marker_tracking=None,
                                 targets=[dict(label="marker", hsv_lower=[140, 80, 50], hsv_upper=[170, 255, 255])])
        cls.camera = D435iCamera(cls.model, cls.vision_config)
        cls.base = FloatingBase(cls.model, cls.data)
        cls.initializer = EpisodeInitializer(cls.model, cls.data, cls.base, cls.episode_config)

    @classmethod
    def tearDownClass(cls):
        cls.camera.close()

    def setUp(self):
        self.initializer.reset(np.random.default_rng(7), sample_id=3)
        self.camera.reset()

    def test_model_dimensions_and_mount(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (15, 14, 7))
        mount = self.model.body("d435i_mount")
        self.assertEqual(mount.parentid[0], self.model.body("link6").id)
        before = self.camera.capture(self.data).world_from_optical.copy()
        self.data.joint("joint1").qpos[0] += 0.2
        self.camera.reset()  # Direct state teleport starts a new capture sequence.
        after = self.camera.capture(self.data).world_from_optical
        self.assertGreater(np.linalg.norm(before[:3, 3] - after[:3, 3]), 0.01)
        wrist = self.data.body("link6")
        mount_local = wrist.xmat.reshape(3, 3).T @ (
            self.data.body("d435i_mount").xpos - wrist.xpos)
        np.testing.assert_allclose(mount_local, [-0.047, 0, 0.065], atol=1e-8)

    def assert_surface_detections(self, frame):
        detections = detect_targets(frame, self.vision_config)
        self.assertEqual([d["label"] for d in detections], ["marker", "marker"])
        for item in detections:
            self.assertIsNotNone(item["depth_m"])
            p = np.array(item["surface_point_world_m"])
            distances = np.linalg.norm(self.initializer.marker_positions_world_m - p, axis=1)
            self.assertAlmostEqual(distances.min(), 0.0075, delta=0.004)

    def test_rgb_depth_and_world_surface_positions(self):
        frame = self.camera.capture(self.data)
        self.assertEqual(frame.rgb.shape, (720, 1280, 3))
        self.assertEqual(frame.aligned_depth_m.shape, (720, 1280))
        self.assertEqual(frame.native_depth_m.shape, (480, 848))
        self.assertGreater(frame.rgb_intrinsics[0, 0], frame.depth_intrinsics[0, 0])
        self.assert_surface_detections(frame)

    def test_calibrated_geometry_fov_quantization_and_invalid_band(self):
        frame = self.camera.capture(self.data)
        for k, size, expected in ((frame.rgb_intrinsics, (1280, 720), (69, 42)),
                                   (frame.depth_intrinsics, (848, 480), (87, 58))):
            fov = np.rad2deg(2*np.arctan(np.array(size)/(2*np.diag(k)[:2])))
            np.testing.assert_allclose(fov, expected)
        c = nominal_calibration()
        rgb_from_depth = np.linalg.inv(frame.world_from_optical) @ frame.world_from_depth_optical
        np.testing.assert_allclose(rgb_from_depth, transform(c["color_from_depth"]), atol=1e-9)
        self.assertEqual(frame.native_depth_z16.dtype, np.uint16)
        valid = frame.native_depth_z16 > 0
        np.testing.assert_allclose(frame.native_depth_m[valid], frame.native_depth_z16[valid]*frame.depth_scale_m,
                                   rtol=2e-7)
        self.assertTrue(np.isnan(frame.native_depth_m[~valid]).all())
        # The left edge lacks stereo overlap, not perfect rendered depth.
        self.assertGreater(np.isnan(frame.native_depth_m[:, :20]).mean(), 0.8)
        imu = self.data.site("d435i_imu_site")
        offset = frame.world_from_depth_optical[:3, :3].T @ (imu.xpos-frame.world_from_depth_optical[:3, 3])
        np.testing.assert_allclose(offset, [0.00552, -0.0051, -0.01174], atol=1e-9)
        self.assertEqual(self.model.body_gravcomp[self.model.body("d435i_mount").id], 0)

    def test_video_cadence_and_no_drift(self):
        first = self.camera.capture(self.data)
        self.data.time = 0.002
        self.assertIs(self.camera.capture(self.data), first)
        self.data.time = 0.04
        second = self.camera.capture(self.data)
        self.assertIsNot(second, first)
        self.data.time = 0.08
        self.assertIsNot(self.camera.capture(self.data), second)

    def test_alignment_uses_source_depth_and_preserves_holes(self):
        depth = np.full((8, 8), np.nan, np.float32)
        depth[3, 3] = 1
        k = np.array([[4., 0, 3.5], [0, 4, 3.5], [0, 0, 1]])
        extrinsics = np.eye(4)
        extrinsics[2, 3] = .1
        aligned, color_z = align_depth_to_color(depth, k, k, extrinsics, (8, 8))
        valid = np.isfinite(aligned)
        self.assertGreater(valid.sum(), 0)
        self.assertLess(valid.sum(), 8)
        np.testing.assert_allclose(aligned[valid], 1)
        np.testing.assert_allclose(color_z[valid], 1.1)

    def test_unsupported_profile_rejected(self):
        config = load_vision_config()
        config["width"], config["height"] = 640, 360
        with self.assertRaises(ValueError):
            D435iCamera(self.model, config)

    def test_device_intrinsics_and_distortion_rendering(self):
        model = make_model(self.episode_config)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        calibration = nominal_calibration()
        calibration["source"] = "synthetic calibration test, not a device measurement"
        # Independent fx/fy, an off-center principal point and nonzero lens distortion.
        calibration["color_intrinsics"] = dict(width=1280, height=720, fx=960, fy=930,
                                                ppx=667, ppy=342, model="modified_brown_conrady",
                                                coeffs=[0.04, -0.01, 0.002, -0.001, 0.001])
        config = dict(self.vision_config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"calibration.json"
            path.write_text(json.dumps(calibration))
            config["calibration_path"] = str(path)
            camera = D435iCamera(model, config)
            try:
                initializer = EpisodeInitializer(model, data, FloatingBase(model, data), self.episode_config)
                initializer.reset(np.random.default_rng(7), sample_id=3)
                frame = camera.capture(data)
                self.assertEqual(frame.rgb_intrinsics[0, 2], 667)
                self.assert_surface_detections(frame)
            finally:
                camera.close()

    def test_brown_variants_projection_inverse(self):
        k = np.array([[900., 0, 639.5], [0, 930, 359.5], [0, 0, 1]])
        points = np.array([[.2, -.1, 1], [-.4, .2, 2], [.3, .15, .8]])
        coeffs = np.array([.04, -.01, .002, -.001, .001])
        for model in ("brown_conrady", "modified_brown_conrady", "inverse_brown_conrady"):
            uv = project(points, k, coeffs, model)
            np.testing.assert_allclose(pixel_rays(uv, k, coeffs, model), points[:, :2]/points[:, 2, None], atol=1e-9)

    def test_detection_after_one_second_of_physics(self):
        for _ in range(round(1 / self.model.opt.timestep)):
            self.base.step()
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assert_surface_detections(self.camera.capture(self.data))

    def test_hidden_targets_do_not_detect_floor(self):
        ids = [self.model.geom(name).id for name in
               ["marker_0", "marker_1"]]
        colors = self.model.geom_rgba[ids].copy()
        try:
            self.model.geom_rgba[ids, 3] = 0
            self.assertEqual(detect_targets(self.camera.capture(self.data), self.vision_config), [])
        finally:
            self.model.geom_rgba[ids] = colors

    def test_invalid_depth_preserves_2d_detection(self):
        frame = self.camera.capture(self.data)
        frame.aligned_depth_m[:] = np.nan
        detections = detect_targets(frame, self.vision_config)
        self.assertEqual(len(detections), 2)
        for item in detections:
            self.assertIsNone(item["surface_point_world_m"])
            self.assertIsNone(item["depth_m"])


if __name__ == "__main__":
    unittest.main()
