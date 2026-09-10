"""Integration checks using actual EGL-rendered RGB/depth, no USB camera."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from config.vision.piper_vision import D435iCamera, ROOT, detect_targets


class VisionIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "xml/agilex/scene.xml"))
        cls.data = mujoco.MjData(cls.model)
        cls.camera = D435iCamera(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.camera.close()

    def setUp(self):
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        mujoco.mj_forward(self.model, self.data)

    def test_model_dimensions_and_mount(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (15, 14, 7))
        mount = self.model.body("d435i_mount")
        self.assertEqual(mount.parentid[0], self.model.body("link6").id)
        before = self.camera.capture(self.data).world_from_optical.copy()
        self.data.joint("joint1").qpos[0] += 0.2
        after = self.camera.capture(self.data).world_from_optical
        self.assertGreater(np.linalg.norm(before[:3, 3] - after[:3, 3]), 0.01)
        wrist = self.data.body("link6")
        mount_local = wrist.xmat.reshape(3, 3).T @ (
            self.data.body("d435i_mount").xpos - wrist.xpos)
        np.testing.assert_allclose(mount_local, [0, -0.065, 0.035], atol=1e-8)

    def assert_surface_detections(self, frame):
        detections = detect_targets(frame)
        self.assertCountEqual([d["label"] for d in detections], ["green_cube", "blue_sphere"])
        for item in detections:
            self.assertIsNotNone(item["depth_m"])
            p = np.array(item["surface_point_world_m"])
            if item["label"] == "green_cube":
                center = self.data.body("target_green_cube").xpos
                self.assertLessEqual(np.max(np.abs(p - center)), 0.031)
                self.assertAlmostEqual(p[2], 0.06, delta=0.002)
            else:
                center = self.data.body("target_blue_sphere").xpos
                self.assertAlmostEqual(np.linalg.norm(p - center), 0.03, delta=0.002)

    def test_rgb_depth_and_world_surface_positions(self):
        frame = self.camera.capture(self.data)
        self.assertEqual(frame.rgb.shape, (360, 640, 3))
        self.assertEqual(frame.aligned_depth_m.shape, (360, 640))
        self.assertGreater(frame.rgb_intrinsics[0, 0], frame.depth_intrinsics[0, 0])
        self.assert_surface_detections(frame)

    def test_detection_after_one_second_of_physics(self):
        for _ in range(round(1 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assert_surface_detections(self.camera.capture(self.data))

    def test_hidden_targets_do_not_detect_floor(self):
        ids = [self.model.geom(name).id for name in
               ["target_green_cube_geom", "target_blue_sphere_geom"]]
        colors = self.model.geom_rgba[ids].copy()
        try:
            self.model.geom_rgba[ids, 3] = 0
            self.assertEqual(detect_targets(self.camera.capture(self.data)), [])
        finally:
            self.model.geom_rgba[ids] = colors

    def test_invalid_depth_preserves_2d_detection(self):
        frame = self.camera.capture(self.data)
        frame.aligned_depth_m[:] = np.nan
        detections = detect_targets(frame)
        self.assertEqual(len(detections), 2)
        for item in detections:
            self.assertIsNone(item["surface_point_world_m"])
            self.assertIsNone(item["depth_m"])


if __name__ == "__main__":
    unittest.main()
