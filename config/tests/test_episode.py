"""CSV reset, full-sphere FOV, real RGB/depth and configurable motion coverage."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/piper-matplotlib")

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.episode import EpisodeInitializer, load_episode_config, make_model, project_path, configure_base
from config.flobase.piper_base import BasePose, FloatingBase
from config.vision.piper_vision import D435iCamera, detect_targets, load_vision_config, project


class EpisodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_episode_config()
        cls.model = make_model(cls.config)
        cls.data = mujoco.MjData(cls.model)
        cls.camera = D435iCamera(cls.model, load_vision_config())
        cls.base = FloatingBase(cls.model, cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.camera.close()

    def setUp(self):
        self.base.set_pose([0, 0, 0])
        self.initializer = EpisodeInitializer(self.model, self.data, self.base, self.config)
        self.camera.reset()

    def assert_fov(self, info):
        points = np.asarray(info["marker_positions_world_m"])
        local = np.asarray(info["marker_positions_camera_m"])
        np.testing.assert_allclose(local[:, 2], info["marker_plane_depth_m"])
        self.assertTrue(np.all((local[:, 2] >= .2) & (local[:, 2] <= 2.8)))
        for name in ("d435i_rgb", "d435i_depth", "d435i_ir_right"):
            camera = self.model.camera(name).id
            optical_rotation = self.data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
            optical = (points - self.data.cam_xpos[camera]) @ optical_rotation
            if name == "d435i_rgb":
                half = np.deg2rad(info["marker_fovy"]) / 2
                margin = optical[:, 2, None] * np.tan(half) - np.abs(optical[:, :2])
                self.assertTrue(np.all(margin >= .0075 / np.cos(half) - 1e-10))
            # Analytical projected extrema of a sphere under pinhole projection.
            k = self.camera.intrinsics(camera)
            radius = .0075
            xy, z = optical[:, :2], optical[:, 2, None]
            root = radius * np.sqrt(z*z + xy*xy - radius*radius)
            lo = (xy*z-root)/(z*z-radius*radius)*np.diag(k)[:2]+k[:2, 2]
            hi = (xy*z+root)/(z*z-radius*radius)*np.diag(k)[:2]+k[:2, 2]
            size = [self.camera.width, self.camera.height] if name == "d435i_rgb" else [self.camera.depth_width, self.camera.depth_height]
            self.assertTrue(np.all(lo >= -.5) and np.all(hi <= np.asarray(size)-.5))

    def test_all_csv_poses_full_sphere_projection_and_joint_control(self):
        rng = np.random.default_rng(20)
        for row in self.initializer.poses:
            with self.subTest(sample_id=row["sample_id"]):
                info = self.initializer.reset(rng, sample_id=row["sample_id"])
                self.assertEqual(self.base.get_pose().position_m[2], 4.0)
                expected = [row[f"q{i}_rad"] for i in range(1, 7)]
                np.testing.assert_allclose(self.data.qpos[self.initializer.qids], expected)
                np.testing.assert_allclose(self.data.ctrl[self.initializer.cids], expected)
                np.testing.assert_allclose(self.data.geom_xpos[self.initializer.gids], info["marker_positions_world_m"])
                self.assertGreater(np.min(self.data.geom_xpos[self.initializer.gids, 2]) - .0075, 0)
                plane = self.data.body("marker_plane")
                np.testing.assert_allclose(plane.xpos, info["marker_plane_position_world_m"])
                np.testing.assert_allclose(plane.xmat.reshape(3, 3)[:, 2], info["camera_zaxis_world"], atol=1e-10)
                for contact in self.data.contact[:self.data.ncon]:
                    if self.initializer.floor_gid in (contact.geom1, contact.geom2):
                        self.assertGreaterEqual(contact.dist, -1e-5)
                self.assert_fov(info)

    def test_scene_material_plane_and_episode_height_replay(self):
        self.assertGreaterEqual(self.model.geom("floor").id, 0)
        self.assertIn(mujoco.mjtTexture.mjTEXTURE_SKYBOX, self.model.tex_type)
        material = self.model.material("marker_material").id
        self.assertEqual(self.model.mat_emission[material], 0)
        self.assertGreater(self.model.mat_specular[material], .8)
        self.assertEqual(self.camera.scene_option.geomgroup[4], 0)
        self.base.set_motion(lambda t: BasePose.from_rpy(
            [0.02*t, 0, .7 + .03*t], [0, 0, .1*t]))
        first = self.initializer.reset(np.random.default_rng(1), sample_id=1)
        lift = first["base_height_offset_m"]
        self.assertAlmostEqual(lift, 3.3)
        self.assertEqual(self.base.get_pose().position_m[2], 4.0)
        for _ in range(5):
            self.base.step()
        expected = self.base.motion(self.data.time-self.model.opt.timestep).position_m + [0, 0, lift]
        np.testing.assert_allclose(self.data.mocap_pos[self.base.mocap_id], expected, atol=1e-12)
        repeated = self.initializer.reset(np.random.default_rng(1), sample_id=1)
        self.assertEqual(first, repeated)

    def test_seed_base_transform_and_world_fixed_markers(self):
        first = self.initializer.reset(np.random.default_rng(7))
        self.assertEqual(first, self.initializer.reset(np.random.default_rng(7)))
        self.base.set_pose([3, -2, 1], rpy_rad=[.3, -.4, .7])
        transformed = self.initializer.reset(np.random.default_rng(7))
        rotation = self.data.body("base_link").xmat.reshape(3, 3)
        np.testing.assert_allclose(transformed["marker_positions_world_m"],
                                   (np.asarray(first["marker_positions_world_m"]) - [0, 0, first["base_height_offset_m"]])
                                   @ rotation.T + [3, -2, 1 + transformed["base_height_offset_m"]])
        np.testing.assert_allclose(transformed["camera_zaxis_world"], rotation @ first["camera_zaxis_world"])
        self.base.set_pose([4, 0, 2], rpy_rad=[0, 0, 1])
        np.testing.assert_allclose(self.data.geom_xpos[self.initializer.gids], transformed["marker_positions_world_m"])

    def test_depth_endpoints_are_detected_after_repeated_resets(self):
        for depth in (.2, 2.8):
            self.initializer.settings["plane_depth_m"] = depth
            for sample_id in (1, 3, 200, 945):
                with self.subTest(depth=depth, sample_id=sample_id):
                    info = self.initializer.reset(np.random.default_rng(sample_id), sample_id=sample_id)
                    self.assert_fov(info)
                    self.camera.reset(0)
                    frame = self.camera.capture(self.data)
                    self.assertEqual(frame.rgb.shape, (720, 1280, 3))
                    np.testing.assert_allclose(np.rad2deg(2 * np.arctan(np.array([1280, 720]) / (2 * np.diag(frame.rgb_intrinsics)[:2]))), [69, 42])
                    detections = detect_targets(frame, self.camera.config)
                    self.assertEqual(len(detections), 6)
                    if depth == .2:
                        self.assertTrue(all(item["depth_m"] is not None for item in detections))
                    for item in detections:
                        if item["center_position_world_m"] is None:
                            self.assertEqual(item["measurement_status"], "insufficient_depth")
                            continue
                        error = np.linalg.norm(np.asarray(info["marker_positions_world_m"])
                                               - item["center_position_world_m"], axis=1).min()
                        self.assertLess(error, .001 if depth == .2 else .03)
                    points = (np.asarray(info["marker_positions_world_m"]) - frame.world_from_optical[:3, 3]) @ frame.world_from_optical[:3, :3]
                    centers = project(points, frame.rgb_intrinsics)
                    detected = np.array([item["center_uv"] for item in detections])
                    self.assertTrue(np.all(np.linalg.norm(centers[:, None] - detected, axis=-1).min(axis=1) < 2))

    def test_custom_positions_callbacks_and_invalid_layouts(self):
        info = self.initializer.reset(np.random.default_rng(7))
        points = np.asarray(info["marker_positions_camera_m"])
        self.initializer.generator = lambda context, rng: points.copy()
        repeat = self.initializer.reset(np.random.default_rng(2), sample_id=info["sample_id"])
        np.testing.assert_allclose(repeat["marker_positions_camera_m"], points)
        for invalid in (np.zeros((5, 3)), np.full((6, 3), np.nan),
                        np.tile([0, 0, 3], (6, 1)), np.tile([1, 0, 1], (6, 1)),
                        np.tile([0, 0, 1], (6, 1))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.initializer.reset(np.random.default_rng(0), marker_positions_m=invalid)
        with self.assertRaises(ValueError):
            self.initializer.reset(np.random.default_rng(0), sample_id=-1)


class RLResetTest(unittest.TestCase):
    def test_seed_reset_controls_sensors_and_configured_callback(self):
        from piper_rl_mujoco import PandaObstacleEnv
        config = {"base": {"mode": "velocity_callback", "position_m": [1, 2, 3],
                            "callback": "config.motion_examples:base_velocity"}}
        env = PandaObstacleEnv(episode_config=config)
        try:
            obs, info = env.reset(seed=7)
            env.step(np.zeros(6))
            _, second = env.reset()
            self.assertNotEqual(info["sample_id"], second["sample_id"])
            repeated, repeated_info = env.reset(seed=7)
            np.testing.assert_array_equal(obs, repeated)
            self.assertEqual(info, repeated_info)
            np.testing.assert_allclose(env.prev_action, 0)
            self.assertEqual(env.data.time, 0)
            np.testing.assert_allclose(env.data.ctrl[env.arm_ctrl_ids], obs[:6], rtol=1e-6)
            np.testing.assert_allclose(env.base.get_pose().position_m, [1, 2, 4])
            np.testing.assert_allclose(info["base_linear_velocity_m_s"], [.02, 0, 0])
            self.assertEqual(len(info["marker_positions_world_m"]), 6)
            frame, detections = env.get_camera_observation()
            self.assertEqual(frame.sim_time, 0)
            self.assertEqual(len(detections), 6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
