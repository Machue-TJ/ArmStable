"""Unified configuration and independent marker-window regression checks."""
import json
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from config.cli import parse_cli
from config.settings import load_project_config
from config.episode import EpisodeInitializer, load_episode_config, make_model
from config.flobase.piper_base import FloatingBase
from config.imu.simulation import load_imu_config
from config.vision.piper_vision import D435iCamera, load_vision_config


class SettingsTest(unittest.TestCase):
    def test_file_defaults_cli_precedence_and_no_shared_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({"episode": {"markers": {"count": 3}},
                                        "vision": {"fps": 15}, "imu": {"noise_enabled": False},
                                        "cli": {"camera": {"frames": 7, "seed": 11}}}))
            args = parse_cli("camera", ["--config", str(path), "--marker-count", "4", "--marker-fovy", "20", "35"])
            self.assertEqual((args.frames, args.seed), (7, 11))
            episode = load_episode_config(args.settings)
            self.assertEqual(episode["markers"]["count"], 4)
            self.assertEqual(episode["markers"]["marker_fovy"], [20, 35])
            self.assertEqual(load_vision_config(path)["fps"], 15)
            self.assertFalse(load_imu_config(path)["noise_enabled"])
            args.settings["episode"]["markers"]["count"] = 99
            self.assertEqual(load_project_config()["episode"]["markers"]["count"], 6)
            # Replacing an override file must invalidate the parsed-JSON cache.
            path.write_text('{"vision": {"fps": 6}}')
            self.assertEqual(load_vision_config(path)["fps"], 6)

    def test_all_entrypoint_defaults_and_separate_training_output(self):
        for command in ("camera", "rl", "imu"):
            args = parse_cli(command, [])
            self.assertIn("vision", args.settings)
        exported = parse_cli("calibration", ["--output", "/tmp/calibration.json"])
        self.assertEqual(exported.depth_size, [848, 480])
        self.assertEqual(exported.color_size, [1280, 720])
        self.assertNotEqual(parse_cli("rl", ["--mode", "train"]).model_path,
                            parse_cli("rl", ["--mode", "test"]).model_path)

    def test_invalid_marker_window_rejected(self):
        for fov in ([0, 30], [30, 180], [30], [float("nan"), 30]):
            with self.subTest(fov=fov), self.assertRaises(ValueError):
                load_episode_config({"markers": {"marker_fovy": fov}})
        with self.assertRaises(ValueError):
            load_vision_config({"fovy_deg": 30})

    def test_non_square_marker_window_preserves_imaging_and_fixed_height(self):
        config = load_episode_config({"markers": {"marker_fovy": [20, 35], "plane_depth_m": 1}})
        model = make_model(config)
        data = mujoco.MjData(model)
        camera = D435iCamera(model)
        try:
            base = FloatingBase(model, data)
            initializer = EpisodeInitializer(model, data, base, config)
            before = model.cam_intrinsic.copy(), model.cam_fovy.copy()
            info = initializer.reset(np.random.default_rng(7))
            np.testing.assert_allclose(base.get_pose().position_m, [0, 0, 4])
            points = np.asarray(info["marker_positions_world_m"])
            optical = (points-data.cam_xpos[camera.rgb_id]) @ (
                data.cam_xmat[camera.rgb_id].reshape(3, 3) @ np.diag([1, -1, -1]))
            half = np.deg2rad([10, 17.5])
            self.assertTrue(np.all(np.abs(optical[:, :2]) + .0075/np.cos(half)
                                   <= optical[:, 2, None]*np.tan(half)))
            np.testing.assert_allclose(model.geom_size[initializer.plane_gid, :2], np.tan(half))
            np.testing.assert_array_equal(model.cam_intrinsic, before[0])
            np.testing.assert_array_equal(model.cam_fovy, before[1])
            frame = camera.capture(data)
            self.assertEqual(frame.rgb.shape, (720, 1280, 3))
            self.assertEqual(frame.native_depth_m.shape, (480, 848))
            # Changing only the allowed window with the same explicit layout
            # must produce exactly the same image, intrinsics and native depth.
            wider = load_episode_config({"markers": {"marker_fovy": [40, 40], "plane_depth_m": 1}})
            other = EpisodeInitializer(model, data, base, wider)
            other.reset(np.random.default_rng(8), sample_id=info["sample_id"],
                        marker_positions_m=info["marker_positions_camera_m"])
            camera.reset(0)
            repeated = camera.capture(data)
            np.testing.assert_array_equal(frame.rgb, repeated.rgb)
            np.testing.assert_array_equal(frame.native_depth_z16, repeated.native_depth_z16)
            np.testing.assert_array_equal(frame.rgb_intrinsics, repeated.rgb_intrinsics)
        finally:
            camera.close()

    def test_repeat_detection_reuses_image_analysis_and_reset_clears_cache(self):
        config = load_episode_config({"markers": {"plane_depth_m": 1}})
        model = make_model(config)
        data = mujoco.MjData(model)
        camera = D435iCamera(model)
        try:
            EpisodeInitializer(model, data, FloatingBase(model, data), config).reset(np.random.default_rng(7))
            frame = camera.capture(data)
            with patch("config.vision.piper_vision.detect_targets", return_value=[]) as detector:
                camera.detect(frame)
                camera.detect(frame)
                self.assertEqual(detector.call_count, 1)
                camera.reset()
                camera.detect(frame)
                self.assertEqual(detector.call_count, 2)
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main()
