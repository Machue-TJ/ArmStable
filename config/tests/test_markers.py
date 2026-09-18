"""Detection quality and temporal IDs use measured RGB-D, not simulator labels."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.episode import load_episode_config, make_model, EpisodeInitializer, project_path
from config.flobase.piper_base import FloatingBase
from config.vision.piper_vision import D435iCamera, load_vision_config
from config.vision.markers import detect_markers, estimate_sphere_center


class MarkerMeasurementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_episode_config({"markers": {"plane_depth_m": 1.0}})
        cls.model = make_model(cls.config)
        cls.data = mujoco.MjData(cls.model)
        cls.camera = D435iCamera(cls.model, load_vision_config())
        cls.initializer = EpisodeInitializer(cls.model, cls.data, FloatingBase(cls.model, cls.data), cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.camera.close()

    def setUp(self):
        self.initializer.reset(np.random.default_rng(200), sample_id=200)
        self.camera.reset(20)

    def test_multi_frame_accuracy_ids_and_no_stale_detections(self):
        previous = None
        errors = []
        for index in range(8):
            self.data.time = index / 30
            frame = self.camera.capture(self.data)
            detections = self.camera.detect(frame)
            self.assertEqual(len(detections), 6)
            ids = [item["track_id"] for item in detections]
            if previous is not None:
                self.assertEqual(ids, previous)
            previous = ids
            self.assertEqual(detections, self.camera.detect(frame))
            for item in detections:
                self.assertEqual(item["measurement_status"], "measured")
                error = np.linalg.norm(self.initializer.marker_positions_world_m - item["center_position_world_m"], axis=1).min()
                errors.append(error)
        self.assertLess(max(errors), .012)
        self.assertLess(np.mean(errors), .004)
        frame.sim_time += 1 / 30
        self.assertEqual(self.camera.marker_tracker.update([], frame), [])
        self.camera.reset()
        self.assertEqual(self.camera.marker_tracker.tracks, {})

    def test_depth_dropout_and_color_distractor(self):
        frame = deepcopy(self.camera.capture(self.data))
        frame.native_depth_m[:] = np.nan
        detections = detect_markers(frame, self.camera.config)
        self.assertEqual(len(detections), 6)
        self.assertTrue(all(item["center_position_world_m"] is None for item in detections))
        # A same-color long rectangle must not be called a sphere.
        frame.rgb[:] = 0
        cv2.rectangle(frame.rgb, (100, 100), (300, 120), (255, 0, 180), -1)
        self.assertEqual(detect_markers(frame, self.camera.config), [])

    def test_known_radius_fit_rejects_large_depth_outliers(self):
        rng = np.random.default_rng(2)
        xy = rng.uniform(-.003, .003, size=(100, 2))
        surface = np.column_stack((xy, 1 - np.sqrt(.0075**2 - np.sum(xy**2, axis=1))))
        surface[:, 2] += rng.normal(0, .0003, len(surface))
        surface[:10, 2] += .05
        center, uncertainty, count = estimate_sphere_center(surface, [0, 0, 1], .0075)
        np.testing.assert_allclose(center, [0, 0, 1], atol=.00015)
        self.assertLess(uncertainty, .0001)
        self.assertLess(count, 100)
        self.assertIsNone(estimate_sphere_center(surface[:2], [0, 0, 1], .0075))


if __name__ == "__main__":
    unittest.main()
