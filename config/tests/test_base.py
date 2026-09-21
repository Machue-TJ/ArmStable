"""Floating-base trajectory, dynamics and arm-index regression checks."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

# Test discovery imports this module before test_vision; choose EGL before MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/piper-matplotlib")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.flobase.piper_base import BasePose, BaseTrajectory, BaseVelocity, FloatingBase
from config.train_sets import BASE_HOLD_S, BASE_SEGMENT_S, base_pose


class TrajectoryTest(unittest.TestCase):
    def test_random_training_pose_hold_bounds_derivatives_and_replay(self):
        for t in (0, .25, .499999, .5):
            pose = base_pose(t)
            np.testing.assert_array_equal(pose.position_m, [0, 0, .1])
            np.testing.assert_array_equal(pose.quat_wxyz, [1, 0, 0, 0])
        dt = .01
        for seed in (7, 123):
            times = np.arange(0, BASE_HOLD_S + 5 * BASE_SEGMENT_S, dt)
            poses = [base_pose(t, seed) for t in times]
            xyz = np.array([p.position_m for p in poses]) - [0, 0, .1]
            quats = np.array([p.quat_wxyz for p in poses])
            rpy = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_euler("xyz")
            coordinates = np.c_[xyz, rpy]
            self.assertTrue(np.all(np.abs(xyz) <= .1 + 1e-12))
            self.assertTrue(np.all(np.abs(rpy) <= np.deg2rad(5) + 1e-12))
            for order, linear, angular in ((1, .046875, 2.34375),
                                            (2, .018043, .90211), (3, .023438, 1.171875)):
                derivative = np.diff(coordinates, n=order, axis=0) / dt**order
                limits = np.r_[np.full(3, linear), np.full(3, np.deg2rad(angular))]
                self.assertTrue(np.all(np.abs(derivative) <= limits + 1e-6))
            # Out-of-order evaluation and reset must reproduce the same path.
            np.testing.assert_array_equal(base_pose(times[50], seed).position_m, poses[50].position_m)
        self.assertGreater(np.linalg.norm(base_pose(6, 7).position_m - base_pose(6, 8).position_m), .01)

    def test_interpolation_endpoints_and_quaternion_shortest_path(self):
        trajectory = BaseTrajectory([1, 3], [[0, 0, 0], [2, 4, 6]],
                                    rpy_rad=[[0, 0, 0], [0, 0, np.pi]])
        pose = trajectory(2)
        np.testing.assert_allclose(pose.position_m, [1, 2, 3])
        np.testing.assert_allclose(pose.quat_wxyz, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-12)
        np.testing.assert_allclose(trajectory(0).position_m, [0, 0, 0])
        np.testing.assert_allclose(trajectory(20).position_m, [2, 4, 6])
        antipodal = BaseTrajectory([0, 1], [[0, 0, 0]] * 2,
                                   quat_wxyz=[[2, 0, 0, 0], [-2, 0, 0, 0]])
        np.testing.assert_allclose(antipodal(0.5).quat_wxyz, [1, 0, 0, 0])
        single = BaseTrajectory([0], [[1, 2, 3]], rpy_rad=[[0, 0, 0]])
        np.testing.assert_allclose(single(100).position_m, [1, 2, 3])

    def test_file_formats(self):
        values = dict(time_s=[0, 2], position_m=[[0, 0, 0], [1, 2, 3]],
                      rpy_rad=[[0, 0, 0], [0.2, 0.4, 0.6]])
        expected = BaseTrajectory(**values)(1)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "base.json").write_text(json.dumps(values))
            np.savez(directory / "base.npz", **values)
            (directory / "base.csv").write_text("time_s,x,y,z,roll,pitch,yaw\n0,0,0,0,0,0,0\n2,1,2,3,0.2,0.4,0.6\n")
            for suffix in ("json", "npz", "csv"):
                with self.subTest(suffix=suffix):
                    pose = BaseTrajectory.load(directory / f"base.{suffix}")(1)
                    np.testing.assert_allclose(pose.position_m, expected.position_m)
                    np.testing.assert_allclose(pose.quat_wxyz, expected.quat_wxyz)
            (directory / "quat.csv").write_text("time_s,x,y,z,qw,qx,qy,qz\n0,1,2,3,1,0,0,0\n")
            np.testing.assert_allclose(BaseTrajectory.load(directory / "quat.csv")(3).position_m, [1, 2, 3])

    def test_invalid_input(self):
        for times in ([], [1, 1], [2, 1], [-1, 0], [0, np.nan]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                BaseTrajectory(times, np.zeros((len(times), 3)), rpy_rad=np.zeros((len(times), 3)))
        for quaternion in ([0, 0, 0, 0], [1, 0, 0], [np.inf, 0, 0, 0]):
            with self.assertRaises(ValueError):
                BasePose([0, 0, 0], quaternion)
        with self.assertRaises(ValueError):
            BaseTrajectory([0], [[0, 0, 0]])
        with self.assertRaises(ValueError):
            BaseTrajectory([0], [[0, 0, 0]], rpy_rad=[[0, 0, 0]], quat_wxyz=[[1, 0, 0, 0]])


class FloatingBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "xml/agilex/scene.xml"))

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.base = FloatingBase(self.model, self.data)

    def test_whole_arm_and_camera_follow_without_changing_joint_angles_or_targets(self):
        names = ["link1", "link3", "ee_center_body", "d435i_mount"]
        original = {name: self.data.body(name).xpos.copy() for name in names}
        camera_before = self.data.cam_xpos.copy()
        rotation_before = self.data.cam_xmat.copy().reshape(-1, 3, 3)
        target = self.data.body("target_green_cube").xpos.copy()
        joints = [self.data.joint(f"joint{i}").qpos.copy() for i in range(1, 9)]
        self.base.set_pose([0.1, -0.2, 0.3], rpy_rad=[0.2, -0.3, 0.4])
        rotation = self.data.body("base_link").xmat.reshape(3, 3)
        for name in names:
            np.testing.assert_allclose(self.data.body(name).xpos,
                                       rotation @ original[name] + [0.1, -0.2, 0.3], atol=1e-10)
        np.testing.assert_allclose(self.data.cam_xpos, camera_before @ rotation.T + [0.1, -0.2, 0.3], atol=1e-10)
        np.testing.assert_allclose(self.data.cam_xmat.reshape(-1, 3, 3), rotation @ rotation_before, atol=1e-10)
        np.testing.assert_allclose(self.data.body("target_green_cube").xpos, target)
        for i, expected in enumerate(joints, 1):
            np.testing.assert_allclose(self.data.joint(f"joint{i}").qpos, expected)

    def test_physics_tracks_translation_and_rotation_and_reset_restarts(self):
        self.base.set_motion(BaseTrajectory([0, 0.5, 1],
                                            [[0, 0, 0.1], [0.05, 0.02, 0.15], [0.05, 0.02, 0.15]],
                                            rpy_rad=[[0, 0, 0], [0.04, 0.06, 0.15], [0.04, 0.06, 0.15]]))
        for _ in range(round(1 / self.model.opt.timestep)):
            self.base.step()
        actual, target = self.base.get_pose(), self.base.motion(1)
        np.testing.assert_allclose(actual.position_m, target.position_m, atol=2e-3)
        self.assertGreater(abs(np.dot(actual.quat_wxyz, target.quat_wxyz)), 0.999)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        mujoco.mj_resetData(self.model, self.data)
        self.base.reset()
        np.testing.assert_allclose(self.base.get_pose().position_m, [0, 0, 0.1])
        self.assertEqual(self.base.start_time, 0)

    def test_callback_uses_simulation_time_and_pose_stops_playback(self):
        self.data.time = 12
        times = []
        def motion(t):
            times.append(t)
            return BasePose.from_rpy([t, 0, 0.2], [0, 0, t])
        self.base.set_motion(motion)
        self.base.step()
        self.base.step()
        np.testing.assert_allclose(times, [0, 0, self.model.opt.timestep], atol=1e-12)
        self.base.set_pose([0, 0, 0.3])
        self.assertIsNone(self.base.motion)
        mujoco.mj_resetData(self.model, self.data)
        self.base.reset()
        np.testing.assert_allclose(self.base.get_pose().position_m, [0, 0, 0.3])

    def test_release_allows_base_motion(self):
        self.base.set_pose([0, 0, 1])
        self.base.release()
        self.data.qvel[self.base.qvel_slice] = [0.2, 0, 0, 0, 0, 0]
        for _ in range(10):
            self.base.step()
        self.assertEqual(self.data.eq_active[self.base.weld_id], 0)
        self.assertGreater(self.base.get_pose().position_m[0], 0.002)

    def test_world_velocity_integration_reset_and_mode_switch(self):
        start = BasePose.from_rpy([0, 0, 1], [0.4, -0.3, 0.2])
        velocity = BaseVelocity([0.02, -0.01, 0.03], [0, 0, 0.2])
        self.base.set_velocity_motion(lambda t: velocity, initial_pose=start)
        np.testing.assert_allclose(self.base.get_velocity().angular_rad_s, velocity.angular_rad_s, atol=1e-12)
        for _ in range(100):
            self.base.step()
        elapsed = 100 * self.model.opt.timestep
        np.testing.assert_allclose(self.data.mocap_pos[self.base.mocap_id],
                                   start.position_m + velocity.linear_m_s * elapsed, atol=1e-12)
        delta = BasePose.from_rpy([0, 0, 0], [0, 0, 0.2 * elapsed]).quat_wxyz
        expected = np.empty(4)
        mujoco.mju_mulQuat(expected, delta, start.quat_wxyz)
        np.testing.assert_allclose(self.data.mocap_quat[self.base.mocap_id], expected, atol=1e-12)
        self.assertLess(np.linalg.norm(self.base.get_pose().position_m - self.base.target_pose.position_m), 0.005)
        mujoco.mj_resetData(self.model, self.data)
        self.base.reset()
        np.testing.assert_allclose(self.base.get_pose().position_m, start.position_m)
        np.testing.assert_allclose(self.base.get_pose().quat_wxyz, start.quat_wxyz)
        self.base.set_pose([0, 0, 0.2])
        self.assertIsNone(self.base.velocity_motion)


class EnvironmentTest(unittest.TestCase):
    def test_arm_control_and_reset_with_remote_base(self):
        from piper_rl_mujoco import PandaObstacleEnv
        env = PandaObstacleEnv(base_motion=ROOT / "config/flobase/base_motion.json")
        try:
            env.base.set_pose([10, -5, 2], rpy_rad=[0.1, 0.2, 0.3])
            observation, _ = env.reset(seed=7)
            self.assertEqual(observation.shape, (9,))
            np.testing.assert_allclose(observation[:6], env.initial_joint_pos)
            np.testing.assert_allclose(env.data.ctrl[env.arm_ctrl_ids], env.initial_joint_pos)
            np.testing.assert_allclose(env.base.get_pose().position_m, [10, -5, 4])
            base_body = env.data.body("base_link")
            local_goal = base_body.xmat.reshape(3, 3).T @ (env.goal - base_body.xpos)
            self.assertGreater(local_goal[0], 0.2)
            self.assertGreater(local_goal[2], 0.2)
            observation, reward, _, _, _ = env.step(np.zeros(6))
            expected_ctrl = env.model.jnt_range[env.arm_joint_ids].mean(axis=1)
            np.testing.assert_allclose(env.data.ctrl[env.arm_ctrl_ids], expected_ctrl)
            self.assertTrue(np.isfinite(observation).all())
            self.assertTrue(np.isfinite(reward))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
