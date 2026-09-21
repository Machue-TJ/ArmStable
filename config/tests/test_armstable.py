"""Sensor-only actor, rigid-pose rewards, control timing and real PPO integration."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/piper-matplotlib")
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from armstable_rl import (ACTOR_DIM, PRIVILEGED_DIM, ActorOnly, ArmStableEnv,
                         AsymmetricPolicy, ImageMarkerSlots, load_task_config,
                         rigid_points, stabilization_reward)


def detection(u, v, depth=1.0):
    return {"center_uv": [u, v], "center_depth_m": depth,
            "center_position_optical_m": None if depth is None else
                [(u - 320) * depth / 900, (v - 240) * depth / 900, depth]}


class GeometryAndSlotsTest(unittest.TestCase):
    def test_slots_survive_reordering_and_mark_missing_without_truth(self):
        tracker = ImageMarkerSlots(80)
        initial = [detection(100 + 100*i, 200) for i in range(6)]
        tracker.reset(initial[::-1])
        moved = [detection(102 + 100*i, 203, None if i == 2 else 1.1) for i in range(6) if i != 4]
        current = tracker.update(moved[::-1])
        np.testing.assert_allclose(current[[0, 1, 3, 5]],
                                   ImageMarkerSlots.measurements([moved[i] for i in (0, 1, 3, 4)]))
        self.assertTrue(np.isnan(current[4]).all())
        self.assertTrue(np.isnan(current[2]).all())
        self.assertTrue(tracker.visible[2])
        np.testing.assert_allclose(tracker.update(initial), ImageMarkerSlots.measurements(initial))
        self.assertTrue(np.isnan(tracker.update([detection(2000, 2000)])).all())

    def test_cube_geometry_and_reward_monotonicity(self):
        settings = load_task_config()["reward"]
        points = rigid_points(np.zeros(3), np.eye(3))
        np.testing.assert_allclose(points[0], 0)
        np.testing.assert_allclose(points[1:, 2], .04)
        self.assertAlmostEqual(np.linalg.norm(points[1] - points[2]), .08)
        xyz = np.array([[.05*i, 0, 1.] for i in range(6)])
        zeros = np.zeros(6)
        def reward(current=xyz, rigid=points, action=zeros):
            return stabilization_reward(xyz, current, points, rigid, action, zeros, zeros, zeros, settings)
        perfect, terms = reward()
        self.assertEqual(terms["rigid"], 1)
        self.assertEqual(terms["marker"], 1)
        self.assertGreater(perfect, reward(xyz + [.02, 0, 0])[0])
        self.assertGreater(perfect, reward(xyz + [0, 0, .2])[0])
        self.assertGreater(perfect, reward(rigid=points + [.01, 0, 0])[0])
        rotated = rigid_points(np.zeros(3), np.diag([-1, -1, 1]))
        self.assertGreater(perfect, reward(rigid=rotated)[0])
        self.assertGreater(perfect, reward(action=np.ones(6))[0])
        self.assertGreater(perfect, reward(np.full((6, 3), np.nan))[0])


class ArmStableIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ArmStableEnv(episode_config={"markers": {"plane_depth_m": 1.0}},
                              task_config={"environment": {"episode_duration_s": 2.0}})

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_seed_history_scaling_action_limits_and_gym_contract(self):
        env = self.env
        first, info = env.reset(seed=7)
        repeat, second_info = env.reset(seed=7)
        self.assertEqual(info, second_info)
        for key in first:
            np.testing.assert_array_equal(first[key], repeat[key])
        self.assertEqual(first["policy"].shape, (102,))
        self.assertEqual(first["privileged"].shape, (9,))
        np.testing.assert_allclose(first["policy"][:18].reshape(6, 3), env.reference_xyz / 3)
        np.testing.assert_allclose(first["policy"][18:36].reshape(6, 3), np.nan_to_num(env.current_xyz / 3))
        np.testing.assert_allclose(env.denormalize_joints(first["policy"][96:]),
                                   env.data.ctrl[env.ctrl_ids], atol=1e-6)
        np.testing.assert_allclose(env.denormalize_joints(-np.ones(6)), env.ranges[:, 0])
        np.testing.assert_allclose(env.denormalize_joints(np.ones(6)), env.ranges[:, 1])
        old_ctrl = env.data.ctrl[env.ctrl_ids].copy()
        old_time = float(env.data.time)
        observation, reward, _, _, info = env.step(np.ones(6))
        delta = env.settings["max_joint_speed_rad_s"] * (env.data.time - old_time)
        self.assertTrue(np.all(np.abs(env.data.ctrl[env.ctrl_ids] - old_ctrl) <= delta + 1e-10))
        np.testing.assert_array_equal(observation["policy"][36:60], first["policy"][42:66])
        np.testing.assert_array_equal(observation["policy"][66:90], first["policy"][72:96])
        np.testing.assert_array_equal(observation["policy"][96:], 1)
        self.assertTrue(np.isfinite(reward))
        self.assertLessEqual(env.data.time - info["frame_time_s"], 1/env.camera.config["fps"])
        check_env(env, warn=True, skip_render_check=True)

    def test_privileged_pose_cannot_change_actor_observation_or_image_matching(self):
        env = self.env
        env.reset(seed=8)
        before = env._observation()
        position, rotation = env._ee_pose()
        with patch.object(env, "_ee_pose", return_value=(position + [1, 2, 3], rotation)):
            after = env._observation()
        np.testing.assert_array_equal(before["policy"], after["policy"])
        self.assertFalse(np.array_equal(before["privileged"], after["privileged"]))
        frame, observed = env._capture()
        np.testing.assert_array_equal(frame.world_from_optical, np.eye(4))
        np.testing.assert_allclose(frame.world_from_depth_optical, env.color_from_depth)
        env.current_xyz[:] = np.nan
        missing = env._observation()["policy"]
        np.testing.assert_array_equal(missing[18:36], 0)
        self.assertTrue(np.isfinite(missing).all())

    def test_base_moves_after_hold_and_time_limit_is_truncation(self):
        env = self.env
        env.reset(seed=7)
        hold = env.prev_action.copy()
        initial_base_target = env.data.mocap_pos[env.base.mocap_id].copy()
        self.assertAlmostEqual(env.data.time, .5)
        np.testing.assert_allclose(env.data.mocap_pos[env.base.mocap_id], initial_base_target, atol=1e-12)
        while True:
            _, _, terminated, truncated, _ = env.step(hold)
            if terminated or truncated:
                break
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertGreater(np.linalg.norm(env.data.mocap_pos[env.base.mocap_id] - initial_base_target), .001)

    def test_control_and_vision_have_separate_cadences_and_cache(self):
        env = self.env
        env.reset(seed=7)
        self.assertEqual(env.settings["control_hz"], 50)
        self.assertEqual(env.camera.config["fps"], 25)
        times, frames = [env.data.time], [env.marker_frame_time]
        hold = env.prev_action.copy()
        from armstable_rl import detect_markers
        with patch("armstable_rl.detect_markers", wraps=detect_markers) as detector:
            for index in range(10):
                _, _, _, _, info = env.step(hold)
                self.assertEqual(info["visual_updated"], index % 2 == 1)
                times.append(info["sim_time_s"])
                frames.append(info["frame_time_s"])
            self.assertEqual(detector.call_count, 5)
        np.testing.assert_allclose(np.diff(times), .02, atol=1e-12)
        np.testing.assert_allclose(np.diff(np.unique(frames)), .04, atol=1e-12)

    def test_episode_motion_seed_and_absolute_pose(self):
        env = self.env
        env.reset(seed=7)
        first = env.base.motion(4).position_m.copy()
        env.reset(seed=7)
        np.testing.assert_array_equal(env.base.motion(4).position_m, first)
        env.reset(seed=8)
        self.assertGreater(np.linalg.norm(env.base.motion(4).position_m - first), .001)
        p0, r0 = env._ee_pose()
        base = env.data.body("base_link")
        local_position = base.xmat.reshape(3, 3).T @ (p0 - base.xpos)
        local_rotation = base.xmat.reshape(3, 3).T @ r0
        motion = env.base.motion
        env.base.set_pose([.08, -.04, 4.03], rpy_rad=[.02, -.03, .05])
        position, rotation = env._ee_pose()
        base_rotation = env.data.body("base_link").xmat.reshape(3, 3)
        np.testing.assert_allclose(position, base.xpos + base_rotation @ local_position)
        np.testing.assert_allclose(rotation, base_rotation @ local_rotation)
        relative = env._observation()["privileged"][3:].reshape(2, 3).T
        reconstructed = np.column_stack((relative, np.cross(relative[:, 0], relative[:, 1])))
        np.testing.assert_allclose(reconstructed, env.reference_rotation.T @ rotation, atol=1e-7)
        env.base.set_motion(motion)

    def test_policy_isolation_ppo_update_save_reload_and_actor_export(self):
        env = self.env
        observation, _ = env.reset(seed=10)
        model = PPO(AsymmetricPolicy, env, n_steps=8, batch_size=8, n_epochs=1,
                    seed=10, device="cpu", verbose=0)
        policy = model.policy
        np.testing.assert_allclose(policy.predict_actor(observation["policy"]),
                                   observation["policy"][60:66], atol=.03)
        tensors, _ = policy.obs_to_tensor(observation)
        changed = {key: value.clone() for key, value in tensors.items()}
        changed["privileged"] += 3
        with torch.no_grad():
            mean1 = policy.get_distribution(tensors).distribution.mean.clone()
            mean2 = policy.get_distribution(changed).distribution.mean.clone()
            value1, value2 = policy.predict_values(tensors), policy.predict_values(changed)
        torch.testing.assert_close(mean1, mean2, rtol=0, atol=0)
        self.assertFalse(torch.equal(value1, value2))
        with torch.no_grad():
            actions, _, rollout_log_prob = policy(tensors)
            _, update_log_prob, _ = policy.evaluate_actions(tensors, actions)
        self.assertTrue(torch.all(actions.abs() <= 1))
        torch.testing.assert_close(rollout_log_prob, update_log_prob, atol=1e-5, rtol=1e-5)
        features = torch.randn(2, ACTOR_DIM + PRIVILEGED_DIM, requires_grad=True)
        policy.mlp_extractor.forward_actor(features).sum().backward()
        torch.testing.assert_close(features.grad[:, ACTOR_DIM:], torch.zeros((2, PRIVILEGED_DIM)))
        actor_before = [p.detach().clone() for p in policy.mlp_extractor.actor.parameters()]
        critic_before = [p.detach().clone() for p in policy.mlp_extractor.critic.parameters()]
        model.learn(total_timesteps=8)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, policy.mlp_extractor.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(critic_before, policy.mlp_extractor.critic.parameters())))
        action = policy.predict_actor(observation["policy"])
        self.assertTrue(np.all(np.abs(action) <= 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.zip"
            model.save(path)
            loaded = PPO.load(path, device="cpu")
            np.testing.assert_array_equal(action, loaded.policy.predict_actor(observation["policy"]))
            exported = torch.jit.script(ActorOnly(loaded.policy).eval())
            exported.save(str(Path(directory) / "actor.pt"))
            inference = torch.jit.load(str(Path(directory) / "actor.pt"))
            with torch.no_grad():
                actual = inference(torch.from_numpy(observation["policy"]))
            np.testing.assert_allclose(actual.numpy(), action, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
