"""Asymmetric PPO for world-frame end-effector stabilization using RGB-D + IMU.

Actor: 102 sensor/history values. Critic: actor input + 9 relative EE pose values.
See config/doc/armstable_rl.md for units, missing data, rewards and commands.
"""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "test", "export"), default="smoke")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--config", help="Existing unified sensor/episode configuration")
    parser.add_argument("--task-config", help="Overrides for config/armstable.json")
    parser.add_argument("--model-path", default="models/armstable_ppo.zip")
    parser.add_argument("--actor-path", default="models/armstable_actor.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    args = parser.parse_args()
    for name in ("n_envs", "total_timesteps", "episodes", "steps", "n_steps", "batch_size", "n_epochs"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.headless or args.mode in ("train", "export"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    return args


# Select the rendering backend before MuJoCo/PyOpenGL imports; --help is cheap.
_args = parse_args() if __name__ == "__main__" else None
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/piper-matplotlib")

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from config.d435i import transform
from config.episode import EpisodeInitializer, configure_base, configure_viewer, load_callback, load_episode_config, make_model
from config.flobase.piper_base import FloatingBase
from config.imu import MujocoD435iIMU
from config.imu.simulation import load_imu_config
from config.settings import load_project_config, merge_settings, project_path, share_device_calibration
from config.train_sets import BASE_HOLD_S, RandomBaseMotion
from config.vision.markers import detect_markers
from config.vision.piper_vision import D435iCamera, load_vision_config, pixel_rays

ACTOR_DIM = 102
PRIVILEGED_DIM = 9
OBSERVATION_FORMAT = "marker_xyz_v1"
# 80 mm cube: center and four corners of the EE +Z face, in metres.
EE_POINTS = np.array([[0, 0, 0], [-.04, -.04, .04], [-.04, .04, .04],
                      [.04, -.04, .04], [.04, .04, .04]])


def load_task_config(value=None):
    with open(project_path("config/armstable.json"), encoding="utf-8") as stream:
        config = json.load(stream)
    if value is not None:
        if not isinstance(value, dict):
            with open(project_path(value), encoding="utf-8") as stream:
                value = json.load(stream)
        if set(value) - set(config):
            raise ValueError("Task config accepts environment, reward and ppo sections")
        config = merge_settings(config, value)
    env, reward = config["environment"], config["reward"]
    for key, value in env.items():
        if key != "base_callback" and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"environment.{key} must be finite and positive")
    for key, value in reward.items():
        if not np.isfinite(value) or value < 0 or ("sigma" in key and value == 0):
            raise ValueError(f"Invalid reward.{key}")
    return config


def rigid_points(position, rotation):
    return EE_POINTS @ np.asarray(rotation).T + position


def stabilization_reward(reference_xyz, current_xyz, reference_points, current_points,
                         action, previous_action, older_action, joint_velocity, settings):
    """Sensor marker alignment plus privileged rigid-body alignment and smoothness."""
    valid = np.isfinite(current_xyz).all(axis=1) & (current_xyz[:, 2] > 0)
    marker_scores = np.zeros(6)
    scales = [settings["marker_xy_sigma_m"], settings["marker_xy_sigma_m"],
              settings["marker_depth_sigma_m"]]
    error = (current_xyz[valid] - reference_xyz[valid]) / scales
    marker_scores[valid] = np.exp(-.5 * np.sum(error**2, axis=1))
    distances2 = np.sum((current_points - reference_points)**2, axis=1)
    terms = {
        "marker": float(marker_scores.mean()),
        "rigid": float(np.exp(-.5 * distances2 / settings["rigid_sigma_m"]**2).mean()),
        "action_rate": float(np.mean((action - previous_action)**2)),
        "action_acceleration": float(np.mean((action - 2 * previous_action + older_action)**2)),
        "joint_velocity": float(np.mean(joint_velocity**2)),
        "missing": float(1 - valid.mean()),
        "rigid_rms_m": float(np.sqrt(distances2.mean())),
    }
    reward = (settings["marker_weight"] * terms["marker"] + settings["rigid_weight"] * terms["rigid"]
              - sum(settings[key + "_weight"] * terms[key] for key in
                    ("action_rate", "action_acceleration", "joint_velocity", "missing")))
    return float(reward), terms


class ImageMarkerSlots:
    """Six sensor-only slots returning RGB-optical (X, Y, depth) in metres.

    Pixels are used only for image association, including during depth holes.
    Neither simulator object IDs nor world poses enter association. An absent
    point or invalid depth yields NaN for the whole metric position.
    """
    def __init__(self, gate_px):
        self.gate_px = gate_px

    @staticmethod
    def measurements(detections):
        points = np.array([item.get("center_position_optical_m")
                           if item.get("center_position_optical_m") is not None else [np.nan]*3
                           for item in detections], dtype=float).reshape(-1, 3)
        points[~np.isfinite(points).all(axis=1) | (points[:, 2] <= 0)] = np.nan
        return points

    @staticmethod
    def image_centers(detections):
        return np.array([item["center_uv"] for item in detections], dtype=float).reshape(-1, 2)

    def reset(self, detections):
        observed = self.measurements(detections)
        pixels = self.image_centers(detections)
        if len(observed) != 6 or not np.isfinite(pixels).all():
            raise ValueError("Reference image must contain exactly six measured marker centers")
        order = np.lexsort((pixels[:, 1], pixels[:, 0]))
        self.last_uv = pixels[order].copy()
        self.visible = np.ones(6, dtype=bool)
        # Limit associations to less than half the initial inter-marker spacing.
        distances = np.linalg.norm(self.last_uv[:, None] - self.last_uv[None], axis=-1)
        np.fill_diagonal(distances, np.inf)
        self.gate = min(self.gate_px, .45 * distances.min())
        return observed[order]

    def update(self, detections):
        observed = self.measurements(detections)
        pixels = self.image_centers(detections)
        result = np.full((6, 3), np.nan)
        self.visible = np.zeros(6, dtype=bool)
        if len(observed):
            costs = np.linalg.norm(self.last_uv[:, None] - pixels[None], axis=-1)
            rows, columns = linear_sum_assignment(np.where(costs <= self.gate, costs, 1e6))
            for row, column in zip(rows, columns):
                if costs[row, column] <= self.gate:
                    result[row] = observed[column]
                    self.last_uv[row] = pixels[column]
                    self.visible[row] = True
        return result


class ArmStableEnv(gym.Env):
    """POMDP observation with separate actor/critic entries for SB3 DictRolloutBuffer."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None, task_config=None, episode_config=None, base_motion=None, visualize=False):
        super().__init__()
        settings = load_project_config(config)
        self.task = load_task_config(task_config)
        self.settings, self.reward_settings = self.task["environment"], self.task["reward"]
        episode = load_episode_config(settings if episode_config is None else episode_config)
        if episode["markers"]["count"] != 6:
            raise ValueError("ArmStableEnv requires six markers")
        self.model = make_model(episode)
        self.data = mujoco.MjData(self.model)
        self.base = FloatingBase(self.model, self.data)
        configure_base(self.base, episode["base"], base_motion or load_callback(self.settings["base_callback"]))
        self.joint_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 7)])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        self.ctrl_ids = np.array([self.model.actuator(f"joint{i}").id for i in range(1, 7)])
        self.ranges = self.model.jnt_range[self.joint_ids].copy()
        self.ee_id = self.model.body("ee_center_body").id
        vision, imu = load_vision_config(settings), load_imu_config(settings)
        share_device_calibration(vision, imu)
        if isinstance(self.base.motion, RandomBaseMotion):
            if self.settings["settle_s"] > BASE_HOLD_S:
                raise ValueError("settle_s must fit inside the random base's 0.5 s hold")
        self.imu = MujocoD435iIMU(self.model, config=imu)
        self.camera = D435iCamera(self.model, vision)
        self.color_from_depth = transform(self.camera.calibration["color_from_depth"])
        self.initializer = EpisodeInitializer(self.model, self.data, self.base, episode)
        self.slots = ImageMarkerSlots(self.settings["marker_gate_px"])
        self.joint_history, self.imu_history = deque(maxlen=5), deque(maxlen=5)
        self.action_space = spaces.Box(-1, 1, (6,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "policy": spaces.Box(-np.inf, np.inf, (ACTOR_DIM,), dtype=np.float32),
            "privileged": spaces.Box(-np.inf, np.inf, (PRIVILEGED_DIM,), dtype=np.float32),
        })
        self.visualize, self.handle = visualize, None
        self._ready = False
        self._sensor_frame = None
        self.peak_base_force_n = self.peak_base_torque_nm = 0.0

    def normalize_joints(self, angles):
        return np.clip(2 * (angles - self.ranges[:, 0]) / np.diff(self.ranges, axis=1).ravel() - 1, -1, 1)

    def denormalize_joints(self, action):
        return self.ranges[:, 0] + .5 * (action + 1) * np.diff(self.ranges, axis=1).ravel()

    def _physics_step(self, target):
        dt = self.model.opt.timestep
        current = self.data.ctrl[self.ctrl_ids]
        self.data.ctrl[self.ctrl_ids] = current + np.clip(target - current,
            -self.settings["max_joint_speed_rad_s"] * dt, self.settings["max_joint_speed_rad_s"] * dt)
        self.base.step()
        # Total constraint wrench at the floating base (includes the drive weld).
        wrench = self.data.qfrc_constraint[self.base.qvel_slice]
        self.peak_base_force_n = max(self.peak_base_force_n, float(np.linalg.norm(wrench[:3])))
        self.peak_base_torque_nm = max(self.peak_base_torque_nm, float(np.linalg.norm(wrench[3:])))
        samples = self.imu.sample(self.data)
        if samples:
            latest = samples[-1]
            # Specific force includes gravity; no true world orientation removal.
            self.latest_imu = np.r_[latest.angular_velocity_rad_s / self.settings["gyro_scale_rad_s"],
                                    latest.acceleration_m_s2 / self.settings["accel_scale_m_s2"]]

    def _capture(self):
        frame = self.camera.capture(self.data)
        if self._sensor_frame is not None and self._sensor_frame.sim_time == frame.sim_time:
            return self._sensor_frame, self._sensor_detections
        # Only fixed, calibrated RGB-to-depth extrinsics reach image processing.
        # Discard simulator world transforms before detection and slot matching.
        sensor_frame = replace(frame, world_from_optical=np.eye(4),
                               world_from_depth_optical=self.color_from_depth)
        self._sensor_frame = sensor_frame
        self._sensor_detections = detect_markers(sensor_frame, self.camera.config)
        return sensor_frame, self._sensor_detections

    def _ee_pose(self):
        body = self.data.body(self.ee_id)
        return body.xpos.copy(), body.xmat.reshape(3, 3).copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self._ready = False
        for attempt in range(10):
            self.episode_info = self.initializer.reset(self.np_random, sample_id=options.get("sample_id"),
                                                       marker_positions_m=options.get("marker_positions_m"))
            sensor_seed = int(self.np_random.integers(2**31))
            self.imu.reset(sensor_seed)
            self.camera.reset(sensor_seed)
            self._sensor_frame = None
            self.latest_imu = np.zeros(6)
            self.imu.sample(self.data)
            target = self.data.qpos[self.qpos_ids].copy()
            while self.data.time < self.settings["settle_s"] - 1e-10:
                self._physics_step(target)
            frame, detections = self._capture()
            measured = ImageMarkerSlots.measurements(detections)
            good_depth = np.isfinite(measured[:, 2]) & (measured[:, 2] > 0)
            if len(measured) == 6 and good_depth.any():
                break
            if options.get("marker_positions_m") is not None or attempt == 9:
                raise RuntimeError("Cannot measure six initial markers with any valid depth; "
                                   "use a nearer marker plane or check RGB-D calibration")
        self.current_xyz = self.slots.reset(detections)
        self.reference_xyz = self.current_xyz.copy()
        missing = ~np.isfinite(self.reference_xyz).all(axis=1)
        # Initial board is frontoparallel and coplanar: measured peer depths may
        # fill initial holes. Never use generated marker coordinates as observations.
        rays = pixel_rays(self.slots.last_uv[missing], frame.rgb_intrinsics,
                          frame.rgb_distortion, frame.rgb_distortion_model)
        self.reference_xyz[missing] = np.c_[rays, np.ones(missing.sum())] * np.median(self.reference_xyz[~missing, 2])
        self.marker_frame_time = frame.sim_time
        self.reference_position, self.reference_rotation = self._ee_pose()
        self.reference_points = rigid_points(self.reference_position, self.reference_rotation)
        self.prev_action = self.normalize_joints(target)
        self.older_action = self.prev_action.copy()
        joints = self.normalize_joints(self.data.qpos[self.qpos_ids])
        self.joint_history.clear()
        self.imu_history.clear()
        for _ in range(5):
            self.joint_history.append(joints.copy())
            self.imu_history.append(self.latest_imu.copy())
        self.start_time = self.next_control_time = float(self.data.time)
        self.lost_time = 0.0
        self._ready = True
        if self.visualize:
            if self.handle is None:
                import mujoco.viewer
                self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            configure_viewer(self.handle, self.data, self.episode_info)
        info = {"sample_id": self.episode_info["sample_id"], "initialization_attempts": attempt + 1,
                "initial_depth_filled_count": int(missing.sum()), "sim_time_s": float(self.data.time),
                "reference_ee_position_world_m": self.reference_position.tolist(),
                "reference_ee_rotation_world": self.reference_rotation.tolist(),
                "base_motion_seed": getattr(self.base.motion, "seed", None)}
        return self._observation(), info

    def _observation(self):
        current = self.current_xyz.copy() / self.settings["depth_scale_m"]
        # Without depth there is no metric X/Y. The complete slot is absent.
        current[~np.isfinite(current).all(axis=1)] = 0
        current = np.nan_to_num(current, nan=0, posinf=0, neginf=0)
        actor = np.concatenate((self.reference_xyz.ravel() / self.settings["depth_scale_m"],
                                current.ravel(), np.asarray(self.joint_history).ravel(),
                                np.asarray(self.imu_history).ravel(), self.prev_action)).astype(np.float32)
        position, rotation = self._ee_pose()
        relative_position = self.reference_rotation.T @ (position - self.reference_position)
        relative_rotation = self.reference_rotation.T @ rotation
        privileged = np.r_[relative_position / self.settings["ee_position_scale_m"],
                            relative_rotation[:, :2].T.ravel()].astype(np.float32)
        return {"policy": actor, "privileged": privileged}

    def step(self, action):
        if not self._ready:
            raise RuntimeError("Call reset() before step(), including after episode completion")
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must contain six finite normalized joint targets")
        action = np.clip(action, -1, 1)
        target = self.denormalize_joints(action)
        before = float(self.data.time)
        self.peak_base_force_n = self.peak_base_torque_nm = 0.0
        self.next_control_time += 1 / self.settings["control_hz"]
        while self.data.time < self.next_control_time - 1e-10:
            self._physics_step(target)
        dt = float(self.data.time) - before
        frame, detections = self._capture()
        visual_updated = frame.sim_time != self.marker_frame_time
        if visual_updated:
            self.current_xyz = self.slots.update(detections)
            self.marker_frame_time = frame.sim_time
        self.joint_history.append(self.normalize_joints(self.data.qpos[self.qpos_ids]))
        self.imu_history.append(self.latest_imu.copy())
        position, rotation = self._ee_pose()
        reward, terms = stabilization_reward(self.reference_xyz, self.current_xyz,
            self.reference_points, rigid_points(position, rotation), action, self.prev_action, self.older_action,
            self.data.qvel[self.qvel_ids], self.reward_settings)
        self.lost_time = self.lost_time + dt if not self.slots.visible.all() else 0.0
        terminated = bool(terms["rigid_rms_m"] > self.settings["failure_rms_m"]
                          or self.lost_time >= self.settings["lost_timeout_s"])
        truncated = bool(self.data.time - self.start_time >= self.settings["episode_duration_s"] - 1e-10)
        if terminated:
            reward -= self.reward_settings["failure_penalty"]
        self.older_action, self.prev_action = self.prev_action.copy(), action.copy()
        observation = self._observation()
        if not all(np.isfinite(value).all() for value in observation.values()) or not np.isfinite(reward):
            raise FloatingPointError("Non-finite physics state or sensor observation")
        self._ready = not (terminated or truncated)
        info = {"reward_terms": terms, "sim_time_s": float(self.data.time), "frame_time_s": frame.sim_time,
                "visible_markers": int(self.slots.visible.sum()),
                "valid_3d_markers": int(np.isfinite(self.current_xyz).all(axis=1).sum()),
                "visual_updated": visual_updated, "frame_age_s": float(self.data.time) - frame.sim_time,
                "applied_joint_targets_rad": self.data.ctrl[self.ctrl_ids].copy(),
                "ee_position_world_m": position.copy(), "ee_rotation_world": rotation.copy(),
                "base_constraint_force_peak_n": self.peak_base_force_n,
                "base_constraint_torque_peak_nm": self.peak_base_torque_nm,
                "ee_position_error_m": float(np.linalg.norm(position - self.reference_position)),
                "ee_rotation_error_rad": float(np.arccos(np.clip((np.trace(self.reference_rotation.T @ rotation)-1)/2, -1, 1))),
                "is_success": bool(truncated and not terminated and terms["rigid_rms_m"] < .02)}
        if self.handle is not None:
            self.handle.sync()
        return observation, reward, terminated, truncated, info

    def close(self):
        self.camera.close()
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class AsymmetricFeatures(BaseFeaturesExtractor):
    """Parameter-free packing; privileged values are sliced off BEFORE actor layers."""
    def __init__(self, observation_space):
        super().__init__(observation_space, ACTOR_DIM + PRIVILEGED_DIM)

    def forward(self, observations):
        return torch.cat((observations["policy"], observations["privileged"]), dim=-1)


class AsymmetricNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim_pi, self.latent_dim_vf = 134, 128
        def mlp(size):
            return nn.Sequential(nn.Linear(size, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU())
        self.actor = mlp(ACTOR_DIM)
        self.critic = mlp(ACTOR_DIM + PRIVILEGED_DIM)

    def forward_actor(self, features):
        actor_input = features[..., :ACTOR_DIM]
        # Encoder shortcut initializes the absolute-angle policy near a hold
        # command at every sampled pose, instead of pulling all joints to zero.
        joint_targets = torch.atanh(actor_input[..., 60:66].clamp(-.999, .999))
        return torch.cat((self.actor(actor_input), joint_targets), dim=-1)

    def forward_critic(self, features):
        return self.critic(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class AsymmetricPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if kwargs.get("use_sde", False):
            raise ValueError("This policy uses a tanh-squashed diagonal Gaussian, not gSDE")
        kwargs["features_extractor_class"] = AsymmetricFeatures
        kwargs.setdefault("log_std_init", -2.0)
        # SB3 builds the same Gaussian mean/log-std heads; its built-in squashed
        # distribution supplies the tanh Jacobian in BOTH rollout and update.
        kwargs["squash_output"] = False
        super().__init__(*args, **kwargs)
        self.action_dist = SquashedDiagGaussianDistribution(self.action_space.shape[0])
        self._squash_output = True

    def _build_mlp_extractor(self):
        self.mlp_extractor = AsymmetricNetwork().to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        with torch.no_grad():
            self.action_net.weight[:, -6:] = torch.eye(6, device=self.device)

    def predict_actor(self, actor_observation, deterministic=True):
        """Deployment inference needs only the 102 sensor values, never EE truth."""
        actor_observation = np.asarray(actor_observation, dtype=np.float32)
        if actor_observation.shape[-1] != ACTOR_DIM:
            raise ValueError("Actor observation must have 102 values")
        dummy = np.zeros((*actor_observation.shape[:-1], PRIVILEGED_DIM), dtype=np.float32)
        return self.predict({"policy": actor_observation, "privileged": dummy}, deterministic=deterministic)[0]


class ActorOnly(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.actor = deepcopy(policy.mlp_extractor.actor)
        self.action = deepcopy(policy.action_net)

    def forward(self, actor_observation):
        joints = torch.atanh(actor_observation[..., 60:66].clamp(-.999, .999))
        features = torch.cat((self.actor(actor_observation), joints), dim=-1)
        return torch.tanh(self.action(features))


class StabilityMetricsCallback(BaseCallback):
    """Log physical errors and reward components alongside PPO optimizer metrics."""
    def _on_rollout_start(self):
        self.metrics = {}

    def _on_step(self):
        for info in self.locals.get("infos", []):
            values = dict(info.get("reward_terms", {}))
            for name in ("ee_position_error_m", "ee_rotation_error_rad", "visible_markers",
                         "valid_3d_markers", "base_constraint_force_peak_n", "base_constraint_torque_peak_nm"):
                if name in info:
                    values[name] = info[name]
            for name, value in values.items():
                self.metrics.setdefault(name, []).append(value)
        return True

    def _on_rollout_end(self):
        for name, values in self.metrics.items():
            self.logger.record("stability/" + name, float(np.mean(values)))


def train_ppo(*, config=None, task_config=None, total_timesteps=1_000_000, n_envs=1,
              model_path="models/armstable_ppo.zip", seed=42, device="cpu"):
    from functools import partial
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    task = load_task_config(task_config)
    if task["ppo"]["n_steps"] * n_envs < 2:
        raise ValueError("PPO rollout must contain at least two transitions")
    if task["ppo"]["batch_size"] < 2:
        raise ValueError("PPO batch_size must be at least two")
    env = make_vec_env(partial(ArmStableEnv, config=config, task_config=task), n_envs=n_envs, seed=seed,
                       vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
                       vec_env_kwargs={"start_method": "spawn"} if n_envs > 1 else None)
    path = project_path(model_path)
    if path.suffix != ".zip":
        path = Path(str(path) + ".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = PPO(AsymmetricPolicy, env, seed=seed, device=device,
                    tensorboard_log=str(project_path("tensorboard/armstable")), **task["ppo"])
        model.learn(total_timesteps=total_timesteps, callback=StabilityMetricsCallback())
        model.save(path)
        with open(str(path) + ".config.json", "w", encoding="utf-8") as stream:
            json.dump({"project": load_project_config(config), "task": task,
                       "observation_format": OBSERVATION_FORMAT,
                       "joint_ranges_rad": env.get_attr("ranges")[0].tolist()}, stream, indent=2)
        return model
    finally:
        env.close()


def main(args):
    # Use a stable module name in saved policies and spawned worker factories.
    if __name__ == "__main__":
        from armstable_rl import main as run
        return run(args)
    model_path = project_path(args.model_path)
    if model_path.suffix != ".zip":
        model_path = Path(str(model_path) + ".zip")
    saved_config = {}
    sidecar = Path(str(model_path) + ".config.json")
    if args.mode in ("test", "export") and sidecar.exists():
        with open(sidecar, encoding="utf-8") as stream:
            saved_config = json.load(stream)
    if args.mode in ("test", "export") and saved_config.get("observation_format") != OBSERVATION_FORMAT:
        raise ValueError("This policy needs a marker_xyz_v1 config sidecar; "
                         "legacy (u,v,depth) policies must be retrained for metric XYZ observations")
    config = args.config if args.config is not None else saved_config.get("project")
    task = load_task_config(args.task_config if args.task_config is not None else saved_config.get("task"))
    for key in ("n_steps", "batch_size", "n_epochs"):
        if getattr(args, key) is not None:
            task["ppo"][key] = getattr(args, key)
    if args.mode == "train":
        train_ppo(config=config, task_config=task, total_timesteps=args.total_timesteps,
                  n_envs=args.n_envs, model_path=model_path, seed=args.seed, device=args.device)
        return
    if args.mode == "export":
        model = PPO.load(model_path, device="cpu")
        actor = torch.jit.script(ActorOnly(model.policy).cpu().eval())
        path = project_path(args.actor_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        actor.save(str(path))
        with open(str(path) + ".config.json", "w", encoding="utf-8") as stream:
            json.dump(saved_config or {"project": load_project_config(config), "task": task}, stream, indent=2)
        print(f"Exported actor: {path} (102 inputs, 6 normalized joint targets)")
        return
    env = ArmStableEnv(config=config, task_config=task, visualize=not args.headless)
    try:
        model = PPO.load(model_path, env=env, device=args.device) if args.mode == "test" else None
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed if episode == 0 else None)
            hold = env.prev_action.copy()
            total_reward = 0.0
            steps = 0
            while True:
                action = model.policy.predict_actor(obs["policy"]) if model is not None else hold
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated or (args.mode == "smoke" and steps >= args.steps):
                    break
            print(json.dumps({"episode": episode + 1, "steps": steps, "reward": total_reward,
                              "terminated": terminated, "truncated": truncated,
                              "visible_markers": info["visible_markers"],
                              "ee_position_error_m": info["ee_position_error_m"],
                              "ee_rotation_error_rad": info["ee_rotation_error_rad"]}))
    finally:
        env.close()


if __name__ == "__main__":
    main(_args)
