from config.cli import parse_cli
from config.settings import load_project_config, project_path, share_device_calibration

# Parse before importing MuJoCo so --headless and --help work without a display.
_cli_args = parse_cli("rl") if __name__ == "__main__" else None

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import warnings
import mujoco.viewer
import time
from typing import Optional
from config.flobase.piper_base import FloatingBase
from config.imu import MujocoD435iIMU
from config.episode import (EpisodeInitializer, configure_base, configure_viewer, make_model,
                            load_episode_config)

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")


class PandaObstacleEnv(gym.Env):
    def __init__(self, visualize: bool = False, base_motion=None, vision_config=None, imu_config=None,
                 episode_config=None, marker_generator=None, config=None):
        super(PandaObstacleEnv, self).__init__()
        self.visualize = visualize
        self.handle = None

        settings = load_project_config(config)
        self.episode_config = load_episode_config(settings if episode_config is None else episode_config)
        self.model = make_model(self.episode_config)
        self.data = mujoco.MjData(self.model)
        self.base = FloatingBase(self.model, self.data)
        configure_base(self.base, self.episode_config["base"], base_motion)
        self.arm_joint_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 7)])
        self.arm_qpos_ids = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_joint_ranges = self.model.jnt_range[self.arm_joint_ids].copy()
        self.arm_ctrl_ids = np.array([self.model.actuator(f"joint{i}").id for i in range(1, 7)])
        self.camera = None
        from config.vision.piper_vision import D435iCamera, load_vision_config
        from config.imu.simulation import load_imu_config
        self.vision_config = load_vision_config(settings if vision_config is None else vision_config)
        imu_settings = load_imu_config(settings if imu_config is None else imu_config)
        share_device_calibration(self.vision_config, imu_settings)
        self.imu = MujocoD435iIMU(self.model, config=imu_settings)
        self.imu_samples = []
        # Renderer allocation remains lazy, but camera intrinsics/extrinsics
        # must be configured before reset validates marker visibility.
        self.camera = D435iCamera(self.model, self.vision_config)
        self.initializer = EpisodeInitializer(self.model, self.data, self.base,
                                               self.episode_config, marker_generator)
        
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        self.initial_ee_pos = np.zeros(3, dtype=np.float32) 
        self.initial_joint_pos = np.zeros(6, dtype=np.float32)
        
        self.goal_size = 0.03
        
        # 约束工作空间
        self.workspace = {
            'x': [-0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.3]
        }
        
        # 动作空间与观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # 6轴关节角度、目标位置
        self.obs_size = 6 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
        
        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng(None)
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.goal_threshold = 0.005
        self.reset()
        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            configure_viewer(self.handle, self.data, self.episode_info)
            self._render_scene()

    def _get_valid_goal(self) -> np.ndarray:
        """Sample reach goals without an unbounded loop for arbitrary CSV poses."""
        candidates = self.np_random.uniform(
            low=[0.2, self.workspace['y'][0], 0.2],
            high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]],
            size=(1024, 3))
        distances = np.linalg.norm(candidates - self.initial_ee_pos, axis=1)
        valid = np.flatnonzero((distances > 0.4) & (distances < 0.5))
        # Some CSV poses cannot meet the old start-to-goal distance band;
        # retain the reach workspace and choose the nearest distance to 0.45 m.
        goal = candidates[valid[0] if len(valid) else np.argmin(np.abs(distances - 0.45))]
        base = self.data.body("base_link")
        return (base.xpos + base.xmat.reshape(3, 3) @ goal).astype(np.float32)

    def _render_scene(self) -> None:
        """渲染目标点"""
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 0
        total_geoms = 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染目标点（蓝色）
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_size, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),
            rgba=goal_rgba
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        options = options or {}
        self.episode_info = self.initializer.reset(
            self.np_random, sample_id=options.get("sample_id"),
            marker_positions_m=options.get("marker_positions_m"))
        self.initial_joint_pos = self.data.qpos[self.arm_qpos_ids].copy()
        self.prev_action[:] = 0
        self.imu.reset(seed)
        self.imu_samples = self.imu.sample(self.data)
        if self.camera is not None:
            self.camera.reset(seed)
        base = self.data.body("base_link")
        self.initial_ee_pos = base.xmat.reshape(3, 3).T @ (self.data.body(self.end_effector_id).xpos - base.xpos)
        
        # 生成目标
        self.goal = self._get_valid_goal()
        if self.visualize:
            self._render_scene()
            if self.handle is not None:
                configure_viewer(self.handle, self.data, self.episode_info)
        
        obs = self._get_observation()
        self.start_t = float(self.data.time)
        return obs, self.get_episode_info()

    def get_episode_info(self):
        """Initialization metadata and current physical base state for RL clients."""
        from copy import deepcopy
        info = deepcopy(self.episode_info)
        pose, velocity = self.base.get_pose(), self.base.get_velocity()
        info.update(base_position_m=pose.position_m.tolist(), base_quat_wxyz=pose.quat_wxyz.tolist(),
                    base_linear_velocity_m_s=velocity.linear_m_s.tolist(),
                    base_angular_velocity_rad_s=velocity.angular_rad_s.tolist())
        return info

    def _get_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.arm_qpos_ids].astype(np.float32)
        return np.concatenate([joint_pos, self.goal])

    def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[float, float, float]:
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)
        
        # 非线性距离奖励
        if dist_to_goal < self.goal_threshold:
            distance_reward = 100.0
        elif dist_to_goal < 2*self.goal_threshold:
            distance_reward = 50.0
        elif dist_to_goal < 3*self.goal_threshold:
            distance_reward = 10.0
        else:
            distance_reward = 1.0 / (1.0 + dist_to_goal)
        
        # 姿态约束：保持末端朝下
        target_orient = np.array([0, 0, -1])
        ee_orient_norm = ee_orient / np.linalg.norm(ee_orient)
        dot_product = np.dot(ee_orient_norm, target_orient)
        angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
        orientation_penalty = 0.3 * angle_error
        
        # 动作相关惩罚
        action_diff = action - self.prev_action
        smooth_penalty = 0.1 * np.linalg.norm(action_diff)
        contact_penalty = float(self.data.ncon)
        total_reward = distance_reward - contact_penalty - smooth_penalty - orientation_penalty

        # 更新上一步动作
        self.prev_action = action.copy()
        
        return total_reward, dist_to_goal, angle_error

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        # 动作缩放
        joint_ranges = self.arm_joint_ranges
        scaled_action = (joint_ranges[:, 0] + (np.asarray(action) + 1) * .5
                         * (joint_ranges[:, 1] - joint_ranges[:, 0])).astype(np.float32)

        # 执行动作
        self.data.ctrl[self.arm_ctrl_ids] = scaled_action
        self.base.step()
        self.imu_samples = self.imu.sample(self.data)
        
        # 计算奖励与状态
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        # The model already exposes the tool's world rotation matrix.
        ee_axis = self.data.body(self.end_effector_id).xmat.reshape(3, 3)[:, 2]
        reward, dist_to_goal,_ = self._calc_reward(ee_pos, ee_axis, self.data.qpos[self.arm_qpos_ids], action)
        terminated = False
        collision = False
        
        # 目标达成
        if dist_to_goal < self.goal_threshold:
            terminated = True

        if not terminated:
            if float(self.data.time) - self.start_t > 20.0:
                reward -= 10.0
                print(f"[超时] 时间过长，奖励减半")
                terminated = True

        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01) 
        
        obs = self._get_observation()
        info = {
            'is_success': terminated and (dist_to_goal < self.goal_threshold),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }
        info.update(self.get_episode_info())
        
        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def get_camera_observation(self):
        """按需获取 RGB、米制深度及颜色目标检测；不改变原 PPO 的 9 维观测。"""
        from config.vision.piper_vision import D435iCamera
        if self.camera is None:
            self.camera = D435iCamera(self.model, self.vision_config)
        frame = self.camera.capture(self.data)
        return frame, self.camera.detect(frame)

    def get_imu_observations(self):
        """本物理步新产生的同步 IMU 样本；尚未到采样时刻时为空列表。"""
        return list(self.imu_samples)

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = None,
    total_timesteps: int = None,
    model_save_path: str = None,
    visualize: bool = False,
    env_kwargs=None,
    seed: int = None,
) -> None:

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import torch
    import torch.nn as nn

    settings = load_project_config((env_kwargs or {}).get("config"))
    runtime, training = settings["cli"]["rl"], settings["training"]
    n_envs = runtime["n_envs"] if n_envs is None else n_envs
    total_timesteps = runtime["total_timesteps"] if total_timesteps is None else total_timesteps
    model_save_path = str(project_path(model_save_path or runtime["train_model_path"]))
    seed = training.pop("seed") if seed is None else seed
    training.pop("seed", None)
    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=training.pop("net_arch"))
    training["tensorboard_log"] = str(project_path(training["tensorboard_log"]))
    ENV_KWARGS = dict(env_kwargs or {}, visualize=visualize)

    # 创建多进程向量环境,生成多个并行环境的基类
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(**ENV_KWARGS),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    
    model = PPO(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs,
                device="cuda" if torch.cuda.is_available() else "cpu", **training)

    print(f"并行环境数: {n_envs}, 总步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = None,
    total_episodes: int = None,
    env_kwargs=None,
    visualize: bool = True,
    seed=None,
) -> None:
    from stable_baselines3 import PPO
    runtime = load_project_config((env_kwargs or {}).get("config"))["cli"]["rl"]
    model_path = str(project_path(model_path or runtime["test_model_path"]))
    total_episodes = runtime["episodes"] if total_episodes is None else total_episodes
    env = PandaObstacleEnv(**dict(env_kwargs or {}, visualize=visualize))
    model = PPO.load(model_path, env=env)

    success_count = 0
    print(f"测试轮数: {total_episodes}")
    
    for ep in range(total_episodes):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")
    
    success_rate = (success_count / total_episodes) * 100
    print(f"总成功率: {success_rate:.1f}%")
    
    env.close()


if __name__ == "__main__":
    import json
    args = _cli_args
    env_kwargs = {"config": args.settings}
    model_path = str(args.model_path)
    if args.mode == "train":
        train_ppo(n_envs=args.n_envs, total_timesteps=args.total_timesteps, model_save_path=model_path,
                  visualize=False, env_kwargs=env_kwargs, seed=args.seed)
    elif args.mode == "test":
        test_ppo(model_path=model_path, total_episodes=args.episodes, env_kwargs=env_kwargs,
                 visualize=not args.headless, seed=args.seed)
    else:
        env = PandaObstacleEnv(**env_kwargs, visualize=not args.headless)
        try:
            for episode in range(args.episodes):
                _, info = env.reset(seed=args.seed if episode == 0 else None)
                print(json.dumps(info, ensure_ascii=False))
                ranges = env.model.jnt_range[env.arm_joint_ids]
                hold_action = 2 * (env.initial_joint_pos - ranges[:, 0]) / np.diff(ranges, axis=1).ravel() - 1
                for _ in range(args.steps):
                    _, _, terminated, truncated, _ = env.step(hold_action)
                    if terminated or truncated:
                        break
        finally:
            env.close()
