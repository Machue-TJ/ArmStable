import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
from pathlib import Path
from typing import Optional
from scipy.spatial.transform import Rotation as R
from config.flobase.piper_base import FloatingBase

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")


class PandaObstacleEnv(gym.Env):
    def __init__(self, visualize: bool = False, base_motion=None):
        super(PandaObstacleEnv, self).__init__()
        self.visualize = visualize
        self.handle = None

        scene_path = Path(__file__).resolve().parent / "xml" / "agilex" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.base = FloatingBase(self.model, self.data)
        if base_motion is not None:
            if callable(base_motion):
                self.base.set_motion(base_motion)
            else:
                self.base.load(base_motion)
        self.arm_joint_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 7)])
        self.arm_qpos_ids = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_ctrl_ids = np.array([self.model.actuator(f"joint{i}").id for i in range(1, 7)])
        self.camera = None
        
        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])
        
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        self.initial_ee_pos = np.zeros(3, dtype=np.float32) 
        self.home_joint_pos = np.array([  # home位姿
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ], dtype=np.float32)
        
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

    def _get_valid_goal(self) -> np.ndarray:
        """生成有效目标点"""
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            if 0.4 < np.linalg.norm(goal - self.initial_ee_pos) < 0.5 and goal[0] > 0.2 and goal[2] > 0.2:
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
        
        # 重置关节到home位姿
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_ids] = self.home_joint_pos
        self.base.reset()
        mujoco.mj_forward(self.model, self.data)
        base = self.data.body("base_link")
        self.initial_ee_pos = base.xmat.reshape(3, 3).T @ (self.data.body(self.end_effector_id).xpos - base.xpos)
        
        # 生成目标
        self.goal = self._get_valid_goal()
        if self.visualize:
            self._render_scene()        
        
        obs = self._get_observation()
        self.start_t = time.time()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.arm_qpos_ids].copy().astype(np.float32)
        # ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        # ee_quat = self.data.body(self.end_effector_id).xquat.copy().astype(np.float32)
        return np.concatenate([joint_pos, self.goal])

    def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
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
        action_magnitude_penalty = 0.05 * np.linalg.norm(action)

        contact_reward = 1.0*self.data.ncon
        
        # 关节角度限制惩罚
        joint_penalty = 0.0
        for i in range(6):
            min_angle, max_angle = self.model.jnt_range[self.arm_joint_ids[i]]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)
        
        # 时间惩罚
        time_penalty = 0.01
        
        # 总奖励计算
        # total_reward = distance_reward - contact_reward - smooth_penalty - action_magnitude_penalty - joint_penalty - time_penalty - orientation_penalty
        total_reward = distance_reward - contact_reward - smooth_penalty - orientation_penalty # - joint_penalty # - orientation_penalty
        # print(f"[奖励] 距离目标: {distance_reward:.3f}, [碰撞]: {contact_reward:.3f}, 动作惩罚: {smooth_penalty:.3f}, 姿态: {orientation_penalty:.3f}  总奖励: {total_reward:.3f}")
        
        # 更新上一步动作
        self.prev_action = action.copy()
        
        return total_reward, dist_to_goal, angle_error

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        # 动作缩放
        joint_ranges = self.model.jnt_range[self.arm_joint_ids]
        scaled_action = np.zeros(6, dtype=np.float32)
        for i in range(6):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        # 执行动作
        self.data.ctrl[self.arm_ctrl_ids] = scaled_action
        self.base.step()
        
        # 计算奖励与状态
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        rot = R.from_quat(ee_quat)
        ee_quat_euler_rad = rot.as_euler('xyz')
        reward, dist_to_goal,_ = self._calc_reward(ee_pos, ee_quat_euler_rad, self.data.qpos[self.arm_qpos_ids], action)
        terminated = False
        collision = False
        
        # 目标达成
        if dist_to_goal < self.goal_threshold:
            terminated = True
        # print(f"[奖励] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")

        if not terminated:
            if time.time() - self.start_t > 20.0:
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
        
        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def get_camera_observation(self):
        """按需获取 RGB、米制深度及颜色目标检测；不改变原 PPO 的 9 维观测。"""
        from config.vision.piper_vision import D435iCamera, detect_targets
        if self.camera is None:
            self.camera = D435iCamera(self.model)
        frame = self.camera.capture(self.data)
        return frame, detect_targets(frame, self.camera.config)

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 40_000_000,
    model_save_path: str = str(Path(__file__).resolve().parent / "models" / "piper_ppo_reach_target"),
    visualize: bool = False
) -> None:

    ENV_KWARGS = {'visualize': visualize}
    
    # 创建多进程向量环境,生成多个并行环境的基类
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(**ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    
    # 策略网络配置
    POLICY_KWARGS = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )
    
    # PPO模型
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        n_steps=2048,          
        batch_size=2048,       
        n_epochs=10,           
        gamma=0.99,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tensorboard/piper_reach_target/"
    )
    
    print(f"并行环境数: {n_envs}, 总步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = str(Path(__file__).resolve().parent / "models" / "piper_ppo_reach_target"),
    total_episodes: int = 5,
) -> None:
    env = PandaObstacleEnv(visualize=True)
    model = PPO.load(model_path, env=env)
    
    record_gif = False
    frames = [] if record_gif else None
    render_scene = None  
    render_context = None 
    pixel_buffer = None 
    viewport = None
    
    success_count = 0
    print(f"测试轮数: {total_episodes}")
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
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
    TRAIN_MODE = False
    model_name = "piper_train_reach_target" if TRAIN_MODE else "piper_ppo_reach_target"
    MODEL_PATH = str(Path(__file__).resolve().parent / "models" / model_name)
    if TRAIN_MODE:
        train_ppo(
            # n_envs=64,                
            n_envs=12,                
            total_timesteps=4_000_000,
            model_save_path=MODEL_PATH,
            visualize = False
        )
    else:
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=15,
        )
