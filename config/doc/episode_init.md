# CSV 初始化、marker 与基座运动

`camera_demo.py`、`PandaObstacleEnv`、PPO 训练和测试共用
`config/settings.json` 的 `episode` 节 与 `config/episode.py`。
路径相对于项目根目录 `Piper_rl` 解析，也支持绝对路径。
自定义 JSON 只需填写要覆盖的字段，但必须保留顶层节名称；如
`{"episode":{"markers":{"count":4,"marker_fovy":[20,35]}}}`。
Python 的 `episode_config=` 则接受 episode 节字典；`config=` 接受完整字典或统一 JSON 路径。

所有 CLI 仅在 `config/cli.py` 中解析一次。`--config` 统一作用于 episode、vision、imu、
cli 和 training；显式 CLI 优先于指定 JSON，指定 JSON 递归覆盖默认配置。
旧的 `--episode-config`、`--imu-config` 已移除，统一使用 `--config`。

## 启动与重置

```bash
# 随机 CSV 姿态、6 个直径 15 mm 的球，保存 RGB、深度和初始化信息
python camera_demo.py --headless --frames 1 --seed 7

# 每 30 帧重新采样；窗口模式也可在 MuJoCo/OpenCV 窗口按 R
python camera_demo.py --frames 300 --reset-every 30

# 自定义数量及相机前方的平面距离
python camera_demo.py --headless --frames 1 --marker-count 10 --marker-depth 1.0

# 不加载策略，直接验证 RL 初始化与 reset
python piper_rl_mujoco.py --mode smoke --headless --episodes 3 --steps 10 --seed 7

python piper_rl_mujoco.py --mode train --n-envs 12 --config config/settings.json
python piper_rl_mujoco.py --mode test --episodes 15 --seed 7
```

每次启动和 `env.reset()` 都随机抽取 CSV 的一行，将 `q1_rad` 到 `q6_rad`
同时写入对应关节角度与位置控制目标。reset 清除物理状态、上一动作、相机缓存和
IMU 历史，并从 `t=0` 重播基座运动。相同 seed 可以复现；不传 seed 的后续 reset
继续随机序列。随机抽样允许偶尔抽到同一行。

CSV 的 `camera_x/y/z_m` 是参考基座坐标系中的 **d435i_mount 原点**，
`camera_zaxis_x/y/z` 是朝前的光轴。生成器先用 reset 后的基座位置和旋转将两者
转到世界坐标。CSV 未提供绕 Z 轴的角度，因此横向轴取模型 RGB 相机的右方向，
再构造正交的右/下/前坐标系。实际 RGB、左右深度光心与 mount 存在偏移；
生成器同时检查 CSV/RGB 光心的 marker 角度窗口，以及三个实际光心的成像视锥。

默认随机选取 `[0.2, 2.8]` m 内的一个 **沿 CSV 相机 Z 轴的距离**，全部球心处于
该深度的同一平面，平面垂直于 Z 轴。距离不是世界 Z 或欧氏距离。
球体半径固定为 0.0075 m，FOV 校验包含整个球体，并检查投影间隔及机械臂遮挡。
0.2 m 处的深度图表面值可能约为 0.184 m，这是光心偏移和球半径造成的正常差异。

`episode.markers.marker_fovy=[30,30]` 表示水平/垂直生成角度，二者可以不同。
例如 `--marker-fovy 20 35` 生成竖向较宽的区域；它不会改变相机内参、畸变或图像尺寸。
若指定范围超过相机实际可见范围，最终取二者及双目视野的交集；范围太小放不下
指定数量的完整球体时会报错。透明平面按两个角度分别计算宽和高。

成像参数继承原 vision_config 中的值，现存放于统一 JSON 的 `vision` 节：RGB
1280×720（名义 69°×42°）、深度 848×480（名义 87°×58°）、30 fps。加载真实
设备标定时以内参/畸变为准。相机外壳不出现在自己的图像中，外部窗口仍显示外壳。

默认 `free_space: false`，保留渐变天空、棋盘地板、灯光及彩色道具，增大远裁剪距离。
保留全部 CSV 姿态；`episode.base.initial_height_m=4.0` 为固定初始高度。
reset 使用 `offset_z = 4 - 初始命令的 z`，之后每个轨迹/回调目标都加相同偏移。
因此轨迹起点 z 为 0.1、下一点为 0.2 时，最终高度为 4.0、4.1 m；不计算机械臂
包围盒、不临时抬升、不累积偏移。`free_space: true` 只去掉地板，初始高度仍为 4 m。
球体采用非自发光的高反射彩色材质、无碰撞，通过独立 mocap body 放置，不增加关节自由度。
淡蓝透明平面在外部窗口显示，不参与碰撞，也不进入相机 RGB/深度。
**marker 在一次 episode 中固定于世界坐标；视野保证适用于生成时。**
随后机械臂或基座运动可能使其离开视野，系统不会自动把球移动回画面。

## 自定义数量、位置与生成函数

创建环境时指定数量；模型会按数量分配球体，后续 reset 复用它们。

```python
from piper_rl_mujoco import PandaObstacleEnv

env = PandaObstacleEnv(episode_config={
    "markers": {"count": 4, "plane_depth_m": 0.8},
})
obs, info = env.reset(seed=7)
print(info["sample_id"], info["marker_positions_world_m"])
frame, detections = env.get_camera_observation()
env.close()
```

`markers.positions_m` 可以直接提供 N×3 数组，坐标是以上 CSV 相机坐标系的
`[右, 下, 前]`，不是世界坐标。所有 Z 必须相同、位于深度范围内，且球体需要通过
实际视野、间隔和遮挡校验。`reset(options={"marker_positions_m": points})`
只覆盖本次的排布；`options={"sample_id": 3}` 可指定 CSV 行用于复现。

更灵活的接口为 `generator(context, rng) -> (N, 3)`：

```python
import numpy as np
from piper_rl_mujoco import PandaObstacleEnv

def my_markers(context, rng):
    points = []
    for _ in range(5000):
        point = np.r_[rng.uniform(-context.half_extent_m, context.half_extent_m, 2),
                      context.plane_depth_m]
        if context.accepts(point, points):
            points.append(point)
        if len(points) == context.count:
            return np.asarray(points)
    raise ValueError("当前数量、深度和间隔无法排布")

env = PandaObstacleEnv(
    episode_config={"markers": {"count": 8, "plane_depth_m": 1.0}},
    marker_generator=my_markers,
)
obs, info = env.reset(seed=7)
env.close()
```

`context` 包含 `count`、`plane_depth_m`、`radius_m`、`half_extent_m`、
`world_from_camera`，以及 `to_world(points)`、`accepts(candidate, previous)`。
`half_extent_m` 是 `[半宽,半高]` 的两元素数组；圆环半径可取其最小值。
context 的世界变换就是固定 4 m 初始高度下的变换，没有临时采样高度。
函数返回值始终是 CSV 相机局部坐标。
返回位置仍会统一验证；越界、重叠、被遮挡或非共面布局会明确报错。
默认生成器在近距离双目重叠区域内拥挤时会重新排布，次数有限，不会无限循环。

函数还可写入可导入模块，然后通过配置或 CLI 指定：

```bash
python camera_demo.py --headless --frames 1 --seed 7 --marker-depth 1.0 \
  --marker-generator config.motion_examples:marker_ring
```

JSON 中对应 `episode.markers.generator: "config.motion_examples:marker_ring"`。
`markers.min_gap_m` 控制球体投影之间的额外间隔；`edge_margin` 控制默认随机生成器
离视锥边缘的采样余量。过大的数量或间隔可能无法塞入指定平面。

## base_target 文件和自定义运动

两个脚本都支持以下选项：

```bash
python camera_demo.py --base-motion config/flobase/base_motion.json
python camera_demo.py --base-callback config.motion_examples:base_pose
python camera_demo.py --base-velocity-callback config.motion_examples:base_velocity \
  --base-pos 0 0 0.1 --base-rpy 0.2 0 0

python piper_rl_mujoco.py --mode smoke --headless --episodes 2 \
  --base-velocity-callback config.motion_examples:base_velocity
```

JSON 的 `base.mode` 可选 `fixed`、`trajectory`、`pose_callback`、`velocity_callback`。
`trajectory` 模式读取 `base.trajectory` 的 JSON/CSV/NPZ 文件，文件格式见
[基座说明](floating_base.md)。两种 callback 模式读取 `base.callback`。
所有回调参数均为从本次 reset 起算的仿真秒数；使用无内部累积状态的时间函数，
即可保证 reset 重播一致。

```python
import numpy as np
from config.flobase.piper_base import BasePose, BaseVelocity

def pose(t):
    # 世界坐标位置 m，roll/pitch/yaw rad；也可返回 BasePose(position, quat_wxyz)
    return BasePose.from_rpy([0.03 * np.sin(t), 0, 0.1], [0, 0, 0.1 * np.sin(t)])

def velocity(t):
    # 线速度 m/s、角速度 rad/s，均沿世界坐标轴
    return BaseVelocity([0.02 * np.cos(t), 0, 0], [0, 0, 0.05])

# 已创建的 env 也可以直接切换；下一次 reset 保留选择并从头开始
env.base.set_motion(pose)
env.base.set_velocity_motion(velocity, initial_pose=BasePose.from_rpy([0, 0, 0.1], [0, 0, 0]))
env.reset(seed=7)
```

速度模式从 `base.position_m` 和 `base.rpy_rad`（或 `quat_wxyz`）出发，按物理步
中点积分平移，用四元数指数积分世界角速度。位姿与速度模式都驱动 `base_target`
及原有 `base_drive` weld；真实 `base_link` 按动力学跟随，有约束跟踪误差。
`base.get_pose()` 和 `base.get_velocity()` 返回实际基座状态。

## RL 数据与训练配置

`train_ppo(env_kwargs={"config": "config/settings.json"})` 和
`test_ppo(env_kwargs={"config": "config/settings.json"})` 将统一配置传入每个环境。
PPO 超参数在 `training` 节；运行模式、轮数、并行数、总步数和模型路径在 `cli.rl` 节。
Python 也可将 callable 放进配置的 `base.callback`/`markers.generator`，或者使用
原有 `PandaObstacleEnv(base_motion=pose_or_path)` 接口。

reset/step 的 `info` 包含 CSV sample_id、初始关节角、相机初始位姿、marker 的
相机/世界坐标、直径与平面距离，以及当前基座位置、四元数、线速度和角速度。
demo 在 `detections.json` 的 `initialization` 中保存相同初始化信息。
当前还输出透明平面位姿/尺寸及基座高度偏移。图像识别采用颜色与圆形筛选、
原始深度多点球心估计及跨帧编号；`center_position_world_m` 为测量球心，
`filtered_center_world_m` 为滤波球心，`track_id` 是视觉轨迹编号而非仿真对象编号。
算法、各函数输入输出及物理模型限制见 [完整改动与函数说明](simulation_functions.md)。

PPO 的默认观测仍是 6 个关节角加 3 个 reach 目标坐标，保持原策略输入维度；
reach 目标与视觉 marker 独立。图像通过 `get_camera_observation()` 获取，
marker/基座状态通过 `info` 获取。当前奖励函数仍用于 reach 任务；若训练视觉跟踪
策略，需要另外定义对应观测与奖励。旧策略的输入尺寸兼容，性能需在随机初始姿态下重新评估。

```bash
python -m unittest discover -s config/tests -v
```

回归检查覆盖全部 1000 行 CSV、完整球体视锥约束、0.2/2.8 m 实际图像检测及缺失深度处理、
seed 复现、远离原点且旋转的基座、marker 世界位置保持、轨迹/速度重置及传感器重置。

恢复真实宽视野后，2.8 m 处 15 mm 球在默认深度图中只有约 2.4 像素直径。
二维可见不等于有足够的深度样本；默认要求至少 4 个有效深度样本，不足时返回
`insufficient_depth`，保留二维检测和编号，三维坐标为 null。
