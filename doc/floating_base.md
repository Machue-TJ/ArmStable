# MuJoCo 浮动基座

`xml/agilex_piper/piper.xml` 的 `base_link` 通过 `base_freejoint`
获得 3 个平移和 3 个旋转自由度，所有子连杆、夹爪和 D435i 相机随之运动。
`base_target` 是 mocap 位姿目标，`base_drive` weld 约束使基座跟随它。
默认目标位于世界原点、姿态为单位旋转，因此原演示可直接启动。

本次 Python 接口面向 MuJoCo。Genesis 示例也引用同一 MJCF，尚未适配或验证
Genesis 对 mocap/weld 的导入及新增基座自由度的控制；请使用以下 MuJoCo 入口。

## 运行

在 `Piper_rl` 目录、已安装 MuJoCo 的 Python 环境中：

```bash
conda activate piper
# 窗口显示整臂运动，同时显示腕部相机
python camera_demo.py --base-motion base_motion.json --frames 150
# 无窗口，按轨迹运动约 1 秒，保存最后一帧及实际基座位姿
python camera_demo.py --headless --base-motion base_motion.json --frames 31
# 指定静态位置和欧拉角
python camera_demo.py --headless --frames 1 --base-pos 0.1 0 0.2 --base-rpy 0 0 0.3
```

`--base-motion` 优先于 `--base-pos/--base-rpy`。输出的 `detections.json`
增加 `base_position_m`、`base_quat_wxyz`，相机世界外参随实际基座更新。
移动后目标可能离开视野，此时检测列表可以为空。

## Python 提供位姿或连续运动

```python
from pathlib import Path
import numpy as np
import mujoco
from piper_base import BasePose, FloatingBase

model = mujoco.MjModel.from_xml_path(str(Path("xml/agilex_piper/scene.xml").resolve()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
base = FloatingBase(model, data)

# 世界坐标系中的绝对位置（米）和 roll/pitch/yaw（弧度）。
base.set_pose([0.1, 0, 0.2], rpy_rad=[0, 0, 0.3])
# 也可使用四元数；顺序固定为 w,x,y,z，内部会归一化。
base.set_pose([0.1, 0, 0.2], quat_wxyz=[1, 0, 0, 0])

# 回调参数为从 set_motion/reset 开始计时的仿真秒数。
def motion(t):
    return BasePose.from_rpy(
        [0.05 * np.sin(t), 0, 0.2],
        [0.05 * np.sin(t), 0, 0.2 * np.sin(t)],
    )

base.set_motion(motion)
for _ in range(2000):
    data.ctrl[model.actuator("joint1").id] = 0.1
    base.step()  # 内含一次 mj_step 和世界变换刷新，不要重复 mj_step

actual = base.get_pose()
print(actual.position_m, actual.quat_wxyz)
```

欧拉角约定为 `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`，所有输入均为绝对世界位姿。
`set_pose` 默认立即移动整个基座、清零基座速度，保留机械臂相对关节角及速度；
适用于初始化或重定位，并停止已有轨迹。
连续在线目标更新可用 `set_pose(..., teleport=False)`，只改变约束目标。
省略旋转参数表示单位旋转。

`set_motion` 将基座初始化到回调 `t=0` 的位姿，后续每次 `base.step()`
在物理步开始时采样一次目标。关节仍按动力学演化；weld 是软约束，
实际位姿存在跟踪误差和延迟，不是逐帧强行覆写自由关节的位置。
急剧变化的目标可能产生较大约束力，示例使用缓慢的小幅运动。

直接操作 MuJoCo 重置时，需要重新初始化控制器：

```python
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
base.reset()  # 轨迹从 t=0 重播；静态模式恢复上次 set_pose
```

`base.release()` 关闭 weld，基座随后由动力学自由演化；再次 `set_pose`、
`set_motion` 或 `reset` 会重新开启约束。模型原有 `gravcomp=1` 保持不变，
所以 release 不等于关闭重力补偿。自由模式可继续调用 `base.step()`。

## 从文件加载

```python
base.load("base_motion.json")
# 或者在 Python 中直接构造同样的轨迹
from piper_base import BaseTrajectory
base.set_motion(BaseTrajectory(
    time_s=[0, 1, 2],
    position_m=[[0, 0, 0.1], [0.1, 0, 0.2], [0, 0, 0.1]],
    rpy_rad=[[0, 0, 0], [0, 0.1, 0.2], [0, 0, 0]],
))
```

JSON 和 NPZ 使用以下字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| `time_s` | N | 非负、严格递增的仿真时间，单位秒 |
| `position_m` | N×3 | 世界坐标 x/y/z，单位米 |
| `rpy_rad` | N×3 | roll/pitch/yaw，单位弧度 |
| `quat_wxyz` | N×4 | 四元数 w/x/y/z，与 `rpy_rad` 二选一 |

最小 JSON 示例：

```json
{
  "time_s": [0, 2],
  "position_m": [[0, 0, 0.1], [0.1, 0, 0.2]],
  "rpy_rad": [[0, 0, 0], [0, 0, 0.3]]
}
```

CSV 使用表头（平移米、旋转弧度）：

```csv
time_s,x,y,z,roll,pitch,yaw
0,0,0,0.1,0,0,0
2,0.1,0,0.2,0,0,0.3
```

也可把 `roll,pitch,yaw` 换为 `qw,qx,qy,qz`。NPZ 可用
`np.savez("motion.npz", time_s=times, position_m=positions, quat_wxyz=quats)` 保存。
禁止 pickle，所有数组需为有限数值；空轨迹、零四元数、重复/逆序时间会报错。

平移采用线性插值，旋转采用最短路径 SLERP；第一帧之前保持第一帧，
最后一帧之后保持最后一帧，不自动循环。单帧文件表示静态位姿。
要表示超过 180° 的连续旋转，应增加中间采样点。

## RL 入口与状态下标

```python
from piper_rl_mujoco import PandaObstacleEnv
env = PandaObstacleEnv(base_motion="base_motion.json")  # 也接受 BasePose 回调
obs, info = env.reset(seed=0)  # 自动重播基座轨迹
env.base.set_pose([0, 0, 0.2], rpy_rad=[0, 0, 0.1])
# env.step(action) 自动推进机械臂和基座
```

模型现在 `nq=15, nv=14, nu=7`。自由关节占 7 个 qpos（xyz+wxyz）、
6 个 qvel；六轴角度不再是 `qpos[:6]`。请通过关节名及
`jnt_qposadr/jnt_dofadr` 定位；MuJoCo RL 入口已改为按名称索引，
`home` 关键帧也已补齐基座状态。状态格式依据
[MuJoCo 自由关节与状态文档](https://mujoco.readthedocs.io/en/stable/computation/)。

reset 在基座局部工作空间内采样目标，再转换为世界坐标；目标在该 episode
内保持世界位置不变。9 维 PPO 观测维度保持兼容，但没有加入基座位姿/速度；
已有策略不能据此获得完整的移动基座状态，运动基座任务的训练需要另行扩展观测。
浮动基座和软约束也改变了动力学，已有策略不保证原性能。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试涵盖三种文件格式、时间和姿态校验、插值、仿真时钟、整臂与相机变换、
动力学跟踪、自由模式、重置，以及六轴控制和 EGL 视觉回归。
