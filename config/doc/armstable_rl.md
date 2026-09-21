# 移动基座下的末端稳定：armstable_rl.py

新脚本复用现有 MuJoCo、CSV 初始化、D435i RGB-D、IMU 和基座控制模块，
独立于 `piper_rl_mujoco.py` 的 reach 任务。目标是在基座运动时，保持 EE 的初始世界位姿。
默认基座回调为 `config.train_sets:base_pose`：前 0.5 s 静止，之后做平滑随机运动。
模型仍使用六个关节的位置执行器。

随机目标的 XYZ 相对初始基座位置分别限制在 ±0.1 m，roll/pitch/yaw 分别限制在
±5°。默认初始世界高度仍为 4 m，即 z 的目标范围为 3.9–4.1 m。
每段时长 8 s，用 minimum-jerk 五次时间缩放 `s(u)=10u³−15u⁴+6u⁵` 连接随机目标。
`s` 单调处于 [0,1]，不会越过端点；各段交接处速度、加速度均为零，jerk 有界。
每轮独立采样运动 seed，相同 reset seed 可复现；运动不依赖回调调用次数。

| 每轴参考量 | 平移上界 | RPY 分量上界 |
|---|---|---|
| 速度 | 0.046875 m/s | 2.34375 °/s |
| 加速度 | 0.018043 m/s² | 0.90211 °/s² |
| jerk | 0.023438 m/s³ | 1.171875 °/s³ |

这些界由最坏相邻端点差 0.2 m / 10° 及多项式导数的最大值直接得到。
角度导数是 RPY 分量导数，不是空间角速度的各分量。轨迹不逐帧叠加白噪声，
不会因目标位置跳变产生冲击；实际基座通过软 weld 跟随，有动态跟踪误差。
轨迹限加速度不能单独保证接触力/约束力上限，后者还取决于载荷、关节动作、
碰撞和约束参数。每个控制周期记录 `base_constraint_force_peak_n` 与
`base_constraint_torque_peak_nm`，它们是基座总约束广义力/力矩范数的峰值。
现有执行器力矩限制继续生效；本改动没有把软约束替换为会瞬移的硬裁剪。
默认 `episode.base.drive_time_constant_s=0.04`，驱动 weld 使用临界阻尼比 1。
它将原 4 ms 响应放缓到 40 ms，降低离散目标驱动造成的实际加速度尖峰，同时允许
毫米级跟踪滞后；该参数必须至少为两个物理步长。
参数含义见 [MuJoCo 约束求解参数](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)。

方法依据：[Modern Robotics 五次时间缩放](https://modernrobotics.northwestern.edu/nu-gm-book-resource/9-1-and-9-2-point-to-point-trajectories-part-2-of-2/)
及 [Ruckig 的速度、加速度、jerk 约束](https://docs.ruckig.com/)。这里使用解析五次轨迹，
没有引入 Ruckig 依赖，也不声称实现其时间最优在线算法。

## 算法与网络

使用项目已安装的 Stable-Baselines3 2.6 PPO：标准 rollout buffer、GAE、优势归一化、
clipped surrogate objective、价值损失、熵项、梯度裁剪和 KL 提前停止。
`spaces.Dict` 分别保存 `policy` 和 `privileged`，自定义 policy 实现非对称网络：

```text
102 维传感器观测 ── 256 ELU ── 256 ELU ── 128 ELU ── Gaussian(6) ── tanh ── 6 维动作

102 维传感器观测 + 9 维 EE 特权位姿 ── 256 ELU ── 256 ELU ── 128 ELU ── Linear(1) ── V
```

actor 与 critic 的可训练参数不共享。无参数 extractor 拼接输入后，actor 在第一层前
只取前 102 维；critic 使用全部 111 维。EE 信息只能影响价值估计和奖励。
使用 SB3 的 `SquashedDiagGaussianDistribution`，rollout 和 update 均含 tanh 的
log-probability Jacobian 修正；不是仅在环境中截断采样结果。初始 `log_std=-2`，
标准差随训练学习；熵用 SB3 支持的采样估计。

动作均值头还有一个当前编码器角度的快捷输入：128 维隐特征与当前六轴归一化角度的
atanh 值拼接后进入输出层，该快捷输入的权重初始化为单位阵。这样未训练策略的
确定性输出接近当前姿态，避免随机初始关节被直接拉到关节中点；最终动作仍是绝对
目标角。该输入完全来自策略已有的第 60:66 维，没有增加观测或使用特权信息。

这是通过五帧历史近似处理部分可观测性的前馈 PPO，没有添加循环网络或额外观测维度。
critic 的九维相对 EE 位姿也并非全部物理状态，因此这里的 V 是带特权信息的价值近似。

接口与算法设计参考：

- [SB3 自定义 policy、Dict 观测和独立网络](https://stable-baselines3.readthedocs.io/en/v2.6.0/guide/custom_policy.html)
- [SB3 PPO 参数与实现](https://stable-baselines3.readthedocs.io/en/v2.6.0/modules/ppo.html)
- [Isaac Lab 的非对称 Actor-Critic 与特权信息](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/transferring-robot-learning-policies-from-simulation-to-reality/05-bridging-the-gap-policy-robustness/01-leveraging-privelaged-information.html)

## 观测定义与数据来源

所有网络输入为 float32。策略的 102 维顺序固定为：

| 索引 | 内容 | 归一化与坐标系 |
|---|---|---|
| 0:18 | 初始 6 个 marker 的 `(X,Y,depth)` | 初始 RGB 光学坐标系，右/下/前，单位 m；全部除以 3 m |
| 18:36 | 当前 6 个 marker 的 `(X,Y,depth)` | 当前 RGB 光学坐标系，右/下/前，单位 m；全部除以 3 m |
| 36:66 | 5 帧 × 6 轴关节角 | 最旧到最新；按每个关节上下限线性映射到 `[-1,1]` |
| 66:96 | 5 帧 × 6 轴 IMU | 最旧到最新；每帧为 gyro xyz / 2 rad/s、accel xyz / 19.6133 m/s² |
| 96:102 | 上一控制周期的目标指令 | 六个归一化绝对关节角目标，速率限制前的策略命令 |

关节读数从 MuJoCo 编码器对应 qpos 读取，内部为弧度；归一化后与使用角度值归一化等价。
IMU 为 D435i IMU 光学坐标系的角速度和**包含重力的比力**，使用现有异步采样、噪声、
量化与同步模块。每个物理步采样，在每个控制周期记录最新同步读数，形成五帧历史。
不使用仿真世界姿态做 IMU 去重力。初始化历史以第一帧读数填满。

critic 的额外 9 维为相对初始 EE 坐标系的平移（除以 0.1 m），以及相对旋转矩阵的
前两列（按列连接，共 6 维）。该表示能表达全部旋转自由度，不含欧拉角跳变。
初始值为 `[0,0,0, 1,0,0, 0,1,0]`。
对于合法旋转矩阵 `R=[r1 r2 r3]`，第三列满足 `r3=r1×r2`，因此前两列足以重建
完整旋转。整个矩阵同样可用，但会多出三个冗余输入；6D 表示适合连续的网络输入，
参见 [Zhou 等的旋转表示论文](https://arxiv.org/abs/1812.07035)。这里只选 EE 位姿是
围绕稳定任务的最小特权设计，并不表示基座速度等状态没有价值；可在后续对比试验中
扩展 critic，而不能让这些仿真真值进入部署 actor。

`_ee_pose()` 直接读取 MuJoCo 中 `ee_center_body.xpos/xmat`，它们是物理仿真更新后的
实际世界位置与旋转，已经包含自由基座运动和机械臂关节运动。其运动学关系为
`T_world_ee = T_world_base * T_base_ee(q)`，不是把基座当作固定原点。
绝对位姿为 `(p_world, R_world)` 或对应 4×4 齐次变换；info 输出
`ee_position_world_m`、`ee_rotation_world`，reset info 保存相应参考位姿。
critic 输入 `R0.T*(p-p0)` 和 `R0.T*R` 的前两列；绝对世界位姿仍用于奖励。
实机若需要同样的绝对位姿，必须获得基座世界定位或相机相对固定 marker 地图的定位；
仅有六轴编码器不能观测自由基座的世界运动，IMU 积分也会漂移。

### marker 检测、固定槽位与缺失值

默认使用真实渲染的 RGB/深度检测结果；不会把生成器的 marker 真值投影后传给策略。
送进检测器的 frame 世界变换先被移除，仅保留固定 RGB-depth 标定外参。
不调用现有带世界位姿辅助匹配的 `MarkerTracker`。
该 tracker 会用 `frame.world_from_optical` 将旧世界点投影到当前画面，再按世界距离
辅助匹配。当前仿真中的这个变换来自 MuJoCo 真值；即使不把位姿直接送进网络，
由真值帮助建立的对应关系也会泄漏信息。若实机有可部署的视觉/惯性定位或外部定位，
可将其估计变换传给 tracker 来复用它；不能将移动的相机坐标系冒充固定世界系。
相机演示仍可用原 tracker，训练 actor 保留不依赖世界位姿的图像关联。

初始六个点按 `(u,v)` 排序，固定为六个槽位；这些槽位是视觉身份，不是九宫格编号
或 MuJoCo geom ID。后续通过上一帧位置、距离门限和 Hungarian 分配保持对应关系。
门限不超过初始最小点距的 45%，减少相邻 marker 互换。快速位移或长时间遮挡仍可能
导致丢失/关联歧义；没有用仿真真值纠正编号。像素坐标仅供检测、关联和绘图，
不作为 marker 位姿送入策略或奖励。球心来自检测器的 `center_position_optical_m`，
包含标定和畸变处理；无畸变时关系为 `X=(u-cx)*depth/fx`、`Y=(v-cy)*depth/fy`。
缺失槽位输出 `(0,0,0)`；深度缺失时无法求米制 X/Y，因此整个三维位置置零，
仍保留内部图像身份用于恢复匹配，不用旧深度拼成伪三维坐标。

reset 在基座静止阶段保持关节目标 0.5 s，于静止段结束时采集初始图像及 EE 参考。
初始必须检测到六个点，且至少一个球心深度有效。初始布局近似正对相机且共面，
所以仅在初始采集时，用有效球心深度的中位数补齐其余初始深度，再按像素射线反投影
得到 `(X,Y,depth)`；有效的单点深度保留自身测量。这个补齐会受深度噪声和初始机械
松弛影响，不是精确真值。info 中的 `initial_depth_filled_count` 记录补齐数。
初始测量不足时最多重新采样 10 次布局，次数也写入 info；明确传入 marker 坐标时
不会替换布局。自定义基座回调应在初始采集期间保持静止。

默认控制频率 **50 Hz**，物理时间步沿用模型（当前 0.002 s），每控制周期 10 个物理步。
图像与 marker 提取频率为 **25 Hz**，每 20 个物理步更新；相邻两次视觉更新之间
复用最新测量，检测器和槽位匹配均不重复运行。IMU 每物理步更新，关节/IMU 历史按
50 Hz 控制时刻记录。info 提供 `visual_updated`、`frame_time_s`、`frame_age_s`。
25 Hz 是此模拟器的采集/提取节奏；实机应从其支持的流配置按时间戳抽帧，不能假定
D435i 的所有 RGB/深度模式原生支持 25 Hz。

## 动作及平滑控制

策略给出六个 `[-1,1]` **绝对关节角目标**，不是角速度或关节增量：

`q_target = q_min + (a+1)/2 * (q_max-q_min)`

每个物理步对执行器目标角做速率限制，默认最大变化速度 0.8 rad/s。
info 的 `applied_joint_targets_rad` 为限制后实际写入执行器的目标。
控制器无碰撞规避或逆运动学前置器，训练结果需要通过评估验证。

## 奖励

1. **视觉位姿项**：固定身份的六个 marker 的米制 `(X,Y,depth)` 与初始值对比。
   每点分数为 `exp(-0.5*((ΔX/σxy)²+(ΔY/σxy)²+(Δdepth/σd)²))`，取六点均值。
   默认 `σxy=0.025 m`、`σd=0.05 m`。缺失或深度无效的点得分为 0；`missing_fraction`
   按三维测量无效的比例计算。没有继续使用像素误差或对数深度误差。

2. **EE 刚体项**：以 EE 为边长 80 mm 立方体的体心，取局部坐标
   `(0,0,0)` 和 `(±0.04,±0.04,+0.04)` 五点。将它们按真实 EE 位姿转到世界坐标，
   与各自初始世界位置比较：
   `r_rigid = mean(exp(-||p_i-p_i0||²/(2σ_rigid²)))`，默认 `σ_rigid=0.02 m`。
   该项同时感知平移与绕三轴的旋转，不只约束末端 Z 轴方向。

3. **平滑项**：六轴平均平方动作差、一阶差的变化、真实关节角速度，分别为
   `mean((a_t-a_t-1)²)`、`mean((a_t-2a_t-1+a_t-2)²)`、`mean(qdot²)`。
   动作差基于归一化目标，角速度单位 rad/s，当前控制频率为 50 Hz。
   前两项是离散动作差的惩罚，没有除以 dt，不能解释为物理速度/加速度。

总奖励为：

```text
r = 1.0*r_marker + 2.0*r_rigid
    - 0.05*action_rate - 0.02*action_acceleration
    - 0.002*joint_velocity - 0.2*missing_fraction
```

五点 RMS 位移超过 0.25 m 或任意 marker 的二维检测持续缺失 1 s 时失败，附加 -5 惩罚并
`terminated=True`。正常 10 s 时限返回 `truncated=True`，让 PPO 正确 bootstrap；
保持稳定不会提前结束。`is_success` 仅表示时限结束且最终 RMS < 20 mm，不等于
整段时间始终满足该误差；评估应同时查看误差曲线与各项均值。
TensorBoard 的 `stability/*` 输出位置/角度误差、五点 RMS、可见点数及奖励分项。

## 配置与运行

工作目录为 `Piper_rl`，使用已有 conda `piper` 环境，无需新增依赖。
`config/armstable.json` 独立保存本任务环境、奖励及 PPO 参数；现有
`config/settings.json` 继续管理场景、marker、RGB-D 与 IMU。
`--task-config` 支持只写覆盖字段的 JSON，`--config` 使用原统一配置格式。

```bash
# 固定关节目标，检查基座运动、观测及奖励；这不是训练策略
python armstable_rl.py --mode smoke --headless --episodes 1 --steps 90 --seed 7

# 短训练：验证完整 rollout / 更新 / 保存流程
python armstable_rl.py --mode train --headless --total-timesteps 32 \
  --n-steps 16 --batch-size 16 --n-epochs 1 --model-path /tmp/armstable_smoke.zip

# 正式训练：按渲染显存和 CPU 资源选择并行数
python armstable_rl.py --mode train --headless --n-envs 4 --total-timesteps 1000000

# 评估；去掉 --headless 可打开 MuJoCo 外部视图
python armstable_rl.py --mode test --headless --episodes 10 --model-path models/armstable_ppo.zip

# 导出只接受 102 维传感器观测的确定性 actor
python armstable_rl.py --mode export --model-path models/armstable_ppo.zip \
  --actor-path models/armstable_actor.pt

python -m unittest config.tests.test_armstable -v
```

MLP 默认在 CPU 训练；可用 `--device` 改变 PyTorch 设备。RGB-D 渲染仍使用 EGL，
多个环境采用 `spawn`，不复用父进程的 OpenGL context。
并行数默认 1，因为每个环境都要渲染 RGB 与双目深度，增加并行数会增加显存需求。
PPO 默认 1024 steps/env、batch 256、10 epochs、γ=0.99、GAE λ=0.95、clip=0.2。

模型保存到独立的 `models/armstable_ppo.zip`，旁边的 `.config.json` 保存完整任务、
传感器配置及关节限位。评估在未显式覆盖配置时读取该文件；导出 actor 同样附带它。
actor 导出为 TorchScript，输入形状 `(102,)` 或 `(batch,102)`，输出对应六轴目标。
实际部署仍需实现相同标定、参考采集、槽位匹配、归一化和动作速率限制。

短训练与测试只能证明程序及梯度链路可运行，不能证明策略已学会稳定控制。
默认已在各轮使用不同的随机基座运动；不同载荷和真实设备上的泛化仍需要评估。
尽管 actor 输入仍为 102 维，当前 marker 的含义已经从像素坐标变为米制 XYZ，
旧权重不兼容，必须重新训练。新模型 sidecar 标记 `observation_format=marker_xyz_v1`，
测试和导出入口会拒绝缺少该标记的旧模型，避免悄悄使用错误的输入定义。
