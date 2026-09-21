# PiPER 仿真改动、接口与原理说明

本说明覆盖本轮及之前的 CSV 初始化、marker、相机配置、基座运动、demo/RL 接入和
Qt 字体修复。以下列出相关生产模块的全部显式函数/方法，包括私有方法；测试函数
按验证项说明。未改动的 IMU 硬件驱动详见 [IMU 函数说明](piper_imu_code_guide.md)。

## 1. 当前行为与文件入口

| 文件 | 职责 |
|---|---|
| `camera_demo.py` | 交互/离屏 demo、R 重置、输出图像/深度/IMU/识别结果 |
| `piper_rl_mujoco.py` | Gym 环境、PPO 训练/测试、无策略 smoke 验证 |
| `config/settings.json` | 全部运行配置、CLI 默认值及 PPO 超参数 |
| `config/settings.py`、`config/cli.py` | 统一配置读取、合并、缓存和命令行解析 |
| `config/episode.py` | 统一模型构造、初始化、视锥/遮挡验证、固定基座高度和外部视角 |
| `config/train_sets.py` | 基座运动回调、默认九宫格 marker 和环形布局示例 |
| `config/flobase/piper_base.py` | 位姿/速度类型、文件轨迹、物理基座跟随 |
| `config/vision/piper_vision.py` | RGB-D 渲染、噪声、对齐、传统检测、任务检测入口 |
| `config/vision/markers.py` | 图像轮廓、多点深度球心估计、跨帧编号 |
| `config/vision/qt_fonts.py` | 修复 OpenCV wheel 的无效 Qt 字体目录 |
| `config/tests/` | 初始姿态、场景、视觉、基座、IMU 及 marker 回归 |

场景保留 `xml/agilex/scene.xml` 中原有的渐变天空、棋盘地板、灯光、绿色方块和蓝色球。
不修改原 XML 文件，任务模型在内存中添加 marker 和辅助平面。之前为避免遮挡而
删除背景和地板的方案已经撤销。原场景 `extent=0.4`、`zfar=10`，实际远裁剪距离
为 4 m；任务将 `zfar` 调为 100，即 40 m，避免天空盒影响远处 marker。

**保留全部 CSV 姿态，基座初始高度统一为 4 m。**
`episode.base.initial_height_m` 设置固定高度；不再计算机械臂/marker/平面的最低点。
`base_height_offset_m = initial_height_m - 初始基座命令z`，reset 后每个目标都加此偏移，
保持相对位置变化、方向和速度不变。初始命令为 z=0.1 时偏移为 3.9 m，实际起点为 4 m。
每轮从初始命令计算，不累积。`free_space` 仅控制地板显示/碰撞，不改变高度策略。
后续任意用户运动仍可能离开相机视野或接触地面。

## 2. 坐标、单位与主要数据

全部长度为 m，时间为仿真 s，关节/欧拉角为 rad，线速度为 m/s，角速度为 rad/s。
四元数采用 MuJoCo 的 `[w,x,y,z]`；RPY 对应 `Rz(yaw) Ry(pitch) Rx(roll)`。
基座速度回调的线速度和角速度都沿世界坐标轴。

CSV 的 `q1_rad…q6_rad` 是六个关节角；`camera_x/y/z_m` 是参考基座坐标系内
`d435i_mount` 的位置，`camera_zaxis_x/y/z` 是朝前的光轴。CSV 未提供完整旋转，
横向 X 轴由实际 RGB 相机的右方向补足，再构造右/下/前的正交相机坐标系。
RGB 和左右深度光心相对 mount 有偏移，不能直接把 CSV 原点当作 RGB 光心。

对基座旋转 `R_b`、位置 `t_b`、固定高度偏移 `o=[0,0,4-t_b(0).z]`：

```text
相机世界位置 = t_b + o + R_b @ camera_position_csv
相机世界前向 = normalize(R_b @ camera_zaxis_csv)
marker 世界坐标 = camera_origin_world + R_camera_world @ [x,y,d]
```

所有 marker 中心共用深度 `d∈[0.2,2.8]`，平面垂直于相机前向。这个距离是沿
CSV 相机 Z 轴的距离，不是世界高度或欧氏距离。球半径为 `r=0.0075`。
`marker_fovy=[H,V]` 默认 `[30,30]`，单位为度。CSV 生成窗口与实际 RGB
光心均按此角度检查完整球体；左右深度及 RGB 成像另按各自的内参检查。对 marker 窗口：

```text
|x| + r/cos(H/2) <= z*tan(H/2)
|y| + r/cos(V/2) <= z*tan(V/2)
```

实际成像视锥从 fx/fy、主点和传感器尺寸构造单位法向 n，要求 `n·球心 >= r`。
生成窗口超出相机时取可见区域交集，不拉伸图像或覆盖标定。
另检查 RGB、左深度、右深度中的角间隔，以及球心和四个边缘方向的遮挡射线。
射线检查降低遮挡风险，有限射线并不等价于对所有表面像素的可见性证明；最终用
实际 RGB/深度回归验证。marker 在一个 episode 中固定于世界坐标，运动后可能出画。

### 初始化信息 `info`

| 字段 | 类型/形状 | 含义 |
|---|---|---|
| `sample_id` | int | CSV 行标识 |
| `joint_angles_rad` | 6 | 初始关节角，同时作为初始位置控制目标 |
| `camera_position_world_m`、`camera_zaxis_world` | 各 3 | 初始化时的 CSV 相机世界位置/前向 |
| `world_from_csv_camera` | 4×4 | 初始化时 CSV 相机坐标到世界的变换 |
| `marker_positions_camera_m`、`marker_positions_world_m` | N×3 | 生成布局的真值；识别算法不读取这些字段 |
| `marker_diameter_m`、`marker_plane_depth_m` | float | 直径和平面深度 |
| `marker_fovy` | 2 | 生成窗口水平/垂直角，单位度 |
| `marker_plane_position_world_m`、`marker_plane_normal_world` | 各 3 | 辅助平面的中心和法向 |
| `marker_plane_half_extent_m` | 2 | 平面半宽/半高 `d*tan([H,V]/2)` |
| `base_height_offset_m` | float | 本 episode 的世界 Z 固定高度偏移 |
| `base_position_m`、`base_quat_wxyz` | 3、4 | RL info 中的当前实际基座位姿 |
| `base_linear_velocity_m_s`、`base_angular_velocity_rad_s` | 各 3 | RL info 中的当前实际世界速度 |

demo 的 `detections.json.initialization` 保存初始化字段；顶层另保存相机内外参、
当前基座状态和检测结果。RL 的 reset/step 都返回上述 info。相机初始化字段保持
初始值，当前相机变换请读取 `RGBDFrame.world_from_optical`。

## 3. 材质、透明平面与真实硬件的对应关系

marker 是直径 15 mm 的独立 mocap sphere，默认为高辨识度洋红色涂层，
`specular=0.9`、`shininess=0.8`、`emission=0`。配合头灯镜面光，它随光照产生
高光与明暗变化，不再靠自发光使球体恒亮。颜色、镜面强度和光泽均可配置。

MuJoCo 原生渲染采用固定管线 Phong 光照。这里是高反射涂层的外观近似，
**并未模拟真实逆反射膜的角度响应、近红外光谱或 D435i 的红外散斑匹配 ASIC**。
不能简单把球的 `reflectance` 设得很高来模拟逆反射：MuJoCo 原生镜面反射属性
主要用于平面和盒子。依据：[MuJoCo 材质/光照文档](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html)。

平面使用淡蓝 RGBA `[0.35,0.7,1,0.18]` 的薄盒子，厚度 0.4 mm，中心严格位于
marker 的共面深度。其法向与生成时相机前向一致，随 reset 更新，episode 内固定。
它是**虚拟观察辅助层**，`contype=conaffinity=0`，放入 geom group 4：

- 外部仿真窗口显示 group 4，因此能看到半透明平面。
- 相机 RGB/深度不渲染 group 4，避免薄平面覆盖球体、污染深度或改变识别。
- marker 使用 group 5，外部窗口和相机都显示。
- 球体也不参与碰撞；它们相当于固定安装的视觉靶标，不模拟自由落体。

真实 D435i 使用主动双目深度，反光过强、低纹理、饱和、遮挡都会影响深度。
官方深度质量规范依赖规定的目标、ROI、距离和配置，不能由这些仿真误差推出硬件
同等精度。实际部署需要真实标定、曝光/照明实验和测量验证。
参考：[RealSense 深度质量测试](https://www.realsenseai.com/wp-content/uploads/2019/11/RealSense_DepthQualityTesting.pdf)、
[D400 深度精度说明](https://support.realsenseai.com/hc/en-us/articles/360059129453-Depth-accuracy-for-Intel-RealSense-D400-Series-Cameras)。

## 4. 识别、球心估计与跨帧稳定性

1. RGB 转 HSV，提取可配置的涂层颜色。闭运算与轮廓填充补齐高光造成的小孔。
2. 通过面积、圆度、凸包填充度和长宽比过滤细长/非圆形干扰；轮廓矩或椭圆拟合
   提供亚像素中心。不会仅因边界像素恰好落在最外列而丢弃完整球体。
3. 将**原始深度像素**反投影，并用 RGB/深度外参转换到 RGB 光学坐标。
   仅取球轮廓内部的样本，每个原始像素使用一次；避免对齐图中重复像素及
   最近深度 splat 的近端偏差。远处小球只占少量像素，不能依赖单个深度值。
4. 已知物理半径 `r`，沿观测中心单位射线 `u` 估计球心 `C=t*u`。球面点 `P`
   给出前表面对应的中心距离：

   ```text
   t = P·u + sqrt(r² - (|P|² - (P·u)²))
   ```

   舍弃侧缘样本，利用中位数/MAD 去除深度异常值，再取多点中心距离中位数。
   同时用观测直径与已知物理尺寸做一致性检查。像素不足或不确定度过高时，保留
   2D 检测，3D 输出为 `null`，不从仿真真值或上一帧编造测量。
5. 跟踪器将上一帧世界球心投影到当前相机，用像素和世界距离门限构造代价矩阵，
   Hungarian 匹配后分配 `track_id`。EMA 只平滑实际测得的世界球心；原始估计仍保留。
   失踪轨迹仅短暂缓存供重关联，不输出虚假的“当前检测”。reset 清空编号和历史。

识别函数只接收图像、深度、内外参和已知球直径，不接收 MuJoCo geom ID、CSV
坐标、marker 真值或随机种子。轨迹编号不等于仿真 marker 编号，也不是跨遮挡
永久身份编码；长时间出画后会新建轨迹。

### 检测结果字典

| 字段 | 输出含义 |
|---|---|
| `label`、`bbox_xywh`、`center_uv`、`area_px` | 类别、像素包围框、亚像素中心、填充像素面积 |
| `circularity` | 轮廓圆度 `4πA/P²` |
| `center_position_optical_m`、`center_position_world_m` | 未平滑的测量球心；不可用时 null |
| `center_depth_m` | RGB 光轴上的球心深度，非欧氏距离 |
| `surface_point_optical_m`、`surface_point_world_m`、`depth_m` | 由拟合球心沿中心视线回退一个半径得到的前表面估计；传统颜色检测分支则返回实际内部像素的表面点 |
| `depth_pixel_uv` | 球心拟合分支为 null，因为结果来自多个原始深度像素；传统分支为单个像素 |
| `depth_sample_count` | MAD 筛选后用于估计的样本数 |
| `center_uncertainty_m` | 样本离散度推导的随机不确定度指标，不包含全部标定/系统误差 |
| `measurement_status` | measured / insufficient_depth / uncertain_depth |
| `diameter_m` | 球心拟合使用的已知物理直径 |
| `track_id` | 当前 episode 的视觉轨迹编号，仅 `camera.detect()` 增加 |
| `filtered_center_world_m` | EMA 滤波后的实际测量球心；本帧无有效 3D 时为 null |

`detect_targets(frame, config)` 是无状态入口；需要稳定编号时使用
`camera.detect(frame)`，demo 和 RL 相机接口已切换到后者。重复请求同一仿真时间的
相机帧不会重复更新滤波器。

## 5. 初始化模块：每个函数的输入、输出、作用和原理

未明确返回值的函数返回 `None`。配置/几何输入不合法时抛 `ValueError`，类型错误
通常抛 `TypeError`；不会静默截断角度或将非法布局改成另一个布局。

| 函数/方法 | 输入 | 输出 | 作用与原理 |
|---|---|---|---|
| `project_path(path)` | 路径字符串或 Path | 绝对/项目根目录下的 Path | 相对路径统一相对于 Piper_rl，避免启动目录影响配置 |
| `load_callback(value)` | callable 或 `模块:函数` 字符串 | callable | 通过 importlib 导入并检查可调用性 |
| `load_episode_config(value=None)` | None、episode 节字典、完整项目字典或统一 JSON 路径 | 完整 episode dict | 递归合并默认配置，校验直径、独立生成角度、深度范围、材质和固定高度 |
| `make_model(config)` | 完整 episode 配置 | MjModel | MjSpec 读取原场景，保留背景/道具，设置远裁剪和材质，按数量增加 mocap 球和平面，再编译 |
| `configure_base(base, settings, motion=None)` | FloatingBase、base 配置；可选旧接口路径/位姿函数 | None | 选择固定、文件、位姿回调或速度回调；旧 motion 参数优先 |
| `configure_viewer(viewer, data, info)` | 含 cam/opt 的 viewer、MjData、初始化 info | None | 用机械臂和四个平面角点的包围盒设置观察中心/距离，避免抬升后机械臂离开外部视角；显示 group 4/5 |
| `MarkerContext.to_world(points_m)` | N×3 的 CSV 相机坐标 | N×3 采样时世界坐标 | 乘固定初始高度下 context 的旋转并加位置，即最终世界坐标 |
| `generate_markers(context, rng)` | MarkerContext、NumPy Generator | N×3 共面坐标 | 限域随机采样，调用 accepts 检查完整球体视野/间隔/遮挡；拥挤时最多重排 20 次 |
| `EpisodeInitializer.__init__(model,data,base,config=None,marker_generator=None)` | 模型/数据/基座、配置、可选生成函数 | 初始化器实例 | 读取缓存 CSV，检查字段/数值/关节限位，缓存 sample_id 索引、关节/执行器/geom 下标和视锥常量 |
| `EpisodeInitializer.reset(rng,sample_id=None,marker_positions_m=None)` | 随机源；可选明确 CSV ID/本次布局 | 初始化 info | 随机选一行或严格选择指定行，再调用 `_reset_row`；不筛掉 CSV 行 |
| `EpisodeInitializer._reset_row(row,rng,marker_positions_m)` | 一行结构化 CSV、随机源、可选 N×3 布局 | 初始化 info | keyframe 清状态，角度和 ctrl 同步；一次性重置至固定 4 m；构造相机基，采样并放置 mocap 和矩形平面 |
| `EpisodeInitializer._accepts(candidate,previous)` | 候选 3 坐标、已接受点列表 | bool | 深度、CSV/三相机视锥、球体角间隔、地面和遮挡射线判定；平面/碰撞代理组不参与可见射线 |
| `_read_pose_table(path,modified_ns)` | CSV 绝对路径、修改时间 | 只读结构化数组 | LRU 缓存，多个环境复用解析结果，文件变化时失效 |
| `EpisodeInitializer._inside_sensor(point,camera_index)` | 相机光学点、RGB/左/右索引 | bool | 完整球体与真实针孔成像边界的有符号距离测试 |
| `EpisodeInitializer._validate_points(points)` | N×3 坐标 | None；不通过抛异常 | 检查有限数、数量、共面及逐点 accepts，应用于用户自定义布局；内置采样器已逐点验证，无需重复射线 |

`PlacementError` 是布局不可行时的 `ValueError` 子类。`MarkerContext` 自动生成的
构造器接收 `count, plane_depth_m, radius_m, half_extent_m, world_from_camera, accepts`，
以及可选的 `rgb_origin_m`（初始 RGB 光心在 CSV 相机坐标系中的位置）。
自定义函数必须返回数量一致的坐标；自定义过密布局仍可能无解，会明确报错。

固定高度仅需读取初始基座命令的 z；不遍历机械臂 geom，也不执行试探性平移。
`MarkerContext.half_extent_m` 现在是 `[半宽,半高]` 数组。矩形范围可直接传入
`rng.uniform(-context.half_extent_m, context.half_extent_m, 2)`；圆环半径取其最小值。

## 6. 基座模块：每个函数的输入、输出、作用和原理

| 函数/方法 | 输入 | 输出 | 作用与原理 |
|---|---|---|---|
| `_array(value,shape,name)` | 数组、期望 shape、错误字段名 | float 数组副本 | 检查形状与有限值，避免外部数组后续修改内部状态 |
| `BasePose.__post_init__()` | 构造后的 position_m(3)、quat_wxyz(4) | None；规范化字段 | 检查位置、非零四元数并归一化 |
| `BasePose.from_rpy(position_m,rpy_rad)` | 位置、roll/pitch/yaw | BasePose | 按 Rz Ry Rx 解析生成 wxyz 四元数 |
| `BaseVelocity.__post_init__()` | linear_m_s(3)、angular_rad_s(3) | None；规范化字段 | 校验世界坐标速度并复制数组 |
| `BaseTrajectory.__init__(time_s,position_m,quat_wxyz=None,rpy_rad=None)` | N 个递增时间、N×3 位置；两种方向表达二选一 | 轨迹对象 | 校验时间/维度，统一到归一化四元数 |
| `BaseTrajectory.load(path)` | JSON/CSV/NPZ | BaseTrajectory | 解析时间/位置/方向；NPZ 禁用 pickle |
| `BaseTrajectory.__call__(time_s)` | 仿真时间 | BasePose | 平移线性插值、旋转最短路径 SLERP；端点外保持端点 |
| `FloatingBase.__init__(model,data)` | 含 base_freejoint/base_target/base_drive 的模型与状态 | 控制器实例 | 缓存自由关节、mocap、weld 下标并初始化保持位姿 |
| `FloatingBase._apply(pose,teleport=False)` | BasePose、是否初始化传送 | None | 在命令位置上加 episode_offset_m，写 mocap；传送时同步 qpos/清速度和 warmstart，再 mj_forward |
| `FloatingBase.set_pose(position_m,quat_wxyz=None,rpy_rad=None,teleport=True)` | 绝对世界位姿 | None | 停止运动，清旧 episode 偏移，启用 weld 并保持；方向省略为单位旋转 |
| `FloatingBase.set_motion(motion)` | `t -> BasePose` | None | 验证 t=0 位姿、清旧偏移并初始化，按仿真时间驱动目标 |
| `FloatingBase.set_velocity_motion(motion,initial_pose=None)` | `t -> BaseVelocity`、可选起始 BasePose | None | 验证速度，从给定/当前位姿开始积分并清旧偏移 |
| `FloatingBase._velocity(value)` | 回调返回值 | BaseVelocity 副本 | 校验类型和有限数，避免把错误返回写入物理状态 |
| `FloatingBase.load(path)` | 轨迹文件路径 | None | `BaseTrajectory.load` 后作为 motion 设置 |
| `FloatingBase.set_episode_offset(translation_m)` | 世界坐标平移 3 向量 | None | 设置每个目标的附加平移并从 t=0 重置，供固定初始高度归一化使用 |
| `FloatingBase.reset()` | 无额外参数 | None | 保留模式与偏移，回到初始命令位姿；速度模式初始化正确的自由关节速度 |
| `FloatingBase.release()` | 无 | None | 关闭 weld，停止运动函数；后续按模型动力学自由演化 |
| `FloatingBase.get_pose()` | 无 | 实际 BasePose | 读取自由关节的实际世界位置/方向，而非理想目标 |
| `FloatingBase.get_velocity()` | 无 | 实际 BaseVelocity | 自由关节平移速度为世界坐标，旋转速度由局部轴转换到世界轴 |
| `FloatingBase.step()` | 无 | None；推进一个 timestep | 位姿模式采样目标；速度模式按中点积分平移、左乘世界角速度四元数指数；然后 mj_step/mj_forward |

`BasePose(position_m,quat_wxyz)` 与 `BaseVelocity(linear_m_s,angular_rad_s)` 的构造器
由 dataclass 生成。`base_target` 是目标，`base_link` 是有质量、惯量的真实自由基座，
通过 `base_drive` weld 跟随，存在动力学误差。不能把目标轨迹当作实际测量。

## 7. 相机模块：每个函数的输入、输出、作用和原理

### 通用投影、渲染和传统检测

| 函数/方法 | 输入 | 输出 | 作用与原理 |
|---|---|---|---|
| `load_vision_config(value=None)` | None、vision 节字典、项目字典或统一 JSON 路径 | 校验后的 vision dict | 从统一配置提取原 D435i 成像参数和检测/跟踪配置 |
| `validate_vision_config(config)` | 视觉配置 | None | 检查支持的 D435i profile；拒绝旧 fovy_deg 成像覆盖、深度窗口、噪声、检测/跟踪门限 |
| `camera_matrix(spec,width,height,fov)` | 标定内参或 None、分辨率、水平/垂直 FOV | 3×3 K | 标定存在时严格匹配分辨率，否则用尺寸/(2 tan(FOV/2)) 计算焦距 |
| `distortion(spec)` | 内参/畸变字典或 None | 5 元素畸变系数 | 验证受支持 Brown 模型，无标定返回零 |
| `deproject(depth,k)` | H×W 光轴深度、K | H×W×3 光学坐标 | 针孔反投影，不自动把光轴 Z 当作欧氏距离 |
| `_distort(xy,coeffs,model)` | 归一化坐标、5 系数、模型名 | 畸变归一化坐标 | 径向与切向 Brown 畸变计算 |
| `pixel_rays(pixels,k,coeffs,model)` | N×2 像素、K、畸变 | N×2 归一化射线 XY | 迭代逆畸变；Z 为 1；不收敛抛异常 |
| `project(points,k,coeffs=None,model=...)` | N×3 光学点、内参、可选畸变 | N×2 像素 | 除以 Z、施加畸变并乘内参；调用方保证 Z>0 |
| `align_depth_to_color(depth,depth_k,color_k,color_from_depth,color_shape,coeffs=None,distortion_model=...)` | 原始深度及深度→RGB 外参等 | `(aligned_source_Z, aligned_color_Z)` 两个图 | 投影像素 footprint，多个源落同一像素取最近深度；保留 NaN；分别保存源深度轴 Z 与 RGB 轴 Z |
| `D435iCamera.__init__(model,config=None)` | MjModel、视觉配置 | 相机对象 | 应用内部外参，设置 RGB/双目内参和渲染组；真正 OpenGL renderer 延迟到 capture 创建 |
| `D435iCamera.reset(seed=None)` | 可选传感器种子 | None | 清采样时钟、图像缓存和 marker tracker，重置噪声随机源 |
| `D435iCamera.detect(frame)` | RGBDFrame | 检测字典列表 | 每个新帧只检测一次，重复读取返回独立缓存副本；再进行可配置关联/滤波 |
| `D435iCamera.intrinsics(camera_id)` | 相机 ID | K 的副本 | 防止调用方修改内部内参 |
| `D435iCamera._pose(data,cid)` | MjData、相机 ID | 4×4 world_from_optical | MuJoCo 相机右/上/后转换为光学右/下/前 |
| `D435iCamera._depth(data,camera_id)` | MjData、深度相机 ID | H×W 理想光轴深度 | MuJoCo 深度渲染，复制结果，供双目可见性计算 |
| `D435iCamera.capture(data)` | 当前物理状态 | RGBDFrame | 按 fps 缓存；渲染 RGB/左右深度；仅对量程内像素做双目投影，加入视差噪声和量化；RGB 畸变映射按标定缓存、Z16 转换、RGB 对齐；未到新采样时刻返回旧帧 |
| `D435iCamera.close()` | 无 | None | 释放 RGB/depth renderer 和帧缓存 |
| `detect_targets(frame,config=None)` | RGBDFrame、可选视觉配置 | 检测列表 | 配有 marker_detection 时转专用球体检测；否则 HSV 连通域，保留原绿色/蓝色等通用颜色检测 |
| `annotate(frame,detections)` | RGBDFrame、检测列表 | BGR uint8 图像 | 绘制包围框、编号和球心深度；缺深度明确标注 invalid |
| `depth_preview(depth,config)` | 米制深度、深度显示范围 | BGR uint8 图像 | 归一化和色表，NaN 显示黑色 |

`RGBDFrame` 为 dataclass：包含 `rgb`、`native_depth_m`、`native_depth_z16`、
`aligned_depth_m`、`aligned_rgb_z_m`、`rgb_intrinsics`、`depth_intrinsics`、
`world_from_optical`、`world_from_depth_optical`、`sim_time`、`depth_scale_m`、
`rgb_distortion`、`rgb_distortion_model`、`calibration_source`。
`calibration_source` 描述所加载内部标定来源；没有真实设备标定时为名义几何。
默认 RGB 1280×720 / 69°×42°，深度 848×480 / 87°×58°，标定存在时以设备内参为准。
改变 `marker_fovy` 不改变这些参数，也不影响图像的分辨率和投影。

模拟深度噪声遵循 `Z=f*b/disparity`：先给视差加噪声、量化，再转换为深度和 Z16。
距离越远，相同视差误差会导致越大的深度误差；仅改变高光外观并不能消除它。

### 专用 marker 与 Qt 字体函数

| 函数/方法 | 输入 | 输出 | 作用与原理 |
|---|---|---|---|
| `native_points_in_color(frame)` | RGBDFrame | `(N×3 RGB光学点, N×2 RGB像素)` | 反投影每个有效原始深度像素，经外参/畸变映射到 RGB；不读取仿真真值 |
| `estimate_sphere_center(points,center_ray,radius_m,min_points=4)` | RGB 光学表面点、中心射线、已知半径、最少样本 | `(球心3向量, 不确定度, 样本数)` 或 None | 已知半径前表面约束 + 中位数/MAD 异常值剔除；侧缘/缺样本拒绝 |
| `detect_markers(frame,config)` | RGBDFrame、marker 检测配置 | 按像素位置排序的检测列表 | HSV、轮廓、椭圆中心、原始深度多点拟合、形状/尺寸/质量门限 |
| `MarkerTracker.__init__(config)` | 跟踪门限和 alpha | 跟踪器 | 复制配置并 reset |
| `MarkerTracker.reset()` | 无 | None | 清 track 字典、编号计数、时钟和缓存 |
| `MarkerTracker.update(detections,frame)` | 本帧检测和相机位姿/时间 | 带 track_id/滤波坐标的新列表 | 过期删除、投影匹配、Hungarian、EMA；空检测返回空列表，同时间返回缓存 |
| `configure_qt_fonts()` | 无；读当前进程环境 | 选中的字体目录 Path 或 None | 仅 Linux 生效；在 import cv2 之后、创建窗口之前检查 QT_QPA_FONTDIR，无效时选择系统 DejaVu/Liberation 等字体，不修改 conda 安装 |

OpenCV wheel 导入时会覆盖 `QT_QPA_FONTDIR`，所以只在 shell 中提前 export 往往
不能修复缺目录问题。模块现在在 cv2 导入后修正。字体路径修复不等于对所有桌面
OpenGL/驱动问题的保证；无窗口环境应使用 `--headless`。

## 8. demo、RL 与示例函数

| 函数/方法 | 输入 | 输出 | 作用与原理 |
|---|---|---|---|
| `camera_demo.main()` | 命令行 | None；生成输出文件 | 后端选择、统一配置、模型/相机/IMU/初始化器、物理步与采集循环、窗口和资源清理 |
| `camera_demo.main.reset_episode()` | 闭包中的 model/data/config/rng 等 | None；更新 episode_info | CSV 重采样、传感器重置、轨迹重播、清本轮 IMU，重新框选视图；R 或 reset-every 调用 |
| `PandaObstacleEnv.__init__(visualize=False,base_motion=None,vision_config=None,imu_config=None,episode_config=None,marker_generator=None,config=None)` | 显示开关、配置与回调 | Gym 环境 | 共用模型/初始化器、配置相机和 IMU、声明动作/观测空间，并初始化一次 |
| `PandaObstacleEnv._get_valid_goal()` | 环境 RNG、初始末端、工作空间 | float32 世界目标 3 向量 | 有界采样 reach 目标；优先距初始末端 0.4–0.5 m，无解时取最接近 0.45 m 的候选，避免无限循环 |
| `PandaObstacleEnv._render_scene()` | 当前 goal、viewer | None | 在外部窗口绘制原 reach 目标，与实际 marker 分离 |
| `PandaObstacleEnv.reset(seed=None,options=None)` | seed；可选 sample_id/marker_positions_m | `(obs,info)` | 重采样关节与 marker、恢复固定 4 m、同步 ctrl、清上一动作/相机/IMU、重播基座、采样 reach 目标 |
| `PandaObstacleEnv.get_episode_info()` | 当前环境状态 | 独立 info dict | 复制初始化真值并附加当前实际基座位姿和速度，防止外部修改内部记录 |
| `PandaObstacleEnv._get_observation()` | 当前状态 | float32(9,) | 六个关节角 + 三个世界 reach 目标坐标 |
| `PandaObstacleEnv._calc_reward(ee_pos,ee_orient,joint_angles,action)` | 末端位置/轴向、关节和动作 | `(reward,distance_to_goal,angle_error)` | 保留原 reach 距离奖励、接触惩罚、动作差与朝下姿态惩罚；更新 prev_action |
| `PandaObstacleEnv.step(action)` | 6 维 [-1,1] 动作 | `(obs,reward,terminated,truncated,info)` | 按关节范围缩放成位置目标，base.step 推进物理，采 IMU，计算奖励/成功/超时并附加 info |
| `PandaObstacleEnv.seed(seed=None)` | seed | `[seed]` | 兼容接口，替换环境随机源 |
| `PandaObstacleEnv.get_camera_observation()` | 无 | `(RGBDFrame,detections)` | 按需采集相机，通过 camera.detect 返回测量球心与轨迹编号 |
| `PandaObstacleEnv.get_imu_observations()` | 无 | 本物理步 IMU 样本列表 | 返回列表副本，尚未到采样时刻可能为空 |
| `PandaObstacleEnv.close()` | 无 | None | 关闭相机和外部窗口 |
| `train_ppo(n_envs=None,total_timesteps=None,model_save_path=None,visualize=False,env_kwargs=None,seed=None)` | 训练规模、路径、统一环境参数 | None；保存模型 | 从统一配置补全默认值，延迟导入 torch/PPO；SubprocVecEnv 独立 seed，学习并保存 |
| `test_ppo(model_path=None,total_episodes=None,env_kwargs=None,visualize=True,seed=None)` | 模型/轮数/环境配置 | None；输出成功率 | 加载策略，每轮调用统一 reset，再循环预测/step |
| `train_sets.base_pose(time_s)` | episode 时间 | BasePose | 前 0.5 s 静止，之后用 8 s 五次段连接随机 XYZ/RPY 目标，每轴限制 ±0.1 m/±5°，适用于 pose_callback |
| `train_sets.base_velocity(time_s)` | episode 时间 | BaseVelocity | 随时间变化的世界线速度/角速度示例，供积分 |
| `train_sets.gen_mkr4train(context,rng)` | MarkerContext、随机源 | 6×3 坐标 | 默认生成器：50 mm 九宫格的 1、2、5、6、8、9 号位置，5 号对准初始 RGB 光轴，平面内随机旋转 |
| `train_sets.marker_ring(context,rng)` | MarkerContext、随机源 | N×3 坐标 | 随机初始相位的均匀圆环，半径取可用半宽/半高较小值的 0.65；最终仍需统一合法性校验 |

RL 仍是原 9 维 reach 任务，不会自动把“达到目标”变成“视觉跟踪”。新数据通过 info
和相机接口提供；如训练视觉策略，需要另外设计观测空间和奖励。原策略尺寸兼容，
不代表在新初始姿态/浮动基座条件下仍有原成功率。

## 9. 配置与调用示例

```bash
conda activate piper
cd /home/armctrl/PiPER/Piper_rl

# 原背景地板 + 透明平面 + 反光球，R 重新初始化
python camera_demo.py --seed 7 --marker-depth 1.0 --frames 300

# 文件轨迹和用户速度回调
python camera_demo.py --base-motion config/flobase/base_motion.json
python camera_demo.py --base-velocity-callback config.train_sets:base_velocity

# 自定义排布函数和数量
python camera_demo.py --marker-count 8 --marker-depth 1.0 \
  --marker-generator config.train_sets:marker_ring

# RL 使用相同初始化和配置
python piper_rl_mujoco.py --mode smoke --headless --episodes 3 --seed 7
python piper_rl_mujoco.py --mode train --n-envs 12 --config config/settings.json
```

```python
from piper_rl_mujoco import PandaObstacleEnv
env = PandaObstacleEnv(episode_config={
    "markers": {
        "count": 6, "plane_depth_m": 1.0,
        "plane_visible": True, "plane_rgba": [0.35, 0.7, 1.0, 0.18],
        "rgba": [0.95, 0.04, 0.65, 1.0], "specular": 0.9, "shininess": 0.8,
    },
})
obs, info = env.reset(seed=7, options={"sample_id": 1})
frame, detections = env.get_camera_observation()
print(info["base_height_offset_m"])
for detection in detections:
    print(detection["track_id"], detection["center_position_world_m"], detection["measurement_status"])
env.close()
```

修改球颜色时同步调整 `settings.json` 中 `vision.marker_detection` 的 HSV 阈值。
默认宽视野下 2.8 m 的 15 mm 球只有约 5 像素 RGB 直径、2.4 像素深度直径；
无法保证每球都有 4 个有效深度样本。二维检测保留，三维不足时明确返回 null。平滑 alpha 越小，静态噪声越小，真实移动目标的跟随延迟越大。

## 10. 验证与结果文件

```bash
python -m unittest discover -s config/tests -v
python camera_demo.py --headless --frames 2 --seed 7
```

测试职责：`test_episode.py` 覆盖全部 1000 行 CSV、完整球体视锥、地面间隔、透明
平面位姿、固定高度偏移重播、0.2/2.8 m 实际图像和缺深度处理；`test_markers.py` 覆盖
多帧球心精度/稳定编号、无效深度不编造 3D、同色细长干扰、深度异常值抑制；
`test_base.py` 验证文件/位姿/速度动力学及 reset；`test_vision.py` 保留传统 profile、
标定、畸变、深度对齐和图像检测；`test_imu.py` 验证 IMU 处理、同步与采样。

`test_settings.py` 额外检查统一 JSON/CLI 覆盖、配置缓存隔离、训练输出路径、非法
角度、非方形 marker 窗口和检测缓存。使用同一布局、不同 marker_fovy 逐像素比较
RGB 和 Z16，验证生成参数不会改变成像。外部预览与测量记录位于
`config/outputs/vision/scene_overview.png`、`config/outputs/marker_validation.json`。
旧的 720×720/30° 检测统计不再适用于当前 D435i 成像配置。

## 11. 统一配置与命令行函数

四份 JSON 已合并为 `config/settings.json`。配置按功能分节，所有命令行参数只由
`config/cli.py` 注册和解析，入口不再做第二次 parse。`--config` 接受统一格式的覆盖
文件；原 `--episode-config` 和 `--imu-config` 改为该选项。覆盖顺序为默认文件、
用户 JSON、显式 CLI；字典递归合并，数组整体替换。

| 节 | 内容 |
|---|---|
| `episode` | CSV、球数量/直径/深度、marker_fovy、材质、平面、基座初始高度和运动 |
| `vision` | 原 D435i 分辨率/帧率/深度噪声/标定，以及 marker 检测/跟踪阈值 |
| `imu` | 型号、异步采样速率、噪声、量化、零偏和设备标定 |
| `cli.camera/rl/imu/calibration` | 对应入口的默认选项；生成器/基座等默认值直接取 episode，避免重复 |
| `training` | PPO 采样、批量、迭代、折扣、学习率、网络、seed 和日志目录 |

`cli.rl.model_path=null` 时，训练用 train_model_path、测试用 test_model_path，避免
默认训练覆盖已有测试模型。标定导出的分辨率默认直接引用 vision 节。

| 函数 | 输入 | 输出 | 作用/原理 |
|---|---|---|---|
| `settings.project_path(path)` | 字符串/Path | Path | 相对项目根目录解析路径 |
| `merge_settings(defaults,overrides)` | 两个字典 | 独立合并字典 | 递归合并，无可变对象共享 |
| `_read_settings(path,modified_ns)` | 路径、mtime | 缓存 JSON 字典（内部使用） | 按路径和修改时间缓存磁盘解析 |
| `load_project_config(value=None)` | None、统一 JSON 路径/完整字典 | 完整项目字典 | 默认值和覆盖递归合并，拒绝未知顶层节 |
| `load_settings_section(name,value=None)` | 节名称、路径/节字典/完整字典 | 节字典 | 底层唯一配置读取通道，功能模块保留可读的命名入口 |
| `share_device_calibration(vision,imu)` | 两个节字典 | None，原位更新路径 | 解析共同标定路径；两个不同设备文件时报错 |
| `cli.parse_cli(command,argv=None)` | camera/rl/imu/calibration、可选字符串列表 | Namespace，含 settings | 单次解析、合并默认值、覆盖 episode、验证计数并在 MuJoCo 导入前选 EGL |
| `load_imu_config(value=None)` | None、统一 JSON 路径/项目字典/imu 节 | IMU 字典 | 复用统一读取器；型号/速率仍由 MujocoD435iIMU 校验 |
| `imu.__main__.main()` | CLI | None，输出 JSON 行 | 中央解析器选硬件/仿真；仿真基座同样从固定高度启动 |
| `export_calibration(serial=None,depth_size=None,color_size=None,fps=None)` | 设备和流选项 | 标定字典 | 缺省流选项取 vision 节，查询 SDK 的真实内外参 |
| `export_calibration.main()` | CLI | None，写 JSON 文件 | 中央解析器处理路径、尺寸；传给标定导出函数 |

IMU 采样类其余函数的输入输出和误差原理见 [IMU 函数说明](piper_imu_code_guide.md)。
物理常数、硬件支持模式和算法公式仍保留在对应模块；运行时可调配置及 CLI/PPO 默认值
集中在 settings.json，避免把同一个默认相机尺寸或初始化高度写在多个入口中。

## 12. 执行时间与可读性

- reset 去掉临时抬升、逐 geom 高度计算和重复布局验证；固定 4 m 只重置一次基座。
- JSON 与 CSV 按路径/修改时间缓存；CSV 为只读数组，多个环境可复用，不共享可变配置。
- 预计算生成角度的三角函数、球体边距、传感器视锥和射线参数；sample_id 用字典查找。
- RGB 畸变 remap 只计算一次；双目匹配只投影深度窗口内的像素。
- 同一图像重复检测直接返回缓存副本；reset 清除检测/跟踪历史。
- torch/PPO 在 train/test 函数中加载，smoke、导入环境和 --help 无需加载训练框架。
- RL 去掉未计入奖励的无效计算、逐关节动作循环和多余四元数转换，沿用原奖励公式。

本机 conda piper 下，相同随机种子 73、连续 100 次初始化的初步比较：旧实现
448 ms，新实现 160 ms（约减少 64%）。这里只计 reset，不含模型构造和图像渲染；
布局约束与相机配置同时修正，数值是本次端到端对照，不是孤立的单函数加速比。

2026-09-18 当前配置的 12 个 CSV 姿态 × 3 个距离测量如下（216 个目标），完整数据见
`config/outputs/marker_validation.json`。只统计本帧成功测量的三维球心，不使用滤波结果：

| 平面距离 | 二维检出/目标数 | 有效三维球心 | 三维平均误差 | 三维最大误差 |
|---|---|---|---|---|
| 0.2 m | 72/72 | 72 | 0.089 mm | 0.150 mm |
| 1.0 m | 72/72 | 72 | 0.837 mm | 2.706 mm |
| 2.8 m | 72/72 | 9 | 9.514 mm | 21.386 mm |

二维完整可见与深度可测是不同条件；远端小球像素不足时保留二维识别，三维输出为空。
这些数值是当前模拟噪声、照明和标定下的回归记录，不代表真实硬件精度。
46 项回归全部通过；另外完成 demo 两帧两次 reset、IMU 采样和单环境 8 步 PPO
训练/保存验证。PPO 小规模验证仅检查配置与执行通路，不评估策略收敛性。
