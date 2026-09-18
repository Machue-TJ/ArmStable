# D435i IMU

所有命令在项目根目录 `Piper_rl`、已有的 `piper` Python 环境中运行。
模块分为独立数据处理、MuJoCo 仿真、真实设备采集三层。

各个类、函数、传感器原理及两份核心文件的调用关系详见 [IMU 源码说明](../doc/piper_imu_code_guide.md)。

## 型号与参考依据

D435i 早期配备 Bosch **BMI055** 六轴 IMU；Intel 的
[PCN 118035-00](https://cdrdv2-public.intel.com/802649/PCN118035-00.pdf)
说明后续批次替换为 **BMI085**。BMI055 加速度采样率为 62.5/250 Hz，
BMI085 为 100/200 Hz。旧 SDK 可能显示兼容采样率，不能仅凭商品名或速率
断定某台相机的芯片。真实设备入口查询 SDK motion profiles，默认选择各路最高
可用速率，并通过 `imu.info` 报告实际选择；不硬编码 BMI055 的寄存器比例系数。

设计参考：

- [librealsense D435i 文档](https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md)：
  motion 数据、光学坐标轴、设备标定及时间戳约定。
- [官方 realsense-ros 同步实现](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_camera/src/base_realsense_node.cpp)：
  参考 `FillImuData_LinearInterpolation` 的设计，将加速度插值到陀螺仪采样时刻。
- [SDK motion_frame 接口](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/hpp/rs_frame.hpp)：
  通过 `get_motion_data()` 读取已转换的浮点向量。
- [MuJoCo 传感器定义](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#sensor-accelerometer)：
  仿真使用 site 局部坐标下的 `gyro` 和 `accelerometer`。

## 输出约定

| 字段 | 含义 |
|---|---|
| `timestamp_s` | 采样时间，秒；真实 SDK 毫秒时间戳在入口转换一次 |
| `timestamp_domain` | SDK 返回的时钟域，或仿真的 `simulation`；不是默认 Unix 时间 |
| `frame_id` | `d435i_imu_optical`，原点为 IMU，X 向右、Y 向下、Z 向前 |
| `angular_velocity_rad_s` | `[wx, wy, wz]`，rad/s，右手规则；不是欧拉角变化率 |
| `acceleration_m_s2` | `[fx, fy, fz]`，m/s²，加速度计比力，保留重力效应 |
| `linear_acceleration_m_s2` | 已去重力的物理加速度，同一坐标轴；缺少外部姿态时为 `None` / JSON `null` |

水平静置、镜头朝前时，比力约为 `[0, -9.80665, 0]`，角速度约为零。
不能直接减去一个固定 Z 轴重力常数。定义 `R` 为光学系到世界系的旋转矩阵，
`g` 为世界系重力，则 `a_optical = f_optical + R.T @ g`。
真实 IMU 的比力在芯片位置测量：SDK 的轴对齐不等于补偿到深度光心的加速度。
本模块没有补偿芯片到光心的旋转杠杆臂，也不生成世界位姿、速度或位置。

## 仿真

```bash
python -m config.imu --backend sim --samples 10
```

每行输出一个 JSON 样本；按独立硬件采样时钟采样，不依赖 RGB 帧率。
默认 `config/settings.json` 的 `imu` 节 选择 BMI085：gyro 400 Hz、accel 200 Hz。
早期设备需修改 `model` 为 `BMI055`，并将 accel 设为 62.5 或 250 Hz。
两种型号 gyro 均可设为 200/400 Hz；量程分别为 ±4 g、±1000°/s。

```python
from config.imu import MujocoD435iIMU

imu = MujocoD435iIMU(model)
imu.sample(data)  # 注册初始状态；等待后置 accel 才能输出同步数据
for _ in range(100):
    base.step()
    for sample in imu.sample(data):  # 必须每个物理步调用，返回列表可能为空
        print(sample.to_dict())
```

`sample()` 在相邻物理状态间插值，以精确的各路时刻采样，然后加入噪声、零偏、
量化、饱和，最后经同步器和处理器输出。每路原始事件保留在 `last_motion_samples`。
`capture()` 只返回本次最新同步样本，或 `None`；需要保留所有数据时使用 `sample()`。
不推进仿真。物理步长必须不大于最快 IMU 周期，漏调时直接报错，避免把陈旧状态
伪造成高频数据。`reset(seed)` 重置全部时钟、缓冲、滤波和随机状态。

IMU 与深度原点的名义偏移为光学坐标下 `(0.00552, -0.0051, -0.01174)` m，
来源为官方 [IMU URDF](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435i_imu_modules.urdf.xacro)。
可用 `calibration_path` 加载实体设备导出的内部外参，详见
[硬件审查说明](../doc/d435i_hardware_audit.md)。仿真计算的是 IMU 位置的比力，包含
转动引起的偏心加速度，不能当成深度光心加速度。

仿真观测与真实硬件一样不自动提供姿态或去重力加速度。只有显式调用
`capture_truth(data)` 才使用仿真真值输出理想值和去重力结果，不应输入部署策略。
该诊断接口不影响采样、噪声和滤波状态。

噪声与量化默认开启，可设 `noise_enabled`、`quantization_enabled` 为 false 做诊断。
白噪声标准差采用 `density * sqrt(fs/2)`，并非实测固件滤波响应：
BMI055 accel 使用 150 µg/√Hz；BMI085 使用较大的 Z 轴典型值 135 µg/√Hz 作为各轴
近似；gyro 使用 0.014°/s/√Hz。参考 [Bosch BMI085 数据表](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi085-ds001.pdf)。
BMI055 accel 为 12 位，BMI085 accel 与两者 gyro 为 16 位。配置中的附加零偏默认零，
表示已标定情形，不伪造某台设备的残余零偏。未模拟温漂、随机游走、时钟抖动和固件内部滤波。

## 真实 D435i

```bash
python -m pip install -r config/imu/requirements-imu.txt
python -m config.imu --backend hardware --samples 100
python -m config.imu --backend hardware --serial YOUR_SERIAL --gyro-fps 400 --accel-fps 200
```

最后一条的加速度速率适用于报告 200 Hz profile 的设备；不支持时会列出可用速率。
设备名称、序列号、速率和 motion correction 状态写到 stderr，样本写到 stdout。
系统需要可用的 librealsense USB/HID 权限；此依赖是可选项，不影响纯数据处理和仿真。

```python
from config.imu import IMUProcessor, RealSenseD435iIMU

processor = IMUProcessor(cutoff_hz=30)  # 默认 None，不滤波
with RealSenseD435iIMU(processor=processor) as imu:
    print(imu.info)
    sample = imu.read(timeout_s=2)
    print(sample.to_dict())
```

采集器使用 SDK 底层 motion sensor 回调，不用视频 frameset 合并 IMU，
也不按帧编号拼接不同采样率的数据。它只占用 motion sensor；其他代码如果也打开
相同 motion sensor，需要改为共用回调，不能重复启动。多台 D435i 时必须指定序列号。
可写且受支持时开启 SDK motion correction；实际标定精度仍取决于设备中的标定数据。
不在 Python 中再次应用 SDK 内参、芯片单位缩放或原始芯片轴变换。

## 处理函数

- `IMUSynchronizer.push(stream, timestamp_s, xyz, timestamp_domain)`：输入 `gyro` 或
  `accel`，返回零个或多个同步 `IMUReading`。等待前后加速度样本，线性插值，
  输出使用陀螺仪时间戳，延迟通常约一个加速度采样周期。不外推；启动时没有前置
  加速度、超出历史或加速度间隔超过默认 0.1 秒的 gyro 会丢弃，并计入 `dropped_gyro`。
  默认每路最多保留 4096 个样本；停止时没有后置加速度的尾部 gyro 不输出。
- `IMUProcessor.process(...)`：减去用户提供的残余零偏，然后可选低通滤波。
  滤波系数使用实际 `dt`：`alpha = 1-exp(-2*pi*cutoff_hz*dt)`；默认间断超过
  0.5 秒时重新初始化滤波。每个对象只用于一路单调递增时间流。
- `estimate_stationary_bias(gyro_samples, accel_samples, expected_specific_force)`：
  用已确认静止的至少 20 个样本估计残余偏差。必须提供该姿态下预期比力，避免把
  重力误当成零偏。检查样本方差，但恒定转动/恒定加速度无法仅靠方差排除。
  完整的多姿态标定请使用 [SDK 校准工具](https://github.com/realsenseai/librealsense/tree/master/tools/rs-imu-calibration)。
- `remove_gravity(f, R, g)`：输入同一采样时刻的外部姿态（例如 VIO）及重力向量。
  六轴数据本身不能直接提供绝对姿态；运动中不应把归一化加速度视作精确重力方向。

非法数据、同一路重复或倒退时间戳、混合时钟域会报错。流重启后重置同步器和处理器；
`RealSenseD435iIMU.start()` 自动重置。USB 回调队列溢出会报错，需加快读取并重启，
不会静默吞掉丢帧。读取超时抛 `TimeoutError`，上下文管理器负责释放设备。

## 验证

```bash
python -m unittest discover -s config/tests -p test_imu.py -v
python -m unittest discover -s config/tests -v
```

覆盖采样率、噪声复现、量程饱和、漏采样检查、静止重力、自由落体、旋转偏心加速度、项目相机坐标对齐、零偏、滤波、
异步插值、时钟异常、缓存边界，以及模拟 SDK 的速率选择、单位、超时和资源释放。
模拟 SDK 测试不能替代接入实体 D435i 后的硬件验证。
