# D435i 硬件一致性审查（2026-09-10）

范围：项目 `Piper_rl` 的 MJCF、相机/IMU 数据路径、演示、浮动基座和 RL 入口。
同工作区的其他独立上游示例仓库不作为本项目运行入口，也未批量修改。
本次将已知的理想化实现改为有硬件约束的仿真；未取得实体相机及支架实测标定，
因此不能宣称已与某台 D435i 完全一致。

## 已修正

| 项目 | 当前行为 |
|---|---|
| RGB / 深度 profile | 分离尺寸；默认 1280×720 / 848×480，30 Hz；验证已实现 profile 组合 |
| 光学 | 分别设置 fx/fy，保持名义水平和垂直 FOV；可导入工厂主点与畸变 |
| 内部几何 | 50 mm 双目基线、RGB 与左深度 15 mm 偏移、非零 IMU 偏移 |
| 深度有效性 | 根据右相机视野与遮挡剔除不可匹配点，保留无效条带和孔洞 |
| 深度误差 | 可调视差噪声、1/32 pixel 视差步长、Z16 量化与零值无效 |
| RGB 对齐 | 从原生深度投影；保留源深度 Z，并另外提供 RGB 光轴 Z |
| 时间 | 视觉周期缓存；IMU 在相邻物理状态之间按各自时钟采样，不受 RGB 帧率限制 |
| IMU | BMI085 默认 400/200 Hz；可切 BMI055；量程、量化、白噪声、偏差和插值同步 |
| 重力 | 相机 75 g 载荷不再自动重力补偿；IMU 比力与去重力诊断输出分开 |
| 坐标 | IMU 原点名称改为 imu_optical，避免误认为加速度已换算到深度光心 |
| RL 入口 | 每个物理步更新 IMU；重置清空采样状态；修正 wxyz/xyzw 顺序，用末端轴而非欧拉角计算方向误差；超时改用仿真时间 |

## 参数依据与性质

厂商规格和参考库：

1. [D435i 产品规格](https://www.realsenseai.com/products/depth-camera-d435i/)：
   名义深度 FOV 87°×58°、RGB FOV 69°×42°，外形约 90×25×25 mm；深度为全局快门，RGB 为滚动快门。
2. [D400 系列数据表](https://dev.realsenseai.com/download/42003/)：
   USB3 profile、75 g 名义重量、IMU ±4 g / ±1000°/s 及采样率。
3. [官方 D435 URDF](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435.urdf.xacro)
   和 [IMU URDF](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435i_imu_modules.urdf.xacro)：
   内部安装名义值；官方同样指出应由设备标定外参替换。
4. [官方深度调优指南](https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/)：
   推荐 848×480，Min-Z 约 0.168 m；视差误差与纹理/场景相关。
5. [子像素深度说明](https://dev.realsenseai.com/docs/white-paper-subpixel-linearity-improvement-for-intel-realsense-depth-cameras/)、
   [SDK align 源码](https://github.com/realsenseai/librealsense/blob/master/src/proc/align.cpp)：深度量化与投影对齐依据。
6. [Bosch BMI085 数据表](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi085-ds001.pdf)、
   [Bosch BMI055 数据表存档](https://datasheet.lcsc.com/lcsc/1811071031_Bosch-Sensortec-BMI055_C189620.pdf)：芯片位数、灵敏度与典型噪声。
7. [SDK D435i 说明](https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md)、
   [官方 ROS IMU 同步](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_camera/src/base_realsense_node.cpp)：比力、坐标与异步线性插值。

默认 0.08 pixel 视差噪声、1 cm 双目遮挡一致性容差、IMU 等效白噪声带宽 fs/2
属于可重复的仿真近似，不能当成出厂噪声标定。当前零偏默认零，无温漂或随机游走。
规格中的采样率容差和时间戳精度没有擅自当成随机高斯抖动。

## 仍需实测或更专门的仿真

- **相机成像**：RGB 目前仍是瞬时快照，不模拟滚动快门逐行读出、曝光积分、
  自动曝光/白平衡、ISP、运动模糊。高速视觉算法不能据此认定实机同样无畸变。
- **主动双目**：几何可见性与视差误差近似不能复刻 IR 投影、材质的红外反射、
  透明/高反光物、弱纹理匹配失败和 D4 匹配/置信度/后处理。左右视图用于几何验证，
  不能冒充真实 Y8 红外测量。未捏造专有 ASIC 行为。
- **具体设备**：工厂内外参、深度比例、芯片批次、残余偏差需要实机读取；
  导出/加载接口已实现，但当前环境无 pyrealsense2，因此尚无实体设备验证。
- **支架与动力学**：link6 到外壳变换仍沿用示例值；外壳按盒体、惯量按均匀分布近似，
  未计入支架、USB 线缆质量和线缆拉力。应做手眼标定并测量负载质量/质心/惯量。
- **机器人与平台**：原机械臂仍采用项目的简化位置伺服、摩擦和重力补偿；浮动基座是
  约束跟踪，不是实测平台控制器。未凭空替换电机参数，也不能保证策略可直接上实机。
  固定道具及原 9 维 PPO 真值观测保持现有任务定义；视觉/IMU 输入策略需单独训练。

加载单机标定、测量支架外参和收集静止/运动 IMU 及不同距离的深度数据后，才能
进一步拟合噪声、偏差和动态误差。相关命令见 [视觉说明](d435i_vision.md) 与
[IMU 说明](../imu/README.md)。

## 验证方式

`python -m unittest discover -s config/tests -v` 检查物理量、时序和 EGL 渲染结果。
`python camera_demo.py --headless --frames 30` 生成 RGB、Z16、对齐深度及 IMU 日志。
修改负载重力和方向奖励会改变旧 PPO 策略的轨迹/回报，需重新评估已有模型。
这些检查证明实现内部一致，不能代替与实体 D435i 的对照实验。
