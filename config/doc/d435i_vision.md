# PiPER 末端 D435i 视觉

`camera_demo.py` 和 RL 使用同一套 D435i 成像参数，统一放在
`config/settings.json` 的 `vision` 节；`episode.markers.marker_fovy` 仅约束生成范围，
默认 `[30,30]`，不改变成像。CSV 初始化和运动接口见 [初始化与运动配置](episode_init.md)。

各类、函数及视觉处理原理的详细解析见 [piper_vision.py 源码说明](piper_vision_code_guide.md)。

在 `Piper_rl` 目录、conda `piper` 环境运行：

```bash
python camera_demo.py --headless --frames 30
python camera_demo.py --base-motion config/flobase/base_motion.json --headless --frames 150
```

默认结果在 `config/outputs/vision/`。图形模式省略 `--headless`，q/Esc 退出。

Linux 的 OpenCV wheel 可能把 `QT_QPA_FONTDIR` 指向不存在的 `cv2/qt/fonts`，
产生 `QFontDatabase: Cannot find font directory`。视觉模块会在导入 OpenCV 后
自动选择系统已有字体（例如 `/usr/share/fonts/truetype/dejavu`），无需修改 conda
目录或重新安装 Qt。这里必须在 `import cv2` 之后设置，因为 OpenCV 导入时会覆盖
该环境变量。字体警告本身不能说明 OpenGL 渲染是否正常；如果窗口仍为空白，应同时
检查运行终端的显示连接、后续错误，以及是否使用了 `--headless`。
视觉与 IMU 参数分别在 `config/settings.json` 的 `vision`、`imu` 节。
硬件依据、修正项和边界见 [硬件审查说明](d435i_hardware_audit.md)。

## 光学与深度

默认 RGB 1280×720、深度 848×480，均为 30 Hz USB3 配置。RGB 名义 FOV 为
69°×42°，深度为 87°×58°；fx/fy 分别计算并直接用于 MuJoCo 投影，不再仅用 fovy
强制方形像素。无单机标定时采用居中主点和零畸变，这不是某台相机的工厂内参。

仿真目前支持 RGB 1280×720 / 1920×1080，深度 848×480 / 1280×720，以及两路
相同的 6/15/30 Hz。超出这组已实现的模式会报错；这不是 D435i 的全部支持列表。
深度 848×480 的 Min-Z 下限设为 0.168 m，1280×720 为 0.28 m。`max_depth_m=3`
是应用裁剪窗口，不是硬件最大可见距离。渲染近远裁剪面同样不代表硬件量程。

左右光心间隔 50 mm，RGB 位于深度左侧 15 mm，均为官方名义外参。
深度来自左视图，经右视图视野及遮挡一致性检查，只保留两侧可观测的点。
这会产生距离相关的左侧无效条带及遮挡孔洞。之后在视差域加入误差，再量化成
Z16（默认每单位 1 mm，零值无效）。视差标准差默认 0.08 pixel 是可调仿真假设，
不是厂家保证值；1/32 pixel 视差步长近似 D4 子像素深度离散化。
原生 float32 米制深度将 Z16 的零值转换为 NaN。

对齐深度从原生深度反投影，经深度到 RGB 的外参和 RGB 内参投影，按像素覆盖
范围写入，冲突保留最近的原生深度。不会从 RGB 相机重新渲染一张完美深度图，
不会自动补齐缺失值。算法沿用 SDK 的投影/覆盖思路，边缘细节仍可能与 SDK 不同。

相机内部支持加载工厂 fx/fy/ppx/ppy、外参和 RGB Brown-Conrady 系列畸变；
SDK modified/inverse Brown 使用对应的前向模型，反投影数值求逆。
深度和右红外要求使用零畸变的已校正 profile。其他畸变模型或分辨率不匹配会报错。

## 输出与接口

| 输出 | 含义 |
|---|---|
| `rgb.png` / `detections.png` | RGB 图像 / 应用层 HSV 颜色检测 |
| `native_depth_z16.npy` | 原生 uint16 深度；零值无效 |
| `native_depth_m.npy` | 原生深度轴 Z，米；无效 NaN |
| `aligned_depth_m.npy` | 映射到 RGB 像素的原生深度轴 Z，沿用 SDK 复制源深度的语义 |
| `aligned_rgb_z_m.npy` | 相同像素对应的 RGB 光轴 Z，用于 RGB 反投影 |
| `imu.jsonl` | 两次视觉采集之间每个物理步累计的同步 IMU 样本 |
| `detections.json` | RGB/深度尺寸、内外参、深度比例、标定来源、检测结果 |

当标定中的两个光轴不平行或存在 Z 向平移时，`aligned_depth_m` 和
`aligned_rgb_z_m` 不能混用。检测使用后者及畸变反投影；`depth_m` 表示 RGB 轴 Z。
坐标为右手光学系 X 右、Y 下、Z 前。世界外参来自仿真真值，仅用于评估；
实机定位必须使用真实手眼标定/状态估计，不能由 IMU 或像素自动得到世界坐标。

```python
from config.vision.piper_vision import D435iCamera, detect_targets
camera = D435iCamera(model)
try:
    frame = camera.capture(data)
    detections = detect_targets(frame, camera.config)
finally:
    camera.close()
```

`capture()` 不推进仿真，在未到下个周期时返回上一帧；重复调用不会突破硬件帧率。
调用者至少按目标视频帧率取帧，若长时间不调用不会补造历史图像。实际采样时刻
记录在 `sim_time`，在当前 2 ms 物理网格上会有最多一个物理步的调度量化。
直接瞬移 qpos、重置到同一时间或改变场景后调用 `camera.reset(seed)`。
返回的缓存帧数组应视作只读，如需修改请先复制。时间倒退会自动重置。

`PandaObstacleEnv.get_camera_observation()` 返回帧和检测，
`get_imu_observations()` 返回本物理步的新 IMU 列表。PPO 仍使用原来的 9 维状态观测，
尚未训练成视觉/惯性策略。HSV 分类是 Python 应用功能，D435i 硬件不原生输出物体标签。

## 实机标定导出

```bash
python -m pip install -r config/imu/requirements-imu.txt
python -m config.vision.export_calibration --output config/vision/device_calibration.json
```

多机时添加 `--serial`，改变分辨率时添加 `--depth-size` / `--color-size`。
该命令只读取标定并保存本地 JSON，不写入相机 NVRAM。随后把统一 JSON 的 `vision` 和 `imu` 节中的
`calibration_path` 都设为 `config/vision/device_calibration.json`。demo/RL 中的配置路径相对项目根目录。
`camera_demo.py` 会将视觉标定路径同时用于 IMU。导出还记录固件、深度控制选项和
IMU 支持速率；记录的曝光/激光参数不是仿真中已实现的功能。

## 验证

```bash
python -m unittest discover -s config/tests -v
```

覆盖 EGL 图像、三维表面定位、非共光心外参、FOV、Z16、无效深度带、
源深度对齐、帧率、坐标转换、IMU 采样与动态传感器。
