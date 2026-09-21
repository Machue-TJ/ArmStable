# 传感器与浮动基座配置

运行以下命令时，工作目录为项目根目录 `Piper_rl`。

| 路径 | 用途 |
|---|---|
| `settings.json` | 唯一运行配置：episode、vision、imu、cli、training |
| `settings.py`、`cli.py` | 配置合并/缓存、所有命令行选项和解析 |
| `episode.py` | demo/RL 共用 CSV 初始化、marker 生成及基座模式 |
| `train_sets.py` | 基座运动回调和默认 50 mm 九宫格 marker 生成函数 |
| `vision/markers.py` | 球体图像检测、多点深度球心估计和跨帧跟踪 |
| `vision/qt_fonts.py` | OpenCV 导入后修正 Qt 字体目录 |
| `flobase/piper_base.py` | 基座位姿、轨迹加载与动力学控制 |
| `vision/piper_vision.py` | RGB-D 采集、坐标变换和目标识别 |
| `imu/` | [D435i IMU 处理、仿真与真实设备采集](imu/README.md) |
| `d435i.py` | 名义内部几何与单机标定加载/应用 |
| `flobase/base_motion.json` | 基座运动示例 |
| `vision/requirements-vision.txt` | 视觉依赖 |
| `doc/` | [相机说明](doc/d435i_vision.md)、[基座说明](doc/floating_base.md) |
| `tests/` | 基座、视觉与 IMU 回归测试 |
| `scripts/run_camera.sh` | conda piper 环境启动脚本 |
| `outputs/vision/` | 生成的图像、深度和检测结果 |

XML 独立放在项目根目录：`xml/agilex/` 存放机械臂模型和网格，
`xml/parts/` 存放相机与视觉目标配置。

两个入口的完整使用方式见 [CSV 初始化、marker 与基座运动](doc/episode_init.md)。
全部改动、各函数的输入输出与算法原理见 [完整函数说明](doc/simulation_functions.md)。

```bash
python camera_demo.py --headless --frames 1
python -m config.imu --backend sim --samples 10
python camera_demo.py --base-motion config/flobase/base_motion.json --frames 150
bash config/scripts/run_camera.sh --headless --frames 1
python -m unittest discover -s config/tests -v
```
