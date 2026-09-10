# 传感器与浮动基座配置

运行以下命令时，工作目录为项目根目录 `Piper_rl`。

| 路径 | 用途 |
|---|---|
| `flobase/piper_base.py` | 基座位姿、轨迹加载与动力学控制 |
| `vision/piper_vision.py` | RGB-D 采集、坐标变换和目标识别 |
| `flobase/base_motion.json` | 基座运动示例 |
| `vision/vision_config.json` | 相机和检测参数 |
| `vision/requirements-vision.txt` | 视觉依赖 |
| `doc/` | [相机说明](doc/d435i_vision.md)、[基座说明](doc/floating_base.md) |
| `tests/` | 基座与视觉回归测试 |
| `scripts/run_camera.sh` | conda piper 环境启动脚本 |
| `outputs/vision/` | 生成的图像、深度和检测结果 |

XML 独立放在项目根目录：`xml/agilex/` 存放机械臂模型和网格，
`xml/parts/` 存放相机与视觉目标配置。

```bash
python camera_demo.py --headless --frames 1
python camera_demo.py --base-motion config/flobase/base_motion.json --frames 150
bash config/scripts/run_camera.sh --headless --frames 1
python -m unittest discover -s config/tests -v
```
