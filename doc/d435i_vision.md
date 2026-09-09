# PiPER 末端 D435i 仿真与目标识别

## 启动

本机 `conda piper` 已有 Python 3.10、MuJoCo 3.3.2、OpenCV 5.0.0.93、
NumPy 2.2.6。离屏采集使用 EGL，无需 USB 相机或 RealSense SDK。

```bash
conda activate piper
cd /home/armctrl/PiPER/Piper_rl
# 离屏采集并保存最后一帧；默认输出 outputs/vision/
python camera_demo.py --headless --frames 1
# 桌面模式：机械臂场景、RGB 检测窗口、对齐深度窗口；q 或 Esc 退出
python camera_demo.py --frames 300
# 无需激活环境的等价入口
bash scripts/run_camera.sh --headless --frames 30
```

程序默认恢复 XML 的 `home` 关键帧，两个目标在此姿态下可见。桌面模式需要
可用的 DISPLAY；不要在该模式设置 `MUJOCO_GL=egl`，可先 `unset MUJOCO_GL`。
只有离屏采集模式在本次配置中自动验证。可用 `--config /path/config.json`
覆盖图像和检测配置，`--output /path/output` 修改输出目录。

新环境安装视觉部分：`python -m pip install -r requirements-vision.txt`。
不要同时安装 opencv-python、opencv-python-headless 和 opencv-contrib-python。
原 RL/Genesis 的其他依赖仍见 requirements.txt。

## 参数位置

| 文件 | 内容 |
|---|---|
| `xml/agilex_piper/d435i.xml` | 相对 link6 的安装外参、外壳、载荷质量/惯量、相机 FOV |
| `xml/agilex_piper/piper.xml` | 在 link6 内 include 相机，固定随腕部运动 |
| `xml/agilex_piper/vision_targets.xml` | 目标形状、颜色、世界位置和碰撞参数 |
| `xml/agilex_piper/scene.xml` | 引入真实可渲染目标，离屏缓冲区尺寸和裁剪参数 |
| `vision_config.json` | 分辨率、帧率、有效深度范围、HSV 阈值、最小连通区域 |
| `piper_vision.py` | RGB/深度采集、内参、光学坐标变换、颜色检测 |
| `camera_demo.py` | 30 Hz 采集示例、窗口显示和文件输出 |

外壳 90 × 25 × 25 mm；相对 link6 平移 `(0, -0.065, 0.035)` m，
四元数按 MuJoCo `wxyz` 顺序为 `(0, 0.8892927216, 0, -0.4573384472)`。
视线朝 link6 的 +Z 并向 +X 倾斜 0.95 rad。光心位于相机 body 的
`(0, 0, -0.014)` m，略在外壳前方；相机不是自动追踪目标的相机。
75 g 为仿真载荷假设，惯量按均匀长方体计算，未计入支架重量。
开启 `gravcomp=1` 与现有机器人设置一致；仍会增加腕部运动惯性和碰撞几何。
安装偏移是示例支架设计，并非实测手眼标定结果。

RGB 垂直 FOV 为 42°，深度垂直 FOV 为 58°；`fovy` 始终使用度，
不受 `compiler angle="radian"` 影响。分辨率为 640 × 360、30 Hz，
按方形像素针孔模型推导水平 FOV（因此不保证精确复现规格表的水平 FOV）。
0.28～3 m 为保守的示例有效深度窗口；不模拟随分辨率变化的近距性能。
`scene.xml` 的 znear/zfar 是相对模型 extent 的渲染裁剪参数，不是上述传感器量程。

规格参考：[D435i 产品页](https://www.realsenseai.com/products/depth-camera-d435i/)。
本模型是理想 RGB-D 近似，没有双目基线、红外投影、畸变、IMU、噪声、
运动模糊、曝光和真实深度缺失机制。RGB 与原生深度相机共光心，视场不同。

## 目标与识别边界

- 绿色方块：中心 `(0.59, -0.105, 0.03)` m，边长 0.06 m。
- 蓝色球：中心 `(0.66, -0.02, 0.03)` m，半径 0.03 m。

目标是固定、有碰撞的识别道具，当前不能抓起。机器人已增加浮动基座，
当前为 `nq=15, nv=14, nu=7`；关键帧和 MuJoCo 控制入口已适配。
基座的平移与旋转会带动腕部相机，视觉定位继续使用当前世界外参。
轨迹播放及 Python 接口见 [浮动基座说明](floating_base.md)。
物体增加了场景碰撞，相机增加了载荷，因此原策略的动力学表现可能变化。

识别采用 OpenCV HSV 分割、形态学去噪和连通域分析。标签通过示例物体的
唯一颜色约定，不是通用物体/形状识别，也不依赖 MuJoCo 的真实目标坐标或
分割 ID。蓝色阈值排除了现有蓝灰地板；更换材质、光照或同色物体需重新调阈值。
被遮挡或离开视野时允许返回空列表。

## 数据与坐标

输出包含：

- `rgb.png` / `detections.png`：原始彩色图 / 检测框与光轴深度。
- `aligned_depth_m.npy`：与 RGB 像素对齐的 float32 深度，单位米。
- `native_depth_m.npy`：58° 垂直 FOV 的原生深度，不可直接与 RGB 像素配对。
- `depth_preview.png`：深度伪彩色预览，黑色为无效，不用于数值测距。
- `detections.json`：仿真时间、RGB/深度内参、光学系到世界系变换、检测结果。

超出配置量程的深度为 NaN。有效深度从连通区域内部、靠近质心的实际像素选取。
JSON 中 `depth_pixel_uv` 指明该像素。输出为该像素可见表面的三维位置，
**不是物体中心**；深度无效时保留二维框，三维位置为 null。

OpenCV 光学系为 x 向右、y 向下、z 向前；MuJoCo 相机为 x 向右、y 向上、
视线沿 -z。使用 `R_world_camera @ diag(1,-1,-1)` 转换。
内参以像素中心为准，`cx=(width-1)/2`、`cy=(height-1)/2`，
`fx=fy=height/(2*tan(fovy/2))`。深度 Z 是光轴距离，反投影为
`[(u-cx)*Z/fx, (v-cy)*Z/fy, Z]`。

参考：[MuJoCo 相机定义](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#body-camera)、
[OpenCV HSV 阈值](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)。

## 与 RL 环境一起使用

`PandaObstacleEnv.get_camera_observation()` 按需返回 `(RGBDFrame, detections)`。
它不改变 PPO 现有的 9 维观测，也不会在每个物理步自动渲染。
RL 目标在 reset 时按基座坐标系采样，再转换到世界坐标系。
这是视觉采集接口；如需策略基于图像学习，还需另行定义图像观测和策略网络。

```python
import os
os.environ["MUJOCO_GL"] = "egl"  # 必须在 import mujoco/环境模块前设置
from piper_rl_mujoco import PandaObstacleEnv
env = PandaObstacleEnv(visualize=False)
try:
    obs, info = env.reset(seed=0)
    frame, detections = env.get_camera_observation()
finally:
    env.close()
```

原 RL reset 仍使用自己的零位；固定识别目标未必在此姿态的视野内。
演示目标与 RL 的随机目标不是同一个对象。使用视觉前应先将手臂移动到观察姿态。
每个采集进程需自己的渲染器，调用方应按约 30 Hz 采集以控制开销。

## 验证

```bash
conda run --no-capture-output -n piper python -m unittest discover -s tests -v
conda run --no-capture-output -n piper python -m pip check
```

测试使用真实 EGL 渲染，检查挂载跟随、图像尺寸、两个目标的表面三维定位、
连续物理仿真后的识别、移除目标后无地板误检，以及无效深度处理。
