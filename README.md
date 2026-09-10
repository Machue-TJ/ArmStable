# Piper RL Demo

使用 MuJoCo 和 PPO 训练 PiPER 机械臂完成 Reach Target 任务：夹爪中心到达指定位置，不约束末端姿态。

## 环境

在项目目录下运行：

```bash
conda activate piper
python -m pip install -r requirements.txt
```

## 目录

- `piper_rl_mujoco.py`：PPO 训练和策略测试入口。
- `camera_demo.py`：相机与基座运动演示入口。
- `config/vision/`：D435i 视觉模块、相机参数和视觉依赖。
- `config/flobase/`：浮动基座控制模块和轨迹配置。
- `config/doc/`、`config/tests/`、`config/scripts/`：说明、测试和辅助启动脚本。
- `xml/agilex/`：机械臂模型与网格资源。
- `xml/parts/`：D435i 相机和视觉目标 XML 配置。
- `models/`：策略模型；附带 `piper_ppo_reach_target.zip`。
- `doc/`：MuJoCo 演示图片。
- `tensorboard/`：训练日志。

## 相机和浮动基座

详细参数与接口见 [D435i 视觉说明](config/doc/d435i_vision.md) 和
[浮动基座说明](config/doc/floating_base.md)。

```bash
python camera_demo.py --headless --frames 1
python camera_demo.py --base-motion config/flobase/base_motion.json --frames 150
python camera_demo.py --headless --frames 1 --base-pos 0.1 0 0.2 --base-rpy 0 0 0.3
```

默认视觉配置为 `config/vision/vision_config.json`，相机输出保存在 `config/outputs/vision/`。
Python 接口从 `config.flobase.piper_base` 和 `config.vision.piper_vision` 导入。

## 验证

```bash
python -m unittest discover -s config/tests -v
python -m pip check
```

## 仓库

[Piper_rl](https://github.com/vanstrong12138/Piper_rl.git)
[Agilex-College](https://github.com/agilexrobotics/Agilex-College.git)

## Mujoco示例

### 在mujoco中实现多个piper并行训练

在脚本末尾设置 `TRAIN_MODE = True`，并配置 `models/` 下的保存路径后运行：
```bash
python piper_rl_mujoco.py
```

开启tensorboard可以看见训练过程中多个piper的奖励变化
```bash
tensorboard  --logdir tensorboard/piper_reach_target/
```
![alt text](doc/image5.png)

### 在mujoco中测试训练好的模型

在脚本末尾设置 `TRAIN_MODE = False`，将 `MODEL_PATH` 指向实际存在的模型。
测试模式默认使用附带的 `models/piper_ppo_reach_target.zip`；训练模式保存到
`models/piper_train_reach_target.zip`。测试新模型时修改 `model_name`。然后运行：
```bash
python piper_rl_mujoco.py
```

可以看到piper成功到达目标位置

![alt text](doc/image6.png)
![alt text](doc/image7.png)

## 参考

[https://github.com/LitchiCheng/mujoco-learning](https://github.com/LitchiCheng/mujoco-learning)
