"""Run from Piper_rl: python -m config.imu --backend sim --samples 10."""
import json
from pathlib import Path
import sys

from . import IMUProcessor, MujocoD435iIMU, RealSenseD435iIMU


def main():
    from config.cli import parse_cli
    args = parse_cli("imu")
    processor = IMUProcessor(cutoff_hz=args.cutoff_hz)
    if args.backend == "hardware":
        with RealSenseD435iIMU(args.serial, args.gyro_fps, args.accel_fps, processor) as imu:
            print(json.dumps(imu.info), file=sys.stderr)
            for _ in range(args.samples):
                print(json.dumps(imu.read().to_dict(), allow_nan=False))
    else:
        import mujoco
        from config.flobase.piper_base import FloatingBase
        root = Path(__file__).resolve().parents[2]
        model = mujoco.MjModel.from_xml_path(str(root / "xml/agilex/scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        base = FloatingBase(model, data)
        from config.episode import configure_base
        from .simulation import load_imu_config
        configure_base(base, args.settings["episode"]["base"])
        base.set_episode_offset([0, 0, args.settings["episode"]["base"]["initial_height_m"] - base.get_pose().position_m[2]])
        config = load_imu_config(args.settings)
        if args.gyro_fps is not None:
            config["gyro_fps"] = args.gyro_fps
        if args.accel_fps is not None:
            config["accel_fps"] = args.accel_fps
        imu = MujocoD435iIMU(model, processor, config)
        count = 0
        while count < args.samples:
            for sample in imu.sample(data):
                print(json.dumps(sample.to_dict(), allow_nan=False))
                count += 1
                if count == args.samples:
                    break
            base.step()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
