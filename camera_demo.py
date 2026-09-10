"""Run with conda run -n piper python camera_demo.py --headless --frames 1."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="PiPER wrist RGB-D camera and color detection")
    parser.add_argument("--headless", action="store_true", help="EGL offscreen rendering, no windows")
    parser.add_argument("--frames", type=int, default=300, help="Number of 30 Hz camera frames")
    parser.add_argument("--config", type=Path, help="Override config/vision/vision_config.json")
    parser.add_argument("--base-motion", type=Path, help="Base trajectory: JSON, CSV or NPZ")
    parser.add_argument("--base-pos", nargs=3, type=float, default=[0, 0, 0], metavar=("X", "Y", "Z"),
                        help="Initial world base position in metres")
    parser.add_argument("--base-rpy", nargs=3, type=float, default=[0, 0, 0], metavar=("ROLL", "PITCH", "YAW"),
                        help="Initial world base rotation in radians")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "config" / "outputs" / "vision")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    # Must select backend BEFORE importing MuJoCo.
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/piper-matplotlib")
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")

    import cv2
    import mujoco
    import numpy as np
    from config.vision.piper_vision import D435iCamera, ROOT, annotate, depth_preview, detect_targets, load_config
    from config.flobase.piper_base import FloatingBase

    config = load_config(args.config)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "xml/agilex/scene.xml"))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    base = FloatingBase(model, data)
    if args.base_motion:
        base.load(args.base_motion)
    else:
        base.set_pose(args.base_pos, rpy_rad=args.base_rpy)
    mujoco.mj_forward(model, data)
    camera = D435iCamera(model, config)
    viewer = None
    try:
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = [0.4, 0, 0.2]
            viewer.cam.distance = 1.4
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -25
            viewer.opt.geomgroup[3] = 0
        wall_start = time.monotonic()
        for index in range(args.frames):
            # Cadence is based on simulated time, never one image per physics step.
            deadline = index / config["fps"]
            while data.time + 1e-9 < deadline:
                base.step()
            frame = camera.capture(data)
            detections = detect_targets(frame, config)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                cv2.imshow("D435i RGB detections", annotate(frame, detections))
                cv2.imshow("D435i aligned depth (metres)", depth_preview(frame.aligned_depth_m, config))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                time.sleep(max(0, wall_start + (index + 1) / config["fps"] - time.monotonic()))
        args.output.mkdir(parents=True, exist_ok=True)
        images = {"rgb.png": cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
                  "detections.png": annotate(frame, detections),
                  "depth_preview.png": depth_preview(frame.aligned_depth_m, config)}
        for name, image in images.items():
            if not cv2.imwrite(str(args.output / name), image):
                raise OSError(f"Could not save {name}")
        np.save(args.output / "aligned_depth_m.npy", frame.aligned_depth_m)
        np.save(args.output / "native_depth_m.npy", frame.native_depth_m)
        report = {"sim_time_s": frame.sim_time, "image_size_wh": [camera.width, camera.height],
                  "base_position_m": base.get_pose().position_m.tolist(),
                  "base_quat_wxyz": base.get_pose().quat_wxyz.tolist(),
                  "rgb_intrinsics": frame.rgb_intrinsics.tolist(),
                  "depth_intrinsics": frame.depth_intrinsics.tolist(),
                  "world_from_rgb_optical": frame.world_from_optical.tolist(),
                  "detections": detections}
        (args.output / "detections.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, indent=2, allow_nan=False))
        print(f"Saved RGB, metric depth and detections to {args.output.resolve()}")
    finally:
        camera.close()
        if viewer is not None:
            viewer.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
