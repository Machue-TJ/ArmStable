"""Run with conda run -n piper python camera_demo.py --headless --frames 1."""
import json
import time
import threading


def main():
    from config.cli import parse_cli
    args = parse_cli("camera")
    from config.episode import (load_episode_config, configure_base, configure_viewer,
                                EpisodeInitializer, make_model)
    from config.settings import share_device_calibration

    import cv2
    import mujoco
    import numpy as np
    from config.vision.piper_vision import D435iCamera, annotate, depth_preview, load_vision_config
    from config.flobase.piper_base import FloatingBase
    from config.imu import MujocoD435iIMU
    from config.imu.simulation import load_imu_config

    episode_config = load_episode_config(args.settings)
    config = load_vision_config(args.settings)
    model = make_model(episode_config)
    data = mujoco.MjData(model)
    base = FloatingBase(model, data)
    configure_base(base, episode_config["base"])
    imu_config = load_imu_config(args.settings)
    share_device_calibration(config, imu_config)
    camera = D435iCamera(model, config)
    imu = MujocoD435iIMU(model, config=imu_config)
    initializer = EpisodeInitializer(model, data, base, episode_config)
    rng = np.random.default_rng(args.seed)
    reset_requested = threading.Event()
    episode_info = None
    episode_count = 0
    imu_samples = []
    viewer = None
    def reset_episode():
        nonlocal episode_info, episode_count
        episode_info = initializer.reset(rng)
        episode_count += 1
        sensor_seed = int(rng.integers(2**32))
        camera.reset(sensor_seed)
        imu.reset(sensor_seed)
        imu.sample(data)
        imu_samples.clear()
        reset_requested.clear()
        if viewer is not None:
            configure_viewer(viewer, data, episode_info)
        print(f"Episode {episode_count}: CSV sample {episode_info['sample_id']}, "
              f"{len(initializer.gids)} markers at {episode_info['marker_plane_depth_m']:.3f} m")
    reset_episode()
    viewer = None
    try:
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(
                model, data, key_callback=lambda key: reset_requested.set() if key in (ord("R"), ord("r")) else None)
            configure_viewer(viewer, data, episode_info)
            # Publish the initial scene before the first RGB-D render/alignment,
            # which can take noticeably longer than subsequent frames.
            viewer.sync()
        wall_start = time.monotonic()
        episode_start_index = 0
        last_sim_time = float(data.time)
        for index in range(args.frames):
            if (reset_requested.is_set() or data.time < last_sim_time
                    or (args.reset_every and index and index % args.reset_every == 0)):
                reset_episode()
                episode_start_index = index
            # Cadence is based on simulated time, never one image per physics step.
            deadline = (index - episode_start_index) / config["fps"]
            while data.time + 1e-9 < deadline:
                base.step()
                imu_samples.extend(sample.to_dict() for sample in imu.sample(data))
            frame = camera.capture(data)
            detections = camera.detect(frame)
            last_sim_time = float(data.time)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                cv2.imshow("D435i RGB detections", annotate(frame, detections))
                cv2.imshow("D435i aligned depth (metres)", depth_preview(frame.aligned_depth_m, config))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key in (ord("r"), ord("R")):
                    reset_requested.set()
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
        np.save(args.output / "native_depth_z16.npy", frame.native_depth_z16)
        np.save(args.output / "aligned_rgb_z_m.npy", frame.aligned_rgb_z_m)
        (args.output / "imu.jsonl").write_text(
            "".join(json.dumps(sample, allow_nan=False) + "\n" for sample in imu_samples), encoding="utf-8")
        report = {"sim_time_s": frame.sim_time, "image_size_wh": [camera.width, camera.height],
                  "episode_count": episode_count, "initialization": episode_info,
                  "depth_size_wh": [camera.depth_width, camera.depth_height],
                  "depth_scale_m": frame.depth_scale_m,
                  "calibration_source": frame.calibration_source,
                  "rgb_distortion": frame.rgb_distortion.tolist(),
                  "rgb_distortion_model": frame.rgb_distortion_model,
                  "world_from_depth_optical": frame.world_from_depth_optical.tolist(),
                  "imu_sample_count": len(imu_samples),
                  "base_position_m": base.get_pose().position_m.tolist(),
                  "base_quat_wxyz": base.get_pose().quat_wxyz.tolist(),
                  "base_linear_velocity_m_s": base.get_velocity().linear_m_s.tolist(),
                  "base_angular_velocity_rad_s": base.get_velocity().angular_rad_s.tolist(),
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
