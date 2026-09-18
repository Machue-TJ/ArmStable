"""Export the connected D435i's SDK intrinsics/extrinsics for simulation."""
import json

import numpy as np


def export_calibration(serial=None, depth_size=None, color_size=None, fps=None):
    from config.settings import load_project_config
    vision = load_project_config()["vision"]
    depth_size = tuple(depth_size or (vision["depth_width"], vision["depth_height"]))
    color_size = tuple(color_size or (vision["width"], vision["height"]))
    fps = vision["fps"] if fps is None else fps
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = [d for d in ctx.query_devices() if "D435I" in d.get_info(rs.camera_info.name).upper()
               and (serial is None or d.get_info(rs.camera_info.serial_number) == serial)]
    if len(devices) != 1:
        raise RuntimeError("Connect exactly one D435i or specify --serial")
    serial = devices[0].get_info(rs.camera_info.serial_number)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, *depth_size, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, *color_size, rs.format.rgb8, fps)
    pipeline = rs.pipeline(ctx)
    active = pipeline.start(config)
    try:
        device = active.get_device()
        depth = active.get_stream(rs.stream.depth)
        color = active.get_stream(rs.stream.color)
        profiles = [p for s in device.query_sensors() for p in s.get_stream_profiles()]
        right = next(p for p in profiles if p.stream_type() == rs.stream.infrared
                     and p.stream_index() == 2 and p.format() == rs.format.y8 and p.fps() == fps
                     and (p.as_video_stream_profile().width(), p.as_video_stream_profile().height()) == depth_size)
        imu = next(p for p in profiles if p.stream_type() == rs.stream.accel
                   and p.format() == rs.format.motion_xyz32f)

        def intrinsics(profile):
            i = profile.as_video_stream_profile().get_intrinsics()
            return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy,
                    "ppx": i.ppx, "ppy": i.ppy, "model": str(i.model), "coeffs": list(i.coeffs)}

        def extrinsics(target):
            e = depth.get_extrinsics_to(target)
            # rs2_extrinsics.rotation is column-major, not row-major.
            return {"rotation": np.asarray(e.rotation).reshape(3, 3, order="F").tolist(),
                    "translation_m": list(e.translation)}

        sensor = device.first_depth_sensor()
        settings = {}
        for name in ("emitter_enabled", "laser_power", "exposure", "gain", "enable_auto_exposure", "visual_preset"):
            option = getattr(rs.option, name)
            if sensor.supports(option):
                settings[name] = sensor.get_option(option)
        return {"source": "connected D435i SDK calibration", "serial": serial,
                "firmware": device.get_info(rs.camera_info.firmware_version), "fps": fps,
                "depth_scale_m": sensor.get_depth_scale(),
                "depth_intrinsics": intrinsics(depth), "color_intrinsics": intrinsics(color),
                "right_intrinsics": intrinsics(right), "color_from_depth": extrinsics(color),
                "right_from_depth": extrinsics(right), "imu_from_depth": extrinsics(imu),
                "depth_options_recorded_only": settings,
                "imu_supported_rates_hz": {name: sorted({p.fps() for p in profiles if p.stream_type() == kind})
                                           for name, kind in (("accel", rs.stream.accel), ("gyro", rs.stream.gyro))}}
    finally:
        pipeline.stop()


def main():
    from config.cli import parse_cli
    args = parse_cli("calibration")
    result = export_calibration(args.serial, tuple(args.depth_size), tuple(args.color_size),
                                args.settings["vision"]["fps"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
