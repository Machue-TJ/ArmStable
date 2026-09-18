"""The only command-line parser in the project. Defaults live in settings.json."""
import argparse
import os
from pathlib import Path

from .settings import DEFAULT_CONFIG, load_project_config, project_path


def parse_cli(command, argv=None):
    """Parse once, load project JSON, then apply explicit command-line overrides.

    Returns Namespace with resolved runtime options and a `settings` dictionary.
    This module intentionally imports neither MuJoCo nor torch.
    """
    parser = argparse.ArgumentParser(description=f"PiPER {command}",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=f"Unified project JSON (default: {DEFAULT_CONFIG})")
    if command in ("camera", "rl"):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--seed", type=int)
        parser.add_argument("--init-poses", type=Path)
        parser.add_argument("--marker-count", type=int)
        parser.add_argument("--marker-depth", type=float)
        parser.add_argument("--marker-fovy", nargs=2, type=float, metavar=("H_DEG", "V_DEG"),
                            help="Marker generation FOV only; leaves D435i imaging unchanged")
        parser.add_argument("--marker-generator")
        sources = parser.add_mutually_exclusive_group()
        sources.add_argument("--base-motion", type=Path)
        sources.add_argument("--base-callback")
        sources.add_argument("--base-velocity-callback")
        parser.add_argument("--base-pos", nargs=3, type=float,
                            help="Reference XYZ; initial Z is normalized to episode.base.initial_height_m")
        parser.add_argument("--base-rpy", nargs=3, type=float)
    if command == "camera":
        parser.add_argument("--frames", type=int)
        parser.add_argument("--reset-every", type=int)
        parser.add_argument("--output", type=Path)
    elif command == "rl":
        parser.add_argument("--mode", choices=("train", "test", "smoke"))
        for option in ("episodes", "steps", "n-envs", "total-timesteps"):
            parser.add_argument(f"--{option}", type=int)
        parser.add_argument("--model-path", type=Path)
    elif command == "imu":
        parser.add_argument("--backend", choices=("sim", "hardware"))
        parser.add_argument("--samples", type=int)
        parser.add_argument("--serial")
        parser.add_argument("--gyro-fps", type=int)
        parser.add_argument("--accel-fps", type=int)
        parser.add_argument("--cutoff-hz", type=float)
    elif command == "calibration":
        parser.add_argument("--serial")
        parser.add_argument("--output", type=Path)
        parser.add_argument("--depth-size", nargs=2, type=int)
        parser.add_argument("--color-size", nargs=2, type=int)
    else:
        raise ValueError(f"Unknown CLI command: {command}")
    explicit = vars(parser.parse_args(argv))
    try:
        settings = load_project_config(explicit.get("config"))
        values = dict(settings["cli"][command], **explicit)
        for name, choices in (("mode", ("train", "test", "smoke")), ("backend", ("sim", "hardware"))):
            if name in values and values[name] not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        episode = settings["episode"]
        for arg, key in (("marker_count", "count"), ("marker_depth", "plane_depth_m"),
                         ("marker_generator", "generator"), ("marker_fovy", "marker_fovy")):
            if arg in values:
                episode["markers"][key] = values[arg]
        if "init_poses" in values:
            episode["init_poses_csv"] = str(values["init_poses"])
        base = episode["base"]
        if "base_pos" in values:
            base["position_m"] = values["base_pos"]
        if "base_rpy" in values:
            base["rpy_rad"] = values["base_rpy"]
            base.pop("quat_wxyz", None)
        for arg, mode, key in (("base_motion", "trajectory", "trajectory"),
                               ("base_callback", "pose_callback", "callback"),
                               ("base_velocity_callback", "velocity_callback", "callback")):
            if arg in values:
                base.update(mode=mode, **{key: str(values[arg])})
        for name in ("frames", "episodes", "steps", "n_envs", "total_timesteps", "samples"):
            if name in values and values[name] < 1:
                raise ValueError(f"{name} must be positive")
        if values.get("reset_every", 0) < 0:
            raise ValueError("reset_every must be nonnegative")
        if command == "calibration":
            if not values.get("output"):
                raise ValueError("--output is required for calibration export")
            values.setdefault("depth_size", [settings["vision"]["depth_width"], settings["vision"]["depth_height"]])
            values.setdefault("color_size", [settings["vision"]["width"], settings["vision"]["height"]])
        if command == "rl" and values["model_path"] is None:
            values["model_path"] = values["train_model_path" if values["mode"] == "train" else "test_model_path"]
        for name in ("output", "model_path"):
            if values.get(name) is not None:
                values[name] = project_path(values[name])
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    if values.get("headless"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/piper-matplotlib")
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/piper-mesa-cache")
    return argparse.Namespace(**values, settings=settings)
