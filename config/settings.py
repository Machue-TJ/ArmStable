"""Shared JSON configuration; no simulator or GUI imports during CLI parsing."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/settings.json"


def project_path(path):
    """Resolve project resources independently of the working directory."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def merge_settings(defaults, overrides):
    """Recursively merge dictionaries; replace lists/scalars without aliasing."""
    result = deepcopy(defaults)
    for key, value in overrides.items():
        result[key] = (merge_settings(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else deepcopy(value))
    return result


@lru_cache(maxsize=16)
def _read_settings(path, modified_ns):
    """Cache parsed JSON by absolute path and modification time."""
    with open(path, encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a JSON object")
    return result


def load_project_config(value=None):
    """Return independent project settings, optionally overlaid by a dict/file."""
    defaults = _read_settings(DEFAULT_CONFIG, DEFAULT_CONFIG.stat().st_mtime_ns)
    if value is None:
        return deepcopy(defaults)
    if not isinstance(value, dict):
        path = project_path(value).resolve()
        value = _read_settings(path, path.stat().st_mtime_ns)
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown project sections: {sorted(unknown)}; use config/settings.json format")
    return merge_settings(defaults, value)


def load_settings_section(name, value=None):
    """Load one section from project JSON, or overlay a section dictionary."""
    if isinstance(value, dict) and name not in value:
        return merge_settings(load_project_config()[name], value)
    return load_project_config(value)[name]


def share_device_calibration(vision, imu):
    """Normalize the common D435i calibration path; reject conflicting files."""
    paths = {str(project_path(p).resolve()) for p in
             (vision.get("calibration_path"), imu.get("calibration_path")) if p}
    if len(paths) > 1:
        raise ValueError("Vision and IMU must use the same device calibration")
    vision["calibration_path"] = imu["calibration_path"] = next(iter(paths), None)
