"""Shared nominal geometry and optional per-device SDK calibration.

All transforms are target_from_source in optical coordinates, metres.
Nominal values are from realsense2_description, not a per-device calibration.
"""
import json
from pathlib import Path

import numpy as np

OPTICAL_FROM_MUJOCO = np.diag([1., -1., -1.])


def nominal_calibration():
    def pose(t):
        return {"rotation": np.eye(3).tolist(), "translation_m": t}
    return {"source": "realsense-ros nominal extrinsics; product nominal FOV",
            "serial": None, "depth_scale_m": 0.001,
            "color_from_depth": pose([0.015, 0, 0]),
            "right_from_depth": pose([-0.05, 0, 0]),
            "imu_from_depth": pose([-0.00552, 0.0051, 0.01174])}


def transform(spec):
    r = np.asarray(spec["rotation"], dtype=float)
    t = np.asarray(spec["translation_m"], dtype=float)
    if (r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all()
            or not np.isfinite(t).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-6)):
        raise ValueError("Extrinsics require a proper rotation and finite translation in metres")
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = r, t
    return result


def load_calibration(path=None):
    if path is None:
        return nominal_calibration()
    with open(Path(path), encoding="utf-8") as stream:
        result = json.load(stream)
    for name in ("color_from_depth", "right_from_depth", "imu_from_depth"):
        transform(result[name])
    if not np.isfinite(result["depth_scale_m"]) or result["depth_scale_m"] <= 0:
        raise ValueError("depth_scale_m must be finite and positive")
    return result


def apply_calibration(model, calibration):
    """Apply internal extrinsics in the existing bracket frame, before simulation.

    This does not modify the unknown link6-to-housing hand-eye transform.
    """
    import mujoco
    c = OPTICAL_FROM_MUJOCO
    depth = model.camera("d435i_depth")
    origin = depth.pos.copy()
    for key, name, is_site in (("color_from_depth", "d435i_rgb", False),
                               ("right_from_depth", "d435i_ir_right", False),
                               ("imu_from_depth", "d435i_imu_site", True)):
        source_from_target = np.linalg.inv(transform(calibration[key]))
        obj = model.site(name) if is_site else model.camera(name)
        obj.pos[:] = origin + c @ source_from_target[:3, 3]
        rotation = c @ source_from_target[:3, :3]
        if not is_site:
            rotation = rotation @ c
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        obj.quat[:] = quat
