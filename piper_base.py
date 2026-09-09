"""World-frame floating base poses and simulation-time trajectory playback."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np


def _array(value, shape, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite numbers")
    return result


@dataclass
class BasePose:
    """Absolute world pose: metres and a MuJoCo-order (w, x, y, z) quaternion."""

    position_m: np.ndarray
    quat_wxyz: np.ndarray

    def __post_init__(self):
        self.position_m = _array(self.position_m, (3,), "position_m")
        self.quat_wxyz = _array(self.quat_wxyz, (4,), "quat_wxyz")
        norm = np.linalg.norm(self.quat_wxyz)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("quat_wxyz must be nonzero with a finite norm")
        self.quat_wxyz /= norm

    @classmethod
    def from_rpy(cls, position_m, rpy_rad):
        """Roll/pitch/yaw in radians, R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
        roll, pitch, yaw = _array(rpy_rad, (3,), "rpy_rad") / 2
        cr, cp, cy = np.cos([roll, pitch, yaw])
        sr, sp, sy = np.sin([roll, pitch, yaw])
        return cls(position_m, [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                                cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


class BaseTrajectory:
    """Linear translation and shortest-path quaternion SLERP; hold at endpoints."""

    def __init__(self, time_s, position_m, *, quat_wxyz=None, rpy_rad=None):
        self.time_s = np.array(time_s, dtype=float, copy=True)
        if (self.time_s.ndim != 1 or self.time_s.size == 0
                or not np.isfinite(self.time_s).all() or self.time_s[0] < 0
                or np.any(np.diff(self.time_s) <= 0)):
            raise ValueError("time_s must be nonempty, finite, nonnegative and strictly increasing")
        count = len(self.time_s)
        self.position_m = _array(position_m, (count, 3), "position_m")
        if (quat_wxyz is None) == (rpy_rad is None):
            raise ValueError("Provide exactly one of quat_wxyz or rpy_rad")
        if rpy_rad is not None:
            angles = _array(rpy_rad, (count, 3), "rpy_rad")
            self.quat_wxyz = np.array([BasePose.from_rpy(p, r).quat_wxyz
                                      for p, r in zip(self.position_m, angles)])
        else:
            quats = _array(quat_wxyz, (count, 4), "quat_wxyz")
            self.quat_wxyz = np.array([BasePose(p, q).quat_wxyz
                                      for p, q in zip(self.position_m, quats)])

    @classmethod
    def load(cls, path):
        """Load JSON/NPZ arrays or CSV with time_s,x,y,z and roll,pitch,yaw/qw,qx,qy,qz."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            values = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                values = {name: archive[name] for name in archive.files}
        elif path.suffix.lower() == ".csv":
            table = np.genfromtxt(path, delimiter=",", names=True, ndmin=1, encoding="utf-8")
            names = set(table.dtype.names or ())
            if not {"time_s", "x", "y", "z"} <= names:
                raise ValueError("CSV requires time_s,x,y,z columns")
            values = {"time_s": table["time_s"],
                      "position_m": np.column_stack([table[k] for k in ("x", "y", "z")])}
            for key, fields in [("rpy_rad", ("roll", "pitch", "yaw")),
                                ("quat_wxyz", ("qw", "qx", "qy", "qz"))]:
                if set(fields) <= names:
                    values[key] = np.column_stack([table[k] for k in fields])
        else:
            raise ValueError("Base trajectory file must be .json, .csv or .npz")
        if not isinstance(values, dict) or not {"time_s", "position_m"} <= values.keys():
            raise ValueError("Trajectory requires time_s and position_m arrays")
        return cls(values["time_s"], values["position_m"],
                   quat_wxyz=values.get("quat_wxyz"), rpy_rad=values.get("rpy_rad"))

    def __call__(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError("Sample time must be finite")
        t = np.clip(time_s, self.time_s[0], self.time_s[-1])
        hi = int(np.searchsorted(self.time_s, t, side="right"))
        if hi == len(self.time_s):
            return BasePose(self.position_m[-1], self.quat_wxyz[-1])
        lo = max(0, hi - 1)
        alpha = (t - self.time_s[lo]) / (self.time_s[hi] - self.time_s[lo])
        q0, q1 = self.quat_wxyz[lo], self.quat_wxyz[hi]
        dot = float(np.dot(q0, q1))
        if dot < 0:
            q1, dot = -q1, -dot
        dot = np.clip(dot, 0, 1)
        if dot > 0.9995:
            quat = (1 - alpha) * q0 + alpha * q1
        else:
            angle = np.arccos(dot)
            quat = (np.sin((1-alpha)*angle)*q0 + np.sin(alpha*angle)*q1) / np.sin(angle)
        return BasePose((1-alpha)*self.position_m[lo] + alpha*self.position_m[hi], quat)


class FloatingBase:
    """Drive a physical freejoint with a mocap weld, or release it for free dynamics.

    set_pose teleports for initialization by default. Motion playback updates only
    the weld target: the arm responds through MuJoCo dynamics, with tracking error.
    """

    def __init__(self, model, data):
        self.model, self.data = model, data
        joint = model.joint("base_freejoint")
        if joint.type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("base_freejoint must be a free joint")
        qadr, vadr = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.qpos_slice = slice(qadr, qadr + 7)
        self.qvel_slice = slice(vadr, vadr + 6)
        self.mocap_id = int(model.body("base_target").mocapid[0])
        self.weld_id = model.equality("base_drive").id
        self.motion = None
        self.start_time = 0.0
        qpos = data.qpos[self.qpos_slice]
        self.hold_pose = BasePose(qpos[:3], qpos[3:])
        self.reset()

    def _apply(self, pose, teleport=False):
        if not isinstance(pose, BasePose):
            raise TypeError("Base motion must return BasePose")
        # Revalidate mutable arrays before writing simulation state.
        pose = BasePose(pose.position_m, pose.quat_wxyz)
        self.data.mocap_pos[self.mocap_id] = pose.position_m
        self.data.mocap_quat[self.mocap_id] = pose.quat_wxyz
        if teleport:
            self.data.qpos[self.qpos_slice] = np.r_[pose.position_m, pose.quat_wxyz]
            self.data.qvel[self.qvel_slice] = 0
            self.data.qacc_warmstart[self.qvel_slice] = 0
        mujoco.mj_forward(self.model, self.data)

    def set_pose(self, position_m, *, quat_wxyz=None, rpy_rad=None, teleport=True):
        """Stop playback and hold a world pose; omitted rotation means identity."""
        if quat_wxyz is not None and rpy_rad is not None:
            raise ValueError("Provide quat_wxyz or rpy_rad, not both")
        pose = (BasePose.from_rpy(position_m, rpy_rad) if rpy_rad is not None
                else BasePose(position_m, [1, 0, 0, 0] if quat_wxyz is None else quat_wxyz))
        self.motion = None
        self.hold_pose = pose
        self.data.eq_active[self.weld_id] = 1
        self._apply(pose, teleport=teleport)

    def set_motion(self, motion: Callable[[float], BasePose]):
        """Start callback/trajectory at t=0 relative to the current simulation time."""
        pose = motion(0.0)
        self._apply(pose, teleport=True)
        self.motion = motion
        self.start_time = float(self.data.time)
        self.data.eq_active[self.weld_id] = 1
        mujoco.mj_forward(self.model, self.data)

    def load(self, path):
        self.set_motion(BaseTrajectory.load(path))

    def reset(self):
        """Call after mj_resetData[Keyframe]; restart motion or restore held pose."""
        self.start_time = float(self.data.time)
        self.data.eq_active[self.weld_id] = 1
        self._apply(self.motion(0.0) if self.motion is not None else self.hold_pose, teleport=True)

    def release(self):
        """Disable the platform constraint; existing model gravity compensation remains."""
        self.motion = None
        self.data.eq_active[self.weld_id] = 0
        mujoco.mj_forward(self.model, self.data)

    def get_pose(self):
        """Read the actual base pose, which may differ from the commanded target."""
        qpos = self.data.qpos[self.qpos_slice]
        return BasePose(qpos[:3], qpos[3:])

    def step(self):
        """Advance one physics step and refresh body/camera world transforms."""
        if self.motion is not None and self.data.eq_active[self.weld_id]:
            self._apply(self.motion(float(self.data.time) - self.start_time))
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
