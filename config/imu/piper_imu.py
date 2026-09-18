"""BMI055/BMI085 data processing following librealsense's D435i conventions.

Vectors: x right, y down, z forward; gyro rad/s, accel specific force m/s².
SDK motion frames already have SI scaling, axis alignment and, when available
and enabled, device calibration. Never apply raw Bosch register scaling here.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result.copy()


def _rotation(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (3, 3) or not np.isfinite(result).all()
            or not np.allclose(result.T @ result, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(result), 1, atol=1e-6)):
        raise ValueError("world_from_optical must be a proper 3x3 rotation")
    return result


def remove_gravity(acceleration_m_s2, world_from_optical,
                   gravity_world_m_s2=(0, 0, -9.80665)):
    """Return physical linear acceleration in optical axes: a = f + R.T @ g.

    Requires externally known orientation at the sample timestamp (e.g. VIO
    or simulation). A six-axis IMU alone does not provide absolute orientation.
    """
    return (_vector(acceleration_m_s2, "acceleration")
            + _rotation(world_from_optical).T @ _vector(gravity_world_m_s2, "gravity"))


def estimate_stationary_bias(gyro_samples, accel_samples, expected_specific_force,
                             max_gyro_std=0.02, max_accel_std=0.15):
    """Estimate residual SDK-output biases during a KNOWN stationary interval.

    expected_specific_force is -R.T @ g, not zero. Variance checks only reject
    noisy/moving windows; they cannot detect constant rotation or acceleration.
    This is a one-pose offset estimate, not the SDK's six-position calibration.
    """
    expected = _vector(expected_specific_force, "expected_specific_force")
    results = []
    for samples, limit in ((gyro_samples, max_gyro_std), (accel_samples, max_accel_std)):
        values = np.asarray(samples, dtype=float)
        if (values.ndim != 2 or values.shape[1] != 3 or len(values) < 20
                or not np.isfinite(values).all() or not np.isfinite(limit) or limit <= 0):
            raise ValueError("Need >=20 finite Nx3 stationary samples and positive thresholds")
        if np.any(values.std(axis=0) > limit):
            raise ValueError("Calibration window is not sufficiently stationary")
        results.append(values.mean(axis=0))
    return results[0], results[1] - expected


@dataclass
class IMUReading:
    timestamp_s: float
    angular_velocity_rad_s: np.ndarray
    acceleration_m_s2: np.ndarray
    timestamp_domain: str = "hardware_clock"
    frame_id: str = "d435i_imu_optical"
    linear_acceleration_m_s2: np.ndarray | None = None

    def to_dict(self):
        return {"timestamp_s": self.timestamp_s, "timestamp_domain": self.timestamp_domain,
                "frame_id": self.frame_id,
                "angular_velocity_rad_s": self.angular_velocity_rad_s.tolist(),
                "acceleration_m_s2": self.acceleration_m_s2.tolist(),
                "linear_acceleration_m_s2": (None if self.linear_acceleration_m_s2 is None
                                              else self.linear_acceleration_m_s2.tolist())}


class IMUProcessor:
    """Residual bias correction and optional first-order low-pass filtering.

    cutoff_hz=None preserves bandwidth. State belongs to one monotonic stream;
    call reset() after a clock reset or changing calibration. Not thread safe.
    """
    def __init__(self, gyro_bias=(0, 0, 0), accel_bias=(0, 0, 0), cutoff_hz=None,
                 reset_gap_s=0.5):
        self.gyro_bias = _vector(gyro_bias, "gyro_bias")
        self.accel_bias = _vector(accel_bias, "accel_bias")
        if cutoff_hz is not None and (not np.isfinite(cutoff_hz) or cutoff_hz <= 0):
            raise ValueError("cutoff_hz must be positive or None")
        if not np.isfinite(reset_gap_s) or reset_gap_s <= 0:
            raise ValueError("reset_gap_s must be positive")
        self.cutoff_hz, self.reset_gap_s = cutoff_hz, reset_gap_s
        self.reset()

    def reset(self):
        self._time = self._domain = self._values = None

    def process(self, timestamp_s, gyro, accel, timestamp_domain="hardware_clock",
                world_from_optical=None, gravity_world_m_s2=(0, 0, -9.80665)):
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("timestamp must be finite")
        if self._time is not None and (timestamp_s <= self._time or timestamp_domain != self._domain):
            raise ValueError("Timestamp must increase in one clock domain; reset after restart")
        values = np.stack((_vector(gyro, "gyro") - self.gyro_bias,
                           _vector(accel, "accel") - self.accel_bias))
        if self._time is not None and self.cutoff_hz is not None:
            dt = timestamp_s - self._time
            if dt <= self.reset_gap_s:
                alpha = -np.expm1(-2 * np.pi * self.cutoff_hz * dt)
                values = self._values + alpha * (values - self._values)
        linear = None if world_from_optical is None else remove_gravity(
            values[1], world_from_optical, gravity_world_m_s2)
        self._time, self._domain, self._values = timestamp_s, timestamp_domain, values.copy()
        return IMUReading(timestamp_s, values[0].copy(), values[1].copy(),
                          timestamp_domain, linear_acceleration_m_s2=linear)


class IMUSynchronizer:
    """Interpolate accel to gyro timestamps, as realsense-ros unite_imu_method=2.

    Accept independent arrival order across streams, strict order within each.
    Emit only bracketed samples; never extrapolate. Gaps > max_gap_s and gyro
    samples older than retained accel history are dropped. Buffers are bounded.
    """
    def __init__(self, max_gap_s=0.1, capacity=4096):
        if not np.isfinite(max_gap_s) or max_gap_s <= 0 or capacity < 2:
            raise ValueError("Require positive max_gap_s and capacity >=2")
        self.max_gap_s, self.capacity = max_gap_s, capacity
        self.reset()

    def reset(self):
        self._accel, self._gyro = deque(), deque()
        self._last = {}
        self._domain = None
        self.dropped_gyro = 0

    def push(self, stream, timestamp_s, xyz, timestamp_domain="hardware_clock"):
        if stream not in ("gyro", "accel"):
            raise ValueError("stream must be gyro or accel")
        t, value = float(timestamp_s), _vector(xyz, stream)
        if not np.isfinite(t) or t <= self._last.get(stream, -np.inf):
            raise ValueError("Each stream must have finite increasing timestamps")
        if self._domain is not None and self._domain != timestamp_domain:
            raise ValueError("Cannot synchronize different timestamp domains; reset first")
        self._domain, self._last[stream] = timestamp_domain, t
        buffer = self._accel if stream == "accel" else self._gyro
        buffer.append((t, value))
        if len(buffer) > self.capacity:
            buffer.popleft()
            self.dropped_gyro += int(stream == "gyro")
        output = []
        while self._gyro and len(self._accel) >= 2:
            tg, gyro = self._gyro[0]
            if tg < self._accel[0][0]:
                self._gyro.popleft()
                self.dropped_gyro += 1
                continue
            while len(self._accel) > 2 and self._accel[1][0] < tg:
                self._accel.popleft()
            (t0, a0), (t1, a1) = self._accel[0], self._accel[1]
            if tg > t1:
                break
            self._gyro.popleft()
            if t1 - t0 > self.max_gap_s:
                self.dropped_gyro += 1
                continue
            accel = a0 + (a1 - a0) * ((tg - t0) / (t1 - t0))
            output.append(IMUReading(tg, gyro.copy(), accel, timestamp_domain))
        return output


class _MujocoIMUBase:
    """Read ideal optical-axis gyro/accelerometer sensors at the model IMU site.

    capture() calls mj_forward for current-state sensors after mj_step. Values
    are instantaneous, without Bosch noise, quantization or hardware cadence.
    """
    def __init__(self, model, processor=None):
        import mujoco
        self.model, self.processor = model, processor or IMUProcessor()
        self._site = model.site("d435i_imu_site").id
        self._slices = []
        for name, kind in (("d435i_gyro", mujoco.mjtSensor.mjSENS_GYRO),
                           ("d435i_accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
            sensor = model.sensor(name)
            if sensor.type[0] != kind or sensor.objid[0] != self._site or sensor.dim[0] != 3:
                raise ValueError(f"Invalid IMU sensor {name}")
            start = int(sensor.adr[0])
            self._slices.append(slice(start, start + 3))

    def capture(self, data):
        import mujoco
        mujoco.mj_forward(self.model, data)
        gyro, accel = (data.sensordata[s].copy() for s in self._slices)
        return self.processor.process(float(data.time), gyro, accel, "simulation",
                                      data.site_xmat[self._site].reshape(3, 3),
                                      self.model.opt.gravity)
