"""Optional pyrealsense2 adapter; importing this module needs no USB SDK."""
from collections import deque
from queue import Empty, Full, Queue
import time

from .piper_imu import IMUProcessor, IMUSynchronizer


class RealSenseD435iIMU:
    """Own only the motion sensor using the SDK low-level sensor callback API.

    SDK callbacks enqueue independent motion samples, never video framesets.
    read() synchronizes/processes on the caller thread. One reader only; do not
    concurrently close/start/read. Overflow is reported, not silently hidden.
    """
    def __init__(self, serial=None, gyro_fps=None, accel_fps=None, processor=None):
        self.serial, self.gyro_fps, self.accel_fps = serial, gyro_fps, accel_fps
        self.processor = processor or IMUProcessor()
        self.synchronizer = IMUSynchronizer()
        self._sensor = None
        self._queue = Queue(maxsize=4096)
        self._ready = deque()
        self._error = None
        self.info = {}

    def start(self):
        if self._sensor is not None:
            raise RuntimeError("IMU already started")
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("Install config/imu/requirements-imu.txt for real hardware") from exc
        self._context = rs.context()
        devices = [d for d in self._context.query_devices()
                   if "D435I" in d.get_info(rs.camera_info.name).upper()
                   and (self.serial is None or d.get_info(rs.camera_info.serial_number) == self.serial)]
        if len(devices) != 1:
            raise RuntimeError("Expected one matching D435i; connect camera or specify serial")
        device = devices[0]
        sensor = next((s for s in device.query_sensors() if s.is_motion_sensor()), None)
        if sensor is None:
            raise RuntimeError("D435i has no available motion sensor")
        profiles = sensor.get_stream_profiles()
        selected, rates = [], {}
        for name, kind, requested in (("gyro", rs.stream.gyro, self.gyro_fps),
                                      ("accel", rs.stream.accel, self.accel_fps)):
            available = [p for p in profiles if p.stream_type() == kind
                         and p.format() == rs.format.motion_xyz32f]
            rates[name] = sorted({p.fps() for p in available})
            if not available or (requested is not None and requested not in rates[name]):
                raise ValueError(f"Unsupported {name} rate {requested}; available: {rates[name]}")
            # Highest reported rate works with both BMI055 and BMI085 revisions.
            rate = requested if requested is not None else max(rates[name])
            selected.append(next(p for p in available if p.fps() == rate))
        self.processor.reset()
        self.synchronizer.reset()
        self._queue, self._ready, self._error = Queue(maxsize=4096), deque(), None
        self.info = {"name": device.get_info(rs.camera_info.name),
                     "serial": device.get_info(rs.camera_info.serial_number),
                     "available_rates_hz": rates,
                     "gyro_fps": selected[0].fps(), "accel_fps": selected[1].fps()}
        option = rs.option.enable_motion_correction
        self.info["motion_correction_enabled"] = None
        if sensor.supports(option):
            if not sensor.is_option_read_only(option):
                sensor.set_option(option, 1)
            self.info["motion_correction_enabled"] = bool(sensor.get_option(option))

        def callback(frame):
            try:
                kind = frame.get_profile().stream_type()
                if kind not in (rs.stream.accel, rs.stream.gyro):
                    return
                xyz = frame.as_motion_frame().get_motion_data()
                self._queue.put_nowait(("gyro" if kind == rs.stream.gyro else "accel",
                                        frame.get_timestamp() * 1e-3, (xyz.x, xyz.y, xyz.z),
                                        str(frame.get_frame_timestamp_domain())))
            except Full:
                self._error = RuntimeError("IMU input queue overflow; read samples faster and restart")
            except Exception as exc:
                self._error = exc

        sensor.open(selected)
        try:
            sensor.start(callback)
        except Exception:
            sensor.close()
            raise
        self._sensor = sensor
        return self

    def read(self, timeout_s=2.0):
        if self._sensor is None:
            raise RuntimeError("Start IMU before reading")
        if not 0 < timeout_s < float("inf"):
            raise ValueError("timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout_s
        while True:
            if self._error is not None:
                raise RuntimeError("IMU callback failed") from self._error
            if self._ready:
                sample = self._ready.popleft()
                return self.processor.process(sample.timestamp_s, sample.angular_velocity_rad_s,
                                              sample.acceleration_m_s2, sample.timestamp_domain)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for synchronized D435i IMU data")
            try:
                event = self._queue.get(timeout=min(remaining, 0.1))
            except Empty:
                continue
            self._ready.extend(self.synchronizer.push(*event))

    def close(self):
        sensor, self._sensor = self._sensor, None
        if sensor is not None:
            try:
                sensor.stop()
            finally:
                sensor.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
