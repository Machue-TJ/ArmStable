"""D435i asynchronous IMU sampling with range, quantization and white noise.

Noise densities use Bosch typical component specifications; effective noise
bandwidth fs/2 is an approximation, not a model of firmware digital filtering.
"""
from config.settings import load_settings_section

import numpy as np

from .piper_imu import _MujocoIMUBase as _TruthSensor, IMUSynchronizer, _vector

PROFILES = {
    "BMI055": {"accel_rates": (62.5, 250), "accel_bits": 12, "accel_noise_ug": 150},
    # Use the larger (Z-axis) typical density for all output axes; chip axes
    # and SDK optical axes are different. A device noise fit should replace it.
    "BMI085": {"accel_rates": (100, 200), "accel_bits": 16, "accel_noise_ug": 135},
}


def load_imu_config(value=None):
    """Load the IMU section from unified project settings or a section dict."""
    return load_settings_section("imu", value)


class MujocoD435iIMU(_TruthSensor):
    """Call sample(data) every physics step; it returns all newly synced samples.

    Sensor vectors are interpolated between adjacent physics states at exact
    independent gyro/accel times, then corrupted and synchronized as on hardware.
    capture_truth(data) is explicitly separate and only for ground-truth checks.
    """
    def __init__(self, model, processor=None, config=None):
        self.config = load_imu_config(config)
        name = self.config["model"]
        if name not in PROFILES:
            raise ValueError("IMU model must be BMI055 or BMI085")
        self.profile = PROFILES[name]
        if self.config["gyro_fps"] not in (200, 400) or self.config["accel_fps"] not in self.profile["accel_rates"]:
            raise ValueError(f"Unsupported sample rates for {name}")
        self._rates = {"gyro": self.config["gyro_fps"], "accel": self.config["accel_fps"]}
        self._bias = {"gyro": _vector(self.config["gyro_bias_rad_s"], "gyro bias"),
                      "accel": _vector(self.config["accel_bias_m_s2"], "accel bias")}
        self._range = {"gyro": np.deg2rad(1000), "accel": 4*9.80665}
        self._bits = {"gyro": 16, "accel": self.profile["accel_bits"]}
        self._density = {"gyro": np.deg2rad(0.014), "accel": self.profile["accel_noise_ug"]*1e-6*9.80665}
        if model.opt.timestep > 1/max(self._rates.values()) + 1e-9:
            raise ValueError("Physics timestep must be <= shortest IMU period")
        if self.config.get("calibration_path"):
            from config.d435i import load_calibration, apply_calibration
            apply_calibration(model, load_calibration(self.config["calibration_path"]))
        super().__init__(model, processor)
        self.synchronizer = IMUSynchronizer()
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(self.config["seed"] if seed is None else seed)
        self.processor.reset()
        self.synchronizer.reset()
        self._previous = self._start = None
        self._indices = {"gyro": 0, "accel": 0}
        self.last_motion_samples = []

    def capture_truth(self, data):
        # Keep truth completely independent of residual bias/filter state.
        import mujoco
        from .piper_imu import IMUReading, remove_gravity
        mujoco.mj_forward(self.model, data)
        gyro, accel = (data.sensordata[s].copy() for s in self._slices)
        linear = remove_gravity(accel, data.site_xmat[self._site].reshape(3, 3), self.model.opt.gravity)
        return IMUReading(float(data.time), gyro, accel, "simulation",
                          linear_acceleration_m_s2=linear)

    def _measure(self, stream, truth):
        value = truth + self._bias[stream]
        if self.config["noise_enabled"]:
            sigma = self._density[stream]*np.sqrt(self._rates[stream]/2)
            value = value + self.rng.normal(0, sigma, 3)
        limit = self._range[stream]
        step = 2*limit / 2**self._bits[stream]
        if self.config["quantization_enabled"]:
            value = np.round(value/step)*step
        return np.clip(value, -limit, limit-step)

    def sample(self, data):
        now = float(data.time)
        if self._previous is not None and now < self._previous.timestamp_s:
            self.reset()
        if self._previous is not None:
            dt = now-self._previous.timestamp_s
            if dt == 0:
                self.last_motion_samples = []
                return []
            if dt > 1/max(self._rates.values())+1e-9:
                raise ValueError("IMU sampling skipped physics states; call sample every physics step")
        current = self.capture_truth(data)
        if self._start is None:
            self._start = now
            self._previous = current
        events = []
        for stream, attr in (("accel", "acceleration_m_s2"), ("gyro", "angular_velocity_rad_s")):
            while True:
                stamp = self._start + self._indices[stream]/self._rates[stream]
                if stamp > now+1e-10:
                    break
                dt = now-self._previous.timestamp_s
                fraction = (stamp-self._previous.timestamp_s)/dt if dt > 0 else 1
                old, new = getattr(self._previous, attr), getattr(current, attr)
                value = old+(new-old)*np.clip(fraction, 0, 1)
                events.append((stream, stamp, self._measure(stream, value), "simulation"))
                self._indices[stream] += 1
        events.sort(key=lambda event: (event[1], event[0]))
        self.last_motion_samples = events
        result = []
        for event in events:
            for sample in self.synchronizer.push(*event):
                result.append(self.processor.process(sample.timestamp_s, sample.angular_velocity_rad_s,
                                                     sample.acceleration_m_s2, "simulation"))
        self._previous = current
        return result

    def capture(self, data):
        """Latest newly produced sample or None; use sample() to retain all samples."""
        samples = self.sample(data)
        return samples[-1] if samples else None
