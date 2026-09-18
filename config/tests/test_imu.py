"""Analytical motion, synchronization and mocked SDK lifecycle checks."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.imu import (IMUProcessor, IMUSynchronizer, MujocoD435iIMU,
                        RealSenseD435iIMU, estimate_stationary_bias, remove_gravity)


class ProcessingTest(unittest.TestCase):
    def test_gravity_and_rotation(self):
        r = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        np.testing.assert_allclose(remove_gravity([0, -9.80665, 0], r), 0, atol=1e-12)
        np.testing.assert_allclose(remove_gravity([1, -9.80665, 2], r), [1, 0, 2])
        with self.assertRaises(ValueError):
            remove_gravity([0, 0, 0], np.diag([1, 1, -1]))

    def test_bias_filter_and_reset(self):
        p = IMUProcessor(gyro_bias=[0.1, 0, 0], accel_bias=[0, 0, 0.2], cutoff_hz=10)
        first = p.process(0, [0.1, 0, 0], [0, 0, 10])
        np.testing.assert_allclose(first.angular_velocity_rad_s, 0)
        np.testing.assert_allclose(first.acceleration_m_s2, [0, 0, 9.8])
        second = p.process(0.01, [1.1, 0, 0], [0, 0, 10])
        self.assertAlmostEqual(second.angular_velocity_rad_s[0], 1 - np.exp(-0.2 * np.pi))
        np.testing.assert_allclose(p.process(1, [1.1, 0, 0], [0, 0, 10]).angular_velocity_rad_s, [1, 0, 0])
        with self.assertRaises(ValueError):
            p.process(1, [0, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            p.process(2, [0, 0, 0], [0, 0, 0], "simulation")
        p.reset()
        self.assertIsNone(p.process(0, [0, 0, 0], [0, 0, 0]).linear_acceleration_m_s2)

    def test_stationary_calibration_preserves_gravity(self):
        gyro = np.tile([0.01, -0.02, 0.03], (30, 1))
        accel = np.tile([0.1, -9.70665, 0.2], (30, 1))
        bg, ba = estimate_stationary_bias(gyro, accel, [0, -9.80665, 0])
        np.testing.assert_allclose(bg, gyro[0])
        np.testing.assert_allclose(ba, [0.1, 0.1, 0.2])
        accel[0] += 10
        with self.assertRaises(ValueError):
            estimate_stationary_bias(gyro, accel, [0, -9.80665, 0])

    def test_sync_interpolation_and_cross_stream_arrival(self):
        sync = IMUSynchronizer()
        self.assertEqual(sync.push("accel", 0, [0, 0, 0]), [])
        self.assertEqual(sync.push("gyro", 0.01, [1, 2, 3]), [])
        result = sync.push("accel", 0.02, [2, 4, 6])[0]
        np.testing.assert_allclose(result.acceleration_m_s2, [1, 2, 3])
        self.assertEqual(result.timestamp_s, 0.01)
        # A gyro callback can arrive after both bracketing accel callbacks.
        late = sync.push("gyro", 0.015, [4, 5, 6])[0]
        np.testing.assert_allclose(late.acceleration_m_s2, [1.5, 3, 4.5])
        with self.assertRaises(ValueError):
            sync.push("gyro", 0.015, [0, 0, 0])
        with self.assertRaises(ValueError):
            sync.push("accel", 0.03, [0, 0, 0], "system_time")

    def test_sync_gaps_and_bounded_history(self):
        sync = IMUSynchronizer(capacity=2)
        sync.push("accel", 0, [0, 0, 0])
        for t in [0.01, 0.02, 0.03]:
            sync.push("gyro", t, [0, 0, 0])
        self.assertEqual(sync.dropped_gyro, 1)
        self.assertEqual(sync.push("accel", 1, [0, 0, 0]), [])
        self.assertEqual(sync.dropped_gyro, 3)


class SimulationTest(unittest.TestCase):
    def test_sampling_rates_noise_quantization_and_reset(self):
        from config.imu.simulation import load_imu_config
        for chip, rate, expected_accels in (("BMI085", 200, 201), ("BMI055", 62.5, 63)):
            config = load_imu_config()
            config.update(model=chip, accel_fps=rate)
            model = self.make_model()
            data = mujoco.MjData(model)
            imu = MujocoD435iIMU(model, config=config)
            gyro_times, accel_times, readings, gyro_values = [], [], [], []
            for tick in range(501):
                data.time = tick*model.opt.timestep
                readings.extend(imu.sample(data))
                for stream, stamp, value, _ in imu.last_motion_samples:
                    if stream == "gyro":
                        gyro_times.append(stamp)
                        gyro_values.append(value)
                    else:
                        accel_times.append(stamp)
            self.assertEqual(len(gyro_times), 401)
            self.assertEqual(len(accel_times), expected_accels)
            np.testing.assert_allclose(np.diff(gyro_times), 1/400, atol=1e-12)
            np.testing.assert_allclose(np.diff(accel_times), 1/rate, atol=1e-12)
            self.assertGreater(np.std(gyro_values), 0.001)
            self.assertIsNone(readings[-1].linear_acceleration_m_s2)
            self.assertEqual(imu.sample(data), [])
            data.time = 0
            imu.sample(data)
            np.testing.assert_array_equal(imu.last_motion_samples[1][2], gyro_values[0])

    def test_range_and_skipped_sampling(self):
        from config.imu.simulation import load_imu_config
        config = load_imu_config()
        config.update(noise_enabled=False)
        model = self.make_model(True)
        model.opt.gravity[:] = 0
        data = mujoco.MjData(model)
        data.qvel[5] = 30
        imu = MujocoD435iIMU(model, config=config)
        imu.sample(data)
        for stream, _, value, _ in imu.last_motion_samples:
            self.assertLessEqual(np.max(np.abs(value)), np.deg2rad(1000) if stream == "gyro" else 4*9.80665)
        data.time = 0.01
        with self.assertRaises(ValueError):
            imu.sample(data)

    def make_model(self, free=False):
        joint = '<freejoint/>' if free else ''
        return mujoco.MjModel.from_xml_string(f'''<mujoco>
          <worldbody><body>{joint}<geom type="sphere" size="0.1" mass="1"/>
            <site name="d435i_imu_site" pos="0.2 0 0" quat="0 1 0 0"/>
          </body></worldbody><sensor>
            <gyro name="d435i_gyro" site="d435i_imu_site"/>
            <accelerometer name="d435i_accel" site="d435i_imu_site"/>
          </sensor></mujoco>''')

    def test_fixed_and_free_fall(self):
        for free in [False, True]:
            model = self.make_model(free)
            sample = MujocoD435iIMU(model).capture_truth(mujoco.MjData(model))
            np.testing.assert_allclose(sample.angular_velocity_rad_s, 0, atol=1e-12)
            np.testing.assert_allclose(sample.acceleration_m_s2,
                                       [0, 0, 0] if free else [0, 0, -9.81], atol=1e-12)
            np.testing.assert_allclose(sample.linear_acceleration_m_s2,
                                       [0, 0, 9.81] if free else [0, 0, 0], atol=1e-12)

    def test_rotating_offset_sensor(self):
        model = self.make_model(True)
        model.opt.gravity[:] = 0
        data = mujoco.MjData(model)
        data.qvel[5] = 2
        sample = MujocoD435iIMU(model).capture_truth(data)
        np.testing.assert_allclose(sample.angular_velocity_rad_s, [0, 0, -2], atol=1e-12)
        np.testing.assert_allclose(sample.acceleration_m_s2, [-0.8, 0, 0], atol=1e-12)

    def test_project_mount_axes_and_output(self):
        root = Path(__file__).resolve().parents[2]
        model = mujoco.MjModel.from_xml_path(str(root / "xml/agilex/scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        sample = MujocoD435iIMU(model).capture_truth(data)
        imu_r = data.site_xmat[model.site("d435i_imu_site").id].reshape(3, 3)
        camera_r = data.cam_xmat[model.camera("d435i_depth").id].reshape(3, 3)
        np.testing.assert_allclose(imu_r, camera_r @ np.diag([1, -1, -1]), atol=1e-9)
        self.assertTrue(np.isfinite(sample.acceleration_m_s2).all())
        self.assertEqual(sample.timestamp_domain, "simulation")


class HardwareAdapterTest(unittest.TestCase):
    def fake_sdk(self, accel_rate=250, fail=False):
        def profile(kind, rate):
            return NS(stream_type=lambda: kind, fps=lambda: rate, format=lambda: "xyz")
        def frame(kind, t, xyz):
            f = NS(get_profile=lambda: profile(kind, 0), get_timestamp=lambda: t,
                   get_frame_timestamp_domain=lambda: "hardware_clock",
                   get_motion_data=lambda: NS(x=xyz[0], y=xyz[1], z=xyz[2]))
            f.as_motion_frame = lambda: f
            return f
        self.closed = self.stopped = False
        def start(callback):
            if fail:
                raise RuntimeError("start failure")
            callback(frame("accel", 1000, [0, -10, 0]))
            callback(frame("gyro", 1005, [0.1, 0.2, 0.3]))
            callback(frame("accel", 1010, [0, -8, 0]))
        sensor = NS(is_motion_sensor=lambda: True,
                    get_stream_profiles=lambda: [profile("gyro", 400), profile("accel", accel_rate)],
                    supports=lambda _: False, open=lambda _: None, start=start,
                    stop=lambda: setattr(self, "stopped", True),
                    close=lambda: setattr(self, "closed", True))
        device = NS(get_info=lambda k: "Intel RealSense D435I" if k == "name" else "123",
                    query_sensors=lambda: [sensor])
        return NS(context=lambda: NS(query_devices=lambda: [device]),
                  camera_info=NS(name="name", serial_number="serial"),
                  stream=NS(gyro="gyro", accel="accel"), format=NS(motion_xyz32f="xyz"),
                  option=NS(enable_motion_correction="correction"))

    def test_both_revisions_units_and_cleanup(self):
        for rate in [250, 200]:
            with patch.dict(sys.modules, {"pyrealsense2": self.fake_sdk(rate)}):
                with RealSenseD435iIMU() as imu:
                    self.assertEqual(imu.info["accel_fps"], rate)
                    sample = imu.read()
                    self.assertAlmostEqual(sample.timestamp_s, 1.005)
                    np.testing.assert_allclose(sample.angular_velocity_rad_s, [0.1, 0.2, 0.3])
                    np.testing.assert_allclose(sample.acceleration_m_s2, [0, -9, 0])
                    with self.assertRaises(TimeoutError):
                        imu.read(timeout_s=0.001)
                self.assertTrue(self.closed and self.stopped)

    def test_profile_error_and_start_cleanup(self):
        with patch.dict(sys.modules, {"pyrealsense2": self.fake_sdk()}):
            with self.assertRaises(ValueError):
                RealSenseD435iIMU(accel_fps=123).start()
        with patch.dict(sys.modules, {"pyrealsense2": self.fake_sdk(fail=True)}):
            with self.assertRaises(RuntimeError):
                RealSenseD435iIMU().start()
            self.assertTrue(self.closed)


if __name__ == "__main__":
    unittest.main()
