"""Editable callbacks shared by camera_demo and RL; all time values are seconds."""
from functools import lru_cache

import numpy as np

from config.flobase.piper_base import BasePose, BaseVelocity


BASE_HOLD_S = 0.5
BASE_SEGMENT_S = 8.0
BASE_LIMITS = np.r_[np.full(3, .1), np.full(3, np.deg2rad(5))]


@lru_cache(maxsize=2048)
def _base_waypoint(seed, index):
    """Index-based sampling makes playback independent of evaluation order."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    point = np.zeros(6) if index == 0 else rng.uniform(-BASE_LIMITS, BASE_LIMITS)
    point.flags.writeable = False
    return point


def base_pose(time_s, seed=0):
    """0.5 s hold, then bounded random waypoints with minimum-jerk interpolation.

    XYZ offsets stay within +/-0.1 m and RPY within +/-5 degrees.
    Eight-second quintic segments have zero velocity/acceleration at joins.
    Worst-case per-axis bounds: 0.046875 m/s, 0.018043 m/s^2, 0.023438 m/s^3;
    RPY bounds: 2.34375 deg/s, 0.90211 deg/s^2, 1.171875 deg/s^3.
    These are command bounds; weld reaction forces depend on full dynamics.
    """
    elapsed = max(0.0, time_s - BASE_HOLD_S) / BASE_SEGMENT_S
    index = int(np.floor(elapsed))
    u = elapsed - index
    blend = u**3 * (10 + u * (-15 + 6 * u))
    start, end = _base_waypoint(int(seed), index), _base_waypoint(int(seed), index + 1)
    pose = (1 - blend) * start + blend * end
    return BasePose.from_rpy(pose[:3] + [0, 0, .1], pose[3:])


class RandomBaseMotion:
    """Episode-owned seed; shared demo/RL resets sample independent trajectories."""
    def __init__(self, seed=0):
        self.seed = seed

    def reset(self, rng):
        self.seed = int(rng.integers(2**32))

    def __call__(self, time_s):
        return base_pose(time_s, self.seed)


def base_velocity(time_s):
    """World velocity and angular velocity, integrated from configured base pose."""
    return BaseVelocity([0.02 * np.cos(time_s), 0, 0], [0, 0, 0.05 * np.cos(time_s)])


def gen_mkr4train(context, rng):
    """Place grid cells 1, 2, 5, 6, 8, 9, with 50 mm center spacing.

    Numbering before rotation is 1/2/3, 4/5/6, 7/8/9 (right/down).
    Cell 5 lies on the initial RGB optical axis. The episode supplies a
    random plane depth; only the in-plane orientation is sampled here.
    """
    from config.episode import PlacementError

    if context.count != 6:
        raise ValueError("gen_mkr4train requires markers.count = 6")
    grid = 0.05 * np.array([[-1, -1], [0, -1], [0, 0], [1, 0], [0, 1], [1, 1]])
    center = np.zeros(2) if context.rgb_origin_m is None else context.rgb_origin_m[:2]
    for _ in range(100):
        angle = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        xy = grid @ np.array([[c, s], [-s, c]]) + center
        points = np.column_stack((xy, np.full(6, context.plane_depth_m)))
        if all(context.accepts(point, points[:i]) for i, point in enumerate(points)):
            return points
    raise PlacementError("Cannot fit the 50 mm training grid at this depth; "
                         "increase depth/marker_fovy or reduce min_gap_m")


def marker_ring(context, rng):
    """Custom coplanar positions, using count/depth supplied by episode settings."""
    phase = rng.uniform(0, 2 * np.pi)
    theta = phase + np.arange(context.count) * 2 * np.pi / context.count
    radius = np.min(context.half_extent_m) * 0.65
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta),
                            np.full(context.count, context.plane_depth_m)))
