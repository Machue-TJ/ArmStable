"""Editable callbacks shared by camera_demo and RL; all time values are seconds."""
import numpy as np

from config.flobase.piper_base import BasePose, BaseVelocity


def base_pose(time_s):
    """Absolute world position and orientation; speed is their time derivative."""
    return BasePose.from_rpy([0.03 * np.sin(time_s), 0, 0.1],
                             [0, 0.04 * np.sin(time_s), 0.1 * np.sin(time_s)])


def base_velocity(time_s):
    """World velocity and angular velocity, integrated from configured base pose."""
    return BaseVelocity([0.02 * np.cos(time_s), 0, 0], [0, 0, 0.05 * np.cos(time_s)])


def marker_ring(context, rng):
    """Custom coplanar positions, using count/depth supplied by episode settings."""
    phase = rng.uniform(0, 2 * np.pi)
    theta = phase + np.arange(context.count) * 2 * np.pi / context.count
    radius = np.min(context.half_extent_m) * 0.65
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta),
                            np.full(context.count, context.plane_depth_m)))
