import math

import numpy as np

from path_plan.mpc_reference import (
    _s_curve_stop_speed,
    limit_acceleration_slew,
    path_reference_horizon,
)


def test_path_reference_brakes_at_the_bspline_endpoint():
    arc = np.array([0.0, 5.0, 10.0])
    path = np.column_stack((arc, np.zeros(3), np.ones(3) * 5.0))
    positions, velocities = path_reference_horizon(
        arc, path, 9.7, 0.1, 20, 3.0, 3.0, 2.0)

    assert positions.shape == velocities.shape == (20, 3)
    assert np.all(np.diff(positions[:, 0]) >= -1.0e-12)
    assert np.all(positions[:, 0] <= 10.0)
    assert np.allclose(positions[-1], [10.0, 0.0, 5.0])
    assert np.allclose(velocities[-1], 0.0)
    stop_distance = 3.0 ** 1.5 / math.sqrt(2.0)
    assert np.isclose(_s_curve_stop_speed(stop_distance, 3.0, 2.0), 3.0)
    assert _s_curve_stop_speed(1.5, 3.0, 2.0) < 3.0


def test_moving_target_relative_braking_stays_on_the_bspline():
    arc = np.array([0.0, 30.0])
    path = np.array([[0.0, 0.0, 5.0], [30.0, 0.0, 5.0]])
    positions, velocities = path_reference_horizon(
        arc, path, 0.0, 0.1, 20, 3.0, 3.0, 2.0,
        target_velocity_xy=np.array([1.0, 0.0]),
        target_range_xy_m=8.0, relative_brake_start_m=10.0)

    relative_speed = np.linalg.norm(
        velocities[:, :2] - np.array([1.0, 0.0]), axis=1)
    assert np.all(relative_speed <= 0.3 + 1.0e-12)
    steps = np.diff(np.vstack(([0.0, 0.0, 5.0], positions)), axis=0)
    assert np.allclose(steps / 0.1, velocities, atol=1.0e-12)


def test_streamed_acceleration_is_slew_limited_at_control_rate():
    acceleration = np.zeros(3)
    desired = np.array([0.2, -0.2, 0.1])
    samples = []
    for _ in range(5):
        acceleration = limit_acceleration_slew(
            acceleration, desired, jerk_m_s3=2.0, elapsed_s=0.02)
        samples.append(acceleration.copy())
    samples = np.asarray(samples)
    assert np.all(np.abs(np.diff(
        np.vstack((np.zeros(3), samples)), axis=0)) <= 0.04 + 1.0e-12)
    assert np.allclose(samples[-1], desired)
