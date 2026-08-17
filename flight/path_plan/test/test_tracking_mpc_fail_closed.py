from types import SimpleNamespace

import numpy as np

import path_plan.mpc as mpc_module
from path_plan.mpc import TrackingMPC


def test_failed_solve_brakes_without_replaying_stale_warm_start(monkeypatch):
    def fail_minimize(_fun, x0, **_kwargs):
        return SimpleNamespace(success=False, x=np.full_like(x0, 99.0))

    monkeypatch.setattr(mpc_module, "minimize", fail_minimize)
    mpc = TrackingMPC(dt_s=0.1, horizon=4, a_max=2.0)
    mpc._warm.fill(1.75)
    velocity = np.array([0.4, -0.8, 0.0])
    reference = np.zeros((mpc.N, 3))

    result = mpc.solve(np.zeros(3), velocity, reference, reference)

    expected_brake = np.clip(
        -velocity / (mpc.N * mpc.dt), -mpc.a_max, mpc.a_max)
    assert not result.success
    assert np.all(np.isfinite(result.acceleration_cmd))
    assert np.all(np.abs(result.acceleration_cmd) <= mpc.a_max)
    assert np.allclose(result.acceleration_cmd, expected_brake)
    assert np.allclose(result.predicted_acc, expected_brake)
    assert np.all(np.isfinite(result.predicted_pos))
    assert np.all(np.isfinite(result.predicted_vel))
    assert np.allclose(mpc._warm, 0.0)


def test_reset_clears_warm_start():
    mpc = TrackingMPC(horizon=3)
    mpc._warm[:] = np.arange(9, dtype=float).reshape(3, 3) + 1.0
    mpc._a_prev[:] = [1.0, 2.0, 3.0]

    mpc.reset()

    assert np.allclose(mpc._warm, 0.0)
    assert np.allclose(mpc._a_prev, 0.0)


def test_first_tracking_acceleration_ramps_with_the_jerk_limit():
    mpc = TrackingMPC(
        dt_s=0.1, horizon=10, v_max=5.0, a_max=3.0, j_max=2.0)
    reference_position = np.full((mpc.N, 3), [20.0, 20.0, 0.0])
    reference_velocity = np.zeros((mpc.N, 3))

    first = mpc.solve(
        np.zeros(3), np.zeros(3),
        reference_position, reference_velocity)
    second = mpc.solve(
        np.zeros(3), np.zeros(3),
        reference_position, reference_velocity)

    assert first.success and second.success
    jerk_step = mpc.j_max * mpc.dt
    assert np.all(np.abs(first.acceleration_cmd) <= jerk_step + 1.0e-6)
    assert np.all(
        np.abs(second.acceleration_cmd - first.acceleration_cmd)
        <= jerk_step + 1.0e-6)


def test_lookahead_output_is_anchored_to_the_last_published_acceleration():
    mpc = TrackingMPC(
        dt_s=0.1, horizon=10, v_max=5.0, a_max=3.0, j_max=2.0)
    positive = np.full((mpc.N, 3), [20.0, 20.0, 0.0])
    negative = -positive
    velocity = np.zeros((mpc.N, 3))

    first = mpc.solve(
        np.zeros(3), np.zeros(3), positive, velocity,
        applied_acceleration=np.zeros(3), output_step=1)
    second = mpc.solve(
        np.zeros(3), np.zeros(3), negative, velocity,
        applied_acceleration=first.acceleration_cmd, output_step=1)

    jerk_step = mpc.j_max * mpc.dt
    assert first.success and second.success
    assert np.allclose(first.acceleration_cmd, first.predicted_acc[1])
    assert np.allclose(second.acceleration_cmd, second.predicted_acc[1])
    assert np.all(np.abs(first.acceleration_cmd) <= jerk_step + 1.0e-6)
    assert np.all(
        np.abs(second.acceleration_cmd - first.acceleration_cmd)
        <= jerk_step + 1.0e-6)
