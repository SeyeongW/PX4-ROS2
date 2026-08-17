import math

import numpy as np
import pytest

from landing_mpc.mpc import LandingMPC
from landing_mpc.parameter_utils import (
    DEFAULT_DECK_Z_M,
    DEFAULT_MARKER_IDS,
    DEFAULT_MARKER_OFFSETS_M,
    DEFAULT_MARKER_SIZES_M,
    derive_control_timing,
    require_between,
    require_leq,
    require_nonnegative,
    require_positive,
    select_control_velocity,
    validate_marker_ladder,
)
from landing_mpc.predictor import predict_const_vel
from landing_mpc.reference import HorizonReference


def test_current_timing_defaults_are_unchanged():
    control_dt, mpc_dt, solve_every = derive_control_timing(50.0, 10.0)
    assert control_dt == pytest.approx(0.02)
    assert mpc_dt == pytest.approx(0.1)
    assert solve_every == 5


def test_mpc_reference_and_predictor_share_the_derived_mpc_period():
    _, mpc_dt, _ = derive_control_timing(50.0, 10.0)
    mpc = LandingMPC(dt_s=mpc_dt, horizon=3)
    reference = HorizonReference(lead_s=mpc_dt)
    predicted_position, _, _ = predict_const_vel(
        np.zeros(3), np.array([2.0, 0.0, 0.0]), mpc_dt, 3)

    assert mpc.dt == pytest.approx(mpc_dt)
    assert reference.lead == pytest.approx(mpc_dt)
    assert predicted_position[0, 0] == pytest.approx(2.0 * mpc_dt)


def test_receding_descent_respects_the_vertical_speed_limit():
    mpc = LandingMPC(
        dt_s=0.1, horizon=20, v_max=1.0, vz_max=0.35,
        a_max=1.0, j_max=2.0)
    position = np.array([0.02, 0.02, 5.25])
    velocity = np.zeros(3)
    target_p, target_v, target_a = predict_const_vel(
        np.zeros(3), np.zeros(3), mpc.dt, mpc.N)

    for _ in range(20):
        result = mpc.solve(
            position, velocity, target_p, target_v, target_a)
        assert result.success
        assert np.min(result.pred_rel_vel[:, 2]) >= -mpc.vz_max - 1.0e-6
        position += velocity * mpc.dt + 0.5 * result.acc_cmd * mpc.dt ** 2
        velocity += result.acc_cmd * mpc.dt
        assert velocity[2] >= -mpc.vz_max - 1.0e-6


def test_landing_lookahead_acceleration_anchors_the_actual_output_step():
    mpc = LandingMPC(
        dt_s=0.1, horizon=10, v_max=3.5, a_max=1.0,
        j_max=2.0, cone_k=0.0, z_ref=0.0)
    target_p, target_v, target_a = predict_const_vel(
        np.zeros(3), np.zeros(3), mpc.dt, mpc.N)
    applied = np.zeros(3)

    first = mpc.solve(
        np.array([5.0, 5.0, 1.0]), np.zeros(3),
        target_p, target_v, target_a,
        applied_acceleration=applied, output_step=1)
    second = mpc.solve(
        np.array([-5.0, -5.0, 1.0]), np.zeros(3),
        target_p, target_v, target_a,
        applied_acceleration=first.accel_cmd, output_step=1)

    jerk_step = mpc.j_max * mpc.dt
    assert first.success and second.success
    assert np.allclose(first.accel_cmd, first.pred_rel_acc[1])
    assert np.allclose(second.accel_cmd, second.pred_rel_acc[1])
    assert np.all(np.abs(first.accel_cmd - applied) <= jerk_step + 1.0e-6)
    assert np.all(
        np.abs(second.accel_cmd - first.accel_cmd)
        <= jerk_step + 1.0e-6)


def test_landing_lookahead_velocity_envelope_respects_the_output_anchor():
    mpc = LandingMPC(
        dt_s=0.1, horizon=20, v_max=3.5, vz_max=0.6,
        a_max=1.0, j_max=2.0, cone_k=0.0, z_ref=1.5)
    target_p, target_v, target_a = predict_const_vel(
        np.zeros(3), np.zeros(3), mpc.dt, mpc.N)
    applied = np.array([0.0, 0.0, -0.8])

    result = mpc.solve(
        np.array([0.0, 0.0, 5.028]),
        np.array([0.0, 0.0, -0.461]),
        target_p, target_v, target_a,
        applied_acceleration=applied, output_step=1)

    assert result.success
    assert np.all(
        np.abs(result.accel_cmd - applied)
        <= mpc.j_max * mpc.dt + 1.0e-6)


def test_landing_handoff_slews_back_inside_its_acceleration_limit():
    mpc = LandingMPC(
        dt_s=0.1, horizon=20, v_max=3.5, vz_max=0.6,
        a_max=1.0, j_max=2.0, cone_k=0.0, z_ref=1.5)
    target_p, target_v, target_a = predict_const_vel(
        np.zeros(3), np.zeros(3), mpc.dt, mpc.N)
    applied = np.array([-1.51, 0.0, 0.0])

    result = mpc.solve(
        np.array([3.0, 0.0, 1.5]), np.zeros(3),
        target_p, target_v, target_a,
        applied_acceleration=applied, output_step=1)

    assert result.success
    assert abs(result.accel_cmd[0] - applied[0]) <= (
        mpc.j_max * mpc.dt + 1.0e-6)
    limits = mpc._acceleration_limits(applied[0], output_step=1)
    assert np.all(np.abs(result.pred_rel_acc[:, 0]) <= limits + 1.0e-6)
    assert abs(result.pred_rel_acc[3, 0]) <= mpc.a_max + 1.0e-6


def test_acquire_brakes_before_a_reversing_target_without_crossing():
    mpc = LandingMPC(
        dt_s=0.1, horizon=20, w_vxy=20.0, v_max=3.5,
        a_max=3.0, j_max=2.0, cone_k=0.0, z_ref=1.5)
    drone_p = np.array([-9.0, 0.0, 5.0])
    drone_v = np.array([3.64, 0.0, 0.0])
    target_p = np.zeros(3)
    target_v = np.array([1.0, 0.0, 0.0])
    applied = np.zeros(3)
    captured_at = None

    for step in range(50):
        prediction = predict_const_vel(
            target_p, target_v, mpc.dt, mpc.N)
        result = mpc.solve(
            drone_p - target_p, drone_v - target_v, *prediction,
            applied_acceleration=applied, output_step=1)
        assert result.success
        applied = result.accel_cmd
        target_accel = (
            np.array([-3.0, 0.0, 0.0])
            if 20 <= step < 27 else np.zeros(3))
        drone_p += drone_v * mpc.dt + 0.5 * applied * mpc.dt ** 2
        drone_v += applied * mpc.dt
        target_p += target_v * mpc.dt + 0.5 * target_accel * mpc.dt ** 2
        target_v += target_accel * mpc.dt
        relative_position = drone_p[0] - target_p[0]
        relative_velocity = drone_v[0] - target_v[0]
        assert relative_position <= 1.0e-3
        if abs(relative_position) <= 0.5 and abs(relative_velocity) <= 0.3:
            captured_at = (step + 1) * mpc.dt
            break

    assert captured_at is not None and captured_at <= 5.0


@pytest.mark.parametrize(
    ('control_rate', 'mpc_rate'),
    [(0.0, 10.0), (50.0, 0.0), (10.0, 50.0), (50.0, 7.0)],
)
def test_invalid_timing_is_rejected(control_rate, mpc_rate):
    with pytest.raises(ValueError):
        derive_control_timing(control_rate, mpc_rate)


def test_current_marker_ladder_defaults_are_valid():
    validate_marker_ladder(
        DEFAULT_MARKER_IDS,
        DEFAULT_MARKER_SIZES_M,
        DEFAULT_MARKER_OFFSETS_M,
    )
    assert math.isfinite(DEFAULT_DECK_Z_M)


@pytest.mark.parametrize(
    ('ids', 'sizes', 'offsets'),
    [
        ([], [], []),
        ([0, 0], [1.3, 1.3], [1.1, 0.0, -1.1, 0.0]),
        ([0], [0.0], [0.0, 0.0]),
        ([0], [1.3], [0.0]),
        ([0, 1], [1.3], [0.0, 0.0, 0.0, 0.0]),
        ([0], [1.3], [float('nan'), 0.0]),
    ],
)
def test_invalid_marker_ladders_are_rejected(ids, sizes, offsets):
    with pytest.raises(ValueError):
        validate_marker_ladder(ids, sizes, offsets)


def test_scalar_validation_contract():
    assert require_positive('rate', 1.0) == 1.0
    assert require_nonnegative('margin', 0.0) == 0.0
    assert require_between('quality', 50, 1, 100) == 50.0
    require_leq('settle', 0.5, 'handoff', 1.0)

    for value in (0.0, -1.0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            require_positive('rate', value)
    with pytest.raises(ValueError):
        require_nonnegative('margin', -0.1)
    with pytest.raises(ValueError):
        require_between('quality', 101, 1, 100)
    with pytest.raises(ValueError):
        require_leq('settle', 1.1, 'handoff', 1.0)


def test_deprecated_marker_velocity_does_not_change_control_feedforward():
    cue_velocity = np.array([3.0, -0.2, 0.0])
    for marker_velocity in (
            np.array([0.0, 0.0, 0.0]),
            np.array([99.0, -99.0, 50.0]),
            None):
        selected = select_control_velocity(cue_velocity, marker_velocity)
        assert selected is cue_velocity
