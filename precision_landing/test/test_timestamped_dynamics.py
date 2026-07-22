"""Focused tests for timestamped vehicle and trailer dynamics."""

import math

import numpy as np
import pytest

from precision_landing.timestamped_dynamics import (
    TrailerPredictionParameters,
    TrailerRk4Predictor,
    VehicleResponseEstimator,
    VehicleResponseParameters,
)


def _trailer_state(
    predictor,
    *,
    sample_time_s=10.0,
    position=(0.0, 0.0, 0.0),
    velocity=(5.0, 0.0, 0.0),
    yaw=0.0,
    yaw_rate=0.0,
    acceleration=0.0,
    covariance=None,
):
    if covariance is None:
        covariance = np.eye(4, dtype=float) * 0.01
    return predictor.state_from_observation(
        sample_time_s,
        position,
        velocity,
        yaw,
        yaw_rate,
        covariance,
        acceleration,
    )


def _advance_first_order_truth(position, velocity, acceleration, command, dt, tau):
    decay = math.exp(-dt / tau)
    velocity_gain = tau * (1.0 - decay)
    position_gain = tau * dt - tau * tau * (1.0 - decay)
    next_position = (
        position
        + velocity * dt
        + position_gain * acceleration
        + (0.5 * dt * dt - position_gain) * command
    )
    next_velocity = (
        velocity
        + velocity_gain * acceleration
        + (dt - velocity_gain) * command
    )
    next_acceleration = command + (acceleration - command) * decay
    return next_position, next_velocity, next_acceleration


def test_vehicle_kalman_uses_irregular_sample_timestamps_and_covariance_is_psd():
    parameters = VehicleResponseParameters(
        actuator_time_constant_s=0.20,
        innovation_nis_gate=100.0,
    )
    estimator = VehicleResponseEstimator(parameters)
    time_s = 100.0
    position = velocity = acceleration = 0.0
    command = 1.5
    first = estimator.update(
        time_s, (position, 0.0, 0.0), (velocity, 0.0, 0.0),
        (command, 0.0, 0.0),
    )
    assert first.accepted

    for dt in (0.017, 0.043, 0.021, 0.036, 0.019, 0.052, 0.025):
        position, velocity, acceleration = _advance_first_order_truth(
            position,
            velocity,
            acceleration,
            command,
            dt,
            parameters.actuator_time_constant_s,
        )
        time_s += dt
        result = estimator.update(
            time_s,
            (position, 0.0, 0.0),
            (velocity, 0.0, 0.0),
            (command, 0.0, 0.0),
        )
        assert result.accepted

    assert result.state is not None
    assert result.state.position_enu_m[0] == pytest.approx(position, abs=2e-3)
    assert result.state.velocity_enu_m_s[0] == pytest.approx(velocity, abs=1e-2)
    covariance = estimator.covariance
    assert covariance is not None
    assert covariance.shape == (3, 4, 4)
    for axis_covariance in covariance:
        np.testing.assert_allclose(
            axis_covariance, axis_covariance.T, atol=1e-12
        )
        assert np.min(np.linalg.eigvalsh(axis_covariance)) >= -1e-12

    # Public diagnostic covariance is a defensive copy.
    covariance[0, 0, 0] = 1.0e9
    assert estimator.covariance[0, 0, 0] < 1.0


def test_vehicle_constant_disturbance_converges_without_imu():
    estimator = VehicleResponseEstimator(
        VehicleResponseParameters(innovation_nis_gate=100.0)
    )
    time_s = 10.0
    position = velocity = 0.0
    true_disturbance = 1.2
    estimator.update(
        time_s, (position, 0.0, 0.0), (velocity, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    irregular_steps = (0.017, 0.031, 0.022, 0.040, 0.019)
    for index in range(300):
        dt = irregular_steps[index % len(irregular_steps)]
        position += velocity * dt + 0.5 * true_disturbance * dt * dt
        velocity += true_disturbance * dt
        time_s += dt
        result = estimator.update(
            time_s,
            (position, 0.0, 0.0),
            (velocity, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        assert result.accepted

    assert result.state is not None
    assert result.state.disturbance_enu_m_s2[0] == pytest.approx(
        true_disturbance, abs=0.03
    )
    assert result.state.acceleration_enu_m_s2[0] == pytest.approx(
        true_disturbance, abs=0.03
    )


def test_vehicle_nis_gate_rejects_outlier_without_posterior_jump():
    estimator = VehicleResponseEstimator(
        VehicleResponseParameters(innovation_nis_gate=16.0)
    )
    estimator.update(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    accepted = estimator.update(
        1.1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    assert accepted.accepted

    outlier = estimator.update(
        1.2, (100.0, -100.0, 0.0), (50.0, -50.0, 0.0)
    )
    assert not outlier.accepted
    assert outlier.reason == "innovation_gate_rejected"
    assert outlier.state is not None
    np.testing.assert_allclose(
        outlier.state.position_enu_m, (0.0, 0.0, 0.0), atol=1e-10
    )
    np.testing.assert_allclose(
        outlier.state.velocity_enu_m_s, (0.0, 0.0, 0.0), atol=1e-10
    )
    assert estimator.last_nis is not None
    assert np.max(estimator.last_nis) > 16.0

    recovered = estimator.update(
        1.3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    assert recovered.accepted


def test_vehicle_rejects_duplicate_out_of_order_reset_and_large_gap():
    estimator = VehicleResponseEstimator(
        VehicleResponseParameters(
            timestamp_reset_threshold_s=0.5,
            maximum_sample_gap_s=0.4,
        )
    )
    estimator.update(10.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    duplicate = estimator.update(
        10.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    out_of_order = estimator.update(
        9.8, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    assert not duplicate.accepted
    assert duplicate.reason == "duplicate_timestamp"
    assert not out_of_order.accepted
    assert out_of_order.reason == "out_of_order_timestamp"
    assert estimator.initialized

    reset = estimator.update(
        9.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    assert not reset.accepted
    assert reset.reset_detected
    assert reset.reason == "timestamp_reset"
    assert not estimator.initialized

    estimator.update(20.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    gap = estimator.update(
        20.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    assert not gap.accepted
    assert gap.reset_detected
    assert gap.reason == "large_sample_gap"
    assert not estimator.initialized


def test_vehicle_response_reanchors_after_consecutive_nis_rejections():
    estimator = VehicleResponseEstimator(
        VehicleResponseParameters(innovation_nis_gate=0.1)
    )
    estimator.update(10.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    for index in range(1, 5):
        rejected = estimator.update(
            10.0 + 0.02 * index,
            (100.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        assert not rejected.accepted
        assert rejected.reason == "innovation_gate_rejected"

    recovered = estimator.update(
        10.10,
        (100.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    assert recovered.accepted
    assert recovered.reason == "innovation_reinitialized"
    assert estimator.consecutive_innovation_rejections == 0
    assert recovered.state is not None
    np.testing.assert_allclose(
        recovered.state.position_enu_m,
        (100.0, 0.0, 0.0),
    )


def test_vehicle_forward_prediction_models_lag_without_mutating_state():
    parameters = VehicleResponseParameters(
        actuator_time_constant_s=0.20,
        maximum_prediction_horizon_s=0.30,
    )
    estimator = VehicleResponseEstimator(parameters)
    estimator.update(
        10.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    )

    result = estimator.predict(10.2)
    assert result.valid
    assert result.state is not None
    assert result.state.acceleration_enu_m_s2[0] == pytest.approx(
        2.0 * (1.0 - math.exp(-1.0))
    )
    assert result.state.velocity_enu_m_s[0] > 0.0
    assert result.state.position_enu_m[0] > 0.0
    assert estimator.state is not None
    assert estimator.state.sample_time_s == 10.0

    stale = estimator.predict(10.31)
    past = estimator.predict(9.9)
    assert not stale.valid
    assert stale.reason == "prediction_horizon_exceeded"
    assert not past.valid
    assert past.reason == "prediction_before_state"


def test_trailer_observation_infers_signed_speed_and_bounded_steering():
    predictor = TrailerRk4Predictor(
        TrailerPredictionParameters(
            wheelbase_m=2.5,
            maximum_steering_rad=0.4,
            minimum_steering_speed_m_s=0.2,
        )
    )
    reverse = _trailer_state(
        predictor,
        velocity=(-3.0, 0.0, 0.0),
        yaw=0.0,
        yaw_rate=2.0,
    )
    stopped = _trailer_state(
        predictor,
        velocity=(0.01, 0.0, 0.0),
        yaw=0.0,
        yaw_rate=10.0,
    )

    assert reverse.speed_m_s == pytest.approx(-3.0)
    assert reverse.steering_rad == pytest.approx(-0.4)
    assert stopped.steering_rad == 0.0


def test_trailer_rk4_straight_constant_speed_prediction():
    predictor = TrailerRk4Predictor(
        TrailerPredictionParameters(maximum_measurement_age_s=3.0)
    )
    state = _trailer_state(predictor)
    result = predictor.predict(state, 12.0)

    assert result.valid
    assert result.state is not None
    np.testing.assert_allclose(
        result.state.position_enu_m, (10.0, 0.0, 0.0), atol=1e-8
    )
    np.testing.assert_allclose(
        result.velocity_enu_m_s, (5.0, 0.0, 0.0), atol=1e-8
    )
    assert result.yaw_rate_rad_s == pytest.approx(0.0)


def test_trailer_rk4_turn_matches_constant_curvature_solution():
    predictor = TrailerRk4Predictor(
        TrailerPredictionParameters(
            wheelbase_m=2.5,
            maximum_measurement_age_s=2.0,
            maximum_integration_step_s=0.01,
        )
    )
    state = _trailer_state(predictor, yaw_rate=1.0)
    result = predictor.predict(state, 11.0)

    assert result.valid
    assert result.state is not None
    assert result.state.position_enu_m[0] == pytest.approx(
        5.0 * math.sin(1.0), abs=1e-7
    )
    assert result.state.position_enu_m[1] == pytest.approx(
        5.0 * (1.0 - math.cos(1.0)), abs=1e-7
    )
    assert result.state.yaw_enu_rad == pytest.approx(1.0, abs=1e-8)
    assert result.yaw_rate_rad_s == pytest.approx(1.0, abs=1e-8)


def test_trailer_rk4_horizon_matches_constant_curvature_samples():
    predictor = TrailerRk4Predictor(
        TrailerPredictionParameters(
            wheelbase_m=2.5,
            maximum_measurement_age_s=2.0,
            maximum_integration_step_s=0.01,
        )
    )
    state = _trailer_state(predictor, yaw_rate=1.0)
    horizon = predictor.predict_horizon(state, 10.0, 0.1, 10)

    assert horizon.valid
    assert horizon.positions_enu_m is not None
    assert horizon.velocities_enu_m_s is not None
    assert horizon.accelerations_enu_m_s2 is not None
    assert horizon.yaws_enu_rad is not None
    assert horizon.yaw_rates_rad_s is not None
    assert horizon.covariances is not None
    times = np.arange(1, 11, dtype=float) * 0.1
    expected_positions = np.column_stack(
        (
            5.0 * np.sin(times),
            5.0 * (1.0 - np.cos(times)),
            np.zeros(10),
        )
    )
    np.testing.assert_allclose(
        horizon.positions_enu_m, expected_positions, atol=1.0e-7
    )
    np.testing.assert_allclose(
        horizon.yaws_enu_rad, times, atol=1.0e-8
    )
    np.testing.assert_allclose(horizon.yaw_rates_rad_s, 1.0, atol=1.0e-8)
    np.testing.assert_allclose(
        np.linalg.norm(horizon.velocities_enu_m_s[:, :2], axis=1),
        5.0,
        atol=1.0e-8,
    )
    assert horizon.covariances.shape == (10, 4, 4)
    for covariance in horizon.covariances:
        np.testing.assert_allclose(covariance, covariance.T, atol=1.0e-12)
        assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-12


def test_trailer_rk4_horizon_keeps_capture_age_rejection_fail_closed():
    predictor = TrailerRk4Predictor(
        TrailerPredictionParameters(maximum_measurement_age_s=1.0)
    )
    state = _trailer_state(predictor, sample_time_s=10.0)

    stale = predictor.predict_horizon(state, 11.01, 0.1, 20)
    invalid_grid = predictor.predict_horizon(state, 10.0, 0.0, 20)

    assert not stale.valid
    assert stale.reason == "stale_measurement"
    assert not invalid_grid.valid
    assert invalid_grid.reason == "invalid_horizon"


def test_trailer_rk4_kinematic_horizon_matches_full_horizon():
    predictor = TrailerRk4Predictor(
        TrailerPredictionParameters(
            wheelbase_m=2.5,
            maximum_measurement_age_s=2.0,
            maximum_integration_step_s=0.02,
        )
    )
    state = _trailer_state(
        predictor, yaw_rate=0.7, acceleration=0.25
    )
    full = predictor.predict_horizon(state, 10.0, 0.1, 20)
    kinematic = predictor.predict_horizon(
        state, 10.0, 0.1, 20, propagate_covariance=False
    )

    assert full.valid and kinematic.valid
    assert full.covariances is not None
    assert kinematic.covariances is None
    np.testing.assert_allclose(
        kinematic.positions_enu_m, full.positions_enu_m, atol=1.0e-12
    )
    np.testing.assert_allclose(
        kinematic.velocities_enu_m_s,
        full.velocities_enu_m_s,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kinematic.accelerations_enu_m_s2,
        full.accelerations_enu_m_s2,
        atol=1.0e-12,
    )


def test_trailer_rk4_wraps_yaw_and_grows_psd_covariance():
    predictor = TrailerRk4Predictor(
        TrailerPredictionParameters(
            maximum_measurement_age_s=2.0,
            maximum_integration_step_s=0.02,
        )
    )
    initial_covariance = np.eye(4, dtype=float) * 1.0e-6
    state = _trailer_state(
        predictor,
        yaw=math.pi - 0.05,
        yaw_rate=0.5,
        covariance=initial_covariance,
    )
    result = predictor.predict(state, 11.0)

    assert result.valid
    assert result.state is not None
    assert -math.pi <= result.state.yaw_enu_rad < math.pi
    assert np.trace(result.state.covariance) > np.trace(initial_covariance)
    np.testing.assert_allclose(
        result.state.covariance,
        result.state.covariance.T,
        atol=1e-12,
    )
    assert np.min(np.linalg.eigvalsh(result.state.covariance)) >= -1e-12


def test_trailer_prediction_rejects_stale_future_and_invalid_covariance():
    predictor = TrailerRk4Predictor(
        TrailerPredictionParameters(
            maximum_measurement_age_s=1.0,
            future_tolerance_s=0.02,
        )
    )
    state = _trailer_state(predictor)

    stale = predictor.predict(state, 11.01)
    future = predictor.predict(state, 9.97)
    assert not stale.valid
    assert stale.reason == "stale_measurement"
    assert not future.valid
    assert future.reason == "measurement_from_future"

    invalid_covariance = np.eye(4, dtype=float)
    invalid_covariance[0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        _trailer_state(predictor, covariance=invalid_covariance)


def test_small_future_skew_is_tolerated_without_backward_integration():
    predictor = TrailerRk4Predictor(
        TrailerPredictionParameters(future_tolerance_s=0.02)
    )
    state = _trailer_state(predictor, sample_time_s=10.0)
    result = predictor.predict(state, 9.99)

    assert result.valid
    assert result.reason == "current"
    assert result.age_s == 0.0
    assert result.state is not None
    np.testing.assert_allclose(
        result.state.position_enu_m, state.position_enu_m
    )
