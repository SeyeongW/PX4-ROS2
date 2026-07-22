"""Focused regression tests for an exogenous nonlinear pad horizon."""

import numpy as np
import pytest

pytest.importorskip("osqp")

from precision_landing.osqp_landing_solver import OsqpLandingSolver


def _solve(solver, **horizon_context):
    return solver.solve(
        vehicle_position_enu_m=(0.0, 0.0, 1.0),
        vehicle_velocity_enu_m_s=(1.0, 0.0, 0.0),
        vehicle_acceleration_enu_m_s2=(0.0, 0.0, 0.0),
        landing_pad_position_enu_m=(0.0, 0.0, 0.0),
        landing_pad_velocity_enu_m_s=(1.0, 0.0, 0.0),
        landing_pad_acceleration_enu_m_s2=(0.0, 1.0, 0.0),
        target_clearance_m=1.0,
        landing_pad_yaw_enu_rad=0.0,
        enforce_landing_constraints=False,
        descent_allowed=False,
        **horizon_context,
    )


def test_osqp_consumes_curved_pad_horizon_without_rebuilding_workspace():
    solver = OsqpLandingSolver(
        dt_s=0.1,
        horizon_steps=8,
        deadline_s=1.0,
        max_horizontal_velocity_m_s=12.0,
    )
    times = np.arange(1, 9, dtype=float) * 0.1
    pad_positions = np.column_stack(
        (times, 0.5 * times**2, np.zeros(8))
    )
    pad_velocities = np.column_stack(
        (np.ones(8), times, np.zeros(8))
    )
    pad_accelerations = np.column_stack(
        (np.zeros(8), np.ones(8), np.zeros(8))
    )

    result = _solve(
        solver,
        landing_pad_positions_horizon_enu_m=pad_positions,
        landing_pad_velocities_horizon_enu_m_s=pad_velocities,
        landing_pad_accelerations_horizon_enu_m_s2=pad_accelerations,
    )

    assert result.success, result.message
    np.testing.assert_allclose(result.landing_pad_positions_m, pad_positions)
    np.testing.assert_allclose(
        result.landing_pad_velocities_m_s, pad_velocities
    )
    np.testing.assert_allclose(
        result.landing_pad_accelerations_m_s2, pad_accelerations
    )
    np.testing.assert_allclose(
        result.relative_positions_m,
        result.positions_m - pad_positions,
        atol=1.0e-9,
    )
    # A scalar constant-acceleration model would not contain the supplied
    # nonlinear sequence.  The lateral command confirms it influenced q/l/u.
    assert result.velocities_m_s[1, 1] > 0.05
    assert solver.workspace_setup_count == 1
    assert solver.workspace_update_count == 1


def test_level_alignment_does_not_force_unreachable_deck_contact():
    """A fast level catch-up remains feasible before descent is authorized."""
    solver = OsqpLandingSolver(
        dt_s=0.1,
        horizon_steps=20,
        deadline_s=0.04,
        maximum_iterations=1000,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        max_horizontal_velocity_m_s=11.0,
        max_horizontal_acceleration_m_s2=3.0,
        max_jerk_m_s3=5.0,
        funnel_minimum_radius_m=0.75,
        landing_pad_length_m=5.0,
        landing_pad_width_m=5.0,
        landing_pad_contact_margin_m=0.75,
    )
    radius_m = 50.0
    speed_m_s = 9.0
    turn_rate_rad_s = speed_m_s / radius_m
    times = np.arange(1, 21, dtype=float) * 0.1
    angles = turn_rate_rad_s * times
    pad_positions = np.column_stack(
        (
            radius_m * np.cos(angles),
            radius_m * np.sin(angles),
            np.zeros(20),
        )
    )
    pad_velocities = np.column_stack(
        (
            -speed_m_s * np.sin(angles),
            speed_m_s * np.cos(angles),
            np.zeros(20),
        )
    )
    centripetal_m_s2 = speed_m_s**2 / radius_m
    pad_accelerations = np.column_stack(
        (
            -centripetal_m_s2 * np.cos(angles),
            -centripetal_m_s2 * np.sin(angles),
            np.zeros(20),
        )
    )

    result = solver.solve(
        vehicle_position_enu_m=(53.0, 0.0, 7.0),
        vehicle_velocity_enu_m_s=(1.0, speed_m_s, 0.0),
        vehicle_acceleration_enu_m_s2=(-centripetal_m_s2, 0.0, 0.0),
        landing_pad_position_enu_m=(radius_m, 0.0, 0.0),
        landing_pad_velocity_enu_m_s=(0.0, speed_m_s, 0.0),
        landing_pad_acceleration_enu_m_s2=(
            -centripetal_m_s2,
            0.0,
            0.0,
        ),
        target_clearance_m=7.0,
        landing_pad_yaw_enu_rad=0.0,
        enforce_landing_constraints=True,
        descent_allowed=False,
        landing_pad_positions_horizon_enu_m=pad_positions,
        landing_pad_velocities_horizon_enu_m_s=pad_velocities,
        landing_pad_accelerations_horizon_enu_m_s2=pad_accelerations,
    )

    assert result.success, result.message


def test_initial_acceleration_viability_envelope_recovers_with_jerk():
    solver = OsqpLandingSolver(
        dt_s=0.1,
        horizon_steps=8,
        deadline_s=1.0,
        maximum_iterations=1000,
        max_horizontal_acceleration_m_s2=3.0,
        max_jerk_m_s3=5.0,
    )
    initial_acceleration_y = 3.535
    result = solver.solve(
        vehicle_position_enu_m=(0.0, 0.0, 1.0),
        vehicle_velocity_enu_m_s=(0.0, 0.0, 0.0),
        vehicle_acceleration_enu_m_s2=(0.0, initial_acceleration_y, 0.0),
        landing_pad_position_enu_m=(0.0, 0.0, 0.0),
        landing_pad_velocity_enu_m_s=(0.0, 0.0, 0.0),
        landing_pad_acceleration_enu_m_s2=(0.0, 0.0, 0.0),
        target_clearance_m=1.0,
        landing_pad_yaw_enu_rad=0.0,
        enforce_landing_constraints=False,
        descent_allowed=False,
    )

    assert result.success, result.message
    assert 3.0 < result.accelerations_m_s2[0, 1] <= (
        initial_acceleration_y + 1.0e-6
    )
    assert np.max(result.accelerations_m_s2[1:, 1]) <= 3.0 + 2.0e-3


def test_horizontal_speed_polygon_matches_vector_command_limit():
    solver = OsqpLandingSolver(
        dt_s=0.1,
        horizon_steps=20,
        deadline_s=1.0,
        maximum_iterations=1000,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        max_horizontal_velocity_m_s=11.0,
    )
    result = solver.solve(
        vehicle_position_enu_m=(0.0, 0.0, 2.0),
        vehicle_velocity_enu_m_s=(0.0, 0.0, 0.0),
        vehicle_acceleration_enu_m_s2=(0.0, 0.0, 0.0),
        landing_pad_position_enu_m=(2.0, 2.0, 0.0),
        # This deliberately requests a diagonal vector whose components fit
        # the obsolete box while its Euclidean speed exceeds 11 m/s.
        landing_pad_velocity_enu_m_s=(9.0, 9.0, 0.0),
        landing_pad_acceleration_enu_m_s2=(0.0, 0.0, 0.0),
        target_clearance_m=2.0,
        landing_pad_yaw_enu_rad=0.0,
        enforce_landing_constraints=False,
        descent_allowed=False,
    )

    assert result.success, result.message
    horizontal_speeds = np.linalg.norm(
        result.velocities_m_s[:, :2], axis=1
    )
    assert np.max(horizontal_speeds) <= 11.0 + 2.0e-3


def test_horizontal_velocity_viability_recovers_previous_9mps_state():
    solver = OsqpLandingSolver(
        dt_s=0.1,
        horizon_steps=20,
        deadline_s=1.0,
        maximum_iterations=1000,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        max_horizontal_velocity_m_s=11.0,
        max_horizontal_acceleration_m_s2=3.0,
        max_jerk_m_s3=5.0,
    )
    result = solver.solve(
        vehicle_position_enu_m=(0.0, 0.0, 9.0),
        vehicle_velocity_enu_m_s=(-0.113, 10.628, -0.109),
        vehicle_acceleration_enu_m_s2=(-2.574, 2.010, -0.053),
        landing_pad_position_enu_m=(0.0, 0.0, 0.0),
        landing_pad_velocity_enu_m_s=(0.0, 9.0, 0.0),
        landing_pad_acceleration_enu_m_s2=(-1.62, 0.0, 0.0),
        target_clearance_m=9.0,
        landing_pad_yaw_enu_rad=0.0,
        enforce_landing_constraints=False,
        descent_allowed=False,
    )

    assert result.success, result.message
    # The production command extracts the 200 ms state. It is executable
    # even though the measured acceleration makes a later transient slightly
    # exceed the nominal boundary before the fastest jerk recovery completes.
    assert np.linalg.norm(result.velocities_m_s[1, :2]) <= 11.0 + 2.0e-3
    assert np.linalg.norm(result.velocities_m_s[-1, :2]) < 11.0


def test_vertical_velocity_viability_recovers_live_descent_boundary():
    solver = OsqpLandingSolver(
        dt_s=0.1,
        horizon_steps=20,
        deadline_s=1.0,
        maximum_iterations=1000,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        max_descent_velocity_m_s=0.7,
        max_vertical_acceleration_m_s2=3.0,
        max_jerk_m_s3=5.0,
    )
    result = solver.solve(
        vehicle_position_enu_m=(0.0, 0.0, 6.769),
        vehicle_velocity_enu_m_s=(0.0, 0.0, -0.705),
        vehicle_acceleration_enu_m_s2=(0.0, 0.0, -0.260),
        landing_pad_position_enu_m=(0.0, 0.0, 0.0),
        landing_pad_velocity_enu_m_s=(0.0, 0.0, 0.0),
        landing_pad_acceleration_enu_m_s2=(0.0, 0.0, 0.0),
        target_clearance_m=6.5,
        landing_pad_yaw_enu_rad=0.0,
        enforce_landing_constraints=False,
        descent_allowed=True,
        relative_descent_speed_m_s=0.7,
    )

    assert result.success, result.message
    # The 200 ms production extraction has already recovered inside the
    # public descent-speed command limit.
    assert result.velocities_m_s[1, 2] >= -0.7 - 2.0e-3
    assert result.velocities_m_s[-1, 2] >= -0.7 - 2.0e-3


def test_acceleration_saturated_recovery_replays_9mps_align_state():
    """The recorded high-speed ALIGN state must remain a feasible catch-up."""
    solver = OsqpLandingSolver(
        dt_s=0.1,
        horizon_steps=20,
        deadline_s=1.0,
        maximum_iterations=1000,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        max_horizontal_velocity_m_s=11.0,
        max_horizontal_acceleration_m_s2=3.0,
        max_jerk_m_s3=5.0,
    )
    result = solver.solve(
        vehicle_position_enu_m=(0.0, 0.0, 6.408),
        vehicle_velocity_enu_m_s=(-7.621, 6.567, -0.019),
        vehicle_acceleration_enu_m_s2=(-2.969, -1.628, -0.004),
        landing_pad_position_enu_m=(0.0, 0.0, 0.0),
        landing_pad_velocity_enu_m_s=(-6.517, 6.167, 0.0),
        landing_pad_acceleration_enu_m_s2=(-1.088, -1.150, 0.0),
        target_clearance_m=6.408,
        landing_pad_yaw_enu_rad=0.0,
        enforce_landing_constraints=False,
        descent_allowed=False,
    )

    assert result.success, result.message
    assert np.linalg.norm(result.velocities_m_s[1, :2]) <= 11.0 + 2.0e-3
    assert np.linalg.norm(result.velocities_m_s[-1, :2]) < 11.0


def test_osqp_scalar_pad_state_retains_constant_acceleration_fallback():
    solver = OsqpLandingSolver(
        dt_s=0.1,
        horizon_steps=5,
        deadline_s=1.0,
        max_horizontal_velocity_m_s=12.0,
    )
    result = _solve(solver)
    times = np.arange(1, 6, dtype=float) * 0.1
    expected_positions = np.column_stack(
        (times, 0.5 * times**2, np.zeros(5))
    )

    np.testing.assert_allclose(
        result.landing_pad_positions_m, expected_positions
    )
    assert solver.workspace_setup_count == 1


def test_osqp_rejects_partial_or_wrong_length_pad_horizon():
    solver = OsqpLandingSolver(
        dt_s=0.1,
        horizon_steps=5,
        deadline_s=1.0,
    )
    with pytest.raises(ValueError, match="must be provided together"):
        _solve(
            solver,
            landing_pad_positions_horizon_enu_m=np.zeros((5, 3)),
        )
    with pytest.raises(ValueError, match="shape"):
        _solve(
            solver,
            landing_pad_positions_horizon_enu_m=np.zeros((4, 3)),
            landing_pad_velocities_horizon_enu_m_s=np.zeros((4, 3)),
            landing_pad_accelerations_horizon_enu_m_s2=np.zeros((4, 3)),
        )
