import math
import time

import numpy as np
import pytest

from autonomy_planner.mission_core import (
    JerkInputMPC,
    MissionCommandGate,
    MissionState,
    RelativeTargetEstimator,
    TouchdownGate,
    TrailerKinematicState,
    body_velocity_to_enu,
    down_optical_to_flu,
    enu_to_ned_vector,
    flu_to_enu_with_px4_quaternion,
    front_optical_to_flu,
    iterative_polyline_intercept,
    iterative_trailer_intercept,
    marker_position_to_flu,
    ned_to_enu_vector,
)


def test_frame_conversions_and_camera_signs():
    enu = np.array([4.0, -2.0, 7.0])
    assert np.allclose(ned_to_enu_vector(enu_to_ned_vector(enu)), enu)
    assert np.allclose(front_optical_to_flu([2.0, 3.0, 5.0]), [5.0, -2.0, -3.0])
    assert np.allclose(down_optical_to_flu([2.0, 3.0, 5.0]), [-3.0, -2.0, -5.0])


def test_marker_frame_is_normalized_exactly_once_and_unknown_frame_rejected():
    base_link = marker_position_to_flu([2.0, 3.0, 5.0], "base_link", "front")
    assert np.allclose(base_link, [2.0, 3.0, 5.0])
    assert np.allclose(
        marker_position_to_flu(
            [2.0, 3.0, 5.0], "front_camera_optical_frame", "front"
        ),
        [5.0, -2.0, -3.0],
    )
    with np.testing.assert_raises(ValueError):
        marker_position_to_flu([1.0, 2.0, 3.0], "map", "front")


def test_child_frame_trailer_velocity_is_rotated_to_world_enu():
    assert np.allclose(
        body_velocity_to_enu([3.0, 0.0, 0.0], math.pi / 2.0),
        [0.0, 3.0, 0.0],
        atol=1.0e-12,
    )


def test_px4_frd_attitude_rotates_full_flu_vector_to_world_enu():
    # Identity PX4 attitude means body forward points North, body right East,
    # and body down Earth-down.  FLU left is therefore West in ENU.
    assert np.allclose(
        flu_to_enu_with_px4_quaternion([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0]),
        [-2.0, 1.0, 3.0],
    )


def test_trailer_motion_and_iterative_intercept_endpoint_clamp():
    start = np.array([200.0, -125.0, 0.0])
    goal = np.array([-128.0, -128.0, 0.0])
    direction = (goal - start) / np.linalg.norm(goal - start)
    state = TrailerKinematicState(0.0, start, 3.0 * direction, np.zeros(3))
    assert math.isclose(np.linalg.norm(state.velocity_enu_m_s), 3.0)
    predicted = state.predict(10.0)
    assert np.allclose(predicted.position_enu_m, start + state.velocity_enu_m_s * 10.0)
    solution = iterative_trailer_intercept(
        state,
        lambda target: float(np.linalg.norm(target - np.array([-120.0, 115.0, 10.0]))),
        5.0,
        endpoint_enu_m=goal,
    )
    assert 0.0 <= solution.eta_s <= 120.0
    route = goal - start
    ratio = np.dot(solution.position_enu_m - start, route) / np.dot(route, route)
    assert 0.0 <= ratio <= 1.0


def test_polyline_intercept_follows_safe_route_instead_of_diagonal_chord():
    route = np.asarray(
        ((200.0, -125.0), (200.0, -100.0), (-128.0, -100.0), (-128.0, -128.0))
    )
    trailer = TrailerKinematicState(
        0.0,
        np.asarray((200.0, -125.0, 0.0)),
        np.zeros(3),
        np.zeros(3),
    )
    solution = iterative_polyline_intercept(
        trailer,
        route,
        lambda _target: 100.0,
        5.0,
        planner_latency_s=8.0,
        iterations=3,
    )
    # At 28 s predicted time the route has completed its short north segment
    # and is travelling west on y=-100, never on the unsafe endpoint chord.
    assert solution.position_enu_m[0] < 200.0
    assert solution.position_enu_m[1] == pytest.approx(-100.0)
    assert solution.trailer_velocity_enu_m_s[0] == pytest.approx(-3.0)


def test_jerk_mpc_exact_rollout_and_bounds():
    mpc = JerkInputMPC(dt_s=0.2, horizon_steps=5, deadline_s=1.0)
    jerk = np.repeat(np.array([[1.0, 0.0, 0.0]]), 5, axis=0)
    p, v, a = mpc.rollout(np.zeros(3), np.zeros(3), np.zeros(3), jerk)
    assert np.allclose(a[-1], [1.0, 0.0, 0.0])
    assert np.allclose(v[-1], [0.5, 0.0, 0.0])
    assert np.allclose(p[-1], [1.0 / 6.0, 0.0, 0.0])
    reference = np.repeat(np.array([[0.8, 0.0, 0.0]]), 5, axis=0)
    result = mpc.solve(np.zeros(3), np.zeros(3), np.zeros(3), reference, np.zeros_like(reference))
    assert np.max(np.abs(result.jerks_m_s3)) <= mpc.jerk_limit_m_s3 + 1e-6
    assert np.max(np.abs(result.accelerations_m_s2)) <= mpc.acceleration_limit_m_s2 + 1e-5


def test_jerk_mpc_infeasible_corridor_falls_back_without_claiming_success():
    mpc = JerkInputMPC(dt_s=0.2, horizon_steps=4, deadline_s=1.0)
    impossible = np.array([10.0, 11.0, 10.0, 11.0, 10.0, 11.0])
    result = mpc.solve(
        np.zeros(3), np.zeros(3), np.zeros(3),
        np.ones((4, 3)), np.zeros((4, 3)), impossible,
    )
    assert not result.success
    assert "HOLD_REPLAN" in result.message


def test_jerk_mpc_respects_vector_limits_and_hard_corridor():
    mpc = JerkInputMPC(
        dt_s=0.1,
        horizon_steps=20,
        deadline_s=1.0,
        maximum_iterations=30,
    )
    reference = np.column_stack(
        (np.linspace(0.05, 1.5, 20), np.zeros(20), np.zeros(20))
    )
    corridor = np.tile(
        np.array([-0.1, 2.0, -0.25, 0.25, -0.25, 0.25]), (20, 1)
    )
    result = mpc.solve(
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        reference,
        np.zeros_like(reference),
        corridor,
    )
    assert result.success
    assert np.max(np.linalg.norm(result.velocities_m_s, axis=1)) <= 5.0 + 1e-5
    assert np.max(np.linalg.norm(result.accelerations_m_s2, axis=1)) <= 3.0 + 1e-5
    assert np.all(result.positions_m >= corridor[:, (0, 2, 4)] - 1e-5)
    assert np.all(result.positions_m <= corridor[:, (1, 3, 5)] + 1e-5)


def test_jerk_mpc_enforces_unmapped_obstacle_sphere():
    mpc = JerkInputMPC(
        dt_s=0.1,
        horizon_steps=20,
        deadline_s=1.0,
        maximum_iterations=40,
    )
    reference = np.column_stack(
        (np.linspace(0.05, 1.5, 20), np.zeros(20), np.zeros(20))
    )
    corridor = np.tile(
        np.array([-0.1, 2.0, -0.8, 0.8, -0.25, 0.25]), (20, 1)
    )
    center = np.array([0.5, 0.0, 0.0])
    radius = 0.2
    result = mpc.solve(
        np.array([0.0, -0.5, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.zeros(3),
        reference,
        np.zeros_like(reference),
        corridor,
        [(center, radius)],
    )
    assert result.success
    assert np.min(np.linalg.norm(result.positions_m - center, axis=1)) >= radius - 1e-5


def test_estimator_uses_real_dt_and_rejects_outlier():
    estimator = RelativeTargetEstimator(process_noise=0.1, innovation_gate=16.0)
    covariance = np.array([0.1, 0.1, 0.1, 0.05])
    assert estimator.update(1.0, [1.0, 0.0, 2.0], 0.0, covariance)
    assert estimator.update(1.2, [1.1, 0.0, 2.0], 0.0, covariance)
    before = estimator.x.copy()
    assert not estimator.update(1.4, [100.0, -100.0, 50.0], math.pi, covariance)
    assert estimator.rejected == 1
    assert np.linalg.norm(estimator.x[:3] - before[:3]) < 2.0


def test_touchdown_requires_continuous_combined_evidence():
    gate = TouchdownGate(dwell_s=1.0)
    assert not gate.update(0.0, 0.1, 0.05, True, False, True)
    assert not gate.update(0.8, 0.1, 0.05, True, False, True)
    gate.reset()
    assert not gate.update(1.0, 0.1, 0.05, True, False, True)
    assert not gate.update(1.8, 0.1, 0.05, True, False, True)
    assert gate.update(2.01, 0.1, 0.05, True, False, True)
    assert not gate.update(3.0, 1.0, 0.0, True, True, True)


def test_land_command_is_return_not_immediate_land():
    gate = MissionCommandGate(MissionState.GLOBAL_TRACK)
    assert gate.request_trailer_land()
    assert gate.state == MissionState.RETURN_REQUESTED
    assert gate.state != MissionState.EMERGENCY_LAND
    gate.request_emergency_land()
    assert gate.state == MissionState.EMERGENCY_LAND
