import io
import multiprocessing
import queue
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from landing_mpc.reference import HorizonReference
from mpc_landing.aruco_landing_node import (
    ArucoLandingNode,
    Phase,
    _header_stamp_seconds,
    _plan_route_worker,
    _planner_worker_init,
)
from path_plan.cju_route import (
    map_to_local,
    rotation_for_heading,
    route_map_info,
    safe_route_target,
    segment_is_free,
)
from path_plan.mpc import TrackingMPC
from path_plan.mpc_reference import (
    limit_acceleration_slew,
    path_reference_horizon,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, **_kwargs):
        self.messages.append(('info', message))

    def warn(self, message, **_kwargs):
        self.messages.append(('warn', message))

    def error(self, message, **_kwargs):
        self.messages.append(('error', message))


def _cruise_state(*, planned, fresh=True):
    logger = _Logger()
    calls = []
    pose = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
        x=0.0, y=0.0, z=5.0)))

    def route_mpc():
        calls.append(('tracking_mpc',))
        return False, float('inf')

    state = SimpleNamespace(
        pose=pose,
        target=np.array([20.0, 0.0, 0.0]),
        target_t=0.0,
        _t_phase=0.0,
        _now=lambda: 1.0,
        _fresh_target=lambda: fresh,
        trailer_lost_search=10.0,
        planned_cruise=planned,
        cruise_timeout=180.0,
        route_timeout=300.0,
        cruise_max_dist=150.0,
        cruise_arrive=1.0,
        _target_range=lambda: 20.0,
        _route_arrival_safe=lambda: False,
        _route_carrot=lambda: (None, float('inf')),
        _route_mpc_command=route_mpc,
        _takeoff_target=lambda: 5.0,
        _alt=lambda: 5.0,
        climb_speed=0.7,
        cruise_kp=0.35,
        cruise_v_max=0.5,
        cruise_log_period=3.0,
        _hold=lambda *args: calls.append(('hold', args)),
        _fly_to=lambda *args, **kwargs: calls.append(('fly', args, kwargs)),
        _due=lambda *_args: False,
        _to=lambda phase: calls.append(('phase', phase)),
        get_logger=lambda: logger,
    )
    return state, calls


def test_planned_cruise_has_no_straight_line_fallback_without_a_carrot():
    state, calls = _cruise_state(planned=True)
    ArucoLandingNode._cruise_to_trailer(state)
    assert any(call[0] == 'tracking_mpc' for call in calls)
    assert any(call[0] == 'hold' for call in calls)
    assert not any(call[0] == 'fly' for call in calls)


def test_legacy_trailer_cruise_still_flies_directly():
    state, calls = _cruise_state(planned=False)
    ArucoLandingNode._cruise_to_trailer(state)
    assert any(call[0] == 'fly' for call in calls)


def test_planned_cruise_does_not_search_from_an_uncertified_lost_position():
    state, calls = _cruise_state(planned=True, fresh=False)
    state._now = lambda: 20.0
    ArucoLandingNode._cruise_to_trailer(state)
    assert ('phase', Phase.LAND) in calls
    assert ('phase', Phase.SEARCH) not in calls


def test_route_timeout_is_separate_from_the_proven_trailer_timeout():
    route, route_calls = _cruise_state(planned=True)
    route._now = lambda: 200.0
    ArucoLandingNode._cruise_to_trailer(route)
    assert ('phase', Phase.LAND) not in route_calls

    trailer, trailer_calls = _cruise_state(planned=False)
    trailer._now = lambda: 200.0
    ArucoLandingNode._cruise_to_trailer(trailer)
    assert ('phase', Phase.LAND) in trailer_calls


def test_nearby_mission_goal_still_needs_an_exact_safe_chord():
    state = SimpleNamespace(
        planned_cruise=True,
        _route_map_info=SimpleNamespace(
            hardware_flight_approved=True,
            horizontal_accuracy='surveyed', mission_goal_xy=(50.0, 50.0)),
        allow_unapproved_route_map=False,
        _route_input_reason=lambda **_kwargs: None,
        _route_goal_local=lambda **_kwargs: np.array([0.5, 0.0]),
        _range_to=lambda _goal: 0.5,
        cruise_arrive=1.0,
        _route_endpoint_reason=lambda **_kwargs: None,
        _route_arrival_safe=lambda **_kwargs: False,
    )
    assert ArucoLandingNode._route_preflight(state) == [
        'nearby mission-goal chord or endpoint is outside the certified map clearance']


def test_return_arrival_uses_the_live_nearby_target_not_an_old_route_endpoint():
    state = SimpleNamespace(
        planned_cruise=True,
        phase=Phase.RETURN,
        _route_flight_health_reason=lambda: None,
        _route_uses_trailer_goal=lambda: True,
        _route_goal_local=lambda **_kwargs: np.array([0.5, 0.0]),
        _fresh_target=lambda: True,
        _range_to=lambda _goal: 0.5,
        cruise_arrive=1.0,
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
            x=0.0, y=0.0))),
        _route_active=(
            object(), np.zeros(2), np.array([20.0, 0.0]), 1, 0.0),
        _route_lib=SimpleNamespace(segment_is_free=lambda *_args: True),
        route_map_yaml='map.yaml',
    )
    assert ArucoLandingNode._route_arrival_safe(
        state, trailer_goal=True)


def test_unapproved_map_needs_the_explicit_route_override():
    state = SimpleNamespace(
        planned_cruise=True,
        _route_map_info=SimpleNamespace(
            hardware_flight_approved=False,
            horizontal_accuracy='not surveyed', mission_goal_xy=(50.0, 50.0)),
        allow_unapproved_route_map=False,
        _route_input_reason=lambda **_kwargs: None,
        _route_goal_local=lambda **_kwargs: np.array([0.5, 0.0]),
        _range_to=lambda _goal: 0.5,
        cruise_arrive=1.0,
        _route_endpoint_reason=lambda **_kwargs: None,
        _route_arrival_safe=lambda **_kwargs: True,
    )
    assert 'not hardware-flight-approved' in (
        ArucoLandingNode._route_preflight(state)[0])
    state.allow_unapproved_route_map = True
    assert ArucoLandingNode._route_preflight(state) == []


def test_far_mission_goal_is_planned_after_mission_not_during_precheck():
    state = SimpleNamespace(
        planned_cruise=True,
        _route_map_info=SimpleNamespace(
            hardware_flight_approved=True,
            horizontal_accuracy='surveyed', mission_goal_xy=(50.0, 50.0)),
        allow_unapproved_route_map=False,
        _route_input_reason=lambda **_kwargs: None,
        _route_goal_local=lambda **_kwargs: np.array([50.0, 50.0]),
        _range_to=lambda _goal: 70.0,
        cruise_arrive=1.0,
        _route_endpoint_reason=lambda **_kwargs: None,
    )
    assert ArucoLandingNode._route_preflight(state) == []


def test_precheck_rejects_a_goal_outside_the_hard_clearance_map():
    checked = []

    def segment(_map, _origin, start, goal):
        checked.append((np.asarray(start), np.asarray(goal)))
        return len(checked) == 1

    state = SimpleNamespace(
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
            x=1.0, y=2.0))),
        _route_synchronized_site_origin=lambda: np.zeros(2),
        _route_goal_local=lambda **_kwargs: np.array([50.0, 50.0]),
        _route_lib=SimpleNamespace(segment_is_free=segment),
        route_map_yaml='map.yaml',
    )
    assert 'goal is blocked' in (
        ArucoLandingNode._route_endpoint_reason(
            state, trailer_goal=False) or '')
    assert len(checked) == 2


def test_route_flight_health_rejects_inaccurate_absolute_gps():
    route_map = (Path(__file__).parents[2]
                 / 'path_plan' / 'config' / 'drone_cju_route.yaml').resolve()
    route_stat = route_map.stat()
    ekf = SimpleNamespace(
        status_fresh=lambda _now: True,
        gps_fresh=lambda _now: True,
        const_pos_mode=False,
        velocity_horiz=True,
        pos_horiz_abs=True,
        gps_glitch=False,
        h_acc=0.41,
    )
    state = SimpleNamespace(
        planned_cruise=True,
        route_map_yaml=str(route_map),
        _route_map_info=SimpleNamespace(vehicle_clearance_m=1.0),
        _route_map_identity=(
            route_stat.st_dev, route_stat.st_ino,
            route_stat.st_size, route_stat.st_mtime_ns),
        _now=lambda: 10.0,
        ekf=ekf,
        route_max_hacc=0.4,
        route_anchor_drift=0.2,
        cruise_v_max=1.5,
        pose=object(),
        pose_t=10.0,
        pose_rx_t=10.0,
        velocity=SimpleNamespace(
            twist=SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0))),
        velocity_t=10.0,
        velocity_rx_t=10.0,
        route_sync_tolerance=0.1,
        route_gps_timeout=3.0,
        route_state_timeout=0.2,
        _route_anchor_drift_reason=lambda: None,
    )
    assert 'exceeds route limit' in (
        ArucoLandingNode._route_flight_health_reason(state) or '')
    ekf.h_acc = 0.3
    state.velocity_t = 9.85
    assert 'pose and velocity are not time-aligned' in (
        ArucoLandingNode._route_flight_health_reason(state) or '')

    state.pose_t = state.velocity_t = 9.79
    assert 'local pose is missing or stale' in (
        ArucoLandingNode._route_flight_health_reason(state) or '')

    state.pose_t = state.velocity_t = 10.0
    state.velocity.twist.linear.x = 1.6
    assert ArucoLandingNode._route_flight_health_reason(state) is None


def test_route_update_invalidates_an_active_anchor_on_health_loss():
    calls = []
    state = SimpleNamespace(
        planned_cruise=True,
        phase=Phase.TAKEOFF,
        _route_flight_health_reason=lambda: 'GPS glitch',
        _invalidate_route=calls.append,
        _plan_future=None,
        _planner_pool=None,
    )
    ArucoLandingNode._route_update(state)
    assert calls == ['GPS glitch']


def test_route_anchor_jump_is_rejected_before_reusing_the_path():
    state = SimpleNamespace(
        _route_active=(None, np.zeros(2), None, None, None),
        _route_pending=None,
        _route_synchronized_site_origin=lambda: np.array([0.201, 0.0]),
        route_anchor_drift=0.2,
    )
    assert 'anchor moved' in (
        ArucoLandingNode._route_anchor_drift_reason(state) or '')


def test_active_route_requires_a_continuously_observable_anchor():
    state = SimpleNamespace(
        _route_active=(None, np.zeros(2), None, None, None),
        _route_pending=None,
        _route_synchronized_site_origin=lambda: None,
        route_anchor_drift=0.2,
    )
    assert 'unavailable' in (
        ArucoLandingNode._route_anchor_drift_reason(state) or '')
    state._route_active = None
    assert ArucoLandingNode._route_anchor_drift_reason(state) is None


def test_synchronized_anchor_survives_between_five_hz_fix_updates():
    state = SimpleNamespace(
        _now=lambda: 0.15,
        _route_observed_origin=np.array([1.0, 2.0]),
        _route_observed_origin_t=0.0,
        _route_observed_origin_rx_t=0.0,
        route_anchor_timeout=0.3,
        route_sync_tolerance=0.1,
    )
    assert np.allclose(
        ArucoLandingNode._route_synchronized_site_origin(state), [1.0, 2.0])
    state._now = lambda: 0.31
    assert ArucoLandingNode._route_synchronized_site_origin(state) is None


def test_only_a_source_time_aligned_pose_fix_pair_updates_the_anchor():
    state = SimpleNamespace(
        planned_cruise=True,
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
            x=10.0, y=20.0))),
        vehicle_fix=SimpleNamespace(
            status=SimpleNamespace(status=0), latitude=36.0, longitude=127.0),
        pose_t=1.05,
        vehicle_fix_t=1.0,
        route_sync_tolerance=0.1,
        _route_map_info=SimpleNamespace(origin_lat=36.0, origin_lon=127.0),
        _enu_offset=lambda *_args: (2.0, 3.0),
        _now=lambda: 1.06,
        _route_observed_origin=None,
        _route_observed_origin_t=float('nan'),
        _route_observed_origin_rx_t=0.0,
    )
    ArucoLandingNode._update_route_site_origin(state)
    assert np.allclose(state._route_observed_origin, [8.0, 17.0])
    assert state._route_observed_origin_t == 1.05

    state.pose_t = 1.11
    ArucoLandingNode._update_route_site_origin(state)
    assert state._route_observed_origin_t == 1.05


def test_route_uses_source_header_time_and_rejects_invalid_stamps():
    message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
        sec=12, nanosec=250_000_000)))
    assert _header_stamp_seconds(message) == 12.25
    message.header.stamp.nanosec = 1_000_000_000
    assert np.isnan(_header_stamp_seconds(message))


def test_route_target_freshness_uses_source_and_receipt_time():
    state = SimpleNamespace(
        target=np.zeros(3), target_t=10.0, target_sample_t=7.0,
        target_timeout=3.0, route_sync_tolerance=0.1,
        planned_cruise=True, _now=lambda: 10.0)
    assert not ArucoLandingNode._fresh_target(state)

    state.target_sample_t = 9.95
    assert ArucoLandingNode._fresh_target(state)
    state.target_sample_t = 10.11
    assert not ArucoLandingNode._fresh_target(state)

    # Non-planned callers retain the original receipt-time contract.
    state.planned_cruise = False
    state.target_sample_t = float('nan')
    assert ArucoLandingNode._fresh_target(state)


def test_fixed_map_goal_is_converted_to_local_enu_not_used_as_local_xy():
    rotation = rotation_for_heading(30.0)
    origin = np.array([12.0, -7.0])
    state = SimpleNamespace(
        phase=Phase.PRECHECK,
        target=np.array([999.0, 999.0, 0.0]),
        _route_uses_trailer_goal=lambda: False,
        _route_synchronized_site_origin=lambda: origin,
        _route_lib=SimpleNamespace(map_to_local=map_to_local),
        _route_map_info=SimpleNamespace(mission_goal_xy=(50.0, 50.0)),
        _route_rotation=rotation,
    )
    goal = ArucoLandingNode._route_goal_local(state, trailer_goal=False)
    assert np.allclose(goal, map_to_local([50.0, 50.0], origin, rotation))
    assert not np.allclose(goal, state.target[:2])
    state.phase = Phase.RETURN_PLAN
    state._route_uses_trailer_goal = (
        lambda: ArucoLandingNode._route_uses_trailer_goal(state))
    assert np.allclose(ArucoLandingNode._route_goal_local(state), [999.0, 999.0])


def test_fixed_mission_route_does_not_require_a_trailer_target_timestamp():
    state = SimpleNamespace(
        _route_flight_health_reason=lambda: None,
        _now=lambda: 3.0,
        vehicle_fix=SimpleNamespace(
            status=SimpleNamespace(status=0), latitude=36.0, longitude=127.0),
        vehicle_fix_t=3.0,
        vehicle_fix_rx_t=3.0,
        route_gps_timeout=3.0,
        route_sync_tolerance=0.1,
        route_vehicle_fix_topic='/mavros/global_position/global',
        _route_synchronized_site_origin=lambda: np.zeros(2),
        _route_goal_local=lambda **_kwargs: np.array([50.0, 50.0]),
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
            x=0.0, y=0.0))),
    )
    assert ArucoLandingNode._route_input_reason(
        state, trailer_goal=False) is None


def test_new_plan_requires_target_and_anchor_samples_to_be_synchronized():
    state = SimpleNamespace(
        _route_flight_health_reason=lambda: None,
        _now=lambda: 3.0,
        vehicle_fix=SimpleNamespace(
            status=SimpleNamespace(status=0), latitude=36.0, longitude=127.0),
        vehicle_fix_t=3.0,
        vehicle_fix_rx_t=3.0,
        route_gps_timeout=3.0,
        pose_t=3.0,
        target_sample_t=2.4,
        route_sync_tolerance=0.5,
        _route_observed_origin_t=3.0,
        _route_synchronized_site_origin=lambda: np.zeros(2),
        _fresh_target=lambda: True,
    )
    assert 'trailer target are not time-aligned' in (
        ArucoLandingNode._route_input_reason(
            state, trailer_goal=True) or '')


def _gate_state(calls, phase=Phase.READY_TO_ARM, blockers=()):
    state = SimpleNamespace(
        phase=phase,
        planned_cruise=True,
        _route_map_info=SimpleNamespace(mission_goal_xy=(50.0, 50.0)),
        _route_preflight=lambda: list(blockers),
        _return_preflight=lambda: [],
        _to=lambda new: calls.append(('phase', new)),
        _begin_route=lambda new: calls.append(('phase', new)),
        _begin_return=lambda: calls.append(('return',)),
        _abort=lambda reason: calls.append(('abort', reason)),
    )
    state._approve = lambda: ArucoLandingNode._approve(state)
    return state


def test_the_arm_is_the_only_gate():
    calls = []
    state = _gate_state(calls)
    ok, message = ArucoLandingNode._approve(state)
    assert ok and calls[-1] == ('phase', Phase.ARMING)
    assert 'arming' in message

    # Every later phase used to be a gate with its own word. None is now, so
    # approving there is a mistake to report, not a transition to make.
    for phase in (Phase.READY, Phase.HOVER, Phase.MISSION, Phase.SEARCH):
        state.phase = phase
        before = list(calls)
        ok, message = ArucoLandingNode._approve(state)
        assert not ok and 'not a gate' in message
        assert calls == before


def test_bare_enter_approves_and_abort_words_land():
    calls = []
    state = _gate_state(calls)
    assert ArucoLandingNode._command(state, '')[0]
    assert calls[-1] == ('phase', Phase.ARMING)

    for word in ('ABORT', 'abort', 'n', 'NO'):
        state.phase = Phase.MISSION
        ok, _message = ArucoLandingNode._command(state, word)
        assert ok and calls[-1][0] == 'abort'


def test_a_route_that_changed_since_precheck_blocks_the_arm():
    calls = []
    state = _gate_state(calls, blockers=['route map is not certified'])
    ok, message = ArucoLandingNode._approve(state)
    assert not ok and 'route changed since PRECHECK' in message
    assert calls == []


def test_ready_starts_the_mission_leg_without_an_operator():
    calls = []
    state = SimpleNamespace(
        phase=Phase.READY,
        planned_cruise=True,
        _drain_stdin_commands=lambda: None,
        _publish_state=lambda: None,
        _send=lambda *_args: None,
        _route_update=lambda: None,
        _hold=lambda *_args: calls.append('hold'),
        _route_preflight=lambda: [],
        _route_map_info=SimpleNamespace(mission_goal_xy=(50.0, 50.0)),
        takeoff_alt=5.0,
        _now=lambda: 1.0,
        _t_phase=0.0,
        route_timeout=300.0,
        get_logger=lambda: _Logger(),
        _begin_route=lambda phase: calls.append(('phase', phase)),
        _begin_return=lambda: calls.append(('return',)),
        _to=lambda phase: calls.append(('phase', phase)),
    )
    ArucoLandingNode._tick(state)
    assert ('phase', Phase.MISSION_PLAN) in calls


def test_ready_gives_up_the_mission_leg_forward_not_backward():
    calls = []
    state = SimpleNamespace(
        phase=Phase.READY,
        planned_cruise=True,
        _drain_stdin_commands=lambda: None,
        _publish_state=lambda: None,
        _send=lambda *_args: None,
        _route_update=lambda: None,
        _hold=lambda *_args: calls.append('hold'),
        _route_preflight=lambda: ['no certified route'],
        _route_map_info=SimpleNamespace(mission_goal_xy=(50.0, 50.0)),
        takeoff_alt=5.0,
        _now=lambda: 400.0,
        _t_phase=0.0,
        route_timeout=300.0,
        get_logger=lambda: _Logger(),
        _begin_route=lambda phase: calls.append(('phase', phase)),
        _begin_return=lambda: calls.append(('return',)),
        _to=lambda phase: calls.append(('phase', phase)),
    )
    ArucoLandingNode._tick(state)
    # Forward to the landing, never back to a phase that re-enters planning.
    assert ('return',) in calls
    assert ('phase', Phase.MISSION_PLAN) not in calls


def test_terminal_thread_queues_instead_of_changing_flight_state(monkeypatch):
    commands = queue.SimpleQueue()
    state = SimpleNamespace(_stdin_commands=commands)
    monkeypatch.setattr(sys, 'stdin', io.StringIO('LAND\n'))
    ArucoLandingNode._stdin_loop(state)
    assert commands.get_nowait() == 'LAND'


def test_land_discards_the_fixed_route_before_return_planning():
    calls = []
    state = SimpleNamespace(
        _route_active=object(),
        _route_pending=object(),
        _route_progress=12.0,
        _route_last_request_t=10.0,
        _route_last_error='old',
        _plan_future=None,
        _reset_path_mpc=lambda: calls.append('reset'),
        _to=lambda phase: calls.append(phase),
    )
    ArucoLandingNode._begin_route(state, Phase.RETURN_PLAN)
    assert state._route_active is None and state._route_pending is None
    assert state._route_progress == 0.0
    assert state._route_last_request_t == float('-inf')
    assert calls == ['reset', Phase.RETURN_PLAN]


def test_takeoff_reaches_ready_without_flying_to_the_trailer():
    calls = []
    state = SimpleNamespace(
        phase=Phase.TAKEOFF,
        planned_cruise=True,
        _drain_stdin_commands=lambda: None,
        _publish_state=lambda: None,
        _send=lambda *_args: calls.append(('setpoint',)),
        _route_update=lambda: None,
        _takeoff_target=lambda: 5.0,
        _alt=lambda: 5.0,
        alt_tol=0.3,
        climb_speed=0.7,
        _route_settled=lambda: True,
        _hold=lambda *_args: calls.append(('hold',)),
        _to=lambda phase: calls.append(('phase', phase)),
    )
    ArucoLandingNode._tick(state)
    assert ('phase', Phase.READY) in calls
    assert ('phase', Phase.RETURN) not in calls
    assert ('phase', Phase.CRUISE) not in calls


def test_takeoff_waits_until_velocity_is_settled():
    calls = []
    state = SimpleNamespace(
        phase=Phase.TAKEOFF,
        planned_cruise=True,
        _drain_stdin_commands=lambda: None,
        _publish_state=lambda: None,
        _send=lambda *_args: None,
        _route_update=lambda: None,
        _takeoff_target=lambda: 5.0,
        _alt=lambda: 5.0,
        alt_tol=0.3,
        climb_speed=0.7,
        _route_settled=lambda: False,
        _hold=lambda *_args: calls.append('hold'),
        _to=lambda phase: calls.append(phase),
    )
    ArucoLandingNode._tick(state)
    assert 'hold' in calls and Phase.READY not in calls


def test_fixed_goal_does_not_hand_over_with_residual_speed():
    calls = []
    state = SimpleNamespace(
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
            x=49.5, y=50.0))),
        _route_goal_local=lambda **_kwargs: np.array([50.0, 50.0]),
        _range_to=lambda _goal: 0.5,
        cruise_arrive=1.0,
        _route_arrival_safe=lambda **_kwargs: True,
        _route_settled=lambda: False,
        _now=lambda: 1.0,
        _t_phase=0.0,
        route_timeout=300.0,
        _takeoff_target=lambda: 5.0,
        _alt=lambda: 5.0,
        climb_speed=0.7,
        _route_mpc_command=lambda: (True, 0.0),
        _hold=lambda *_args: calls.append('hold'),
        _due=lambda *_args: False,
        _route_map_info=SimpleNamespace(mission_goal_xy=(50.0, 50.0)),
        cruise_log_period=3.0,
        get_logger=lambda: _Logger(),
        _to=lambda phase: calls.append(phase),
    )
    ArucoLandingNode._mission_to_goal(state)
    assert Phase.HOVER not in calls
    state._route_settled = lambda: True
    ArucoLandingNode._mission_to_goal(state)
    assert Phase.HOVER in calls


def test_hover_keeps_the_tracking_mpc_reference_and_returns_by_itself():
    calls = []
    state = SimpleNamespace(
        phase=Phase.HOVER,
        planned_cruise=True,
        _drain_stdin_commands=lambda: None,
        _publish_state=lambda: None,
        _route_mpc_command=lambda: calls.append('tracking_mpc') or (True, 0.0),
        _hold=lambda: calls.append('hold'),
        _return_preflight=lambda: [],
        _now=lambda: 1.0,
        _t_phase=0.0,
        trailer_lost_search=10.0,
        get_logger=lambda: _Logger(),
        _begin_return=lambda: calls.append('return'),
        _to=lambda phase: calls.append(phase),
    )
    ArucoLandingNode._tick(state)
    assert calls == ['tracking_mpc', 'return']

    # The trailer link is the one thing it waits on, and only for as long as a
    # lost target is tolerated anywhere else — then it lands where it is.
    calls.clear()
    state._return_preflight = lambda: ['no trailer fix']
    state._now = lambda: 100.0
    ArucoLandingNode._tick(state)
    assert Phase.LAND in calls


def test_route_settle_uses_full_measured_velocity():
    state = SimpleNamespace(
        _now=lambda: 10.0,
        velocity=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(
            x=0.1, y=0.1, z=0.1))),
        velocity_t=10.0,
        velocity_rx_t=10.0,
        route_state_timeout=0.2,
        route_sync_tolerance=0.1,
        arrival_speed=0.2,
    )
    assert ArucoLandingNode._route_settled(state)
    state.velocity.twist.linear.z = 0.2
    assert not ArucoLandingNode._route_settled(state)
    state.velocity.twist.linear.z = 0.0
    state.velocity_rx_t = 9.79
    assert not ArucoLandingNode._route_settled(state)


def test_mpc_setpoint_precedes_completed_route_commit_work():
    calls = []
    state = SimpleNamespace(
        phase=Phase.MISSION,
        planned_cruise=True,
        _drain_stdin_commands=lambda: None,
        _publish_state=lambda: None,
        _mission_to_goal=lambda: calls.append('setpoint'),
        _route_update=lambda: calls.append('route_update'),
    )
    ArucoLandingNode._tick(state)
    assert calls == ['setpoint', 'route_update']


def test_return_planning_timeout_lands_instead_of_hovering_forever():
    calls = []
    state = SimpleNamespace(
        phase=Phase.RETURN_PLAN,
        planned_cruise=True,
        _drain_stdin_commands=lambda: None,
        _publish_state=lambda: None,
        _send=lambda *_args: calls.append('heartbeat'),
        _route_update=lambda: None,
        _hold=lambda: calls.append('hold'),
        _route_goal_range=lambda **_kwargs: 20.0,
        _route_arrival_safe=lambda **_kwargs: False,
        _route_active=None,
        cruise_arrive=1.0,
        _fresh_target=lambda: True,
        _now=lambda: 301.0,
        _t_phase=0.0,
        trailer_lost_search=10.0,
        route_timeout=300.0,
        get_logger=lambda: _Logger(),
        _to=lambda phase: calls.append(phase),
    )
    ArucoLandingNode._tick(state)
    assert Phase.LAND in calls


def test_phase_reentry_restores_the_operator_prompt():
    state = SimpleNamespace(
        phase=Phase.MISSION,
        _prompted=Phase.READY.value,
        _announced=Phase.READY.value,
        _now=lambda: 10.0,
        planned_cruise=False,
        get_logger=lambda: _Logger(),
    )
    ArucoLandingNode._to(state, Phase.READY)
    assert state._prompted == state._announced == ''


def test_route_worker_is_spawn_safe():
    map_yaml = str(
        Path(__file__).parents[2]
        / 'path_plan' / 'config' / 'drone_cju_route.yaml')
    info = route_map_info(map_yaml)
    rotation = rotation_for_heading(info.heading_deg_enu)
    origin = np.zeros(2)
    start = map_to_local([5.0, 0.0], origin, rotation)
    goal = map_to_local([50.0, 50.0], origin, rotation)
    with ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_planner_worker_init) as pool:
        plan = pool.submit(
            _plan_route_worker, map_yaml, start.tolist(), goal.tolist(),
            origin.tolist()).result(timeout=15.0)
    assert np.allclose(plan.path_local_xy[0], start, atol=1.0e-6)
    assert np.allclose(plan.path_local_xy[-1], goal, atol=1.0e-6)


def test_one_metre_route_streams_bounded_tracking_mpc_commands():
    """Exercise the real 2-D CJU geometry through JO TrackingMPC at 50 Hz."""
    map_yaml = str(
        Path(__file__).parents[2]
        / 'path_plan' / 'config' / 'drone_cju_route.yaml')
    info = route_map_info(map_yaml)
    rotation = rotation_for_heading(info.heading_deg_enu)
    origin = np.zeros(2)
    start = map_to_local([5.0, 0.0], origin, rotation)
    goal = map_to_local([50.0, 50.0], origin, rotation)
    plan = _plan_route_worker(map_yaml, start, goal, origin)

    path = np.column_stack((
        plan.path_local_xy, np.full(len(plan.arc_m), 5.0)))
    mpc = TrackingMPC(
        dt_s=0.1, horizon=20, v_max=1.5, a_max=0.5, j_max=2.0,
        q_pos=4.0, q_vel=0.4, r_acc=0.05, q_terminal=20.0)
    stream = HorizonReference(lead_s=0.1)
    position = np.r_[start, 5.0]
    measured_velocity = np.zeros(3)
    progress = elapsed = 0.0
    last_acceleration = np.zeros(3)
    solve_t = None

    while elapsed < 300.0 and np.linalg.norm(position[:2] - goal) > 1.0:
        progress, carrot, _cross_track = safe_route_target(
            map_yaml, origin, plan.arc_m, plan.path_local_xy,
            position[:2], progress, 6.0, 0.25)
        assert carrot is not None
        target_range = float(np.linalg.norm(position[:2] - goal))
        if solve_t is None or elapsed - solve_t >= 0.1 - 1.0e-9:
            reference_p, reference_v = path_reference_horizon(
                plan.arc_m, path, progress, 0.1, 20, 1.5, 0.5, 2.0,
                target_velocity_xy=np.zeros(2),
                target_range_xy_m=target_range)
            result = mpc.solve(
                position, measured_velocity,
                reference_p, reference_v,
                applied_acceleration=last_acceleration, output_step=1)
            assert result.success
            chain = np.vstack((position[:2], result.predicted_pos[:, :2]))
            assert all(segment_is_free(map_yaml, origin, a, b)
                       for a, b in zip(chain[:-1], chain[1:]))
            stream.set_plan(
                position, measured_velocity,
                result.predicted_pos, result.predicted_vel,
                result.predicted_acc, 0.1,
                np.zeros(3), np.zeros(3), np.zeros(3))
            solve_t = elapsed

        command_position, velocity, acceleration = stream.sample(
            elapsed - solve_t)
        assert np.max(np.abs(velocity)) <= 1.5 + 1.0e-4
        assert segment_is_free(
            map_yaml, origin, position[:2], command_position[:2])

        previous_acceleration = last_acceleration.copy()
        last_acceleration = limit_acceleration_slew(
            last_acceleration, acceleration, 2.0, 0.02)
        assert np.max(np.abs(
            last_acceleration - previous_acceleration)) <= 0.04 + 1.0e-12
        position += measured_velocity * 0.02 \
            + 0.5 * last_acceleration * 0.02 ** 2
        measured_velocity += last_acceleration * 0.02
        elapsed += 0.02

    assert np.linalg.norm(position[:2] - goal) <= 1.0
    assert elapsed < 300.0


def test_integration_accepts_jo_axiswise_velocity_contract():
    """The hardware wrapper must not reinterpret JO's axis-wise MPC bound."""
    sent = []
    state = SimpleNamespace(
        _route_carrot=lambda: (np.zeros(2), 0.0),
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
            x=0.0, y=0.0))),
        velocity=SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0))),
        _path_mpc=SimpleNamespace(dt=0.1, N=20, j_max=2.0),
        _path_reference=SimpleNamespace(
            ready=lambda: True,
            sample=lambda _elapsed: (
                np.array([0.1, 0.1, 5.0]),
                np.array([1.1, 1.1, 0.0]),
                np.zeros(3))),
        _route_active=(object(),),
        _alt=lambda: 5.0,
        _now=lambda: 1.0,
        _path_last_solve_t=1.0,
        _path_solve_t=1.0,
        _route_prediction_is_safe=lambda _positions: True,
        _send_pva=lambda _p, v, _a, _j: sent.append(np.asarray(v)) or True,
    )
    assert ArucoLandingNode._route_mpc_command(state) == (True, 0.0)
    assert np.linalg.norm(sent[0][:2]) > 1.5


def test_trailer_velocity_braking_is_used_only_on_the_return_leg():
    captured = {}

    def horizon(*_args, **kwargs):
        captured.update(kwargs)
        return np.zeros((2, 3)), np.zeros((2, 3))

    result = SimpleNamespace(
        success=True,
        predicted_pos=np.zeros((2, 3)),
        predicted_vel=np.zeros((2, 3)),
        predicted_acc=np.zeros((2, 3)),
    )
    path_reference = SimpleNamespace(
        lead=0.1,
        set_plan=lambda *_args: None,
        ready=lambda: True,
        sample=lambda _elapsed: (np.zeros(3), np.zeros(3), np.zeros(3)),
    )
    plan = SimpleNamespace(
        arc_m=np.array([0.0, 1.0]),
        path_local_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
    )
    state = SimpleNamespace(
        _route_carrot=lambda: (np.zeros(2), 0.0),
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
            x=0.0, y=0.0))),
        velocity=SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0))),
        _path_mpc=SimpleNamespace(
            dt=0.1, N=2, j_max=2.0,
            solve=lambda *_args, **_kwargs: result),
        _path_reference=path_reference,
        _route_active=(plan,),
        _alt=lambda: 5.0,
        _takeoff_target=lambda: 5.0,
        _now=lambda: 1.0,
        _path_last_solve_t=None,
        _path_solve_t=None,
        _path_reference_horizon=horizon,
        cruise_v_max=1.5,
        cruise_accel=0.5,
        _route_progress=0.0,
        _route_uses_trailer_goal=lambda: False,
        _route_goal_range=lambda: 20.0,
        _target_velocity=SimpleNamespace(v=np.array([0.7, 0.0])),
        _last_mpc_acceleration=np.zeros(3),
        _route_prediction_is_safe=lambda _positions: True,
        _reset_path_mpc=lambda: None,
        _send_pva=lambda *_args: True,
    )
    assert ArucoLandingNode._route_mpc_command(state)[0]
    assert captured['target_velocity_xy'] is None

    state._route_uses_trailer_goal = lambda: True
    state._path_last_solve_t = None
    captured.clear()
    assert ArucoLandingNode._route_mpc_command(state)[0]
    assert np.allclose(captured['target_velocity_xy'], [0.7, 0.0])


def test_planned_trailer_descend_uses_proven_p_controller():
    calls = []
    state = SimpleNamespace(_descend_p=lambda: calls.append('p'))
    ArucoLandingNode._descend(state)
    assert calls == ['p']


def test_marker_acquisition_counts_new_pose_frames_not_timer_ticks():
    state = SimpleNamespace(
        marker=object(), detected=True, _marker_seq=1,
        _acq_last_marker_seq=0, _acq_streak=0, acquire_frames=3,
        _fresh_marker=lambda: True)
    for _ in range(5):
        assert not ArucoLandingNode._marker_acquired(state)
    assert state._acq_streak == 1
    state._marker_seq = 2
    assert not ArucoLandingNode._marker_acquired(state)
    state._marker_seq = 3
    assert ArucoLandingNode._marker_acquired(state)

    # aruco_pose_node publishes pose before detected=True. A timer between the
    # two callbacks must not consume that new pose while detected is still false.
    state.detected = False
    state._marker_seq = 4
    assert not ArucoLandingNode._marker_acquired(state)
    state.detected = True
    assert not ArucoLandingNode._marker_acquired(state)
    assert state._acq_streak == 1


def _arming_state(*, armed, preflight_ok):
    calls = []
    preflight_calls = []
    logger = _Logger()

    def check_preflight(**kwargs):
        preflight_calls.append(kwargs)
        return preflight_ok

    state = SimpleNamespace(
        phase=Phase.ARMING,
        planned_cruise=False,
        _drain_stdin_commands=lambda: None,
        state=SimpleNamespace(armed=armed, mode='OFFBOARD'),
        _publish_state=lambda: None,
        _send=lambda *_args: calls.append(('setpoint',)),
        _preflight_ok=check_preflight,
        _t_prestream=0.0,
        _checks_logged=True,
        _prompted=Phase.READY_TO_ARM.value,
        _announced=Phase.READY_TO_ARM.value,
        _z_ground=None,
        _yaw_hold=None,
        _alt=lambda: 0.0,
        _yaw_now=lambda: 0.0,
        _takeoff_target=lambda: 5.0,
        takeoff_alt=5.0,
        _to=lambda phase: calls.append(('phase', phase)),
        get_logger=lambda: logger,
    )
    return state, calls, preflight_calls


def test_arming_rechecks_live_inputs_and_requires_fresh_approval():
    state, calls, preflight_calls = _arming_state(
        armed=False, preflight_ok=False)
    ArucoLandingNode._tick(state)
    assert preflight_calls == [{}]
    assert ('phase', Phase.PRECHECK) in calls
    assert state._t_prestream is None
    assert state._prompted == state._announced == ''


def test_armed_confirmation_lands_if_inputs_changed_during_arm_request():
    state, calls, preflight_calls = _arming_state(
        armed=True, preflight_ok=False)
    ArucoLandingNode._tick(state)
    assert preflight_calls == [{'allow_armed': True}]
    assert ('phase', Phase.LAND) in calls
    assert ('phase', Phase.TAKEOFF) not in calls


def test_armed_confirmation_takes_off_only_with_live_inputs():
    state, calls, preflight_calls = _arming_state(
        armed=True, preflight_ok=True)
    ArucoLandingNode._tick(state)
    assert preflight_calls == [{'allow_armed': True}]
    assert ('phase', Phase.TAKEOFF) in calls
