import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from landing_mpc.mpc import LandingMPC
from landing_mpc.predictor import predict_const_vel
from landing_mpc.reference import HorizonReference
from mpc_landing.aruco_landing_node import (
    ArucoLandingNode,
    Phase,
    _header_stamp_seconds,
    _plan_route_worker,
    _planner_worker_init,
)
from path_plan.cju_route import map_to_local, rotation_for_heading, route_map_info


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


def test_nearby_target_still_needs_an_exact_safe_chord():
    state = SimpleNamespace(
        planned_cruise=True,
        _route_map_info=SimpleNamespace(
            hardware_flight_approved=True,
            horizontal_accuracy='surveyed'),
        allow_unapproved_route_map=False,
        _route_input_reason=lambda: None,
        _target_range=lambda: 0.5,
        cruise_arrive=1.0,
        _route_arrival_safe=lambda: False,
    )
    assert ArucoLandingNode._route_preflight(state) == [
        'nearby trailer chord or endpoint is outside the certified map clearance']


def test_unapproved_map_needs_the_explicit_route_override():
    state = SimpleNamespace(
        planned_cruise=True,
        _route_map_info=SimpleNamespace(
            hardware_flight_approved=False,
            horizontal_accuracy='not surveyed'),
        allow_unapproved_route_map=False,
        _route_input_reason=lambda: None,
        _target_range=lambda: 0.5,
        cruise_arrive=1.0,
        _route_arrival_safe=lambda: True,
    )
    assert 'not hardware-flight-approved' in (
        ArucoLandingNode._route_preflight(state)[0])
    state.allow_unapproved_route_map = True
    assert ArucoLandingNode._route_preflight(state) == []


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
        _route_map_info=SimpleNamespace(clearance_margin_m=1.0),
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
    state.route_anchor_drift = 0.5
    assert 'error budget' in (
        ArucoLandingNode._route_flight_health_reason(state) or '')


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


def test_route_velocity_checks_a_turn_aware_stopping_envelope():
    pose = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
        x=0.0, y=0.0)))
    velocity = SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(
        x=1.0, y=0.0)))

    checked = []

    def envelope_free(_map, _origin, center, radius, **_kwargs):
        checked.append((np.asarray(center), float(radius)))
        return False

    state = SimpleNamespace(
        planned_cruise=True,
        phase=Phase.CRUISE,
        _route_flight_health_reason=lambda: None,
        pose=pose,
        velocity=velocity,
        _route_active=(None, np.zeros(2), None, None, None),
        _route_synchronized_site_origin=lambda: None,
        _now=lambda: 0.0,
        pose_t=0.0,
        velocity_t=0.0,
        rate_hz=20.0,
        cruise_v_max=1.5,
        cruise_accel=0.5,
        route_brake_decel=0.25,
        route_reaction_time=0.5,
        _v_cmd=np.zeros(2),
        _route_lib=SimpleNamespace(
            stopping_envelope_is_free=envelope_free),
        route_map_yaml='unused.yaml',
    )
    assert not ArucoLandingNode._route_velocity_is_safe(
        state, np.array([0.0, 1.0]))
    assert len(checked) == 1
    assert np.allclose(checked[0][0], [0.0, 0.0])
    assert checked[0][1] == 2.5

    state._now = lambda: 0.1
    assert not ArucoLandingNode._route_velocity_is_safe(
        state, np.array([0.0, 1.0]))
    assert checked[-1][1] == 2.6

    state.velocity.twist.linear.x = 0.0
    state._now = lambda: 0.0
    assert not ArucoLandingNode._route_velocity_is_safe(
        state, np.zeros(2))
    assert checked[-1][1] == 0.0

    state._v_cmd = np.array([1.0, 0.0])
    assert not ArucoLandingNode._route_velocity_is_safe(
        state, np.zeros(2))
    assert checked[-1][1] == 2.5


def test_route_bisects_the_first_slew_step_instead_of_deadlocking():
    pose = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
        x=0.0, y=0.0)))
    velocity = SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(
        x=0.0, y=0.0)))
    radius_limit = 0.013627

    def envelope_free(_map, _origin, _center, radius, **_kwargs):
        return float(radius) <= radius_limit

    state = SimpleNamespace(
        planned_cruise=True,
        phase=Phase.CRUISE,
        _route_flight_health_reason=lambda: None,
        pose=pose,
        velocity=velocity,
        _v_cmd=np.zeros(2),
        _route_active=(None, np.zeros(2), None, None, None),
        _route_synchronized_site_origin=lambda: None,
        _now=lambda: 0.0,
        pose_t=0.0,
        velocity_t=0.0,
        route_reaction_time=0.5,
        route_brake_decel=0.25,
        _route_lib=SimpleNamespace(
            stopping_envelope_is_free=envelope_free),
        route_map_yaml='unused.yaml',
    )
    state._route_velocity_is_safe = lambda command: \
        ArucoLandingNode._route_velocity_is_safe(state, command)
    preferred = np.array([0.025, 0.0])
    result = ArucoLandingNode._route_safe_velocity(
        state, preferred, np.zeros(2))
    assert 0.0 < result[0] < preferred[0]
    assert ArucoLandingNode._route_velocity_is_safe(state, result)


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

    # The hardware-proven trailer mode retains its receipt-time contract.
    state.planned_cruise = False
    state.target_sample_t = float('nan')
    assert ArucoLandingNode._fresh_target(state)


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
        ArucoLandingNode._route_input_reason(state) or '')


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


def test_descend_dispatch_preserves_trailer_p_and_selects_route_mpc():
    calls = []
    state = SimpleNamespace(
        planned_cruise=True,
        _descend_mpc=lambda: calls.append('mpc'),
        _descend_p=lambda: calls.append('p'))
    ArucoLandingNode._descend(state)
    state.planned_cruise = False
    ArucoLandingNode._descend(state)
    assert calls == ['mpc', 'p']


def test_landing_mpc_streams_bounded_recovery_from_high_relative_speed():
    sent, holds = [], []
    mpc = LandingMPC(
        dt_s=0.1, horizon=20,
        w_xy=6.0, w_z=3.0, w_vxy=20.0, w_vz=1.5,
        w_a=0.05, w_terminal=40.0, w_jerk=0.5,
        v_max=0.8, a_max=0.6, vz_max=0.35,
        cone_k=2.0, j_max=1.5)
    state = SimpleNamespace(
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(
            x=0.0, y=0.0, z=3.0))),
        velocity=SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0))),
        marker=np.array([0.6, 0.0, 0.0]),
        marker_t=10.0,
        marker_vel=SimpleNamespace(v=np.array([1.0, 0.0])),
        marker_lost_abort=5.0,
        _landing_mpc=mpc,
        _landing_reference=HorizonReference(lead_s=0.1),
        _landing_solve_t=None,
        _landing_last_solve_t=None,
        _landing_acquiring=True,
        _landing_hold_z=3.0,
        _last_mpc_acceleration=np.zeros(3),
        _last_mpc_acceleration_t=None,
        route_landing_cone=2.0,
        route_landing_v_max=0.8,
        route_landing_vz_max=0.35,
        route_landing_entry_xy=0.5,
        route_landing_entry_speed=0.3,
        cruise_v_max=1.5,
        touch_alt=0.4,
        touch_xy=0.2,
        descend_log_period=1.0,
        _fresh_marker=lambda: True,
        _now=lambda: 10.0,
        _alt=lambda: 3.0,
        _predict_const_vel=predict_const_vel,
        _route_prediction_is_safe=lambda _positions: True,
        _route_velocity_is_safe=lambda _velocity: True,
        _send_pva=lambda p, v, a, j: (
            sent.append((np.asarray(p), np.asarray(v), np.asarray(a), j))
            or True),
        _hold=lambda: holds.append(True),
        _to=lambda _phase: None,
        _due=lambda *_args: False,
        get_logger=lambda: _Logger(),
    )
    state._reset_mpc_output = lambda: ArucoLandingNode._reset_mpc_output(
        state)

    ArucoLandingNode._descend_mpc(state)

    assert mpc.w_xy == 6.0 and mpc.w_vxy == 20.0
    assert len(sent) == 1 and holds == []
    commanded_velocity = sent[0][1]
    assert np.linalg.norm(commanded_velocity[:2]) <= state.cruise_v_max
    # The current relative speed is physically outside 0.8 m/s. LandingMPC's
    # feasible envelope brings it back under the bound over time; the wrapper
    # must stream that recovery instead of deadlocking in HOLD.
    relative_speed = np.linalg.norm(
        commanded_velocity[:2] - state.marker_vel.v)
    assert state.route_landing_v_max < relative_speed < 1.0


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
