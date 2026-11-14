"""A final static-point occlusion may not hide stale control or target changes."""
import ast
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from balloon_locator import stl_mission


def scenario():
    return dict(now=20., phase='spline_tracking', aircraft_type='stl_pacer_quad',
        options={'target_is_virtual_waypoint': True, 'target_is_pass_through_gate': False},
        locked=dict(valid=True, source='camera_monocular_size', frame_id='world', target_id=1,
                    position=[0., -40., 10.], calibration_revision='sim1', evidence={'target_is_static': True}),
        plan={'goal': [0., -40., 10.]},
        own={'stamp': 20., 'position': [0., -39.5, 10.], 'velocity': [0., -.1, 0.]},
        loss_since=19., loss_position=[0., -39.3, 10.],
        latest_estimate=dict(stamp=20., valid=False, frame_id='world', target_id=1,
                             source='camera_monocular_size', reason='no_detection'))


def test_only_frozen_terminal_coordinate_is_used_without_relabeling_measurement():
    data = scenario()
    before = deepcopy(data)
    assert stl_mission.terminal_static_hold_allowed(**data)
    assert data == before
    assert data['latest_estimate']['valid'] is False


@pytest.mark.parametrize('change', [
    'far', 'loss_started_far', 'timeout', 'takeoff', 'gate', 'physical_target',
    'moving_target', 'different_goal', 'wrong_aircraft', 'stale_own', 'future_own',
    'fresh_conflict', 'target_id', 'sensor_frame', 'calibration', 'reset', 'no_start', 'tf',
])
def test_terminal_exception_rejects_unsafe_or_unrelated_cases(change):
    data = scenario()
    if change == 'far': data['own']['position'][1] = -38.9
    elif change == 'loss_started_far': data['loss_position'][1] = -38.9
    elif change == 'timeout': data['loss_since'] = 4.9
    elif change == 'takeoff': data['phase'] = 'takeoff'
    elif change == 'gate': data['plan']['gate'] = {'width_m': 1.2}
    elif change == 'physical_target': data['options']['target_is_virtual_waypoint'] = False
    elif change == 'moving_target': data['locked']['evidence']['target_is_static'] = False
    elif change == 'different_goal': data['plan']['goal'][1] = -41.
    elif change == 'wrong_aircraft': data['aircraft_type'] = 'tailsitter20cm_reference'
    elif change == 'stale_own': data['own']['stamp'] = 19.7
    elif change == 'future_own': data['own']['stamp'] = 20.1
    elif change == 'fresh_conflict': data['latest_estimate']['valid'] = True
    elif change == 'target_id': data['latest_estimate']['target_id'] = 2
    elif change == 'sensor_frame': data['latest_estimate']['frame_id'] = 'local_ned'
    elif change == 'calibration': data['latest_estimate']['reason'] = 'balloon_calibration_unconfirmed'
    elif change == 'reset': data['latest_estimate']['reason'] = 'camera_acquisition_reset'
    elif change == 'no_start': data['latest_estimate']['reason'] = 'waiting_for_acquisition_start'
    elif change == 'tf': data['latest_estimate']['reason'] = 'camera_tf_unavailable'
    assert not stl_mission.terminal_static_hold_allowed(**data)


@pytest.mark.parametrize('distance,speed,expected', [(.4, .1, True), (.6, .1, False), (.4, .3, False)])
def test_arrived_can_hold_only_inside_goal_region_at_low_speed(distance, speed, expected):
    data = scenario()
    data.update(phase='arrived', loss_since=1.)
    data['own'].update(position=[0., -40.+distance, 10.], velocity=[speed, 0., 0.])
    assert bool(stl_mission.terminal_static_hold_allowed(**data)) is expected


def source_method(name):
    """Execute the actual nested method without starting ROS, PX4 or Gazebo."""
    tree = ast.parse(Path(stl_mission.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    cls = next(n for n in main.body if isinstance(n, ast.ClassDef) and n.name == 'Mission')
    tick = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(stl_mission))
    namespace['Trigger'] = SimpleNamespace(Request=lambda: object())
    exec(compile(ast.Module(body=[tick], type_ignores=[]), stl_mission.__file__, 'exec'), namespace)
    return namespace[name]


def source_tick():
    return source_method('tick')


def reset_state(height=.4076, backend='mavros'):
    state = SimpleNamespace(backend=backend, phase='landed', locked={'target_id': 1},
        plan={'goal': [0., -40., 10.]}, events=[], start=1., landing=object())
    state.now = lambda: 20.
    state.own_history = [dict(stamp=20., position=[2., .2, height], velocity=[0., 0., .055])]
    state.controller_history = [dict(stamp=20., connected=True, armed=False,
        controller_ready=True, landed=True, landed_state=1, fcu_state_stamp=20.)]
    state.active = lambda: False
    state.grounded = lambda now: source_method('grounded')(state, now)
    state.invalidate_plan = lambda: setattr(state, 'plan', None)
    state.estimate = lambda now: None
    state.transition = lambda phase: setattr(state, 'phase', phase)
    state.publish_status = lambda now: None
    state.save = lambda: None
    return state


@pytest.mark.parametrize('height', [.4076, 3.5])
def test_mavros_grounded_reset_uses_fresh_landed_feedback_not_absolute_altitude(height):
    state = reset_state(height)
    own_before = deepcopy(state.own_history)
    assert state.grounded(20.)
    response = source_method('action')(state, 'reset', SimpleNamespace())
    assert response.success, response.message
    assert state.phase == 'detecting'
    assert state.locked is state.plan is state.start is state.landing is None
    assert state.own_history == own_before  # Reset must never move the aircraft.


@pytest.mark.parametrize('condition', [
    'stale_controller', 'future_controller', 'missing_controller', 'disconnected',
    'not_ready', 'armed', 'airborne', 'unknown_landed', 'expired_extended',
    'stale_fcu', 'future_fcu', 'missing_fcu_stamp', 'moving', 'stale_own', 'future_own',
])
def test_mavros_grounded_and_reset_reject_unverified_landing_even_near_world_ground(condition):
    state = reset_state(.1)
    controller = state.controller_history[-1]
    own = state.own_history[-1]
    if condition == 'stale_controller': controller['stamp'] = 19.69
    elif condition == 'future_controller': controller['stamp'] = 20.01
    elif condition == 'missing_controller': state.controller_history.clear()
    elif condition == 'disconnected': controller['connected'] = False
    elif condition == 'not_ready': controller['controller_ready'] = False
    elif condition == 'armed': controller['armed'] = True
    elif condition == 'airborne': controller.update(landed=False, landed_state=2)
    elif condition == 'unknown_landed': controller['landed_state'] = 0
    elif condition == 'expired_extended': controller['landed'] = False
    elif condition == 'stale_fcu': controller['fcu_state_stamp'] = 17.99
    elif condition == 'future_fcu': controller['fcu_state_stamp'] = 20.01
    elif condition == 'missing_fcu_stamp': controller.pop('fcu_state_stamp')
    elif condition == 'moving': own['velocity'] = [.151, 0., 0.]
    elif condition == 'stale_own': own['stamp'] = 19.79
    elif condition == 'future_own': own['stamp'] = 20.01
    assert not state.grounded(20.)
    response = source_method('action')(state, 'reset', SimpleNamespace())
    assert not response.success
    assert response.message == 'reset_requires_grounded_vehicle_no_pose_teleport'
    assert state.phase == 'landed' and state.plan and state.locked


@pytest.mark.parametrize('height,expected', [(.1, True), (.4076, False), (3.5, False)])
def test_native_grounded_reset_retains_height_guard(height, expected):
    state = reset_state(height, backend='native')
    state.controller_history.clear()
    assert state.grounded(20.) is expected
    response = source_method('action')(state, 'reset', SimpleNamespace())
    assert response.success is expected


def test_reset_still_rejects_active_mission_with_landed_feedback():
    state = reset_state()
    state.active = lambda: True
    response = source_method('action')(state, 'reset', SimpleNamespace())
    assert not response.success
    assert response.message == 'abort_and_land_before_reset'


@pytest.mark.parametrize('phase', ['coordinate_ready', 'preview_ready'])
def test_continuous_path_never_implies_operator_lock_or_launch(phase):
    state = tick_state()
    state.phase = phase
    state.locked = None
    state.options['continuous_takeoff'] = True
    state.estimate = lambda now: {'valid': True}
    state.lock_block_reason = lambda now: ''
    state.can_launch = lambda now: True
    calls = []
    state.lock_estimate = lambda now: calls.append('lock')
    state.start_client = SimpleNamespace(call_async=lambda request: calls.append('arm'))
    source_tick()(state)
    assert state.phase == phase
    assert calls == []


def tick_state():
    data = scenario()
    state = SimpleNamespace(**{k: v for k, v in data.items()
        if k not in ('now', 'loss_since', 'loss_position', 'latest_estimate')})
    state.clock = 20.
    state.now = lambda: state.clock
    state.last_now = 20.
    state.own_history = [state.own]
    state.estimates = deque([data['latest_estimate']])
    state.sensor_unavailable_since = None
    state.terminal_loss_position = state.terminal_loss_eligible = None
    state.advance_preview_reference = state.retry_ground_reference_preview = lambda now: None
    state.estimate = lambda now: None
    state.coordinate_matches = state.calibration_changed = lambda now: False
    state.active = lambda: False
    state.landing_reasons = []
    state.begin_landing = lambda reason: state.landing_reasons.append(reason)
    state.publish_status = lambda now, sample: None
    state.last_save = 20.
    state.save = state.publish_plan = lambda: None
    return state


def test_real_tick_retains_outage_deadline_and_never_revives_aged_conflict():
    tick = source_tick()
    state = tick_state()
    state.estimates[-1]['valid'] = True  # A fresh, nonmatching coordinate.
    tick(state)
    assert state.terminal_loss_eligible is False
    state.clock = state.own['stamp'] = 21.1
    tick(state)  # The conflicting observation is now stale, but remains disqualifying.
    assert state.coordinate_mode == 'unavailable'
    assert state.sensor_unavailable_since == 20.
    assert state.landing_reasons == ['sensor_coordinate_unavailable']


def test_real_tick_terminal_mode_still_executes_controller_loss_watchdog():
    state = tick_state()
    state.active = lambda: True
    state.ready = lambda now: False
    state.backend = 'native'
    state.begin_emergency_landing = lambda reason: state.landing_reasons.append(reason)
    source_tick()(state)
    assert state.coordinate_mode == 'locked_static_terminal'
    assert state.landing_reasons == ['own_or_controller_lost_watchdog_descent']


def test_real_tick_terminal_timeout_aborts_without_refreshing_deadline():
    tick = source_tick()
    state = tick_state()
    tick(state)
    assert state.coordinate_mode == 'locked_static_terminal'
    state.clock = state.own['stamp'] = 35.1
    state.estimates[-1]['stamp'] = 35.1
    tick(state)
    assert state.sensor_unavailable_since == 20.
    assert state.coordinate_mode == 'unavailable'
    assert state.landing_reasons == ['sensor_coordinate_unavailable']


def start_future(message='fresh_landed_EKF_reference_and_prestream_required', *, success=False, error=None, done=True):
    return SimpleNamespace(done=lambda: done, exception=lambda: error,
        result=lambda: SimpleNamespace(success=success, message=message))


def arming_state():
    state = tick_state()
    state.phase = 'arming'
    state.active = lambda: True
    state.backend = 'mavros'
    state.ready = state.grounded = lambda now: True
    state.controller = dict(origin_world=[0., 0., 0.], north_yaw_world_rad=1.57,
        armed=False, fcu_system_status=3, landed_state=1, mode='STANDBY')
    state.lock_frame = dict(origin_world=[0., 0., 0.], north_yaw_world_rad=1.57)
    state.arm_requested = 20.
    state.arm_retry_count = 0
    state.arm_retry_at = None
    state.arm_future = start_future()
    state.events = []
    state.start_requests = []

    def start(request):
        state.start_requests.append(state.clock)
        return start_future()

    state.start_client = SimpleNamespace(service_is_ready=lambda: True, call_async=start)
    method = source_method('retry_transient_start')
    state.retry_transient_start = lambda now, message: method(state, now, message)
    return state


def tick_at(state, time):
    state.clock = state.own['stamp'] = time
    source_tick()(state)


def test_real_tick_retries_only_explicit_prearm_refusal_with_bounded_spacing():
    state = arming_state()
    tick_at(state, 20.)
    assert state.arm_retry_at == pytest.approx(20.35)
    tick_at(state, 20.2)
    assert not state.start_requests
    tick_at(state, 20.36)
    tick_at(state, 20.40)
    tick_at(state, 20.76)
    tick_at(state, 20.80)
    tick_at(state, 21.16)
    tick_at(state, 21.20)
    assert state.start_requests == [20.36, 20.76, 21.16]
    assert state.arm_retry_count == 3
    assert state.arm_requested == 20.  # Original launch deadline never restarts.
    assert state.landing_reasons == ['PX4_start_rejected: fresh_landed_EKF_reference_and_prestream_required']


@pytest.mark.parametrize('condition', ['armed', 'unhealthy', 'airborne', 'not_grounded', 'service_missing', 'late'])
def test_retry_never_bypasses_current_preflight_conditions(condition):
    state = arming_state()
    if condition == 'armed': state.controller['armed'] = True
    elif condition == 'unhealthy': state.controller['fcu_system_status'] = 2
    elif condition == 'airborne': state.controller['landed_state'] = 2
    elif condition == 'not_grounded': state.grounded = lambda now: False
    elif condition == 'service_missing': state.start_client.service_is_ready = lambda: False
    tick_at(state, 23. if condition == 'late' else 20.)
    assert not state.start_requests and state.arm_retry_at is None
    assert state.landing_reasons == ['PX4_start_rejected: fresh_landed_EKF_reference_and_prestream_required']


@pytest.mark.parametrize('message,error', [
    ('duplicate_start_or_airborne_vehicle', None),
    ('MAVROS_mode_and_arming_services_required', None),
    ('PX4_arm_rejected', None),
    ('', RuntimeError('start_service_disconnected')),
])
def test_real_tick_retains_rejection_reason_and_never_retries_other_failures(message, error):
    state = arming_state()
    state.arm_future = start_future(message, error=error)
    tick_at(state, 20.)
    assert not state.start_requests and state.arm_retry_at is None
    assert state.landing_reasons == ['PX4_start_rejected: '+(str(error) if error else message)]


def test_accepted_start_is_not_repeated_and_original_15s_timeout_remains():
    state = arming_state()
    state.arm_future = start_future('OFFBOARD_arming_requested_wait_for_armed_state', success=True)
    tick_at(state, 20.)
    assert not state.start_requests and not state.landing_reasons
    tick_at(state, 35.01)
    assert not state.start_requests
    assert state.landing_reasons == ['PX4_arming_timeout']


def flying_gate_state():
    """Actual planner and tick methods, with ROS publishers/services replaced."""
    state = tick_state()
    state.options.update(continuous_takeoff=True, takeoff_join_speed=.15,
                         target_is_pass_through_gate=True, max_speed=4.,
                         max_acceleration=2., max_jerk=6., body_radius=.3,
                         gate_exit_distance=5., standoff=0., controller_profile='aggressive_multicopter')
    state.plan = stl_mission.build_static_plan([-2., 0., .2], [0., -40., 10.], **state.options)
    state.start = 20.
    state.clock = state.start + state.plan['segments'][0]['duration_s'] + .5
    sample = stl_mission.sample_plan(state.plan, state.clock-state.start)
    state.own.update(stamp=state.clock, position=sample['position'], velocity=sample['velocity'])
    state.phase = 'spline_tracking'
    state.backend = 'mavros'
    state.active = state.ready = lambda *args: True
    state.coordinate_matches = lambda now: True
    state.estimate = lambda now: dict(state.locked, stamp=now)
    state.controller = dict(origin_world=[0., 0., 0.], north_yaw_world_rad=1.57,
                            controller_ready=True, setpoint_frame='jol/flight_ned',
                            armed=True, mode='OFFBOARD')
    state.lock_frame = dict(origin_world=[0., 0., 0.], north_yaw_world_rad=1.57)
    state.reference_flight = {}
    state.reference_history = deque()
    state.transfer_start = state.clock
    state.checkpoint = dict(outcome='pending')
    state.gate_feedback = None
    state.dwell = None
    state.events, state.samples, state.flown = [], [], []
    state.commands, state.return_requests, state.reset_requests = [], [], []
    state.vision_last_target = None
    state.return_client = SimpleNamespace(service_is_ready=lambda: True,
                                         call_async=lambda _: state.return_requests.append(True))
    state.balloon_reset_client = SimpleNamespace(service_is_ready=lambda: True,
                                                call_async=lambda _: state.reset_requests.append(True))
    state.command = lambda sample: state.commands.append(deepcopy(sample))
    state.publish_status = lambda now, sample=None: None
    state.aircraft_metadata = lambda: {}
    state.publish_path = lambda *args: None
    state.flown_pub = None
    def transition(phase, reason=''):
        state.phase, state.reason, state.dwell = phase, reason, None
        state.events.append(dict(phase=phase, reason=reason))
    state.transition = transition
    def landing(reason):
        state.landing_reasons.append(reason)
        transition('autopilot_landing', reason)
    state.begin_landing = landing
    state.plan_time = lambda now: source_method('plan_time')(state, now)
    state.check_gate = lambda now: source_method('check_gate')(state, now)
    return state


def test_fresh_ground_estimate_keeps_actual_spline_and_locked_gate():
    state = flying_gate_state()
    original = deepcopy(state.plan)
    expected = stl_mission.sample_plan(state.plan, state.clock-state.start)
    source_tick()(state)
    assert state.phase == 'spline_tracking'
    assert state.plan == original
    assert state.checkpoint['outcome'] == 'pending'
    assert not state.reset_requests
    np.testing.assert_allclose(state.commands[-1]['position'], expected['position'])


@pytest.mark.parametrize('checkpoint', ['pending', 'passed'])
def test_goal_proximity_requires_observed_gate_pass(checkpoint):
    state = flying_gate_state()
    state.clock = state.start + state.plan['duration_s'] + .1
    state.own.update(stamp=state.clock, position=list(state.plan['goal']), velocity=[0., 0., 0.])
    state.checkpoint = dict(outcome=checkpoint)
    tick_at(state, state.clock)
    assert state.phase == 'spline_tracking'
    for _ in range(12):
        tick_at(state, state.clock + .1)
        if state.phase != 'spline_tracking':
            break
    if checkpoint == 'passed':
        assert state.phase == 'arrived'
    else:
        assert state.phase == 'autopilot_landing'
        assert state.landing_reasons == ['checkpoint_crossing_unobserved_return_to_takeoff']
    assert not state.reset_requests


def test_arrived_holds_five_metres_beyond_gate_without_automatic_rtl():
    state = flying_gate_state()
    state.phase = 'arrived'
    state.checkpoint = dict(outcome='passed')
    state.clock = state.start + state.plan['duration_s'] + .1
    state.own.update(stamp=state.clock, position=list(state.plan['goal']), velocity=[0., 0., 0.])
    # A prior camera field must not replace the planned post-gate hold point.
    state.vision_last_target = np.array(state.locked['position'])
    tick_at(state, state.clock)
    tick_at(state, state.clock + 5.1)
    assert state.phase == 'arrived'
    assert not state.return_requests
    assert np.linalg.norm(np.array(state.plan['goal'])-state.locked['position']) == pytest.approx(5.)
    for command in state.commands:
        np.testing.assert_allclose(command['position'], state.plan['goal'])
        np.testing.assert_allclose(command['velocity'], [0., 0., 0.])
