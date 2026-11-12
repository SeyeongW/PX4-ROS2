"""A final static-point occlusion may not hide stale control or target changes."""
import ast
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

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
