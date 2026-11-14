"""Legacy simulation guards: deterministic fake I/O, never UDP/serial/arming."""
from collections import deque
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location('legacy_guidance_under_test',
    Path(__file__).resolve().parents[2] / 'tools' / 'tailsitter_guidance.py')
guidance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guidance)


@pytest.fixture
def clock(monkeypatch, tmp_path):
    ticks = [100.0]
    monkeypatch.setattr(guidance, 'time', SimpleNamespace(
        monotonic=lambda: ticks[0], time=lambda: ticks[0],
        sleep=lambda dt: ticks.__setitem__(0, ticks[0] + dt)))
    monkeypatch.setattr(guidance, 'APPROVAL_FLAG_FILE', tmp_path / 'approval')
    monkeypatch.setattr(guidance, 'AUTO_APPROVE', False)
    return ticks


def estimate(stamp=10.0):
    return dict(valid=True, stamp=stamp, frame_id='world', source='camera_monocular_size',
        position_semantics='known_sphere_center', calibration_revision='sim-calibration',
        calibration_source='gazebo_balloon_locator_camera_info_tf',
        evidence=dict(ground_truth_used=False, target_is_static=True,
            image_source_stamp=stamp, camera_tf_stamp=stamp),
        position=[-2., 10., 5.2], covariance=(np.eye(3) * .01).tolist(), maximum_position_std_m=.1)


def message(kind, source=(1, 1), **values):
    defaults = dict(time_boot_ms=1000, x=0., y=0., z=-5., vx=0., vy=0., vz=0.,
        q1=1., q2=0., q3=0., q4=0., base_mode=128, custom_mode=6 << 16,
        autopilot=guidance.mavutil.mavlink.MAV_AUTOPILOT_PX4)
    defaults.update(values)
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: source[0],
        get_srcComponent=lambda: source[1], **defaults)


def fresh_telemetry(clock):
    state = guidance.TelemetryState(1, 1)
    for kind in ('LOCAL_POSITION_NED', 'ATTITUDE_QUATERNION', 'HEARTBEAT'):
        assert state.accept(message(kind), clock[0])
    return state


def test_cancel_expiry_and_old_approval_file_cannot_launch(clock):
    import os
    path = guidance.APPROVAL_FLAG_FILE
    path.write_text('approved')
    os.utime(path, (99., 99.))
    vision = guidance.VisionListener()
    assert not vision.is_approved()
    vision.set_approved(True)
    assert vision.is_approved()
    vision.set_approved(False)
    assert not vision.is_approved()
    vision.set_approved(True)
    clock[0] += guidance.APPROVAL_TTL + .1
    vision.set_approved(True)  # a repeated true must not renew the request
    assert not vision.is_approved()
    vision.set_approved(False)
    vision.set_approved(True)
    assert vision.is_approved()
    vision.begin_mission()
    vision.set_approved(False)
    vision.set_approved(True)  # cannot silently restart an aborted mission
    assert not vision.is_approved() and vision.cancel_requested


def test_fresh_file_approval_expires_and_does_not_override_cancel(clock):
    import os
    vision = guidance.VisionListener()
    clock[0] += 1
    guidance.APPROVAL_FLAG_FILE.write_text('approved')
    os.utime(guidance.APPROVAL_FLAG_FILE, (clock[0], clock[0]))
    assert vision.is_approved()
    vision.begin_mission()
    guidance.APPROVAL_FLAG_FILE.unlink()
    assert vision.cancellation_pending()
    vision.set_approved(False)
    assert not vision.is_approved()


@pytest.mark.parametrize('change', [
    dict(valid=False), dict(stamp=-189.), dict(stamp=11.),
    dict(frame_id='local_ned'), dict(source='gazebo_truth'),
    dict(calibration_revision=''), dict(maximum_position_std_m=float('nan')),
    dict(maximum_position_std_m=2.8), dict(position=[float('inf'), 0, 0]),
    dict(covariance=(np.eye(3) * 9.).tolist()),
])
def test_invalid_estimate_cannot_become_lock(clock, change):
    vision = guidance.VisionListener()
    data = estimate()
    data.update(change)
    assert not vision.update_estimate(data, 10.)
    assert not vision.target_ready()


def test_estimate_replay_does_not_refresh_measurement_age(clock):
    vision = guidance.VisionListener()
    assert vision.update_estimate(estimate(), 10.)
    assert vision.target_ready()
    clock[0] += .6
    assert not vision.update_estimate(estimate(), 10.6)
    clock[0] += .3
    assert not vision.target_ready()
    assert not vision.update_estimate(estimate(9.9), 10.)
    assert vision.update_estimate(estimate(10.9), 10.9)
    assert vision.target_ready()
    assert not vision.update_estimate(dict(valid=False), 10.9)
    assert not vision.target_ready()


def test_telemetry_rejects_foreign_replay_nan_and_expires(clock):
    telemetry = fresh_telemetry(clock)
    telemetry.require_fresh(clock[0], armed=True)
    clock[0] += .4
    assert not telemetry.accept(message('LOCAL_POSITION_NED', source=(2, 1), time_boot_ms=1100), clock[0])
    assert not telemetry.accept(message('LOCAL_POSITION_NED', time_boot_ms=1000), clock[0])
    assert not telemetry.accept(message('LOCAL_POSITION_NED', time_boot_ms=1200, x=float('nan')), clock[0])
    assert not telemetry.accept(message('ATTITUDE_QUATERNION', time_boot_ms=1200, q1=0.), clock[0])
    clock[0] += .2
    with pytest.raises(guidance.GuidanceAbort, match='telemetry_stale'):
        telemetry.require_fresh(clock[0], armed=True)


@pytest.mark.parametrize('failure', ['cancel', 'target', 'telemetry', 'mode'])
def test_control_guard_emits_no_setpoint_after_fault(clock, failure):
    telemetry = fresh_telemetry(clock)
    vision = guidance.VisionListener()
    vision.update_estimate(estimate(), 10.)
    conn = SimpleNamespace(recv_match=lambda **kw: None)
    if failure == 'cancel':
        vision.set_approved(False)
    elif failure == 'target':
        vision.update_estimate(dict(valid=False), 10.)
    elif failure == 'telemetry':
        clock[0] += .6
    else:
        telemetry.offboard = False
    sent = []
    with pytest.raises(guidance.GuidanceAbort):
        guidance.require_guidance(conn, telemetry, vision, armed=True)
        sent.append('attitude')
    assert sent == []


@pytest.mark.parametrize('previous,current,expected', [
    ([-1., 0., 0.], [1., 0., 0.], True),
    ([-1., .149, 0.], [1., .149, 0.], True),
    ([-1., .151, 0.], [1., .151, 0.], False),
    ([-1., 5., 0.], [1., 5., 0.], False),
    ([1., 0., 0.], [2., 0., 0.], False),
    ([1., 0., 0.], [-1., 0., 0.], False),
])
def test_pass_requires_forward_plane_crossing_and_bounded_miss(previous, current, expected):
    passed, _ = guidance.plane_crossing(previous, current, [0, 0, 0], [1, 0, 0])
    assert passed is expected


def test_safe_mode_requires_observed_matching_fc_heartbeat(clock, capsys):
    commands = []
    queue = deque([
        message('COMMAND_ACK', command=guidance.mavutil.mavlink.MAV_CMD_DO_SET_MODE, result=0),
        message('HEARTBEAT', source=(2, 1), custom_mode=(4 << 16) | (5 << 24)),
    ])
    def receive(**kwargs):
        guidance.time.sleep(.1)
        return queue.popleft() if queue else None
    conn = SimpleNamespace(target_system=1, target_component=1, recv_match=receive,
        mav=SimpleNamespace(command_long_send=lambda *args: commands.append(args)))
    assert not guidance.transfer_to_px4(conn, 'RTL', timeout=.5)
    assert 'FC 모드 확인=False' in capsys.readouterr().out
    queue.append(message('HEARTBEAT', custom_mode=(4 << 16) | (3 << 24)))
    assert guidance.transfer_to_px4(conn, 'HOLD', timeout=.5)
    assert commands[-1][5:7] == (4, 3)


@pytest.fixture
def fake_flight(clock, monkeypatch):
    """Drive the real run_mission loop using deterministic FC messages, no sockets."""
    commands, positions, attitudes = [], [], []
    state = dict(last_tick=None, pending=deque(), z=-5., x=0., y=0., stop=False)
    def receive(**kwargs):
        if state['stop']:
            return None
        if state['last_tick'] != clock[0]:
            state['last_tick'] = clock[0]
            state['pending'].extend(message(kind, time_boot_ms=round(clock[0] * 1000),
                x=state['x'], y=state['y'], z=state['z']) for kind in ('LOCAL_POSITION_NED', 'ATTITUDE_QUATERNION', 'HEARTBEAT'))
        return state['pending'].popleft() if state['pending'] else None
    conn = SimpleNamespace(target_system=1, target_component=1, recv_match=receive,
        mav=SimpleNamespace(command_long_send=lambda *a: commands.append(a),
            set_position_target_local_ned_send=lambda *a: positions.append((clock[0], a)),
            set_attitude_target_send=lambda *a: attitudes.append((clock[0], a))))
    vision = guidance.VisionListener()
    vision.set_approved(True)
    monkeypatch.setattr(vision, 'target_ready', lambda: True)
    monkeypatch.setattr(vision, 'get_ground_ukf', lambda: ([-2., 10., 5.2], .1, 3, True, 'ready', 0., 1, 10.))
    monkeypatch.setattr(guidance, 'set_param', lambda *a: None)
    return conn, vision, state, commands, positions, attitudes


def test_takeoff_timeout_never_transitions_to_attitude_guidance(fake_flight):
    conn, vision, state, _, positions, attitudes = fake_flight
    state['z'] = 0.
    with pytest.raises(guidance.GuidanceAbort, match='takeoff_altitude_timeout'):
        guidance.run_mission(conn, vision)
    assert positions and not attitudes


@pytest.mark.parametrize('fault', ['cancel', 'telemetry', 'target'])
def test_actual_loop_stops_on_midflight_fault(fake_flight, clock, monkeypatch, fault):
    conn, vision, state, _, _, attitudes = fake_flight
    onset = []
    original_sleep = guidance.time.sleep
    def sleep(dt):
        original_sleep(dt)
        if attitudes and not onset:
            onset.append(clock[0])
            if fault == 'cancel':
                vision.set_approved(False)
            elif fault == 'telemetry':
                state['stop'] = True
            else:
                monkeypatch.setattr(vision, 'target_ready', lambda: False)
    monkeypatch.setattr(guidance.time, 'sleep', sleep)
    with pytest.raises(guidance.GuidanceAbort):
        guidance.run_mission(conn, vision)
    assert attitudes
    assert attitudes[-1][0] <= onset[0] + guidance.TELEMETRY_MAX_AGE
    assert clock[0] <= onset[0] + guidance.TELEMETRY_MAX_AGE + .04


def test_actual_pass_requests_hold_only_after_two_metres_beyond_plane(fake_flight, monkeypatch):
    conn, vision, state, _, _, attitudes = fake_flight
    transfer = []
    original_sleep = guidance.time.sleep
    def sleep(dt):
        original_sleep(dt)
        if attitudes:
            state['x'] += .25
    monkeypatch.setattr(guidance.time, 'sleep', sleep)
    monkeypatch.setattr(guidance, 'transfer_to_px4', lambda c, mode: transfer.append((mode, state['x'])) or True)
    guidance.run_mission(conn, vision)
    assert transfer == [('HOLD', 12.)]


def test_actual_loop_reports_five_metre_plane_miss_as_failure(fake_flight, monkeypatch):
    conn, vision, state, _, _, attitudes = fake_flight
    original_sleep = guidance.time.sleep
    def sleep(dt):
        original_sleep(dt)
        if attitudes:
            state['x'] += .25
            state['y'] = 5.
    monkeypatch.setattr(guidance.time, 'sleep', sleep)
    monkeypatch.setattr(guidance, 'transfer_to_px4', lambda *args: pytest.fail('miss requested HOLD'))
    with pytest.raises(guidance.GuidanceAbort, match='waypoint_missed:radial_error=5.000m'):
        guidance.run_mission(conn, vision)


def test_abort_entrypoint_requests_rtl_and_closes_transport(clock, monkeypatch):
    monkeypatch.setenv('FLY_EXPERIMENTAL_TAILSITTER', '1')
    actions = []
    conn = SimpleNamespace(target_system=1, target_component=1,
        wait_heartbeat=lambda **kw: message('HEARTBEAT'), close=lambda: actions.append('close'))
    monkeypatch.setattr(guidance, 'start_ros_vision_thread', lambda v: None)
    monkeypatch.setattr(guidance.mavutil, 'mavlink_connection', lambda address: conn)
    monkeypatch.setattr(guidance, 'threading', SimpleNamespace(Lock=lambda: __import__('threading').Lock(),
        Thread=lambda **kw: SimpleNamespace(start=lambda: None)))
    def abort(*args):
        raise guidance.GuidanceAbort('approval_cancelled')
    monkeypatch.setattr(guidance, 'run_mission', abort)
    monkeypatch.setattr(guidance, 'transfer_to_px4', lambda c, mode: actions.append(mode) or True)
    guidance.main()
    assert actions == ['RTL', 'close']


def test_legacy_opt_in_is_required_before_any_ros_or_mavlink_connection(monkeypatch):
    monkeypatch.delenv('FLY_EXPERIMENTAL_TAILSITTER', raising=False)
    monkeypatch.setattr(guidance, 'start_ros_vision_thread', lambda *a: pytest.fail('started ROS'))
    monkeypatch.setattr(guidance.mavutil, 'mavlink_connection', lambda *a: pytest.fail('opened MAVLink'))
    with pytest.raises(SystemExit, match='explicit experimental_tailsitter'):
        guidance.main()


def test_cancel_at_last_prestream_sleep_prevents_mode_and_arm(fake_flight, monkeypatch):
    conn, vision, _, commands, positions, _ = fake_flight
    original_sleep = guidance.time.sleep
    def sleep(dt):
        original_sleep(dt)
        if len(positions) == 35:
            vision.set_approved(False)
    monkeypatch.setattr(guidance.time, 'sleep', sleep)
    with pytest.raises(guidance.GuidanceAbort, match='approval_cancelled'):
        guidance.run_mission(conn, vision)
    assert all(command[2] not in (guidance.mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        guidance.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM) for command in commands)


def test_cancel_after_mode_request_prevents_arm_even_if_already_offboard(fake_flight, monkeypatch):
    conn, vision, _, commands, _, _ = fake_flight
    original_send = conn.mav.command_long_send
    def command(*args):
        original_send(*args)
        if args[2] == guidance.mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            vision.set_approved(False)
    monkeypatch.setattr(conn.mav, 'command_long_send', command)
    with pytest.raises(guidance.GuidanceAbort, match='approval_cancelled'):
        guidance.run_mission(conn, vision)
    assert all(command[2] != guidance.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM for command in commands)


def test_delayed_onboard_pixels_cannot_steer_without_exposure_pose(fake_flight, monkeypatch):
    conn, vision, state, _, _, attitudes = fake_flight
    transfer = []
    # The old code accepted this 200ms-old image and projected it using latest R_bn.
    monkeypatch.setattr(vision, 'get', lambda: (320., 180., 20., 20., 20., 0, .2, 100.))
    monkeypatch.setattr(guidance, 'pixel_to_los_ned',
        lambda *args: pytest.fail('pixel guidance used a pose not matched to image exposure'))
    original_sleep = guidance.time.sleep
    def sleep(dt):
        original_sleep(dt)
        if attitudes:
            state['x'] += .25
    monkeypatch.setattr(guidance.time, 'sleep', sleep)
    monkeypatch.setattr(guidance, 'transfer_to_px4', lambda c, mode: transfer.append(mode) or True)
    guidance.run_mission(conn, vision)
    assert transfer == ['HOLD']
