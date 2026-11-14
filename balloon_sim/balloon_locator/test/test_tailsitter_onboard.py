"""Onboard regressions use fake transports only; no socket or flight is opened."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ONBOARD = Path(__file__).resolve().parents[1] / 'onboard'
sys.path.insert(0, str(ONBOARD))
import common


def load_controller():
    spec = importlib.util.spec_from_file_location('tested_tailsitter', ONBOARD / 'tailsitter_onboard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def message(kind, **values):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: 1,
                           get_srcComponent=lambda: 1, **values)


def test_default_cli_opens_no_transport(monkeypatch):
    module = load_controller()
    def forbidden(*args, **kwargs):
        pytest.fail('default CLI attempted hardware I/O')
    monkeypatch.setattr(module.mavutil, 'mavlink_connection', forbidden)
    monkeypatch.setattr(module.socket, 'socket', forbidden)
    assert module.main([]) == 0


@pytest.mark.parametrize('yaw', [0., .7, 2.4, -2.8])
def test_nose_crossing_horizontal_has_no_roll_flip(yaw):
    module = load_controller()
    noses = [np.array([np.cos(yaw), np.sin(yaw), z]) for z in (-1e-4, 0., 1e-4)]
    quaternions = [module.nose_dir_to_quat_frd(n) for n in noses]
    for first, second in zip(quaternions, quaternions[1:]):
        angle = 2. * np.arccos(np.clip(abs(np.dot(first, second)), 0., 1.))
        assert angle < .001
    for n, q in zip(noses, quaternions):
        rotation = module.quat_to_rot(*q)
        np.testing.assert_allclose(-rotation[:, 2], n / np.linalg.norm(n), atol=1e-12)
        np.testing.assert_allclose(rotation[:, 1], [-np.sin(yaw), np.cos(yaw), 0.], atol=1e-12)


def test_local_ned_is_validated_ordered_and_required_for_feedback():
    received = dict(GLOBAL_POSITION_INT=1., HEARTBEAT=1., EXTENDED_SYS_STATE=1., ATTITUDE_QUATERNION=1.)
    stamps = {}
    local = message('LOCAL_POSITION_NED', time_boot_ms=1000, x=3., y=8., z=-2., vx=0., vy=0., vz=0.)
    assert not common.telemetry_fresh(received, 1., .5, attitude=True, local_position=True)
    assert common.record_telemetry(local, received, stamps, 1.)
    assert common.telemetry_fresh(received, 1., .5, attitude=True, local_position=True)
    for timestamp in (1000, 999, float('nan'), -1, 1000.5):
        local.time_boot_ms = timestamp
        assert not common.record_telemetry(local, received, stamps, 1.1)
    local.time_boot_ms = 1100
    local.x = float('nan')
    assert not common.record_telemetry(local, received, stamps, 1.1)
    received.update(GLOBAL_POSITION_INT=2., HEARTBEAT=2., EXTENDED_SYS_STATE=2., ATTITUDE_QUATERNION=2.)
    assert not common.telemetry_fresh(received, 2., .5, attitude=True, local_position=True)


@pytest.mark.parametrize('response,expected', [
    ('ack_and_mode', True), ('mode_and_ack', True), ('ack_only', False),
    ('mode_only', False), ('reject', False), ('silence', False), ('wrong_vehicle', False)])
@pytest.mark.parametrize('submode', [common.PX4_AUTO_LOITER, 5])
def test_auto_handoff_requires_ack_and_observed_mode(monkeypatch, response, expected, submode):
    module = load_controller()
    clock = [100.]
    def now():
        clock[0] += .05
        return clock[0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=now, sleep=lambda _: None))
    ack = message('COMMAND_ACK', command=common.M.MAV_CMD_DO_SET_MODE,
                  result=common.M.MAV_RESULT_DENIED if response == 'reject' else common.M.MAV_RESULT_ACCEPTED)
    mode = message('HEARTBEAT', base_mode=common.M.MAV_MODE_FLAG_SAFETY_ARMED,
                   custom_mode=(common.PX4_AUTO << 16) | (submode << 24))
    pending = {'ack_and_mode': [ack, mode], 'mode_and_ack': [mode, ack],
               'ack_only': [ack], 'mode_only': [mode], 'reject': [ack],
               'silence': [], 'wrong_vehicle': [ack, mode]}[response]
    if response == 'wrong_vehicle':
        ack.get_srcSystem = mode.get_srcSystem = lambda: 2
    sent = []
    mav = SimpleNamespace(target_system=1, target_component=1,
                          recv_match=lambda **_: pending.pop(0) if pending else None,
                          mav=SimpleNamespace(command_long_send=lambda *p: sent.append(p)))
    assert module.confirm_auto_mode(mav, submode, lambda _: None) is expected
    assert len(sent) == 1
    assert clock[0] < 106.


@pytest.mark.parametrize('fault_stage', ['TAKEOFF', 'TRANSITION', 'TS_UKF_APPROACH'])
@pytest.mark.parametrize('fault_kind', ['guidance', 'local_telemetry', 'ground_only'])
def test_expired_guidance_stops_every_active_stage(monkeypatch, fault_stage, fault_kind):
    module = load_controller()
    clock = [100.]
    state = {'stage': 'WAIT_TARGET', 'mode': 0, 'armed': False, 'slot': 0, 'fault_at': None}
    sent, attitude_times, acks, logs, handoffs, takeoff_noses, mode_times = [], [], [], [], [], [], []
    def now():
        clock[0] += .005
        return clock[0]
    def log(line, **_):
        logs.append(line)
        for stage in ('TAKEOFF', 'TRANSITION', 'TS_UKF_APPROACH', 'TS_LOS_PURSUIT'):
            if f'[STAGE -> {stage}]' in line:
                state['stage'] = stage
                if stage == fault_stage:
                    state['fault_at'] = clock[0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=now, sleep=lambda _: None))
    monkeypatch.setattr(module, 'print', log, raising=False)
    def receive(**_):
        if clock[0] > 112.:
            pytest.fail('expired guidance did not stop the actual onboard loop')
        if acks:
            return acks.pop(0)
        state['slot'] += 1
        slot = state['slot'] % 7
        altitude = 20. if state['armed'] and fault_stage != 'TAKEOFF' else 0.
        if slot == 0:
            return message('HEARTBEAT', base_mode=128 if state['armed'] else 0, custom_mode=state['mode'])
        if slot == 1:
            return message('GLOBAL_POSITION_INT', time_boot_ms=state['slot'] * 20,
                           lat=375000000, lon=1270000000, relative_alt=int(altitude * 1000), vx=0, vy=0, vz=0)
        if slot == 2:
            return message('HOME_POSITION', latitude=375000000, longitude=1270000000, altitude=143000)
        if slot == 3:
            return message('EXTENDED_SYS_STATE', landed_state=common.M.MAV_LANDED_STATE_IN_AIR if state['armed'] else common.M.MAV_LANDED_STATE_ON_GROUND)
        if slot == 4:
            return message('ATTITUDE_QUATERNION', time_boot_ms=state['slot'] * 20, q1=1., q2=0., q3=0., q4=0.)
        if slot == 5:
            if fault_kind == 'local_telemetry' and state['fault_at'] is not None:
                return None
            return message('LOCAL_POSITION_NED', time_boot_ms=state['slot'] * 20,
                           x=3., y=8., z=2. - altitude, vx=0., vy=0., vz=0.)
        if fault_kind in ('guidance', 'ground_only') and state['fault_at'] is not None:
            return None
        packet = message('COMMAND_INT', command=common.M.MAV_CMD_USER_1, target_system=1,
                         target_component=191, frame=common.M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                         param1=30., param2=.3, param3=0., param4=0., x=375001000, y=1270000000, z=30.)
        packet.get_srcSystem, packet.get_srcComponent = lambda: 250, lambda: 191
        return packet
    def send(*p):
        sent.append(p)
        if p[2] == common.M.MAV_CMD_DO_SET_MODE:
            state['mode'] = int(p[5]) << 16 | int(p[6]) << 24
            mode_times.append(clock[0])
        elif p[2] == common.M.MAV_CMD_COMPONENT_ARM_DISARM:
            assert state['mode'] >> 16 & 0xFF == common.PX4_OFFBOARD
            state['armed'] = bool(p[4])
        else:
            return
        acks.append(message('COMMAND_ACK', command=p[2], result=common.M.MAV_RESULT_ACCEPTED))
    def attitude(*p):
        attitude_times.append(clock[0])
        if state['stage'] == 'TAKEOFF':
            takeoff_noses.append(-module.quat_to_rot(*p[4])[:, 2])
    mav = SimpleNamespace(target_system=1, target_component=1, recv_match=receive, close=lambda: None,
                          mav=SimpleNamespace(command_long_send=send, heartbeat_send=lambda *p: None,
                                              set_attitude_target_send=attitude))
    def no_vision(*_):
        raise BlockingIOError
    import struct
    vis_sent_at = [-1.]
    def vision(*_):
        if (state['fault_at'] is None or fault_kind == 'ground_only') and clock[0] - vis_sent_at[0] > .02:
            vis_sent_at[0] = clock[0]
            return struct.pack('<4f', 1., 0., 0., 3.), ('127.0.0.1', 15760)
        return no_vision()
    sock = SimpleNamespace(bind=lambda *p: None, setblocking=lambda *p: None, recvfrom=vision, close=lambda: None)
    monkeypatch.setattr(module.mavutil, 'mavlink_connection', lambda *p, **kw: mav)
    monkeypatch.setattr(module.socket, 'socket', lambda *p: sock)
    monkeypatch.setattr(module, 'wait_fcu', lambda *p, **kw: None)
    monkeypatch.setattr(module, 'confirm_auto_mode', lambda *p: handoffs.append(p[1]) or True)
    original_control = module.compute_tailsitter_cmd
    def control(**kwargs):
        assert kwargs.get('los_ray_ned') is None, 'untimestamped image must not steer the aircraft'
        assert not kwargs.get('allow_dive', False)
        return original_control(**kwargs)
    monkeypatch.setattr(module, 'compute_tailsitter_cmd', control)
    target_timeout = '2.' if fault_kind == 'local_telemetry' else '.3'
    assert module.main(['--execute', '--timeout', target_timeout, '--lost', '3.']) == 1
    assert state['fault_at'] is not None
    limit = .6 if fault_kind == 'local_telemetry' else .4
    assert clock[0] - state['fault_at'] < limit
    assert not attitude_times or attitude_times[-1] - state['fault_at'] <= limit
    reason = 'FCU telemetry stale' if fault_kind == 'local_telemetry' else 'guidance_stale'
    assert any(reason in line for line in logs)
    assert not any('[STAGE -> TS_LOS_PURSUIT]' in line for line in logs)
    for nose in takeoff_noses:
        np.testing.assert_allclose(nose, [0., 0., -1.], atol=1e-12)
    assert not mode_times or mode_times[0] - attitude_times[0] >= 1.
