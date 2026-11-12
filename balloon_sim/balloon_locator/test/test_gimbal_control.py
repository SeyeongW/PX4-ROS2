"""Gimbal commands require current observations; fake UDP never touches hardware."""
import time
from types import SimpleNamespace

import pytest
import rclpy
from geometry_msgs.msg import Vector3, Vector3Stamped
from rclpy.parameter import Parameter
from rclpy.time import Time
from vision_msgs.msg import Detection2D, Detection2DArray

from balloon_locator.gimbal_tracker_node import GimbalTrackerNode
from balloon_locator.siyi_gimbal_node import SiyiGimbalNode
from balloon_locator import siyi_protocol as sp


@pytest.fixture
def tracker():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = GimbalTrackerNode()
    commands = []
    node.pub = SimpleNamespace(publish=commands.append)
    node.get_clock().set_ros_time_override(Time(nanoseconds=2000000000))
    try:
        yield node, commands
    finally:
        node.destroy_node()
        rclpy.shutdown()


def detection(stamp, x=1200., y=540.):
    msg = Detection2DArray()
    msg.header.stamp = Time(nanoseconds=round(stamp * 1e9)).to_msg()
    det = Detection2D()
    det.bbox.center.position.x = x
    det.bbox.center.position.y = y
    msg.detections = [det]
    return msg


@pytest.mark.parametrize('stamp,x', [(1.,1200.), (3.,1200.), (2.,float('nan'))])
def test_tracker_ignores_stale_future_and_nonfinite_detection(tracker, stamp, x):
    node, commands = tracker
    node._on_det(detection(stamp, x))
    node._tick()
    assert not commands


def test_tracker_replay_does_not_reverse_command(tracker):
    node, commands = tracker
    node._on_det(detection(2.))
    node._tick()
    assert commands[-1].x > 0
    node._on_det(detection(1.9, 100.))
    node._tick()
    assert commands[-1].x > 0


@pytest.mark.parametrize('cause', ['no_detection', 'wall_timeout', 'clock_reset'])
def test_tracker_stops_on_loss_pause_and_clock_reset(tracker, cause):
    node, commands = tracker
    node._on_det(detection(2.))
    node._tick()
    if cause == 'no_detection':
        node.get_clock().set_ros_time_override(Time(nanoseconds=2100000000))
        msg = detection(2.1)
        msg.detections = []
        node._on_det(msg)
    elif cause == 'wall_timeout':
        node.last_time = time.monotonic() - 2.
    else:
        node.get_clock().set_ros_time_override(Time(nanoseconds=1000000000))
    node._tick()
    assert commands[-1].x == commands[-1].y == 0.


def test_siyi_malformed_payload_does_not_stop_polling():
    packets = iter([(sp.build_frame(sp.CMD_ATTITUDE, b'\x00'), ('192.168.144.25', 37260)),
                    (sp.build_frame(sp.CMD_CURRENT_ZOOM, bytes([2, 0])), ('192.168.144.25', 37260))])
    def receive(_):
        try:
            return next(packets)
        except StopIteration:
            raise BlockingIOError
    zooms = []
    fake = SimpleNamespace(rate_active=False, addr=('192.168.144.25', 37260),
        _send=lambda *args:None, sock=SimpleNamespace(recvfrom=receive),
        get_logger=lambda:SimpleNamespace(warn=lambda *args, **kwargs:None),
        pub_zoom=SimpleNamespace(publish=zooms.append))
    SiyiGimbalNode._poll(fake)
    assert len(zooms) == 1 and zooms[0].data == 2.


def test_siyi_nonfinite_rate_cannot_reach_hardware():
    sent = []
    fake = SimpleNamespace(_send=lambda *args:sent.append(args))
    SiyiGimbalNode._on_rate(fake, Vector3(x=float('nan')))
    assert not sent


def test_siyi_receive_error_recovers_on_next_poll():
    errors = []
    def receive(_):
        raise ConnectionResetError('simulated UDP receive failure')
    fake = SimpleNamespace(rate_active=False, _send=lambda *args:None,
        sock=SimpleNamespace(recvfrom=receive),
        get_logger=lambda:SimpleNamespace(warn=lambda *args, **kwargs:errors.append(args)))
    SiyiGimbalNode._poll(fake)
    assert len(errors) == 1
    def empty(_):
        raise BlockingIOError
    fake.sock.recvfrom = empty
    SiyiGimbalNode._poll(fake)
    assert len(errors) == 1


def attitude(stamp=2., yaw=0., pitch=20.):
    msg = Vector3Stamped()
    msg.header.stamp = Time(nanoseconds=round(stamp * 1e9)).to_msg()
    msg.vector = Vector3(x=yaw, y=pitch)
    return msg


def empty_detection(stamp=2.):
    msg = detection(stamp)
    msg.detections = []
    return msg


def enable_search(node):
    assert node.set_parameters([Parameter('enable_search', value=True)])[0].successful
    locks = []
    node.pub_locked = SimpleNamespace(publish=locks.append)
    return locks


@pytest.mark.parametrize('missing', ['both', 'image', 'attitude'])
def test_search_requires_current_image_processing_and_attitude(tracker, missing):
    node, commands = tracker
    locks = enable_search(node)
    if missing == 'image':
        node._on_att(attitude())
    if missing == 'attitude':
        node._on_det(empty_detection())
    node._tick()
    assert commands[-1].x == commands[-1].y == 0.
    assert not locks[-1].data


def test_fresh_empty_detection_permits_search_without_fabricated_lock(tracker):
    node, commands = tracker
    locks = enable_search(node)
    node._on_att(attitude())
    node._on_det(empty_detection())
    node._tick()
    assert commands[-1].x != 0. or commands[-1].y != 0.
    assert not locks[-1].data


@pytest.mark.parametrize('fault', ['image_age', 'attitude_age', 'image_wall',
                                  'attitude_wall', 'attitude_nan', 'clock_reset'])
def test_search_sensor_fault_stops_and_clears_lock(tracker, fault):
    node, commands = tracker
    locks = enable_search(node)
    node._on_att(attitude())
    node._on_det(detection(2.))
    node.locked = True
    node._tick()
    assert commands[-1].x != 0.
    if fault in ('image_age', 'attitude_age'):
        node.get_clock().set_ros_time_override(Time(nanoseconds=2700000000))
        if fault == 'image_age':
            node._on_att(attitude(2.7))
        else:
            node._on_det(detection(2.7))
    elif fault == 'image_wall':
        node.last_frame_time = time.monotonic() - 2.
    elif fault == 'attitude_wall':
        node.last_att_time = time.monotonic() - 2.
    elif fault == 'attitude_nan':
        node._on_att(attitude(2.01, yaw=float('nan')))
    else:
        node.get_clock().set_ros_time_override(Time(nanoseconds=1000000000))
    node._tick()
    assert commands[-1].x == commands[-1].y == 0.
    assert not locks[-1].data and not node.locked


def test_siyi_watchdog_retries_failed_stop_on_next_poll():
    sent = []
    def send(cmd, data=b''):
        sent.append((cmd, data))
        return len([item for item in sent if item[0] == sp.CMD_ROTATE]) >= 2
    def empty(_):
        raise BlockingIOError
    fake = SimpleNamespace(rate_active=True, rate_cmd=(10., 0.),
        rate_cmd_time=time.monotonic() - 2., cmd_timeout=.3,
        _send=send, sock=SimpleNamespace(recvfrom=empty),
        get_logger=lambda:SimpleNamespace(warn=lambda *args, **kwargs:None))
    SiyiGimbalNode._poll(fake)
    assert fake.rate_active
    SiyiGimbalNode._poll(fake)
    assert not fake.rate_active
    assert [item for item in sent if item[0] == sp.CMD_ROTATE] == [
        (sp.CMD_ROTATE, sp.encode_rotate(0, 0))] * 2


@pytest.mark.parametrize('success', [False, True])
def test_siyi_center_reports_only_send_result(success):
    fake = SimpleNamespace(_send=lambda *args:success)
    response = SiyiGimbalNode._on_center(fake, None, SimpleNamespace())
    assert response.success is success
    assert 'unconfirmed' in response.message


@pytest.mark.parametrize('key,value', [
    ('search_yaw_rate', float('nan')), ('search_pitch_rate', 101.),
    ('search_period_sec', 0.), ('search_yaw_limit_deg', 181.),
    ('search_pitch_min_deg', 50.), ('search_pitch_max_deg', -91.),
    ('search_resume_timeout_sec', -1.), ('lock_confirm_frames', 0),
])
def test_runtime_rejects_invalid_search_configuration(tracker, key, value):
    node, _ = tracker
    before = node.get_parameter(key).value
    result = node.set_parameters([Parameter(key, value=value)])[0]
    assert not result.successful
    assert node.get_parameter(key).value == before


@pytest.mark.parametrize('search', [False, True])
def test_low_max_rate_preserves_tracking_and_clamps_search(tracker, search):
    node, commands = tracker
    assert node.set_parameters([Parameter('max_rate', value=10.)])[0].successful
    if search:
        enable_search(node)
        node._on_att(attitude())
        node._on_det(empty_detection())
    else:
        node._on_det(detection(2.))
    node._tick()
    assert commands[-1].x == 10.
    assert abs(commands[-1].y) <= 10.


@pytest.mark.parametrize('stamp', [0., 1., 3.])
def test_search_rejects_unstamped_old_or_future_attitude(tracker, stamp):
    node, commands = tracker
    enable_search(node)
    node._on_att(attitude(stamp))
    node._on_det(empty_detection())
    node._tick()
    assert commands[-1].x == commands[-1].y == 0.


@pytest.mark.parametrize('stream', ['image', 'attitude'])
def test_replayed_stream_cannot_refresh_wall_watchdog(tracker, stream):
    node, commands = tracker
    enable_search(node)
    node._on_att(attitude())
    node._on_det(empty_detection())
    node._tick()
    if stream == 'image':
        node.last_frame_time = time.monotonic() - 2.
        node._on_det(empty_detection())
    else:
        node.last_att_time = time.monotonic() - 2.
        node._on_att(attitude())
    node._tick()
    assert commands[-1].x == commands[-1].y == 0.


def test_new_empty_frames_resume_search_after_target_loss_grace(tracker):
    node, commands = tracker
    locks = enable_search(node)
    node._on_att(attitude())
    node._on_det(detection(2.))
    node.locked = True
    node._tick()
    node.get_clock().set_ros_time_override(Time(nanoseconds=2100000000))
    node._on_det(empty_detection(2.1))
    node._tick()
    assert commands[-1].x == commands[-1].y == 0. and locks[-1].data
    node.last_time = time.monotonic() - 3.
    node._tick()
    assert commands[-1].x != 0. or commands[-1].y != 0.
    assert not locks[-1].data


@pytest.mark.parametrize('outcome', ['success', 'oserror', 'short'])
def test_siyi_send_result_uses_actual_datagram_result(outcome):
    frames = []
    def sendto(frame, address):
        frames.append((sp.parse_frame(frame), address))
        if outcome == 'oserror':
            raise OSError('fake network down')
        return len(frame) if outcome == 'success' else len(frame) - 1
    fake = SimpleNamespace(seq=65535, addr=('192.168.144.25', 37260),
        sock=SimpleNamespace(sendto=sendto),
        get_logger=lambda:SimpleNamespace(warn=lambda *args, **kwargs:None))
    assert SiyiGimbalNode._send(fake, sp.CMD_ROTATE, sp.encode_rotate(0, 0)) is (outcome == 'success')
    assert frames[0][0].seq == 0 and frames[0][1] == fake.addr


def test_failed_direct_stop_cannot_be_overwritten_by_motion_before_retry():
    sent = []
    def send(cmd, data=b''):
        sent.append((cmd, data))
        return len(sent) > 1
    def empty(_):
        raise BlockingIOError
    fake = SimpleNamespace(stop_pending=False, rate_active=True, cmd_timeout=.3,
        _send=send, sock=SimpleNamespace(recvfrom=empty))
    SiyiGimbalNode._on_rate(fake, Vector3())
    assert fake.stop_pending
    SiyiGimbalNode._on_rate(fake, Vector3(x=20.))
    assert len(sent) == 1
    SiyiGimbalNode._poll(fake)
    assert not fake.stop_pending and not fake.rate_active
    assert sent[:2] == [(sp.CMD_ROTATE, sp.encode_rotate(0, 0))] * 2


def test_siyi_ignores_foreign_sender_before_parsing():
    packets = iter([(sp.build_frame(sp.CMD_CURRENT_ZOOM, bytes([2, 0])),
                     ('192.168.144.99', 37260))])
    def receive(_):
        try:
            return next(packets)
        except StopIteration:
            raise BlockingIOError
    zooms = []
    fake = SimpleNamespace(rate_active=False, addr=('192.168.144.25', 37260),
        _send=lambda *args:True, sock=SimpleNamespace(recvfrom=receive),
        pub_zoom=SimpleNamespace(publish=zooms.append))
    SiyiGimbalNode._poll(fake)
    assert not zooms
