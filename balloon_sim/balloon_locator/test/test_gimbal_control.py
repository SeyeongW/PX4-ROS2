"""Gimbal commands require current observations; fake UDP never touches hardware."""
import time
from types import SimpleNamespace

import pytest
import rclpy
from geometry_msgs.msg import Vector3
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
