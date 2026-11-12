"""Offline approval UI checks; no camera, OpenCV window, serial or FC."""
import json
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.time import Time
from std_msgs.msg import Bool, String

from balloon_locator.detection_viewer_node import DetectionViewerNode


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    import balloon_locator.detection_viewer_node as module
    clock = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    # Even an explicitly configured stale simulation flag must not affect hardware UI.
    flag = tmp_path / 'legacy_approval'
    flag.write_text('approved\n')
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true', '-p',
                    f'simulation_approval_file:={flag}'])
    node = DetectionViewerNode()
    node.get_clock().set_ros_time_override(Time(nanoseconds=10_000_000_000))
    sent = []
    node.approve_pub = SimpleNamespace(publish=lambda msg: sent.append(msg.data))
    try:
        yield node, sent, clock, flag
    finally:
        node.destroy_node()
        rclpy.shutdown()


def status(node, **updates):
    packet = dict(stamp=10., approval_state='waiting', coordinate_valid=True,
                  broadcasting=False, fc=dict(heartbeat_fresh=False, armed=None))
    packet.update(updates)
    node._on_bridge_status(String(data=json.dumps(packet)))


def test_request_is_not_confirmed_broadcast_or_flight_and_cancel_is_false(viewer):
    node, sent, _, flag = viewer
    assert 'UNKNOWN' in node._status_lines()[1]
    assert not node._request_approval(True)
    assert not sent
    status(node)
    assert node._request_approval(True)
    assert sent == [True]
    line, fc = node._status_lines()
    assert 'REQUEST SENT' in line and 'UNKNOWN' in fc
    assert flag.read_text() == 'approved\n'
    status(node, approval_state='approved', broadcasting=True,
           fc=dict(heartbeat_fresh=True, armed=True))
    line, fc = node._status_lines()
    assert 'TRANSMITTING' in line and 'ARMED' in fc and 'not confirmed' in fc
    node._request_approval(False)
    assert sent == [True, False]
    line, _ = node._status_lines()
    assert 'CANCEL REQUEST SENT' in line
    status(node, approval_state='cancelled')
    line, _ = node._status_lines()
    assert 'CANCELLED' in line and 'PAUSED' in line


def test_stale_malformed_bridge_status_cannot_enable_or_claim_state(viewer):
    node, sent, clock, _ = viewer
    status(node, approval_state='approved', broadcasting=True, fc=dict(heartbeat_fresh=True, armed=False))
    clock[0] += 2.
    assert not node._request_approval(True)
    assert 'UNKNOWN' in node._status_lines()[1]
    node._on_bridge_status(String(data='[]'))
    assert not node._request_approval(True)
    assert not sent


def test_approval_bool_means_request_only_and_restart_does_not_reuse_flag(viewer):
    node, _, _, _ = viewer
    node._on_approve_msg(Bool(data=True))
    line, fc = node._status_lines()
    assert 'REQUEST' in line and 'UNKNOWN' in fc
    node._on_approve_msg(Bool(data=False))
    assert not node.approval_requested
    assert not node.simulation_mode
    assert 'TRANSMITTING' not in node._status_lines()[0]


def test_viewer_restart_does_not_reuse_approval_request_or_flag(viewer):
    node, _, _, _ = viewer
    node._on_approve_msg(Bool(data=True))
    restarted = DetectionViewerNode()
    try:
        assert not restarted.approval_requested
        assert not restarted._coordinate_ready()
        assert 'NO CURRENT BRIDGE' in restarted._status_lines()[0]
    finally:
        restarted.destroy_node()


def test_explicit_simulation_topic_does_not_publish_hardware_approval():
    rclpy.init(args=['--ros-args', '-p', 'simulation_mode:=true', '-p', 'use_sim_time:=true'])
    node = DetectionViewerNode()
    try:
        assert node.approve_pub.topic_name == '/mission/approve_launch'
        assert node.simulation_mode
        assert node.simulation_flag is None
        assert 'SIMULATION' in node._status_lines()[0]
    finally:
        node.destroy_node()
        rclpy.shutdown()
