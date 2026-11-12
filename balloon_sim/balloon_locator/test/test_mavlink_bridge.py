"""Hardware configuration checks without opening UDP or serial connections."""
import pytest
import rclpy

from balloon_locator.mavlink_bridge_node import MavlinkBridgeNode, mavutil


@pytest.mark.parametrize('bad_setting',[
    'cam_lat_deg:=91.0', 'cam_lon_deg:=181.0', 'rate_hz:=0.0',
    'max_age_sec:=0.0', 'target_system:=0',
])
def test_invalid_settings_fail_before_opening_hardware(monkeypatch,bad_setting):
    def forbidden_connection(*args,**kwargs):
        pytest.fail('configuration error reached a hardware connection')
    monkeypatch.setattr(mavutil,'mavlink_connection',forbidden_connection)
    rclpy.init(args=['--ros-args','-p','cam_lat_deg:=37.0','-p','cam_lon_deg:=127.0',
                    '-p','cam_height_above_home_m:=1.5','-p',bad_setting])
    try:
        with pytest.raises(RuntimeError,match='invalid_camera_reference'):
            MavlinkBridgeNode()
    finally:
        rclpy.shutdown()


@pytest.fixture
def bridge(monkeypatch):
    from types import SimpleNamespace
    from collections import deque
    from rclpy.time import Time
    import balloon_locator.mavlink_bridge_node as module
    sent, diagnostics, incoming = [], [], deque()
    clock = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    connection = SimpleNamespace(
        mav=SimpleNamespace(command_int_send=lambda *args: sent.append(args), heartbeat_send=lambda *args: None),
        recv_match=lambda **kwargs: incoming.popleft() if incoming else None,
        close=lambda: None)
    monkeypatch.setattr(mavutil, 'mavlink_connection', lambda *a, **k: connection)
    rclpy.init(args=['--ros-args','-p','cam_lat_deg:=37.0','-p','cam_lon_deg:=127.0',
                    '-p','cam_height_above_home_m:=1.5','-p','require_launch_approval:=true',
                    '-p','approval_timeout_sec:=2.0','-p','use_sim_time:=true'])
    node = MavlinkBridgeNode()
    node.get_clock().set_ros_time_override(Time(nanoseconds=10_000_000_000))
    node.pub_status = SimpleNamespace(publish=lambda msg: diagnostics.append(__import__('json').loads(msg.data)))
    node.point, node.point_stamp = (40.,0.,-10.), 9.9
    node.t_point = node.t_status = clock[0]
    node.status = dict(valid=True, frame_id='local_ned', pose_source='gimbal', measurement_stamp=9.9,
        measurement_wall_age=.1,n=30,sigma_m=.3,calibration_valid=True,station_offset_ned=[0.,0.,0.])
    try:
        yield node, sent, diagnostics, clock, incoming
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_approval_cancel_expiry_and_repeated_true_cannot_renew(bridge):
    from std_msgs.msg import Bool
    node, sent, statuses, clock, _ = bridge
    node._send()
    assert not sent and not node.launch_approved
    node._on_approval(Bool(data=True))
    node._send()
    assert len(sent) == 1 and statuses[-1]['broadcasting']
    clock[0] += .5
    node._on_approval(Bool(data=True))
    clock[0] += 1.6
    node.t_point = node.t_status = clock[0]
    node._send()
    assert not node.launch_approved and statuses[-1]['approval_state'] == 'expired'
    assert len(sent) == 1
    node._on_approval(Bool(data=True))
    node._send()
    assert len(sent) == 1  # held/repeated true cannot re-approve after expiry
    node._on_approval(Bool(data=False))
    node._on_approval(Bool(data=True))
    node._send()
    assert len(sent) == 2
    node._on_approval(Bool(data=False))
    node._send()
    assert len(sent) == 2 and statuses[-1]['approval_state'] == 'cancelled'


def test_stale_or_invalid_coordinate_cannot_accept_or_resume_approval(bridge):
    from std_msgs.msg import Bool
    node, sent, statuses, clock, _ = bridge
    node.status['measurement_wall_age'] = 5.
    node._on_approval(Bool(data=True))
    assert not node.launch_approved
    node.status['measurement_wall_age'] = .1
    node._send()
    assert not sent
    node._on_approval(Bool(data=False))
    node._on_approval(Bool(data=True))
    node.status['calibration_valid'] = False
    node._send()
    assert not sent and not node.launch_approved
    node.status['calibration_valid'] = True
    node._send()
    assert not sent


def test_fc_heartbeat_source_and_age_do_not_imply_flight(bridge):
    from types import SimpleNamespace
    node, _, statuses, clock, incoming = bridge
    def heartbeat(system, component, armed):
        return SimpleNamespace(get_srcSystem=lambda:system,get_srcComponent=lambda:component,
            base_mode=128 if armed else 0,autopilot=12,type=2)
    incoming.extend([heartbeat(2,1,True),heartbeat(1,191,True)])
    node._send()
    assert statuses[-1]['fc']['armed'] is None
    incoming.append(heartbeat(1,1,True))
    node._send()
    assert statuses[-1]['fc']['heartbeat_fresh']
    assert statuses[-1]['fc']['armed'] is True
    assert 'airborne' not in statuses[-1]['fc']
    clock[0] += 4.
    node._send()
    assert statuses[-1]['fc']['armed'] is None


def test_transport_failure_revokes_approval_without_retrying(bridge):
    from std_msgs.msg import Bool
    node, sent, statuses, _, _ = bridge
    node._on_approval(Bool(data=True))
    def failed(*args):
        raise OSError('disconnected')
    node.mav.mav.command_int_send = failed
    node._send()
    assert statuses[-1]['approval_state'] == 'transport_error'
    assert not node.launch_approved and not statuses[-1]['broadcasting']
    node.mav.mav.command_int_send = lambda *args: sent.append(args)
    node._send()
    assert not sent


@pytest.mark.parametrize('value', ['0.0', '-1.0', '.nan', '.inf'])
def test_invalid_approval_timeout_never_opens_transport(monkeypatch, value):
    monkeypatch.setattr(mavutil, 'mavlink_connection', lambda *a, **k: pytest.fail('opened transport'))
    rclpy.init(args=['--ros-args','-p','cam_lat_deg:=37.0','-p','cam_lon_deg:=127.0',
                    '-p','cam_height_above_home_m:=1.5','-p',f'approval_timeout_sec:={value}'])
    try:
        with pytest.raises(RuntimeError, match='invalid_camera_reference'):
            MavlinkBridgeNode()
    finally:
        rclpy.shutdown()


def test_restart_resets_approval_even_with_valid_coordinates(bridge):
    from std_msgs.msg import Bool
    node, _, _, _, _ = bridge
    node._on_approval(Bool(data=True))
    assert node.launch_approved
    restarted = MavlinkBridgeNode()
    try:
        assert not restarted.launch_approved
        assert restarted.approval_state == 'waiting'
        assert restarted.approval_deadline == 0.
        assert restarted.point is None and restarted.status is None
    finally:
        restarted.destroy_node()


def test_false_cancels_before_expiry_and_receive_error_is_visible(bridge):
    from std_msgs.msg import Bool
    node, sent, statuses, _, _ = bridge
    node._on_approval(Bool(data=True))
    node._send()
    assert len(sent) == 1
    node._on_approval(Bool(data=False))
    node._send()
    assert len(sent) == 1 and statuses[-1]['approval_state'] == 'cancelled'
    node._on_approval(Bool(data=True))
    def disconnected(**kwargs):
        raise OSError('serial unplugged')
    node.mav.recv_match = disconnected
    node._send()
    assert len(sent) == 1 and statuses[-1]['approval_state'] == 'transport_error'
    assert not statuses[-1]['fc']['heartbeat_fresh']


@pytest.mark.parametrize('first', ['point', 'status'])
def test_coordinate_callbacks_in_either_order_pause_without_cancelling(bridge, first):
    import json
    from geometry_msgs.msg import PointStamped
    from rclpy.time import Time
    from std_msgs.msg import Bool, String
    node, sent, statuses, _, _ = bridge
    node._on_approval(Bool(data=True))
    node._send()
    assert len(sent) == 1
    point = PointStamped()
    point.header.frame_id = node.expect_frame
    point.header.stamp = Time(nanoseconds=9_950_000_000).to_msg()
    point.point.x, point.point.y, point.point.z = node.point
    packet = dict(node.status, measurement_stamp=9.95)
    updates = dict(point=lambda: node._on_point(point),
                   status=lambda: node._on_status(String(data=json.dumps(packet))))
    updates[first]()
    node._send()
    assert node.launch_approved  # normal asynchronous callbacks must not revoke
    assert len(sent) == 1 and not statuses[-1]['broadcasting']
    updates['status' if first == 'point' else 'point']()
    node._send()
    assert node.launch_approved and len(sent) == 2
    assert statuses[-1]['broadcasting']


def test_unmatched_pair_cannot_hide_stale_point_or_invalid_status(bridge):
    from std_msgs.msg import Bool
    node, sent, statuses, _, _ = bridge
    node._on_approval(Bool(data=True))
    node.point_stamp = 8.0  # receipt looks fresh but exposure is stale
    node._send()
    assert not node.launch_approved and not sent
    assert statuses[-1]['approval_state'] == 'coordinates_invalid'
