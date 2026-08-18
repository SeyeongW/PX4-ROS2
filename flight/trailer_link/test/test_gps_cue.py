import collections
import math
import threading
from collections import deque
from types import SimpleNamespace

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus

from trailer_link.geodesy import RelativeTarget, enu_offset
from trailer_link.radio_probe import GPS_MSG, POSITION_MSG, verdict
from trailer_link.trailer_gps_node import TrailerGpsNode, gpi_velocity_enu
from trailer_link.trailer_target_node import (
    LOCAL_ENU_FRAME,
    TrailerTargetNode,
    _cue_qos,
)


class _Sink:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Logger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, _message):
        pass


class _Clock:
    def now(self):
        return self

    def to_msg(self):
        return Time(sec=123, nanosec=456)


def _stamp(message, nanoseconds=100_000_000_000):
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(
        nanoseconds, 1_000_000_000)
    return message


def _fix(lat, lon, alt, *, epoch=100_000_000_000, valid=True):
    message = _stamp(NavSatFix(), epoch)
    message.status.status = (
        NavSatStatus.STATUS_FIX if valid else NavSatStatus.STATUS_NO_FIX)
    message.latitude, message.longitude, message.altitude = lat, lon, alt
    return message


def _target_node(now=100.0):
    node = object.__new__(TrailerTargetNode)
    node.target = RelativeTarget(
        stale_after=1.0, max_input_skew=0.5, max_distance=200.0)
    node.trailer_velocity_frame = 'earth_enu'
    node.vehicle_pose_frame = 'map'
    node.output_frame = LOCAL_ENU_FRAME
    node.deck_z = 1.811
    node.stale_after = 1.0
    node.max_input_skew = 0.5
    node.max_speed = 20.0
    node.min_source_rate = 0.0
    node.rate_window = 2.0
    node.stats_period = 2.0
    node._velocity = None
    node._velocity_stamp = None
    node._fix_epoch_ns = None
    node._velocity_epoch_ns = None
    node._last_published_epoch_ns = None
    node._fix_source_times = deque()
    node._vehicle_streams = {
        'fix': node._new_vehicle_stream(),
        'pose': node._new_vehicle_stream(),
    }
    node._vehicle_anchor_epochs = None
    node._last_block = 'starting up'
    node._published = 0
    node.cue_pub = _Sink()
    node.cue_velocity_pub = _Sink()
    node._test_now = now
    node._now = lambda: node._test_now
    node.get_logger = lambda: _Logger()
    return node


def _qualify_vehicle_anchor(node, vehicle, pose, *, step_ns=100_000_000):
    final_ns = (vehicle.header.stamp.sec * 1_000_000_000
                + vehicle.header.stamp.nanosec)
    _stamp(vehicle, final_ns - step_ns)
    _stamp(pose, final_ns - step_ns)
    node._on_vehicle_fix(vehicle)
    node._on_vehicle_pose(pose)
    _stamp(vehicle, final_ns)
    _stamp(pose, final_ns)
    node._on_vehicle_fix(vehicle)
    node._on_vehicle_pose(pose)


def test_wgs84_axes_and_gpi_velocity_are_enu():
    latitude, longitude = 36.64, 127.49
    east, north = enu_offset(
        latitude, longitude, latitude, longitude + 1.0e-5)
    assert math.isclose(east, 0.894, abs_tol=0.002)
    assert abs(north) < 1.0e-9
    east, north = enu_offset(
        latitude, longitude, latitude + 1.0e-5, longitude)
    assert abs(east) < 1.0e-9
    assert math.isclose(north, 1.110, abs_tol=0.002)
    assert gpi_velocity_enu(200.0, 300.0, 40.0) == (3.0, 2.0, 0.0)


def test_relative_target_refuses_no_fix_stale_and_anchor_skew():
    target = RelativeTarget(
        stale_after=1.0, max_input_skew=0.5, max_distance=200.0)
    target.on_trailer_fix(10.0, lat=36.64, lon=127.49, alt=50.0,
                          has_fix=False)
    target.on_vehicle_fix(10.0, lat=36.64, lon=127.49, alt=50.0,
                          has_fix=True)
    target.on_vehicle_local(10.0, x=0.0, y=0.0, z=5.0)
    assert target.solve(10.1) is None
    target.trailer_has_fix = True
    assert target.solve(11.1) is None
    target.on_trailer_fix(20.0, lat=36.64, lon=127.49, alt=50.0,
                          has_fix=True)
    target.on_vehicle_fix(20.0, lat=36.64, lon=127.49, alt=50.0,
                          has_fix=True)
    target.on_vehicle_local(20.6, x=0.0, y=0.0, z=5.0)
    assert target.solve(20.7) is None
    target.on_vehicle_fix(20.6, lat=36.64, lon=127.49, alt=50.0,
                          has_fix=True)
    assert target.solve(20.7) == (0.0, 0.0, 5.0)


def test_radio_sample_publishes_position_and_velocity_with_one_epoch():
    node = object.__new__(TrailerGpsNode)
    node.max_speed = 20.0
    node.frame_id = 'trailer_gps'
    node.velocity_frame = 'earth_enu'
    node.fix_pub = _Sink()
    node.velocity_pub = _Sink()
    node.get_clock = lambda: _Clock()
    node.get_logger = lambda: _Logger()
    sample = SimpleNamespace(
        lat=366400000, lon=1274900000, alt=50000,
        vx=200, vy=300, vz=40,
    )
    assert node._publish_pair(sample, 3)
    fix = node.fix_pub.messages[0]
    velocity = node.velocity_pub.messages[0]
    assert fix.header.stamp == velocity.header.stamp
    assert fix.header.frame_id == 'trailer_gps'
    assert velocity.header.frame_id == 'earth_enu'
    assert (velocity.vector.x, velocity.vector.y, velocity.vector.z) == (
        3.0, 2.0, 0.0)


def test_radio_handle_expires_fix_quality_and_reconnect_resets_it():
    node = object.__new__(TrailerGpsNode)
    node.max_speed = 20.0
    node.frame_id = 'trailer_gps'
    node.velocity_frame = 'earth_enu'
    node.fix_status_timeout = 1.5
    node.fix_pub = _Sink()
    node.velocity_pub = _Sink()
    node.get_clock = lambda: _Clock()
    node.get_logger = lambda: _Logger()
    node._lock = threading.Lock()
    node._link_up = False
    node._sysid = None
    node._fix_type = 0
    node._sats = 0
    node._t_last_fix_status = -math.inf
    node._n_msgs = node._n_gpi = node._n_pairs = 0
    node._t_last_rx = 0.0
    now = [10.0]
    node._now = lambda: now[0]

    gps = SimpleNamespace(
        get_type=lambda: 'GPS_RAW_INT', get_srcSystem=lambda: 1,
        fix_type=3, satellites_visible=14,
    )
    position = SimpleNamespace(
        get_type=lambda: 'GLOBAL_POSITION_INT', get_srcSystem=lambda: 1,
        lat=366400000, lon=1274900000, alt=50000,
        vx=200, vy=300, vz=0,
    )
    node._handle(gps)
    node._handle(position)
    assert node.fix_pub.messages[-1].status.status == NavSatStatus.STATUS_FIX

    now[0] = 12.0
    node._handle(position)
    assert (node.fix_pub.messages[-1].status.status
            == NavSatStatus.STATUS_NO_FIX)
    node._reset_link_state()
    assert node._fix_type == 0
    assert node._sysid is None
    assert not node._link_up


def test_target_publishes_one_atomic_epoch_with_fixed_deck_height():
    node = _target_node()
    trailer = _fix(36.64001, 127.49001, 10.0)
    vehicle = _fix(36.64, 127.49, 1000.0)
    velocity = _stamp(Vector3Stamped())
    velocity.header.frame_id = 'earth_enu'
    velocity.vector.x, velocity.vector.y = 3.0, 2.0
    pose = PoseStamped()
    _stamp(pose)
    pose.header.frame_id = 'map'
    pose.pose.position.x, pose.pose.position.y = 10.0, -4.0
    pose.pose.position.z = 6.0

    node._on_trailer_fix(trailer)
    node._on_trailer_velocity(velocity)
    _qualify_vehicle_anchor(node, vehicle, pose)
    node._tick()

    assert len(node.cue_pub.messages) == 1
    assert len(node.cue_velocity_pub.messages) == 1
    cue = node.cue_pub.messages[0]
    speed = node.cue_velocity_pub.messages[0]
    assert cue.header.frame_id == speed.header.frame_id == LOCAL_ENU_FRAME
    assert cue.header.stamp == speed.header.stamp
    assert cue.point.x > 10.0
    assert cue.point.y > -4.0
    assert cue.point.z == 1.811
    assert (speed.vector.x, speed.vector.y, speed.vector.z) == (3.0, 2.0, 0.0)

    node._tick()
    assert len(node.cue_pub.messages) == 1


def test_target_rejects_mismatched_epoch_bad_frame_and_stale_velocity():
    node = _target_node()
    trailer = _fix(36.64, 127.49, 50.0)
    vehicle = _fix(36.64, 127.49, 50.0)
    velocity = _stamp(Vector3Stamped(), 98_000_000_000)
    velocity.header.frame_id = 'earth_enu'
    pose = PoseStamped()
    _stamp(pose)
    pose.header.frame_id = 'map'
    node._on_trailer_fix(trailer)
    node._on_trailer_velocity(velocity)
    _qualify_vehicle_anchor(node, vehicle, pose)
    node._tick()
    assert not node.cue_pub.messages

    velocity.header.stamp = trailer.header.stamp
    velocity.header.frame_id = 'body'
    node._on_trailer_velocity(velocity)
    node._tick()
    assert not node.cue_pub.messages

    velocity.header.frame_id = 'earth_enu'
    node._on_trailer_velocity(velocity)
    node._test_now = 101.1
    node._tick()
    assert not node.cue_pub.messages


def test_target_rejects_stale_mavros_source_stamps_despite_fresh_delivery():
    node = _target_node()
    trailer = _fix(36.64, 127.49, 50.0)
    velocity = _stamp(Vector3Stamped())
    velocity.header.frame_id = 'earth_enu'
    stale_vehicle = _fix(36.64, 127.49, 50.0, epoch=1_000_000_000)
    stale_pose = _stamp(PoseStamped(), 1_000_000_000)
    stale_pose.header.frame_id = 'map'
    node._on_trailer_fix(trailer)
    node._on_trailer_velocity(velocity)
    node._on_vehicle_fix(stale_vehicle)
    node._on_vehicle_pose(stale_pose)
    node._tick()
    assert not node.cue_pub.messages


def test_vehicle_anchor_tolerates_latency_but_requires_source_progress():
    node = _target_node()
    trailer = _fix(36.64001, 127.49001, 50.0)
    velocity = _stamp(Vector3Stamped())
    velocity.header.frame_id = 'earth_enu'
    vehicle = _fix(36.64, 127.49, 50.0, epoch=98_900_000_000)
    pose = _stamp(PoseStamped(), 98_900_000_000)
    pose.header.frame_id = 'map'

    node._on_trailer_fix(trailer)
    node._on_trailer_velocity(velocity)
    node._on_vehicle_fix(vehicle)
    node._on_vehicle_pose(pose)
    _stamp(vehicle, 99_100_000_000)
    _stamp(pose, 99_100_000_000)
    node._on_vehicle_fix(vehicle)
    node._on_vehicle_pose(pose)
    node._tick()
    assert len(node.cue_pub.messages) == 1
    assert node.cue_pub.messages[0].header.stamp == trailer.header.stamp

    node._test_now = 101.01
    trailer = _fix(36.64001, 127.49001, 50.0,
                   epoch=101_010_000_000)
    velocity = _stamp(Vector3Stamped(), 101_010_000_000)
    velocity.header.frame_id = 'earth_enu'
    node._on_trailer_fix(trailer)
    node._on_trailer_velocity(velocity)
    node._tick()
    assert len(node.cue_pub.messages) == 1


def test_vehicle_anchor_regression_and_pair_skew_fail_closed():
    node = _target_node()
    vehicle = _fix(36.64, 127.49, 50.0)
    pose = _stamp(PoseStamped())
    pose.header.frame_id = 'map'
    _qualify_vehicle_anchor(node, vehicle, pose)
    assert node._vehicle_anchor_epochs is not None

    _stamp(vehicle, 99_000_000_000)
    node._on_vehicle_fix(vehicle)
    assert node._vehicle_anchor_epochs is None

    node = _target_node()
    vehicle = _fix(36.64, 127.49, 50.0, epoch=99_600_000_000)
    pose = _stamp(PoseStamped(), 99_000_000_000)
    pose.header.frame_id = 'map'
    node._on_vehicle_fix(vehicle)
    node._on_vehicle_pose(pose)
    _stamp(vehicle, 99_800_000_000)
    _stamp(pose, 99_200_000_000)
    node._on_vehicle_fix(vehicle)
    node._on_vehicle_pose(pose)
    assert node._vehicle_stream_block(100.0) is not None
    assert node._vehicle_anchor_epochs is None


def test_cue_qos_matches_reliable_mission_input():
    assert _cue_qos().reliability == ReliabilityPolicy.RELIABLE


def test_target_requires_a_measured_four_hz_position_stream():
    node = _target_node()
    node.min_source_rate = 4.0
    node._fix_source_times = deque([10.0, 10.2, 10.4])
    assert math.isclose(node._source_rate_hz(), 5.0)
    node._fix_source_times = deque([10.0, 10.5, 11.0])
    assert math.isclose(node._source_rate_hz(), 2.0)


def test_radio_probe_distinguishes_no_fix_from_disabled_position_stream():
    traffic = collections.Counter({GPS_MSG: 20, 'HEARTBEAT': 10})
    no_fix = verdict(traffic, 2, 5)
    stream_off = verdict(traffic, 3, 14)
    assert 'NO POSITION YET' in no_fix[0]
    assert 'STREAM IS SWITCHED OFF' in stream_off[0]
    assert 'SRx_POSITION' in stream_off[1]
    assert 'SRx_POSITION' not in no_fix[1]


def test_radio_probe_recognizes_position_stream():
    headline, _ = verdict(
        collections.Counter({GPS_MSG: 20, POSITION_MSG: 50}), 3, 14)
    assert 'POSITION IS ARRIVING' in headline
