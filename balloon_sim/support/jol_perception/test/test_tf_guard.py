"""Sparse-TF interpolation must not invent a rotating sensor pose."""
import math

import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from tf2_ros import Buffer, TransformException

from jol_perception.tf_guard import BracketedBuffer, guarded_tf


def transform(child, stamp, yaw=0., parent='world'):
    message = TransformStamped()
    message.header.frame_id, message.child_frame_id = parent, child
    message.header.stamp = Time(nanoseconds=round(stamp*1e9)).to_msg()
    message.transform.rotation.z = math.sin(yaw/2)
    message.transform.rotation.w = math.cos(yaw/2)
    return message


def test_sparse_bracket_is_rejected_until_delayed_nearby_samples_arrive():
    ordinary, guarded = Buffer(), BracketedBuffer(.03)
    for stamp, yaw in [(1., 0.), (1.5, .5)]:
        msg = transform('sensor', stamp, yaw)
        ordinary.set_transform(msg, 'test')
        guarded.set_transform(msg, 'test')
    when = Time(seconds=1.25)
    # tf2 succeeds despite a 0.5s gap: asking for a timestamp is not proof that
    # nearby rig measurements supported that interpolation.
    ordinary.lookup_transform('world', 'sensor', when)
    with pytest.raises(TransformException, match='interpolation gap'):
        guarded.lookup_transform('world', 'sensor', when)
    guarded.lookup_transform('world', 'sensor', Time(seconds=1.5))  # True exact sample is valid.
    for stamp, yaw in [(1.245, .398), (1.255, .402)]:
        guarded.set_transform(transform('sensor', stamp, yaw), 'late_test')
    result = guarded.lookup_transform('world', 'sensor', when)
    assert math.isclose(2*math.atan2(result.transform.rotation.z, result.transform.rotation.w), .4)
    assert math.isclose(guarded.last_support['sensor']['gap_s'], .01)
    assert guarded.guard_rejections == 1


def test_only_requested_path_dynamic_edges_need_brackets_and_static_identity_are_valid():
    guarded = BracketedBuffer()
    guarded.set_transform_static(transform('rig', 0.), 'test')
    guarded.set_transform_static(transform('sensor', 0., parent='moving'), 'test')
    guarded.set_transform_static(transform('static_sensor', 0.), 'test')
    for stamp in (1., 1.02):
        guarded.set_transform(transform('moving', stamp, parent='rig'), 'test')
    for stamp in (1., 2.):
        guarded.set_transform(transform('unrelated', stamp), 'test')
    guarded.lookup_transform('sensor', 'world', Time(seconds=1.01))  # Inverse path is guarded too.
    assert set(guarded.last_support) == {'moving'}
    guarded.lookup_transform('world', 'static_sensor', Time(seconds=50.))
    assert guarded.last_support == {}
    guarded.lookup_transform('same_frame', 'same_frame', Time(seconds=1.))
    with pytest.raises(TransformException, match='latest TF is forbidden'):
        guarded.lookup_transform('world', 'sensor', Time())
    with pytest.raises(TransformException, match='missing timestamp brackets'):
        guarded.lookup_transform('world', 'sensor', Time(seconds=1.03))


def test_clear_and_parent_change_cannot_reuse_old_sample_support():
    guarded = BracketedBuffer()
    guarded.set_transform_static(transform('rig', 0.), 'test')
    guarded.set_transform(transform('sensor', 10., parent='rig'), 'test')
    guarded.clear()
    assert guarded.dynamic_times == {}
    guarded.lookup_transform('world', 'rig', Time(seconds=2.))  # Static calibration survives clear.
    with pytest.raises(TransformException):
        guarded.lookup_transform('world', 'sensor', Time(seconds=10.))
    guarded.set_transform(transform('sensor', 2., parent='rig'), 'test')
    guarded.lookup_transform('world', 'sensor', Time(seconds=2.))
    guarded.set_transform(transform('sensor', 2.01, parent='world'), 'test')
    assert guarded.dynamic_times['sensor'] == [2010000000]
    with pytest.raises(TransformException):
        guarded.lookup_transform('world', 'sensor', Time(seconds=2.))


def test_ros_clock_reset_clears_support_and_listener_uses_deeper_queue():
    rclpy.init()
    node = Node('jol_tf_guard_clock_test')
    try:
        node.set_parameters([Parameter('use_sim_time', value=True)])
        buffer, listener = guarded_tf(node)
        assert listener.tf_sub.qos_profile.depth == 2000
        assert node.get_parameter('tf_max_interpolation_gap').value == .03
        node.get_clock().set_ros_time_override(Time(seconds=100.))
        buffer.set_transform(transform('sensor', 100.), 'test')
        assert buffer.dynamic_times
        node.get_clock().set_ros_time_override(Time(seconds=2.))
        assert buffer.dynamic_times == {}
        buffer.set_transform(transform('sensor', 100.), 'late_old_epoch')
        assert buffer.dynamic_times == {}
        assert buffer.future_rejections == 1
        with pytest.raises(TransformException):
            buffer.lookup_transform('world', 'sensor', Time(seconds=100.))
        buffer.set_transform(transform('sensor', 2.), 'new_epoch')
        buffer.lookup_transform('world', 'sensor', Time(seconds=2.))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_invalid_static_transform_does_not_corrupt_existing_parent_path():
    buffer = BracketedBuffer()
    buffer.set_transform_static(transform('sensor', 0.), 'valid')
    invalid = transform('sensor', 0., parent='disconnected')
    invalid.transform.rotation.w = 0.
    buffer.set_transform_static(invalid, 'invalid')
    assert buffer.parents['sensor'] == 'world'
    buffer.lookup_transform('world', 'sensor', Time(seconds=1.))
