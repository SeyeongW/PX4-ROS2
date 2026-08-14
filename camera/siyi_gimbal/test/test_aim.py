"""Aiming the gimbal somewhere other than nadir, end to end over a real socket.

The mission node sweeps the camera to look for a marker and then keeps it on
the marker while flying there (`mpc_landing_node`'s SEARCH). That only works if
an aim SURVIVES: this node's whole reason for re-sending is that a gimbal is a
separate computer on a lossy link, and a re-assert that dragged the camera back
to nadir every two seconds would be worse than not re-asserting at all.

Same rig as `test_arm_triggers_nadir.py` — a fake gimbal that decodes what the
node actually puts on the wire, so these cover the framing too.
"""

import math
import time

import pytest
import rclpy
from geometry_msgs.msg import Vector3Stamped
from mavros_msgs.msg import State
from rclpy.parameter import Parameter

from siyi_gimbal.siyi_gimbal_node import SiyiGimbalNode

from test_arm_triggers_nadir import PORT, FakeGimbal, _settle


@pytest.fixture
def rig():
    rclpy.init()
    gimbal = FakeGimbal()
    node = SiyiGimbalNode(parameter_overrides=[
        Parameter('transport', value='udp'),
        Parameter('gimbal_host', value='127.0.0.1'),
        Parameter('gimbal_port', value=PORT),
        Parameter('nadir_on_start', value=False),
        # Fast enough that a test does not sit through two real seconds to see
        # a re-assert, and no attitude feedback, so the re-assert is
        # unconditional — which is the case that matters here.
        Parameter('reassert_period_s', value=0.15),
        Parameter('poll_attitude', value=False),
    ])
    yield node, gimbal
    node.destroy_node()
    rclpy.shutdown()
    gimbal.close()


def _aim(pitch, yaw):
    m = Vector3Stamped()
    m.vector.y, m.vector.z = float(pitch), float(yaw)
    return m


def _spin(node, seconds):
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.02)


def test_an_aim_is_commanded_as_sent(rig):
    node, gimbal = rig
    node._on_aim(_aim(-45.0, 60.0))
    angles = _settle(gimbal, 1)
    assert angles[0] == pytest.approx((60.0, -45.0))
    assert gimbal.bad_frames == 0


def test_the_reassert_holds_the_aim_not_nadir(rig):
    """The bug this exists to prevent: the camera yanked out of its search sector."""
    node, gimbal = rig
    node._on_aim(_aim(-40.0, -90.0))
    _settle(gimbal, 1)
    _spin(node, 0.6)
    assert len(gimbal.angles()) > 1, 'nothing was re-asserted'
    assert all(a == pytest.approx((-90.0, -40.0)) for a in gimbal.angles()), \
        f'the re-assert did not hold the aim: {gimbal.angles()}'


def test_a_repeated_aim_does_not_re_send(rig):
    """A mission tracking a marker republishes every tick; the link is 115200 baud."""
    node, gimbal = rig
    node._on_aim(_aim(-40.0, 30.0))
    _settle(gimbal, 1)
    before = len(gimbal.angles())
    for _ in range(20):
        node._on_aim(_aim(-40.1, 30.2))       # inside aim_deadband_deg
    time.sleep(0.05)
    assert len(gimbal.angles()) == before, 'a dead-band aim went on the wire'


def test_a_moved_aim_does_re_send(rig):
    node, gimbal = rig
    node._on_aim(_aim(-40.0, 30.0))
    _settle(gimbal, 1)
    node._on_aim(_aim(-40.0, 75.0))
    angles = _settle(gimbal, 2)
    assert angles[-1] == pytest.approx((75.0, -40.0))


def test_nan_releases_the_aim_back_to_nadir(rig):
    node, gimbal = rig
    node._on_aim(_aim(-40.0, 30.0))
    _settle(gimbal, 1)
    node._on_aim(_aim(math.nan, math.nan))
    angles = _settle(gimbal, 2)
    assert angles[-1] == pytest.approx((0.0, -90.0))
    assert node._aim is None


def test_arming_takes_the_gimbal_back_from_a_search(rig):
    """A flight starts at nadir whatever the last mission left the camera doing."""
    node, gimbal = rig
    node._on_aim(_aim(-35.0, -120.0))
    _settle(gimbal, 1)
    node._on_state(State(connected=True, armed=False))
    node._on_state(State(connected=True, armed=True))
    angles = _settle(gimbal, 2)
    assert angles[-1] == pytest.approx((0.0, -90.0))
    assert node._aim is None


def test_an_out_of_travel_aim_is_still_sent_clamped(rig):
    """The protocol owns the limits; refusing here would silently stop a sweep."""
    node, gimbal = rig
    node._on_aim(_aim(-40.0, 200.0))
    angles = _settle(gimbal, 1)
    assert angles[0][0] == pytest.approx(135.0)
