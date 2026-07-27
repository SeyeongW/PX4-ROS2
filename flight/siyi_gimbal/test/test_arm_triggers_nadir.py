"""The headline behaviour, end to end over a real socket: arm => look down.

A fake gimbal binds the SIYI port on loopback and decodes whatever the node
actually puts on the wire, so this covers the parts the protocol unit tests
cannot: that the node reacts to the ARMED edge, that it addresses and frames the
datagram correctly, and that it does not fire before arming.
"""

import socket
import struct
import threading
import time

import pytest
import rclpy
from mavros_msgs.msg import State

from siyi_gimbal import protocol as p
from siyi_gimbal.siyi_gimbal_node import SiyiGimbalNode

PORT = 37799          # not the real 37260, so a gimbal on the bench is untouched


class FakeGimbal:
    """Binds the UDP port and records decoded commands."""

    def __init__(self, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port))
        self.sock.settimeout(0.2)
        self.commands = []          # (cmd_id, payload)
        self.bad_frames = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _addr = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                return
            parsed = p.decode(data)
            if parsed is None:
                self.bad_frames += 1
                continue
            self.commands.append(parsed)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.sock.close()

    def angles(self):
        """Every SET_ANGLE received, decoded back to (yaw, pitch) degrees."""
        out = []
        for cmd, payload in self.commands:
            if cmd == p.SET_ANGLE:
                yaw_raw, pitch_raw = struct.unpack('<hh', payload)
                out.append((-yaw_raw / 10.0, pitch_raw / 10.0))
        return out


@pytest.fixture
def rig():
    rclpy.init()
    gimbal = FakeGimbal()
    node = SiyiGimbalNode()
    # Point it at the fake rather than the factory address. Done after
    # construction so the node's own declared defaults stay the thing under test.
    node.host, node.port = '127.0.0.1', PORT
    yield node, gimbal
    node.destroy_node()
    rclpy.shutdown()
    gimbal.close()


def _settle(gimbal, want, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end and len(gimbal.angles()) < want:
        time.sleep(0.02)
    return gimbal.angles()


def test_nothing_is_commanded_before_arming(rig):
    node, gimbal = rig
    for _ in range(5):
        node._on_state(State(connected=True, armed=False))
        time.sleep(0.02)
    time.sleep(0.3)
    assert gimbal.angles() == [], 'gimbal must not be moved while disarmed'


def test_arming_points_the_camera_straight_down(rig):
    node, gimbal = rig
    node._on_state(State(connected=True, armed=False))
    time.sleep(0.1)
    node._on_state(State(connected=True, armed=True))    # the edge under test

    angles = _settle(gimbal, 1)
    assert angles, 'arming produced no SET_ANGLE'
    yaw, pitch = angles[0]
    assert pitch == pytest.approx(-90.0), f'expected nadir, got pitch {pitch}'
    assert yaw == pytest.approx(0.0)
    assert gimbal.bad_frames == 0, 'the gimbal could not decode what we sent'


def test_repeated_armed_messages_do_not_re_command(rig):
    """Only the EDGE matters; /mavros/state repeats at 1 Hz forever."""
    node, gimbal = rig
    node._on_state(State(connected=True, armed=False))
    node._on_state(State(connected=True, armed=True))
    _settle(gimbal, 1)
    for _ in range(10):
        node._on_state(State(connected=True, armed=True))
    time.sleep(0.3)
    assert len(gimbal.angles()) == 1, 'a repeat of ARMED must not re-command'


def test_disarming_centres_the_gimbal(rig):
    node, gimbal = rig
    node._on_state(State(connected=True, armed=False))
    node._on_state(State(connected=True, armed=True))
    _settle(gimbal, 1)
    node._on_state(State(connected=True, armed=False))
    end = time.time() + 2.0
    while time.time() < end:
        if any(c == p.CENTER for c, _ in gimbal.commands):
            break
        time.sleep(0.02)
    assert any(c == p.CENTER for c, _ in gimbal.commands), \
        'disarming should recentre so the camera is not left staring down'
    assert node._want_nadir is False
