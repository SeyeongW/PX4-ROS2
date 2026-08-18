import importlib.util
import math
import random
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
from gz.math7 import Angle, SphericalCoordinates, Vector3d
from gz.msgs10.odometry_pb2 import Odometry


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'simulation/gazebo/tools/trailer_mavlink_emulator.py'
SPEC = importlib.util.spec_from_file_location('trailer_mavlink_emulator', SCRIPT)
EMULATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EMULATOR)


def _spherical():
    return SphericalCoordinates(
        SphericalCoordinates.EARTH_WGS84,
        Angle(math.radians(36.6533752165)),
        Angle(math.radians(127.4955858675)),
        74.0,
        Angle(0.0),
    )


def _odometry(timestamp=1.0):
    message = Odometry()
    message.header.stamp.sec = int(timestamp)
    message.header.stamp.nsec = int(round(timestamp % 1.0 * 1.0e9))
    message.pose.position.x = -5.669916246818
    message.pose.position.y = 65.542579972196
    message.pose.orientation.w = 1.0
    message.twist.linear.x = 3.2
    message.twist.linear.y = -1.5
    return message


def test_gazebo_enu_packs_wgs84_and_mavlink_ned_velocity():
    spherical = _spherical()
    packed = EMULATOR.mavlink_sample(
        spherical, _odometry(), 2.051, (0.0, 0.0), (0.0, 0.0))
    recovered = spherical.position_transform(
        Vector3d(math.radians(packed['lat'] * 1.0e-7),
                 math.radians(packed['lon'] * 1.0e-7),
                 packed['alt'] * 1.0e-3),
        SphericalCoordinates.SPHERICAL,
        SphericalCoordinates.LOCAL2,
    )
    assert recovered.x() == pytest.approx(-5.669916246818, abs=0.015)
    assert recovered.y() == pytest.approx(65.542579972196, abs=0.015)
    assert packed['vx'] == -150  # north
    assert packed['vy'] == 320   # east
    assert packed['vz'] == 0


class _MavlinkSink:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args))


def test_emulator_queues_one_hz_status_and_five_hz_delayed_position():
    emulator = object.__new__(EMULATOR.Emulator)
    emulator.args = SimpleNamespace(
        antenna_z_m=2.051,
        position_noise_std_m=0.0,
        velocity_noise_std_m_s=0.0,
        delay_s=0.25,
        dropout_probability=0.0,
        status_rate_hz=1.0,
        position_rate_hz=5.0,
        fix_type=6,
        eph_cm=5,
        epv_cm=10,
        satellites=18,
    )
    emulator.spherical = _spherical()
    emulator.rng = random.Random(1)
    emulator.queue = deque()
    emulator.next_status_t = emulator.next_position_t = 1.0
    emulator.mavlink = _MavlinkSink()
    emulator.position_count = emulator.status_count = 0

    emulator._queue_packets(_odometry(), 1.0)
    assert [entry[1] for entry in emulator.queue] == ['status', 'position']
    emulator._send_due(1.249)
    assert not emulator.mavlink.calls
    emulator._send_due(1.25)
    assert [call[0] for call in emulator.mavlink.calls] == [
        'heartbeat_send', 'gps_raw_int_send', 'global_position_int_send']
    assert emulator.status_count == emulator.position_count == 1

    emulator._queue_packets(_odometry(1.2), 1.2)
    assert [entry[1] for entry in emulator.queue] == ['position']
