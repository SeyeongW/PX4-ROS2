#!/usr/bin/env python3
"""Turn Gazebo trailer odometry into the field MAVLink GPS stream."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import pty
import random
import signal
import threading
import time
import tty
from collections import deque

from gz.math7 import Angle, SphericalCoordinates, Vector3d
from gz.msgs10.odometry_pb2 import Odometry
from gz.transport13 import Node
from pymavlink import mavutil


def _yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _clamp_i16(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


def mavlink_sample(spherical: SphericalCoordinates, message: Odometry,
                   antenna_z_m: float, position_noise_m: tuple[float, float],
                   velocity_noise_m_s: tuple[float, float]) -> dict[str, int]:
    """Encode world ENU pose and body velocity as MAVLink GPS fields."""
    position = message.pose.position
    orientation = message.pose.orientation
    linear = message.twist.linear
    east = float(position.x) + position_noise_m[0]
    north = float(position.y) + position_noise_m[1]
    up = float(position.z) + antenna_z_m
    spherical_position = spherical.position_transform(
        Vector3d(east, north, up),
        SphericalCoordinates.LOCAL2,
        SphericalCoordinates.SPHERICAL,
    )

    yaw = _yaw(orientation.x, orientation.y,
               orientation.z, orientation.w)
    velocity_east = (math.cos(yaw) * float(linear.x)
                     - math.sin(yaw) * float(linear.y)
                     + velocity_noise_m_s[0])
    velocity_north = (math.sin(yaw) * float(linear.x)
                      + math.cos(yaw) * float(linear.y)
                      + velocity_noise_m_s[1])
    speed = math.hypot(velocity_east, velocity_north)
    course = (math.degrees(math.atan2(velocity_east, velocity_north))
              % 360.0 if speed >= 0.01 else None)
    return {
        'lat': int(round(math.degrees(spherical_position.x()) * 1.0e7)),
        'lon': int(round(math.degrees(spherical_position.y()) * 1.0e7)),
        'alt': int(round(spherical_position.z() * 1000.0)),
        'relative_alt': int(round(up * 1000.0)),
        # GLOBAL_POSITION_INT velocity is NED in cm/s.
        'vx': _clamp_i16(velocity_north * 100.0),
        'vy': _clamp_i16(velocity_east * 100.0),
        'vz': 0,
        'speed': max(0, min(65535, int(round(speed * 100.0)))),
        'course': (65535 if course is None
                   else max(0, min(35999, int(round(course * 100.0))))),
    }


class Emulator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop = threading.Event()
        self.rng = random.Random(args.seed)
        self.queue = deque()
        self.next_position_t: float | None = None
        self.next_status_t: float | None = None
        self.last_sim_t: float | None = None
        self.position_count = 0
        self.status_count = 0

        self.spherical = SphericalCoordinates(
            SphericalCoordinates.EARTH_WGS84,
            Angle(math.radians(args.latitude_deg)),
            Angle(math.radians(args.longitude_deg)),
            args.elevation_m,
            Angle(math.radians(args.heading_deg)),
        )
        self.master_fd, self.slave_fd = pty.openpty()
        tty.setraw(self.slave_fd)
        self.slave_path = os.ttyname(self.slave_fd)
        self.pty_link = pathlib.Path(args.pty_link)
        self.pty_link.parent.mkdir(parents=True, exist_ok=True)
        self.pty_link.unlink(missing_ok=True)
        self.pty_link.symlink_to(self.slave_path)
        self.stream = os.fdopen(self.master_fd, 'wb', buffering=0)
        self.mavlink = mavutil.mavlink.MAVLink(
            self.stream, srcSystem=args.sysid, srcComponent=1)

    @staticmethod
    def _sim_time(message: Odometry) -> float:
        stamp = message.header.stamp
        return float(stamp.sec) + float(stamp.nsec) * 1.0e-9

    def _reset_schedule(self, now: float) -> None:
        self.queue.clear()
        self.next_status_t = now
        self.next_position_t = now

    def _queue_packets(self, message: Odometry, now: float) -> None:
        sample = mavlink_sample(
            self.spherical,
            message,
            self.args.antenna_z_m,
            (self.rng.gauss(0.0, self.args.position_noise_std_m),
             self.rng.gauss(0.0, self.args.position_noise_std_m)),
            (self.rng.gauss(0.0, self.args.velocity_noise_std_m_s),
             self.rng.gauss(0.0, self.args.velocity_noise_std_m_s)),
        )
        if now + 1.0e-9 >= self.next_status_t:
            self.queue.append((now + self.args.delay_s, 'status', now, sample))
            self.next_status_t = now + 1.0 / self.args.status_rate_hz
        if now + 1.0e-9 >= self.next_position_t:
            if self.rng.random() >= self.args.dropout_probability:
                self.queue.append(
                    (now + self.args.delay_s, 'position', now, sample))
            # Do not burst old packets after a pause or dropout.
            self.next_position_t = now + 1.0 / self.args.position_rate_hz

    def _send_due(self, now: float) -> None:
        while self.queue and self.queue[0][0] <= now + 1.0e-9:
            _, kind, source_t, sample = self.queue.popleft()
            boot_ms = int(round(source_t * 1000.0)) & 0xFFFFFFFF
            if kind == 'status':
                self.mavlink.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GROUND_ROVER,
                    mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                    0, 0, mavutil.mavlink.MAV_STATE_ACTIVE)
                self.mavlink.gps_raw_int_send(
                    int(round(source_t * 1.0e6)), self.args.fix_type,
                    sample['lat'], sample['lon'], sample['alt'],
                    self.args.eph_cm, self.args.epv_cm,
                    sample['speed'], sample['course'], self.args.satellites)
                self.status_count += 1
            else:
                self.mavlink.global_position_int_send(
                    boot_ms, sample['lat'], sample['lon'], sample['alt'],
                    sample['relative_alt'], sample['vx'], sample['vy'],
                    sample['vz'], sample['course'])
                self.position_count += 1

    def on_odometry(self, message: Odometry) -> None:
        if self.stop.is_set():
            return
        now = self._sim_time(message)
        if not math.isfinite(now) or now <= 0.0:
            return
        if self.last_sim_t is None or now < self.last_sim_t:
            self._reset_schedule(now)
        self.last_sim_t = now
        self._queue_packets(message, now)
        self._send_due(now)

    def run(self) -> int:
        print(f'PTY {self.pty_link} -> {self.slave_path}', flush=True)
        time.sleep(self.args.reader_warmup_s)
        node = Node()
        if not node.subscribe(Odometry, self.args.odometry_topic,
                              self.on_odometry):
            raise RuntimeError(
                f'cannot subscribe to {self.args.odometry_topic}')
        while not self.stop.wait(2.0):
            print(f'MAVLink GPS: {self.position_count / 2.0:.1f} Hz GPI, '
                  f'{self.status_count / 2.0:.1f} Hz GPS_RAW', flush=True)
            self.position_count = self.status_count = 0
        return 0

    def close(self) -> None:
        self.stop.set()
        self.pty_link.unlink(missing_ok=True)
        try:
            self.stream.close()
        finally:
            os.close(self.slave_fd)


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--odometry-topic', required=True)
    parser.add_argument('--pty-link', required=True)
    parser.add_argument('--latitude-deg', required=True, type=float)
    parser.add_argument('--longitude-deg', required=True, type=float)
    parser.add_argument('--elevation-m', required=True, type=float)
    parser.add_argument('--heading-deg', default=0.0, type=float)
    parser.add_argument('--antenna-z-m', required=True, type=float)
    parser.add_argument('--sysid', default=1, type=int)
    parser.add_argument('--position-rate-hz', default=5.0, type=float)
    parser.add_argument('--status-rate-hz', default=1.0, type=float)
    parser.add_argument('--position-noise-std-m', default=0.0, type=float)
    parser.add_argument('--velocity-noise-std-m-s', default=0.0, type=float)
    parser.add_argument('--delay-s', default=0.0, type=float)
    parser.add_argument('--dropout-probability', default=0.0, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--fix-type', default=6, type=int)
    parser.add_argument('--satellites', default=18, type=int)
    parser.add_argument('--eph-cm', default=5, type=int)
    parser.add_argument('--epv-cm', default=10, type=int)
    parser.add_argument('--reader-warmup-s', default=0.5, type=float)
    args = parser.parse_args(argv)
    finite = (args.latitude_deg, args.longitude_deg, args.elevation_m,
              args.heading_deg, args.antenna_z_m, args.position_rate_hz,
              args.status_rate_hz, args.position_noise_std_m,
              args.velocity_noise_std_m_s, args.delay_s,
              args.dropout_probability, args.reader_warmup_s)
    if not all(math.isfinite(value) for value in finite):
        parser.error('numeric parameters must be finite')
    if not (-90.0 <= args.latitude_deg <= 90.0
            and -180.0 <= args.longitude_deg <= 180.0):
        parser.error('invalid spherical origin')
    if (args.position_rate_hz <= 0.0 or args.status_rate_hz <= 0.0
            or min(args.position_noise_std_m,
                   args.velocity_noise_std_m_s,
                   args.delay_s, args.reader_warmup_s) < 0.0
            or not 0.0 <= args.dropout_probability <= 1.0
            or not 1 <= args.sysid <= 254
            or not 3 <= args.fix_type <= 6
            or not 0 <= args.satellites <= 255):
        parser.error('invalid rate, noise, delay, dropout, fix, or sysid')
    return args


def main(argv=None) -> int:
    emulator = Emulator(_arguments(argv))
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _s, _f: emulator.stop.set())
    try:
        return emulator.run()
    finally:
        emulator.close()


if __name__ == '__main__':
    raise SystemExit(main())
