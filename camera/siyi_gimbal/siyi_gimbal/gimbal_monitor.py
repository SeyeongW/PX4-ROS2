"""gimbal_monitor — live gimbal angles, and what frame they are reported in.

Run it, tilt the airframe by hand, and read the answer off the screen.

WHY THIS EXISTS
---------------
The gimbal reports an attitude (SIYI command 0x0D). Everything downstream — the
TF chain that turns a marker pixel into a world position — depends on knowing
WHICH FRAME that attitude is in, and the vendor does not say:

  EARTH-referenced   the gimbal is IMU-stabilised and reports where it is
                     pointing in the world. Tilt the airframe and the reported
                     pitch DOES NOT MOVE. Use it directly; composing it with the
                     vehicle attitude would apply the vehicle's rotation twice.

  BODY-referenced    it reports joint angles relative to its mount. Tilt the
                     airframe and the joint counter-rotates to hold the view, so
                     the reported pitch moves by MINUS the tilt — which means
                     (gimbal + vehicle) stays constant. This is what the SITL
                     stack assumes, and it must be composed with the vehicle
                     attitude to mean anything in the world.

Getting this backwards does not error. It produces a marker position that is
wrong by exactly the airframe's tilt, which looks like a mysterious tracking
bias and is very hard to find later. So it is measured here, once, on the bench.

READING IT
----------
Tilt the vehicle nose-up/nose-down through 15-20 deg and watch the two right
columns. Exactly one of them should stay still:

    gimbal-only steady  -> EARTH frame
    sum steady          -> BODY frame

Vehicle attitude comes from /mavros/imu/data when MAVROS is running. Without it
the gimbal columns still work, but the frame question cannot be answered — you
need something to tilt against.

    ros2 run siyi_gimbal gimbal_monitor
    ros2 run siyi_gimbal gimbal_monitor --ros-args -p transport:=udp
"""

from __future__ import annotations

import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import Imu

from . import protocol as siyi
from . import transport as tp


def _sensor_qos() -> QoSProfile:
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


def _rpy_deg(q) -> tuple[float, float, float]:
    """Quaternion -> roll, pitch, yaw in degrees."""
    w, x, y, z = q.w, q.x, q.y, q.z
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class GimbalMonitor(Node):
    def __init__(self):
        super().__init__('gimbal_monitor')
        p = self.declare_parameter
        p('transport', 'serial')
        p('serial_device', '/dev/ttyTHS1')
        p('serial_baud', 115200)
        p('gimbal_host', '192.168.144.25')
        p('gimbal_port', 37260)
        p('poll_hz', 10.0)

        g = self.get_parameter
        try:
            self.link = tp.make(str(g('transport').value),
                                host=str(g('gimbal_host').value),
                                port=int(g('gimbal_port').value),
                                device=str(g('serial_device').value),
                                baud=int(g('serial_baud').value))
        except (RuntimeError, ValueError) as exc:
            self.get_logger().fatal(str(exc))
            raise SystemExit(1) from None

        self.gimbal: tuple[float, float, float] | None = None
        self.vehicle: tuple[float, float, float] | None = None
        self._seq = 0
        self._rx = 0
        self._t0 = time.time()
        # Ranges, so a tilt shows up as a SPREAD rather than something you have
        # to eyeball frame by frame. This is the actual measurement.
        self._span = {'gimbal_pitch': [None, None], 'sum_pitch': [None, None]}

        self.create_subscription(Imu, '/mavros/imu/data', self._on_imu,
                                 _sensor_qos())
        # Say it EARLY. Telling someone their vehicle attitude was missing only
        # in the verdict, after they have already tilted the airframe through a
        # full test, wastes the one thing this tool is asking them to do.
        self.create_timer(3.0, self._nag)
        hz = float(g('poll_hz').value)
        self.create_timer(1.0 / hz, self._poll)
        self.create_timer(0.05, self._drain)
        self.create_timer(0.2, self._draw)

        print(f'\n  gimbal_monitor — {self.link.description}\n'
              f'  tilt the airframe and watch which column stays still\n')
        print(f'  {"gimbal r/p/y":>26} | {"vehicle r/p/y":>26} | '
              f'{"g.pitch":>8} {"g+v pitch":>10}')
        print('  ' + '-' * 78)

    def _on_imu(self, msg: Imu) -> None:
        self.vehicle = _rpy_deg(msg.orientation)

    def _nag(self) -> None:
        """Keep saying it until the vehicle attitude shows up."""
        if self.vehicle is not None:
            return
        print('\n  !! NO VEHICLE ATTITUDE on /mavros/imu/data — do not bother '
              'tilting yet.\n'
              '     The frame cannot be decided without it. Check MAVROS is up '
              'and in this\n'
              '     ROS_DOMAIN_ID:  ros2 topic hz /mavros/imu/data\n')

    def _poll(self) -> None:
        self._seq = (self._seq + 1) % 0xFFFF
        try:
            self.link.send(siyi.request_attitude(self._seq))
        except OSError:
            pass

    def _drain(self) -> None:
        try:
            packets = self.link.read()
        except OSError:
            return
        for data in packets:
            parsed = siyi.decode(data)
            if parsed is None or parsed[0] != siyi.ACQUIRE_GIMBAL_ATTITUDE:
                continue
            att = siyi.parse_attitude(parsed[1])
            if att is not None:
                self.gimbal = att
                self._rx += 1

    def _track(self, key: str, value: float) -> None:
        lo, hi = self._span[key]
        self._span[key] = [value if lo is None else min(lo, value),
                           value if hi is None else max(hi, value)]

    def _draw(self) -> None:
        if self.gimbal is None:
            sys.stdout.write(
                f'\r  waiting for the gimbal…  polled {self._seq}, '
                f'got {self._rx}   ')
            sys.stdout.flush()
            return
        gr, gp, gy = self.gimbal
        gtxt = f'{gr:+7.1f} {gp:+7.1f} {gy:+7.1f}'
        self._track('gimbal_pitch', gp)

        if self.vehicle is None:
            vtxt = f'{"(no /mavros/imu/data)":>26}'
            sumtxt = f'{"—":>10}'
        else:
            vr, vp, vy = self.vehicle
            vtxt = f'{vr:+7.1f} {vp:+7.1f} {vy:+7.1f}'
            total = gp + vp
            self._track('sum_pitch', total)
            sumtxt = f'{total:+10.1f}'

        sys.stdout.write(f'\r  {gtxt:>26} | {vtxt:>26} | {gp:+8.1f} {sumtxt}  ')
        sys.stdout.flush()

    def verdict(self) -> None:
        """Print the ranges, and read the frame off them."""
        g_lo, g_hi = self._span['gimbal_pitch']
        s_lo, s_hi = self._span['sum_pitch']
        print('\n\n  ' + '=' * 78)
        if g_lo is None:
            print('  no gimbal attitude was received — nothing to conclude')
            print('  ' + '=' * 78)
            return
        g_span = g_hi - g_lo
        print(f'  gimbal pitch ranged over  {g_span:6.1f} deg  '
              f'({g_lo:+.1f} .. {g_hi:+.1f})')
        if s_lo is None:
            print('  vehicle attitude never arrived, so the frame cannot be '
                  'decided — start MAVROS and tilt the airframe.')
            print('  ' + '=' * 78)
            return
        s_span = s_hi - s_lo
        print(f'  (gimbal+vehicle) ranged over {s_span:6.1f} deg  '
              f'({s_lo:+.1f} .. {s_hi:+.1f})')
        print()
        if max(g_span, s_span) < 3.0:
            print('  INCONCLUSIVE — nothing moved enough. Tilt the airframe '
                  'through 15-20 deg while this runs.')
        elif g_span < s_span * 0.5:
            print('  => EARTH-referenced. The gimbal holds its world angle, so '
                  'use its attitude DIRECTLY;')
            print('     composing it with the vehicle attitude would apply the '
                  'airframe rotation twice.')
        elif s_span < g_span * 0.5:
            print('  => BODY-referenced (joint angles). COMPOSE it with the '
                  'vehicle attitude,')
            print('     exactly as the SITL chain does in frame.gimbal_tf_chain.')
        else:
            print('  AMBIGUOUS — both moved. Tilt in pitch only, keep it slow, '
                  'and try again.')
        print('  ' + '=' * 78)


def main(args=None):
    rclpy.init(args=args)
    node = GimbalMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.verdict()
        node.link.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
