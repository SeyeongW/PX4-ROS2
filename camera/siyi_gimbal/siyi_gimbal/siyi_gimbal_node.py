"""siyi_gimbal_node — point a SIYI A8 mini straight down the moment the vehicle arms.

One job, and it is a safety-adjacent one: the landing camera must be looking at
the ground before the vehicle leaves it. Doing that by hand is a step that gets
forgotten exactly once.

    /mavros/state.armed  false -> true   ==>   SET_ANGLE(yaw 0, pitch -90)

The command is then **re-asserted on a slow timer** rather than sent once. A
gimbal is a separate computer on the other end of a UDP link with no delivery
guarantee: a dropped datagram, a power blip, a mode change or someone bumping
the manual control all leave it pointing somewhere else, and a fire-and-forget
command cannot tell the difference between "obeyed" and "never arrived". If the
gimbal is reporting its attitude, the re-assert is skipped while it is already
where it should be, so a healthy link is quiet.

ALL PARAMETERS ARE DECLARED HERE, IN `_declare`, WITH THEIR VALUES.
The launch file passes none — same rule as `precland_hw`.

Interfaces
----------
Subscribes  /mavros/state                mavros_msgs/State
Publishes   ~/attitude                   geometry_msgs/Vector3Stamped  (r,p,y deg)
            ~/status                     std_msgs/String
Services    ~/look_down, ~/center        std_srvs/Trigger   (manual override)
Talks to    the gimbal over UDP, default 192.168.144.25:37260
"""

from __future__ import annotations

import socket

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import String
from std_srvs.srv import Trigger

from mavros_msgs.msg import State

from . import protocol as siyi
from . import siyi_commands as cmds


def _sensor_qos() -> QoSProfile:
    """MAVROS publishes BEST_EFFORT; a RELIABLE subscriber would get nothing."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class SiyiGimbalNode(Node):
    def __init__(self):
        super().__init__('siyi_gimbal_node')
        self._declare()
        g = self.get_parameter
        self.host = str(g('gimbal_host').value)
        self.port = int(g('gimbal_port').value)
        self.nadir_pitch = float(g('nadir_pitch_deg').value)
        self.nadir_yaw = float(g('nadir_yaw_deg').value)
        self.reassert_s = float(g('reassert_period_s').value)
        self.attitude_tol = float(g('attitude_tolerance_deg').value)
        self.poll_attitude = bool(g('poll_attitude').value)
        self.poll_s = float(g('attitude_poll_period_s').value)
        self.disarm_centers = bool(g('center_on_disarm').value)

        self._seq = 0
        self._armed: bool | None = None
        self._want_nadir = False
        self._attitude: tuple[float, float, float] | None = None
        self._sent = 0
        self._rx_bad = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

        self.att_pub = self.create_publisher(Vector3Stamped, '~/attitude', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.create_subscription(State, '/mavros/state', self._on_state,
                                 _sensor_qos())
        self.create_service(Trigger, '~/look_down', self._on_look_down)
        self.create_service(Trigger, '~/center', self._on_center)

        self.create_timer(self.reassert_s, self._reassert)
        self.create_timer(0.05, self._drain_socket)
        if self.poll_attitude:
            self.create_timer(self.poll_s, self._poll)
        self.create_timer(2.0, self._publish_status)

        ok, why = siyi.clamped(self.nadir_yaw, self.nadir_pitch)
        if not ok:
            self.get_logger().warn(f'requested nadir is outside the A8 mini '
                                   f'travel and will be clamped: {why}')
        self.get_logger().info(
            f'siyi_gimbal_node: {self.host}:{self.port} — will look down '
            f'(yaw {self.nadir_yaw:.0f}, pitch {self.nadir_pitch:.0f} deg) '
            f'the moment /mavros/state reports ARMED')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these may be set. The launch file passes none."""
        p = self.declare_parameter
        # SIYI's factory address. The A8 mini is a UDP peer on the vehicle
        # network, not a MAVLink mount, so this is its IP and not a MAVROS topic.
        p('gimbal_host', '192.168.144.25')
        p('gimbal_port', 37260)
        # Straight down, level in yaw. Pitch is negative-down on this gimbal.
        p('nadir_pitch_deg', -90.0)
        p('nadir_yaw_deg', 0.0)
        # How often to re-send while armed. Slow on purpose: this is a
        # correction for lost datagrams, not a control loop.
        p('reassert_period_s', 2.0)
        # Close enough to count as already pointing down, so a healthy gimbal
        # is not commanded needlessly.
        p('attitude_tolerance_deg', 3.0)
        # Attitude feedback. Without it the node still works, it just cannot
        # tell whether the command landed, so it re-sends unconditionally.
        p('poll_attitude', True)
        p('attitude_poll_period_s', 0.5)
        # Recentre when the vehicle disarms, so the camera is not left staring
        # at the ground on the bench.
        p('center_on_disarm', True)

    # ---------------------------------------------------------------- helpers
    def _send(self, packet: bytes, what: str) -> None:
        try:
            self.sock.sendto(packet, (self.host, self.port))
            self._sent += 1
        except OSError as exc:
            # Log, do not raise: the gimbal being unreachable must not take the
            # node down, or it cannot recover when the link comes back.
            self.get_logger().warn(f'{what}: send failed ({exc})',
                                   throttle_duration_sec=5.0)

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) % 0xFFFF
        return self._seq

    def _command_nadir(self, why: str) -> None:
        self._send(siyi.set_angle(self.nadir_yaw, self.nadir_pitch,
                                  self._next_seq()), 'look_down')
        self.get_logger().info(
            f'-> gimbal: look down (yaw {self.nadir_yaw:.0f}, '
            f'pitch {self.nadir_pitch:.0f} deg) [{why}]')

    def _at_nadir(self) -> bool:
        if self._attitude is None:
            return False
        _roll, pitch, yaw = self._attitude
        return (abs(pitch - self.nadir_pitch) <= self.attitude_tol
                and abs(yaw - self.nadir_yaw) <= self.attitude_tol)

    # -------------------------------------------------------------- callbacks
    def _on_state(self, msg: State) -> None:
        armed = bool(msg.armed)
        if armed == self._armed:
            return
        first = self._armed is None
        self._armed = armed
        if armed:
            self._want_nadir = True
            self._command_nadir('vehicle ARMED' if not first
                                else 'vehicle already ARMED at startup')
        else:
            self._want_nadir = False
            if not first and self.disarm_centers:
                self._send(siyi.encode(siyi.CENTER, bytes([cmds.TRIGGER]), self._next_seq()),
                           'center')
                self.get_logger().info('-> gimbal: centre [vehicle DISARMED]')

    def _reassert(self) -> None:
        """Re-send while armed, unless feedback says it is already there."""
        if not self._want_nadir:
            return
        if self.poll_attitude and self._at_nadir():
            return
        self._command_nadir('re-assert')

    def _poll(self) -> None:
        self._send(siyi.request_attitude(self._next_seq()), 'attitude request')

    def _drain_socket(self) -> None:
        while True:
            try:
                data, _addr = self.sock.recvfrom(1024)
            except (BlockingIOError, OSError):
                return
            parsed = siyi.decode(data)
            if parsed is None:
                self._rx_bad += 1
                continue
            cmd, payload = parsed
            if cmd != siyi.ACQUIRE_GIMBAL_ATTITUDE:
                continue
            att = siyi.parse_attitude(payload)
            if att is None:
                continue
            self._attitude = att
            m = Vector3Stamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'gimbal'
            m.vector.x, m.vector.y, m.vector.z = att      # roll, pitch, yaw (deg)
            self.att_pub.publish(m)

    # ---------------------------------------------------------------- services
    def _on_look_down(self, _req, res):
        self._command_nadir('manual service call')
        res.success, res.message = True, 'commanded nadir'
        return res

    def _on_center(self, _req, res):
        self._want_nadir = False
        self._send(siyi.encode(siyi.CENTER, bytes([cmds.TRIGGER]), self._next_seq()), 'center')
        res.success = True
        res.message = 'commanded centre; nadir hold released until next ARM'
        self.get_logger().info('-> gimbal: centre [manual service call]')
        return res

    # ------------------------------------------------------------------ status
    def _publish_status(self) -> None:
        armed = 'unknown' if self._armed is None else ('ARMED' if self._armed
                                                       else 'disarmed')
        if self._attitude is None:
            att = 'no feedback'
        else:
            r, pi, y = self._attitude
            att = f'r{r:+.1f} p{pi:+.1f} y{y:+.1f}'
        holding = 'holding nadir' if self._want_nadir else 'idle'
        self.status_pub.publish(String(
            data=f'{holding} | vehicle {armed} | gimbal {att} | '
                 f'sent {self._sent} | bad_rx {self._rx_bad}'))
        if self._want_nadir and self._attitude is None:
            self.get_logger().warn(
                f'no attitude feedback from {self.host}:{self.port} — commands '
                f'are being sent blind; check the gimbal IP and that the '
                f'vehicle network is up', throttle_duration_sec=10.0)


def main(args=None):
    rclpy.init(args=args)
    node = SiyiGimbalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
