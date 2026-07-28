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

TWO WAYS TO REACH THE GIMBAL
----------------------------
The A8 mini can be wired either way and speaks the same protocol on both:

    transport:=serial   /dev/ttyTHS1 @115200 — the Jetson's header UART
    transport:=udp      192.168.144.25:37260 — the vehicle ethernet

Only the pipe differs; see `transport.py`. Default is serial, because that is
how the aircraft is actually wired.

Interfaces
----------
Subscribes  /mavros/state                mavros_msgs/State
Publishes   ~/attitude                   geometry_msgs/Vector3Stamped  (r,p,y deg)
            ~/status                     std_msgs/String
Services    ~/look_down, ~/center        std_srvs/Trigger   (manual override)
Talks to    the gimbal over serial or UDP — see `transport`
"""

from __future__ import annotations

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
from . import transport as tp


def _sensor_qos() -> QoSProfile:
    """MAVROS publishes BEST_EFFORT; a RELIABLE subscriber would get nothing."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class SiyiGimbalNode(Node):
    def __init__(self, **kwargs):
        # **kwargs so a caller can inject parameter_overrides — the tests use it
        # to select the UDP transport, since a desktop has no gimbal UART.
        super().__init__('siyi_gimbal_node', **kwargs)
        self._declare()
        g = self.get_parameter
        self.transport_kind = str(g('transport').value)
        self.host = str(g('gimbal_host').value)
        self.port = int(g('gimbal_port').value)
        self.device = str(g('serial_device').value)
        self.baud = int(g('serial_baud').value)
        self.nadir_pitch = float(g('nadir_pitch_deg').value)
        self.nadir_yaw = float(g('nadir_yaw_deg').value)
        self.reassert_s = float(g('reassert_period_s').value)
        self.attitude_tol = float(g('attitude_tolerance_deg').value)
        self.poll_attitude = bool(g('poll_attitude').value)
        self.poll_s = float(g('attitude_poll_period_s').value)
        self.nadir_on_start = bool(g('nadir_on_start').value)
        self.disarm_centers = bool(g('center_on_disarm').value)

        self._seq = 0
        self._armed: bool | None = None
        self._want_nadir = False
        self._attitude: tuple[float, float, float] | None = None
        self._sent = 0
        self._rx_bad = 0

        # A bad port or a missing permission is an OPERATOR problem, so it gets
        # one readable line rather than a traceback that buries the reason.
        try:
            self.link = tp.make(self.transport_kind, host=self.host,
                                port=self.port, device=self.device,
                                baud=self.baud)
        except (RuntimeError, ValueError) as exc:
            self.get_logger().fatal(str(exc))
            raise SystemExit(1) from None

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

        if self.nadir_on_start:
            self._want_nadir = True
            self._command_nadir('startup — down for preflight')

        ok, why = siyi.clamped(self.nadir_yaw, self.nadir_pitch)
        if not ok:
            self.get_logger().warn(f'requested nadir is outside the A8 mini '
                                   f'travel and will be clamped: {why}')
        self.get_logger().info(
            f'siyi_gimbal_node: {self.link.description} — will look down '
            f'(yaw {self.nadir_yaw:.0f}, pitch {self.nadir_pitch:.0f} deg) '
            f'the moment /mavros/state reports ARMED')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these may be set. The launch file passes none."""
        p = self.declare_parameter
        # HOW the gimbal is wired: 'serial' or 'udp'. Defaults to serial
        # because that is how the aircraft is built — the gimbal hangs off the
        # Jetson's header UART, not the vehicle ethernet.
        p('transport', 'serial')
        # serial: the Jetson's 40-pin header UART. /dev/ttyUSB0 for a USB-TTL
        # adapter on a desktop.
        p('serial_device', '/dev/ttyTHS1')
        p('serial_baud', 115200)
        # udp: SIYI's factory address, used when the gimbal is on the vehicle
        # network instead. Same protocol, different pipe.
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
        # Look down AS SOON AS THIS NODE STARTS, not only once armed.
        # Preflight is when you want to see whether the camera can actually
        # find the marker — checking that after arming is checking it too late.
        # The arm trigger still fires and still re-asserts; this only moves the
        # first command earlier.
        p('nadir_on_start', True)
        # Recentre when the vehicle disarms. Ignored while nadir_on_start is
        # set: recentring after a landing would leave the next preflight
        # looking at the horizon, which is the problem nadir_on_start exists to
        # remove.
        p('center_on_disarm', True)

    # ---------------------------------------------------------------- helpers
    def _send(self, packet: bytes, what: str) -> None:
        try:
            self.link.send(packet)
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
            # Stay pointed down when nadir_on_start is set: the next preflight
            # wants the same view this one did.
            self._want_nadir = self.nadir_on_start
            if not first and self.disarm_centers and not self.nadir_on_start:
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
        try:
            packets = self.link.read()
        except OSError as exc:
            self.get_logger().warn(f'link read failed ({exc})',
                                   throttle_duration_sec=5.0)
            return
        for data in packets:
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
            data=f'{holding} | {self.link.description} | vehicle {armed} | '
                 f'gimbal {att} | sent {self._sent} | bad_rx {self._rx_bad}'))
        if self._want_nadir and self._attitude is None:
            self.get_logger().warn(
                f'no attitude feedback over {self.link.description} — commands '
                f'are being sent blind. serial: check the wiring, the baud and '
                f'that nothing else holds the port; udp: check the gimbal IP '
                f'and that the vehicle network is up',
                throttle_duration_sec=10.0)


def main(args=None):
    rclpy.init(args=args)
    node = SiyiGimbalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.link.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
