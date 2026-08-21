"""trailer_gps_node — the trailer's position, off a 900 MHz MAVLink radio.

Step 1 of the trailer-coordinate pipeline (handoff Part C): open the telemetry
radio on the Jetson's serial port, prove the link, and turn the trailer's
GLOBAL_POSITION_INT into a ROS2 fix the rest of the pipeline can consume.

WHY IT LOGS SO MUCH ON THE BENCH
--------------------------------
The link and the GPS fix are two separate things, and conflating them wastes a
field trip. The radio, the USB serial, and the pymavlink parse can ALL be
verified indoors, with no sky and no fix, because HEARTBEAT (~1 Hz, always) and
GPS_RAW_INT (carries fix_type even at 0) arrive regardless. Only the actual
lat/lon needs a fix: ArduPilot does not emit a valid GLOBAL_POSITION_INT until
its EKF has a position, so indoors that message simply does not come — which is
correct, not a fault. So this node reports both, separately:

    link      HEARTBEAT seen -> the radio chain is alive
    gps       fix_type / sats -> the module's own progress, 0 -> 2D -> 3D
    position  GLOBAL_POSITION_INT rate -> only non-zero once fixed
    quality   message rate (Hz) and packet loss over the link

so a silent /trailer/fix is never ambiguous: the log already says whether it is
the link, the fix, or the position stream that is missing.

Bench first (1a): confirm `link up` and watch `fix_type` climb. Field next (1b):
confirm the position rate is non-zero and lat/lon track the moving trailer.

    ros2 run trailer_link trailer_gps_node
    ros2 run trailer_link trailer_gps_node --ros-args -p serial_device:=/dev/ttyUSB0 -p baud:=57600

Publishes
    /trailer/fix    sensor_msgs/NavSatFix    (only while GLOBAL_POSITION_INT flows)

Needs pymavlink:  pip install pymavlink
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import NavSatFix, NavSatStatus

try:
    from pymavlink import mavutil
except ImportError as exc:                       # pragma: no cover
    raise ImportError(
        'trailer_gps_node needs pymavlink — `pip install pymavlink`') from exc


#: MAV_GPS_FIX_TYPE, for the human-readable log.
_FIX_NAMES = {
    0: 'no GPS', 1: 'no fix', 2: '2D fix', 3: '3D fix', 4: 'DGPS',
    5: 'RTK float', 6: 'RTK fixed', 7: 'static', 8: 'PPP',
}


def fix_status(fix_type: int) -> int:
    """MAV_GPS_FIX_TYPE -> NavSatStatus.status.

    A 2D fix has no usable altitude and only rough horizontal position, so it is
    still reported NO_FIX here — the pipeline wants a real 3D position or nothing.
    """
    if fix_type >= 3:
        return NavSatStatus.STATUS_FIX
    return NavSatStatus.STATUS_NO_FIX


def _sensor_qos() -> QoSProfile:
    """Sensor-style, so a downstream node subscribing BEST_EFFORT is compatible."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class TrailerGpsNode(Node):
    def __init__(self):
        super().__init__('trailer_gps_node')
        self._declare()
        self._read_params()

        self.fix_pub = self.create_publisher(NavSatFix, self.publish_topic,
                                             _sensor_qos())

        # --- link/telemetry state, shared with the rx thread
        self._lock = threading.Lock()
        self._link_up = False
        self._sysid: int | None = None
        self._fix_type = 0
        self._sats = 0
        self._n_msgs = 0            # messages since last stats log
        self._n_gpi = 0            # GLOBAL_POSITION_INT since last stats log
        self._t_last_rx = 0.0
        self._master = None

        # The serial read is blocking, so it lives in its own daemon thread —
        # the node's executor (and the stats timer) never wait on the radio.
        self._stop = False
        threading.Thread(target=self._rx_loop, daemon=True).start()
        self.create_timer(self.stats_period, self._log_stats)

        self.get_logger().info(
            f'trailer_gps_node: {self.serial_device} @ {self.baud} baud -> '
            f'{self.publish_topic}. Bench check: wait for "link up", then watch '
            f'fix_type climb; position needs a 3D fix.')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        p = self.declare_parameter
        # The radio B end on the Jetson. SiK 900 MHz telemetry defaults to 57600.
        p('serial_device', '/dev/ttyUSB0')
        p('baud', 57600)
        p('publish_topic', '/trailer/fix')
        p('frame_id', 'trailer_gps')
        # 0 accepts any source system; set to the trailer's MAV_SYS_ID to ignore
        # anything else that might share the link.
        p('target_sysid', 0)
        # How often to print the link/gps/quality summary.
        p('stats_period_s', 2.0)
        # Louder than a warning if nothing arrives for this long — a dead link.
        p('link_timeout_s', 5.0)

    def _read_params(self) -> None:
        g = self.get_parameter
        self.serial_device = str(g('serial_device').value)
        self.baud = int(g('baud').value)
        self.publish_topic = str(g('publish_topic').value)
        self.frame_id = str(g('frame_id').value)
        self.target_sysid = int(g('target_sysid').value)
        self.stats_period = float(g('stats_period_s').value)
        self.link_timeout = float(g('link_timeout_s').value)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------- rx thread
    def _connect(self):
        """Open the serial link, retrying — the radio may be plugged in late."""
        while not self._stop and rclpy.ok():
            try:
                return mavutil.mavlink_connection(self.serial_device,
                                                  baud=self.baud)
            except Exception as e:                       # noqa: BLE001
                self.get_logger().error(
                    f'cannot open {self.serial_device}: {e} — is the radio '
                    f'plugged in? (ls /dev/ttyUSB*)', throttle_duration_sec=5.0)
                self._sleep(2.0)
        return None

    def _rx_loop(self) -> None:
        self._master = self._connect()
        if self._master is None:
            return
        while not self._stop and rclpy.ok():
            try:
                msg = self._master.recv_match(blocking=True, timeout=1.0)
            except Exception as e:                       # noqa: BLE001
                self.get_logger().error(f'serial read failed: {e} — reopening',
                                        throttle_duration_sec=5.0)
                self._master = self._connect()
                continue
            if msg is None:
                continue
            if self.target_sysid and msg.get_srcSystem() != self.target_sysid:
                continue
            self._handle(msg)

    def _handle(self, msg) -> None:
        mtype = msg.get_type()
        now = self._now()
        with self._lock:
            self._n_msgs += 1
            self._t_last_rx = now
            if not self._link_up:
                self._link_up = True
                self._sysid = msg.get_srcSystem()
                self.get_logger().info(
                    f'link up — receiving MAVLink from system {self._sysid} '
                    f'(first message: {mtype})')
        if mtype == 'GPS_RAW_INT':
            with self._lock:
                self._fix_type = int(msg.fix_type)
                self._sats = int(msg.satellites_visible)
        elif mtype == 'GLOBAL_POSITION_INT':
            with self._lock:
                self._n_gpi += 1
                fix_type = self._fix_type
            self._publish_fix(msg, fix_type)

    def _publish_fix(self, msg, fix_type: int) -> None:
        m = NavSatFix()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.status.status = fix_status(fix_type)
        m.status.service = NavSatStatus.SERVICE_GPS
        m.latitude = msg.lat * 1e-7          # GLOBAL_POSITION_INT: degE7
        m.longitude = msg.lon * 1e-7
        m.altitude = msg.alt * 1e-3          # mm -> m (AMSL)
        # No per-axis GPS covariance on this message; mark it unknown rather than
        # publishing a fake zero the downstream might trust as certainty.
        m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.fix_pub.publish(m)

    # ------------------------------------------------------------------ stats
    def _log_stats(self) -> None:
        with self._lock:
            n, n_gpi = self._n_msgs, self._n_gpi
            self._n_msgs = self._n_gpi = 0
            link_up, sysid = self._link_up, self._sysid
            fix_type, sats = self._fix_type, self._sats
            t_last = self._t_last_rx
        if not link_up:
            self.get_logger().warn(
                f'no MAVLink yet on {self.serial_device} — link DOWN. Check the '
                f'radio pair, the USB device, and the baud ({self.baud}).')
            return
        silent = self._now() - t_last
        if silent > self.link_timeout:
            self.get_logger().error(
                f'link STALLED — {silent:.1f} s since the last message '
                f'(was system {sysid}). Radio dropout or power?')
            return
        rate = n / self.stats_period
        gpi_rate = n_gpi / self.stats_period
        loss = self._loss_pct()
        fix = _FIX_NAMES.get(fix_type, f'fix {fix_type}')
        pos = (f'position {gpi_rate:.1f} Hz' if n_gpi
               else 'position — NONE (needs a 3D fix)')
        self.get_logger().info(
            f'link OK {rate:.1f} Hz (loss {loss}) | gps {fix}, {sats} sats | '
            f'{pos}')

    def _loss_pct(self) -> str:
        """Packet-loss %, if pymavlink is tracking it on this connection."""
        try:
            return f'{self._master.packet_loss():.0f}%'
        except Exception:                                # noqa: BLE001
            return 'n/a'

    def _sleep(self, s: float) -> None:
        """Interruptible sleep for the rx thread, so shutdown is prompt."""
        end = self._now() + s
        while not self._stop and rclpy.ok() and self._now() < end:
            import time
            time.sleep(0.05)

    def destroy_node(self):
        self._stop = True
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrailerGpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
