"""Receive the trailer's position over the field-used MAVLink radio link.

This intentionally keeps the committed ``wang`` serial/read/diagnostic path.
The only control-facing addition is a velocity message from the same
``GLOBAL_POSITION_INT`` sample and with the same ROS source stamp as the fix.
It does not change autopilot parameters or request streams at runtime.
"""

from __future__ import annotations

import math
import threading

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import NavSatFix, NavSatStatus

try:
    from pymavlink import mavutil
except ImportError as exc:  # pragma: no cover - deployment dependency
    raise ImportError(
        'trailer_gps_node needs pymavlink (`pip install pymavlink`)') from exc


_FIX_NAMES = {
    0: 'no GPS', 1: 'no fix', 2: '2D fix', 3: '3D fix', 4: 'DGPS',
    5: 'RTK float', 6: 'RTK fixed', 7: 'static', 8: 'PPP',
}


def fix_status(fix_type: int) -> int:
    """Map MAV_GPS_FIX_TYPE to the conservative NavSatFix status."""
    if int(fix_type) >= 3:
        return NavSatStatus.STATUS_FIX
    return NavSatStatus.STATUS_NO_FIX


def gpi_velocity_enu(vx_cm_s: float, vy_cm_s: float,
                     vz_cm_s: float = 0.0) -> tuple[float, float, float]:
    """Convert GLOBAL_POSITION_INT NED cm/s to horizontal ENU m/s.

    The landing deck height is calibrated separately, so vertical GPS speed is
    intentionally not passed into control.
    """
    values = tuple(float(value) for value in
                   (vx_cm_s, vy_cm_s, vz_cm_s))
    if not all(math.isfinite(value) for value in values):
        raise ValueError('GLOBAL_POSITION_INT velocity must be finite')
    vx_north, vy_east, _ = values
    return 0.01 * vy_east, 0.01 * vx_north, 0.0


def _sensor_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


class TrailerGpsNode(Node):
    """Field trailer MAVLink receiver with a paired ENU velocity output."""

    def __init__(self) -> None:
        super().__init__('trailer_gps_node')
        p = self.declare_parameter
        self.serial_device = str(p('serial_device', '/dev/ttyUSB0').value)
        self.baud = int(p('baud', 57600).value)
        # Keep the field-used parameter names and defaults compatible.
        self.publish_topic = str(p('publish_topic', '/trailer/fix').value)
        self.frame_id = str(p('frame_id', 'trailer_gps').value)
        self.velocity_topic = str(
            p('velocity_topic', '/trailer/velocity_enu').value)
        self.velocity_frame = str(
            p('velocity_frame_id', 'earth_enu').value)
        self.target_sysid = int(p('target_sysid', 0).value)
        self.max_speed = float(p('max_speed_m_s', 20.0).value)
        self.stats_period = float(p('stats_period_s', 2.0).value)
        self.link_timeout = float(p('link_timeout_s', 5.0).value)
        self.fix_status_timeout = float(
            p('fix_status_timeout_s', 1.5).value)
        if not self.serial_device or self.baud <= 0:
            raise ValueError('serial_device and positive baud are required')
        if not 0 <= self.target_sysid <= 254:
            raise ValueError('target_sysid must be 0..254')
        limits = (self.max_speed, self.stats_period, self.link_timeout,
                  self.fix_status_timeout)
        if not all(math.isfinite(value) and value > 0.0
                   for value in limits):
            raise ValueError('GPS limits and periods must be positive')

        qos = _sensor_qos()
        self.fix_pub = self.create_publisher(
            NavSatFix, self.publish_topic, qos)
        self.velocity_pub = self.create_publisher(
            Vector3Stamped, self.velocity_topic, qos)

        self._lock = threading.Lock()
        self._link_up = False
        self._sysid: int | None = None
        self._fix_type = 0
        self._sats = 0
        self._t_last_fix_status = -math.inf
        self._n_msgs = 0
        self._n_gpi = 0
        self._n_pairs = 0
        self._t_last_rx = 0.0
        self._master = None
        self._stop = False
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()
        self.create_timer(self.stats_period, self._log_stats)
        self.get_logger().info(
            f'trailer radio {self.serial_device}@{self.baud} -> '
            f'{self.publish_topic} + {self.velocity_topic}; '
            'GLOBAL_POSITION_INT must already be enabled on the trailer')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _connect(self):
        while not self._stop and rclpy.ok():
            try:
                return mavutil.mavlink_connection(
                    self.serial_device, baud=self.baud)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(
                    f'cannot open {self.serial_device}: {exc}; is the radio '
                    'plugged in?', throttle_duration_sec=5.0)
                self._sleep(2.0)
        return None

    def _rx_loop(self) -> None:
        self._master = self._connect()
        if self._master is None:
            return
        while not self._stop and rclpy.ok():
            try:
                message = self._master.recv_match(blocking=True, timeout=1.0)
            except Exception as exc:  # noqa: BLE001
                if self._stop or not rclpy.ok():
                    break
                self.get_logger().error(
                    f'serial read failed: {exc}; reopening',
                    throttle_duration_sec=5.0)
                self._close_master()
                self._reset_link_state()
                self._master = self._connect()
                continue
            if message is None:
                continue
            if (self.target_sysid
                    and message.get_srcSystem() != self.target_sysid):
                continue
            self._handle(message)

    def _handle(self, message) -> None:
        message_type = message.get_type()
        now = self._now()
        with self._lock:
            self._n_msgs += 1
            self._t_last_rx = now
            if not self._link_up:
                self._link_up = True
                self._sysid = message.get_srcSystem()
                self.get_logger().info(
                    f'link up from MAVLink system {self._sysid} '
                    f'(first message: {message_type})')
        if message_type == 'GPS_RAW_INT':
            with self._lock:
                self._fix_type = int(message.fix_type)
                self._sats = int(message.satellites_visible)
                self._t_last_fix_status = now
            return
        if message_type != 'GLOBAL_POSITION_INT':
            return
        with self._lock:
            self._n_gpi += 1
            fix_type = (self._fix_type
                        if now - self._t_last_fix_status
                        <= self.fix_status_timeout else 0)
        if self._publish_pair(message, fix_type):
            with self._lock:
                self._n_pairs += 1

    def _publish_pair(self, message, fix_type: int | None = None) -> bool:
        lat = float(message.lat) * 1.0e-7
        lon = float(message.lon) * 1.0e-7
        alt = float(message.alt) * 1.0e-3  # AMSL; diagnostic only downstream.
        try:
            velocity = gpi_velocity_enu(
                message.vx, message.vy, message.vz)
        except (AttributeError, TypeError, ValueError):
            self.get_logger().error('refusing invalid trailer GPS velocity')
            return False
        if (not all(math.isfinite(value)
                    for value in (lat, lon, alt, *velocity))
                or not -90.0 <= lat <= 90.0
                or not -180.0 <= lon <= 180.0
                or math.hypot(*velocity[:2]) > self.max_speed):
            self.get_logger().error('refusing invalid trailer GPS sample')
            return False

        stamp = self.get_clock().now().to_msg()
        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = self.frame_id
        fix.status.status = fix_status(
            self._fix_type if fix_type is None else fix_type)
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude, fix.longitude, fix.altitude = lat, lon, alt
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        speed = Vector3Stamped()
        speed.header.stamp = stamp
        speed.header.frame_id = self.velocity_frame
        speed.vector.x, speed.vector.y, speed.vector.z = velocity
        self.fix_pub.publish(fix)
        self.velocity_pub.publish(speed)
        return True

    def _reset_link_state(self) -> None:
        """Forget prior-link quality before accepting a new serial session."""
        with self._lock:
            self._link_up = False
            self._sysid = None
            self._fix_type = 0
            self._sats = 0
            self._t_last_fix_status = -math.inf

    def _log_stats(self) -> None:
        with self._lock:
            messages, gpi, pairs = self._n_msgs, self._n_gpi, self._n_pairs
            self._n_msgs = self._n_gpi = self._n_pairs = 0
            link_up, sysid = self._link_up, self._sysid
            fix_type, sats = self._fix_type, self._sats
            last_rx = self._t_last_rx
        if not link_up:
            self.get_logger().warn(
                f'no MAVLink on {self.serial_device}; link DOWN')
            return
        silent = self._now() - last_rx
        if silent > self.link_timeout:
            self.get_logger().error(
                f'link STALLED for {silent:.1f}s (system {sysid})')
            return
        loss = self._loss_pct()
        self.get_logger().info(
            f'link {messages / self.stats_period:.1f}Hz (loss {loss}) | '
            f'gps {_FIX_NAMES.get(fix_type, fix_type)}, {sats} sats | '
            f'position {gpi / self.stats_period:.1f}Hz | '
            f'paired output {pairs / self.stats_period:.1f}Hz')

    def _loss_pct(self) -> str:
        try:
            return f'{self._master.packet_loss():.0f}%'
        except Exception:  # noqa: BLE001
            return 'n/a'

    def _sleep(self, seconds: float) -> None:
        import time
        end = time.monotonic() + seconds
        while not self._stop and rclpy.ok() and time.monotonic() < end:
            time.sleep(0.05)

    def _close_master(self) -> None:
        master, self._master = self._master, None
        if master is not None:
            try:
                master.close()
            except Exception:  # noqa: BLE001
                pass

    def destroy_node(self):
        self._stop = True
        self._close_master()
        if (self._thread.is_alive()
                and self._thread is not threading.current_thread()):
            self._thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrailerGpsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
