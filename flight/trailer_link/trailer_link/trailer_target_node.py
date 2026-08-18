"""trailer_target_node — the trailer as a point the mission can fly to.

Step 2 of the trailer-coordinate pipeline (Part C), and the only consumer of
`trailer_gps_node`'s fix. It takes three streams that all exist already:

    /trailer/fix                        the trailer's lat/lon (trailer_gps_node)
    /mavros/global_position/global      the VEHICLE's lat/lon
    /mavros/local_position/pose         the VEHICLE's local ENU position

and publishes one thing the mission node can use without knowing any geodesy:

    /trailer/target_local               the trailer, in the vehicle's local frame

The arithmetic and the refusals live in `geodesy.py`, ROS-free and unit-tested;
this node is the plumbing around them.

SILENCE IS THE ERROR SIGNAL
---------------------------
The target is published ONLY while it is fully valid — every input fresh, the
trailer holding a 3D fix, and the distance inside the sanity limit. There is no
"stale" or "degraded" flag for a downstream to misread, and no last-known value
that keeps arriving after the radio has died. A consumer therefore needs one
rule, the same one the marker pipeline uses: if the last message is older than
its timeout, there is no target.

Nothing is hidden by that silence — the throttled log always says which input is
missing, so a quiet topic is never a mystery:

    ros2 topic echo /trailer/target_local
    ros2 run trailer_link trailer_target_node

Publishes
    /trailer/target_local   geometry_msgs/PointStamped   (frame: map, ENU)
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import NavSatFix, NavSatStatus

from .geodesy import (DEFAULT_MAX_DISTANCE_M, DEFAULT_STALE_AFTER_S,
                      RelativeTarget)


def _sensor_qos() -> QoSProfile:
    """MAVROS publishes telemetry BEST_EFFORT; a RELIABLE subscriber gets nothing."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class TrailerTargetNode(Node):
    def __init__(self):
        super().__init__('trailer_target_node')
        self._declare()
        self._read_params()

        self.target = RelativeTarget(stale_after=self.stale_after,
                                     max_distance=self.max_distance)
        self._last_block: str | None = 'starting up'
        self._published = 0

        self.create_subscription(NavSatFix, self.trailer_fix_topic,
                                 self._on_trailer_fix, _sensor_qos())
        self.create_subscription(NavSatFix, self.vehicle_fix_topic,
                                 self._on_vehicle_fix, _sensor_qos())
        self.create_subscription(PoseStamped, self.vehicle_pose_topic,
                                 self._on_vehicle_pose, _sensor_qos())

        # Published sensor-style, to match every other perception topic in the
        # stack — a mission node subscribing BEST_EFFORT is then compatible.
        self.target_pub = self.create_publisher(PointStamped,
                                                self.publish_topic, _sensor_qos())

        # Publishing on its own timer, not on an input callback: the output rate
        # is then a property of this node rather than of whichever radio packet
        # happened to arrive, and a consumer's freshness timeout means the same
        # thing at every stage of the flight.
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.create_timer(self.stats_period, self._log_stats)

        self.get_logger().info(
            f'trailer_target_node: {self.trailer_fix_topic} + '
            f'{self.vehicle_fix_topic} -> {self.publish_topic} at '
            f'{self.rate_hz:.0f} Hz, refusing anything past '
            f'{self.max_distance:.0f} m')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these numbers may be set."""
        p = self.declare_parameter
        p('trailer_fix_topic', '/trailer/fix')
        p('vehicle_fix_topic', '/mavros/global_position/global')
        p('vehicle_pose_topic', '/mavros/local_position/pose')
        p('publish_topic', '/trailer/target_local')
        p('map_frame', 'map')
        p('rate_hz', 10.0)
        # Inputs older than this are absent, not fact (geodesy.py).
        p('stale_after_s', DEFAULT_STALE_AFTER_S)
        # A target farther than this is refused as a bad fix, NOT clamped.
        p('max_distance_m', DEFAULT_MAX_DISTANCE_M)
        p('stats_period_s', 2.0)

    def _read_params(self) -> None:
        g = self.get_parameter
        self.trailer_fix_topic = str(g('trailer_fix_topic').value)
        self.vehicle_fix_topic = str(g('vehicle_fix_topic').value)
        self.vehicle_pose_topic = str(g('vehicle_pose_topic').value)
        self.publish_topic = str(g('publish_topic').value)
        self.map_frame = str(g('map_frame').value)
        self.rate_hz = float(g('rate_hz').value)
        self.stale_after = float(g('stale_after_s').value)
        self.max_distance = float(g('max_distance_m').value)
        self.stats_period = float(g('stats_period_s').value)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------- callbacks
    def _on_trailer_fix(self, m: NavSatFix) -> None:
        # STATUS_NO_FIX still carries a lat/lon field; it is simply not a
        # position. trailer_gps_node already maps a 2D fix to NO_FIX, so this is
        # the single place the pipeline decides whether the number means anything.
        self.target.on_trailer_fix(
            self._now(), lat=m.latitude, lon=m.longitude, alt=m.altitude,
            has_fix=m.status.status >= NavSatStatus.STATUS_FIX)

    def _on_vehicle_fix(self, m: NavSatFix) -> None:
        self.target.on_vehicle_fix(self._now(), lat=m.latitude,
                                   lon=m.longitude, alt=m.altitude)

    def _on_vehicle_pose(self, m: PoseStamped) -> None:
        self.target.on_vehicle_local(self._now(), x=m.pose.position.x,
                                     y=m.pose.position.y, z=m.pose.position.z)

    # ------------------------------------------------------------------- loop
    def _tick(self) -> None:
        now = self._now()
        solved = self.target.solve(now)
        if solved is None:
            self._note_block(self.target.blocking_reason(now))
            return
        self._note_block(None)
        self._published += 1

        m = PointStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.point.x, m.point.y, m.point.z = solved
        self.target_pub.publish(m)

    def _note_block(self, reason: str | None) -> None:
        """Say it once when it changes, so the log records transitions not spam."""
        if reason == self._last_block:
            return
        self._last_block = reason
        if reason is None:
            self.get_logger().info('trailer target VALID — publishing')
        else:
            self.get_logger().warn(f'no trailer target: {reason}')

    def _log_stats(self) -> None:
        n, self._published = self._published, 0
        rate = n / self.stats_period
        self.get_logger().info(
            f'{self.target.summary(self._now())} | out {rate:.1f} Hz')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrailerTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
