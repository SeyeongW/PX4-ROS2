"""ROS 2 / Gazebo adapter for the jerk-limited landing-trailer route.

The node commands Gazebo's VelocityControl system with the continuous S-curve
reference from :mod:`autonomy_planner.trailer_route`.  Gazebo dynamic pose is
used only as simulated trailer odometry; it is intentionally separate from
camera-based ArUco observations used by a precision-landing controller.

Published ROS interfaces (world ENU):

* ``/trailer/odometry`` (``nav_msgs/Odometry``)
* ``/trailer/pose`` (``geometry_msgs/PoseStamped``)
* ``/trailer/velocity`` (``geometry_msgs/TwistStamped``)
* TF ``trailer/odom -> trailer/base_link -> trailer/landing_deck ->
  trailer/aruco_marker``

The module is runnable after sourcing the workspace with
``python3 -m autonomy_planner.trailer_motion_node``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Optional

import numpy as np

from .trailer_route import JerkLimitedTrailerRoute, JerkLimitedWaypointRoute

try:  # Imports remain optional so the pure route module works without ROS.
    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
    from nav_msgs.msg import Odometry
    from rclpy.clock import Clock, ClockType
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.time import Time as RosTime
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    ROS_AVAILABLE = True
    ROS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host ROS install
    ROS_AVAILABLE = False
    ROS_IMPORT_ERROR = exc
    Node = object  # type: ignore[assignment,misc]

try:
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.twist_pb2 import Twist as GzTwist
    from gz.transport13 import Node as GzNode

    GZ_AVAILABLE = True
    GZ_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host gz install
    GZ_AVAILABLE = False
    GZ_IMPORT_ERROR = exc


@dataclass(frozen=True)
class _ActualPose:
    position_enu_m: np.ndarray
    quaternion_xyzw: np.ndarray
    velocity_enu_m_s: np.ndarray
    measurement_time_s: float
    received_wall_s: float


class _GazeboPoseReceiver:
    """Thread-safe dynamic-pose receiver with finite-difference velocity."""

    def __init__(self, entity_name: str, velocity_filter_alpha: float = 0.35) -> None:
        self.entity_name = entity_name
        self.velocity_filter_alpha = float(velocity_filter_alpha)
        if not 0.0 < self.velocity_filter_alpha <= 1.0:
            raise ValueError("velocity_filter_alpha must be in (0, 1]")
        self._lock = threading.Lock()
        self._value: Optional[_ActualPose] = None

    def callback(self, message: "Pose_V") -> None:
        for pose in message.pose:
            if pose.name != self.entity_name:
                continue
            now = time.monotonic()
            position = np.array(
                [pose.position.x, pose.position.y, pose.position.z],
                dtype=np.float64,
            )
            quaternion = np.array(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(quaternion))
            if not np.all(np.isfinite(position)) or not math.isfinite(norm) or norm < 1e-9:
                return
            quaternion /= norm
            stamp = message.header.stamp
            measurement_time_s = float(stamp.sec) + 1e-9 * float(stamp.nsec)
            if measurement_time_s <= 0.0:
                measurement_time_s = now
            with self._lock:
                previous = self._value
                if previous is None:
                    velocity = np.zeros(3, dtype=np.float64)
                else:
                    dt = measurement_time_s - previous.measurement_time_s
                    if 1e-4 < dt < 1.0:
                        raw_velocity = (position - previous.position_enu_m) / dt
                        alpha = self.velocity_filter_alpha
                        velocity = (
                            alpha * raw_velocity
                            + (1.0 - alpha) * previous.velocity_enu_m_s
                        )
                    else:
                        velocity = previous.velocity_enu_m_s.copy()
                self._value = _ActualPose(
                    position, quaternion, velocity, measurement_time_s, now
                )
            return

    def get(self) -> Optional[_ActualPose]:
        with self._lock:
            value = self._value
            if value is None:
                return None
            return _ActualPose(
                value.position_enu_m.copy(),
                value.quaternion_xyzw.copy(),
                value.velocity_enu_m_s.copy(),
                value.measurement_time_s,
                value.received_wall_s,
            )


def _yaw_from_quaternion(quaternion_xyzw: np.ndarray) -> float:
    x, y, z, w = quaternion_xyzw
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


class TrailerMotionNode(Node):  # pragma: no cover - exercised in Gazebo
    """Drive the trailer reference and expose native ROS odometry and TF."""

    def __init__(self) -> None:
        if not GZ_AVAILABLE:
            raise RuntimeError(f"Gazebo Python transport unavailable: {GZ_IMPORT_ERROR}")
        super().__init__("trailer_motion")

        world = str(self.declare_parameter("world", "applepark_city").value)
        entity = str(self.declare_parameter("entity", "trailer").value)
        command_topic = str(
            self.declare_parameter(
                "command_topic", "/model/trailer/cmd_vel"
            ).value
        )
        rate_hz = float(self.declare_parameter("rate_hz", 50.0).value)
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")

        start = self.declare_parameter(
            "start_enu_m", [200.0, -125.0, 0.0]
        ).value
        goal = self.declare_parameter(
            "goal_enu_m", [-128.0, -128.0, 0.0]
        ).value
        route_waypoints = self.declare_parameter(
            "route_waypoints_enu_m",
            [
                200.0, -125.0, 0.0,
                200.0, -100.0, 0.0,
                -128.0, -100.0, 0.0,
                -128.0, -128.0, 0.0,
            ],
        ).value
        cruise_speed = float(
            self.declare_parameter("cruise_speed_m_s", 3.0).value
        )
        max_acceleration = float(
            self.declare_parameter("max_acceleration_m_s2", 1.0).value
        )
        max_jerk = float(
            self.declare_parameter("max_jerk_m_s3", 0.5).value
        )
        if route_waypoints:
            if len(route_waypoints) < 6 or len(route_waypoints) % 3 != 0:
                raise ValueError(
                    "route_waypoints_enu_m must be a flat [x,y,z,...] array "
                    "containing at least two waypoints"
                )
            self.route = JerkLimitedWaypointRoute(
                np.asarray(route_waypoints, dtype=np.float64).reshape(-1, 3),
                cruise_speed_m_s=cruise_speed,
                max_acceleration_m_s2=max_acceleration,
                max_jerk_m_s3=max_jerk,
            )
        else:
            self.route = JerkLimitedTrailerRoute(
                start_enu_m=start,
                goal_enu_m=goal,
                cruise_speed_m_s=cruise_speed,
                max_acceleration_m_s2=max_acceleration,
                max_jerk_m_s3=max_jerk,
            )
        self.start_tolerance_m = float(
            self.declare_parameter("start_tolerance_m", 0.75).value
        )
        self.pose_stale_timeout_s = float(
            self.declare_parameter("pose_stale_timeout_s", 0.5).value
        )
        velocity_filter_alpha = float(
            self.declare_parameter("velocity_filter_alpha", 0.35).value
        )
        self.odom_frame = str(
            self.declare_parameter("odom_frame", "trailer/odom").value
        )
        self.base_frame = str(
            self.declare_parameter("base_frame", "trailer/base_link").value
        )
        self.deck_frame = str(
            self.declare_parameter("deck_frame", "trailer/landing_deck").value
        )
        self.marker_frame = str(
            self.declare_parameter("marker_frame", "trailer/aruco_marker").value
        )
        self.deck_height_m = float(
            self.declare_parameter("deck_height_m", 1.25).value
        )
        marker_offset = self.declare_parameter(
            "marker_offset_deck_m", [0.0, 0.0, 0.001]
        ).value
        if len(marker_offset) != 3:
            raise ValueError("marker_offset_deck_m must contain three values")
        self.marker_offset_deck_m = tuple(float(value) for value in marker_offset)
        self.marker_yaw_rad = float(
            self.declare_parameter("marker_yaw_rad", 0.0).value
        )

        odometry_topic = str(
            self.declare_parameter("odometry_topic", "/trailer/odometry").value
        )
        pose_topic = str(
            self.declare_parameter("pose_topic", "/trailer/pose").value
        )
        velocity_topic = str(
            self.declare_parameter("velocity_topic", "/trailer/velocity").value
        )
        self.odom_publisher = self.create_publisher(Odometry, odometry_topic, 10)
        self.pose_publisher = self.create_publisher(PoseStamped, pose_topic, 10)
        self.velocity_publisher = self.create_publisher(
            TwistStamped, velocity_topic, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.gz_node = GzNode()
        self.command_publisher = self.gz_node.advertise(command_topic, GzTwist)
        self.receiver = _GazeboPoseReceiver(entity, velocity_filter_alpha)
        pose_topic = f"/world/{world}/dynamic_pose/info"
        self.gz_node.subscribe(Pose_V, pose_topic, self.receiver.callback)

        self._profile_start_sim_s: Optional[float] = None
        self._last_error_log_wall_s = -math.inf
        self._publish_static_transforms()
        # This adapter must keep receiving/publishing even if an ABI-mismatched
        # ros_gz bridge cannot provide /clock.  Route progress itself uses the
        # Gazebo dynamic-pose timestamp, while the callback timer is steady-wall.
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            1.0 / rate_hz, self._tick, clock=self._steady_clock
        )

        segment_count = len(getattr(self.route, "segments", (self.route,)))
        self.get_logger().info(
            "trailer route ready: "
            f"start={self.route.start_enu_m.tolist()} "
            f"goal={self.route.goal_enu_m.tolist()} "
            f"distance={self.route.distance_m:.3f} m "
            f"segments={segment_count} "
            f"peak_speed={self.route.peak_speed_m_s:.3f} m/s "
            f"duration={self.route.duration_s:.3f} s"
        )
        self.get_logger().info(
            f"Gazebo pose={pose_topic} command={command_topic}; "
            "ground truth is published as trailer odometry only"
        )

    def _transform(self, parent: str, child: str) -> "TransformStamped":
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.rotation.w = 1.0
        return transform

    def _publish_static_transforms(self) -> None:
        base_to_deck = self._transform(self.base_frame, self.deck_frame)
        base_to_deck.transform.translation.z = self.deck_height_m
        deck_to_marker = self._transform(self.deck_frame, self.marker_frame)
        (
            deck_to_marker.transform.translation.x,
            deck_to_marker.transform.translation.y,
            deck_to_marker.transform.translation.z,
        ) = self.marker_offset_deck_m
        deck_to_marker.transform.rotation.z = math.sin(self.marker_yaw_rad / 2.0)
        deck_to_marker.transform.rotation.w = math.cos(self.marker_yaw_rad / 2.0)
        self.static_tf_broadcaster.sendTransform([base_to_deck, deck_to_marker])

    def _publish_command(self, velocity_enu_m_s: np.ndarray) -> None:
        command = GzTwist()
        command.linear.x = float(velocity_enu_m_s[0])
        command.linear.y = float(velocity_enu_m_s[1])
        command.linear.z = float(velocity_enu_m_s[2])
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = 0.0
        self.command_publisher.publish(command)

    @staticmethod
    def _copy_pose(message_pose: object, actual: _ActualPose) -> None:
        message_pose.position.x = float(actual.position_enu_m[0])
        message_pose.position.y = float(actual.position_enu_m[1])
        message_pose.position.z = float(actual.position_enu_m[2])
        message_pose.orientation.x = float(actual.quaternion_xyzw[0])
        message_pose.orientation.y = float(actual.quaternion_xyzw[1])
        message_pose.orientation.z = float(actual.quaternion_xyzw[2])
        message_pose.orientation.w = float(actual.quaternion_xyzw[3])

    def _publish_actual_state(self, actual: _ActualPose) -> None:
        stamp = RosTime(
            nanoseconds=int(round(actual.measurement_time_s * 1e9))
        ).to_msg()
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.odom_frame
        self._copy_pose(pose.pose, actual)
        self.pose_publisher.publish(pose)

        velocity = TwistStamped()
        velocity.header.stamp = stamp
        velocity.header.frame_id = self.odom_frame
        velocity.twist.linear.x = float(actual.velocity_enu_m_s[0])
        velocity.twist.linear.y = float(actual.velocity_enu_m_s[1])
        velocity.twist.linear.z = float(actual.velocity_enu_m_s[2])
        self.velocity_publisher.publish(velocity)

        odometry = Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = self.odom_frame
        odometry.child_frame_id = self.base_frame
        self._copy_pose(odometry.pose.pose, actual)
        yaw = _yaw_from_quaternion(actual.quaternion_xyzw)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        vx, vy, vz = actual.velocity_enu_m_s
        odometry.twist.twist.linear.x = float(cos_yaw * vx + sin_yaw * vy)
        odometry.twist.twist.linear.y = float(-sin_yaw * vx + cos_yaw * vy)
        odometry.twist.twist.linear.z = float(vz)
        self.odom_publisher.publish(odometry)

        world_to_base = self._transform(self.odom_frame, self.base_frame)
        world_to_base.header.stamp = stamp
        world_to_base.transform.translation.x = float(actual.position_enu_m[0])
        world_to_base.transform.translation.y = float(actual.position_enu_m[1])
        world_to_base.transform.translation.z = float(actual.position_enu_m[2])
        world_to_base.transform.rotation.x = float(actual.quaternion_xyzw[0])
        world_to_base.transform.rotation.y = float(actual.quaternion_xyzw[1])
        world_to_base.transform.rotation.z = float(actual.quaternion_xyzw[2])
        world_to_base.transform.rotation.w = float(actual.quaternion_xyzw[3])
        self.tf_broadcaster.sendTransform(world_to_base)

    def _log_throttled_error(self, text: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log_wall_s >= 2.0:
            self.get_logger().error(text)
            self._last_error_log_wall_s = now

    def _tick(self) -> None:
        actual = self.receiver.get()
        if actual is None:
            self._publish_command(np.zeros(3, dtype=np.float64))
            self._log_throttled_error("waiting for the configured trailer Gazebo pose")
            return
        self._publish_actual_state(actual)

        age = time.monotonic() - actual.received_wall_s
        if age > self.pose_stale_timeout_s:
            self._publish_command(np.zeros(3, dtype=np.float64))
            self._log_throttled_error(
                f"trailer pose stale ({age:.3f} s); commanding zero velocity"
            )
            return

        if self._profile_start_sim_s is None:
            start_error = float(
                np.linalg.norm(actual.position_enu_m[:2] - self.route.start_enu_m[:2])
            )
            if start_error > self.start_tolerance_m:
                self._publish_command(np.zeros(3, dtype=np.float64))
                self._log_throttled_error(
                    f"trailer is {start_error:.3f} m from required start "
                    f"({self.route.start_enu_m[0]:.1f}, {self.route.start_enu_m[1]:.1f}); "
                    "motion is inhibited"
                )
                return
            self._profile_start_sim_s = actual.measurement_time_s
            self.get_logger().info("trailer start pose verified; beginning S-curve")

        elapsed_s = max(
            0.0, actual.measurement_time_s - self._profile_start_sim_s
        )
        reference = self.route.sample(elapsed_s)
        self._publish_command(reference.velocity_enu_m_s)
        if reference.complete:
            endpoint_error = float(
                np.linalg.norm(actual.position_enu_m[:2] - self.route.goal_enu_m[:2])
            )
            if endpoint_error > 1.0:
                self._log_throttled_error(
                    f"reference complete but trailer endpoint error is {endpoint_error:.3f} m"
                )

    def stop(self) -> None:
        for _ in range(5):
            self._publish_command(np.zeros(3, dtype=np.float64))
            time.sleep(0.01)


def main(args: Optional[list[str]] = None) -> None:  # pragma: no cover - ROS entry
    if not ROS_AVAILABLE:
        raise RuntimeError(f"ROS 2 Python modules unavailable: {ROS_IMPORT_ERROR}")
    rclpy.init(args=args)
    node: Optional[TrailerMotionNode] = None
    try:
        node = TrailerMotionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
