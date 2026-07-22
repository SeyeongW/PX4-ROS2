"""Gazebo waypoint trailer driver embedded in the landing ROS node."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from gz.msgs10.odometry_pb2 import Odometry as GzOdometry
from gz.msgs10.twist_pb2 import Twist as GzTwist
from gz.transport13 import Node as GzNode


_START_PHASES = {
    "marker_track_down",
    "precision_align",
    "precision_descent",
    "final_approach",
    "touchdown_confirm",
}
_STOP_PHASES = {"landed", "abort_climb", "failsafe_hold"}


@dataclass(frozen=True)
class _TrailerSample:
    stamp_sec: int
    stamp_nanosec: int
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class _RoundedCorner:
    entry: tuple[float, float]
    exit: tuple[float, float]
    center: tuple[float, float]
    incoming: tuple[float, float]
    outgoing: tuple[float, float]


class MovingTrailerBridge:
    """Drive one Gazebo trailer and publish its measured odometry."""

    def __init__(
        self,
        node: Node,
        phase_provider: Callable[[], str],
        odometry_callback: Callable[[Odometry], None] | None = None,
    ) -> None:
        self._node = node
        self._phase_provider = phase_provider
        self._odometry_callback = odometry_callback
        self._origin = tuple(
            float(node.get_parameter(name).value)
            for name in (
                "trailer_world_origin_x_enu_m",
                "trailer_world_origin_y_enu_m",
                "trailer_world_origin_z_enu_m",
            )
        )
        self._speed = float(node.get_parameter("trailer_speed_m_s").value)
        self._acceleration = float(
            node.get_parameter("trailer_acceleration_m_s2").value
        )
        flattened = [
            float(value)
            for value in node.get_parameter("trailer_waypoints_enu_m").value
        ]
        self._waypoint_radius = float(
            node.get_parameter("trailer_waypoint_radius_m").value
        )
        self._corner_radius = float(
            node.get_parameter("trailer_corner_radius_m").value
        )
        rate_hz = float(
            node.get_parameter("trailer_state_publish_rate_hz").value
        )
        self._warmup_samples = int(
            node.get_parameter("trailer_odometry_warmup_samples").value
        )
        self._future_tolerance_s = float(
            node.get_parameter("target_fusion_future_tolerance_s").value
        )
        if (
            not all(math.isfinite(value) for value in self._origin)
            or not math.isfinite(self._speed)
            or self._speed <= 0.0
            or not math.isfinite(self._acceleration)
            or self._acceleration <= 0.0
            or len(flattened) != 8
            or not all(math.isfinite(value) for value in flattened)
            or not all(0.0 <= value <= 200.0 for value in flattened)
            or not math.isfinite(self._waypoint_radius)
            or self._waypoint_radius <= 0.0
            or not math.isfinite(self._corner_radius)
            or self._corner_radius < 9.0
            or not math.isfinite(rate_hz)
            or rate_hz <= 0.0
            or self._warmup_samples < 1
            or not math.isfinite(self._future_tolerance_s)
            or self._future_tolerance_s < 0.0
        ):
            raise ValueError("invalid moving trailer parameters")
        self._waypoints = tuple(
            (flattened[index], flattened[index + 1])
            for index in range(0, len(flattened), 2)
        )
        segment_directions: list[tuple[float, float]] = []
        segment_lengths: list[float] = []
        for start, end in zip(
            self._waypoints,
            self._waypoints[1:] + self._waypoints[:1],
        ):
            delta_x = end[0] - start[0]
            delta_y = end[1] - start[1]
            absolute_x = abs(delta_x)
            absolute_y = abs(delta_y)
            if (absolute_x <= 1.0e-9) == (absolute_y <= 1.0e-9):
                raise ValueError(
                    "trailer route must be a four-corner axis-aligned loop"
                )
            length = math.hypot(delta_x, delta_y)
            segment_lengths.append(length)
            segment_directions.append((delta_x / length, delta_y / length))
        if any(
            length <= 2.0 * self._corner_radius
            for length in segment_lengths
        ):
            raise ValueError(
                "trailer side length must exceed twice the corner radius"
            )
        if (
            abs(segment_lengths[0] - segment_lengths[2]) > 1.0e-6
            or abs(segment_lengths[1] - segment_lengths[3]) > 1.0e-6
        ):
            raise ValueError("trailer route opposite sides must be equal")

        rounded_corners: list[_RoundedCorner] = []
        for index, corner in enumerate(self._waypoints):
            incoming = segment_directions[(index - 1) % 4]
            outgoing = segment_directions[index]
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            if cross <= 0.999999 or abs(dot) > 1.0e-6:
                raise ValueError(
                    "trailer route corners must all be 90-degree CCW turns"
                )
            entry = (
                corner[0] - self._corner_radius * incoming[0],
                corner[1] - self._corner_radius * incoming[1],
            )
            exit_point = (
                corner[0] + self._corner_radius * outgoing[0],
                corner[1] + self._corner_radius * outgoing[1],
            )
            center = (
                entry[0] + self._corner_radius * outgoing[0],
                entry[1] + self._corner_radius * outgoing[1],
            )
            rounded_corners.append(
                _RoundedCorner(
                    entry=entry,
                    exit=exit_point,
                    center=center,
                    incoming=incoming,
                    outgoing=outgoing,
                )
            )
        self._rounded_corners = tuple(rounded_corners)

        self._lock = threading.Lock()
        self._samples: deque[_TrailerSample] = deque(maxlen=1024)
        self._sample_count = 0
        self._last_published_stamp: tuple[int, int] | None = None
        self._motion_latched = False
        self._terminal_stop = False
        self._commanded_speed = 0.0
        self._last_command_monotonic = time.monotonic()
        # The trailer spawns at corner A and initially travels east toward the
        # tangent entry of corner B. Later laps round all four corners.
        self._corner_index = 1
        self._on_corner_arc = False

        latest_odometry_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._odometry_publisher = node.create_publisher(
            Odometry,
            str(node.get_parameter("trailer_odometry_topic").value),
            latest_odometry_qos,
        )
        self._gz = GzNode()
        command_topic = str(
            node.get_parameter("trailer_gz_command_topic").value
        ).strip()
        self._command_publisher = (
            self._gz.advertise(command_topic, GzTwist)
            if command_topic
            else None
        )
        self._gz.subscribe(
            GzOdometry,
            str(node.get_parameter("trailer_gz_odometry_topic").value),
            self._on_gz_odometry,
        )
        if self._command_publisher is None:
            node.get_logger().info(
                "embedded trailer odometry bridge: Gazebo command owner is external"
            )
        else:
            node.get_logger().info(
                "embedded trailer: 4-corner CCW loop, cruise=%.3f m/s, "
                "corner radius=%.2f m"
                % (self._speed, self._corner_radius)
            )

    def _on_gz_odometry(self, message: GzOdometry) -> None:
        stamp = message.header.stamp
        pose = message.pose
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return
        sample = _TrailerSample(
            int(stamp.sec),
            int(stamp.nsec),
            (float(values[0]), float(values[1]), float(values[2])),
            (float(values[3]), float(values[4]), float(values[5]), float(values[6])),
        )
        with self._lock:
            self._samples.append(sample)
            self._sample_count += 1

    def _update_phase(self) -> bool:
        phase = self._phase_provider()
        if phase in _STOP_PHASES:
            self._terminal_stop = True
            return False
        if phase in _START_PHASES and not self._motion_latched:
            self._motion_latched = True
            self._node.get_logger().info(
                "phase=%s: trailer waypoint motion started at %.3f m/s"
                % (phase, self._speed)
            )
        return self._motion_latched and not self._terminal_stop

    def _waypoint_direction(
        self, sample: _TrailerSample
    ) -> tuple[float, float] | None:
        position_x, position_y, _ = sample.position
        corner = self._rounded_corners[self._corner_index]
        if not self._on_corner_arc:
            remaining = (
                (corner.entry[0] - position_x) * corner.incoming[0]
                + (corner.entry[1] - position_y) * corner.incoming[1]
            )
            if remaining > self._waypoint_radius:
                return corner.incoming
            self._on_corner_arc = True
            self._node.get_logger().info(
                "trailer corner %d/4 arc entry -> (%.2f, %.2f)"
                % (
                    self._corner_index + 1,
                    corner.entry[0],
                    corner.entry[1],
                )
            )

        exit_progress = (
            (position_x - corner.exit[0]) * corner.outgoing[0]
            + (position_y - corner.exit[1]) * corner.outgoing[1]
        )
        if exit_progress >= 0.0:
            direction = corner.outgoing
            self._corner_index = (self._corner_index + 1) % 4
            self._on_corner_arc = False
            return direction

        radial_x = position_x - corner.center[0]
        radial_y = position_y - corner.center[1]
        radial_distance = math.hypot(radial_x, radial_y)
        if radial_distance <= 1.0e-9:
            return corner.incoming
        radial_unit_x = radial_x / radial_distance
        radial_unit_y = radial_y / radial_distance
        tangent_x = -radial_unit_y
        tangent_y = radial_unit_x
        radial_error_ratio = (
            radial_distance - self._corner_radius
        ) / self._corner_radius
        radial_correction = max(
            -0.20,
            min(0.20, -0.50 * radial_error_ratio),
        )
        direction_x = tangent_x + radial_correction * radial_unit_x
        direction_y = tangent_y + radial_correction * radial_unit_y
        direction_norm = math.hypot(direction_x, direction_y)
        return direction_x / direction_norm, direction_y / direction_norm

    def _publish_command(
        self, moving: bool, sample: _TrailerSample | None, immediate: bool = False
    ) -> None:
        if self._command_publisher is None:
            return
        now = time.monotonic()
        dt = max(0.0, min(now - self._last_command_monotonic, 0.20))
        self._last_command_monotonic = now
        direction = (
            self._waypoint_direction(sample)
            if moving and sample is not None
            else None
        )
        target_speed = self._speed if direction is not None else 0.0
        if immediate:
            self._commanded_speed = 0.0
        else:
            change = self._acceleration * dt
            error = target_speed - self._commanded_speed
            self._commanded_speed += max(-change, min(change, error))
        command = GzTwist()
        if direction is not None:
            command.linear.x = self._commanded_speed * direction[0]
            command.linear.y = self._commanded_speed * direction[1]
        self._command_publisher.publish(command)

    def tick(self, current_sample_time_s: float) -> None:
        """Forward the newest Gazebo sample from the node's 50 Hz control tick."""
        # Gazebo odometry payload stamps and PASSTHROUGH PX4 state stamps share
        # the raw simulation sample domain.  ROS /clock can lag that domain
        # under Python bridge load, so select against the newest complete PX4
        # state watermark rather than against the delivered ROS clock.
        eligible_limit_s = float(current_sample_time_s)
        with self._lock:
            sample_count = self._sample_count
            sample = None
            for candidate in self._samples:
                candidate_stamp_s = (
                    float(candidate.stamp_sec)
                    + float(candidate.stamp_nanosec) * 1.0e-9
                )
                if candidate_stamp_s <= eligible_limit_s:
                    sample = candidate
                else:
                    break
            if sample is not None:
                selected_stamp = (sample.stamp_sec, sample.stamp_nanosec)
                while self._samples:
                    oldest = self._samples[0]
                    if (oldest.stamp_sec, oldest.stamp_nanosec) > selected_stamp:
                        break
                    self._samples.popleft()
        if self._command_publisher is not None:
            moving = self._update_phase()
            self._publish_command(
                moving, sample, immediate=self._terminal_stop
            )
        if sample is None or sample_count <= self._warmup_samples:
            return
        stamp_key = (sample.stamp_sec, sample.stamp_nanosec)
        if stamp_key <= (0, 0) or stamp_key == self._last_published_stamp:
            return
        self._last_published_stamp = stamp_key
        message = Odometry()
        message.header.stamp.sec = sample.stamp_sec
        message.header.stamp.nanosec = sample.stamp_nanosec
        message.header.frame_id = "map"
        message.child_frame_id = "trailer/base_link"
        message.pose.pose.position.x = sample.position[0] - self._origin[0]
        message.pose.pose.position.y = sample.position[1] - self._origin[1]
        message.pose.pose.position.z = sample.position[2] - self._origin[2]
        orientation = message.pose.pose.orientation
        orientation.x, orientation.y, orientation.z, orientation.w = (
            sample.orientation
        )
        # The vehicle receives trailer position for interception, but never
        # the simulator's ground-truth speed.  Twist is deliberately zero
        # with very large covariance; production control estimates XY speed
        # only from timestamped ArUco world poses.
        for index in (0, 7, 14, 35):
            message.pose.covariance[index] = 1.0e-6
        for index in (0, 7, 14, 21, 28, 35):
            message.twist.covariance[index] = 1.0e6
        # The production mission consumes this sample synchronously in the
        # control tick.  Republishing to ROS and subscribing from the same
        # single-threaded executor previously starved this callback behind the
        # 50 Hz state/heartbeat traffic and created multi-second target gaps.
        if self._odometry_callback is not None:
            self._odometry_callback(message)
        self._odometry_publisher.publish(message)

    def stop(self) -> None:
        """Stop the simulated trailer immediately during node shutdown."""
        if self._command_publisher is not None:
            self._terminal_stop = True
            self._publish_command(False, None, immediate=True)
