#!/usr/bin/env python3
"""Drive the seo-branch moving platform along a map-defined ENU route.

This talks directly to Gazebo Transport, so it needs neither MAVROS nor a ROS
workspace build.  The platform is kinematic rather than a wheeled vehicle.
The mountain route is therefore kept on the exact-flat central corridor; a
heightmap safeguard raises the horizontal body before any off-route terrain
can penetrate its 5x5 m footprint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
from PIL import Image
import yaml

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node


GAZEBO = Path(__file__).resolve().parent
REPO = GAZEBO.parent
STOP = threading.Event()


def quaternion_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return roll, pitch, math.atan2(siny, cosy)


def conservative_stop_speed(
    measured_speed_m_s: float | None,
    commanded_speed_m_s: float,
) -> float:
    if measured_speed_m_s is None:
        return commanded_speed_m_s
    return max(measured_speed_m_s, commanded_speed_m_s)


def pose_v_sim_time_s(message: Pose_V) -> float | None:
    """Return Gazebo simulation time carried by a Pose_V message."""
    stamp = message.header.stamp
    value = float(stamp.sec) + float(stamp.nsec) * 1.0e-9
    return value if math.isfinite(value) and value >= 0.0 else None


class PoseReceiver:
    def __init__(self, entity_name: str):
        self.entity_name = entity_name
        self.lock = threading.Lock()
        self.value: tuple[float, float, float, float, float, float] | None = None
        self.received_monotonic: float | None = None
        self.received_sim_time_s: float | None = None
        self.xy_speed_m_s: float | None = None

    def callback(self, message: Pose_V) -> None:
        for pose in message.pose:
            if pose.name != self.entity_name:
                continue
            orientation = pose.orientation
            roll, pitch, yaw = quaternion_rpy(
                orientation.x, orientation.y, orientation.z, orientation.w
            )
            now = time.monotonic()
            sim_time_s = pose_v_sim_time_s(message)
            with self.lock:
                if self.value is not None and self.received_monotonic is not None:
                    dt = None
                    if (sim_time_s is not None
                            and self.received_sim_time_s is not None
                            and sim_time_s > self.received_sim_time_s):
                        dt = sim_time_s - self.received_sim_time_s
                    elif sim_time_s is None or self.received_sim_time_s is None:
                        dt = now - self.received_monotonic
                    if dt is not None and dt > 1.0e-6:
                        self.xy_speed_m_s = math.hypot(
                            pose.position.x - self.value[0],
                            pose.position.y - self.value[1],
                        ) / dt
                self.value = (
                    pose.position.x, pose.position.y, pose.position.z,
                    roll, pitch, yaw,
                )
                self.received_monotonic = now
                self.received_sim_time_s = sim_time_s
            return

    def get(self) -> tuple[float, float, float, float, float, float] | None:
        with self.lock:
            return self.value

    def age_s(self, now: float) -> float:
        with self.lock:
            if self.received_monotonic is None:
                return math.inf
            return max(0.0, now - self.received_monotonic)

    def speed_xy(self) -> float | None:
        with self.lock:
            return self.xy_speed_m_s

    def simulation_time_s(self) -> float | None:
        with self.lock:
            return self.received_sim_time_s


@dataclass(frozen=True)
class _RoundedCorner:
    entry: tuple[float, float]
    exit: tuple[float, float]
    center: tuple[float, float]
    incoming: tuple[float, float]
    outgoing: tuple[float, float]


class RoundedSquareRoute:
    """Constant-speed four-waypoint loop with tangent rounded corners."""

    def __init__(
        self,
        waypoints: list[tuple[float, float]],
        corner_radius_m: float,
        waypoint_tolerance_m: float,
    ) -> None:
        if len(waypoints) != 4:
            raise ValueError("rounded route requires exactly four waypoints")
        if not math.isfinite(corner_radius_m) or corner_radius_m <= 0.0:
            raise ValueError("corner radius must be finite and positive")
        if not math.isfinite(waypoint_tolerance_m) \
                or waypoint_tolerance_m <= 0.0:
            raise ValueError("waypoint tolerance must be finite and positive")

        directions: list[tuple[float, float]] = []
        lengths: list[float] = []
        for start, end in zip(waypoints, waypoints[1:] + waypoints[:1]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if (abs(dx) <= 1.0e-9) == (abs(dy) <= 1.0e-9):
                raise ValueError(
                    "rounded route must be an axis-aligned square"
                )
            length = math.hypot(dx, dy)
            lengths.append(length)
            directions.append((dx / length, dy / length))
        if any(length <= 2.0 * corner_radius_m for length in lengths):
            raise ValueError(
                "route side length must exceed twice the corner radius"
            )
        if abs(lengths[0] - lengths[2]) > 1.0e-6 \
                or abs(lengths[1] - lengths[3]) > 1.0e-6:
            raise ValueError("opposite route sides must be equal")

        corners: list[_RoundedCorner] = []
        for index, point in enumerate(waypoints):
            incoming = directions[(index - 1) % 4]
            outgoing = directions[index]
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            if cross <= 0.999999 or abs(dot) > 1.0e-6:
                raise ValueError(
                    "rounded route must use 90-degree counter-clockwise turns"
                )
            entry = (
                point[0] - corner_radius_m * incoming[0],
                point[1] - corner_radius_m * incoming[1],
            )
            exit_point = (
                point[0] + corner_radius_m * outgoing[0],
                point[1] + corner_radius_m * outgoing[1],
            )
            center = (
                entry[0] + corner_radius_m * outgoing[0],
                entry[1] + corner_radius_m * outgoing[1],
            )
            corners.append(
                _RoundedCorner(
                    entry=entry,
                    exit=exit_point,
                    center=center,
                    incoming=incoming,
                    outgoing=outgoing,
                )
            )
        self._corners = tuple(corners)
        self._corner_radius_m = corner_radius_m
        self._waypoint_tolerance_m = waypoint_tolerance_m
        # The trailer starts at waypoint A and first travels toward B.
        self._corner_index = 1
        self._on_arc = False
        self.completed_loops = 0
        self.reached_corners = 0

    def direction(self, x: float, y: float) -> tuple[float, float]:
        corner = self._corners[self._corner_index]
        if not self._on_arc:
            remaining = (
                (corner.entry[0] - x) * corner.incoming[0]
                + (corner.entry[1] - y) * corner.incoming[1]
            )
            if remaining > self._waypoint_tolerance_m:
                return corner.incoming
            self._on_arc = True
            print(
                f"corner_arc_entry={self._corner_index} "
                f"x={x:.3f} y={y:.3f}"
            )

        exit_progress = (
            (x - corner.exit[0]) * corner.outgoing[0]
            + (y - corner.exit[1]) * corner.outgoing[1]
        )
        if exit_progress >= 0.0:
            completed_corner = self._corner_index
            self.reached_corners += 1
            if completed_corner == 0:
                self.completed_loops += 1
            print(
                f"corner_arc_exit={completed_corner} "
                f"x={x:.3f} y={y:.3f}"
            )
            self._corner_index = (self._corner_index + 1) % 4
            self._on_arc = False
            return corner.outgoing

        radial_x = x - corner.center[0]
        radial_y = y - corner.center[1]
        radial_distance = math.hypot(radial_x, radial_y)
        if radial_distance <= 1.0e-9:
            return corner.incoming
        radial_unit_x = radial_x / radial_distance
        radial_unit_y = radial_y / radial_distance
        tangent_x = -radial_unit_y
        tangent_y = radial_unit_x
        radial_error_ratio = (
            radial_distance - self._corner_radius_m
        ) / self._corner_radius_m
        radial_correction = max(
            -0.20,
            min(0.20, -0.50 * radial_error_ratio),
        )
        direction_x = tangent_x + radial_correction * radial_unit_x
        direction_y = tangent_y + radial_correction * radial_unit_y
        norm = math.hypot(direction_x, direction_y)
        return direction_x / norm, direction_y / norm


class StopTurnWaypointRoute:
    """Straight waypoint patrol that brakes to a stop before each turn."""

    def __init__(
        self,
        waypoints: list[tuple[float, float]],
        waypoint_tolerance_m: float,
        turn_speed_tolerance_m_s: float,
        start_index: int = 0,
        stop_waypoint_indices: list[int] | None = None,
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("stop-turn route requires at least two waypoints")
        if not 0 <= start_index < len(waypoints):
            raise ValueError("stop-turn start index is outside the route")
        if not math.isfinite(waypoint_tolerance_m) \
                or waypoint_tolerance_m <= 0.0:
            raise ValueError("waypoint tolerance must be finite and positive")
        if not math.isfinite(turn_speed_tolerance_m_s) \
                or turn_speed_tolerance_m_s < 0.0:
            raise ValueError(
                "turn speed tolerance must be finite and non-negative"
            )
        self._waypoints = tuple(waypoints)
        self._waypoint_tolerance_m = waypoint_tolerance_m
        self._turn_speed_tolerance_m_s = turn_speed_tolerance_m_s
        if stop_waypoint_indices is None:
            stop_waypoint_indices = list(range(len(waypoints)))
        if any(not 0 <= index < len(waypoints) for index in stop_waypoint_indices):
            raise ValueError("stop waypoint index is outside the route")
        self._stop_waypoint_indices = frozenset(stop_waypoint_indices)
        self._start_index = start_index
        # The entity is spawned at start_index; the first target is the next
        # waypoint, so the spawn itself is not counted as a completed corner.
        self._target_index = (start_index + 1) % len(waypoints)
        self._stopping = False
        self._settled_zero_cycle = False
        self.completed_loops = 0
        self.reached_waypoints = 0

    @property
    def target_index(self) -> int:
        return self._target_index

    def command(
        self,
        x: float,
        y: float,
        current_speed_m_s: float,
        cruise_speed_m_s: float,
        acceleration_m_s2: float,
    ) -> tuple[float, float]:
        target_x, target_y = self._waypoints[self._target_index]
        dx, dy = target_x - x, target_y - y
        distance = math.hypot(dx, dy)

        if self._stopping:
            if current_speed_m_s > self._turn_speed_tolerance_m_s:
                return 0.0, 0.0
            if not self._settled_zero_cycle:
                self._settled_zero_cycle = True
                return 0.0, 0.0
            self._advance(x, y)
            target_x, target_y = self._waypoints[self._target_index]
            dx, dy = target_x - x, target_y - y
            distance = math.hypot(dx, dy)

        stop_here = self._target_index in self._stop_waypoint_indices
        start = self._waypoints[(self._target_index - 1) % len(self._waypoints)]
        leg_x = target_x - start[0]
        leg_y = target_y - start[1]
        leg_length = math.hypot(leg_x, leg_y)
        remaining_along = (
            dx * leg_x + dy * leg_y
        ) / max(leg_length, 1.0e-9)
        if distance <= self._waypoint_tolerance_m \
                or remaining_along <= self._waypoint_tolerance_m:
            if stop_here:
                self._stopping = True
                self._settled_zero_cycle = False
                return 0.0, 0.0
            self._advance(x, y)
            target_x, target_y = self._waypoints[self._target_index]
            dx, dy = target_x - x, target_y - y
            distance = math.hypot(dx, dy)
            stop_here = self._target_index in self._stop_waypoint_indices

        if distance <= 1.0e-9:
            return 0.0, 0.0
        target_speed = cruise_speed_m_s
        if stop_here:
            braking_distance = max(distance - self._waypoint_tolerance_m, 0.0)
            target_speed = min(
                cruise_speed_m_s,
                math.sqrt(2.0 * acceleration_m_s2 * braking_distance),
            )
        return target_speed * dx / distance, target_speed * dy / distance

    def _advance(self, x: float, y: float) -> None:
        completed_index = self._target_index
        self.reached_waypoints += 1
        self._target_index = (self._target_index + 1) % len(self._waypoints)
        if completed_index == self._start_index:
            self.completed_loops += 1
        self._stopping = False
        self._settled_zero_cycle = False
        print(
            f"waypoint_reached={completed_index} "
            f"x={x:.3f} y={y:.3f}"
        )


class LinearShuttleRoute:
    """Straight endpoint-to-endpoint motion with yaw-preserving reversal."""

    def __init__(
        self,
        endpoints: list[tuple[float, float]],
        declared_leg_length_m: float,
        creep_speed_m_s: float,
        brake_margin_m: float,
    ) -> None:
        if len(endpoints) != 2:
            raise ValueError("linear shuttle requires exactly two endpoints")
        if not all(
            math.isfinite(value)
            for endpoint in endpoints
            for value in endpoint
        ):
            raise ValueError("linear shuttle endpoints must be finite")
        delta_x = endpoints[1][0] - endpoints[0][0]
        delta_y = endpoints[1][1] - endpoints[0][1]
        measured_leg_length_m = math.hypot(delta_x, delta_y)
        if measured_leg_length_m <= 0.0:
            raise ValueError("linear shuttle endpoints must be distinct")
        if not math.isfinite(declared_leg_length_m) \
                or declared_leg_length_m <= 0.0:
            raise ValueError(
                "shuttle_leg_length_m must be finite and positive"
            )
        if not math.isclose(
            measured_leg_length_m,
            declared_leg_length_m,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                f"linear shuttle endpoint distance is "
                f"{measured_leg_length_m:.6f} m, declared "
                f"{declared_leg_length_m:.6f} m"
            )
        if not math.isfinite(creep_speed_m_s) or creep_speed_m_s <= 0.0:
            raise ValueError(
                "turnaround_creep_speed_m_s must be finite and positive"
            )
        if not math.isfinite(brake_margin_m) or brake_margin_m < 0.0:
            raise ValueError(
                "turnaround_brake_margin_m must be finite and non-negative"
            )

        self.start = endpoints[0]
        self.end = endpoints[1]
        self.leg_length_m = measured_leg_length_m
        self.unit_x = delta_x / measured_leg_length_m
        self.unit_y = delta_y / measured_leg_length_m
        self.normal_x = -self.unit_y
        self.normal_y = self.unit_x
        self.heading_rad = math.atan2(self.unit_y, self.unit_x)
        self.creep_speed_m_s = creep_speed_m_s
        self.brake_margin_m = brake_margin_m
        self._direction_sign = 1.0
        self.completed_legs = 0
        self.completed_loops = 0
        self.reached_endpoints = 0
        self.max_cross_track_error_m = 0.0
        self.max_endpoint_overshoot_m = 0.0

    def start_error_m(self, x: float, y: float) -> float:
        return math.hypot(x - self.start[0], y - self.start[1])

    def command(
        self,
        x: float,
        y: float,
        cruise_speed_m_s: float,
        acceleration_m_s2: float,
    ) -> tuple[float, float]:
        relative_x = x - self.start[0]
        relative_y = y - self.start[1]
        along_track_m = (
            relative_x * self.unit_x + relative_y * self.unit_y
        )
        cross_track_m = (
            relative_x * self.normal_x + relative_y * self.normal_y
        )
        self.max_cross_track_error_m = max(
            self.max_cross_track_error_m,
            abs(cross_track_m),
        )
        # Measure the true outside excursion independently of the active leg.
        # After an endpoint flip, the common acceleration limiter needs a few
        # samples to remove the old velocity, so the largest excursion can
        # occur after the first crossing sample.
        outside_excursion_m = max(
            -along_track_m,
            along_track_m - self.leg_length_m,
            0.0,
        )
        self.max_endpoint_overshoot_m = max(
            self.max_endpoint_overshoot_m,
            outside_excursion_m,
        )

        if self._direction_sign > 0.0:
            remaining_m = self.leg_length_m - along_track_m
            endpoint_name = "end"
        else:
            remaining_m = along_track_m
            endpoint_name = "start"

        if remaining_m <= 1.0e-9:
            overshoot_m = max(-remaining_m, 0.0)
            self.completed_legs += 1
            self.reached_endpoints += 1
            self.completed_loops = self.completed_legs // 2
            print(
                f"shuttle_endpoint_reached={endpoint_name} "
                f"x={x:.6f} y={y:.6f} "
                f"overshoot_m={overshoot_m:.6f} "
                f"completed_legs={self.completed_legs}"
            )
            self._direction_sign *= -1.0
            remaining_m = (
                along_track_m
                if self._direction_sign < 0.0
                else self.leg_length_m - along_track_m
            )

        braking_distance_m = max(
            remaining_m - self.brake_margin_m,
            0.0,
        )
        target_speed_m_s = max(
            self.creep_speed_m_s,
            min(
                cruise_speed_m_s,
                math.sqrt(
                    2.0 * acceleration_m_s2 * braking_distance_m
                ),
            ),
        )
        return (
            self._direction_sign * target_speed_m_s * self.unit_x,
            self._direction_sign * target_speed_m_s * self.unit_y,
        )


class CircularRoute:
    """Constant-speed circular route without waypoint corner transients."""

    def __init__(
        self,
        center: tuple[float, float],
        radius_m: float,
        direction: str,
    ) -> None:
        if not all(math.isfinite(value) for value in center):
            raise ValueError("circle center must be finite")
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("circle radius must be finite and positive")
        normalized_direction = direction.strip().lower().replace("-", "_")
        if normalized_direction not in ("counterclockwise", "clockwise"):
            raise ValueError(
                "circle direction must be counterclockwise or clockwise"
            )
        self.center = center
        self.radius_m = radius_m
        self._direction_sign = (
            1.0 if normalized_direction == "counterclockwise" else -1.0
        )
        self._previous_angle: float | None = None
        self._angle_travelled = 0.0
        self.completed_loops = 0

    def direction(self, x: float, y: float) -> tuple[float, float]:
        radial_x = x - self.center[0]
        radial_y = y - self.center[1]
        radial_distance = math.hypot(radial_x, radial_y)
        if radial_distance <= 1.0e-9:
            raise ValueError("trailer cannot be at the circle center")

        angle = math.atan2(radial_y, radial_x)
        if self._previous_angle is not None:
            angle_step = math.atan2(
                math.sin(angle - self._previous_angle),
                math.cos(angle - self._previous_angle),
            )
            forward_step = self._direction_sign * angle_step
            # Ignore tiny reverse motion caused by Gazebo pose jitter.
            if forward_step > 0.0:
                self._angle_travelled += forward_step
                self.completed_loops = int(
                    self._angle_travelled / (2.0 * math.pi)
                )
        self._previous_angle = angle

        radial_unit_x = radial_x / radial_distance
        radial_unit_y = radial_y / radial_distance
        tangent_x = -self._direction_sign * radial_unit_y
        tangent_y = self._direction_sign * radial_unit_x

        # The tangent field is the actual route.  A small smooth radial term
        # only cancels numerical / physics drift without creating waypoints.
        radial_error_ratio = (
            radial_distance - self.radius_m
        ) / self.radius_m
        radial_correction = max(
            -0.20,
            min(0.20, -0.50 * radial_error_ratio),
        )
        direction_x = tangent_x + radial_correction * radial_unit_x
        direction_y = tangent_y + radial_correction * radial_unit_y
        norm = math.hypot(direction_x, direction_y)
        return direction_x / norm, direction_y / norm


class StadiumRoute:
    """Constant-speed capsule-shaped stadium centerline.

    The local +x axis follows ``heading_deg`` in the ENU plane.  A
    counter-clockwise lap travels along the local y=-radius straight toward
    +x, around the +x semicircle, back along y=+radius, and around the -x
    semicircle.  The returned vector is tangent to the nearest centerline
    point with a small cross-track correction.
    """

    def __init__(
        self,
        center: tuple[float, float],
        straight_length_m: float,
        curve_radius_m: float,
        heading_deg: float,
        direction: str,
    ) -> None:
        if not all(math.isfinite(value) for value in center):
            raise ValueError("stadium center must be finite")
        if not math.isfinite(straight_length_m) or straight_length_m <= 0.0:
            raise ValueError(
                "stadium straight length must be finite and positive"
            )
        if not math.isfinite(curve_radius_m) or curve_radius_m <= 0.0:
            raise ValueError(
                "stadium curve radius must be finite and positive"
            )
        if not math.isfinite(heading_deg):
            raise ValueError("stadium heading must be finite")
        normalized_direction = direction.strip().lower().replace("-", "_")
        if normalized_direction not in ("counterclockwise", "clockwise"):
            raise ValueError(
                "stadium direction must be counterclockwise or clockwise"
            )

        self.center = center
        self.straight_length_m = straight_length_m
        self.curve_radius_m = curve_radius_m
        self.heading_deg = heading_deg
        self.perimeter_m = (
            2.0 * self.straight_length_m
            + 2.0 * math.pi * self.curve_radius_m
        )
        self._half_straight_m = 0.5 * self.straight_length_m
        heading_rad = math.radians(heading_deg)
        self._cos_heading = math.cos(heading_rad)
        self._sin_heading = math.sin(heading_rad)
        self._direction_sign = (
            1.0 if normalized_direction == "counterclockwise" else -1.0
        )
        self._previous_progress_m: float | None = None
        self._distance_travelled_m = 0.0
        self.completed_loops = 0
        self.max_cross_track_error_m = 0.0

    def _to_local(self, x: float, y: float) -> tuple[float, float]:
        dx = x - self.center[0]
        dy = y - self.center[1]
        return (
            self._cos_heading * dx + self._sin_heading * dy,
            -self._sin_heading * dx + self._cos_heading * dy,
        )

    def _vector_to_world(
        self, local_x: float, local_y: float
    ) -> tuple[float, float]:
        return (
            self._cos_heading * local_x - self._sin_heading * local_y,
            self._sin_heading * local_x + self._cos_heading * local_y,
        )

    def _project_local(
        self, x: float, y: float
    ) -> tuple[float, float, float, float, float]:
        """Return nearest point, CCW tangent, and CCW path progress."""

        half = self._half_straight_m
        radius = self.curve_radius_m
        straight = self.straight_length_m

        if x > half:
            radial_x = x - half
            radial_y = y
            radial_distance = math.hypot(radial_x, radial_y)
            if radial_distance <= 1.0e-9:
                unit_x, unit_y = 1.0, 0.0
            else:
                unit_x = radial_x / radial_distance
                unit_y = radial_y / radial_distance
            nearest_x = half + radius * unit_x
            nearest_y = radius * unit_y
            tangent_x, tangent_y = -unit_y, unit_x
            angle = math.atan2(unit_y, unit_x)
            progress_m = straight + radius * (angle + math.pi / 2.0)
        elif x < -half:
            radial_x = x + half
            radial_y = y
            radial_distance = math.hypot(radial_x, radial_y)
            if radial_distance <= 1.0e-9:
                unit_x, unit_y = -1.0, 0.0
            else:
                unit_x = radial_x / radial_distance
                unit_y = radial_y / radial_distance
            nearest_x = -half + radius * unit_x
            nearest_y = radius * unit_y
            tangent_x, tangent_y = -unit_y, unit_x
            angle = math.atan2(unit_y, unit_x)
            if angle < math.pi / 2.0:
                angle += 2.0 * math.pi
            progress_m = (
                2.0 * straight
                + math.pi * radius
                + radius * (angle - math.pi / 2.0)
            )
        elif y <= 0.0:
            nearest_x, nearest_y = x, -radius
            tangent_x, tangent_y = 1.0, 0.0
            progress_m = x + half
        else:
            nearest_x, nearest_y = x, radius
            tangent_x, tangent_y = -1.0, 0.0
            progress_m = (
                straight + math.pi * radius + (half - x)
            )

        return (
            nearest_x,
            nearest_y,
            tangent_x,
            tangent_y,
            progress_m % self.perimeter_m,
        )

    def direction(self, x: float, y: float) -> tuple[float, float]:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("trailer stadium position must be finite")
        local_x, local_y = self._to_local(x, y)
        (
            nearest_x,
            nearest_y,
            tangent_x,
            tangent_y,
            ccw_progress_m,
        ) = self._project_local(local_x, local_y)

        progress_m = (
            ccw_progress_m
            if self._direction_sign > 0.0
            else (-ccw_progress_m) % self.perimeter_m
        )
        if self._previous_progress_m is not None:
            progress_step_m = (
                progress_m
                - self._previous_progress_m
                + 0.5 * self.perimeter_m
            ) % self.perimeter_m - 0.5 * self.perimeter_m
            # Ignore small reverse motion caused by pose / control jitter.
            if progress_step_m > 0.0:
                self._distance_travelled_m += progress_step_m
                self.completed_loops = int(
                    self._distance_travelled_m / self.perimeter_m
                )
        self._previous_progress_m = progress_m

        tangent_x *= self._direction_sign
        tangent_y *= self._direction_sign
        correction_x = nearest_x - local_x
        correction_y = nearest_y - local_y
        correction_distance = math.hypot(correction_x, correction_y)
        self.max_cross_track_error_m = max(
            self.max_cross_track_error_m, correction_distance
        )
        if correction_distance > 1.0e-9:
            correction_strength = min(
                0.20,
                0.50 * correction_distance / self.curve_radius_m,
            )
            tangent_x += correction_strength * correction_x / correction_distance
            tangent_y += correction_strength * correction_y / correction_distance
        norm = math.hypot(tangent_x, tangent_y)
        return self._vector_to_world(tangent_x / norm, tangent_y / norm)


class Terrain:
    def __init__(self, document: dict[str, object]):
        terrain = document["terrain"]
        assert isinstance(terrain, dict)
        self.flat = (
            str(document["map"]["name"]).startswith("city")  # type: ignore[index]
            or str(terrain.get("type", "")) == "flat_box"
        )
        self.pixels: np.ndarray | None = None
        if not self.flat:
            image_path = REPO / str(terrain["image"])
            self.pixels = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
            self.half_size = 150.0
            self.height_scale = 40.0

    def height(self, x: float, y: float) -> float:
        if self.flat:
            return 0.0
        assert self.pixels is not None
        last = self.pixels.shape[0] - 1
        column = min(max((x + self.half_size) / (2.0 * self.half_size) * last, 0.0), last)
        row = min(max((self.half_size - y) / (2.0 * self.half_size) * last, 0.0), last)
        c0, r0 = int(math.floor(column)), int(math.floor(row))
        c1, r1 = min(c0 + 1, last), min(r0 + 1, last)
        tx, ty = column - c0, row - r0
        value = (
            (1.0 - tx) * (1.0 - ty) * self.pixels[r0, c0]
            + tx * (1.0 - ty) * self.pixels[r0, c1]
            + (1.0 - tx) * ty * self.pixels[r1, c0]
            + tx * ty * self.pixels[r1, c1]
        )
        return float(value / 255.0 * self.height_scale)

    def safe_root_z(self, x: float, y: float) -> float:
        # The body bottom is model-root + 0.05 m.  Sampling a 7x7, 6x6 m
        # envelope gives 0.5 m look-ahead around the physical 5x5 footprint.
        highest = max(
            self.height(x + dx, y + dy)
            for dx in np.linspace(-3.0, 3.0, 7)
            for dy in np.linspace(-3.0, 3.0, 7)
        )
        return max(0.0, highest + 0.08 - 0.05)


def publish_velocity(publisher: object, vx: float, vy: float, vz: float) -> None:
    message = Twist()
    message.linear.x = vx
    message.linear.y = vy
    message.linear.z = vz
    message.angular.x = 0.0
    message.angular.y = 0.0
    message.angular.z = 0.0
    publisher.publish(message)  # type: ignore[attr-defined]


def world_to_model_xy(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    """Express a horizontal world velocity in the model's fixed body axes."""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return cosine * vx + sine * vy, -sine * vx + cosine * vy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "map",
        choices=("city", "mountain", "mpc-landing-moving", "cju-track"),
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        help="coordinate YAML selected by the parent launcher",
    )
    parser.add_argument("--loops", type=int, default=0,
                        help="route repetitions; 0 repeats until interrupted")
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="wall-clock timeout in seconds; 0 disables")
    parser.add_argument(
        "--rate", type=float,
        help="command rate; defaults to trailer.command_rate_hz or 20 Hz",
    )
    parser.add_argument("--speed", type=float, default=0.0,
                        help="override the YAML cruise speed; 0 uses YAML")
    parser.add_argument("--start-index", type=int, default=0,
                        help="resume a validation route at this waypoint index")
    parser.add_argument("--route", choices=("flat", "slope"), default="flat",
                        help="flat operational route or mountain slope safeguard test")
    parser.add_argument("--no-terrain-follow", action="store_true")
    args = parser.parse_args()
    if args.loops < 0:
        parser.error("--loops must be non-negative")
    if args.timeout < 0.0:
        parser.error("--timeout must be non-negative")
    if args.rate is not None and args.rate <= 0.0:
        parser.error("--rate must be positive")

    coordinate_path = (
        args.coordinates.expanduser().resolve()
        if args.coordinates is not None
        else GAZEBO / "maps" / f"{args.map}_coordinates.yaml"
    )
    if not coordinate_path.is_file():
        parser.error(f"coordinate map does not exist: {coordinate_path}")
    document = yaml.safe_load(coordinate_path.read_text(encoding="utf-8"))
    trailer = document["trailer"]
    entity = str(trailer["entity_name"])
    topic = str(trailer["command_topic"])
    pose_topic = str(trailer["pose_topic"])
    route_type = str(trailer.get("route_type", "waypoints")).strip().lower()
    shuttle_route = None
    shuttle_spawn_tolerance_m = 0.0
    shuttle_yaw_tolerance_rad = 0.0
    shuttle_cross_track_limit_m = 0.0
    shuttle_endpoint_excursion_limit_m = 0.0
    shuttle_min_cruise_ratio = 0.0
    circle_route = None
    stadium_route = None
    waypoints: list[tuple[float, float]] = []
    if route_type == "linear_shuttle":
        if args.route != "flat":
            parser.error(
                "linear shuttle is only available for the flat route"
            )
        if args.start_index != 0:
            parser.error("linear shuttle requires --start-index 0")
        try:
            endpoints = [
                (float(point[0]), float(point[1]))
                for point in trailer["shuttle_endpoints_enu_m"]
            ]
            shuttle_route = LinearShuttleRoute(
                endpoints,
                float(trailer["shuttle_leg_length_m"]),
                float(trailer["turnaround_creep_speed_m_s"]),
                float(trailer["turnaround_brake_margin_m"]),
            )
            declared_cycle_length_m = float(trailer["route_length_m"])
            if not math.isclose(
                declared_cycle_length_m,
                2.0 * shuttle_route.leg_length_m,
                abs_tol=1.0e-6,
            ):
                raise ValueError(
                    f"route_length_m is {declared_cycle_length_m:.6f} m, "
                    f"expected {2.0 * shuttle_route.leg_length_m:.6f} m"
                )
            shuttle_spawn_tolerance_m = float(
                trailer["shuttle_spawn_tolerance_m"]
            )
            shuttle_yaw_tolerance_rad = math.radians(
                float(trailer["shuttle_yaw_tolerance_deg"])
            )
            shuttle_cross_track_limit_m = float(
                trailer["shuttle_max_cross_track_error_m"]
            )
            shuttle_endpoint_excursion_limit_m = float(
                trailer["shuttle_max_endpoint_excursion_m"]
            )
            shuttle_min_cruise_ratio = float(
                trailer["shuttle_min_cruise_ratio"]
            )
            positive_limits = (
                shuttle_spawn_tolerance_m,
                shuttle_yaw_tolerance_rad,
                shuttle_cross_track_limit_m,
                shuttle_endpoint_excursion_limit_m,
            )
            if not all(
                math.isfinite(value) and value > 0.0
                for value in positive_limits
            ):
                raise ValueError(
                    "linear shuttle validation tolerances must be finite "
                    "and positive"
                )
            if not (
                math.isfinite(shuttle_min_cruise_ratio)
                and 0.0 < shuttle_min_cruise_ratio <= 1.0
            ):
                raise ValueError(
                    "shuttle_min_cruise_ratio must be in (0, 1]"
                )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            parser.error(f"invalid linear shuttle route: {exc}")
    elif route_type == "circle":
        if args.route != "flat":
            parser.error("circle route is only available for the flat route")
        if args.start_index != 0:
            parser.error("circle route requires --start-index 0")
        try:
            center_values = trailer["circle_center_enu_m"]
            center = (float(center_values[0]), float(center_values[1]))
            circle_route = CircularRoute(
                center,
                float(trailer["circle_radius_m"]),
                str(trailer.get("circle_direction", "counterclockwise")),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            parser.error(f"invalid circle route: {exc}")
    elif route_type == "stadium":
        if args.route != "flat":
            parser.error("stadium route is only available for the flat route")
        if args.start_index != 0:
            parser.error("stadium route requires --start-index 0")
        try:
            center_values = trailer["stadium_center_enu_m"]
            center = (float(center_values[0]), float(center_values[1]))
            stadium_route = StadiumRoute(
                center,
                float(trailer["stadium_straight_length_m"]),
                float(trailer["stadium_curve_radius_m"]),
                float(trailer["stadium_heading_deg"]),
                str(trailer.get(
                    "stadium_direction", "counterclockwise"
                )),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            parser.error(f"invalid stadium route: {exc}")
    elif route_type == "waypoints":
        route_key = (
            "slope_validation_waypoints_enu_m"
            if args.route == "slope" else "waypoints_enu_m"
        )
        if route_key not in trailer:
            parser.error(
                f"the {args.map} map does not define a {args.route} route"
            )
        waypoints = [
            (float(point[0]), float(point[1]))
            for point in trailer[route_key]
        ]
        if not 0 <= args.start_index < len(waypoints):
            parser.error(f"--start-index must be in 0..{len(waypoints) - 1}")
    else:
        parser.error(f"unsupported trailer route_type: {route_type}")
    speed = args.speed if args.speed > 0.0 else float(trailer["cruise_speed_m_s"])
    command_rate_hz = (
        float(args.rate)
        if args.rate is not None
        else float(trailer.get("command_rate_hz", 20.0))
    )
    tolerance = float(trailer["waypoint_tolerance_m"])
    acceleration = float(trailer.get("acceleration_m_s2", 1.0))
    try:
        simulation_speed_factor = float(
            os.environ.get("PX4_SIM_SPEED_FACTOR", "1.0")
        )
    except ValueError:
        parser.error("PX4_SIM_SPEED_FACTOR must be numeric")
    if not math.isfinite(speed) or speed <= 0.0:
        parser.error("trailer cruise speed must be finite and positive")
    if not math.isfinite(command_rate_hz) or command_rate_hz <= 0.0:
        parser.error("trailer command rate must be finite and positive")
    if not math.isfinite(acceleration) or acceleration <= 0.0:
        parser.error("trailer acceleration must be finite and positive")
    if (not math.isfinite(simulation_speed_factor)
            or simulation_speed_factor <= 0.0):
        parser.error("PX4_SIM_SPEED_FACTOR must be finite and positive")
    if shuttle_route is not None \
            and shuttle_route.creep_speed_m_s > speed:
        parser.error(
            "turnaround_creep_speed_m_s must not exceed cruise speed"
        )
    rounded_route = None
    if route_type == "waypoints" and "corner_radius_m" in trailer:
        if args.start_index != 0:
            parser.error("rounded square route requires --start-index 0")
        try:
            rounded_route = RoundedSquareRoute(
                waypoints,
                float(trailer["corner_radius_m"]),
                tolerance,
            )
        except ValueError as exc:
            parser.error(str(exc))
    stop_turn_route = None
    if route_type == "waypoints" and bool(trailer.get("stop_at_waypoints", False)):
        if rounded_route is not None:
            parser.error("stop-turn and rounded routes are mutually exclusive")
        try:
            stop_turn_route = StopTurnWaypointRoute(
                waypoints,
                tolerance,
                float(trailer.get("turn_speed_tolerance_m_s", 0.20)),
                args.start_index,
                [int(index) for index in trailer.get(
                    "stop_waypoint_indices",
                    range(len(waypoints)),
                )],
            )
        except ValueError as exc:
            parser.error(str(exc))
    terrain = Terrain(document)

    node = Node()
    publisher = node.advertise(topic, Twist)
    receiver = PoseReceiver(entity)
    node.subscribe(Pose_V, pose_topic, receiver.callback)
    for handled in (signal.SIGINT, signal.SIGTERM):
        signal.signal(handled, lambda *_: STOP.set())

    print(
        f"trailer_map={args.map} route={args.route} "
        f"route_type={route_type} entity={entity} waypoints={len(waypoints)}"
    )
    if shuttle_route is not None:
        print(
            f"shuttle_start=({shuttle_route.start[0]:.6f},"
            f"{shuttle_route.start[1]:.6f}) "
            f"shuttle_end=({shuttle_route.end[0]:.6f},"
            f"{shuttle_route.end[1]:.6f}) "
            f"leg_length_m={shuttle_route.leg_length_m:.3f} "
            f"cycle_length_m={2.0 * shuttle_route.leg_length_m:.3f}"
        )
    if circle_route is not None:
        print(
            f"circle_center=({circle_route.center[0]:.9f},"
            f"{circle_route.center[1]:.9f}) "
            f"circle_radius_m={circle_route.radius_m:.3f}"
        )
    if stadium_route is not None:
        print(
            f"stadium_center=({stadium_route.center[0]:.9f},"
            f"{stadium_route.center[1]:.9f}) "
            f"stadium_straight_length_m="
            f"{stadium_route.straight_length_m:.3f} "
            f"stadium_curve_radius_m={stadium_route.curve_radius_m:.3f} "
            f"stadium_heading_deg={stadium_route.heading_deg:.3f} "
            f"stadium_perimeter_m={stadium_route.perimeter_m:.3f}"
        )
    print(
        f"command_topic={topic} pose_topic={pose_topic} "
        f"command_rate_hz={command_rate_hz:.3f} "
        f"simulation_speed_factor={simulation_speed_factor:.3f}"
    )
    start = time.monotonic()
    while receiver.get() is None and not STOP.is_set():
        if time.monotonic() - start > 15.0:
            print(f"ERROR: no pose received for {entity}", file=sys.stderr)
            return 2
        time.sleep(0.05)
    initial_pose = receiver.get()
    if initial_pose is None:
        print("TRAILER WAYPOINT VALIDATION: INTERRUPTED", file=sys.stderr)
        return 130
    initial_yaw = initial_pose[5]
    if shuttle_route is not None:
        start_error_m = shuttle_route.start_error_m(
            initial_pose[0],
            initial_pose[1],
        )
        if start_error_m > shuttle_spawn_tolerance_m:
            print(
                "ERROR: linear shuttle spawn differs from its start by "
                f"{start_error_m:.6f} m "
                f"(tolerance {shuttle_spawn_tolerance_m:.6f} m)",
                file=sys.stderr,
            )
            return 2
        initial_heading_error_rad = math.atan2(
            math.sin(initial_yaw - shuttle_route.heading_rad),
            math.cos(initial_yaw - shuttle_route.heading_rad),
        )
        if abs(initial_heading_error_rad) > shuttle_yaw_tolerance_rad:
            print(
                "ERROR: linear shuttle initial yaw differs from route "
                f"heading by {math.degrees(initial_heading_error_rad):.6f} "
                "deg (tolerance "
                f"{math.degrees(shuttle_yaw_tolerance_rad):.6f} deg)",
                file=sys.stderr,
            )
            return 2

    index = args.start_index
    completed_loops = 0
    reached = 0
    previous_velocity = np.zeros(3, dtype=np.float64)
    previous_time = time.monotonic()
    previous_sim_time = receiver.simulation_time_s()
    max_abs_z_error = max_abs_roll = max_abs_pitch = 0.0
    max_abs_yaw_error = 0.0
    max_forward_along_speed_m_s = 0.0
    max_reverse_along_speed_m_s = 0.0
    tracking_error_samples = 0
    excessive_tilt_samples = 0
    period = 1.0 / command_rate_hz
    # Poll twice as fast as the requested simulation-time command rate.  If
    # polling and the pose publisher share the same period, scheduler phase can
    # miss a just-late stamp and accidentally degrade 50 Hz control to 25 Hz.
    poll_period = period / (2.0 * max(1.0, simulation_speed_factor))

    try:
        while not STOP.is_set():
            now = time.monotonic()
            if args.timeout > 0.0 and now - start >= args.timeout:
                print("ERROR: trailer route timed out", file=sys.stderr)
                return 3
            pose = receiver.get()
            if pose is None:
                time.sleep(poll_period)
                continue
            sim_time = receiver.simulation_time_s()
            sim_elapsed = None
            if sim_time is not None and previous_sim_time is not None:
                if sim_time < previous_sim_time:
                    previous_sim_time = sim_time
                else:
                    sim_elapsed = sim_time - previous_sim_time
                    # Publish at the YAML rate in simulation time even when
                    # Gazebo runs faster (or cannot sustain the requested RTF).
                    if sim_elapsed + 1.0e-9 < period:
                        time.sleep(poll_period)
                        continue
            x, y, z, roll, pitch, yaw = pose
            pose_fresh = receiver.age_s(now) <= max(0.10, 2.5 * period)
            if not pose_fresh:
                desired_vx = desired_vy = 0.0
            elif shuttle_route is not None:
                desired_vx, desired_vy = shuttle_route.command(
                    x,
                    y,
                    speed,
                    acceleration,
                )
                reached = shuttle_route.reached_endpoints
                completed_loops = shuttle_route.completed_loops
                if args.loops > 0 and completed_loops >= args.loops:
                    break
            elif stadium_route is not None:
                direction_x, direction_y = stadium_route.direction(x, y)
                completed_loops = stadium_route.completed_loops
                if args.loops > 0 and completed_loops >= args.loops:
                    break
                desired_vx = speed * direction_x
                desired_vy = speed * direction_y
            elif circle_route is not None:
                direction_x, direction_y = circle_route.direction(x, y)
                completed_loops = circle_route.completed_loops
                if args.loops > 0 and completed_loops >= args.loops:
                    break
                desired_vx = speed * direction_x
                desired_vy = speed * direction_y
            elif rounded_route is not None:
                direction_x, direction_y = rounded_route.direction(x, y)
                reached = rounded_route.reached_corners
                completed_loops = rounded_route.completed_loops
                if args.loops > 0 and completed_loops >= args.loops:
                    break
                desired_vx = speed * direction_x
                desired_vy = speed * direction_y
            elif stop_turn_route is not None:
                measured_speed = receiver.speed_xy()
                desired_vx, desired_vy = stop_turn_route.command(
                    x,
                    y,
                    conservative_stop_speed(
                        measured_speed,
                        float(np.linalg.norm(previous_velocity[:2])),
                    ),
                    speed,
                    acceleration,
                )
                reached = stop_turn_route.reached_waypoints
                completed_loops = stop_turn_route.completed_loops
                if args.loops > 0 and completed_loops >= args.loops:
                    break
            else:
                target_x, target_y = waypoints[index]
                dx, dy = target_x - x, target_y - y
                distance = math.hypot(dx, dy)
                if distance <= tolerance:
                    reached += 1
                    print(
                        f"waypoint_reached={index} "
                        f"x={x:.3f} y={y:.3f} z={z:.3f}"
                    )
                    index += 1
                    if index == len(waypoints):
                        completed_loops += 1
                        if args.loops > 0 and completed_loops >= args.loops:
                            break
                        index = 0
                    target_x, target_y = waypoints[index]
                    dx, dy = target_x - x, target_y - y
                    distance = math.hypot(dx, dy)

                desired_xy_speed = min(speed, 1.25 * distance)
                if distance > 1e-9:
                    desired_vx = desired_xy_speed * dx / distance
                    desired_vy = desired_xy_speed * dy / distance
                else:
                    desired_vx = desired_vy = 0.0
            expected_z = (
                terrain.safe_root_z(x, y)
                if not args.no_terrain_follow else float(trailer["spawn_pose_enu"]["z"])
            )
            desired_vz = max(-2.5, min(2.5, 5.0 * (expected_z - z)))

            elapsed = (
                sim_elapsed
                if sim_elapsed is not None and sim_elapsed > 0.0
                else (now - previous_time) * simulation_speed_factor
            )
            dt = min(max(elapsed, 1e-3), 2.0 * period)
            desired = np.array([desired_vx, desired_vy, desired_vz], dtype=np.float64)
            delta = desired - previous_velocity
            if stop_turn_route is not None:
                horizontal_norm = float(np.linalg.norm(delta[:2]))
                horizontal_limit = acceleration * dt
                if horizontal_norm > horizontal_limit:
                    delta[:2] *= horizontal_limit / horizontal_norm
                delta[2] = float(np.clip(delta[2], -3.0 * dt, 3.0 * dt))
            else:
                max_delta = np.array(
                    [acceleration, acceleration, 3.0], dtype=np.float64
                ) * dt
                delta = np.clip(delta, -max_delta, max_delta)
            command = previous_velocity + delta
            if shuttle_route is not None:
                along_speed_m_s = (
                    float(command[0]) * shuttle_route.unit_x
                    + float(command[1]) * shuttle_route.unit_y
                )
                max_forward_along_speed_m_s = max(
                    max_forward_along_speed_m_s,
                    along_speed_m_s,
                )
                max_reverse_along_speed_m_s = max(
                    max_reverse_along_speed_m_s,
                    -along_speed_m_s,
                )
            body_vx, body_vy = world_to_model_xy(
                float(command[0]), float(command[1]), yaw
            )
            publish_velocity(
                publisher, body_vx, body_vy, float(command[2])
            )
            previous_velocity = command
            previous_time = now
            previous_sim_time = sim_time

            z_error = abs(z - expected_z)
            max_abs_z_error = max(max_abs_z_error, z_error)
            max_abs_roll = max(max_abs_roll, abs(roll))
            max_abs_pitch = max(max_abs_pitch, abs(pitch))
            yaw_error = math.atan2(
                math.sin(yaw - initial_yaw),
                math.cos(yaw - initial_yaw),
            )
            max_abs_yaw_error = max(
                max_abs_yaw_error,
                abs(yaw_error),
            )
            if z_error > 0.10:
                tracking_error_samples += 1
            if abs(roll) > math.radians(3.0) or abs(pitch) > math.radians(3.0):
                excessive_tilt_samples += 1
            time.sleep(max(0.0, poll_period - (time.monotonic() - now)))
    finally:
        for _ in range(5):
            publish_velocity(publisher, 0.0, 0.0, 0.0)
            time.sleep(0.02 / max(1.0, simulation_speed_factor))

    route_complete = args.loops > 0 and completed_loops >= args.loops
    if STOP.is_set() and not route_complete:
        print("TRAILER WAYPOINT VALIDATION: INTERRUPTED", file=sys.stderr)
        print(f"completed_loops={completed_loops} reached_waypoints={reached}")
        if stadium_route is not None:
            print(
                "max_cross_track_error_m="
                f"{stadium_route.max_cross_track_error_m:.6f}"
            )
        return 130

    z_limit = 0.40 if args.route == "slope" else 0.10
    stable = (
        max_abs_z_error <= z_limit
        and max_abs_roll <= math.radians(3.0)
        and max_abs_pitch <= math.radians(3.0)
        and excessive_tilt_samples == 0
        and (
            shuttle_route is None
            or (
                max_abs_yaw_error <= shuttle_yaw_tolerance_rad
                and shuttle_route.max_cross_track_error_m
                <= shuttle_cross_track_limit_m
                and shuttle_route.max_endpoint_overshoot_m
                <= shuttle_endpoint_excursion_limit_m
                and max_forward_along_speed_m_s
                >= speed * shuttle_min_cruise_ratio
                and max_reverse_along_speed_m_s
                >= speed * shuttle_min_cruise_ratio
            )
        )
    )
    print(f"TRAILER WAYPOINT VALIDATION: {'PASS' if stable else 'FAIL'}")
    print(f"completed_loops={completed_loops} reached_waypoints={reached}")
    print(f"max_abs_z_error_m={max_abs_z_error:.6f}")
    print(f"max_abs_roll_deg={math.degrees(max_abs_roll):.6f}")
    print(f"max_abs_pitch_deg={math.degrees(max_abs_pitch):.6f}")
    print(f"max_abs_yaw_error_deg={math.degrees(max_abs_yaw_error):.6f}")
    if shuttle_route is not None:
        print(f"completed_legs={shuttle_route.completed_legs}")
        print(
            "max_cross_track_error_m="
            f"{shuttle_route.max_cross_track_error_m:.6f}"
        )
        print(
            "max_endpoint_overshoot_m="
            f"{shuttle_route.max_endpoint_overshoot_m:.6f}"
        )
        print(
            "max_forward_along_speed_m_s="
            f"{max_forward_along_speed_m_s:.6f}"
        )
        print(
            "max_reverse_along_speed_m_s="
            f"{max_reverse_along_speed_m_s:.6f}"
        )
    if stadium_route is not None:
        print(
            "max_cross_track_error_m="
            f"{stadium_route.max_cross_track_error_m:.6f}"
        )
    print(f"terrain_tracking_error_samples={tracking_error_samples}")
    print(f"excessive_tilt_samples={excessive_tilt_samples}")
    print(f"impulse_launch_detected={'NO' if stable else 'YES'}")
    return 0 if stable else 4


if __name__ == "__main__":
    raise SystemExit(main())
