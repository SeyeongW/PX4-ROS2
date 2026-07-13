"""Jerk-limited motion profiles for the moving landing trailer.

The profile is a symmetric seven-phase S-curve.  Jerk, acceleration and speed
are bounded analytically; the endpoint is reached with zero velocity and zero
acceleration, then held.  This module deliberately has no ROS dependencies so
the route contract can be unit-tested without Gazebo or a running ROS graph.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TrailerRouteSample:
    """One world-ENU reference sample from :class:`JerkLimitedTrailerRoute`."""

    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray
    acceleration_enu_m_s2: np.ndarray
    jerk_enu_m_s3: np.ndarray
    distance_m: float
    yaw_rad: float
    phase: str
    complete: bool


class JerkLimitedTrailerRoute:
    """Time-optimal, jerk-limited 1-D S-curve embedded in world ENU.

    The default route implements the mission contract exactly: the trailer
    starts at ``(200, -125)`` and travels to ``(-128, -128)``.  A requested
    3 m/s cruise speed is retained whenever the route is long enough.  Shorter
    routes automatically use a lower peak speed instead of violating the
    acceleration or jerk bounds.
    """

    DEFAULT_START_ENU_M = (200.0, -125.0, 0.0)
    DEFAULT_GOAL_ENU_M = (-128.0, -128.0, 0.0)

    def __init__(
        self,
        start_enu_m: Sequence[float] = DEFAULT_START_ENU_M,
        goal_enu_m: Sequence[float] = DEFAULT_GOAL_ENU_M,
        cruise_speed_m_s: float = 3.0,
        max_acceleration_m_s2: float = 1.0,
        max_jerk_m_s3: float = 0.5,
    ) -> None:
        self.start_enu_m = self._vector3("start_enu_m", start_enu_m)
        self.goal_enu_m = self._vector3("goal_enu_m", goal_enu_m)
        self.cruise_speed_m_s = self._positive(
            "cruise_speed_m_s", cruise_speed_m_s
        )
        self.max_acceleration_m_s2 = self._positive(
            "max_acceleration_m_s2", max_acceleration_m_s2
        )
        self.max_jerk_m_s3 = self._positive("max_jerk_m_s3", max_jerk_m_s3)

        delta = self.goal_enu_m - self.start_enu_m
        self.distance_m = float(np.linalg.norm(delta))
        if self.distance_m <= 1e-9:
            self.direction_enu = np.zeros(3, dtype=np.float64)
            self.yaw_rad = 0.0
            self.peak_speed_m_s = 0.0
            self.peak_acceleration_m_s2 = 0.0
            self.jerk_ramp_time_s = 0.0
            self.constant_acceleration_time_s = 0.0
            self.cruise_time_s = 0.0
            self._phases: tuple[tuple[str, float, float], ...] = ()
            self.duration_s = 0.0
            return

        self.direction_enu = delta / self.distance_m
        self.yaw_rad = math.atan2(self.direction_enu[1], self.direction_enu[0])
        self._build_profile()

    @staticmethod
    def _vector3(name: str, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain three finite values")
        return array

    @staticmethod
    def _positive(name: str, value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    def _acceleration_shape(self, speed_m_s: float) -> tuple[float, float, float]:
        """Return jerk-ramp time, constant-accel time and attained accel."""

        acceleration_threshold_speed = (
            self.max_acceleration_m_s2**2 / self.max_jerk_m_s3
        )
        if speed_m_s <= acceleration_threshold_speed:
            ramp = math.sqrt(speed_m_s / self.max_jerk_m_s3)
            return ramp, 0.0, self.max_jerk_m_s3 * ramp
        ramp = self.max_acceleration_m_s2 / self.max_jerk_m_s3
        constant = speed_m_s / self.max_acceleration_m_s2 - ramp
        return ramp, constant, self.max_acceleration_m_s2

    def _minimum_distance_for_speed(self, speed_m_s: float) -> float:
        ramp, constant, _ = self._acceleration_shape(speed_m_s)
        # Acceleration and deceleration are point-symmetric.  Their combined
        # distance is peak_speed * acceleration_duration.
        return speed_m_s * (2.0 * ramp + constant)

    def _reachable_peak_speed(self) -> float:
        requested = self.cruise_speed_m_s
        if self.distance_m >= self._minimum_distance_for_speed(requested):
            return requested

        transition_speed = self.max_acceleration_m_s2**2 / self.max_jerk_m_s3
        transition_distance = self._minimum_distance_for_speed(transition_speed)
        if self.distance_m <= transition_distance:
            # D = 2 * v^(3/2) / sqrt(j)
            return (
                self.distance_m * math.sqrt(self.max_jerk_m_s3) / 2.0
            ) ** (2.0 / 3.0)

        # With saturated acceleration:
        # D = v^2/a + v*a/j.  Solve the positive quadratic root.
        a = self.max_acceleration_m_s2
        j = self.max_jerk_m_s3
        linear = a * a / j
        return 0.5 * (-linear + math.sqrt(linear * linear + 4.0 * a * self.distance_m))

    def _build_profile(self) -> None:
        self.peak_speed_m_s = self._reachable_peak_speed()
        (
            self.jerk_ramp_time_s,
            self.constant_acceleration_time_s,
            self.peak_acceleration_m_s2,
        ) = self._acceleration_shape(self.peak_speed_m_s)

        non_cruise_distance = self._minimum_distance_for_speed(self.peak_speed_m_s)
        remaining = max(0.0, self.distance_m - non_cruise_distance)
        self.cruise_time_s = remaining / self.peak_speed_m_s

        ramp = self.jerk_ramp_time_s
        constant = self.constant_acceleration_time_s
        cruise = self.cruise_time_s
        jerk = self.max_jerk_m_s3
        candidates = (
            ("accel_jerk_up", ramp, jerk),
            ("accel_hold", constant, 0.0),
            ("accel_jerk_down", ramp, -jerk),
            ("cruise", cruise, 0.0),
            ("decel_jerk_down", ramp, -jerk),
            ("decel_hold", constant, 0.0),
            ("decel_jerk_up", ramp, jerk),
        )
        self._phases = tuple(phase for phase in candidates if phase[1] > 1e-12)
        self.duration_s = sum(duration for _, duration, _ in self._phases)

    @staticmethod
    def _integrate(
        distance: float,
        speed: float,
        acceleration: float,
        jerk: float,
        duration: float,
    ) -> tuple[float, float, float]:
        distance += (
            speed * duration
            + 0.5 * acceleration * duration**2
            + jerk * duration**3 / 6.0
        )
        speed += acceleration * duration + 0.5 * jerk * duration**2
        acceleration += jerk * duration
        return distance, speed, acceleration

    def sample(self, elapsed_s: float) -> TrailerRouteSample:
        """Evaluate the reference at elapsed time, holding after completion."""

        elapsed = float(elapsed_s)
        if not math.isfinite(elapsed):
            raise ValueError("elapsed_s must be finite")
        if self.distance_m <= 1e-9 or elapsed >= self.duration_s:
            return TrailerRouteSample(
                position_enu_m=self.goal_enu_m.copy(),
                velocity_enu_m_s=np.zeros(3, dtype=np.float64),
                acceleration_enu_m_s2=np.zeros(3, dtype=np.float64),
                jerk_enu_m_s3=np.zeros(3, dtype=np.float64),
                distance_m=self.distance_m,
                yaw_rad=self.yaw_rad,
                phase="hold",
                complete=True,
            )

        remaining = max(0.0, elapsed)
        distance = speed = acceleration = 0.0
        active_phase = self._phases[0][0]
        active_jerk = self._phases[0][2]
        for name, duration, jerk in self._phases:
            active_phase = name
            active_jerk = jerk
            if remaining <= duration:
                distance, speed, acceleration = self._integrate(
                    distance, speed, acceleration, jerk, remaining
                )
                break
            distance, speed, acceleration = self._integrate(
                distance, speed, acceleration, jerk, duration
            )
            remaining -= duration

        # Round-off at phase boundaries must never command motion past the goal.
        distance = min(max(distance, 0.0), self.distance_m)
        speed = min(max(speed, 0.0), self.peak_speed_m_s)
        if abs(acceleration) < 1e-12:
            acceleration = 0.0
        return TrailerRouteSample(
            position_enu_m=self.start_enu_m + self.direction_enu * distance,
            velocity_enu_m_s=self.direction_enu * speed,
            acceleration_enu_m_s2=self.direction_enu * acceleration,
            jerk_enu_m_s3=self.direction_enu * active_jerk,
            distance_m=distance,
            yaw_rad=self.yaw_rad,
            phase=active_phase,
            complete=False,
        )

    def predict_position(self, elapsed_s: float, lead_time_s: float) -> np.ndarray:
        """Return a bounded future trailer position for intercept replanning."""

        lead = float(lead_time_s)
        if not math.isfinite(lead) or lead < 0.0:
            raise ValueError("lead_time_s must be finite and non-negative")
        return self.sample(float(elapsed_s) + lead).position_enu_m


class JerkLimitedWaypointRoute:
    """Piecewise S-curve route that stops at every configured waypoint.

    Each segment is an independent :class:`JerkLimitedTrailerRoute`.  Thus
    position, velocity and acceleration are continuous at a corner, with
    exactly zero velocity and acceleration before the heading changes.  This
    avoids the unsafe diagonal shortcut through city buildings while retaining
    the same scalar speed, acceleration and jerk limits on every segment.
    """

    def __init__(
        self,
        waypoints_enu_m: Sequence[Sequence[float]],
        cruise_speed_m_s: float = 3.0,
        max_acceleration_m_s2: float = 1.0,
        max_jerk_m_s3: float = 0.5,
    ) -> None:
        points = np.asarray(waypoints_enu_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            raise ValueError("waypoints_enu_m must have shape (N, 3), N >= 2")
        if not np.all(np.isfinite(points)):
            raise ValueError("waypoints_enu_m must contain only finite values")
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(segment_lengths <= 1e-9):
            raise ValueError("consecutive waypoints must be distinct")

        self.waypoints_enu_m = points.copy()
        self.start_enu_m = points[0].copy()
        self.goal_enu_m = points[-1].copy()
        self.cruise_speed_m_s = float(cruise_speed_m_s)
        self.max_acceleration_m_s2 = float(max_acceleration_m_s2)
        self.max_jerk_m_s3 = float(max_jerk_m_s3)
        self.segments = tuple(
            JerkLimitedTrailerRoute(
                start_enu_m=start,
                goal_enu_m=goal,
                cruise_speed_m_s=cruise_speed_m_s,
                max_acceleration_m_s2=max_acceleration_m_s2,
                max_jerk_m_s3=max_jerk_m_s3,
            )
            for start, goal in zip(points[:-1], points[1:])
        )
        self._segment_end_times_s = np.cumsum(
            [segment.duration_s for segment in self.segments]
        )
        self._segment_start_distances_m = np.concatenate((
            np.zeros(1, dtype=np.float64),
            np.cumsum([segment.distance_m for segment in self.segments[:-1]]),
        ))
        self.duration_s = float(self._segment_end_times_s[-1])
        self.distance_m = float(sum(segment.distance_m for segment in self.segments))
        self.peak_speed_m_s = max(
            segment.peak_speed_m_s for segment in self.segments
        )
        self.direction_enu = self.segments[0].direction_enu.copy()
        self.yaw_rad = self.segments[0].yaw_rad

    def sample(self, elapsed_s: float) -> TrailerRouteSample:
        elapsed = float(elapsed_s)
        if not math.isfinite(elapsed):
            raise ValueError("elapsed_s must be finite")
        if elapsed >= self.duration_s:
            final = self.segments[-1].sample(self.segments[-1].duration_s)
            return TrailerRouteSample(
                final.position_enu_m,
                final.velocity_enu_m_s,
                final.acceleration_enu_m_s2,
                final.jerk_enu_m_s3,
                self.distance_m,
                final.yaw_rad,
                "hold",
                True,
            )

        bounded_elapsed = max(0.0, elapsed)
        index = min(
            bisect_right(self._segment_end_times_s, bounded_elapsed),
            len(self.segments) - 1,
        )
        start_time = 0.0 if index == 0 else self._segment_end_times_s[index - 1]
        local = self.segments[index].sample(bounded_elapsed - start_time)
        return TrailerRouteSample(
            local.position_enu_m,
            local.velocity_enu_m_s,
            local.acceleration_enu_m_s2,
            local.jerk_enu_m_s3,
            float(self._segment_start_distances_m[index] + local.distance_m),
            local.yaw_rad,
            f"segment_{index + 1}/{local.phase}",
            False,
        )

    def predict_position(self, elapsed_s: float, lead_time_s: float) -> np.ndarray:
        lead = float(lead_time_s)
        if not math.isfinite(lead) or lead < 0.0:
            raise ValueError("lead_time_s must be finite and non-negative")
        return self.sample(float(elapsed_s) + lead).position_enu_m
