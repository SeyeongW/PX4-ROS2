"""Capture-time vehicle-state interpolation for ArUco measurements.

The module has no ROS imports.  The MAVROS boundary supplies PX4-synchronized
sample stamps, local ENU position/velocity, and body-FLU to world-ENU attitude.
Camera stamps remain in the ROS clock domain and are mapped with the estimated
constant clock offset before querying the history.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .marker_measurement import flu_to_enu_rotation_matrix


_EPS = 1.0e-9


class StateInterpolationError(ValueError):
    """A capture cannot be represented by the available state history."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason)


class StateTimeReset(RuntimeError):
    """A state sample clock moved backwards far enough to indicate a reset."""


@dataclass(frozen=True)
class OffsetObservation:
    accepted: bool
    reset_detected: bool
    residual_s: float | None


class TimestampOffsetEstimator:
    """Estimate ``sample_time = ros_time + offset`` with reset detection."""

    def __init__(
        self,
        *,
        smoothing_factor: float = 0.10,
        minimum_samples: int = 5,
        reset_threshold_s: float = 0.50,
        median_window_size: int = 101,
        offset_reset_threshold_s: float | None = None,
    ) -> None:
        offset_reset_threshold = (
            reset_threshold_s
            if offset_reset_threshold_s is None
            else offset_reset_threshold_s
        )
        if not 0.0 < smoothing_factor <= 1.0:
            raise ValueError("smoothing_factor must be in (0, 1]")
        if (
            minimum_samples < 1
            or reset_threshold_s <= 0.0
            or offset_reset_threshold <= 0.0
            or median_window_size < minimum_samples
        ):
            raise ValueError(
                "offset window, sample count, and reset threshold are invalid"
            )
        self.smoothing_factor = float(smoothing_factor)
        self.minimum_samples = int(minimum_samples)
        self.reset_threshold_s = float(reset_threshold_s)
        self.offset_reset_threshold_s = float(offset_reset_threshold)
        self.median_window_size = int(median_window_size)
        self.reset()

    def reset(self) -> None:
        self.offset_s: float | None = None
        self.sample_count = 0
        self.last_ros_time_s: float | None = None
        self.last_sample_time_s: float | None = None
        self._candidates: deque[float] = deque(
            maxlen=self.median_window_size
        )

    @property
    def ready(self) -> bool:
        return (
            self.offset_s is not None
            and self.sample_count >= self.minimum_samples
        )

    def observe(
        self, ros_time_s: float, sample_time_s: float
    ) -> OffsetObservation:
        ros_time = float(ros_time_s)
        sample_time = float(sample_time_s)
        if (
            not math.isfinite(ros_time)
            or not math.isfinite(sample_time)
            or ros_time <= 0.0
            or sample_time <= 0.0
        ):
            return OffsetObservation(False, False, None)

        candidate = sample_time - ros_time
        residual = (
            None if self.offset_s is None else candidate - self.offset_s
        )
        ros_reset = (
            self.last_ros_time_s is not None
            and ros_time < self.last_ros_time_s - self.reset_threshold_s
        )
        sample_reset = (
            self.last_sample_time_s is not None
            and sample_time
            < self.last_sample_time_s - self.reset_threshold_s
        )
        if ros_reset or sample_reset:
            self.reset()
            return OffsetObservation(False, True, residual)

        self._candidates.append(candidate)
        robust_candidate = float(np.median(self._candidates))
        persistent_offset_jump = (
            self.offset_s is not None
            and len(self._candidates) == self.median_window_size
            and abs(robust_candidate - self.offset_s)
            > self.offset_reset_threshold_s
        )
        if persistent_offset_jump:
            self.reset()
            return OffsetObservation(False, True, residual)

        if self.offset_s is None:
            self.offset_s = robust_candidate
        else:
            alpha = self.smoothing_factor
            self.offset_s = (
                (1.0 - alpha) * self.offset_s
                + alpha * robust_candidate
            )
        self.sample_count += 1
        self.last_ros_time_s = ros_time
        self.last_sample_time_s = sample_time
        return OffsetObservation(True, False, residual)

    def to_sample_time(self, ros_time_s: float) -> float:
        ros_time = float(ros_time_s)
        if not self.ready or self.offset_s is None:
            raise StateInterpolationError("time_offset_uninitialized")
        if not math.isfinite(ros_time) or ros_time <= 0.0:
            raise StateInterpolationError("invalid_ros_capture_time")
        return ros_time + self.offset_s


@dataclass(frozen=True)
class TimesyncObservation:
    accepted: bool
    reset_detected: bool


class MavrosTimesyncTracker:
    """Validate MAVROS' already-filtered companion-to-PX4 time offset."""

    def __init__(
        self,
        *,
        minimum_samples: int = 5,
        reset_threshold_s: float = 0.50,
        maximum_round_trip_time_ms: float = 50.0,
    ) -> None:
        if (
            minimum_samples < 1
            or reset_threshold_s <= 0.0
            or maximum_round_trip_time_ms <= 0.0
        ):
            raise ValueError("timesync limits must be positive")
        self.minimum_samples = int(minimum_samples)
        self.reset_threshold_ns = int(reset_threshold_s * 1.0e9)
        self.maximum_round_trip_time_ms = float(maximum_round_trip_time_ms)
        self.reset()

    def reset(self) -> None:
        self.estimated_offset_ns: int | None = None
        self.last_remote_timestamp_ns: int | None = None
        self.sample_count = 0

    @property
    def ready(self) -> bool:
        return (
            self.estimated_offset_ns is not None
            and self.sample_count >= self.minimum_samples
        )

    def observe(
        self,
        remote_timestamp_ns: int,
        estimated_offset_ns: int,
        round_trip_time_ms: float,
    ) -> TimesyncObservation:
        remote = int(remote_timestamp_ns)
        offset = int(estimated_offset_ns)
        round_trip = float(round_trip_time_ms)
        if (
            remote <= 0
            # MAVROS publishes INT64_MIN while its time synchroniser has no
            # estimate.  Treating that sentinel as an offset turns an
            # ordinary boot-time header into a timestamp about 292 years in
            # the future and creates a false reset when the filter converges.
            or offset == -(1 << 63)
            or not math.isfinite(round_trip)
            or round_trip < 0.0
            or round_trip > self.maximum_round_trip_time_ms
        ):
            return TimesyncObservation(False, False)

        remote_reset = (
            self.last_remote_timestamp_ns is not None
            and remote
            < self.last_remote_timestamp_ns - self.reset_threshold_ns
        )
        if remote_reset:
            self.reset()
            return TimesyncObservation(False, True)

        # In lockstep SITL the PX4 simulation clock and companion wall clock
        # may run at different rates.  MAVROS' estimated offset can therefore
        # rebase while both clocks remain healthy.  The recovered PX4 state
        # history, not this filter value alone, detects a real clock reset.
        self.estimated_offset_ns = offset
        self.last_remote_timestamp_ns = remote
        self.sample_count += 1
        return TimesyncObservation(True, False)

    def recover_sample_time_s(self, seconds: int, nanoseconds: int) -> float:
        if not self.ready or self.estimated_offset_ns is None:
            raise StateInterpolationError("mavros_timesync_uninitialized")
        return mavros_header_to_px4_time_s(
            seconds,
            nanoseconds,
            self.estimated_offset_ns,
        )


@dataclass(frozen=True)
class _PositionSample:
    sample_time_s: float
    position_enu_m: np.ndarray


@dataclass(frozen=True)
class _AttitudeSample:
    sample_time_s: float
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class _VelocitySample:
    sample_time_s: float
    velocity_enu_m_s: np.ndarray


@dataclass(frozen=True)
class InterpolatedVehicleState:
    sample_time_s: float
    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray
    quaternion_xyzw: np.ndarray


def mavros_header_to_px4_time_s(
    seconds: int,
    nanoseconds: int,
    estimated_offset_ns: int,
) -> float:
    """Recover a PX4 sample stamp from a MAVROS-synchronised ROS header.

    MAVROS publishes ``header = remote PX4 time + estimated_offset``.  Integer
    nanosecond subtraction is deliberately used before conversion to seconds,
    so an epoch-sized MAVROS header does not lose capture-time precision.
    """

    sec = int(seconds)
    nanosec = int(nanoseconds)
    offset = int(estimated_offset_ns)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("MAVROS header timestamp is invalid")
    header_ns = sec * 1_000_000_000 + nanosec
    sample_ns = header_ns - offset
    if sample_ns <= 0:
        raise ValueError("recovered PX4 sample timestamp must be positive")
    return float(sample_ns) * 1.0e-9


def _finite_vector(
    values: Sequence[float], length: int, name: str
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return vector.copy()


def normalize_quaternion_xyzw(values: Sequence[float]) -> np.ndarray:
    quaternion = _finite_vector(values, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < _EPS:
        raise ValueError("quaternion norm must be positive")
    return quaternion / norm


def quaternion_slerp_xyzw(
    first_xyzw: Sequence[float],
    second_xyzw: Sequence[float],
    fraction: float,
) -> np.ndarray:
    """Shortest-path quaternion SLERP, including equivalent ``q``/``-q``."""

    first = normalize_quaternion_xyzw(first_xyzw)
    second = normalize_quaternion_xyzw(second_xyzw)
    amount = float(fraction)
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ValueError("SLERP fraction must be finite and in [0, 1]")
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_xyzw(first + amount * (second - first))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return normalize_quaternion_xyzw(
        math.sin((1.0 - amount) * angle) / sine * first
        + math.sin(amount * angle) / sine * second
    )


def quaternion_multiply_xyzw(
    left_xyzw: Sequence[float], right_xyzw: Sequence[float]
) -> np.ndarray:
    """Compose rotations as ``left * right`` in ROS xyzw ordering."""

    left = normalize_quaternion_xyzw(left_xyzw)
    right = normalize_quaternion_xyzw(right_xyzw)
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return normalize_quaternion_xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def quaternion_yaw_enu_rad(quaternion_xyzw: Sequence[float]) -> float:
    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def quaternion_world_z_tilt_rad(
    quaternion_xyzw: Sequence[float],
) -> float:
    """Return the directed marker-normal tilt away from world ENU +Z."""
    rotation = flu_to_enu_rotation_matrix(quaternion_xyzw)
    normal_world = rotation[:, 2]
    cosine = float(np.clip(normal_world[2], -1.0, 1.0))
    return math.acos(cosine)


def marker_world_pose_at_capture(
    state: InterpolatedVehicleState,
    marker_position_flu: Sequence[float],
    marker_orientation_xyzw: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Transform a production ``base_link`` marker pose into world ENU.

    The production tf2 output is the vehicle-to-marker translation expressed
    in body FLU.  Consequently rigid-transform composition is
    ``p_world_marker = p_world_vehicle + R_world_body @ p_body_marker``.
    """

    relative = _finite_vector(marker_position_flu, 3, "marker position")
    rotation = flu_to_enu_rotation_matrix(state.quaternion_xyzw)
    position = state.position_enu_m + rotation @ relative
    orientation = quaternion_multiply_xyzw(
        state.quaternion_xyzw, marker_orientation_xyzw
    )
    yaw = quaternion_yaw_enu_rad(orientation)
    return position, orientation, yaw, rotation


class VehicleStateHistory:
    """Bounded position, velocity, and attitude histories in PX4 time."""

    def __init__(
        self,
        *,
        duration_s: float = 3.0,
        reset_threshold_s: float = 0.50,
    ) -> None:
        if duration_s <= 0.0 or reset_threshold_s <= 0.0:
            raise ValueError(
                "history duration and reset threshold must be positive"
            )
        self.duration_s = float(duration_s)
        self.reset_threshold_s = float(reset_threshold_s)
        self.reset()

    def reset(self) -> None:
        self._positions: list[_PositionSample] = []
        self._velocities: list[_VelocitySample] = []
        self._attitudes: list[_AttitudeSample] = []

    @property
    def position_sample_count(self) -> int:
        return len(self._positions)

    @property
    def velocity_sample_count(self) -> int:
        return len(self._velocities)

    @property
    def attitude_sample_count(self) -> int:
        return len(self._attitudes)

    @property
    def latest_common_time_s(self) -> float | None:
        if not self._positions or not self._velocities or not self._attitudes:
            return None
        return min(
            self._positions[-1].sample_time_s,
            self._velocities[-1].sample_time_s,
            self._attitudes[-1].sample_time_s,
        )

    def add_position(
        self,
        sample_time_s: float,
        position_enu_m: Sequence[float],
    ) -> bool:
        stamp = self._valid_stamp(sample_time_s)
        sample = _PositionSample(
            stamp,
            _finite_vector(position_enu_m, 3, "position"),
        )
        accepted = self._append(self._positions, sample)
        if accepted:
            self._prune(self._positions, stamp)
        return accepted

    def add_velocity(
        self,
        sample_time_s: float,
        velocity_enu_m_s: Sequence[float],
    ) -> bool:
        stamp = self._valid_stamp(sample_time_s)
        sample = _VelocitySample(
            stamp,
            _finite_vector(velocity_enu_m_s, 3, "velocity"),
        )
        accepted = self._append(self._velocities, sample)
        if accepted:
            self._prune(self._velocities, stamp)
        return accepted

    def add_attitude(
        self,
        sample_time_s: float,
        quaternion_xyzw: Sequence[float],
    ) -> bool:
        stamp = self._valid_stamp(sample_time_s)
        sample = _AttitudeSample(
            stamp,
            normalize_quaternion_xyzw(quaternion_xyzw),
        )
        accepted = self._append(self._attitudes, sample)
        if accepted:
            self._prune(self._attitudes, stamp)
        return accepted

    @staticmethod
    def _valid_stamp(sample_time_s: float) -> float:
        stamp = float(sample_time_s)
        if not math.isfinite(stamp) or stamp <= 0.0:
            raise ValueError("sample timestamp must be finite and positive")
        return stamp

    def _append(self, samples: list, sample: object) -> bool:
        stamp = float(getattr(sample, "sample_time_s"))
        if samples:
            latest = float(samples[-1].sample_time_s)
            if stamp < latest - self.reset_threshold_s:
                raise StateTimeReset(
                    f"state sample time reset from {latest:.9f} to {stamp:.9f}"
                )
            if stamp < latest:
                return False
            if abs(stamp - latest) <= _EPS:
                samples[-1] = sample
                return True
        samples.append(sample)
        return True

    def _prune(self, samples: list, newest_s: float) -> None:
        oldest_allowed = newest_s - self.duration_s
        first_kept = 0
        while (
            first_kept + 1 < len(samples)
            and samples[first_kept].sample_time_s < oldest_allowed
        ):
            first_kept += 1
        if first_kept:
            del samples[:first_kept]

    @staticmethod
    def _bracket(samples: list, target_s: float, maximum_gap_s: float):
        timestamps = [sample.sample_time_s for sample in samples]
        index = bisect_left(timestamps, target_s)
        if index < len(samples) and abs(timestamps[index] - target_s) <= _EPS:
            return samples[index], samples[index], 0.0
        if index == 0 or index >= len(samples):
            raise StateInterpolationError("capture_outside_history")
        first = samples[index - 1]
        second = samples[index]
        gap = float(second.sample_time_s - first.sample_time_s)
        if gap <= 0.0 or gap > maximum_gap_s:
            raise StateInterpolationError("interpolation_gap_too_large")
        fraction = (target_s - first.sample_time_s) / gap
        return first, second, float(fraction)

    def interpolate(
        self,
        sample_time_s: float,
        *,
        current_sample_time_s: float,
        maximum_gap_s: float,
        maximum_capture_age_s: float,
        maximum_state_age_s: float,
        future_tolerance_s: float = 0.0,
    ) -> InterpolatedVehicleState:
        target = self._valid_stamp(sample_time_s)
        current = self._valid_stamp(current_sample_time_s)
        if min(
            maximum_gap_s,
            maximum_capture_age_s,
            maximum_state_age_s,
        ) <= 0.0 or future_tolerance_s < 0.0:
            raise ValueError("interpolation limits are invalid")
        if target > current + future_tolerance_s:
            raise StateInterpolationError("capture_time_in_future")
        if current - target > maximum_capture_age_s:
            raise StateInterpolationError("capture_time_stale")
        if not self._positions or not self._velocities or not self._attitudes:
            raise StateInterpolationError("state_history_uninitialized")

        oldest_common = max(
            self._positions[0].sample_time_s,
            self._velocities[0].sample_time_s,
            self._attitudes[0].sample_time_s,
        )
        latest_common = min(
            self._positions[-1].sample_time_s,
            self._velocities[-1].sample_time_s,
            self._attitudes[-1].sample_time_s,
        )
        if current - latest_common > maximum_state_age_s:
            raise StateInterpolationError("state_history_stale")
        if target < oldest_common - _EPS:
            raise StateInterpolationError("capture_outside_history")
        if target > latest_common + _EPS:
            raise StateInterpolationError("capture_ahead_of_state_history")

        position_first, position_second, position_fraction = self._bracket(
            self._positions, target, maximum_gap_s
        )
        velocity_first, velocity_second, velocity_fraction = self._bracket(
            self._velocities, target, maximum_gap_s
        )
        attitude_first, attitude_second, attitude_fraction = self._bracket(
            self._attitudes, target, maximum_gap_s
        )
        position = (
            position_first.position_enu_m
            + position_fraction
            * (
                position_second.position_enu_m
                - position_first.position_enu_m
            )
        )
        velocity = (
            velocity_first.velocity_enu_m_s
            + velocity_fraction
            * (
                velocity_second.velocity_enu_m_s
                - velocity_first.velocity_enu_m_s
            )
        )
        attitude = quaternion_slerp_xyzw(
            attitude_first.quaternion_xyzw,
            attitude_second.quaternion_xyzw,
            attitude_fraction,
        )
        return InterpolatedVehicleState(
            target,
            position.copy(),
            velocity.copy(),
            attitude,
        )
