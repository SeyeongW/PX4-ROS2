"""Timestamp-ordered probabilistic fusion for the landing-marker state.

This module intentionally has no ROS imports.  Both ArUco poses and trailer
odometry are converted to a common PX4 sample-time / world-ENU contract by the
ROS node before (or while) they enter this estimator.  A short reorder window
allows delayed camera captures to be processed before newer odometry, while a
control query predicts a copy of the filter and never advances canonical time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
from typing import Mapping, Sequence

import numpy as np


_STATE_SIZE = 8
_POSITION_INDICES = (0, 1, 2)
_VELOCITY_INDICES = (3, 4, 5)
_YAW_INDEX = 6
_YAW_RATE_INDEX = 7
_VISION_INDICES = (0, 1, 2, 6)
_COVARIANCE_TOLERANCE = 1.0e-9


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)`` without losing finite validation."""
    angle = float(angle_rad)
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _finite_vector(
    value: Sequence[float], size: int, description: str
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{description} must be a finite {size}-vector")
    return vector.copy()


def validate_covariance(
    value: Sequence[Sequence[float]] | Sequence[float],
    size: int,
    description: str,
) -> np.ndarray:
    """Return a finite, symmetric positive-semidefinite covariance."""
    try:
        covariance = np.asarray(value, dtype=float).reshape(size, size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{description} covariance must contain {size * size} values"
        ) from exc
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{description} covariance must be finite")
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        raise ValueError(f"{description} covariance must be symmetric")
    if np.any(np.diag(covariance) < 0.0):
        raise ValueError(
            f"{description} covariance variances must be nonnegative"
        )
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    if float(np.min(eigenvalues)) < -_COVARIANCE_TOLERANCE:
        raise ValueError(
            f"{description} covariance must be positive semidefinite"
        )
    return 0.5 * (covariance + covariance.T)


def _apply_variance_floor(
    covariance: np.ndarray, floors: Sequence[float]
) -> np.ndarray:
    result = np.asarray(covariance, dtype=float).copy()
    floor_vector = _finite_vector(floors, result.shape[0], "variance floors")
    if np.any(floor_vector <= 0.0):
        raise ValueError("variance floors must be positive")
    scale = np.sqrt(floor_vector)
    normalized = result / np.outer(scale, scale)
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (normalized + normalized.T)
    )
    # Clamp generalized covariance eigenvalues, not only diagonal entries.
    # Thus an already-noisier message remains unchanged, while a perfectly
    # correlated PSD input cannot claim zero uncertainty in its null space.
    floored = (
        eigenvectors
        @ np.diag(np.maximum(eigenvalues, 1.0))
        @ eigenvectors.T
    )
    result = floored * np.outer(scale, scale)
    return 0.5 * (result + result.T)


@dataclass(frozen=True)
class TrailerMarkerMeasurement:
    """Marker-centered odometry measurement and propagated covariance."""

    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray
    yaw_enu_rad: float
    yaw_rate_rad_s: float
    covariance: np.ndarray
    jacobian: np.ndarray


def trailer_odometry_to_marker(
    trailer_position_enu_m: Sequence[float],
    trailer_velocity: Sequence[float],
    trailer_yaw_enu_rad: float,
    trailer_yaw_rate_rad_s: float,
    pose_covariance_6x6: Sequence[Sequence[float]] | Sequence[float],
    twist_covariance_6x6: Sequence[Sequence[float]] | Sequence[float],
    marker_offset_trailer_m: Sequence[float],
    *,
    twist_in_body_frame: bool = True,
) -> TrailerMarkerMeasurement:
    """Move trailer-root odometry to the physical landing-marker center.

    ``nav_msgs/Odometry.twist`` is normally expressed in ``child_frame_id``.
    The optional world-frame path exists for sources which explicitly publish
    map-frame twist; the active experiment uses the default body-frame path.
    Pose/twist cross covariance is unavailable in ``nav_msgs/Odometry`` and is
    therefore zero in the raw block-diagonal covariance before propagation.
    """
    position = _finite_vector(
        trailer_position_enu_m, 3, "trailer position"
    )
    velocity_input = _finite_vector(
        trailer_velocity, 3, "trailer velocity"
    )
    offset = _finite_vector(
        marker_offset_trailer_m, 3, "trailer marker offset"
    )
    yaw = wrap_angle(trailer_yaw_enu_rad)
    yaw_rate = float(trailer_yaw_rate_rad_s)
    if not math.isfinite(yaw_rate):
        raise ValueError("trailer yaw rate must be finite")

    pose_covariance = validate_covariance(
        pose_covariance_6x6, 6, "trailer pose"
    )
    twist_covariance = validate_covariance(
        twist_covariance_6x6, 6, "trailer twist"
    )
    selected = (0, 1, 2, 5)
    raw_covariance = np.zeros((8, 8), dtype=float)
    raw_covariance[:4, :4] = pose_covariance[np.ix_(selected, selected)]
    raw_covariance[4:, 4:] = twist_covariance[np.ix_(selected, selected)]

    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
    rotated_offset = rotation @ offset
    offset_tangent = np.asarray(
        (-rotated_offset[1], rotated_offset[0], 0.0), dtype=float
    )
    if twist_in_body_frame:
        velocity_enu = rotation @ velocity_input
        velocity_rotation = rotation
        velocity_yaw_derivative = np.asarray(
            (-velocity_enu[1], velocity_enu[0], 0.0), dtype=float
        )
    else:
        velocity_enu = velocity_input.copy()
        velocity_rotation = np.eye(3, dtype=float)
        velocity_yaw_derivative = np.zeros(3, dtype=float)

    marker_position = position + rotated_offset
    marker_velocity = velocity_enu + yaw_rate * offset_tangent

    # Raw variable order: [position(3), yaw, velocity(3), yaw_rate].
    # Output order: [marker position(3), marker velocity(3), yaw, yaw_rate].
    jacobian = np.zeros((8, 8), dtype=float)
    jacobian[:3, :3] = np.eye(3, dtype=float)
    jacobian[:3, 3] = offset_tangent
    jacobian[3:6, 4:7] = velocity_rotation
    jacobian[3:6, 7] = offset_tangent
    offset_second_derivative = np.asarray(
        (-rotated_offset[0], -rotated_offset[1], 0.0), dtype=float
    )
    jacobian[3:6, 3] = (
        velocity_yaw_derivative
        + yaw_rate * offset_second_derivative
    )
    jacobian[6, 3] = 1.0
    jacobian[7, 7] = 1.0
    covariance = jacobian @ raw_covariance @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)

    return TrailerMarkerMeasurement(
        marker_position,
        marker_velocity,
        yaw,
        yaw_rate,
        covariance,
        jacobian,
    )


@dataclass(frozen=True)
class FusionParameters:
    """Estimator tuning and fail-closed timing limits."""

    reorder_window_s: float = 0.30
    maximum_measurement_age_s: float = 0.75
    future_tolerance_s: float = 0.02
    vision_timeout_s: float = 0.60
    odometry_timeout_s: float = 1.00
    predict_only_timeout_s: float = 1.50
    process_acceleration_variance: float = 0.25
    process_yaw_acceleration_variance: float = 0.25
    vision_position_variance_floor: float = 4.0e-4
    vision_yaw_variance_floor: float = math.radians(2.0) ** 2
    odometry_position_variance_floor: float = 2.5e-3
    odometry_velocity_variance_floor: float = 1.0e-2
    odometry_yaw_variance_floor: float = math.radians(3.0) ** 2
    odometry_yaw_rate_variance_floor: float = math.radians(5.0) ** 2
    initial_velocity_variance: float = 25.0
    initial_yaw_rate_variance: float = 4.0
    vision_nis_gate: float = 18.47
    odometry_nis_gate: float = 26.12
    maximum_position_innovation_m: float = 20.0
    maximum_velocity_innovation_m_s: float = 10.0
    maximum_yaw_rate_rad_s: float = 3.0

    def __post_init__(self) -> None:
        """Reject non-finite or internally inconsistent tuning."""
        values = {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("fusion parameters must be finite")
        positive = set(values) - {"future_tolerance_s"}
        if any(values[name] <= 0.0 for name in positive):
            raise ValueError("fusion parameters must be positive")
        if values["future_tolerance_s"] < 0.0:
            raise ValueError("future tolerance must be nonnegative")
        if self.predict_only_timeout_s < max(
            self.vision_timeout_s, self.odometry_timeout_s
        ):
            raise ValueError(
                "predict-only timeout cannot be shorter than sensor timeout"
            )


@dataclass
class SensorStatistics:
    """Per-source acceptance and rejection diagnostics."""

    received: int = 0
    queued: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicate: int = 0
    late: int = 0
    invalid_covariance: int = 0
    invalid_value: int = 0
    stale: int = 0
    future: int = 0
    jump: int = 0
    nis: int = 0
    last_received_stamp_s: float | None = None
    last_accepted_stamp_s: float | None = None
    last_rejection_reason: str | None = None
    last_nis: float | None = None
    last_position_innovation_enu_m: np.ndarray | None = None
    last_position_innovation_norm_m: float | None = None
    last_yaw_innovation_rad: float | None = None
    last_position_marginal_nis: float | None = None
    last_yaw_marginal_nis: float | None = None


@dataclass(frozen=True)
class SubmissionResult:
    """Immediate result of validating/enqueuing a measurement."""

    queued: bool
    reason: str


@dataclass(frozen=True)
class TargetEstimate:
    """A non-mutating prediction of the fused marker state."""

    valid: bool
    mode: str
    stamp_s: float
    position_enu_m: np.ndarray | None = None
    velocity_enu_m_s: np.ndarray | None = None
    yaw_enu_rad: float | None = None
    yaw_rate_rad_s: float | None = None
    covariance: np.ndarray | None = None
    filter_stamp_s: float | None = None
    vision_fresh: bool = False
    odometry_fresh: bool = False
    last_measurement_age_s: float | None = None


@dataclass(order=True)
class _QueuedMeasurement:
    stamp_s: float
    serial: int
    source: str = field(compare=False)
    kind: str = field(compare=False)
    value: np.ndarray = field(compare=False)
    covariance: np.ndarray = field(compare=False)


class TimestampedTargetEstimator:
    """Eight-state CV Kalman filter with timestamp-ordered sensor updates."""

    VISION_SOURCES = ("vision_front", "vision_down")
    ODOMETRY_SOURCE = "odometry"

    def __init__(self, parameters: FusionParameters | None = None) -> None:
        """Construct an empty estimator with immutable tuning."""
        self.parameters = parameters or FusionParameters()
        self.statistics: dict[str, SensorStatistics] = {
            source: SensorStatistics()
            for source in (*self.VISION_SOURCES, self.ODOMETRY_SOURCE)
        }
        self.reset()

    def reset(self) -> None:
        """Clear filter, reorder queue, dedupe memory, and counters."""
        self.x = np.zeros(_STATE_SIZE, dtype=float)
        self.P = np.eye(_STATE_SIZE, dtype=float)
        self.filter_stamp_s: float | None = None
        self._queue: list[_QueuedMeasurement] = []
        self._initial_candidate: _QueuedMeasurement | None = None
        self._serial = 0
        self._seen_keys: dict[str, set[int]] = {
            source: set() for source in self.statistics
        }
        self._seen_order: dict[str, deque[int]] = {
            source: deque() for source in self.statistics
        }
        for source in self.statistics:
            self.statistics[source] = SensorStatistics()

    @property
    def pending_count(self) -> int:
        """Return the number of measurements awaiting the watermark."""
        return len(self._queue) + int(self._initial_candidate is not None)

    @property
    def initialized(self) -> bool:
        """Return whether at least one measurement initialized the state."""
        return self.filter_stamp_s is not None

    def submit_vision(
        self,
        source: str,
        stamp_s: float,
        stamp_key_ns: int,
        position_enu_m: Sequence[float],
        yaw_enu_rad: float,
        covariance_4x4: Sequence[Sequence[float]] | Sequence[float],
        received_px4_time_s: float,
    ) -> SubmissionResult:
        """Validate and enqueue one front/down ArUco pose measurement."""
        if source not in self.VISION_SOURCES:
            raise ValueError(
                "vision source must be vision_front or vision_down"
            )
        statistics = self.statistics[source]
        statistics.received += 1
        timestamp_result = self._register_timestamp(
            source, stamp_s, stamp_key_ns, received_px4_time_s
        )
        if timestamp_result is not None:
            return timestamp_result
        try:
            position = _finite_vector(position_enu_m, 3, "vision position")
            yaw = wrap_angle(yaw_enu_rad)
            covariance = validate_covariance(
                covariance_4x4, 4, "vision pose"
            )
            covariance = _apply_variance_floor(
                covariance,
                (
                    self.parameters.vision_position_variance_floor,
                    self.parameters.vision_position_variance_floor,
                    self.parameters.vision_position_variance_floor,
                    self.parameters.vision_yaw_variance_floor,
                ),
            )
        except ValueError as exc:
            category = (
                "invalid_covariance"
                if "covariance" in str(exc) or "variance" in str(exc)
                else "invalid_value"
            )
            return self._reject(source, category, str(exc))
        value = np.concatenate((position, np.asarray((yaw,), dtype=float)))
        return self._enqueue(source, "vision", stamp_s, value, covariance)

    def submit_odometry(
        self,
        stamp_s: float,
        stamp_key_ns: int,
        trailer_position_enu_m: Sequence[float],
        trailer_velocity: Sequence[float],
        trailer_yaw_enu_rad: float,
        trailer_yaw_rate_rad_s: float,
        pose_covariance_6x6: Sequence[Sequence[float]] | Sequence[float],
        twist_covariance_6x6: Sequence[Sequence[float]] | Sequence[float],
        marker_offset_trailer_m: Sequence[float],
        received_px4_time_s: float,
        *,
        twist_in_body_frame: bool = True,
    ) -> SubmissionResult:
        """Validate and enqueue trailer odometry at marker center."""
        source = self.ODOMETRY_SOURCE
        self.statistics[source].received += 1
        timestamp_result = self._register_timestamp(
            source, stamp_s, stamp_key_ns, received_px4_time_s
        )
        if timestamp_result is not None:
            return timestamp_result
        try:
            measurement = trailer_odometry_to_marker(
                trailer_position_enu_m,
                trailer_velocity,
                trailer_yaw_enu_rad,
                trailer_yaw_rate_rad_s,
                pose_covariance_6x6,
                twist_covariance_6x6,
                marker_offset_trailer_m,
                twist_in_body_frame=twist_in_body_frame,
            )
            covariance = validate_covariance(
                measurement.covariance, 8, "marker odometry"
            )
            covariance = _apply_variance_floor(
                covariance,
                (
                    self.parameters.odometry_position_variance_floor,
                    self.parameters.odometry_position_variance_floor,
                    self.parameters.odometry_position_variance_floor,
                    self.parameters.odometry_velocity_variance_floor,
                    self.parameters.odometry_velocity_variance_floor,
                    self.parameters.odometry_velocity_variance_floor,
                    self.parameters.odometry_yaw_variance_floor,
                    self.parameters.odometry_yaw_rate_variance_floor,
                ),
            )
        except ValueError as exc:
            category = (
                "invalid_covariance"
                if "covariance" in str(exc) or "variance" in str(exc)
                else "invalid_value"
            )
            return self._reject(source, category, str(exc))
        if float(np.linalg.norm(measurement.velocity_enu_m_s)) > (
            self.parameters.maximum_velocity_innovation_m_s
        ):
            return self._reject(
                source, "jump", "odometry velocity exceeds physical limit"
            )
        if abs(measurement.yaw_rate_rad_s) > (
            self.parameters.maximum_yaw_rate_rad_s
        ):
            return self._reject(
                source, "jump", "odometry yaw rate exceeds physical limit"
            )
        value = np.concatenate(
            (
                measurement.position_enu_m,
                measurement.velocity_enu_m_s,
                np.asarray(
                    (
                        measurement.yaw_enu_rad,
                        measurement.yaw_rate_rad_s,
                    ),
                    dtype=float,
                ),
            )
        )
        return self._enqueue(source, "odometry", stamp_s, value, covariance)

    def _register_timestamp(
        self,
        source: str,
        stamp_s: float,
        stamp_key_ns: int,
        received_s: float,
    ) -> SubmissionResult | None:
        statistics = self.statistics[source]
        try:
            stamp = float(stamp_s)
            received = float(received_s)
            key = int(stamp_key_ns)
        except (TypeError, ValueError, OverflowError):
            return self._reject(source, "invalid_value", "invalid timestamp")
        if (
            not math.isfinite(stamp)
            or not math.isfinite(received)
            or stamp <= 0.0
            or key <= 0
        ):
            return self._reject(source, "invalid_value", "invalid timestamp")
        statistics.last_received_stamp_s = stamp
        if key in self._seen_keys[source]:
            return self._reject(source, "duplicate", "duplicate timestamp")
        self._remember_stamp(source, key)
        age = received - stamp
        if age > self.parameters.maximum_measurement_age_s:
            return self._reject(source, "stale", "stale timestamp")
        if age < -self.parameters.future_tolerance_s:
            return self._reject(source, "future", "future timestamp")
        if (
            self.filter_stamp_s is not None
            and stamp < self.filter_stamp_s - 1.0e-9
        ):
            return self._reject(source, "late", "late timestamp")
        if (
            self._initial_candidate is not None
            and stamp < self._initial_candidate.stamp_s - 1.0e-9
        ):
            return self._reject(source, "late", "late timestamp")
        return None

    def _remember_stamp(self, source: str, key: int) -> None:
        keys = self._seen_keys[source]
        order = self._seen_order[source]
        keys.add(key)
        order.append(key)
        while len(order) > 512:
            keys.discard(order.popleft())

    def _enqueue(
        self,
        source: str,
        kind: str,
        stamp_s: float,
        value: np.ndarray,
        covariance: np.ndarray,
    ) -> SubmissionResult:
        self._serial += 1
        heapq.heappush(
            self._queue,
            _QueuedMeasurement(
                float(stamp_s),
                self._serial,
                source,
                kind,
                value.copy(),
                covariance.copy(),
            ),
        )
        self.statistics[source].queued += 1
        return SubmissionResult(True, "queued")

    def _reject(
        self, source: str, category: str, reason: str
    ) -> SubmissionResult:
        statistics = self.statistics[source]
        statistics.rejected += 1
        if hasattr(statistics, category):
            current = int(getattr(statistics, category))
            setattr(statistics, category, current + 1)
        statistics.last_rejection_reason = str(reason)
        return SubmissionResult(False, str(reason))

    def flush(self, current_px4_time_s: float) -> int:
        """Process queued samples older than the reorder watermark."""
        current = float(current_px4_time_s)
        if not math.isfinite(current):
            raise ValueError("current PX4 time must be finite")
        watermark = current - self.parameters.reorder_window_s
        processed = 0
        while self._queue and self._queue[0].stamp_s <= watermark + 1.0e-12:
            measurement = heapq.heappop(self._queue)
            if (
                self.filter_stamp_s is not None
                and measurement.stamp_s < self.filter_stamp_s - 1.0e-9
            ):
                self._reject(
                    measurement.source,
                    "late",
                    "measurement older than filter time",
                )
                continue
            self._process(measurement)
            processed += 1
        candidate = self._initial_candidate
        if (
            self.filter_stamp_s is None
            and candidate is not None
            and current - candidate.stamp_s
            > self.parameters.maximum_measurement_age_s
        ):
            self._reject(
                candidate.source,
                "stale",
                "unconfirmed initial measurement expired",
            )
            self._initial_candidate = None
        return processed

    def _process(self, measurement: _QueuedMeasurement) -> None:
        if self.filter_stamp_s is None:
            self._process_initial_measurement(measurement)
            return
        self._predict_in_place(measurement.stamp_s)
        if measurement.kind == "vision":
            indices = _VISION_INDICES
            maximum_nis = self.parameters.vision_nis_gate
        else:
            indices = tuple(range(_STATE_SIZE))
            maximum_nis = self.parameters.odometry_nis_gate

        innovation = measurement.value - self.x[list(indices)]
        yaw_measurement_index = 3 if measurement.kind == "vision" else 6
        innovation[yaw_measurement_index] = wrap_angle(
            innovation[yaw_measurement_index]
        )
        if float(np.linalg.norm(innovation[:3])) > (
            self.parameters.maximum_position_innovation_m
        ):
            self._reject(
                measurement.source, "jump", "position innovation jump"
            )
            return
        if measurement.kind == "odometry":
            if float(np.linalg.norm(innovation[3:6])) > (
                self.parameters.maximum_velocity_innovation_m_s
            ):
                self._reject(
                    measurement.source, "jump", "velocity innovation jump"
                )
                return
            if abs(float(measurement.value[7])) > (
                self.parameters.maximum_yaw_rate_rad_s
            ):
                self._reject(
                    measurement.source, "jump", "yaw-rate physical limit"
                )
                return

        h = np.zeros((len(indices), _STATE_SIZE), dtype=float)
        for row, column in enumerate(indices):
            h[row, column] = 1.0
        innovation_covariance = h @ self.P @ h.T + measurement.covariance
        self._record_innovation_diagnostics(
            measurement.source,
            innovation,
            innovation_covariance,
            yaw_measurement_index,
        )
        try:
            solved = np.linalg.solve(innovation_covariance, innovation)
            nis = float(innovation.T @ solved)
        except np.linalg.LinAlgError:
            self._reject(
                measurement.source,
                "invalid_covariance",
                "singular innovation covariance",
            )
            return
        self.statistics[measurement.source].last_nis = nis
        if not math.isfinite(nis) or nis > maximum_nis:
            self._reject(measurement.source, "nis", "NIS gate exceeded")
            return

        cross_covariance = self.P @ h.T
        kalman_gain = np.linalg.solve(
            innovation_covariance, cross_covariance.T
        ).T
        self.x = self.x + kalman_gain @ innovation
        self.x[_YAW_INDEX] = wrap_angle(self.x[_YAW_INDEX])
        identity = np.eye(_STATE_SIZE, dtype=float)
        residual_transform = identity - kalman_gain @ h
        self.P = (
            residual_transform @ self.P @ residual_transform.T
            + kalman_gain @ measurement.covariance @ kalman_gain.T
        )
        self.P = 0.5 * (self.P + self.P.T)
        self._accept(measurement)

    def _record_innovation_diagnostics(
        self,
        source: str,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
        yaw_measurement_index: int,
    ) -> None:
        """Record finite marginal diagnostics without affecting the filter.

        Position and yaw marginal NIS values use their own covariance blocks.
        They are intentionally diagnostic projections: when the full
        innovation covariance couples position and yaw, the marginals need
        not sum to the joint ``last_nis`` value used by the unchanged gate.
        """

        statistics = self.statistics[source]
        position_innovation = np.asarray(innovation[:3], dtype=float).copy()
        yaw_innovation = float(innovation[yaw_measurement_index])
        statistics.last_position_innovation_enu_m = position_innovation
        statistics.last_position_innovation_norm_m = float(
            np.linalg.norm(position_innovation)
        )
        statistics.last_yaw_innovation_rad = yaw_innovation
        statistics.last_position_marginal_nis = None
        statistics.last_yaw_marginal_nis = None

        try:
            position_covariance = innovation_covariance[:3, :3]
            position_nis = float(
                position_innovation.T
                @ np.linalg.solve(position_covariance, position_innovation)
            )
            yaw_variance = float(
                innovation_covariance[
                    yaw_measurement_index, yaw_measurement_index
                ]
            )
            yaw_nis = yaw_innovation * yaw_innovation / yaw_variance
        except (IndexError, np.linalg.LinAlgError, ZeroDivisionError):
            return
        if math.isfinite(position_nis):
            statistics.last_position_marginal_nis = position_nis
        if math.isfinite(yaw_nis):
            statistics.last_yaw_marginal_nis = yaw_nis

    def _process_initial_measurement(
        self, measurement: _QueuedMeasurement
    ) -> None:
        """Confirm vision once; initialize structured odometry directly."""
        candidate = self._initial_candidate
        if candidate is None:
            if measurement.kind == "odometry":
                self._initialize(measurement)
                return
            self._initial_candidate = measurement
            return

        gap_s = measurement.stamp_s - candidate.stamp_s
        if gap_s > self.parameters.maximum_measurement_age_s:
            self._reject(
                candidate.source,
                "stale",
                "initialization confirmation gap exceeded",
            )
            self._initial_candidate = measurement
            return

        consistent, category, reason, nis = self._initially_consistent(
            candidate, measurement
        )
        self.statistics[candidate.source].last_nis = nis
        if not consistent:
            self._reject(candidate.source, category, reason)
            self._initial_candidate = measurement
            return

        self._initial_candidate = None
        self._initialize(candidate)
        self._process(measurement)

    def _initially_consistent(
        self,
        first: _QueuedMeasurement,
        second: _QueuedMeasurement,
    ) -> tuple[bool, str, str, float | None]:
        first_value, first_covariance = self._pose_projection(first)
        second_value, second_covariance = self._pose_projection(second)
        innovation = second_value - first_value
        innovation[3] = wrap_angle(innovation[3])
        if float(np.linalg.norm(innovation[:3])) > (
            self.parameters.maximum_position_innovation_m
        ):
            return False, "jump", "initial position jump", None

        dt = max(0.0, second.stamp_s - first.stamp_s)
        confirmation_uncertainty = np.zeros((4, 4), dtype=float)
        confirmation_uncertainty[:3, :3] = (
            np.eye(3, dtype=float)
            * self.parameters.initial_velocity_variance
            * dt * dt
        )
        confirmation_uncertainty[3, 3] = (
            self.parameters.initial_yaw_rate_variance * dt * dt
        )
        innovation_covariance = (
            first_covariance
            + second_covariance
            + confirmation_uncertainty
        )
        self._record_innovation_diagnostics(
            first.source,
            innovation,
            innovation_covariance,
            3,
        )
        try:
            nis = float(
                innovation.T
                @ np.linalg.solve(innovation_covariance, innovation)
            )
        except np.linalg.LinAlgError:
            return (
                False,
                "invalid_covariance",
                "singular initialization covariance",
                None,
            )
        if not math.isfinite(nis) or nis > self.parameters.vision_nis_gate:
            return False, "nis", "initialization NIS gate exceeded", nis
        return True, "", "", nis

    @staticmethod
    def _pose_projection(
        measurement: _QueuedMeasurement,
    ) -> tuple[np.ndarray, np.ndarray]:
        if measurement.kind == "vision":
            return measurement.value.copy(), measurement.covariance.copy()
        indices = _VISION_INDICES
        return (
            measurement.value[list(indices)].copy(),
            measurement.covariance[np.ix_(indices, indices)].copy(),
        )

    def _initialize(self, measurement: _QueuedMeasurement) -> None:
        self.x.fill(0.0)
        self.P.fill(0.0)
        if measurement.kind == "vision":
            self.x[:3] = measurement.value[:3]
            self.x[_YAW_INDEX] = wrap_angle(measurement.value[3])
            self.P[np.ix_(_VISION_INDICES, _VISION_INDICES)] = (
                measurement.covariance
            )
            self.P[3, 3] = self.parameters.initial_velocity_variance
            self.P[4, 4] = self.parameters.initial_velocity_variance
            self.P[5, 5] = self.parameters.initial_velocity_variance
            self.P[7, 7] = self.parameters.initial_yaw_rate_variance
        else:
            self.x = measurement.value.copy()
            self.x[_YAW_INDEX] = wrap_angle(self.x[_YAW_INDEX])
            self.P = measurement.covariance.copy()
        self.filter_stamp_s = measurement.stamp_s
        self._accept(measurement)

    def _accept(self, measurement: _QueuedMeasurement) -> None:
        statistics = self.statistics[measurement.source]
        statistics.accepted += 1
        statistics.last_accepted_stamp_s = measurement.stamp_s
        statistics.last_rejection_reason = None

    def _predict_in_place(self, target_stamp_s: float) -> None:
        if self.filter_stamp_s is None:
            return
        dt = float(target_stamp_s) - self.filter_stamp_s
        if dt < -1.0e-9:
            raise ValueError("filter time cannot move backwards")
        if dt <= 0.0:
            self.filter_stamp_s = max(
                self.filter_stamp_s, float(target_stamp_s)
            )
            return
        self.x, self.P = self._predict_copy(self.x, self.P, dt)
        self.filter_stamp_s = float(target_stamp_s)

    def _predict_copy(
        self, state: np.ndarray, covariance: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        transition = np.eye(_STATE_SIZE, dtype=float)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        transition[6, 7] = dt
        predicted_state = transition @ state
        predicted_state[_YAW_INDEX] = wrap_angle(
            predicted_state[_YAW_INDEX]
        )

        process = np.zeros((_STATE_SIZE, _STATE_SIZE), dtype=float)
        linear_block = (
            self.parameters.process_acceleration_variance
            * np.asarray(
                (
                    (dt ** 3 / 3.0, dt ** 2 / 2.0),
                    (dt ** 2 / 2.0, dt),
                ),
                dtype=float,
            )
        )
        for position_index, velocity_index in zip(
            _POSITION_INDICES, _VELOCITY_INDICES
        ):
            process[np.ix_((position_index, velocity_index),
                           (position_index, velocity_index))] = linear_block
        yaw_block = (
            self.parameters.process_yaw_acceleration_variance
            * np.asarray(
                ((dt ** 3 / 3.0, dt ** 2 / 2.0),
                 (dt ** 2 / 2.0, dt)),
                dtype=float,
            )
        )
        process[np.ix_((_YAW_INDEX, _YAW_RATE_INDEX),
                       (_YAW_INDEX, _YAW_RATE_INDEX))] = yaw_block
        predicted_covariance = (
            transition @ covariance @ transition.T + process
        )
        return predicted_state, 0.5 * (
            predicted_covariance + predicted_covariance.T
        )

    def source_fresh(
        self, source: str, current_px4_time_s: float, timeout_s: float
    ) -> bool:
        """Return freshness from the source's last accepted sample stamp."""
        if source not in self.statistics:
            raise ValueError(f"unknown fusion source: {source}")
        stamp = self.statistics[source].last_accepted_stamp_s
        if stamp is None:
            return False
        age = float(current_px4_time_s) - stamp
        return 0.0 <= age <= float(timeout_s)

    def estimate(self, current_px4_time_s: float) -> TargetEstimate:
        """Return current prediction without moving canonical filter time."""
        current = float(current_px4_time_s)
        if not math.isfinite(current):
            raise ValueError("current PX4 time must be finite")
        self.flush(current)
        if self.filter_stamp_s is None:
            return TargetEstimate(False, "uninitialized", current)
        if current < self.filter_stamp_s - 1.0e-9:
            return TargetEstimate(
                False,
                "time_before_filter",
                current,
                filter_stamp_s=self.filter_stamp_s,
            )

        vision_fresh = any(
            self.source_fresh(
                source, current, self.parameters.vision_timeout_s
            )
            for source in self.VISION_SOURCES
        )
        odometry_fresh = self.source_fresh(
            self.ODOMETRY_SOURCE,
            current,
            self.parameters.odometry_timeout_s,
        )
        accepted_stamps = [
            statistics.last_accepted_stamp_s
            for statistics in self.statistics.values()
            if statistics.last_accepted_stamp_s is not None
        ]
        last_accepted = max(accepted_stamps) if accepted_stamps else None
        last_age = None if last_accepted is None else current - last_accepted
        if vision_fresh and odometry_fresh:
            mode = "vision_odometry"
            valid = True
        elif vision_fresh:
            mode = "vision_only"
            valid = True
        elif odometry_fresh:
            mode = "odometry_only"
            valid = True
        elif (
            last_age is not None
            and 0.0 <= last_age <= self.parameters.predict_only_timeout_s
        ):
            mode = "predict_only"
            valid = True
        else:
            mode = "invalid"
            valid = False

        dt = max(0.0, current - self.filter_stamp_s)
        state, covariance = self._predict_copy(self.x, self.P, dt)
        return TargetEstimate(
            valid,
            mode,
            current,
            state[:3].copy(),
            state[3:6].copy(),
            wrap_angle(state[6]),
            float(state[7]),
            covariance.copy(),
            self.filter_stamp_s,
            vision_fresh,
            odometry_fresh,
            last_age,
        )

    def statistics_snapshot(self) -> Mapping[str, SensorStatistics]:
        """Expose read-only-by-convention statistics for ROS status output."""
        return dict(self.statistics)
