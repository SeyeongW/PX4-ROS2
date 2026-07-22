"""Fail-closed landing descent and touchdown safety gates.

The helpers in this module deliberately have no ROS dependencies.  Callers
must provide one coherent, timestamped snapshot.  A :class:`TouchdownGate`
never commands disarm; it only reports whether independent geometric and PX4
land-detector evidence has remained continuously valid for the configured
dwell period.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_COVARIANCE_RELATIVE_TOLERANCE = 1.0e-9
_COVARIANCE_ABSOLUTE_TOLERANCE = 1.0e-12
_COMPARISON_TOLERANCE = 1.0e-9


def _finite_float(value: object) -> float | None:
    """Return a finite float, excluding booleans, or ``None``."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _strict_bool(value: object) -> bool | None:
    """Accept only real Boolean values, including NumPy Boolean scalars."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


@dataclass(frozen=True)
class DescentPermissionLimits:
    """Limits used to decide whether any downward command is permissible."""

    maximum_vision_age_s: float
    maximum_odometry_age_s: float
    maximum_position_variance_m2: float
    maximum_relative_horizontal_speed_m_s: float
    maximum_relative_height_m: float

    def __post_init__(self) -> None:
        """Require finite, positive safety bounds."""
        for name in (
            "maximum_vision_age_s",
            "maximum_odometry_age_s",
            "maximum_position_variance_m2",
            "maximum_relative_horizontal_speed_m_s",
            "maximum_relative_height_m",
        ):
            value = _finite_float(getattr(self, name))
            if value is None or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class DescentPermissionInput:
    """One immutable snapshot of all descent-permission evidence."""

    target_estimate_valid: bool
    vision_age_s: float | None
    odometry_age_s: float | None
    position_covariance_m2: np.ndarray
    horizontal_error_m: float
    funnel_radius_m: float
    relative_horizontal_speed_m_s: float
    relative_height_m: float | None
    solver_command_valid: bool

    def __post_init__(self) -> None:
        """Deep-copy covariance so later producer mutation cannot alter it."""
        covariance = np.asarray(
            self.position_covariance_m2, dtype=float
        ).copy()
        covariance.setflags(write=False)
        object.__setattr__(self, "position_covariance_m2", covariance)


@dataclass(frozen=True)
class DescentPermissionDecision:
    """Fail-closed descent decision and deterministic diagnostic reasons."""

    allowed: bool
    reason: str
    failed_checks: tuple[str, ...]

    def __post_init__(self) -> None:
        """Keep the summary consistent with the complete failure list."""
        if self.allowed:
            if self.reason != "allowed" or self.failed_checks:
                raise ValueError("allowed decisions cannot contain failures")
        elif not self.failed_checks or self.reason != self.failed_checks[0]:
            raise ValueError("blocked decisions require ordered failures")


def _covariance_rejection(
    covariance: np.ndarray,
    maximum_variance_m2: float,
) -> str | None:
    """Validate a 3-D position covariance and apply its variance limit."""
    if covariance.shape != (3, 3):
        return "position_covariance_invalid"
    if not np.all(np.isfinite(covariance)):
        return "position_covariance_invalid"
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=_COVARIANCE_RELATIVE_TOLERANCE,
        atol=_COVARIANCE_ABSOLUTE_TOLERANCE,
    ):
        return "position_covariance_invalid"
    diagonal = np.diag(covariance)
    if np.any(diagonal < 0.0):
        return "position_covariance_invalid"
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues < -_COVARIANCE_ABSOLUTE_TOLERANCE):
        return "position_covariance_invalid"
    # Correlation can make the largest principal-axis variance exceed every
    # diagonal term.  Gate the covariance ellipsoid, not only world axes.
    if float(np.max(eigenvalues)) > maximum_variance_m2:
        return "position_covariance_exceeded"
    return None


def evaluate_descent_permission(
    snapshot: DescentPermissionInput,
    limits: DescentPermissionLimits,
) -> DescentPermissionDecision:
    """Evaluate every descent gate and return all failures in fixed order.

    Both vision and trailer odometry must be fresh while descending.  The
    production profile has no lidar input, so height must be a finite,
    plausible vehicle-to-fused-target relative height.
    """
    if not isinstance(snapshot, DescentPermissionInput):
        raise TypeError("snapshot must be DescentPermissionInput")
    if not isinstance(limits, DescentPermissionLimits):
        raise TypeError("limits must be DescentPermissionLimits")

    failures: list[str] = []
    target_valid = _strict_bool(snapshot.target_estimate_valid)
    if target_valid is not True:
        failures.append("target_estimate_invalid")

    vision_age = _finite_float(snapshot.vision_age_s)
    if (
        vision_age is None
        or vision_age < 0.0
        or vision_age > limits.maximum_vision_age_s
    ):
        failures.append("vision_stale")

    odometry_age = _finite_float(snapshot.odometry_age_s)
    if (
        odometry_age is None
        or odometry_age < 0.0
        or odometry_age > limits.maximum_odometry_age_s
    ):
        failures.append("odometry_stale")

    covariance_rejection = _covariance_rejection(
        snapshot.position_covariance_m2,
        limits.maximum_position_variance_m2,
    )
    if covariance_rejection is not None:
        failures.append(covariance_rejection)

    horizontal_error = _finite_float(snapshot.horizontal_error_m)
    funnel_radius = _finite_float(snapshot.funnel_radius_m)
    if horizontal_error is None or horizontal_error < 0.0:
        failures.append("horizontal_error_invalid")
    elif funnel_radius is None or funnel_radius < 0.0:
        failures.append("funnel_radius_invalid")
    elif horizontal_error > funnel_radius + _COMPARISON_TOLERANCE:
        failures.append("outside_landing_funnel")

    relative_speed = _finite_float(
        snapshot.relative_horizontal_speed_m_s
    )
    if relative_speed is None or relative_speed < 0.0:
        failures.append("relative_horizontal_speed_invalid")
    elif (
        relative_speed
        > limits.maximum_relative_horizontal_speed_m_s
        + _COMPARISON_TOLERANCE
    ):
        failures.append("relative_horizontal_speed_exceeded")

    relative_height = _finite_float(snapshot.relative_height_m)
    relative_height_valid = bool(
        relative_height is not None
        and 0.0 <= relative_height <= limits.maximum_relative_height_m
    )
    if not relative_height_valid:
        failures.append("height_source_invalid")

    solver_valid = _strict_bool(snapshot.solver_command_valid)
    if solver_valid is not True:
        failures.append("solver_command_invalid")

    if failures:
        return DescentPermissionDecision(False, failures[0], tuple(failures))
    return DescentPermissionDecision(True, "allowed", ())


@dataclass(frozen=True)
class TouchdownGateLimits:
    """Thresholds for continuous, multi-signal touchdown confirmation."""

    maximum_deck_distance_m: float
    maximum_relative_vertical_speed_m_s: float
    dwell_time_s: float
    maximum_sample_age_s: float
    maximum_future_skew_s: float
    maximum_inter_sample_gap_s: float
    confirmation_timeout_s: float
    minimum_consecutive_samples: int = 2

    def __post_init__(self) -> None:
        """Validate positive timing and geometry limits."""
        for name in (
            "maximum_deck_distance_m",
            "maximum_relative_vertical_speed_m_s",
            "dwell_time_s",
            "maximum_sample_age_s",
            "maximum_inter_sample_gap_s",
            "confirmation_timeout_s",
        ):
            value = _finite_float(getattr(self, name))
            if value is None or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        future_skew = _finite_float(self.maximum_future_skew_s)
        if future_skew is None or future_skew < 0.0:
            raise ValueError(
                "maximum_future_skew_s must be finite and nonnegative"
            )
        object.__setattr__(self, "maximum_future_skew_s", future_skew)
        timeout = float(self.confirmation_timeout_s)
        if timeout <= float(self.dwell_time_s):
            raise ValueError("confirmation timeout must exceed dwell time")
        samples = int(self.minimum_consecutive_samples)
        if samples < 2 or samples != self.minimum_consecutive_samples:
            raise ValueError("minimum_consecutive_samples must be >= 2")
        object.__setattr__(self, "minimum_consecutive_samples", samples)


@dataclass(frozen=True)
class TouchdownGateInput:
    """One coherent touchdown observation in a monotonic time domain."""

    now_s: float
    sample_stamp_s: float
    deck_distance_m: float
    relative_vertical_speed_m_s: float
    ground_contact: bool
    landed: bool
    at_rest: bool
    reset_epoch: int = 0


@dataclass(frozen=True)
class TouchdownGateDecision:
    """Touchdown-gate state returned without causing any flight action."""

    confirmed: bool
    conditions_met: bool
    timed_out: bool
    reason: str
    dwell_elapsed_s: float
    consecutive_samples: int


class TouchdownGate:
    """Require fresh continuous evidence before reporting touchdown.

    Distinct sample timestamps are required, so repeatedly polling one
    momentary ``ground_contact`` value cannot satisfy the dwell.  Confirmation
    is not latched: any invalid, stale, reset, or subsequently false input
    immediately produces a fail-closed decision.
    """

    def __init__(self, limits: TouchdownGateLimits) -> None:
        """Create an empty touchdown-evidence accumulator."""
        if not isinstance(limits, TouchdownGateLimits):
            raise TypeError("limits must be TouchdownGateLimits")
        self.limits = limits
        self._last_now_s: float | None = None
        self._last_sample_stamp_s: float | None = None
        self._candidate_since_stamp_s: float | None = None
        self._last_qualifying_stamp_s: float | None = None
        self._monitoring_started_s: float | None = None
        self._reset_epoch: int | None = None
        self._consecutive_samples = 0
        self._timed_out = False

    def reset(self, reset_epoch: int | None = None) -> None:
        """Clear all accumulated evidence, for example on phase entry/reset."""
        if reset_epoch is not None:
            epoch = self._valid_epoch(reset_epoch)
            if epoch is None:
                raise ValueError("reset_epoch must be a nonnegative integer")
            self._reset_epoch = epoch
        else:
            self._reset_epoch = None
        self._last_now_s = None
        self._last_sample_stamp_s = None
        self._candidate_since_stamp_s = None
        self._last_qualifying_stamp_s = None
        self._monitoring_started_s = None
        self._consecutive_samples = 0
        self._timed_out = False

    @staticmethod
    def _valid_epoch(value: object) -> int | None:
        """Return a nonnegative integral epoch, excluding Boolean values."""
        if isinstance(value, (bool, np.bool_)):
            return None
        try:
            epoch = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return None
        try:
            unchanged = float(value) == float(epoch)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return None
        return epoch if epoch >= 0 and unchanged else None

    def _clear_candidate(self) -> None:
        """Discard dwell evidence without restarting the approach timeout."""
        self._candidate_since_stamp_s = None
        self._last_qualifying_stamp_s = None
        self._consecutive_samples = 0

    def _decision(
        self,
        reason: str,
        *,
        confirmed: bool = False,
        conditions_met: bool = False,
        dwell_elapsed_s: float = 0.0,
    ) -> TouchdownGateDecision:
        """Build a normalized immutable decision."""
        return TouchdownGateDecision(
            confirmed=confirmed,
            conditions_met=conditions_met,
            timed_out=self._timed_out,
            reason=reason,
            dwell_elapsed_s=max(0.0, float(dwell_elapsed_s)),
            consecutive_samples=self._consecutive_samples,
        )

    def _time_reset_decision(
        self,
        now_s: float,
        sample_stamp_s: float,
        epoch: int,
    ) -> TouchdownGateDecision:
        """Fail this sample and establish a clean post-reset time baseline."""
        self._clear_candidate()
        self._last_now_s = now_s
        self._last_sample_stamp_s = sample_stamp_s
        self._monitoring_started_s = now_s
        self._reset_epoch = epoch
        self._timed_out = False
        return self._decision("time_reset")

    def update(self, observation: TouchdownGateInput) -> TouchdownGateDecision:
        """Consume one observation and return a fail-closed gate decision."""
        if not isinstance(observation, TouchdownGateInput):
            raise TypeError("observation must be TouchdownGateInput")
        now_s = _finite_float(observation.now_s)
        sample_stamp_s = _finite_float(observation.sample_stamp_s)
        epoch = self._valid_epoch(observation.reset_epoch)
        if (
            now_s is None
            or sample_stamp_s is None
            or now_s < 0.0
            or sample_stamp_s < 0.0
            or epoch is None
        ):
            self._clear_candidate()
            return self._decision("invalid_timestamp")

        if self._reset_epoch is None:
            self._reset_epoch = epoch
        elif epoch != self._reset_epoch:
            return self._time_reset_decision(now_s, sample_stamp_s, epoch)
        if self._last_now_s is not None and now_s < self._last_now_s:
            return self._time_reset_decision(now_s, sample_stamp_s, epoch)
        self._last_now_s = now_s
        if self._monitoring_started_s is None:
            self._monitoring_started_s = now_s

        if self._timed_out or (
            now_s - self._monitoring_started_s
            >= self.limits.confirmation_timeout_s
        ):
            self._timed_out = True
            self._clear_candidate()
            return self._decision("confirmation_timeout")

        if (
            self._last_sample_stamp_s is not None
            and sample_stamp_s < self._last_sample_stamp_s
        ):
            self._clear_candidate()
            return self._decision("out_of_order_sample")
        duplicate = (
            self._last_sample_stamp_s is not None
            and sample_stamp_s == self._last_sample_stamp_s
        )

        sample_age_s = now_s - sample_stamp_s
        if sample_age_s < -self.limits.maximum_future_skew_s:
            self._clear_candidate()
            return self._decision("future_sample")
        if sample_age_s > self.limits.maximum_sample_age_s:
            self._clear_candidate()
            return self._decision("stale_sample")
        if not duplicate:
            # Only a time-valid observation may advance the ordering baseline;
            # one corrupted future stamp must not poison all later evidence.
            self._last_sample_stamp_s = sample_stamp_s

        deck_distance = _finite_float(observation.deck_distance_m)
        vertical_speed = _finite_float(
            observation.relative_vertical_speed_m_s
        )
        if deck_distance is None or deck_distance < 0.0:
            self._clear_candidate()
            return self._decision("deck_distance_invalid")
        if vertical_speed is None:
            self._clear_candidate()
            return self._decision("relative_vertical_speed_invalid")
        if (
            deck_distance
            > self.limits.maximum_deck_distance_m + _COMPARISON_TOLERANCE
        ):
            self._clear_candidate()
            return self._decision("deck_distance_exceeded")
        if (
            abs(vertical_speed)
            > self.limits.maximum_relative_vertical_speed_m_s
            + _COMPARISON_TOLERANCE
        ):
            self._clear_candidate()
            return self._decision("relative_vertical_speed_exceeded")

        for name in ("ground_contact", "landed", "at_rest"):
            signal = _strict_bool(getattr(observation, name))
            if signal is None:
                self._clear_candidate()
                return self._decision(f"{name}_invalid")
            if not signal:
                self._clear_candidate()
                return self._decision(f"{name}_false")

        if duplicate:
            elapsed = 0.0
            if self._candidate_since_stamp_s is not None:
                elapsed = sample_stamp_s - self._candidate_since_stamp_s
            return self._decision(
                "duplicate_sample",
                conditions_met=True,
                dwell_elapsed_s=elapsed,
            )

        gap_restarted = False
        if self._candidate_since_stamp_s is None:
            self._candidate_since_stamp_s = sample_stamp_s
            self._consecutive_samples = 1
        elif (
            self._last_qualifying_stamp_s is None
            or sample_stamp_s - self._last_qualifying_stamp_s
            > self.limits.maximum_inter_sample_gap_s
        ):
            self._candidate_since_stamp_s = sample_stamp_s
            self._consecutive_samples = 1
            gap_restarted = True
        else:
            self._consecutive_samples += 1
        self._last_qualifying_stamp_s = sample_stamp_s
        dwell_elapsed_s = sample_stamp_s - self._candidate_since_stamp_s
        if (
            dwell_elapsed_s + _COMPARISON_TOLERANCE
            >= self.limits.dwell_time_s
            and self._consecutive_samples
            >= self.limits.minimum_consecutive_samples
        ):
            return self._decision(
                "touchdown_confirmed",
                confirmed=True,
                conditions_met=True,
                dwell_elapsed_s=dwell_elapsed_s,
            )
        reason = "qualifying_sample_gap" if gap_restarted else "dwell_pending"
        return self._decision(
            reason,
            conditions_met=True,
            dwell_elapsed_s=dwell_elapsed_s,
        )
