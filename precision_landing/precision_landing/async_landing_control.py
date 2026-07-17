"""ROS-independent asynchronous landing-controller execution.

The flight-control thread owns state estimation, mission state, and command
publication.  This module owns only one background controller invocation at a
time.  Every invocation receives a deep-frozen snapshot and returns its result
through an explicit polling boundary; worker threads never touch a ROS node or
publisher.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

import numpy as np

from .landing_controller import (
    LEGACY_MPC_CONTROLLER_TYPE,
    P_FEEDFORWARD_CONTROLLER_TYPE,
    RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
    LandingControlCommand,
    LandingControlInput,
    LandingController,
)


_COMMAND_TOLERANCE = 1.0e-6
_READ_ONLY_ZERO = np.zeros(3, dtype=float)
_READ_ONLY_ZERO.setflags(write=False)


def _freeze_array(value: np.ndarray) -> np.ndarray:
    """Return one independent, finite, read-only array."""
    array = np.asarray(value, dtype=float).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("snapshot arrays must be finite")
    array.setflags(write=False)
    return array


def _freeze_control_input(value: LandingControlInput) -> LandingControlInput:
    """Deep-copy an input and protect every nested ndarray from mutation."""
    if not isinstance(value, LandingControlInput):
        raise TypeError("control_input must be LandingControlInput")
    frozen = LandingControlInput(
        vehicle_position_enu_m=value.vehicle_position_enu_m,
        vehicle_velocity_enu_m_s=value.vehicle_velocity_enu_m_s,
        vehicle_acceleration_enu_m_s2=(
            value.vehicle_acceleration_enu_m_s2
        ),
        vehicle_yaw_enu_rad=value.vehicle_yaw_enu_rad,
        reference_positions_enu_m=value.reference_positions_enu_m,
        reference_velocities_enu_m_s=value.reference_velocities_enu_m_s,
        target_yaw_enu_rad=value.target_yaw_enu_rad,
        horizontal_velocity_limit_m_s=(
            value.horizontal_velocity_limit_m_s
        ),
        vertical_velocity_limit_m_s=value.vertical_velocity_limit_m_s,
        landing_pad_position_enu_m=value.landing_pad_position_enu_m,
        landing_pad_velocity_enu_m_s=value.landing_pad_velocity_enu_m_s,
        landing_pad_acceleration_enu_m_s2=(
            value.landing_pad_acceleration_enu_m_s2
        ),
        target_clearance_m=value.target_clearance_m,
        landing_constraints_enabled=value.landing_constraints_enabled,
        descent_allowed=value.descent_allowed,
        camera_fov_constraint_enabled=(
            value.camera_fov_constraint_enabled
        ),
        relative_descent_speed_m_s=value.relative_descent_speed_m_s,
    )
    for name in (
        "vehicle_position_enu_m",
        "vehicle_velocity_enu_m_s",
        "vehicle_acceleration_enu_m_s2",
        "reference_positions_enu_m",
        "reference_velocities_enu_m_s",
        "landing_pad_position_enu_m",
        "landing_pad_velocity_enu_m_s",
        "landing_pad_acceleration_enu_m_s2",
    ):
        array = getattr(frozen, name)
        if array is not None:
            array.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class LandingCommandLimits:
    """Command-level limits rechecked before cache application."""

    max_horizontal_velocity_m_s: float
    max_ascent_velocity_m_s: float
    max_descent_velocity_m_s: float
    max_total_velocity_m_s: float | None = None

    def __post_init__(self) -> None:
        """Require finite positive limits."""
        for name in (
            "max_horizontal_velocity_m_s",
            "max_ascent_velocity_m_s",
            "max_descent_velocity_m_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        total = self.max_total_velocity_m_s
        if total is not None:
            total = float(total)
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "max_total_velocity_m_s must be finite and positive"
                )
            object.__setattr__(self, "max_total_velocity_m_s", total)


@dataclass(frozen=True)
class LandingSolveSnapshot:
    """Immutable vehicle, target, covariance, time, and constraint state."""

    generation: int
    created_monotonic_s: float
    source_stamp_s: float | None
    phase: str
    time_reset_count: int
    control_input: LandingControlInput
    target_covariance: np.ndarray | None
    solver_deadline_s: float
    constraint_limits: LandingCommandLimits

    def __post_init__(self) -> None:
        """Deep-freeze nested state and validate snapshot metadata."""
        generation = int(self.generation)
        created = float(self.created_monotonic_s)
        reset_count = int(self.time_reset_count)
        deadline = float(self.solver_deadline_s)
        if generation < 1:
            raise ValueError("snapshot generation must be positive")
        if not math.isfinite(created) or created < 0.0:
            raise ValueError("snapshot timestamp must be finite and positive")
        if reset_count < 0:
            raise ValueError("time reset count must be nonnegative")
        if not math.isfinite(deadline) or deadline <= 0.0:
            raise ValueError("solver deadline must be finite and positive")
        if not str(self.phase):
            raise ValueError("snapshot phase must be nonempty")
        source_stamp = self.source_stamp_s
        if source_stamp is not None:
            source_stamp = float(source_stamp)
            if not math.isfinite(source_stamp) or source_stamp < 0.0:
                raise ValueError(
                    "source timestamp must be finite and nonnegative"
                )
        covariance = self.target_covariance
        if covariance is not None:
            covariance = _freeze_array(covariance)
            if (
                covariance.ndim != 2
                or covariance.shape[0] != covariance.shape[1]
            ):
                raise ValueError("target covariance must be square")
            if not np.allclose(
                covariance,
                covariance.T,
                rtol=1.0e-9,
                atol=1.0e-12,
            ):
                raise ValueError("target covariance must be symmetric")
        if not isinstance(self.constraint_limits, LandingCommandLimits):
            raise TypeError("constraint_limits has an invalid type")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "created_monotonic_s", created)
        object.__setattr__(self, "source_stamp_s", source_stamp)
        object.__setattr__(self, "phase", str(self.phase))
        object.__setattr__(self, "time_reset_count", reset_count)
        object.__setattr__(
            self, "control_input", _freeze_control_input(self.control_input)
        )
        object.__setattr__(self, "target_covariance", covariance)
        object.__setattr__(self, "solver_deadline_s", deadline)

    @property
    def vehicle_position_enu_m(self) -> np.ndarray:
        """Return the frozen captured vehicle position."""
        return self.control_input.vehicle_position_enu_m

    @property
    def vehicle_velocity_enu_m_s(self) -> np.ndarray:
        """Return the frozen captured vehicle velocity."""
        return self.control_input.vehicle_velocity_enu_m_s

    @property
    def vehicle_acceleration_enu_m_s2(self) -> np.ndarray:
        """Return the frozen captured vehicle acceleration."""
        return self.control_input.vehicle_acceleration_enu_m_s2

    @property
    def target_position_enu_m(self) -> np.ndarray:
        """Return the pad position or the first absolute reference."""
        pad = self.control_input.landing_pad_position_enu_m
        reference = self.control_input.reference_positions_enu_m
        return pad if pad is not None else (
            reference if reference.ndim == 1 else reference[0]
        )

    @property
    def target_velocity_enu_m_s(self) -> np.ndarray:
        """Return the pad velocity or the first velocity reference."""
        pad = self.control_input.landing_pad_velocity_enu_m_s
        reference = self.control_input.reference_velocities_enu_m_s
        return pad if pad is not None else (
            reference if reference.ndim == 1 else reference[0]
        )

    @property
    def target_acceleration_enu_m_s2(self) -> np.ndarray:
        """Return captured pad acceleration or zero for absolute flight."""
        acceleration = self.control_input.landing_pad_acceleration_enu_m_s2
        return _READ_ONLY_ZERO if acceleration is None else acceleration

    @property
    def target_yaw_enu_rad(self) -> float:
        """Return the captured target yaw."""
        return self.control_input.target_yaw_enu_rad


@dataclass(frozen=True)
class LandingSolveResult:
    """One completed worker invocation and its immutable input identity."""

    snapshot: LandingSolveSnapshot
    command: LandingControlCommand | None
    started_monotonic_s: float
    finished_monotonic_s: float
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate worker timing and result type."""
        started = float(self.started_monotonic_s)
        finished = float(self.finished_monotonic_s)
        if (
            not math.isfinite(started)
            or not math.isfinite(finished)
            or started < self.snapshot.created_monotonic_s
            or finished < started
        ):
            raise ValueError("worker result timestamps are invalid")
        if self.command is not None and not isinstance(
            self.command, LandingControlCommand
        ):
            raise TypeError("worker command has an invalid type")
        if self.command is None and not self.error:
            raise ValueError("missing worker command requires an error")
        object.__setattr__(self, "started_monotonic_s", started)
        object.__setattr__(self, "finished_monotonic_s", finished)
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))

    @property
    def elapsed_s(self) -> float:
        """Return full controller wall time in the worker."""
        return self.finished_monotonic_s - self.started_monotonic_s

    @property
    def queue_age_s(self) -> float:
        """Return snapshot age when its worker invocation began."""
        return self.started_monotonic_s - self.snapshot.created_monotonic_s


class AsyncLandingControllerWorker:
    """Own exactly one background landing-controller invocation."""

    def __init__(
        self,
        controller: LandingController,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create one reusable executor without starting any solve."""
        if not callable(getattr(controller, "compute", None)):
            raise TypeError("controller must provide compute()")
        self._controller = controller
        self._clock = clock
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="landing_mpc"
        )
        self._lock = threading.Lock()
        self._future: Future[LandingSolveResult] | None = None
        self._snapshot: LandingSolveSnapshot | None = None
        self._closed = False

    @staticmethod
    def _compute(
        controller: LandingController,
        snapshot: LandingSolveSnapshot,
        clock: Callable[[], float],
    ) -> LandingSolveResult:
        """Run one controller call and capture all ordinary exceptions."""
        started = float(clock())
        try:
            command = controller.compute(snapshot.control_input)
            if not isinstance(command, LandingControlCommand):
                raise TypeError(
                    "controller returned an invalid command type"
                )
            error = None
        except Exception as exc:
            command = None
            error = f"{type(exc).__name__}: {exc}"
        finished = max(started, float(clock()))
        return LandingSolveResult(
            snapshot,
            command,
            started,
            finished,
            error,
        )

    @property
    def solver_running(self) -> bool:
        """Return whether the single worker call is currently executing."""
        with self._lock:
            return self._future is not None and not self._future.done()

    @property
    def has_outstanding(self) -> bool:
        """Return whether a running or unconsumed result occupies the slot."""
        with self._lock:
            return self._future is not None

    @property
    def queue_age_s(self) -> float | None:
        """Return current age of the outstanding snapshot."""
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return None
        return max(0.0, float(self._clock()) - snapshot.created_monotonic_s)

    def submit(self, snapshot: LandingSolveSnapshot) -> bool:
        """Start one solve only when no running/unconsumed solve exists."""
        if not isinstance(snapshot, LandingSolveSnapshot):
            raise TypeError("snapshot has an invalid type")
        with self._lock:
            if self._closed:
                raise RuntimeError("landing solver worker is closed")
            if self._future is not None:
                return False
            self._snapshot = snapshot
            try:
                self._future = self._executor.submit(
                    self._compute,
                    self._controller,
                    snapshot,
                    self._clock,
                )
            except Exception:
                self._snapshot = None
                raise
        return True

    def take_completed(self) -> LandingSolveResult | None:
        """Consume a completed result without waiting for a running solve."""
        with self._lock:
            future = self._future
            if future is None or not future.done():
                return None
            self._future = None
            self._snapshot = None
        return future.result()

    def close(self) -> None:
        """Stop accepting work and join the single executor thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class SolutionAcceptanceLimits:
    """Freshness and state-change limits for applying a solved command."""

    maximum_solution_age_s: float
    maximum_vehicle_position_change_m: float
    maximum_vehicle_velocity_change_m_s: float
    maximum_vehicle_acceleration_change_m_s2: float
    maximum_target_position_change_m: float
    maximum_target_velocity_change_m_s: float
    maximum_target_acceleration_change_m_s2: float
    maximum_target_yaw_change_rad: float

    def __post_init__(self) -> None:
        """Require finite positive acceptance limits."""
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


def _angle_distance(left: float, right: float) -> float:
    """Return the absolute shortest angular distance."""
    difference = (float(left) - float(right) + math.pi) % (
        2.0 * math.pi
    ) - math.pi
    return abs(difference)


def _norm_change(left: np.ndarray, right: np.ndarray) -> float:
    """Return Euclidean change between two finite vectors."""
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def _command_constraint_rejection(
    snapshot: LandingSolveSnapshot,
    command: LandingControlCommand,
) -> str | None:
    """Recheck command modes, finite values, and velocity limits."""
    fields = (
        (command.position_setpoint_enu_m, command.position_enabled),
        (command.velocity_setpoint_enu_m_s, command.velocity_enabled),
        (command.acceleration_setpoint_enu_m_s2, command.acceleration_enabled),
    )
    for vector, enabled in fields:
        if np.asarray(vector).shape != (3,):
            return "invalid_command_shape"
        if enabled and not np.all(np.isfinite(vector)):
            return "nonfinite_enabled_command"
        if not enabled and not np.all(np.isnan(vector)):
            return "non_nan_disabled_command"
    primary = command.primary_controller_type
    if primary in (LEGACY_MPC_CONTROLLER_TYPE, RELATIVE_LANDING_MPC_CONTROLLER_TYPE):
        if not command.mpc_attempted or command.mpc_success is not True:
            return "solver_failed"
        if primary == RELATIVE_LANDING_MPC_CONTROLLER_TYPE:
            if (
                command.position_enabled
                or not command.velocity_enabled
                or not command.acceleration_enabled
            ):
                return "invalid_mpc_control_mode"
        elif (
            not command.position_enabled
            or not command.velocity_enabled
            or command.acceleration_enabled
        ):
            return "invalid_mpc_control_mode"
    if command.controller_type == P_FEEDFORWARD_CONTROLLER_TYPE and not (
        not command.position_enabled
        and command.velocity_enabled
        and not command.acceleration_enabled
    ):
        return "invalid_p_control_mode"
    if not command.velocity_enabled:
        return "velocity_setpoint_disabled"
    velocity = command.velocity_setpoint_enu_m_s
    limits = snapshot.constraint_limits
    if (
        float(np.linalg.norm(velocity[:2]))
        > limits.max_horizontal_velocity_m_s + _COMMAND_TOLERANCE
    ):
        return "horizontal_velocity_constraint"
    if velocity[2] > limits.max_ascent_velocity_m_s + _COMMAND_TOLERANCE:
        return "ascent_velocity_constraint"
    if velocity[2] < -limits.max_descent_velocity_m_s - _COMMAND_TOLERANCE:
        return "descent_velocity_constraint"
    total_limit = limits.max_total_velocity_m_s
    if (
        total_limit is not None
        and float(np.linalg.norm(velocity)) > total_limit + _COMMAND_TOLERANCE
    ):
        return "total_velocity_constraint"
    if not math.isfinite(command.yaw_enu_rad):
        return "nonfinite_command_yaw"
    return None


def landing_snapshot_change_rejection(
    captured: LandingSolveSnapshot,
    current: LandingSolveSnapshot,
    limits: SolutionAcceptanceLimits,
) -> str | None:
    """Return why a captured command no longer matches current state."""
    prediction_dt_s = (
        current.created_monotonic_s - captured.created_monotonic_s
    )
    if not math.isfinite(prediction_dt_s) or prediction_dt_s < 0.0:
        return "snapshot_time_invalid"
    prediction_dt_squared_s2 = prediction_dt_s * prediction_dt_s
    predicted_vehicle_position = (
        captured.vehicle_position_enu_m
        + captured.vehicle_velocity_enu_m_s * prediction_dt_s
        + 0.5
        * captured.vehicle_acceleration_enu_m_s2
        * prediction_dt_squared_s2
    )
    predicted_target_position = (
        captured.target_position_enu_m
        + captured.target_velocity_enu_m_s * prediction_dt_s
        + 0.5
        * captured.target_acceleration_enu_m_s2
        * prediction_dt_squared_s2
    )
    state_checks = (
        (
            predicted_vehicle_position,
            current.vehicle_position_enu_m,
            limits.maximum_vehicle_position_change_m,
            "vehicle_position_changed",
        ),
        (
            captured.vehicle_velocity_enu_m_s,
            current.vehicle_velocity_enu_m_s,
            limits.maximum_vehicle_velocity_change_m_s,
            "vehicle_velocity_changed",
        ),
        (
            captured.vehicle_acceleration_enu_m_s2,
            current.vehicle_acceleration_enu_m_s2,
            limits.maximum_vehicle_acceleration_change_m_s2,
            "vehicle_acceleration_changed",
        ),
        (
            predicted_target_position,
            current.target_position_enu_m,
            limits.maximum_target_position_change_m,
            "target_position_changed",
        ),
        (
            captured.target_velocity_enu_m_s,
            current.target_velocity_enu_m_s,
            limits.maximum_target_velocity_change_m_s,
            "target_velocity_changed",
        ),
        (
            captured.target_acceleration_enu_m_s2,
            current.target_acceleration_enu_m_s2,
            limits.maximum_target_acceleration_change_m_s2,
            "target_acceleration_changed",
        ),
    )
    for previous, latest, maximum, reason in state_checks:
        if _norm_change(previous, latest) > maximum:
            return reason
    if _angle_distance(
        captured.target_yaw_enu_rad,
        current.target_yaw_enu_rad,
    ) > limits.maximum_target_yaw_change_rad:
        return "target_yaw_changed"
    previous_input = captured.control_input
    current_input = current.control_input
    if (
        previous_input.landing_constraints_enabled
        != current_input.landing_constraints_enabled
        or previous_input.descent_allowed != current_input.descent_allowed
        or previous_input.camera_fov_constraint_enabled
        != current_input.camera_fov_constraint_enabled
    ):
        return "landing_constraints_changed"
    if (
        abs(
            previous_input.relative_descent_speed_m_s
            - current_input.relative_descent_speed_m_s
        )
        > limits.maximum_target_velocity_change_m_s
    ):
        return "landing_descent_reference_changed"
    if captured.constraint_limits != current.constraint_limits or not (
        math.isclose(
            previous_input.horizontal_velocity_limit_m_s,
            current_input.horizontal_velocity_limit_m_s,
            rel_tol=0.0,
            abs_tol=_COMMAND_TOLERANCE,
        )
        and math.isclose(
            previous_input.vertical_velocity_limit_m_s,
            current_input.vertical_velocity_limit_m_s,
            rel_tol=0.0,
            abs_tol=_COMMAND_TOLERANCE,
        )
    ):
        return "command_limits_changed"
    previous_clearance = previous_input.target_clearance_m
    current_clearance = current_input.target_clearance_m
    if (previous_clearance is None) != (current_clearance is None):
        return "landing_clearance_changed"
    if (
        previous_clearance is not None
        and current_clearance is not None
        and abs(previous_clearance - current_clearance)
        > limits.maximum_target_position_change_m
    ):
        return "landing_clearance_changed"
    return None


def validate_landing_solve_result(
    result: LandingSolveResult,
    current_snapshot: LandingSolveSnapshot,
    limits: SolutionAcceptanceLimits,
    *,
    now_monotonic_s: float,
    current_phase: str,
    current_time_reset_count: int,
    last_applied_generation: int,
) -> str | None:
    """Return a rejection reason, or ``None`` for a fresh safe command."""
    now = float(now_monotonic_s)
    snapshot = result.snapshot
    command = result.command
    if not math.isfinite(now) or now < snapshot.created_monotonic_s:
        return "invalid_solution_time"
    if snapshot.generation <= int(last_applied_generation):
        return "out_of_order_solution"
    if snapshot.phase != str(current_phase):
        return "phase_changed"
    if snapshot.time_reset_count != int(current_time_reset_count):
        return "time_reset_changed"
    if current_snapshot.phase != str(current_phase):
        return "current_snapshot_phase_mismatch"
    if current_snapshot.time_reset_count != int(current_time_reset_count):
        return "current_snapshot_time_reset_mismatch"
    if now - snapshot.created_monotonic_s > limits.maximum_solution_age_s:
        return "stale_solution_age"
    if result.elapsed_s > snapshot.solver_deadline_s:
        return "worker_deadline_missed"
    if result.error is not None or command is None:
        return "solver_exception"
    if command.deadline_missed:
        return "solver_deadline_missed"
    if command.solve_time_s > snapshot.solver_deadline_s:
        return "reported_deadline_missed"
    if not command.valid:
        return f"invalid_solver_command: {command.status}"
    change_rejection = landing_snapshot_change_rejection(
        snapshot, current_snapshot, limits
    )
    if change_rejection is not None:
        return change_rejection
    return _command_constraint_rejection(snapshot, command)


def is_stale_solution_rejection(reason: str) -> bool:
    """Return whether a rejection represents stale or changed state."""
    return str(reason) in {
        "invalid_solution_time",
        "out_of_order_solution",
        "phase_changed",
        "time_reset_changed",
        "current_snapshot_phase_mismatch",
        "current_snapshot_time_reset_mismatch",
        "stale_solution_age",
        "vehicle_position_changed",
        "vehicle_velocity_changed",
        "vehicle_acceleration_changed",
        "target_position_changed",
        "target_velocity_changed",
        "target_acceleration_changed",
        "target_yaw_changed",
        "landing_constraints_changed",
        "command_limits_changed",
        "landing_clearance_changed",
    }
