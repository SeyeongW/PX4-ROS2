"""ROS-independent landing-controller interfaces and implementations.

The estimator and mission state machine produce one common reference.  This
module turns that reference into a transport-neutral command.  Production uses
the relative-coordinate OSQP landing controller, with the small
P-plus-target-velocity controller retained only for the transit phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import math
import time
from typing import Protocol, Sequence

import numpy as np


RELATIVE_LANDING_MPC_CONTROLLER_TYPE = "relative_landing_mpc"
P_FEEDFORWARD_CONTROLLER_TYPE = "p_feedforward"
HOLD_FALLBACK = "hold"

# Use one immutable future state for all commanded axes and feed-forward
# terms.  The 200 ms extraction remains the measured stable policy for the
# current PX4 velocity-loop latency; a direct 100 ms trial increased the
# 9 m/s catch-up excursion from about 1 m to more than 4 m.
RELATIVE_MPC_COMMAND_HORIZON_INDEX = 1
RELATIVE_MPC_COMMAND_LEAD_MS = 200
RELATIVE_MPC_ACCELERATION_HORIZON_INDEX = (
    RELATIVE_MPC_COMMAND_HORIZON_INDEX
)
RELATIVE_MPC_ACCELERATION_LEAD_MS = RELATIVE_MPC_COMMAND_LEAD_MS


def _three_vector(value: Sequence[float], description: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{description} must be a finite three-vector")
    return vector.copy()


def _reference_array(
    value: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    description: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (3,):
        result = array.copy()
    elif array.ndim == 2 and array.shape[1] == 3 and len(array) > 0:
        result = array.copy()
    else:
        raise ValueError(
            f"{description} must have shape (3,) or a nonempty (N,3)"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{description} must be finite")
    return result


def _horizon_array(value: np.ndarray, description: str) -> np.ndarray:
    """Return a defensive copy of one nonempty finite xyz horizon."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 1:
        raise ValueError(f"{description} must have nonempty shape (N,3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{description} must be finite")
    return array.copy()


def _first_reference(reference: np.ndarray) -> np.ndarray:
    return reference.copy() if reference.ndim == 1 else reference[0].copy()


@dataclass(frozen=True)
class LandingControlInput:
    """Common vehicle/reference state plus optional landing-pad context."""

    vehicle_position_enu_m: np.ndarray
    vehicle_velocity_enu_m_s: np.ndarray
    vehicle_acceleration_enu_m_s2: np.ndarray
    vehicle_yaw_enu_rad: float
    reference_positions_enu_m: np.ndarray
    reference_velocities_enu_m_s: np.ndarray
    target_yaw_enu_rad: float
    horizontal_velocity_limit_m_s: float
    vertical_velocity_limit_m_s: float
    landing_pad_position_enu_m: np.ndarray | None = None
    landing_pad_velocity_enu_m_s: np.ndarray | None = None
    landing_pad_acceleration_enu_m_s2: np.ndarray | None = None
    landing_pad_positions_horizon_enu_m: np.ndarray | None = None
    landing_pad_velocities_horizon_enu_m_s: np.ndarray | None = None
    landing_pad_accelerations_horizon_enu_m_s2: np.ndarray | None = None
    target_clearance_m: float | None = None
    landing_constraints_enabled: bool = False
    descent_allowed: bool = False
    camera_fov_constraint_enabled: bool = True
    relative_descent_speed_m_s: float = 0.0

    def __post_init__(self) -> None:
        """Copy and validate every value at the controller boundary."""
        object.__setattr__(
            self,
            "vehicle_position_enu_m",
            _three_vector(self.vehicle_position_enu_m, "vehicle position"),
        )
        object.__setattr__(
            self,
            "vehicle_velocity_enu_m_s",
            _three_vector(self.vehicle_velocity_enu_m_s, "vehicle velocity"),
        )
        object.__setattr__(
            self,
            "vehicle_acceleration_enu_m_s2",
            _three_vector(
                self.vehicle_acceleration_enu_m_s2,
                "vehicle acceleration",
            ),
        )
        object.__setattr__(
            self,
            "reference_positions_enu_m",
            _reference_array(
                self.reference_positions_enu_m, "reference positions"
            ),
        )
        object.__setattr__(
            self,
            "reference_velocities_enu_m_s",
            _reference_array(
                self.reference_velocities_enu_m_s,
                "reference velocities",
            ),
        )
        for name in ("vehicle_yaw_enu_rad", "target_yaw_enu_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in (
            "horizontal_velocity_limit_m_s",
            "vertical_velocity_limit_m_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        pad_names = (
            "landing_pad_position_enu_m",
            "landing_pad_velocity_enu_m_s",
            "landing_pad_acceleration_enu_m_s2",
        )
        provided = [getattr(self, name) is not None for name in pad_names]
        if any(provided) and not all(provided):
            raise ValueError(
                "landing-pad position, velocity, and acceleration must be "
                "provided together"
            )
        if all(provided):
            for name in pad_names:
                object.__setattr__(
                    self,
                    name,
                    _three_vector(
                        getattr(self, name), name.replace("_", " ")
                    ),
                )
            if self.target_clearance_m is None:
                raise ValueError(
                    "landing-pad context requires a target clearance"
                )
            clearance = float(self.target_clearance_m)
            if not math.isfinite(clearance) or clearance < 0.0:
                raise ValueError(
                    "target clearance must be finite and nonnegative"
                )
            object.__setattr__(self, "target_clearance_m", clearance)
        elif self.target_clearance_m is not None:
            raise ValueError(
                "target clearance cannot be set without landing-pad state"
            )
        horizon_names = (
            "landing_pad_positions_horizon_enu_m",
            "landing_pad_velocities_horizon_enu_m_s",
            "landing_pad_accelerations_horizon_enu_m_s2",
        )
        horizon_provided = [
            getattr(self, name) is not None for name in horizon_names
        ]
        if any(horizon_provided) and not all(horizon_provided):
            raise ValueError(
                "landing-pad position, velocity, and acceleration horizons "
                "must be provided together"
            )
        if all(horizon_provided):
            if not all(provided):
                raise ValueError(
                    "landing-pad horizons require current landing-pad state"
                )
            horizon_lengths: set[int] = set()
            for name in horizon_names:
                horizon = _horizon_array(
                    getattr(self, name), name.replace("_", " ")
                )
                object.__setattr__(self, name, horizon)
                horizon_lengths.add(len(horizon))
            if len(horizon_lengths) != 1:
                raise ValueError(
                    "landing-pad horizons must have the same length"
                )
        constraints_enabled = bool(self.landing_constraints_enabled)
        if constraints_enabled and not all(provided):
            raise ValueError(
                "landing constraints require landing-pad context"
            )
        object.__setattr__(
            self, "landing_constraints_enabled", constraints_enabled
        )
        descent_allowed = bool(self.descent_allowed)
        if descent_allowed and not all(provided):
            raise ValueError("descent permission requires landing-pad context")
        object.__setattr__(self, "descent_allowed", descent_allowed)
        object.__setattr__(
            self,
            "camera_fov_constraint_enabled",
            bool(self.camera_fov_constraint_enabled),
        )
        descent_speed = float(self.relative_descent_speed_m_s)
        if not math.isfinite(descent_speed) or descent_speed < 0.0:
            raise ValueError(
                "relative descent speed must be finite and nonnegative"
            )
        if descent_speed > 0.0 and not descent_allowed:
            raise ValueError(
                "relative descent speed requires descent permission"
            )
        object.__setattr__(
            self, "relative_descent_speed_m_s", descent_speed
        )

    @property
    def has_landing_pad_context(self) -> bool:
        """Return whether the complete relative-landing state is present."""
        return self.landing_pad_position_enu_m is not None

@dataclass(frozen=True)
class LandingControlCommand:
    """Transport-neutral setpoints, control mode, and solver diagnostics."""

    position_setpoint_enu_m: np.ndarray
    velocity_setpoint_enu_m_s: np.ndarray
    acceleration_setpoint_enu_m_s2: np.ndarray
    yaw_enu_rad: float
    position_enabled: bool
    velocity_enabled: bool
    acceleration_enabled: bool
    valid: bool
    degraded: bool
    status: str
    solve_time_s: float
    controller_type: str
    primary_controller_type: str
    iterations: int = 0
    deadline_missed: bool = False
    mpc_attempted: bool = False
    mpc_success: bool | None = None
    fallback_used: str | None = None
    acceleration_enabled_axes: tuple[bool, bool, bool] | None = None
    extraction_policy: str = "direct"

    def __post_init__(self) -> None:
        """Allow NaN only in fields disabled by the cached control mode."""
        fields = (
            (
                "position_setpoint_enu_m",
                self.position_setpoint_enu_m,
                self.position_enabled,
            ),
            (
                "velocity_setpoint_enu_m_s",
                self.velocity_setpoint_enu_m_s,
                self.velocity_enabled,
            ),
        )
        for name, value, enabled in fields:
            vector = np.asarray(value, dtype=float)
            if vector.shape != (3,):
                raise ValueError(f"{name} must be a three-vector")
            if enabled and not np.all(np.isfinite(vector)):
                raise ValueError(f"enabled {name} must be finite")
            if not enabled and not np.all(np.isnan(vector)):
                raise ValueError(f"disabled {name} must contain only NaN")
            object.__setattr__(self, name, vector.copy())
        acceleration = np.asarray(
            self.acceleration_setpoint_enu_m_s2, dtype=float
        )
        if acceleration.shape != (3,):
            raise ValueError(
                "acceleration_setpoint_enu_m_s2 must be a three-vector"
            )
        axes = self.acceleration_enabled_axes
        if axes is None:
            axes = (bool(self.acceleration_enabled),) * 3
        if len(axes) != 3:
            raise ValueError("acceleration_enabled_axes must have three axes")
        axes = tuple(bool(enabled) for enabled in axes)
        if bool(self.acceleration_enabled) != any(axes):
            raise ValueError(
                "acceleration_enabled must match the per-axis enable mask"
            )
        for index, enabled in enumerate(axes):
            if enabled and not math.isfinite(float(acceleration[index])):
                raise ValueError(
                    "enabled acceleration axes must contain finite values"
                )
            if not enabled and not math.isnan(float(acceleration[index])):
                raise ValueError(
                    "disabled acceleration axes must contain NaN"
                )
        object.__setattr__(
            self, "acceleration_setpoint_enu_m_s2", acceleration.copy()
        )
        object.__setattr__(self, "acceleration_enabled_axes", axes)
        yaw = float(self.yaw_enu_rad)
        solve_time = float(self.solve_time_s)
        if not math.isfinite(yaw):
            raise ValueError("command yaw must be finite")
        if not math.isfinite(solve_time) or solve_time < 0.0:
            raise ValueError(
                "command solve time must be finite and nonnegative"
            )
        if int(self.iterations) < 0:
            raise ValueError("command iterations must be nonnegative")
        if not str(self.status):
            raise ValueError("command status must be nonempty")
        if not str(self.controller_type) or not str(
            self.primary_controller_type
        ):
            raise ValueError("command controller type must be nonempty")
        if not str(self.extraction_policy):
            raise ValueError("command extraction policy must be nonempty")
        if self.mpc_success is not None and not self.mpc_attempted:
            raise ValueError(
                "MPC success cannot be set without an MPC attempt"
            )
        object.__setattr__(self, "yaw_enu_rad", yaw)
        object.__setattr__(self, "solve_time_s", solve_time)
        object.__setattr__(self, "iterations", int(self.iterations))

    @classmethod
    def hold(
        cls,
        position_enu_m: Sequence[float],
        yaw_enu_rad: float,
        *,
        valid: bool,
        degraded: bool,
        status: str,
        solve_time_s: float = 0.0,
        primary_controller_type: str = "hold",
        iterations: int = 0,
        deadline_missed: bool = False,
        mpc_attempted: bool = False,
        mpc_success: bool | None = None,
        fallback_used: str | None = None,
    ) -> "LandingControlCommand":
        """Build the existing finite position/zero-velocity HOLD command."""
        return cls(
            _three_vector(position_enu_m, "hold position"),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            float(yaw_enu_rad),
            True,
            True,
            True,
            bool(valid),
            bool(degraded),
            str(status),
            float(solve_time_s),
            "hold",
            str(primary_controller_type),
            int(iterations),
            bool(deadline_missed),
            bool(mpc_attempted),
            mpc_success,
            fallback_used,
        )


class LandingController(ABC):
    """Common landing-controller interface."""

    @abstractmethod
    def compute(
        self, control_input: LandingControlInput
    ) -> LandingControlCommand:
        """Compute one command from the shared state and reference."""


class PFeedforwardLandingController(LandingController):
    """P position feedback plus target-velocity feed-forward."""

    def __init__(self, horizontal_gain: float, vertical_gain: float) -> None:
        """Configure separate horizontal and vertical proportional gains."""
        horizontal = float(horizontal_gain)
        vertical = float(vertical_gain)
        if (
            not math.isfinite(horizontal)
            or not math.isfinite(vertical)
            or horizontal <= 0.0
            or vertical <= 0.0
        ):
            raise ValueError("P gains must be finite and positive")
        self.horizontal_gain = horizontal
        self.vertical_gain = vertical

    def compute(
        self, control_input: LandingControlInput
    ) -> LandingControlCommand:
        """Compute velocity feed-forward minus gain times position error."""
        started = time.monotonic()
        reference_position = _first_reference(
            control_input.reference_positions_enu_m
        )
        reference_velocity = _first_reference(
            control_input.reference_velocities_enu_m_s
        )
        position_error = (
            control_input.vehicle_position_enu_m - reference_position
        )
        gains = np.asarray(
            (
                self.horizontal_gain,
                self.horizontal_gain,
                self.vertical_gain,
            ),
            dtype=float,
        )
        velocity = reference_velocity - gains * position_error
        horizontal_norm = float(np.linalg.norm(velocity[:2]))
        horizontal_limit = control_input.horizontal_velocity_limit_m_s
        if horizontal_norm > horizontal_limit:
            velocity[:2] *= horizontal_limit / horizontal_norm
        velocity[2] = float(
            np.clip(
                velocity[2],
                -control_input.vertical_velocity_limit_m_s,
                control_input.vertical_velocity_limit_m_s,
            )
        )
        elapsed = time.monotonic() - started
        disabled = np.full(3, np.nan, dtype=float)
        return LandingControlCommand(
            disabled,
            velocity,
            disabled,
            control_input.target_yaw_enu_rad,
            False,
            True,
            False,
            True,
            False,
            "p_feedforward",
            elapsed,
            P_FEEDFORWARD_CONTROLLER_TYPE,
            P_FEEDFORWARD_CONTROLLER_TYPE,
        )


class _RelativeMpcSolver(Protocol):
    def solve(
        self,
        vehicle_position_enu_m: Sequence[float],
        vehicle_velocity_enu_m_s: Sequence[float],
        vehicle_acceleration_enu_m_s2: Sequence[float],
        landing_pad_position_enu_m: Sequence[float],
        landing_pad_velocity_enu_m_s: Sequence[float],
        landing_pad_acceleration_enu_m_s2: Sequence[float],
        target_clearance_m: float,
        landing_pad_yaw_enu_rad: float,
        *,
        enforce_landing_constraints: bool,
        descent_allowed: bool,
        camera_fov_constraint_enabled: bool = True,
        relative_descent_speed_m_s: float = 0.0,
        landing_pad_positions_horizon_enu_m: np.ndarray | None = None,
        landing_pad_velocities_horizon_enu_m_s: np.ndarray | None = None,
        landing_pad_accelerations_horizon_enu_m_s2: np.ndarray | None = None,
    ) -> object:
        """Return a RelativeLandingMPCResult-compatible object."""


class RelativeMpcLandingController(LandingController):
    """Landing-only relative MPC with explicit P control during transit."""

    def __init__(
        self,
        solver: _RelativeMpcSolver,
        transit_controller: PFeedforwardLandingController,
        *,
        failure_fallback: str = HOLD_FALLBACK,
    ) -> None:
        """Configure the relative solver and fixed HOLD failure policy."""
        if failure_fallback != HOLD_FALLBACK:
            raise ValueError("production MPC failure fallback must be hold")
        self.solver = solver
        self.transit_controller = transit_controller

    def compute(
        self, control_input: LandingControlInput
    ) -> LandingControlCommand:
        """Compute relative landing control or explicit P transit control."""
        if not control_input.has_landing_pad_context:
            transit = self.transit_controller.compute(control_input)
            return replace(transit, status="p_feedforward_transit")

        pad_position = control_input.landing_pad_position_enu_m
        pad_velocity = control_input.landing_pad_velocity_enu_m_s
        pad_acceleration = control_input.landing_pad_acceleration_enu_m_s2
        clearance = control_input.target_clearance_m
        if (
            pad_position is None
            or pad_velocity is None
            or pad_acceleration is None
            or clearance is None
        ):
            raise RuntimeError("incomplete landing-pad context")
        solver_started = time.monotonic()
        try:
            result = self.solver.solve(
                control_input.vehicle_position_enu_m,
                control_input.vehicle_velocity_enu_m_s,
                control_input.vehicle_acceleration_enu_m_s2,
                pad_position,
                pad_velocity,
                pad_acceleration,
                clearance,
                control_input.target_yaw_enu_rad,
                enforce_landing_constraints=(
                    control_input.landing_constraints_enabled
                ),
                descent_allowed=control_input.descent_allowed,
                camera_fov_constraint_enabled=(
                    control_input.camera_fov_constraint_enabled
                ),
                relative_descent_speed_m_s=(
                    control_input.relative_descent_speed_m_s
                ),
                landing_pad_positions_horizon_enu_m=(
                    control_input.landing_pad_positions_horizon_enu_m
                ),
                landing_pad_velocities_horizon_enu_m_s=(
                    control_input.landing_pad_velocities_horizon_enu_m_s
                ),
                landing_pad_accelerations_horizon_enu_m_s2=(
                    control_input.landing_pad_accelerations_horizon_enu_m_s2
                ),
            )
            solve_time = float(getattr(result, "solve_time_s"))
            iterations = int(getattr(result, "iterations"))
            deadline_missed = bool(getattr(result, "deadline_missed"))
            message = str(getattr(result, "message"))
            success = bool(getattr(result, "success"))
            if not math.isfinite(solve_time) or solve_time < 0.0:
                raise ValueError("solver returned an invalid solve time")
            if iterations < 0:
                raise ValueError("solver returned invalid iterations")
            if success and deadline_missed:
                clear_warm_start = getattr(
                    self.solver, "clear_warm_start", None
                )
                if callable(clear_warm_start):
                    try:
                        clear_warm_start()
                    except Exception:
                        pass
                success = False
                message = f"deadline-missed solver result: {message}"
        except Exception as exc:  # fail closed at the flight-control boundary
            clear_warm_start = getattr(
                self.solver, "clear_warm_start", None
            )
            if callable(clear_warm_start):
                try:
                    clear_warm_start()
                except Exception:
                    pass
            result = None
            solve_time = max(0.0, time.monotonic() - solver_started)
            iterations = 0
            deadline_missed = False
            message = f"solver exception: {type(exc).__name__}: {exc}"
            success = False
        if success:
            try:
                if result is None:
                    raise RuntimeError(
                        "successful solver result is unavailable"
                    )
                positions = np.asarray(
                    getattr(result, "positions_m"), dtype=float
                )
                velocities = np.asarray(
                    getattr(result, "velocities_m_s"), dtype=float
                )
                accelerations = np.asarray(
                    getattr(result, "accelerations_m_s2"), dtype=float
                )
                horizons = (positions, velocities, accelerations)
                if any(
                    array.ndim != 2
                    or array.shape[1] != 3
                    or len(array) < 1
                    or not np.all(np.isfinite(array))
                    for array in horizons
                ):
                    raise ValueError(
                        "solver returned invalid command horizons"
                    )
            except Exception as exc:
                clear_warm_start = getattr(
                    self.solver, "clear_warm_start", None
                )
                if callable(clear_warm_start):
                    try:
                        clear_warm_start()
                    except Exception:
                        pass
                message = (
                    "solver result rejected: "
                    f"{type(exc).__name__}: {exc}"
                )
                success = False
        if success:
            disabled = np.full(3, np.nan, dtype=float)
            # Extract velocity and horizontal acceleration from the same
            # future state. Mixing t+200 ms velocity with t+100 ms
            # acceleration applies mutually inconsistent turn geometry at
            # the PX4 boundary and caused the high-speed lateral oscillation.
            command_index = min(
                RELATIVE_MPC_COMMAND_HORIZON_INDEX,
                len(velocities) - 1,
            )
            velocity_command = velocities[command_index].copy()
            # The no-descent viability row intentionally admits
            # a few cm/s beyond the nominal bound while arresting an already
            # measured descent.  Clamp only that numerical recovery tail at the
            # transport boundary so a safe OSQP result is not discarded for a
            # sub-sample vertical overshoot.
            velocity_command[2] = float(
                np.clip(
                    velocity_command[2],
                    -float(
                        getattr(
                            self.solver,
                            "max_descent_velocity_m_s",
                            control_input.vertical_velocity_limit_m_s,
                        )
                    ),
                    float(
                        getattr(
                            self.solver,
                            "max_ascent_velocity_m_s",
                            control_input.vertical_velocity_limit_m_s,
                        )
                    ),
                )
            )
            acceleration_index = min(
                RELATIVE_MPC_ACCELERATION_HORIZON_INDEX,
                len(accelerations) - 1,
            )
            acceleration_command = accelerations[
                acceleration_index
            ].copy()
            acceleration_command[2] = np.nan
            extraction_policy = (
                f"mpc_horizon_lead_{RELATIVE_MPC_COMMAND_LEAD_MS}ms"
            )
            if (
                float(clearance) <= 1.0e-6
                and control_input.descent_allowed
                and control_input.relative_descent_speed_m_s > 0.0
            ):
                # At physical deck contact the measured vertical state cannot
                # follow the optimizer's first dynamically reachable sample.
                # Re-solving from that same blocked state therefore kept
                # publishing only a few cm/s of descent and left PX4 at hover
                # thrust indefinitely.  The zero-clearance context is used
                # only by the guarded terminal-contact path (fresh relative
                # height, low relative vertical speed, bounded XY tracking).
                # Keep the OSQP horizontal command and request its already
                # bounded terminal relative descent directly so PX4 unloads
                # the motors and its normal land detector can assert contact.
                velocity_command[2] = float(
                    pad_velocity[2]
                    - control_input.relative_descent_speed_m_s
                )
                pad_velocity_horizon = (
                    control_input.landing_pad_velocities_horizon_enu_m_s
                )
                if pad_velocity_horizon is not None:
                    # The selected MPC sample is 200 ms in the future.  Its
                    # absolute XY velocity therefore contains a future
                    # tangent on a turning deck (0.324 m/s different from
                    # now at 9 m/s on the 50 m circle).  At contact, apply
                    # only the optimizer's relative-velocity correction at
                    # the current measured pad velocity.  Static and linear
                    # motion are unchanged because both pad velocities are
                    # identical in those cases.
                    future_pad_velocity = np.asarray(
                        pad_velocity_horizon[command_index], dtype=float
                    )
                    velocity_command[:2] = (
                        pad_velocity[:2]
                        + velocity_command[:2]
                        - future_pad_velocity[:2]
                    )
                extraction_policy = "contact_settle_vertical_override"
            return LandingControlCommand(
                disabled,
                velocity_command,
                acceleration_command,
                control_input.target_yaw_enu_rad,
                False,
                True,
                True,
                True,
                False,
                f"relative_landing_mpc[{extraction_policy}]: {message}",
                solve_time,
                RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
                RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
                iterations,
                deadline_missed,
                True,
                True,
                acceleration_enabled_axes=(True, True, False),
                extraction_policy=extraction_policy,
            )

        return LandingControlCommand.hold(
            control_input.vehicle_position_enu_m,
            control_input.vehicle_yaw_enu_rad,
            valid=False,
            degraded=True,
            status=f"relative_landing_mpc_failed_hold: {message}",
            solve_time_s=solve_time,
            primary_controller_type=RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
            iterations=iterations,
            deadline_missed=deadline_missed,
            mpc_attempted=True,
            mpc_success=False,
            fallback_used=HOLD_FALLBACK,
        )
