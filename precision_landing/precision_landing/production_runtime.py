"""ROS-independent runtime pieces for the single production landing stack.

Production intentionally has one controller contract: Relative Landing MPC
solved by OSQP, with a moving-deck HOLD on failure.  P/feed-forward remains an
internal transit implementation used before a complete landing-pad context is
available; it is not a selectable landing controller or failure fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import RLock
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .async_landing_control import LandingCommandLimits
from .landing_controller import (
    HOLD_FALLBACK,
    RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
    LandingControlCommand,
    LandingControlInput,
    PFeedforwardLandingController,
    RelativeMpcLandingController,
)
from .osqp_landing_solver import OsqpLandingSolver


PRODUCTION_CONTROLLER_TYPE = RELATIVE_LANDING_MPC_CONTROLLER_TYPE
PRODUCTION_SOLVER_NAME = "osqp"
PRODUCTION_FAILURE_FALLBACK = HOLD_FALLBACK


@dataclass(frozen=True)
class LandingConfig:
    """Immutable view of the parameter values resolved at node startup."""

    _values: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_values",
            MappingProxyType(dict(self._values)),
        )

    def value(self, name: str) -> object:
        return self._values[name]

    def float(self, name: str) -> float:
        return float(self.value(name))

    def int(self, name: str) -> int:
        return int(self.value(name))

    def bool(self, name: str) -> bool:
        return bool(self.value(name))

    def string(self, name: str) -> str:
        return str(self.value(name))


@dataclass(frozen=True)
class TrailerPositionSample:
    """Position-only trailer sample exposed to the production mission.

    The simulator and real vehicle adapters share this contract.  Velocity is
    deliberately absent: approach may use the measured position, while target
    motion for precision flight is estimated from accepted ArUco captures.
    """

    sample_time_s: float
    position_enu_m: np.ndarray
    yaw_enu_rad: float

    def __post_init__(self) -> None:
        stamp = float(self.sample_time_s)
        position = np.asarray(self.position_enu_m, dtype=float)
        yaw = float(self.yaw_enu_rad)
        if (
            not math.isfinite(stamp)
            or stamp <= 0.0
            or position.shape != (3,)
            or not np.all(np.isfinite(position))
            or not math.isfinite(yaw)
        ):
            raise ValueError("trailer position sample must be finite")
        object.__setattr__(self, "sample_time_s", stamp)
        object.__setattr__(self, "position_enu_m", position.copy())
        object.__setattr__(self, "yaw_enu_rad", yaw)


@dataclass(frozen=True)
class ProductionLandingStack:
    """Fixed Relative-OSQP controller and its immutable runtime contract."""

    solver: OsqpLandingSolver
    controller: RelativeMpcLandingController
    solver_deadline_s: float

    @classmethod
    def create(
        cls,
        *,
        solver_parameters: Mapping[str, object],
        solver_deadline_s: float,
        maximum_iterations: int,
        absolute_tolerance: float,
        relative_tolerance: float,
        transit_horizontal_gain: float,
        transit_vertical_gain: float,
    ) -> "ProductionLandingStack":
        """Build the only production solver/controller combination."""
        solver = OsqpLandingSolver(
            **dict(solver_parameters),
            deadline_s=float(solver_deadline_s),
            maximum_iterations=int(maximum_iterations),
            absolute_tolerance=float(absolute_tolerance),
            relative_tolerance=float(relative_tolerance),
        )
        transit = PFeedforwardLandingController(
            float(transit_horizontal_gain),
            float(transit_vertical_gain),
        )
        controller = RelativeMpcLandingController(
            solver,
            transit,
            failure_fallback=PRODUCTION_FAILURE_FALLBACK,
        )
        return cls(solver, controller, float(solver_deadline_s))

    def command_limits(
        self, control_input: LandingControlInput
    ) -> LandingCommandLimits:
        """Return the OSQP bounds captured with an immutable solve snapshot."""
        if not control_input.has_landing_pad_context:
            raise ValueError("OSQP snapshot requires landing-pad context")
        return LandingCommandLimits(
            self.solver.max_horizontal_velocity_m_s,
            self.solver.max_ascent_velocity_m_s,
            self.solver.max_descent_velocity_m_s,
        )


class LatestCommandCache:
    """Small thread-safe handoff between control and heartbeat callbacks."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._command: LandingControlCommand | None = None

    def load(self) -> LandingControlCommand | None:
        """Return the latest immutable command reference."""
        with self._lock:
            return self._command

    def store(self, command: LandingControlCommand) -> None:
        """Atomically replace the cached command."""
        if not isinstance(command, LandingControlCommand):
            raise TypeError("landing command has an invalid type")
        with self._lock:
            self._command = command

    def clear(self) -> None:
        """Atomically revoke the current command without creating a command."""
        with self._lock:
            self._command = None
