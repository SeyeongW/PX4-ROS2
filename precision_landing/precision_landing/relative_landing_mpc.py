"""Shared result and constraint helpers for the production relative MPC.

The live solver is :class:`OsqpLandingSolver`.  This module intentionally
contains only the transport-neutral result schema and the no-descent recovery
helper shared by that solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


_CONSTRAINT_TOLERANCE = 1.0e-5
# The production descent limit is 0.7 m/s. With the configured jerk/vertical
# acceleration bounds, a previously authorized descent can require roughly
# 0.7 s and 0.4 m to arrest. This envelope permits only that fastest reachable
# braking trajectory after descent permission is revoked.
_NO_DESCENT_RECOVERY_TIME_S = 0.80
_NO_DESCENT_RECOVERY_DISTANCE_M = 0.50


def _finite_three_vector(
    value: Sequence[float], description: str
) -> np.ndarray:
    """Return a defensive copy of one finite xyz vector."""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{description} must be a finite three-vector")
    return vector.copy()


def _recoverable_no_descent_velocity_floor(
    free_relative_velocities: np.ndarray,
    free_absolute_accelerations: np.ndarray,
    position_map: np.ndarray,
    velocity_map: np.ndarray,
    *,
    current_relative_position_z_m: float,
    current_relative_velocity_z_m_s: float,
    dt_s: float,
    max_vertical_acceleration_m_s2: float,
    max_jerk_m_s3: float,
) -> np.ndarray:
    """Return the fastest dynamically reachable arrest of existing descent.

    A hard ``relative_vz >= 0`` bound is infeasible when a disturbance has
    already produced a small downward velocity that finite jerk cannot cancel
    in one sample.  This floor applies maximum admissible upward jerk until
    the relative descent can be arrested, then returns to the strict zero
    lower bound.  It therefore admits no additional avoidable descent.
    """
    relative_velocities = np.asarray(
        free_relative_velocities, dtype=float
    )
    absolute_accelerations = np.asarray(
        free_absolute_accelerations, dtype=float
    )
    position_mapping = np.asarray(position_map, dtype=float)
    velocity_mapping = np.asarray(velocity_map, dtype=float)
    if (
        relative_velocities.ndim != 2
        or relative_velocities.shape[1] != 3
        or absolute_accelerations.shape != relative_velocities.shape
        or position_mapping.shape
        != (relative_velocities.shape[0], relative_velocities.shape[0])
        or velocity_mapping.shape
        != (relative_velocities.shape[0], relative_velocities.shape[0])
        or not np.all(np.isfinite(relative_velocities))
        or not np.all(np.isfinite(absolute_accelerations))
        or not np.all(np.isfinite(position_mapping))
        or not np.all(np.isfinite(velocity_mapping))
    ):
        raise ValueError("no-descent recovery inputs must be finite horizons")
    current_relative_velocity = float(current_relative_velocity_z_m_s)
    current_relative_position = float(current_relative_position_z_m)
    dt = float(dt_s)
    acceleration_limit = float(max_vertical_acceleration_m_s2)
    jerk_limit = float(max_jerk_m_s3)
    if (
        not math.isfinite(dt)
        or dt <= 0.0
        or not math.isfinite(current_relative_position)
        or not math.isfinite(current_relative_velocity)
        or not math.isfinite(acceleration_limit)
        or acceleration_limit <= 0.0
        or not math.isfinite(jerk_limit)
        or jerk_limit <= 0.0
    ):
        raise ValueError("no-descent recovery limits must be positive")

    acceleration = float(absolute_accelerations[0, 2])
    recovery_jerk = np.empty(relative_velocities.shape[0], dtype=float)
    for step in range(len(recovery_jerk)):
        acceleration_room_jerk = (acceleration_limit - acceleration) / dt
        jerk = min(jerk_limit, max(-jerk_limit, acceleration_room_jerk))
        recovery_jerk[step] = jerk
        acceleration += dt * jerk
    fastest_recovery = (
        relative_velocities[:, 2] + velocity_mapping @ recovery_jerk
    )
    nonnegative_suffix = np.logical_and.accumulate(
        (fastest_recovery >= 0.0)[::-1]
    )[::-1]
    recovery_indices = np.flatnonzero(nonnegative_suffix)
    if recovery_indices.size == 0:
        return np.zeros(relative_velocities.shape[0], dtype=float)
    recovery_index = int(recovery_indices[0])
    recovery_time = (recovery_index + 1) * dt
    relative_acceleration = (
        relative_velocities[0, 2] - current_relative_velocity
    ) / dt
    times = dt * np.arange(
        1, relative_velocities.shape[0] + 1, dtype=float
    )
    recovery_displacement = (
        current_relative_velocity * times
        + 0.5 * relative_acceleration * times**2
        + position_mapping @ recovery_jerk
    )
    # This exception is intentionally bounded: only a descent that can be
    # arrested within the configured viability envelope may follow the
    # fastest reachable braking trajectory. A larger or prolonged descent
    # keeps the strict zero floor and therefore remains fail-closed.
    if (
        recovery_time > _NO_DESCENT_RECOVERY_TIME_S + 1.0e-12
        or np.min(recovery_displacement[:recovery_index + 1])
        < -_NO_DESCENT_RECOVERY_DISTANCE_M - 1.0e-12
        or current_relative_position
        + np.min(recovery_displacement[:recovery_index + 1])
        < 0.0
    ):
        return np.zeros(relative_velocities.shape[0], dtype=float)
    floor = np.minimum(fastest_recovery, 0.0)
    # Avoid an exactly active jerk bound at the first recovery sample.  The
    # shared post-solve tolerance already permits this 10 um/s numerical
    # interior, while an exactly active bound makes numerical solvers brittle.
    floor[floor < 0.0] -= _CONSTRAINT_TOLERANCE
    return floor


@dataclass(frozen=True)
class RelativeLandingMPCResult:
    """Absolute/relative horizons and fail-closed solver diagnostics."""

    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    accelerations_m_s2: np.ndarray
    jerks_m_s3: np.ndarray
    relative_positions_m: np.ndarray
    relative_velocities_m_s: np.ndarray
    relative_accelerations_m_s2: np.ndarray
    landing_pad_positions_m: np.ndarray
    landing_pad_velocities_m_s: np.ndarray
    landing_pad_accelerations_m_s2: np.ndarray
    success: bool
    deadline_missed: bool
    solve_time_s: float
    iterations: int
    message: str
    cost: float
    constraint_margins: Mapping[str, float]
    minimum_constraint_margin: float
    landing_constraints_enforced: bool
    alignment_gate_passed: bool
    descent_allowed_requested: bool
    descent_allowed: bool
