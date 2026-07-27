"""Relative-coordinate jerk MPC for landing on a moving platform.

The controller in this module is deliberately independent of ROS.  It uses a
constant-acceleration landing-pad prediction and optimizes UAV jerk with
SciPy's SLSQP implementation.  Both relative trajectories (for landing costs
and geometry) and absolute UAV trajectories (for command publication) are
returned to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np
from scipy import optimize


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
    # interior, while SLSQP otherwise stalls at its infeasible zero seed.
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

    @property
    def pad_positions_m(self) -> np.ndarray:
        """Return the constant-acceleration pad position horizon."""
        return self.landing_pad_positions_m

    @property
    def pad_velocities_m_s(self) -> np.ndarray:
        """Return the constant-acceleration pad velocity horizon."""
        return self.landing_pad_velocities_m_s

    @property
    def pad_accelerations_m_s2(self) -> np.ndarray:
        """Return the constant pad acceleration horizon."""
        return self.landing_pad_accelerations_m_s2


class RelativeLandingMPC:
    """Finite-horizon relative landing MPC with UAV jerk as its input."""

    def __init__(
        self,
        *,
        dt_s: float = 0.10,
        horizon_steps: int = 20,
        max_horizontal_velocity_m_s: float = 5.0,
        max_ascent_velocity_m_s: float = 2.0,
        max_descent_velocity_m_s: float = 0.7,
        max_horizontal_acceleration_m_s2: float = 3.0,
        max_vertical_acceleration_m_s2: float = 3.0,
        max_jerk_m_s3: float = 5.0,
        deadline_s: float = 0.040,
        maximum_iterations: int = 30,
        funnel_minimum_radius_m: float = 0.20,
        funnel_slope: float = 0.75,
        camera_horizontal_fov_rad: float = 1.396,
        camera_vertical_fov_rad: float = 1.1231652391,
        camera_fov_margin: float = 0.85,
        landing_pad_length_m: float = 5.0,
        landing_pad_width_m: float = 5.0,
        landing_pad_contact_margin_m: float = 0.75,
        alignment_position_tolerance_m: float = 0.35,
        alignment_velocity_tolerance_m: float = 0.35,
    ) -> None:
        """Configure timing, vehicle limits, and landing geometry."""
        positive_values = {
            "dt_s": dt_s,
            "max_horizontal_velocity_m_s": max_horizontal_velocity_m_s,
            "max_ascent_velocity_m_s": max_ascent_velocity_m_s,
            "max_descent_velocity_m_s": max_descent_velocity_m_s,
            "max_horizontal_acceleration_m_s2": (
                max_horizontal_acceleration_m_s2
            ),
            "max_vertical_acceleration_m_s2": (
                max_vertical_acceleration_m_s2
            ),
            "max_jerk_m_s3": max_jerk_m_s3,
            "deadline_s": deadline_s,
            "funnel_slope": funnel_slope,
            "camera_horizontal_fov_rad": camera_horizontal_fov_rad,
            "camera_vertical_fov_rad": camera_vertical_fov_rad,
            "camera_fov_margin": camera_fov_margin,
            "landing_pad_length_m": landing_pad_length_m,
            "landing_pad_width_m": landing_pad_width_m,
            "alignment_position_tolerance_m": (
                alignment_position_tolerance_m
            ),
            "alignment_velocity_tolerance_m": (
                alignment_velocity_tolerance_m
            ),
        }
        for name, raw_value in positive_values.items():
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if horizon_steps < 2:
            raise ValueError("horizon_steps must be at least two")
        if maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        minimum_radius = float(funnel_minimum_radius_m)
        contact_margin = float(landing_pad_contact_margin_m)
        if not math.isfinite(minimum_radius) or minimum_radius < 0.0:
            raise ValueError(
                "funnel_minimum_radius_m must be finite and nonnegative"
            )
        if not math.isfinite(contact_margin) or contact_margin < 0.0:
            raise ValueError(
                "landing_pad_contact_margin_m must be finite and "
                "nonnegative"
            )
        if camera_horizontal_fov_rad >= math.pi:
            raise ValueError("camera_horizontal_fov_rad must be below pi")
        if camera_vertical_fov_rad >= math.pi:
            raise ValueError("camera_vertical_fov_rad must be below pi")
        if camera_fov_margin > 1.0:
            raise ValueError("camera_fov_margin cannot exceed one")
        if contact_margin >= 0.5 * min(
            landing_pad_length_m, landing_pad_width_m
        ):
            raise ValueError(
                "landing pad contact margin leaves no usable deck"
            )

        self.dt_s = float(dt_s)
        self.horizon_steps = int(horizon_steps)
        self.max_horizontal_velocity_m_s = float(
            max_horizontal_velocity_m_s
        )
        self.max_ascent_velocity_m_s = float(max_ascent_velocity_m_s)
        self.max_descent_velocity_m_s = float(max_descent_velocity_m_s)
        self.max_horizontal_acceleration_m_s2 = float(
            max_horizontal_acceleration_m_s2
        )
        self.max_vertical_acceleration_m_s2 = float(
            max_vertical_acceleration_m_s2
        )
        self.max_jerk_m_s3 = float(max_jerk_m_s3)
        self.deadline_s = float(deadline_s)
        self.maximum_iterations = int(maximum_iterations)
        self.funnel_minimum_radius_m = minimum_radius
        self.funnel_slope = float(funnel_slope)
        self.camera_horizontal_fov_rad = float(
            camera_horizontal_fov_rad
        )
        self.camera_vertical_fov_rad = float(camera_vertical_fov_rad)
        self.camera_fov_margin = float(camera_fov_margin)
        self.landing_pad_length_m = float(landing_pad_length_m)
        self.landing_pad_width_m = float(landing_pad_width_m)
        self.landing_pad_contact_margin_m = contact_margin
        self.alignment_position_tolerance_m = float(
            alignment_position_tolerance_m
        )
        self.alignment_velocity_tolerance_m = float(
            alignment_velocity_tolerance_m
        )

        # Relative landing weights.  Terminal velocity is intentionally strong
        # because matching the platform at contact is as important as position.
        self._horizontal_position_weight = 18.0
        self._horizontal_velocity_weight = 8.0
        self._clearance_weight = 20.0
        self._vertical_velocity_weight = 8.0
        self._acceleration_weight = 0.5
        self._jerk_weight = 0.06
        self._jerk_change_weight = 0.10
        self._terminal_horizontal_position_weight = 55.0
        self._terminal_clearance_weight = 65.0
        self._terminal_horizontal_velocity_weight = 30.0
        self._terminal_vertical_velocity_weight = 35.0

        self._warm = np.zeros((self.horizon_steps, 3), dtype=float)
        self._previous_jerk = np.zeros(3, dtype=float)
        self._build_condensed_dynamics()

    def _build_condensed_dynamics(self) -> None:
        """Build exact zero-order-held jerk maps for the horizon."""
        count = self.horizon_steps
        dt = self.dt_s
        self._position_map = np.zeros((count, count), dtype=float)
        self._velocity_map = np.zeros((count, count), dtype=float)
        self._acceleration_map = np.zeros((count, count), dtype=float)
        for row in range(count):
            for column in range(row + 1):
                lag = row - column
                self._position_map[row, column] = dt**3 * (
                    1.0 / 6.0 + 0.5 * lag + 0.5 * lag * lag
                )
                self._velocity_map[row, column] = dt**2 * (lag + 0.5)
                self._acceleration_map[row, column] = dt
        identity_xyz = np.eye(3, dtype=float)
        self._position_flat_map = np.kron(
            self._position_map, identity_xyz
        )
        self._velocity_flat_map = np.kron(
            self._velocity_map, identity_xyz
        )
        self._acceleration_flat_map = np.kron(
            self._acceleration_map, identity_xyz
        )
        self._position_axis_maps = tuple(
            self._position_flat_map[axis::3]
            for axis in range(3)
        )
        self._velocity_axis_maps = tuple(
            self._velocity_flat_map[axis::3]
            for axis in range(3)
        )
        self._acceleration_axis_maps = tuple(
            self._acceleration_flat_map[axis::3]
            for axis in range(3)
        )
        self._jerk_difference = np.eye(count, dtype=float)
        self._jerk_difference[1:, :-1] -= np.eye(count - 1, dtype=float)

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
    ) -> RelativeLandingMPCResult:
        """Solve one relative landing problem and return a fresh horizon."""
        vehicle_position = _finite_three_vector(
            vehicle_position_enu_m, "vehicle position"
        )
        vehicle_velocity = _finite_three_vector(
            vehicle_velocity_enu_m_s, "vehicle velocity"
        )
        vehicle_acceleration = _finite_three_vector(
            vehicle_acceleration_enu_m_s2, "vehicle acceleration"
        )
        pad_position = _finite_three_vector(
            landing_pad_position_enu_m, "landing-pad position"
        )
        pad_velocity = _finite_three_vector(
            landing_pad_velocity_enu_m_s, "landing-pad velocity"
        )
        pad_acceleration = _finite_three_vector(
            landing_pad_acceleration_enu_m_s2,
            "landing-pad acceleration",
        )
        clearance = float(target_clearance_m)
        pad_yaw = float(landing_pad_yaw_enu_rad)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                "target clearance must be finite and nonnegative"
            )
        if not math.isfinite(pad_yaw):
            raise ValueError("landing-pad yaw must be finite")
        constraints_enforced = bool(enforce_landing_constraints)
        descent_requested = bool(descent_allowed)
        fov_constraint_enabled = bool(camera_fov_constraint_enabled)
        descent_speed = float(relative_descent_speed_m_s)
        if not math.isfinite(descent_speed) or descent_speed < 0.0:
            raise ValueError(
                "relative descent speed must be finite and nonnegative"
            )

        relative_position = vehicle_position - pad_position
        relative_velocity = vehicle_velocity - pad_velocity
        relative_acceleration = vehicle_acceleration - pad_acceleration
        alignment_gate_passed = bool(
            np.linalg.norm(relative_position[:2])
            <= self.alignment_position_tolerance_m
            and np.linalg.norm(relative_velocity[:2])
            <= self.alignment_velocity_tolerance_m
        )
        effective_descent_allowed = bool(
            descent_requested and alignment_gate_passed
        )

        steps = np.arange(
            1, self.horizon_steps + 1, dtype=float
        )[:, None]
        times = steps * self.dt_s
        pad_positions = (
            pad_position
            + times * pad_velocity
            + 0.5 * times**2 * pad_acceleration
        )
        pad_velocities = pad_velocity + times * pad_acceleration
        pad_accelerations = np.repeat(
            pad_acceleration[None, :], self.horizon_steps, axis=0
        )
        free_relative_positions = (
            relative_position
            + times * relative_velocity
            + 0.5 * times**2 * relative_acceleration
        )
        free_relative_velocities = (
            relative_velocity + times * relative_acceleration
        )
        free_relative_accelerations = np.repeat(
            relative_acceleration[None, :],
            self.horizon_steps,
            axis=0,
        )
        free_accelerations = (
            free_relative_accelerations + pad_accelerations
        )
        no_descent_velocity_floor = (
            _recoverable_no_descent_velocity_floor(
                free_relative_velocities,
                free_accelerations,
                self._position_map,
                self._velocity_map,
                current_relative_position_z_m=relative_position[2],
                current_relative_velocity_z_m_s=relative_velocity[2],
                dt_s=self.dt_s,
                max_vertical_acceleration_m_s2=(
                    self.max_vertical_acceleration_m_s2
                ),
                max_jerk_m_s3=self.max_jerk_m_s3,
            )
        )

        started = time.monotonic()
        deadline_checks_enabled = True

        class _SolveDeadline(RuntimeError):
            pass

        def enforce_deadline() -> None:
            if (
                deadline_checks_enabled
                and time.monotonic() - started > self.deadline_s
            ):
                raise _SolveDeadline(
                    "relative landing MPC wall-clock deadline exceeded"
                )

        def predicted(
            flat: np.ndarray,
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]:
            jerk = flat.reshape(self.horizon_steps, 3)
            relative_positions = (
                free_relative_positions + self._position_map @ jerk
            )
            relative_velocities = (
                free_relative_velocities + self._velocity_map @ jerk
            )
            relative_accelerations = (
                free_relative_accelerations
                + self._acceleration_map @ jerk
            )
            positions = relative_positions + pad_positions
            velocities = relative_velocities + pad_velocities
            accelerations = relative_accelerations + pad_accelerations
            return (
                positions,
                velocities,
                accelerations,
                relative_positions,
                relative_velocities,
                relative_accelerations,
                jerk,
            )

        def objective(flat: np.ndarray) -> float:
            enforce_deadline()
            (
                _positions,
                _velocities,
                _accelerations,
                relative_positions,
                relative_velocities,
                relative_accelerations,
                jerk,
            ) = predicted(flat)
            horizontal_position = relative_positions[:, :2]
            horizontal_velocity = relative_velocities[:, :2]
            clearance_error = relative_positions[:, 2] - clearance
            vertical_velocity = relative_velocities[:, 2]
            jerk_difference = self._jerk_difference @ jerk
            jerk_difference[0] -= self._previous_jerk
            terminal_position = relative_positions[-1]
            terminal_velocity = relative_velocities[-1]
            return float(
                self._horizontal_position_weight
                * np.sum(horizontal_position**2)
                + self._horizontal_velocity_weight
                * np.sum(horizontal_velocity**2)
                + self._clearance_weight * np.sum(clearance_error**2)
                + self._vertical_velocity_weight
                * np.sum(vertical_velocity**2)
                + self._acceleration_weight
                * np.sum(relative_accelerations**2)
                + self._jerk_weight * np.sum(jerk**2)
                + self._jerk_change_weight
                * np.sum(jerk_difference**2)
                + self._terminal_horizontal_position_weight
                * np.dot(terminal_position[:2], terminal_position[:2])
                + self._terminal_clearance_weight
                * (terminal_position[2] - clearance) ** 2
                + self._terminal_horizontal_velocity_weight
                * np.dot(terminal_velocity[:2], terminal_velocity[:2])
                + self._terminal_vertical_velocity_weight
                * terminal_velocity[2] ** 2
            )

        def objective_jacobian(flat: np.ndarray) -> np.ndarray:
            enforce_deadline()
            (
                _positions,
                _velocities,
                _accelerations,
                relative_positions,
                relative_velocities,
                relative_accelerations,
                jerk,
            ) = predicted(flat)
            position_error = relative_positions.copy()
            position_error[:, 2] -= clearance
            position_weights = np.asarray(
                (
                    self._horizontal_position_weight,
                    self._horizontal_position_weight,
                    self._clearance_weight,
                ),
                dtype=float,
            )
            velocity_weights = np.asarray(
                (
                    self._horizontal_velocity_weight,
                    self._horizontal_velocity_weight,
                    self._vertical_velocity_weight,
                ),
                dtype=float,
            )
            jerk_difference = self._jerk_difference @ jerk
            jerk_difference[0] -= self._previous_jerk
            gradient = (
                2.0
                * self._position_map.T
                @ (position_error * position_weights)
                + 2.0
                * self._velocity_map.T
                @ (relative_velocities * velocity_weights)
                + 2.0
                * self._acceleration_weight
                * self._acceleration_map.T
                @ relative_accelerations
                + 2.0 * self._jerk_weight * jerk
                + 2.0
                * self._jerk_change_weight
                * self._jerk_difference.T
                @ jerk_difference
            )
            terminal_position_error = relative_positions[-1].copy()
            terminal_position_error[2] -= clearance
            terminal_position_weights = np.asarray(
                (
                    self._terminal_horizontal_position_weight,
                    self._terminal_horizontal_position_weight,
                    self._terminal_clearance_weight,
                ),
                dtype=float,
            )
            terminal_velocity_weights = np.asarray(
                (
                    self._terminal_horizontal_velocity_weight,
                    self._terminal_horizontal_velocity_weight,
                    self._terminal_vertical_velocity_weight,
                ),
                dtype=float,
            )
            gradient += (
                2.0
                * self._position_map[-1, :, None]
                * (
                    terminal_position_error
                    * terminal_position_weights
                )[None, :]
            )
            gradient += (
                2.0
                * self._velocity_map[-1, :, None]
                * (
                    relative_velocities[-1]
                    * terminal_velocity_weights
                )[None, :]
            )
            return gradient.reshape(-1)

        def dynamic_constraint_values(flat: np.ndarray) -> np.ndarray:
            enforce_deadline()
            positions = predicted(flat)
            velocities = positions[1]
            accelerations = positions[2]
            jerk = positions[6]
            return np.concatenate(
                (
                    self.max_horizontal_velocity_m_s**2
                    - np.sum(velocities[:, :2] ** 2, axis=1),
                    self.max_ascent_velocity_m_s - velocities[:, 2],
                    velocities[:, 2] + self.max_descent_velocity_m_s,
                    self.max_horizontal_acceleration_m_s2**2
                    - np.sum(accelerations[:, :2] ** 2, axis=1),
                    self.max_vertical_acceleration_m_s2
                    - accelerations[:, 2],
                    accelerations[:, 2]
                    + self.max_vertical_acceleration_m_s2,
                    self.max_jerk_m_s3**2
                    - np.sum(jerk**2, axis=1),
                )
            )

        def dynamic_constraint_jacobian(flat: np.ndarray) -> np.ndarray:
            enforce_deadline()
            values = predicted(flat)
            velocities = values[1]
            accelerations = values[2]
            jerk = values[6]
            count = self.horizon_steps
            jacobian = np.zeros((7 * count, 3 * count), dtype=float)
            velocity_x, velocity_y, velocity_z = (
                self._velocity_axis_maps
            )
            acceleration_x, acceleration_y, acceleration_z = (
                self._acceleration_axis_maps
            )
            jacobian[:count] = -2.0 * (
                velocities[:, 0, None] * velocity_x
                + velocities[:, 1, None] * velocity_y
            )
            jacobian[count:2 * count] = -velocity_z
            jacobian[2 * count:3 * count] = velocity_z
            jacobian[3 * count:4 * count] = -2.0 * (
                accelerations[:, 0, None] * acceleration_x
                + accelerations[:, 1, None] * acceleration_y
            )
            jacobian[4 * count:5 * count] = -acceleration_z
            jacobian[5 * count:6 * count] = acceleration_z
            jerk_rows = jacobian[6 * count:].reshape(count, count, 3)
            indices = np.arange(count)
            jerk_rows[indices, indices] = -2.0 * jerk
            return jacobian

        cosine = math.cos(pad_yaw)
        sine = math.sin(pad_yaw)
        pad_from_enu = np.asarray(
            ((cosine, sine), (-sine, cosine)), dtype=float
        )
        usable_half_length = (
            0.5 * self.landing_pad_length_m
            - self.landing_pad_contact_margin_m
        )
        usable_half_width = (
            0.5 * self.landing_pad_width_m
            - self.landing_pad_contact_margin_m
        )
        conservative_half_fov = 0.5 * min(
            self.camera_horizontal_fov_rad,
            self.camera_vertical_fov_rad,
        )
        fov_slope = self.camera_fov_margin * math.tan(
            conservative_half_fov
        )

        def safety_constraint_values(flat: np.ndarray) -> np.ndarray:
            enforce_deadline()
            values = predicted(flat)
            relative_positions = values[3]
            relative_velocities = values[4]
            margins: list[np.ndarray] = []
            if constraints_enforced:
                altitude = relative_positions[:, 2]
                horizontal_squared = np.sum(
                    relative_positions[:, :2] ** 2, axis=1
                )
                funnel_radius = (
                    self.funnel_minimum_radius_m
                    + self.funnel_slope * altitude
                )
                fov_radius = fov_slope * altitude
                terminal_pad = (
                    pad_from_enu @ relative_positions[-1, :2]
                )
                margins.extend(
                    (altitude, funnel_radius**2 - horizontal_squared)
                )
                if fov_constraint_enabled:
                    margins.append(fov_radius**2 - horizontal_squared)
                margins.append(
                    np.asarray(
                        (
                            usable_half_length - terminal_pad[0],
                            usable_half_length + terminal_pad[0],
                            usable_half_width - terminal_pad[1],
                            usable_half_width + terminal_pad[1],
                        ),
                        dtype=float,
                    )
                )
            if effective_descent_allowed:
                # Descent permission is conditional throughout the horizon,
                # not only at the current state.  Keeping both horizontal
                # relative states inside their alignment gates prevents an
                # optimizer from starting a descent and drifting outside the
                # landing corridor one step later.
                margins.extend(
                    (
                        np.full(
                            self.horizon_steps,
                            self.alignment_position_tolerance_m**2,
                            dtype=float,
                        )
                        - np.sum(relative_positions[:, :2] ** 2, axis=1),
                        np.full(
                            self.horizon_steps,
                            self.alignment_velocity_tolerance_m**2,
                            dtype=float,
                        )
                        - np.sum(relative_velocities[:, :2] ** 2, axis=1),
                    )
                )
            else:
                margins.append(
                    relative_velocities[:, 2]
                    - no_descent_velocity_floor
                )
            return np.concatenate(margins)

        def safety_constraint_jacobian(flat: np.ndarray) -> np.ndarray:
            enforce_deadline()
            values = predicted(flat)
            relative_positions = values[3]
            row_count = 0
            if constraints_enforced:
                row_count = 2 * self.horizon_steps + 4
                if fov_constraint_enabled:
                    row_count += self.horizon_steps
            row_count += (
                2 * self.horizon_steps
                if effective_descent_allowed
                else self.horizon_steps
            )
            jacobian = np.zeros(
                (row_count, 3 * self.horizon_steps), dtype=float
            )
            position_x, position_y, position_z = (
                self._position_axis_maps
            )
            output_row = 0
            if constraints_enforced:
                jacobian[
                    output_row:output_row + self.horizon_steps
                ] = position_z
                output_row += self.horizon_steps

                funnel_radius = (
                    self.funnel_minimum_radius_m
                    + self.funnel_slope * relative_positions[:, 2]
                )
                funnel_jacobian = (
                    -2.0 * relative_positions[:, 0, None] * position_x
                    - 2.0 * relative_positions[:, 1, None] * position_y
                    + 2.0
                    * self.funnel_slope
                    * funnel_radius[:, None]
                    * position_z
                )
                jacobian[
                    output_row:output_row + self.horizon_steps
                ] = funnel_jacobian
                output_row += self.horizon_steps

                if fov_constraint_enabled:
                    fov_jacobian = (
                        -2.0 * relative_positions[:, 0, None] * position_x
                        - 2.0 * relative_positions[:, 1, None] * position_y
                        + 2.0
                        * fov_slope**2
                        * relative_positions[:, 2, None]
                        * position_z
                    )
                    jacobian[
                        output_row:output_row + self.horizon_steps
                    ] = fov_jacobian
                    output_row += self.horizon_steps

                pad_gradients = np.asarray(
                    (
                        -pad_from_enu[0],
                        pad_from_enu[0],
                        -pad_from_enu[1],
                        pad_from_enu[1],
                    ),
                    dtype=float,
                )
                for rectangle_row, gradient_xy in enumerate(pad_gradients):
                    jacobian[output_row + rectangle_row] = (
                        gradient_xy[0] * position_x[-1]
                        + gradient_xy[1] * position_y[-1]
                    )
                output_row += 4

            if effective_descent_allowed:
                velocity_x, velocity_y, _velocity_z = (
                    self._velocity_axis_maps
                )
                jacobian[
                    output_row:output_row + self.horizon_steps
                ] = -2.0 * (
                    relative_positions[:, 0, None] * position_x
                    + relative_positions[:, 1, None] * position_y
                )
                output_row += self.horizon_steps
                relative_velocities = values[4]
                jacobian[
                    output_row:output_row + self.horizon_steps
                ] = -2.0 * (
                    relative_velocities[:, 0, None] * velocity_x
                    + relative_velocities[:, 1, None] * velocity_y
                )
            else:
                jacobian[
                    output_row:output_row + self.horizon_steps
                ] = self._velocity_axis_maps[2]
            return jacobian

        constraints: list[dict[str, object]] = [
            {
                "type": "ineq",
                "fun": dynamic_constraint_values,
                "jac": dynamic_constraint_jacobian,
            }
        ]
        constraints.append(
            {
                "type": "ineq",
                "fun": safety_constraint_values,
                "jac": safety_constraint_jacobian,
            }
        )

        result = None
        message = "solver not run"
        timed_out = False

        def callback(_flat: np.ndarray) -> None:
            enforce_deadline()

        try:
            result = optimize.minimize(
                objective,
                self._warm.reshape(-1),
                jac=objective_jacobian,
                method="SLSQP",
                bounds=[
                    (-self.max_jerk_m_s3, self.max_jerk_m_s3)
                ]
                * (self.horizon_steps * 3),
                constraints=constraints,
                callback=callback,
                options={
                    "maxiter": self.maximum_iterations,
                    "ftol": 1.0e-5,
                    "disp": False,
                },
            )
            message = str(result.message)
        except _SolveDeadline as exc:
            timed_out = True
            message = str(exc)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            message = f"relative landing solver exception: {exc}"

        # Constraint functions are reused for the mandatory post-solve audit.
        # That audit must report a missed deadline, not raise another timeout.
        deadline_checks_enabled = False
        elapsed = time.monotonic() - started
        deadline_missed = timed_out or elapsed > self.deadline_s
        valid = bool(
            result is not None
            and result.success
            and np.all(np.isfinite(result.x))
            and not deadline_missed
        )
        if valid and result is not None:
            candidate = predicted(result.x)
            dynamic_values = dynamic_constraint_values(result.x)
            safety_values = safety_constraint_values(result.x)
            valid = bool(
                all(np.all(np.isfinite(array)) for array in candidate)
                and np.all(dynamic_values >= -_CONSTRAINT_TOLERANCE)
                and np.all(safety_values >= -_CONSTRAINT_TOLERANCE)
            )
        else:
            candidate = predicted(np.zeros(self.horizon_steps * 3))
            dynamic_values = dynamic_constraint_values(
                np.zeros(self.horizon_steps * 3)
            )
            safety_values = safety_constraint_values(
                np.zeros(self.horizon_steps * 3)
            )

        if valid and result is not None:
            jerk = candidate[6]
            self._warm[:-1] = jerk[1:]
            self._warm[-1] = jerk[-1]
            self._previous_jerk = jerk[0].copy()
            iterations = int(result.nit)
            cost = float(result.fun)
        else:
            # The caller will HOLD or invoke its configured P fallback.  A
            # deterministic zero-jerk rollout is returned for diagnostics;
            # neither this rollout nor a stale warm solution is executable.
            zero = np.zeros(self.horizon_steps * 3, dtype=float)
            candidate = predicted(zero)
            dynamic_values = dynamic_constraint_values(zero)
            safety_values = safety_constraint_values(zero)
            iterations = (
                0
                if result is None
                else int(getattr(result, "nit", 0))
            )
            cost = float("nan")
            if result is not None and bool(result.success):
                message = f"hard constraint recheck failed: {message}"

        constraint_margins = self._constraint_margin_summary(
            dynamic_values,
            safety_values,
            constraints_enforced=constraints_enforced,
            descent_blocked=not effective_descent_allowed,
            descent_alignment_enforced=effective_descent_allowed,
            camera_fov_constraint_enabled=fov_constraint_enabled,
        )
        minimum_margin = min(constraint_margins.values())
        return RelativeLandingMPCResult(
            candidate[0].copy(),
            candidate[1].copy(),
            candidate[2].copy(),
            candidate[6].copy(),
            candidate[3].copy(),
            candidate[4].copy(),
            candidate[5].copy(),
            pad_positions.copy(),
            pad_velocities.copy(),
            pad_accelerations.copy(),
            valid,
            deadline_missed,
            elapsed,
            iterations,
            message,
            cost,
            constraint_margins,
            minimum_margin,
            constraints_enforced,
            alignment_gate_passed,
            descent_requested,
            effective_descent_allowed,
        )

    def _constraint_margin_summary(
        self,
        dynamic_values: np.ndarray,
        safety_values: np.ndarray,
        *,
        constraints_enforced: bool,
        descent_blocked: bool,
        descent_alignment_enforced: bool,
        camera_fov_constraint_enabled: bool = True,
    ) -> dict[str, float]:
        """Reduce every active constraint family to its minimum margin."""
        count = self.horizon_steps
        names = (
            "horizontal_velocity",
            "ascent_velocity",
            "descent_velocity",
            "horizontal_acceleration",
            "vertical_acceleration_upper",
            "vertical_acceleration_lower",
            "jerk",
        )
        summary = {
            name: float(
                np.min(
                    dynamic_values[index * count:(index + 1) * count]
                )
            )
            for index, name in enumerate(names)
        }
        offset = 0
        if constraints_enforced:
            summary["relative_altitude"] = float(
                np.min(safety_values[offset:offset + count])
            )
            offset += count
            summary["landing_funnel"] = float(
                np.min(safety_values[offset:offset + count])
            )
            offset += count
            if camera_fov_constraint_enabled:
                summary["camera_fov"] = float(
                    np.min(safety_values[offset:offset + count])
                )
                offset += count
            summary["landing_pad_contact"] = float(
                np.min(safety_values[offset:offset + 4])
            )
            offset += 4
        if descent_blocked:
            summary["no_descent"] = float(
                np.min(safety_values[offset:offset + count])
            )
        elif descent_alignment_enforced:
            summary["descent_alignment_position"] = float(
                np.min(safety_values[offset:offset + count])
            )
            offset += count
            summary["descent_alignment_velocity"] = float(
                np.min(safety_values[offset:offset + count])
            )
        return summary
