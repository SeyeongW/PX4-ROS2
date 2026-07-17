"""Sparse OSQP solver for relative-coordinate moving-pad landing MPC.

The solver has the same ``solve`` contract as :class:`RelativeLandingMPC`,
but condenses the linear jerk dynamics into one quadratic program.  Its
Hessian and constraint matrix are created once.  A control cycle updates only
the linear cost and constraint bounds before warm-starting the existing OSQP
workspace.

Nonlinear circular limits from the SLSQP reference are represented by fixed,
inscribed regular polygons.  The terminal deck polygon is itself contained in
the usable rectangle for every deck yaw, which lets the OSQP constraint matrix
remain immutable while retaining fail-closed geometry.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np
from scipy import sparse

try:
    import osqp
except ImportError:  # pragma: no cover - exercised on systems without OSQP.
    osqp = None

from .relative_landing_mpc import (
    RelativeLandingMPC,
    RelativeLandingMPCResult,
    _finite_three_vector,
    _recoverable_no_descent_velocity_floor,
)


_BASE_CONSTRAINT_TOLERANCE = 1.0e-5


class SlsqpReferenceSolver(RelativeLandingMPC):
    """Offline SLSQP reference with the common landing-solver contract."""


class OsqpLandingSolver:
    """Reusable sparse-QP solver for relative moving-platform landing."""

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
        deadline_s: float = 0.020,
        maximum_iterations: int = 400,
        absolute_tolerance: float = 1.0e-4,
        relative_tolerance: float = 1.0e-4,
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
        polygon_facets: int = 8,
    ) -> None:
        """Configure limits and create the immutable OSQP workspace."""
        if osqp is None:
            raise ImportError(
                "OsqpLandingSolver requires the optional 'osqp' package"
            )

        positive_values = {
            "dt_s": dt_s,
            "max_horizontal_velocity_m_s": (
                max_horizontal_velocity_m_s
            ),
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
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
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
        if polygon_facets < 4:
            raise ValueError("polygon_facets must be at least four")
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
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
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
        self.polygon_facets = int(polygon_facets)

        self._horizontal_position_weight = 18.0
        self._horizontal_velocity_weight = 16.0
        self._clearance_weight = 20.0
        self._vertical_velocity_weight = 8.0
        self._acceleration_weight = 0.5
        self._jerk_weight = 0.06
        self._jerk_change_weight = 0.10
        self._terminal_horizontal_position_weight = 55.0
        self._terminal_clearance_weight = 65.0
        self._terminal_horizontal_velocity_weight = 50.0
        self._terminal_vertical_velocity_weight = 35.0

        self._previous_jerk = np.zeros(3, dtype=float)
        self._build_condensed_dynamics()
        self._build_objective_matrix()
        self._build_constraint_matrix()
        self._q = np.zeros(self._variable_count, dtype=float)
        self._lower = np.full(self._constraint_count, -np.inf)
        self._upper = np.full(self._constraint_count, np.inf)
        self._zero_primal = np.zeros(self._variable_count, dtype=float)
        self._zero_dual = np.zeros(self._constraint_count, dtype=float)
        self._workspace_setup_count = 0
        self._workspace_update_count = 0
        self._workspace_solve_count = 0
        self._setup_workspace()

    @property
    def workspace_setup_count(self) -> int:
        """Return how many OSQP workspaces this instance has created."""
        return self._workspace_setup_count

    @property
    def workspace_update_count(self) -> int:
        """Return the number of successful in-place q/l/u updates."""
        return self._workspace_update_count

    @property
    def workspace_solve_count(self) -> int:
        """Return the number of calls made to the persistent workspace."""
        return self._workspace_solve_count

    def clear_warm_start(self) -> None:
        """Clear all primal, dual, and jerk-history state after rejection."""
        self._workspace.warm_start(
            x=self._zero_primal, y=self._zero_dual
        )
        self._previous_jerk.fill(0.0)

    @property
    def constraint_matrix(self) -> sparse.csc_matrix:
        """Return the immutable sparse constraint matrix."""
        return self._constraint_matrix

    @property
    def hessian_matrix(self) -> sparse.csc_matrix:
        """Return the immutable upper-triangular QP Hessian."""
        return self._hessian

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
                self._velocity_map[row, column] = dt**2 * (
                    lag + 0.5
                )
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
        difference = np.eye(count, dtype=float)
        difference[1:, :-1] -= np.eye(count - 1, dtype=float)
        self._jerk_difference_flat = np.kron(difference, identity_xyz)
        self._variable_count = 3 * count

    def _build_objective_matrix(self) -> None:
        """Build the fixed positive-definite jerk Hessian."""
        count = self.horizon_steps
        position_weights = np.tile(
            (
                self._horizontal_position_weight,
                self._horizontal_position_weight,
                self._clearance_weight,
            ),
            count,
        )
        velocity_weights = np.tile(
            (
                self._horizontal_velocity_weight,
                self._horizontal_velocity_weight,
                self._vertical_velocity_weight,
            ),
            count,
        )
        position_weights[-3:] += np.asarray(
            (
                self._terminal_horizontal_position_weight,
                self._terminal_horizontal_position_weight,
                self._terminal_clearance_weight,
            )
        )
        velocity_weights[-3:] += np.asarray(
            (
                self._terminal_horizontal_velocity_weight,
                self._terminal_horizontal_velocity_weight,
                self._terminal_vertical_velocity_weight,
            )
        )
        self._position_weights = position_weights
        self._velocity_weights = velocity_weights
        self._position_gradient_map = 2.0 * (
            self._position_flat_map.T * position_weights[None, :]
        )
        self._velocity_gradient_map = 2.0 * (
            self._velocity_flat_map.T * velocity_weights[None, :]
        )
        self._acceleration_gradient_map = (
            2.0 * self._acceleration_weight * self._acceleration_flat_map.T
        )
        weighted_position_map = (
            position_weights[:, None] * self._position_flat_map
        )
        weighted_velocity_map = (
            velocity_weights[:, None] * self._velocity_flat_map
        )
        dense_hessian = 2.0 * (
            self._position_flat_map.T @ weighted_position_map
            + self._velocity_flat_map.T @ weighted_velocity_map
            + self._acceleration_weight
            * self._acceleration_flat_map.T
            @ self._acceleration_flat_map
            + self._jerk_weight * np.eye(self._variable_count)
            + self._jerk_change_weight
            * self._jerk_difference_flat.T
            @ self._jerk_difference_flat
        )
        dense_hessian += 1.0e-10 * np.eye(self._variable_count)
        self._hessian = sparse.csc_matrix(np.triu(dense_hessian))

    def _append_rows(
        self,
        name: str,
        matrix: np.ndarray,
        matrices: list[sparse.csc_matrix],
        row_offset: int,
    ) -> int:
        """Append a constraint family and record its row slice."""
        row_count = matrix.shape[0]
        self._row_slices[name] = slice(row_offset, row_offset + row_count)
        matrices.append(sparse.csc_matrix(matrix))
        return row_offset + row_count

    def _polygon_matrix(
        self,
        source_map: np.ndarray,
        *,
        altitude_slope: float = 0.0,
    ) -> np.ndarray:
        """Return fixed polygon rows for each point in one xyz horizon."""
        count = self.horizon_steps
        rows = np.empty(
            (count * self.polygon_facets, self._variable_count),
            dtype=float,
        )
        for step in range(count):
            for facet, direction in enumerate(self._facet_directions):
                row = step * self.polygon_facets + facet
                rows[row] = (
                    direction[0] * source_map[3 * step]
                    + direction[1] * source_map[3 * step + 1]
                    - altitude_slope * source_map[3 * step + 2]
                )
        return rows

    def _build_constraint_matrix(self) -> None:
        """Build every potentially active constraint row exactly once."""
        angles = 2.0 * math.pi * np.arange(
            self.polygon_facets, dtype=float
        ) / float(self.polygon_facets)
        self._facet_directions = np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        self._polygon_apothem_scale = math.cos(
            math.pi / float(self.polygon_facets)
        )
        conservative_half_fov = 0.5 * min(
            self.camera_horizontal_fov_rad,
            self.camera_vertical_fov_rad,
        )
        self._fov_slope = self.camera_fov_margin * math.tan(
            conservative_half_fov
        )
        usable_half_length = (
            0.5 * self.landing_pad_length_m
            - self.landing_pad_contact_margin_m
        )
        usable_half_width = (
            0.5 * self.landing_pad_width_m
            - self.landing_pad_contact_margin_m
        )
        self._deck_radius = min(usable_half_length, usable_half_width)

        matrices: list[sparse.csc_matrix] = []
        self._row_slices: dict[str, slice] = {}
        row_offset = 0
        row_offset = self._append_rows(
            "velocity", self._velocity_flat_map, matrices, row_offset
        )
        row_offset = self._append_rows(
            "acceleration",
            self._acceleration_flat_map,
            matrices,
            row_offset,
        )
        row_offset = self._append_rows(
            "jerk",
            np.eye(self._variable_count, dtype=float),
            matrices,
            row_offset,
        )
        row_offset = self._append_rows(
            "relative_altitude",
            self._position_flat_map[2::3],
            matrices,
            row_offset,
        )
        funnel_slope = (
            self._polygon_apothem_scale * self.funnel_slope
        )
        row_offset = self._append_rows(
            "landing_funnel",
            self._polygon_matrix(
                self._position_flat_map,
                altitude_slope=funnel_slope,
            ),
            matrices,
            row_offset,
        )
        fov_slope = self._polygon_apothem_scale * self._fov_slope
        row_offset = self._append_rows(
            "camera_fov",
            self._polygon_matrix(
                self._position_flat_map,
                altitude_slope=fov_slope,
            ),
            matrices,
            row_offset,
        )
        terminal_map = np.zeros(
            (self.polygon_facets, self._variable_count), dtype=float
        )
        for facet, direction in enumerate(self._facet_directions):
            terminal_map[facet] = (
                direction[0] * self._position_flat_map[-3]
                + direction[1] * self._position_flat_map[-2]
            )
        row_offset = self._append_rows(
            "landing_pad_contact", terminal_map, matrices, row_offset
        )
        self._alignment_position_matrix = self._polygon_matrix(
            self._position_flat_map
        )
        self._alignment_velocity_matrix = self._polygon_matrix(
            self._velocity_flat_map
        )
        self._alignment_position_jerk_reach = self.max_jerk_m_s3 * np.sum(
            np.abs(self._alignment_position_matrix), axis=1
        )
        self._alignment_velocity_jerk_reach = self.max_jerk_m_s3 * np.sum(
            np.abs(self._alignment_velocity_matrix), axis=1
        )
        row_offset = self._append_rows(
            "descent_alignment_position",
            self._alignment_position_matrix,
            matrices,
            row_offset,
        )
        row_offset = self._append_rows(
            "descent_alignment_velocity",
            self._alignment_velocity_matrix,
            matrices,
            row_offset,
        )
        row_offset = self._append_rows(
            "no_descent",
            self._velocity_flat_map[2::3],
            matrices,
            row_offset,
        )
        self._constraint_matrix = sparse.vstack(
            matrices, format="csc"
        )
        self._constraint_count = row_offset

    def _setup_workspace(self) -> None:
        """Create one workspace with compatibility across OSQP versions."""
        settings = {
            "verbose": False,
            "max_iter": self.maximum_iterations,
            "eps_abs": self.absolute_tolerance,
            "eps_rel": self.relative_tolerance,
            "time_limit": self.deadline_s,
            "adaptive_rho": True,
            "check_termination": 5,
            "scaled_termination": False,
        }
        major_version = int(str(osqp.__version__).split(".", 1)[0])
        self._osqp_major_version = major_version
        if major_version >= 1:
            settings.update(warm_starting=True, polishing=False)
        else:  # pragma: no cover - retained for Ubuntu OSQP 0.6 packages.
            settings.update(warm_start=True, polish=False)
        self._workspace = osqp.OSQP()
        self._workspace.setup(
            P=self._hessian,
            q=self._q,
            A=self._constraint_matrix,
            l=self._lower,
            u=self._upper,
            **settings,
        )
        self._workspace_setup_count += 1

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
        """Update and solve one QP, returning only a fresh verified result."""
        try:
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
        except ValueError:
            self.clear_warm_start()
            raise
        clearance = float(target_clearance_m)
        pad_yaw = float(landing_pad_yaw_enu_rad)
        if not math.isfinite(clearance) or clearance < 0.0:
            self.clear_warm_start()
            raise ValueError(
                "target clearance must be finite and nonnegative"
            )
        if not math.isfinite(pad_yaw):
            self.clear_warm_start()
            raise ValueError("landing-pad yaw must be finite")
        numerical_inputs = (
            vehicle_position,
            vehicle_velocity,
            vehicle_acceleration,
            pad_position,
            pad_velocity,
            pad_acceleration,
        )
        if (
            clearance > 1.0e6
            or any(np.max(np.abs(value)) > 1.0e6 for value in numerical_inputs)
        ):
            self.clear_warm_start()
            raise ValueError("landing MPC input exceeds numerical range")

        constraints_enforced = bool(enforce_landing_constraints)
        descent_requested = bool(descent_allowed)
        fov_constraint_enabled = bool(camera_fov_constraint_enabled)
        descent_speed = float(relative_descent_speed_m_s)
        if not math.isfinite(descent_speed) or descent_speed < 0.0:
            self.clear_warm_start()
            raise ValueError(
                "relative descent speed must be finite and nonnegative"
            )
        relative_position = vehicle_position - pad_position
        relative_velocity = vehicle_velocity - pad_velocity
        relative_acceleration = vehicle_acceleration - pad_acceleration
        # The QP approximates each circular alignment limit with an inscribed
        # polygon. Use that same conservative geometry for the activation
        # gate. A circular-only gate can activate polygon constraints while
        # the first predicted sample is still outside the polygon and cannot
        # reach it within one jerk-limited step, making a safe boundary state
        # spuriously primal-infeasible.
        alignment_position_bound = (
            self._polygon_apothem_scale
            * self.alignment_position_tolerance_m
        )
        alignment_velocity_bound = (
            self._polygon_apothem_scale
            * self.alignment_velocity_tolerance_m
        )
        alignment_gate_passed = bool(
            np.all(
                self._facet_directions @ relative_position[:2]
                <= alignment_position_bound + _BASE_CONSTRAINT_TOLERANCE
            )
            and np.all(
                self._facet_directions @ relative_velocity[:2]
                <= alignment_velocity_bound + _BASE_CONSTRAINT_TOLERANCE
            )
        )
        started = time.monotonic()
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
        free_positions = free_relative_positions + pad_positions
        free_velocities = free_relative_velocities + pad_velocities
        free_accelerations = (
            free_relative_accelerations + pad_accelerations
        )

        # The current state can be inside the alignment polygon while the
        # first predicted sample is already outside and cannot be brought
        # back by any jerk-bounded command.  Activating the hard descent rows
        # in that case makes an otherwise recoverable level solve infeasible.
        if alignment_gate_passed:
            free_position_projection = (
                free_relative_positions[:, :2] @ self._facet_directions.T
            ).reshape(-1)
            free_velocity_projection = (
                free_relative_velocities[:, :2] @ self._facet_directions.T
            ).reshape(-1)
            alignment_gate_passed = bool(
                np.all(
                    free_position_projection[:self.polygon_facets]
                    - self._alignment_position_jerk_reach[
                        :self.polygon_facets
                    ]
                    <= alignment_position_bound
                    + _BASE_CONSTRAINT_TOLERANCE
                )
                and np.all(
                    free_velocity_projection[:self.polygon_facets]
                    - self._alignment_velocity_jerk_reach[
                        :self.polygon_facets
                    ]
                    <= alignment_velocity_bound
                    + _BASE_CONSTRAINT_TOLERANCE
                )
            )
        effective_descent_allowed = bool(
            descent_requested and alignment_gate_passed
        )
        (
            clearance_reference,
            vertical_velocity_reference,
        ) = self._vertical_reference_horizon(
            current_relative_position_z_m=float(relative_position[2]),
            current_relative_velocity_z_m_s=float(relative_velocity[2]),
            target_clearance_m=clearance,
            requested_descent_speed_m_s=(
                descent_speed if effective_descent_allowed else 0.0
            ),
        )

        result = None
        message = "OSQP solver not run"
        try:
            with np.errstate(over="raise", invalid="raise"):
                self._update_linear_cost(
                    free_relative_positions,
                    free_relative_velocities,
                    free_relative_accelerations,
                    clearance_reference,
                    vertical_velocity_reference,
                )
                self._update_constraint_bounds(
                    free_relative_positions,
                    free_relative_velocities,
                    free_positions,
                    free_velocities,
                    free_accelerations,
                    current_relative_position_z_m=relative_position[2],
                    current_relative_velocity_z_m_s=relative_velocity[2],
                    constraints_enforced=constraints_enforced,
                    effective_descent_allowed=effective_descent_allowed,
                    camera_fov_constraint_enabled=fov_constraint_enabled,
                )
            if not np.all(np.isfinite(self._q)):
                raise FloatingPointError("QP linear cost is not finite")
            if np.any(np.isnan(self._lower)) or np.any(np.isnan(self._upper)):
                raise FloatingPointError("QP bounds contain NaN")
            if np.any(self._lower > self._upper):
                raise ValueError("QP lower bound exceeds upper bound")
            self._workspace.update(
                q=self._q, l=self._lower, u=self._upper
            )
            self._workspace_update_count += 1
            self._workspace_solve_count += 1
            if self._osqp_major_version >= 1:
                result = self._workspace.solve(raise_error=False)
            else:  # pragma: no cover - Ubuntu OSQP 0.6 compatibility.
                result = self._workspace.solve()
            message = str(result.info.status)
        except Exception as exc:  # OSQP version-specific numerical failures
            message = f"OSQP landing solver exception: {exc}"

        elapsed = time.monotonic() - started
        solver_timed_out = bool(
            result is not None
            and (
                int(result.info.status_val) == 8
                or "time limit" in str(result.info.status).lower()
            )
        )
        deadline_missed = bool(
            solver_timed_out or elapsed > self.deadline_s
        )
        status_solved = bool(
            result is not None
            and int(result.info.status_val) == 1
        )
        finite_solution = bool(
            result is not None
            and result.x is not None
            and np.asarray(result.x).shape == (self._variable_count,)
            and np.all(np.isfinite(result.x))
        )
        valid = bool(
            status_solved and finite_solution and not deadline_missed
        )
        if valid and result is not None:
            flat_jerk = np.asarray(result.x, dtype=float)
            candidate = self._predicted(
                flat_jerk,
                free_relative_positions,
                free_relative_velocities,
                free_relative_accelerations,
                pad_positions,
                pad_velocities,
                pad_accelerations,
            )
            valid = self._post_solve_valid(flat_jerk, candidate)
            if not valid:
                message = f"hard constraint recheck failed: {message}"
        else:
            flat_jerk = self._zero_primal
            candidate = self._predicted(
                flat_jerk,
                free_relative_positions,
                free_relative_velocities,
                free_relative_accelerations,
                pad_positions,
                pad_velocities,
                pad_accelerations,
            )

        # A successful command is audited only after every expensive piece of
        # post-processing, including diagnostic reduction, defensive copies,
        # and warm-start preparation.  This prevents an apparently fresh OSQP
        # solve from becoming stale while its command is being assembled.
        if valid and result is not None:
            jerk = candidate[6]
            iterations = int(result.info.iter)
            cost = self._objective_cost(
                candidate[3],
                candidate[4],
                candidate[5],
                jerk,
                clearance_reference,
                vertical_velocity_reference,
            )
            constraint_margins = self._constraint_margin_summary(
                candidate,
                constraints_enforced=constraints_enforced,
                effective_descent_allowed=effective_descent_allowed,
                camera_fov_constraint_enabled=fov_constraint_enabled,
            )
            minimum_margin = min(constraint_margins.values())
            command_arrays = tuple(array.copy() for array in candidate)
            pad_arrays = (
                pad_positions.copy(),
                pad_velocities.copy(),
                pad_accelerations.copy(),
            )
            if not math.isfinite(cost) or not math.isfinite(minimum_margin):
                valid = False
                message = f"non-finite post-solve diagnostic: {message}"
            else:
                shifted = np.empty_like(jerk)
                shifted[:-1] = jerk[1:]
                shifted[-1] = jerk[-1]
                try:
                    # Bounds switch between level recovery and authorized
                    # descent.  Retaining duals from the former active set
                    # produced false `primal infeasible` results on the next
                    # feasible solve; keep the useful shifted primal only.
                    self._workspace.warm_start(
                        x=shifted.reshape(-1), y=self._zero_dual
                    )
                    self._previous_jerk = jerk[0].copy()
                except Exception as exc:
                    valid = False
                    message = f"OSQP warm-start exception: {exc}"

            # This is the final success audit.  No executable trajectory is
            # returned when any part of the complete solve call exceeds its
            # deadline.
            elapsed = time.monotonic() - started
            deadline_missed = bool(
                solver_timed_out or elapsed > self.deadline_s
            )
            valid = bool(valid and not deadline_missed)
            if deadline_missed:
                message = f"deadline exceeded: {message}"
            if valid:
                return RelativeLandingMPCResult(
                    command_arrays[0],
                    command_arrays[1],
                    command_arrays[2],
                    command_arrays[6],
                    command_arrays[3],
                    command_arrays[4],
                    command_arrays[5],
                    pad_arrays[0],
                    pad_arrays[1],
                    pad_arrays[2],
                    True,
                    False,
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
        else:
            elapsed = time.monotonic() - started
            deadline_missed = bool(
                solver_timed_out or elapsed > self.deadline_s
            )
            if deadline_missed:
                message = f"deadline exceeded: {message}"

        # Clear the workspace's primal and dual iterates after every failure.
        # The returned zero-jerk rollout is diagnostic only; neither it nor a
        # previous successful solution is executable.
        self.clear_warm_start()
        flat_jerk = self._zero_primal
        candidate = self._predicted(
            flat_jerk,
            free_relative_positions,
            free_relative_velocities,
            free_relative_accelerations,
            pad_positions,
            pad_velocities,
            pad_accelerations,
        )
        iterations = 0 if result is None else int(result.info.iter)
        constraint_margins = self._constraint_margin_summary(
            candidate,
            constraints_enforced=constraints_enforced,
            effective_descent_allowed=effective_descent_allowed,
            camera_fov_constraint_enabled=fov_constraint_enabled,
        )
        minimum_margin = min(constraint_margins.values())
        elapsed = time.monotonic() - started
        deadline_missed = bool(
            deadline_missed or solver_timed_out or elapsed > self.deadline_s
        )
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
            False,
            deadline_missed,
            elapsed,
            iterations,
            message,
            float("nan"),
            constraint_margins,
            minimum_margin,
            constraints_enforced,
            alignment_gate_passed,
            descent_requested,
            effective_descent_allowed,
        )

    def _update_linear_cost(
        self,
        free_relative_positions: np.ndarray,
        free_relative_velocities: np.ndarray,
        free_relative_accelerations: np.ndarray,
        clearance_reference: np.ndarray,
        vertical_velocity_reference: np.ndarray,
    ) -> None:
        """Update q in-place from the current free relative rollout."""
        position_error = free_relative_positions.copy()
        position_error[:, 2] -= clearance_reference
        velocity_error = free_relative_velocities.copy()
        velocity_error[:, 2] -= vertical_velocity_reference
        self._q[:] = (
            self._position_gradient_map @ position_error.reshape(-1)
            + self._velocity_gradient_map
            @ velocity_error.reshape(-1)
            + self._acceleration_gradient_map
            @ free_relative_accelerations.reshape(-1)
        )
        previous_target = np.zeros(self._variable_count, dtype=float)
        previous_target[:3] = self._previous_jerk
        self._q -= (
            2.0
            * self._jerk_change_weight
            * self._jerk_difference_flat.T
            @ previous_target
        )

    def _vertical_reference_horizon(
        self,
        *,
        current_relative_position_z_m: float,
        current_relative_velocity_z_m_s: float,
        target_clearance_m: float,
        requested_descent_speed_m_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a continuous acceleration-limited relative-Z trajectory.

        A scalar clearance together with a zero vertical-speed target asks
        the optimizer to stop at a newly lowered point on every control tick.
        That produced the measured climb/descent cycle.  This reference starts
        at the measured relative state, ramps toward the requested descent
        speed, and brakes to zero at the clearance floor.  The QP's existing
        jerk constraints remain the authoritative command smoothness limit.
        """
        position = float(current_relative_position_z_m)
        velocity = float(current_relative_velocity_z_m_s)
        floor = float(target_clearance_m)
        speed = float(requested_descent_speed_m_s)
        values = (position, velocity, floor, speed)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vertical reference inputs must be finite")
        if floor < 0.0 or speed < 0.0:
            raise ValueError("vertical reference floor/speed must be nonnegative")

        positions = np.empty(self.horizon_steps, dtype=float)
        velocities = np.empty(self.horizon_steps, dtype=float)
        if speed <= 1.0e-9:
            positions.fill(position)
            velocities.fill(0.0)
            return positions, velocities

        # One metre per second squared reaches the 0.55--0.60 m/s landing
        # rates gently while leaving substantial margin to the configured
        # 3 m/s^2 command bound.  Braking speed makes the last approach taper
        # continuously instead of striking the floor with a reference step.
        reference_acceleration = min(
            1.0, self.max_vertical_acceleration_m_s2
        )
        velocity = float(
            np.clip(
                velocity,
                -self.max_descent_velocity_m_s,
                self.max_ascent_velocity_m_s,
            )
        )
        for index in range(self.horizon_steps):
            remaining = max(0.0, position - floor)
            braking_speed = math.sqrt(
                max(0.0, 2.0 * reference_acceleration * remaining)
            )
            target_velocity = -min(speed, braking_speed)
            maximum_delta = reference_acceleration * self.dt_s
            velocity += float(
                np.clip(
                    target_velocity - velocity,
                    -maximum_delta,
                    maximum_delta,
                )
            )
            next_position = position + velocity * self.dt_s
            if next_position <= floor:
                next_position = floor
                velocity = 0.0
            position = next_position
            positions[index] = position
            velocities[index] = velocity
        return positions, velocities

    def _polygon_upper_bounds(
        self,
        free_values: np.ndarray,
        *,
        radius: float,
        altitude_slope: float = 0.0,
    ) -> np.ndarray:
        """Return upper bounds for a horizon of inscribed polygons."""
        bounds = np.empty(
            self.horizon_steps * self.polygon_facets, dtype=float
        )
        for step in range(self.horizon_steps):
            horizontal = free_values[step, :2]
            altitude = free_values[step, 2]
            start = step * self.polygon_facets
            stop = start + self.polygon_facets
            bounds[start:stop] = (
                self._polygon_apothem_scale * radius
                - self._facet_directions @ horizontal
                + altitude_slope * altitude
            )
        return bounds

    def _update_constraint_bounds(
        self,
        free_relative_positions: np.ndarray,
        free_relative_velocities: np.ndarray,
        free_positions: np.ndarray,
        free_velocities: np.ndarray,
        free_accelerations: np.ndarray,
        *,
        current_relative_position_z_m: float,
        current_relative_velocity_z_m_s: float,
        constraints_enforced: bool,
        effective_descent_allowed: bool,
        camera_fov_constraint_enabled: bool = True,
    ) -> None:
        """Update l/u in-place without changing the sparse A matrix."""
        del free_positions
        self._lower.fill(-np.inf)
        self._upper.fill(np.inf)

        velocity_slice = self._row_slices["velocity"]
        velocity_limits = np.tile(
            (
                self.max_horizontal_velocity_m_s,
                self.max_horizontal_velocity_m_s,
                self.max_ascent_velocity_m_s,
            ),
            self.horizon_steps,
        )
        velocity_lower_limits = np.tile(
            (
                -self.max_horizontal_velocity_m_s,
                -self.max_horizontal_velocity_m_s,
                -self.max_descent_velocity_m_s,
            ),
            self.horizon_steps,
        )
        self._lower[velocity_slice] = (
            velocity_lower_limits - free_velocities.reshape(-1)
        )
        self._upper[velocity_slice] = (
            velocity_limits - free_velocities.reshape(-1)
        )

        acceleration_slice = self._row_slices["acceleration"]
        acceleration_limits = np.tile(
            (
                self.max_horizontal_acceleration_m_s2,
                self.max_horizontal_acceleration_m_s2,
                self.max_vertical_acceleration_m_s2,
            ),
            self.horizon_steps,
        )
        self._lower[acceleration_slice] = (
            -acceleration_limits - free_accelerations.reshape(-1)
        )
        self._upper[acceleration_slice] = (
            acceleration_limits - free_accelerations.reshape(-1)
        )

        jerk_slice = self._row_slices["jerk"]
        self._lower[jerk_slice] = -self.max_jerk_m_s3
        self._upper[jerk_slice] = self.max_jerk_m_s3

        if constraints_enforced:
            altitude_slice = self._row_slices["relative_altitude"]
            self._lower[altitude_slice] = -free_relative_positions[:, 2]

            funnel_slice = self._row_slices["landing_funnel"]
            funnel_slope = (
                self._polygon_apothem_scale * self.funnel_slope
            )
            self._upper[funnel_slice] = self._polygon_upper_bounds(
                free_relative_positions,
                radius=self.funnel_minimum_radius_m,
                altitude_slope=funnel_slope,
            )

            if camera_fov_constraint_enabled:
                fov_slice = self._row_slices["camera_fov"]
                fov_slope = self._polygon_apothem_scale * self._fov_slope
                self._upper[fov_slice] = self._polygon_upper_bounds(
                    free_relative_positions,
                    radius=0.0,
                    altitude_slope=fov_slope,
                )

            deck_slice = self._row_slices["landing_pad_contact"]
            terminal_horizontal = free_relative_positions[-1, :2]
            self._upper[deck_slice] = (
                self._polygon_apothem_scale * self._deck_radius
                - self._facet_directions @ terminal_horizontal
            )

        if effective_descent_allowed:
            position_slice = self._row_slices[
                "descent_alignment_position"
            ]
            position_bounds = self._polygon_upper_bounds(
                free_relative_positions,
                radius=self.alignment_position_tolerance_m,
            )
            self._upper[
                position_slice.start:
                position_slice.start + self.polygon_facets
            ] = position_bounds[:self.polygon_facets]
            velocity_alignment_slice = self._row_slices[
                "descent_alignment_velocity"
            ]
            velocity_bounds = self._polygon_upper_bounds(
                free_relative_velocities,
                radius=self.alignment_velocity_tolerance_m,
            )
            self._upper[
                velocity_alignment_slice.start:
                velocity_alignment_slice.start + self.polygon_facets
            ] = velocity_bounds[:self.polygon_facets]
        else:
            no_descent_slice = self._row_slices["no_descent"]
            no_descent_velocity_floor = (
                _recoverable_no_descent_velocity_floor(
                    free_relative_velocities,
                    free_accelerations,
                    self._position_map,
                    self._velocity_map,
                    current_relative_position_z_m=(
                        current_relative_position_z_m
                    ),
                    current_relative_velocity_z_m_s=(
                        current_relative_velocity_z_m_s
                    ),
                    dt_s=self.dt_s,
                    max_vertical_acceleration_m_s2=(
                        self.max_vertical_acceleration_m_s2
                    ),
                    max_jerk_m_s3=self.max_jerk_m_s3,
                )
            )
            self._lower[no_descent_slice] = (
                no_descent_velocity_floor
                - free_relative_velocities[:, 2]
            )

    def _predicted(
        self,
        flat_jerk: np.ndarray,
        free_relative_positions: np.ndarray,
        free_relative_velocities: np.ndarray,
        free_relative_accelerations: np.ndarray,
        pad_positions: np.ndarray,
        pad_velocities: np.ndarray,
        pad_accelerations: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Roll out absolute and relative trajectories for one jerk vector."""
        jerk = np.asarray(flat_jerk, dtype=float).reshape(
            self.horizon_steps, 3
        )
        relative_positions = (
            free_relative_positions + self._position_map @ jerk
        )
        relative_velocities = (
            free_relative_velocities + self._velocity_map @ jerk
        )
        relative_accelerations = (
            free_relative_accelerations + self._acceleration_map @ jerk
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

    def _post_solve_valid(
        self,
        flat_jerk: np.ndarray,
        candidate: tuple[np.ndarray, ...],
    ) -> bool:
        """Reject non-finite output and every violated active QP bound."""
        if not all(np.all(np.isfinite(array)) for array in candidate):
            return False
        values = np.asarray(self._constraint_matrix @ flat_jerk)
        if not np.all(np.isfinite(values)):
            return False

        # OSQP tests the primal residual r = Ax - z, where z is the
        # projection of Ax onto [l, u], against
        #
        #   eps_abs + eps_rel * max(||Ax||_inf, ||z||_inf).
        #
        # Reconstruct that same unscaled residual here.  The previous fixed
        # ``5 * eps_abs`` audit was stricter than a legitimately ``solved``
        # result whenever the relative term dominated, causing deterministic
        # reject/retry islands during a long approach.  This remains a hard
        # post-solve audit: status other than exact ``solved`` is rejected
        # before this method, and violations beyond OSQP's configured primal
        # feasibility tolerance still fail closed.
        finite_lower = np.isfinite(self._lower)
        finite_upper = np.isfinite(self._upper)
        projected = values.copy()
        projected[finite_lower] = np.maximum(
            projected[finite_lower], self._lower[finite_lower]
        )
        projected[finite_upper] = np.minimum(
            projected[finite_upper], self._upper[finite_upper]
        )
        primal_residual = values - projected
        primal_scale = max(
            float(np.linalg.norm(values, ord=np.inf)),
            float(np.linalg.norm(projected, ord=np.inf)),
        )
        tolerance = max(
            _BASE_CONSTRAINT_TOLERANCE,
            self.absolute_tolerance
            + self.relative_tolerance * primal_scale,
        )
        return bool(
            float(np.linalg.norm(primal_residual, ord=np.inf)) <= tolerance
        )

    def _objective_cost(
        self,
        relative_positions: np.ndarray,
        relative_velocities: np.ndarray,
        relative_accelerations: np.ndarray,
        jerk: np.ndarray,
        clearance_reference: np.ndarray,
        vertical_velocity_reference: np.ndarray,
    ) -> float:
        """Evaluate the full reference objective, including constants."""
        position_error = relative_positions.copy()
        position_error[:, 2] -= clearance_reference
        velocity_error = relative_velocities.copy()
        velocity_error[:, 2] -= vertical_velocity_reference
        jerk_difference = self._jerk_difference_flat @ jerk.reshape(-1)
        jerk_difference[:3] -= self._previous_jerk
        return float(
            np.dot(
                self._position_weights,
                position_error.reshape(-1) ** 2,
            )
            + np.dot(
                self._velocity_weights,
                velocity_error.reshape(-1) ** 2,
            )
            + self._acceleration_weight
            * np.sum(relative_accelerations**2)
            + self._jerk_weight * np.sum(jerk**2)
            + self._jerk_change_weight * np.sum(jerk_difference**2)
        )

    def _constraint_margin_summary(
        self,
        candidate: tuple[np.ndarray, ...],
        *,
        constraints_enforced: bool,
        effective_descent_allowed: bool,
        camera_fov_constraint_enabled: bool = True,
    ) -> dict[str, float]:
        """Reduce every active linear constraint family to one margin."""
        velocities = candidate[1]
        accelerations = candidate[2]
        jerk = candidate[6]
        summary = {
            "horizontal_velocity": float(
                self.max_horizontal_velocity_m_s
                - np.max(np.abs(velocities[:, :2]))
            ),
            "ascent_velocity": float(
                self.max_ascent_velocity_m_s
                - np.max(velocities[:, 2])
            ),
            "descent_velocity": float(
                np.min(velocities[:, 2])
                + self.max_descent_velocity_m_s
            ),
            "horizontal_acceleration": float(
                self.max_horizontal_acceleration_m_s2
                - np.max(np.abs(accelerations[:, :2]))
            ),
            "vertical_acceleration_upper": float(
                self.max_vertical_acceleration_m_s2
                - np.max(accelerations[:, 2])
            ),
            "vertical_acceleration_lower": float(
                np.min(accelerations[:, 2])
                + self.max_vertical_acceleration_m_s2
            ),
            "jerk": float(
                self.max_jerk_m_s3 - np.max(np.abs(jerk))
            ),
        }
        values = np.asarray(
            self._constraint_matrix @ jerk.reshape(-1)
        )
        if constraints_enforced:
            active_geometry_constraints = [
                "relative_altitude",
                "landing_funnel",
                "landing_pad_contact",
            ]
            if camera_fov_constraint_enabled:
                active_geometry_constraints.insert(2, "camera_fov")
            for name in active_geometry_constraints:
                rows = self._row_slices[name]
                lower_margin = values[rows] - self._lower[rows]
                upper_margin = self._upper[rows] - values[rows]
                finite_margins = np.concatenate(
                    (
                        lower_margin[np.isfinite(lower_margin)],
                        upper_margin[np.isfinite(upper_margin)],
                    )
                )
                summary[name] = float(np.min(finite_margins))
        if effective_descent_allowed:
            names = (
                "descent_alignment_position",
                "descent_alignment_velocity",
            )
        elif constraints_enforced:
            names = ("no_descent",)
        else:
            names = ()
        for name in names:
            rows = self._row_slices[name]
            lower_margin = values[rows] - self._lower[rows]
            upper_margin = self._upper[rows] - values[rows]
            finite_margins = np.concatenate(
                (
                    lower_margin[np.isfinite(lower_margin)],
                    upper_margin[np.isfinite(upper_margin)],
                )
            )
            summary[name] = float(np.min(finite_margins))
        return summary
