"""Command-computation role for the precision-landing ROS facade.

This Python role owns asynchronous OSQP snapshots, validation, reference
construction and command caching.  It never creates a ROS node or publisher;
the facade heartbeat remains the sole MAVROS setpoint publisher.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from .async_landing_control import (
    LandingCommandLimits,
    LandingSolveResult,
    LandingSolveSnapshot,
    is_stale_solution_rejection,
    landing_snapshot_change_rejection,
    validate_landing_solve_result,
)
from .capture_state_history import StateInterpolationError
from .landing_controller import (
    HOLD_FALLBACK,
    P_FEEDFORWARD_CONTROLLER_TYPE,
    LandingControlCommand,
    LandingControlInput,
)
from .production_runtime import PRODUCTION_CONTROLLER_TYPE
from .runtime_types import (
    ACQUISITION_GUIDANCE_CONTROLLER_TYPE,
    LandingPhase,
)
from .target_fusion import TargetEstimate, validate_covariance


_ACQUISITION_COMMAND_MAX_AGE_S = 0.25


class CommandRuntimeMixin:
    """OSQP/reference/cache processing owned by the one ROS facade."""

    def _stage_position_only_acquisition_command(
        self,
        now: float,
        clearance_m: float,
        *,
        use_down_marker: bool,
        target_velocity_enu_m_s: np.ndarray | None = None,
    ) -> tuple[float, float] | None:
        """Pursue live trailer position while ArUco learns target speed."""
        vehicle_position = self.vehicle_position_enu
        vehicle_velocity = self.vehicle_velocity_enu
        if vehicle_position is None:
            return None
        vehicle_position = np.asarray(vehicle_position, dtype=float)
        measured_velocity = np.asarray(vehicle_velocity, dtype=float)
        if (
            vehicle_position.shape != (3,)
            or measured_velocity.shape != (3,)
            or not np.all(np.isfinite(vehicle_position))
            or not np.all(np.isfinite(measured_velocity))
        ):
            return None

        # Before optical motion-model qualification, keep one continuous guidance
        # reference: the latest trailer marker position.  Switching this servo
        # between single-frame ArUco and odometry positions on every detector
        # Bool displaced the command reference and pushed the marker back out
        # of view.  ArUco is still the sole source of target velocity and the
        # handoff qualification; trailer twist is never consumed here.
        raw_pose = self._latest_trailer_marker_pose(now)
        if raw_pose is None:
            return None
        target_position = np.asarray(raw_pose[0], dtype=float).copy()
        target_yaw = float(raw_pose[1])
        error_xy = target_position[:2] - vehicle_position[:2]
        distance_xy = float(np.linalg.norm(error_xy))
        if not math.isfinite(distance_xy):
            return None

        maximum_speed = self._float("position_only_pursuit_max_speed_m_s")
        desired_velocity = np.zeros(3, dtype=float)
        optical_velocity = target_velocity_enu_m_s
        optical_velocity_valid = bool(
            optical_velocity is not None
            and np.asarray(optical_velocity).shape == (3,)
            and np.all(np.isfinite(optical_velocity))
        )
        # Position-only pursuit is sufficient to reach the trailer, but it has
        # a permanent e = v_target / Kp lag once the deck is moving.  That lag
        # repeatedly pushed the marker out of the downward FOV before the
        # motion model could qualify. As soon as successive ArUco
        # captures qualify an optical velocity, use that camera-only estimate
        # as feed-forward while keeping this level acquisition controller in
        # charge.  Trailer odometry twist is never read or substituted here.
        # Once optical feed-forward removes the moving-target steady-state
        # lag, use the lower tracking gain.  Retaining the position-only gain
        # of 3/s amplified small odometry/vision disagreement into the
        # observed left-right velocity oscillation.
        horizontal_gain = self._float(
            "landing_p_horizontal_gain"
            if optical_velocity_valid
            else "position_only_pursuit_gain_s_inv"
        )
        correction_velocity = horizontal_gain * error_xy
        if (
            not optical_velocity_valid
            and (
                use_down_marker
                or self._down_marker_detection_live(now)
            )
        ):
            # Only after the first live camera observation, bound unknown-
            # motion correction to a speed that can stop inside the capture
            # radius. Applying a distance envelope before detection leaves a
            # 9 m/s target in the measured 6.8 m pursuit equilibrium.
            capture_speed = math.sqrt(
                2.0
                * self._float(
                    "relative_landing_mpc_max_horizontal_acceleration_m_s2"
                )
                * self._float("approach_capture_radius_m")
            )
            correction_speed = float(np.linalg.norm(correction_velocity))
            if (
                correction_speed > capture_speed
                and correction_speed > 1.0e-9
            ):
                correction_velocity *= capture_speed / correction_speed
        desired_velocity[:2] = correction_velocity
        if optical_velocity_valid:
            assert optical_velocity is not None
            desired_velocity[:2] += np.asarray(
                optical_velocity[:2], dtype=float
            )
            # Preserve the aruco_capture_* provenance set by the optical
            # estimator; acquisition must not disguise it as odometry data.
        else:
            self.control_target_velocity_source = (
                "acquisition_odometry_position_servo"
            )

        desired_horizontal_speed = float(
            np.linalg.norm(desired_velocity[:2])
        )
        if desired_horizontal_speed > maximum_speed:
            desired_velocity[:2] *= maximum_speed / desired_horizontal_speed
        vertical_target = target_position[2] + max(0.0, float(clearance_m))
        vertical_error = vertical_target - float(vehicle_position[2])
        vertical_limit = self._float("landing_p_vertical_velocity_limit_m_s")
        desired_velocity[2] = float(
            np.clip(
                self._float("landing_p_vertical_gain") * vertical_error,
                -vertical_limit,
                vertical_limit,
            )
        )

        # Keep the absolute acquisition command continuous when the moving
        # target crosses the camera axis.  Seed from measured velocity after
        # a gap so this is a bounded command slew, not an accumulated outer
        # velocity integrator.
        previous_update = self.last_position_servo_update_monotonic_s
        if previous_update is None or float(now) - previous_update > 0.25:
            seed = measured_velocity.copy()
            seed_speed = float(np.linalg.norm(seed[:2]))
            if seed_speed > maximum_speed:
                seed[:2] *= maximum_speed / seed_speed
            self.last_position_servo_velocity_enu = seed
        dt = (
            1.0 / self._float("control_rate_hz")
            if previous_update is None
            else max(0.0, min(float(now) - previous_update, 0.20))
        )
        change = desired_velocity - self.last_position_servo_velocity_enu
        horizontal_change = float(np.linalg.norm(change[:2]))
        maximum_change = (
            self._float("position_only_pursuit_acceleration_m_s2") * dt
        )
        if horizontal_change > maximum_change and horizontal_change > 1.0e-9:
            change[:2] *= maximum_change / horizontal_change
        change[2] = float(
            np.clip(change[2], -maximum_change, maximum_change)
        )
        command_velocity = self.last_position_servo_velocity_enu + change
        self.last_position_servo_velocity_enu = command_velocity.copy()
        self.last_position_servo_update_monotonic_s = float(now)

        self._invalidate_solver_context(clear_cached=True)
        disabled = np.full(3, np.nan, dtype=float)
        command = LandingControlCommand(
            disabled,
            command_velocity,
            disabled,
            target_yaw,
            False,
            True,
            False,
            valid=True,
            degraded=False,
            status=self.control_target_velocity_source,
            solve_time_s=0.0,
            controller_type=ACQUISITION_GUIDANCE_CONTROLLER_TYPE,
            primary_controller_type=PRODUCTION_CONTROLLER_TYPE,
        )
        self._record_landing_command(command)
        self._cache_command(command)
        self._acquisition_command_monotonic_s = time.monotonic()
        self._acquisition_command_phase = self.phase.value
        self._acquisition_command_time_reset_count = self.time_reset_count
        self.last_requested_clearance_m = float(clearance_m)
        self.last_landing_horizontal_error_m = distance_xy
        # Report relative speed only when it comes from successive camera
        # captures.  A position-servo residual is not sensor truth.
        self.last_landing_relative_horizontal_speed_m_s = (
            float(
                np.linalg.norm(
                    measured_velocity[:2]
                    - np.asarray(optical_velocity[:2], dtype=float)
                )
            )
            if optical_velocity_valid and optical_velocity is not None
            else None
        )
        yaw_error = abs(
            math.atan2(
                math.sin(self.vehicle_yaw_enu_rad - target_yaw),
                math.cos(self.vehicle_yaw_enu_rad - target_yaw),
            )
        )
        return distance_xy, yaw_error
    def _solver_tick(self) -> None:
        """Apply one result and start at most one immutable snapshot."""
        worker = self.solver_worker
        if worker is None:
            return
        result = worker.take_completed()
        now = time.monotonic()
        with self._solver_context_lock:
            if result is not None:
                self._apply_solver_result(result, now)
            self._expire_cached_solver_command(now)
            if worker.has_outstanding:
                return
            snapshot = self._pending_solver_snapshot
            if snapshot is None:
                return
            if (
                snapshot.phase != self.phase.value
                or snapshot.time_reset_count != self.time_reset_count
            ):
                self._invalidate_solver_context(clear_cached=False)
                self.stale_solution_rejected_count += 1
                return
            # Claim this exact generation before submitting.  A control tick
            # may publish a newer pending snapshot afterwards; it must not be
            # erased when this older submission completes.
            self._pending_solver_snapshot = None
            try:
                worker.submit(snapshot)
            except Exception as exc:
                self.consecutive_solver_failures += 1
                fallback_snapshot = self._fresh_fallback_snapshot(
                    snapshot, time.monotonic()
                )
                command = self._async_failure_command(
                    fallback_snapshot,
                    "worker submit exception: "
                    f"{type(exc).__name__}: {exc}",
                    0.0,
                )
                self._record_landing_command(command)
                self._cached_solver_snapshot = (
                    fallback_snapshot if command.valid else None
                )
                self._cache_command(command)
    def _apply_solver_result(
        self, result: LandingSolveResult, now: float
    ) -> None:
        """Validate one completed result and update only the command cache."""
        if result.snapshot.generation <= getattr(
            self, "_revoked_solver_generation", 0
        ):
            # A phase transition, HOLD, time reset, or synchronous P command
            # explicitly revoked this generation.  Its late worker result is
            # expected scheduling fallout, not a solver/staleness failure.
            self.production_stack.solver.clear_warm_start()
            return
        current = self._latest_solver_snapshot
        if current is None:
            rejection = "no_current_snapshot"
        else:
            rejection = validate_landing_solve_result(
                result,
                current,
                self.solution_acceptance_limits,
                now_monotonic_s=now,
                current_phase=self.phase.value,
                current_time_reset_count=self.time_reset_count,
                last_applied_generation=(
                    self._last_processed_solver_generation
                ),
            )
        self._last_processed_solver_generation = max(
            self._last_processed_solver_generation,
            result.snapshot.generation,
        )
        if rejection is not None:
            # The OSQP workspace advances its shifted primal and previous
            # jerk while constructing a successful result.  If the async
            # snapshot audit rejects that result, PX4 never applies its jerk;
            # retaining it would make the next objective depend on a command
            # that was never flown.
            self.production_stack.solver.clear_warm_start()
            context_invalidated = rejection in {
                "no_current_snapshot",
                "out_of_order_solution",
                "phase_changed",
                "time_reset_changed",
                "current_snapshot_phase_mismatch",
                "current_snapshot_time_reset_mismatch",
            }
            if rejection == "no_current_snapshot" or (
                is_stale_solution_rejection(rejection)
            ):
                self.stale_solution_rejected_count += 1
            if context_invalidated:
                # A hold, phase transition, time reset, or newer generation
                # deliberately revoked this solve.  Never let its late result
                # replace the command selected by that newer safety context.
                return
            if (
                self.consecutive_solver_failures == 0
                and result.command is not None
                and result.command.mpc_attempted
            ):
                control_input = result.snapshot.control_input
                pad_position = control_input.landing_pad_position_enu_m
                pad_velocity = control_input.landing_pad_velocity_enu_m_s
                pad_acceleration = (
                    control_input.landing_pad_acceleration_enu_m_s2
                )
                relative_z = (
                    math.nan
                    if pad_position is None
                    else float(
                        control_input.vehicle_position_enu_m[2]
                        - pad_position[2]
                    )
                )
                relative_vz = (
                    math.nan
                    if pad_velocity is None
                    else float(
                        control_input.vehicle_velocity_enu_m_s[2]
                        - pad_velocity[2]
                    )
                )

                def compact(vector: object) -> str:
                    if vector is None:
                        return "none"
                    return np.array2string(
                        np.asarray(vector, dtype=float),
                        precision=3,
                        separator=",",
                        suppress_small=True,
                    )

                self.get_logger().warning(
                    "first OSQP result rejection: "
                    f"rejection={rejection} "
                    f"status={result.command.status} "
                    f"command_v={compact(result.command.velocity_setpoint_enu_m_s)} "
                    f"vehicle_v={compact(control_input.vehicle_velocity_enu_m_s)} "
                    f"vehicle_a={compact(control_input.vehicle_acceleration_enu_m_s2)} "
                    f"pad_v={compact(pad_velocity)} "
                    f"pad_a={compact(pad_acceleration)} "
                    f"relative_z={relative_z:.3f} "
                    f"relative_vz={relative_vz:.3f} "
                    f"constraints={control_input.landing_constraints_enabled} "
                    f"descent={control_input.descent_allowed}"
                )
            self.consecutive_solver_failures += 1
            fallback_snapshot = self._fresh_fallback_snapshot(current, now)
            command = self._async_failure_command(
                fallback_snapshot,
                rejection,
                result.elapsed_s,
            )
            self._record_landing_command(command)
            self._cached_solver_snapshot = (
                fallback_snapshot if command.valid else None
            )
            self._cache_command(command)
            return
        command = result.command
        if command is None:
            raise RuntimeError("validated solver result has no command")
        self.consecutive_solver_failures = 0
        self._record_landing_command(command)
        self._cached_solver_snapshot = result.snapshot
        self._cache_command(command)
    def _expire_cached_solver_command(self, now: float) -> None:
        """Replace an aged or phase-invalid solved command with fallback."""
        command = self.command_cache
        if (
            command is not None
            and command.controller_type
            == ACQUISITION_GUIDANCE_CONTROLLER_TYPE
        ):
            if self._cached_acquisition_command_valid(now):
                return
            self.stale_solution_rejected_count += 1
            velocity = self.vehicle_velocity_enu
            if (
                velocity is None
                or np.asarray(velocity).shape != (3,)
                or not np.all(np.isfinite(velocity))
            ):
                self.command_cache = None
                return
            # A brief camera/control callback overrun must never inject an
            # absolute zero-velocity position HOLD into a 9 m/s acquisition.
            # Coast at the measured horizontal velocity and level Z until the
            # next synchronous acquisition tick refreshes the command.
            disabled = np.full(3, np.nan, dtype=float)
            coast_velocity = np.asarray(velocity, dtype=float).copy()
            coast_velocity[2] = 0.0
            hold = LandingControlCommand(
                disabled,
                coast_velocity,
                disabled,
                self.vehicle_yaw_enu_rad,
                False,
                True,
                False,
                valid=False,
                degraded=True,
                status="aruco_acquisition_command_stale_coast",
                solve_time_s=0.0,
                controller_type="hold",
                primary_controller_type=PRODUCTION_CONTROLLER_TYPE,
            )
            self._record_landing_command(hold)
            self._cache_command(hold)
            return
        if (
            command is not None
            and command.controller_type == "transition_coast"
        ):
            if self._cached_transition_command_valid(now):
                return
            self.stale_solution_rejected_count += 1
            position = self.vehicle_position_enu
            if (
                position is None
                or np.asarray(position).shape != (3,)
                or not np.all(np.isfinite(position))
            ):
                self.command_cache = None
                return
            hold = LandingControlCommand.hold(
                position,
                self.vehicle_yaw_enu_rad,
                valid=False,
                degraded=True,
                status="phase_transition_command_stale",
                primary_controller_type=PRODUCTION_CONTROLLER_TYPE,
            )
            self._record_landing_command(hold)
            self._cache_command(hold)
            return
        if (
            command is not None
            and command.controller_type == P_FEEDFORWARD_CONTROLLER_TYPE
            and command.primary_controller_type
            == P_FEEDFORWARD_CONTROLLER_TYPE
        ):
            if self._cached_p_command_valid(now):
                return
            position = self.vehicle_position_enu
            if (
                position is None
                or np.asarray(position).shape != (3,)
                or not np.all(np.isfinite(position))
            ):
                self.command_cache = None
                return
            hold = LandingControlCommand.hold(
                position,
                self.vehicle_yaw_enu_rad,
                valid=False,
                degraded=True,
                status="p_feedforward_command_stale",
                primary_controller_type=P_FEEDFORWARD_CONTROLLER_TYPE,
            )
            self._record_landing_command(hold)
            self._cache_command(hold)
            return
        snapshot = self._cached_solver_snapshot
        if snapshot is None:
            return
        phase_changed = snapshot.phase != self.phase.value
        reset_changed = snapshot.time_reset_count != self.time_reset_count
        age = max(0.0, now - snapshot.created_monotonic_s)
        stale = age > self.solution_acceptance_limits.maximum_solution_age_s
        current = self._latest_solver_snapshot
        state_rejection: str | None = None
        if current is None:
            state_rejection = "current_snapshot_missing"
        else:
            if current.phase != self.phase.value:
                state_rejection = "current_snapshot_phase_mismatch"
            elif current.time_reset_count != self.time_reset_count:
                state_rejection = "current_snapshot_time_reset_mismatch"
            else:
                state_rejection = landing_snapshot_change_rejection(
                    snapshot,
                    current,
                    self.solution_acceptance_limits,
                )
        if not (phase_changed or reset_changed or stale or state_rejection):
            return
        self.stale_solution_rejected_count += 1
        self.consecutive_solver_failures += 1
        reason = (
            "cached_solution_phase_changed"
            if phase_changed
            else (
                "cached_solution_time_reset"
                if reset_changed
                else (
                    "cached_solution_stale"
                    if stale
                    else f"cached_solution_{state_rejection}"
                )
            )
        )
        fallback_snapshot = self._fresh_fallback_snapshot(current, now)
        command = self._async_failure_command(
            fallback_snapshot, reason, 0.0
        )
        self._record_landing_command(command)
        self._cached_solver_snapshot = (
            fallback_snapshot if command.valid else None
        )
        self._cache_command(command)
    def _invalidate_solver_context(self, *, clear_cached: bool) -> None:
        """Revoke queued/current snapshots and cached control context."""
        with self._solver_context_lock:
            self._revoked_solver_generation = max(
                getattr(self, "_revoked_solver_generation", 0),
                self._solver_snapshot_generation,
            )
            self._last_processed_solver_generation = max(
                self._last_processed_solver_generation,
                self._solver_snapshot_generation,
            )
            self._pending_solver_snapshot = None
            self._latest_solver_snapshot = None
            self._control_tick_snapshot_candidate = None
            self._cached_p_command_monotonic_s = None
            self._cached_p_command_phase = None
            self._cached_p_command_time_reset_count = None
            self._acquisition_command_monotonic_s = None
            self._acquisition_command_phase = None
            self._acquisition_command_time_reset_count = None
            self._transition_command_monotonic_s = None
            self._transition_command_phase = None
            self._transition_command_time_reset_count = None
            if clear_cached:
                self._cached_solver_snapshot = None
    def _fresh_fallback_snapshot(
        self,
        snapshot: LandingSolveSnapshot | None,
        now: float,
    ) -> LandingSolveSnapshot | None:
        """Return only a current-phase snapshot suitable for failure HOLD."""
        if snapshot is None:
            return None
        age = float(now) - snapshot.created_monotonic_s
        if (
            not math.isfinite(age)
            or age < 0.0
            or age
            > self.solution_acceptance_limits.maximum_solution_age_s
            or snapshot.phase != self.phase.value
            or snapshot.time_reset_count != self.time_reset_count
        ):
            return None
        return snapshot
    def _cache_time_reset_hold(self) -> None:
        """Replace any pre-reset command without publishing from a callback."""
        position = self.vehicle_position_enu
        yaw = float(self.vehicle_yaw_enu_rad)
        if (
            position is None
            or np.asarray(position).shape != (3,)
            or not np.all(np.isfinite(position))
            or not math.isfinite(yaw)
        ):
            self.command_cache = None
            return
        self._cache_command(
            LandingControlCommand.hold(
                position,
                yaw,
                valid=False,
                degraded=True,
                status="time_reset_hold",
            )
        )
        self._time_reset_hold_pending = False
    def _async_failure_command(
        self,
        current: LandingSolveSnapshot | None,
        reason: str,
        elapsed_s: float,
    ) -> LandingControlCommand:
        """Build a fresh moving-deck HOLD without reusing solver output."""
        deadline_missed = "deadline" in str(reason)
        elapsed = max(0.0, float(elapsed_s))
        position = self.vehicle_position_enu
        if (
            position is None
            or np.asarray(position).shape != (3,)
            or not np.all(np.isfinite(position))
        ):
            position = (
                np.zeros(3, dtype=float)
                if current is None
                else current.vehicle_position_enu_m
            )
        yaw = float(self.vehicle_yaw_enu_rad)
        if not np.isfinite(yaw):
            yaw = (
                0.0
                if current is None
                else current.control_input.vehicle_yaw_enu_rad
            )
        if current is not None and current.control_input.has_landing_pad_context:
            control_input = current.control_input
            pad_position = np.asarray(
                control_input.landing_pad_position_enu_m,
                dtype=float,
            )
            pad_velocity = np.asarray(
                control_input.landing_pad_velocity_enu_m_s,
                dtype=float,
            )
            if (
                pad_position.shape == (3,)
                and pad_velocity.shape == (3,)
                and np.all(np.isfinite(pad_position))
                and np.all(np.isfinite(pad_velocity))
            ):
                horizontal_limit = float(
                    control_input.horizontal_velocity_limit_m_s
                )
                vertical_limit = float(
                    control_input.vertical_velocity_limit_m_s
                )
                pad_position = np.asarray(
                    control_input.landing_pad_position_enu_m, dtype=float
                )
                pad_velocity = np.asarray(
                    control_input.landing_pad_velocity_enu_m_s, dtype=float
                )
                # Preserve the flight-verified moving-deck recovery contract:
                # one rejected asynchronous solve still follows the current
                # pad velocity and removes relative position error.  A second
                # 50 Hz slew limiter reduced this bounded command below the
                # 1.62 m/s^2 centripetal acceleration required at 9 m/s.
                safe_velocity = pad_velocity.copy()
                safe_velocity[:2] += self._float(
                    "landing_p_horizontal_gain"
                ) * (pad_position[:2] - np.asarray(position)[:2])
                horizontal_speed = float(np.linalg.norm(safe_velocity[:2]))
                if horizontal_speed > horizontal_limit:
                    safe_velocity[:2] *= horizontal_limit / horizontal_speed
                safe_velocity[2] = float(
                    np.clip(safe_velocity[2], -vertical_limit, vertical_limit)
                )
                disabled = np.full(3, np.nan, dtype=float)
                return LandingControlCommand(
                    position_setpoint_enu_m=disabled,
                    velocity_setpoint_enu_m_s=safe_velocity,
                    acceleration_setpoint_enu_m_s2=disabled,
                    yaw_enu_rad=yaw,
                    position_enabled=False,
                    velocity_enabled=True,
                    acceleration_enabled=False,
                    valid=False,
                    degraded=True,
                    status=f"async_solver_failed_coast_hold: {reason}",
                    solve_time_s=elapsed,
                    controller_type="hold",
                    primary_controller_type=PRODUCTION_CONTROLLER_TYPE,
                    deadline_missed=deadline_missed,
                    mpc_attempted=True,
                    mpc_success=False,
                    fallback_used=HOLD_FALLBACK,
                )
        return LandingControlCommand.hold(
            position,
            yaw,
            valid=False,
            degraded=True,
            status=f"async_solver_failed_hold: {reason}",
            solve_time_s=elapsed,
            primary_controller_type=PRODUCTION_CONTROLLER_TYPE,
            deadline_missed=deadline_missed,
            mpc_attempted=True,
            mpc_success=False,
            fallback_used=HOLD_FALLBACK,
        )
    def _cached_solver_command_valid(self, now: float) -> bool:
        """Return whether mission gates may rely on the cached command."""
        command = self.command_cache
        if (
            command is not None
            and command.controller_type == P_FEEDFORWARD_CONTROLLER_TYPE
            and command.primary_controller_type
            == P_FEEDFORWARD_CONTROLLER_TYPE
        ):
            return self._cached_p_command_valid(now)
        snapshot = self._cached_solver_snapshot
        current = self._latest_solver_snapshot
        if (
            snapshot is None
            or current is None
            or command is None
            or not command.valid
        ):
            return False
        if not (
            snapshot.phase == self.phase.value == current.phase
            and snapshot.time_reset_count
            == self.time_reset_count
            == current.time_reset_count
            and 0.0
            <= now - snapshot.created_monotonic_s
            <= self.solution_acceptance_limits.maximum_solution_age_s
        ):
            return False
        return (
            landing_snapshot_change_rejection(
                snapshot,
                current,
                self.solution_acceptance_limits,
            )
            is None
        )
    def _cached_p_command_valid(self, now: float) -> bool:
        """Validate one synchronous P command without a solver snapshot."""
        command = self.command_cache
        created = self._cached_p_command_monotonic_s
        if (
            command is None
            or created is None
            or not command.valid
            or command.controller_type != P_FEEDFORWARD_CONTROLLER_TYPE
            or command.primary_controller_type
            != P_FEEDFORWARD_CONTROLLER_TYPE
            or command.position_enabled
            or not command.velocity_enabled
            or command.acceleration_enabled
            or command.acceleration_enabled_axes
            != (False, False, False)
            or command.mpc_attempted
            or command.mpc_success is not None
            or self._cached_p_command_phase != self.phase.value
            or self._cached_p_command_time_reset_count
            != self.time_reset_count
        ):
            return False
        age = float(now) - created
        return bool(
            math.isfinite(age)
            and 0.0 <= age
            <= self.solution_acceptance_limits.maximum_solution_age_s
        )
    def _cached_acquisition_command_valid(self, now: float) -> bool:
        """Reject a stale position-only intercept command fail-closed."""
        command = self.command_cache
        created = self._acquisition_command_monotonic_s
        if (
            command is None
            or created is None
            or not command.valid
            or command.controller_type
            != ACQUISITION_GUIDANCE_CONTROLLER_TYPE
            or command.position_enabled
            or not command.velocity_enabled
            or command.acceleration_enabled
            or command.acceleration_enabled_axes
            != (False, False, False)
            or command.mpc_attempted
            or command.mpc_success is not None
            or self.phase
            not in {LandingPhase.APPROACH, LandingPhase.MARKER_TRACK_DOWN}
            or self._acquisition_command_phase != self.phase.value
            or self._acquisition_command_time_reset_count
            != self.time_reset_count
        ):
            return False
        age = float(now) - created
        return bool(
            math.isfinite(age)
            and 0.0 <= age
            <= max(
                self.solution_acceptance_limits.maximum_solution_age_s,
                _ACQUISITION_COMMAND_MAX_AGE_S,
            )
        )
    def _cached_transition_command_valid(self, now: float) -> bool:
        """Bound a phase-transition coast to one normal command lifetime."""
        command = self.command_cache
        created = self._transition_command_monotonic_s
        if (
            command is None
            or created is None
            or not command.valid
            or command.controller_type != "transition_coast"
            or command.position_enabled
            or not command.velocity_enabled
            or command.acceleration_enabled
            or command.acceleration_enabled_axes
            != (False, False, False)
            or self._transition_command_phase != self.phase.value
            or self._transition_command_time_reset_count
            != self.time_reset_count
        ):
            return False
        age = float(now) - created
        return bool(
            math.isfinite(age)
            and 0.0 <= age
            <= self.solution_acceptance_limits.maximum_solution_age_s
        )
    def _stage_trailer_relative_command(
        self,
        now: float,
        clearance_m: float,
        camera: str | None,
        *,
        disable_descent: bool = False,
        require_solver_success: bool = True,
        relative_descent_speed_m_s: float | None = None,
    ) -> tuple[float, float, float] | None:
        self.last_requested_clearance_m = float(clearance_m)
        try:
            control_px4_time_s = self._control_epoch_px4_time_s()
        except ValueError:
            return None
        estimate = self._landing_target_enu(
            now,
            camera,
            control_px4_time_s=control_px4_time_s,
        )
        if estimate is None:
            return None
        (
            deck,
            trailer_velocity,
            trailer_yaw,
            target_estimate_snapshot,
        ) = estimate
        target = deck.copy()
        target[2] = deck[2] + max(0.0, float(clearance_m))
        # Record geometry from this control tick's prediction, independently
        # of whether the asynchronous solver has completed the newly staged
        # snapshot.  Otherwise status keeps reporting the last successful
        # snapshot and makes a moving target look artificially latched.
        horizontal_error = float(
            np.linalg.norm(self.vehicle_position_enu[:2] - deck[:2])
        )
        relative_speed = float(
            np.linalg.norm(
                self.vehicle_velocity_enu[:2] - trailer_velocity[:2]
            )
        )
        yaw_error = abs(
            math.atan2(
                math.sin(self.vehicle_yaw_enu_rad - trailer_yaw),
                math.cos(self.vehicle_yaw_enu_rad - trailer_yaw),
            )
        )
        self.last_landing_horizontal_error_m = horizontal_error
        self.last_landing_relative_horizontal_speed_m_s = relative_speed

        if relative_descent_speed_m_s is None:
            if disable_descent:
                relative_descent_speed_m_s = 0.0
            elif self.phase == LandingPhase.PRECISION_DESCENT:
                relative_descent_speed_m_s = self._float(
                    "precision_descent_speed_m_s"
                )
            elif self.phase == LandingPhase.FINAL_APPROACH:
                relative_descent_speed_m_s = (
                    self._final_descent_speed(now)
                )
            elif self.phase == LandingPhase.TOUCHDOWN_CONFIRM:
                relative_descent_speed_m_s = self._float(
                    "touchdown_contact_settle_speed_m_s"
                )
            else:
                relative_descent_speed_m_s = 0.0

        success = self._stage_control_command(
            target,
            trailer_velocity,
            trailer_yaw,
            disable_descent=disable_descent,
            relative_descent_speed_m_s=relative_descent_speed_m_s,
            control_px4_time_s=control_px4_time_s,
            target_estimate_snapshot=target_estimate_snapshot,
        )
        if not success and require_solver_success:
            return None
        return horizontal_error, relative_speed, yaw_error
    def _estimate_landing_pad_acceleration(
        self, pad_velocity_enu_m_s: np.ndarray
    ) -> np.ndarray:
        """Return acceleration coherent with the robust ArUco motion fit."""
        velocity = np.asarray(pad_velocity_enu_m_s, dtype=float)
        acceleration = np.asarray(
            self.landing_pad_acceleration_enu, dtype=float
        )
        if (
            velocity.shape != (3,)
            or acceleration.shape != (3,)
            or not np.all(np.isfinite(velocity))
            or not np.all(np.isfinite(acceleration))
        ):
            return np.zeros(3, dtype=float)
        return acceleration.copy()
    def _landing_pad_prediction_horizon(
        self,
        pad_position_enu_m: np.ndarray,
        pad_velocity_enu_m_s: np.ndarray,
        pad_acceleration_enu_m_s2: np.ndarray,
        target_estimate_snapshot: TargetEstimate | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Build one same-epoch ArUco-derived RK4 pad horizon.

        The fused estimator owns the current pad position.  Camera history
        owns speed, course, turn rate, and tangential acceleration.  Rebase
        those motion terms onto the fused current position before propagating
        so an old raw image coordinate can never become the MPC horizon's
        origin.
        """
        position = np.asarray(pad_position_enu_m, dtype=float)
        velocity = np.asarray(pad_velocity_enu_m_s, dtype=float)
        acceleration = np.asarray(pad_acceleration_enu_m_s2, dtype=float)
        estimate = (
            target_estimate_snapshot
            if target_estimate_snapshot is not None
            else self.last_target_estimate
        )
        cached_motion_state = self._trailer_prediction_cache
        if (
            position.shape != (3,)
            or velocity.shape != (3,)
            or acceleration.shape != (3,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(velocity))
            or not np.all(np.isfinite(acceleration))
            or estimate is None
            or not estimate.valid
            or estimate.position_enu_m is None
            or estimate.covariance is None
            or cached_motion_state is None
        ):
            self.last_trailer_prediction_reason = (
                "rk4_horizon_state_unavailable"
            )
            return None
        epoch_s = float(estimate.stamp_s)
        estimate_position = np.asarray(estimate.position_enu_m, dtype=float)
        estimate_covariance = np.asarray(estimate.covariance, dtype=float)
        if (
            not math.isfinite(epoch_s)
            or epoch_s <= 0.0
            or estimate_position.shape != (3,)
            or not np.all(np.isfinite(estimate_position))
            or estimate_covariance.shape != (8, 8)
            or not np.all(np.isfinite(estimate_covariance))
            or np.linalg.norm(position - estimate_position) > 1.0e-4
        ):
            self.last_trailer_prediction_reason = (
                "rk4_horizon_epoch_mismatch"
            )
            return None

        # Bring the independent ArUco motion covariance to the exact fused
        # estimate epoch before rebasing its mean position.
        covariance_prediction = self.trailer_motion_predictor.predict(
            cached_motion_state, epoch_s
        )
        if (
            not covariance_prediction.valid
            or covariance_prediction.state is None
        ):
            self.last_trailer_prediction_reason = (
                "rk4_horizon_"
                f"{covariance_prediction.reason}"
            )
            return None
        motion_covariance = np.asarray(
            covariance_prediction.state.covariance, dtype=float
        )
        try:
            position_covariance = validate_covariance(
                estimate_covariance[:2, :2],
                2,
                "fused target horizontal position",
            )
        except ValueError:
            self.last_trailer_prediction_reason = (
                "rk4_horizon_invalid_position_covariance"
            )
            return None
        # Bicycle yaw is the velocity course/phase, not the physical ArUco
        # orientation.  Preserve the camera-motion model's course variance;
        # the fused marker-yaw variance has different semantics.
        yaw_variance = float(motion_covariance[2, 2])
        speed_variance = float(motion_covariance[3, 3])
        speed_covariance_exceeded = bool(
            math.isfinite(speed_variance)
            and speed_variance
            > self._float("vision_velocity_maximum_variance_m2_s2")
        )
        if (
            not math.isfinite(yaw_variance)
            or yaw_variance < 0.0
            or not math.isfinite(speed_variance)
            or speed_variance < 0.0
            or (
                speed_covariance_exceeded
                and (
                    not self._terminal_motion_covariance_bridge_allowed(
                        estimate
                    )
                    or self.control_target_velocity_source
                    != "aruco_terminal_hold"
                )
            )
        ):
            self.last_trailer_prediction_reason = (
                "rk4_horizon_motion_covariance_exceeded"
            )
            return None
        rebased_covariance = np.zeros((4, 4), dtype=float)
        rebased_covariance[:2, :2] = position_covariance
        rebased_covariance[2, 2] = max(1.0e-9, yaw_variance)
        rebased_covariance[3, 3] = max(1.0e-9, speed_variance)

        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        if horizontal_speed > 1.0e-3:
            course_yaw = math.atan2(
                float(velocity[1]), float(velocity[0])
            )
            tangential_acceleration = float(
                np.dot(velocity[:2], acceleration[:2])
                / horizontal_speed
            )
            turn_rate = float(
                (
                    velocity[0] * acceleration[1]
                    - velocity[1] * acceleration[0]
                )
                / (horizontal_speed * horizontal_speed)
            )
        else:
            course_yaw = 0.0
            tangential_acceleration = 0.0
            turn_rate = 0.0
        try:
            rebased_state = (
                self.trailer_motion_predictor.state_from_observation(
                    epoch_s,
                    position,
                    velocity,
                    course_yaw,
                    turn_rate,
                    rebased_covariance,
                    tangential_acceleration,
                )
            )
            solver = self.production_stack.solver
            horizon = self.trailer_motion_predictor.predict_horizon(
                rebased_state,
                epoch_s,
                solver.dt_s,
                solver.horizon_steps,
                propagate_covariance=False,
            )
        except ValueError:
            self.last_trailer_prediction_reason = (
                "rk4_horizon_invalid_rebased_state"
            )
            return None
        if (
            not horizon.valid
            or horizon.positions_enu_m is None
            or horizon.velocities_enu_m_s is None
            or horizon.accelerations_enu_m_s2 is None
        ):
            self.last_trailer_prediction_reason = (
                f"rk4_horizon_{horizon.reason}"
            )
            return None
        self.last_trailer_prediction_reason = (
            "rk4_horizon_degraded_terminal_covariance"
            if speed_covariance_exceeded
            else "rk4_horizon_valid"
        )
        return (
            horizon.positions_enu_m,
            horizon.velocities_enu_m_s,
            horizon.accelerations_enu_m_s2,
        )
    def _relative_landing_context(
        self,
        reference_position_enu: np.ndarray,
        reference_velocity_enu: np.ndarray,
        *,
        disable_descent: bool = False,
        relative_descent_speed_m_s: float = 0.0,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        bool,
        bool,
        bool,
        float,
    ] | None:
        """Build pad state and constraint flags without changing phases."""
        landing_phases = {
            LandingPhase.APPROACH,
            LandingPhase.MARKER_TRACK_DOWN,
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
        }
        if (
            self.phase not in landing_phases
            or self.last_requested_clearance_m is None
        ):
            return None
        requested_clearance = max(
            0.0, float(self.last_requested_clearance_m)
        )
        pad_position = (
            reference_position_enu.copy()
            if reference_position_enu.ndim == 1
            else reference_position_enu[0].copy()
        )
        pad_velocity = (
            reference_velocity_enu.copy()
            if reference_velocity_enu.ndim == 1
            else reference_velocity_enu[0].copy()
        )
        pad_position[2] -= requested_clearance
        pad_acceleration = self._estimate_landing_pad_acceleration(
            pad_velocity
        )
        # A level recenter is requested precisely because the vehicle can be
        # outside the funnel. Keeping funnel/deck rows hard in that state can
        # make the recovery horizon unreachable and drive OSQP to inaccurate
        # or infeasible. Dynamic limits and the hard no-descent row remain.
        constraints_enabled = bool(
            not disable_descent
            and self.phase
            in {
                LandingPhase.PRECISION_DESCENT,
                LandingPhase.FINAL_APPROACH,
                LandingPhase.TOUCHDOWN_CONFIRM,
            }
        )
        phase_allows_descent = self.phase in {
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            # ON_GROUND has already been observed before this phase. Keeping
            # the bounded contact target active unloads thrust during the
            # multi-signal dwell while Relative OSQP continues deck matching.
            LandingPhase.TOUCHDOWN_CONFIRM,
        }
        # Safety code may revoke descent for one solve without changing the
        # mission phase. This override is disable-only, so it can never
        # authorize descent from a level phase.
        descent_allowed = bool(
            phase_allows_descent
            and not disable_descent
        )
        # A finite camera FOV is meaningful only while the marker is actually
        # visible. Once the single large board is intentionally bridged by
        # fresh deck-relative geometry near contact, keeping the
        # FOV polygon active makes the QP mathematically infeasible as height
        # approaches zero. Disable only that row family; funnel, deck,
        # velocity, acceleration, jerk and descent-alignment bounds remain.
        camera_fov_constraint_enabled = self.phase not in {
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
        }
        clearance = requested_clearance
        return (
            pad_position,
            pad_velocity,
            pad_acceleration,
            clearance,
            constraints_enabled,
            descent_allowed,
            camera_fov_constraint_enabled,
            (
                max(0.0, float(relative_descent_speed_m_s))
                if descent_allowed
                else 0.0
            ),
        )
    def _measured_target_clearance(
        self,
        estimate: TargetEstimate | None,
        fallback_clearance_m: float,
    ) -> float:
        """Return current vehicle-to-deck clearance or a safe fallback."""
        fallback = max(0.0, float(fallback_clearance_m))
        if self.vehicle_position_enu is None:
            return fallback
        vehicle_position = np.asarray(self.vehicle_position_enu, dtype=float)
        if (
            estimate is not None
            and bool(getattr(estimate, "valid", False))
            and estimate.position_enu_m is not None
        ):
            deck_position = np.asarray(estimate.position_enu_m, dtype=float)
        else:
            return fallback
        if (
            vehicle_position.shape != (3,)
            or deck_position.shape != (3,)
            or not np.all(np.isfinite(vehicle_position))
            or not np.all(np.isfinite(deck_position))
        ):
            return fallback
        clearance = float(vehicle_position[2] - deck_position[2])
        return max(0.0, clearance) if math.isfinite(clearance) else fallback
    def _descent_mpc_clearance_target(
        self,
        nominal_clearance_m: float,
        minimum_clearance_m: float,
        descent_speed_m_s: float,
    ) -> float:
        """Return the floor used by the solver's continuous Z horizon."""
        nominal = float(nominal_clearance_m)
        minimum = float(minimum_clearance_m)
        speed = float(descent_speed_m_s)
        if (
            not math.isfinite(nominal)
            or not math.isfinite(minimum)
            or minimum < 0.0
            or not math.isfinite(speed)
            or speed < 0.0
        ):
            return max(0.0, nominal)
        return min(max(0.0, nominal), minimum)
    def _final_descent_speed(self, now: float) -> float:
        """Taper terminal descent continuously before landing-gear contact."""
        nominal = self._float("final_descent_speed_m_s")
        settle = self._float("touchdown_contact_settle_speed_m_s")
        if (
            not self._fresh_landing_height(now)
            or self.landing_height_distance_m is None
        ):
            return nominal
        distance = float(self.landing_height_distance_m)
        contact_clearance = self._float("touchdown_contact_clearance_m")
        # Continue the configured terminal speed until a continuous braking
        # envelope becomes limiting. Contact still uses the independent
        # settle-speed and touchdown relative-vz gates below.
        braking_distance = max(0.0, distance - contact_clearance)
        braking_speed = math.sqrt(
            2.0
            * self._float("final_descent_braking_acceleration_m_s2")
            * braking_distance
        )
        return min(nominal, max(settle, braking_speed))
    def _final_approach_height_reached(
        self,
        now: float,
        estimate: TargetEstimate | None,
    ) -> bool:
        """Require measured, not merely commanded, terminal height."""
        final_height = self._float("final_approach_height_m")
        measured_clearance = self._measured_target_clearance(
            estimate, math.inf
        )
        if (
            not math.isfinite(final_height)
            or final_height <= 0.0
            or not math.isfinite(measured_clearance)
            or measured_clearance > final_height + 1.0e-6
        ):
            return False
        if not self._fresh_landing_height(now):
            return False
        height = self.landing_height_distance_m
        return bool(
            height is not None
            and math.isfinite(float(height))
            and float(height) <= final_height + 1.0e-6
        )
    def _snapshot_solver_deadline_s(
        self, control_input: LandingControlInput
    ) -> float:
        """Return the fixed Relative-OSQP worker deadline."""
        if not control_input.has_landing_pad_context:
            raise ValueError("OSQP snapshot requires landing-pad context")
        return self.production_stack.solver_deadline_s
    def _snapshot_command_limits(
        self, control_input: LandingControlInput
    ) -> LandingCommandLimits:
        """Capture fixed OSQP bounds for post-solve revalidation."""
        return self.production_stack.command_limits(control_input)
    def _control_epoch_px4_time_s(self) -> float:
        """Return one PX4-domain epoch shared by target and vehicle state."""
        try:
            ros_now_s = self.get_clock().now().nanoseconds * 1.0e-9
            epoch_s = self._ros_time_to_px4_sample_time(ros_now_s)
        except StateInterpolationError:
            epoch_s = self._current_px4_sample_time()
        estimator_state = self.vehicle_response_estimator.state
        if estimator_state is not None:
            epoch_s = max(float(epoch_s), estimator_state.sample_time_s)
        epoch_s = float(epoch_s)
        if not math.isfinite(epoch_s) or epoch_s <= 0.0:
            raise ValueError("invalid control epoch")
        return epoch_s

    def _vehicle_state_for_control(
        self,
        *,
        require_fresh_prediction: bool,
        solve_px4_time_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one timestamp-aligned vehicle snapshot for a solve.

        The response observer is fed only MAVROS message sample timestamps.
        Querying it does not advance the filter. Precision solves fail closed
        if that timestamped state cannot be projected to the current PX4 clock;
        takeoff/acquisition retain the measured position/velocity fallback.
        """
        if self.vehicle_position_enu is None:
            raise ValueError("vehicle position unavailable")
        callback_latest = (
            np.asarray(self.vehicle_position_enu, dtype=float).copy(),
            np.asarray(self.vehicle_velocity_enu, dtype=float).copy(),
            np.asarray(self.vehicle_acceleration_enu, dtype=float).copy(),
        )
        paired_position = self.vehicle_control_position_enu
        paired_velocity = self.vehicle_control_velocity_enu
        paired_time_s = self.vehicle_control_sample_time_s
        paired_valid = bool(
            paired_position is not None
            and paired_velocity is not None
            and paired_time_s is not None
            and math.isfinite(float(paired_time_s))
            and np.asarray(paired_position).shape == (3,)
            and np.asarray(paired_velocity).shape == (3,)
            and np.all(np.isfinite(paired_position))
            and np.all(np.isfinite(paired_velocity))
        )
        if not self.vehicle_response_estimator.initialized:
            if require_fresh_prediction:
                raise ValueError("vehicle response estimator uninitialized")
            return callback_latest
        if solve_px4_time_s is None:
            solve_px4_time_s = self._control_epoch_px4_time_s()
        else:
            solve_px4_time_s = float(solve_px4_time_s)
            if not math.isfinite(solve_px4_time_s) or solve_px4_time_s <= 0.0:
                raise ValueError("invalid vehicle control epoch")
        estimator_state = self.vehicle_response_estimator.state
        if (
            estimator_state is not None
            and solve_px4_time_s < estimator_state.sample_time_s - 1.0e-9
        ):
            raise ValueError("vehicle control epoch is behind estimator")
        applied_acceleration = self.last_command_acceleration_enu
        if applied_acceleration is not None:
            applied_array = np.asarray(applied_acceleration, dtype=float)
            applied_acceleration = (
                applied_array
                if applied_array.shape == (3,)
                and np.all(np.isfinite(applied_array))
                else None
            )
        prediction = self.vehicle_response_estimator.predict(
            solve_px4_time_s, applied_acceleration
        )
        if not prediction.valid or prediction.state is None:
            if require_fresh_prediction:
                raise ValueError(
                    f"vehicle response prediction {prediction.reason}"
                )
            return callback_latest
        if not paired_valid:
            if require_fresh_prediction:
                raise ValueError("timestamp-paired vehicle state unavailable")
            return callback_latest
        state = prediction.state
        assert paired_position is not None
        assert paired_velocity is not None
        assert paired_time_s is not None
        projection_dt_s = float(solve_px4_time_s) - float(paired_time_s)
        if (
            not math.isfinite(projection_dt_s)
            or projection_dt_s < -1.0e-6
            or projection_dt_s
            > self.vehicle_response_estimator.parameters.maximum_prediction_horizon_s
        ):
            if require_fresh_prediction:
                raise ValueError("timestamp-paired vehicle state stale")
            return callback_latest
        projection_dt_s = max(0.0, projection_dt_s)
        acceleration = state.acceleration_enu_m_s2.copy()
        # The observer deliberately allows larger diagnostic transients than
        # the landing model. Clamp only the controller-facing initial state to
        # the dynamics that the Relative MPC can realize.
        horizontal_acceleration_limit = self._float(
            "relative_landing_mpc_max_horizontal_acceleration_m_s2"
        )
        horizontal_acceleration = float(np.linalg.norm(acceleration[:2]))
        if horizontal_acceleration > horizontal_acceleration_limit:
            acceleration[:2] *= (
                horizontal_acceleration_limit / horizontal_acceleration
            )
        vertical_acceleration_limit = self._float(
            "relative_landing_mpc_max_vertical_acceleration_m_s2"
        )
        acceleration[2] = float(
            np.clip(
                acceleration[2],
                -vertical_acceleration_limit,
                vertical_acceleration_limit,
            )
        )
        position = (
            np.asarray(paired_position, dtype=float)
            + projection_dt_s * np.asarray(paired_velocity, dtype=float)
            + 0.5 * projection_dt_s**2 * acceleration
        )
        velocity = (
            np.asarray(paired_velocity, dtype=float)
            + projection_dt_s * acceleration
        )
        return (
            position,
            velocity,
            acceleration,
        )
    def _stage_solver_snapshot(
        self, control_input: LandingControlInput
    ) -> LandingSolveSnapshot:
        """Queue one deep-frozen snapshot at the control-tick boundary."""
        with self._solver_context_lock:
            self._solver_snapshot_generation += 1
            estimate = self.last_target_estimate
            if control_input.has_landing_pad_context and estimate is not None:
                source_stamp = float(estimate.stamp_s)
                covariance = estimate.covariance
            else:
                source_stamp = None
                covariance = None
            snapshot = LandingSolveSnapshot(
                generation=self._solver_snapshot_generation,
                created_monotonic_s=time.monotonic(),
                source_stamp_s=source_stamp,
                phase=self.phase.value,
                time_reset_count=self.time_reset_count,
                control_input=control_input,
                target_covariance=covariance,
                solver_deadline_s=self._snapshot_solver_deadline_s(
                    control_input
                ),
                constraint_limits=self._snapshot_command_limits(
                    control_input
                ),
            )
            if self._control_tick_snapshot_staging:
                # Latest request wins inside one mission decision, but remains
                # invisible to the asynchronous solver until the callback
                # finishes and commits it once.
                self._control_tick_snapshot_candidate = snapshot
            else:
                self._pending_solver_snapshot = snapshot
                self._latest_solver_snapshot = snapshot
        return snapshot
    def _record_landing_command(
        self, command: LandingControlCommand
    ) -> None:
        """Keep only the MPC diagnostics published in compact status."""
        if command.mpc_attempted:
            self.last_mpc_message = command.status
            self.last_mpc_success = bool(command.mpc_success)
            self.last_mpc_solve_time_s = command.solve_time_s
    def _cache_synchronous_p_failure(self, reason: str) -> bool:
        """Cache a fail-closed internal transit-controller error."""
        position = self.vehicle_position_enu
        if (
            position is None
            or np.asarray(position).shape != (3,)
            or not np.all(np.isfinite(position))
        ):
            position = np.zeros(3, dtype=float)
        yaw = float(self.vehicle_yaw_enu_rad)
        if not math.isfinite(yaw):
            yaw = 0.0
        command = LandingControlCommand.hold(
            position,
            yaw,
            valid=False,
            degraded=True,
            status=f"p_feedforward_compute_failed: {reason}",
            primary_controller_type=P_FEEDFORWARD_CONTROLLER_TYPE,
        )
        self.consecutive_solver_failures = 0
        self._record_landing_command(command)
        self._cache_command(command)
        return False
    def _compute_synchronous_p_command(
        self, control_input: LandingControlInput
    ) -> bool:
        """Compute P control on the heartbeat thread and cache only."""
        self._invalidate_solver_context(clear_cached=True)
        try:
            command = self.landing_controller.compute(control_input)
            if not isinstance(command, LandingControlCommand):
                raise TypeError("P controller returned an invalid command")
            if (
                not command.valid
                or command.controller_type
                != P_FEEDFORWARD_CONTROLLER_TYPE
                or command.primary_controller_type
                != P_FEEDFORWARD_CONTROLLER_TYPE
                or command.position_enabled
                or not command.velocity_enabled
                or command.acceleration_enabled
                or command.acceleration_enabled_axes
                != (False, False, False)
                or command.mpc_attempted
                or command.mpc_success is not None
            ):
                raise ValueError("P controller returned an invalid mode")
            velocity = command.velocity_setpoint_enu_m_s
            tolerance = 1.0e-6
            if (
                float(np.linalg.norm(velocity[:2]))
                > control_input.horizontal_velocity_limit_m_s + tolerance
                or abs(float(velocity[2]))
                > control_input.vertical_velocity_limit_m_s + tolerance
            ):
                raise ValueError("P controller exceeded velocity limits")
        except Exception as exc:
            return self._cache_synchronous_p_failure(
                f"{type(exc).__name__}: {exc}"
            )
        self.consecutive_solver_failures = 0
        self._record_landing_command(command)
        self._cache_command(command)
        self._cached_p_command_monotonic_s = time.monotonic()
        self._cached_p_command_phase = self.phase.value
        self._cached_p_command_time_reset_count = self.time_reset_count
        return self._cached_p_command_valid(time.monotonic())
    def _stage_control_command(
        self,
        reference_position_enu: Sequence[float],
        reference_velocity_enu: Sequence[float],
        yaw_enu_rad: float,
        *,
        disable_descent: bool = False,
        relative_descent_speed_m_s: float = 0.0,
        horizontal_velocity_limit_m_s: float | None = None,
        control_px4_time_s: float | None = None,
        target_estimate_snapshot: TargetEstimate | None = None,
    ) -> bool:
        if self.vehicle_position_enu is None:
            return False
        reference_position = np.asarray(reference_position_enu, dtype=float)
        reference_velocity = np.asarray(reference_velocity_enu, dtype=float)
        relative_context = self._relative_landing_context(
            reference_position,
            reference_velocity,
            disable_descent=disable_descent,
            relative_descent_speed_m_s=relative_descent_speed_m_s,
        )
        # Relative MPC owns every complete landing context.  The controller's
        # internal P/feed-forward transit path is used only before that context
        # exists; it is not a selectable production landing controller.
        synchronous_p_path = relative_context is None
        try:
            (
                control_vehicle_position,
                control_vehicle_velocity,
                control_vehicle_acceleration,
            ) = self._vehicle_state_for_control(
                require_fresh_prediction=not synchronous_p_path,
                solve_px4_time_s=control_px4_time_s,
            )
        except Exception as exc:
            self._invalidate_solver_context(clear_cached=True)
            if synchronous_p_path:
                return self._cache_synchronous_p_failure(
                    f"vehicle state rejected: {type(exc).__name__}: {exc}"
                )
            self.consecutive_solver_failures += 1
            command = self._async_failure_command(
                None,
                f"vehicle state rejected: {type(exc).__name__}: {exc}",
                0.0,
            )
            self._record_landing_command(command)
            self._cache_command(command)
            return False
        landing_context: dict[str, object] = {}
        if relative_context is not None:
            (
                pad_position,
                pad_velocity,
                pad_acceleration,
                clearance,
                constraints_enabled,
                descent_allowed,
                camera_fov_constraint_enabled,
                descent_speed,
            ) = relative_context
            pad_horizon = self._landing_pad_prediction_horizon(
                pad_position,
                pad_velocity,
                pad_acceleration,
                target_estimate_snapshot,
            )
            precision_horizon_required = self.phase in {
                LandingPhase.PRECISION_ALIGN,
                LandingPhase.PRECISION_DESCENT,
                LandingPhase.FINAL_APPROACH,
                LandingPhase.TOUCHDOWN_CONFIRM,
            }
            if pad_horizon is None and precision_horizon_required:
                self._invalidate_solver_context(clear_cached=True)
                self.consecutive_solver_failures += 1
                command = self._async_failure_command(
                    None,
                    "landing-pad RK4 horizon unavailable: "
                    f"{self.last_trailer_prediction_reason}",
                    0.0,
                )
                self._record_landing_command(command)
                self._cache_command(command)
                return False
            landing_context = {
                "landing_pad_position_enu_m": pad_position,
                "landing_pad_velocity_enu_m_s": pad_velocity,
                "landing_pad_acceleration_enu_m_s2": pad_acceleration,
                "target_clearance_m": clearance,
                "landing_constraints_enabled": constraints_enabled,
                "descent_allowed": descent_allowed,
                "camera_fov_constraint_enabled": (
                    camera_fov_constraint_enabled
                ),
                "relative_descent_speed_m_s": descent_speed,
            }
            if self._contact_entry_latched or self.phase == (
                LandingPhase.TOUCHDOWN_CONFIRM
            ):
                # The response observer sees a collision impulse at contact.
                # It is not a realizable free-flight initial acceleration and
                # can make every jerk-limited contact QP infeasible.  Once the
                # one-way contact gate has latched, the vehicle is constrained
                # by the deck; use the same RK4 pad acceleration that defines
                # the external horizon for that constrained initial state.
                control_vehicle_acceleration = np.asarray(
                    pad_acceleration, dtype=float
                ).copy()
                control_vehicle_velocity = np.asarray(
                    control_vehicle_velocity, dtype=float
                ).copy()
                control_vehicle_velocity[2] = float(pad_velocity[2])
            # Keep the flight-verified scalar pad-state solver contract.  The
            # RK4 result above is a bounded validity check for the accepted
            # ArUco motion model; timestamp lead is handled by projecting the
            # current target and vehicle to one shared control epoch.
        try:
            if horizontal_velocity_limit_m_s is None:
                horizontal_velocity_limit_m_s = self._float(
                    "landing_p_horizontal_velocity_limit_m_s"
                )
            else:
                horizontal_velocity_limit_m_s = float(
                    horizontal_velocity_limit_m_s
                )
            if relative_context is not None:
                # The solver-failure moving-deck HOLD consumes this shared
                # limit.  Feeding it the 1 m/s P-controller cap forced a
                # 3 m/s trailer command down to exactly 1 m/s at every failed
                # solve, creating the measured brake/catch-up oscillation.
                horizontal_velocity_limit_m_s = self._float(
                    "relative_landing_mpc_max_horizontal_velocity_m_s"
                )
            control_input = LandingControlInput(
                vehicle_position_enu_m=control_vehicle_position,
                vehicle_velocity_enu_m_s=control_vehicle_velocity,
                vehicle_acceleration_enu_m_s2=(
                    control_vehicle_acceleration
                ),
                vehicle_yaw_enu_rad=self.vehicle_yaw_enu_rad,
                reference_positions_enu_m=reference_position,
                reference_velocities_enu_m_s=reference_velocity,
                target_yaw_enu_rad=float(yaw_enu_rad),
                horizontal_velocity_limit_m_s=(
                    horizontal_velocity_limit_m_s
                ),
                vertical_velocity_limit_m_s=self._float(
                    "landing_p_vertical_velocity_limit_m_s"
                ),
                **landing_context,
            )
            if synchronous_p_path:
                success = self._compute_synchronous_p_command(control_input)
            else:
                self._stage_solver_snapshot(control_input)
                success = self._cached_solver_command_valid(
                    time.monotonic()
                )
        except Exception as exc:
            self._invalidate_solver_context(clear_cached=True)
            if synchronous_p_path:
                return self._cache_synchronous_p_failure(
                    f"input rejected: {type(exc).__name__}: {exc}"
                )
            self.consecutive_solver_failures += 1
            command = self._async_failure_command(
                None,
                f"solver snapshot rejected: {type(exc).__name__}: {exc}",
                0.0,
            )
            self._record_landing_command(command)
            self._cache_command(command)
            return False
        return success
    def _cache_hold_command(self, position_enu: Sequence[float]) -> None:
        self._invalidate_solver_context(clear_cached=True)
        self._cache_command(
            LandingControlCommand.hold(
                position_enu,
                self.vehicle_yaw_enu_rad,
                valid=True,
                degraded=False,
                status="hold",
            )
        )
    def _cache_command(self, command: LandingControlCommand) -> None:
        if not isinstance(command, LandingControlCommand):
            raise TypeError("landing command has an invalid type")
        self._cached_p_command_monotonic_s = None
        self._cached_p_command_phase = None
        self._cached_p_command_time_reset_count = None
        self._acquisition_command_monotonic_s = None
        self._acquisition_command_phase = None
        self._acquisition_command_time_reset_count = None
        self._transition_command_monotonic_s = None
        self._transition_command_phase = None
        self._transition_command_time_reset_count = None
        self.command_cache = command
