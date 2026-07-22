"""Mission and safety role for the precision-landing ROS facade.

This Python role owns phase decisions, descent gates, dropout policy and
touchdown handling.  It creates no ROS node and publishes no setpoints; all
effects flow through the facade's single command cache and heartbeat.
"""

from __future__ import annotations

import math
import time

import numpy as np
from mavros_msgs.msg import ExtendedState
from std_srvs.srv import Trigger

from .landing_controller import (
    HOLD_FALLBACK,
    P_FEEDFORWARD_CONTROLLER_TYPE,
    LandingControlCommand,
)
from .landing_safety import (
    DescentPermissionDecision,
    DescentPermissionInput,
    TouchdownGateDecision,
    TouchdownGateInput,
    evaluate_descent_permission,
)
from .production_runtime import PRODUCTION_CONTROLLER_TYPE
from .runtime_types import LandingPhase
from .target_fusion import TargetEstimate, validate_covariance


class MissionRuntimeMixin:
    """Mission state and safety processing owned by the one ROS facade."""

    def _on_start(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if not self._activation_qualified():
            response.success = False
            response.message = (
                "production start rejected: Stage-10 flight qualification "
                "is false"
            )
            return response
        if self.start_requested and self.phase not in (
            LandingPhase.READY,
            LandingPhase.WAITING,
        ):
            response.success = False
            response.message = f"already active in {self.phase.value}"
            return response
        self.start_requested = True
        response.success = True
        response.message = "precision landing start latched"
        return response
    def _activation_qualified(self) -> bool:
        """Allow flight only after qualification or in explicit validation."""
        return bool(
            self._bool("production_qualified")
            or self._bool("validation_mode")
        )
    def _on_hold(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if self.vehicle_position_enu is None:
            response.success = False
            response.message = "local position unavailable"
            return response
        self._enter_failsafe_hold("operator hold request")
        response.success = True
        response.message = "safe hold selected"
        return response
    def _ready(self, now: float) -> bool:
        return bool(
            self.mavros_state is not None
            and self.mavros_state.connected
            and self.vehicle_position_enu is not None
            and np.all(np.isfinite(self.vehicle_position_enu))
            and np.all(np.isfinite(self.vehicle_velocity_enu))
            and now - self.pose_stamp <= self._float("pose_timeout_s")
            and now - self.velocity_stamp
            <= self._float("velocity_timeout_s")
            and self._trailer_fresh(now)
        )
    def _offboard_setpoint_active(self) -> bool:
        """Return whether the current phase expects MAVROS raw setpoints."""
        return self.phase in {
            LandingPhase.PRESTREAM,
            LandingPhase.ARMING,
            LandingPhase.TAKEOFF,
            LandingPhase.APPROACH,
            LandingPhase.MARKER_TRACK_DOWN,
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
            LandingPhase.ABORT_CLIMB,
            LandingPhase.FAILSAFE_HOLD,
        }
    def _strict_relative_deck_clearance_m(
        self, now: float
    ) -> float | None:
        """Return height above the canonical fused marker estimate."""
        if (
            self.vehicle_position_enu is None
            or self.pose_stamp <= 0.0
            or self.velocity_stamp <= 0.0
            or not 0.0 <= now - self.pose_stamp
            <= self._float("pose_timeout_s")
            or not 0.0 <= now - self.velocity_stamp
            <= self._float("velocity_timeout_s")
        ):
            return None
        estimate = self._current_target_estimate()
        if (
            estimate is None
            or not estimate.valid
            or estimate.position_enu_m is None
        ):
            return None
        vehicle = np.asarray(self.vehicle_position_enu, dtype=float)
        deck = np.asarray(estimate.position_enu_m, dtype=float)
        if (
            vehicle.shape != (3,)
            or deck.shape != (3,)
            or not np.all(np.isfinite(vehicle))
            or not np.all(np.isfinite(deck))
        ):
            return None
        clearance = float(vehicle[2] - deck[2])
        if not math.isfinite(clearance) or clearance < 0.0:
            return None
        return clearance
    def _fresh_landing_height(self, now: float) -> bool:
        """Refresh camera/odometry relative height fail-closed."""
        distance = self._strict_relative_deck_clearance_m(now)
        valid = bool(
            distance is not None
            and math.isfinite(float(distance))
            and 0.0 <= float(distance)
            <= self._float("descent_max_relative_height_m")
        )
        self.landing_height_distance_m = (
            float(distance) if valid and distance is not None else None
        )
        if valid and self.landing_height_distance_m is not None:
            self.last_valid_landing_height_distance_m = (
                self.landing_height_distance_m
            )
            self.last_valid_landing_height_stamp = float(now)
        return valid
    def _contact_evidence_height_m(self) -> float:
        """Return the relative-height contact-evidence threshold."""
        return self._float("touchdown_contact_evidence_clearance_m")
    def _target_source_age(
        self, estimate: TargetEstimate, source: str
    ) -> float | None:
        """Return one accepted sensor age in the estimator time domain."""
        stamp = self.target_estimator.statistics[
            source
        ].last_accepted_stamp_s
        if stamp is None:
            return None
        age = float(estimate.stamp_s) - float(stamp)
        return age if math.isfinite(age) and age >= 0.0 else None
    def _vision_motion_source_age(
        self, estimate: TargetEstimate
    ) -> float | None:
        """Age of the DOWN ArUco sample that actually drives velocity."""
        stamp = self.vision_motion_estimator.statistics[
            "vision_down"
        ].last_accepted_stamp_s
        if stamp is None:
            return None
        age = float(estimate.stamp_s) - float(stamp)
        return age if math.isfinite(age) and age >= 0.0 else None
    def _solver_cache_safe_for_descent(self, now: float) -> bool:
        """Require a fresh successful command from the current phase."""
        command = self.command_cache
        if (
            command is not None
            and command.controller_type == P_FEEDFORWARD_CONTROLLER_TYPE
            and command.primary_controller_type
            == P_FEEDFORWARD_CONTROLLER_TYPE
        ):
            return bool(
                self.consecutive_solver_failures
                < self._int("max_consecutive_solver_failures")
                and self._cached_p_command_valid(now)
            )
        if (
            self.consecutive_solver_failures
            >= self._int("max_consecutive_solver_failures")
            or not self._cached_solver_command_valid(now)
        ):
            return False
        assert command is not None
        if command.primary_controller_type == PRODUCTION_CONTROLLER_TYPE:
            return bool(command.mpc_success is True)
        return False
    def _descent_permission(
        self,
        now: float,
        estimate: TargetEstimate | None = None,
        *,
        require_solver_command: bool = True,
    ) -> DescentPermissionDecision:
        """Evaluate one coherent Stage-9 descent-safety snapshot."""
        target = (
            self._current_target_estimate()
            if estimate is None
            else estimate
        )
        control_target_velocity = self._control_target_velocity_enu(
            target, now
        )
        valid = bool(
            target is not None
            and target.valid
            and target.position_enu_m is not None
            and control_target_velocity is not None
            and self.vehicle_position_enu is not None
        )
        if valid and target is not None:
            target_position = np.asarray(target.position_enu_m, dtype=float)
            assert control_target_velocity is not None
            target_velocity = control_target_velocity
            relative = self.vehicle_position_enu - target_position
            horizontal_error = float(np.linalg.norm(relative[:2]))
            relative_speed = float(
                np.linalg.norm(
                    self.vehicle_velocity_enu[:2] - target_velocity[:2]
                )
            )
            relative_height = float(relative[2])
            funnel_radius = min(
                self._float("precision_lateral_gate_m"),
                max(
                    self._float(
                        "relative_landing_mpc_funnel_minimum_radius_m"
                    ),
                    self._float("relative_landing_mpc_funnel_slope")
                    * max(0.0, relative_height),
                ),
            )
            vision_motion = self.last_vision_motion_estimate
            covariance = (
                np.asarray(vision_motion.covariance, dtype=float)[:3, :3]
                if vision_motion is not None
                and vision_motion.valid
                and vision_motion.covariance is not None
                else np.full((3, 3), np.nan, dtype=float)
            )
            # Compensate only the intentional reorder delay, then combine it
            # with immediate callback freshness.  Rejected pose sequences
            # therefore cannot keep descent enabled just by arriving.
            callback_age = (
                None
                if self.down_marker_stamp <= 0.0
                else max(0.0, now - self.down_marker_stamp)
            )
            accepted_age = self._vision_motion_source_age(target)
            vision_age = (
                None
                if (
                    callback_age is None
                    or accepted_age is None
                    or not self._down_marker_detection_live(now)
                )
                else max(
                    callback_age,
                    max(
                        0.0,
                        accepted_age
                        - self._float("target_fusion_reorder_window_s"),
                    ),
                )
            )
            odometry_age = self._target_source_age(target, "odometry")
        else:
            horizontal_error = math.nan
            relative_speed = math.nan
            relative_height = None
            funnel_radius = math.nan
            covariance = np.full((3, 3), np.nan, dtype=float)
            vision_age = None
            odometry_age = None
        decision = evaluate_descent_permission(
            DescentPermissionInput(
                target_estimate_valid=valid,
                vision_age_s=vision_age,
                odometry_age_s=odometry_age,
                position_covariance_m2=covariance,
                horizontal_error_m=horizontal_error,
                funnel_radius_m=funnel_radius,
                relative_horizontal_speed_m_s=relative_speed,
                relative_height_m=relative_height,
                # While forming the next MPC request, physical descent gates
                # must not toggle merely because the preceding asynchronous
                # result has just expired.  The finished command is still
                # validated before it reaches the cache, and the heartbeat
                # cannot publish a not-yet-solved request.  ALIGN retains the
                # default requirement before entering a descent phase.
                solver_command_valid=(
                    self._solver_cache_safe_for_descent(now)
                    if require_solver_command
                    else True
                ),
            ),
            self.descent_permission_limits,
        )
        self.last_descent_permission = decision
        return decision
    def _terminal_occlusion_estimate_safe(
        self, estimate: TargetEstimate | None
    ) -> bool:
        """Validate the odometry-backed position used after optical loss."""
        if (
            estimate is None
            or not estimate.valid
            or not estimate.odometry_fresh
            or estimate.position_enu_m is None
            or estimate.covariance is None
        ):
            return False
        position = np.asarray(estimate.position_enu_m, dtype=float)
        covariance = np.asarray(estimate.covariance, dtype=float)
        if (
            position.shape != (3,)
            or not np.all(np.isfinite(position))
            or covariance.shape != (8, 8)
        ):
            return False
        try:
            position_covariance = validate_covariance(
                covariance[:3, :3], 3, "terminal target position"
            )
        except ValueError:
            return False
        return bool(
            float(np.max(np.linalg.eigvalsh(position_covariance)))
            <= self._float("descent_max_position_variance_m2")
        )
    def _px4_land_detector_signals(
        self, now: float
    ) -> tuple[bool, bool, bool, float, str]:
        """Adapt independent PX4 bits or PX4's composite landed result.

        MAVROS ``ON_GROUND`` is one already-dwelled PX4 land-detector verdict,
        not three statistically independent observations.  The repeated tuple
        only adapts that composite verdict to ``TouchdownGateInput``; its source
        label prevents the generic multi-sample logic from claiming otherwise.
        """
        timeout = self._float("land_detector_sample_timeout_s")
        extended_fresh = bool(
            self.extended_state is not None
            and 0.0 <= now - self.extended_state_stamp <= timeout
        )
        on_ground = bool(
            extended_fresh
            and self.extended_state is not None
            and self.extended_state.landed_state
            == ExtendedState.LANDED_STATE_ON_GROUND
        )

        def signal(
            topic_parameter: str,
            value: bool | None,
            stamp: float,
        ) -> tuple[bool, float, str]:
            topic = self._string(topic_parameter).strip()
            if topic:
                fresh = bool(
                    value is not None and 0.0 <= now - stamp <= timeout
                )
                return bool(value) if fresh else False, stamp, "bool_topic"
            return on_ground, self.extended_state_stamp, "extended_state"

        ground, ground_stamp, ground_source = signal(
            "px4_ground_contact_topic",
            self.px4_ground_contact_override,
            self.px4_ground_contact_override_stamp,
        )
        landed, landed_stamp, landed_source = signal(
            "px4_landed_topic",
            self.px4_landed_override,
            self.px4_landed_override_stamp,
        )
        at_rest, rest_stamp, rest_source = signal(
            "px4_at_rest_topic",
            self.px4_at_rest_override,
            self.px4_at_rest_override_stamp,
        )
        sample_stamp = min(ground_stamp, landed_stamp, rest_stamp)
        sources = {ground_source, landed_source, rest_source}
        source = (
            "bridged_px4_bool_topics"
            if "bool_topic" in sources
            else "px4_composite_on_ground"
        )
        return ground, landed, at_rest, sample_stamp, source
    def _critical_flight_failure(self, now: float) -> str | None:
        """Return one active-flight failure requiring fail-closed HOLD."""
        state = self.mavros_state
        if state is None or not state.connected:
            return "px4_connection_lost"
        if (
            self.vehicle_position_enu is None
            or not np.all(np.isfinite(self.vehicle_position_enu))
            or self.pose_stamp <= 0.0
            or now - self.pose_stamp > self._float("pose_timeout_s")
            or not np.all(np.isfinite(self.vehicle_velocity_enu))
            or self.velocity_stamp <= 0.0
            or now - self.velocity_stamp > self._float("velocity_timeout_s")
        ):
            return "px4_local_position_invalid"
        offboard_required = self.phase in {
            LandingPhase.TAKEOFF,
            LandingPhase.APPROACH,
            LandingPhase.MARKER_TRACK_DOWN,
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
            LandingPhase.ABORT_CLIMB,
        }
        if offboard_required and state.armed and state.mode != "OFFBOARD":
            return "offboard_lost"
        if (
            offboard_required
            and self.consecutive_solver_failures
            >= self._int("max_consecutive_solver_failures")
            and self.phase != LandingPhase.FINAL_APPROACH
        ):
            return "solver_failure_limit"
        # TAKEOFF is a vehicle-local maneuver. A transient trailer-odometry
        # gap must not disarm its setpoint stream one second after arming.
        # Approach/search use the latest measured trailer position directly;
        # the fused state becomes mandatory only for precision control.
        position_measurement_required = self.phase in {
            LandingPhase.APPROACH,
            LandingPhase.MARKER_TRACK_DOWN,
        }
        fused_target_required = self.phase in {
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
        }
        if position_measurement_required:
            position_source_live = self._trailer_fresh(now)
            if (
                not position_source_live
                and self.phase == LandingPhase.MARKER_TRACK_DOWN
            ):
                position_source_live = (
                    self._fresh_down_vision_pose(now) is not None
                )
            if not position_source_live:
                return "target_sensors_timeout"
        if fused_target_required:
            estimate = self._current_target_estimate()
            if self._target_sensors_loss_persisted(now, estimate):
                return "target_sensors_timeout"
        elif not fused_target_required:
            self._target_sensors_loss_started_monotonic_s = None
        return None
    def _enter_failsafe_hold(self, reason: str) -> None:
        """Latch one position and invalidate every descent-capable cache."""
        if self.phase != LandingPhase.FAILSAFE_HOLD:
            if (
                self.vehicle_position_enu is not None
                and np.all(np.isfinite(self.vehicle_position_enu))
            ):
                self.failsafe_hold_position_enu = (
                    self.vehicle_position_enu.copy()
                )
            self._transition(LandingPhase.FAILSAFE_HOLD, reason)
        self.last_safety_failure_reason = str(reason)
        self._invalidate_solver_context(clear_cached=True)
        self.touchdown_gate.reset(self.time_reset_count)
        if self.failsafe_hold_position_enu is None:
            self.command_cache = None
        else:
            self._cache_hold_command(self.failsafe_hold_position_enu)
    def _start_abort_climb(
        self,
        reason: str,
        estimate: TargetEstimate | None = None,
    ) -> None:
        """Latch a non-descending vertical escape target."""
        ground, landed, at_rest, _stamp, _source = (
            self._px4_land_detector_signals(time.monotonic())
        )
        if ground or landed or at_rest:
            self._enter_failsafe_hold(
                f"abort climb inhibited by contact evidence: {reason}"
            )
            return
        if (
            self.vehicle_position_enu is None
            or not np.all(np.isfinite(self.vehicle_position_enu))
        ):
            self._enter_failsafe_hold(f"abort unavailable: {reason}")
            return
        target = self.vehicle_position_enu.copy()
        deck_height: float | None = None
        if (
            estimate is not None
            and estimate.position_enu_m is not None
            and np.all(np.isfinite(estimate.position_enu_m))
        ):
            deck_height = float(estimate.position_enu_m[2])
        if deck_height is None:
            target[2] += self._float("abort_climb_height_m")
        else:
            target[2] = max(
                target[2],
                deck_height + self._float("abort_climb_height_m"),
            )
        self.abort_climb_target_enu = target
        self.last_safety_failure_reason = str(reason)
        self.touchdown_gate.reset(self.time_reset_count)
        self._transition(LandingPhase.ABORT_CLIMB, reason)
    def _hold_descent_reference(
        self,
        now: float,
        estimate: TargetEstimate | None,
    ) -> None:
        """Stop relative descent while retaining moving-deck XY tracking."""
        del estimate
        level_clearance_m = self.descent_clearance_m
        if self.phase in {
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
        }:
            level_clearance_m = self._float(
                "touchdown_contact_clearance_m"
            )
        if self.phase in {
            LandingPhase.MARKER_TRACK_DOWN,
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
        }:
            staged = self._stage_trailer_relative_command(
                now,
                level_clearance_m,
                "down",
                disable_descent=True,
                require_solver_success=False,
                relative_descent_speed_m_s=0.0,
            )
            if staged is not None:
                return
            # Preserve any still-bounded moving-pad command while the next
            # relative solve is staging. A world-frame zero-velocity hold on
            # a 3 m/s deck creates an immediate catch-up oscillation.
            if self.command_cache is not None:
                return
        if self.vehicle_position_enu is not None:
            self._cache_hold_command(self.vehicle_position_enu)
    def _target_sensors_loss_persisted(
        self,
        now: float,
        estimate: TargetEstimate | None,
    ) -> bool:
        """Require one continuous outage before declaring total target loss."""
        if estimate is not None and bool(getattr(estimate, "valid", False)):
            self._target_sensors_loss_started_monotonic_s = None
            return False
        started = self._target_sensors_loss_started_monotonic_s
        if started is None or now < started:
            self._target_sensors_loss_started_monotonic_s = now
            return False
        return bool(
            now - started
            >= self._float("target_sensors_loss_abort_dwell_s")
        )
    def _cache_target_sensors_loss_level_coast(self) -> None:
        """Stop descent while retaining the last bounded deck-following XY."""
        command = self.command_cache
        velocity: np.ndarray | None = None
        if command is not None and command.velocity_enabled:
            candidate = np.asarray(
                command.velocity_setpoint_enu_m_s, dtype=float
            )
            if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                velocity = candidate.copy()
        if velocity is None and self.vehicle_velocity_enu is not None:
            candidate = np.asarray(self.vehicle_velocity_enu, dtype=float)
            if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                velocity = candidate.copy()
        if velocity is None:
            if self.vehicle_position_enu is not None:
                self._cache_hold_command(self.vehicle_position_enu)
            return

        pad_vertical_velocity = 0.0
        snapshot = self._latest_solver_snapshot or self._cached_solver_snapshot
        if snapshot is not None:
            pad_velocity = getattr(
                snapshot.control_input,
                "landing_pad_velocity_enu_m_s",
                None,
            )
            if pad_velocity is not None:
                candidate = np.asarray(pad_velocity, dtype=float)
                if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                    pad_vertical_velocity = float(candidate[2])
        maximum_vertical_delta = (
            min(
                0.6,
                self._float(
                    "relative_landing_mpc_max_vertical_acceleration_m_s2"
                ),
            )
            / self._float("control_rate_hz")
        )
        velocity[2] += float(
            np.clip(
                pad_vertical_velocity - velocity[2],
                -maximum_vertical_delta,
                maximum_vertical_delta,
            )
        )
        self._invalidate_solver_context(clear_cached=True)
        disabled = np.full(3, np.nan, dtype=float)
        coast = LandingControlCommand(
            position_setpoint_enu_m=disabled,
            velocity_setpoint_enu_m_s=velocity,
            acceleration_setpoint_enu_m_s2=disabled,
            yaw_enu_rad=float(self.vehicle_yaw_enu_rad),
            position_enabled=False,
            velocity_enabled=True,
            acceleration_enabled=False,
            valid=False,
            degraded=True,
            status="target_sensors_transient_level_coast",
            solve_time_s=0.0,
            controller_type="hold",
            primary_controller_type=PRODUCTION_CONTROLLER_TYPE,
            mpc_attempted=False,
            mpc_success=None,
            fallback_used=HOLD_FALLBACK,
        )
        self._record_landing_command(coast)
        self._cache_command(coast)
    def _reset_terminal_contact_context(self) -> None:
        """Clear the one-shot terminal-contact latch on reset/phase change."""
        self._contact_settle_started_monotonic_s = None
        self._contact_settle_started_time_reset_count = None
        self._contact_settle_min_height_m = None
        self._contact_entry_latched = False
        self._contact_compression_started_monotonic_s = None
        self._contact_height_violation_started_monotonic_s = None
        self.terminal_contact_bridge_active = False
    def _contact_settle_elapsed_s(self, now: float) -> float | None:
        """Return total contact-latch age without refreshing it per command."""
        started = getattr(self, "_contact_settle_started_monotonic_s", None)
        epoch = getattr(
            self, "_contact_settle_started_time_reset_count", None
        )
        if started is None or epoch != self.time_reset_count:
            return None
        elapsed = float(now) - float(started)
        return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else None
    def _contact_height_within_latch(self, now: float) -> bool:
        """Keep contact tracking while relative height stays on the deck."""
        if self._fresh_landing_height(now):
            distance = self.landing_height_distance_m
        elif (
            self.last_valid_landing_height_distance_m is not None
            and 0.0 <= now - self.last_valid_landing_height_stamp
            <= self._float("landing_height_dropout_grace_s")
        ):
            distance = self.last_valid_landing_height_distance_m
        else:
            return False
        maximum = self._float(
            "touchdown_contact_latch_exit_distance_m"
        )
        if (
            distance is None
            or not math.isfinite(float(distance))
            or not math.isfinite(maximum)
            or maximum <= 0.0
        ):
            return False
        # The touchdown gate separately requires bounded vertical speed and
        # PX4 contact/landed/at-rest; this only bounds deck geometry.
        return bool(float(distance) <= maximum + 1.0e-6)
    def _terminal_contact_bridge_permission(
        self,
        now: float,
        estimate: TargetEstimate | None,
        decision: DescentPermissionDecision,
    ) -> tuple[bool, str]:
        """Allow only a short, already-latched optical-occlusion bridge."""
        if self.phase != LandingPhase.FINAL_APPROACH:
            return False, "phase_invalid"
        elapsed = self._contact_settle_elapsed_s(now)
        if elapsed is None:
            return False, "not_latched"
        if elapsed > self._float("touchdown_contact_settle_timeout_s"):
            return False, "mode_handoff_timeout"
        failures = set(decision.failed_checks)
        contact_transients = {
            "outside_landing_funnel",
            "relative_horizontal_speed_exceeded",
        }
        permitted_failures = {
            "vision_stale",
            "solver_command_invalid",
            "position_covariance_invalid",
            "position_covariance_exceeded",
        } | contact_transients
        if (
            not failures.intersection(
                {
                    "vision_stale",
                    "position_covariance_invalid",
                    "position_covariance_exceeded",
                }
            )
            or failures.difference(permitted_failures)
        ):
            return False, "non_vision_gate_failed"
        if failures.intersection(contact_transients) and not (
            self._terminal_horizontal_kinematics_safe(estimate)
        ):
            return False, "contact_horizontal_kinematics_invalid"
        if not self._terminal_occlusion_estimate_safe(estimate):
            return False, "odometry_position_confidence_invalid"
        floor = self._float("touchdown_contact_clearance_m")
        if not self._touchdown_contact_ready(now, estimate, floor):
            return False, "contact_conditions_invalid"
        if not self._contact_height_within_latch(now):
            return False, "deck_height_envelope_exceeded"
        return True, "vision_occluded_at_contact"
    def _terminal_horizontal_kinematics_safe(
        self,
        estimate: TargetEstimate | None,
    ) -> bool:
        """Recheck contact XY state from the newest vehicle/target samples."""
        target_velocity = self._control_target_velocity_enu(
            estimate, time.monotonic()
        )
        if (
            estimate is None
            or not bool(getattr(estimate, "valid", False))
            or estimate.position_enu_m is None
            or target_velocity is None
            or self.vehicle_position_enu is None
        ):
            return False
        vehicle_position = np.asarray(self.vehicle_position_enu, dtype=float)
        vehicle_velocity = np.asarray(self.vehicle_velocity_enu, dtype=float)
        target_position = np.asarray(estimate.position_enu_m, dtype=float)
        vectors = (
            vehicle_position,
            vehicle_velocity,
            target_position,
            target_velocity,
        )
        if any(
            vector.shape != (3,) or not np.all(np.isfinite(vector))
            for vector in vectors
        ):
            return False
        horizontal_error = float(
            np.linalg.norm(vehicle_position[:2] - target_position[:2])
        )
        relative_speed = float(
            np.linalg.norm(vehicle_velocity[:2] - target_velocity[:2])
        )
        return bool(
            horizontal_error <= self._float("precision_lateral_gate_m")
            and relative_speed <= self._float("relative_speed_gate_m_s")
        )
    def _touchdown_contact_ready(
        self,
        now: float,
        estimate: TargetEstimate | None,
        clearance_m: float,
    ) -> bool:
        """Require coherent low/slow geometry before contact settling."""
        target_velocity = self._control_target_velocity_enu(estimate, now)
        floor = self._float("touchdown_contact_clearance_m")
        maximum_deck_distance = (
            self._float("touchdown_max_deck_distance_m")
            + self._float("touchdown_contact_height_rebound_tolerance_m")
        )
        maximum_relative_speed = self._float(
            "touchdown_max_relative_vertical_speed_m_s"
        )
        clearance = float(clearance_m)
        measured_clearance = self._measured_target_clearance(
            estimate, clearance
        )
        if (
            self.phase != LandingPhase.FINAL_APPROACH
            or not math.isfinite(floor)
            or floor < 0.0
            or not math.isfinite(maximum_deck_distance)
            or maximum_deck_distance <= 0.0
            or not math.isfinite(maximum_relative_speed)
            or maximum_relative_speed <= 0.0
            or not math.isfinite(clearance)
            or not math.isfinite(measured_clearance)
            or measured_clearance > maximum_deck_distance + 1.0e-6
            or not self._fresh_landing_height(now)
            or self.landing_height_distance_m is None
            or self.landing_height_distance_m
            > maximum_deck_distance + 1.0e-6
            or estimate is None
            or not bool(getattr(estimate, "valid", False))
            or estimate.position_enu_m is None
            or target_velocity is None
            or estimate.yaw_enu_rad is None
            or self.vehicle_position_enu is None
        ):
            return False
        vehicle_position = np.asarray(self.vehicle_position_enu, dtype=float)
        vehicle_velocity = np.asarray(self.vehicle_velocity_enu, dtype=float)
        target_position = np.asarray(estimate.position_enu_m, dtype=float)
        target_yaw = float(estimate.yaw_enu_rad)
        if (
            vehicle_position.shape != (3,)
            or vehicle_velocity.shape != (3,)
            or target_position.shape != (3,)
            or target_velocity.shape != (3,)
            or not np.all(np.isfinite(vehicle_position))
            or not np.all(np.isfinite(vehicle_velocity))
            or not np.all(np.isfinite(target_position))
            or not np.all(np.isfinite(target_velocity))
            or not math.isfinite(target_yaw)
        ):
            return False
        relative_vertical_speed = abs(
            float(vehicle_velocity[2] - target_velocity[2])
        )
        return bool(
            relative_vertical_speed <= maximum_relative_speed
        )
    def _terminal_contact_entry_ready(
        self,
        now: float,
        estimate: TargetEstimate | None,
    ) -> bool:
        """Authorize the one-way transition into the contact controller."""
        if (
            self.phase != LandingPhase.FINAL_APPROACH
            or estimate is None
            or not bool(getattr(estimate, "valid", False))
            or not bool(getattr(estimate, "odometry_fresh", False))
            or estimate.position_enu_m is None
            or estimate.yaw_enu_rad is None
            or self.vehicle_position_enu is None
            or not self._fresh_landing_height(now)
            or self.landing_height_distance_m is None
            or not self._terminal_horizontal_kinematics_safe(estimate)
        ):
            return False
        target_velocity = self._control_target_velocity_enu(estimate, now)
        vehicle_velocity = np.asarray(self.vehicle_velocity_enu, dtype=float)
        if (
            target_velocity is None
            or target_velocity.shape != (3,)
            or vehicle_velocity.shape != (3,)
            or not np.all(np.isfinite(target_velocity))
            or not np.all(np.isfinite(vehicle_velocity))
        ):
            return False
        floor = self._float("touchdown_contact_clearance_m")
        maximum_deck_distance = (
            self._float("touchdown_max_deck_distance_m")
            + self._float("touchdown_contact_height_rebound_tolerance_m")
        )
        measured_clearance = self._measured_target_clearance(
            estimate, math.inf
        )
        if (
            not math.isfinite(measured_clearance)
            or measured_clearance > maximum_deck_distance + 1.0e-6
        ):
            return False

        # Prefer the normal low/slow gate.  At actual gear contact the EKF
        # vertical velocity can cross that gate for only one or two 50 Hz
        # ticks before the collision rebound.  The independent near-deck
        # clause catches that same physical entry while the aircraft is still
        # descending, but it only requests the proven 0.12 m/s settle command;
        # the stronger compression still requires the original low-speed
        # physical-contact evidence below.
        slow_contact_entry = self._touchdown_contact_ready(
            now, estimate, floor
        )
        relative_vertical_velocity = float(
            vehicle_velocity[2] - target_velocity[2]
        )
        near_deck_entry = bool(
            math.isfinite(float(self.landing_height_distance_m))
            and float(self.landing_height_distance_m)
            <= self._contact_evidence_height_m()
            and abs(relative_vertical_velocity)
            <= self._float(
                "touchdown_contact_entry_max_relative_vertical_speed_m_s"
            )
        )
        return bool(slow_contact_entry or near_deck_entry)
    def _terminal_contact_tracking_ready(
        self,
        now: float,
        estimate: TargetEstimate | None,
    ) -> bool:
        """Keep a latched contact solve inside the bounded deck envelope."""
        return bool(
            self.phase == LandingPhase.FINAL_APPROACH
            and self._contact_entry_latched
            and estimate is not None
            and bool(getattr(estimate, "valid", False))
            and bool(getattr(estimate, "odometry_fresh", False))
            and estimate.position_enu_m is not None
            and estimate.yaw_enu_rad is not None
            and self.vehicle_position_enu is not None
            and self._control_target_velocity_enu(estimate, now) is not None
            and self._fresh_landing_height(now)
            and self._contact_height_within_latch(now)
        )
    def _arm_terminal_contact_entry(self) -> None:
        """Latch contact and revoke every older ordinary FINAL solve."""
        if self._contact_entry_latched:
            return
        self._contact_entry_latched = True
        self.terminal_contact_bridge_active = True
        # Keep publishing the last bounded downward command while the first
        # contact QP is queued, but make every in-flight ordinary FINAL result
        # ineligible.  Otherwise a late pre-contact solve can overwrite the
        # settle request with the upward rebound seen in the failed ULog.
        self._invalidate_solver_context(clear_cached=True)
    def _stage_touchdown_contact_osqp(
        self,
        now: float,
        estimate: TargetEstimate | None,
        *,
        descent_allowed: bool,
        relative_descent_speed_m_s: float | None = None,
    ) -> bool:
        """Keep the Relative OSQP in control through contact and dwell."""
        floor = self._float("touchdown_contact_clearance_m")
        contact_entry_ready = self._terminal_contact_entry_ready(
            now, estimate
        )
        if contact_entry_ready:
            self._arm_terminal_contact_entry()
        final_contact_ready = bool(
            contact_entry_ready
            or self._terminal_contact_tracking_ready(now, estimate)
        )
        touchdown_dwell_tracking = bool(
            self.phase == LandingPhase.TOUCHDOWN_CONFIRM
            and estimate is not None
            and bool(getattr(estimate, "valid", False))
            and bool(getattr(estimate, "odometry_fresh", False))
            and estimate.position_enu_m is not None
            and estimate.yaw_enu_rad is not None
            and self.vehicle_position_enu is not None
            and self._fresh_landing_height(now)
        )
        if not descent_allowed or not (
            final_contact_ready or touchdown_dwell_tracking
        ):
            return False
        assert estimate is not None
        target_velocity = self._control_target_velocity_enu(estimate, now)
        if target_velocity is None:
            return False
        settle_speed = (
            self._float("touchdown_contact_settle_speed_m_s")
            if relative_descent_speed_m_s is None
            else float(relative_descent_speed_m_s)
        )
        if relative_descent_speed_m_s is None:
            vehicle_velocity = np.asarray(
                self.vehicle_velocity_enu, dtype=float
            )
            deck_height = self.landing_height_distance_m
            relative_vertical_speed = (
                math.inf
                if vehicle_velocity.shape != (3,)
                or not np.all(np.isfinite(vehicle_velocity))
                else abs(float(vehicle_velocity[2] - target_velocity[2]))
            )
            physical_contact_evidence = bool(
                self._fresh_landing_height(now)
                and deck_height is not None
                and math.isfinite(float(deck_height))
                and float(deck_height)
                <= self._contact_evidence_height_m()
                and relative_vertical_speed
                <= self._float(
                    "touchdown_contact_entry_max_relative_vertical_speed_m_s"
                )
            )
            if physical_contact_evidence:
                if self._contact_compression_started_monotonic_s is None:
                    self._contact_compression_started_monotonic_s = now
            compression_started = (
                self._contact_compression_started_monotonic_s
            )
            if compression_started is not None:
                # Physical support is latched before increasing the downward
                # demand.  Ramp from the low settle speed instead of stepping
                # directly from 0.12 to 0.50 m/s; the final value still gives
                # PX4's normal land detector enough authority to unload the
                # motors while the deck prevents further motion.
                compression_speed = self._float(
                    "touchdown_contact_compression_speed_m_s"
                )
                ramp_rate = self._float(
                    "touchdown_contact_compression_ramp_rate_m_s2"
                )
                ramp_elapsed = max(0.0, now - compression_started)
                settle_speed = min(
                    compression_speed,
                    settle_speed + ramp_rate * ramp_elapsed,
                )
        if (
            not math.isfinite(settle_speed)
            or settle_speed < 0.0
        ):
            return False

        # Lower only the optimizer's contact target.  OSQP retains horizontal
        # relative-position/velocity costs, jerk limits, the funnel/FOV/deck
        # constraints and the single async command-cache path.  The previous
        # direct P+feed-forward velocity command discarded every one of those
        # guarantees exactly when the marker became occluded.
        contact_target_clearance = self._descent_mpc_clearance_target(
            floor,
            0.0,
            settle_speed,
        )
        generation_before = self._solver_snapshot_generation
        staged = self._stage_trailer_relative_command(
            now,
            contact_target_clearance,
            "down",
            require_solver_success=False,
            relative_descent_speed_m_s=settle_speed,
        )
        if staged is None:
            return False
        snapshot_staged = bool(
            self._solver_snapshot_generation > generation_before
        )

        created = time.monotonic()
        solver_valid = self._solver_cache_safe_for_descent(created)
        transition_valid = self._cached_transition_command_valid(created)
        if solver_valid:
            if self._contact_settle_started_monotonic_s is None:
                self._contact_settle_started_monotonic_s = created
                self._contact_settle_started_time_reset_count = (
                    self.time_reset_count
                )
            deck_height = self.landing_height_distance_m
            if (
                deck_height is not None
                and math.isfinite(float(deck_height))
            ):
                if self._contact_settle_min_height_m is None:
                    self._contact_settle_min_height_m = float(
                        deck_height
                    )
                else:
                    self._contact_settle_min_height_m = min(
                        self._contact_settle_min_height_m,
                        float(deck_height),
                    )
        self.descent_clearance_m = floor
        # A fresh phase-transition coast or a newly queued immutable contact
        # snapshot is an acceptable short bridge while the first contact QP
        # runs.  Neither starts the contact timeout or claims a successful
        # solver result; the next tick still has to observe a validated cache.
        return bool(solver_valid or transition_valid or snapshot_staged)
    def _stage_level_recovery_reference(self, now: float) -> None:
        """Queue a level moving-pad solve without changing mission phase."""
        level_clearance_m = self.descent_clearance_m
        if self.phase in {
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
        }:
            level_clearance_m = self._float(
                "touchdown_contact_clearance_m"
            )
        self._stage_trailer_relative_command(
            now,
            level_clearance_m,
            "down",
            disable_descent=True,
            require_solver_success=False,
            relative_descent_speed_m_s=0.0,
        )
    def _handle_descent_denial(
        self,
        decision: DescentPermissionDecision,
        estimate: TargetEstimate | None,
    ) -> None:
        """Apply altitude-dependent marker and sensor-loss policy."""
        self.last_safety_failure_reason = decision.reason
        failures = set(decision.failed_checks)
        if failures == {"solver_command_invalid"} and (
            self.consecutive_solver_failures
            < self._int("max_consecutive_solver_failures")
        ):
            self._stage_level_recovery_reference(time.monotonic())
            return
        if "solver_command_invalid" in failures and (
            self.consecutive_solver_failures
            >= self._int("max_consecutive_solver_failures")
        ):
            self._enter_failsafe_hold("solver_failure_limit")
            return
        relative_height: float | None = None
        if (
            estimate is not None
            and estimate.position_enu_m is not None
            and self.vehicle_position_enu is not None
        ):
            height = float(
                self.vehicle_position_enu[2] - estimate.position_enu_m[2]
            )
            if math.isfinite(height):
                relative_height = height
        recoverable_tracking_failures = {
            "vision_stale",
            "position_covariance_invalid",
            "position_covariance_exceeded",
            "outside_landing_funnel",
            "relative_horizontal_speed_exceeded",
            "solver_command_invalid",
        }
        if (
            self.phase
            in {
                LandingPhase.PRECISION_DESCENT,
                LandingPhase.FINAL_APPROACH,
            }
            and failures
            and failures.issubset(recoverable_tracking_failures)
            and estimate is not None
            and bool(getattr(estimate, "valid", False))
            and bool(getattr(estimate, "odometry_fresh", False))
        ):
            # Do not change phase for a recoverable moving-target alignment
            # error. Keep the same relative OSQP and stop only vertical
            # motion; a phase change revokes the async command and creates a
            # 3 m/s world-frame brake/catch-up transient.
            loss_now = self._mission_time_s()
            vision_lost = "vision_stale" in failures
            if vision_lost:
                if self.marker_loss_started_mission_s is None:
                    self.marker_loss_started_mission_s = loss_now
            else:
                # Lateral/funnel and relative-speed misses are reasons to stop
                # vertical motion and recenter, not evidence that the optical
                # motion model has expired.  Starting the marker-loss timer for
                # those ordinary alignment errors revoked Relative-OSQP after
                # 1.5 s even while ArUco frames were live, then left a 5 m/s
                # deck faster than the position-only reacquisition controller.
                self.marker_loss_started_mission_s = None
            if self.phase == LandingPhase.FINAL_APPROACH:
                self.terminal_contact_bridge_active = True
            if vision_lost:
                assert self.marker_loss_started_mission_s is not None
                elapsed = loss_now - self.marker_loss_started_mission_s
                maximum_loss_s = (
                    self._vision_motion_continuation_limit_s()
                )
                if elapsed >= maximum_loss_s:
                    if self.phase == LandingPhase.FINAL_APPROACH:
                        self._start_abort_climb(
                            "terminal ArUco motion model expired", estimate
                        )
                    else:
                        self._transition(
                            LandingPhase.MARKER_TRACK_DOWN,
                            "ArUco motion model expired during level tracking",
                        )
                    return
            staged = self._stage_trailer_relative_command(
                time.monotonic(),
                (
                    self._float("touchdown_contact_clearance_m")
                    if self.phase == LandingPhase.FINAL_APPROACH
                    else self.descent_clearance_m
                ),
                "down",
                disable_descent=True,
                require_solver_success=False,
                relative_descent_speed_m_s=0.0,
            )
            self.last_marker_loss_policy = "level_track_recenter"
            if staged is not None:
                return
            return
        if "target_estimate_invalid" in failures:
            if self.phase != LandingPhase.FINAL_APPROACH:
                # Keep following the live trailer position and reacquire the
                # marker instead of freezing the last world-frame velocity.
                self._transition(
                    LandingPhase.MARKER_TRACK_DOWN,
                    "ArUco motion unavailable; resume marker acquisition",
                )
                return
            # The fused target can remain valid while the ArUco-derived
            # velocity used by control is unavailable.  A shared fused-sensor
            # timer is reset on every valid odometry update and therefore can
            # never bound this failure.  Track the control-motion outage on
            # the marker-loss clock instead.
            loss_now = self._mission_time_s()
            if self.marker_loss_started_mission_s is None:
                self.marker_loss_started_mission_s = loss_now
            elapsed = loss_now - self.marker_loss_started_mission_s
            if elapsed >= self._float("vision_velocity_terminal_hold_s"):
                self._start_abort_climb(
                    "terminal ArUco motion unavailable", estimate
                )
            else:
                self._cache_target_sensors_loss_level_coast()
                self.last_safety_failure_reason = (
                    "target_sensors_transient_level_coast"
                )
            return
        high = self._float("marker_loss_high_altitude_m")
        low = self._float("marker_loss_low_altitude_m")
        if "vision_stale" in failures:
            loss_now = self._mission_time_s()
            if self.marker_loss_started_mission_s is None:
                self.marker_loss_started_mission_s = loss_now
        recoverable_high_altitude_failures = {
            "vision_stale",
            "odometry_stale",
            "position_covariance_invalid",
            "position_covariance_exceeded",
            "outside_landing_funnel",
            "relative_horizontal_speed_exceeded",
            "height_source_invalid",
            "solver_command_invalid",
        }
        if (
            relative_height is not None
            and relative_height > high
            and failures.intersection(recoverable_high_altitude_failures)
        ):
            self.last_marker_loss_policy = "high_altitude_hold_reacquire"
            self._transition(
                LandingPhase.PRECISION_ALIGN,
                f"high-altitude descent hold: {decision.reason}",
            )
            return
        if (
            relative_height is not None
            and relative_height > low
            and failures.intersection(
                {
                    "outside_landing_funnel",
                    "relative_horizontal_speed_exceeded",
                }
            )
        ):
            # A rounded waypoint can briefly increase relative speed. Stop
            # descent and recenter; abort remains reserved for genuinely
            # low-altitude or persistent sensor failures.
            self.last_marker_loss_policy = "mid_altitude_recenter"
            self._transition(
                LandingPhase.PRECISION_ALIGN,
                f"descent hold/recenter: {decision.reason}",
            )
            return
        if "vision_stale" in failures:
            if relative_height is not None and relative_height > high:
                self.last_marker_loss_policy = "high_altitude_hold_reacquire"
                self._transition(
                    LandingPhase.PRECISION_ALIGN,
                    "high-altitude marker loss: hold/reacquire",
                )
            elif relative_height is not None and relative_height > low:
                self.last_marker_loss_policy = "mid_altitude_hold_reacquire"
                self._transition(
                    LandingPhase.PRECISION_ALIGN,
                    "mid-altitude marker loss: hold/reacquire",
                )
            else:
                loss_now = self._mission_time_s()
                elapsed = (
                    0.0
                    if self.marker_loss_started_mission_s is None
                    else loss_now - self.marker_loss_started_mission_s
                )
                if elapsed >= self._float(
                    "low_altitude_marker_reacquire_timeout_s"
                ):
                    self.last_marker_loss_policy = (
                        "low_altitude_reacquire_timeout_abort"
                    )
                    self._start_abort_climb(
                        "low-altitude marker reacquisition timeout",
                        estimate,
                    )
                else:
                    self.last_marker_loss_policy = (
                        "low_altitude_hold_reacquire"
                    )
                    self._transition(
                        LandingPhase.PRECISION_ALIGN,
                        "low-altitude marker loss: hold/reacquire",
                    )
            return
        if "height_source_invalid" in failures:
            if (
                relative_height is not None
                and relative_height > self._float("marker_loss_low_altitude_m")
            ):
                self._start_abort_climb(
                    "landing height source invalid", estimate
                )
            else:
                self._enter_failsafe_hold(
                    "low-altitude landing height source invalid"
                )
            return
        if failures.intersection(
            {
                "odometry_stale",
                "position_covariance_invalid",
                "position_covariance_exceeded",
                "outside_landing_funnel",
                "relative_horizontal_speed_exceeded",
            }
        ):
            self._start_abort_climb(
                f"descent gate blocked: {decision.reason}", estimate
            )
            return
        self._enter_failsafe_hold(
            f"unclassified descent gate failure: {decision.reason}"
        )
    def _update_touchdown_gate(
        self,
        now: float,
        estimate: TargetEstimate | None,
    ) -> TouchdownGateDecision:
        """Feed one coherent low-altitude sample into the pure gate."""
        target_velocity = self._control_target_velocity_enu(estimate, now)
        if target_velocity is not None:
            relative_vertical_speed = float(
                self.vehicle_velocity_enu[2]
                - target_velocity[2]
            )
        else:
            relative_vertical_speed = math.nan
        deck_distance = (
            float(self.landing_height_distance_m)
            if self._fresh_landing_height(now)
            and self.landing_height_distance_m is not None
            else math.nan
        )
        ground, landed, at_rest, sample_stamp, _source = (
            self._px4_land_detector_signals(now)
        )
        decision = self.touchdown_gate.update(
            TouchdownGateInput(
                now_s=now,
                sample_stamp_s=sample_stamp,
                deck_distance_m=deck_distance,
                relative_vertical_speed_m_s=relative_vertical_speed,
                ground_contact=ground,
                landed=landed,
                at_rest=at_rest,
                reset_epoch=self.time_reset_count,
            )
        )
        self.last_touchdown_decision = decision
        return decision
    def _control_tick(self) -> None:
        """Run one atomic mission decision and expose one solver snapshot.

        Several safety branches may ask for a level or descending solve while
        one state-machine decision is being formed.  Keep those requests
        private to this callback and publish only the final request after the
        decision returns.  The solver timer therefore never observes an
        intermediate level request followed by a descent request from the
        same 20 ms control tick.
        """
        with self._solver_context_lock:
            self._control_tick_snapshot_candidate = None
            self._control_tick_snapshot_staging = True
            try:
                self._control_tick_impl()
            except BaseException:
                self._control_tick_snapshot_candidate = None
                raise
            finally:
                self._control_tick_snapshot_staging = False
            snapshot = self._control_tick_snapshot_candidate
            self._control_tick_snapshot_candidate = None
            if snapshot is not None and (
                snapshot.phase == self.phase.value
                and snapshot.time_reset_count == self.time_reset_count
            ):
                self._pending_solver_snapshot = snapshot
                self._latest_solver_snapshot = snapshot

    def _control_tick_impl(self) -> None:
        now = time.monotonic()
        mission_now = self._mission_time_s()
        state = self.mavros_state

        if self._control_tick_blocked(now, state):
            return

        # Preserve the intentional same-tick fall-through through the startup
        # phases.  In particular, an accepted READY request immediately stages
        # PRESTREAM, and an already-armed ARMING phase immediately stages the
        # first TAKEOFF command from the same state snapshot.
        if self.phase == LandingPhase.WAITING:
            self._tick_waiting()
        if self.phase == LandingPhase.READY:
            self._tick_ready()
            if self.phase == LandingPhase.READY:
                return
        if self.phase == LandingPhase.PRESTREAM:
            self._tick_prestream(mission_now)
            return
        if self.phase == LandingPhase.ARMING:
            self._tick_arming(now, state)
            if self.phase == LandingPhase.ARMING:
                return

        if self.phase == LandingPhase.TAKEOFF:
            self._tick_takeoff(mission_now)
        elif self.phase == LandingPhase.ABORT_CLIMB:
            self._tick_abort_climb(mission_now)
        elif self.phase == LandingPhase.APPROACH:
            self._tick_approach(now)
        elif self.phase == LandingPhase.MARKER_TRACK_DOWN:
            self._tick_marker_track_down(now, mission_now)
        elif self.phase == LandingPhase.PRECISION_ALIGN:
            self._tick_precision_align(now, mission_now)
        elif self.phase == LandingPhase.PRECISION_DESCENT:
            self._tick_precision_descent(now, mission_now)
        elif self.phase == LandingPhase.FINAL_APPROACH:
            self._tick_final_approach(now, mission_now)
        elif self.phase == LandingPhase.TOUCHDOWN_CONFIRM:
            self._tick_touchdown_confirm(now, state)

    def _control_tick_blocked(self, now: float, state) -> bool:
        if self.phase == LandingPhase.LANDED:
            return True
        if self.phase == LandingPhase.FAILSAFE_HOLD:
            if self.failsafe_hold_position_enu is not None:
                self._cache_hold_command(self.failsafe_hold_position_enu)
            return True

        if self.phase == LandingPhase.TOUCHDOWN_CONFIRM and state is not None:
            if not state.armed:
                if self.touchdown_disarm_requested:
                    self.command_cache = None
                    self._invalidate_solver_context(clear_cached=True)
                    self.start_requested = False
                    self._transition(
                        LandingPhase.LANDED,
                        "normal disarm confirmed after touchdown dwell",
                    )
                else:
                    self._enter_failsafe_hold(
                        "unexpected disarm before touchdown confirmation"
                    )
                return True

        if self.phase in {LandingPhase.WAITING, LandingPhase.READY}:
            if not self._ready(now):
                self.phase = LandingPhase.WAITING
                return True
        elif self.phase in {LandingPhase.PRESTREAM, LandingPhase.ARMING}:
            if not self._ready(now):
                self._enter_failsafe_hold("preflight state became invalid")
                return True
        else:
            failure = self._critical_flight_failure(now)
            if failure is not None:
                self._enter_failsafe_hold(failure)
                return True

        unexpected_disarm_phases = {
            LandingPhase.TAKEOFF,
            LandingPhase.APPROACH,
            LandingPhase.MARKER_TRACK_DOWN,
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            LandingPhase.ABORT_CLIMB,
        }
        if (
            self.phase in unexpected_disarm_phases
            and state is not None
            and not state.armed
        ):
            self._enter_failsafe_hold("unexpected in-air disarm observed")
            return True

        return False

    def _tick_waiting(self) -> None:
        self._transition(
            LandingPhase.READY, "PX4 and experiment sensors ready"
        )

    def _tick_ready(self) -> None:
        if not self.start_requested:
            return
        self.takeoff_origin = self.vehicle_position_enu.copy()
        self._transition(LandingPhase.PRESTREAM, "start request accepted")

    def _tick_prestream(self, mission_now: float) -> None:
        self._cache_hold_command(self.vehicle_position_enu)
        if mission_now - self.phase_started >= self._float(
            "offboard_prestream_s"
        ):
            self._transition(LandingPhase.ARMING, "setpoint stream warm")

    def _tick_arming(self, now: float, state) -> None:
        self._cache_hold_command(self.vehicle_position_enu)
        if state is None:
            return
        if state.mode != "OFFBOARD":
            self._request_mode("OFFBOARD", now)
            return
        if not state.armed:
            self._request_arm(True, now)
            return
        self._transition(LandingPhase.TAKEOFF, "armed in OFFBOARD")

    def _tick_takeoff(self, mission_now: float) -> None:
        if self.phase == LandingPhase.TAKEOFF:
            if self.takeoff_origin is None:
                self.takeoff_origin = self.vehicle_position_enu.copy()
            target = self.takeoff_origin.copy()
            takeoff_height = self._float("takeoff_height_m")
            takeoff_elapsed_s = max(0.0, mission_now - self.phase_started)
            takeoff_reference_rate_m_s = 1.5 * self._float(
                "landing_p_vertical_velocity_limit_m_s"
            )
            commanded_height = min(
                takeoff_height,
                takeoff_reference_rate_m_s * takeoff_elapsed_s,
            )
            target[2] = self.takeoff_origin[2] + commanded_height
            self._stage_control_command(
                target, np.zeros(3), self.vehicle_yaw_enu_rad
            )
            takeoff_settled = bool(
                commanded_height >= takeoff_height - 1.0e-6
                and abs(
                    self.vehicle_position_enu[2]
                    - (self.takeoff_origin[2] + takeoff_height)
                )
                <= self._float("takeoff_tolerance_m")
                and abs(float(self.vehicle_velocity_enu[2])) <= 0.10
            )
            if self._gate_dwell(
                takeoff_settled,
                mission_now,
                max(0.20, self._float("precision_alignment_dwell_s")),
            ):
                self._transition(
                    LandingPhase.APPROACH,
                    "takeoff height and vertical speed settled",
                )
            return

    def _tick_abort_climb(self, mission_now: float) -> None:
        if self.phase == LandingPhase.ABORT_CLIMB:
            target = self.abort_climb_target_enu
            if target is None:
                self._enter_failsafe_hold("abort climb target unavailable")
                return
            success = self._stage_control_command(
                target, np.zeros(3), self.vehicle_yaw_enu_rad
            )
            if (
                success
                and self.vehicle_position_enu[2]
                >= target[2] - self._float("abort_climb_tolerance_m")
            ):
                self._enter_failsafe_hold("abort climb completed")
            elif mission_now - self.phase_started > self._float(
                "abort_climb_timeout_s"
            ):
                self._enter_failsafe_hold("abort climb timeout")
            return

    def _tick_approach(self, now: float) -> None:
        if self.phase == LandingPhase.APPROACH:
            down_marker_visible = bool(
                self._down_marker_world_enu(now) is not None
            )
            result = self._stage_position_only_acquisition_command(
                now,
                self._float("marker_search_height_m"),
                use_down_marker=down_marker_visible,
            )
            marker_search_height = self._float("marker_search_height_m")
            measured_clearance = self._measured_target_clearance(
                self._current_target_estimate(),
                0.0,
            )
            search_height_ready = bool(
                math.isfinite(measured_clearance)
                and measured_clearance
                >= marker_search_height
                - self._float("takeoff_tolerance_m")
            )
            if (
                result is not None
                and search_height_ready
                and (
                    down_marker_visible
                    or result[0]
                    <= self._float("approach_capture_radius_m")
                )
            ):
                # APPROACH is odometry-only transit to the trailer.  Enter
                # the explicit search phase as soon as that transit is
                # complete; requiring vision here made the operator-visible
                # MARKER_SEARCH state begin only after the marker was already
                # acquired.  MARKER_TRACK_DOWN keeps level odometry tracking
                # while searching and gates TRACK_AND_LAND on the detector's
                # debounced pose plus the alignment dwell below.
                self.descent_clearance_m = marker_search_height
                self._transition(
                    LandingPhase.MARKER_TRACK_DOWN,
                    "trailer approach complete; begin down-marker search",
                )
            return

    def _tick_marker_track_down(
        self, now: float, mission_now: float
    ) -> None:
        if self.phase == LandingPhase.MARKER_TRACK_DOWN:
            estimate = self._current_target_estimate()
            # Keep the latched search clearance fixed while one continuous
            # acquisition controller matches marker position, speed, and yaw.
            tracking_clearance = self.descent_clearance_m
            marker_pose = self._down_marker_world_enu(now)
            marker_velocity = self._control_target_velocity_enu(
                estimate, now
            )
            if not self._trailer_fresh(now):
                self._transition(
                    LandingPhase.APPROACH,
                    "down marker and trailer odometry stale",
                )
                return

            acquisition_result = (
                self._stage_position_only_acquisition_command(
                    now,
                    tracking_clearance,
                    use_down_marker=marker_pose is not None,
                    target_velocity_enu_m_s=marker_velocity,
                )
            )
            acquisition_sensor_ready = bool(
                marker_velocity is not None
                # Do not hand control to Relative MPC from a short-window
                # Kalman derivative.  The same robust ArUco position model
                # must first classify stationary/straight/turning motion and
                # provide a coherent course, turn rate, and acceleration.
                # A raw marker pose is intentionally not required on every
                # state-machine tick: the 9 m/s circular case can lose one
                # frame behind the landing gear while its qualified optical
                # motion cache is still inside its bounded dropout window. Requiring
                # that frame here prevented Relative OSQP from removing the
                # remaining lateral lag.  This authorizes level ALIGN only;
                # the existing live-marker and descent gates still fail
                # closed before any vertical descent is allowed.
                and self.vision_position_model_qualified
                and self._vision_position_covariance_safe()
                and self.control_target_velocity_source
                in {"aruco_capture_kalman", "aruco_capture_hold"}
                and self._vehicle_acceleration_ready_for_relative_mpc()
            )
            acquisition_ready = bool(
                acquisition_sensor_ready
                and acquisition_result is not None
                and self.last_landing_relative_horizontal_speed_m_s
                is not None
                and acquisition_result[0]
                <= self._float("marker_track_entry_lateral_gate_m")
                and self.last_landing_relative_horizontal_speed_m_s
                <= self._float(
                    "marker_track_entry_relative_speed_m_s"
                )
                and acquisition_result[1]
                <= self._float("marker_track_entry_yaw_error_rad")
            )
            if not acquisition_ready:
                self.marker_tracking_acquisition_since_s = None
            else:
                if self.marker_tracking_acquisition_since_s is None:
                    self.marker_tracking_acquisition_since_s = mission_now
                if (
                    mission_now - self.marker_tracking_acquisition_since_s
                    >= self._float("precision_alignment_dwell_s")
                ):
                    # The robust motion fit proves observability, while this
                    # existing one-second dwell proves that the acquisition
                    # controller has actually matched position, speed and yaw.
                    # A single good sample is not a safe 9 m/s MPC handoff.
                    self.descent_clearance_m = tracking_clearance
                    self._transition(
                        LandingPhase.PRECISION_ALIGN,
                        "qualified ArUco motion and acquisition dwell ready",
                    )
                    return
            self._gate_dwell(
                False,
                mission_now,
                self._float("precision_alignment_dwell_s"),
            )
            return

    def _tick_precision_align(
        self, now: float, mission_now: float
    ) -> None:
        if self.phase == LandingPhase.PRECISION_ALIGN:
            if (
                self.terminal_recovery_started_mission_s is not None
                and mission_now
                - self.terminal_recovery_started_mission_s
                >= self._float("final_approach_timeout_s")
            ):
                # Terminal recovery has one bounded state lifetime.  Do not
                # let intermittent marker frames or deck-height noise restart
                # or bypass that timeout.
                self._start_abort_climb(
                    "terminal optical recovery timeout",
                    self._current_target_estimate(),
                )
                return
            if self._down_marker_world_enu(now) is None:
                estimate = self._current_target_estimate()
                measured_clearance = self._measured_target_clearance(
                    estimate,
                    self.descent_clearance_m,
                )
                if measured_clearance > (
                    self._float("terminal_occlusion_max_clearance_m")
                ):
                    # A single rejected camera frame must stop descent but
                    # must not restart the whole acquisition sequence. Coast
                    # horizontally for at most one bounded optical
                    # reacquisition window using only the bounded ArUco
                    # velocity cache plus fresh trailer position.  No trailer
                    # twist is introduced.  A longer loss returns to SEARCH.
                    if self.marker_loss_started_mission_s is None:
                        self.marker_loss_started_mission_s = mission_now
                    cached_velocity = self._control_target_velocity_enu(
                        estimate, now
                    )
                    if (
                        estimate is None
                        or not estimate.odometry_fresh
                        or cached_velocity is None
                    ):
                        self._transition(
                            LandingPhase.MARKER_TRACK_DOWN,
                            "optical velocity cache unavailable",
                        )
                        return
                    self._stage_trailer_relative_command(
                        now,
                        self.descent_clearance_m,
                        None,
                        require_solver_success=False,
                    )
                    self.last_marker_loss_policy = (
                        "alignment_short_loss_level_track"
                    )
                    self._gate_dwell(
                        False,
                        mission_now,
                        self._float("precision_alignment_dwell_s"),
                    )
                    if (
                        mission_now - self.marker_loss_started_mission_s
                        >= self._vision_motion_continuation_limit_s()
                    ):
                        self._transition(
                            LandingPhase.MARKER_TRACK_DOWN,
                            "qualified optical motion cache expired",
                        )
                    return
                # A stale camera frame must stop vertical descent, but it must
                # not turn a moving-deck landing into an absolute world-frame
                # hold.  Continue feeding the newest odometry-backed target
                # prediction to the relative MPC at the latched alignment
                # clearance.
                # PRECISION_ALIGN makes ``descent_allowed`` false in
                # ``_relative_landing_context``, so this preserves horizontal
                # position/velocity matching without authorizing descent.
                tracking_requested = False
                tracking_result = None
                if estimate is not None and estimate.odometry_fresh:
                    tracking_clearance = self.descent_clearance_m
                    # The solver is asynchronous: a false return can simply
                    # mean that the just-staged snapshot has not completed yet.
                    # Do not invalidate that snapshot with an immediate HOLD.
                    generation_before = self._solver_snapshot_generation
                    tracking_result = self._stage_trailer_relative_command(
                        now,
                        tracking_clearance,
                        None,
                    )
                    tracking_requested = bool(
                        tracking_result is not None
                        or self._solver_snapshot_generation
                        > generation_before
                    )
                if not tracking_requested:
                    self._hold_descent_reference(now, estimate)
                self.last_marker_loss_policy = (
                    "alignment_odometry_track_reacquire"
                    if tracking_requested
                    else "alignment_hold_reacquire"
                )
                if self.marker_loss_started_mission_s is None:
                    self.marker_loss_started_mission_s = mission_now
                relative_height = None
                estimate_position = (
                    getattr(estimate, "position_enu_m", None)
                    if estimate is not None
                    else None
                )
                if (
                    estimate_position is not None
                    and self.vehicle_position_enu is not None
                ):
                    relative_height = float(
                        self.vehicle_position_enu[2]
                        - estimate_position[2]
                    )
                occlusion_height = self._float(
                    "terminal_occlusion_max_clearance_m"
                )
                occlusion_height_limit = occlusion_height
                terminal_recovery_envelope = bool(
                    relative_height is not None
                    and math.isfinite(relative_height)
                    and self.descent_clearance_m <= occlusion_height
                    and relative_height <= occlusion_height
                    and estimate is not None
                    and estimate.odometry_fresh
                    and self._fresh_landing_height(now)
                    and self.landing_height_distance_m is not None
                    and math.isfinite(
                        float(self.landing_height_distance_m)
                    )
                    and float(self.landing_height_distance_m)
                    <= occlusion_height_limit
                )
                if terminal_recovery_envelope:
                    if self.terminal_recovery_started_mission_s is None:
                        self.terminal_recovery_started_mission_s = mission_now
                    # One large marker necessarily leaves the finite camera
                    # FOV near contact. FINAL_APPROACH already contains the
                    # strict geometry + covariance + odometry optical bridge,
                    # but only re-enter it after unconstrained ALIGN has
                    # recovered any corner-induced position/speed error.
                    horizontal_error = math.inf
                    relative_speed = math.inf
                    recovery_target_velocity = (
                        self._control_target_velocity_enu(estimate, now)
                    )
                    if (
                        estimate_position is not None
                        and recovery_target_velocity is not None
                        and self.vehicle_position_enu is not None
                    ):
                        horizontal_error = float(
                            np.linalg.norm(
                                self.vehicle_position_enu[:2]
                                - np.asarray(estimate_position, dtype=float)[:2]
                            )
                        )
                        relative_speed = float(
                            np.linalg.norm(
                                self.vehicle_velocity_enu[:2]
                                - recovery_target_velocity[:2]
                            )
                        )
                    # Do not reactivate the low-altitude hard constraints
                    # from one instantaneous sample.  The previous code
                    # could switch ALIGN -> FINAL only 20 ms after entering
                    # ALIGN, before its asynchronous solve had completed;
                    # that revoked the new snapshot and repeatedly injected
                    # invalid commands.  Require a validated ALIGN command,
                    # tighter terminal geometry, and a continuous dwell.
                    terminal_reentry_position_gate_m = self._float(
                        "precision_lateral_gate_m"
                    )
                    terminal_reentry_speed_gate_m_s = min(
                        0.20,
                        self._float("relative_speed_gate_m_s"),
                    )
                    terminal_reentry_ready = bool(
                        (
                            tracking_result is not None
                            or self._solver_cache_safe_for_descent(
                                time.monotonic()
                            )
                        )
                        and math.isfinite(horizontal_error)
                        and math.isfinite(relative_speed)
                        and horizontal_error
                        <= terminal_reentry_position_gate_m
                        and relative_speed
                        <= terminal_reentry_speed_gate_m_s
                    )
                    if self._gate_dwell(
                        terminal_reentry_ready,
                        mission_now,
                        self._float("precision_alignment_dwell_s"),
                    ):
                        self._transition(
                            LandingPhase.FINAL_APPROACH,
                            "terminal optical recovery aligned; enter guarded bridge",
                        )
                        return
                    loss_elapsed = (
                        0.0
                        if self.terminal_recovery_started_mission_s is None
                        else mission_now
                        - self.terminal_recovery_started_mission_s
                    )
                    if loss_elapsed >= self._float(
                        "final_approach_timeout_s"
                    ):
                        self._start_abort_climb(
                            "terminal optical recovery timeout",
                            estimate,
                        )
                    return
                self._gate_dwell(
                    False,
                    mission_now,
                    self._float("precision_alignment_dwell_s"),
                )
                if (
                    relative_height is not None
                    and math.isfinite(relative_height)
                    and relative_height
                    <= self._float("marker_loss_low_altitude_m")
                    and mission_now - self.marker_loss_started_mission_s
                    >= self._float(
                        "low_altitude_marker_reacquire_timeout_s"
                    )
                ):
                    self.last_marker_loss_policy = (
                        "low_altitude_reacquire_timeout_abort"
                    )
                    self._start_abort_climb(
                        "low-altitude marker reacquisition timeout",
                        estimate,
                    )
                return
            self.marker_loss_started_mission_s = None
            estimate = self._current_target_estimate()
            # Keep the clearance captured on entry to alignment. Replacing it
            # with the measured clearance every tick ratcheted small upward
            # tracking excursions into a steadily rising altitude reference.
            alignment_clearance = self.descent_clearance_m
            result = self._stage_trailer_relative_command(
                now, alignment_clearance, "down"
            )
            # Synchronous P control above timestamps its cache after the
            # tick-start ``now``.  A fresh monotonic sample prevents a valid
            # same-tick command from appearing to have a negative age.
            decision = self._descent_permission(time.monotonic(), estimate)
            within = bool(
                result is not None
                and decision.allowed
                and self._vision_position_covariance_safe()
                and result[2]
                <= self._float("marker_track_entry_yaw_error_rad")
            )
            if not within:
                failures = set(decision.failed_checks)
                failures.discard("solver_command_invalid")
                # Keep the staged, level Relative-MPC recentering command for
                # recoverable lateral/speed misses. Holding it here caused the
                # vehicle to oscillate around the deck without ever aligning.
                recoverable = {
                    "outside_landing_funnel",
                    "relative_horizontal_speed_exceeded",
                    # Freeze descent through a short camera gap while the
                    # fresh trailer odometry keeps horizontal tracking alive.
                    "vision_stale",
                    # These gates block vertical motion. They must not replace
                    # the already staged level moving-pad MPC command with a
                    # zero-velocity world-frame hold.
                    "position_covariance_invalid",
                    "position_covariance_exceeded",
                    "height_source_invalid",
                }
                if failures and not failures.issubset(recoverable):
                    self._hold_descent_reference(now, estimate)
            if self._gate_dwell(
                within,
                mission_now,
                self._float("precision_descent_gate_dwell_s"),
            ):
                if self._final_approach_height_reached(now, estimate):
                    # At this height the single large marker is close to the
                    # landing-gear/FOV occlusion boundary. Enter the guarded
                    # odometry bridge directly instead of spending an
                    # extra tick in PRECISION_DESCENT and losing the marker
                    # before FINAL_APPROACH owns the expected optical gap.
                    measured_clearance = self._measured_target_clearance(
                        estimate, self.descent_clearance_m
                    )
                    if math.isfinite(measured_clearance):
                        self.descent_clearance_m = max(
                            self._float("touchdown_contact_clearance_m"),
                            min(self.descent_clearance_m, measured_clearance),
                        )
                    self._transition(
                        LandingPhase.FINAL_APPROACH,
                        "terminal height aligned; enter guarded bridge",
                    )
                else:
                    self._transition(
                        LandingPhase.PRECISION_DESCENT,
                        "all descent-safety gates held",
                    )
            return

    def _tick_precision_descent(
        self, now: float, mission_now: float
    ) -> None:
        if self.phase == LandingPhase.PRECISION_DESCENT:
            estimate = self._current_target_estimate()
            # Decide level versus descent once from the physical sensor and
            # geometry gates.  Solver-cache readiness controls command
            # publication, not the constraint mode of the solve that is being
            # requested; coupling the two created a level/descent alternation
            # whenever an otherwise valid asynchronous command expired.
            decision = self._descent_permission(
                now,
                estimate,
                require_solver_command=False,
            )
            if not decision.allowed:
                failures = set(decision.failed_checks)
                if (
                    "vision_stale" in failures
                    and failures.issubset(
                        {"vision_stale", "solver_command_invalid"}
                    )
                    and estimate is not None
                    and estimate.odometry_fresh
                ):
                    # Freeze the current clearance but keep this phase for a
                    # bounded optical gap.  Repeated DESCENT <-> ALIGN phase
                    # changes invalidated the 50 Hz async cache and produced
                    # the observed climb/descent jerk.  The level solve uses
                    # fresh trailer position plus only the bounded ArUco
                    # velocity cache; no trailer twist is exposed.
                    measured_clearance = self._measured_target_clearance(
                        estimate,
                        self.descent_clearance_m,
                    )
                    if math.isfinite(measured_clearance):
                        self.descent_clearance_m = max(
                            self._float("touchdown_contact_clearance_m"),
                            min(
                                self.descent_clearance_m,
                                measured_clearance,
                            ),
                        )
                    occlusion_clearance = self._float(
                        "terminal_occlusion_max_clearance_m"
                    )
                    terminal_gap_entry = bool(
                        self.vision_position_model_qualified
                        and math.isfinite(measured_clearance)
                        and measured_clearance
                        <= occlusion_clearance + 1.0e-6
                        and self.descent_clearance_m
                        <= occlusion_clearance + 1.0e-6
                        and self._fresh_landing_height(now)
                        and self.landing_height_distance_m is not None
                        and math.isfinite(
                            float(self.landing_height_distance_m)
                        )
                        and float(self.landing_height_distance_m)
                        <= occlusion_clearance + 1.0e-6
                        and self._terminal_occlusion_estimate_safe(estimate)
                        and self._control_target_velocity_enu(estimate, now)
                        is not None
                    )
                    if terminal_gap_entry:
                        # The final-phase boundary and the close-range optical
                        # bridge must meet exactly. Otherwise a marker hidden
                        # by the landing gear in the last few centimetres of
                        # PRECISION_DESCENT can belong to neither controller.
                        self._transition(
                            LandingPhase.FINAL_APPROACH,
                            "terminal height reached during optical gap",
                        )
                        return
                    if self.marker_loss_started_mission_s is None:
                        self.marker_loss_started_mission_s = mission_now
                    self._stage_trailer_relative_command(
                        now,
                        self.descent_clearance_m,
                        None,
                        disable_descent=True,
                        require_solver_success=False,
                    )
                    self.last_marker_loss_policy = (
                        "descent_short_loss_level_track"
                    )
                    self.last_safety_failure_reason = "vision_stale"
                    if (
                        mission_now - self.marker_loss_started_mission_s
                        >= self._vision_motion_continuation_limit_s()
                    ):
                        self._transition(
                            LandingPhase.MARKER_TRACK_DOWN,
                            "down marker loss exceeded bounded level tracking",
                        )
                    return
                self._handle_descent_denial(decision, estimate)
                return
            self.marker_loss_started_mission_s = None
            # A short marker-loss recovery can clamp the descent reference
            # below the nominal FINAL entry height.  Never raise that valid
            # lower reference back to the configured threshold: doing so sent
            # an unnecessary climb command while matching a moving trailer.
            precision_descent_floor_m = min(
                self._float("final_approach_height_m"),
                self.descent_clearance_m,
            )
            candidate_clearance = max(
                precision_descent_floor_m,
                self.descent_clearance_m
                - self._float("precision_descent_speed_m_s")
                / self._float("control_rate_hz"),
            )
            # Advance the reference at the control rate, not only when the
            # asynchronous solver happens to return a result.  Gating
            # this assignment on ``result is not None`` reduced a configured
            # 0.45 m/s descent to roughly 0.13 m/s in the live run.
            self.descent_clearance_m = candidate_clearance
            mpc_clearance = self._descent_mpc_clearance_target(
                candidate_clearance,
                precision_descent_floor_m,
                self._float("precision_descent_speed_m_s"),
            )
            self._stage_trailer_relative_command(
                now, mpc_clearance, "down"
            )
            # FINAL owns the expected close-range optical gap only after both
            # the commanded clearance ramp and measured deck clearance reach
            # the terminal envelope.  Entering on measurement alone switches
            # solver context while the reference is still several metres high
            # and caused the observed moving-target divergence.
            if (
                self.descent_clearance_m
                <= self._float("final_approach_height_m") + 1.0e-6
                and self._final_approach_height_reached(now, estimate)
            ):
                measured_clearance = self._measured_target_clearance(
                    estimate, self.descent_clearance_m
                )
                if math.isfinite(measured_clearance):
                    self.descent_clearance_m = max(
                        self._float("touchdown_contact_clearance_m"),
                        min(self.descent_clearance_m, measured_clearance),
                    )
                self._transition(
                    LandingPhase.FINAL_APPROACH,
                    "measured terminal height",
                )
            return

    def _tick_final_approach(
        self, now: float, mission_now: float
    ) -> None:
        if self.phase == LandingPhase.FINAL_APPROACH:
            estimate = self._current_target_estimate()
            decision = self._descent_permission(
                now,
                estimate,
                require_solver_command=False,
            )

            # Synchronize a lagging command-side clearance to the measured
            # target-relative height on FINAL entry. Never raise the reference
            # after a bounce: doing so would command another climb instead of
            # continuing to match the moving deck horizontally.
            measured_clearance = self._measured_target_clearance(
                estimate, self.descent_clearance_m
            )
            if math.isfinite(measured_clearance):
                self.descent_clearance_m = max(
                    self._float("touchdown_contact_clearance_m"),
                    min(self.descent_clearance_m, measured_clearance),
                )

            if not self._fresh_landing_height(now):
                contact_settle_active = bool(
                    self._contact_settle_elapsed_s(now) is not None
                )
                contact_height_grace = bool(
                    contact_settle_active
                    and self._contact_height_within_latch(now)
                )
                if contact_settle_active and not contact_height_grace:
                    self._hold_descent_reference(now, estimate)
                    self._enter_failsafe_hold(
                        "terminal_contact_height_invalid"
                    )
                    return
                if contact_height_grace:
                    # At moving-deck contact the base-link estimate can cross
                    # the marker plane for a few samples. Keep the already
                    # latched level/contact controller alive for the existing
                    # bounded dropout grace so PX4's land detector can be
                    # consumed; never resume open-loop descent from this path.
                    self.last_safety_failure_reason = (
                        "terminal_contact_height_transient"
                    )
                else:
                    self._hold_descent_reference(now, estimate)
                    self.last_safety_failure_reason = "final_height_invalid"
                    if mission_now - self.phase_started >= self._float(
                        "final_approach_timeout_s"
                    ):
                        self._start_abort_climb(
                            "final approach height timeout", estimate
                        )
                    return
            final_descent_speed = self._final_descent_speed(now)
            candidate_clearance = max(
                self._float("touchdown_contact_clearance_m"),
                self.descent_clearance_m
                - final_descent_speed
                / self._float("control_rate_hz"),
            )
            mpc_clearance = self._descent_mpc_clearance_target(
                candidate_clearance,
                self._float("touchdown_contact_clearance_m"),
                final_descent_speed,
            )
            contact_ready = self._touchdown_contact_ready(
                now, estimate, candidate_clearance
            )
            contact_entry_ready = self._terminal_contact_entry_ready(
                now, estimate
            )
            if contact_entry_ready:
                self._arm_terminal_contact_entry()
            contact_tracking_latched = (
                self._terminal_contact_tracking_ready(now, estimate)
            )
            settle_elapsed = self._contact_settle_elapsed_s(now)
            latched_contact_evidence = False
            if settle_elapsed is not None:
                maximum_deck_distance = self._float(
                    "touchdown_max_deck_distance_m"
                )
                settle_timeout = self._float(
                    "touchdown_contact_settle_timeout_s"
                )
                latched_contact_evidence = bool(
                    settle_elapsed <= settle_timeout
                    and self._fresh_landing_height(now)
                    and self.landing_height_distance_m is not None
                    and math.isfinite(
                        float(self.landing_height_distance_m)
                    )
                    and float(self.landing_height_distance_m)
                    <= maximum_deck_distance
                )
            px4_landed = False
            if (
                contact_ready
                or self._contact_entry_latched
                or settle_elapsed is not None
            ):
                _ground, px4_landed, _at_rest, _stamp, _source = (
                    self._px4_land_detector_signals(now)
                )
            if px4_landed and (
                contact_ready
                or self._contact_entry_latched
                or latched_contact_evidence
            ):
                self.descent_clearance_m = self._float(
                    "touchdown_contact_clearance_m"
                )
                self.touchdown_gate.reset(self.time_reset_count)
                self.touchdown_disarm_requested = False
                self._transition(
                    LandingPhase.TOUCHDOWN_CONFIRM,
                    "PX4 ON_GROUND confirmed; begin touchdown dwell",
                )
                # Stay in OFFBOARD and keep matching the deck horizontally
                # until the multi-signal touchdown dwell permits a normal
                # disarm.  A world HOLD or PX4 AUTO.LAND would brake a 3 m/s
                # vehicle away from the moving deck.
                self._stage_touchdown_contact_osqp(
                    now,
                    estimate,
                    descent_allowed=True,
                    relative_descent_speed_m_s=self._float(
                        "touchdown_contact_settle_speed_m_s"
                    ),
                )
                return
            if (
                settle_elapsed is not None
                and settle_elapsed
                > self._float("touchdown_contact_settle_timeout_s")
            ):
                self._start_abort_climb(
                    "touchdown contact confirmation timeout",
                    estimate,
                )
                return
            if self._contact_entry_latched:
                # Contact is a one-way controller transition.  Collision
                # rebound can make the raw relative-vz gate false for a few
                # samples; returning to the ordinary clearance MPC in that
                # window produced an explicit climb command and lifted the
                # vehicle off the moving deck.  Keep staging only the contact
                # QP until PX4 reports ON_GROUND or the deck envelope is
                # genuinely lost.
                if not self._contact_height_within_latch(now):
                    violation_started = (
                        self._contact_height_violation_started_monotonic_s
                    )
                    if violation_started is None:
                        self._contact_height_violation_started_monotonic_s = (
                            float(now)
                        )
                        violation_started = float(now)
                    violation_elapsed = max(
                        0.0, float(now) - float(violation_started)
                    )
                    if violation_elapsed < self._float(
                        "touchdown_contact_latch_exit_dwell_s"
                    ):
                        self._stage_trailer_relative_command(
                            now,
                            self.descent_clearance_m,
                            "down",
                            disable_descent=True,
                            require_solver_success=False,
                            relative_descent_speed_m_s=0.0,
                        )
                        self.last_safety_failure_reason = (
                            "touchdown_contact_height_transient"
                        )
                        return
                    self._start_abort_climb(
                        "touchdown contact deck envelope lost",
                        estimate,
                    )
                    return
                self._contact_height_violation_started_monotonic_s = None
                if not self._terminal_horizontal_kinematics_safe(estimate):
                    # Contact entry does not waive the existing XY and
                    # relative-speed gates.  Recenter level on the moving
                    # deck before asking the tightly constrained terminal QP
                    # to descend again; otherwise an already-slipping state
                    # can make its hard funnel immediately infeasible.
                    self._stage_level_recovery_reference(now)
                    self.last_safety_failure_reason = (
                        "terminal_contact_level_recenter"
                    )
                    return
                if contact_tracking_latched and (
                    self._stage_touchdown_contact_osqp(
                        now,
                        estimate,
                        descent_allowed=True,
                    )
                ):
                    self.descent_clearance_m = self._float(
                        "touchdown_contact_clearance_m"
                    )
                    self.last_safety_failure_reason = None
                    return
                # A transient target/solver gap may stop descent, but it must
                # never fall back to the pre-contact MPC.  Retain horizontal
                # deck motion and level the vertical command while the normal
                # timeout/failsafe policy evaluates the persistent failure.
                self._cache_target_sensors_loss_level_coast()
                self.last_safety_failure_reason = (
                    "terminal_contact_tracking_unavailable"
                )
                return
            if not decision.allowed:
                # A single large marker necessarily leaves a finite-FOV
                # camera near contact.  After FINAL_APPROACH has already been
                # entered with fresh vision, bridge only that expected optical
                # occlusion using the same covariance-gated trailer estimate,
                # fresh relative geometry and constrained Relative MPC. Any
                # failed gate still holds/aborts below.
                failures = set(decision.failed_checks)
                optical_loss_failures = {
                    "vision_stale",
                    "position_covariance_invalid",
                    "position_covariance_exceeded",
                }
                if (
                    failures == {"solver_command_invalid"}
                    and self.consecutive_solver_failures
                    < self._int("max_consecutive_solver_failures")
                ):
                    # FINAL invalidates the ALIGN solve by design. Preserve the
                    # bounded transition coast while the first constrained
                    # FINAL snapshot runs; replacing it with a world-position
                    # HOLD would brake immediately against the moving deck.
                    staged = self._stage_trailer_relative_command(
                        now,
                        mpc_clearance,
                        "down",
                        disable_descent=True,
                        require_solver_success=False,
                    )
                    if staged is not None:
                        self.last_safety_failure_reason = (
                            "final_solver_staging"
                        )
                        return
                if (
                    failures.intersection(optical_loss_failures)
                    and self.marker_loss_started_mission_s is None
                ):
                    self.marker_loss_started_mission_s = mission_now
                occlusion_clearance = self._float(
                    "terminal_occlusion_max_clearance_m"
                )
                occlusion_height_limit = occlusion_clearance
                measured_clearance = self._measured_target_clearance(
                    estimate, math.inf
                )
                cached_velocity_age = (
                    None
                    if (
                        estimate is None
                        or self.last_vision_velocity_stamp_px4_s is None
                    )
                    else float(estimate.stamp_s)
                    - float(self.last_vision_velocity_stamp_px4_s)
                )
                cached_velocity = self.last_vision_velocity_enu
                aruco_velocity_bounded = bool(
                    cached_velocity is not None
                    and np.asarray(cached_velocity, dtype=float).shape == (3,)
                    and np.all(np.isfinite(cached_velocity))
                    and self.control_target_velocity_source
                    in {
                        "aruco_capture_kalman",
                        "aruco_capture_hold",
                        "aruco_terminal_hold",
                    }
                    and cached_velocity_age is not None
                    and 0.0 <= cached_velocity_age
                    <= self._float("vision_velocity_terminal_hold_s")
                )
                near_occlusion_envelope = bool(
                    settle_elapsed is None
                    and (
                        self.marker_loss_started_mission_s is None
                        or mission_now - self.marker_loss_started_mission_s
                        < self._float("final_approach_timeout_s")
                    )
                    and estimate is not None
                    and estimate.odometry_fresh
                    and self._terminal_occlusion_estimate_safe(estimate)
                    and aruco_velocity_bounded
                    and math.isfinite(measured_clearance)
                    and measured_clearance <= occlusion_clearance
                    and self._fresh_landing_height(now)
                    and self.landing_height_distance_m is not None
                    and math.isfinite(
                        float(self.landing_height_distance_m)
                    )
                    and float(self.landing_height_distance_m)
                    <= occlusion_height_limit
                )
                # Level recentering uses fused deck-relative geometry; marker
                # visibility is not required after the contact latch.
                level_recenter_envelope = bool(
                    settle_elapsed is None
                    and (
                        self.marker_loss_started_mission_s is None
                        or mission_now - self.marker_loss_started_mission_s
                        < self._float("final_approach_timeout_s")
                    )
                    and estimate is not None
                    and estimate.odometry_fresh
                    and self._terminal_occlusion_estimate_safe(estimate)
                    and aruco_velocity_bounded
                    and math.isfinite(measured_clearance)
                    and measured_clearance
                    <= self._float("final_approach_height_m") + 0.50
                    and self._fresh_landing_height(now)
                    and self.landing_height_distance_m is not None
                    and math.isfinite(
                        float(self.landing_height_distance_m)
                    )
                )
                if (
                    near_occlusion_envelope
                    and self.consecutive_solver_failures
                    >= self._int("max_consecutive_solver_failures")
                ):
                    # Keep the one terminal controller and rebuild a level
                    # moving-pad solve. FINAL->ALIGN invalidates the async
                    # cache and previously let an old downward command coast
                    # with no no-descent row active.
                    self.consecutive_solver_failures = 0
                    self.terminal_contact_bridge_active = True
                    self._stage_trailer_relative_command(
                        now,
                        mpc_clearance,
                        "down",
                        disable_descent=True,
                        require_solver_success=False,
                        relative_descent_speed_m_s=0.0,
                    )
                    self.last_safety_failure_reason = (
                        "terminal_constrained_level_recenter"
                    )
                    return
                if (
                    failures.intersection(optical_loss_failures)
                    and failures.issubset(
                        optical_loss_failures | {"solver_command_invalid"}
                    )
                    and "solver_command_invalid" in failures
                    and near_occlusion_envelope
                    and self.consecutive_solver_failures
                    < self._int("max_consecutive_solver_failures")
                ):
                    # Entering FINAL_APPROACH invalidates the ALIGN snapshot.
                    # Keep matching the moving deck while the first constrained
                    # final solve completes instead of bouncing back to ALIGN.
                    self.terminal_contact_bridge_active = True
                    self._stage_trailer_relative_command(
                        now,
                        mpc_clearance,
                        "down",
                    )
                    self.last_safety_failure_reason = (
                        "terminal_occlusion_solver_staging"
                    )
                    return
                optical_bridge = bool(
                    failures.intersection(optical_loss_failures)
                    and failures.issubset(optical_loss_failures)
                    and near_occlusion_envelope
                )
                if optical_bridge:
                    self.terminal_contact_bridge_active = True
                    if contact_ready:
                        if self._stage_touchdown_contact_osqp(
                            now, estimate, descent_allowed=True
                        ):
                            self.descent_clearance_m = self._float(
                                "touchdown_contact_clearance_m"
                            )
                            self.last_safety_failure_reason = None
                        else:
                            self._hold_descent_reference(now, estimate)
                        return
                    result = self._stage_trailer_relative_command(
                        now, mpc_clearance, "down"
                    )
                    self.descent_clearance_m = candidate_clearance
                    if result is not None:
                        self.last_safety_failure_reason = None
                    else:
                        # The asynchronous worker has already retained a
                        # bounded moving-pad coast command.  Do not overwrite
                        # it with a zero-velocity world HOLD while staging.
                        self.last_safety_failure_reason = (
                            "terminal_solver_staging"
                        )
                    return
                recoverable_terminal_geometry = bool(
                    level_recenter_envelope
                    and (
                        failures.intersection(
                            {
                                "outside_landing_funnel",
                                "relative_horizontal_speed_exceeded",
                            }
                        )
                        or (
                            failures.intersection(optical_loss_failures)
                            and not near_occlusion_envelope
                        )
                    )
                    and failures.issubset(
                        {
                            "vision_stale",
                            "solver_command_invalid",
                            "position_covariance_invalid",
                            "position_covariance_exceeded",
                            "outside_landing_funnel",
                            "relative_horizontal_speed_exceeded",
                        }
                    )
                )
                if recoverable_terminal_geometry:
                    # Stay in FINAL so the close-range optical bridge keeps
                    # the moving-pad reference and its bounded ArUco-derived
                    # velocity.  Switching to ALIGN here eventually expired
                    # that velocity cache and replaced 3 m/s deck tracking
                    # with an absolute world-frame HOLD.  Recenter level with
                    # FOV disabled, while every funnel/deck/dynamic constraint
                    # remains active, and resume descent only after the safety
                    # gates recover.
                    self.terminal_contact_bridge_active = True
                    self._stage_trailer_relative_command(
                        now,
                        mpc_clearance,
                        "down",
                        disable_descent=True,
                        require_solver_success=False,
                    )
                    self.last_safety_failure_reason = (
                        "terminal_geometry_level_recenter"
                    )
                    return
                bridge_allowed, bridge_reason = (
                    self._terminal_contact_bridge_permission(
                        now, estimate, decision
                    )
                )
                if bridge_allowed:
                    self.terminal_contact_bridge_active = True
                    if self._stage_touchdown_contact_osqp(
                        now,
                        estimate,
                        descent_allowed=True,
                    ):
                        self.descent_clearance_m = self._float(
                            "touchdown_contact_clearance_m"
                        )
                        self.last_safety_failure_reason = None
                    else:
                        self._hold_descent_reference(now, estimate)
                        self._enter_failsafe_hold(
                            "terminal_contact_bridge_command_invalid"
                        )
                    return
                if settle_elapsed is not None:
                    # Near contact, keep one continuous FINAL controller.
                    # Leaving for ALIGN invalidates the moving command and can
                    # never reacquire a marker already hidden by the landing
                    # gear.  Stop descent and recenter level with the same
                    # ArUco-derived motion plus measured trailer position.
                    if bridge_reason in {
                        "contact_conditions_invalid",
                        "contact_horizontal_kinematics_invalid",
                    }:
                        self.terminal_contact_bridge_active = True
                        self._stage_trailer_relative_command(
                            now,
                            mpc_clearance,
                            "down",
                            disable_descent=True,
                            require_solver_success=False,
                        )
                        self.last_safety_failure_reason = (
                            "terminal_contact_level_recenter"
                        )
                        return
                    self._hold_descent_reference(now, estimate)
                    self._enter_failsafe_hold(
                        f"terminal_contact_bridge_{bridge_reason}"
                    )
                    return
                self._hold_descent_reference(now, estimate)
                self._handle_descent_denial(decision, estimate)
                return
            # A close-range bridge is a one-way latch until touchdown or a
            # mission phase reset. Re-enabling FOV/reference switching on one
            # recovered frame made the QP active set and lateral reference
            # toggle at camera rate.
            self.marker_loss_started_mission_s = None
            if (
                settle_elapsed is None
                and mission_now - self.phase_started
                >= self._float("final_approach_timeout_s")
            ):
                self._start_abort_climb("final approach timeout", estimate)
                return
            if contact_ready:
                if self._stage_touchdown_contact_osqp(
                    now,
                    estimate,
                    descent_allowed=decision.allowed,
                ):
                    # The diagnostic floor remains fixed while OSQP lowers its
                    # bounded contact target and keeps matching deck motion.
                    self.descent_clearance_m = self._float(
                        "touchdown_contact_clearance_m"
                    )
                    self.last_safety_failure_reason = None
                else:
                    self._hold_descent_reference(now, estimate)
                    self.last_safety_failure_reason = (
                        "contact_settle_command_invalid"
                    )
                return
            result = self._stage_trailer_relative_command(
                now, mpc_clearance, "down"
            )
            self.descent_clearance_m = candidate_clearance
            return

    def _tick_touchdown_confirm(self, now: float, state) -> None:
        if self.phase == LandingPhase.TOUCHDOWN_CONFIRM:
            # Entry into TOUCHDOWN_CONFIRM is one-way and occurs only after the
            # guarded final-contact path has already observed PX4 ON_GROUND.
            # MAVROS ExtendedState.ON_GROUND is PX4's consolidated, internally
            # dwelled landing verdict.  Once it remains asserted, loss of the
            # now-occluded marker, fused estimate, or a fresh OSQP result must
            # not keep the motors armed or cause a takeoff command.
            ground, landed, at_rest, _sample_stamp, source = (
                self._px4_land_detector_signals(now)
            )
            if (
                source == "px4_composite_on_ground"
                and ground
                and landed
                and at_rest
            ):
                self.touchdown_disarm_requested = True
                self.command_cache = None
                self._invalidate_solver_context(clear_cached=True)
                if state is not None and state.armed:
                    self._request_arm(
                    False, now
                )
                return

            estimate = self._current_target_estimate()
            tracking_valid = self._stage_touchdown_contact_osqp(
                now,
                estimate,
                descent_allowed=True,
                # PX4 has already asserted ON_GROUND.  Keep matching the
                # moving deck in XY with only the low settle demand; the
                # stronger contact-compression demand must stop here.
                relative_descent_speed_m_s=self._float(
                    "touchdown_contact_settle_speed_m_s"
                ),
            )
            if not tracking_valid:
                self.last_safety_failure_reason = (
                    "touchdown_deck_tracking_invalid"
                )
                # Solver staging alone must never accumulate a disarm dwell.
                self.touchdown_gate.reset(self.time_reset_count)
            if self.touchdown_disarm_requested:
                # MAVROS service dispatch is asynchronous.  Retry at the
                # existing one-second rate until /mavros/state confirms that
                # the vehicle is disarmed; never resume compression meanwhile.
                if state is not None and state.armed:
                    self._request_arm(
                    False, now
                )
                return
            decision = self._update_touchdown_gate(now, estimate)
            # MAVROS ExtendedState.ON_GROUND is PX4's already-dwelled land
            # detector result.  When no independent PX4 contact topics are
            # configured, all three gate bits are conservatively derived from
            # that result and share its low-rate timestamp.  Requiring another
            # distinct-sample dwell here kept the motors armed at idle for
            # one or more seconds after PX4 had declared the vehicle landed.
            # Keep the full geometry/relative-speed checks above, but consume
            # this native confirmation immediately.  Independent bool-topic
            # inputs still use the normal multi-sample dwell.
            if decision.confirmed:
                # Latch the completed touchdown authorization before the
                # rate-limited MAVROS request.  Otherwise a duplicate land
                # detector sample can erase the one-shot confirmation while
                # PX4's automatic landed disarm wins the race.
                self.touchdown_disarm_requested = True
                self._request_arm(False, now)
                return
            if decision.timed_out:
                if state is not None and state.armed:
                    self._start_abort_climb(
                        "touchdown confirmation timeout", estimate
                    )
                else:
                    self._enter_failsafe_hold(
                        "touchdown timeout while disarmed"
                    )
            return
    def _gate_dwell(self, condition: bool, now: float, dwell_s: float) -> bool:
        if not condition:
            self.within_gate_since = None
            return False
        if self.within_gate_since is None:
            self.within_gate_since = now
            return False
        return now - self.within_gate_since >= dwell_s
    def _mission_time_s(self) -> float:
        """Return phase time in the domain that advances mission dynamics."""
        if self._mission_uses_ros_time:
            return self.get_clock().now().nanoseconds * 1.0e-9
        return time.monotonic()
    def _transition(self, phase: LandingPhase, explanation: str) -> None:
        if phase == self.phase:
            return
        previous = self.phase
        transition_mission_time_s = self._mission_time_s()
        if (
            phase == LandingPhase.PRECISION_ALIGN
            and previous == LandingPhase.FINAL_APPROACH
        ):
            # A terminal recenter gets its own bounded recovery window.
            # Reusing the timestamp of the first optical occlusion meant a
            # later corner could enter ALIGN with most of its timeout already
            # consumed and abort while the error was visibly converging.
            self.terminal_recovery_started_mission_s = (
                transition_mission_time_s
            )
        elif (
            phase == LandingPhase.FINAL_APPROACH
            and previous == LandingPhase.PRECISION_ALIGN
        ):
            self.terminal_recovery_started_mission_s = None
            # FINAL_APPROACH owns the bounded odometry optical bridge.
            # Carrying the older low-altitude marker-loss timestamp into this
            # phase made it abort 80 ms after a successful 15 s recenter.
            self.marker_loss_started_mission_s = transition_mission_time_s
        elif phase not in {
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.FINAL_APPROACH,
        }:
            self.terminal_recovery_started_mission_s = None
        if (
            phase == LandingPhase.PRECISION_ALIGN
            and previous
            in {
                LandingPhase.PRECISION_DESCENT,
                LandingPhase.FINAL_APPROACH,
            }
        ):
            # Freeze at the lower of the last authorized target and measured
            # clearance on a descent denial.  A recovery must not ratchet the
            # target upward after camera/odometry disagreement, while the
            # contact floor still prevents pressing through the deck.
            estimate = self._current_target_estimate()
            measured = self._measured_target_clearance(
                estimate,
                self.descent_clearance_m,
            )
            if math.isfinite(measured):
                self.descent_clearance_m = max(
                    self._float("touchdown_contact_clearance_m"),
                    min(self.descent_clearance_m, measured),
                )
        if (
            phase == LandingPhase.FINAL_APPROACH
            and previous
            in {
                LandingPhase.PRECISION_ALIGN,
                LandingPhase.PRECISION_DESCENT,
            }
        ):
            # Enter FINAL from the aircraft's measured clearance when it has
            # already descended below the nominal phase boundary.  Retaining
            # the older 3.1 m reference commanded a corrective climb of about
            # 0.5 m in SITL before descending again, producing the observed
            # vertical jerk and wasting several seconds.
            estimate = self._current_target_estimate()
            measured = self._measured_target_clearance(
                estimate,
                self.descent_clearance_m,
            )
            if math.isfinite(measured):
                self.descent_clearance_m = max(
                    self._float("touchdown_contact_clearance_m"),
                    min(self.descent_clearance_m, measured),
                )
        self.phase = phase
        if phase == LandingPhase.MARKER_TRACK_DOWN:
            self.marker_tracking_acquisition_since_s = None
        elif previous == LandingPhase.MARKER_TRACK_DOWN:
            self.marker_tracking_acquisition_since_s = None
        reset_terminal = getattr(
            self, "_reset_terminal_contact_context", None
        )
        if reset_terminal is not None:
            reset_terminal()
        # Revoke queued, in-flight, and cached work atomically with the phase
        # change. During moving-deck tracking, bridge planned transitions with
        # the measured velocity instead of a zero-velocity world HOLD. The
        # heartbeat remains the only publisher and the next phase immediately
        # stages a fresh MPC snapshot.
        self._invalidate_solver_context(clear_cached=True)
        position = self.vehicle_position_enu
        velocity = self.vehicle_velocity_enu
        yaw = float(self.vehicle_yaw_enu_rad)
        tracking_phases = {
            LandingPhase.APPROACH,
            LandingPhase.MARKER_TRACK_DOWN,
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
        }
        coast_transition = bool(
            phase in tracking_phases
            and previous in tracking_phases | {LandingPhase.TAKEOFF}
            and velocity is not None
            and np.asarray(velocity).shape == (3,)
            and np.all(np.isfinite(velocity))
        )
        if phase == LandingPhase.LANDED:
            # LANDED is terminal: no hold or stale MPC command may remain in
            # the cache even though the heartbeat phase gate also blocks it.
            self.command_cache = None
        elif (
            position is not None
            and np.asarray(position).shape == (3,)
            and np.all(np.isfinite(position))
            and math.isfinite(yaw)
        ):
            if coast_transition:
                disabled = np.full(3, np.nan, dtype=float)
                transition_velocity = np.asarray(
                    velocity, dtype=float
                ).copy()
                if phase == LandingPhase.TOUCHDOWN_CONFIRM:
                    # Do not copy a contact-bounce upward velocity into the
                    # transition coast.  Match the deck with only the bounded
                    # low settle demand while the first touchdown QP runs.
                    estimate = self._current_target_estimate()
                    target_velocity = self._control_target_velocity_enu(
                        estimate,
                        time.monotonic(),
                    )
                    settle_speed = self._float(
                        "touchdown_contact_settle_speed_m_s"
                    )
                    if (
                        target_velocity is not None
                        and target_velocity.shape == (3,)
                        and np.all(np.isfinite(target_velocity))
                    ):
                        transition_velocity[2] = (
                            target_velocity[2] - settle_speed
                        )
                    else:
                        transition_velocity[2] = -settle_speed
                elif (
                    phase == LandingPhase.PRECISION_ALIGN
                    and previous
                    in {
                        LandingPhase.PRECISION_DESCENT,
                        LandingPhase.FINAL_APPROACH,
                    }
                ):
                    # Recovery must stop relative descent immediately while
                    # retaining horizontal deck velocity.  Copying the full
                    # measured vehicle velocity here allowed the last downward
                    # command to coast through every ALIGN transition.
                    estimate = self._current_target_estimate()
                    target_velocity = self._control_target_velocity_enu(
                        estimate,
                        time.monotonic(),
                    )
                    if (
                        target_velocity is not None
                        and target_velocity.shape == (3,)
                        and np.all(np.isfinite(target_velocity))
                    ):
                        transition_velocity[2] = target_velocity[2]
                    else:
                        transition_velocity[2] = max(
                            0.0, float(transition_velocity[2])
                        )
                transition_command = LandingControlCommand(
                    disabled,
                    transition_velocity,
                    disabled,
                    yaw,
                    False,
                    True,
                    False,
                    valid=True,
                    degraded=False,
                    status="phase_transition_coast",
                    solve_time_s=0.0,
                    controller_type="transition_coast",
                    primary_controller_type="transition_coast",
                )
            else:
                transition_command = LandingControlCommand.hold(
                    position,
                    yaw,
                    valid=True,
                    degraded=False,
                    status="phase_transition_hold",
                )
            self._cache_command(transition_command)
            if transition_command.controller_type == "transition_coast":
                self._transition_command_monotonic_s = time.monotonic()
                self._transition_command_phase = self.phase.value
                self._transition_command_time_reset_count = (
                    self.time_reset_count
                )
        else:
            self.command_cache = None
        self.phase_started = transition_mission_time_s
        self.last_transition_reason = str(explanation)
        self.within_gate_since = None
        self.get_logger().info(
            f"landing phase {previous.value} -> {phase.value}: {explanation}"
        )
