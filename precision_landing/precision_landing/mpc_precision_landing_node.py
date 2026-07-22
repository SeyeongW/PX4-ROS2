#!/usr/bin/env python3
"""Single production ArUco landing mission for PX4 through MAVROS.

One launcher process holds the repository control-authority lock because this
node is the sole publisher of ``/mavros/setpoint_raw/local``.

ArUco pose, yaw, and covariance are transformed with the vehicle state at the
image capture stamp and fused with marker-offset-corrected trailer odometry.
One fixed Relative-OSQP controller runs on a background worker; P/feed-forward
is used only internally while transiting without a complete landing context.
The sole MAVROS heartbeat publisher consumes a thread-safe command cache.
Every downward step is gated on fused-target confidence and touchdown requires
continuous geometric, kinematic, and PX4 land-detector evidence before a
normal disarm request.
Solver deadlines and data-freshness checks remain on ``time.monotonic()``;
mission phase elapsed time follows the ROS clock in simulation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
from threading import RLock
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    TwistStamped,
)
from mavros_msgs.msg import (
    ExtendedState,
    PositionTarget,
    State,
    TimesyncStatus,
)
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .async_landing_control import (
    AsyncLandingControllerWorker,
    LandingSolveSnapshot,
    SolutionAcceptanceLimits,
)
from .capture_state_history import (
    MavrosTimesyncTracker,
    TimestampOffsetEstimator,
    VehicleStateHistory,
)
from .marker_measurement import (
    CaptureStampGuard,
)
from .landing_controller import (
    LandingControlCommand,
)
from .landing_safety import (
    DescentPermissionDecision,
    DescentPermissionLimits,
    TouchdownGate,
    TouchdownGateDecision,
    TouchdownGateLimits,
)
from .production_runtime import (
    PRODUCTION_CONTROLLER_TYPE,
    PRODUCTION_SOLVER_NAME,
    LandingConfig,
    LatestCommandCache,
    ProductionLandingStack,
    TrailerPositionSample,
)
from .status_payload import dumps_strict_json, sanitize_json_value
from .command_runtime import CommandRuntimeMixin
from .mission_runtime import MissionRuntimeMixin
from .runtime_types import LandingPhase
from .target_runtime import TargetRuntimeMixin
from .target_fusion import (
    FusionParameters,
    TargetEstimate,
    TimestampedTargetEstimator,
)
from .timestamped_dynamics import (
    TrailerBicycleState,
    TrailerPredictionParameters,
    TrailerRk4Predictor,
    VehicleResponseEstimator,
    VehicleResponseParameters,
)


class MpcPrecisionLandingNode(
    TargetRuntimeMixin,
    CommandRuntimeMixin,
    MissionRuntimeMixin,
    Node,
):
    """Single-publisher MPC controller for the trailer experiment."""

    def __init__(self) -> None:
        super().__init__("mpc_precision_landing")
        parameter_names = self._declare_parameters()
        self.config = LandingConfig(
            {
                name: self.get_parameter(name).value
                for name in (*parameter_names, "use_sim_time")
            }
        )
        self.mavros_state_timestamp_source = self._string(
            "mavros_state_timestamp_source"
        ).strip()
        if self.mavros_state_timestamp_source not in {
            "timesync_status",
            "ros_header",
        }:
            raise ValueError(
                "mavros_state_timestamp_source must be timesync_status or "
                "ros_header"
            )
        self._mission_uses_ros_time = self._bool("use_sim_time")
        relative_solver_parameters = {
            "dt_s": self._float("relative_landing_mpc_dt_s"),
            "horizon_steps": self._int(
                "relative_landing_mpc_horizon_steps"
            ),
            "max_horizontal_velocity_m_s": self._float(
                "relative_landing_mpc_max_horizontal_velocity_m_s"
            ),
            "max_ascent_velocity_m_s": self._float(
                "relative_landing_mpc_max_ascent_velocity_m_s"
            ),
            "max_descent_velocity_m_s": self._float(
                "relative_landing_mpc_max_descent_velocity_m_s"
            ),
            "max_horizontal_acceleration_m_s2": self._float(
                "relative_landing_mpc_max_horizontal_acceleration_m_s2"
            ),
            "max_vertical_acceleration_m_s2": self._float(
                "relative_landing_mpc_max_vertical_acceleration_m_s2"
            ),
            "max_jerk_m_s3": self._float(
                "relative_landing_mpc_max_jerk_m_s3"
            ),
            "funnel_minimum_radius_m": self._float(
                "relative_landing_mpc_funnel_minimum_radius_m"
            ),
            "funnel_slope": self._float(
                "relative_landing_mpc_funnel_slope"
            ),
            "camera_horizontal_fov_rad": self._float(
                "relative_landing_mpc_camera_horizontal_fov_rad"
            ),
            "camera_vertical_fov_rad": self._float(
                "relative_landing_mpc_camera_vertical_fov_rad"
            ),
            "camera_fov_margin": self._float(
                "relative_landing_mpc_camera_fov_margin"
            ),
            "landing_pad_length_m": self._float(
                "relative_landing_mpc_landing_pad_length_m"
            ),
            "landing_pad_width_m": self._float(
                "relative_landing_mpc_landing_pad_width_m"
            ),
            "landing_pad_contact_margin_m": self._float(
                "relative_landing_mpc_landing_pad_contact_margin_m"
            ),
            "alignment_position_tolerance_m": self._float(
                "relative_landing_mpc_alignment_position_tolerance_m"
            ),
            "alignment_velocity_tolerance_m": self._float(
                "relative_landing_mpc_alignment_velocity_tolerance_m_s"
            ),
        }
        self.production_stack = ProductionLandingStack.create(
            solver_parameters=relative_solver_parameters,
            solver_deadline_s=0.001 * self._float("landing_mpc_deadline_ms"),
            maximum_iterations=self._int("landing_mpc_max_iterations"),
            absolute_tolerance=self._float("landing_mpc_absolute_tolerance"),
            relative_tolerance=self._float("landing_mpc_relative_tolerance"),
            transit_horizontal_gain=self._float("landing_p_horizontal_gain"),
            transit_vertical_gain=self._float("landing_p_vertical_gain"),
        )
        self.landing_controller = self.production_stack.controller
        self.landing_mpc_solver_name = PRODUCTION_SOLVER_NAME
        self.solution_acceptance_limits = SolutionAcceptanceLimits(
            maximum_solution_age_s=(
                0.001 * self._float("mpc_solution_max_age_ms")
            ),
            maximum_vehicle_position_change_m=self._float(
                "mpc_max_vehicle_position_change_m"
            ),
            maximum_vehicle_velocity_change_m_s=self._float(
                "mpc_max_vehicle_velocity_change_m_s"
            ),
            maximum_vehicle_acceleration_change_m_s2=self._float(
                "mpc_max_vehicle_acceleration_change_m_s2"
            ),
            maximum_target_position_change_m=self._float(
                "mpc_max_target_position_change_m"
            ),
            maximum_target_velocity_change_m_s=self._float(
                "mpc_max_target_velocity_change_m_s"
            ),
            maximum_target_acceleration_change_m_s2=self._float(
                "mpc_max_target_acceleration_change_m_s2"
            ),
            maximum_target_yaw_change_rad=self._float(
                "mpc_max_target_yaw_change_rad"
            ),
        )
        self.descent_permission_limits = DescentPermissionLimits(
            maximum_vision_age_s=self._float("descent_max_vision_age_s"),
            maximum_odometry_age_s=self._float("trailer_timeout_s"),
            maximum_position_variance_m2=self._float(
                "descent_max_position_variance_m2"
            ),
            maximum_relative_horizontal_speed_m_s=self._float(
                "relative_speed_gate_m_s"
            ),
            maximum_relative_height_m=self._float(
                "descent_max_relative_height_m"
            ),
        )
        self.touchdown_gate = TouchdownGate(
            TouchdownGateLimits(
                maximum_deck_distance_m=self._float(
                    "touchdown_max_deck_distance_m"
                ),
                maximum_relative_vertical_speed_m_s=self._float(
                    "touchdown_max_relative_vertical_speed_m_s"
                ),
                dwell_time_s=self._float("touchdown_dwell_s"),
                maximum_sample_age_s=self._float(
                    "land_detector_sample_timeout_s"
                ),
                maximum_future_skew_s=self._float(
                    "land_detector_future_tolerance_s"
                ),
                maximum_inter_sample_gap_s=self._float(
                    "land_detector_max_sample_gap_s"
                ),
                confirmation_timeout_s=self._float(
                    "touchdown_confirmation_timeout_s"
                ),
                minimum_consecutive_samples=self._int(
                    "touchdown_minimum_consecutive_samples"
                ),
            )
        )
        heartbeat_rate = self._float("control_rate_hz")
        solver_rate = self._float("mpc_solver_rate_hz")
        if not 20.0 <= heartbeat_rate <= 50.0:
            raise ValueError("control_rate_hz must be within [20, 50]")
        if not 10.0 <= solver_rate <= 50.0:
            raise ValueError("mpc_solver_rate_hz must be within [10, 50]")
        marker_entry_yaw_error = self._float(
            "marker_track_entry_yaw_error_rad"
        )
        if not 0.0 < marker_entry_yaw_error <= math.pi:
            raise ValueError(
                "marker_track_entry_yaw_error_rad must be within (0, pi]"
            )
        capture_radius = self._float("approach_capture_radius_m")
        acquisition_values = (
            self._float("position_only_pursuit_max_speed_m_s"),
            self._float("position_only_pursuit_acceleration_m_s2"),
            capture_radius,
            self._float("position_only_pursuit_gain_s_inv"),
            self._float("marker_detection_status_timeout_s"),
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in acquisition_values
        ):
            raise ValueError(
                "ArUco acquisition limits must be finite and positive"
            )
        if self._int("max_consecutive_solver_failures") < 1:
            raise ValueError(
                "max_consecutive_solver_failures must be positive"
            )
        target_loss_dwell = self._float(
            "target_sensors_loss_abort_dwell_s"
        )
        if (
            not math.isfinite(target_loss_dwell)
            or not 0.0 < target_loss_dwell <= 5.0
        ):
            raise ValueError(
                "target_sensors_loss_abort_dwell_s must be within (0, 5]"
            )
        final_descent_braking_acceleration = self._float(
            "final_descent_braking_acceleration_m_s2"
        )
        if (
            not math.isfinite(final_descent_braking_acceleration)
            or final_descent_braking_acceleration <= 0.0
            or final_descent_braking_acceleration
            > self._float(
                "relative_landing_mpc_max_vertical_acceleration_m_s2"
            )
        ):
            raise ValueError(
                "final descent braking acceleration must be finite, positive, "
                "and within the vertical acceleration limit"
            )
        contact_settle_speed = self._float(
            "touchdown_contact_settle_speed_m_s"
        )
        contact_settle_timeout = self._float(
            "touchdown_contact_settle_timeout_s"
        )
        contact_evidence_height = self._float(
            "touchdown_contact_evidence_clearance_m"
        )
        contact_compression_speed = self._float(
            "touchdown_contact_compression_speed_m_s"
        )
        contact_compression_ramp_rate = self._float(
            "touchdown_contact_compression_ramp_rate_m_s2"
        )
        contact_height_rebound_tolerance = self._float(
            "touchdown_contact_height_rebound_tolerance_m"
        )
        contact_latch_exit_distance = self._float(
            "touchdown_contact_latch_exit_distance_m"
        )
        contact_latch_exit_dwell = self._float(
            "touchdown_contact_latch_exit_dwell_s"
        )
        contact_clearance = self._float(
            "touchdown_contact_clearance_m"
        )
        contact_deck_distance = self._float(
            "touchdown_max_deck_distance_m"
        )
        contact_relative_speed = self._float(
            "touchdown_max_relative_vertical_speed_m_s"
        )
        if (
            not math.isfinite(contact_settle_speed)
            or contact_settle_speed <= 0.0
            or not math.isfinite(contact_settle_timeout)
            or contact_settle_timeout <= 0.0
            or contact_settle_timeout
            > self._float("final_approach_timeout_s")
            or contact_settle_timeout
            > self._float("touchdown_confirmation_timeout_s")
            or not math.isfinite(contact_height_rebound_tolerance)
            or contact_height_rebound_tolerance < 0.0
            or not math.isfinite(contact_clearance)
            or contact_clearance < 0.0
            or not math.isfinite(contact_deck_distance)
            or contact_deck_distance <= 0.0
            or not math.isfinite(contact_relative_speed)
            or contact_relative_speed <= 0.0
            or contact_settle_speed
            > self._float("landing_p_vertical_velocity_limit_m_s")
            or contact_settle_speed > contact_relative_speed
            or not math.isfinite(contact_evidence_height)
            or contact_evidence_height
            < 0.0
            or contact_evidence_height > contact_deck_distance
            or not math.isfinite(contact_compression_speed)
            or contact_compression_speed < contact_settle_speed
            or contact_compression_speed
            > self._float("landing_p_vertical_velocity_limit_m_s")
            or not math.isfinite(contact_compression_ramp_rate)
            or contact_compression_ramp_rate <= 0.0
            or contact_compression_ramp_rate
            > self._float(
                "relative_landing_mpc_max_vertical_acceleration_m_s2"
            )
            or contact_height_rebound_tolerance > contact_deck_distance
            or not math.isfinite(contact_latch_exit_distance)
            or contact_latch_exit_distance
            <= contact_deck_distance + contact_height_rebound_tolerance
            or contact_latch_exit_distance > 1.0
            or not math.isfinite(contact_latch_exit_dwell)
            or not 0.0 < contact_latch_exit_dwell <= 2.0
        ):
            raise ValueError(
                "touchdown contact-settle limits must be finite, bounded by "
                "the final/confirmation timeouts and deck distance, and no "
                "greater than the vertical command or touchdown-speed limits"
            )
        descent_vision_age = self._float("descent_max_vision_age_s")
        if not 0.0 < descent_vision_age <= self._float("marker_timeout_s"):
            raise ValueError(
                "descent_max_vision_age_s must be positive and no greater "
                "than marker_timeout_s"
            )
        # Descent freshness now uses callback age and subtracts only the
        # estimator's intentional reorder delay.  It therefore need not
        # include that delay or the separate alignment dwell.
        maximum_marker_tilt = self._float(
            "maximum_marker_world_tilt_rad"
        )
        if (
            not math.isfinite(maximum_marker_tilt)
            or maximum_marker_tilt <= 0.0
            or maximum_marker_tilt >= 0.5 * math.pi
        ):
            raise ValueError(
                "maximum_marker_world_tilt_rad must be in (0, pi/2)"
            )
        if self._float("state_history_max_capture_age_s") > self._float(
            "target_fusion_maximum_measurement_age_s"
        ):
            raise ValueError(
                "state capture age cannot exceed fusion admission age"
            )
        if (
            self._float("marker_loss_low_altitude_m")
            >= self._float("marker_loss_high_altitude_m")
        ):
            raise ValueError(
                "marker loss altitude thresholds must be increasing"
            )
        if self._float("low_altitude_marker_reacquire_timeout_s") <= 0.0:
            raise ValueError(
                "low-altitude marker reacquisition timeout must be positive"
            )
        terminal_occlusion_clearance = self._float(
            "terminal_occlusion_max_clearance_m"
        )
        if (
            not 0.0 < terminal_occlusion_clearance
            <= self._float("final_approach_height_m")
        ):
            raise ValueError(
                "terminal optical-occlusion clearance must be positive and "
                "at or below final approach height"
            )
        land_detector_topics = tuple(
            self._string(name).strip()
            for name in (
                "px4_ground_contact_topic",
                "px4_landed_topic",
                "px4_at_rest_topic",
            )
        )
        if any(land_detector_topics) and not all(land_detector_topics):
            raise ValueError(
                "PX4 land-detector Bool bridge topics must be configured "
                "together"
            )

        self.phase = LandingPhase.WAITING
        self.phase_started = self._mission_time_s()
        self.last_transition_reason = "initialized"
        self.start_requested = self._bool("auto_start")
        if self.start_requested and not self._activation_qualified():
            self.start_requested = False
            self.get_logger().error(
                "auto_start rejected: production controller is not qualified"
            )
        self.takeoff_origin: np.ndarray | None = None
        self.takeoff_from_trailer = False
        self.takeoff_trailer_offset_body_m = np.zeros(3, dtype=float)
        self.takeoff_feedforward_velocity_enu = np.zeros(3, dtype=float)
        self.takeoff_trailer_previous_sample_time_s: float | None = None
        self.takeoff_trailer_previous_position_enu: np.ndarray | None = None
        self.takeoff_trailer_velocity_initialized = False
        self.descent_clearance_m = self._float("precision_align_height_m")
        self.failsafe_hold_position_enu: np.ndarray | None = None
        self.abort_climb_target_enu: np.ndarray | None = None
        self.last_descent_permission: DescentPermissionDecision | None = None
        self.last_touchdown_decision: TouchdownGateDecision | None = None
        self.touchdown_disarm_requested = False
        self.last_safety_failure_reason: str | None = None
        self.last_marker_loss_policy: str | None = None
        self.marker_loss_started_mission_s: float | None = None
        self.terminal_recovery_started_mission_s: float | None = None
        self.last_mode_request = 0.0
        self.last_arm_request = 0.0
        self.within_gate_since: float | None = None

        self.mavros_state: State | None = None
        self.extended_state: ExtendedState | None = None
        self.extended_state_stamp = 0.0
        self.px4_ground_contact_override: bool | None = None
        self.px4_ground_contact_override_stamp = 0.0
        self.px4_landed_override: bool | None = None
        self.px4_landed_override_stamp = 0.0
        self.px4_at_rest_override: bool | None = None
        self.px4_at_rest_override_stamp = 0.0
        self.vehicle_position_enu: np.ndarray | None = None
        self.vehicle_velocity_enu = np.zeros(3, dtype=float)
        self.vehicle_acceleration_enu = np.zeros(3, dtype=float)
        self.vehicle_control_position_enu: np.ndarray | None = None
        self.vehicle_control_velocity_enu: np.ndarray | None = None
        self.vehicle_control_sample_time_s: float | None = None
        self.vehicle_yaw_enu_rad = 0.0
        self.pose_stamp = 0.0
        self.velocity_stamp = 0.0
        self.vehicle_response_estimator = VehicleResponseEstimator(
            VehicleResponseParameters(
                maximum_acceleration_m_s2=8.0,
                maximum_disturbance_m_s2=3.0,
                maximum_sample_gap_s=0.50,
                maximum_prediction_horizon_s=0.25,
            )
        )
        self.last_vehicle_response_reason = "uninitialized"
        self.state_history = VehicleStateHistory(
            duration_s=self._float("state_history_duration_s"),
            reset_threshold_s=self._float("time_reset_threshold_s"),
        )
        self.camera_time_offset = TimestampOffsetEstimator(
            smoothing_factor=self._float(
                "camera_time_offset_smoothing_factor"
            ),
            minimum_samples=self._int("camera_time_offset_minimum_samples"),
            reset_threshold_s=self._float("time_reset_threshold_s"),
            offset_reset_threshold_s=self._float(
                "camera_time_offset_reset_threshold_s"
            ),
            median_window_size=self._int(
                "camera_time_offset_median_window_size"
            ),
        )
        self.mavros_timesync = MavrosTimesyncTracker(
            minimum_samples=self._int("mavros_timesync_minimum_samples"),
            reset_threshold_s=self._float("time_reset_threshold_s"),
            maximum_round_trip_time_ms=self._float(
                "mavros_timesync_max_round_trip_time_ms"
            ),
        )
        self.mavros_timesync_stamp = 0.0
        self.time_reset_count = 0
        self.last_time_reset_reason: str | None = None
        self.landing_height_distance_m: float | None = None
        self.last_valid_landing_height_distance_m: float | None = None
        self.last_valid_landing_height_stamp = 0.0

        self.trailer_state: TrailerPositionSample | None = None
        self.trailer_state_stamp = 0.0
        fusion_parameters = self._fusion_parameters()
        self.target_estimator = TimestampedTargetEstimator(
            fusion_parameters
        )
        # This second estimator receives ArUco poses only.  Its velocity is
        # therefore camera-observed target motion, never trailer odometry
        # twist.  The primary estimator remains responsible for fused target
        # position/yaw and covariance.
        self.vision_motion_estimator = TimestampedTargetEstimator(
            replace(
                fusion_parameters,
                # The camera-only filter must follow the direction change of
                # the moving target without exposing trailer twist. Keep the
                # lower process noise on the primary position fusion filter.
                process_acceleration_variance=self._float(
                    "vision_motion_process_acceleration_variance"
                ),
            )
        )
        self.last_target_estimate: TargetEstimate | None = None
        self.last_vision_motion_estimate: TargetEstimate | None = None
        self.last_vision_velocity_enu: np.ndarray | None = None
        self.last_vision_velocity_stamp_px4_s: float | None = None
        self.last_vision_velocity_source_stamp_px4_s: float | None = None
        self.last_vision_motion_model_stamp_px4_s: float | None = None
        self.vision_motion_model_covariance: np.ndarray | None = None
        self._trailer_prediction_cache: TrailerBicycleState | None = None
        self.last_vision_motion_capture_px4_time_s: float | None = None
        self.vision_velocity_history: deque[tuple[float, np.ndarray]] = (
            deque()
        )
        self.vision_position_history: deque[tuple[float, np.ndarray]] = (
            deque()
        )
        self.vision_position_candidates: deque[
            tuple[float, np.ndarray]
        ] = deque()
        self.vision_position_model_qualified = False
        self.vision_motion_model_turn_rate_rad_s = 0.0
        self.vision_motion_model_tangential_acceleration_m_s2 = 0.0
        self.vision_motion_model_span_s = 0.0
        self.marker_tracking_acquisition_since_s: float | None = None
        self.control_target_velocity_source = "unavailable"
        self.trailer_motion_predictor = TrailerRk4Predictor(
            TrailerPredictionParameters(
                wheelbase_m=2.80,
                maximum_acceleration_m_s2=self._float(
                    "relative_landing_mpc_max_pad_acceleration_m_s2"
                ),
                maximum_measurement_age_s=self._float(
                    "vision_velocity_terminal_hold_s"
                ),
                maximum_integration_step_s=0.10,
            )
        )
        self.last_trailer_prediction_reason = "uninitialized"
        self.down_marker_capture_guard = CaptureStampGuard()
        self.down_marker_stamp = 0.0
        self.down_marker_detected_stamp = 0.0
        self.down_marker_positive_stamp = 0.0
        self.down_marker_sequence = 0
        self.down_marker_invalid_covariance_count = 0
        self._pending_marker_messages: dict[
            str, tuple[int, PoseWithCovarianceStamped]
        ] = {}
        self._pending_vehicle_response_sample: (
            tuple[float, np.ndarray] | None
        ) = None
        self._pending_trailer_odometry: tuple[int, Odometry] | None = None

        self.last_command_acceleration_enu: np.ndarray | None = None
        self._command_cache_store = LatestCommandCache()
        self.last_landing_horizontal_error_m: float | None = None
        self.last_landing_relative_horizontal_speed_m_s: float | None = None
        self.last_requested_clearance_m: float | None = None
        self.landing_pad_max_acceleration_m_s2 = self._float(
            "relative_landing_mpc_max_pad_acceleration_m_s2"
        )
        if self.landing_pad_max_acceleration_m_s2 <= 0.0:
            raise ValueError("pad acceleration limit must be positive")
        self.landing_pad_acceleration_enu = np.zeros(3, dtype=float)
        self.last_mpc_message = "not_started"
        self.last_mpc_success = False
        self.last_mpc_solve_time_s = 0.0
        # The control, solver and heartbeat timers use separate callback
        # groups.  Keep the complete async-solver context under one lock so a
        # pending snapshot cannot be replaced and then cleared by another
        # callback between two Python statements.  The heartbeat never takes
        # this lock; it only reads the independently locked command cache.
        self._solver_context_lock = RLock()
        self._solver_snapshot_generation = 0
        self._last_processed_solver_generation = 0
        self._revoked_solver_generation = 0
        self._pending_solver_snapshot: LandingSolveSnapshot | None = None
        self._latest_solver_snapshot: LandingSolveSnapshot | None = None
        self._cached_solver_snapshot: LandingSolveSnapshot | None = None
        self._control_tick_snapshot_candidate: (
            LandingSolveSnapshot | None
        ) = None
        self._control_tick_snapshot_staging = False
        self._cached_p_command_monotonic_s: float | None = None
        self._cached_p_command_phase: str | None = None
        self._cached_p_command_time_reset_count: int | None = None
        self._acquisition_command_monotonic_s: float | None = None
        self._acquisition_command_phase: str | None = None
        self._acquisition_command_time_reset_count: int | None = None
        self.last_position_servo_velocity_enu = np.zeros(3, dtype=float)
        self.last_position_servo_update_monotonic_s: float | None = None
        self._transition_command_monotonic_s: float | None = None
        self._transition_command_phase: str | None = None
        self._transition_command_time_reset_count: int | None = None
        self._contact_settle_started_monotonic_s: float | None = None
        self._contact_settle_started_time_reset_count: int | None = None
        self._contact_settle_min_height_m: float | None = None
        self._contact_entry_latched = False
        self._contact_compression_started_monotonic_s: float | None = None
        self._contact_height_violation_started_monotonic_s: (
            float | None
        ) = None
        self._target_sensors_loss_started_monotonic_s: float | None = None
        self.terminal_contact_bridge_active = False
        self._time_reset_hold_pending = False
        self.stale_solution_rejected_count = 0
        self.consecutive_solver_failures = 0
        self._heartbeat_timestamps: deque[float] = deque(maxlen=101)
        self._setpoint_timestamps: deque[float] = deque(maxlen=101)
        self.solver_worker = AsyncLandingControllerWorker(
            self.landing_controller
        )

        # The heartbeat cache is latest-command-only.  A depth-10 publisher at
        # 50 Hz can leave 0.2 s of obsolete contact-compression setpoints in
        # transport after touchdown, so keep exactly one current command.
        setpoint_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.setpoint_pub = self.create_publisher(
            PositionTarget, "/mavros/setpoint_raw/local", setpoint_qos
        )
        self.status_pub = self.create_publisher(String, "/autonomy/status", 10)
        self.create_subscription(State, "/mavros/state", self._on_state, 10)
        self.create_subscription(
            ExtendedState,
            "/mavros/extended_state",
            self._on_extended_state,
            qos_profile_sensor_data,
        )
        for parameter, callback in (
            ("px4_ground_contact_topic", self._on_px4_ground_contact),
            ("px4_landed_topic", self._on_px4_landed),
            ("px4_at_rest_topic", self._on_px4_at_rest),
        ):
            topic = self._string(parameter).strip()
            if topic:
                self.create_subscription(
                    Bool, topic, callback, qos_profile_sensor_data
                )
        self.create_subscription(
            PoseStamped,
            self._string("vehicle_pose_topic"),
            self._on_pose,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TwistStamped,
            self._string("vehicle_velocity_topic"),
            self._on_velocity,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            self._string("vehicle_attitude_topic"),
            self._on_attitude,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TimesyncStatus,
            self._string("mavros_timesync_status_topic"),
            self._on_timesync_status,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            self._string("trailer_odometry_topic"),
            self._on_trailer_odometry,
            qos_profile_sensor_data,
        )
        latest_perception_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            self._string("down_marker_pose_topic"),
            self._on_down_marker_pose,
            latest_perception_qos,
        )
        self.create_subscription(
            Bool,
            self._string("down_marker_detected_topic"),
            self._on_down_marker_detected,
            latest_perception_qos,
        )
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.create_service(
            Trigger, "/autonomy/start_precision_landing", self._on_start
        )
        self.create_service(
            Trigger, "/autonomy/hold_precision_landing", self._on_hold
        )
        # Mission dynamics follow ROS/Gazebo time. Heartbeat and solver
        # scheduling use a steady wall clock so a slow simulation cannot
        # starve PX4's offboard stream or age an otherwise current solution.
        self._steady_timer_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._heartbeat_callback_group = MutuallyExclusiveCallbackGroup()
        self._solver_callback_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(
            1.0 / self._float("control_rate_hz"), self._control_tick
        )
        self.create_timer(
            1.0 / self._float("control_rate_hz"),
            self._heartbeat_tick,
            callback_group=self._heartbeat_callback_group,
            clock=self._steady_timer_clock,
        )
        self.create_timer(
            1.0 / self._float("mpc_solver_rate_hz"),
            self._solver_tick,
            callback_group=self._solver_callback_group,
            clock=self._steady_timer_clock,
        )
        self.create_timer(0.5, self._status_tick)
        # All values are structural in the one production profile.  ROS launch
        # overrides are applied before this callback is installed; changing a
        # horizon, rate, geometry or safety threshold in flight would otherwise
        # leave already-constructed solver/history objects inconsistent.
        self.add_on_set_parameters_callback(
            self._reject_runtime_parameter_changes
        )
        self.get_logger().info(
            "landing control ready inactive=%s controller=%s solver=%s "
            "heartbeat=%.1fHz worker=%.1fHz stack=%s"
            % (
                not self.start_requested,
                PRODUCTION_CONTROLLER_TYPE,
                self.landing_mpc_solver_name,
                self._float("control_rate_hz"),
                self._float("mpc_solver_rate_hz"),
                self._string("control_stack_name"),
            )
        )

    def _declare_parameters(self) -> tuple[str, ...]:
        defaults: dict[str, object] = {
            "control_stack_name": "px4_mavros_precision_landing",
            "baseline_run_id": "",
            "uses_sim_ground_truth_control_input": False,
            "validation_mode": False,
            "production_qualified": False,
            "landing_mpc_absolute_tolerance": 2.0e-4,
            "landing_mpc_relative_tolerance": 2.0e-4,
            "mpc_solver_rate_hz": 50.0,
            # The published Relative-MPC state is at t + 200 ms. A
            # longer cache lifetime would replay a trajectory point after its
            # planned time and recreate the moving-target brake/catch-up lag.
            "mpc_solution_max_age_ms": 80.0,
            "mpc_max_vehicle_position_change_m": 0.30,
            "mpc_max_vehicle_velocity_change_m_s": 0.50,
            "mpc_max_vehicle_acceleration_change_m_s2": 1.0,
            "mpc_max_target_position_change_m": 0.90,
            "mpc_max_target_velocity_change_m_s": 0.50,
            "mpc_max_target_acceleration_change_m_s2": 1.0,
            "mpc_max_target_yaw_change_rad": 0.2617993877991494,
            "landing_p_horizontal_gain": 0.8,
            "landing_p_vertical_gain": 0.8,
            "landing_p_horizontal_velocity_limit_m_s": 1.0,
            "landing_p_vertical_velocity_limit_m_s": 0.5,
            "relative_landing_mpc_horizon_steps": 20,
            "relative_landing_mpc_dt_s": 0.10,
            "relative_landing_mpc_max_horizontal_velocity_m_s": 11.0,
            "relative_landing_mpc_max_ascent_velocity_m_s": 2.0,
            "relative_landing_mpc_max_descent_velocity_m_s": 0.7,
            "relative_landing_mpc_max_horizontal_acceleration_m_s2": 3.0,
            "relative_landing_mpc_max_vertical_acceleration_m_s2": 3.0,
            "relative_landing_mpc_max_jerk_m_s3": 5.0,
            "relative_landing_mpc_funnel_minimum_radius_m": 1.75,
            "relative_landing_mpc_funnel_slope": 0.75,
            "relative_landing_mpc_camera_horizontal_fov_rad": 1.396,
            "relative_landing_mpc_camera_vertical_fov_rad": (
                1.1231652391
            ),
            "relative_landing_mpc_camera_fov_margin": 0.85,
            "relative_landing_mpc_landing_pad_length_m": 5.0,
            "relative_landing_mpc_landing_pad_width_m": 5.0,
            "relative_landing_mpc_landing_pad_contact_margin_m": 0.75,
            "relative_landing_mpc_alignment_position_tolerance_m": 0.75,
            "relative_landing_mpc_alignment_velocity_tolerance_m_s": 0.35,
            "relative_landing_mpc_max_pad_acceleration_m_s2": 2.0,
            "down_marker_pose_topic": (
                "/perception/down/marker_pose_covariance"
            ),
            "down_marker_detected_topic": (
                "/perception/down/aruco_detected"
            ),
            "marker_detection_status_timeout_s": 0.50,
            "trailer_odometry_topic": "/trailer/odometry",
            "vehicle_pose_topic": "/mavros/local_position/pose",
            "vehicle_velocity_topic": "/mavros/local_position/velocity_local",
            "vehicle_attitude_topic": "/mavros/imu/data",
            "mavros_timesync_status_topic": "/mavros/timesync_status",
            "mavros_state_timestamp_source": "timesync_status",
            "px4_ground_contact_topic": "",
            "px4_landed_topic": "",
            "px4_at_rest_topic": "",
            "auto_start": False,
            "control_rate_hz": 50.0,
            "pose_timeout_s": 1.0,
            "velocity_timeout_s": 1.0,
            "trailer_timeout_s": 1.5,
            "target_sensors_loss_abort_dwell_s": 1.0,
            "marker_timeout_s": 1.0,
            "maximum_marker_world_tilt_rad": math.radians(20.0),
            "target_fusion_reorder_window_s": 0.30,
            "target_fusion_maximum_measurement_age_s": 0.75,
            "target_fusion_future_tolerance_s": 0.05,
            "target_fusion_predict_only_timeout_s": 1.50,
            "target_fusion_process_acceleration_variance": 0.25,
            "target_fusion_process_yaw_acceleration_variance": 0.25,
            "target_fusion_vision_position_variance_floor": 1.0e-2,
            "target_fusion_vision_yaw_variance_floor": (
                math.radians(2.0) ** 2
            ),
            "target_fusion_odometry_position_variance_floor": 2.5e-3,
            "target_fusion_odometry_yaw_variance_floor": (
                math.radians(3.0) ** 2
            ),
            "target_fusion_initial_velocity_variance": 25.0,
            "target_fusion_initial_yaw_rate_variance": 4.0,
            "vision_velocity_minimum_accepted_samples": 3,
            "vision_velocity_maximum_variance_m2_s2": 1.0,
            "vision_motion_process_acceleration_variance": 1.0,
            "vision_velocity_dropout_hold_s": 3.0,
            "vision_velocity_terminal_hold_s": 20.0,
            "vision_velocity_reacquisition_gap_s": 3.0,
            "position_only_pursuit_max_speed_m_s": 10.5,
            "position_only_pursuit_acceleration_m_s2": 6.0,
            "position_only_pursuit_gain_s_inv": 3.0,
            "target_fusion_vision_nis_gate": 18.47,
            "target_fusion_odometry_nis_gate": 26.12,
            "target_fusion_maximum_position_innovation_m": 20.0,
            "trailer_marker_offset_x_m": 0.0,
            "trailer_marker_offset_y_m": 0.0,
            "trailer_marker_offset_z_m": 2.051,
            "state_history_duration_s": 3.0,
            "state_history_max_interpolation_gap_s": 0.25,
            "state_history_max_capture_age_s": 0.75,
            "state_history_max_state_age_s": 0.50,
            "state_history_future_tolerance_s": 0.10,
            "camera_time_offset_smoothing_factor": 0.10,
            "camera_time_offset_minimum_samples": 5,
            "camera_time_offset_median_window_size": 101,
            "camera_time_offset_reset_threshold_s": 2.0,
            "mavros_timesync_minimum_samples": 5,
            "mavros_timesync_max_round_trip_time_ms": 50.0,
            "mavros_timesync_timeout_s": 0.50,
            "time_reset_threshold_s": 0.50,
            "offboard_prestream_s": 2.0,
            "takeoff_height_m": 6.0,
            "takeoff_tolerance_m": 0.4,
            "marker_search_height_m": 11.0,
            "precision_align_height_m": 3.0,
            "final_approach_height_m": 5.2,
            "terminal_occlusion_max_clearance_m": 5.2,
            "precision_descent_speed_m_s": 0.70,
            "final_descent_speed_m_s": 0.55,
            "final_descent_braking_acceleration_m_s2": 0.05,
            "precision_lateral_gate_m": 0.75,
            "marker_track_entry_lateral_gate_m": 2.50,
            "marker_track_entry_relative_speed_m_s": 1.00,
            "marker_track_entry_yaw_error_rad": math.radians(10.0),
            "approach_capture_radius_m": 2.5,
            "relative_speed_gate_m_s": 0.35,
            "precision_alignment_dwell_s": 1.0,
            "precision_descent_gate_dwell_s": 0.2,
            "descent_max_vision_age_s": 1.0,
            "descent_max_position_variance_m2": 0.09,
            "descent_max_relative_height_m": 20.0,
            "max_consecutive_solver_failures": 10,
            "marker_loss_high_altitude_m": 2.0,
            "marker_loss_low_altitude_m": 0.80,
            "low_altitude_marker_reacquire_timeout_s": 1.0,
            "abort_climb_height_m": 3.0,
            "abort_climb_tolerance_m": 0.30,
            "abort_climb_timeout_s": 10.0,
            "final_approach_timeout_s": 30.0,
            "touchdown_contact_clearance_m": 0.22,
            "touchdown_contact_settle_speed_m_s": 0.12,
            "touchdown_contact_evidence_clearance_m": 0.23,
            "touchdown_contact_entry_max_relative_vertical_speed_m_s": 0.35,
            "landing_height_dropout_grace_s": 2.0,
            "touchdown_contact_compression_speed_m_s": 0.50,
            "touchdown_contact_compression_ramp_rate_m_s2": 0.25,
            "touchdown_contact_settle_timeout_s": 15.0,
            "touchdown_contact_height_rebound_tolerance_m": 0.05,
            "touchdown_contact_latch_exit_distance_m": 0.50,
            "touchdown_contact_latch_exit_dwell_s": 0.70,
            "touchdown_max_deck_distance_m": 0.25,
            "touchdown_max_relative_vertical_speed_m_s": 0.18,
            "touchdown_dwell_s": 0.60,
            "touchdown_confirmation_timeout_s": 15.0,
            "touchdown_minimum_consecutive_samples": 2,
            "land_detector_sample_timeout_s": 1.50,
            "land_detector_future_tolerance_s": 0.02,
            "land_detector_max_sample_gap_s": 1.50,
            "landing_mpc_max_iterations": 400,
            "landing_mpc_deadline_ms": 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        return tuple(defaults)

    def _float(self, name: str) -> float:
        return self.config.float(name)

    def _int(self, name: str) -> int:
        return self.config.int(name)

    def _bool(self, name: str) -> bool:
        return self.config.bool(name)

    def _string(self, name: str) -> str:
        return self.config.string(name)

    @staticmethod
    def _reject_runtime_parameter_changes(_parameters: object) -> SetParametersResult:
        return SetParametersResult(
            successful=False,
            reason="precision landing parameters are startup-only; restart the node",
        )

    @property
    def command_cache(self) -> LandingControlCommand | None:
        """Return the command shared with the isolated heartbeat callback."""
        return self._command_cache_store.load()

    @command_cache.setter
    def command_cache(self, command: LandingControlCommand | None) -> None:
        if command is None:
            self._command_cache_store.clear()
        else:
            self._command_cache_store.store(command)

    def _fusion_parameters(self) -> FusionParameters:
        return FusionParameters(
            reorder_window_s=self._float(
                "target_fusion_reorder_window_s"
            ),
            maximum_measurement_age_s=self._float(
                "target_fusion_maximum_measurement_age_s"
            ),
            future_tolerance_s=self._float(
                "target_fusion_future_tolerance_s"
            ),
            vision_timeout_s=self._float("marker_timeout_s"),
            odometry_timeout_s=self._float("trailer_timeout_s"),
            predict_only_timeout_s=self._float(
                "target_fusion_predict_only_timeout_s"
            ),
            process_acceleration_variance=self._float(
                "target_fusion_process_acceleration_variance"
            ),
            process_yaw_acceleration_variance=self._float(
                "target_fusion_process_yaw_acceleration_variance"
            ),
            vision_position_variance_floor=self._float(
                "target_fusion_vision_position_variance_floor"
            ),
            vision_yaw_variance_floor=self._float(
                "target_fusion_vision_yaw_variance_floor"
            ),
            odometry_position_variance_floor=self._float(
                "target_fusion_odometry_position_variance_floor"
            ),
            odometry_yaw_variance_floor=self._float(
                "target_fusion_odometry_yaw_variance_floor"
            ),
            initial_velocity_variance=self._float(
                "target_fusion_initial_velocity_variance"
            ),
            initial_yaw_rate_variance=self._float(
                "target_fusion_initial_yaw_rate_variance"
            ),
            vision_nis_gate=self._float("target_fusion_vision_nis_gate"),
            odometry_nis_gate=self._float(
                "target_fusion_odometry_nis_gate"
            ),
            maximum_position_innovation_m=self._float(
                "target_fusion_maximum_position_innovation_m"
            ),
        )

    def _on_state(self, message: State) -> None:
        self.mavros_state = message

    def _on_extended_state(self, message: ExtendedState) -> None:
        self.extended_state = message
        self.extended_state_stamp = time.monotonic()

    def _on_px4_ground_contact(self, message: Bool) -> None:
        """Store an optional bridged PX4 land-detector ground bit."""
        self.px4_ground_contact_override = bool(message.data)
        self.px4_ground_contact_override_stamp = time.monotonic()

    def _on_px4_landed(self, message: Bool) -> None:
        """Store an optional bridged PX4 land-detector landed bit."""
        self.px4_landed_override = bool(message.data)
        self.px4_landed_override_stamp = time.monotonic()

    def _on_px4_at_rest(self, message: Bool) -> None:
        """Store an optional bridged PX4 land-detector at-rest bit."""
        self.px4_at_rest_override = bool(message.data)
        self.px4_at_rest_override_stamp = time.monotonic()

    def _heartbeat_tick(self) -> None:
        """Continuously publish only the latest validated command cache.

        Result collection, validation and cache expiry belong to the solver
        timer.  Keeping those operations out of this callback prevents OSQP
        rejection logging or fallback construction from delaying the PX4
        offboard stream.
        """
        now = time.monotonic()
        self._heartbeat_timestamps.append(now)
        command = self.command_cache
        if command is not None and self._offboard_setpoint_active():
            self._publish_setpoint(command)

    @staticmethod
    def _measured_rate_hz(
        timestamps: deque[float], now: float
    ) -> float:
        """Return a rolling two-second event rate."""
        while timestamps and now - timestamps[0] > 2.0:
            timestamps.popleft()
        if len(timestamps) < 2:
            return 0.0
        elapsed = timestamps[-1] - timestamps[0]
        return 0.0 if elapsed <= 0.0 else (len(timestamps) - 1) / elapsed

    def _publish_setpoint(
        self, command: LandingControlCommand
    ) -> None:
        """Encode one fresh cached command on the sole MAVROS publisher."""
        position = command.position_setpoint_enu_m
        velocity = command.velocity_setpoint_enu_m_s
        acceleration = command.acceleration_setpoint_enu_m_s2
        message = PositionTarget()
        message.header.stamp = self.get_clock().now().to_msg()
        message.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        message.type_mask = PositionTarget.IGNORE_YAW_RATE
        if not command.position_enabled:
            message.type_mask |= (
                PositionTarget.IGNORE_PX
                | PositionTarget.IGNORE_PY
                | PositionTarget.IGNORE_PZ
            )
        if not command.velocity_enabled:
            message.type_mask |= (
                PositionTarget.IGNORE_VX
                | PositionTarget.IGNORE_VY
                | PositionTarget.IGNORE_VZ
            )
        acceleration_ignore_bits = (
            PositionTarget.IGNORE_AFX,
            PositionTarget.IGNORE_AFY,
            PositionTarget.IGNORE_AFZ,
        )
        for enabled, ignore_bit in zip(
            command.acceleration_enabled_axes,
            acceleration_ignore_bits,
        ):
            if not enabled:
                message.type_mask |= ignore_bit
        message.position.x, message.position.y, message.position.z = position
        message.velocity.x, message.velocity.y, message.velocity.z = velocity
        message.acceleration_or_force.x = acceleration[0]
        message.acceleration_or_force.y = acceleration[1]
        message.acceleration_or_force.z = acceleration[2]
        message.yaw = command.yaw_enu_rad
        self.setpoint_pub.publish(message)
        published_at = time.monotonic()
        self._setpoint_timestamps.append(published_at)
        # Disabled MAVROS acceleration axes are represented by NaN in the
        # command object.  Preserve enabled XY feed-forward for the vehicle
        # response observer while mapping disabled axes to a neutral zero;
        # rejecting the whole vector because AFZ is disabled silently removed
        # every horizontal command from the actuator-lag model.
        observer_acceleration = np.zeros(3, dtype=float)
        for axis, enabled in enumerate(
            command.acceleration_enabled_axes
        ):
            if enabled:
                observer_acceleration[axis] = acceleration[axis]
        self.last_command_acceleration_enu = observer_acceleration

    def _request_mode(self, mode: str, now: float) -> bool:
        if now - self.last_mode_request < 1.0:
            return False
        self.last_mode_request = now
        request = SetMode.Request()
        request.custom_mode = str(mode)
        self.mode_client.call_async(request)
        return True

    def _request_arm(self, arm: bool, now: float) -> bool:
        """Send one normal arm/disarm request when its retry window opens."""
        if now - self.last_arm_request < 1.0:
            return False
        self.last_arm_request = now
        request = CommandBool.Request()
        request.value = bool(arm)
        self.arm_client.call_async(request)
        return True

    def _status_tick(self) -> None:
        """Publish the small operator/supervisor contract only.

        Detailed estimator and solver internals stay in their owning modules;
        serialising hundreds of mostly write-only fields here needlessly kept
        the state callback group busy and made camera/control latency harder
        to diagnose.
        """
        now = time.monotonic()
        state = self.mavros_state
        target = self._current_target_estimate()
        command = self.command_cache
        worker = self.solver_worker
        fusion_statistics = self.target_estimator.statistics
        vision_statistics = fusion_statistics["vision_down"]
        odometry_statistics = fusion_statistics["odometry"]

        queue_age_s = None if worker is None else worker.queue_age_s
        if queue_age_s is None and self._pending_solver_snapshot is not None:
            queue_age_s = max(
                0.0,
                now - self._pending_solver_snapshot.created_monotonic_s,
            )
        solution_age_s = (
            None
            if self._cached_solver_snapshot is None
            else max(
                0.0,
                now - self._cached_solver_snapshot.created_monotonic_s,
            )
        )
        ground_contact, px4_landed, px4_at_rest, _, detector_source = (
            self._px4_land_detector_signals(now)
        )
        descent = self.last_descent_permission
        touchdown = self.last_touchdown_decision

        def accepted_age(statistics: object) -> float | None:
            stamp = statistics.last_accepted_stamp_s
            if target is None or stamp is None:
                return None
            return max(0.0, target.stamp_s - stamp)

        def compact_source(statistics: object) -> dict[str, object]:
            return {
                "accepted": statistics.accepted,
                "rejected": statistics.rejected,
                "duplicate": statistics.duplicate,
                "nis_rejected": statistics.nis,
                "age_s": accepted_age(statistics),
                "last_rejection_reason": statistics.last_rejection_reason,
                "last_nis": statistics.last_nis,
                "last_position_innovation_enu_m": (
                    statistics.last_position_innovation_enu_m
                ),
            }

        fields = {
            "schema_version": 2,
            "control_stack": self._string("control_stack_name"),
            "baseline_run_id": self._string("baseline_run_id"),
            "validation_mode": self._bool("validation_mode"),
            "production_qualified": self._bool("production_qualified"),
            "activation_qualified": self._activation_qualified(),
            "landing_controller_type": PRODUCTION_CONTROLLER_TYPE,
            "landing_mpc_solver_effective": PRODUCTION_SOLVER_NAME,
            "uses_sim_ground_truth_control_input": self._bool(
                "uses_sim_ground_truth_control_input"
            ),
            "phase": self.phase.value,
            "last_transition_reason": self.last_transition_reason,
            "active": self.start_requested,
            "connected": None if state is None else bool(state.connected),
            "armed": None if state is None else bool(state.armed),
            "flight_mode": None if state is None else str(state.mode),
            "target_estimate_valid": (
                False if target is None else target.valid
            ),
            "target_estimate_mode": (
                "time_unavailable" if target is None else target.mode
            ),
            "target_estimate_position_enu_m": (
                None if target is None else target.position_enu_m
            ),
            "target_estimate_covariance_diagonal": (
                None
                if target is None or target.covariance is None
                else np.diag(target.covariance)
            ),
            "target_vision_fresh": (
                False if target is None else target.vision_fresh
            ),
            "target_odometry_fresh": (
                False if target is None else target.odometry_fresh
            ),
            "target_sources": {
                "vision_down": compact_source(vision_statistics),
                "odometry": compact_source(odometry_statistics),
            },
            "down_marker_capture_stamp_s": (
                self.down_marker_capture_guard.capture_stamp_s
            ),
            "down_marker_duplicate_count": (
                self.down_marker_capture_guard.duplicate_count
                + vision_statistics.duplicate
            ),
            "down_marker_invalid_covariance_count": (
                self.down_marker_invalid_covariance_count
                + vision_statistics.invalid_covariance
            ),
            "vision_motion_model_qualified": (
                self.vision_position_model_qualified
            ),
            "vision_motion_model_span_s": self.vision_motion_model_span_s,
            "control_target_velocity_source": (
                self.control_target_velocity_source
            ),
            "trailer_motion_predictor_status": (
                self.last_trailer_prediction_reason
            ),
            "vehicle_response_estimator_status": (
                self.last_vehicle_response_reason
            ),
            "landing_horizontal_error_m": (
                self.last_landing_horizontal_error_m
            ),
            "landing_relative_horizontal_speed_m_s": (
                self.last_landing_relative_horizontal_speed_m_s
            ),
            "landing_height_m": self.landing_height_distance_m,
            "landing_height_fresh": self._fresh_landing_height(now),
            "landing_mpc": self.last_mpc_message,
            "landing_mpc_success": self.last_mpc_success,
            "landing_mpc_solve_ms": 1000.0 * self.last_mpc_solve_time_s,
            "solver_running": bool(
                worker is not None and worker.solver_running
            ),
            "solver_queue_age_ms": (
                None if queue_age_s is None else 1000.0 * queue_age_s
            ),
            "solution_age_ms": (
                None if solution_age_s is None else 1000.0 * solution_age_s
            ),
            "consecutive_solver_failures": (
                self.consecutive_solver_failures
            ),
            "stale_solution_rejected_count": (
                self.stale_solution_rejected_count
            ),
            "command_extraction_policy": (
                None if command is None else command.extraction_policy
            ),
            "command_acceleration_enabled_axes": (
                None
                if command is None
                else list(command.acceleration_enabled_axes)
            ),
            "descent_permission_allowed": (
                None if descent is None else descent.allowed
            ),
            "descent_permission_reason": (
                None if descent is None else descent.reason
            ),
            "last_safety_failure_reason": (
                self.last_safety_failure_reason
            ),
            "marker_loss_policy": self.last_marker_loss_policy,
            "land_detector_source": detector_source,
            "px4_ground_contact": ground_contact,
            "px4_landed": px4_landed,
            "px4_at_rest": px4_at_rest,
            "touchdown_confirmed": (
                False if touchdown is None else touchdown.confirmed
            ),
            "touchdown_disarm_requested": (
                self.touchdown_disarm_requested
            ),
            "heartbeat_rate_hz": self._measured_rate_hz(
                self._heartbeat_timestamps, now
            ),
            "setpoint_rate_hz": self._measured_rate_hz(
                self._setpoint_timestamps, now
            ),
        }
        message = String()
        message.data = dumps_strict_json(sanitize_json_value(fields))
        self.status_pub.publish(message)

    def destroy_node(self) -> None:
        """Join the solver worker before destroying ROS entities."""
        if self.solver_worker is not None:
            self.solver_worker.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MpcPrecisionLandingNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # ROS 2 Humble can invalidate the context between executor waits when
        # launch sends SIGINT.  Suppress only that shutdown race; live-context
        # runtime errors must still surface.
        if rclpy.ok():
            raise
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
