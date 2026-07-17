#!/usr/bin/env python3
"""MPC ArUco landing controller for PX4 through MAVROS.

This is a separate control stack from ``precision_landing_node.py``.  The
legacy P/feed-forward node remains a frozen comparison baseline.  A launcher
process must hold the repository control-authority lock before starting either
stack, because both publish ``/mavros/setpoint_raw/local``.

Stage 3 keeps a bounded PX4-time history sourced through MAVROS.  ArUco pose,
yaw, and covariance are transformed with the position and attitude
interpolated to the image capture stamp.  Stage 4 fuses those captures with
marker-offset-corrected trailer odometry in timestamp order; control reads a
non-mutating current-time prediction.  Stage 5 routes the unchanged mission
reference through a ROS-independent selectable landing controller and caches
its control-mode flags before the sole MAVROS heartbeat publisher.  Stage 6
adds a separate moving-pad relative-coordinate MPC while preserving the
legacy absolute MPC for takeoff and comparison.  Stage 7 selects a persistent
sparse OSQP workspace for the real-time relative MPC while retaining SLSQP as
an offline reference.  Stage 8 runs the selected controller on one background
worker from immutable snapshots, while the sole MAVROS heartbeat publisher
reuses only validated cached commands.  Stage 9 gates every downward step on
fused-target confidence and confirms touchdown from continuous geometric,
kinematic, and PX4 land-detector evidence before a normal disarm request.
Solver deadlines and data-freshness checks remain on ``time.monotonic()``;
mission phase elapsed time follows the ROS clock in simulation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from enum import Enum
import math
import os
import time
from typing import Sequence

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
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
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger

from .async_landing_control import (
    AsyncLandingControllerWorker,
    LandingCommandLimits,
    LandingSolveResult,
    LandingSolveSnapshot,
    SolutionAcceptanceLimits,
    is_stale_solution_rejection,
    landing_snapshot_change_rejection,
    validate_landing_solve_result,
)
from .capture_state_history import (
    MavrosTimesyncTracker,
    StateInterpolationError,
    StateTimeReset,
    TimestampOffsetEstimator,
    VehicleStateHistory,
    marker_world_pose_at_capture,
    mavros_header_to_px4_time_s,
    quaternion_world_z_tilt_rad,
)
from .marker_measurement import (
    CaptureStampGuard,
    build_estimator_covariance,
    extract_pose_covariance_6x6,
)
from .moving_trailer import MovingTrailerBridge
from .landing_controller import (
    HOLD_FALLBACK,
    LEGACY_MPC_CONTROLLER_TYPE,
    P_FEEDFORWARD_CONTROLLER_TYPE,
    P_FEEDFORWARD_FALLBACK,
    RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
    LandingControlCommand,
    LandingControlInput,
    PFeedforwardLandingController,
    create_landing_controller,
)
from .landing_safety import (
    DescentPermissionDecision,
    DescentPermissionInput,
    DescentPermissionLimits,
    TouchdownGate,
    TouchdownGateDecision,
    TouchdownGateInput,
    TouchdownGateLimits,
    evaluate_descent_permission,
)
from .mpc_core import (
    JerkInputMPC,
    TrailerKinematicState,
)
from .osqp_landing_solver import OsqpLandingSolver, SlsqpReferenceSolver
from .status_payload import dumps_strict_json, sanitize_json_value
from .target_fusion import (
    FusionParameters,
    TargetEstimate,
    TimestampedTargetEstimator,
    validate_covariance,
)


class LandingPhase(Enum):
    WAITING = "waiting_for_px4_and_sensors"
    READY = "ready_inactive"
    PRESTREAM = "offboard_setpoint_prestream"
    ARMING = "arming_and_entering_offboard"
    TAKEOFF = "vertical_takeoff"
    APPROACH = "stationary_trailer_approach"
    MARKER_TRACK_DOWN = "marker_track_down"
    PRECISION_ALIGN = "precision_align"
    PRECISION_DESCENT = "precision_descent"
    FINAL_APPROACH = "final_approach"
    TOUCHDOWN_CONFIRM = "touchdown_confirm"
    LANDED = "landed"
    ABORT_CLIMB = "abort_climb"
    FAILSAFE_HOLD = "failsafe_hold"


ACQUISITION_GUIDANCE_CONTROLLER_TYPE = "aruco_acquisition_guidance"


class MpcPrecisionLandingNode(Node):
    """Single-publisher MPC controller for the trailer experiment."""

    def __init__(self) -> None:
        super().__init__("mpc_precision_landing")
        self._declare_parameters()
        self.mavros_state_timestamp_source = str(
            self.get_parameter("mavros_state_timestamp_source").value
        ).strip()
        if self.mavros_state_timestamp_source not in {
            "timesync_status",
            "ros_header",
        }:
            raise ValueError(
                "mavros_state_timestamp_source must be timesync_status or "
                "ros_header"
            )
        self._mission_uses_ros_time = bool(
            self.get_parameter("use_sim_time").value
        )
        self.mpc = JerkInputMPC(
            dt_s=self._float("landing_mpc_dt_s"),
            horizon_steps=self._int("landing_mpc_horizon_steps"),
            velocity_limit_m_s=self._float("max_velocity_m_s"),
            acceleration_limit_m_s2=self._float("max_acceleration_m_s2"),
            jerk_limit_m_s3=self._float("max_jerk_m_s3"),
            deadline_s=0.001 * self._float("legacy_mpc_deadline_ms"),
            maximum_iterations=self._int("legacy_mpc_max_iterations"),
        )
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
        self.landing_mpc_solver_name = str(
            self.get_parameter("landing_mpc_solver").value
        )
        if self.landing_mpc_solver_name == "osqp":
            self.relative_landing_mpc = OsqpLandingSolver(
                **relative_solver_parameters,
                deadline_s=0.001
                * self._float("landing_mpc_deadline_ms"),
                maximum_iterations=self._int(
                    "landing_mpc_max_iterations"
                ),
                absolute_tolerance=self._float(
                    "landing_mpc_absolute_tolerance"
                ),
                relative_tolerance=self._float(
                    "landing_mpc_relative_tolerance"
                ),
            )
        elif self.landing_mpc_solver_name == "slsqp":
            self.relative_landing_mpc = SlsqpReferenceSolver(
                **relative_solver_parameters,
                deadline_s=0.001
                * self._float("relative_landing_mpc_deadline_ms"),
                maximum_iterations=self._int(
                    "relative_landing_mpc_max_iterations"
                ),
            )
        else:
            raise ValueError("landing_mpc_solver must be osqp or slsqp")
        self.landing_controller = create_landing_controller(
            str(self.get_parameter("landing_controller_type").value),
            mpc_solver=self.mpc,
            mpc_failure_fallback=str(
                self.get_parameter("mpc_failure_fallback").value
            ),
            p_horizontal_gain=self._float("landing_p_horizontal_gain"),
            p_vertical_gain=self._float("landing_p_vertical_gain"),
            relative_mpc_solver=self.relative_landing_mpc,
        )
        self.async_p_fallback_controller = PFeedforwardLandingController(
            self._float("landing_p_horizontal_gain"),
            self._float("landing_p_vertical_gain"),
        )
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
            maximum_lidar_age_s=self._float("descent_lidar_timeout_s"),
            maximum_lidar_distance_m=self._float(
                "descent_max_lidar_distance_m"
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
        contact_settle_speed = self._float(
            "touchdown_contact_settle_speed_m_s"
        )
        contact_settle_timeout = self._float(
            "touchdown_contact_settle_timeout_s"
        )
        contact_evidence_lidar = self._float(
            "touchdown_contact_evidence_lidar_m"
        )
        contact_compression_speed = self._float(
            "touchdown_contact_compression_speed_m_s"
        )
        contact_compression_ramp_rate = self._float(
            "touchdown_contact_compression_ramp_rate_m_s2"
        )
        contact_lidar_rebound_tolerance = self._float(
            "touchdown_contact_lidar_rebound_tolerance_m"
        )
        auto_land_timeout = self._float("touchdown_auto_land_timeout_s")
        auto_land_mode = str(
            self.get_parameter("touchdown_auto_land_mode").value
        ).strip()
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
            or not math.isfinite(contact_lidar_rebound_tolerance)
            or contact_lidar_rebound_tolerance < 0.0
            or not math.isfinite(contact_clearance)
            or contact_clearance < 0.0
            or not math.isfinite(contact_deck_distance)
            or contact_deck_distance <= 0.0
            or not math.isfinite(contact_relative_speed)
            or contact_relative_speed <= 0.0
            or contact_settle_speed
            > self._float("landing_p_vertical_velocity_limit_m_s")
            or contact_settle_speed > contact_relative_speed
            or not math.isfinite(contact_evidence_lidar)
            or contact_evidence_lidar
            < self._float("descent_min_lidar_distance_m")
            or contact_evidence_lidar > contact_deck_distance
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
            or contact_lidar_rebound_tolerance > contact_deck_distance
            or not math.isfinite(auto_land_timeout)
            or auto_land_timeout <= 0.0
            or auto_land_timeout > self._float("final_approach_timeout_s")
            or auto_land_mode != "AUTO.LAND"
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
        terminal_occlusion_lidar = self._float(
            "terminal_occlusion_max_lidar_m"
        )
        if (
            not 0.0 < terminal_occlusion_clearance
            < self._float("final_approach_height_m")
            or not 0.0 < terminal_occlusion_lidar
            < terminal_occlusion_clearance
        ):
            raise ValueError(
                "terminal optical-occlusion bounds must be positive, below "
                "final approach height, and lidar must be below clearance"
            )
        land_detector_topics = tuple(
            str(self.get_parameter(name).value).strip()
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
        self.descent_clearance_m = self._float("precision_align_height_m")
        self.failsafe_hold_position_enu: np.ndarray | None = None
        self.abort_climb_target_enu: np.ndarray | None = None
        self.last_descent_permission: DescentPermissionDecision | None = None
        self.last_touchdown_decision: TouchdownGateDecision | None = None
        self.last_land_detector_source = "unavailable"
        self.touchdown_disarm_requested = False
        self.normal_disarm_request_count = 0
        self.last_safety_failure_reason: str | None = None
        self.last_marker_loss_policy: str | None = None
        self.marker_loss_started_mission_s: float | None = None
        self.terminal_recovery_started_mission_s: float | None = None
        self.last_mode_request = 0.0
        self.last_arm_request = 0.0
        self.last_land_request = 0.0
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
        self.vehicle_quaternion_xyzw: np.ndarray | None = None
        self.vehicle_yaw_enu_rad = 0.0
        self.pose_stamp = 0.0
        self.velocity_stamp = 0.0
        self._previous_velocity: np.ndarray | None = None
        self._previous_velocity_stamp = 0.0
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
        self.last_camera_time_offset_residual_s: float | None = None
        self.time_reset_count = 0
        self.last_time_reset_reason: str | None = None
        self.down_lidar_range_m: float | None = None
        self.down_lidar_stamp = 0.0

        self.trailer_state: TrailerKinematicState | None = None
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
            fusion_parameters
        )
        self.last_target_estimate: TargetEstimate | None = None
        self.last_vision_motion_estimate: TargetEstimate | None = None
        self.last_vision_velocity_enu: np.ndarray | None = None
        self.last_vision_velocity_stamp_px4_s: float | None = None
        self.last_vision_velocity_source_stamp_px4_s: float | None = None
        self.last_vision_motion_capture_px4_time_s: float | None = None
        self.vision_velocity_history: deque[tuple[float, np.ndarray]] = (
            deque()
        )
        self.vision_motion_model_acceleration_enu = np.zeros(3, dtype=float)
        self.vision_motion_model_span_s = 0.0
        self.last_position_servo_velocity_enu = np.zeros(3, dtype=float)
        self.last_position_servo_update_monotonic_s: float | None = None
        self.marker_tracking_acquisition_since_s: float | None = None
        self.marker_tracking_latched = False
        self.control_target_velocity_source = "unavailable"
        self.front_marker_capture_guard = CaptureStampGuard()
        self.down_marker_capture_guard = CaptureStampGuard()
        self.front_marker_stamp = 0.0
        self.down_marker_stamp = 0.0
        self.down_marker_detected = False
        self.down_marker_detected_stamp = 0.0
        self.down_marker_positive_stamp = 0.0
        self.front_marker_sequence = 0
        self.down_marker_sequence = 0
        self.front_marker_quality: float | None = None
        self.down_marker_quality: float | None = None
        self.front_marker_quality_stamp = 0.0
        self.down_marker_quality_stamp = 0.0
        self.front_marker_covariance_diagonal: np.ndarray | None = None
        self.down_marker_covariance_diagonal: np.ndarray | None = None
        self.front_marker_invalid_covariance_count = 0
        self.down_marker_invalid_covariance_count = 0
        self.front_marker_timing_rejection_count = 0
        self.down_marker_timing_rejection_count = 0
        self.front_marker_last_timing_rejection: str | None = None
        self.down_marker_last_timing_rejection: str | None = None
        self.front_marker_world_tilt_rad: float | None = None
        self.down_marker_world_tilt_rad: float | None = None
        self.front_marker_tilt_rejection_count = 0
        self.down_marker_tilt_rejection_count = 0
        self._pending_marker_messages: dict[
            str, tuple[int, PoseWithCovarianceStamped]
        ] = {}
        self.front_marker_deferred_count = 0
        self.down_marker_deferred_count = 0
        self.front_marker_capture_px4_time_s: float | None = None
        self.down_marker_capture_px4_time_s: float | None = None
        self.front_marker_capture_delay_s: float | None = None
        self.down_marker_capture_delay_s: float | None = None
        self.front_marker_capture_vehicle_position_enu: (
            np.ndarray | None
        ) = None
        self.down_marker_capture_vehicle_position_enu: (
            np.ndarray | None
        ) = None
        self.front_marker_capture_vehicle_velocity_enu: (
            np.ndarray | None
        ) = None
        self.down_marker_capture_vehicle_velocity_enu: (
            np.ndarray | None
        ) = None

        self.last_reference_enu: np.ndarray | None = None
        self.last_command_position_enu: np.ndarray | None = None
        self.last_command_velocity_enu: np.ndarray | None = None
        self.last_command_acceleration_enu: np.ndarray | None = None
        self.last_command_yaw_enu_rad: float | None = None
        self.last_command_stamp = 0.0
        self.command_cache: LandingControlCommand | None = None
        self._pending_command: LandingControlCommand | None = None
        self.command_generation = 0
        self.last_landing_control_status = "not_started"
        self.last_landing_control_valid = False
        self.last_landing_control_degraded = False
        self.last_landing_control_type = str(
            self.get_parameter("landing_controller_type").value
        )
        self.last_landing_control_solve_time_s = 0.0
        self.landing_control_compute_count = 0
        self.landing_control_valid_count = 0
        self.landing_control_degraded_count = 0
        self.landing_control_invalid_count = 0
        self.last_landing_horizontal_error_m: float | None = None
        self.last_landing_relative_horizontal_speed_m_s: float | None = None
        self.last_requested_clearance_m: float | None = None
        self.last_landing_pad_position_enu: np.ndarray | None = None
        self.last_landing_pad_velocity_enu: np.ndarray | None = None
        self.last_landing_pad_acceleration_enu: np.ndarray | None = None
        self.last_landing_relative_position_enu: np.ndarray | None = None
        self.last_landing_relative_velocity_enu: np.ndarray | None = None
        self.last_landing_relative_acceleration_enu: np.ndarray | None = None
        self.last_relative_mpc_constraints_enabled = False
        self.last_relative_mpc_descent_allowed = False
        self.landing_pad_acceleration_smoothing_factor = self._float(
            "relative_landing_mpc_pad_acceleration_smoothing_factor"
        )
        self.landing_pad_max_acceleration_m_s2 = self._float(
            "relative_landing_mpc_max_pad_acceleration_m_s2"
        )
        if not 0.0 < self.landing_pad_acceleration_smoothing_factor <= 1.0:
            raise ValueError(
                "pad acceleration smoothing factor must be in (0,1]"
            )
        if self.landing_pad_max_acceleration_m_s2 <= 0.0:
            raise ValueError("pad acceleration limit must be positive")
        self.landing_pad_acceleration_enu = np.zeros(3, dtype=float)
        self.landing_pad_acceleration_source = "aruco_not_initialized"
        self._previous_landing_pad_velocity: np.ndarray | None = None
        self._previous_landing_pad_velocity_stamp_s: float | None = None
        self.last_mpc_message = "not_started"
        self.last_mpc_success = False
        self.last_mpc_solve_time_s = 0.0
        self.last_mpc_iterations = 0
        self.last_mpc_deadline_missed = False
        self.mpc_solve_count = 0
        self.mpc_success_count = 0
        self.mpc_failure_count = 0
        self.mpc_deadline_miss_count = 0
        self._solver_snapshot_generation = 0
        self._last_processed_solver_generation = 0
        self._revoked_solver_generation = 0
        self._pending_solver_snapshot: LandingSolveSnapshot | None = None
        self._latest_solver_snapshot: LandingSolveSnapshot | None = None
        self._cached_solver_snapshot: LandingSolveSnapshot | None = None
        self._cached_p_command_monotonic_s: float | None = None
        self._cached_p_command_phase: str | None = None
        self._cached_p_command_time_reset_count: int | None = None
        self._acquisition_command_monotonic_s: float | None = None
        self._acquisition_command_phase: str | None = None
        self._acquisition_command_time_reset_count: int | None = None
        self._transition_command_monotonic_s: float | None = None
        self._transition_command_phase: str | None = None
        self._transition_command_time_reset_count: int | None = None
        self._contact_settle_started_monotonic_s: float | None = None
        self._contact_settle_started_time_reset_count: int | None = None
        self._contact_settle_min_lidar_range_m: float | None = None
        self._contact_entry_latched = False
        self._contact_compression_started_monotonic_s: float | None = None
        self._target_sensors_loss_started_monotonic_s: float | None = None
        self.terminal_contact_bridge_active = False
        self.terminal_contact_bridge_activation_count = 0
        self.last_terminal_contact_bridge_reason: str | None = None
        self.terminal_auto_land_requested = False
        self.terminal_auto_land_request_count = 0
        self.terminal_auto_land_engaged_monotonic_s: float | None = None
        self._terminal_auto_land_cancel_reason: str | None = None
        self._terminal_auto_land_kinematics_unsafe_since: float | None = None
        self._time_reset_hold_pending = False
        self._last_solver_result_snapshot_stamp_s: float | None = None
        self._last_solver_queue_age_s: float | None = None
        self.stale_solution_rejected_count = 0
        self.consecutive_solver_failures = 0
        self._heartbeat_timestamps: deque[float] = deque(maxlen=101)
        self._setpoint_timestamps: deque[float] = deque(maxlen=101)
        self.last_status_publish = 0.0
        configured_controller = str(
            self.get_parameter("landing_controller_type").value
        )
        self.solver_worker: AsyncLandingControllerWorker | None = None
        if configured_controller != P_FEEDFORWARD_CONTROLLER_TYPE:
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
            topic = str(self.get_parameter(parameter).value).strip()
            if topic:
                self.create_subscription(
                    Bool, topic, callback, qos_profile_sensor_data
                )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("vehicle_pose_topic").value),
            self._on_pose,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("vehicle_velocity_topic").value),
            self._on_velocity,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("vehicle_attitude_topic").value),
            self._on_attitude,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TimesyncStatus,
            str(self.get_parameter("mavros_timesync_status_topic").value),
            self._on_timesync_status,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            "/down_lidar",
            self._on_down_lidar,
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
            str(self.get_parameter("front_marker_pose_topic").value),
            self._on_front_marker_pose,
            latest_perception_qos,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter("down_marker_pose_topic").value),
            self._on_down_marker_pose,
            latest_perception_qos,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("front_marker_quality_topic").value),
            self._on_front_marker_quality,
            latest_perception_qos,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("down_marker_quality_topic").value),
            self._on_down_marker_quality,
            latest_perception_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("down_marker_detected_topic").value),
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
        self.create_timer(
            1.0 / self._float("control_rate_hz"), self._control_tick
        )
        self.create_timer(
            1.0 / self._float("control_rate_hz"),
            self._heartbeat_tick,
            clock=self._steady_timer_clock,
        )
        self.create_timer(
            1.0 / self._float("mpc_solver_rate_hz"),
            self._solver_tick,
            clock=self._steady_timer_clock,
        )
        self.create_timer(0.5, self._status_tick)
        self.trailer_bridge = MovingTrailerBridge(
            self,
            lambda: self.phase.value,
            odometry_callback=self._on_trailer_odometry,
        )
        self.get_logger().info(
            "landing control ready inactive=%s controller=%s solver=%s "
            "heartbeat=%.1fHz worker=%.1fHz stack=%s authority=%s"
            % (
                not self.start_requested,
                self.get_parameter("landing_controller_type").value,
                self.landing_mpc_solver_name,
                self._float("control_rate_hz"),
                self._float("mpc_solver_rate_hz"),
                self.get_parameter("control_stack_name").value,
                self.get_parameter("vehicle_authority_id").value,
            )
        )

    def _declare_parameters(self) -> None:
        defaults: dict[str, object] = {
            "control_stack_name": "px4_mavros_precision_landing",
            "baseline_run_id": "",
            "vehicle_authority_id": "vehicle_1",
            "uses_sim_ground_truth_control_input": False,
            "validation_mode": False,
            "production_qualified": False,
            "landing_controller_type": "p_feedforward",
            "landing_mpc_solver": "osqp",
            "landing_mpc_absolute_tolerance": 2.0e-4,
            "landing_mpc_relative_tolerance": 2.0e-4,
            "mpc_solver_rate_hz": 50.0,
            # The first published Relative-MPC state is at t + 100 ms.  A
            # longer cache lifetime would replay a trajectory point after its
            # planned time and recreate the moving-target brake/catch-up lag.
            "mpc_solution_max_age_ms": 80.0,
            "mpc_max_vehicle_position_change_m": 0.30,
            "mpc_max_vehicle_velocity_change_m_s": 0.50,
            "mpc_max_vehicle_acceleration_change_m_s2": 1.0,
            "mpc_max_target_position_change_m": 0.50,
            "mpc_max_target_velocity_change_m_s": 0.50,
            "mpc_max_target_acceleration_change_m_s2": 1.0,
            "mpc_max_target_yaw_change_rad": 0.2617993877991494,
            "mpc_failure_fallback": "hold",
            "landing_p_horizontal_gain": 0.8,
            "landing_p_vertical_gain": 0.8,
            "landing_p_horizontal_velocity_limit_m_s": 1.0,
            "landing_p_vertical_velocity_limit_m_s": 0.5,
            "relative_landing_mpc_horizon_steps": 20,
            "relative_landing_mpc_dt_s": 0.10,
            "relative_landing_mpc_max_iterations": 30,
            "relative_landing_mpc_deadline_ms": 40.0,
            "relative_landing_mpc_max_horizontal_velocity_m_s": 5.0,
            "relative_landing_mpc_max_ascent_velocity_m_s": 2.0,
            "relative_landing_mpc_max_descent_velocity_m_s": 0.7,
            "relative_landing_mpc_max_horizontal_acceleration_m_s2": 3.0,
            "relative_landing_mpc_max_vertical_acceleration_m_s2": 3.0,
            "relative_landing_mpc_max_jerk_m_s3": 5.0,
            "relative_landing_mpc_funnel_minimum_radius_m": 0.75,
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
            "relative_landing_mpc_pad_acceleration_smoothing_factor": 0.20,
            "relative_landing_mpc_max_pad_acceleration_m_s2": 3.0,
            "front_marker_pose_topic": (
                "/perception/front/marker_pose_covariance"
            ),
            "down_marker_pose_topic": (
                "/perception/down/marker_pose_covariance"
            ),
            "front_marker_quality_topic": "/perception/front/quality",
            "down_marker_quality_topic": "/perception/down/quality",
            "down_marker_detected_topic": (
                "/perception/down/aruco_detected"
            ),
            "marker_detection_status_timeout_s": 0.50,
            "trailer_odometry_topic": "/trailer/odometry",
            "trailer_gz_odometry_topic": "/model/trailer/odometry",
            "trailer_gz_command_topic": "/model/trailer/cmd_vel",
            "trailer_world_origin_x_enu_m": 30.0,
            "trailer_world_origin_y_enu_m": 30.0,
            "trailer_world_origin_z_enu_m": 0.24,
            "trailer_speed_m_s": 3.0,
            "trailer_acceleration_m_s2": 1.0,
            "trailer_waypoints_enu_m": [
                37.071067811865476,
                37.071067811865476,
                137.07106781186548,
                37.071067811865476,
                137.07106781186548,
                137.07106781186548,
                37.071067811865476,
                137.07106781186548,
            ],
            "trailer_waypoint_radius_m": 0.2,
            "trailer_corner_radius_m": 20.0,
            "trailer_state_publish_rate_hz": 50.0,
            "trailer_odometry_warmup_samples": 12,
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
            "target_fusion_vision_position_variance_floor": 6.4e-3,
            "target_fusion_vision_yaw_variance_floor": (
                math.radians(2.0) ** 2
            ),
            "target_fusion_odometry_position_variance_floor": 2.5e-3,
            "target_fusion_odometry_velocity_variance_floor": 1.0e-2,
            "target_fusion_odometry_yaw_variance_floor": (
                math.radians(3.0) ** 2
            ),
            "target_fusion_odometry_yaw_rate_variance_floor": (
                math.radians(5.0) ** 2
            ),
            "target_fusion_initial_velocity_variance": 25.0,
            "target_fusion_initial_yaw_rate_variance": 4.0,
            "vision_velocity_minimum_accepted_samples": 3,
            "vision_velocity_maximum_variance_m2_s2": 1.0,
            "vision_velocity_dropout_hold_s": 1.5,
            "vision_velocity_terminal_hold_s": 30.0,
            "vision_velocity_reacquisition_gap_s": 1.5,
            "position_only_pursuit_max_speed_m_s": 4.2,
            "position_only_pursuit_acceleration_m_s2": 2.5,
            "position_only_pursuit_gain_s_inv": 3.0,
            "target_fusion_vision_nis_gate": 18.47,
            "target_fusion_odometry_nis_gate": 26.12,
            "target_fusion_maximum_position_innovation_m": 20.0,
            "target_fusion_maximum_velocity_innovation_m_s": 10.0,
            "target_fusion_maximum_yaw_rate_rad_s": 3.0,
            "trailer_marker_offset_x_m": 0.0,
            "trailer_marker_offset_y_m": 0.0,
            "trailer_marker_offset_z_m": 2.051,
            "trailer_odometry_twist_in_body_frame": True,
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
            "marker_search_height_m": 6.0,
            "precision_align_height_m": 3.0,
            "final_approach_height_m": 3.1,
            "terminal_occlusion_max_clearance_m": 3.00,
            "terminal_occlusion_max_lidar_m": 2.90,
            "precision_descent_speed_m_s": 0.60,
            "final_descent_speed_m_s": 0.55,
            "landing_motion_lead_time_s": 0.05,
            "precision_lateral_gate_m": 0.75,
            "marker_track_entry_lateral_gate_m": 0.50,
            "marker_track_entry_relative_speed_m_s": 0.35,
            "marker_track_entry_yaw_error_rad": math.radians(10.0),
            "approach_capture_radius_m": 2.5,
            "relative_speed_gate_m_s": 0.35,
            "precision_alignment_dwell_s": 1.0,
            "precision_descent_gate_dwell_s": 0.2,
            "descent_max_vision_age_s": 1.0,
            "descent_max_position_variance_m2": 0.09,
            "descent_lidar_timeout_s": 0.50,
            "descent_min_lidar_distance_m": 0.02,
            "descent_max_lidar_distance_m": 20.0,
            "descent_max_relative_height_m": 20.0,
            "final_approach_requires_lidar": True,
            "max_consecutive_solver_failures": 10,
            "marker_loss_high_altitude_m": 2.0,
            "marker_loss_low_altitude_m": 0.80,
            "low_altitude_marker_reacquire_timeout_s": 2.0,
            "abort_climb_height_m": 3.0,
            "abort_climb_tolerance_m": 0.30,
            "abort_climb_timeout_s": 10.0,
            "final_approach_timeout_s": 18.0,
            "touchdown_contact_clearance_m": 0.22,
            "touchdown_contact_settle_speed_m_s": 0.12,
            "touchdown_contact_evidence_lidar_m": 0.04,
            "touchdown_contact_compression_speed_m_s": 0.50,
            "touchdown_contact_compression_ramp_rate_m_s2": 0.25,
            "touchdown_contact_settle_timeout_s": 15.0,
            "touchdown_contact_lidar_rebound_tolerance_m": 0.05,
            "touchdown_auto_land_mode": "AUTO.LAND",
            "touchdown_auto_land_timeout_s": 8.0,
            "touchdown_max_deck_distance_m": 0.25,
            "touchdown_max_relative_vertical_speed_m_s": 0.18,
            "touchdown_dwell_s": 1.0,
            "touchdown_confirmation_timeout_s": 15.0,
            "touchdown_minimum_consecutive_samples": 2,
            "land_detector_sample_timeout_s": 1.50,
            "land_detector_future_tolerance_s": 0.02,
            "land_detector_max_sample_gap_s": 1.50,
            "landing_mpc_horizon_steps": 20,
            "landing_mpc_dt_s": 0.10,
            "landing_mpc_max_iterations": 400,
            "landing_mpc_deadline_ms": 20.0,
            "legacy_mpc_max_iterations": 12,
            "legacy_mpc_deadline_ms": 40.0,
            "max_velocity_m_s": 5.0,
            "max_acceleration_m_s2": 3.0,
            "max_jerk_m_s3": 5.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

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
            odometry_velocity_variance_floor=self._float(
                "target_fusion_odometry_velocity_variance_floor"
            ),
            odometry_yaw_variance_floor=self._float(
                "target_fusion_odometry_yaw_variance_floor"
            ),
            odometry_yaw_rate_variance_floor=self._float(
                "target_fusion_odometry_yaw_rate_variance_floor"
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
            maximum_velocity_innovation_m_s=self._float(
                "target_fusion_maximum_velocity_innovation_m_s"
            ),
            maximum_yaw_rate_rad_s=self._float(
                "target_fusion_maximum_yaw_rate_rad_s"
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

    def _on_timesync_status(self, message: TimesyncStatus) -> None:
        if self.mavros_state_timestamp_source == "ros_header":
            return
        observation = self.mavros_timesync.observe(
            message.remote_timestamp_ns,
            message.estimated_offset_ns,
            message.round_trip_time_ms,
        )
        if observation.reset_detected:
            self._handle_time_reset(
                "MAVROS/PX4 timesync reset",
                reset_timesync=True,
            )
            observation = self.mavros_timesync.observe(
                message.remote_timestamp_ns,
                message.estimated_offset_ns,
                message.round_trip_time_ms,
            )
        if observation.accepted:
            self.mavros_timesync_stamp = time.monotonic()

    def _on_pose(self, message: PoseStamped) -> None:
        position = message.pose.position
        orientation = message.pose.orientation
        values = np.asarray(
            (position.x, position.y, position.z, orientation.x,
             orientation.y, orientation.z, orientation.w),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            return
        quaternion = values[3:].copy()
        if float(np.linalg.norm(quaternion)) < 1.0e-9:
            return
        self._record_position_history(message.header.stamp, values[:3])
        self.vehicle_position_enu = values[:3].copy()
        self.vehicle_quaternion_xyzw = quaternion
        self.vehicle_yaw_enu_rad = self._quaternion_yaw(orientation)
        self.pose_stamp = time.monotonic()
        if self._time_reset_hold_pending:
            self._cache_time_reset_hold()
        self._retry_pending_marker_measurements()

    def _on_velocity(self, message: TwistStamped) -> None:
        now = time.monotonic()
        linear = message.twist.linear
        velocity = np.asarray((linear.x, linear.y, linear.z), dtype=float)
        if not np.all(np.isfinite(velocity)):
            return
        self._record_velocity_history(message.header.stamp, velocity)
        if self._previous_velocity is not None:
            dt = now - self._previous_velocity_stamp
            if 0.01 <= dt <= 0.5:
                measured = (velocity - self._previous_velocity) / dt
                limits = np.asarray(
                    (
                        self._float(
                            "relative_landing_mpc_max_horizontal_acceleration_m_s2"
                        ),
                        self._float(
                            "relative_landing_mpc_max_horizontal_acceleration_m_s2"
                        ),
                        self._float(
                            "relative_landing_mpc_max_vertical_acceleration_m_s2"
                        ),
                    ),
                    dtype=float,
                )
                measured = np.clip(measured, -limits, limits)
                self.vehicle_acceleration_enu = (
                    0.8 * self.vehicle_acceleration_enu + 0.2 * measured
                )
                self.vehicle_acceleration_enu = np.clip(
                    self.vehicle_acceleration_enu, -limits, limits
                )
        self.vehicle_velocity_enu = velocity
        self.velocity_stamp = now
        self._previous_velocity = velocity.copy()
        self._previous_velocity_stamp = now
        self._retry_pending_marker_measurements()

    def _on_attitude(self, message: Imu) -> None:
        orientation = message.orientation
        quaternion = np.asarray(
            (
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
            dtype=float,
        )
        if (
            not np.all(np.isfinite(quaternion))
            or float(np.linalg.norm(quaternion)) < 1.0e-9
        ):
            return
        try:
            sample_time_s = self._px4_sample_time(message.header.stamp)
            self.state_history.add_attitude(sample_time_s, quaternion)
        except StateTimeReset as exc:
            self._handle_time_reset(str(exc), reset_timesync=True)
        except (StateInterpolationError, ValueError):
            return
        self._retry_pending_marker_measurements()

    def _record_position_history(
        self, stamp: object, position_enu: np.ndarray
    ) -> None:
        try:
            sample_time_s = self._px4_sample_time(stamp)
            accepted = self.state_history.add_position(
                sample_time_s, position_enu
            )
        except StateTimeReset as exc:
            self._handle_time_reset(str(exc), reset_timesync=True)
            return
        except (StateInterpolationError, ValueError):
            return
        if not accepted:
            return
        if self.mavros_state_timestamp_source == "ros_header":
            # PASSTHROUGH state and Gazebo sensor payloads already share the
            # raw simulation sample domain.  ROS /clock delivery may lag that
            # domain under load, so callback-time offset observations would
            # measure bridge backlog rather than a clock offset.
            return

        ros_now_s = self.get_clock().now().nanoseconds * 1.0e-9
        observation = self.camera_time_offset.observe(
            ros_now_s, sample_time_s
        )
        self.last_camera_time_offset_residual_s = observation.residual_s
        if observation.reset_detected:
            self._handle_time_reset(
                "camera/PX4 time-offset reset",
                reset_timesync=False,
            )
            self.camera_time_offset.observe(ros_now_s, sample_time_s)
            self.state_history.add_position(sample_time_s, position_enu)

    def _record_velocity_history(
        self, stamp: object, velocity_enu: np.ndarray
    ) -> None:
        try:
            sample_time_s = self._px4_sample_time(stamp)
            self.state_history.add_velocity(sample_time_s, velocity_enu)
        except StateTimeReset as exc:
            self._handle_time_reset(str(exc), reset_timesync=True)
        except (StateInterpolationError, ValueError):
            return

    def _px4_sample_time(self, stamp: object) -> float:
        seconds = int(getattr(stamp, "sec"))
        nanoseconds = int(getattr(stamp, "nanosec"))
        if self.mavros_state_timestamp_source == "ros_header":
            # PASSTHROUGH preserves the PX4 sample stamp without callback-time
            # distortion.  Its boot origin can still differ from Gazebo
            # /clock, so retain this raw sample domain and estimate the one
            # constant camera-to-PX4 offset from position callbacks below.
            return mavros_header_to_px4_time_s(
                seconds, nanoseconds, 0
            )
        if (
            self.mavros_timesync_stamp <= 0.0
            or time.monotonic() - self.mavros_timesync_stamp
            > self._float("mavros_timesync_timeout_s")
        ):
            raise StateInterpolationError("mavros_timesync_stale")
        return self.mavros_timesync.recover_sample_time_s(
            seconds,
            nanoseconds,
        )

    def _ros_time_to_px4_sample_time(self, ros_time_s: float) -> float:
        """Map one ROS stamp into the active PX4 sample-time domain.

        PASSTHROUGH PX4 state and Gazebo sensor payloads use the same raw
        simulation sample domain.  The ROS /clock topic can be delivered late
        without changing those payload stamps.  Real PX4 profiles retain the
        observed companion-to-PX4 mapping.
        """
        value = float(ros_time_s)
        if not math.isfinite(value) or value <= 0.0:
            raise StateInterpolationError("invalid_ros_capture_time")
        if self.mavros_state_timestamp_source == "ros_header":
            return value
        return self.camera_time_offset.to_sample_time(value)

    def _current_px4_sample_time(self) -> float:
        """Return the newest complete PX4 sample-domain state watermark."""
        latest = self.state_history.latest_common_time_s
        if latest is not None and math.isfinite(float(latest)):
            return float(latest)
        ros_now_s = self.get_clock().now().nanoseconds * 1.0e-9
        return self._ros_time_to_px4_sample_time(ros_now_s)

    def _handle_time_reset(
        self, reason: str, *, reset_timesync: bool
    ) -> None:
        self.state_history.reset()
        self.camera_time_offset.reset()
        if reset_timesync:
            self.mavros_timesync.reset()
            self.mavros_timesync_stamp = 0.0
        self.target_estimator.reset()
        self.vision_motion_estimator.reset()
        self.last_target_estimate = None
        self._target_sensors_loss_started_monotonic_s = None
        self.last_vision_motion_estimate = None
        self.last_vision_velocity_enu = None
        self.last_vision_velocity_stamp_px4_s = None
        self.last_vision_velocity_source_stamp_px4_s = None
        self.last_vision_motion_capture_px4_time_s = None
        self.vision_velocity_history.clear()
        self.vision_motion_model_acceleration_enu.fill(0.0)
        self.vision_motion_model_span_s = 0.0
        self.last_position_servo_velocity_enu.fill(0.0)
        self.last_position_servo_update_monotonic_s = None
        self.marker_tracking_acquisition_since_s = None
        self.marker_tracking_latched = False
        self.control_target_velocity_source = "unavailable"
        self.trailer_state = None
        self.trailer_state_stamp = 0.0
        self.front_marker_capture_guard = CaptureStampGuard()
        self.down_marker_capture_guard = CaptureStampGuard()
        self._pending_marker_messages.clear()
        self.front_marker_stamp = 0.0
        self.down_marker_stamp = 0.0
        self.down_marker_detected = False
        self.down_marker_detected_stamp = 0.0
        self.down_marker_positive_stamp = 0.0
        self.front_marker_sequence = 0
        self.down_marker_sequence = 0
        self.front_marker_covariance_diagonal = None
        self.down_marker_covariance_diagonal = None
        self.front_marker_capture_px4_time_s = None
        self.down_marker_capture_px4_time_s = None
        self.front_marker_capture_delay_s = None
        self.down_marker_capture_delay_s = None
        self.front_marker_capture_vehicle_position_enu = None
        self.down_marker_capture_vehicle_position_enu = None
        self.front_marker_capture_vehicle_velocity_enu = None
        self.down_marker_capture_vehicle_velocity_enu = None
        self.last_camera_time_offset_residual_s = None
        self.landing_pad_acceleration_enu = np.zeros(3, dtype=float)
        self.landing_pad_acceleration_source = "aruco_not_initialized"
        self._previous_landing_pad_velocity = None
        self._previous_landing_pad_velocity_stamp_s = None
        self._reset_terminal_contact_context()
        self._invalidate_solver_context(clear_cached=True)
        self.command_cache = None
        self._pending_command = None
        self._time_reset_hold_pending = True
        self.time_reset_count += 1
        self.touchdown_gate.reset(self.time_reset_count)
        self.last_descent_permission = None
        self.last_touchdown_decision = None
        self.marker_loss_started_mission_s = None
        self.terminal_recovery_started_mission_s = None
        self.touchdown_disarm_requested = False
        self.last_time_reset_reason = str(reason)
        self.get_logger().warning(
            f"PX4 capture-time state reset: {self.last_time_reset_reason}"
        )

    def _on_down_lidar(self, message: LaserScan) -> None:
        now = time.monotonic()
        sensor_min = float(message.range_min)
        sensor_max = float(message.range_max)
        configured_min = self._float("descent_min_lidar_distance_m")
        configured_max = self._float("descent_max_lidar_distance_m")
        minimum = max(
            configured_min,
            sensor_min if math.isfinite(sensor_min) else configured_min,
        )
        maximum = min(
            configured_max,
            sensor_max
            if math.isfinite(sensor_max) and sensor_max > 0.0
            else configured_max,
        )
        valid = [
            float(value)
            for value in message.ranges
            if math.isfinite(value) and minimum <= float(value) <= maximum
        ]
        self.down_lidar_stamp = now
        if not valid or maximum < minimum:
            self.down_lidar_range_m = None
            return
        self.down_lidar_range_m = min(valid)

    def _on_trailer_odometry(self, message: Odometry) -> None:
        frame = str(message.header.frame_id).strip().lstrip("/")
        child_frame = str(message.child_frame_id).strip().lstrip("/")
        if frame != "map" or child_frame != "trailer/base_link":
            self.get_logger().warning(
                "Rejecting trailer odometry frames: %r -> %r"
                % (frame, child_frame)
            )
            return
        stamp = message.header.stamp
        stamp_key_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if stamp_key_ns <= 0:
            self.get_logger().warning(
                "Rejecting trailer odometry with zero timestamp"
            )
            return
        ros_stamp_s = stamp_key_ns * 1.0e-9
        try:
            measurement_px4_time_s = self._ros_time_to_px4_sample_time(
                ros_stamp_s
            )
            current_px4_time_s = self._current_px4_sample_time()
        except StateInterpolationError:
            return

        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        values = np.asarray(
            (
                position.x,
                position.y,
                position.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
            dtype=float,
        )
        if (
            not np.all(np.isfinite(values))
            or float(np.linalg.norm(values[3:7])) < 1.0e-9
        ):
            return
        yaw = self._quaternion_yaw(orientation)
        # Keep the latest measured trailer position independent of Kalman
        # admission.  Approach/search guidance is position-only and must keep
        # following a fresh physical sample even if the probabilistic fusion
        # path rejects one update.  No simulator twist is exposed here.
        now = time.monotonic()
        self.trailer_state = TrailerKinematicState(
            measurement_px4_time_s,
            values[:3],
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            yaw,
        )
        self.trailer_state_stamp = now
        # Position is available for interception, but trailer twist is not a
        # production sensor.  Ignore the message field even if a future
        # bridge accidentally populates it and mark the synthetic zero-speed
        # placeholder as effectively uninformative in the fusion update.
        unknown_velocity = np.zeros(3, dtype=float)
        unknown_twist_covariance = np.eye(6, dtype=float) * 1.0e6
        result = self.target_estimator.submit_odometry(
            measurement_px4_time_s,
            stamp_key_ns,
            values[:3],
            unknown_velocity,
            yaw,
            0.0,
            message.pose.covariance,
            unknown_twist_covariance,
            np.asarray(
                (
                    self._float("trailer_marker_offset_x_m"),
                    self._float("trailer_marker_offset_y_m"),
                    self._float("trailer_marker_offset_z_m"),
                ),
                dtype=float,
            ),
            current_px4_time_s,
            twist_in_body_frame=self._bool(
                "trailer_odometry_twist_in_body_frame"
            ),
        )
        if not result.queued:
            self.get_logger().warning(
                f"Rejecting trailer odometry fusion input: {result.reason}"
            )
            return

    def _on_front_marker_pose(
        self, message: PoseWithCovarianceStamped
    ) -> None:
        self._on_marker_pose("front", message)

    def _on_down_marker_pose(
        self, message: PoseWithCovarianceStamped
    ) -> None:
        self._on_marker_pose("down", message)

    def _on_front_marker_quality(self, message: Float32) -> None:
        value = float(message.data)
        self.front_marker_quality = value if math.isfinite(value) else None
        self.front_marker_quality_stamp = time.monotonic()

    def _on_down_marker_quality(self, message: Float32) -> None:
        value = float(message.data)
        self.down_marker_quality = value if math.isfinite(value) else None
        self.down_marker_quality_stamp = time.monotonic()

    def _on_down_marker_detected(self, message: Bool) -> None:
        """Track the detector's current frame result, including misses."""
        received_at = time.monotonic()
        self.down_marker_detected = bool(message.data)
        self.down_marker_detected_stamp = received_at
        if self.down_marker_detected:
            self.down_marker_positive_stamp = received_at

    def _down_marker_detection_live(self, now: float) -> bool:
        """Require a recent positive result and a live detector stream.

        A single missed 10 Hz frame is common when the landing gear clips a
        corner of the marker.  Keep the last positive observation for the
        configured 0.3 s status window, while still failing closed if either
        the positive track or the detector stream itself stops.
        """
        stream_stamp = float(
            getattr(self, "down_marker_detected_stamp", 0.0)
        )
        positive_stamp = float(
            getattr(self, "down_marker_positive_stamp", 0.0)
        )
        stream_age = float(now) - stream_stamp
        positive_age = float(now) - positive_stamp
        timeout_s = self._float("marker_detection_status_timeout_s")
        return bool(
            stream_stamp > 0.0
            and positive_stamp > 0.0
            and math.isfinite(stream_age)
            and math.isfinite(positive_age)
            and 0.0 <= stream_age <= timeout_s
            and 0.0 <= positive_age <= timeout_s
        )

    def _on_marker_pose(
        self,
        camera: str,
        message: PoseWithCovarianceStamped,
        *,
        _registered: bool = False,
    ) -> None:
        """Transform and queue one unique capture for ordered fusion."""

        if camera == "front":
            guard = self.front_marker_capture_guard
        elif camera == "down":
            guard = self.down_marker_capture_guard
        else:
            raise ValueError("camera must be front or down")

        stamp = message.header.stamp
        if not _registered:
            try:
                unique_capture = guard.register(stamp.sec, stamp.nanosec)
            except ValueError as exc:
                self.get_logger().warning(
                    f"Rejecting {camera} marker measurement timestamp: {exc}"
                )
                return
            if not unique_capture:
                return
            # A newer registered capture supersedes an older deferred one for
            # the same camera. This bounded latest-wins queue cannot grow when
            # MAVROS state delivery stalls.
            self._pending_marker_messages.pop(camera, None)

        frame = str(message.header.frame_id).strip().lstrip("/")
        if frame != "base_link":
            self.get_logger().warning(
                "Rejecting %s marker outside production base_link: %r"
                % (camera, frame)
            )
            return
        try:
            covariance_6x6 = extract_pose_covariance_6x6(
                message.pose.covariance
            )
        except ValueError as exc:
            self._increment_invalid_covariance(camera)
            self.get_logger().warning(
                f"Rejecting {camera} marker covariance: {exc}"
            )
            return

        if camera == "front":
            self.front_marker_covariance_diagonal = np.diag(
                covariance_6x6
            ).copy()
        else:
            self.down_marker_covariance_diagonal = np.diag(
                covariance_6x6
            ).copy()

        relative_flu = np.asarray(
            (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(relative_flu)):
            return
        marker_orientation_xyzw = np.asarray(
            (
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
                message.pose.pose.orientation.w,
            ),
            dtype=float,
        )
        if (
            not np.all(np.isfinite(marker_orientation_xyzw))
            or float(np.linalg.norm(marker_orientation_xyzw)) < 1.0e-9
        ):
            self.get_logger().warning(
                f"Rejecting {camera} marker orientation"
            )
            return

        try:
            capture_ros_time_s = (
                int(stamp.sec) + int(stamp.nanosec) * 1.0e-9
            )
            if capture_ros_time_s <= 0.0:
                raise StateInterpolationError("invalid_ros_capture_time")
            capture_px4_time_s = self._ros_time_to_px4_sample_time(
                capture_ros_time_s
            )
            current_px4_time_s = self._current_px4_sample_time()
            capture_state = self.state_history.interpolate(
                capture_px4_time_s,
                current_sample_time_s=current_px4_time_s,
                maximum_gap_s=self._float(
                    "state_history_max_interpolation_gap_s"
                ),
                maximum_capture_age_s=self._float(
                    "state_history_max_capture_age_s"
                ),
                maximum_state_age_s=self._float(
                    "state_history_max_state_age_s"
                ),
                future_tolerance_s=self._float(
                    "state_history_future_tolerance_s"
                ),
            )
            latest_common_px4_time_s = (
                self.state_history.latest_common_time_s
            )
            if latest_common_px4_time_s is None:
                raise StateInterpolationError("state_history_uninitialized")
            # The capture has already been bracketed by PX4 samples.  Use
            # that sample-domain watermark for fusion admission as well;
            # a separately scheduled /clock callback can lag one camera frame
            # and otherwise reclassify this proven capture as future data.
            fusion_received_px4_time_s = max(
                current_px4_time_s,
                float(latest_common_px4_time_s),
            )
            (
                marker_world,
                marker_world_orientation,
                marker_world_yaw,
                rotation_enu_from_flu,
            ) = marker_world_pose_at_capture(
                capture_state,
                relative_flu,
                marker_orientation_xyzw,
            )
        except StateInterpolationError as exc:
            if exc.reason in {
                "capture_ahead_of_state_history",
                "state_history_stale",
            }:
                self._defer_marker_measurement(camera, message)
                return
            self._reject_marker_timing(camera, exc.reason)
            return
        except ValueError as exc:
            self.get_logger().warning(
                f"Rejecting {camera} marker capture transform: {exc}"
            )
            return

        marker_world_tilt_rad = quaternion_world_z_tilt_rad(
            marker_world_orientation
        )
        if camera == "front":
            self.front_marker_world_tilt_rad = marker_world_tilt_rad
        else:
            self.down_marker_world_tilt_rad = marker_world_tilt_rad
        if marker_world_tilt_rad > self._float(
            "maximum_marker_world_tilt_rad"
        ):
            if camera == "front":
                self.front_marker_tilt_rejection_count += 1
            else:
                self.down_marker_tilt_rejection_count += 1
            self.get_logger().warning(
                "Rejecting %s marker world tilt %.3f rad"
                % (camera, marker_world_tilt_rad)
            )
            return

        try:
            estimator_covariance = build_estimator_covariance(
                covariance_6x6,
                rotation_enu_from_flu,
                distance_m=float(np.linalg.norm(relative_flu)),
                camera=camera,
            )
        except ValueError as exc:
            self._increment_invalid_covariance(camera)
            self.get_logger().warning(
                f"Rejecting {camera} marker covariance transform: {exc}"
            )
            return

        stamp_key_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        submission = self.target_estimator.submit_vision(
            f"vision_{camera}",
            capture_px4_time_s,
            stamp_key_ns,
            marker_world,
            marker_world_yaw,
            estimator_covariance,
            fusion_received_px4_time_s,
        )
        if not submission.queued:
            self.get_logger().warning(
                "Rejecting %s marker fusion input: %s"
                % (camera, submission.reason)
            )
            return
        if camera == "down":
            previous_capture = self.last_vision_motion_capture_px4_time_s
            if (
                previous_capture is not None
                and capture_px4_time_s - previous_capture
                > self._float("vision_velocity_reacquisition_gap_s")
            ):
                # A new optical acquisition must prove motion again.  Do not
                # let a cumulative accepted counter or an old velocity cache
                # make one post-dropout pose look like a continuous track.
                self.vision_motion_estimator.reset()
                self.last_vision_motion_estimate = None
                # A new optical epoch must identify velocity again from its
                # own three captures. The acquisition coast keeps the UAV's
                # current motion while that happens, so an old trajectory can
                # neither qualify alignment nor silently resume descent.
                self.last_vision_velocity_enu = None
                self.last_vision_velocity_stamp_px4_s = None
                self.last_vision_velocity_source_stamp_px4_s = None
                self.vision_velocity_history.clear()
                self.vision_motion_model_acceleration_enu.fill(0.0)
                self.vision_motion_model_span_s = 0.0
                self.landing_pad_acceleration_enu.fill(0.0)
                self._previous_landing_pad_velocity = None
                self._previous_landing_pad_velocity_stamp_s = None
            if (
                previous_capture is None
                or capture_px4_time_s > previous_capture
            ):
                self.last_vision_motion_capture_px4_time_s = (
                    capture_px4_time_s
                )
            motion_submission = self.vision_motion_estimator.submit_vision(
                "vision_down",
                capture_px4_time_s,
                stamp_key_ns,
                marker_world,
                marker_world_yaw,
                estimator_covariance,
                fusion_received_px4_time_s,
            )
            if not motion_submission.queued:
                self.get_logger().warning(
                    "Rejecting down marker velocity input: %s"
                    % motion_submission.reason
                )
        now = time.monotonic()
        capture_delay_s = (
            fusion_received_px4_time_s - capture_px4_time_s
        )
        if camera == "front":
            self.front_marker_stamp = now
            self.front_marker_capture_px4_time_s = capture_px4_time_s
            self.front_marker_capture_delay_s = capture_delay_s
            self.front_marker_capture_vehicle_position_enu = (
                capture_state.position_enu_m.copy()
            )
            self.front_marker_capture_vehicle_velocity_enu = (
                capture_state.velocity_enu_m_s.copy()
            )
        else:
            self.down_marker_stamp = now
            self.down_marker_capture_px4_time_s = capture_px4_time_s
            self.down_marker_capture_delay_s = capture_delay_s
            self.down_marker_capture_vehicle_position_enu = (
                capture_state.position_enu_m.copy()
            )
            self.down_marker_capture_vehicle_velocity_enu = (
                capture_state.velocity_enu_m_s.copy()
            )

    def _defer_marker_measurement(
        self,
        camera: str,
        message: PoseWithCovarianceStamped,
    ) -> None:
        """Hold one latest capture until real PX4 state brackets its stamp."""
        stamp = message.header.stamp
        stamp_key_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        previous = self._pending_marker_messages.get(camera)
        if previous is None or previous[0] != stamp_key_ns:
            if camera == "front":
                self.front_marker_deferred_count += 1
            elif camera == "down":
                self.down_marker_deferred_count += 1
            else:
                raise ValueError("camera must be front or down")
        self._pending_marker_messages[camera] = (stamp_key_ns, message)

    def _retry_pending_marker_measurements(self) -> None:
        """Retry deferred captures after a position, velocity, or attitude update."""
        if not self._pending_marker_messages:
            return
        pending = tuple(self._pending_marker_messages.items())
        self._pending_marker_messages.clear()
        for camera, (_stamp_key_ns, message) in pending:
            self._on_marker_pose(camera, message, _registered=True)

    def _increment_invalid_covariance(self, camera: str) -> None:
        if camera == "front":
            self.front_marker_invalid_covariance_count += 1
        else:
            self.down_marker_invalid_covariance_count += 1

    def _reject_marker_timing(self, camera: str, reason: str) -> None:
        if camera == "front":
            self.front_marker_timing_rejection_count += 1
            self.front_marker_last_timing_rejection = str(reason)
        else:
            self.down_marker_timing_rejection_count += 1
            self.down_marker_last_timing_rejection = str(reason)
        self.get_logger().warning(
            f"Rejecting {camera} marker capture time: {reason}"
        )

    @staticmethod
    def _quaternion_yaw(orientation: object) -> float:
        x = float(getattr(orientation, "x"))
        y = float(getattr(orientation, "y"))
        z = float(getattr(orientation, "z"))
        w = float(getattr(orientation, "w"))
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if not math.isfinite(norm) or norm < 1.0e-9:
            raise ValueError(
                "orientation quaternion must be finite and nonzero"
            )
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

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

    def _trailer_fresh(self, now: float) -> bool:
        age = float(now) - float(self.trailer_state_stamp)
        return bool(
            self.trailer_state is not None
            and self.trailer_state_stamp > 0.0
            and math.isfinite(age)
            and 0.0 <= age <= self._float("trailer_timeout_s")
        )

    def _current_target_estimate(self) -> TargetEstimate | None:
        try:
            current_px4_time_s = self._current_px4_sample_time()
            estimate = self.target_estimator.estimate(current_px4_time_s)
        except (StateInterpolationError, ValueError):
            return None
        if (
            not estimate.valid
            and self.phase
            in {
                LandingPhase.FINAL_APPROACH,
                LandingPhase.TOUCHDOWN_CONFIRM,
            }
        ):
            # The probabilistic filter can reject otherwise-live simulator
            # position samples after the optical marker leaves the landing-
            # gear FOV.  FINAL must not turn that admission decision into a
            # total target loss: production already uses the measured trailer
            # position as its continuous marker-centre anchor and deliberately
            # ignores trailer twist.  Combine that fresh position only with
            # the bounded ArUco-derived speed/turn model learned before the
            # occlusion.  Lidar, geometry and timeout gates below remain in
            # force, so this path cannot start precision landing by itself.
            now = time.monotonic()
            raw_pose = self._latest_trailer_marker_pose(now)
            target_velocity = self._control_target_velocity_enu(estimate, now)
            if (
                raw_pose is not None
                and target_velocity is not None
                and self._fresh_lidar(now)
                and self.vehicle_position_enu is not None
            ):
                raw_position, raw_yaw = raw_pose
                measured_clearance = float(
                    self.vehicle_position_enu[2] - raw_position[2]
                )
                if math.isfinite(measured_clearance):
                    position_variance = self._float(
                        "target_fusion_odometry_position_variance_floor"
                    )
                    diagonal = np.asarray(
                        (
                            position_variance,
                            position_variance,
                            position_variance,
                            1.0,
                            1.0,
                            1.0,
                            self._float(
                                "target_fusion_odometry_yaw_variance_floor"
                            ),
                            self._float(
                                "target_fusion_odometry_yaw_rate_variance_floor"
                            ),
                        ),
                        dtype=float,
                    )
                    estimate = TargetEstimate(
                        True,
                        "terminal_raw_position_aruco_motion",
                        float(current_px4_time_s),
                        np.asarray(raw_position, dtype=float).copy(),
                        np.asarray(target_velocity, dtype=float).copy(),
                        float(raw_yaw),
                        0.0,
                        np.diag(diagonal),
                        estimate.filter_stamp_s,
                        False,
                        True,
                        max(0.0, now - self.trailer_state_stamp),
                    )
        self.last_target_estimate = estimate
        self.front_marker_sequence = self.target_estimator.statistics[
            "vision_front"
        ].accepted
        self.down_marker_sequence = self.vision_motion_estimator.statistics[
            "vision_down"
        ].accepted
        return estimate

    def _marker_world_enu(self, camera: str, now: float) -> np.ndarray | None:
        if camera == "front":
            source = "vision_front"
        elif camera == "down":
            source = "vision_down"
        else:
            raise ValueError("camera must be front or down")
        if camera == "down":
            if not self._down_marker_detection_live(now):
                return None
            try:
                current_px4_time_s = self._current_px4_sample_time()
                estimate = self.vision_motion_estimator.estimate(
                    current_px4_time_s
                )
            except (StateInterpolationError, ValueError):
                return None
            self.last_vision_motion_estimate = estimate
            self.down_marker_sequence = (
                self.vision_motion_estimator.statistics["vision_down"].accepted
            )
            fresh = self.vision_motion_estimator.source_fresh(
                source,
                current_px4_time_s,
                self._float("marker_timeout_s"),
            )
        else:
            del now
            estimate = self._current_target_estimate()
            if estimate is None:
                return None
            fresh = self.target_estimator.source_fresh(
                source,
                estimate.stamp_s,
                self._float("marker_timeout_s"),
            )
        if estimate is None or not estimate.valid or not fresh:
            return None
        position = estimate.position_enu_m
        if position is None:
            return None
        return position.copy() if np.all(np.isfinite(position)) else None

    def _latest_trailer_marker_pose(
        self, now: float
    ) -> tuple[np.ndarray, float] | None:
        """Return the newest measured trailer position without any speed."""
        state = self.trailer_state
        age = float(now) - float(self.trailer_state_stamp)
        if (
            state is None
            or not math.isfinite(age)
            or age < 0.0
            or age > self._float("trailer_timeout_s")
        ):
            return None
        yaw = float(state.yaw_enu_rad)
        offset = np.asarray(
            (
                self._float("trailer_marker_offset_x_m"),
                self._float("trailer_marker_offset_y_m"),
                self._float("trailer_marker_offset_z_m"),
            ),
            dtype=float,
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        rotated_offset = np.asarray(
            (
                cosine * offset[0] - sine * offset[1],
                sine * offset[0] + cosine * offset[1],
                offset[2],
            ),
            dtype=float,
        )
        position = np.asarray(state.position_enu_m, dtype=float) + rotated_offset
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return None
        return position, yaw

    def _landing_target_enu(
        self, now: float, preferred_camera: str | None
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        estimate = self._current_target_estimate()
        target_velocity = self._control_target_velocity_enu(estimate, now)
        if target_velocity is None:
            return None
        if (
            estimate is not None
            and estimate.valid
            and estimate.position_enu_m is not None
            and estimate.yaw_enu_rad is not None
        ):
            target_position = estimate.position_enu_m.copy()
            target_yaw = float(estimate.yaw_enu_rad)
        else:
            # SEARCH must not discard a fresh measured position merely because
            # the covariance filter is recovering.  Relative MPC still starts
            # only after ArUco has independently identified target velocity.
            raw_trailer_pose = self._latest_trailer_marker_pose(now)
            if raw_trailer_pose is None:
                return None
            target_position, target_yaw = raw_trailer_pose
        raw_trailer_pose = self._latest_trailer_marker_pose(now)
        if (
            raw_trailer_pose is not None
            and self.phase
            in {
                LandingPhase.MARKER_TRACK_DOWN,
                LandingPhase.PRECISION_ALIGN,
                LandingPhase.PRECISION_DESCENT,
                LandingPhase.FINAL_APPROACH,
                LandingPhase.TOUCHDOWN_CONFIRM,
            }
        ):
            # Keep one continuous marker-centre position anchor throughout
            # precision flight.  Switching XY between vision and trailer pose
            # at close-range occlusion injected a 0.5--1.5 m lateral step and
            # produced the visible left/right chase.  Trailer position is an
            # allowed mission input; its twist remains deliberately hidden.
            # ArUco alone still identifies target velocity/direction and gates
            # entry into descent.
            target_position, target_yaw = raw_trailer_pose
        terminal_level_recovery = bool(
            preferred_camera is None
            and self.phase == LandingPhase.PRECISION_ALIGN
            and self.terminal_recovery_started_mission_s is not None
            and self.descent_clearance_m
            <= self._float("terminal_occlusion_max_clearance_m")
            + 1.0e-6
        )
        if terminal_level_recovery:
            # Trailer position remains an allowed measured input, while its
            # twist is deliberately hidden.  During a close-range marker gap,
            # use that fresh position directly and combine it only with the
            # bounded ArUco-derived velocity selected above.  The fused pose
            # can momentarily lag or jump when FINAL changes sensor context;
            # using it here produced an opposite-direction command before the
            # previous run fell into world HOLD.
            raw_trailer_pose = self._latest_trailer_marker_pose(now)
            if raw_trailer_pose is None:
                return None
            target_position, target_yaw = raw_trailer_pose
        if preferred_camera == "down":
            # During precision tracking center on the optical marker itself;
            # odometry remains the position-only reacquisition source.
            vision_pose = self._fresh_down_vision_pose(now)
            vision_usable = bool(
                vision_pose is not None
                and self._vision_position_covariance_safe()
            )
            terminal_bridge = bool(
                self.phase
                in {
                    LandingPhase.FINAL_APPROACH,
                    LandingPhase.TOUCHDOWN_CONFIRM,
                }
                and self.terminal_contact_bridge_active
            )
            if vision_usable and vision_pose is not None:
                vision_position, vision_yaw = vision_pose
                target_yaw = vision_yaw
                if raw_trailer_pose is None:
                    target_position = vision_position.copy()
            elif terminal_bridge:
                if raw_trailer_pose is not None:
                    target_position, target_yaw = raw_trailer_pose
            elif self.phase in {
                LandingPhase.FINAL_APPROACH,
                LandingPhase.TOUCHDOWN_CONFIRM,
            }:
                # A single large marker necessarily leaves the downward FOV
                # near contact. Keep using only the newest measured trailer
                # position while velocity remains the independently cached
                # ArUco-derived value selected above. Trailer twist is never
                # substituted into the control reference.
                if raw_trailer_pose is not None:
                    target_position, target_yaw = raw_trailer_pose
        if self.phase in {
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
        }:
            # Aim slightly ahead along the camera-derived direction so the
            # vehicle reaches the moving deck rather than the marker's old
            # position after camera, solver and PX4 response latency. Trailer
            # twist remains unused; both direction and speed are ArUco-only.
            lead_time_s = self._float("landing_motion_lead_time_s")
            if math.isfinite(lead_time_s) and lead_time_s > 0.0:
                target_position = target_position.copy()
                target_position[:2] += (
                    target_velocity[:2] * lead_time_s
                )
        return (
            target_position,
            target_velocity,
            target_yaw,
        )

    def _fresh_down_vision_pose(
        self, now: float
    ) -> tuple[np.ndarray, float] | None:
        """Return the fresh vision-only marker pose used by control/gates."""
        if not self._down_marker_detection_live(now):
            return None
        motion = self.last_vision_motion_estimate
        if (
            motion is None
            or not motion.valid
            or not motion.vision_fresh
            or motion.position_enu_m is None
            or motion.yaw_enu_rad is None
        ):
            return None
        callback_age = (
            math.inf
            if self.down_marker_stamp <= 0.0
            else float(now) - self.down_marker_stamp
        )
        accepted_age = self._vision_motion_source_age(motion)
        effective_accepted_age = (
            math.inf
            if accepted_age is None
            else max(
                0.0,
                accepted_age
                - self._float("target_fusion_reorder_window_s"),
            )
        )
        if (
            not math.isfinite(callback_age)
            or callback_age < 0.0
            or max(callback_age, effective_accepted_age)
            > self._float("descent_max_vision_age_s")
        ):
            return None
        position = np.asarray(motion.position_enu_m, dtype=float)
        yaw = float(motion.yaw_enu_rad)
        if (
            position.shape != (3,)
            or not np.all(np.isfinite(position))
            or not math.isfinite(yaw)
        ):
            return None
        return position.copy(), yaw

    def _vision_position_covariance_safe(self) -> bool:
        """Return whether the current vision-only position covariance passes."""
        motion = self.last_vision_motion_estimate
        if motion is None or not motion.valid or motion.covariance is None:
            return False
        covariance = np.asarray(motion.covariance, dtype=float)
        if covariance.shape != (8, 8):
            return False
        try:
            position_covariance = validate_covariance(
                covariance[:3, :3], 3, "vision target position"
            )
        except ValueError:
            return False
        return bool(
            float(np.max(np.linalg.eigvalsh(position_covariance)))
            <= self._float("descent_max_position_variance_m2")
        )

    def _update_vision_velocity_model(
        self,
        capture_stamp_s: float,
        measured_velocity_enu_m_s: np.ndarray,
    ) -> None:
        """Fit a robust local motion model from ArUco-derived velocities."""
        stamp = float(capture_stamp_s)
        measured = np.asarray(measured_velocity_enu_m_s, dtype=float).copy()
        if (
            not math.isfinite(stamp)
            or measured.shape != (3,)
            or not np.all(np.isfinite(measured))
        ):
            return
        measured[2] = 0.0
        history = self.vision_velocity_history
        if history and stamp <= history[-1][0] + 1.0e-9:
            return
        history.append((stamp, measured))
        window_s = 5.0
        while history and stamp - history[0][0] > window_s:
            history.popleft()

        stamps = np.asarray([sample[0] for sample in history], dtype=float)
        velocities = np.asarray(
            [sample[1][:2] for sample in history], dtype=float
        )
        span_s = 0.0 if len(stamps) < 2 else float(stamps[-1] - stamps[0])
        self.vision_motion_model_span_s = max(0.0, span_s)
        fitted_velocity = np.median(velocities, axis=0)
        fitted_acceleration = np.zeros(2, dtype=float)
        if len(stamps) >= 5 and span_s >= 0.8:
            relative_times = stamps - stamps[-1]
            design = np.column_stack(
                (np.ones(len(relative_times), dtype=float), relative_times)
            )
            coefficients = np.linalg.lstsq(
                design, velocities, rcond=None
            )[0]
            residual_norm = np.linalg.norm(
                velocities - design @ coefficients, axis=1
            )
            median_residual = float(np.median(residual_norm))
            mad = float(
                np.median(np.abs(residual_norm - median_residual))
            )
            robust_scale = max(1.0e-3, 1.4826 * mad)
            retained = residual_norm <= median_residual + 3.0 * robust_scale
            if int(np.count_nonzero(retained)) >= 4:
                coefficients = np.linalg.lstsq(
                    design[retained], velocities[retained], rcond=None
                )[0]
            fitted_velocity = coefficients[0]
            fitted_acceleration = coefficients[1]

        maximum_speed = self._float(
            "relative_landing_mpc_max_horizontal_velocity_m_s"
        )
        speed = float(np.linalg.norm(fitted_velocity))
        if speed > maximum_speed:
            fitted_velocity *= maximum_speed / speed
            speed = maximum_speed

        # Reconstruct one internally consistent planar turn model.  A raw
        # adjacent-frame derivative had random turn-rate sign and tangential
        # acceleration, which grew into metre-per-second errors during the
        # expected close-marker occlusion.
        coherent_acceleration = np.zeros(2, dtype=float)
        if speed > 1.0e-3:
            tangent = fitted_velocity / speed
            normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
            turn_rate = float(
                np.cross(
                    np.append(fitted_velocity, 0.0),
                    np.append(fitted_acceleration, 0.0),
                )[2]
                / (speed * speed)
            )
            turn_rate = float(np.clip(turn_rate, -0.20, 0.20))
            tangential = float(np.dot(fitted_acceleration, tangent))
            if abs(tangential) < 0.15:
                tangential = 0.0
            tangential = float(np.clip(tangential, -0.40, 0.40))
            coherent_acceleration = (
                tangential * tangent + turn_rate * speed * normal
            )
            maximum_acceleration = self._float(
                "relative_landing_mpc_max_pad_acceleration_m_s2"
            )
            acceleration_norm = float(
                np.linalg.norm(coherent_acceleration)
            )
            if acceleration_norm > maximum_acceleration:
                coherent_acceleration *= (
                    maximum_acceleration / acceleration_norm
                )

        self.last_vision_velocity_enu = np.asarray(
            (fitted_velocity[0], fitted_velocity[1], 0.0), dtype=float
        )
        self.vision_motion_model_acceleration_enu = np.asarray(
            (
                coherent_acceleration[0],
                coherent_acceleration[1],
                0.0,
            ),
            dtype=float,
        )

    def _predict_vision_motion_velocity(
        self, age_s: float
    ) -> np.ndarray | None:
        """Propagate the coherent ArUco speed/heading model to now."""
        velocity = self.last_vision_velocity_enu
        if velocity is None:
            return None
        base_velocity = np.asarray(velocity, dtype=float)
        base_acceleration = np.asarray(
            self.vision_motion_model_acceleration_enu, dtype=float
        )
        age = float(age_s)
        if (
            base_velocity.shape != (3,)
            or base_acceleration.shape != (3,)
            or not np.all(np.isfinite(base_velocity))
            or not np.all(np.isfinite(base_acceleration))
            or not math.isfinite(age)
            or age < 0.0
        ):
            return None
        predicted = base_velocity.copy()
        predicted_acceleration = base_acceleration.copy()
        speed = float(np.linalg.norm(base_velocity[:2]))
        if speed > 1.0e-3:
            tangent = base_velocity[:2] / speed
            turn_rate = float(
                (
                    base_velocity[0] * base_acceleration[1]
                    - base_velocity[1] * base_acceleration[0]
                )
                / (speed * speed)
            )
            tangential = float(
                np.dot(base_velocity[:2], base_acceleration[:2]) / speed
            )
            angle = turn_rate * age
            cosine = math.cos(angle)
            sine = math.sin(angle)
            direction = np.asarray(
                (
                    cosine * tangent[0] - sine * tangent[1],
                    sine * tangent[0] + cosine * tangent[1],
                ),
                dtype=float,
            )
            normal = np.asarray((-direction[1], direction[0]), dtype=float)
            predicted_speed = max(0.0, speed + tangential * age)
            maximum_speed = self._float(
                "relative_landing_mpc_max_horizontal_velocity_m_s"
            )
            predicted_speed = min(maximum_speed, predicted_speed)
            predicted[:2] = predicted_speed * direction
            predicted_acceleration[:2] = (
                tangential * direction
                + turn_rate * predicted_speed * normal
            )
        else:
            predicted[:2] += base_acceleration[:2] * age
        predicted[2] = 0.0
        predicted_acceleration[2] = 0.0
        self.landing_pad_acceleration_enu = predicted_acceleration
        self.landing_pad_acceleration_source = "aruco_robust_motion_fit"
        return predicted

    def _control_target_velocity_enu(
        self,
        estimate: TargetEstimate | None,
        now: float,
    ) -> np.ndarray | None:
        """Return the ArUco-observed velocity used by MPC and safety gates."""
        del estimate
        try:
            current_px4_time_s = self._current_px4_sample_time()
            motion = self.vision_motion_estimator.estimate(
                current_px4_time_s
            )
        except (StateInterpolationError, ValueError):
            motion = None
            current_px4_time_s = math.nan
        self.last_vision_motion_estimate = motion

        statistics = self.vision_motion_estimator.statistics["vision_down"]
        accepted = int(statistics.accepted)
        accepted_stamp = statistics.last_accepted_stamp_s
        fresh_velocity = False
        candidate: np.ndarray | None = None
        if (
            motion is not None
            and motion.valid
            and motion.vision_fresh
            and motion.velocity_enu_m_s is not None
            and motion.covariance is not None
            and accepted >= self._int(
                "vision_velocity_minimum_accepted_samples"
            )
            and accepted_stamp is not None
            and math.isfinite(current_px4_time_s)
            and self._down_marker_detection_live(now)
            and accepted_stamp
            != self.last_vision_velocity_source_stamp_px4_s
        ):
            candidate = np.asarray(motion.velocity_enu_m_s, dtype=float)
            covariance = np.asarray(motion.covariance, dtype=float)
            velocity_variances = (
                np.diag(covariance)[3:6]
                if covariance.shape == (8, 8)
                else np.full(3, math.inf, dtype=float)
            )
            stream_age = (
                math.inf
                if self.down_marker_stamp <= 0.0
                else max(0.0, time.monotonic() - self.down_marker_stamp)
            )
            fresh_velocity = bool(
                candidate.shape == (3,)
                and np.all(np.isfinite(candidate))
                and np.all(np.isfinite(velocity_variances))
                and np.all(
                    velocity_variances
                    <= self._float(
                        "vision_velocity_maximum_variance_m2_s2"
                    )
                )
                and stream_age
                <= self._float("descent_max_vision_age_s")
            )
        if fresh_velocity and candidate is not None:
            candidate = candidate.copy()
            # The trailer is constrained to the flat map.  Differentiating
            # ArUco depth would turn PnP Z jitter into vertical jerk.
            candidate[2] = 0.0
            maximum_xy_speed = self._float(
                "relative_landing_mpc_max_horizontal_velocity_m_s"
            )
            if np.linalg.norm(candidate[:2]) <= maximum_xy_speed:
                self._update_vision_velocity_model(
                    float(accepted_stamp), candidate
                )
                # Cache age is tied to the accepted camera capture.  Repeated
                # control ticks can no longer re-certify a stale solution by
                # sliding this timestamp forward to "now".
                self.last_vision_velocity_stamp_px4_s = float(accepted_stamp)
                self.last_vision_velocity_source_stamp_px4_s = float(
                    accepted_stamp
                )
                self.control_target_velocity_source = (
                    "aruco_capture_kalman"
                )

        cached = self.last_vision_velocity_enu
        cached_stamp = self.last_vision_velocity_stamp_px4_s
        if (
            cached is not None
            and cached_stamp is not None
            and math.isfinite(current_px4_time_s)
        ):
            age = current_px4_time_s - cached_stamp
            effective_age = max(
                0.0,
                age - self._float("target_fusion_reorder_window_s"),
            )
            # FINAL may temporarily enter level ALIGN to recover lateral
            # geometry.  That transition must stop descent without shrinking
            # the bounded ArUco-velocity continuation from the terminal
            # window to the ordinary 1.5 s dropout window.  Doing so made the
            # moving-deck reference disappear during the expected close-range
            # marker occlusion and replaced 3 m/s tracking with a world HOLD.
            terminal_recovery = bool(
                self.phase == LandingPhase.PRECISION_ALIGN
                and self.terminal_recovery_started_mission_s is not None
                and self.descent_clearance_m
                <= self._float("terminal_occlusion_max_clearance_m")
                + 1.0e-6
            )
            terminal = bool(
                self.phase
                in {
                    LandingPhase.FINAL_APPROACH,
                    LandingPhase.TOUCHDOWN_CONFIRM,
                }
                or terminal_recovery
            )
            maximum_age = self._float(
                "vision_velocity_terminal_hold_s"
                if terminal
                else "vision_velocity_dropout_hold_s"
            )
            if 0.0 <= age and effective_age <= maximum_age:
                # Propagation uses the full capture age. The estimator reorder
                # delay is a freshness allowance, not time that can be removed
                # from circular-motion kinematics.
                predicted = self._predict_vision_motion_velocity(age)
                if predicted is None:
                    self.control_target_velocity_source = "unavailable"
                    return None
                if effective_age <= self._float(
                    "descent_max_vision_age_s"
                ):
                    self.control_target_velocity_source = (
                        "aruco_capture_hold"
                    )
                else:
                    self.control_target_velocity_source = (
                        "aruco_terminal_hold"
                        if terminal
                        else "aruco_dropout_hold"
                    )
                return predicted

        self.control_target_velocity_source = "unavailable"
        return None

    def _publish_position_only_acquisition_setpoint(
        self,
        now: float,
        clearance_m: float,
        *,
        use_down_marker: bool,
        target_velocity_enu_m_s: np.ndarray | None = None,
    ) -> tuple[float, float] | None:
        """Pursue live trailer position while ArUco learns target speed."""
        del use_down_marker
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

        # Before the one-second optical handoff, keep one continuous guidance
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
        # one-second handoff dwell could finish.  As soon as successive ArUco
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
        desired_velocity[:2] = horizontal_gain * error_xy
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
            self.vehicle_yaw_enu_rad,
            False,
            True,
            False,
            valid=True,
            degraded=False,
            status=self.control_target_velocity_source,
            solve_time_s=0.0,
            controller_type=ACQUISITION_GUIDANCE_CONTROLLER_TYPE,
            primary_controller_type=RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
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
        self.last_reference_enu = target_position.copy()
        self.last_reference_enu[2] = vertical_target
        yaw_error = abs(
            math.atan2(
                math.sin(self.vehicle_yaw_enu_rad - target_yaw),
                math.cos(self.vehicle_yaw_enu_rad - target_yaw),
            )
        )
        return distance_xy, yaw_error

    def _heartbeat_tick(self) -> None:
        """Continuously publish only the latest validated command cache."""
        now = time.monotonic()
        self._heartbeat_timestamps.append(now)
        self._pending_command = None
        takeover_active = getattr(
            self, "_terminal_auto_land_mode_active", lambda: False
        )()
        if takeover_active:
            self._accept_terminal_auto_land_engagement(now)
            return
        worker = self.solver_worker
        result = None if worker is None else worker.take_completed()
        if result is not None:
            self._apply_solver_result(result, time.monotonic())
        self._expire_cached_solver_command(time.monotonic())
        command = self.command_cache
        if command is not None and self._offboard_setpoint_active():
            self._publish_setpoint(command)

    def _solver_tick(self) -> None:
        """Start at most one queued immutable snapshot on the worker."""
        worker = self.solver_worker
        if worker is None or worker.has_outstanding:
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
        try:
            submitted = worker.submit(snapshot)
        except Exception as exc:
            self._pending_solver_snapshot = None
            self.consecutive_solver_failures += 1
            fallback_snapshot = self._fresh_fallback_snapshot(
                snapshot, time.monotonic()
            )
            command = self._async_failure_command(
                fallback_snapshot,
                f"worker submit exception: {type(exc).__name__}: {exc}",
                0.0,
            )
            self._record_landing_command(command)
            self._cached_solver_snapshot = (
                fallback_snapshot if command.valid else None
            )
            self._cache_command(command)
            return
        if submitted:
            self._last_solver_queue_age_s = max(
                0.0, time.monotonic() - snapshot.created_monotonic_s
            )
            # ``latest`` remains the current-state comparison snapshot while
            # this exact immutable input is owned by the worker.  Clearing it
            # here made every fast result fail as ``no_current_snapshot``.
            self._pending_solver_snapshot = None

    def _apply_solver_result(
        self, result: LandingSolveResult, now: float
    ) -> None:
        """Validate one completed result and update only the command cache."""
        self._last_solver_result_snapshot_stamp_s = (
            result.snapshot.created_monotonic_s
        )
        self._last_solver_queue_age_s = result.queue_age_s
        if result.snapshot.generation <= getattr(
            self, "_revoked_solver_generation", 0
        ):
            # A phase transition, HOLD, time reset, or synchronous P command
            # explicitly revoked this generation.  Its late worker result is
            # expected scheduling fallout, not a solver/staleness failure.
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
            position = self.vehicle_position_enu
            if (
                position is None
                or np.asarray(position).shape != (3,)
                or not np.all(np.isfinite(position))
            ):
                self.command_cache = None
                self._pending_command = None
                return
            hold = LandingControlCommand.hold(
                position,
                self.vehicle_yaw_enu_rad,
                valid=False,
                degraded=True,
                status="aruco_acquisition_command_stale",
                primary_controller_type=RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
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
                self._pending_command = None
                return
            hold = LandingControlCommand.hold(
                position,
                self.vehicle_yaw_enu_rad,
                valid=False,
                degraded=True,
                status="phase_transition_command_stale",
                primary_controller_type=RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
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
                self._pending_command = None
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
        """Return only a current-phase snapshot suitable for P fallback."""
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
            self._pending_command = None
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
        """Build a fresh configured fallback without reusing solver output."""
        configured = str(
            self.get_parameter("landing_controller_type").value
        )
        mpc_attempted = configured in {
            LEGACY_MPC_CONTROLLER_TYPE,
            RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
        }
        deadline_missed = "deadline" in str(reason)
        elapsed = max(0.0, float(elapsed_s))
        fallback = str(self.get_parameter("mpc_failure_fallback").value)
        if (
            fallback == P_FEEDFORWARD_FALLBACK
            and current is not None
            and not current.control_input.landing_constraints_enabled
        ):
            try:
                command = self.async_p_fallback_controller.compute(
                    current.control_input
                )
                control_input = current.control_input
                if (
                    configured == RELATIVE_LANDING_MPC_CONTROLLER_TYPE
                    and control_input.has_landing_pad_context
                    and not control_input.descent_allowed
                ):
                    pad_velocity = (
                        control_input.landing_pad_velocity_enu_m_s
                    )
                    if pad_velocity is None:
                        raise ValueError(
                            "landing-pad velocity unavailable for fallback"
                        )
                    pad_vertical_velocity = float(pad_velocity[2])
                    if (
                        pad_vertical_velocity
                        > control_input.vertical_velocity_limit_m_s
                    ):
                        raise ValueError(
                            "landing pad climbs faster than fallback limit"
                        )
                    safe_velocity = (
                        command.velocity_setpoint_enu_m_s.copy()
                    )
                    safe_velocity[2] = max(
                        safe_velocity[2], pad_vertical_velocity
                    )
                    command = replace(
                        command,
                        velocity_setpoint_enu_m_s=safe_velocity,
                    )
                return replace(
                    command,
                    degraded=True,
                    status=f"async_solver_failed_p_feedforward: {reason}",
                    solve_time_s=elapsed,
                    primary_controller_type=configured,
                    deadline_missed=deadline_missed,
                    mpc_attempted=mpc_attempted,
                    mpc_success=False if mpc_attempted else None,
                    fallback_used=P_FEEDFORWARD_FALLBACK,
                )
            except Exception:
                pass
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
            pad_velocity = np.asarray(
                control_input.landing_pad_velocity_enu_m_s,
                dtype=float,
            )
            previous_velocity = None
            cached_command = self.command_cache
            if cached_command is not None and cached_command.velocity_enabled:
                candidate_velocity = np.asarray(
                    cached_command.velocity_setpoint_enu_m_s,
                    dtype=float,
                )
                if candidate_velocity.shape == (3,) and np.all(
                    np.isfinite(candidate_velocity)
                ):
                    previous_velocity = candidate_velocity.copy()
            if previous_velocity is None:
                previous_velocity = np.asarray(
                    control_input.vehicle_velocity_enu_m_s,
                    dtype=float,
                ).copy()
            safe_velocity = previous_velocity.copy()
            if safe_velocity.shape == (3,) and np.all(
                np.isfinite(safe_velocity)
            ) and pad_velocity.shape == (3,) and np.all(
                np.isfinite(pad_velocity)
            ):
                horizontal_limit = float(
                    control_input.horizontal_velocity_limit_m_s
                )
                vertical_limit = float(
                    control_input.vertical_velocity_limit_m_s
                )
                control_dt_s = 1.0 / self._float("control_rate_hz")
                horizontal_acceleration_limit = min(
                    1.5,
                    self._float(
                        "relative_landing_mpc_max_horizontal_acceleration_m_s2"
                    ),
                )
                vertical_acceleration_limit = min(
                    0.6,
                    self._float(
                        "relative_landing_mpc_max_vertical_acceleration_m_s2"
                    ),
                )
                desired_velocity = pad_velocity.copy()
                desired_horizontal_speed = float(
                    np.linalg.norm(desired_velocity[:2])
                )
                if desired_horizontal_speed > horizontal_limit:
                    desired_velocity[:2] *= (
                        horizontal_limit / desired_horizontal_speed
                    )
                desired_velocity[2] = float(
                    np.clip(
                        desired_velocity[2],
                        -vertical_limit,
                        vertical_limit,
                    )
                )
                horizontal_delta = (
                    desired_velocity[:2] - previous_velocity[:2]
                )
                maximum_horizontal_delta = (
                    horizontal_acceleration_limit * control_dt_s
                )
                horizontal_delta_norm = float(
                    np.linalg.norm(horizontal_delta)
                )
                if horizontal_delta_norm > maximum_horizontal_delta:
                    horizontal_delta *= (
                        maximum_horizontal_delta / horizontal_delta_norm
                    )
                safe_velocity[:2] = (
                    previous_velocity[:2] + horizontal_delta
                )
                maximum_vertical_delta = (
                    vertical_acceleration_limit * control_dt_s
                )
                safe_velocity[2] = float(
                    previous_velocity[2]
                    + np.clip(
                        desired_velocity[2] - previous_velocity[2],
                        -maximum_vertical_delta,
                        maximum_vertical_delta,
                    )
                )
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
                    primary_controller_type=configured,
                    deadline_missed=deadline_missed,
                    mpc_attempted=mpc_attempted,
                    mpc_success=False if mpc_attempted else None,
                    fallback_used=HOLD_FALLBACK,
                )
        return LandingControlCommand.hold(
            position,
            yaw,
            valid=False,
            degraded=True,
            status=f"async_solver_failed_hold: {reason}",
            solve_time_s=elapsed,
            primary_controller_type=configured,
            deadline_missed=deadline_missed,
            mpc_attempted=mpc_attempted,
            mpc_success=False if mpc_attempted else None,
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
            <= self.solution_acceptance_limits.maximum_solution_age_s
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

    def _offboard_setpoint_active(self) -> bool:
        """Return whether the current phase expects MAVROS raw setpoints."""
        return not self._terminal_auto_land_mode_active() and self.phase in {
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

    def _fresh_lidar(self, now: float) -> bool:
        """Return whether the latest lidar range is finite and in bounds."""
        distance = self.down_lidar_range_m
        return bool(
            distance is not None
            and math.isfinite(distance)
            and self._float("descent_min_lidar_distance_m")
            <= distance
            <= self._float("descent_max_lidar_distance_m")
            and 0.0 <= now - self.down_lidar_stamp
            <= self._float("descent_lidar_timeout_s")
        )

    def _target_source_age(
        self, estimate: TargetEstimate, source: str
    ) -> float | None:
        """Return one accepted sensor age in the estimator time domain."""
        if (
            source == "odometry"
            and estimate.mode == "terminal_raw_position_aruco_motion"
        ):
            age = time.monotonic() - self.trailer_state_stamp
            return age if math.isfinite(age) and age >= 0.0 else None
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
        if command.primary_controller_type in {
            LEGACY_MPC_CONTROLLER_TYPE,
            RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
        }:
            return bool(command.mpc_success is True)
        return False

    def _descent_permission(
        self,
        now: float,
        estimate: TargetEstimate | None = None,
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
            vision_pose = self._fresh_down_vision_pose(now)
            vision_usable = bool(
                vision_pose is not None
                and self._vision_position_covariance_safe()
            )
            if vision_usable and vision_pose is not None:
                # Check the exact optical target that the precision controller
                # is centering on. The old fused-position comparison produced
                # false outside-funnel denials at centimetre camera error.
                target_position = vision_pose[0].copy()
                raw_trailer_pose = self._latest_trailer_marker_pose(now)
                if raw_trailer_pose is not None:
                    target_position[2] = raw_trailer_pose[0][2]
            elif (
                self.phase
                in {
                    LandingPhase.FINAL_APPROACH,
                    LandingPhase.TOUCHDOWN_CONFIRM,
                }
                and (
                    self.terminal_contact_bridge_active
                    or not self._vision_position_covariance_safe()
                )
            ):
                raw_trailer_pose = self._latest_trailer_marker_pose(now)
                if raw_trailer_pose is not None:
                    target_position = raw_trailer_pose[0]
            elif self.phase in {
                LandingPhase.FINAL_APPROACH,
                LandingPhase.TOUCHDOWN_CONFIRM,
            }:
                raw_trailer_pose = self._latest_trailer_marker_pose(now)
                if raw_trailer_pose is not None:
                    target_position = raw_trailer_pose[0]
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
        lidar_age = (
            None
            if self.down_lidar_stamp <= 0.0
            else max(0.0, now - self.down_lidar_stamp)
        )
        decision = evaluate_descent_permission(
            DescentPermissionInput(
                target_estimate_valid=valid,
                vision_age_s=vision_age,
                odometry_age_s=odometry_age,
                position_covariance_m2=covariance,
                horizontal_error_m=horizontal_error,
                funnel_radius_m=funnel_radius,
                relative_horizontal_speed_m_s=relative_speed,
                lidar_distance_m=self.down_lidar_range_m,
                lidar_age_s=lidar_age,
                relative_height_m=relative_height,
                solver_command_valid=self._solver_cache_safe_for_descent(now),
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
        """Adapt optional PX4 bits or conservative MAVROS ON_GROUND."""
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
            topic = str(self.get_parameter(topic_parameter).value).strip()
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
            else "mavros_on_ground_implication"
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
        auto_land_mode = str(
            self.get_parameter("touchdown_auto_land_mode").value
        )
        takeover_requested = bool(
            getattr(self, "terminal_auto_land_requested", False)
        )
        takeover_engaged_once = bool(
            getattr(self, "terminal_auto_land_engaged_monotonic_s", None)
            is not None
        )
        if offboard_required and state.armed:
            if takeover_engaged_once and state.mode != auto_land_mode:
                return "terminal_auto_land_lost"
            if (
                takeover_requested
                and not takeover_engaged_once
                and state.mode not in {"OFFBOARD", auto_land_mode}
            ):
                return "offboard_lost"
            if not takeover_requested and state.mode != "OFFBOARD":
                return "offboard_lost"
        if (
            offboard_required
            and not self._terminal_auto_land_mode_active()
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
        if (
            position_measurement_required
            and not self._terminal_auto_land_mode_active()
        ):
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
        if fused_target_required and not self._terminal_auto_land_mode_active():
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
            self._pending_command = None
        else:
            self._publish_hold(self.failsafe_hold_position_enu)

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
        if self.phase in {
            LandingPhase.MARKER_TRACK_DOWN,
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
        }:
            staged = self._publish_trailer_relative_setpoint(
                now,
                self.descent_clearance_m,
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
            self._publish_hold(self.vehicle_position_enu)

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
                self._publish_hold(self.vehicle_position_enu)
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
            primary_controller_type=RELATIVE_LANDING_MPC_CONTROLLER_TYPE,
            mpc_attempted=False,
            mpc_success=None,
            fallback_used=HOLD_FALLBACK,
        )
        self._record_landing_command(coast)
        self._cache_command(coast)

    def _reset_terminal_contact_context(
        self, *, preserve_auto_land: bool = False
    ) -> None:
        """Clear the one-shot terminal-contact latch on reset/phase change."""
        self._contact_settle_started_monotonic_s = None
        self._contact_settle_started_time_reset_count = None
        self._contact_settle_min_lidar_range_m = None
        self._contact_entry_latched = False
        self._contact_compression_started_monotonic_s = None
        self.terminal_contact_bridge_active = False
        self.last_terminal_contact_bridge_reason = None
        self._terminal_auto_land_cancel_reason = None
        self._terminal_auto_land_kinematics_unsafe_since = None
        if not preserve_auto_land:
            self.terminal_auto_land_requested = False
            self.terminal_auto_land_engaged_monotonic_s = None

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

    def _contact_lidar_within_latch(self) -> bool:
        """Keep contact tracking while the aircraft remains over the deck."""
        distance = self.down_lidar_range_m
        maximum = self._float("touchdown_max_deck_distance_m")
        if (
            distance is None
            or not math.isfinite(float(distance))
            or not math.isfinite(maximum)
            or maximum <= 0.0
        ):
            return False
        # The old running-minimum + 5 cm test aborted the first real contact
        # approach even though the recorded range was still monotonically
        # falling from 0.24 m to 0.10 m. The touchdown gate already requires
        # fresh lidar, bounded vertical speed and PX4 contact/landed/at-rest;
        # here only ensure that the vehicle has not left the deck envelope.
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
        if not self._contact_lidar_within_latch():
            return False, "lidar_rebound_exceeded"
        return True, "vision_occluded_at_contact"

    def _terminal_auto_land_mode_active(self) -> bool:
        """Return whether the explicitly requested PX4 handoff owns descent."""
        state = self.mavros_state
        mode = str(self.get_parameter("touchdown_auto_land_mode").value)
        return bool(
            getattr(self, "terminal_auto_land_requested", False)
            and state is not None
            and state.armed
            and state.mode == mode
        )

    def _terminal_auto_land_engaged(self) -> bool:
        """Return whether LAND is active in a terminal mission phase."""
        return bool(
            self.phase
            in {
                LandingPhase.FINAL_APPROACH,
                LandingPhase.TOUCHDOWN_CONFIRM,
                LandingPhase.FAILSAFE_HOLD,
            }
            and self._terminal_auto_land_mode_active()
        )

    def _accept_terminal_auto_land_engagement(self, now: float) -> None:
        """Atomically revoke OFFBOARD work when PX4 reports AUTO.LAND."""
        if not self._terminal_auto_land_mode_active():
            return
        if self.terminal_auto_land_engaged_monotonic_s is None:
            self.terminal_auto_land_engaged_monotonic_s = float(now)
            self._invalidate_solver_context(clear_cached=True)
            self.command_cache = None
            self._pending_command = None

    def _request_terminal_auto_land(self, now: float) -> bool:
        """Request normal PX4 LAND only after a fully gated contact latch."""
        if self._contact_settle_elapsed_s(now) is None:
            return False
        self.terminal_auto_land_requested = True
        if self._terminal_auto_land_engaged():
            self._accept_terminal_auto_land_engagement(now)
            return True
        mode = str(self.get_parameter("touchdown_auto_land_mode").value)
        if self._request_mode(mode, now):
            self.terminal_auto_land_request_count += 1
        return True

    def _terminal_auto_land_safety_permission(
        self,
        now: float,
        estimate: TargetEstimate | None,
        decision: DescentPermissionDecision,
    ) -> tuple[bool, str]:
        """Monitor all non-optical gates while PX4 performs final landing."""
        if not self._terminal_auto_land_engaged():
            return False, "auto_land_not_engaged"
        engaged = self.terminal_auto_land_engaged_monotonic_s
        if engaged is None:
            self.terminal_auto_land_engaged_monotonic_s = float(now)
            engaged = float(now)
        elapsed = float(now) - float(engaged)
        if (
            not math.isfinite(elapsed)
            or elapsed < 0.0
            or elapsed > self._float("touchdown_auto_land_timeout_s")
        ):
            return False, "auto_land_timeout"
        failures = set(decision.failed_checks)
        contact_transients = {
            "outside_landing_funnel",
            "relative_horizontal_speed_exceeded",
        }
        unsafe = failures.difference(
            {
                "vision_stale",
                "solver_command_invalid",
                "position_covariance_invalid",
                "position_covariance_exceeded",
            }
            | contact_transients
        )
        if unsafe:
            return False, "auto_land_non_vision_gate_failed"
        if not self._terminal_horizontal_kinematics_safe(estimate):
            return False, "auto_land_horizontal_kinematics_invalid"
        if not self._terminal_occlusion_estimate_safe(estimate):
            return False, "auto_land_odometry_position_confidence_invalid"
        floor = self._float("touchdown_contact_clearance_m")
        if not self._touchdown_contact_ready(now, estimate, floor) and not (
            self._terminal_auto_land_contact_envelope_safe(now, estimate)
        ):
            return False, "auto_land_contact_conditions_invalid"
        if not self._contact_lidar_within_latch():
            return False, "auto_land_lidar_rebound_exceeded"
        return True, "px4_auto_land_active"

    def _terminal_auto_land_kinematics_failure_persisted(
        self, now: float
    ) -> bool:
        """Reject only a sustained post-handoff fused-kinematics failure."""
        started = self._terminal_auto_land_kinematics_unsafe_since
        if started is None or not math.isfinite(float(started)):
            self._terminal_auto_land_kinematics_unsafe_since = float(now)
            return False
        return bool(
            float(now) - float(started)
            >= self._float("touchdown_dwell_s")
        )

    def _clear_terminal_auto_land_kinematics_failure(self) -> None:
        self._terminal_auto_land_kinematics_unsafe_since = None

    def _terminal_auto_land_contact_envelope_safe(
        self,
        now: float,
        estimate: TargetEstimate | None,
    ) -> bool:
        """Bound the short PX4 LAND transient after a fully gated latch."""
        target_velocity = self._control_target_velocity_enu(estimate, now)
        if (
            estimate is None
            or not bool(getattr(estimate, "valid", False))
            or not bool(getattr(estimate, "odometry_fresh", False))
            or target_velocity is None
            or not self._fresh_lidar(now)
            or not self._terminal_horizontal_kinematics_safe(estimate)
        ):
            return False
        vehicle_velocity = np.asarray(self.vehicle_velocity_enu, dtype=float)
        if (
            vehicle_velocity.shape != (3,)
            or target_velocity.shape != (3,)
            or not np.all(np.isfinite(vehicle_velocity))
            or not np.all(np.isfinite(target_velocity))
        ):
            return False
        relative_vertical_speed = abs(
            float(vehicle_velocity[2] - target_velocity[2])
        )
        return bool(
            relative_vertical_speed
            <= self._float("landing_p_vertical_velocity_limit_m_s")
        )

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

    def _fail_terminal_auto_land(self, now: float, reason: str) -> None:
        """Fail closed without fighting PX4 after it accepted AUTO.LAND."""
        del now
        self._terminal_auto_land_cancel_reason = str(reason)
        self.last_safety_failure_reason = str(reason)
        self._enter_failsafe_hold(str(reason))

    def _touchdown_contact_ready(
        self,
        now: float,
        estimate: TargetEstimate | None,
        clearance_m: float,
    ) -> bool:
        """Require coherent low/slow geometry before contact settling."""
        target_velocity = self._control_target_velocity_enu(estimate, now)
        floor = self._float("touchdown_contact_clearance_m")
        maximum_deck_distance = self._float(
            "touchdown_max_deck_distance_m"
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
            or measured_clearance
            > floor + maximum_deck_distance + 1.0e-6
            or not self._fresh_lidar(now)
            or self.down_lidar_range_m is None
            or self.down_lidar_range_m > maximum_deck_distance
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
            or not self._fresh_lidar(now)
            or self.down_lidar_range_m is None
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
        maximum_deck_distance = self._float(
            "touchdown_max_deck_distance_m"
        )
        measured_clearance = self._measured_target_clearance(
            estimate, math.inf
        )
        if (
            not math.isfinite(measured_clearance)
            or measured_clearance
            > floor + maximum_deck_distance + 1.0e-6
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
            math.isfinite(float(self.down_lidar_range_m))
            and float(self.down_lidar_range_m)
            <= self._float("touchdown_contact_evidence_lidar_m")
            and relative_vertical_velocity <= 0.05
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
            and self._fresh_lidar(now)
            and self._contact_lidar_within_latch()
        )

    def _arm_terminal_contact_entry(self) -> None:
        """Latch contact and revoke every older ordinary FINAL solve."""
        if self._contact_entry_latched:
            return
        self._contact_entry_latched = True
        was_active = self.terminal_contact_bridge_active
        self.terminal_contact_bridge_active = True
        self.last_terminal_contact_bridge_reason = (
            "terminal_contact_entry_latched"
        )
        if not was_active:
            self.terminal_contact_bridge_activation_count += 1
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
            and self._fresh_lidar(now)
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
            lidar = self.down_lidar_range_m
            relative_vertical_speed = (
                math.inf
                if vehicle_velocity.shape != (3,)
                or not np.all(np.isfinite(vehicle_velocity))
                else abs(float(vehicle_velocity[2] - target_velocity[2]))
            )
            physical_contact_evidence = bool(
                self._fresh_lidar(now)
                and lidar is not None
                and math.isfinite(float(lidar))
                and float(lidar)
                <= self._float("touchdown_contact_evidence_lidar_m")
                and relative_vertical_speed <= 0.05
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
        staged = self._publish_trailer_relative_setpoint(
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
            lidar = self.down_lidar_range_m
            if lidar is not None and math.isfinite(float(lidar)):
                if self._contact_settle_min_lidar_range_m is None:
                    self._contact_settle_min_lidar_range_m = float(lidar)
                else:
                    self._contact_settle_min_lidar_range_m = min(
                        self._contact_settle_min_lidar_range_m,
                        float(lidar),
                    )
        self.descent_clearance_m = floor
        # A fresh phase-transition coast or a newly queued immutable contact
        # snapshot is an acceptable short bridge while the first contact QP
        # runs.  Neither starts the contact timeout or claims a successful
        # solver result; the next tick still has to observe a validated cache.
        return bool(solver_valid or transition_valid or snapshot_staged)

    def _stage_level_recovery_reference(self, now: float) -> None:
        """Queue a level moving-pad solve without changing mission phase."""
        self._publish_trailer_relative_setpoint(
            now,
            self.descent_clearance_m,
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
            if self.marker_loss_started_mission_s is None:
                self.marker_loss_started_mission_s = loss_now
            if self.phase == LandingPhase.FINAL_APPROACH:
                self.terminal_contact_bridge_active = True
                self.last_terminal_contact_bridge_reason = (
                    "terminal_level_recenter"
                )
            staged = self._publish_trailer_relative_setpoint(
                time.monotonic(),
                self.descent_clearance_m,
                "down",
                disable_descent=True,
                require_solver_success=False,
                relative_descent_speed_m_s=0.0,
            )
            self.last_marker_loss_policy = "level_track_recenter"
            if staged is not None:
                return
            elapsed = loss_now - self.marker_loss_started_mission_s
            if elapsed < self._float("vision_velocity_reacquisition_gap_s"):
                return
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
        if "target_estimate_invalid" in failures:
            loss_now = time.monotonic()
            if self._target_sensors_loss_persisted(loss_now, estimate):
                self._enter_failsafe_hold("target_sensors_timeout")
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
            float(self.down_lidar_range_m)
            if self._fresh_lidar(now)
            and self.down_lidar_range_m is not None
            else math.nan
        )
        ground, landed, at_rest, sample_stamp, source = (
            self._px4_land_detector_signals(now)
        )
        self.last_land_detector_source = source
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
        now = time.monotonic()
        mission_now = self._mission_time_s()
        # Keep the simulator-only Gazebo->ROS position bridge in the same
        # deterministic 50 Hz path as mission state evaluation.  The former
        # daemon worker could silently stop while Gazebo odometry remained
        # live, which made a healthy moving target look lost.
        latest_px4_sample_time_s = self.state_history.latest_common_time_s
        trailer_sample_limit_s = (
            mission_now
            if latest_px4_sample_time_s is None
            else float(latest_px4_sample_time_s)
        )
        self.trailer_bridge.tick(trailer_sample_limit_s)
        # A direct trailer sample is stamped from inside ``tick``. Refresh the
        # tick time afterwards so its age cannot be transiently negative and
        # make a newly received target fail the readiness/freshness gate.
        now = time.monotonic()
        state = self.mavros_state
        if self.phase == LandingPhase.LANDED:
            return
        if self.phase == LandingPhase.FAILSAFE_HOLD:
            if self.failsafe_hold_position_enu is not None:
                self._publish_hold(self.failsafe_hold_position_enu)
            return

        if self.phase == LandingPhase.TOUCHDOWN_CONFIRM and state is not None:
            if not state.armed:
                if self.touchdown_disarm_requested:
                    self.command_cache = None
                    self._pending_command = None
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
                return

        if self.phase in {LandingPhase.WAITING, LandingPhase.READY}:
            if not self._ready(now):
                self.phase = LandingPhase.WAITING
                return
        elif self.phase in {LandingPhase.PRESTREAM, LandingPhase.ARMING}:
            if not self._ready(now):
                self._enter_failsafe_hold("preflight state became invalid")
                return
        else:
            failure = self._critical_flight_failure(now)
            if failure is not None:
                self._enter_failsafe_hold(failure)
                return

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
            return

        if self.phase == LandingPhase.WAITING:
            self._transition(
                LandingPhase.READY, "PX4 and experiment sensors ready"
            )
        if self.phase == LandingPhase.READY:
            if not self.start_requested:
                return
            self.takeoff_origin = self.vehicle_position_enu.copy()
            self._transition(LandingPhase.PRESTREAM, "start request accepted")

        if self.phase == LandingPhase.PRESTREAM:
            self._publish_hold(self.vehicle_position_enu)
            if mission_now - self.phase_started >= self._float(
                "offboard_prestream_s"
            ):
                self._transition(LandingPhase.ARMING, "setpoint stream warm")
            return

        if self.phase == LandingPhase.ARMING:
            self._publish_hold(self.vehicle_position_enu)
            if state is None:
                return
            if state.mode != "OFFBOARD":
                self._request_mode("OFFBOARD", now)
                return
            if not state.armed:
                self._request_arm(True, now)
                return
            self._transition(LandingPhase.TAKEOFF, "armed in OFFBOARD")

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
            self._solve_and_publish(
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

        if self.phase == LandingPhase.ABORT_CLIMB:
            target = self.abort_climb_target_enu
            if target is None:
                self._enter_failsafe_hold("abort climb target unavailable")
                return
            success = self._solve_and_publish(
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

        if self.phase == LandingPhase.APPROACH:
            down_marker_visible = bool(
                self._marker_world_enu("down", now) is not None
            )
            result = self._publish_position_only_acquisition_setpoint(
                now,
                self._float("marker_search_height_m"),
                use_down_marker=down_marker_visible,
            )
            if (
                result is not None
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
                self.descent_clearance_m = self._measured_target_clearance(
                    self._current_target_estimate(),
                    self._float("marker_search_height_m"),
                )
                self._transition(
                    LandingPhase.MARKER_TRACK_DOWN,
                    "trailer approach complete; begin down-marker search",
                )
            return

        if self.phase == LandingPhase.MARKER_TRACK_DOWN:
            estimate = self._current_target_estimate()
            # Latch search clearance when odometry transit completes.
            # Recomputing it from every measured vehicle pose turned any
            # upward disturbance into a permanent reference increase and
            # produced vertical bobbing while marker acquisition was still
            # in progress.
            tracking_clearance = self.descent_clearance_m
            marker_pose = self._marker_world_enu("down", now)
            marker_velocity = self._control_target_velocity_enu(
                estimate, now
            )
            acquisition_sensor_ready = bool(
                marker_pose is not None
                and marker_velocity is not None
                and self._vision_position_covariance_safe()
                and self.control_target_velocity_source
                in {"aruco_capture_kalman", "aruco_capture_hold"}
            )
            if not self.marker_tracking_latched:
                # Keep one controller active while the optical motion estimate
                # matures.  Switching P/OSQP on every detector Bool produced a
                # repeatable 1 m -> 8 m limit cycle before either controller
                # could settle.  One continuous second of pose and ArUco-only
                # velocity is required before Relative OSQP owns XY motion.
                if not self._trailer_fresh(now):
                    self._transition(
                        LandingPhase.APPROACH,
                        "down marker and trailer odometry stale",
                    )
                    return
                acquisition_result = (
                    self._publish_position_only_acquisition_setpoint(
                        now,
                        tracking_clearance,
                        use_down_marker=marker_pose is not None,
                        target_velocity_enu_m_s=marker_velocity,
                    )
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
                elif self.marker_tracking_acquisition_since_s is None:
                    self.marker_tracking_acquisition_since_s = mission_now
                elif (
                    mission_now - self.marker_tracking_acquisition_since_s
                    >= self._float("precision_alignment_dwell_s")
                ):
                    # This dwell already proves continuous ArUco pose,
                    # camera-derived speed, lateral alignment, relative speed,
                    # and yaw.  Enter the level ALIGN state immediately;
                    # making MARKER_TRACK_DOWN repeat the same gate under a
                    # second controller left the mission oscillating forever
                    # between acquisition and OSQP despite a stable track.
                    self.descent_clearance_m = tracking_clearance
                    self._transition(
                        LandingPhase.PRECISION_ALIGN,
                        "ArUco track and deck-relative motion held",
                    )
                    return
                self._gate_dwell(
                    False,
                    mission_now,
                    self._float("precision_alignment_dwell_s"),
                )
                return

            if marker_velocity is None:
                # The bounded ArUco velocity cache expired. Revoke MPC
                # ownership and return to live trailer-position pursuit; no
                # simulator/trailer velocity is substituted.
                self.marker_tracking_latched = False
                self.marker_tracking_acquisition_since_s = None
                self._publish_position_only_acquisition_setpoint(
                    now,
                    tracking_clearance,
                    use_down_marker=False,
                )
                self.last_marker_loss_policy = (
                    "marker_search_odometry_track_reacquire"
                )
                self._gate_dwell(
                    False,
                    mission_now,
                    self._float("precision_alignment_dwell_s"),
                )
                return

            # After the stable handoff, a short optical gap keeps level
            # Relative OSQP tracking from the bounded ArUco velocity cache and
            # fresh odometry position instead of toggling back to P per frame.
            result = self._publish_trailer_relative_setpoint(
                now,
                tracking_clearance,
                "down" if marker_pose is not None else None,
                require_solver_success=False,
            )
            if result is None:
                self.marker_tracking_latched = False
                self.marker_tracking_acquisition_since_s = None
                self._publish_position_only_acquisition_setpoint(
                    now, tracking_clearance, use_down_marker=False
                )
                self._gate_dwell(
                    False,
                    mission_now,
                    self._float("precision_alignment_dwell_s"),
                )
                return
            alignment_vision_pose = self._fresh_down_vision_pose(now)
            alignment_vision_ready = bool(
                alignment_vision_pose is not None
                and self._vision_position_covariance_safe()
            )
            alignment_entry_ready = bool(
                result is not None
                and estimate is not None
                and self._trailer_fresh(now)
                and alignment_vision_ready
                and self.control_target_velocity_source
                in {"aruco_capture_kalman", "aruco_capture_hold"}
                and self._solver_cache_safe_for_descent(time.monotonic())
                and result[0]
                <= self._float("marker_track_entry_lateral_gate_m")
                and result[1]
                <= self._float("marker_track_entry_relative_speed_m_s")
                and result[2]
                <= self._float("marker_track_entry_yaw_error_rad")
            )
            if self._gate_dwell(
                alignment_entry_ready,
                mission_now,
                self._float("precision_alignment_dwell_s"),
            ):
                # Seed the first authorized descent from the measured level.
                # Later recovery cycles preserve the last clearance that was
                # commanded while every safety gate was valid.
                self.descent_clearance_m = tracking_clearance
                self._transition(
                    LandingPhase.PRECISION_ALIGN,
                    "lateral and relative-speed gates held",
                )
            return

        if self.phase == LandingPhase.PRECISION_ALIGN:
            if (
                self.terminal_recovery_started_mission_s is not None
                and mission_now
                - self.terminal_recovery_started_mission_s
                >= self._float("final_approach_timeout_s")
            ):
                # Terminal recovery has one bounded state lifetime.  Do not
                # let intermittent marker frames or lidar-edge noise restart
                # or bypass that timeout.
                self._start_abort_climb(
                    "terminal optical recovery timeout",
                    self._current_target_estimate(),
                )
                return
            if self._marker_world_enu("down", now) is None:
                estimate = self._current_target_estimate()
                measured_clearance = self._measured_target_clearance(
                    estimate,
                    self.descent_clearance_m,
                )
                if measured_clearance > (
                    self._float("terminal_occlusion_max_clearance_m")
                ):
                    # A single rejected camera frame must stop descent but
                    # must not restart the whole one-second acquisition
                    # sequence.  Coast horizontally for at most one optical
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
                    self._publish_trailer_relative_setpoint(
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
                        >= self._float(
                            "vision_velocity_reacquisition_gap_s"
                        )
                    ):
                        self._transition(
                            LandingPhase.MARKER_TRACK_DOWN,
                            "down marker loss exceeded one second",
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
                    tracking_result = self._publish_trailer_relative_setpoint(
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
                occlusion_lidar = self._float(
                    "terminal_occlusion_max_lidar_m"
                )
                terminal_recovery_envelope = bool(
                    relative_height is not None
                    and math.isfinite(relative_height)
                    and self.descent_clearance_m <= occlusion_height
                    and relative_height <= occlusion_height
                    and estimate is not None
                    and estimate.odometry_fresh
                    and self._fresh_lidar(now)
                    and self.down_lidar_range_m is not None
                    and math.isfinite(float(self.down_lidar_range_m))
                    and float(self.down_lidar_range_m) <= occlusion_lidar
                )
                if terminal_recovery_envelope:
                    if self.terminal_recovery_started_mission_s is None:
                        self.terminal_recovery_started_mission_s = mission_now
                    # One large marker necessarily leaves the finite camera
                    # FOV near contact. FINAL_APPROACH already contains the
                    # strict lidar + covariance + odometry optical bridge,
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
            result = self._publish_trailer_relative_setpoint(
                now, alignment_clearance, "down"
            )
            # Synchronous P control above timestamps its cache after the
            # tick-start ``now``.  A fresh monotonic sample prevents a valid
            # same-tick command from appearing to have a negative age.
            decision = self._descent_permission(time.monotonic(), estimate)
            within = bool(
                result is not None
                and decision.allowed
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
                    # lidar + odometry bridge directly instead of spending an
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

        if self.phase == LandingPhase.PRECISION_DESCENT:
            estimate = self._current_target_estimate()
            # A phase transition invalidates the previous asynchronous result.
            # Queue a level moving-pad solve before checking descent permission
            # so the heartbeat keeps coasting with the 3 m/s deck while the
            # first constrained result is pending. A world-frame HOLD here
            # creates an immediate relative-speed error and infeasible QPs.
            if not self._solver_cache_safe_for_descent(now):
                self._publish_trailer_relative_setpoint(
                    now,
                    self.descent_clearance_m,
                    (
                        "down"
                        if self._marker_world_enu("down", now) is not None
                        else None
                    ),
                )
            decision = self._descent_permission(now, estimate)
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
                    if self.marker_loss_started_mission_s is None:
                        self.marker_loss_started_mission_s = mission_now
                    self._publish_trailer_relative_setpoint(
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
                        >= self._float(
                            "vision_velocity_reacquisition_gap_s"
                        )
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
            self._publish_trailer_relative_setpoint(
                now, mpc_clearance, "down"
            )
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

        if self.phase == LandingPhase.FINAL_APPROACH:
            estimate = self._current_target_estimate()
            decision = self._descent_permission(now, estimate)

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

            # Once PX4 confirms the explicitly requested AUTO.LAND handoff,
            # it owns thrust reduction and land detection.  Do not let a
            # stale OFFBOARD command race or fight that native landing path.
            if self._terminal_auto_land_mode_active():
                self._accept_terminal_auto_land_engagement(now)
                _ground, px4_landed, _at_rest, _stamp, source = (
                    self._px4_land_detector_signals(now)
                )
                self.last_land_detector_source = source
                if px4_landed:
                    self.descent_clearance_m = self._float(
                        "touchdown_contact_clearance_m"
                    )
                    self.touchdown_gate.reset(self.time_reset_count)
                    self.touchdown_disarm_requested = False
                    self._transition(
                        LandingPhase.TOUCHDOWN_CONFIRM,
                        "PX4 AUTO.LAND reported ON_GROUND; begin dwell",
                    )
                    return
                takeover_safe, takeover_reason = (
                    self._terminal_auto_land_safety_permission(
                        now, estimate, decision
                    )
                )
                if not takeover_safe:
                    if (
                        takeover_reason
                        == "auto_land_horizontal_kinematics_invalid"
                        and not self._terminal_auto_land_kinematics_failure_persisted(
                            now
                        )
                    ):
                        self.last_safety_failure_reason = (
                            "terminal_auto_land_horizontal_kinematics_transient"
                        )
                        return
                    self._fail_terminal_auto_land(
                        now, f"terminal_{takeover_reason}"
                    )
                    return
                self._clear_terminal_auto_land_kinematics_failure()
                self.last_safety_failure_reason = None
                return

            if self._bool("final_approach_requires_lidar") and not (
                self._fresh_lidar(now)
            ):
                if self._contact_settle_elapsed_s(now) is not None:
                    self._hold_descent_reference(now, estimate)
                    self._enter_failsafe_hold(
                        "terminal_contact_lidar_invalid"
                    )
                    return
                self._hold_descent_reference(now, estimate)
                self.last_safety_failure_reason = "final_lidar_invalid"
                if mission_now - self.phase_started >= self._float(
                    "final_approach_timeout_s"
                ):
                    self._start_abort_climb(
                        "final approach lidar timeout", estimate
                    )
                return
            final_descent_speed = self._final_descent_speed_from_lidar(now)
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
                    and self._fresh_lidar(now)
                    and self.down_lidar_range_m is not None
                    and math.isfinite(float(self.down_lidar_range_m))
                    and float(self.down_lidar_range_m)
                    <= maximum_deck_distance
                )
            px4_landed = False
            if (
                contact_ready
                or self._contact_entry_latched
                or settle_elapsed is not None
            ):
                _ground, px4_landed, _at_rest, _stamp, source = (
                    self._px4_land_detector_signals(now)
                )
                self.last_land_detector_source = source
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
                and not self._terminal_auto_land_mode_active()
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
                if not self._contact_lidar_within_latch():
                    self._start_abort_climb(
                        "touchdown contact deck envelope lost",
                        estimate,
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
                # fresh lidar and constrained Relative MPC.  Any additional
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
                    staged = self._publish_trailer_relative_setpoint(
                        now,
                        self.descent_clearance_m,
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
                occlusion_lidar = self._float(
                    "terminal_occlusion_max_lidar_m"
                )
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
                    and self._fresh_lidar(now)
                    and self.down_lidar_range_m is not None
                    and math.isfinite(float(self.down_lidar_range_m))
                    and float(self.down_lidar_range_m)
                    <= occlusion_lidar
                )
                # Level recentering must not require the lidar ray to already
                # hit the deck.  With lateral error the ray can see the ground
                # even though trailer odometry shows that the aircraft is
                # inside the terminal-height band.  Requiring lidar <= the
                # deck threshold here created a catch-22: recentering was
                # needed before the ray could hit the deck.  Descent below
                # still retains the strict deck-lidar envelope.
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
                    and self._fresh_lidar(now)
                    and self.down_lidar_range_m is not None
                    and math.isfinite(float(self.down_lidar_range_m))
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
                    self.last_terminal_contact_bridge_reason = (
                        "terminal_constrained_level_recenter"
                    )
                    self._publish_trailer_relative_setpoint(
                        now,
                        self.descent_clearance_m,
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
                    was_active = self.terminal_contact_bridge_active
                    self.terminal_contact_bridge_active = True
                    self.last_terminal_contact_bridge_reason = (
                        "single_marker_near_contact_occlusion_staging"
                    )
                    if not was_active:
                        self.terminal_contact_bridge_activation_count += 1
                    self._publish_trailer_relative_setpoint(
                        now,
                        self.descent_clearance_m,
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
                    was_active = self.terminal_contact_bridge_active
                    self.terminal_contact_bridge_active = True
                    self.last_terminal_contact_bridge_reason = (
                        "single_marker_near_contact_occlusion"
                    )
                    if not was_active:
                        self.terminal_contact_bridge_activation_count += 1
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
                    result = self._publish_trailer_relative_setpoint(
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
                    was_active = self.terminal_contact_bridge_active
                    self.terminal_contact_bridge_active = True
                    self.last_terminal_contact_bridge_reason = (
                        "terminal_geometry_level_recenter"
                    )
                    if not was_active:
                        self.terminal_contact_bridge_activation_count += 1
                    self._publish_trailer_relative_setpoint(
                        now,
                        self.descent_clearance_m,
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
                    was_active = self.terminal_contact_bridge_active
                    self.terminal_contact_bridge_active = True
                    self.last_terminal_contact_bridge_reason = bridge_reason
                    if not was_active:
                        self.terminal_contact_bridge_activation_count += 1
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
                        "lidar_rebound_exceeded",
                    }:
                        self.terminal_contact_bridge_active = True
                        self.last_terminal_contact_bridge_reason = (
                            "terminal_contact_level_recenter"
                        )
                        self._publish_trailer_relative_setpoint(
                            now,
                            self.descent_clearance_m,
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
            result = self._publish_trailer_relative_setpoint(
                now, mpc_clearance, "down"
            )
            self.descent_clearance_m = candidate_clearance
            return

        if self.phase == LandingPhase.TOUCHDOWN_CONFIRM:
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
                if (
                    state is not None
                    and state.armed
                    and self._request_arm(False, now)
                ):
                    self.normal_disarm_request_count += 1
                return
            decision = self._update_touchdown_gate(now, estimate)
            if decision.confirmed:
                if self._request_arm(False, now):
                    self.normal_disarm_request_count += 1
                    self.touchdown_disarm_requested = True
                return
            if decision.timed_out:
                if self._terminal_auto_land_mode_active():
                    self._fail_terminal_auto_land(
                        now, "touchdown_confirmation_timeout_auto_land"
                    )
                    return
                if state is not None and state.armed:
                    self._start_abort_climb(
                        "touchdown confirmation timeout", estimate
                    )
                else:
                    self._enter_failsafe_hold(
                        "touchdown timeout while disarmed"
                    )
            return

    def _publish_trailer_relative_setpoint(
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
        estimate = self._landing_target_enu(now, camera)
        if estimate is None:
            return None
        deck, trailer_velocity, trailer_yaw = estimate
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
                    self._final_descent_speed_from_lidar(now)
                )
            elif self.phase == LandingPhase.TOUCHDOWN_CONFIRM:
                relative_descent_speed_m_s = self._float(
                    "touchdown_contact_settle_speed_m_s"
                )
            else:
                relative_descent_speed_m_s = 0.0

        success = self._solve_and_publish(
            target,
            trailer_velocity,
            trailer_yaw,
            disable_descent=disable_descent,
            relative_descent_speed_m_s=relative_descent_speed_m_s,
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
            self.landing_pad_acceleration_source = "unavailable"
            return np.zeros(3, dtype=float)
        return acceleration.copy()

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
        # fresh lidar + measured trailer position near contact, keeping the
        # FOV polygon active makes the QP mathematically infeasible as height
        # approaches zero. Disable only that row family; funnel, deck,
        # velocity, acceleration, jerk and descent-alignment bounds remain.
        camera_fov_constraint_enabled = not bool(
            constraints_enabled
            and (
                (
                    self.phase == LandingPhase.FINAL_APPROACH
                    and self.terminal_contact_bridge_active
                )
                or self.phase == LandingPhase.TOUCHDOWN_CONFIRM
            )
        )
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
        trailer_pose = self._latest_trailer_marker_pose(time.monotonic())
        if trailer_pose is not None:
            deck_position = np.asarray(trailer_pose[0], dtype=float)
        elif (
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

    def _final_descent_speed_from_lidar(self, now: float) -> float:
        """Taper terminal descent continuously before landing-gear contact."""
        nominal = self._float("final_descent_speed_m_s")
        settle = self._float("touchdown_contact_settle_speed_m_s")
        if not self._fresh_lidar(now) or self.down_lidar_range_m is None:
            return nominal
        distance = float(self.down_lidar_range_m)
        # The live vehicle contacts the deck at roughly 0.03 m lidar range.
        # A deliberately conservative 0.03 m/s^2 braking envelope is active
        # throughout FINAL.  The previous 0.08 value still reached the 0.25 m
        # contact gate at 0.34 m/s and physical contact at 0.26 m/s, so the
        # low/slow contact controller never became eligible before rebound.
        braking_distance = max(0.0, distance - 0.05)
        braking_speed = math.sqrt(2.0 * 0.03 * braking_distance)
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
        if not self._bool("final_approach_requires_lidar"):
            return True
        lidar = getattr(self, "down_lidar_range_m", None)
        return bool(
            self._fresh_lidar(now)
            and lidar is not None
            and math.isfinite(float(lidar))
            and float(lidar) <= final_height + 1.0e-6
        )

    def _snapshot_solver_deadline_s(
        self, control_input: LandingControlInput
    ) -> float:
        """Return the active controller's full worker deadline."""
        configured = str(
            self.get_parameter("landing_controller_type").value
        )
        if (
            configured == RELATIVE_LANDING_MPC_CONTROLLER_TYPE
            and control_input.has_landing_pad_context
        ):
            parameter = (
                "landing_mpc_deadline_ms"
                if self.landing_mpc_solver_name == "osqp"
                else "relative_landing_mpc_deadline_ms"
            )
        elif configured == P_FEEDFORWARD_CONTROLLER_TYPE:
            parameter = "landing_mpc_deadline_ms"
        else:
            parameter = "legacy_mpc_deadline_ms"
        return 0.001 * self._float(parameter)

    def _snapshot_command_limits(
        self, control_input: LandingControlInput
    ) -> LandingCommandLimits:
        """Capture command-level bounds for post-solve revalidation."""
        configured = str(
            self.get_parameter("landing_controller_type").value
        )
        if configured == P_FEEDFORWARD_CONTROLLER_TYPE:
            return LandingCommandLimits(
                control_input.horizontal_velocity_limit_m_s,
                control_input.vertical_velocity_limit_m_s,
                control_input.vertical_velocity_limit_m_s,
            )
        if (
            configured == RELATIVE_LANDING_MPC_CONTROLLER_TYPE
            and control_input.has_landing_pad_context
        ):
            solver = self.relative_landing_mpc
            return LandingCommandLimits(
                solver.max_horizontal_velocity_m_s,
                solver.max_ascent_velocity_m_s,
                solver.max_descent_velocity_m_s,
            )
        limit = float(self.mpc.velocity_limit_m_s)
        return LandingCommandLimits(limit, limit, limit, limit)

    def _stage_solver_snapshot(
        self, control_input: LandingControlInput
    ) -> LandingSolveSnapshot:
        """Replace the latest-wins queue with one deep-frozen snapshot."""
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
            constraint_limits=self._snapshot_command_limits(control_input),
        )
        self._pending_solver_snapshot = snapshot
        self._latest_solver_snapshot = snapshot
        return snapshot

    def _record_landing_command(
        self, command: LandingControlCommand
    ) -> None:
        """Update controller/solver counters on the ROS executor thread."""
        self.landing_control_compute_count += 1
        self.last_landing_control_status = command.status
        self.last_landing_control_valid = command.valid
        self.last_landing_control_degraded = command.degraded
        self.last_landing_control_type = command.controller_type
        self.last_landing_control_solve_time_s = command.solve_time_s
        if command.valid:
            self.landing_control_valid_count += 1
        else:
            self.landing_control_invalid_count += 1
        if command.degraded:
            self.landing_control_degraded_count += 1
        if command.mpc_attempted:
            self.mpc_solve_count += 1
            self.last_mpc_message = command.status
            self.last_mpc_success = bool(command.mpc_success)
            self.last_mpc_solve_time_s = command.solve_time_s
            self.last_mpc_iterations = command.iterations
            self.last_mpc_deadline_missed = command.deadline_missed
            if command.deadline_missed:
                self.mpc_deadline_miss_count += 1
            if command.mpc_success:
                self.mpc_success_count += 1
            else:
                self.mpc_failure_count += 1

    def _cache_synchronous_p_failure(self, reason: str) -> bool:
        """Cache a fail-closed P error without charging solver counters."""
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

    def _solve_and_publish(
        self,
        reference_position_enu: Sequence[float],
        reference_velocity_enu: Sequence[float],
        yaw_enu_rad: float,
        *,
        disable_descent: bool = False,
        relative_descent_speed_m_s: float = 0.0,
    ) -> bool:
        if self.vehicle_position_enu is None:
            return False
        configured_controller = str(
            self.get_parameter("landing_controller_type").value
        )
        reference_position = np.asarray(reference_position_enu, dtype=float)
        reference_velocity = np.asarray(reference_velocity_enu, dtype=float)
        relative_context = self._relative_landing_context(
            reference_position,
            reference_velocity,
            disable_descent=disable_descent,
            relative_descent_speed_m_s=relative_descent_speed_m_s,
        )
        synchronous_p_path = bool(
            configured_controller == P_FEEDFORWARD_CONTROLLER_TYPE
            or (
                configured_controller
                == RELATIVE_LANDING_MPC_CONTROLLER_TYPE
                and relative_context is None
            )
        )
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
            self.last_landing_pad_position_enu = pad_position.copy()
            self.last_landing_pad_velocity_enu = pad_velocity.copy()
            self.last_landing_pad_acceleration_enu = pad_acceleration.copy()
            self.last_landing_relative_position_enu = (
                self.vehicle_position_enu - pad_position
            )
            self.last_landing_relative_velocity_enu = (
                self.vehicle_velocity_enu - pad_velocity
            )
            self.last_landing_relative_acceleration_enu = (
                self.vehicle_acceleration_enu - pad_acceleration
            )
            self.last_relative_mpc_constraints_enabled = (
                constraints_enabled
            )
            self.last_relative_mpc_descent_allowed = descent_allowed
        else:
            self.last_landing_pad_position_enu = None
            self.last_landing_pad_velocity_enu = None
            self.last_landing_pad_acceleration_enu = None
            self.last_landing_relative_position_enu = None
            self.last_landing_relative_velocity_enu = None
            self.last_landing_relative_acceleration_enu = None
            self.last_relative_mpc_constraints_enabled = False
            self.last_relative_mpc_descent_allowed = False
        try:
            horizontal_velocity_limit_m_s = self._float(
                "landing_p_horizontal_velocity_limit_m_s"
            )
            if (
                configured_controller
                == RELATIVE_LANDING_MPC_CONTROLLER_TYPE
                and relative_context is not None
            ):
                # The solver-failure moving-deck HOLD consumes this shared
                # limit.  Feeding it the 1 m/s P-controller cap forced a
                # 3 m/s trailer command down to exactly 1 m/s at every failed
                # solve, creating the measured brake/catch-up oscillation.
                horizontal_velocity_limit_m_s = self._float(
                    "relative_landing_mpc_max_horizontal_velocity_m_s"
                )
            control_input = LandingControlInput(
                vehicle_position_enu_m=self.vehicle_position_enu,
                vehicle_velocity_enu_m_s=self.vehicle_velocity_enu,
                vehicle_acceleration_enu_m_s2=(
                    self.vehicle_acceleration_enu
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
        self.last_reference_enu = (
            reference_position.copy()
            if reference_position.ndim == 1
            else reference_position[0].copy()
        )
        return success

    def _publish_hold(self, position_enu: Sequence[float]) -> None:
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
        self._pending_command = command
        self.command_generation += 1

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
        if not command.acceleration_enabled:
            message.type_mask |= (
                PositionTarget.IGNORE_AFX
                | PositionTarget.IGNORE_AFY
                | PositionTarget.IGNORE_AFZ
            )
        message.position.x, message.position.y, message.position.z = position
        message.velocity.x, message.velocity.y, message.velocity.z = velocity
        message.acceleration_or_force.x = acceleration[0]
        message.acceleration_or_force.y = acceleration[1]
        message.acceleration_or_force.z = acceleration[2]
        message.yaw = command.yaw_enu_rad
        self.setpoint_pub.publish(message)
        published_at = time.monotonic()
        self._setpoint_timestamps.append(published_at)
        self.last_command_position_enu = position.copy()
        self.last_command_velocity_enu = velocity.copy()
        self.last_command_acceleration_enu = acceleration.copy()
        self.last_command_yaw_enu_rad = command.yaw_enu_rad
        self.last_command_stamp = published_at

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
            # FINAL_APPROACH owns the bounded lidar/odometry optical bridge.
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
        preserve_auto_land = False
        if getattr(self, "terminal_auto_land_requested", False):
            state = getattr(self, "mavros_state", None)
            auto_land_mode = str(
                self.get_parameter("touchdown_auto_land_mode").value
            )
            preserve_auto_land = bool(
                state is not None
                and state.armed
                and state.mode == auto_land_mode
                and phase
                in {
                    LandingPhase.TOUCHDOWN_CONFIRM,
                    LandingPhase.FAILSAFE_HOLD,
                }
            )
        self.phase = phase
        if phase == LandingPhase.MARKER_TRACK_DOWN:
            self.marker_tracking_acquisition_since_s = None
            self.marker_tracking_latched = False
        elif previous == LandingPhase.MARKER_TRACK_DOWN:
            self.marker_tracking_acquisition_since_s = None
            self.marker_tracking_latched = False
        reset_terminal = getattr(
            self, "_reset_terminal_contact_context", None
        )
        if reset_terminal is not None:
            reset_terminal(preserve_auto_land=preserve_auto_land)
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
            self._pending_command = None
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
            self._pending_command = None
        self.phase_started = transition_mission_time_s
        self.last_transition_reason = str(explanation)
        self.within_gate_since = None
        self.get_logger().info(
            f"landing phase {previous.value} -> {phase.value}: {explanation}"
        )

    def _status_tick(self) -> None:
        now = time.monotonic()
        self.last_status_publish = now
        state = self.mavros_state
        trailer = self.trailer_state
        target = self._current_target_estimate()
        command = self.command_cache
        fusion_statistics = self.target_estimator.statistics
        pending_snapshot = self._pending_solver_snapshot
        worker = self.solver_worker
        outstanding_queue_age_s = (
            None if worker is None else worker.queue_age_s
        )
        if outstanding_queue_age_s is not None:
            solver_queue_age_s = outstanding_queue_age_s
        elif pending_snapshot is not None:
            solver_queue_age_s = max(
                0.0, now - pending_snapshot.created_monotonic_s
            )
        else:
            solver_queue_age_s = None
        cached_snapshot = self._cached_solver_snapshot
        if cached_snapshot is not None:
            solution_age_s = max(
                0.0,
                now - cached_snapshot.created_monotonic_s,
            )
        elif self._cached_p_command_monotonic_s is not None:
            solution_age_s = max(
                0.0,
                now - self._cached_p_command_monotonic_s,
            )
        elif self._acquisition_command_monotonic_s is not None:
            solution_age_s = max(
                0.0,
                now - self._acquisition_command_monotonic_s,
            )
        else:
            solution_age_s = None
        heartbeat_rate_hz = self._measured_rate_hz(
            self._heartbeat_timestamps, now
        )
        setpoint_rate_hz = self._measured_rate_hz(
            self._setpoint_timestamps, now
        )
        ground_contact, px4_landed, px4_at_rest, detector_stamp, (
            detector_source
        ) = self._px4_land_detector_signals(now)
        detector_age_s = (
            None
            if detector_stamp <= 0.0
            else max(0.0, now - detector_stamp)
        )
        descent_decision = self.last_descent_permission
        touchdown_decision = self.last_touchdown_decision

        def source_age(source: str) -> float | None:
            accepted_stamp = fusion_statistics[source].last_accepted_stamp_s
            if target is None or accepted_stamp is None:
                return None
            return max(0.0, target.stamp_s - accepted_stamp)

        def source_statistics(source: str) -> dict[str, object]:
            statistics = fusion_statistics[source]
            return {
                "received": statistics.received,
                "queued": statistics.queued,
                "accepted": statistics.accepted,
                "rejected": statistics.rejected,
                "duplicate": statistics.duplicate,
                "late": statistics.late,
                "invalid_covariance": statistics.invalid_covariance,
                "invalid_value": statistics.invalid_value,
                "stale": statistics.stale,
                "future": statistics.future,
                "jump": statistics.jump,
                "nis_rejected": statistics.nis,
                "last_received_stamp_s": (
                    statistics.last_received_stamp_s
                ),
                "last_accepted_stamp_s": (
                    statistics.last_accepted_stamp_s
                ),
                "last_accepted_age_s": source_age(source),
                "last_rejection_reason": statistics.last_rejection_reason,
                "last_nis": statistics.last_nis,
                "last_position_innovation_enu_m": (
                    statistics.last_position_innovation_enu_m
                ),
                "last_position_innovation_norm_m": (
                    statistics.last_position_innovation_norm_m
                ),
                "last_yaw_innovation_rad": (
                    statistics.last_yaw_innovation_rad
                ),
                "last_position_marginal_nis": (
                    statistics.last_position_marginal_nis
                ),
                "last_yaw_marginal_nis": (
                    statistics.last_yaw_marginal_nis
                ),
            }

        configured_controller = str(
            self.get_parameter("landing_controller_type").value
        )
        if configured_controller == "relative_landing_mpc":
            effective_solver: str | None = self.landing_mpc_solver_name
        elif configured_controller == "legacy_mpc":
            effective_solver = "legacy_slsqp"
        else:
            effective_solver = None
        fields = {
            "schema_version": 1,
            "control_stack": str(
                self.get_parameter("control_stack_name").value
            ),
            "baseline_run_id": str(
                self.get_parameter("baseline_run_id").value
            ),
            "vehicle_authority_id": str(
                self.get_parameter("vehicle_authority_id").value
            ),
            "validation_mode": self._bool("validation_mode"),
            "production_qualified": self._bool("production_qualified"),
            "activation_qualified": self._activation_qualified(),
            "landing_controller_type": configured_controller,
            "landing_mpc_solver": self.landing_mpc_solver_name,
            "landing_mpc_solver_effective": effective_solver,
            "landing_mpc_workspace_setup_count": getattr(
                self.relative_landing_mpc,
                "workspace_setup_count",
                None,
            ),
            "landing_mpc_workspace_update_count": getattr(
                self.relative_landing_mpc,
                "workspace_update_count",
                None,
            ),
            "landing_mpc_workspace_solve_count": getattr(
                self.relative_landing_mpc,
                "workspace_solve_count",
                None,
            ),
            "mpc_failure_fallback": str(
                self.get_parameter("mpc_failure_fallback").value
            ),
            "command_transport": "mavros_position_target",
            "command_topics": [
                "/mavros/setpoint_raw/local",
                "/mavros/set_mode",
                "/mavros/cmd/arming",
            ],
            "phase": self.phase.value,
            "last_transition_reason": self.last_transition_reason,
            "active": self.start_requested,
            "connected": None if state is None else bool(state.connected),
            "armed": None if state is None else bool(state.armed),
            "flight_mode": None if state is None else str(state.mode),
            "vehicle_position_enu_m": self.vehicle_position_enu,
            "vehicle_velocity_enu_m_s": self.vehicle_velocity_enu,
            "vehicle_acceleration_enu_m_s2": self.vehicle_acceleration_enu,
            "vehicle_attitude_source": (
                "interpolated_mavros_imu_px4_time_stage3"
            ),
            "capture_time_attitude_interpolation": True,
            "state_history_duration_s": self._float(
                "state_history_duration_s"
            ),
            "state_history_position_sample_count": (
                self.state_history.position_sample_count
            ),
            "state_history_velocity_sample_count": (
                self.state_history.velocity_sample_count
            ),
            "state_history_attitude_sample_count": (
                self.state_history.attitude_sample_count
            ),
            "state_history_latest_common_px4_time_s": (
                self.state_history.latest_common_time_s
            ),
            "camera_to_px4_time_offset_ready": self.camera_time_offset.ready,
            "camera_to_px4_time_offset_s": self.camera_time_offset.offset_s,
            "camera_to_px4_time_offset_sample_count": (
                self.camera_time_offset.sample_count
            ),
            "camera_to_px4_time_offset_residual_s": (
                self.last_camera_time_offset_residual_s
            ),
            "mavros_timesync_ready": self.mavros_timesync.ready,
            "mavros_timesync_sample_count": self.mavros_timesync.sample_count,
            "mavros_timesync_estimated_offset_ns": (
                self.mavros_timesync.estimated_offset_ns
            ),
            "mavros_timesync_age_s": (
                None
                if self.mavros_timesync_stamp <= 0.0
                else max(0.0, now - self.mavros_timesync_stamp)
            ),
            "time_reset_count": self.time_reset_count,
            "last_time_reset_reason": self.last_time_reset_reason,
            "trailer_position_enu_m": (
                None if trailer is None else trailer.position_enu_m
            ),
            "trailer_velocity_enu_m_s": (
                None
            ),
            "trailer_acceleration_enu_m_s2": (
                None
            ),
            "trailer_velocity_available_to_controller": False,
            "control_target_velocity_source": (
                self.control_target_velocity_source
            ),
            "control_target_velocity_enu_m_s": (
                self.last_vision_velocity_enu
                if self.control_target_velocity_source.startswith("aruco_")
                else self.last_position_servo_velocity_enu
                if self.control_target_velocity_source.startswith(
                    "acquisition_"
                )
                else None
            ),
            "trailer_yaw_enu_rad": (
                None if trailer is None else trailer.yaw_enu_rad
            ),
            "trailer_age_s": (
                None
                if trailer is None or self.trailer_state_stamp <= 0.0
                else max(0.0, now - self.trailer_state_stamp)
            ),
            "trailer_fresh": self._trailer_fresh(now),
            "target_estimate_valid": (
                False if target is None else target.valid
            ),
            "target_estimate_mode": (
                "time_unavailable" if target is None else target.mode
            ),
            "target_estimate_position_enu_m": (
                None if target is None else target.position_enu_m
            ),
            "target_estimate_velocity_enu_m_s": (
                None
            ),
            "target_estimate_yaw_enu_rad": (
                None if target is None else target.yaw_enu_rad
            ),
            "target_estimate_yaw_rate_rad_s": (
                None if target is None else target.yaw_rate_rad_s
            ),
            "target_estimate_covariance_diagonal": (
                None
                if target is None or target.covariance is None
                else np.diag(target.covariance)
            ),
            "target_filter_stamp_px4_s": (
                None if target is None else target.filter_stamp_s
            ),
            "target_last_measurement_age_s": (
                None if target is None else target.last_measurement_age_s
            ),
            "target_vision_fresh": (
                False if target is None else target.vision_fresh
            ),
            "target_odometry_fresh": (
                False if target is None else target.odometry_fresh
            ),
            "target_fusion_pending_count": self.target_estimator.pending_count,
            "target_fusion_sources": {
                source: source_statistics(source)
                for source in (
                    "vision_front",
                    "vision_down",
                    "odometry",
                )
            },
            "trailer_marker_offset_m": [
                self._float("trailer_marker_offset_x_m"),
                self._float("trailer_marker_offset_y_m"),
                self._float("trailer_marker_offset_z_m"),
            ],
            "reference_position_enu_m": self.last_reference_enu,
            "command_position_enu_m": self.last_command_position_enu,
            "command_velocity_enu_m_s": self.last_command_velocity_enu,
            "command_acceleration_enu_m_s2": (
                self.last_command_acceleration_enu
            ),
            "command_yaw_enu_rad": self.last_command_yaw_enu_rad,
            "command_generation": self.command_generation,
            "command_position_enabled": (
                None if command is None else command.position_enabled
            ),
            "command_velocity_enabled": (
                None if command is None else command.velocity_enabled
            ),
            "command_acceleration_enabled": (
                None if command is None else command.acceleration_enabled
            ),
            "command_controller_type": (
                None if command is None else command.controller_type
            ),
            "command_primary_controller_type": (
                None
                if command is None
                else command.primary_controller_type
            ),
            "command_fallback_used": (
                None if command is None else command.fallback_used
            ),
            "command_age_s": (
                None
                if self.last_command_stamp <= 0.0
                else max(0.0, now - self.last_command_stamp)
            ),
            "landing_control_status": self.last_landing_control_status,
            "landing_control_valid": self.last_landing_control_valid,
            "landing_control_degraded": self.last_landing_control_degraded,
            "landing_control_effective_type": (
                self.last_landing_control_type
            ),
            "landing_control_solve_ms": (
                1000.0 * self.last_landing_control_solve_time_s
            ),
            "landing_control_compute_count": (
                self.landing_control_compute_count
            ),
            "landing_control_valid_count": self.landing_control_valid_count,
            "landing_control_degraded_count": (
                self.landing_control_degraded_count
            ),
            "landing_control_invalid_count": (
                self.landing_control_invalid_count
            ),
            "landing_horizontal_error_m": self.last_landing_horizontal_error_m,
            "landing_relative_horizontal_speed_m_s": (
                self.last_landing_relative_horizontal_speed_m_s
            ),
            "requested_clearance_m": self.last_requested_clearance_m,
            "touchdown_contact_settle_active": (
                self._contact_settle_elapsed_s(now) is not None
                and command is not None
                and command.controller_type
                == RELATIVE_LANDING_MPC_CONTROLLER_TYPE
                and self._cached_solver_command_valid(now)
            ),
            "touchdown_contact_settle_speed_m_s": self._float(
                "touchdown_contact_settle_speed_m_s"
            ),
            "touchdown_contact_settle_latched": (
                self._contact_settle_elapsed_s(now) is not None
            ),
            "touchdown_contact_settle_elapsed_s": (
                self._contact_settle_elapsed_s(now)
            ),
            "touchdown_contact_settle_timeout_s": self._float(
                "touchdown_contact_settle_timeout_s"
            ),
            "terminal_contact_bridge_active": (
                self.terminal_contact_bridge_active
            ),
            "terminal_contact_bridge_activation_count": (
                self.terminal_contact_bridge_activation_count
            ),
            "terminal_contact_bridge_reason": (
                self.last_terminal_contact_bridge_reason
            ),
            "terminal_auto_land_requested": (
                self.terminal_auto_land_requested
            ),
            "terminal_auto_land_engaged": (
                self._terminal_auto_land_mode_active()
            ),
            "terminal_auto_land_request_count": (
                self.terminal_auto_land_request_count
            ),
            "terminal_auto_land_engaged_age_s": (
                None
                if self.terminal_auto_land_engaged_monotonic_s is None
                else max(
                    0.0,
                    now - self.terminal_auto_land_engaged_monotonic_s,
                )
            ),
            "terminal_auto_land_timeout_s": self._float(
                "touchdown_auto_land_timeout_s"
            ),
            "offboard_setpoint_suppressed_for_auto_land": (
                self._terminal_auto_land_mode_active()
            ),
            "landing_pad_position_enu_m": (
                self.last_landing_pad_position_enu
            ),
            "landing_pad_velocity_enu_m_s": (
                self.last_landing_pad_velocity_enu
            ),
            "landing_pad_acceleration_enu_m_s2": (
                self.last_landing_pad_acceleration_enu
            ),
            "landing_pad_acceleration_source": (
                self.landing_pad_acceleration_source
            ),
            "landing_relative_position_enu_m": (
                self.last_landing_relative_position_enu
            ),
            "landing_relative_velocity_enu_m_s": (
                self.last_landing_relative_velocity_enu
            ),
            "landing_relative_acceleration_enu_m_s2": (
                self.last_landing_relative_acceleration_enu
            ),
            "relative_mpc_constraints_enabled": (
                self.last_relative_mpc_constraints_enabled
            ),
            "relative_mpc_descent_requested": (
                self.last_relative_mpc_descent_allowed
            ),
            "landing_mpc": self.last_mpc_message,
            "landing_mpc_success": self.last_mpc_success,
            "landing_mpc_solve_ms": 1000.0 * self.last_mpc_solve_time_s,
            "landing_mpc_iterations": self.last_mpc_iterations,
            "landing_mpc_deadline_missed": self.last_mpc_deadline_missed,
            "landing_mpc_solve_count": self.mpc_solve_count,
            "landing_mpc_success_count": self.mpc_success_count,
            "landing_mpc_failure_count": self.mpc_failure_count,
            "landing_mpc_deadline_miss_count": self.mpc_deadline_miss_count,
            "solver_running": bool(
                worker is not None and worker.solver_running
            ),
            "solver_queue_age_ms": (
                None
                if solver_queue_age_s is None
                else 1000.0 * solver_queue_age_s
            ),
            "solution_age_ms": (
                None
                if solution_age_s is None
                else 1000.0 * solution_age_s
            ),
            "stale_solution_rejected_count": (
                self.stale_solution_rejected_count
            ),
            "consecutive_solver_failures": (
                self.consecutive_solver_failures
            ),
            "descent_permission_allowed": (
                None
                if descent_decision is None
                else descent_decision.allowed
            ),
            "descent_permission_reason": (
                None
                if descent_decision is None
                else descent_decision.reason
            ),
            "descent_permission_failed_checks": (
                []
                if descent_decision is None
                else list(descent_decision.failed_checks)
            ),
            "last_safety_failure_reason": (
                self.last_safety_failure_reason
            ),
            "marker_loss_policy": self.last_marker_loss_policy,
            "lidar_fresh": self._fresh_lidar(now),
            "land_detector_source": detector_source,
            "land_detector_age_s": detector_age_s,
            "px4_ground_contact": ground_contact,
            "px4_landed": px4_landed,
            "px4_at_rest": px4_at_rest,
            "touchdown_conditions_met": (
                False
                if touchdown_decision is None
                else touchdown_decision.conditions_met
            ),
            "touchdown_confirmed": (
                False
                if touchdown_decision is None
                else touchdown_decision.confirmed
            ),
            "touchdown_confirmation_timed_out": (
                False
                if touchdown_decision is None
                else touchdown_decision.timed_out
            ),
            "touchdown_gate_reason": (
                None
                if touchdown_decision is None
                else touchdown_decision.reason
            ),
            "touchdown_dwell_elapsed_s": (
                None
                if touchdown_decision is None
                else touchdown_decision.dwell_elapsed_s
            ),
            "touchdown_consecutive_samples": (
                0
                if touchdown_decision is None
                else touchdown_decision.consecutive_samples
            ),
            "touchdown_disarm_requested": (
                self.touchdown_disarm_requested
            ),
            "normal_disarm_request_count": (
                self.normal_disarm_request_count
            ),
            "heartbeat_rate_hz": heartbeat_rate_hz,
            "setpoint_rate_hz": setpoint_rate_hz,
            "front_marker_age_s": (
                source_age("vision_front")
            ),
            "down_marker_age_s": (
                source_age("vision_down")
            ),
            "front_marker_capture_stamp_s": (
                self.front_marker_capture_guard.capture_stamp_s
            ),
            "down_marker_capture_stamp_s": (
                self.down_marker_capture_guard.capture_stamp_s
            ),
            "front_marker_capture_px4_time_s": (
                self.front_marker_capture_px4_time_s
            ),
            "down_marker_capture_px4_time_s": (
                self.down_marker_capture_px4_time_s
            ),
            "front_marker_capture_delay_s": (
                self.front_marker_capture_delay_s
            ),
            "down_marker_capture_delay_s": (
                self.down_marker_capture_delay_s
            ),
            "front_marker_capture_vehicle_position_enu_m": (
                self.front_marker_capture_vehicle_position_enu
            ),
            "down_marker_capture_vehicle_position_enu_m": (
                self.down_marker_capture_vehicle_position_enu
            ),
            "front_marker_capture_vehicle_velocity_enu_m_s": (
                self.front_marker_capture_vehicle_velocity_enu
            ),
            "down_marker_capture_vehicle_velocity_enu_m_s": (
                self.down_marker_capture_vehicle_velocity_enu
            ),
            "front_marker_quality": self.front_marker_quality,
            "down_marker_quality": self.down_marker_quality,
            "front_marker_quality_stamp": self.front_marker_quality_stamp,
            "down_marker_quality_stamp": self.down_marker_quality_stamp,
            "front_marker_covariance_diagonal": (
                self.front_marker_covariance_diagonal
            ),
            "down_marker_covariance_diagonal": (
                self.down_marker_covariance_diagonal
            ),
            "front_marker_duplicate_count": (
                self.front_marker_capture_guard.duplicate_count
                + fusion_statistics["vision_front"].duplicate
            ),
            "down_marker_duplicate_count": (
                self.down_marker_capture_guard.duplicate_count
                + fusion_statistics["vision_down"].duplicate
            ),
            "front_marker_invalid_covariance_count": (
                self.front_marker_invalid_covariance_count
                + fusion_statistics["vision_front"].invalid_covariance
            ),
            "down_marker_invalid_covariance_count": (
                self.down_marker_invalid_covariance_count
                + fusion_statistics["vision_down"].invalid_covariance
            ),
            "front_marker_timing_rejection_count": (
                self.front_marker_timing_rejection_count
            ),
            "down_marker_timing_rejection_count": (
                self.down_marker_timing_rejection_count
            ),
            "front_marker_last_timing_rejection": (
                self.front_marker_last_timing_rejection
            ),
            "down_marker_last_timing_rejection": (
                self.down_marker_last_timing_rejection
            ),
            "front_marker_world_tilt_rad": self.front_marker_world_tilt_rad,
            "down_marker_world_tilt_rad": self.down_marker_world_tilt_rad,
            "front_marker_tilt_rejection_count": (
                self.front_marker_tilt_rejection_count
            ),
            "down_marker_tilt_rejection_count": (
                self.down_marker_tilt_rejection_count
            ),
            "front_marker_deferred_count": self.front_marker_deferred_count,
            "down_marker_deferred_count": self.down_marker_deferred_count,
            "pending_marker_measurement_count": len(
                self._pending_marker_messages
            ),
            "front_marker_sequence": self.front_marker_sequence,
            "down_marker_sequence": self.down_marker_sequence,
            "front_marker_estimator_initialized": (
                fusion_statistics["vision_front"].accepted > 0
            ),
            "down_marker_estimator_initialized": (
                fusion_statistics["vision_down"].accepted > 0
            ),
            "front_marker_accepted_count": (
                fusion_statistics["vision_front"].accepted
            ),
            "down_marker_accepted_count": (
                fusion_statistics["vision_down"].accepted
            ),
            "front_marker_rejected_count": (
                fusion_statistics["vision_front"].rejected
            ),
            "down_marker_rejected_count": (
                fusion_statistics["vision_down"].rejected
            ),
            "down_lidar_m": self.down_lidar_range_m,
            "uses_sim_ground_truth_control_input": self._bool(
                "uses_sim_ground_truth_control_input"
            ),
            "gz_partition": os.environ.get("GZ_PARTITION", ""),
        }
        message = String()
        message.data = dumps_strict_json(sanitize_json_value(fields))
        self.status_pub.publish(message)

    def destroy_node(self) -> None:
        """Join the solver worker before destroying ROS entities."""
        self.trailer_bridge.stop()
        if self.solver_worker is not None:
            self.solver_worker.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MpcPrecisionLandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
