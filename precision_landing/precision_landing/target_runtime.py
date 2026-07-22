"""Target-state role for the single precision-landing ROS facade.

This is a Python role mixin, not a ROS node.  It owns timestamp alignment,
measurement ingestion, fused-target access, ArUco motion fitting and RK4
prediction while the facade retains subscriptions and publication authority.
"""

from __future__ import annotations

import math
import time

import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from mavros_msgs.msg import TimesyncStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from .capture_state_history import (
    StateInterpolationError,
    StateTimeReset,
    marker_world_pose_at_capture,
    mavros_header_to_px4_time_s,
    quaternion_world_z_tilt_rad,
)
from .marker_measurement import (
    CaptureStampGuard,
    build_estimator_covariance,
    extract_pose_covariance_6x6,
)
from .production_runtime import TrailerPositionSample
from .runtime_types import LandingPhase
from .target_fusion import TargetEstimate, validate_covariance


class TargetRuntimeMixin:
    """Target and vehicle-state processing owned by the one ROS facade."""

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
        self.vehicle_yaw_enu_rad = self._quaternion_yaw(orientation)
        self.pose_stamp = time.monotonic()
        if self._time_reset_hold_pending:
            self._cache_time_reset_hold()
        self._retry_pending_vehicle_response_sample()
        self._retry_pending_trailer_odometry()
        self._retry_pending_marker_measurements()
    def _on_velocity(self, message: TwistStamped) -> None:
        now = time.monotonic()
        linear = message.twist.linear
        velocity = np.asarray((linear.x, linear.y, linear.z), dtype=float)
        if not np.all(np.isfinite(velocity)):
            return
        self._record_velocity_history(message.header.stamp, velocity)
        try:
            sample_time_s = self._px4_sample_time(message.header.stamp)
        except (StateInterpolationError, ValueError):
            sample_time_s = None
        if sample_time_s is not None:
            # Pose and velocity carry PX4 sample stamps but their ROS
            # callbacks can arrive in either order.  Keep one latest velocity
            # until position history can represent that exact sample time;
            # never combine it with the callback-time latest position.
            self._pending_vehicle_response_sample = (
                float(sample_time_s),
                velocity.copy(),
            )
            self._retry_pending_vehicle_response_sample()
        self.vehicle_velocity_enu = velocity
        self.velocity_stamp = now
        self._retry_pending_trailer_odometry()
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
        self._retry_pending_trailer_odometry()
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
    def _retry_pending_vehicle_response_sample(self) -> None:
        """Update response dynamics from a same-sample position/velocity pair."""
        pending = self._pending_vehicle_response_sample
        if pending is None:
            return
        sample_time_s, velocity = pending
        try:
            position = self.state_history.interpolate_position(
                sample_time_s,
                maximum_gap_s=self._float(
                    "state_history_max_interpolation_gap_s"
                ),
            )
        except StateInterpolationError:
            latest_position_time_s = (
                self.state_history.latest_position_time_s
            )
            if (
                latest_position_time_s is not None
                and latest_position_time_s >= sample_time_s
            ):
                # Position history has already passed this sample and cannot
                # acquire a new bracket without accepting out-of-order state.
                self._pending_vehicle_response_sample = None
                self.last_vehicle_response_reason = (
                    "position_pair_unavailable"
                )
            return
        except ValueError:
            self._pending_vehicle_response_sample = None
            self.last_vehicle_response_reason = "position_pair_invalid"
            return

        applied_acceleration = self.last_command_acceleration_enu
        if applied_acceleration is not None:
            applied_array = np.asarray(applied_acceleration, dtype=float)
            applied_acceleration = (
                applied_array
                if applied_array.shape == (3,)
                and np.all(np.isfinite(applied_array))
                else None
            )
        self._pending_vehicle_response_sample = None
        # Preserve the exact timestamp-paired measurement independently of
        # the response observer. A rejected Kalman update must not substitute
        # an open-loop p/v prediction for measured MPC state.
        self.vehicle_control_position_enu = np.asarray(
            position, dtype=float
        ).copy()
        self.vehicle_control_velocity_enu = np.asarray(
            velocity, dtype=float
        ).copy()
        self.vehicle_control_sample_time_s = float(sample_time_s)
        update = self.vehicle_response_estimator.update(
            sample_time_s,
            position,
            velocity,
            applied_acceleration,
        )
        self.last_vehicle_response_reason = update.reason
        if update.accepted and update.state is not None:
            self.vehicle_acceleration_enu = (
                update.state.acceleration_enu_m_s2.copy()
            )
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
        self.vehicle_response_estimator.reset()
        self.last_vehicle_response_reason = "time_reset"
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
        self.last_vision_motion_model_stamp_px4_s = None
        self.last_vision_motion_capture_px4_time_s = None
        self.vision_motion_model_covariance = None
        self._trailer_prediction_cache = None
        self.vision_velocity_history.clear()
        self.vision_position_history.clear()
        self.vision_position_candidates.clear()
        self.vision_position_model_qualified = False
        self.vision_motion_model_turn_rate_rad_s = 0.0
        self.vision_motion_model_tangential_acceleration_m_s2 = 0.0
        self.vision_motion_model_span_s = 0.0
        self.last_position_servo_velocity_enu.fill(0.0)
        self.last_position_servo_update_monotonic_s = None
        self.marker_tracking_acquisition_since_s = None
        self.control_target_velocity_source = "unavailable"
        self.trailer_state = None
        self.trailer_state_stamp = 0.0
        self.takeoff_feedforward_velocity_enu.fill(0.0)
        self.takeoff_trailer_previous_sample_time_s = None
        self.takeoff_trailer_previous_position_enu = None
        self.takeoff_trailer_velocity_initialized = False
        self.down_marker_capture_guard = CaptureStampGuard()
        self._pending_marker_messages.clear()
        self._pending_vehicle_response_sample = None
        self.vehicle_control_position_enu = None
        self.vehicle_control_velocity_enu = None
        self.vehicle_control_sample_time_s = None
        self._pending_trailer_odometry = None
        self.down_marker_stamp = 0.0
        self.down_marker_detected_stamp = 0.0
        self.down_marker_positive_stamp = 0.0
        self.down_marker_sequence = 0
        self.landing_pad_acceleration_enu = np.zeros(3, dtype=float)
        self._reset_terminal_contact_context()
        self._invalidate_solver_context(clear_cached=True)
        self.command_cache = None
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
    def _on_trailer_odometry(
        self,
        message: Odometry,
        *,
        _from_pending: bool = False,
    ) -> None:
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
        if not _from_pending:
            now = time.monotonic()
            self.trailer_state = TrailerPositionSample(
                measurement_px4_time_s,
                values[:3],
                yaw,
            )
            self.trailer_state_stamp = now

        # Gazebo/MAVROS callbacks from the same simulation step can be
        # scheduled before the final PX4 state stream advances its common
        # watermark.  Defer that valid sample rather than classifying it as a
        # permanently bad future measurement.  The bounded slot is
        # latest-wins and is retried whenever position/velocity/attitude moves.
        if measurement_px4_time_s > (
            current_px4_time_s
            + self._float("target_fusion_future_tolerance_s")
        ):
            self._defer_trailer_odometry(stamp_key_ns, message)
            return
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
            twist_in_body_frame=False,
        )
        if not result.queued:
            if result.reason == "future timestamp":
                self._defer_trailer_odometry(stamp_key_ns, message)
                return
            self.get_logger().warning(
                f"Rejecting trailer odometry fusion input: {result.reason}"
            )
            return
    def _defer_trailer_odometry(
        self,
        stamp_key_ns: int,
        message: Odometry,
    ) -> None:
        pending = self._pending_trailer_odometry
        if pending is None or int(stamp_key_ns) >= pending[0]:
            self._pending_trailer_odometry = (
                int(stamp_key_ns),
                message,
            )
    def _retry_pending_trailer_odometry(self) -> None:
        pending = self._pending_trailer_odometry
        if pending is None:
            return
        self._pending_trailer_odometry = None
        _stamp_key_ns, message = pending
        self._on_trailer_odometry(message, _from_pending=True)
    def _on_down_marker_pose(
        self, message: PoseWithCovarianceStamped
    ) -> None:
        self._on_marker_pose(message)
    def _on_down_marker_detected(self, message: Bool) -> None:
        """Track the detector's current frame result, including misses."""
        received_at = time.monotonic()
        detected = bool(message.data)
        self.down_marker_detected_stamp = received_at
        if detected:
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
        message: PoseWithCovarianceStamped,
        *,
        _registered: bool = False,
    ) -> None:
        """Transform and queue one unique capture for ordered fusion."""
        guard = self.down_marker_capture_guard

        stamp = message.header.stamp
        if not _registered:
            try:
                unique_capture = guard.register(stamp.sec, stamp.nanosec)
            except ValueError as exc:
                self.get_logger().warning(
                    f"Rejecting down marker measurement timestamp: {exc}"
                )
                return
            if not unique_capture:
                return
            # A newer registered capture supersedes an older deferred one for
            # the same camera. This bounded latest-wins queue cannot grow when
            # MAVROS state delivery stalls.
            self._pending_marker_messages.pop("down", None)

        frame = str(message.header.frame_id).strip().lstrip("/")
        if frame != "base_link":
            self.get_logger().warning(
                "Rejecting down marker outside production base_link: %r"
                % frame
            )
            return
        try:
            covariance_6x6 = extract_pose_covariance_6x6(
                message.pose.covariance
            )
        except ValueError as exc:
            self._increment_invalid_covariance()
            self.get_logger().warning(
                f"Rejecting down marker covariance: {exc}"
            )
            return

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
                "Rejecting down marker orientation"
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
                self._defer_marker_measurement(message)
                return
            self._reject_marker_timing(exc.reason)
            return
        except ValueError as exc:
            self.get_logger().warning(
                f"Rejecting down marker capture transform: {exc}"
            )
            return

        marker_world_tilt_rad = quaternion_world_z_tilt_rad(
            marker_world_orientation
        )
        if marker_world_tilt_rad > self._float(
            "maximum_marker_world_tilt_rad"
        ):
            self.get_logger().warning(
                "Rejecting down marker world tilt %.3f rad"
                % marker_world_tilt_rad
            )
            return

        try:
            estimator_covariance = build_estimator_covariance(
                covariance_6x6,
                rotation_enu_from_flu,
                distance_m=float(np.linalg.norm(relative_flu)),
                camera="down",
            )
        except ValueError as exc:
            self._increment_invalid_covariance()
            self.get_logger().warning(
                f"Rejecting down marker covariance transform: {exc}"
            )
            return

        stamp_key_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        submission = self.target_estimator.submit_vision(
            "vision_down",
            capture_px4_time_s,
            stamp_key_ns,
            marker_world,
            marker_world_yaw,
            estimator_covariance,
            fusion_received_px4_time_s,
        )
        if not submission.queued:
            self.get_logger().warning(
                "Rejecting down marker fusion input: %s"
                % submission.reason
            )
            return
        previous_capture = self.last_vision_motion_capture_px4_time_s
        last_accepted_capture = self.vision_motion_estimator.statistics[
            "vision_down"
        ].last_accepted_stamp_s
        reacquisition_gap_s = self._vision_motion_continuation_limit_s()
        accepted_stream_lost = bool(
            last_accepted_capture is not None
            and capture_px4_time_s - float(last_accepted_capture)
            > reacquisition_gap_s
        )
        callback_stream_lost_before_initialization = bool(
            last_accepted_capture is None
            and previous_capture is not None
            and capture_px4_time_s - previous_capture > reacquisition_gap_s
        )
        if accepted_stream_lost or callback_stream_lost_before_initialization:
            # Reset on an accepted-sample outage, not merely a callback
            # outage. A continuous run of NIS-rejected camera frames
            # otherwise kept the old CV prediction alive forever and made
            # every later valid frame fail the same innovation gate.
            self.vision_motion_estimator.reset()
            self.last_vision_motion_estimate = None
            self.vision_velocity_history.clear()
            self.vision_position_history.clear()
            self.vision_position_candidates.clear()
            self.vision_position_model_qualified = False
            self.vision_motion_model_span_s = 0.0
            self.landing_pad_acceleration_enu.fill(0.0)
        if previous_capture is None or capture_px4_time_s > previous_capture:
            self.last_vision_motion_capture_px4_time_s = capture_px4_time_s
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
        else:
            self._record_vision_position_sample(
                capture_px4_time_s, marker_world
            )
        now = time.monotonic()
        self.down_marker_stamp = now
    def _defer_marker_measurement(
        self,
        message: PoseWithCovarianceStamped,
    ) -> None:
        """Hold one latest capture until real PX4 state brackets its stamp."""
        stamp = message.header.stamp
        stamp_key_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        self._pending_marker_messages["down"] = (stamp_key_ns, message)
    def _retry_pending_marker_measurements(self) -> None:
        """Retry deferred captures after a position, velocity, or attitude update."""
        if not self._pending_marker_messages:
            return
        pending = tuple(self._pending_marker_messages.items())
        self._pending_marker_messages.clear()
        for _camera, (_stamp_key_ns, message) in pending:
            self._on_marker_pose(message, _registered=True)
    def _increment_invalid_covariance(self) -> None:
        self.down_marker_invalid_covariance_count += 1
    def _reject_marker_timing(self, reason: str) -> None:
        self.get_logger().warning(
            f"Rejecting down marker capture time: {reason}"
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
    def _trailer_fresh(self, now: float) -> bool:
        age = float(now) - float(self.trailer_state_stamp)
        return bool(
            self.trailer_state is not None
            and self.trailer_state_stamp > 0.0
            and math.isfinite(age)
            and 0.0 <= age <= self._float("trailer_timeout_s")
        )
    def _current_target_estimate(
        self, current_px4_time_s: float | None = None
    ) -> TargetEstimate | None:
        try:
            if current_px4_time_s is None:
                estimate_time_s = self._current_px4_sample_time()
            else:
                estimate_time_s = float(current_px4_time_s)
                if (
                    not math.isfinite(estimate_time_s)
                    or estimate_time_s <= 0.0
                ):
                    raise ValueError("invalid target estimate time")
            estimate = self.target_estimator.estimate(estimate_time_s)
        except (StateInterpolationError, ValueError):
            return None
        self.last_target_estimate = estimate
        self.down_marker_sequence = self.vision_motion_estimator.statistics[
            "vision_down"
        ].accepted
        return estimate
    def _down_marker_world_enu(self, now: float) -> np.ndarray | None:
        """Return the latest fresh optical marker position."""
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
        self.down_marker_sequence = self.vision_motion_estimator.statistics[
            "vision_down"
        ].accepted
        fresh = self.vision_motion_estimator.source_fresh(
            "vision_down",
            current_px4_time_s,
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
        self,
        now: float,
        preferred_camera: str | None,
        *,
        control_px4_time_s: float | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        TargetEstimate | None,
    ] | None:
        if preferred_camera not in {None, "down"}:
            raise ValueError("production supports only the down camera")
        estimate = self._current_target_estimate(control_px4_time_s)
        target_velocity = self._control_target_velocity_enu(
            estimate,
            now,
            current_px4_time_s=control_px4_time_s,
        )
        if target_velocity is None:
            return None
        precision_phases = {
            LandingPhase.PRECISION_ALIGN,
            LandingPhase.PRECISION_DESCENT,
            LandingPhase.FINAL_APPROACH,
            LandingPhase.TOUCHDOWN_CONFIRM,
        }
        if (
            estimate is not None
            and estimate.valid
            and estimate.position_enu_m is not None
            and estimate.yaw_enu_rad is not None
        ):
            target_position = estimate.position_enu_m.copy()
            target_yaw = float(estimate.yaw_enu_rad)
        else:
            # Raw trailer position is acquisition-only.  Once precision flight
            # begins, every position/yaw reference must come from the canonical
            # timestamp-ordered estimator; never synthesize or overwrite an
            # estimate after a covariance/NIS rejection.
            if self.phase in precision_phases:
                return None
            raw_trailer_pose = self._latest_trailer_marker_pose(now)
            if raw_trailer_pose is None:
                return None
            target_position, target_yaw = raw_trailer_pose
        return (
            target_position,
            target_velocity,
            target_yaw,
            estimate,
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
    def _record_vision_position_sample(
        self, capture_stamp_s: float, marker_position_enu_m: np.ndarray
    ) -> None:
        """Retain a validated ArUco pose until fusion accepts or rejects it."""
        stamp = float(capture_stamp_s)
        position = np.asarray(marker_position_enu_m, dtype=float)
        if (
            not math.isfinite(stamp)
            or position.shape != (3,)
            or not np.all(np.isfinite(position))
        ):
            return
        candidates = self.vision_position_candidates
        if candidates and stamp <= candidates[-1][0] + 1.0e-9:
            return
        candidates.append((stamp, position.copy()))
        while candidates and stamp - candidates[0][0] > 4.0:
            candidates.popleft()
    def _promote_accepted_vision_position_sample(
        self, accepted_capture_stamp_s: float
    ) -> bool:
        """Move only an estimator-accepted ArUco pose into the fit history."""
        accepted_stamp = float(accepted_capture_stamp_s)
        if not math.isfinite(accepted_stamp):
            return False
        matched_position: np.ndarray | None = None
        for stamp, position in self.vision_position_candidates:
            if abs(stamp - accepted_stamp) <= 1.0e-6:
                matched_position = position
                break
        if matched_position is None:
            return False
        history = self.vision_position_history
        if history and accepted_stamp <= history[-1][0] + 1.0e-9:
            return False
        history.append((accepted_stamp, matched_position.copy()))
        while history and accepted_stamp - history[0][0] > 5.2:
            history.popleft()
        while (
            self.vision_position_candidates
            and self.vision_position_candidates[0][0]
            <= accepted_stamp + 1.0e-9
        ):
            self.vision_position_candidates.popleft()
        return True
    def _fit_vision_position_motion_model(
        self, maximum_capture_stamp_s: float
    ) -> tuple[
        np.ndarray, np.ndarray, float, float, float, bool
    ] | None:
        """Fit velocity and turn rate directly from recent ArUco positions.

        The trailer velocity topic is deliberately not used.  A robust local
        quadratic p(t)=c0+c1*t+c2*t^2 yields velocity c1 and acceleration 2*c2
        at the newest accepted camera capture.  Constant-turn rate then follows
        from omega=(vx*ay-vy*ax)/|v|^2.
        """
        accepted_stamp = float(maximum_capture_stamp_s)
        if not math.isfinite(accepted_stamp):
            return None
        samples = [
            (stamp, position)
            for stamp, position in self.vision_position_history
            if stamp <= accepted_stamp + 1.0e-6
            and accepted_stamp - stamp <= 5.0
        ]
        # Sparse detections can span the full 4.8 s observability window yet
        # still fit a biased circle (the failed 9 m/s run qualified 14 points
        # at 0.217 rad/s for a 0.180 rad/s route).  Twenty accepted camera
        # positions are still only about two seconds on the normal 10 Hz
        # stream, preserve stationary/slow-target operation, and prevent that
        # under-sampled fit from owning the dropout prediction.
        if len(samples) < 20:
            return None
        stamps = np.asarray([sample[0] for sample in samples], dtype=float)
        positions = np.asarray(
            [sample[1][:2] for sample in samples], dtype=float
        )
        span_s = float(stamps[-1] - stamps[0])
        if span_s < 2.0:
            return None
        relative_times = stamps - stamps[-1]
        design = np.column_stack(
            (
                np.ones(len(stamps), dtype=float),
                relative_times,
                relative_times * relative_times,
            )
        )
        coefficients = np.linalg.lstsq(design, positions, rcond=None)[0]
        residual_norms = np.linalg.norm(
            positions - design @ coefficients, axis=1
        )
        median_residual = float(np.median(residual_norms))
        mad = float(
            np.median(np.abs(residual_norms - median_residual))
        )
        robust_scale = max(0.005, 1.4826 * mad)
        retained = residual_norms <= median_residual + 3.0 * robust_scale
        if int(np.count_nonzero(retained)) >= 8:
            coefficients = np.linalg.lstsq(
                design[retained], positions[retained], rcond=None
            )[0]
        else:
            retained = np.ones(len(stamps), dtype=bool)

        velocity_xy = np.asarray(coefficients[1], dtype=float)
        acceleration_xy = np.asarray(2.0 * coefficients[2], dtype=float)
        if (
            velocity_xy.shape != (2,)
            or acceleration_xy.shape != (2,)
            or not np.all(np.isfinite(velocity_xy))
            or not np.all(np.isfinite(acceleration_xy))
        ):
            return None
        speed = float(np.linalg.norm(velocity_xy))
        maximum_speed = self._float(
            "relative_landing_mpc_max_horizontal_velocity_m_s"
        )
        if speed > maximum_speed + 0.75:
            return None
        # Endpoint derivatives of a quadratic are noise-sensitive on a
        # stationary marker. Use the full-window linear slope to identify that
        # case before accepting an apparent turn rate.
        linear_coefficients = np.linalg.lstsq(
            design[retained, :2], positions[retained], rcond=None
        )[0]
        robust_linear_speed = float(
            np.linalg.norm(linear_coefficients[1])
        )
        model_observable = False
        if robust_linear_speed < 0.25:
            velocity_xy.fill(0.0)
            acceleration_xy.fill(0.0)
            turn_rate = 0.0
            tangential = 0.0
            model_observable = span_s >= 2.8
        else:
            # A quadratic endpoint derivative systematically straightens a
            # fast circular trajectory (about 8% at 10 m/s, R=50 m over this
            # window). Fit the circle geometry and regress its unwrapped
            # phase instead. Straight motion or an ill-conditioned short arc
            # falls back to the quadratic model below.
            circle_applied = False
            circle_positions = positions[retained]
            circle_times = relative_times[retained]
            if len(circle_positions) >= 8:
                position_mean = np.mean(circle_positions, axis=0)
                centered = circle_positions - position_mean
                circle_design = np.column_stack(
                    (
                        2.0 * centered[:, 0],
                        2.0 * centered[:, 1],
                        np.ones(len(centered), dtype=float),
                    )
                )
                circle_rhs = np.sum(centered * centered, axis=1)
                circle_coefficients = np.linalg.lstsq(
                    circle_design, circle_rhs, rcond=None
                )[0]
                center_relative = np.asarray(
                    circle_coefficients[:2], dtype=float
                )
                radius_squared = float(
                    circle_coefficients[2]
                    + np.dot(center_relative, center_relative)
                )
                if radius_squared > 0.0 and math.isfinite(radius_squared):
                    circle_center = position_mean + center_relative
                    radial_vectors = circle_positions - circle_center
                    radii = np.linalg.norm(radial_vectors, axis=1)
                    radius = float(math.sqrt(radius_squared))
                    radial_residual = float(
                        np.sqrt(np.mean((radii - radius) ** 2))
                    )
                    angles = np.unwrap(
                        np.arctan2(
                            radial_vectors[:, 1], radial_vectors[:, 0]
                        )
                    )
                    angle_design = np.column_stack(
                        (
                            np.ones(len(circle_times), dtype=float),
                            circle_times,
                        )
                    )
                    angle_coefficients = np.linalg.lstsq(
                        angle_design, angles, rcond=None
                    )[0]
                    angular_residual = angles - (
                        angle_design @ angle_coefficients
                    )
                    circle_turn_rate = float(angle_coefficients[1])
                    circle_speed = abs(circle_turn_rate) * radius
                    circle_time_span = float(
                        circle_times[-1] - circle_times[0]
                    )
                    # Judge observability with the fitted angular rate, not
                    # the two noisy endpoint angles.  On the 5 m/s baseline a
                    # bad short-arc circle claimed enough endpoint coverage
                    # while implying only 0.0815 rad/s; the real 0.1000 rad/s
                    # deck then diverged after close-range marker occlusion.
                    # Require nearly the full camera window as well as angular
                    # coverage. A noisy 3.4 s arc at 3 m/s once fit the wrong
                    # turn direction while still claiming 0.31 rad coverage;
                    # 4.8 s makes both 3 and 5 m/s turns observable before the
                    # model is allowed to drive close-range prediction.
                    fitted_angular_span = (
                        abs(circle_turn_rate) * circle_time_span
                    )
                    speed_consistent = abs(
                        circle_speed - robust_linear_speed
                    ) <= max(0.75, 0.15 * robust_linear_speed)
                    if (
                        np.all(np.isfinite(circle_center))
                        and np.all(np.isfinite(angular_residual))
                        and 5.0 <= radius <= 250.0
                        and radial_residual <= 0.75
                        and float(np.sqrt(np.mean(angular_residual**2)))
                        <= 0.08
                        and 0.02 <= abs(circle_turn_rate) <= 0.25
                        and circle_time_span >= 4.8
                        and fitted_angular_span >= 0.28
                        and circle_speed <= maximum_speed + 0.75
                        and speed_consistent
                    ):
                        turn_rate = float(
                            np.clip(circle_turn_rate, -0.25, 0.25)
                        )
                        speed = min(maximum_speed, circle_speed)
                        newest_angle = float(angle_coefficients[0])
                        radial_direction = np.asarray(
                            (math.cos(newest_angle), math.sin(newest_angle)),
                            dtype=float,
                        )
                        tangent_direction = np.asarray(
                            (-radial_direction[1], radial_direction[0]),
                            dtype=float,
                        )
                        velocity_xy = (
                            math.copysign(speed, turn_rate)
                            * tangent_direction
                        )
                        acceleration_xy = (
                            -(turn_rate * turn_rate)
                            * radius
                            * radial_direction
                        )
                        tangential = 0.0
                        circle_applied = True
                        model_observable = True

            if not circle_applied:
                turn_rate = float(
                    (
                        velocity_xy[0] * acceleration_xy[1]
                        - velocity_xy[1] * acceleration_xy[0]
                    )
                    / (speed * speed)
                )
                turn_rate = float(np.clip(turn_rate, -0.25, 0.25))
                tangential = float(
                    np.dot(velocity_xy, acceleration_xy) / speed
                )
                if abs(tangential) < 0.15:
                    tangential = 0.0
                tangential = float(np.clip(tangential, -0.40, 0.40))
                speed = min(maximum_speed, speed)
                direction = velocity_xy / float(np.linalg.norm(velocity_xy))
                velocity_xy = speed * direction
                normal = np.asarray(
                    (-direction[1], direction[0]), dtype=float
                )
                acceleration_xy = (
                    tangential * direction + turn_rate * speed * normal
                )
                model_observable = bool(
                    # A 3 m/s target on the 50 m route accumulates only
                    # 0.168 rad in 2.8 s.  At that span camera noise can make
                    # the quadratic fallback look falsely straight and lock
                    # the wrong turn direction.  Let the circle fit observe
                    # its 0.28 rad first; true straight motion can tolerate
                    # this bounded extra acquisition time.
                    span_s >= 4.8
                    and abs(turn_rate) <= 0.02
                    and abs(tangential) <= 0.15
                )

        maximum_acceleration = self._float(
            "relative_landing_mpc_max_pad_acceleration_m_s2"
        )
        acceleration_norm = float(np.linalg.norm(acceleration_xy))
        if acceleration_norm > maximum_acceleration:
            acceleration_xy *= maximum_acceleration / acceleration_norm
            speed = float(np.linalg.norm(velocity_xy))
            if speed > 1.0e-3:
                turn_rate = float(
                    (
                        velocity_xy[0] * acceleration_xy[1]
                        - velocity_xy[1] * acceleration_xy[0]
                    )
                    / (speed * speed)
                )
                tangential = float(
                    np.dot(velocity_xy, acceleration_xy) / speed
                )

        velocity = np.asarray(
            (velocity_xy[0], velocity_xy[1], 0.0), dtype=float
        )
        acceleration = np.asarray(
            (acceleration_xy[0], acceleration_xy[1], 0.0), dtype=float
        )
        return (
            velocity,
            acceleration,
            turn_rate,
            tangential,
            float(stamps[-1]),
            model_observable,
        )
    def _update_vision_velocity_model(
        self,
        model_stamp_s: float,
        measured_velocity_enu_m_s: np.ndarray | None,
        accepted_capture_stamp_s: float,
        *,
        allow_position_model: bool,
    ) -> bool:
        """Fit a robust planar constant-turn model from ArUco velocity."""
        stamp = float(model_stamp_s)
        if not math.isfinite(stamp):
            return False
        measured = (
            None
            if measured_velocity_enu_m_s is None
            else np.asarray(measured_velocity_enu_m_s, dtype=float).copy()
        )
        if measured is not None and (
            measured.shape != (3,) or not np.all(np.isfinite(measured))
        ):
            return False
        if measured is not None:
            measured[2] = 0.0
        measured_speed = (
            0.0 if measured is None else float(np.linalg.norm(measured[:2]))
        )
        position_model = (
            self._fit_vision_position_motion_model(
                accepted_capture_stamp_s
            )
            if allow_position_model
            else None
        )
        if position_model is not None:
            was_position_model_qualified = (
                self.vision_position_model_qualified
            )
            (
                fitted_velocity,
                coherent_acceleration,
                turn_rate,
                tangential,
                position_model_stamp,
                model_observable,
            ) = position_model
            if self.vision_position_model_qualified and not model_observable:
                # A short/noisy quadratic fallback must not replace the phase
                # of a circle, straight line, or stationary model that has
                # already passed its observability gate.  Rebase that proven
                # CTRV analytically at the new capture time instead.  This
                # keeps position, velocity, acceleration, and model epoch
                # coherent without freezing the epoch or introducing RK4
                # integration error.
                previous_model_stamp = (
                    self.last_vision_motion_model_stamp_px4_s
                )
                if (
                    previous_model_stamp is not None
                    and math.isfinite(float(previous_model_stamp))
                    and position_model_stamp
                    >= float(previous_model_stamp) - 1.0e-9
                ):
                    propagated_velocity = (
                        self._predict_vision_motion_velocity(
                            max(
                                0.0,
                                position_model_stamp
                                - float(previous_model_stamp),
                            )
                        )
                    )
                    if propagated_velocity is not None:
                        self.last_vision_velocity_enu = (
                            propagated_velocity.copy()
                        )
                        self.last_vision_motion_model_stamp_px4_s = (
                            position_model_stamp
                        )
                        self.vision_motion_model_span_s = min(
                            5.0,
                            max(
                                0.0,
                                position_model_stamp
                                - self.vision_position_history[0][0],
                            ),
                        )
                        return True
            if self.vision_position_model_qualified:
                previous_speed = float(
                    np.linalg.norm(self.last_vision_velocity_enu[:2])
                )
                fitted_speed = float(np.linalg.norm(fitted_velocity[:2]))
                fitted_speed = previous_speed + float(
                    np.clip(fitted_speed - previous_speed, -0.35, 0.35)
                )
                fitted_norm = float(np.linalg.norm(fitted_velocity[:2]))
                if fitted_norm > 1.0e-6:
                    fitted_velocity[:2] *= fitted_speed / fitted_norm
                previous_turn_rate = float(
                    self.vision_motion_model_turn_rate_rad_s
                )
                turn_rate = previous_turn_rate + float(
                    np.clip(turn_rate - previous_turn_rate, -0.02, 0.02)
                )
                if fitted_speed > 1.0e-3:
                    direction = fitted_velocity[:2] / fitted_speed
                    normal = np.asarray(
                        (-direction[1], direction[0]), dtype=float
                    )
                    coherent_acceleration[:2] = (
                        tangential * direction
                        + turn_rate * fitted_speed * normal
                    )
                else:
                    coherent_acceleration[:2] = 0.0
            self.last_vision_velocity_enu = fitted_velocity
            self.vision_motion_model_turn_rate_rad_s = turn_rate
            self.vision_motion_model_tangential_acceleration_m_s2 = tangential
            self.last_vision_motion_model_stamp_px4_s = position_model_stamp
            self.vision_position_model_qualified = bool(
                self.vision_position_model_qualified or model_observable
            )
            self.vision_motion_model_span_s = min(
                5.0,
                max(
                    0.0,
                    position_model_stamp
                    - self.vision_position_history[0][0],
                ),
            )
            if (
                self.vision_position_model_qualified
                and not was_position_model_qualified
            ):
                self.get_logger().info(
                    "ArUco position CTRV qualified: "
                    f"speed={np.linalg.norm(fitted_velocity[:2]):.3f}m/s "
                    f"turn_rate={turn_rate:.4f}rad/s "
                    f"span={self.vision_motion_model_span_s:.2f}s "
                    f"samples={len(self.vision_position_history)}"
                )
            return True
        if measured is None or measured_speed <= 1.0e-3:
            return False
        history = self.vision_velocity_history
        if history and stamp <= history[-1][0] + 1.0e-9:
            return False

        # Do not let one noisy pose derivative replace an already qualified
        # turning model.  These bounds are well outside the expected change
        # over one camera period but reject the velocity spikes that otherwise
        # collapse the fitted curvature immediately before marker loss.
        if len(history) >= 4:
            historical_speeds = np.asarray(
                [np.linalg.norm(sample[1][:2]) for sample in history],
                dtype=float,
            )
            median_speed = float(np.median(historical_speeds))
            if abs(measured_speed - median_speed) > 0.75:
                return False
            previous_stamp, previous_velocity = history[-1]
            previous_heading = math.atan2(
                float(previous_velocity[1]), float(previous_velocity[0])
            )
            predicted_heading = (
                previous_heading
                + self.vision_motion_model_turn_rate_rad_s
                * max(0.0, stamp - previous_stamp)
            )
            measured_heading = math.atan2(
                float(measured[1]), float(measured[0])
            )
            heading_residual = math.atan2(
                math.sin(measured_heading - predicted_heading),
                math.cos(measured_heading - predicted_heading),
            )
            if abs(heading_residual) > 0.20:
                return False

        history.append((stamp, measured))
        # A five-second Cartesian v=v0+a*t fit spans 0.5 rad on the current
        # 5 m/s, R=50 m trajectory and progressively straightens the command.
        # Fit speed and unwrapped heading over a short local window instead.
        window_s = 2.0
        while history and stamp - history[0][0] > window_s:
            history.popleft()

        stamps = np.asarray([sample[0] for sample in history], dtype=float)
        velocities = np.asarray(
            [sample[1][:2] for sample in history], dtype=float
        )
        span_s = 0.0 if len(stamps) < 2 else float(stamps[-1] - stamps[0])
        self.vision_motion_model_span_s = max(0.0, span_s)
        speeds = np.linalg.norm(velocities, axis=1)
        headings = np.unwrap(np.arctan2(velocities[:, 1], velocities[:, 0]))
        fitted_speed = float(np.median(speeds))
        fitted_heading = float(headings[-1])
        turn_rate = self.vision_motion_model_turn_rate_rad_s
        tangential = self.vision_motion_model_tangential_acceleration_m_s2
        if len(stamps) >= 5 and span_s >= 0.8:
            relative_times = stamps - stamps[-1]
            design = np.column_stack(
                (np.ones(len(relative_times), dtype=float), relative_times)
            )
            heading_coefficients = np.linalg.lstsq(
                design, headings, rcond=None
            )[0]
            heading_residuals = headings - design @ heading_coefficients
            absolute_residual = np.abs(heading_residuals)
            median_residual = float(np.median(absolute_residual))
            mad = float(
                np.median(np.abs(absolute_residual - median_residual))
            )
            robust_scale = max(1.0e-3, 1.4826 * mad)
            retained = (
                absolute_residual
                <= median_residual + 3.0 * robust_scale
            )
            if int(np.count_nonzero(retained)) >= 4:
                heading_coefficients = np.linalg.lstsq(
                    design[retained], headings[retained], rcond=None
                )[0]
            else:
                retained = np.ones(len(stamps), dtype=bool)
            fitted_heading = float(heading_coefficients[0])
            turn_rate = float(np.clip(heading_coefficients[1], -0.25, 0.25))

            retained_speeds = speeds[retained]
            fitted_speed = float(np.median(retained_speeds))
            speed_coefficients = np.linalg.lstsq(
                design[retained], retained_speeds, rcond=None
            )[0]
            tangential = float(speed_coefficients[1])
            if abs(tangential) < 0.15:
                tangential = 0.0
            tangential = float(np.clip(tangential, -0.40, 0.40))

        maximum_speed = self._float(
            "relative_landing_mpc_max_horizontal_velocity_m_s"
        )
        fitted_speed = min(maximum_speed, max(0.0, fitted_speed))
        direction = np.asarray(
            (math.cos(fitted_heading), math.sin(fitted_heading)), dtype=float
        )
        normal = np.asarray((-direction[1], direction[0]), dtype=float)
        fitted_velocity = fitted_speed * direction
        coherent_acceleration = (
            tangential * direction + turn_rate * fitted_speed * normal
        )
        maximum_acceleration = self._float(
            "relative_landing_mpc_max_pad_acceleration_m_s2"
        )
        acceleration_norm = float(np.linalg.norm(coherent_acceleration))
        if acceleration_norm > maximum_acceleration:
            coherent_acceleration *= maximum_acceleration / acceleration_norm
            # Preserve a consistent exact-turn model after clipping.
            if fitted_speed > 1.0e-3:
                turn_rate = float(
                    (
                        fitted_velocity[0] * coherent_acceleration[1]
                        - fitted_velocity[1] * coherent_acceleration[0]
                    )
                    / (fitted_speed * fitted_speed)
                )
                tangential = float(
                    np.dot(fitted_velocity, coherent_acceleration)
                    / fitted_speed
                )

        self.last_vision_velocity_enu = np.asarray(
            (fitted_velocity[0], fitted_velocity[1], 0.0), dtype=float
        )
        self.vision_motion_model_turn_rate_rad_s = turn_rate
        self.vision_motion_model_tangential_acceleration_m_s2 = tangential
        self.last_vision_motion_model_stamp_px4_s = stamp
        return True
    def _capture_vision_motion_model_covariance(
        self, motion: TargetEstimate | None
    ) -> None:
        """Freeze uncertainty at the capture used to refresh the motion fit.

        ``vision_motion_estimator.estimate(now)`` predicts its covariance
        forward on every control tick.  Feeding that already-grown covariance
        back into an RK4 propagation would count the same dropout age twice and
        prematurely invalidate an otherwise bounded terminal continuation.
        """
        covariance = np.diag((0.01, 0.01, 0.01, 0.25)).astype(float)
        velocity = self.last_vision_velocity_enu
        if velocity is None:
            self.vision_motion_model_covariance = covariance
            return
        speed = float(np.linalg.norm(np.asarray(velocity)[:2]))
        course_yaw = (
            math.atan2(float(velocity[1]), float(velocity[0]))
            if speed > 1.0e-3
            else 0.0
        )
        if motion is not None and motion.covariance is not None:
            fused = np.asarray(motion.covariance, dtype=float)
            if fused.shape == (8, 8) and np.all(np.isfinite(fused)):
                heading = np.asarray(
                    (math.cos(course_yaw), math.sin(course_yaw)), dtype=float
                )
                candidate = covariance.copy()
                candidate[0:2, 0:2] = fused[0:2, 0:2]
                candidate[2, 2] = max(1.0e-6, float(fused[6, 6]))
                if not self.vision_position_model_qualified:
                    # Before the position fit is observable, velocity really
                    # is the Kalman state and must retain its covariance.  A
                    # qualified circle/line fit, however, estimates speed from
                    # a separate 4.8 s robust position window.  Reusing the
                    # Kalman velocity variance for that independent model can
                    # start above the rejection threshold and invalidate it
                    # immediately.  Keep the conservative 0.25 (m/s)^2 model
                    # variance above and let RK4 process noise grow it once.
                    candidate[3, 3] = max(
                        1.0e-6,
                        float(heading @ fused[3:5, 3:5] @ heading),
                    )
                candidate = 0.5 * (candidate + candidate.T)
                if np.min(np.linalg.eigvalsh(candidate)) >= -1.0e-9:
                    covariance = candidate
        self.vision_motion_model_covariance = covariance
        # A new camera fit defines a new model epoch.  Do not mix a prediction
        # from the preceding speed/turn fit with this covariance snapshot.
        self._trailer_prediction_cache = None
    def _terminal_motion_covariance_bridge_allowed(
        self, estimate: TargetEstimate | None
    ) -> bool:
        """Keep only a qualified near-deck motion mean after optical loss.

        Growing RK4 covariance still blocks ordinary descent.  Near the deck,
        however, discarding the already-qualified speed/course mean before the
        configured terminal hold expires turns a bounded moving-pad prediction
        into a world-frame stop.  This narrow bridge additionally requires a
        fresh, covariance-bounded shared trailer position on every call.
        """
        terminal_recovery = bool(
            self.phase == LandingPhase.PRECISION_ALIGN
            and self.terminal_recovery_started_mission_s is not None
            and self.descent_clearance_m
            <= self._float("terminal_occlusion_max_clearance_m") + 1.0e-6
        )
        if (
            not self.vision_position_model_qualified
            or not (
                self.phase
                in {
                    LandingPhase.FINAL_APPROACH,
                    LandingPhase.TOUCHDOWN_CONFIRM,
                }
                or terminal_recovery
            )
            or estimate is None
            or not bool(getattr(estimate, "valid", False))
            or not bool(getattr(estimate, "odometry_fresh", False))
            or estimate.position_enu_m is None
            or estimate.covariance is None
            or self.vehicle_position_enu is None
        ):
            return False
        target_position = np.asarray(estimate.position_enu_m, dtype=float)
        vehicle_position = np.asarray(self.vehicle_position_enu, dtype=float)
        covariance = np.asarray(estimate.covariance, dtype=float)
        if (
            target_position.shape != (3,)
            or vehicle_position.shape != (3,)
            or covariance.shape != (8, 8)
            or not np.all(np.isfinite(target_position))
            or not np.all(np.isfinite(vehicle_position))
        ):
            return False
        try:
            position_covariance = validate_covariance(
                covariance[:3, :3], 3, "terminal target position"
            )
        except ValueError:
            return False
        clearance = float(vehicle_position[2] - target_position[2])
        minimum_clearance = (
            -self._float("touchdown_contact_latch_exit_distance_m")
            if self._contact_entry_latched
            or self.phase == LandingPhase.TOUCHDOWN_CONFIRM
            else 0.0
        )
        return bool(
            math.isfinite(clearance)
            and minimum_clearance <= clearance
            <= self._float("terminal_occlusion_max_clearance_m") + 1.0e-6
            and float(np.max(np.linalg.eigvalsh(position_covariance)))
            <= self._float("descent_max_position_variance_m2")
        )

    def _predict_vision_motion_velocity(
        self,
        age_s: float,
        *,
        allow_terminal_covariance_bridge: bool = False,
    ) -> np.ndarray | None:
        """Propagate the accepted ArUco motion with the RK4 bicycle model."""
        velocity = self.last_vision_velocity_enu
        if velocity is None:
            return None
        base_velocity = np.asarray(velocity, dtype=float)
        turn_rate = float(self.vision_motion_model_turn_rate_rad_s)
        tangential = float(
            self.vision_motion_model_tangential_acceleration_m_s2
        )
        age = float(age_s)
        if (
            base_velocity.shape != (3,)
            or not np.all(np.isfinite(base_velocity))
            or not math.isfinite(turn_rate)
            or not math.isfinite(tangential)
            or not math.isfinite(age)
            or age < 0.0
        ):
            return None
        speed = float(np.linalg.norm(base_velocity[:2]))
        if speed > 1.0e-3:
            course_yaw = math.atan2(
                float(base_velocity[1]), float(base_velocity[0])
            )
        else:
            course_yaw = 0.0
        model_stamp = self.last_vision_motion_model_stamp_px4_s
        if model_stamp is None or not math.isfinite(float(model_stamp)):
            self.last_trailer_prediction_reason = "model_stamp_unavailable"
            return None
        base_position = (
            self.vision_position_history[-1][1].copy()
            if self.vision_position_history
            else np.zeros(3, dtype=float)
        )
        covariance = (
            np.diag((0.01, 0.01, 0.01, 0.25)).astype(float)
            if self.vision_motion_model_covariance is None
            else self.vision_motion_model_covariance.copy()
        )
        target_time_s = float(model_stamp) + age
        cached_state = self._trailer_prediction_cache
        use_cached_state = bool(
            cached_state is not None
            and cached_state.sample_time_s >= float(model_stamp) - 1.0e-9
            and cached_state.sample_time_s <= target_time_s + 1.0e-9
        )
        try:
            state = (
                cached_state
                if use_cached_state and cached_state is not None
                else self.trailer_motion_predictor.state_from_observation(
                    float(model_stamp),
                    base_position,
                    base_velocity,
                    course_yaw,
                    turn_rate,
                    covariance,
                    tangential,
                )
            )
            prediction = self.trailer_motion_predictor.predict(
                state, target_time_s
            )
        except ValueError:
            self.last_trailer_prediction_reason = "invalid_model_state"
            return None
        self.last_trailer_prediction_reason = prediction.reason
        if (
            not prediction.valid
            or prediction.state is None
            or prediction.velocity_enu_m_s is None
            or prediction.acceleration_enu_m_s2 is None
        ):
            return None
        self._trailer_prediction_cache = prediction.state
        predicted_covariance = prediction.state.covariance
        if (
            not np.all(np.isfinite(predicted_covariance))
            or float(predicted_covariance[2, 2]) < 0.0
            or float(predicted_covariance[3, 3]) < 0.0
        ):
            self.last_trailer_prediction_reason = "invalid_prediction_covariance"
            return None
        speed_covariance_exceeded = bool(
            float(predicted_covariance[3, 3])
            > self._float("vision_velocity_maximum_variance_m2_s2")
        )
        if speed_covariance_exceeded and not allow_terminal_covariance_bridge:
            self.last_trailer_prediction_reason = "speed_covariance_exceeded"
            return None
        if speed_covariance_exceeded:
            self.last_trailer_prediction_reason = (
                "speed_covariance_degraded_terminal_bridge"
            )
        predicted = prediction.velocity_enu_m_s.copy()
        maximum_speed = self._float(
            "relative_landing_mpc_max_horizontal_velocity_m_s"
        )
        predicted_speed = float(np.linalg.norm(predicted[:2]))
        if predicted_speed > maximum_speed and predicted_speed > 1.0e-9:
            predicted[:2] *= maximum_speed / predicted_speed
        predicted[2] = 0.0
        self.landing_pad_acceleration_enu = (
            prediction.acceleration_enu_m_s2.copy()
        )
        self.landing_pad_acceleration_enu[2] = 0.0
        return predicted
    def _vision_motion_continuation_limit_s(self) -> float:
        """Return one coherent lifetime for the qualified optical model.

        SEARCH keeps the short ordinary reacquisition window. Once the
        the vehicle reaches the guarded terminal envelope, a temporary optical
        gap must not discard that validated speed/direction model after three
        seconds. Ordinary ALIGN/DESCENT still use the short limit; only FINAL,
        touchdown, or an explicitly latched terminal recenter may use the
        longer bridge.
        """
        ordinary_limit_s = min(
            self._float("vision_velocity_dropout_hold_s"),
            self._float("vision_velocity_reacquisition_gap_s"),
        )
        terminal_recovery = bool(
            self.phase == LandingPhase.PRECISION_ALIGN
            and self.terminal_recovery_started_mission_s is not None
            and self.descent_clearance_m
            <= self._float("terminal_occlusion_max_clearance_m") + 1.0e-6
        )
        terminal_tracking = bool(
            self.vision_position_model_qualified
            and (
                self.phase
                in {
                    LandingPhase.FINAL_APPROACH,
                    LandingPhase.TOUCHDOWN_CONFIRM,
                }
                or terminal_recovery
            )
        )
        if not terminal_tracking:
            return ordinary_limit_s
        return max(
            ordinary_limit_s,
            self._float("vision_velocity_terminal_hold_s"),
        )
    def _control_target_velocity_enu(
        self,
        estimate: TargetEstimate | None,
        now: float,
        *,
        current_px4_time_s: float | None = None,
    ) -> np.ndarray | None:
        """Return the ArUco-observed velocity used by MPC and safety gates."""
        try:
            if current_px4_time_s is None:
                current_px4_time_s = self._current_px4_sample_time()
            else:
                current_px4_time_s = float(current_px4_time_s)
                if (
                    not math.isfinite(current_px4_time_s)
                    or current_px4_time_s <= 0.0
                ):
                    raise ValueError("invalid target velocity time")
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
        accepted_stamp_is_new = bool(
            accepted_stamp is not None
            and math.isfinite(float(accepted_stamp))
            and math.isfinite(current_px4_time_s)
            and accepted_stamp
            != self.last_vision_velocity_source_stamp_px4_s
        )
        position_model_updated = False
        if accepted_stamp_is_new:
            # Accepted raw ArUco positions are the primary motion source.
            # Their history must not depend on the Kalman velocity variance;
            # that dependency could prevent CTRV qualification forever.
            promoted = self._promote_accepted_vision_position_sample(
                float(accepted_stamp)
            )
            if promoted:
                position_model_updated = self._update_vision_velocity_model(
                    current_px4_time_s,
                    None,
                    float(accepted_stamp),
                    allow_position_model=True,
                )
                if position_model_updated:
                    self.last_vision_velocity_stamp_px4_s = float(
                        accepted_stamp
                    )
                    self._capture_vision_motion_model_covariance(motion)
                    self.control_target_velocity_source = (
                        "aruco_position_ctrv"
                    )

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
            and accepted_stamp_is_new
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
                model_updated = position_model_updated
                if not model_updated:
                    # Kalman velocity is only the startup fallback while the
                    # accepted raw-position window is still too short.
                    model_updated = self._update_vision_velocity_model(
                        current_px4_time_s,
                        candidate,
                        float(accepted_stamp),
                        allow_position_model=False,
                    )
                if model_updated:
                    self.last_vision_velocity_stamp_px4_s = float(
                        accepted_stamp
                    )
                    self._capture_vision_motion_model_covariance(motion)
                    if not position_model_updated:
                        self.control_target_velocity_source = (
                            "aruco_capture_kalman"
                        )
        if accepted_stamp_is_new:
            # Consume the accepted capture once. A rejected fallback must not
            # be retried at 50 Hz, while a successful position fit already
            # refreshed the independent model/cache timestamps above.
            self.last_vision_velocity_source_stamp_px4_s = float(
                accepted_stamp
            )

        cached = self.last_vision_velocity_enu
        cached_stamp = self.last_vision_velocity_stamp_px4_s
        model_stamp = self.last_vision_motion_model_stamp_px4_s
        if (
            cached is not None
            and cached_stamp is not None
            and model_stamp is not None
            and math.isfinite(current_px4_time_s)
        ):
            age = current_px4_time_s - cached_stamp
            model_age = current_px4_time_s - model_stamp
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
            maximum_age = self._vision_motion_continuation_limit_s()
            if (
                0.0 <= age
                and 0.0 <= model_age
                and effective_age <= maximum_age
                and model_age <= maximum_age
            ):
                acquisition_level_hold = bool(
                    self.phase == LandingPhase.MARKER_TRACK_DOWN
                    and effective_age
                    <= self._float("descent_max_vision_age_s")
                    and estimate is not None
                    and bool(getattr(estimate, "valid", False))
                    and bool(getattr(estimate, "odometry_fresh", False))
                )
                # The Kalman velocity passed to the model is already predicted
                # at current_px4_time_s.  Propagating it again from the older
                # capture stamp double-counted camera latency and distorted
                # the estimated turn. Camera age remains the freshness gate;
                # the separate model age drives only CTRV propagation.  Do
                # not invalidate this already-qualified motion solely because
                # position covariance grows during a short optical gap:
                # ``_descent_permission`` still blocks vertical motion on
                # freshness and position covariance, while this bounded CTRV
                # continuation keeps level XY tracking alive for reacquisition.
                predicted = self._predict_vision_motion_velocity(
                    model_age,
                    allow_terminal_covariance_bridge=(
                        acquisition_level_hold
                        or (
                            terminal
                            and self._terminal_motion_covariance_bridge_allowed(
                                estimate
                            )
                        )
                    ),
                )
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
    def _vehicle_acceleration_ready_for_relative_mpc(self) -> bool:
        """Require the exact MPC input state to satisfy its accel bounds."""
        if (
            self.vehicle_response_estimator.consecutive_innovation_rejections
            != 0
        ):
            return False
        try:
            _position, _velocity, acceleration = (
                self._vehicle_state_for_control(
                    require_fresh_prediction=True
                )
            )
        except (StateInterpolationError, ValueError):
            return False
        acceleration = np.asarray(acceleration, dtype=float)
        if acceleration.shape != (3,) or not np.all(
            np.isfinite(acceleration)
        ):
            return False
        return bool(
            np.max(np.abs(acceleration[:2]))
            <= self._float(
                "relative_landing_mpc_max_horizontal_acceleration_m_s2"
            )
            and abs(float(acceleration[2]))
            <= self._float(
                "relative_landing_mpc_max_vertical_acceleration_m_s2"
            )
        )
