"""Execute the city A*/SFC/B-spline mission through PX4 native DDS topics."""

from __future__ import annotations

from enum import Enum
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Sequence

import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavigationPath
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLandDetected,
    VehicleAttitude,
    VehicleLocalPosition,
    VehicleStatus,
)
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .gz_sensor_adapter import GazeboSensorAdapter
from .mission_core import (
    JerkInputMPC,
    RelativeTargetEstimator,
    TouchdownGate,
    TrailerKinematicState,
    body_velocity_to_enu,
    enu_to_ned_vector,
    flu_to_enu_with_px4_quaternion,
    iterative_polyline_intercept,
    marker_position_to_flu,
)
from .planner_core import (
    CityMap,
    FrameTransform,
    OccupancyGrid,
)
from .planner_3d import PlanningOutputMode
from .runtime_plan import (
    ActivePlan3D,
    CorridorViolationError,
    LatestPlanWorker3D,
    PlanWorkerStatus,
    RuntimePlanAdapter3D,
)


class MissionPhase(Enum):
    WAITING = "waiting_for_px4_and_sensors"
    PRESTREAM = "offboard_setpoint_prestream"
    ARMING = "arming_and_entering_offboard"
    TAKEOFF = "vertical_takeoff_10m"
    GLOBAL_PLAN = "global_plan"
    TRACKING = "astar_sfc_bspline_mpc_tracking"
    HOLDING = "goal_hold"
    RETURN_REQUESTED = "return_requested"
    TRAILER_INTERCEPT_PLAN = "trailer_intercept_plan"
    TRAILER_INTERCEPT_TRACK = "trailer_intercept_track"
    MARKER_SEARCH_FRONT = "marker_search_front"
    MARKER_ACQUIRE_FRONT = "marker_acquire_front"
    CAMERA_HANDOFF = "camera_handoff"
    MARKER_TRACK_DOWN = "marker_track_down"
    VELOCITY_MATCH = "velocity_match"
    PRECISION_ALIGN = "precision_align"
    PRECISION_DESCENT = "precision_descent"
    FINAL_APPROACH = "final_approach"
    TOUCHDOWN_CONFIRM = "touchdown_confirm"
    DISARM = "disarm_after_touchdown"
    HOLD_REPLAN = "hold_replan"
    ABORT_CLIMB = "abort_climb"
    SENSOR_HOLD = "sensor_stale_hold"
    FAILSAFE_HOLD = "failsafe_hold"
    ABORT_LAND = "abort_auto_land"


def _source_map_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    uav_map = root / "gazebo/maps/city_coordinates_uav.yaml"
    return uav_map if uav_map.is_file() else root / "gazebo/maps/city_coordinates.yaml"


def _unit_xy(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float)[:2]
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-9:
        return np.asarray((1.0, 0.0), dtype=float)
    return value / norm


class CityAutonomyNode(Node):
    """PX4 OFFBOARD state machine for the requested city mission."""

    def __init__(self) -> None:
        super().__init__("city_autonomy")
        self._declare_parameters()
        self.map_path = Path(str(self.get_parameter("map_yaml").value)).expanduser()
        self.city = CityMap.from_yaml(self.map_path)
        self.transform = FrameTransform.from_city_map(self.city)
        self.grid = OccupancyGrid.from_city_map(
            self.city,
            resolution_m=self._float("grid_resolution_m"),
            inflation_m=self._float("obstacle_inflation_m"),
        )
        self.goal_enu = np.asarray(
            (self._float("goal_enu_x_m"), self._float("goal_enu_y_m")),
            dtype=float,
        )
        self.runtime_planner = RuntimePlanAdapter3D(self.city)
        planning_z = max(
            self.runtime_planner.minimum_planning_center_z_m,
            self.city.px4_origin_enu_m[2] + self._float("takeoff_height_m"),
        )
        start_enu = np.asarray(
            (*self.city.px4_origin_enu_m[:2], planning_z), dtype=float
        )
        goal_enu = np.asarray((*self.goal_enu, planning_z), dtype=float)
        self.active_plan: ActivePlan3D = self.runtime_planner.plan_active(
            start_enu,
            goal_enu,
            self._planning_output_mode(),
        )
        self.path_xyz = np.asarray(
            self.active_plan.sampled_position_enu_m, dtype=float
        )
        self.path_xy = self.path_xyz[:, :2]
        self.path_lengths = np.asarray(
            self.active_plan.cumulative_length_m, dtype=float
        )
        self.plan_worker = LatestPlanWorker3D(
            self.runtime_planner.plan_active,
            cache_entries=4,
            thread_name="city-global-plan-worker",
        )
        self.pending_plan_request_id: int | None = None
        self.pending_plan_goal_xy: np.ndarray | None = None
        self.pending_plan_resume_phase = MissionPhase.TRACKING
        self.pending_plan_kind = ""
        self.pending_plan_started = 0.0
        self.pending_waypoint_index: int | None = None

        self.mpc = JerkInputMPC(
            dt_s=self._float("local_mpc_dt_s"),
            horizon_steps=self._int("local_mpc_horizon_steps"),
            velocity_limit_m_s=self._float("max_horizontal_speed_m_s"),
            acceleration_limit_m_s2=self._float("max_acceleration_m_s2"),
            jerk_limit_m_s3=self._float("max_jerk_m_s3"),
            deadline_s=0.001 * self._float("local_mpc_deadline_ms"),
            maximum_iterations=self._int("local_mpc_max_outer_iterations"),
        )
        self.landing_mpc = JerkInputMPC(
            dt_s=self._float("landing_mpc_dt_s"),
            horizon_steps=self._int("landing_mpc_horizon_steps"),
            velocity_limit_m_s=self._float("max_horizontal_speed_m_s"),
            acceleration_limit_m_s2=self._float("max_acceleration_m_s2"),
            jerk_limit_m_s3=self._float("max_jerk_m_s3"),
            deadline_s=0.001 * self._float("landing_mpc_deadline_ms"),
            maximum_iterations=self._int("landing_mpc_max_iterations"),
        )

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", px4_qos
        )
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", px4_qos
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", px4_qos
        )
        self.path_pub = self.create_publisher(
            NavigationPath, "/autonomy/global_path", path_qos
        )
        self.status_pub = self.create_publisher(String, "/autonomy/status", 10)
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self._on_local_position,
            px4_qos,
        )
        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self._on_vehicle_attitude,
            px4_qos,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self._on_vehicle_status,
            px4_qos,
        )
        self.create_subscription(
            VehicleCommandAck,
            "/fmu/out/vehicle_command_ack",
            self._on_command_ack,
            px4_qos,
        )
        self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self._on_land_detected,
            px4_qos,
        )
        self.create_subscription(
            Odometry,
            "/trailer/odometry",
            self._on_trailer_odometry,
            10,
        )
        self.create_subscription(
            PoseStamped,
            "/perception/front/marker_pose",
            self._on_front_marker_pose,
            10,
        )
        self.create_subscription(
            PoseStamped,
            "/perception/down/marker_pose",
            self._on_down_marker_pose,
            10,
        )
        self.create_service(
            Trigger,
            "/autonomy/land_on_trailer",
            self._on_land_on_trailer,
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_service(
            Trigger,
            "/autonomy/emergency_land",
            self._on_emergency_land,
            callback_group=ReentrantCallbackGroup(),
        )

        self.sensors = GazeboSensorAdapter(
            self.city.gazebo_world_name,
            self.city.runtime_entity_name,
        )

        self.control_time_s = 0.0
        self.last_control_sample_dt_s = 0.0
        self.estimated_sim_rtf = math.nan
        self._previous_sample_wall_s = 0.0
        self._previous_sample_time_s = 0.0
        self.phase = MissionPhase.WAITING
        self.phase_started = time.monotonic()
        self.phase_started_control_s = self.control_time_s
        self.local_position: VehicleLocalPosition | None = None
        self.vehicle_status: VehicleStatus | None = None
        self.vehicle_attitude: VehicleAttitude | None = None
        self.land_detected: VehicleLandDetected | None = None
        self.local_position_stamp = 0.0
        self.local_position_valid = False
        self.vehicle_status_stamp = 0.0
        self.vehicle_attitude_stamp = 0.0
        self.land_detected_stamp = 0.0
        self.runtime_origin_ned: np.ndarray | None = None
        self.xy_reset_counter: int | None = None
        self.z_reset_counter: int | None = None
        self.ekf_reset_sequence_valid = True
        self.takeoff_start_ned: np.ndarray | None = None
        self.takeoff_target_ned: np.ndarray | None = None
        self.takeoff_within_since: float | None = None
        self.hold_target_ned: np.ndarray | None = None
        self.path_index = 0
        self.filtered_altitude_world_m = (
            max(
                self.city.px4_origin_enu_m[2] + self._float("takeoff_height_m"),
                float(self.path_xyz[0, 2]),
            )
        )
        self.avoidance_offset_m = 0.0
        self.avoidance_sign = 0.0
        self.avoidance_until = 0.0
        self.last_mode_command = 0.0
        self.last_arm_command = 0.0
        self.last_land_command = 0.0
        self.last_disarm_command = 0.0
        self.last_status_publish = 0.0
        self.last_command_acceleration = np.zeros(3, dtype=float)
        self.last_reference_ned = np.zeros(3, dtype=float)
        self.last_mpc_message = "not_started"
        self.last_mpc_solve_time_s = 0.0
        self.last_mpc_iterations = 0
        self.last_mpc_success = False
        self.last_landing_mpc_message = "not_started"
        self.last_landing_mpc_solve_time_s = 0.0
        self.last_landing_mpc_iterations = 0
        self.last_landing_mpc_success = False
        self.last_landing_control_failure = "not_started"
        self.estimated_acceleration_ned = np.zeros(3, dtype=float)
        self._previous_velocity_ned: np.ndarray | None = None
        self._previous_velocity_stamp = 0.0
        self._return_request = threading.Event()
        self._emergency_land_request = threading.Event()
        self.trailer_state: TrailerKinematicState | None = None
        self.trailer_state_stamp = 0.0
        self.front_marker_pose: PoseStamped | None = None
        self.front_marker_stamp = 0.0
        self.down_marker_pose: PoseStamped | None = None
        self.down_marker_stamp = 0.0
        self.front_marker_seen_since: float | None = None
        self.down_marker_seen_since: float | None = None
        self.front_marker_sequence = 0
        self.down_marker_sequence = 0
        self.phase_front_marker_sequence = 0
        self.phase_down_marker_sequence = 0
        self.front_marker_estimator = RelativeTargetEstimator()
        self.down_marker_estimator = RelativeTargetEstimator()
        self.last_intercept_plan = 0.0
        self.intercept_target_enu: np.ndarray | None = None
        self.waypoint_hold_started: float | None = None
        self.landing_gate_since: float | None = None
        self.landing_descent_start_clearance_m: float | None = None
        self.landing_abort_count = 0
        self.hold_replan_resume_phase = MissionPhase.TRACKING
        self.plan_join_resume_phase = MissionPhase.TRACKING
        self.plan_join_within_since: float | None = None
        self.waypoint_index = 0
        flat_waypoints = np.asarray(
            self.get_parameter("waypoints_enu_xy_m").value, dtype=float
        )
        if flat_waypoints.size < 2 or flat_waypoints.size % 2:
            raise ValueError("waypoints_enu_xy_m must contain x,y pairs")
        self.waypoints_enu = flat_waypoints.reshape(-1, 2)
        self.goal_enu = self.waypoints_enu[0].copy()
        flat_trailer_route = np.asarray(
            self.get_parameter("trailer_route_waypoints_enu_xy_m").value,
            dtype=float,
        )
        if flat_trailer_route.size < 4 or flat_trailer_route.size % 2:
            raise ValueError(
                "trailer_route_waypoints_enu_xy_m must contain x,y pairs"
            )
        self.trailer_route_waypoints_enu = flat_trailer_route.reshape(-1, 2)
        self.touchdown_gate = TouchdownGate(
            maximum_range_m=self._float("touchdown_max_range_m"),
            maximum_vertical_speed_m_s=self._float("touchdown_max_vertical_speed_m_s"),
            dwell_s=self._float("touchdown_confirm_s"),
        )
        self._has_setpoint = False
        self._setpoint_lock = threading.Lock()
        self._last_velocity_ned: np.ndarray | None = None
        self._last_acceleration_ned: np.ndarray | None = None
        self._last_yaw_ned: float | None = None
        self._sensor_wait_warning_sent = False
        self.sensor_resume_phase = MissionPhase.TRACKING

        self._publish_planned_path()
        period = 1.0 / self._float("control_rate_hz")
        self.timer = self.create_timer(period, self._control_tick)
        self.heartbeat_timer = self.create_timer(
            0.05,
            self._heartbeat_tick,
            callback_group=ReentrantCallbackGroup(),
        )
        common_metrics = self.active_plan.waypoint_plan.metrics
        self.get_logger().info(
            "3-D global plan ready: expanded=%d, waypoints=%d, SFC=%d, "
            "samples=%d, length=%.1f m, mode=%s, minimum residual "
            "clearance=%.3f m"
            % (
                common_metrics.expanded_nodes,
                common_metrics.output_waypoint_count,
                common_metrics.corridor_box_count,
                len(self.path_xyz),
                self.path_lengths[-1],
                self.active_plan.mode.value,
                common_metrics.minimum_clearance_m,
            )
        )
        self.get_logger().info(
            "Mission coordinates: ENU goal=(%.1f, %.1f), PX4 NED goal=(%.1f, %.1f, %.1f)"
            % (
                self.goal_enu[0],
                self.goal_enu[1],
                *self.transform.enu_to_ned(
                    (self.goal_enu[0], self.goal_enu[1], self.filtered_altitude_world_m)
                ),
            )
        )

    def _declare_parameters(self) -> None:
        defaults: dict[str, object] = {
            "map_yaml": str(_source_map_path()),
            "goal_enu_x_m": 200.0,
            "goal_enu_y_m": -125.0,
            "waypoints_enu_xy_m": [200.0, -125.0, -128.0, -128.0],
            "repeat_waypoints": True,
            "guidance_mode": "bspline",
            "body_length_x_m": 1.0,
            "body_width_y_m": 1.0,
            "body_height_z_m": 0.4,
            "body_yaw_invariant_radius_m": 0.7071067811865476,
            "map_margin_m": 0.10,
            "localization_margin_m": 0.15,
            "tracking_margin_m": 0.20,
            "control_margin_m": 0.15,
            "extra_margin_m": 0.1428932188134524,
            "hard_inflation_radius_m": 1.45,
            "preferred_clearance_m": 2.15,
            "city_spacing_scale_xy": 2.5,
            "building_footprint_scale_xy": 0.9,
            "grid_resolution_coarse_m": 6.0,
            "grid_resolution_fine_m": 2.0,
            "astar_max_expanded_coarse": 200000,
            "astar_max_expanded_fine": 350000,
            "astar_timeout_coarse_s": 8.0,
            "astar_timeout_fine_s": 18.0,
            "sfc_max_iterations": 30,
            "sfc_max_repair_passes": 3,
            "bspline_max_iterations": 200,
            "bspline_time_scaling_retries": 5,
            "local_mpc_horizon_steps": 20,
            "local_mpc_dt_s": 0.10,
            "local_mpc_max_outer_iterations": 20,
            "local_mpc_max_qp_iterations": 100,
            "local_mpc_deadline_ms": 40.0,
            "landing_mpc_horizon_steps": 20,
            "landing_mpc_dt_s": 0.10,
            "landing_mpc_max_iterations": 12,
            "landing_mpc_deadline_ms": 40.0,
            "global_replan_rate_hz": 1.0,
            "global_replan_target_shift_m": 2.0,
            "waypoint_hold_s": 2.0,
            "takeoff_height_m": 10.0,
            "takeoff_rate_m_s": 1.5,
            "ground_clearance_m": 10.0,
            "ground_clearance_tolerance_m": 0.5,
            "grid_resolution_m": 1.0,
            "obstacle_inflation_m": 5.0,
            "clearance_cost_weight": 4.0,
            "corridor_max_radius_m": 10.0,
            "bspline_sample_spacing_m": 0.5,
            "cruise_speed_m_s": 4.0,
            "max_horizontal_speed_m_s": 5.0,
            "max_vertical_speed_m_s": 2.0,
            "max_acceleration_m_s2": 3.0,
            "max_jerk_m_s3": 5.0,
            "path_lookahead_m": 6.0,
            "waypoint_capture_radius_m": 2.0,
            "goal_tolerance_m": 1.5,
            "takeoff_tolerance_m": 0.4,
            "takeoff_hold_s": 1.5,
            "offboard_prestream_s": 2.0,
            "control_rate_hz": 20.0,
            "mpc_horizon_steps": 8,
            "mpc_dt_s": 0.2,
            "mpc_deadline_s": 0.09,
            "depth_trigger_distance_m": 10.0,
            "depth_emergency_distance_m": 4.0,
            "planner_reaction_time_s": 0.10,
            "braking_acceleration_m_s2": 3.0,
            "dynamic_obstacle_uncertainty_m": 0.50,
            "local_bypass_offset_m": 7.0,
            "local_obstacle_ttl_s": 2.0,
            "down_lidar_timeout_s": 1.5,
            "depth_timeout_s": 1.5,
            "local_position_timeout_s": 0.5,
            "vehicle_status_timeout_s": 1.0,
            "sensor_hold_land_timeout_s": 10.0,
            "trailer_timeout_s": 1.0,
            "trailer_replan_period_s": 1.0,
            "trailer_replan_distance_m": 2.0,
            "trailer_endpoint_x_m": -128.0,
            "trailer_endpoint_y_m": -128.0,
            "trailer_route_waypoints_enu_xy_m": [
                200.0,
                -125.0,
                200.0,
                -100.0,
                -128.0,
                -100.0,
                -128.0,
                -128.0,
            ],
            "trailer_cruise_speed_m_s": 3.0,
            "trailer_max_acceleration_m_s2": 1.0,
            "global_planner_latency_budget_s": 8.0,
            "trailer_deck_height_m": 1.25,
            "trailer_approach_radius_m": 10.0,
            "marker_timeout_s": 0.6,
            "marker_acquire_dwell_s": 0.5,
            "camera_handoff_overlap_s": 0.35,
            "marker_search_height_m": 8.0,
            "velocity_match_height_m": 5.0,
            "precision_align_height_m": 3.0,
            "final_approach_height_m": 0.8,
            "precision_descent_speed_m_s": 0.35,
            "final_descent_speed_m_s": 0.12,
            "precision_lateral_gate_m": 0.35,
            "relative_speed_gate_m_s": 0.35,
            "touchdown_max_range_m": 0.22,
            "touchdown_max_vertical_speed_m_s": 0.18,
            "touchdown_confirm_s": 1.0,
            "land_detected_timeout_s": 0.5,
            "landing_max_abort_count": 3,
            "max_world_altitude_m": 120.0,
            "require_sensors_before_arm": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _planning_output_mode(self) -> PlanningOutputMode:
        mode = str(self.get_parameter("guidance_mode").value).strip().lower()
        if mode in ("bspline", "sfc_bspline"):
            return PlanningOutputMode.SFC_BSPLINE
        if mode in ("px4_waypoint_preview", "native_waypoint_preview"):
            # This preview follows the certified polyline through OFFBOARD and
            # is useful only for dry-run visualization.  It is intentionally
            # not called PX4 L1: true multicopter L1-style crossing requires
            # AUTO.MISSION triplets and is provided by px4_l1_mission.
            return PlanningOutputMode.PX4_NATIVE_WAYPOINTS
        if mode in ("px4_l1", "l1"):
            raise ValueError(
                "guidance_mode=px4_l1 cannot run through TrajectorySetpoint; "
                "use autonomy_planner.px4_l1_mission for PX4 AUTO.MISSION"
            )
        raise ValueError(
            "guidance_mode must be 'bspline' or 'px4_waypoint_preview'"
        )

    @staticmethod
    def _cumulative_lengths(points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return np.zeros(len(points), dtype=float)
        return np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        )

    def _on_local_position(self, message: VehicleLocalPosition) -> None:
        if not (message.xy_valid and message.z_valid):
            self.local_position_valid = False
            self.local_position_stamp = time.monotonic()
            return
        reset_delta = np.zeros(3, dtype=float)
        reset_detected = False
        if (
            self.xy_reset_counter is not None
            and message.xy_reset_counter != self.xy_reset_counter
        ):
            if (int(message.xy_reset_counter) - self.xy_reset_counter) % 256 != 1:
                self.ekf_reset_sequence_valid = False
                self.get_logger().error("Missed one or more PX4 XY reset events")
            reset_delta[:2] = np.asarray(message.delta_xy, dtype=float)
            reset_detected = True
        if (
            self.z_reset_counter is not None
            and message.z_reset_counter != self.z_reset_counter
        ):
            if (int(message.z_reset_counter) - self.z_reset_counter) % 256 != 1:
                self.ekf_reset_sequence_valid = False
                self.get_logger().error("Missed one or more PX4 Z reset events")
            reset_delta[2] = float(message.delta_z)
            reset_detected = True
        if reset_detected:
            self._shift_for_ekf_reset(reset_delta)
        self.xy_reset_counter = int(message.xy_reset_counter)
        self.z_reset_counter = int(message.z_reset_counter)
        self.local_position = message
        self.local_position_valid = True
        now = time.monotonic()
        sample_us = int(
            getattr(message, "timestamp_sample", 0)
            or getattr(message, "timestamp", 0)
        )
        sample_time_s = 1.0e-6 * sample_us
        if sample_time_s > self._previous_sample_time_s:
            sample_dt = sample_time_s - self._previous_sample_time_s
            wall_dt = now - self._previous_sample_wall_s
            if self._previous_sample_time_s > 0.0 and wall_dt > 1.0e-4:
                self.last_control_sample_dt_s = sample_dt
                measured_rtf = sample_dt / wall_dt
                if math.isfinite(measured_rtf) and measured_rtf > 0.0:
                    self.estimated_sim_rtf = (
                        measured_rtf
                        if not math.isfinite(self.estimated_sim_rtf)
                        else 0.95 * self.estimated_sim_rtf + 0.05 * measured_rtf
                    )
            self.control_time_s = sample_time_s
            self._previous_sample_time_s = sample_time_s
            self._previous_sample_wall_s = now
        velocity = np.asarray((message.vx, message.vy, message.vz), dtype=float)
        if self._previous_velocity_ned is not None:
            dt = sample_time_s - self._previous_velocity_stamp
            if 0.01 <= dt <= 0.5 and np.all(np.isfinite(velocity)):
                measured = (velocity - self._previous_velocity_ned) / dt
                measured = np.clip(
                    measured,
                    -2.0 * self._float("max_acceleration_m_s2"),
                    2.0 * self._float("max_acceleration_m_s2"),
                )
                self.estimated_acceleration_ned = (
                    0.8 * self.estimated_acceleration_ned + 0.2 * measured
                )
        if np.all(np.isfinite(velocity)):
            self._previous_velocity_ned = velocity
            self._previous_velocity_stamp = sample_time_s
        self.local_position_stamp = now

    def _shift_for_ekf_reset(self, delta_ned: np.ndarray) -> None:
        """Keep world-fixed references fixed when PX4 shifts its local frame."""
        if self.runtime_origin_ned is None:
            return
        self.runtime_origin_ned += delta_ned
        for attribute in (
            "takeoff_start_ned",
            "takeoff_target_ned",
            "hold_target_ned",
        ):
            target = getattr(self, attribute)
            if target is not None:
                target += delta_ned
        self.get_logger().warning(
            "Applied PX4 EKF local-frame reset delta NED=(%.3f, %.3f, %.3f)"
            % tuple(delta_ned)
        )

    def _capture_runtime_origin(self, current_ned: np.ndarray) -> None:
        """Tie the known Gazebo spawn point to PX4's current local frame."""
        self.runtime_origin_ned = current_ned.copy()
        self.takeoff_start_ned = current_ned.copy()
        takeoff_world = np.asarray(self.city.px4_origin_enu_m, dtype=float)
        takeoff_world[2] += self._float("takeoff_height_m")
        self.takeoff_target_ned = self._enu_to_px4_ned(takeoff_world)
        self.hold_target_ned = current_ned.copy()
        goal_world = np.asarray(
            (self.goal_enu[0], self.goal_enu[1], takeoff_world[2]), dtype=float
        )
        goal_ned = self._enu_to_px4_ned(goal_world)
        self.get_logger().info(
            "Runtime PX4 origin captured at NED=(%.3f, %.3f, %.3f); "
            "goal NED=(%.1f, %.1f, %.1f)"
            % (*self.runtime_origin_ned, *goal_ned)
        )

    def _enu_to_px4_ned(self, point_enu: Sequence[float]) -> np.ndarray:
        mapped = self.transform.enu_to_ned(point_enu)
        if self.runtime_origin_ned is not None:
            mapped += self.runtime_origin_ned
        return mapped

    def _px4_ned_to_enu(self, point_ned: Sequence[float]) -> np.ndarray:
        point = np.asarray(point_ned, dtype=float)
        if self.runtime_origin_ned is not None:
            point = point - self.runtime_origin_ned
        return self.transform.ned_to_enu(point)

    def _on_vehicle_status(self, message: VehicleStatus) -> None:
        self.vehicle_status = message
        self.vehicle_status_stamp = time.monotonic()

    def _on_vehicle_attitude(self, message: VehicleAttitude) -> None:
        quaternion = np.asarray(message.q, dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if quaternion.shape != (4,) or not math.isfinite(norm) or norm < 1.0e-9:
            return
        self.vehicle_attitude = message
        self.vehicle_attitude_stamp = time.monotonic()

    def _on_land_detected(self, message: VehicleLandDetected) -> None:
        self.land_detected = message
        self.land_detected_stamp = time.monotonic()

    def _on_trailer_odometry(self, message: Odometry) -> None:
        now = time.monotonic()
        position = np.asarray(
            (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            ),
            dtype=float,
        )
        velocity_body = np.asarray(
            (
                message.twist.twist.linear.x,
                message.twist.twist.linear.y,
                message.twist.twist.linear.z,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(position)) or not np.all(
            np.isfinite(velocity_body)
        ):
            return
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        # nav_msgs/Odometry defines twist in child_frame_id.  The trailer
        # adapter publishes pose in world ENU and twist in trailer/base_link,
        # so rotate it once before feeding the world-frame interceptor.
        velocity = body_velocity_to_enu(velocity_body, yaw)
        acceleration = np.zeros(3, dtype=float)
        if self.trailer_state is not None:
            dt = now - self.trailer_state_stamp
            if 0.01 <= dt <= 1.0:
                acceleration = np.clip(
                    (velocity - self.trailer_state.velocity_enu_m_s) / dt,
                    -2.0,
                    2.0,
                )
        self.trailer_state = TrailerKinematicState(
            now, position, velocity, acceleration, yaw
        )
        self.trailer_state_stamp = now

    def _on_front_marker_pose(self, message: PoseStamped) -> None:
        position = message.pose.position
        if all(math.isfinite(value) for value in (position.x, position.y, position.z)):
            now = time.monotonic()
            relative_flu = self._marker_position_flu("front", message)
            if relative_flu is None:
                return
            marker_world = self._marker_world_from_flu(relative_flu, now)
            if marker_world is None:
                return
            distance = max(0.2, float(np.linalg.norm(relative_flu)))
            position_variance = (0.015 + 0.006 * distance) ** 2
            yaw_variance = math.radians(8.0) ** 2
            accepted = self.front_marker_estimator.update(
                now,
                marker_world,
                self._quaternion_yaw(message.pose.orientation),
                np.asarray(
                    (position_variance, position_variance, position_variance, yaw_variance)
                ),
            )
            if accepted:
                self.front_marker_pose = message
                self.front_marker_stamp = now
                self.front_marker_sequence += 1

    def _on_down_marker_pose(self, message: PoseStamped) -> None:
        position = message.pose.position
        if all(math.isfinite(value) for value in (position.x, position.y, position.z)):
            now = time.monotonic()
            relative_flu = self._marker_position_flu("down", message)
            if relative_flu is None:
                return
            marker_world = self._marker_world_from_flu(relative_flu, now)
            if marker_world is None:
                return
            distance = max(0.2, float(np.linalg.norm(relative_flu)))
            position_variance = (0.01 + 0.004 * distance) ** 2
            yaw_variance = math.radians(5.0) ** 2
            accepted = self.down_marker_estimator.update(
                now,
                marker_world,
                self._quaternion_yaw(message.pose.orientation),
                np.asarray(
                    (position_variance, position_variance, position_variance, yaw_variance)
                ),
            )
            if accepted:
                self.down_marker_pose = message
                self.down_marker_stamp = now
                self.down_marker_sequence += 1

    def _marker_position_flu(
        self, camera: str, message: PoseStamped
    ) -> np.ndarray | None:
        """Return one marker observation in aircraft base-link FLU.

        The configured ArUco nodes transform their output to ``base_link``.
        Applying an optical conversion again mirrored/swapped the landing
        error and destabilized terminal guidance.  Optical-frame input remains
        supported for diagnostics, but unknown frames are rejected closed.
        """

        position = message.pose.position
        vector = (position.x, position.y, position.z)
        try:
            return marker_position_to_flu(
                vector, str(message.header.frame_id), camera
            )
        except ValueError as exc:
            self.get_logger().error(f"Rejecting {camera} marker pose: {exc}")
            return None

    def _marker_world_from_flu(
        self, relative_flu: Sequence[float], now: float
    ) -> np.ndarray | None:
        """Transform a timestamp-near observation into the inertial ENU map."""

        if (
            self.vehicle_attitude is None
            or now - self.vehicle_attitude_stamp > 0.25
            or self.local_position is None
            or now - self.local_position_stamp > self._float(
                "local_position_timeout_s"
            )
        ):
            return None
        try:
            world_vector = flu_to_enu_with_px4_quaternion(
                relative_flu, self.vehicle_attitude.q
            )
        except ValueError:
            return None
        current_ned = np.asarray(
            (
                self.local_position.x,
                self.local_position.y,
                self.local_position.z,
            ),
            dtype=float,
        )
        return self._px4_ned_to_enu(current_ned) + world_vector

    @staticmethod
    def _quaternion_yaw(orientation: object) -> float:
        return math.atan2(
            2.0
            * (
                float(getattr(orientation, "w")) * float(getattr(orientation, "z"))
                + float(getattr(orientation, "x")) * float(getattr(orientation, "y"))
            ),
            1.0
            - 2.0
            * (
                float(getattr(orientation, "y")) ** 2
                + float(getattr(orientation, "z")) ** 2
            ),
        )

    def _on_land_on_trailer(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        allowed = self.phase in (
            MissionPhase.GLOBAL_PLAN,
            MissionPhase.TRACKING,
            MissionPhase.HOLDING,
            MissionPhase.HOLD_REPLAN,
            MissionPhase.FAILSAFE_HOLD,
        )
        if not allowed:
            response.success = False
            response.message = f"land rejected in phase {self.phase.value}"
            return response
        self._return_request.set()
        response.success = True
        response.message = (
            "land accepted: waypoint loop will stop and the drone will return "
            "to the moving trailer; this is not an in-place PX4 LAND command"
        )
        return response

    def _on_emergency_land(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        self._emergency_land_request.set()
        response.success = True
        response.message = "emergency landing requested at the current location"
        return response

    def _on_command_ack(self, message: VehicleCommandAck) -> None:
        if message.command in (
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
        ):
            self.get_logger().info(
                f"PX4 command ack: command={message.command} result={message.result}"
            )

    def _transition(self, phase: MissionPhase, explanation: str) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        self.phase_started = time.monotonic()
        self.phase_started_control_s = self.control_time_s
        self.phase_front_marker_sequence = self.front_marker_sequence
        self.phase_down_marker_sequence = self.down_marker_sequence
        self.landing_gate_since = None
        self.get_logger().info(f"PHASE -> {phase.value}: {explanation}")

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds // 1000)

    def _publish_offboard_heartbeat(self) -> None:
        message = OffboardControlMode()
        message.timestamp = self._now_us()
        message.position = True
        message.velocity = False
        message.acceleration = False
        message.attitude = False
        message.body_rate = False
        message.thrust_and_torque = False
        message.direct_actuator = False
        self.offboard_pub.publish(message)

    def _heartbeat_tick(self) -> None:
        # This timer is the *only* TrajectorySetpoint publisher.  Control and
        # solver callbacks atomically replace an immutable desired reference;
        # they never publish out-of-order samples from another executor thread.
        status = self.vehicle_status
        disarm_confirmed = bool(
            self.phase == MissionPhase.DISARM
            and status is not None
            and status.arming_state != VehicleStatus.ARMING_STATE_ARMED
        )
        if self.phase != MissionPhase.ABORT_LAND and not disarm_confirmed:
            self._publish_offboard_heartbeat()
            with self._setpoint_lock:
                has_setpoint = self._has_setpoint
                position = self.last_reference_ned.copy()
                velocity = (
                    None
                    if self._last_velocity_ned is None
                    else self._last_velocity_ned.copy()
                )
                acceleration = (
                    None
                    if self._last_acceleration_ned is None
                    else self._last_acceleration_ned.copy()
                )
                yaw = self._last_yaw_ned
            if has_setpoint:
                self._publish_setpoint_message(
                    position, velocity, acceleration, yaw
                )

    def _publish_setpoint_message(
        self,
        position_ned: Sequence[float],
        velocity_ned: Sequence[float] | None,
        acceleration_ned: Sequence[float] | None,
        yaw_ned: float | None,
    ) -> None:
        """Publish one cached reference from the single heartbeat thread."""

        nan = float("nan")
        message = TrajectorySetpoint()
        message.timestamp = self._now_us()
        message.position = [float(value) for value in position_ned]
        message.velocity = (
            [nan, nan, nan]
            if velocity_ned is None
            else [float(value) for value in velocity_ned]
        )
        message.acceleration = (
            [nan, nan, nan]
            if acceleration_ned is None
            else [float(value) for value in acceleration_ned]
        )
        message.jerk = [nan, nan, nan]
        message.yaw = nan if yaw_ned is None else float(yaw_ned)
        message.yawspeed = nan
        self.trajectory_pub.publish(message)

    def _publish_setpoint(
        self,
        position_ned: Sequence[float],
        velocity_ned: Sequence[float] | None = None,
        acceleration_ned: Sequence[float] | None = None,
        yaw_ned: float | None = None,
    ) -> None:
        """Atomically cache the next reference for the heartbeat publisher."""

        position = np.asarray(position_ned, dtype=float)
        velocity = (
            None if velocity_ned is None else np.asarray(velocity_ned, dtype=float)
        )
        acceleration = (
            None
            if acceleration_ned is None
            else np.asarray(acceleration_ned, dtype=float)
        )
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("PX4 position setpoint must be a finite NED three-vector")
        if velocity is not None and (
            velocity.shape != (3,) or not np.all(np.isfinite(velocity))
        ):
            raise ValueError("PX4 velocity feed-forward must be finite")
        if acceleration is not None and (
            acceleration.shape != (3,) or not np.all(np.isfinite(acceleration))
        ):
            raise ValueError("PX4 acceleration feed-forward must be finite")
        with self._setpoint_lock:
            self.last_reference_ned = position.copy()
            self._last_velocity_ned = None if velocity is None else velocity.copy()
            self._last_acceleration_ned = (
                None if acceleration is None else acceleration.copy()
            )
            self._last_yaw_ned = yaw_ned
            self._has_setpoint = True

    def _vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
    ) -> None:
        message = VehicleCommand()
        message.timestamp = self._now_us()
        message.command = int(command)
        message.param1 = float(param1)
        message.param2 = float(param2)
        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self.command_pub.publish(message)

    def _request_offboard_and_arm(self, now: float) -> None:
        status = self.vehicle_status
        if status is None or status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_mode_command >= 1.0:
                self._vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
                )
                self.last_mode_command = now
        if status is None or status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_arm_command >= 1.0:
                self._vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
                )
                self.last_arm_command = now

    def _request_auto_land(self, now: float) -> None:
        status = self.vehicle_status
        if status is None or status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            return
        if status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
            return
        if now - self.last_land_command >= 1.0:
            self._vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.last_land_command = now

    def _request_normal_disarm(self, now: float) -> None:
        status = self.vehicle_status
        if status is None or status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            return
        if now - self.last_disarm_command >= 1.0:
            # param2 deliberately remains zero: no PX4 magic/force disarm.
            self._vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=0.0,
                param2=0.0,
            )
            self.last_disarm_command = now

    def _sensor_state(self, now: float) -> tuple[bool, bool]:
        depth = self.sensors.depth()
        down = self.sensors.downward()
        depth_ready = (
            depth is not None
            and now - depth.stamp_monotonic <= self._float("depth_timeout_s")
        )
        down_ready = (
            down is not None
            and now - down.stamp_monotonic <= self._float("down_lidar_timeout_s")
        )
        return depth_ready, down_ready

    def _current_ned(self) -> tuple[np.ndarray, np.ndarray] | None:
        message = self.local_position
        if (
            message is None
            or not self.local_position_valid
            or time.monotonic() - self.local_position_stamp
            > self._float("local_position_timeout_s")
            or not message.v_xy_valid
            or not message.v_z_valid
        ):
            return None
        position = np.asarray((message.x, message.y, message.z), dtype=float)
        velocity = np.asarray((message.vx, message.vy, message.vz), dtype=float)
        if not np.all(np.isfinite(position)):
            return None
        if not np.all(np.isfinite(velocity)):
            velocity[:] = 0.0
        return position, velocity

    def _publish_planned_path(self) -> None:
        message = NavigationPath()
        message.header.frame_id = "map_enu"
        message.header.stamp = self.get_clock().now().to_msg()
        for x, y, z in self.path_xyz:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        self.path_pub.publish(message)

    def _install_plan(
        self, plan: ActivePlan3D, goal_enu_xy: Sequence[float]
    ) -> None:
        """Atomically replace the active certified 3-D path and SFC."""
        path_xyz = np.asarray(plan.sampled_position_enu_m, dtype=float)
        self.active_plan = plan
        self.goal_enu = np.asarray(goal_enu_xy, dtype=float)[:2].copy()
        self.path_xyz = path_xyz
        self.path_xy = path_xyz[:, :2]
        self.path_lengths = np.asarray(plan.cumulative_length_m, dtype=float)
        self.path_index = 0
        self.filtered_altitude_world_m = max(
            self.filtered_altitude_world_m, float(path_xyz[0, 2])
        )
        self._publish_planned_path()

    def _submit_replan_from_world(
        self,
        current_world_enu: Sequence[float],
        goal_enu_xy: Sequence[float],
        resume_phase: MissionPhase,
        kind: str,
    ) -> bool:
        """Submit one latest-only global plan without blocking PX4 control."""

        if self.pending_plan_request_id is not None:
            return False
        current = np.asarray(current_world_enu, dtype=float)
        goal_xy = np.asarray(goal_enu_xy, dtype=float)[:2]
        minimum_z = self.runtime_planner.minimum_planning_center_z_m
        start = np.asarray((current[0], current[1], max(current[2], minimum_z)))
        goal = np.asarray((goal_xy[0], goal_xy[1], max(current[2], minimum_z)))
        try:
            request_id = self.plan_worker.submit(
                start,
                goal,
                self._planning_output_mode(),
            )
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(f"could not submit 3-D replan: {exc}")
            return False
        self.pending_plan_request_id = request_id
        self.pending_plan_goal_xy = goal_xy.copy()
        self.pending_plan_resume_phase = resume_phase
        self.pending_plan_kind = str(kind)
        self.pending_plan_started = time.monotonic()
        self.get_logger().info(
            "submitted asynchronous 3-D plan %d: kind=%s goal=(%.1f, %.1f)"
            % (request_id, kind, goal_xy[0], goal_xy[1])
        )
        return True

    def _cancel_pending_plan(self, reason: str) -> None:
        request_id = self.pending_plan_request_id
        if request_id is None:
            return
        self.plan_worker.cancel(request_id)
        self.pending_plan_request_id = None
        self.pending_plan_goal_xy = None
        self.pending_plan_kind = ""
        self.pending_waypoint_index = None
        self.get_logger().info(f"cancelled global plan {request_id}: {reason}")

    def _poll_pending_replan(self, current_ned: np.ndarray, now: float) -> None:
        """Install only a completed, current latest plan on the control thread."""

        request_id = self.pending_plan_request_id
        if request_id is None:
            return
        outcome = self.plan_worker.latest_outcome
        if outcome is None or outcome.request_id != request_id:
            return
        goal_xy = self.pending_plan_goal_xy
        resume_phase = self.pending_plan_resume_phase
        kind = self.pending_plan_kind
        elapsed = max(0.0, now - self.pending_plan_started)
        self.pending_plan_request_id = None
        self.pending_plan_goal_xy = None
        self.pending_plan_kind = ""
        if (
            outcome.status is PlanWorkerStatus.SUCCEEDED
            and outcome.plan is not None
            and goal_xy is not None
        ):
            self._install_plan(outcome.plan, goal_xy)
            metrics = outcome.plan.waypoint_plan.metrics
            self.get_logger().info(
                "asynchronous 3-D plan %d accepted in %.3fs: "
                "expanded=%d SFC=%d length=%.1fm mode=%s"
                % (
                    request_id,
                    elapsed,
                    metrics.expanded_nodes,
                    metrics.corridor_box_count,
                    self.path_lengths[-1],
                    outcome.plan.mode.value,
                )
            )
            if kind == "intercept":
                self.last_intercept_plan = self.control_time_s
                if goal_xy is not None:
                    self.intercept_target_enu = np.asarray(
                        (goal_xy[0], goal_xy[1], self.path_xyz[-1, 2]),
                        dtype=float,
                    )
            elif kind == "waypoint" and self.pending_waypoint_index is not None:
                self.waypoint_index = self.pending_waypoint_index
            self.pending_waypoint_index = None
            self._begin_plan_join(
                resume_phase,
                f"asynchronous {kind} plan ready; joining certified corridor",
            )
            return

        error = outcome.error or outcome.status.value
        self.get_logger().error(
            f"asynchronous 3-D plan {request_id} failed after {elapsed:.3f}s: {error}"
        )
        self.hold_target_ned = current_ned.copy()
        self.pending_waypoint_index = None
        if kind == "waypoint":
            self.waypoint_hold_started = self.control_time_s
            self._transition(
                MissionPhase.HOLDING,
                "waypoint replan failed; retaining stationary hold before retry",
            )
        elif self.phase == MissionPhase.TRAILER_INTERCEPT_TRACK:
            self.last_intercept_plan = self.control_time_s
            self.get_logger().warning(
                "retaining the previously certified intercept path after "
                "a background refresh failure"
            )
        else:
            self._transition(
                MissionPhase.FAILSAFE_HOLD,
                "moving-trailer global plan failed; holding safely",
            )

    def _select_trailer_intercept(
        self, current_world: np.ndarray
    ) -> np.ndarray | None:
        """Choose a future target on the exact collision-free trailer route."""

        if self.trailer_state is None:
            return None

        def estimated_path_length(target: np.ndarray) -> float:
            return 1.15 * float(np.linalg.norm(target[:2] - current_world[:2]))

        solution = iterative_polyline_intercept(
            self.trailer_state,
            self.trailer_route_waypoints_enu,
            estimated_path_length,
            self._float("cruise_speed_m_s"),
            trailer_cruise_speed_m_s=self._float("trailer_cruise_speed_m_s"),
            trailer_acceleration_m_s2=self._float(
                "trailer_max_acceleration_m_s2"
            ),
            planner_latency_s=self._float("global_planner_latency_budget_s"),
            iterations=4,
        )
        self.get_logger().info(
            "route-constrained trailer intercept=(%.1f, %.1f), ETA=%.1fs"
            % (
                solution.position_enu_m[0],
                solution.position_enu_m[1],
                solution.eta_s,
            )
        )
        return solution.position_enu_m.copy()

    def _trailer_fresh(self, now: float) -> bool:
        return (
            self.trailer_state is not None
            and now - self.trailer_state_stamp <= self._float("trailer_timeout_s")
        )

    def _marker_world_enu(self, camera: str, now: float) -> np.ndarray | None:
        if camera == "front":
            stamp = self.front_marker_stamp
            estimator = self.front_marker_estimator
        elif camera == "down":
            stamp = self.down_marker_stamp
            estimator = self.down_marker_estimator
        else:
            raise ValueError("camera must be front or down")
        if stamp <= 0.0 or now - stamp > self._float("marker_timeout_s"):
            return None
        estimate = estimator.x[:3].copy()
        return estimate if np.all(np.isfinite(estimate)) else None

    def _landing_target_enu(
        self,
        current_world: np.ndarray,
        now: float,
        preferred_camera: str | None,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        if not self._trailer_fresh(now) or self.trailer_state is None:
            return None
        trailer = self.trailer_state.predict(max(0.0, now - self.trailer_state_stamp))
        target = trailer.position_enu_m.copy()
        if preferred_camera is not None:
            marker_world = self._marker_world_enu(preferred_camera, now)
            if marker_world is not None:
                # Blend vision with odometry.  The marker dominates laterally;
                # deck height stays tied to the configured physical model.
                target[:2] = 0.85 * marker_world[:2] + 0.15 * target[:2]
        target[2] = trailer.position_enu_m[2] + self._float("trailer_deck_height_m")
        return target, trailer.velocity_enu_m_s.copy(), trailer.yaw_enu_rad

    def _publish_trailer_relative_setpoint(
        self,
        current_ned: np.ndarray,
        now: float,
        clearance_m: float,
        camera: str | None,
    ) -> tuple[float, float] | None:
        current_world = self._px4_ned_to_enu(current_ned)
        estimate = self._landing_target_enu(current_world, now, camera)
        if estimate is None:
            self.last_landing_control_failure = "trailer_or_target_state_stale"
            return None
        deck, trailer_velocity, trailer_yaw = estimate
        horizon = self._int("landing_mpc_horizon_steps")
        dt = self._float("landing_mpc_dt_s")
        trailer_acceleration = (
            np.zeros(3, dtype=float)
            if self.trailer_state is None
            else self.trailer_state.acceleration_enu_m_s2
        )
        references_world: list[np.ndarray] = []
        reference_velocities_world: list[np.ndarray] = []
        bounds_ned: list[np.ndarray] = []
        horizontal_radius = (
            self._float("trailer_approach_radius_m")
            + self._float("hard_inflation_radius_m")
        )
        vertical_top = self._float("marker_search_height_m") + 3.0
        for step in range(horizon):
            future_s = dt * (step + 1)
            predicted_deck = (
                deck
                + trailer_velocity * future_s
                + 0.5 * trailer_acceleration * future_s**2
            )
            predicted_velocity = trailer_velocity + trailer_acceleration * future_s
            target = predicted_deck.copy()
            target[2] = deck[2] + max(0.0, float(clearance_m))
            references_world.append(target)
            reference_velocities_world.append(predicted_velocity)

            minimum_world = np.asarray(
                (
                    predicted_deck[0] - horizontal_radius,
                    predicted_deck[1] - horizontal_radius,
                    predicted_deck[2] + 0.02,
                ),
                dtype=float,
            )
            maximum_world = np.asarray(
                (
                    predicted_deck[0] + horizontal_radius,
                    predicted_deck[1] + horizontal_radius,
                    predicted_deck[2] + vertical_top,
                ),
                dtype=float,
            )
            corners = np.asarray(
                [
                    self._enu_to_px4_ned((x, y, z))
                    for x in (minimum_world[0], maximum_world[0])
                    for y in (minimum_world[1], maximum_world[1])
                    for z in (minimum_world[2], maximum_world[2])
                ],
                dtype=float,
            )
            bounds_ned.append(
                np.asarray(
                    (
                        np.min(corners[:, 0]),
                        np.max(corners[:, 0]),
                        np.min(corners[:, 1]),
                        np.max(corners[:, 1]),
                        np.min(corners[:, 2]),
                        np.max(corners[:, 2]),
                    ),
                    dtype=float,
                )
            )

        references_ned = np.vstack(
            [self._enu_to_px4_ned(point) for point in references_world]
        )
        reference_velocities_ned = np.vstack(
            [enu_to_ned_vector(value) for value in reference_velocities_world]
        )
        current_velocity_ned = (
            np.zeros(3, dtype=float)
            if self.local_position is None
            else np.asarray(
                (
                    self.local_position.vx,
                    self.local_position.vy,
                    self.local_position.vz,
                ),
                dtype=float,
            )
        )
        result = self.landing_mpc.solve(
            current_ned,
            current_velocity_ned,
            self.estimated_acceleration_ned,
            references_ned,
            reference_velocities_ned,
            np.vstack(bounds_ned),
        )
        self.last_landing_mpc_message = result.message
        self.last_landing_mpc_solve_time_s = result.solve_time_s
        self.last_landing_mpc_iterations = result.iterations
        self.last_landing_mpc_success = result.success
        if not result.success:
            self.hold_target_ned = current_ned.copy()
            self._publish_setpoint(self.hold_target_ned, np.zeros(3))
            self.last_landing_control_failure = f"landing_mpc:{result.message}"
            return None
        self.last_landing_control_failure = ""
        self.last_command_acceleration = result.accelerations_m_s2[0]
        self._publish_setpoint(
            result.positions_m[0],
            result.velocities_m_s[0],
            None,
            self.transform.yaw_enu_to_ned(trailer_yaw),
        )
        horizontal_error = float(np.linalg.norm(current_world[:2] - deck[:2]))
        relative_speed = float(
            np.linalg.norm(
                np.asarray(
                    (self.local_position.vy, self.local_position.vx), dtype=float
                )
                - trailer_velocity[:2]
            )
        ) if self.local_position is not None else math.inf
        return horizontal_error, relative_speed

    def _gate_dwell(self, condition: bool, now: float, dwell_s: float) -> bool:
        if not condition:
            self.landing_gate_since = None
            return False
        if self.landing_gate_since is None:
            self.landing_gate_since = now
            return False
        if now - self.landing_gate_since >= dwell_s:
            self.landing_gate_since = None
            return True
        return False

    def _run_active_path_tracking(
        self,
        current_ned: np.ndarray,
        velocity_ned: np.ndarray,
        now: float,
        resume_phase: MissionPhase,
    ) -> bool:
        """Run one bounded local-MPC step; return True at the active goal."""
        try:
            (
                references,
                reference_velocities,
                yaw_ned,
                _emergency,
                corridor_bounds_ned,
                obstacle_spheres_ned,
            ) = self._tracking_references(current_ned, velocity_ned, now)
        except CorridorViolationError as exc:
            self.hold_target_ned = current_ned.copy()
            self.hold_replan_resume_phase = resume_phase
            self._publish_setpoint(self.hold_target_ned, np.zeros(3))
            self._transition(
                MissionPhase.HOLD_REPLAN,
                f"SFC reference rejected: {exc}",
            )
            return False
        result = self.mpc.solve(
            current_ned,
            velocity_ned,
            self.estimated_acceleration_ned,
            references,
            reference_velocities,
            corridor_bounds_ned,
            obstacle_spheres_ned,
        )
        self.last_command_acceleration = result.accelerations_m_s2[0]
        self.last_mpc_message = result.message
        self.last_mpc_solve_time_s = result.solve_time_s
        self.last_mpc_iterations = result.iterations
        self.last_mpc_success = result.success
        if not result.success:
            self.hold_target_ned = current_ned.copy()
            self.hold_replan_resume_phase = resume_phase
            self._publish_setpoint(self.hold_target_ned, np.zeros(3))
            self._transition(
                MissionPhase.HOLD_REPLAN,
                f"jerk MPC rejected: {result.message}",
            )
            return False

        # PX4 remains the sole low-level feedback controller.  The optimized
        # state is sent as position plus velocity feed-forward; acceleration is
        # intentionally not sent a second time.
        self._publish_setpoint(
            result.positions_m[0],
            result.velocities_m_s[0],
            None,
            yaw_ned,
        )
        current_world = self._px4_ned_to_enu(current_ned)
        goal_error = float(np.linalg.norm(current_world[:2] - self.goal_enu))
        return bool(
            goal_error <= self._float("goal_tolerance_m")
            and self.path_index >= len(self.path_xy) - 3
        )

    def _nearest_progress_index(self, current_xy: np.ndarray) -> int:
        begin = max(0, self.path_index - 4)
        end = min(len(self.path_xy), self.path_index + 120)
        local = self.path_xy[begin:end]
        nearest = begin + int(np.argmin(np.linalg.norm(local - current_xy, axis=1)))
        self.path_index = max(self.path_index, nearest)
        return self.path_index

    def _point_at_distance(self, distance_m: float) -> np.ndarray:
        distance = float(np.clip(distance_m, 0.0, self.path_lengths[-1]))
        upper = int(np.searchsorted(self.path_lengths, distance, side="right"))
        if upper <= 0:
            return self.path_xyz[0].copy()
        if upper >= len(self.path_xyz):
            return self.path_xyz[-1].copy()
        lower = upper - 1
        span = self.path_lengths[upper] - self.path_lengths[lower]
        ratio = 0.0 if span <= 1.0e-9 else (distance - self.path_lengths[lower]) / span
        return self.path_xyz[lower] + ratio * (
            self.path_xyz[upper] - self.path_xyz[lower]
        )

    def _path_tangent(self, distance_m: float) -> np.ndarray:
        before = self._point_at_distance(max(0.0, distance_m - 1.0))
        after = self._point_at_distance(min(self.path_lengths[-1], distance_m + 1.0))
        return _unit_xy(after[:2] - before[:2])

    def _choose_avoidance_sign(
        self,
        current_xy: np.ndarray,
        nominal_xy: np.ndarray,
        tangent: np.ndarray,
    ) -> float:
        observation = self.sensors.depth()
        preferred = 1.0
        if observation is not None:
            preferred = (
                1.0
                if observation.left_near_points <= observation.right_near_points
                else -1.0
            )
        normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
        offset = self._float("local_bypass_offset_m")
        for sign in (preferred, -preferred):
            candidate = nominal_xy + sign * offset * normal
            if self.grid.is_free(candidate) and self.grid.segment_is_free(
                current_xy, candidate
            ):
                return sign
        return 0.0

    def _altitude_reference(
        self, current_world: np.ndarray, now: float, planned_world_z_m: float
    ) -> float:
        nominal = max(
            self.runtime_planner.minimum_planning_center_z_m,
            float(planned_world_z_m),
        )
        target = nominal
        downward = self.sensors.downward()
        if (
            downward is not None
            and now - downward.stamp_monotonic <= self._float("down_lidar_timeout_s")
            and math.isfinite(downward.range_m)
        ):
            surface_world_z = current_world[2] - downward.range_m
            target = max(nominal, surface_world_z + self._float("ground_clearance_m"))
        target = float(np.clip(target, nominal, self._float("max_world_altitude_m")))
        maximum_step = self._float("max_vertical_speed_m_s") / self._float(
            "control_rate_hz"
        )
        delta = float(
            np.clip(target - self.filtered_altitude_world_m, -maximum_step, maximum_step)
        )
        self.filtered_altitude_world_m += delta
        return self.filtered_altitude_world_m

    def _tracking_references(
        self,
        current_ned: np.ndarray,
        current_velocity_ned: np.ndarray,
        now: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        bool,
        np.ndarray,
        list[tuple[np.ndarray, float]],
    ]:
        current_world = self._px4_ned_to_enu(current_ned)
        index = self._nearest_progress_index(current_world[:2])
        progress = self.path_lengths[index]
        tangent = self._path_tangent(progress + self._float("path_lookahead_m"))
        normal = np.asarray((-tangent[1], tangent[0]), dtype=float)

        depth = self.sensors.depth()
        depth_fresh = (
            depth is not None
            and now - depth.stamp_monotonic <= self._float("depth_timeout_s")
        )
        obstacle = depth_fresh and depth.nearest_m < self._float(
            "depth_trigger_distance_m"
        )
        emergency = depth_fresh and depth.nearest_m < self._float(
            "depth_emergency_distance_m"
        )
        nominal_point = self._point_at_distance(
            progress + self._float("path_lookahead_m")
        )
        nominal_xy = nominal_point[:2]
        if obstacle:
            chosen = self._choose_avoidance_sign(current_world[:2], nominal_xy, tangent)
            self.avoidance_sign = chosen
            self.avoidance_until = now + self._float("local_obstacle_ttl_s")
            if chosen == 0.0:
                emergency = True
        elif now >= self.avoidance_until:
            self.avoidance_sign = 0.0

        downward = self.sensors.downward()
        down_fresh = (
            downward is not None
            and now - downward.stamp_monotonic <= self._float("down_lidar_timeout_s")
        )
        if down_fresh and downward is not None:
            minimum_clearance = self._float("ground_clearance_m") - self._float(
                "ground_clearance_tolerance_m"
            )
            if downward.range_m < minimum_clearance:
                emergency = True

        target_offset = self.avoidance_sign * self._float("local_bypass_offset_m")
        self.avoidance_offset_m += 0.2 * (target_offset - self.avoidance_offset_m)
        if abs(self.avoidance_offset_m) < 0.02:
            self.avoidance_offset_m = 0.0

        altitude = self._altitude_reference(
            current_world, now, float(nominal_point[2])
        )
        horizon = self._int("local_mpc_horizon_steps")
        dt = self._float("local_mpc_dt_s")
        speed = 0.0 if emergency else self._float("cruise_speed_m_s")
        references_world: list[np.ndarray] = []
        for step in range(horizon):
            distance = progress + max(0.75, speed * dt * (step + 1))
            point = self._point_at_distance(distance)
            xy = point[:2]
            candidate_xy = xy + self.avoidance_offset_m * normal
            if self.grid.is_free(candidate_xy):
                xy = candidate_xy
            references_world.append(
                np.asarray((xy[0], xy[1], max(altitude, point[2])), dtype=float)
            )

        if emergency:
            # Stop horizontal motion and climb while the depth obstacle is
            # inside the emergency envelope.
            emergency_altitude = min(
                self._float("max_world_altitude_m"),
                max(altitude, current_world[2] + 3.0),
            )
            references_world = [
                np.asarray((current_world[0], current_world[1], emergency_altitude))
                for _ in range(horizon)
            ]

        references_ned = np.vstack(
            [self._enu_to_px4_ned(point) for point in references_world]
        )
        corridor_mapping = self.active_plan.horizon_sfc_bounds_ned(
            references_world, self.transform
        )
        corridor_bounds_ned = np.asarray(
            corridor_mapping.bounds_px4_local_ned_m, dtype=float
        ).copy()
        if self.runtime_origin_ned is not None:
            # The certificate is expressed in the declared map-local NED
            # frame.  PX4 can initialize its EKF local origin at a non-zero
            # NED value; references already include that delta, therefore the
            # box bounds must receive exactly the same translation.
            corridor_bounds_ned[:, 0:2] += self.runtime_origin_ned[0]
            corridor_bounds_ned[:, 2:4] += self.runtime_origin_ned[1]
            corridor_bounds_ned[:, 4:6] += self.runtime_origin_ned[2]
        obstacle_spheres_ned: list[tuple[np.ndarray, float]] = []
        if obstacle and depth is not None and math.isfinite(depth.nearest_m):
            horizontal_speed = float(np.linalg.norm(current_velocity_ned[:2]))
            dynamic_radius = (
                self._float("hard_inflation_radius_m")
                + horizontal_speed * self._float("planner_reaction_time_s")
                + horizontal_speed**2
                / (2.0 * self._float("braking_acceleration_m_s2"))
                + self._float("dynamic_obstacle_uncertainty_m")
            )
            obstacle_world = np.asarray(
                (
                    current_world[0] + tangent[0] * depth.nearest_m,
                    current_world[1] + tangent[1] * depth.nearest_m,
                    current_world[2],
                )
            )
            obstacle_spheres_ned.append(
                (self._enu_to_px4_ned(obstacle_world), dynamic_radius)
            )
        velocities = np.vstack(
            (
                (references_ned[0] - current_ned) / dt,
                np.diff(references_ned, axis=0) / dt,
            )
        )
        horizontal = np.linalg.norm(velocities[:, :2], axis=1)
        maximum = self._float("max_horizontal_speed_m_s")
        for row, value in enumerate(horizontal):
            if value > maximum:
                velocities[row, :2] *= maximum / value
        velocities[:, 2] = np.clip(
            velocities[:, 2],
            -self._float("max_vertical_speed_m_s"),
            self._float("max_vertical_speed_m_s"),
        )
        yaw_ned = math.atan2(float(tangent[0]), float(tangent[1]))
        return (
            references_ned,
            velocities,
            yaw_ned,
            bool(emergency),
            corridor_bounds_ned,
            obstacle_spheres_ned,
        )

    def _begin_plan_join(
        self, resume_phase: MissionPhase, explanation: str
    ) -> None:
        """Enter a bounded join to the first point of a certified 3-D plan."""

        self.plan_join_resume_phase = resume_phase
        self.plan_join_within_since = None
        self._transition(MissionPhase.GLOBAL_PLAN, explanation)

    def _run_plan_join(self, current_ned: np.ndarray, now: float) -> None:
        """Climb/translate to the active SFC start before enabling local MPC.

        The requested takeoff is exactly 10 m AGL, whereas the complete
        1 m-class vehicle plus 10 m underside-clearance contract requires a
        slightly higher centre altitude before an SFC box can exist.  Initial
        joining therefore climbs vertically first.  Subsequent replans require
        the entire short join segment to be hard-inflated collision free.
        """

        current_world = self._px4_ned_to_enu(current_ned)
        target_world = np.asarray(self.path_xyz[0], dtype=float)
        minimum_z = self.runtime_planner.minimum_planning_center_z_m
        if current_world[2] < minimum_z - 0.05:
            target_world = np.asarray(
                (current_world[0], current_world[1], self.path_xyz[0, 2]),
                dtype=float,
            )
        elif not self.runtime_planner.occupancy.segment_is_free(
            current_world, target_world, inflated=True, sample_step_m=0.20
        ):
            self.hold_target_ned = current_ned.copy()
            self._publish_setpoint(self.hold_target_ned, np.zeros(3))
            self.hold_replan_resume_phase = MissionPhase.GLOBAL_PLAN
            self._transition(
                MissionPhase.HOLD_REPLAN,
                "active-plan join segment failed the hard 3-D occupancy check",
            )
            return

        delta = target_world - current_world
        horizontal_norm = float(np.linalg.norm(delta[:2]))
        control_rate = self._float("control_rate_hz")
        horizontal_step = self._float("max_horizontal_speed_m_s") / control_rate
        vertical_step = self._float("max_vertical_speed_m_s") / control_rate
        reference_world = current_world.copy()
        if horizontal_norm > horizontal_step > 0.0:
            reference_world[:2] += delta[:2] * horizontal_step / horizontal_norm
        else:
            reference_world[:2] = target_world[:2]
        reference_world[2] += float(
            np.clip(delta[2], -vertical_step, vertical_step)
        )
        self._publish_setpoint(self._enu_to_px4_ned(reference_world))

        position_error = float(np.linalg.norm(current_world - self.path_xyz[0]))
        speed = 0.0 if self.local_position is None else float(
            np.linalg.norm(
                (self.local_position.vx, self.local_position.vy, self.local_position.vz)
            )
        )
        joined = position_error <= 0.40 and speed <= 0.60
        if not joined:
            self.plan_join_within_since = None
        elif self.plan_join_within_since is None:
            self.plan_join_within_since = now
        elif now - self.plan_join_within_since >= 0.50:
            self.filtered_altitude_world_m = float(self.path_xyz[0, 2])
            self._transition(
                self.plan_join_resume_phase,
                "certified 3-D path start reached; enabling SFC-constrained MPC",
            )

    def _control_tick(self) -> None:
        now = time.monotonic()
        motion_now = self.control_time_s
        airborne_phases = (
            MissionPhase.TAKEOFF,
            MissionPhase.GLOBAL_PLAN,
            MissionPhase.TRACKING,
            MissionPhase.HOLDING,
            MissionPhase.RETURN_REQUESTED,
            MissionPhase.TRAILER_INTERCEPT_PLAN,
            MissionPhase.TRAILER_INTERCEPT_TRACK,
            MissionPhase.MARKER_SEARCH_FRONT,
            MissionPhase.MARKER_ACQUIRE_FRONT,
            MissionPhase.CAMERA_HANDOFF,
            MissionPhase.MARKER_TRACK_DOWN,
            MissionPhase.VELOCITY_MATCH,
            MissionPhase.PRECISION_ALIGN,
            MissionPhase.PRECISION_DESCENT,
            MissionPhase.FINAL_APPROACH,
            MissionPhase.TOUCHDOWN_CONFIRM,
            MissionPhase.HOLD_REPLAN,
            MissionPhase.ABORT_CLIMB,
            MissionPhase.SENSOR_HOLD,
            MissionPhase.FAILSAFE_HOLD,
        )
        state = self._current_ned()
        if state is None:
            status = self.vehicle_status
            armed_during_arming = (
                self.phase == MissionPhase.ARMING
                and status is not None
                and status.arming_state == VehicleStatus.ARMING_STATE_ARMED
            )
            if self.phase in airborne_phases or armed_during_arming:
                self._transition(
                    MissionPhase.ABORT_LAND,
                    "PX4 local position became invalid or stale",
                )
            if self.phase == MissionPhase.ABORT_LAND:
                self._request_auto_land(now)
            self._publish_status(now, None)
            return
        current_ned, velocity_ned = state

        if self._emergency_land_request.is_set():
            self._emergency_land_request.clear()
            self._transition(
                MissionPhase.ABORT_LAND,
                "operator requested the separate emergency-land command",
            )

        if self._return_request.is_set() and self.phase in (
            MissionPhase.GLOBAL_PLAN,
            MissionPhase.TRACKING,
            MissionPhase.HOLDING,
            MissionPhase.HOLD_REPLAN,
            MissionPhase.FAILSAFE_HOLD,
        ):
            self._return_request.clear()
            self._cancel_pending_plan("operator LAND changed mission ownership")
            self._transition(
                MissionPhase.RETURN_REQUESTED,
                "operator LAND means return to the moving trailer",
            )

        status = self.vehicle_status
        status_fresh = (
            status is not None
            and now - self.vehicle_status_stamp
            <= self._float("vehicle_status_timeout_s")
        )
        if self.phase in airborne_phases:
            abort_reason = ""
            if not status_fresh:
                abort_reason = "PX4 vehicle status became stale"
            elif status is not None and status.failsafe:
                abort_reason = "PX4 reported failsafe"
            elif (
                status is not None
                and status.arming_state != VehicleStatus.ARMING_STATE_ARMED
            ):
                abort_reason = "unexpected disarm; automatic re-arm is prohibited"
            elif (
                status is not None
                and status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ):
                abort_reason = "unexpected OFFBOARD exit; automatic re-entry is prohibited"
            if abort_reason:
                self._transition(MissionPhase.ABORT_LAND, abort_reason)

        if not self.ekf_reset_sequence_valid:
            disarmed = (
                status is None
                or status.arming_state != VehicleStatus.ARMING_STATE_ARMED
            )
            if self.phase in (
                MissionPhase.WAITING,
                MissionPhase.PRESTREAM,
                MissionPhase.ARMING,
            ) and disarmed:
                self.get_logger().warning(
                    "Re-anchoring at the stationary spawn after missed EKF reset events"
                )
                self._capture_runtime_origin(current_ned)
                self.ekf_reset_sequence_valid = True
            else:
                self._transition(
                    MissionPhase.ABORT_LAND,
                    "missed PX4 EKF reset events; world/local transform is unsafe",
                )

        if self.phase == MissionPhase.ABORT_LAND:
            self._request_auto_land(now)
            self._publish_status(now, current_ned)
            return

        depth_ready, down_ready = self._sensor_state(now)
        sensors_ready = depth_ready and down_ready
        if self.phase == MissionPhase.WAITING:
            if self._bool("require_sensors_before_arm") and not sensors_ready:
                self._publish_setpoint(current_ned)
                if not self._sensor_wait_warning_sent:
                    self.get_logger().warning(
                        "Waiting for native Gazebo depth and downward-lidar data before arming"
                    )
                    self._sensor_wait_warning_sent = True
                self._publish_status(now, current_ned)
                return
            if self.runtime_origin_ned is None:
                self._capture_runtime_origin(current_ned)
            self._transition(MissionPhase.PRESTREAM, "PX4 local position and sensors ready")

        if (
            self.phase in (MissionPhase.PRESTREAM, MissionPhase.ARMING)
            and self._bool("require_sensors_before_arm")
            and not sensors_ready
        ):
            if status is not None and status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self._transition(
                    MissionPhase.ABORT_LAND,
                    "required sensor became stale during arming",
                )
                self._request_auto_land(now)
            else:
                self._transition(MissionPhase.WAITING, "required sensor became stale")
                self._publish_setpoint(current_ned)
            self._publish_status(now, current_ned)
            return

        if (
            self.phase in airborne_phases
            and self.phase not in (
                MissionPhase.SENSOR_HOLD,
                MissionPhase.FAILSAFE_HOLD,
                MissionPhase.ABORT_CLIMB,
            )
            and self._bool("require_sensors_before_arm")
            and not sensors_ready
        ):
            terminal_phases = (
                MissionPhase.MARKER_SEARCH_FRONT,
                MissionPhase.MARKER_ACQUIRE_FRONT,
                MissionPhase.CAMERA_HANDOFF,
                MissionPhase.MARKER_TRACK_DOWN,
                MissionPhase.VELOCITY_MATCH,
                MissionPhase.PRECISION_ALIGN,
                MissionPhase.PRECISION_DESCENT,
                MissionPhase.FINAL_APPROACH,
                MissionPhase.TOUCHDOWN_CONFIRM,
            )
            if self.phase in terminal_phases:
                self.sensor_resume_phase = MissionPhase.ABORT_CLIMB
                self.landing_abort_count += 1
                self.touchdown_gate.reset()
            else:
                self.sensor_resume_phase = self.phase
            self.hold_target_ned = current_ned.copy()
            self._transition(
                MissionPhase.SENSOR_HOLD,
                "depth or downward-lidar data became stale",
            )

        if self.phase == MissionPhase.SENSOR_HOLD:
            if sensors_ready:
                self._transition(self.sensor_resume_phase, "required sensors recovered")
            else:
                if self.hold_target_ned is None:
                    self.hold_target_ned = current_ned.copy()
                self._publish_setpoint(self.hold_target_ned)
                if now - self.phase_started >= self._float("sensor_hold_land_timeout_s"):
                    self._transition(
                        MissionPhase.ABORT_LAND,
                        "required sensors did not recover within hold timeout",
                    )
                    self._request_auto_land(now)
                self._publish_status(now, current_ned)
                return

        # Completion is polled without waiting.  Installation and all mission
        # state mutation stay on this control callback, while the worker owns
        # the slow A*/SFC/B-spline computation.
        self._poll_pending_replan(current_ned, now)

        if self.phase == MissionPhase.PRESTREAM:
            self._publish_setpoint(current_ned)
            if motion_now - self.phase_started_control_s >= self._float(
                "offboard_prestream_s"
            ):
                self._transition(MissionPhase.ARMING, "setpoint stream established")

        elif self.phase == MissionPhase.ARMING:
            assert self.takeoff_start_ned is not None
            # Hold the contact point while PX4 accepts OFFBOARD and arm.  The
            # vertical reference ramp starts only after the armed-state edge.
            self._publish_setpoint(self.takeoff_start_ned)
            self._request_offboard_and_arm(now)
            status = self.vehicle_status
            if (
                status is not None
                and status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
                and status.arming_state == VehicleStatus.ARMING_STATE_ARMED
            ):
                self._transition(MissionPhase.TAKEOFF, "armed in PX4 OFFBOARD")

        elif self.phase == MissionPhase.TAKEOFF:
            assert self.takeoff_target_ned is not None
            assert self.takeoff_start_ned is not None
            # Do not step the PX4 position target from ground level straight to
            # 10 m.  A bounded vertical reference avoids a large initial thrust
            # transient and makes the same takeoff reproducible across Gazebo
            # real-time factors.
            climb_m = min(
                self._float("takeoff_height_m"),
                self._float("takeoff_rate_m_s")
                * max(0.0, motion_now - self.phase_started_control_s),
            )
            takeoff_reference = self.takeoff_start_ned.copy()
            takeoff_reference[2] -= climb_m
            self._publish_setpoint(takeoff_reference)
            error = float(np.linalg.norm(current_ned - self.takeoff_target_ned))
            if error <= self._float("takeoff_tolerance_m"):
                if self.takeoff_within_since is None:
                    self.takeoff_within_since = motion_now
                elif motion_now - self.takeoff_within_since >= self._float(
                    "takeoff_hold_s"
                ):
                    self._begin_plan_join(
                        MissionPhase.TRACKING,
                        "10 m takeoff complete; joining the certified 3-D corridor",
                    )
            else:
                self.takeoff_within_since = None

        elif self.phase == MissionPhase.GLOBAL_PLAN:
            self._run_plan_join(current_ned, motion_now)

        elif self.phase == MissionPhase.TRACKING:
            if self._run_active_path_tracking(
                current_ned,
                velocity_ned,
                now,
                MissionPhase.TRACKING,
            ):
                goal_world = np.asarray(
                    (
                        self.goal_enu[0],
                        self.goal_enu[1],
                        self.filtered_altitude_world_m,
                    )
                )
                self.hold_target_ned = self._enu_to_px4_ned(goal_world)
                self.waypoint_hold_started = motion_now
                self._transition(MissionPhase.HOLDING, "ENU goal reached")

        elif self.phase == MissionPhase.HOLDING:
            if self.hold_target_ned is None:
                self.hold_target_ned = current_ned.copy()
            self._publish_setpoint(self.hold_target_ned)
            if self.waypoint_hold_started is None:
                self.waypoint_hold_started = motion_now
            if (
                self._bool("repeat_waypoints")
                and motion_now - self.waypoint_hold_started
                >= self._float("waypoint_hold_s")
            ):
                next_index = (self.waypoint_index + 1) % len(self.waypoints_enu)
                next_goal = self.waypoints_enu[next_index]
                current_world = self._px4_ned_to_enu(current_ned)
                if self._submit_replan_from_world(
                    current_world,
                    next_goal,
                    MissionPhase.TRACKING,
                    "waypoint",
                ):
                    self.pending_waypoint_index = next_index
                    self.waypoint_hold_started = None
                    self._transition(
                        MissionPhase.HOLD_REPLAN,
                        f"planning waypoint loop index {next_index}",
                    )
                else:
                    self.waypoint_hold_started = motion_now

        elif self.phase == MissionPhase.RETURN_REQUESTED:
            self.hold_target_ned = current_ned.copy()
            self._publish_setpoint(self.hold_target_ned, np.zeros(3))
            if not self._trailer_fresh(now) or self.trailer_state is None:
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    "trailer state unavailable; return cannot be planned safely",
                )
            else:
                current_world = self._px4_ned_to_enu(current_ned)
                intercept = self._select_trailer_intercept(current_world)
                if intercept is None:
                    self._transition(
                        MissionPhase.FAILSAFE_HOLD,
                        "route-constrained trailer prediction unavailable",
                    )
                else:
                    self.intercept_target_enu = intercept
                    if self._submit_replan_from_world(
                        current_world,
                        intercept[:2],
                        MissionPhase.TRAILER_INTERCEPT_TRACK,
                        "intercept",
                    ):
                        self._transition(
                            MissionPhase.TRAILER_INTERCEPT_PLAN,
                            "asynchronous route-constrained intercept planning",
                        )
                    else:
                        self._transition(
                            MissionPhase.FAILSAFE_HOLD,
                            "could not submit moving-trailer intercept plan",
                        )

        elif self.phase == MissionPhase.TRAILER_INTERCEPT_PLAN:
            if self.hold_target_ned is None:
                self.hold_target_ned = current_ned.copy()
            self._publish_setpoint(self.hold_target_ned, np.zeros(3))
            if self.pending_plan_request_id is None:
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    "intercept planner ended without an installable plan",
                )

        elif self.phase == MissionPhase.TRAILER_INTERCEPT_TRACK:
            if not self._trailer_fresh(now) or self.trailer_state is None:
                self.hold_target_ned = current_ned.copy()
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    "trailer state became stale during intercept",
                )
            else:
                current_world = self._px4_ned_to_enu(current_ned)
                trailer_prediction = self.trailer_state.predict(
                    max(0.0, now - self.trailer_state_stamp)
                )
                distance_to_trailer = float(
                    np.linalg.norm(
                        current_world[:2] - trailer_prediction.position_enu_m[:2]
                    )
                )
                if distance_to_trailer <= self._float("trailer_approach_radius_m"):
                    self._cancel_pending_plan(
                        "terminal camera guidance acquired mission ownership"
                    )
                    self._transition(
                        MissionPhase.MARKER_SEARCH_FRONT,
                        "inside trailer approach radius; acquiring front ArUco",
                    )
                else:
                    self._run_active_path_tracking(
                        current_ned,
                        velocity_ned,
                        now,
                        MissionPhase.TRAILER_INTERCEPT_TRACK,
                    )
                    replan_period = max(
                        self._float("trailer_replan_period_s"),
                        1.0 / self._float("global_replan_rate_hz"),
                    )
                    if (
                        self.pending_plan_request_id is None
                        and motion_now - self.last_intercept_plan >= replan_period
                    ):
                        updated = self._select_trailer_intercept(current_world)
                        shift = (
                            math.inf
                            if updated is None or self.intercept_target_enu is None
                            else float(
                                np.linalg.norm(
                                    updated[:2] - self.intercept_target_enu[:2]
                                )
                            )
                        )
                        if (
                            updated is not None
                            and shift >= self._float("trailer_replan_distance_m")
                        ):
                            self._submit_replan_from_world(
                                current_world,
                                updated[:2],
                                MissionPhase.TRAILER_INTERCEPT_TRACK,
                                "intercept",
                            )

        elif self.phase == MissionPhase.MARKER_SEARCH_FRONT:
            result = self._publish_trailer_relative_setpoint(
                current_ned,
                now,
                self._float("marker_search_height_m"),
                None,
            )
            if result is None:
                self.hold_target_ned = current_ned.copy()
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    f"marker-search controller failed: {self.last_landing_control_failure}",
                )
            else:
                front_fresh = self._marker_world_enu("front", now) is not None
                if self._gate_dwell(
                    front_fresh,
                    motion_now,
                    self._float("marker_acquire_dwell_s"),
                ):
                    self._transition(
                        MissionPhase.MARKER_ACQUIRE_FRONT,
                        "front marker continuously acquired",
                    )

        elif self.phase == MissionPhase.MARKER_ACQUIRE_FRONT:
            result = self._publish_trailer_relative_setpoint(
                current_ned,
                now,
                self._float("marker_search_height_m"),
                "front",
            )
            if result is None:
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    f"front approach controller failed: {self.last_landing_control_failure}",
                )
            elif self._marker_world_enu("front", now) is None:
                self._transition(
                    MissionPhase.MARKER_SEARCH_FRONT,
                    "front marker lost before camera handoff",
                )
            elif result is not None and result[0] <= 3.0:
                self._transition(
                    MissionPhase.CAMERA_HANDOFF,
                    "front-guided approach centered; requiring overlap with down camera",
                )

        elif self.phase == MissionPhase.CAMERA_HANDOFF:
            result = self._publish_trailer_relative_setpoint(
                current_ned,
                now,
                self._float("marker_search_height_m"),
                "front",
            )
            both_fresh = (
                self._marker_world_enu("front", now) is not None
                and self._marker_world_enu("down", now) is not None
            )
            if result is None:
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    f"camera-handoff controller failed: {self.last_landing_control_failure}",
                )
            elif self._gate_dwell(
                both_fresh,
                motion_now,
                self._float("camera_handoff_overlap_s"),
            ):
                self._transition(
                    MissionPhase.MARKER_TRACK_DOWN,
                    "front/down overlap validated; down camera owns terminal guidance",
                )
            elif self._marker_world_enu("front", now) is None:
                self._transition(
                    MissionPhase.MARKER_SEARCH_FRONT,
                    "handoff failed because front marker became stale",
                )

        elif self.phase == MissionPhase.MARKER_TRACK_DOWN:
            result = self._publish_trailer_relative_setpoint(
                current_ned,
                now,
                self._float("velocity_match_height_m"),
                "down",
            )
            if result is None:
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    f"down-track controller failed: {self.last_landing_control_failure}",
                )
            elif self._marker_world_enu("down", now) is None:
                self.landing_abort_count += 1
                self._transition(
                    MissionPhase.ABORT_CLIMB,
                    "down marker lost after handoff",
                )
            elif result is not None and self._gate_dwell(
                result[0] <= 1.0,
                motion_now,
                self._float("marker_acquire_dwell_s"),
            ):
                self._transition(
                    MissionPhase.VELOCITY_MATCH,
                    "down marker centered; matching trailer velocity",
                )

        elif self.phase == MissionPhase.VELOCITY_MATCH:
            result = self._publish_trailer_relative_setpoint(
                current_ned,
                now,
                self._float("velocity_match_height_m"),
                "down",
            )
            if result is None:
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    f"velocity-match controller failed: {self.last_landing_control_failure}",
                )
            elif self._marker_world_enu("down", now) is None:
                self.landing_abort_count += 1
                self._transition(MissionPhase.ABORT_CLIMB, "marker lost in velocity match")
            elif result is not None and self._gate_dwell(
                result[0] <= 0.75
                and result[1] <= self._float("relative_speed_gate_m_s"),
                motion_now,
                0.8,
            ):
                self._transition(
                    MissionPhase.PRECISION_ALIGN,
                    "relative velocity and lateral error are inside gates",
                )

        elif self.phase == MissionPhase.PRECISION_ALIGN:
            result = self._publish_trailer_relative_setpoint(
                current_ned,
                now,
                self._float("precision_align_height_m"),
                "down",
            )
            if result is None:
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    f"precision-align controller failed: {self.last_landing_control_failure}",
                )
            elif self._marker_world_enu("down", now) is None:
                self.landing_abort_count += 1
                self._transition(MissionPhase.ABORT_CLIMB, "marker lost during alignment")
            elif result is not None and self._gate_dwell(
                result[0] <= self._float("precision_lateral_gate_m")
                and result[1] <= self._float("relative_speed_gate_m_s"),
                motion_now,
                0.8,
            ):
                self.landing_descent_start_clearance_m = self._float(
                    "precision_align_height_m"
                )
                self._transition(
                    MissionPhase.PRECISION_DESCENT,
                    "alignment stable; starting bounded precision descent",
                )

        elif self.phase == MissionPhase.PRECISION_DESCENT:
            initial = (
                self._float("precision_align_height_m")
                if self.landing_descent_start_clearance_m is None
                else self.landing_descent_start_clearance_m
            )
            clearance = max(
                self._float("final_approach_height_m"),
                initial
                - self._float("precision_descent_speed_m_s")
                * max(0.0, motion_now - self.phase_started_control_s),
            )
            result = self._publish_trailer_relative_setpoint(
                current_ned, now, clearance, "down"
            )
            if result is None:
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    f"precision-descent controller failed: {self.last_landing_control_failure}",
                )
            elif self._marker_world_enu("down", now) is None:
                self.landing_abort_count += 1
                self._transition(MissionPhase.ABORT_CLIMB, "marker lost during descent")
            elif result is not None and result[0] > 2.0 * self._float(
                "precision_lateral_gate_m"
            ):
                self.landing_abort_count += 1
                self._transition(MissionPhase.ABORT_CLIMB, "lateral gate violated in descent")
            elif clearance <= self._float("final_approach_height_m") + 1.0e-6:
                self._transition(
                    MissionPhase.FINAL_APPROACH,
                    "precision descent complete; entering slow final approach",
                )

        elif self.phase in (
            MissionPhase.FINAL_APPROACH,
            MissionPhase.TOUCHDOWN_CONFIRM,
        ):
            final_initial = (
                self._float("final_approach_height_m")
                if self.phase == MissionPhase.FINAL_APPROACH
                or self.landing_descent_start_clearance_m is None
                else self.landing_descent_start_clearance_m
            )
            clearance = max(
                0.05,
                final_initial
                - self._float("final_descent_speed_m_s")
                * max(0.0, motion_now - self.phase_started_control_s),
            )
            landing_result = self._publish_trailer_relative_setpoint(
                current_ned, now, clearance, "down"
            )
            marker_fresh = self._marker_world_enu("down", now) is not None
            lateral_speed_safe = bool(
                landing_result is not None
                and landing_result[0] <= self._float("precision_lateral_gate_m")
                and landing_result[1] <= self._float("relative_speed_gate_m_s")
            )
            if not marker_fresh or not lateral_speed_safe:
                self.touchdown_gate.reset()
                self.landing_abort_count += 1
                self._transition(
                    MissionPhase.ABORT_CLIMB,
                    "terminal marker/lateral/relative-speed gate violated",
                )
            else:
                downward = self.sensors.downward()
                range_to_deck = None if downward is None else downward.range_m
                vertical_speed_enu = -float(velocity_ned[2])
                trailer_vertical_speed = (
                    0.0
                    if self.trailer_state is None
                    else float(self.trailer_state.velocity_enu_m_s[2])
                )
                land_state_fresh = bool(
                    self.land_detected is not None
                    and now - self.land_detected_stamp
                    <= self._float("land_detected_timeout_s")
                )
                detected = self.land_detected if land_state_fresh else None
                if detected is None:
                    self.touchdown_gate.reset()
                touchdown = self.touchdown_gate.update(
                    motion_now,
                    range_to_deck,
                    vertical_speed_enu - trailer_vertical_speed,
                    bool(detected is not None and detected.ground_contact),
                    bool(detected is not None and detected.landed),
                    bool(detected is not None and detected.at_rest),
                )
                if self.phase == MissionPhase.FINAL_APPROACH and clearance <= 0.20:
                    self.landing_descent_start_clearance_m = clearance
                    self._transition(
                        MissionPhase.TOUCHDOWN_CONFIRM,
                        "near deck; waiting for continuous multi-signal touchdown proof",
                    )
                if touchdown:
                    self._transition(
                        MissionPhase.DISARM,
                        "touchdown continuously confirmed; normal disarm allowed",
                    )

        elif self.phase == MissionPhase.DISARM:
            self._request_normal_disarm(now)

        elif self.phase == MissionPhase.ABORT_CLIMB:
            result = self._publish_trailer_relative_setpoint(
                current_ned,
                now,
                self._float("marker_search_height_m"),
                None,
            )
            if self.landing_abort_count > self._int("landing_max_abort_count"):
                self.hold_target_ned = current_ned.copy()
                self._transition(
                    MissionPhase.FAILSAFE_HOLD,
                    "precision landing retry limit reached; holding for operator",
                )
            elif result is not None:
                current_world = self._px4_ned_to_enu(current_ned)
                target = self._landing_target_enu(current_world, now, None)
                if target is not None:
                    clearance = current_world[2] - target[0][2]
                    if clearance >= self._float("marker_search_height_m") - 0.5:
                        self._transition(
                            MissionPhase.MARKER_SEARCH_FRONT,
                            "abort climb complete; restarting marker acquisition",
                        )

        elif self.phase == MissionPhase.FAILSAFE_HOLD:
            if self.hold_target_ned is None:
                self.hold_target_ned = current_ned.copy()
            self._publish_setpoint(self.hold_target_ned, np.zeros(3))

        elif self.phase == MissionPhase.HOLD_REPLAN:
            if self.hold_target_ned is None:
                self.hold_target_ned = current_ned.copy()
            self._publish_setpoint(self.hold_target_ned, np.zeros(3))
            if (
                self.pending_plan_request_id is None
                and now - self.phase_started >= 1.0
            ):
                self._transition(
                    self.hold_replan_resume_phase,
                    "retrying local MPC from a stationary safe hold",
                )

        self._publish_status(now, current_ned)

    def _publish_status(self, now: float, current_ned: np.ndarray | None) -> None:
        if now - self.last_status_publish < 0.5:
            return
        self.last_status_publish = now
        depth = self.sensors.depth()
        down = self.sensors.downward()
        current_world = (
            None
            if current_ned is None
            else [float(value) for value in self._px4_ned_to_enu(current_ned)]
        )
        payload = {
            "phase": self.phase.value,
            "goal_enu_m": [float(value) for value in self.goal_enu],
            "current_enu_m": current_world,
            "path_index": self.path_index,
            "path_points": len(self.path_xy),
            "path_length_m": float(self.path_lengths[-1]),
            "bspline": self.active_plan.mode is PlanningOutputMode.SFC_BSPLINE,
            "active_plan_mode": self.active_plan.mode.value,
            "sfc_boxes": len(self.active_plan.corridor.boxes),
            "guidance_mode": str(self.get_parameter("guidance_mode").value),
            "depth_nearest_m": (
                None if depth is None or not math.isfinite(depth.nearest_m) else depth.nearest_m
            ),
            "down_lidar_m": None if down is None else down.range_m,
            "avoidance_offset_m": self.avoidance_offset_m,
            "mpc": self.last_mpc_message,
            "mpc_success": self.last_mpc_success,
            "mpc_solve_ms": 1000.0 * self.last_mpc_solve_time_s,
            "mpc_iterations": self.last_mpc_iterations,
            "landing_mpc": self.last_landing_mpc_message,
            "landing_mpc_success": self.last_landing_mpc_success,
            "landing_mpc_solve_ms": 1000.0
            * self.last_landing_mpc_solve_time_s,
            "landing_mpc_iterations": self.last_landing_mpc_iterations,
            "waypoint_index": self.waypoint_index,
            "trailer_fresh": self._trailer_fresh(now),
            "front_marker_age_s": (
                None if self.front_marker_stamp <= 0.0 else now - self.front_marker_stamp
            ),
            "down_marker_age_s": (
                None if self.down_marker_stamp <= 0.0 else now - self.down_marker_stamp
            ),
            "landing_abort_count": self.landing_abort_count,
            "px4_sample_dt_s": self.last_control_sample_dt_s,
            "estimated_sim_rtf": (
                None
                if not math.isfinite(self.estimated_sim_rtf)
                else self.estimated_sim_rtf
            ),
            "vehicle_attitude_age_s": (
                None
                if self.vehicle_attitude_stamp <= 0.0
                else now - self.vehicle_attitude_stamp
            ),
            "gz_partition": os.environ.get("GZ_PARTITION", ""),
        }
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(message)

    def destroy_node(self) -> bool:
        """Stop the background planner before tearing down ROS publishers."""

        self.plan_worker.close(timeout_s=2.0)
        return super().destroy_node()


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node: CityAutonomyNode | None = None
    executor: MultiThreadedExecutor | None = None
    try:
        node = CityAutonomyNode()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
