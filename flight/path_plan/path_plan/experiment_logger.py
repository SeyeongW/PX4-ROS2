#!/usr/bin/env python3
"""Common CSV experiment logger for hardware, CJU and Gazebo runs.

The node is observation-only: it subscribes to vehicle, planner, controller and
ArUco diagnostics and never publishes a command.  One timestamped event CSV and
one one-row summary CSV are closed on mission ``DONE``, disarm, or Ctrl-C.

Metric contract
---------------
* path length: 3-D distance flown while armed/mission-active.
* tracking error: horizontal distance to the B-spline active at that sample.
* clearance: vehicle-centre distance to the raw obstacle surface (the
  configured vehicle clearance remains visible instead of being subtracted).
* A*/MPC times: arithmetic means of existing ``/path_plan/*_stats`` events.
* SFC: actual optimizer-box generation time, horizontal widths/count, and
  continuous vehicle excursions outside the active horizontal box union.
* replan count: accepted plans after the first plan of each mission leg.
* ArUco rate: accepted/processed detector frames during SEARCH/ACQUIRE/DESCEND.
* relative XY error: mean marker-to-vehicle horizontal error in that window.
* landing XY error: last fresh vision error at the normal P-control handover.
* touchdown relative speed: onboard vehicle/marker velocity estimate at
  contact.

The last two are onboard estimates, not external ground truth.  On the real
aircraft the marker leaves view before physical contact, so the CSV records
that limitation in the corresponding ``*_source`` summary columns.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path as FsPath

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import (PointStamped, PoseStamped, TwistStamped,
                               Vector3Stamped)
from mavros_msgs.msg import ExtendedState, State
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Bool, Float32MultiArray, String

from trailer_link.geodesy import enu_offset

from .cju_route import local_to_map, route_map_info, rotation_for_heading
from .ros_msgs import msg_to_corridor, msg_to_trajectory, path_to_positions
from .world_model import WorldModel

try:  # Only the Gazebo-native profile needs this optional message type.
    from px4_msgs.msg import VehicleLocalPosition
except ImportError:  # pragma: no cover - hardware images may omit px4_msgs
    VehicleLocalPosition = None


_TRACKING_PHASES = {'MISSION', 'RETURN', 'CRUISE'}
_ARUCO_PHASES = {'SEARCH', 'ACQUIRE', 'DESCEND'}
_ACTIVE_PHASES = _TRACKING_PHASES | _ARUCO_PHASES | {'TAKEOFF', 'APPROACH'}
_PLAN_PHASES = {'MISSION_PLAN', 'RETURN_PLAN'}

SUMMARY_METRICS = (
    'path_length_m',
    'tracking_error_mean_m',
    'tracking_error_max_m',
    'tracking_error_rmse_m',
    'min_clearance_m',
    'astar_plan_time_ms',
    'astar_expanded_nodes_mean',
    'astar_expanded_nodes_max',
    'mpc_solve_time_ms',
    'sfc_generation_time_ms',
    'sfc_min_width_m',
    'sfc_avg_width_m',
    'sfc_corridor_count',
    'sfc_violation_count',
    'replan_count',
    'aruco_detection_rate_pct',
    'relative_xy_error_m',
    'landing_xy_error_m',
    'touchdown_relative_speed_m_s',
    'max_speed_m_s',
    'max_accel_m_s2',
    'accel_rms_m_s2',
)

TIMESERIES_FIELDS = (
    'timestamp_utc', 'elapsed_s', 'event', 'phase',
    'x_m', 'y_m', 'z_m',
    'vx_m_s', 'vy_m_s', 'vz_m_s', 'speed_m_s',
    'ax_m_s2', 'ay_m_s2', 'az_m_s2', 'accel_mag_m_s2',
    'path_length_m', 'tracking_error_m', 'min_clearance_m',
    'astar_plan_time_ms', 'astar_expanded_nodes', 'mpc_solve_time_ms',
    'sfc_generation_time_ms', 'sfc_min_width_m', 'sfc_avg_width_m',
    'sfc_corridor_count', 'sfc_violation_count', 'replan_count',
    'aruco_detected', 'aruco_detection_rate_pct',
    'relative_xy_error_m', 'landing_xy_error_m',
    'touchdown_relative_speed_m_s',
)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _stamp_seconds(message) -> float:
    try:
        stamp = message.header.stamp
        value = float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)
        return value if value > 0.0 and math.isfinite(value) else float('nan')
    except (AttributeError, TypeError, ValueError):
        return float('nan')


def _polyline_distance_xy(point_xy, path_xyz) -> float | None:
    """Exact point-to-polyline XY distance, independent of path sample rate."""
    point = np.asarray(point_xy, float)
    path = np.asarray(path_xyz, float)
    if (point.shape != (2,) or path.ndim != 2 or path.shape[1] < 2
            or len(path) < 2 or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(path[:, :2]))):
        return None
    start = path[:-1, :2]
    delta = path[1:, :2] - start
    length2 = np.einsum('ij,ij->i', delta, delta)
    valid = length2 > 1.0e-12
    if not np.any(valid):
        return float(np.linalg.norm(point - path[0, :2]))
    fraction = np.zeros(len(delta))
    fraction[valid] = np.clip(
        np.einsum('ij,ij->i', point - start[valid], delta[valid])
        / length2[valid], 0.0, 1.0)
    closest = start + fraction[:, None] * delta
    return float(np.linalg.norm(closest - point, axis=1).min())


def _horizontal_corridor_widths(boxes_min, boxes_max) -> np.ndarray:
    """Per-box horizontal width: the smaller of its X and Y spans."""
    low = np.asarray(boxes_min, float)
    high = np.asarray(boxes_max, float)
    if (low.ndim != 2 or low.shape[1] != 3 or high.shape != low.shape
            or not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high)) or np.any(high < low)):
        return np.empty(0, float)
    return np.min(high[:, :2] - low[:, :2], axis=1)


@dataclass
class ExperimentMetrics:
    """Small ROS-free accumulator used by the node and its regression test."""

    path_length_m: float = 0.0
    pose_samples: int = 0
    tracking_errors: list[float] = field(default_factory=list)
    clearances: list[float] = field(default_factory=list)
    astar_times_ms: list[float] = field(default_factory=list)
    astar_expanded_nodes: list[float] = field(default_factory=list)
    mpc_times_ms: list[float] = field(default_factory=list)
    sfc_times_ms: list[float] = field(default_factory=list)
    sfc_min_widths_m: list[float] = field(default_factory=list)
    sfc_width_sum_m: float = 0.0
    sfc_corridor_count: int = 0
    sfc_violation_count: int = 0
    sfc_evaluation_count: int = 0
    successful_plans: int = 0
    replan_count: int = 0
    aruco_true: int = 0
    aruco_total: int = 0
    relative_xy_errors: list[float] = field(default_factory=list)
    landing_xy_error_m: float | None = None
    touchdown_relative_speed_m_s: float | None = None
    max_speed_m_s: float | None = None
    max_accel_m_s2: float | None = None
    _accel_sq_sum: float = 0.0
    _accel_count: int = 0
    _last_position: np.ndarray | None = None

    def start_at(self, position=None) -> None:
        if position is not None:
            point = np.asarray(position, float)
            if point.shape == (3,) and np.all(np.isfinite(point)):
                self._last_position = point.copy()

    def add_pose(self, position) -> None:
        point = np.asarray(position, float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return
        if self._last_position is not None:
            self.path_length_m += float(np.linalg.norm(
                point - self._last_position))
        self._last_position = point.copy()
        self.pose_samples += 1

    def add_tracking_error(self, value) -> None:
        if _finite(value) and float(value) >= 0.0:
            self.tracking_errors.append(float(value))

    def add_clearance(self, value) -> None:
        if _finite(value) and float(value) >= 0.0:
            self.clearances.append(float(value))

    def add_astar(self, plan_time_ms, *, initial_for_leg: bool,
                  expanded_nodes=None) -> None:
        if not _finite(plan_time_ms) or float(plan_time_ms) < 0.0:
            return
        self.astar_times_ms.append(float(plan_time_ms))
        # A*'s search effort — the compute-load counterpart to the plan time,
        # and the number that climbs on a denser obstacle map.
        if _finite(expanded_nodes) and float(expanded_nodes) >= 0.0:
            self.astar_expanded_nodes.append(float(expanded_nodes))
        self.successful_plans += 1
        if not initial_for_leg:
            self.replan_count += 1

    def add_mpc(self, solve_time_ms) -> None:
        if _finite(solve_time_ms) and float(solve_time_ms) >= 0.0:
            self.mpc_times_ms.append(float(solve_time_ms))

    def add_sfc(self, generation_time_ms, min_width_m, avg_width_m,
                corridor_count) -> None:
        values = (generation_time_ms, min_width_m, avg_width_m,
                  corridor_count)
        if not all(_finite(value) for value in values):
            return
        generation = float(generation_time_ms)
        minimum = float(min_width_m)
        average = float(avg_width_m)
        count = int(round(float(corridor_count)))
        if (generation < 0.0 or minimum < 0.0 or average < 0.0
                or count <= 0):
            return
        self.sfc_times_ms.append(generation)
        self.sfc_min_widths_m.append(minimum)
        self.sfc_width_sum_m += average * count
        self.sfc_corridor_count += count

    def add_sfc_violation(self) -> None:
        self.sfc_violation_count += 1

    def add_sfc_evaluation(self) -> None:
        self.sfc_evaluation_count += 1

    def add_detection(self, detected: bool) -> None:
        self.aruco_total += 1
        self.aruco_true += int(bool(detected))

    def add_relative_xy(self, value) -> None:
        if _finite(value) and float(value) >= 0.0:
            self.relative_xy_errors.append(float(value))

    def add_velocity(self, velocity) -> None:
        """Track the peak ground speed from a local-ENU velocity sample."""
        vector = np.asarray(velocity, float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            return
        speed = float(np.linalg.norm(vector))
        if self.max_speed_m_s is None or speed > self.max_speed_m_s:
            self.max_speed_m_s = speed

    def add_acceleration(self, acceleration) -> None:
        """Track peak and RMS of the kinematic acceleration magnitude."""
        vector = np.asarray(acceleration, float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            return
        magnitude = float(np.linalg.norm(vector))
        if self.max_accel_m_s2 is None or magnitude > self.max_accel_m_s2:
            self.max_accel_m_s2 = magnitude
        self._accel_sq_sum += magnitude * magnitude
        self._accel_count += 1

    def accel_rms_m_s2(self) -> float | None:
        if not self._accel_count:
            return None
        return float(math.sqrt(self._accel_sq_sum / self._accel_count))

    def detection_rate(self) -> float | None:
        if not self.aruco_total:
            return None
        return 100.0 * self.aruco_true / self.aruco_total

    def summary(self) -> dict[str, float | int | None]:
        tracking = np.asarray(self.tracking_errors, float)
        return {
            'path_length_m': (self.path_length_m
                              if self.pose_samples else None),
            'tracking_error_mean_m': (float(tracking.mean())
                                      if len(tracking) else None),
            'tracking_error_max_m': (float(tracking.max())
                                     if len(tracking) else None),
            'tracking_error_rmse_m': (float(np.sqrt(np.mean(tracking ** 2)))
                                      if len(tracking) else None),
            'min_clearance_m': (min(self.clearances)
                                if self.clearances else None),
            'astar_plan_time_ms': _mean(self.astar_times_ms),
            'astar_expanded_nodes_mean': _mean(self.astar_expanded_nodes),
            'astar_expanded_nodes_max': (max(self.astar_expanded_nodes)
                                         if self.astar_expanded_nodes else None),
            'mpc_solve_time_ms': _mean(self.mpc_times_ms),
            'sfc_generation_time_ms': _mean(self.sfc_times_ms),
            'sfc_min_width_m': (min(self.sfc_min_widths_m)
                                if self.sfc_min_widths_m else None),
            'sfc_avg_width_m': (
                self.sfc_width_sum_m / self.sfc_corridor_count
                if self.sfc_corridor_count else None),
            'sfc_corridor_count': (self.sfc_corridor_count
                                   if self.sfc_times_ms else None),
            'sfc_violation_count': (self.sfc_violation_count
                                    if self.sfc_evaluation_count else None),
            'replan_count': (self.replan_count
                             if self.successful_plans else None),
            'aruco_detection_rate_pct': self.detection_rate(),
            'relative_xy_error_m': _mean(self.relative_xy_errors),
            'landing_xy_error_m': self.landing_xy_error_m,
            'touchdown_relative_speed_m_s':
                self.touchdown_relative_speed_m_s,
            'max_speed_m_s': self.max_speed_m_s,
            'max_accel_m_s2': self.max_accel_m_s2,
            'accel_rms_m_s2': self.accel_rms_m_s2(),
        }


class ClearanceCalculator:
    """Raw obstacle clearance in the coordinate frame declared by one map."""

    def __init__(self, map_yaml: str, *, pose_is_map_frame: bool):
        self.path = FsPath(map_yaml).expanduser().resolve(strict=True)
        self.pose_is_map_frame = bool(pose_is_map_frame)
        document = yaml.safe_load(self.path.read_text(encoding='utf-8'))
        route_map = (isinstance(document, dict)
                     and int(document.get('schema_version', 0)) == 1
                     and isinstance(document.get('mission'), dict))
        self.kind = 'route' if route_map else 'city'
        self.origin_local_xy = None
        if self.kind == 'route':
            info = route_map_info(str(self.path))
            self.origin_lat, self.origin_lon = info.origin_lat, info.origin_lon
            self.rotation = rotation_for_heading(info.heading_deg_enu)
            lows, highs = [], []
            for obstacle in document['mission'].get('obstacles', []):
                centre = np.asarray(obstacle['center_m'], float)
                size = np.asarray(obstacle['size_m'], float)
                if (centre.shape != (3,) or size.shape != (3,)
                        or not np.all(np.isfinite(centre))
                        or not np.all(np.isfinite(size))
                        or np.any(size <= 0.0)):
                    raise ValueError('route obstacle geometry is invalid')
                lows.append(centre[:2] - 0.5 * size[:2])
                highs.append(centre[:2] + 0.5 * size[:2])
            self.boxes_min_xy = np.asarray(lows, float).reshape(-1, 2)
            self.boxes_max_xy = np.asarray(highs, float).reshape(-1, 2)
            self.world = None
        else:
            self.world = WorldModel.from_city_yaml(
                self.path, inflation_xy_m=0.0, ground_clearance_m=-1.0e4,
                ceiling_m=1.0e4, overfly_allowed=False)

    def update_anchor(self, local_xy, latitude: float,
                      longitude: float) -> None:
        if self.kind != 'route' or self.pose_is_map_frame:
            return
        local = np.asarray(local_xy, float)
        if (local.shape != (2,) or not np.all(np.isfinite(local))
                or not _finite(latitude) or not _finite(longitude)):
            return
        east, north = enu_offset(
            self.origin_lat, self.origin_lon,
            float(latitude), float(longitude))
        self.origin_local_xy = local - np.array([east, north])

    def clearance(self, position) -> float | None:
        point = np.asarray(position, float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return None
        if self.kind == 'city':
            if not self.pose_is_map_frame:
                return None
            value = self.world.clearance(point)
            return value if math.isfinite(value) else None
        if not len(self.boxes_min_xy):
            return None
        if self.pose_is_map_frame:
            map_xy = point[:2]
        elif self.origin_local_xy is not None:
            map_xy = local_to_map(
                point[:2], self.origin_local_xy, self.rotation)
        else:
            return None
        gap = (np.maximum(self.boxes_min_xy - map_xy, 0.0)
               + np.maximum(map_xy - self.boxes_max_xy, 0.0))
        return float(np.linalg.norm(gap, axis=1).min())

    def map_position(self, position) -> np.ndarray | None:
        """Express a vehicle position in the corridor/map coordinate frame."""
        point = np.asarray(position, float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return None
        if self.pose_is_map_frame:
            return point.copy()
        if self.kind != 'route' or self.origin_local_xy is None:
            return None
        return np.r_[local_to_map(
            point[:2], self.origin_local_xy, self.rotation), point[2]]


class CsvSink:
    def __init__(self, output_dir: str):
        self.started_wall = datetime.now(timezone.utc)
        self.run_id = self.started_wall.strftime('%Y%m%dT%H%M%S_%fZ')
        self.output_dir = FsPath(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries_path = (
            self.output_dir / f'{self.run_id}_timeseries.csv')
        self.summary_path = self.output_dir / f'{self.run_id}_summary.csv'
        self._stream = self.timeseries_path.open(
            'w', encoding='utf-8', newline='')
        self._writer = csv.DictWriter(
            self._stream, fieldnames=TIMESERIES_FIELDS)
        self._writer.writeheader()
        self._last_flush = time.monotonic()
        self.closed = False

    def write(self, elapsed_s: float, event: str, phase: str,
              **values) -> None:
        if self.closed:
            return
        row = {name: '' for name in TIMESERIES_FIELDS}
        row.update({
            'timestamp_utc': datetime.now(timezone.utc).isoformat(
                timespec='milliseconds'),
            'elapsed_s': max(0.0, float(elapsed_s)),
            'event': str(event),
            'phase': str(phase),
        })
        row.update({key: value for key, value in values.items()
                    if key in row and value is not None and _finite(value)})
        self._writer.writerow(row)
        if time.monotonic() - self._last_flush >= 1.0:
            self._stream.flush()
            self._last_flush = time.monotonic()

    def close(self, summary: dict, *, ended_wall: datetime,
              metadata: dict) -> None:
        if self.closed:
            return
        self._stream.flush()
        self._stream.close()
        fields = (
            'run_id', 'started_at_utc', 'ended_at_utc', 'end_reason',
            *SUMMARY_METRICS,
            'successful_plan_count', 'pose_sample_count',
            'tracking_sample_count', 'sfc_evaluation_count',
            'aruco_sample_count',
            'landing_xy_error_source', 'touchdown_relative_speed_source',
        )
        row = {name: '' for name in fields}
        row.update({
            'run_id': self.run_id,
            'started_at_utc': self.started_wall.isoformat(
                timespec='milliseconds'),
            'ended_at_utc': ended_wall.isoformat(timespec='milliseconds'),
        })
        row.update({key: value for key, value in summary.items()
                    if key in row and value is not None})
        row.update({key: value for key, value in metadata.items()
                    if key in row and value is not None})
        with self.summary_path.open(
                'w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        self.closed = True


class ExperimentLoggerNode(Node):
    def __init__(self):
        super().__init__('experiment_logger')
        p = self.declare_parameter
        output_dir = str(p('output_dir', '~/px4_experiment_logs').value)
        self.pose_source = str(p('pose_source', 'mavros_pose').value)
        self.state_topic = str(
            p('state_topic', '/aruco_landing_node/state').value)
        self.detected_topic = str(
            p('aruco_detected_topic',
              '/perception/down/aruco_detected').value)
        self.marker_position_topic = str(
            p('marker_position_topic', '/marker/position').value)
        self.marker_velocity_topic = str(
            p('marker_velocity_topic', '/marker/velocity').value)
        self.pose_fix_sync_s = float(
            p('pose_fix_sync_tolerance_s', 0.20).value)
        self.marker_fresh_s = float(p('marker_fresh_s', 1.5).value)
        self.handoff_max_xy = float(
            p('landing_handoff_max_xy_m', 0.20).value)
        map_yaml = str(p('map_yaml', '').value)
        pose_is_map_frame = bool(p('pose_is_map_frame', False).value)
        px4_position_topic = str(
            p('px4_local_position_topic',
              '/fmu/out/vehicle_local_position_v1').value)

        valid_pose_sources = {
            'mavros_pose', 'path_plan_odometry', 'px4_local_position'}
        if self.pose_source not in valid_pose_sources:
            raise ValueError(
                'pose_source must be mavros_pose, path_plan_odometry, or '
                'px4_local_position')
        if self.pose_source == 'px4_local_position' \
                and VehicleLocalPosition is None:
            raise RuntimeError(
                'pose_source=px4_local_position requires px4_msgs')

        self.metrics = ExperimentMetrics()
        self.sink = CsvSink(output_dir)
        self.pose_is_map_frame = pose_is_map_frame
        self.phase = ''
        self.recording = False
        self.was_armed = False
        self.have_mavros_state = False
        self.finalized = False
        self._start_ros_s = None
        self._position = None
        self._position_t = float('nan')
        self._velocity = None
        self._velocity_t = float('-inf')
        # Kinematic acceleration, finite-differenced from the velocity stream
        # (gravity-free, matching the double-integrator MPC's accel input).
        self._acceleration = None
        self._prev_velocity = None
        self._prev_velocity_t = float('-inf')
        self._active_path = None
        self._active_corridor_min = None
        self._active_corridor_max = None
        self._sfc_outside = False
        self._sfc_seen_inside = False
        self._fix = None
        self._fix_t = float('nan')
        self._marker_position = None
        self._marker_t = float('-inf')
        self._marker_velocity = None
        self._marker_velocity_t = float('-inf')
        self._marker_valid = False
        self._marker_valid_seen = False
        self._detected = False
        self._latest_relative_xy = None
        self._latest_relative_t = float('-inf')
        self._landing_source = ''
        self._touchdown_source = ''
        self._airborne_seen = False
        self._plan_leg = None
        self._plans_in_leg = 0

        self.clearance = None
        if map_yaml:
            try:
                self.clearance = ClearanceCalculator(
                    map_yaml, pose_is_map_frame=pose_is_map_frame)
            except Exception as exc:
                self.get_logger().error(
                    f'clearance disabled; map could not be read: {exc}')

        sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(State, '/mavros/state',
                                 self._on_mavros_state, sensor)
        self.create_subscription(ExtendedState, '/mavros/extended_state',
                                 self._on_extended_state, sensor)
        self.create_subscription(String, self.state_topic,
                                 self._on_phase, 10)
        self.create_subscription(Bool, self.detected_topic,
                                 self._on_detected, sensor)
        self.create_subscription(PointStamped, self.marker_position_topic,
                                 self._on_marker_position, 10)
        self.create_subscription(Vector3Stamped, self.marker_velocity_topic,
                                 self._on_marker_velocity, 10)
        self.create_subscription(Bool, '/marker/valid',
                                 self._on_marker_valid, 10)
        self.create_subscription(Float32MultiArray, '/path_plan/astar_stats',
                                 self._on_astar_stats, latched)
        self.create_subscription(Float32MultiArray, '/path_plan/mpc_stats',
                                 self._on_mpc_stats, 10)
        self.create_subscription(Float32MultiArray, '/path_plan/sfc_stats',
                                 self._on_sfc_stats, latched)
        self.create_subscription(Float32MultiArray,
                                 '/path_plan/sfc_corridor',
                                 self._on_sfc_corridor, latched)
        self.create_subscription(Float32MultiArray, '/path_plan/trajectory',
                                 self._on_trajectory, latched)
        self.create_subscription(Float32MultiArray,
                                 '/path_plan/active_path_xy',
                                 self._on_active_path_xy, latched)
        self.create_subscription(Path, '/path_plan/trajectory_path',
                                 self._on_path, latched)
        # Existing Gazebo bspline_node keeps this private name.
        self.create_subscription(Path, '/bspline_optimizer/trajectory_path',
                                 self._on_path, latched)
        self.create_subscription(NavSatFix, '/mavros/global_position/global',
                                 self._on_fix, sensor)

        if self.pose_source == 'mavros_pose':
            self.create_subscription(
                PoseStamped, '/mavros/local_position/pose',
                self._on_pose, sensor)
            self.create_subscription(
                TwistStamped, '/mavros/local_position/velocity_local',
                self._on_velocity, sensor)
        elif self.pose_source == 'path_plan_odometry':
            self.create_subscription(Odometry, '/path_plan/odometry',
                                     self._on_odometry, sensor)
        elif self.pose_source == 'px4_local_position':
            self.create_subscription(VehicleLocalPosition, px4_position_topic,
                                     self._on_px4_position, sensor)

        self.get_logger().info(
            f'experiment logger ready: {self.sink.timeseries_path} + '
            f'{self.sink.summary_path} | pose={self.pose_source}')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _elapsed(self) -> float:
        now = self._now()
        if self._start_ros_s is None:
            self._start_ros_s = now
        return now - self._start_ros_s

    def _base_values(self) -> dict:
        rate = self.metrics.detection_rate()
        return {
            'path_length_m': (self.metrics.path_length_m
                              if self.metrics.pose_samples else None),
            'replan_count': (self.metrics.replan_count
                             if self.metrics.successful_plans else None),
            'sfc_violation_count': (
                self.metrics.sfc_violation_count
                if self.metrics.sfc_evaluation_count else None),
            'aruco_detection_rate_pct': rate,
        }

    def _write(self, event: str, **values) -> None:
        if self.finalized:
            return
        row = self._base_values()
        row.update(values)
        self.sink.write(self._elapsed(), event, self.phase, **row)

    def _start_recording(self, reason: str) -> None:
        if self.recording:
            return
        self.recording = True
        self.metrics.start_at(self._position)
        self._write('start', **({'x_m': self._position[0],
                                 'y_m': self._position[1],
                                 'z_m': self._position[2]}
                                if self._position is not None else {}))
        self.get_logger().info(f'experiment recording started ({reason})')

    def _on_mavros_state(self, message: State) -> None:
        self.have_mavros_state = True
        if message.armed:
            self.was_armed = True
            self._start_recording('armed')
        elif self.was_armed and self.recording:
            self._snapshot_touchdown('mavros_disarm')
            self.finalize('disarmed')

    def _on_extended_state(self, message: ExtendedState) -> None:
        in_air = message.landed_state == ExtendedState.LANDED_STATE_IN_AIR
        on_ground = (
            message.landed_state == ExtendedState.LANDED_STATE_ON_GROUND)
        if in_air:
            self._airborne_seen = True
        elif on_ground and self._airborne_seen and self.recording:
            self._snapshot_touchdown('mavros_on_ground')
            self._airborne_seen = False

    def _on_phase(self, message: String) -> None:
        previous = self.phase
        self.phase = str(message.data).strip().upper()
        if self.phase in _PLAN_PHASES and self.phase != previous:
            self._plan_leg = self.phase
            self._plans_in_leg = 0
        if (not self.have_mavros_state and self.phase in _ACTIVE_PHASES):
            self._start_recording(f'phase_{self.phase.lower()}')
        if previous == 'DESCEND' and self.phase in {'LAND', 'TOUCHDOWN'}:
            self._snapshot_landing_handoff(trusted=(self.phase == 'TOUCHDOWN'))
            if self.phase == 'TOUCHDOWN':
                self._snapshot_touchdown('phase_touchdown')
        if self.phase == 'DONE':
            # Gazebo's KF remains valid through touchdown; prefer that final
            # estimate only if the transition sample was unavailable. Hardware
            # normally keeps the earlier vision handover.
            if (self.metrics.landing_xy_error_m is None
                    and self._marker_valid_seen and self._marker_valid):
                self._snapshot_landing_handoff(trusted=True)
            self._snapshot_touchdown('mission_done')
            self.finalize('mission_done')

    def _on_pose(self, message: PoseStamped) -> None:
        p = message.pose.position
        self._handle_pose(
            np.array([p.x, p.y, p.z], float), _stamp_seconds(message))

    def _ingest_velocity(self, velocity: np.ndarray) -> None:
        """Store the velocity and finite-difference it into acceleration.

        Acceleration is the velocity derivative — the kinematic accel the
        double-integrator MPC treats as its control input, gravity-free — so a
        plain difference over the sample interval is the right estimate. Absurd
        gaps (a dropout, a first sample) are skipped rather than differentiated.
        """
        now = self._now()
        if (self._prev_velocity is not None
                and math.isfinite(self._prev_velocity_t)):
            dt = now - self._prev_velocity_t
            if 1.0e-3 <= dt <= 0.5:
                self._acceleration = (velocity - self._prev_velocity) / dt
                if self.recording:
                    self.metrics.add_acceleration(self._acceleration)
        if self.recording:
            self.metrics.add_velocity(velocity)
        self._velocity = velocity
        self._velocity_t = now
        self._prev_velocity = velocity.copy()
        self._prev_velocity_t = now

    def _on_velocity(self, message: TwistStamped) -> None:
        v = message.twist.linear
        self._ingest_velocity(np.array([v.x, v.y, v.z], float))

    def _on_odometry(self, message: Odometry) -> None:
        p = message.pose.pose.position
        v = message.twist.twist.linear
        self._ingest_velocity(np.array([v.x, v.y, v.z], float))
        self._handle_pose(
            np.array([p.x, p.y, p.z], float), _stamp_seconds(message))

    def _on_px4_position(self, message) -> None:
        # PX4 NED -> ROS ENU, exactly as landing_mpc's mission manager.
        self._ingest_velocity(
            np.array([message.vy, message.vx, -message.vz], float))
        self._handle_pose(
            np.array([message.y, message.x, -message.z], float), self._now())

    def _handle_pose(self, position: np.ndarray, source_t: float) -> None:
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return
        self._position = position
        self._position_t = source_t if math.isfinite(source_t) else self._now()
        self._update_clearance_anchor()
        if not self.recording:
            return
        self.metrics.add_pose(position)
        tracking = None
        if (self._active_path is not None
                and self.phase in _TRACKING_PHASES):
            tracking = _polyline_distance_xy(position[:2], self._active_path)
            self.metrics.add_tracking_error(tracking)
        clearance = (self.clearance.clearance(position)
                     if self.clearance is not None else None)
        self.metrics.add_clearance(clearance)
        if (self._active_corridor_min is not None
                and self.phase in _TRACKING_PHASES):
            corridor_position = (
                self.clearance.map_position(position)
                if self.clearance is not None else
                position if self.pose_is_map_frame else None)
            if corridor_position is not None:
                inside = bool(np.any(np.all(
                    (corridor_position[:2]
                     >= self._active_corridor_min[:, :2] - 1.0e-6)
                    & (corridor_position[:2]
                       <= self._active_corridor_max[:, :2] + 1.0e-6),
                    axis=1)))
                if inside:
                    self._sfc_seen_inside = True
                if self._sfc_seen_inside:
                    self.metrics.add_sfc_evaluation()
                    if not inside and not self._sfc_outside:
                        self.metrics.add_sfc_violation()
                self._sfc_outside = not inside
        vel = self._velocity
        acc = self._acceleration
        kinematics = {}
        if vel is not None and np.all(np.isfinite(vel)):
            kinematics.update(
                vx_m_s=float(vel[0]), vy_m_s=float(vel[1]),
                vz_m_s=float(vel[2]), speed_m_s=float(np.linalg.norm(vel)))
        if acc is not None and np.all(np.isfinite(acc)):
            kinematics.update(
                ax_m_s2=float(acc[0]), ay_m_s2=float(acc[1]),
                az_m_s2=float(acc[2]),
                accel_mag_m_s2=float(np.linalg.norm(acc)))
        self._write('pose', x_m=position[0], y_m=position[1], z_m=position[2],
                    tracking_error_m=tracking, min_clearance_m=clearance,
                    **kinematics)

    def _on_fix(self, message: NavSatFix) -> None:
        if message.status.status < NavSatStatus.STATUS_FIX:
            return
        if not (_finite(message.latitude) and _finite(message.longitude)):
            return
        self._fix = (float(message.latitude), float(message.longitude))
        stamp = _stamp_seconds(message)
        self._fix_t = stamp if math.isfinite(stamp) else self._now()
        self._update_clearance_anchor()

    def _update_clearance_anchor(self) -> None:
        if (self.clearance is None or self._position is None
                or self._fix is None or not math.isfinite(self._position_t)
                or not math.isfinite(self._fix_t)
                or abs(self._position_t - self._fix_t) > self.pose_fix_sync_s):
            return
        self.clearance.update_anchor(
            self._position[:2], self._fix[0], self._fix[1])

    def _on_trajectory(self, message: Float32MultiArray) -> None:
        try:
            _time, positions, _velocity = msg_to_trajectory(message)
            if len(positions) >= 2 and np.all(np.isfinite(positions)):
                self._active_path = np.asarray(positions, float)
                self._write('path')
        except (TypeError, ValueError):
            return

    def _on_path(self, message: Path) -> None:
        positions = path_to_positions(message)
        if len(positions) >= 2 and np.all(np.isfinite(positions)):
            self._active_path = positions
            self._write('path')

    def _on_active_path_xy(self, message: Float32MultiArray) -> None:
        try:
            positions = np.asarray(message.data, float).reshape(-1, 2)
        except (TypeError, ValueError):
            return
        if len(positions) >= 2 and np.all(np.isfinite(positions)):
            self._active_path = positions
            self._write('path')

    def _on_astar_stats(self, message: Float32MultiArray) -> None:
        if not message.data:
            return
        plan_ms = 1000.0 * float(message.data[0])
        # astar_stats payload is [plan_time_s, expanded_nodes, path_point_count].
        expanded = (float(message.data[1]) if len(message.data) > 1 else None)
        if self._plan_leg is not None:
            initial = self._plans_in_leg == 0
            self._plans_in_leg += 1
        else:
            initial = self.metrics.successful_plans == 0
        self.metrics.add_astar(
            plan_ms, initial_for_leg=initial, expanded_nodes=expanded)
        self._write('astar', astar_plan_time_ms=plan_ms,
                    astar_expanded_nodes=expanded)

    def _on_mpc_stats(self, message: Float32MultiArray) -> None:
        if (not self.recording or not message.data
                or self.phase not in _TRACKING_PHASES):
            return
        solve_ms = float(message.data[0])
        self.metrics.add_mpc(solve_ms)
        self._write('mpc', mpc_solve_time_ms=solve_ms)

    def _on_sfc_stats(self, message: Float32MultiArray) -> None:
        if len(message.data) < 4:
            return
        generation, minimum, average, count = map(float, message.data[:4])
        before = len(self.metrics.sfc_times_ms)
        self.metrics.add_sfc(generation, minimum, average, count)
        if len(self.metrics.sfc_times_ms) == before:
            return
        self._write(
            'sfc', sfc_generation_time_ms=generation,
            sfc_min_width_m=minimum, sfc_avg_width_m=average,
            sfc_corridor_count=int(round(count)))

    def _on_sfc_corridor(self, message: Float32MultiArray) -> None:
        try:
            low, high = msg_to_corridor(message)
        except (TypeError, ValueError):
            return
        widths = _horizontal_corridor_widths(low, high)
        if not len(widths):
            return
        self._active_corridor_min = low
        self._active_corridor_max = high
        self._sfc_outside = False
        # A hardware replan is safely spliced from the current pose to the
        # original plan. Count only excursions after the vehicle has first
        # entered this optimizer corridor, not that certified join chord.
        self._sfc_seen_inside = False

    def _on_detected(self, message: Bool) -> None:
        self._detected = bool(message.data)
        if not self.recording or self.phase not in _ARUCO_PHASES:
            return
        self.metrics.add_detection(self._detected)
        self._write('aruco', aruco_detected=int(self._detected))

    def _on_marker_valid(self, message: Bool) -> None:
        self._marker_valid_seen = True
        self._marker_valid = bool(message.data)

    def _on_marker_position(self, message: PointStamped) -> None:
        p = message.point
        marker = np.array([p.x, p.y, p.z], float)
        if not np.all(np.isfinite(marker)):
            return
        self._marker_position = marker
        self._marker_t = self._now()
        if (not self.recording or self.phase not in _ARUCO_PHASES
                or self._position is None
                or (self._marker_valid_seen and not self._marker_valid)):
            return
        relative = float(np.linalg.norm(marker[:2] - self._position[:2]))
        self._latest_relative_xy = relative
        self._latest_relative_t = self._marker_t
        self.metrics.add_relative_xy(relative)
        self._write('marker', relative_xy_error_m=relative)

    def _on_marker_velocity(self, message: Vector3Stamped) -> None:
        v = message.vector
        velocity = np.array([v.x, v.y, v.z], float)
        if np.all(np.isfinite(velocity)):
            self._marker_velocity = velocity
            self._marker_velocity_t = self._now()

    def _snapshot_landing_handoff(self, *, trusted: bool) -> None:
        if (self.metrics.landing_xy_error_m is not None
                or self._latest_relative_xy is None
                or self._now() - self._latest_relative_t > self.marker_fresh_s
                or (not trusted
                    and self._latest_relative_xy > self.handoff_max_xy)):
            return
        self.metrics.landing_xy_error_m = self._latest_relative_xy
        self._landing_source = ('onboard_touchdown_estimate' if trusted
                                else 'vision_handoff')
        self._write('landing_handoff',
                    landing_xy_error_m=self.metrics.landing_xy_error_m)

    def _snapshot_touchdown(self, source: str) -> None:
        now = self._now()
        if (self.metrics.touchdown_relative_speed_m_s is not None
                or self._velocity is None or self._marker_velocity is None
                or now - self._velocity_t > self.marker_fresh_s
                or now - self._marker_velocity_t > self.marker_fresh_s
                or (self._marker_valid_seen and not self._marker_valid)):
            return
        relative = (np.asarray(self._velocity)
                    - np.asarray(self._marker_velocity))
        if not np.all(np.isfinite(relative)):
            return
        self.metrics.touchdown_relative_speed_m_s = float(
            np.linalg.norm(relative))
        self._touchdown_source = 'onboard_estimate_' + source
        self._write('touchdown', touchdown_relative_speed_m_s=(
            self.metrics.touchdown_relative_speed_m_s))

    def finalize(self, reason: str) -> None:
        if self.finalized:
            return
        self._write(
            'end', landing_xy_error_m=self.metrics.landing_xy_error_m,
            touchdown_relative_speed_m_s=(
                self.metrics.touchdown_relative_speed_m_s))
        metadata = {
            'end_reason': reason,
            'successful_plan_count': self.metrics.successful_plans,
            'pose_sample_count': self.metrics.pose_samples,
            'tracking_sample_count': len(self.metrics.tracking_errors),
            'sfc_evaluation_count': self.metrics.sfc_evaluation_count,
            'aruco_sample_count': self.metrics.aruco_total,
            'landing_xy_error_source': self._landing_source,
            'touchdown_relative_speed_source': self._touchdown_source,
        }
        self.sink.close(
            self.metrics.summary(), ended_wall=datetime.now(timezone.utc),
            metadata=metadata)
        self.finalized = True
        if rclpy.ok():
            self.get_logger().info(
                f'experiment CSV saved: {self.sink.timeseries_path} and '
                f'{self.sink.summary_path} ({reason})')


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentLoggerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finalize('shutdown')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
