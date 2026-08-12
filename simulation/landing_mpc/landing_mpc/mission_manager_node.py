"""mission_manager_node — ONE job: sequence the landing mission and decide
WHICH target source the controller should believe.

The reason this node exists: even the 1.3 m long-range markers are unresolved
at long range (at 90 m they are ~1 px per ArUco cell), so vision cannot fly
the approach.  The approach is flown to the target's REPORTED coordinates
(`/marker/cue`), and
vision (`/marker/position`, from the KF) takes over only once the vehicle is
close enough for the camera to actually see the marker.  Measured detection
rate: 0.3% while chasing at 50-90 m, 60-80% hovering overhead.

This node is the single Offboard setpoint authority — never run it alongside
another setpoint publisher.

Subscribes
    /mission/command     String          takeoff → mission → land
    /marker/cue          PointStamped    long-range target position (ENU)
    /marker/cue_velocity Vector3Stamped  long-range target velocity
    /marker/position     PointStamped    vision/KF target position (ENU)
    /marker/valid        Bool            vision usable (KF not over-coasting)
    /fmu/out/vehicle_local_position_v1
Publishes
    /fmu/in/goto_setpoint, /fmu/in/trajectory_setpoint,
    /fmu/in/offboard_control_mode,
    /fmu/in/vehicle_command
    /mission/state       String          current phase (observability)

Phases

  Phase 0
    PRECHECK  validate PX4 feedback, cue, planner and Offboard readiness
  Phase 1
    TAKEOFF   PX4 NAV_TAKEOFF to takeoff_alt
    READY     hold after takeoff until the explicit mission command
  Phase 2
    MISSION_PLAN plan A* and a geometry-only B-spline without blocking Offboard
    MISSION   follow that spatial path with PX4 Goto control to the map goal
    HOVER     hold over the map goal and wait for land
  Phase 3
    RETURN_PLAN plan A* and a geometry-only B-spline around map obstacles
    RETURN    refresh it as the trailer moves, until a direct live segment is safe
    PRECLAND  publish the corrected landing target and hand all flight control,
              touchdown detection and auto-disarm to PX4 precision landing
    ABORT     cue lost -> hold, then re-enter through trailer A*.

THE CUE FOLLOWS, THE MARKER CENTRES. RETURN is only an obstacle bypass toward
a recent cue snapshot and replans when that snapshot moves. As soon as the live cue has a clear YAML segment the
node stops publishing Offboard setpoints, supplies only LandingTargetPose, and
requests PX4 PRECLAND. PX4 then owns speed, acceleration, jerk, attitude,
descent, contact detection and disarm. Vision remains a slowly-filtered
horizontal correction to the continuous cue; it never becomes a flight
controller.
"""

from __future__ import annotations

import math
import multiprocessing
import re
import signal
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped, Pose, PoseArray, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool, String

from px4_msgs.msg import (GotoSetpoint, LandingTargetPose, OffboardControlMode,
                          TrajectorySetpoint, VehicleCommand, VehicleLandDetected,
                          VehicleLocalPosition, VehicleStatus)

from .frame import LOCAL_ENU_FRAME_ID, enu_to_ned
from .parameter_utils import (
    require_finite,
    require_nonempty,
    require_positive,
)


def _planner_worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _px4_qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


def _planned_path_qos():
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=1)


def _plan_global_path(map_yaml, start_local_enu=None, goal_local_enu=None):
    """Build one exact-safe A* -> geometry-only B-spline path."""
    from path_plan.astar import AStarPlanner3D
    from path_plan.bspline_optimizer import BsplineOptimizer
    from path_plan.uniform_bspline import UniformBspline
    from path_plan.world_model import WorldModel

    document = yaml.safe_load(Path(map_yaml).read_text(encoding='utf-8'))
    mission = document['mission']
    spawn_pose = document['spawn']['gazebo_spawn_pose_enu']
    frame_name = mission['coordinate_frame']
    if document['terrain']['coordinate_frame'] != frame_name:
        raise ValueError('mission and terrain must use the same frame')
    frame = document['frames'][frame_name]

    heading = math.radians(float(frame['heading_deg_enu']))
    rotation = np.array([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)],
    ])
    frame_origin = np.asarray(frame['origin_enu_m'][:2], float)
    spawn = np.asarray([spawn_pose['x'], spawn_pose['y']], float)
    altitude = float(mission['cruise_altitude_m'])
    start_local = (np.zeros(3) if start_local_enu is None
                   else np.asarray(start_local_enu, float).copy())
    if start_local.shape != (3,) or not np.all(np.isfinite(start_local)):
        raise ValueError('start_local_enu must contain three finite values')
    start = np.r_[((spawn + start_local[:2]) - frame_origin) @ rotation,
                  altitude]
    if goal_local_enu is None:
        goal_xy = np.asarray(mission['goal_m'], float)
    else:
        goal_local = np.asarray(goal_local_enu, float)
        if goal_local.shape != (3,) or not np.all(np.isfinite(goal_local)):
            raise ValueError('goal_local_enu must contain three finite values')
        goal_xy = ((spawn + goal_local[:2]) - frame_origin) @ rotation
    goal = np.r_[goal_xy, altitude]
    clearance = float(mission['obstacle_clearance_m'])
    spline_margin = float(mission.get('bspline_clearance_margin_m', 0.5))
    control_spacing = float(mission.get('bspline_control_spacing_m', 2.0))
    sample_spacing = float(mission.get('bspline_sample_spacing_m', 0.1))
    values = (clearance, control_spacing, sample_spacing)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError('CJU planner and spatial follower values must be positive')
    if not math.isfinite(spline_margin) or spline_margin < 0.0:
        raise ValueError('bspline_clearance_margin_m must be non-negative')
    if sample_spacing > 0.25:
        raise ValueError('bspline_sample_spacing_m must be <= 0.25')

    obstacle_lows, obstacle_highs = [], []
    for obstacle in mission['obstacles']:
        centre = np.asarray(obstacle['center_m'], float)
        half_size = 0.5 * np.asarray(obstacle['size_m'], float)
        low, high = centre - half_size, centre + half_size
        # Deliberately forbid overflight: this mission verifies lateral
        # avoidance of the configured ten-metre barriers.
        low[2], high[2] = -1.0e4, 1.0e4
        obstacle_lows.append(low)
        obstacle_highs.append(high)

    terrain_size = np.asarray(document['terrain']['size_m'], float)
    if (terrain_size.shape != (2,) or not np.all(np.isfinite(terrain_size))
            or np.any(terrain_size <= 0.0)):
        raise ValueError('terrain.size_m must contain two positive dimensions')
    terrain_center = np.asarray(document['terrain']['center_m'], float)
    if terrain_center.shape != (2,) or not np.all(np.isfinite(terrain_center)):
        raise ValueError('terrain.center_m must contain two finite values')
    half_terrain = 0.5 * terrain_size

    def make_world(clearance_xy_m, z_half_width):
        return WorldModel.from_boxes(
            obstacle_lows, obstacle_highs,
            [*(terrain_center - half_terrain), altitude - z_half_width],
            [*(terrain_center + half_terrain), altitude + z_half_width],
            xy_clearance_m=clearance_xy_m)

    # A* and the optimizer receive the same planning reserve used by final
    # acceptance. The optimizer needs non-zero z thickness for its 3-D SFC
    # seeds; the final spline is projected onto the cruise-altitude plane.
    planning_clearance = clearance + spline_margin
    planner_world = make_world(planning_clearance, 0.0)
    spline_world = make_world(planning_clearance, 0.5)
    if not bool(planner_world.is_free(start)[0]):
        raise RuntimeError('CJU A* exact start is blocked or out of bounds')
    if not bool(planner_world.is_free(goal)[0]):
        raise RuntimeError('CJU A* exact goal is blocked or out of bounds')
    planner = AStarPlanner3D(
        planner_world,
        resolution_m=float(mission['planner_resolution_m']),
        clearance_pref_m=planning_clearance,
        altitude_pref_m=altitude,
    )
    result = planner.plan(start, goal)
    if not result.success:
        raise RuntimeError(f'CJU A* failed: {result.message}')
    # A* searches a one-metre grid, while a moving cue is generally between
    # cells. Preserve the exact endpoints and validate both grid connectors;
    # otherwise the final handoff can be up to sqrt(0.5^2 + 0.5^2) m wrong.
    points = [start]
    for point in result.waypoints_m:
        if not np.allclose(points[-1], point, atol=1.0e-9, rtol=0.0):
            points.append(np.asarray(point, float))
    if np.allclose(points[-1], goal, atol=1.0e-9, rtol=0.0):
        points[-1] = goal
    else:
        points.append(goal)
    waypoints = np.asarray(points, float)
    if not all(planner_world.segment_is_free(a, b, step_m=0.1)
               for a, b in zip(waypoints[:-1], waypoints[1:])):
        raise RuntimeError('CJU A* returned a colliding shortcut')

    optimized = BsplineOptimizer(
        spline_world,
        cruise_speed_m_s=None,
        ctrl_spacing_m=control_spacing,
    ).optimize(waypoints)
    if not optimized.accepted:
        raise RuntimeError(
            'CJU B-spline optimization rejected: '
            f'solver={optimized.solver_success} '
            f'status={optimized.solver_status}, '
            f'finite={optimized.solution_finite}, '
            f'collision_free={optimized.collision_free}: '
            f'{optimized.solver_message}')
    control_points = optimized.spline.q.copy()
    control_points[:, 2] = altitude
    spline = UniformBspline(control_points, optimized.spline.ts)

    # Sample by DISTANCE, never by time.  The dense first pass only measures
    # curve arc length; it does not define a flight-time profile.
    guide_length = float(np.linalg.norm(
        np.diff(waypoints, axis=0), axis=1).sum())
    dense_count = max(200, int(math.ceil(guide_length / sample_spacing)) * 4)
    _, dense_positions, _, _ = spline.sample(dense_count)
    dense_positions[:, 2] = altitude
    dense_arc = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(dense_positions, axis=0), axis=1))]
    keep = np.r_[True, np.diff(dense_arc) > 1.0e-9]
    dense_arc = dense_arc[keep]
    dense_positions = dense_positions[keep]
    if len(dense_arc) < 2 or dense_arc[-1] <= 0.0:
        raise RuntimeError('CJU B-spline has no spatial extent')
    sample_arc = np.r_[np.arange(0.0, dense_arc[-1], sample_spacing),
                       dense_arc[-1]]
    positions = np.column_stack([
        np.interp(sample_arc, dense_arc, dense_positions[:, axis])
        for axis in range(3)])
    positions[0], positions[-1] = start, goal
    positions[:, 2] = altitude
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(positions, axis=0), axis=1))]
    if not (np.all(np.isfinite(positions)) and np.all(np.diff(arc) > 0.0)
            and np.allclose(positions[0], start, atol=1.0e-6)
            and np.allclose(positions[-1], goal, atol=1.0e-6)):
        raise RuntimeError('CJU B-spline spatial contract failed')
    if not all(spline_world.segment_is_free(a, b)
               for a, b in zip(positions[:-1], positions[1:])):
        raise RuntimeError(
            'CJU B-spline failed exact planning-clearance validation')
    world_xy = positions[:, :2] @ rotation.T + frame_origin
    local_positions = np.column_stack((world_xy - spawn, positions[:, 2]))
    return arc, local_positions, result.expanded


def _path_position(arc_m, path, distance_m):
    distance = float(np.clip(distance_m, arc_m[0], arc_m[-1]))
    return np.array([
        np.interp(distance, arc_m, path[:, axis]) for axis in range(3)])


def _splice_path_from_current(segment_is_free, arc_m, path, current,
                              lookahead_m):
    """Join a completed rolling path from the vehicle's current position."""
    arc = np.asarray(arc_m, float)
    points = np.asarray(path, float)
    current = np.asarray(current, float)
    projection_s, _, _ = _spatial_path_target(
        arc, points, current, 0.0, float(arc[-1]), 1.0)
    # Half a normal carrot avoids both a near-perpendicular projection join
    # and a long corner-cut. One failed connector keeps the prior safe route.
    join_s = min(projection_s + 0.5 * lookahead_m, float(arc[-1]))
    join = _path_position(arc, points, join_s)
    if not segment_is_free(current, join):
        return None
    tail = points[arc > join_s + 1.0e-9]
    joined = np.vstack((current, join, tail))
    keep = np.r_[True, np.linalg.norm(
        np.diff(joined, axis=0), axis=1) > 1.0e-9]
    joined = joined[keep]
    if len(joined) < 2:
        return None
    joined_arc = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(joined, axis=0), axis=1))]
    if not (np.all(np.isfinite(joined))
            and np.all(np.diff(joined_arc) > 0.0)):
        return None
    return joined_arc, joined


def _retarget_path_tail(segment_is_free, arc_m, path, progress_m, goal,
                        lookahead_m, cross_track_limit_m, sample_spacing_m):
    """Replace only the unflown tail with one exact-safe live-goal segment."""
    arc = np.asarray(arc_m, float)
    points = np.asarray(path, float)
    target = np.asarray(goal, float)
    if (points.ndim != 2 or len(points) != len(arc) or len(points) < 2
            or target.shape != (points.shape[1],)
            or not np.all(np.isfinite(np.column_stack((arc, points))))
            or not np.all(np.isfinite(target))
            or not math.isfinite(sample_spacing_m)
            or sample_spacing_m <= 0.0):
        return None

    progress = float(np.clip(progress_m, 0.0, arc[-1]))
    # Preserve every segment the follower can currently project onto.  Scanning
    # from the old endpoint backwards then changes the smallest possible tail.
    projection_window = max(2.0 * lookahead_m, 2.0 * cross_track_limit_m)
    first_join = min(
        len(points) - 1,
        int(np.searchsorted(
            arc, min(arc[-1], progress + projection_window), side='right')))
    for index in range(len(points) - 1, first_join - 1, -1):
        join = points[index]
        if not segment_is_free(join, target):
            continue
        distance = float(np.linalg.norm(target - join))
        count = max(1, int(math.ceil(distance / sample_spacing_m)))
        connector = join + np.linspace(0.0, 1.0, count + 1)[:, None] * (
            target - join)
        candidate = np.vstack((points[:index + 1], connector[1:]))
        keep = np.r_[True, np.linalg.norm(
            np.diff(candidate, axis=0), axis=1) > 1.0e-9]
        candidate = candidate[keep]
        if len(candidate) < 2:
            continue
        candidate_arc = np.r_[0.0, np.cumsum(np.linalg.norm(
            np.diff(candidate, axis=0), axis=1))]
        if (np.all(np.isfinite(candidate_arc))
                and np.all(np.diff(candidate_arc) > 0.0)):
            return candidate_arc, candidate
    return None


def _spatial_path_target(arc_m, path, position, progress_m, lookahead_m,
                         cross_track_limit_m):
    """Advance only from measured spatial progress and return one lookahead."""
    arc = np.asarray(arc_m, float)
    points = np.asarray(path, float)
    current = np.asarray(position, float)
    progress = float(np.clip(progress_m, 0.0, arc[-1]))
    if progress >= arc[-1]:
        cross_track = float(np.linalg.norm(current - points[-1]))
        return progress, points[-1].copy(), cross_track
    window = max(2.0 * lookahead_m, 2.0 * cross_track_limit_m)
    first = max(0, int(np.searchsorted(arc, progress, side='right')) - 1)
    last = min(len(arc) - 1, int(np.searchsorted(
        arc, min(arc[-1], progress + window), side='right')))
    best_distance = math.inf
    candidate = progress
    for index in range(first, last):
        a, b = points[index], points[index + 1]
        delta = b - a
        length2 = float(delta @ delta)
        if length2 <= 0.0:
            continue
        fraction = float(np.clip((current - a) @ delta / length2, 0.0, 1.0))
        projection = a + fraction * delta
        # The global route is flown at a fixed altitude, so cross-track is a
        # horizontal path error.  Vertical tracking error is controlled
        # independently and must not flip the spatial follower's mode.
        distance = float(np.linalg.norm((current - projection)[:2]))
        if distance < best_distance:
            best_distance = distance
            candidate = max(
                progress,
                float(arc[index] + fraction * (arc[index + 1] - arc[index])))
    cross_track = best_distance
    # Keep the carrot long on-track, then shorten it continuously as the
    # vehicle approaches the configured cross-track limit. At the limit the
    # target is the route projection, so PX4 rejoins instead of cutting the
    # obstacle reserve or receiving a discontinuous six-metre reversal.
    progress = candidate
    tracking_fraction = float(np.clip(
        1.0 - cross_track / cross_track_limit_m, 0.0, 1.0))
    target_s = min(
        progress + lookahead_m * tracking_fraction, float(arc[-1]))
    return progress, _path_position(arc, points, target_s), cross_track


def _safe_spatial_path_target(map_yaml, arc_m, path, position, progress_m,
                              lookahead_m, cross_track_limit_m):
    """Return the furthest exact-safe carrot on an accepted spatial path."""
    progress, target, _ = _spatial_path_target(
        arc_m, path, position, progress_m, lookahead_m,
        cross_track_limit_m)
    if _mission_segment_is_free(map_yaml, position, target):
        return progress, target
    for offset in np.linspace(lookahead_m, 0.0, 21)[1:]:
        candidate = _path_position(
            arc_m, path, min(progress + offset, arc_m[-1]))
        if _mission_segment_is_free(map_yaml, position, candidate):
            return progress, candidate
    return progress, None


@lru_cache(maxsize=8)
def _mission_collision_contract(map_yaml, planning=False):
    """Load the immutable per-run map snapshot once for 50 Hz checks."""
    from path_plan.world_model import WorldModel

    document = yaml.safe_load(Path(map_yaml).read_text(encoding='utf-8'))
    mission = document['mission']
    frame = document['frames'][mission['coordinate_frame']]
    spawn_pose = document['spawn']['gazebo_spawn_pose_enu']
    heading = math.radians(float(frame['heading_deg_enu']))
    rotation = np.array([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)],
    ])
    spawn = np.asarray([spawn_pose['x'], spawn_pose['y']], float)
    origin = np.asarray(frame['origin_enu_m'][:2], float)
    terrain_center = np.asarray(document['terrain']['center_m'], float)
    terrain_half = 0.5 * np.asarray(document['terrain']['size_m'], float)
    clearance = float(mission['obstacle_clearance_m'])
    if planning:
        clearance += float(mission.get('bspline_clearance_margin_m', 0.5))
    lows, highs = [], []
    for obstacle in mission['obstacles']:
        center = np.asarray(obstacle['center_m'][:2], float)
        half = 0.5 * np.asarray(obstacle['size_m'][:2], float)
        lows.append([*(center - half), -1.0e4])
        highs.append([*(center + half), 1.0e4])
    altitude = float(mission['cruise_altitude_m'])
    world = WorldModel.from_boxes(
        lows,
        highs,
        [*(terrain_center - terrain_half), altitude],
        [*(terrain_center + terrain_half), altitude],
        xy_clearance_m=clearance,
    )
    return rotation, spawn, origin, altitude, world


def _mission_segment_is_free(map_yaml, start_local_enu, goal_local_enu):
    """Check one cruise-altitude local segment against the same YAML AABBs."""
    start = np.asarray(start_local_enu, float)[:2]
    goal = np.asarray(goal_local_enu, float)[:2]
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(goal))):
        return False
    rotation, spawn, origin, altitude, world = (
        _mission_collision_contract(str(map_yaml)))
    map_xy = (np.vstack((start, goal)) + spawn - origin) @ rotation
    return world.segment_is_free(
        [*map_xy[0], altitude], [*map_xy[1], altitude])


def _mission_planning_segment_is_free(
        map_yaml, start_local_enu, goal_local_enu):
    """Check one local segment against clearance plus the planning reserve."""
    start = np.asarray(start_local_enu, float)[:2]
    goal = np.asarray(goal_local_enu, float)[:2]
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(goal))):
        return False
    rotation, spawn, origin, altitude, world = (
        _mission_collision_contract(str(map_yaml), True))
    map_xy = (np.vstack((start, goal)) + spawn - origin) @ rotation
    return world.segment_is_free(
        [*map_xy[0], altitude], [*map_xy[1], altitude])


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')
        p = self.declare_parameter
        self.control_rate_hz = require_positive(
            'control_rate_hz', p('control_rate_hz', 50.0).value)
        self.dt = 1.0 / self.control_rate_hz
        self.takeoff_alt = require_finite(
            'takeoff_alt', p('takeoff_alt', 6.0).value)
        # This is a route-completion gate, not a commanded speed. PX4 owns the
        # actual profile through Goto/Position Control.
        self.settle_v_tol = require_positive(
            'settle_vel_tol_m_s',
            p('settle_vel_tol_m_s', 0.5).value)
        # ArUco remains a measurement only. It slowly corrects the continuous
        # trailer cue before that target is handed to PX4 PRECLAND.
        self.align_deg = require_positive(
            'vision_align_depression_deg',
            p('vision_align_depression_deg', 60.0).value)
        if self.align_deg >= 90.0:
            raise ValueError(
                'vision_align_depression_deg must be < 90, '
                f'got {self.align_deg}')
        self.bias_tau = require_positive(
            'bias_tau_s', p('bias_tau_s', 1.5).value)
        self.bias_rate = require_positive(
            'bias_rate_max_m_s',
            p('bias_rate_max_m_s', 0.3).value)
        self.bias_max = require_positive(
            'bias_max_m', p('bias_max_m', 5.0).value)
        self.vis_fresh = require_positive(
            'vision_fresh_s', p('vision_fresh_s', 0.5).value)
        self.cue_timeout_s = require_positive(
            'cue_timeout_s', p('cue_timeout_s', 2.0).value)
        self.auto_start = bool(p('auto_start', True).value)
        self.mission_map_yaml = str(p('mission_map_yaml', '').value).strip()
        self.precheck_timeout = require_positive(
            'precheck_timeout_s', p('precheck_timeout_s', 1.0).value)
        self.precheck_warmup = require_positive(
            'precheck_warmup_s', p('precheck_warmup_s', 1.5).value)
        self.mission_tolerance = require_positive(
            'mission_waypoint_tolerance_m',
            p('mission_waypoint_tolerance_m', 0.7).value)
        self.mission_command_topic = require_nonempty(
            'mission_command_topic',
            p('mission_command_topic', '/mission/command').value)

        self.get_logger().info(
            f'control timing: {self.control_rate_hz:g} Hz; PX4 owns all '
            'takeoff, route and landing dynamics')

        self._mission_arc_m = None
        self._mission_path = None
        self._mission_progress_m = 0.0
        self._mission_lookahead = 6.0
        self._mission_cross_track = 0.25
        self._mission_sample_spacing = 0.1
        self._precland_handoff = 6.0
        self._return_replan_min_period = 2.0
        self._precland_target_timeout = 0.5
        self._planner_pool = None
        self._plan_future = None
        if self.mission_map_yaml:
            mission_document = yaml.safe_load(
                Path(self.mission_map_yaml).read_text(encoding='utf-8'))
            mission_config = mission_document['mission']
            self._mission_lookahead = require_positive(
                'mission.mpc_path_lookahead_m',
                mission_config.get('mpc_path_lookahead_m', 6.0))
            self._mission_cross_track = require_positive(
                'mission.mpc_path_cross_track_m',
                mission_config.get('mpc_path_cross_track_m', 0.25))
            self._mission_sample_spacing = require_positive(
                'mission.bspline_sample_spacing_m',
                mission_config.get('bspline_sample_spacing_m', 0.1))
            self._precland_handoff = require_positive(
                'mission.precland_handoff_m',
                mission_config.get('precland_handoff_m', 6.0))
            self._return_replan_min_period = require_positive(
                'mission.return_replan_min_period_s',
                mission_config.get('return_replan_min_period_s', 2.0))
            self._precland_target_timeout = require_positive(
                'px4_vehicle.sitl_parameter_overrides.PLD_BTOUT',
                mission_document.get('px4_vehicle', {}).get(
                    'sitl_parameter_overrides', {}).get('PLD_BTOUT', 0.5))
            # A* is CPU-bound Python. A thread delayed the 50 Hz Offboard
            # heartbeat by hundreds of milliseconds in measurement; a spawned
            # worker process keeps ROS/DDS state out of the child and the
            # control timer responsive.
            self._planner_pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_planner_worker_init)
            self.get_logger().info(
                'CJU A* -> geometry-only B-spline -> PX4 Goto ready')

        self.state = 'PRECHECK'
        self._takeoff_requested = False
        self._hold_pos = np.zeros(3)
        self._launch_ground = None
        self._plan_start = None
        self._plan_goal = None
        self._last_safe_goto = None
        self._precland_goto = None
        self._last_return_plan_t = None
        self.p_d = None
        self.v_d = np.zeros(3)
        self._local_valid = False
        self._t_position = None
        self._ref_alt = None
        self.cue = None
        self.cue_v = np.zeros(3)
        self._t_cue = None
        self._t_cue_v = None
        self._t_cue_source = None
        self._t_cue_v_source = None
        self.vis = None
        self.vis_valid = False
        self._t_vis = None
        self._bias = np.zeros(3)             # learned (vision - cue) offset
        self.k = 0
        self._status = None
        self._t_status = None
        self._precheck_since = None
        self._last_engage_cmd = None
        self._native_takeoff_accepted = False
        self._last_offboard_cmd = None
        self._last_precland_cmd = None
        self._precland_since = None
        self._native_precland_accepted = False
        self._last_landing_source_t = None
        self._last_precheck_report = None
        self.engaged = False
        self.armed = None                    # from VehicleStatus; None = unknown
        self.landed = None                   # PX4 VehicleLandDetected verdict

        self.sp_pub = self.create_publisher(TrajectorySetpoint,
                                            '/fmu/in/trajectory_setpoint', _px4_qos())
        self.goto_pub = self.create_publisher(
            GotoSetpoint, '/fmu/in/goto_setpoint', _px4_qos())
        self.ocm_pub = self.create_publisher(OffboardControlMode,
                                             '/fmu/in/offboard_control_mode', _px4_qos())
        self.cmd_pub = self.create_publisher(VehicleCommand,
                                             '/fmu/in/vehicle_command', _px4_qos())
        self.landing_target_pub = self.create_publisher(
            LandingTargetPose, '/fmu/in/landing_target_pose', _px4_qos())
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        self.planned_path_pub = self.create_publisher(
            PoseArray, '/mission/planned_path', _planned_path_qos())
        self.vehicle_position_pub = self.create_publisher(
            PointStamped, '/mission/vehicle_position', 10)
        self.create_subscription(String, self.mission_command_topic,
                                 self._on_command, 10)

        self.create_subscription(PointStamped, '/marker/cue', self._on_cue, 10)
        self.create_subscription(Vector3Stamped, '/marker/cue_velocity',
                                 self._on_cue_v, 10)
        self.create_subscription(PointStamped, '/marker/position', self._on_vis, 10)
        self.create_subscription(Bool, '/marker/valid', self._on_valid, 10)
        self.local_pos_topic = require_nonempty(
            'local_pos_topic',
            p('local_pos_topic',
              '/fmu/out/vehicle_local_position_v1').value)
        self.create_subscription(VehicleLocalPosition,
                                 self.local_pos_topic,
                                 self._on_pos, _px4_qos())
        # The two feedback topics are RESOLVED, not guessed — see `_resolve_px4`.
        # Empty means "ask the graph"; set them to pin a name by hand.
        self._land_topic = str(p('land_topic', '').value)
        self._status_topic = str(p('status_topic', '').value)
        self._pending = [
            ('/fmu/out/vehicle_land_detected', self._land_topic,
             'px4_msgs/msg/VehicleLandDetected', VehicleLandDetected,
             self._on_land),
            ('/fmu/out/vehicle_status', self._status_topic,
             'px4_msgs/msg/VehicleStatus', VehicleStatus, self._on_status),
        ]
        self._bind_px4_feedback()
        self.create_timer(self.dt, self._tick)
        if not self.get_parameter('use_sim_time').value:
            self.get_logger().warn('use_sim_time=false — reference timing will '
                                   'drift from physics under RTF<1')
        else:
            # FROZEN-CLOCK WATCHDOG.  With use_sim_time the 50 Hz `_tick` runs
            # on the SIM clock, so if /clock never advances (the sensor bridge
            # isn't delivering Gazebo's clock, or Gazebo is paused/gone) the
            # whole mission silently sits in PRECHECK and "it won't take off" has
            # no visible cause.  This timer runs on the WALL clock, so it fires
            # regardless, and shouts if the sim clock has not moved.
            import time as _t
            from rclpy.clock import Clock, ClockType
            self._wall_t0 = _t.monotonic()
            self._sim_t0 = self._now()
            # A SYSTEM_TIME clock keeps firing while the sim clock is frozen.
            self.create_timer(2.0, self._clock_watchdog,
                              clock=Clock(clock_type=ClockType.SYSTEM_TIME))
        self.get_logger().info(
            'mission_manager: YAML A* + geometry-only B-spline supplies route '
            'positions; PX4 Goto/PRECLAND owns all vehicle dynamics and landing')

    # ------------------------------------------------------------- callbacks
    def _on_command(self, message):
        if self.auto_start:
            self.get_logger().warn(
                f'{self.mission_command_topic} ignored while auto_start=true')
            return
        command = message.data.strip()
        # The CLI republishes until the state changes so DDS discovery cannot
        # lose a one-shot word. Accepted repeats are intentionally quiet.
        if command == 'takeoff' and self._takeoff_requested:
            return
        already_accepted = {
            'takeoff': ('TAKEOFF', 'READY', 'MISSION_PLAN', 'MISSION', 'HOVER',
                        'RETURN_PLAN', 'RETURN', 'PRECLAND', 'DONE'),
            'mission': ('MISSION_PLAN', 'MISSION', 'HOVER', 'RETURN_PLAN',
                        'RETURN', 'PRECLAND', 'DONE'),
            'land': ('RETURN_PLAN', 'RETURN', 'PRECLAND', 'DONE'),
        }
        if self.state in already_accepted.get(command, ()):
            return
        allowed = {
            'takeoff': self.state == 'PRECHECK',
            'mission': self.state == 'READY',
            'land': self.state == 'HOVER',
        }
        if not allowed.get(command, False):
            expected = {
                'PRECHECK': 'takeoff',
                'READY': 'mission',
                'HOVER': 'land',
            }.get(self.state, '없음')
            self.get_logger().warn(
                f'command {command!r} rejected in {self.state}; '
                f'expected {expected!r}')
            return

        if command == 'takeoff':
            self._takeoff_requested = True
        elif command == 'mission':
            if self._planner_pool is None or self.p_d is None:
                self.get_logger().error('dynamic mission planner is unavailable')
                return
            self._start_global_plan(None, return_route=False)
        else:
            if not self._cue_fresh():
                self.get_logger().warn(
                    'landing rejected: trailer cue unavailable or stale')
                return
            distance = float(np.linalg.norm(
                self.p_d[:2] - self.cue[:2]))
            if (distance <= self._precland_handoff
                    and _mission_segment_is_free(
                        self.mission_map_yaml, self.p_d, self.cue)):
                self._enter_precland(distance)
            else:
                self._start_global_plan(self.cue, return_route=True)

    def _on_cue(self, m):
        stamp = self._cue_stamp(m)
        cue = np.array([m.point.x, m.point.y, m.point.z])
        if stamp is not None and np.all(np.isfinite(cue)):
            self.cue = cue
            self._t_cue = self._now()
            self._t_cue_source = stamp
        else:
            self.get_logger().warn(
                'invalid trailer position ignored', throttle_duration_sec=5.0)

    def _on_cue_v(self, m):
        stamp = self._cue_stamp(m)
        velocity = np.array([m.vector.x, m.vector.y, m.vector.z])
        if stamp is not None and np.all(np.isfinite(velocity)):
            self.cue_v = velocity
            self._t_cue_v = self._now()
            self._t_cue_v_source = stamp
        else:
            self.cue_v = np.zeros(3)
            self._t_cue_v = None
            self._t_cue_v_source = None
            self.get_logger().warn(
                'invalid trailer velocity ignored', throttle_duration_sec=5.0)

    def _on_vis(self, m):
        point = np.array([m.point.x, m.point.y, m.point.z])
        if np.all(np.isfinite(point)):
            self.vis = point
            self._t_vis = self._now()

    def _on_valid(self, m):
        self.vis_valid = bool(m.data)

    def _on_pos(self, m):
        self.p_d = np.array([m.y, m.x, -m.z])
        self.v_d = np.array([m.vy, m.vx, -m.vz])
        self._local_valid = bool(
            m.xy_valid and m.z_valid and m.v_xy_valid and m.v_z_valid
            and np.all(np.isfinite(self.p_d))
            and np.all(np.isfinite(self.v_d)))
        self._ref_alt = (
            float(m.ref_alt)
            if m.z_global and math.isfinite(float(m.ref_alt)) else None)
        self._t_position = self._now()
        if self._local_valid:
            position = PointStamped()
            position.header.stamp = self.get_clock().now().to_msg()
            position.header.frame_id = LOCAL_ENU_FRAME_ID
            position.point.x, position.point.y, position.point.z = map(
                float, self.p_d)
            self.vehicle_position_pub.publish(position)

    def _clock_watchdog(self):
        """Wall-time check that the SIM clock is actually advancing."""
        import time as _t
        wall = _t.monotonic() - self._wall_t0
        sim = self._now() - self._sim_t0
        # After 3 s of wall time the sim clock should have moved noticeably.
        if wall > 3.0 and sim < 0.1:
            self.get_logger().error(
                f'/clock is NOT advancing ({sim:.2f}s sim in {wall:.1f}s wall) '
                f'— the mission timer is frozen, so it will never arm or take '
                f'off. The sensor bridge publishes /clock: check it is up and '
                f'seeing Gazebo (ros2 topic hz /clock). Nothing here is wrong; '
                f'the sim clock source is.')
        elif sim > 1.0 and self.p_d is None:
            self.get_logger().warn(
                'clock OK but no vehicle_local_position yet — waiting for '
                f'{self.get_parameter("local_pos_topic").value} (PX4 uXRCE '
                'agent up?)')
        elif sim > 1.0 and self._pending:
            self.get_logger().warn(
                'still waiting for ' + ', '.join(b for b, *_ in self._pending)
                + ' — no publisher of any message version yet, so the '
                  'touchdown cannot be confirmed and the vehicle will not '
                  'disarm itself')
        elif sim > 1.0 and self.p_d is not None and self.cue is None:
            self.get_logger().warn(
                'clock + position OK but no /marker/cue — trailer_cue_node not '
                'publishing; the mission arms only once it has a cue.')

    def _on_status(self, m):
        self._status = m
        self._t_status = self._now()
        self.armed = m.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def _on_land(self, m):
        """Observe PX4's final verdict; this node never substitutes its own."""
        self.landed = bool(m.landed)

    # ---------------------------------------------------------------- helpers
    def _resolve_px4(self, base, type_name):
        """The advertised name of a PX4 topic, across message versions.

        PX4's uXRCE client appends each message's MESSAGE_VERSION to the DDS
        topic name, so ONE logical topic has different names on different
        firmware: on this tree it is `/fmu/out/vehicle_status_v4`, on an older
        one `_v1`, and `/fmu/out/vehicle_land_detected` carries no suffix at
        all because that message is unversioned.  A hardcoded guess is
        therefore correct on exactly one firmware, and it fails SILENTLY —
        subscribing to a name nobody publishes is not an error, it is just
        permanent silence.

        That is not hypothetical: the defaults here were `vehicle_status_v1`
        and `vehicle_land_detected_v1`, neither of which this PX4 publishes, so
        `self.armed` and `self.landed` never updated. Native PRECLAND completion
        would then be unobservable, so graph resolution is fail-closed.

        So ask the graph instead of asserting.  A candidate must have the right
        type AND a live publisher — the second test matters because OUR OWN
        subscription puts the wrong name in the graph with the right type, so
        type alone would happily confirm the mistake it is meant to catch.
        """
        for name, types in self.get_topic_names_and_types():
            if (type_name in types
                    and re.fullmatch(re.escape(base) + r'(_v\d+)?', name)
                    and self.count_publishers(name) > 0):
                return name
        return None

    def _bind_px4_feedback(self):
        """Subscribe to whatever PX4 is actually publishing, once it appears.

        PX4 usually starts after this node, so resolution has to be retried
        rather than done once at construction.  Runs on the WALL clock: with
        use_sim_time a frozen /clock would otherwise stop the retry too, and
        the resulting silence would look exactly like the bug above.
        """
        still = []
        for base, override, type_name, msg_type, cb in self._pending:
            name = override or self._resolve_px4(base, type_name)
            if name is None:
                still.append((base, override, type_name, msg_type, cb))
                continue
            self.create_subscription(msg_type, name, cb, _px4_qos())
            self.get_logger().info(f'{base} -> subscribed on {name}')
        self._pending = still
        if self._pending and not hasattr(self, '_bind_timer'):
            from rclpy.clock import Clock, ClockType
            self._bind_timer = self.create_timer(
                2.0, self._retry_bind, clock=Clock(clock_type=ClockType.SYSTEM_TIME))

    def _retry_bind(self):
        self._bind_px4_feedback()
        if not self._pending:
            self._bind_timer.cancel()

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _cue_stamp(self, message):
        if message.header.frame_id != LOCAL_ENU_FRAME_ID:
            return None
        stamp = (float(message.header.stamp.sec)
                 + float(message.header.stamp.nanosec) * 1.0e-9)
        return stamp if math.isfinite(stamp) else None

    def _cue_fresh(self):
        if (self.cue is None or self._t_cue is None
                or self._t_cue_v is None):
            return False
        now = self._now()
        position_age = now - self._t_cue
        velocity_age = now - self._t_cue_v
        return (np.all(np.isfinite(self.cue))
                and np.all(np.isfinite(self.cue_v))
                and 0.0 <= position_age <= self.cue_timeout_s
                and 0.0 <= velocity_age <= self.cue_timeout_s)

    def _landing_target_fresh(self):
        """Use the source timestamp so PX4 can enforce PLD_BTOUT honestly."""
        if (not self._cue_fresh() or self._t_cue_source is None
                or self._t_cue_v_source is None):
            return False
        now = self._now()
        return (
            0.0 <= now - self._t_cue_source <= self._precland_target_timeout
            and 0.0 <= now - self._t_cue_v_source
            <= self._precland_target_timeout)

    def _precheck_issues(self):
        """Return fail-closed Phase 0 blockers; an empty list permits arming."""
        now = self._now()
        issues = []
        position_age = (math.inf if self._t_position is None
                        else now - self._t_position)
        if (self.p_d is None or not 0.0 <= position_age <= self.precheck_timeout
                or not self._local_valid):
            issues.append('local position invalid/stale')
        if self._ref_alt is None:
            issues.append('global altitude reference unavailable')
        status_age = (math.inf if self._t_status is None
                      else now - self._t_status)
        if (self._status is None
                or not 0.0 <= status_age <= self.precheck_timeout):
            issues.append('vehicle status unavailable/stale')
        else:
            if not self._status.pre_flight_checks_pass:
                issues.append('PX4 preflight checks failed')
            if self._status.failsafe:
                issues.append('PX4 failsafe active')
            if self._status.failure_detector_status != VehicleStatus.FAILURE_NONE:
                issues.append('PX4 failure detector active')
            if (not self.engaged
                    and self._status.arming_state
                    != VehicleStatus.ARMING_STATE_DISARMED):
                issues.append('vehicle is not confirmed disarmed')
        if not self._cue_fresh():
            issues.append('trailer cue invalid/stale')
        if not self.auto_start and self._planner_pool is None:
            issues.append('YAML global planner unavailable')
        if self._pending:
            issues.append('PX4 feedback topics unresolved')
        return issues

    def _cmd(self, command, p1=math.nan, p2=math.nan, p3=math.nan, p4=math.nan,
             p5=math.nan, p6=math.nan, p7=math.nan):
        c = VehicleCommand()
        c.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        c.command = command
        for index, value in enumerate((p1, p2, p3, p4, p5, p6, p7), 1):
            setattr(c, f'param{index}', float(value))
        c.target_system = c.target_component = 1
        c.source_system = c.source_component = 1
        c.from_external = True
        self.cmd_pub.publish(c)

    def _send_takeoff(self):
        """Ask PX4 Navigator to use MIS_TAKEOFF_ALT and MPC_TKO_SPEED."""
        if self._ref_alt is None:
            return False
        self._cmd(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            p7=self._ref_alt + self.takeoff_alt)
        return True

    def _ocm(self):
        m = OffboardControlMode()
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        m.position = True
        self.ocm_pub.publish(m)

    def _send(self, pos, vel=None, acc=None):
        pp = enu_to_ned(np.asarray(pos, float))
        s = TrajectorySetpoint()
        s.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        # Generated px4_msgs arrays otherwise default to zero.  Explicit NaNs
        # mean "uncontrolled/feed-forward absent" for every omitted field.
        s.velocity = [float('nan')] * 3
        s.acceleration = [float('nan')] * 3
        s.jerk = [float('nan')] * 3
        s.position = [float(pp[0]), float(pp[1]), float(pp[2])]
        if vel is not None:
            v = enu_to_ned(vel)
            s.velocity = [float(v[0]), float(v[1]), float(v[2])]
        if acc is not None:
            a = enu_to_ned(acc)
            s.acceleration = [float(a[0]), float(a[1]), float(a[2])]
        s.yaw = float('nan')
        s.yawspeed = float('nan')
        self.sp_pub.publish(s)

    def _send_goto(self, pos):
        """Send geometry only; PX4 creates the velocity/acceleration profile."""
        p_ned = enu_to_ned(np.asarray(pos, float))
        message = GotoSetpoint()
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        message.position = [float(value) for value in p_ned]
        message.flag_control_heading = False
        message.heading = float('nan')
        message.flag_set_max_horizontal_speed = False
        message.max_horizontal_speed = float('nan')
        message.flag_set_max_vertical_speed = False
        message.max_vertical_speed = float('nan')
        message.flag_set_max_heading_rate = False
        message.max_heading_rate = float('nan')
        self.goto_pub.publish(message)

    def _publish_landing_target(self):
        """Publish one live target measurement; never a flight setpoint."""
        if self.p_d is None or not self._landing_target_fresh():
            return False
        # Do not turn one old measurement into a fresh 50 Hz stream. PX4 must
        # observe an actual publication gap and enforce PLD_BTOUT on source loss.
        if self._t_cue_source == self._last_landing_source_t:
            return True
        target, target_v = self._target()
        absolute = enu_to_ned(target)
        relative = enu_to_ned(target - self.p_d)
        relative_v = enu_to_ned(target_v - self.v_d)
        message = LandingTargetPose()
        # Zero asks uXRCE/PX4 to stamp the one-shot sample at receipt. Source
        # time remains local for freshness/deduplication across clock domains.
        message.timestamp = 0
        message.is_static = False
        message.rel_pos_valid = True
        message.rel_vel_valid = True
        message.x_rel, message.y_rel, message.z_rel = map(float, relative)
        message.vx_rel, message.vy_rel = map(float, relative_v[:2])
        message.cov_x_rel = message.cov_y_rel = 0.04
        message.cov_vx_rel = message.cov_vy_rel = 0.04
        message.abs_pos_valid = True
        message.x_abs, message.y_abs, message.z_abs = map(float, absolute)
        self.landing_target_pub.publish(message)
        self._last_landing_source_t = self._t_cue_source
        return True

    def _enter_precland(self, distance):
        """End Offboard authority and let PX4 own the complete landing."""
        if not self._publish_landing_target():
            return False
        candidate = getattr(self, '_last_safe_goto', None)
        self._precland_goto = (
            np.asarray(candidate, float).copy()
            if candidate is not None
            and _mission_segment_is_free(
                self.mission_map_yaml, self.p_d, candidate)
            else None)
        self._hold_pos = self.p_d.copy()
        self._publish_planned_path(None)
        self._set_state(
            'PRECLAND', f'(PX4 precision-land handoff, d={distance:.1f} m)')
        return True

    def _set_state(self, s, why=''):
        if s != self.state:
            self.get_logger().info(f'{self.state} -> {s}  {why}')
            self.state = s
            if s == 'ABORT' and getattr(self, 'p_d', None) is not None:
                self._hold_pos = self.p_d.copy()
                self._publish_planned_path(None)
            if s in ('MISSION_PLAN', 'RETURN_PLAN'):
                self._last_offboard_cmd = None
            if s == 'TAKEOFF':
                self._native_takeoff_accepted = False
            if s == 'PRECLAND':
                self._precland_since = self._now()
                self._last_precland_cmd = None
                self._native_precland_accepted = False

    def _start_global_plan(self, goal_local_enu, *, return_route):
        """Start one serialized map-goal or trailer A*/B-spline leg."""
        if self._planner_pool is None or self.p_d is None:
            raise RuntimeError('global planner is unavailable')
        rolling_return = (
            return_route and self.state == 'RETURN'
            and self._mission_path is not None)
        start = self.p_d.copy()
        self._hold_pos = start.copy()
        goal = (None if goal_local_enu is None
                else np.asarray(goal_local_enu, float).copy())
        if goal is not None and not np.all(np.isfinite(goal)):
            raise ValueError('global plan goal must be finite')
        if not rolling_return:
            self._publish_planned_path(None)
        if self._plan_future is not None:
            self._plan_future.cancel()
        self._plan_future = self._planner_pool.submit(
            _plan_global_path, self.mission_map_yaml, list(start),
            None if goal is None else goal.tolist())
        self._plan_start = start
        self._plan_goal = goal
        if return_route:
            self._last_return_plan_t = self._now()
        if not rolling_return:
            self._mission_arc_m = None
            self._mission_path = None
            self._mission_progress_m = 0.0
        if return_route:
            state, destination = 'RETURN_PLAN', 'moving trailer'
        else:
            state, destination = 'MISSION_PLAN', 'map (50,50)'
        self._set_state(state, f'(A* + geometry B-spline -> {destination})')

    def _publish_planned_path(self, path):
        """Publish only geometry accepted by the flight authority."""
        message = PoseArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = LOCAL_ENU_FRAME_ID
        if path is not None:
            points = np.asarray(path, float)
            if (points.ndim != 2 or points.shape[1] != 3
                    or not np.all(np.isfinite(points))):
                raise ValueError('planned path must be finite Nx3 local ENU')
            for point in points:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(
                    float, point)
                pose.orientation.w = 1.0
                message.poses.append(pose)
        self.planned_path_pub.publish(message)

    def _depression(self, target):
        """Angle below horizontal from the vehicle to `target`, in degrees.

        90 deg is straight down.  This is the axis the measured detection rate
        is stratified along, so it is the axis the marker is trusted along.
        """
        d = self.p_d - target
        return math.degrees(math.atan2(max(0.0, d[2]),
                                       max(float(math.hypot(d[0], d[1])), 1e-6)))

    def _target(self):
        """Return the live cue plus a fresh, horizontal ArUco correction."""
        cue = self.cue if self.cue is not None else self.p_d
        if (self.vis_valid and self.vis is not None and self.cue is not None
                and self._t_vis is not None
                and self._now() - self._t_vis <= self.vis_fresh
                and self._depression(self.vis) >= self.align_deg):
            err = self.vis[:2] - cue[:2]
            if float(np.linalg.norm(err)) <= self.bias_max:
                alpha = min(1.0, self.dt / max(self.bias_tau, 1e-3))
                step = alpha * (err - self._bias[:2])
                cap = self.bias_rate * self.dt
                n = float(np.linalg.norm(step))
                if n > cap:
                    step *= cap / n
                self._bias[:2] += step
        return cue + self._bias, self.cue_v.copy()

    # ------------------------------------------------------------------ phases
    def _tick(self):
        self.k += 1
        self.state_pub.publish(String(data=self.state))
        if self.state == 'DONE' or self.p_d is None:
            return

        nav_state = None if self._status is None else self._status.nav_state
        takeoff_modes = (
            VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF,
            VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
        )
        if self.state == 'TAKEOFF' and nav_state in takeoff_modes:
            self._native_takeoff_accepted = True
        if (self.state == 'PRECLAND'
                and nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND):
            self._native_precland_accepted = True
        native_takeoff = (
            self.state == 'TAKEOFF'
            and getattr(self, '_native_takeoff_accepted', False)
        )
        native_precland = (
            self.state == 'PRECLAND'
            and getattr(self, '_native_precland_accepted', False)
        )
        # Stop Offboard only after PX4 has actually accepted the native mode.
        if not native_takeoff and not native_precland:
            self._ocm()

        if self.state in ('RETURN_PLAN', 'RETURN') and not self._cue_fresh():
            if self._plan_future is not None:
                self._plan_future.cancel()
                self._plan_future = None
            self._hold_pos = self.p_d.copy()
            self._set_state('ABORT', '(trailer cue stale)')
            return

        if self.state == 'PRECHECK':
            if self._local_valid and not self.engaged:
                self._launch_ground = self.p_d.copy()
                self._hold_pos = self.p_d.copy()
            if self._launch_ground is None:
                return
            self._send(self._hold_pos)
            now = self._now()
            issues = self._precheck_issues()
            if issues:
                self._precheck_since = None
                if (self._last_precheck_report is None
                        or now - self._last_precheck_report >= 2.0):
                    self.get_logger().warn(
                        'PRECHECK waiting: ' + '; '.join(issues))
                    self._last_precheck_report = now
                return
            if self._precheck_since is None:
                self._precheck_since = now
            requested = self.auto_start or self._takeoff_requested
            if not requested or now - self._precheck_since < self.precheck_warmup:
                return
            if (self.armed is True and nav_state
                    == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
                self._send_takeoff()
                self._last_engage_cmd = now
                self._set_state('TAKEOFF', '(PX4 NAV_TAKEOFF)')
                return
            if (self._last_engage_cmd is None
                    or now - self._last_engage_cmd >= 1.0):
                self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.engaged = True
                self._last_engage_cmd = now
            return

        if self.state == 'TAKEOFF':
            now = self._now()
            if not native_takeoff:
                self._send(self._hold_pos)
                if (self._last_engage_cmd is None
                        or now - self._last_engage_cmd >= 1.0):
                    self._send_takeoff()
                    self._last_engage_cmd = now
                return
            if nav_state not in takeoff_modes:
                return
            if (self.armed is True
                    and abs(self.p_d[2] - self.takeoff_alt) <= 0.15
                    and abs(self.v_d[2]) <= 0.15):
                if self.auto_start and self._planner_pool is not None:
                    self._start_global_plan(None, return_route=False)
                elif self.auto_start and self._cue_fresh():
                    distance = float(np.linalg.norm(
                        self.p_d[:2] - self.cue[:2]))
                    self._enter_precland(distance)
                else:
                    self._hold_pos = self.p_d.copy()
                    self._set_state('READY', f'(PX4 takeoff at {self.p_d[2]:.1f} m)')
            return

        if self.state in ('READY', 'HOVER'):
            self._send(self._hold_pos)
            return

        if self.state in ('MISSION_PLAN', 'RETURN_PLAN'):
            return_route = self.state == 'RETURN_PLAN'
            rolling_path = (
                return_route
                and getattr(self, '_mission_path', None) is not None)
            rolling_path_safe = False
            if rolling_path:
                live_distance = float(np.linalg.norm(
                    self.p_d[:2] - self.cue[:2]))
                if (live_distance <= self._precland_handoff
                        and _mission_segment_is_free(
                            self.mission_map_yaml, self.p_d, self.cue)
                        and self._enter_precland(live_distance)):
                    self._plan_future.cancel()
                    self._plan_future = None
                    return
                progress, safe_target = _safe_spatial_path_target(
                    self.mission_map_yaml, self._mission_arc_m,
                    self._mission_path, self.p_d,
                    self._mission_progress_m, self._mission_lookahead,
                    self._mission_cross_track)
                if safe_target is not None:
                    self._mission_progress_m = progress
                    self._last_safe_goto = np.asarray(
                        safe_target, float).copy()
                    self._send_goto(safe_target)
                    rolling_path_safe = True
                else:
                    self._hold_pos = self.p_d.copy()
                    self._send(self._hold_pos)
            else:
                self._send(self._hold_pos)
            now = self._now()
            if nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                if (self._last_offboard_cmd is None
                        or now - self._last_offboard_cmd >= 1.0):
                    self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self._last_offboard_cmd = now
                return
            if not self._plan_future.done():
                return
            try:
                arc_m, path, expanded = self._plan_future.result()
            except Exception as exc:
                self._plan_future = None
                self.get_logger().error(
                    f'global A*/B-spline replan failed: {exc}')
                if rolling_path and rolling_path_safe:
                    self._set_state(
                        'RETURN', '(planner failure; keep prior safe route)')
                    return
                fallback = 'ABORT' if return_route else 'READY'
                self._hold_pos = self.p_d.copy()
                self._set_state(fallback, '(planner failure; hold)')
                return
            self._plan_future = None
            if (not rolling_path
                    and float(np.linalg.norm(
                        self.p_d[:2] - self._plan_start[:2]))
                    > self.mission_tolerance):
                if return_route and not self._cue_fresh():
                    self._hold_pos = self.p_d.copy()
                    self._set_state('ABORT', '(cue stale after planning)')
                else:
                    goal = self.cue if return_route else None
                    self._start_global_plan(goal, return_route=return_route)
                return
            replacement_progress = 0.0
            if rolling_path:
                replacement = _splice_path_from_current(
                    lambda start, goal: _mission_planning_segment_is_free(
                        self.mission_map_yaml, start, goal),
                    arc_m, path, self.p_d,
                    self._mission_lookahead)
                if replacement is None:
                    self.get_logger().error(
                        'global A*/B-spline replan failed: '
                        'replacement route has no exact-safe splice')
                    if rolling_path_safe:
                        self._set_state(
                            'RETURN', '(keep prior exact-safe route)')
                    else:
                        self._hold_pos = self.p_d.copy()
                        self._set_state(
                            'ABORT', '(replacement route is unreachable)')
                    return
                arc_m, path = replacement
                replacement_progress, replacement_target = (
                    _safe_spatial_path_target(
                        self.mission_map_yaml, arc_m, path, self.p_d, 0.0,
                        self._mission_lookahead,
                        self._mission_cross_track))
                if replacement_target is None:
                    self.get_logger().error(
                        'global A*/B-spline replan failed: '
                        'replacement route has no exact-safe splice')
                    if rolling_path_safe:
                        self._set_state(
                            'RETURN', '(keep prior exact-safe route)')
                    else:
                        self._hold_pos = self.p_d.copy()
                        self._set_state(
                            'ABORT', '(replacement route is unreachable)')
                    return
            self._mission_arc_m = arc_m
            self._mission_path = path
            self._mission_progress_m = replacement_progress
            self._publish_planned_path(path)
            self.get_logger().info(
                f'global A*/B-spline: {len(path)} samples, '
                f'{arc_m[-1]:.1f} m, {expanded} A* expansions')
            next_state = 'RETURN' if return_route else 'MISSION'
            self._set_state(
                next_state, '(validated geometry B-spline -> PX4 Goto)')
            return

        if self.state in ('MISSION', 'RETURN'):
            return_route = self.state == 'RETURN'
            if return_route:
                live_distance = float(np.linalg.norm(
                    self.p_d[:2] - self.cue[:2]))
                if (live_distance <= self._precland_handoff
                        and _mission_segment_is_free(
                            self.mission_map_yaml, self.p_d, self.cue)):
                    self._enter_precland(live_distance)
                    return
            progress, safe_target = _safe_spatial_path_target(
                self.mission_map_yaml, self._mission_arc_m,
                self._mission_path, self.p_d,
                self._mission_progress_m, self._mission_lookahead,
                self._mission_cross_track)
            if safe_target is None:
                hold = self.p_d.copy()
                hold[2] = self.takeoff_alt
                self._send_goto(hold)
                goal = self.cue if return_route else None
                self._start_global_plan(goal, return_route=return_route)
                return
            self._mission_progress_m = progress
            self._last_safe_goto = np.asarray(safe_target, float).copy()
            self._send_goto(safe_target)
            at_end = (
                self._mission_arc_m[-1] - self._mission_progress_m
                <= self.mission_tolerance
            )
            if return_route:
                now = self._now()
                retarget_due = (
                    self._last_return_plan_t is None
                    or now - self._last_return_plan_t
                    >= self._return_replan_min_period)
            if return_route and (retarget_due or at_end):
                self._last_return_plan_t = now
                target = np.asarray(self.cue, float).copy()
                target[2] = self._mission_path[-1, 2]
                replacement = _retarget_path_tail(
                    lambda start, goal: _mission_planning_segment_is_free(
                        self.mission_map_yaml, start, goal),
                    self._mission_arc_m, self._mission_path,
                    self._mission_progress_m, target,
                    self._mission_lookahead, self._mission_cross_track,
                    self._mission_sample_spacing)
                if replacement is not None:
                    self._mission_arc_m, self._mission_path = replacement
                    self._publish_planned_path(self._mission_path)
                    self.get_logger().info(
                        'return tail retarget accepted: '
                        f'goal=({target[0]:.3f},{target[1]:.3f})')
                    return
                self.get_logger().warn(
                    'return tail retarget rejected; '
                    'keep prior route while full planner runs')
                self._start_global_plan(self.cue, return_route=True)
                return

            settled = (
                float(np.linalg.norm(
                    self.p_d - self._mission_path[-1]))
                <= self.mission_tolerance
                and float(np.linalg.norm(self.v_d)) <= self.settle_v_tol
            )
            if at_end and settled:
                self._hold_pos = self._mission_path[-1].copy()
                if not return_route:
                    self._set_state(
                        'HOVER', '(B-spline goal (50,50), altitude 5 m)')
            return

        if self.state == 'PRECLAND':
            published = self._publish_landing_target()
            now = self._now()
            if self.landed is True and self.armed is False:
                self._set_state('DONE', '(PX4 landed and auto-disarmed)')
                return
            if not native_precland:
                # Preserve PX4's Goto smoother until the native-mode handoff.
                bridge_goto = getattr(self, '_precland_goto', None)
                if (published and bridge_goto is not None
                        and _mission_segment_is_free(
                            self.mission_map_yaml, self.p_d, bridge_goto)):
                    self._send_goto(bridge_goto)
                else:
                    self._precland_goto = None
                    self._hold_pos = self.p_d.copy()
                    self._send(self._hold_pos)
                if (published and now - self._precland_since >= 0.1
                        and (self._last_precland_cmd is None
                             or now - self._last_precland_cmd >= 1.0)):
                    self._cmd(VehicleCommand.VEHICLE_CMD_NAV_PRECLAND)
                    self._last_precland_cmd = now
                elif not published and self.k % 100 == 0:
                    self.get_logger().warn(
                        'PRECLAND target stale; PX4 owns search/fallback')
            return

        if self.state == 'ABORT':
            self._send(np.array([
                self._hold_pos[0], self._hold_pos[1], self.takeoff_alt]))
            if (self._planner_pool is None and self._cue_fresh()
                    and self.p_d[2] >= self.takeoff_alt - 0.5):
                distance = float(np.linalg.norm(
                    self.p_d[:2] - self.cue[:2]))
                self._enter_precland(distance)
            return

    def destroy_node(self):
        if self._plan_future is not None:
            self._plan_future.cancel()
        if self._planner_pool is not None:
            self._planner_pool.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
