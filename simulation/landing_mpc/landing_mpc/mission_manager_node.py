"""Sequence the CJU mission as its only companion-side PX4 setpoint authority.

Long-range motion follows the target's reported coordinates (``/marker/cue``)
because the 1.3 m ArUco marker is unresolved at tens of metres. A fresh, valid
vision/KF position slowly corrects the cue. Distinct KF-accepted camera fixes
within a configured window qualify terminal control.

This node is the single Offboard setpoint authority — never run it alongside
another setpoint publisher.

Subscribes
    /mission/command     String          takeoff → mission → land
    /marker/cue          PointStamped    long-range target position (ENU)
    /marker/cue_velocity Vector3Stamped  long-range target velocity
    /marker/position     PointStamped    vision/KF target position (ENU)
    /marker/velocity     Vector3Stamped  vision/KF target velocity (ENU)
    /marker/valid        Bool            vision usable (KF not over-coasting)
    /marker/entry_valid  Bool            3 accepted fixes within 0.5 s
    /fmu/out/vehicle_local_position_v1
Publishes
    /fmu/in/goto_setpoint, /fmu/in/trajectory_setpoint,
    /fmu/in/offboard_control_mode,
    /fmu/in/vehicle_command
    /mission/state       String          current phase (observability)
    /mission/landing_diagnostics String  range/speed/vision/commit/turnaround
    /mission/active_plan_markers MarkerArray atomic path + active-path SFC

Phases

  Phase 0
    PRECHECK  validate PX4 feedback, cue, planner and Offboard readiness
  Phase 1
    TAKEOFF   PX4 NAV_TAKEOFF to takeoff_alt
    READY     hold after takeoff until the explicit mission command
  Phase 2
    MISSION_PLAN plan A* and a geometry-only B-spline without blocking Offboard
    MISSION   follow the accepted B-spline with TrackingMPC to the map goal
    HOVER     hold over the map goal and wait for land
  Phase 3
    RETURN_PLAN plan A* and a geometry-only B-spline around map obstacles
    RETURN    asynchronously rebuild the moving-target A*/SFC/B-spline route
    LANDING_ACQUIRE  use LandingMPC to align at fixed altitude
    LANDING_DESCEND  use LandingMPC to descend on the corrected moving cue
    PRECLAND  hand final approach, contact and auto-disarm to PX4
    ABORT     hold after loss of the continuous trailer cue

The route and landing controllers are separate wang MPC implementations and
never publish independently. ArUco remains a measurement source; this manager
alone selects the active controller and publishes the resulting setpoint.
"""

from __future__ import annotations

import math
import multiprocessing
import re
import signal
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from std_msgs.msg import Bool, Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from px4_msgs.msg import (GotoSetpoint, LandingTargetPose, OffboardControlMode,
                          TrajectorySetpoint, VehicleCommand, VehicleLandDetected,
                          VehicleLocalPosition, VehicleStatus)

from path_plan.mpc import TrackingMPC

from .frame import LOCAL_ENU_FRAME_ID, enu_to_ned
from .mpc import LandingMPC
from .parameter_utils import (
    derive_control_timing,
    require_finite,
    require_nonnegative,
    require_nonempty,
    require_positive,
)
from .predictor import predict_const_vel
from .reference import HorizonReference


DEFAULT_ENTRY_FIX_WINDOW_S = 0.5


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


def _mission_frame_contract(document):
    """Resolve the mission map frame and the PX4-local ENU origin."""
    mission = document['mission']
    frame_name = mission['coordinate_frame']
    frame = document['frames'][frame_name]
    source = mission.get('obstacle_source', 'mission_obstacles')
    if source == 'city_buildings':
        if (frame_name != 'gazebo_world'
                or frame.get('convention') != 'ENU'):
            raise ValueError(
                'city_buildings must use the gazebo_world ENU frame')
        heading_deg = frame.get('heading_deg_enu', 0.0)
        origin_value = frame.get('origin_enu_m', [0.0, 0.0, 0.0])
    else:
        heading_deg = frame['heading_deg_enu']
        origin_value = frame['origin_enu_m']
    heading = math.radians(float(heading_deg))
    rotation = np.array([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)],
    ])
    origin = np.asarray(origin_value[:2], float)
    spawn_pose = document['spawn']['gazebo_spawn_pose_enu']
    spawn = np.asarray([spawn_pose['x'], spawn_pose['y']], float)
    if (origin.shape != (2,) or not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(spawn))
            or not math.isfinite(heading)):
        raise ValueError('mission frame and spawn must be finite')
    return rotation, spawn, origin


def _mission_obstacle_geometry(document, map_yaml, altitude):
    """Return raw AABBs and XY bounds in the configured mission frame."""
    from path_plan.world_model import WorldModel

    mission = document['mission']
    source = mission.get('obstacle_source', 'mission_obstacles')
    if source == 'city_buildings':
        # City footprints and map bounds are already Gazebo-world ENU. Reuse
        # the path_plan parser so both pipelines keep identical raw AABBs.
        city = WorldModel.from_city_yaml(
            map_yaml, xy_clearance_m=0.0,
            ceiling_m=altitude + 1.0, overfly_allowed=False)
        return (city.boxes_min, city.boxes_max,
                city.bounds_min[:2], city.bounds_max[:2])
    if source != 'mission_obstacles':
        raise ValueError(f'unsupported mission.obstacle_source: {source}')

    lows, highs = [], []
    for obstacle in mission['obstacles']:
        centre = np.asarray(obstacle['center_m'], float)
        half_size = 0.5 * np.asarray(obstacle['size_m'], float)
        low, high = centre - half_size, centre + half_size
        # Deliberately forbid overflight: this mission verifies lateral
        # avoidance of the configured ten-metre barriers.
        low[2], high[2] = -1.0e4, 1.0e4
        lows.append(low)
        highs.append(high)

    terrain = document['terrain']
    if terrain['coordinate_frame'] != mission['coordinate_frame']:
        raise ValueError('mission and terrain must use the same frame')
    terrain_size = np.asarray(terrain['size_m'], float)
    if (terrain_size.shape != (2,) or not np.all(np.isfinite(terrain_size))
            or np.any(terrain_size <= 0.0)):
        raise ValueError('terrain.size_m must contain two positive dimensions')
    terrain_center = np.asarray(terrain['center_m'], float)
    if terrain_center.shape != (2,) or not np.all(np.isfinite(terrain_center)):
        raise ValueError('terrain.center_m must contain two finite values')
    half_terrain = 0.5 * terrain_size
    return (np.asarray(lows, float), np.asarray(highs, float),
            terrain_center - half_terrain, terrain_center + half_terrain)


def _plan_global_path(map_yaml, start_local_enu=None, goal_local_enu=None,
                      *, include_diagnostics=False):
    """Build one exact-safe A* -> geometry-only B-spline path."""
    from path_plan.astar import AStarPlanner3D
    from path_plan.bspline_optimizer import BsplineOptimizer
    from path_plan.sfc import SafeFlightCorridor
    from path_plan.uniform_bspline import UniformBspline
    from path_plan.world_model import WorldModel

    document = yaml.safe_load(Path(map_yaml).read_text(encoding='utf-8'))
    mission = document['mission']
    rotation, spawn, frame_origin = _mission_frame_contract(document)
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
    clearance = float(mission['vehicle_clearance_xy_m'])
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

    obstacle_lows, obstacle_highs, bounds_low, bounds_high = (
        _mission_obstacle_geometry(document, map_yaml, altitude))

    def make_world(clearance_xy_m, z_half_width):
        return WorldModel.from_boxes(
            obstacle_lows, obstacle_highs,
            [*bounds_low, altitude - z_half_width],
            [*bounds_high, altitude + z_half_width],
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
    sfc_started = perf_counter()
    positions, active_corridor = SafeFlightCorridor(
        spline_world).cover_polyline(positions)
    sfc_generation_time_ms = 1000.0 * (perf_counter() - sfc_started)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(positions, axis=0), axis=1))]
    world_xy = positions[:, :2] @ rotation.T + frame_origin
    local_positions = np.column_stack((world_xy - spawn, positions[:, 2]))
    result_tuple = (arc, local_positions, result.expanded)
    if not include_diagnostics:
        return result_tuple
    diagnostics = {
        # These boxes certify the final active polyline.  The optimizer's soft
        # control-point corridor shapes the B-spline but does not certify the
        # current-position splice added when a rolling plan completes.
        'sfc_boxes_min_map': active_corridor.boxes_min.copy(),
        'sfc_boxes_max_map': active_corridor.boxes_max.copy(),
        'sfc_generation_time_ms': sfc_generation_time_ms,
    }
    return (*result_tuple, diagnostics)


def _path_position(arc_m, path, distance_m):
    distance = float(np.clip(distance_m, arc_m[0], arc_m[-1]))
    return np.array([
        np.interp(distance, arc_m, path[:, axis]) for axis in range(3)])


def _s_curve_stop_speed(remaining_m, acceleration_m_s2, jerk_m_s3):
    """Maximum speed that can stop in ``remaining_m`` with zero end accel."""
    remaining = max(0.0, float(remaining_m))
    acceleration = float(acceleration_m_s2)
    jerk = float(jerk_m_s3)
    if acceleration <= 0.0 or jerk <= 0.0:
        raise ValueError('acceleration and jerk limits must be positive')

    # Symmetric S-curve: triangular below the acceleration limit, otherwise
    # jerk down, hold -a_max, jerk back to zero acceleration at the endpoint.
    triangular_distance = acceleration ** 3 / jerk ** 2
    if remaining <= triangular_distance:
        return (remaining * math.sqrt(jerk)) ** (2.0 / 3.0)
    ratio = acceleration ** 2 / jerk
    return 0.5 * (math.sqrt(ratio ** 2 + 8.0 * acceleration * remaining)
                  - ratio)


def _s_curve_stop_distance(speed_m_s, acceleration_m_s2, jerk_m_s3):
    """Distance needed to stop with the same symmetric S-curve contract.

    Its rest-to-speed duration is ``2 * distance / speed`` because the
    symmetric velocity profile has mean speed ``speed / 2``.
    """
    speed = max(0.0, float(speed_m_s))
    acceleration = float(acceleration_m_s2)
    jerk = float(jerk_m_s3)
    if acceleration <= 0.0 or jerk <= 0.0:
        raise ValueError('acceleration and jerk limits must be positive')
    triangular_speed = acceleration ** 2 / jerk
    if speed <= triangular_speed:
        return speed ** 1.5 / math.sqrt(jerk)
    return (speed ** 2 / (2.0 * acceleration)
            + speed * acceleration / (2.0 * jerk))


def _relative_braking_path_speed(nominal_speed_m_s, tangent_xy,
                                 target_velocity_xy, range_xy_m,
                                 start_range_m, target_relative_speed_m_s):
    """Reduce moving-target relative speed without leaving the B-spline.

    The path geometry remains authoritative.  Inside ``start_range_m`` this
    selects the fastest forward speed whose target-relative velocity is below
    the requested capture speed.  A target component normal to the path cannot
    be cancelled without leaving the validated spline, so the closest feasible
    tangent speed is used instead.
    """
    nominal = float(nominal_speed_m_s)
    tangent = np.asarray(tangent_xy, float)
    target_velocity = np.asarray(target_velocity_xy, float)
    distance = max(0.0, float(range_xy_m))
    start = float(start_range_m)
    target_relative = float(target_relative_speed_m_s)
    norm = float(np.linalg.norm(tangent))
    if (nominal < 0.0 or tangent.shape != (2,)
            or target_velocity.shape != (2,)
            or not np.all(np.isfinite(np.r_[nominal, tangent,
                                             target_velocity, distance, start,
                                             target_relative]))
            or start <= 0.0 or target_relative < 0.0 or norm <= 1.0e-9):
        raise ValueError('invalid moving-target B-spline braking input')
    if distance >= start or nominal == 0.0:
        return nominal

    direction = tangent / norm
    target_along = float(target_velocity @ direction)
    target_cross_sq = max(
        0.0, float(target_velocity @ target_velocity) - target_along ** 2)
    nominal_relative = nominal * direction - target_velocity

    relative_limit = min(
        float(np.linalg.norm(nominal_relative)), target_relative)

    # ||speed*tangent - v_target|| <= relative_limit.  If the target's
    # cross-path velocity already exceeds the cap, cancelling the along-path
    # component is the best solution that preserves the validated geometry.
    if relative_limit ** 2 <= target_cross_sq:
        desired = target_along
    else:
        desired = target_along + math.sqrt(
            max(0.0, relative_limit ** 2 - target_cross_sq))
    return float(np.clip(desired, 0.0, nominal))


def _limit_acceleration_slew(previous, desired, jerk_m_s3, elapsed_s):
    """Rate-limit the acceleration that is actually streamed to PX4."""
    old = np.asarray(previous, float)
    new = np.asarray(desired, float)
    limit = float(jerk_m_s3) * max(0.0, float(elapsed_s))
    if (old.shape != (3,) or new.shape != (3,)
            or not np.all(np.isfinite(np.r_[old, new]))
            or jerk_m_s3 <= 0.0):
        raise ValueError('invalid acceleration slew input')
    return old + np.clip(new - old, -limit, limit)


_PATH_HARD_SPEED_LIMIT_M_S = 12.0
# Use the same 12 m/s ceiling for path tracking, rendezvous, and PX4.
_GPS_PREACQUIRE_SPEED_LIMIT_M_S = 12.0


def _path_speed_envelope(arc_m, path, cruise_speed_m_s, acceleration_m_s2,
                         terminal_speed_m_s=0.0):
    """Cap speed by curvature, then propagate feasible accel/decel speeds."""
    arc = np.asarray(arc_m, float)
    points = np.asarray(path, float)
    acceleration = float(acceleration_m_s2)
    terminal_speed = float(terminal_speed_m_s)
    if not math.isfinite(terminal_speed) or terminal_speed < 0.0:
        raise ValueError('terminal_speed_m_s must be finite and non-negative')
    speeds = np.full(
        len(arc), min(float(cruise_speed_m_s), _PATH_HARD_SPEED_LIMIT_M_S))

    incoming = points[1:-1] - points[:-2]
    outgoing = points[2:] - points[1:-1]
    chord = points[2:] - points[:-2]
    denominator = (np.linalg.norm(incoming, axis=1)
                   * np.linalg.norm(outgoing, axis=1)
                   * np.linalg.norm(chord, axis=1))
    curvature = np.divide(
        2.0 * np.linalg.norm(np.cross(incoming, outgoing), axis=1),
        denominator, out=np.zeros_like(denominator),
        where=denominator > 1.0e-12)
    curved = curvature > 1.0e-12
    interior = speeds[1:-1]
    interior[curved] = np.minimum(
        interior[curved], np.sqrt(acceleration / curvature[curved]))

    speeds[-1] = min(speeds[-1], terminal_speed)
    intervals = np.diff(arc)
    for index in range(1, len(speeds)):
        speeds[index] = min(
            speeds[index],
            math.sqrt(speeds[index - 1] ** 2
                      + 2.0 * acceleration * intervals[index - 1]))
    for index in range(len(speeds) - 2, -1, -1):
        speeds[index] = min(
            speeds[index],
            math.sqrt(speeds[index + 1] ** 2
                      + 2.0 * acceleration * intervals[index]))
    return speeds


def _path_reference_horizon(arc_m, path, progress_m, dt_s, horizon,
                            cruise_speed_m_s, acceleration_m_s2,
                            jerk_m_s3, target_velocity_xy=None,
                            target_range_xy_m=math.inf,
                            relative_brake_start_m=10.0,
                            target_relative_speed_m_s=0.3):
    """Time-parameterise a geometry path with jerk-aware endpoint braking."""
    arc = np.asarray(arc_m, float)
    points = np.asarray(path, float)
    dt = float(dt_s)
    count = int(horizon)
    speed_limit = min(
        float(cruise_speed_m_s), _PATH_HARD_SPEED_LIMIT_M_S)
    accel = float(acceleration_m_s2)
    jerk = float(jerk_m_s3)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) != len(arc)
            or len(points) < 2 or count < 1 or dt <= 0.0
            or speed_limit <= 0.0 or accel <= 0.0 or jerk <= 0.0
            or not np.all(np.isfinite(np.column_stack((arc, points))))
            or not np.all(np.diff(arc) > 0.0)):
        raise ValueError('invalid geometry path or MPC timing')

    distance = float(np.clip(progress_m, arc[0], arc[-1]))
    brake_range = max(0.0, float(target_range_xy_m))
    target_velocity = (None if target_velocity_xy is None else
                       np.asarray(target_velocity_xy, float))
    if (target_velocity is not None
            and (target_velocity.shape != (2,)
                 or not np.all(np.isfinite(target_velocity)))):
        raise ValueError('target_velocity_xy must contain two finite values')
    target_speed = (0.0 if target_velocity is None else min(
        speed_limit, float(np.linalg.norm(target_velocity))))
    speed_envelope = _path_speed_envelope(
        arc, points, speed_limit, acceleration_m_s2,
        terminal_speed_m_s=target_speed)
    query = np.empty(count)
    speeds = np.empty(count)
    for index in range(count):
        remaining = max(0.0, float(arc[-1] - distance))
        segment = int(np.clip(
            np.searchsorted(arc, distance, side='right') - 1,
            0, len(points) - 2))
        distance_from_left = distance - float(arc[segment])
        distance_to_right = float(arc[segment + 1]) - distance
        terminal_cap = (
            _s_curve_stop_speed(remaining, accel, jerk)
            if target_velocity is None else min(
                speed_limit,
                target_speed + _s_curve_stop_speed(
                    remaining, accel, jerk)))
        speed = min(
            speed_limit,
            math.sqrt(speed_envelope[segment] ** 2
                      + 2.0 * accel * distance_from_left),
            math.sqrt(speed_envelope[segment + 1] ** 2
                      + 2.0 * accel * distance_to_right),
            terminal_cap)
        if (target_velocity is not None
                and brake_range < relative_brake_start_m):
            tangent_xy = points[segment + 1, :2] - points[segment, :2]
            speed = _relative_braking_path_speed(
                speed, tangent_xy, target_velocity, brake_range,
                relative_brake_start_m, target_relative_speed_m_s)
        distance = min(float(arc[-1]), distance + speed * dt)
        query[index] = distance
        speeds[index] = speed if distance < arc[-1] - 1.0e-9 else 0.0

    reference_positions = np.column_stack([
        np.interp(query, arc, points[:, axis]) for axis in range(3)])
    segment = np.clip(np.searchsorted(arc, query, side='right') - 1,
                      0, len(points) - 2)
    tangent = points[segment + 1] - points[segment]
    lengths = np.linalg.norm(tangent, axis=1)
    unit = np.divide(tangent, lengths[:, None],
                     out=np.zeros_like(tangent),
                     where=lengths[:, None] > 1.0e-9)
    reference_velocities = unit * speeds[:, None]
    return reference_positions, reference_velocities


def _mpc_prediction_is_safe(
        map_yaml, current, predicted_positions, planning=False):
    """Accept a controller horizon only when every commanded chord is free."""
    predicted = np.asarray(predicted_positions, float)
    start = np.asarray(current, float)
    if (predicted.ndim != 2 or predicted.shape[1] != 3 or start.shape != (3,)
            or not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(start))):
        return False
    chain = np.vstack((start, predicted))
    segment_is_free = (_mission_planning_segment_is_free
                       if planning else _mission_segment_is_free)
    return all(segment_is_free(map_yaml, a, b)
               for a, b in zip(chain[:-1], chain[1:]))


def _mpc_horizontal_speed_is_safe(predicted_velocities, limit_m_s):
    """Accept only horizons whose complete XY speed stays under the cap."""
    velocities = np.asarray(predicted_velocities, float)
    limit = float(limit_m_s)
    if (velocities.ndim != 2 or velocities.shape[1] != 3
            or velocities.shape[0] == 0 or not math.isfinite(limit)
            or limit <= 0.0 or not np.all(np.isfinite(velocities))):
        return False
    speeds = np.linalg.norm(velocities[:, :2], axis=1)
    return bool(np.all(speeds <= limit + 1.0e-6))


def _mpc_stream_speed_is_safe(command_velocity, limit_m_s):
    """Keep every high-rate XY velocity feed-forward under the same cap."""
    command = np.asarray(command_velocity, float)
    limit = float(limit_m_s)
    if (command.shape != (3,)
            or not np.all(np.isfinite(np.r_[command, limit]))
            or limit <= 0.0):
        return False
    commanded_speed = float(np.linalg.norm(command[:2]))
    return bool(commanded_speed <= limit + 1.0e-6)


def _splice_path_from_current(segment_is_free, arc_m, path, current,
                              lookahead_m):
    """Join a completed rolling path from the vehicle's current position."""
    arc = np.asarray(arc_m, float)
    points = np.asarray(path, float)
    current = np.asarray(current, float).copy()
    # The global route is planar. Preserve the measured XY anchor but project
    # ordinary altitude-tracking error onto the route plane before certifying
    # its SFC; the vertical controller independently closes that error.
    current[2] = points[0, 2]
    projection_s, _, _ = _spatial_path_target(
        arc, points, current, 0.0, float(arc[-1]), 1.0)
    # Prefer the longest exact-safe connector.  The old fixed half-lookahead
    # splice created a fresh 3.9--4.3 m/s curvature bottleneck after nearly
    # every rolling swap.  A longer chord aligns with the replacement route;
    # blocked chords fall back without weakening the planning-clearance gate.
    join_s = None
    join = None
    for scale in (2.0, 1.5, 1.0, 0.5):
        candidate_s = min(
            projection_s + scale * lookahead_m, float(arc[-1]))
        candidate = _path_position(arc, points, candidate_s)
        if segment_is_free(current, candidate):
            join_s, join = candidate_s, candidate
            break
    if join_s is None:
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
    """Return the furthest planning-clearance carrot on an accepted path."""
    progress, target, _ = _spatial_path_target(
        arc_m, path, position, progress_m, lookahead_m,
        cross_track_limit_m)
    if _mission_planning_segment_is_free(map_yaml, position, target):
        return progress, target
    for offset in np.linspace(lookahead_m, 0.0, 21)[1:]:
        candidate = _path_position(
            arc_m, path, min(progress + offset, arc_m[-1]))
        if _mission_planning_segment_is_free(map_yaml, position, candidate):
            return progress, candidate
    return progress, None


@lru_cache(maxsize=8)
def _mission_collision_contract(map_yaml, planning=False):
    """Load the immutable per-run map snapshot once for 50 Hz checks."""
    from path_plan.world_model import WorldModel

    document = yaml.safe_load(Path(map_yaml).read_text(encoding='utf-8'))
    mission = document['mission']
    rotation, spawn, origin = _mission_frame_contract(document)
    clearance = float(mission['vehicle_clearance_xy_m'])
    if planning:
        clearance += float(mission.get('bspline_clearance_margin_m', 0.5))
    altitude = float(mission['cruise_altitude_m'])
    lows, highs, bounds_low, bounds_high = _mission_obstacle_geometry(
        document, map_yaml, altitude)
    world = WorldModel.from_boxes(
        lows,
        highs,
        [*bounds_low, altitude],
        [*bounds_high, altitude],
        xy_clearance_m=clearance,
    )
    return rotation, spawn, origin, altitude, world


def _active_path_sfc(map_yaml, path_local_enu):
    """Return a refined local path and free-box cover at planning clearance."""
    from path_plan.sfc import SafeFlightCorridor
    from path_plan.world_model import WorldModel

    path = np.asarray(path_local_enu, float)
    if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2
            or not np.all(np.isfinite(path))):
        raise ValueError('active path must be finite Nx3 local ENU')
    rotation, spawn, origin, altitude, planar_world = (
        _mission_collision_contract(str(map_yaml), True))
    if not np.allclose(path[:, 2], altitude, atol=1.0e-6, rtol=0.0):
        raise ValueError('active path must stay at mission cruise altitude')

    # The 50 Hz segment checker is planar.  SFC boxes need real z thickness
    # for MarkerArray/RViz, so use the same +/-0.5 m optimizer slab without
    # changing the horizontal 1.5 m planning-clearance contract.
    world = WorldModel.from_boxes(
        planar_world.boxes_min, planar_world.boxes_max,
        [*planar_world.bounds_min[:2], altitude - 0.5],
        [*planar_world.bounds_max[:2], altitude + 0.5],
        xy_clearance_m=planar_world.xy_clearance_m)
    map_xy = (path[:, :2] + spawn - origin) @ rotation
    map_path = np.column_stack((map_xy, np.full(len(path), altitude)))
    sfc_started = perf_counter()
    refined_map, corridor = SafeFlightCorridor(world).cover_polyline(map_path)
    sfc_generation_time_ms = 1000.0 * (perf_counter() - sfc_started)
    local_xy = refined_map[:, :2] @ rotation.T + origin - spawn
    refined_local = np.column_stack((
        local_xy, np.full(len(refined_map), altitude)))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(refined_local, axis=0), axis=1))]
    if (len(refined_local) < 2 or not np.all(np.diff(arc) > 0.0)
            or not all(world.box_is_free(lo, hi) for lo, hi in zip(
                corridor.boxes_min, corridor.boxes_max))):
        raise ValueError('active-path SFC certification failed')
    diagnostics = {
        'sfc_boxes_min_map': corridor.boxes_min.copy(),
        'sfc_boxes_max_map': corridor.boxes_max.copy(),
        'sfc_generation_time_ms': sfc_generation_time_ms,
    }
    return arc, refined_local, diagnostics


@lru_cache(maxsize=8)
def _mission_frame_id(map_yaml):
    document = yaml.safe_load(Path(map_yaml).read_text(encoding='utf-8'))
    return str(document['mission']['coordinate_frame'])


def _active_plan_marker_message(
        map_yaml, path_local_enu, diagnostics, plan_seq, stamp):
    """Serialize one atomic map-frame path/SFC snapshot for the live UI."""
    frame_id = _mission_frame_id(str(map_yaml))
    clear = Marker()
    clear.header.frame_id = frame_id
    clear.header.stamp = stamp
    clear.ns = 'active_plan'
    clear.id = 0
    clear.action = Marker.DELETEALL
    message = MarkerArray(markers=[clear])
    if path_local_enu is None:
        return message

    path = np.asarray(path_local_enu, float)
    boxes_min = np.asarray(diagnostics['sfc_boxes_min_map'], float)
    boxes_max = np.asarray(diagnostics['sfc_boxes_max_map'], float)
    if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2
            or not np.all(np.isfinite(path))
            or boxes_min.ndim != 2 or boxes_min.shape[1] != 3
            or boxes_max.shape != boxes_min.shape or not len(boxes_min)
            or not np.all(np.isfinite(np.r_[boxes_min, boxes_max]))
            or np.any(boxes_max <= boxes_min)):
        raise ValueError('active plan requires a finite path and positive SFC')
    sequence = int(plan_seq)
    if sequence <= 0 or sequence > 2_000_000_000:
        raise ValueError('active plan sequence must be a positive int32')
    rotation, spawn, origin, altitude, _ = _mission_collision_contract(
        str(map_yaml), True)
    path_map = np.column_stack((
        (path[:, :2] + spawn - origin) @ rotation,
        np.full(len(path), altitude)))

    line = Marker()
    line.header.frame_id = frame_id
    line.header.stamp = stamp
    line.ns = 'active_path'
    line.id = sequence
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.pose.orientation.w = 1.0
    line.scale.x = 0.12
    line.color.r, line.color.g, line.color.b, line.color.a = (
        0.18, 0.62, 0.27, 1.0)
    line.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                   for p in path_map]
    message.markers.append(line)

    for index, (low, high) in enumerate(zip(boxes_min, boxes_max)):
        box = Marker()
        box.header.frame_id = frame_id
        box.header.stamp = stamp
        box.ns = 'active_sfc'
        box.id = index
        box.type = Marker.CUBE
        box.action = Marker.ADD
        centre = 0.5 * (low + high)
        extent = high - low
        box.pose.position.x, box.pose.position.y, box.pose.position.z = map(
            float, centre)
        box.pose.orientation.w = 1.0
        box.scale.x, box.scale.y, box.scale.z = map(float, extent)
        box.color.r, box.color.g, box.color.b, box.color.a = (
            0.13, 0.55, 0.90, 0.10)
        message.markers.append(box)
    return message


def _mission_segment_is_free(map_yaml, start_local_enu, goal_local_enu):
    """Check one cruise-altitude local segment against the same YAML AABBs."""
    start = np.asarray(start_local_enu, float)[:2]
    goal = np.asarray(goal_local_enu, float)[:2]
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(goal))):
        return False
    if not str(map_yaml).strip():
        return True
    rotation, spawn, origin, altitude, world = (
        _mission_collision_contract(str(map_yaml)))
    map_xy = (np.vstack((start, goal)) + spawn - origin) @ rotation
    return world.segment_is_free(
        [*map_xy[0], altitude], [*map_xy[1], altitude])


def _forward_endpoint_eta_s(position_xy, velocity_xy, endpoints_xy,
                            endpoint_tolerance_m, minimum_speed_m_s,
                            route_match_tolerance_m=None):
    """Seconds remaining on the active cyclic waypoint segment, or zero."""
    position = np.asarray(position_xy, float)
    velocity = np.asarray(velocity_xy, float)
    endpoints = np.asarray(endpoints_xy, float)
    if (position.shape != (2,) or velocity.shape != (2,)
            or endpoints.ndim != 2 or endpoints.shape[1] != 2
            or len(endpoints) < 2
            or not np.all(np.isfinite(np.r_[position, velocity,
                                             endpoints.ravel()]))):
        return 0.0
    speed = float(np.linalg.norm(velocity))
    if speed < float(minimum_speed_m_s):
        return 0.0
    starts = np.roll(endpoints, 1, axis=0)
    segments = endpoints - starts
    lengths = np.linalg.norm(segments, axis=1)
    directions = np.divide(
        segments, lengths[:, None], out=np.zeros_like(segments),
        where=lengths[:, None] > 1.0e-12)
    offsets = position - starts
    along = np.sum(offsets * directions, axis=1)
    cross_track = np.abs(
        offsets[:, 0] * directions[:, 1]
        - offsets[:, 1] * directions[:, 0])
    alignment = directions @ (velocity / speed)
    remaining = lengths - along
    tolerance = float(endpoint_tolerance_m)
    route_tolerance = (tolerance if route_match_tolerance_m is None
                       else float(route_match_tolerance_m))
    valid = ((lengths > 1.0e-12) & (alignment > 0.0)
             & (cross_track <= route_tolerance) & (along >= -tolerance)
             & (remaining > tolerance))
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return 0.0
    active = min(
        candidates, key=lambda index: (cross_track[index], -alignment[index]))
    return float(remaining[active] / speed)


def _forward_cyclic_route_position(
        position_xy, velocity_xy, waypoints_xy, lead_s,
        endpoint_tolerance_m, minimum_speed_m_s,
        route_match_tolerance_m=None):
    """Project a moving target through turns on a closed waypoint route."""
    position = np.asarray(position_xy, float)
    velocity = np.asarray(velocity_xy, float)
    waypoints = np.asarray(waypoints_xy, float)
    lead = float(lead_s)
    if (position.shape != (2,) or velocity.shape != (2,)
            or waypoints.ndim != 2 or waypoints.shape[1] != 2
            or len(waypoints) < 2 or not math.isfinite(lead) or lead < 0.0
            or not np.all(np.isfinite(np.r_[position, velocity,
                                             waypoints.ravel()]))):
        return None
    speed = float(np.linalg.norm(velocity))
    if speed < float(minimum_speed_m_s):
        return None
    starts = np.roll(waypoints, 1, axis=0)
    segments = waypoints - starts
    lengths = np.linalg.norm(segments, axis=1)
    route_length = float(np.sum(lengths))
    if route_length <= 1.0e-12:
        return None
    directions = np.divide(
        segments, lengths[:, None], out=np.zeros_like(segments),
        where=lengths[:, None] > 1.0e-12)
    offsets = position - starts
    along = np.sum(offsets * directions, axis=1)
    cross_track = np.abs(
        offsets[:, 0] * directions[:, 1]
        - offsets[:, 1] * directions[:, 0])
    alignment = directions @ (velocity / speed)
    tolerance = float(endpoint_tolerance_m)
    route_tolerance = (tolerance if route_match_tolerance_m is None
                       else float(route_match_tolerance_m))
    remaining = lengths - along
    valid = ((lengths > 1.0e-12) & (alignment > 0.0)
             & (cross_track <= route_tolerance) & (along >= -tolerance)
             & (remaining > tolerance))
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return None
    active = min(
        candidates, key=lambda index: (cross_track[index], -alignment[index]))
    distance = (max(0.0, float(along[active]))
                + (speed * lead) % route_length)
    for _ in range(len(waypoints) + 1):
        if distance <= lengths[active] + 1.0e-9:
            return starts[active] + min(distance, lengths[active]) * (
                directions[active])
        distance -= lengths[active]
        active = (active + 1) % len(waypoints)
    return None


def _landing_staging_point(waypoints_xy, speed_m_s, required_runway_s):
    """Choose a point on the longest cyclic leg with runway left to land."""
    waypoints = np.asarray(waypoints_xy, float)
    speed = float(speed_m_s)
    required = float(required_runway_s)
    if (waypoints.ndim != 2 or waypoints.shape[1] != 2
            or len(waypoints) < 2 or speed <= 0.0 or required <= 0.0
            or not np.all(np.isfinite(np.r_[waypoints.ravel(),
                                             speed, required]))):
        return None
    ends = np.roll(waypoints, -1, axis=0)
    legs = ends - waypoints
    lengths = np.linalg.norm(legs, axis=1)
    index = int(np.argmax(lengths))
    runway = speed * required
    if lengths[index] < runway:
        return None
    direction = legs[index] / lengths[index]
    return ends[index] - runway * direction


def _mission_planning_segment_is_free(
        map_yaml, start_local_enu, goal_local_enu):
    """Check one local segment against clearance plus the planning reserve."""
    start = np.asarray(start_local_enu, float)[:2]
    goal = np.asarray(goal_local_enu, float)[:2]
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(goal))):
        return False
    if not str(map_yaml).strip():
        return True
    rotation, spawn, origin, altitude, world = (
        _mission_collision_contract(str(map_yaml), True))
    map_xy = (np.vstack((start, goal)) + spawn - origin) @ rotation
    return world.segment_is_free(
        [*map_xy[0], altitude], [*map_xy[1], altitude])


def _prewarm_mission_collision_contracts(map_yaml):
    """Populate the exact runtime and planning cache keys before Offboard."""
    _mission_collision_contract(str(map_yaml))
    _mission_collision_contract(str(map_yaml), True)


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')
        p = self.declare_parameter
        self.control_rate_hz = require_positive(
            'control_rate_hz', p('control_rate_hz', 50.0).value)
        self.mpc_rate_hz = require_positive(
            'mpc_rate_hz', p('mpc_rate_hz', 10.0).value)
        self.dt, self.mpc_dt, self._mpc_solve_every = derive_control_timing(
            self.control_rate_hz, self.mpc_rate_hz)
        self._mpc_horizon = int(p('mpc_horizon', 20).value)
        if self._mpc_horizon < 2:
            raise ValueError('mpc_horizon must be >= 2')
        self._path_mpc_speed = require_positive(
            'path_mpc_speed_m_s', p('path_mpc_speed_m_s', 3.0).value)
        self._path_mpc_v_max = require_positive(
            'path_mpc_v_max_m_s', p('path_mpc_v_max_m_s', 5.0).value)
        self._path_mpc_a_max = require_positive(
            'path_mpc_a_max_m_s2', p('path_mpc_a_max_m_s2', 3.0).value)
        self._path_mpc_jerk = require_positive(
            'path_mpc_jerk_m_s3', p('path_mpc_jerk_m_s3', 2.0).value)
        self._path_speed_profile_a_max = require_positive(
            'path_speed_profile_a_max_m_s2',
            p('path_speed_profile_a_max_m_s2',
              self._path_mpc_a_max).value)
        self._mission_path_speed_profile_a_max = require_positive(
            'mission_path_speed_profile_a_max_m_s2',
            p('mission_path_speed_profile_a_max_m_s2',
              self._path_speed_profile_a_max).value)
        self._path_terminal_goto_enabled = bool(
            p('path_terminal_goto_enabled', False).value)
        self._path_relative_brake_distance = require_positive(
            'path_mpc_relative_brake_distance_m',
            p('path_mpc_relative_brake_distance_m', 10.0).value)
        self._path_target_relative_speed = require_positive(
            'path_mpc_target_relative_speed_m_s',
            p('path_mpc_target_relative_speed_m_s', 0.3).value)
        self._landing_mpc_v_max = require_positive(
            'landing_mpc_v_max_m_s', p('landing_mpc_v_max_m_s', 3.5).value)
        self._landing_mpc_a_max = require_positive(
            'landing_mpc_a_max_m_s2', p('landing_mpc_a_max_m_s2', 1.0).value)
        self._landing_mpc_vz_max = require_positive(
            'landing_mpc_vz_max_m_s', p('landing_mpc_vz_max_m_s', 0.6).value)
        self._landing_mpc_jerk = require_positive(
            'landing_mpc_jerk_m_s3', p('landing_mpc_jerk_m_s3', 2.0).value)
        self._landing_mpc_cone = require_positive(
            'landing_mpc_cone_k', p('landing_mpc_cone_k', 2.0).value)
        self._landing_mpc_handoff_height = require_positive(
            'landing_mpc_precland_height_m',
            p('landing_mpc_precland_height_m', 1.5).value)
        self._landing_mpc_bias_ok = require_positive(
            'landing_mpc_bias_converged_m',
            p('landing_mpc_bias_converged_m', 0.3).value)
        self._landing_gps_preacquire_range = require_nonnegative(
            'landing_gps_preacquire_range_m',
            p('landing_gps_preacquire_range_m', 0.0).value)
        self._landing_target_min_speed = require_nonnegative(
            'landing_target_min_speed_m_s',
            p('landing_target_min_speed_m_s', 0.0).value)
        self._precland_commit_height = require_positive(
            'precland_blind_commit_height_m',
            p('precland_blind_commit_height_m', 0.65).value)
        self._precland_commit_grace = require_positive(
            'precland_blind_commit_grace_s',
            p('precland_blind_commit_grace_s', 8.0).value)
        self.takeoff_alt = require_finite(
            'takeoff_alt', p('takeoff_alt', 6.0).value)
        # This is a route-completion gate, not the TrackingMPC reference speed.
        self.settle_v_tol = require_positive(
            'settle_vel_tol_m_s',
            p('settle_vel_tol_m_s', 0.2).value)
        # ArUco remains a measurement only. It slowly corrects the continuous
        # trailer cue before that target is handed to PX4 PRECLAND.
        self.bias_tau = require_positive(
            'bias_tau_s', p('bias_tau_s', 1.5).value)
        self.bias_rate = require_positive(
            'bias_rate_max_m_s',
            p('bias_rate_max_m_s', 0.3).value)
        self.bias_max = require_positive(
            'bias_max_m', p('bias_max_m', 5.0).value)
        self.vis_fresh = require_positive(
            'vision_fresh_s', p('vision_fresh_s', 0.5).value)
        self._entry_fix_window = require_positive(
            'entry_fix_window_s',
            p('entry_fix_window_s', DEFAULT_ENTRY_FIX_WINDOW_S).value)
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
        self.marker_detection_topic = require_nonempty(
            'marker_detection_topic',
            p('marker_detection_topic', '/aruco/detected').value)

        self.get_logger().info(
            f'control timing: {self.control_rate_hz:g} Hz setpoints, '
            f'{self.mpc_rate_hz:g} Hz MPC solves')

        self._mission_arc_m = None
        self._mission_path = None
        self._mission_progress_m = 0.0
        self._path_terminal_goto = None
        self._active_plan_seq = 0
        self._active_sfc_diagnostics = None
        self._mission_lookahead = 6.0
        self._mission_cross_track = 0.25
        self._mission_sample_spacing = 0.1
        self._mission_hover_duration = 0.0
        self._return_replan_min_period = 2.0
        self._return_plan_lead_s = self._return_replan_min_period
        self._landing_alignment_margin_s = 0.0
        self._precland_target_timeout = 0.5
        self._landing_xy_tol = 0.5
        self._landing_v_tol = 0.3
        self._terminal_waypoints_local = None
        self._terminal_route_cyclic = False
        self._terminal_waypoint_tolerance = 0.2
        self._terminal_route_match_tolerance = 0.2
        self._terminal_min_cruise_speed = 0.0
        self._precland_runway_required_s = 0.0
        self._return_staging_goal = None
        self._return_staging_enabled = True
        self._return_staging = False
        self._return_staging_arrived = False
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
            self._mission_hover_duration = require_nonnegative(
                'mission.hover_duration_s',
                mission_config.get('hover_duration_s', 0.0))
            self._return_replan_min_period = require_positive(
                'mission.return_replan_min_period_s',
                mission_config.get('return_replan_min_period_s', 2.0))
            self._return_staging_enabled = mission_config.get(
                'return_staging_enabled', True)
            if not isinstance(self._return_staging_enabled, bool):
                raise ValueError(
                    'mission.return_staging_enabled must be boolean')
            self._return_plan_lead_s = self._return_replan_min_period
            self._landing_alignment_margin_s = require_nonnegative(
                'mission.landing_alignment_margin_s',
                mission_config.get('landing_alignment_margin_s', 0.0))
            self._precland_target_timeout = require_positive(
                'px4_vehicle.sitl_parameter_overrides.PLD_BTOUT',
                mission_document.get('px4_vehicle', {}).get(
                    'sitl_parameter_overrides', {}).get('PLD_BTOUT', 0.5))
            px4_config = mission_document.get(
                'px4_vehicle', {}).get('sitl_parameter_overrides', {})
            self._landing_xy_tol = require_positive(
                'px4_vehicle.sitl_parameter_overrides.PLD_HACC_RAD',
                px4_config.get('PLD_HACC_RAD', 0.5))
            self._landing_v_tol = require_positive(
                'px4_vehicle.sitl_parameter_overrides.PLD_VEL_THR',
                px4_config.get('PLD_VEL_THR', 0.3))
            trailer_config = mission_document.get('trailer', {})
            route_type = trailer_config.get('route_type')
            route_key = {
                'linear_shuttle': 'shuttle_endpoints_enu_m',
                'waypoints': 'waypoints_enu_m',
            }.get(route_type)
            if route_key is not None:
                waypoints = np.asarray(trailer_config[route_key], float)
                spawn_pose = mission_document['spawn'][
                    'gazebo_spawn_pose_enu']
                spawn_xy = np.asarray(
                    [spawn_pose['x'], spawn_pose['y']], float)
                valid_shape = (
                    waypoints.ndim == 2 and waypoints.shape[1] == 2
                    and len(waypoints) >= 2
                    and (route_type != 'linear_shuttle'
                         or len(waypoints) == 2))
                if not valid_shape or not np.all(np.isfinite(waypoints)):
                    raise ValueError(
                        f'trailer.{route_key} must contain finite XY points')
                self._terminal_waypoints_local = waypoints - spawn_xy
                self._terminal_route_cyclic = bool(
                    route_type == 'waypoints'
                    and trailer_config.get('patrol_mode') == 'repeat')
                self._terminal_waypoint_tolerance = require_positive(
                    'trailer.waypoint_tolerance_m',
                    trailer_config.get('waypoint_tolerance_m', 0.2))
                self._terminal_route_match_tolerance = require_positive(
                    'mission.terminal_route_match_tolerance_m',
                    mission_config.get(
                        'terminal_route_match_tolerance_m',
                        self._terminal_waypoint_tolerance))
                cruise_speed = require_positive(
                    'trailer.cruise_speed_m_s',
                    trailer_config['cruise_speed_m_s'])
                turn_tolerance = self._landing_v_tol
                if (route_type == 'waypoints'
                        and self._return_staging_enabled):
                    turn_tolerance = require_nonnegative(
                        'trailer.turn_speed_tolerance_m_s',
                        trailer_config.get(
                            'turn_speed_tolerance_m_s', turn_tolerance))
                self._terminal_min_cruise_speed = max(
                    1.0e-3, cruise_speed - turn_tolerance)
                disarm_delay = require_positive(
                    'px4_vehicle.sitl_parameter_overrides.COM_DISARM_LAND',
                    px4_config.get('COM_DISARM_LAND', 2.0))
                self._precland_runway_required_s = (
                    self._precland_commit_grace
                    + self._precland_target_timeout + disarm_delay)
                if (route_type == 'waypoints'
                        and self._return_staging_enabled):
                    speed_change_s = (
                        2.0 * _s_curve_stop_distance(
                            cruise_speed, self._path_mpc_a_max,
                            self._landing_mpc_jerk) / cruise_speed)
                    staging_runway_s = (
                        self._precland_runway_required_s
                        + max(0.0, self.takeoff_alt
                              - self._precland_commit_height)
                        / self._landing_mpc_vz_max
                        + speed_change_s
                        # KF entry qualification precedes descent and is not
                        # PX4's PLD_BTOUT target-loss allowance above.
                        + self._entry_fix_window
                        + self._landing_alignment_margin_s)
                    staging_map = _landing_staging_point(
                        waypoints, cruise_speed, staging_runway_s)
                    if staging_map is not None:
                        self._return_staging_goal = np.r_[
                            staging_map - spawn_xy, self.takeoff_alt]
            # Build both immutable worlds before timers start.  A cold city
            # load takes about one second and must never block 50 Hz Offboard.
            _prewarm_mission_collision_contracts(self.mission_map_yaml)
            # A* is CPU-bound Python. A thread delayed the 50 Hz Offboard
            # heartbeat by hundreds of milliseconds in measurement; a spawned
            # worker process keeps ROS/DDS state out of the child and the
            # control timer responsive.
            self._planner_pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_planner_worker_init)
            self.get_logger().info(
                'CJU A* -> geometry-only B-spline -> TrackingMPC ready')

        # Keep the two wang controllers separate.  They solve different
        # problems and never publish directly; this node remains the one PX4
        # setpoint authority.
        self._path_mpc = TrackingMPC(
            dt_s=self.mpc_dt, horizon=self._mpc_horizon,
            v_max=self._path_mpc_v_max, a_max=self._path_mpc_a_max,
            j_max=self._path_mpc_jerk)
        self._path_reference = HorizonReference(lead_s=self.mpc_dt)
        self._path_solve_t = None
        self._path_last_solve_t = None
        self._path_failure_hold = None
        self._landing_mpc = LandingMPC(
            dt_s=self.mpc_dt, horizon=self._mpc_horizon,
            w_vxy=20.0,
            v_max=self._landing_mpc_v_max,
            a_max=self._landing_mpc_a_max,
            vz_max=self._landing_mpc_vz_max,
            cone_k=self._landing_mpc_cone,
            z_ref=self._landing_mpc_handoff_height,
            j_max=self._landing_mpc_jerk)
        # Preserve one 50 Hz state-age tick in the position preview.  Removing
        # the full 0.1 s lead put the moving-deck command behind PX4 and
        # let its position P term cancel the forward velocity feed-forward.
        self._landing_reference = HorizonReference(
            lead_s=self.mpc_dt, state_age_s=self.dt)
        self._landing_solve_t = None
        self._landing_last_solve_t = None
        self._landing_failure_hold = None
        self._gps_preacquire_active = False
        self._landing_hold_z = None
        self._last_mpc_setpoint = None
        self._last_sent_acceleration = np.zeros(3)
        self._last_sent_acceleration_t = None
        self._mpc_solve_count = 0
        self._mpc_solve_total_s = 0.0
        self._mpc_solve_max_s = 0.0
        self._marker_metric_frames = 0
        self._marker_metric_hits = 0
        self._touchdown_metric_recorded = False
        self._touchdown_metric_candidate = None
        self._ground_contact_seen = False
        self._landing_contact_confirmed = False
        self._landing_error_3d_m = math.nan
        self._landing_xy_error_m = math.nan
        self._touchdown_relative_speed_3d_m_s = math.nan
        self._touchdown_relative_vertical_speed_m_s = math.nan
        self._experiment_metrics_logged = False
        self._precland_attempts = 0
        self._precland_recoveries = 0
        self._landing_descend_recoveries = 0

        self.state = 'PRECHECK'
        self._hover_since = None
        self._takeoff_requested = False
        self._hold_pos = np.zeros(3)
        self._launch_ground = None
        self._plan_start = None
        self._plan_goal = None
        self._last_safe_goto = None
        self._precland_goto = None
        self._precland_commit_until = None
        self._landing_recovery_since = None
        self._last_return_plan_t = None
        self._return_intercept_deadline_t = None
        self._return_intercept_lock_active = False
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
        self.vis_v = None
        self.vis_valid = False
        self.vis_entry_valid = False
        self._t_vis = None
        self._bias = np.zeros(3)             # learned (vision - cue) offset
        self._landing_bias_locked = False    # latch calibration before descent
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
        self.landing_diagnostics_pub = self.create_publisher(
            String, '/mission/landing_diagnostics', 10)
        self.planned_path_pub = self.create_publisher(
            PoseArray, '/mission/planned_path', _planned_path_qos())
        self.sfc_pub = self.create_publisher(
            Float32MultiArray, '/mission/sfc_boxes_map', _planned_path_qos())
        self.active_plan_pub = self.create_publisher(
            MarkerArray, '/mission/active_plan_markers', _planned_path_qos())
        self.vehicle_position_pub = self.create_publisher(
            PointStamped, '/mission/vehicle_position', 10)
        self.create_subscription(String, self.mission_command_topic,
                                 self._on_command, 10)

        self.create_subscription(PointStamped, '/marker/cue', self._on_cue, 10)
        self.create_subscription(Vector3Stamped, '/marker/cue_velocity',
                                 self._on_cue_v, 10)
        self.create_subscription(PointStamped, '/marker/position', self._on_vis, 10)
        self.create_subscription(Vector3Stamped, '/marker/velocity',
                                 self._on_vis_v, 10)
        self.create_subscription(Bool, '/marker/valid', self._on_valid, 10)
        self.create_subscription(Bool, '/marker/entry_valid',
                                 self._on_entry_valid, 10)
        self.create_subscription(Bool, self.marker_detection_topic,
                                 self._on_marker_detection,
                                 qos_profile_sensor_data)
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
            'mission_manager: B-spline TrackingMPC and ArUco LandingMPC are '
            'separate; PX4 PRECLAND retains contact and auto-disarm authority')

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
                        'RETURN_PLAN', 'RETURN', 'LANDING_ACQUIRE',
                        'LANDING_DESCEND', 'PRECLAND', 'DONE'),
            'mission': ('MISSION_PLAN', 'MISSION', 'HOVER', 'RETURN_PLAN',
                        'RETURN', 'LANDING_ACQUIRE', 'LANDING_DESCEND',
                        'PRECLAND', 'DONE'),
            'land': ('RETURN_PLAN', 'RETURN', 'LANDING_ACQUIRE',
                     'LANDING_DESCEND', 'PRECLAND', 'DONE'),
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

        if command == 'land':
            dwell = float(getattr(self, '_mission_hover_duration', 0.0))
            hover_since = getattr(self, '_hover_since', None)
            if (dwell > 0.0
                    and (hover_since is None
                         or self._now() - hover_since < dwell)):
                # The launcher republishes at 5 Hz, so the same command is
                # accepted on the first sample after the configured dwell.
                return

        if command == 'takeoff':
            self._takeoff_requested = True
        elif command == 'mission':
            if self._planner_pool is None or self.p_d is None:
                self.get_logger().error('dynamic mission planner is unavailable')
                return
            self._start_global_plan(None, return_route=False)
        else:
            staging_goal = getattr(self, '_return_staging_goal', None)
            if not getattr(self, '_return_staging_enabled', True):
                staging_goal = None
            if staging_goal is not None:
                self._return_staging = True
                self._return_staging_arrived = False
                self._start_global_plan(staging_goal, return_route=True)
            elif not self._cue_fresh():
                self.get_logger().warn(
                    'landing rejected: trailer cue unavailable or stale')
                return
            elif self._landing_mpc_entry_ready():
                self._enter_landing_mpc()
            else:
                self._return_staging = False
                self._return_staging_arrived = False
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

    def _on_vis_v(self, m):
        velocity = np.array([m.vector.x, m.vector.y, m.vector.z])
        if np.all(np.isfinite(velocity)):
            self.vis_v = velocity

    def _on_valid(self, m):
        self.vis_valid = bool(m.data)

    def _on_entry_valid(self, m):
        self.vis_entry_valid = bool(m.data)

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
        if self.state not in (
                'LANDING_ACQUIRE', 'LANDING_DESCEND', 'PRECLAND'):
            return
        if bool(getattr(m, 'ground_contact', False)):
            self._ground_contact_seen = True
        if getattr(self, '_touchdown_metric_recorded', False):
            return
        if not bool(getattr(m, 'ground_contact', False)):
            self._touchdown_metric_candidate = None
            return

        # Keep the first sample of this uninterrupted contact, but only commit
        # it once PX4's native land detector confirms the stronger second
        # stage.  A brief bounce therefore cannot become the touchdown metric.
        if getattr(self, '_touchdown_metric_candidate', None) is None:
            self._touchdown_metric_candidate = (
                MissionManagerNode._touchdown_snapshot(self))
        if not (bool(getattr(m, 'maybe_landed', False)) or self.landed):
            return
        self._landing_contact_confirmed = True
        candidate = self._touchdown_metric_candidate
        if candidate is None:
            return
        (self._landing_error_3d_m,
         self._landing_xy_error_m,
         self._touchdown_relative_speed_3d_m_s,
         self._touchdown_relative_vertical_speed_m_s) = candidate
        self._touchdown_metric_recorded = True

    def _touchdown_snapshot(self):
        cue = getattr(self, 'cue', None)
        position = getattr(self, 'p_d', None)
        if cue is None or position is None:
            return None
        target = np.asarray(cue, float) + np.asarray(
            getattr(self, '_bias', np.zeros(3)), float)
        velocity = np.asarray(getattr(self, 'v_d', np.zeros(3)), float)
        target_velocity = np.asarray(
            getattr(self, 'cue_v', np.zeros(3)), float)
        relative_position = np.asarray(position, float) - target
        relative_velocity = velocity - target_velocity
        if not np.all(np.isfinite(np.r_[relative_position, relative_velocity])):
            return None
        return (
            float(np.linalg.norm(relative_position)),
            float(np.linalg.norm(relative_position[:2])),
            float(np.linalg.norm(relative_velocity)),
            abs(float(relative_velocity[2])),
        )

    def _on_marker_detection(self, message):
        """Count stable ArUco results only during precision landing."""
        if (self.state not in (
                'LANDING_ACQUIRE', 'LANDING_DESCEND', 'PRECLAND')
                or getattr(self, '_touchdown_metric_recorded', False)):
            return
        self._marker_metric_frames += 1
        self._marker_metric_hits += int(bool(message.data))

    def _record_mpc_solve(self, elapsed_s):
        elapsed_s = float(elapsed_s)
        if not np.isfinite(elapsed_s) or elapsed_s < 0.0:
            return
        self._mpc_solve_count = getattr(self, '_mpc_solve_count', 0) + 1
        self._mpc_solve_total_s = (
            getattr(self, '_mpc_solve_total_s', 0.0) + elapsed_s)
        self._mpc_solve_max_s = max(
            getattr(self, '_mpc_solve_max_s', 0.0), elapsed_s)

    def _log_experiment_metrics(self, final=True):
        if final and getattr(self, '_experiment_metrics_logged', False):
            return
        if final:
            self._experiment_metrics_logged = True
        values = {
            'marker_hits': getattr(self, '_marker_metric_hits', 0),
            'marker_frames': getattr(self, '_marker_metric_frames', 0),
            'landing_error_3d_m': getattr(
                self, '_landing_error_3d_m', math.nan),
            'landing_xy_error_m': getattr(
                self, '_landing_xy_error_m', math.nan),
            'touchdown_relative_speed_3d_m_s': getattr(
                self, '_touchdown_relative_speed_3d_m_s', math.nan),
            'touchdown_relative_vertical_speed_m_s': getattr(
                self, '_touchdown_relative_vertical_speed_m_s', math.nan),
            'mpc_count': getattr(self, '_mpc_solve_count', 0),
            'mpc_total_ms': 1000.0 * getattr(
                self, '_mpc_solve_total_s', 0.0),
            'mpc_max_ms': 1000.0 * getattr(
                self, '_mpc_solve_max_s', 0.0),
            'precland_attempts': getattr(self, '_precland_attempts', 0),
            'precland_recoveries': getattr(self, '_precland_recoveries', 0),
            'landing_descend_recoveries': getattr(
                self, '_landing_descend_recoveries', 0),
        }
        payload = ' '.join(
            f'{key}={value:.9g}' if isinstance(value, float)
            else f'{key}={value}'
            for key, value in values.items())
        self.get_logger().info(f'EXPERIMENT_METRICS {payload}')

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

    def _cue_at_now(self):
        """Project the latest paired GPS sample across its bounded ZOH age."""
        cue = np.asarray(self.cue, float).copy()
        velocity = np.asarray(
            getattr(self, 'cue_v', np.zeros_like(cue)), float)
        source_time = getattr(self, '_t_cue_source', None)
        if (source_time is None or not np.all(np.isfinite(cue))
                or not np.all(np.isfinite(velocity))):
            return cue
        max_age = min(
            float(getattr(self, 'cue_timeout_s', 0.5)),
            float(getattr(self, '_precland_target_timeout', 0.5)))
        age = float(np.clip(self._now() - source_time, 0.0, max_age))
        return cue + velocity * age

    def _landing_target_fresh(self):
        """Use the source timestamp so PX4 can enforce PLD_BTOUT honestly."""
        if (not self._cue_fresh() or self._t_cue_source is None
                or self._t_cue_v_source is None):
            return False
        now = self._now()
        # Gazebo pose and /clock callbacks can be delivered in either order.
        # Accept at most one control tick of negative age so a newly received
        # 50 Hz cue cannot inject a one-tick hold; larger future stamps remain
        # fail-closed just like stale samples.
        future_tolerance = self.dt + 1.0e-6
        return (
            -future_tolerance <= now - self._t_cue_source
            <= self._precland_target_timeout
            and -future_tolerance <= now - self._t_cue_v_source
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
            acceleration = np.asarray(acc, float)
            a = enu_to_ned(acceleration)
            s.acceleration = [float(a[0]), float(a[1]), float(a[2])]
            remembered_acceleration = acceleration.copy()
        else:
            remembered_acceleration = np.zeros(3)
        s.yaw = float('nan')
        s.yawspeed = float('nan')
        self.sp_pub.publish(s)
        self._last_sent_acceleration = remembered_acceleration
        self._last_sent_acceleration_t = s.timestamp * 1.0e-6

    def _send_goto(self, pos):
        """Send geometry only; PX4 creates the velocity/acceleration profile."""
        p_ned = enu_to_ned(np.asarray(pos, float))
        message = GotoSetpoint()
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        message.position = [float(value) for value in p_ned]
        message.flag_control_heading = False
        message.heading = float('nan')
        message.flag_set_max_horizontal_speed = True
        message.max_horizontal_speed = min(
            float(getattr(
                self, '_path_mpc_v_max', _PATH_HARD_SPEED_LIMIT_M_S)),
            _PATH_HARD_SPEED_LIMIT_M_S)
        message.flag_set_max_vertical_speed = False
        message.max_vertical_speed = float('nan')
        message.flag_set_max_heading_rate = False
        message.max_heading_rate = float('nan')
        self.goto_pub.publish(message)
        self._last_sent_acceleration = np.zeros(3)
        self._last_sent_acceleration_t = message.timestamp * 1.0e-6

    def _follow_path(self):
        """Track the accepted B-spline with wang's dedicated TrackingMPC."""
        static_terminal = (
            getattr(self, '_path_terminal_goto_enabled', False)
            and (getattr(self, 'state', '') in ('MISSION', 'HOVER')
                 or (getattr(self, 'state', '') == 'RETURN'
                     and getattr(self, '_return_staging', False))))
        terminal = getattr(self, '_path_terminal_goto', None)
        if static_terminal and terminal is not None:
            progress, _, _ = _spatial_path_target(
                self._mission_arc_m, self._mission_path, self.p_d,
                self._mission_progress_m, self._mission_lookahead,
                self._mission_cross_track)
            self._mission_progress_m = progress
            if not _mission_segment_is_free(
                    self.mission_map_yaml, self.p_d, terminal):
                return False
            self._last_safe_goto = np.asarray(terminal, float).copy()
            self._path_failure_hold = None
            self._send_goto(terminal)
            return True

        safety_planning = True
        progress, safe_target = _safe_spatial_path_target(
            self.mission_map_yaml, self._mission_arc_m,
            self._mission_path, self.p_d,
            self._mission_progress_m, self._mission_lookahead,
            self._mission_cross_track)
        if safe_target is None:
            recovery_progress, _, _ = _spatial_path_target(
                self._mission_arc_m, self._mission_path, self.p_d,
                self._mission_progress_m, self._mission_lookahead,
                self._mission_cross_track)
            recovery = _path_position(
                self._mission_arc_m, self._mission_path,
                min(float(self._mission_arc_m[-1]),
                    recovery_progress + 0.5 * self._mission_lookahead))
            direction = recovery[:2] - self.p_d[:2]
            velocity = np.asarray(
                getattr(self, 'v_d', np.zeros(3)), float)
            forward = (float(np.linalg.norm(velocity[:2])) < 0.5
                       or float(direction @ velocity[:2]) >= 0.0)
            if (forward and _mission_segment_is_free(
                    self.mission_map_yaml, self.p_d, recovery)):
                progress, safe_target = recovery_progress, recovery
                safety_planning = False
                self.get_logger().warn(
                    'outside planning reserve; MPC rejoin on hard-safe path',
                    throttle_duration_sec=2.0)
            else:
                return False
        self._mission_progress_m = progress
        self._last_safe_goto = np.asarray(safe_target, float).copy()

        if static_terminal:
            endpoint = np.asarray(self._mission_path[-1], float)
            remaining = max(
                0.0, float(self._mission_arc_m[-1] - progress))
            speed_xy = float(np.linalg.norm(self.v_d[:2]))
            stop_distance = _s_curve_stop_distance(
                speed_xy,
                getattr(self, '_path_speed_profile_a_max',
                        getattr(self, '_path_mpc_a_max', 3.0)),
                getattr(getattr(self, '_path_mpc', None), 'j_max', 2.0))
            direction = endpoint[:2] - self.p_d[:2]
            forward = (speed_xy < 0.5
                       or float(direction @ self.v_d[:2]) >= 0.0)
            at_end = remaining <= getattr(
                self, 'mission_tolerance', 0.7)
            handoff_speed = max(
                getattr(self, 'settle_v_tol', 0.2),
                0.5 * getattr(self, '_path_mpc_speed', speed_xy))
            if (remaining <= stop_distance + 0.5 * self._mission_lookahead
                    and speed_xy <= handoff_speed
                    and (forward or at_end)
                    and _mission_planning_segment_is_free(
                        self.mission_map_yaml, self.p_d, endpoint)):
                self._path_terminal_goto = endpoint.copy()
                if getattr(self, '_path_mpc', None) is not None:
                    self._path_mpc.reset()
                    self._path_reference.reset()
                    self._path_solve_t = None
                    self._path_last_solve_t = None
                self._path_failure_hold = None
                self._last_safe_goto = endpoint.copy()
                self.get_logger().info(
                    'terminal braking handoff to fixed PX4 Goto: '
                    f'remaining={remaining:.2f}m, vxy={speed_xy:.2f}m/s, '
                    f'stop={stop_distance:.2f}m, '
                    f'limit={handoff_speed:.2f}m/s')
                self._send_goto(endpoint)
                return True

        # Tests and non-CJU fallback users may deliberately omit the controller.
        if getattr(self, '_path_mpc', None) is None:
            self._path_failure_hold = None
            self._send_goto(safe_target)
            return True

        now = self._now()
        prior_reference_ready = (
            self._path_reference.ready()
            and self._path_solve_t is not None
            and now - self._path_solve_t + self._path_reference.lead
            < self._path_reference.T - 1.0e-6)
        solve_due = (
            not self._path_reference.ready()
            or getattr(self, '_path_last_solve_t', None) is None
            or now < self._path_last_solve_t
            or now - self._path_last_solve_t >= self.mpc_dt - 1.0e-6)
        if solve_due:
            self._path_last_solve_t = now
            output_step = min(
                int(self._path_reference.lead / self.mpc_dt),
                self._path_mpc.N - 1)
            solve_started = perf_counter()
            try:
                result, speed_safe, prediction_safe, hard_safe = (
                    MissionManagerNode._solve_tracking_path_mpc(
                        self, self._path_mpc,
                        self._mission_arc_m, self._mission_path, progress,
                        output_step, safety_planning,
                        allow_hard_recovery=True))
            finally:
                MissionManagerNode._record_mpc_solve(
                    self, perf_counter() - solve_started)
            dynamics_well_formed = (
                result.success
                and np.all(np.isfinite(result.predicted_vel))
                and np.all(np.isfinite(result.predicted_acc)))
            if (speed_safe and safety_planning and not prediction_safe
                    and hard_safe):
                # Any configured extra planning margin is the controller's
                # braking/rejoin reserve. Preserve the continuous MPC output
                # while its complete horizon still satisfies hard clearance.
                safety_planning = False
                prediction_safe = True
                self.get_logger().warn(
                    'TrackingMPC using hard-clearance recovery horizon',
                    throttle_duration_sec=2.0)
            accepted = speed_safe and prediction_safe
            if accepted:
                zeros = np.zeros(3)
                self._path_reference.set_plan(
                    self.p_d, self.v_d,
                    result.predicted_pos, result.predicted_vel,
                    result.predicted_acc, self.mpc_dt,
                    zeros, zeros, zeros)
                self._path_solve_t = now
            elif dynamics_well_formed and prior_reference_ready:
                # Keep the still-live certified P/V/A while a replacement
                # horizon brakes around a corner.  Its next streamed chord is
                # checked again below; an abrupt position hold at speed caused
                # the vehicle to drift outside the one-metre hard clearance.
                self.get_logger().warn(
                    'TrackingMPC recovery: keep prior certified PVA',
                    throttle_duration_sec=2.0)
            else:
                self._path_mpc.reset()
                self._path_reference.reset()
                self._path_solve_t = None
                self._path_last_solve_t = None
                if safety_planning:
                    self.get_logger().warn(
                        'TrackingMPC rejected; use certified path carrot',
                        throttle_duration_sec=2.0)
                else:
                    if getattr(self, '_path_failure_hold', None) is None:
                        self._path_failure_hold = self.p_d.copy()
                    self._hold_pos = self._path_failure_hold.copy()
                    self.get_logger().warn(
                        'TrackingMPC rejected outside planning reserve; hold',
                        throttle_duration_sec=2.0)
                    return False

        if self._path_reference.ready() and self._path_solve_t is not None:
            pos, vel, acc = self._path_reference.sample(
                self._now() - self._path_solve_t)
            segment_is_free = (_mission_planning_segment_is_free
                               if safety_planning
                               else _mission_segment_is_free)
            if (segment_is_free(self.mission_map_yaml, self.p_d, pos)
                    and np.all(np.isfinite(np.r_[pos, vel, acc]))
                    and _mpc_stream_speed_is_safe(
                        vel, self._path_mpc.v_max)):
                self._last_mpc_setpoint = np.asarray(pos, float).copy()
                last_acceleration = getattr(
                    self, '_last_sent_acceleration', np.zeros(3))
                last_time = getattr(self, '_last_sent_acceleration_t', None)
                elapsed = (getattr(self, 'dt', self.mpc_dt)
                           if last_time is None else max(
                               0.0, self._now() - last_time))
                streamed_acceleration = _limit_acceleration_slew(
                    last_acceleration, acc, self._path_mpc.j_max, elapsed)
                self._path_failure_hold = None
                self._send(pos, vel, streamed_acceleration)
                return True
            self._path_reference.reset()
            self._path_mpc.reset()

        if not safety_planning:
            if getattr(self, '_path_failure_hold', None) is None:
                self._path_failure_hold = self.p_d.copy()
            self._hold_pos = self._path_failure_hold.copy()
            return False
        # One manager remains the sole authority; Goto is only the bounded
        # fallback already proven by the existing route follower.
        self._send_goto(safe_target)
        self._path_failure_hold = None
        return True

    def _tracking_path_reference(
            self, arc_m, path, progress_m, speed_limit_m_s=None):
        """Build the shared geometry reference for live and shadow MPC."""
        return_braking = (
            getattr(self, 'state', '') in ('RETURN', 'RETURN_PLAN')
            and self._cue_fresh())
        target_velocity = self.cue_v[:2] if return_braking else None
        target_range = (float(np.linalg.norm(
            (self.cue - self.p_d)[:2])) if return_braking else math.inf)
        profile_acceleration = getattr(
            self, '_path_speed_profile_a_max', self._path_mpc_a_max)
        if getattr(self, 'state', '') in ('MISSION', 'HOVER'):
            profile_acceleration = getattr(
                self, '_mission_path_speed_profile_a_max',
                profile_acceleration)
        return _path_reference_horizon(
            arc_m, path, progress_m, self.mpc_dt, self._path_mpc.N,
            (self._path_mpc_speed if speed_limit_m_s is None
             else float(speed_limit_m_s)),
            profile_acceleration,
            self._path_mpc.j_max,
            target_velocity_xy=target_velocity,
            target_range_xy_m=target_range,
            relative_brake_start_m=getattr(
                self, '_path_relative_brake_distance', 10.0),
            target_relative_speed_m_s=getattr(
                self, '_path_target_relative_speed', 0.3))

    def _solve_tracking_path_mpc(
            self, controller, arc_m, path, progress_m, output_step,
            safety_planning, allow_hard_recovery):
        """Solve with one bounded retry for each rejected limit."""
        speed_limit = None
        speed_retried = False
        clearance_retried = False
        while True:
            reference_p, reference_v = MissionManagerNode._tracking_path_reference(
                self, arc_m, path, progress_m,
                speed_limit_m_s=speed_limit)
            result = controller.solve(
                self.p_d, self.v_d, reference_p, reference_v,
                applied_acceleration=getattr(
                    self, '_last_sent_acceleration', np.zeros(3)),
                output_step=output_step)
            well_formed = (
                result.success
                and np.all(np.isfinite(result.predicted_vel))
                and np.all(np.isfinite(result.predicted_acc)))
            speed_safe = (
                well_formed and _mpc_horizontal_speed_is_safe(
                    result.predicted_vel, controller.v_max))
            if well_formed and not speed_safe and not speed_retried:
                peak_speed = float(np.max(np.linalg.norm(
                    result.predicted_vel[:, :2], axis=1)))
                retry_speed = max(
                    0.5, min(
                        self._path_mpc_speed,
                        self._path_mpc_speed * 0.9
                        * controller.v_max / peak_speed))
                speed_limit = min(
                    retry_speed,
                    self._path_mpc_speed if speed_limit is None
                    else speed_limit)
                speed_retried = True
                continue

            prediction_safe = (
                speed_safe and _mpc_prediction_is_safe(
                    self.mission_map_yaml, self.p_d,
                    result.predicted_pos, planning=safety_planning))
            hard_safe = prediction_safe
            if (speed_safe and safety_planning and not prediction_safe
                    and allow_hard_recovery):
                hard_safe = _mpc_prediction_is_safe(
                    self.mission_map_yaml, self.p_d,
                    result.predicted_pos, planning=False)
            clearance_safe = (
                prediction_safe or (allow_hard_recovery and hard_safe))
            if (speed_safe and not clearance_safe
                    and not clearance_retried):
                # The B-spline is certified, but a fast controller horizon can
                # still cut inside a tight corner.  Keep one bounded retry at
                # 75% of the rejected horizon; tying it to measured speed made
                # a transient slowdown collapse the pursuit reference to
                # 0.5--4 m/s.  The collision check remains the publication
                # gate, so this changes performance without weakening safety.
                current_limit = (
                    self._path_mpc_speed if speed_limit is None
                    else speed_limit)
                predicted_peak = float(np.max(np.linalg.norm(
                    result.predicted_vel[:, :2], axis=1)))
                speed_limit = max(
                    0.5, 0.75 * min(current_limit, predicted_peak))
                clearance_retried = True
                self.get_logger().warn(
                    'TrackingMPC clearance retry at '
                    f'{speed_limit:.2f}m/s',
                    throttle_duration_sec=2.0)
                continue
            return result, speed_safe, prediction_safe, hard_safe

    def _shadow_tracking_plan(self, arc_m, path, progress_m):
        """Prepare a replacement horizon without mutating live control."""
        if getattr(self, '_path_mpc', None) is None:
            return None
        candidate = deepcopy(self._path_mpc)
        output_step = min(
            int(self._path_reference.lead / self.mpc_dt), candidate.N - 1)
        result, speed_safe, prediction_safe, _ = (
            MissionManagerNode._solve_tracking_path_mpc(
                self, candidate, arc_m, path, progress_m, output_step,
                safety_planning=True, allow_hard_recovery=False))
        accepted = (
            result.success
            and speed_safe
            and prediction_safe)
        return (candidate, result) if accepted else None

    def _vision_measurement_fresh(self):
        return bool(
            self.vis_entry_valid
            and MissionManagerNode._vision_track_usable(self))

    def _vision_track_usable(self):
        """Keep an entered KF track usable through its bounded coast."""
        return bool(
            self.vis_valid and self.vis is not None and self.cue is not None
            and self._t_vis is not None
            and self._now() - self._t_vis <= self.vis_fresh)

    def _vision_correction_converged(self):
        if (not self._vision_measurement_fresh()
                or not MissionManagerNode._vision_velocity_converged(self)):
            return False
        corrected = MissionManagerNode._cue_at_now(self)[:2] + self._bias[:2]
        return float(np.linalg.norm(self.vis[:2] - corrected)) <= (
            self._landing_mpc_bias_ok)

    def _vision_track_converged(self):
        """Accept a coast only while the existing KF remains cue-consistent."""
        if (not MissionManagerNode._vision_track_usable(self)
                or not MissionManagerNode._vision_velocity_converged(self)):
            return False
        corrected = MissionManagerNode._cue_at_now(self)[:2] + self._bias[:2]
        return float(np.linalg.norm(self.vis[:2] - corrected)) <= (
            self._landing_mpc_bias_ok)

    def _vision_velocity_converged(self):
        """Reject the KF bootstrap transient before learning cue bias."""
        velocity = getattr(self, 'vis_v', None)
        cue_velocity = getattr(self, 'cue_v', None)
        return bool(
            velocity is not None and cue_velocity is not None
            and np.all(np.isfinite(np.r_[velocity, cue_velocity]))
            and float(np.linalg.norm(
                np.asarray(velocity, float)[:2]
                - np.asarray(cue_velocity, float)[:2]))
            <= self._landing_v_tol)

    def _precland_target_allowed(self):
        """Require vision except for one bounded, aligned camera-blind commit."""
        now = self._now()
        target = MissionManagerNode._cue_at_now(self) + self._bias
        p_rel = self.p_d - target
        v_rel = self.v_d - self.cue_v
        horizontal_aligned = (
            float(np.linalg.norm(p_rel[:2])) <= self._landing_xy_tol)
        velocity_aligned = (
            float(np.linalg.norm(v_rel[:2])) <= self._landing_v_tol)
        aligned = (
            0.0 <= float(p_rel[2]) <= self._precland_commit_height
            and horizontal_aligned and velocity_aligned)
        if self._vision_measurement_fresh():
            if aligned:
                # The centred 0.30 m marker fills the image below about 0.19 m.
                # The SITL estimator reads ~0.42 m high there, hence 0.65 m.
                # The low marker disappears about 3 s before contact and PX4
                # ground_contact can lag another 4 s; eight seconds leaves a
                # one-second margin without relaxing the alignment gates.
                self._precland_commit_until = (
                    now + self._precland_commit_grace)
            return True
        if getattr(self, '_touchdown_metric_candidate', None) is not None:
            return True
        # A qualified low-altitude handoff remains valid across a few
        # millimetres of vertical estimator rebound.  Horizontal alignment and
        # relative speed remain live safety gates.
        return bool(
            self._precland_commit_until is not None
            and now <= self._precland_commit_until
            and horizontal_aligned and velocity_aligned)

    def _terminal_runway_status(
            self, descent_height_m=None, preparation_s=0.0):
        """Return whether PX4 can finish before the next route turn."""
        waypoints = getattr(self, '_terminal_waypoints_local', None)
        if waypoints is None:
            return True, math.inf
        cue_now = MissionManagerNode._cue_at_now(self)
        eta_s = _forward_endpoint_eta_s(
            cue_now[:2], self.cue_v[:2], waypoints,
            self._terminal_waypoint_tolerance,
            self._terminal_min_cruise_speed,
            self._terminal_route_match_tolerance)
        required_s = (
            self._precland_runway_required_s
            + max(0.0, float(preparation_s)))
        if descent_height_m is not None:
            descent_m = max(
                0.0, float(descent_height_m)
                - self._precland_commit_height)
            required_s += descent_m / self._landing_mpc_vz_max
        return eta_s >= required_s, eta_s

    def _landing_diagnostics(self):
        if self.p_d is None or self.cue is None:
            return 'landing data: waiting'
        cue_now = MissionManagerNode._cue_at_now(self)
        distance = float(np.linalg.norm(self.p_d[:2] - cue_now[:2]))
        relative_speed = float(np.linalg.norm(
            self.v_d[:2] - self.cue_v[:2]))
        runway_clear, eta_s = MissionManagerNode._terminal_runway_status(self)
        deadline = getattr(self, '_precland_commit_until', None)
        commit_s = 0.0 if deadline is None else max(
            0.0, float(deadline) - self._now())
        eta_text = 'n/a' if math.isinf(eta_s) else f'{eta_s:.1f}s'
        vision_residual = math.nan
        vision_dv = math.nan
        vision_usable = MissionManagerNode._vision_track_usable(self)
        if self.vis is not None and vision_usable:
            vision_residual = float(np.linalg.norm(
                self.vis[:2] - (cue_now[:2] + self._bias[:2])))
        if vision_usable and getattr(self, 'vis_v', None) is not None:
            vision_dv = float(np.linalg.norm(
                self.vis_v[:2] - self.cue_v[:2]))
        vision_text = (
            ('stale' if self.vis is not None and not vision_usable else 'n/a')
            if not math.isfinite(vision_residual)
            else f'{vision_residual:.2f}m/{vision_dv:.2f}m/s')
        return (
            f'horizontal range={distance:.2f}m | '
            f'relative XY speed={relative_speed:.2f}m/s | '
            f'ArUco entry={int(bool(self.vis_entry_valid))} '
            f'(res/dv={vision_text}) | '
            f'blind commit={commit_s:.1f}s | '
            f'turnaround={"CLEAR" if runway_clear else "HOLD"} '
            f'(endpoint ETA {eta_text})')

    def _landing_mpc_entry_ready(self):
        """Enter a planning-safe ACQUIRE; direct mode aims there before vision."""
        cue_now = MissionManagerNode._cue_at_now(self)
        direct_acquire = getattr(
            self, '_landing_target_min_speed', 0.0) > 0.0
        direct_range = float(getattr(
            self, '_landing_gps_preacquire_range', 0.0))
        direct_ready = (
            direct_acquire
            and direct_range > 0.0
            and float(np.linalg.norm(
                (self.p_d - cue_now)[:2])) <= direct_range)
        takeoff_direct = (
            getattr(self, 'state', None) == 'TAKEOFF'
            and bool(getattr(self, 'auto_start', False)))
        runway_clear = True
        preparation_s = (
            getattr(self, '_entry_fix_window', 0.0)
            + getattr(self, '_landing_alignment_margin_s', 0.0))
        if direct_ready and not takeoff_direct:
            approaching = float(
                (cue_now - self.p_d)[:2]
                @ (self.cue_v - self.v_d)[:2]) < 0.0
            runway_clear, _ = MissionManagerNode._terminal_runway_status(
                self, self.p_d[2] - self.cue[2], preparation_s)
            direct_ready = approaching and runway_clear
        elif direct_acquire and not takeoff_direct:
            runway_clear, _ = MissionManagerNode._terminal_runway_status(
                self, self.p_d[2] - self.cue[2], preparation_s)
        # A visual shortcut beyond the GPS range is allowed only after both
        # position and velocity agree with the independent trailer cue.  Three
        # accepted frames alone still include the KF bootstrap transient.
        vision_ready = self._vision_correction_converged()
        return (
            self._landing_target_fresh()
            and (not direct_acquire or takeoff_direct or runway_clear)
            and (direct_ready or vision_ready)
            and _mission_planning_segment_is_free(
                self.mission_map_yaml, self.p_d, cue_now))

    def _gps_preacquire_entry_ready(self):
        """Start altitude-hold GPS matching only at the certified staging point."""
        if (not getattr(self, '_return_staging', False)
                or not getattr(self, '_return_staging_arrived', False)
                or self._landing_gps_preacquire_range <= 0.0
                or not self._landing_target_fresh()):
            return False
        cue_now = MissionManagerNode._cue_at_now(self)
        distance = float(np.linalg.norm((self.p_d - cue_now)[:2]))
        approaching = float(
            (cue_now - self.p_d)[:2]
            @ (self.cue_v - self.v_d)[:2]) < 0.0
        runway_clear, _ = MissionManagerNode._terminal_runway_status(
            self, self.p_d[2] - self.cue[2],
            getattr(self, '_entry_fix_window', 0.0)
            + getattr(self, '_landing_alignment_margin_s', 0.0))
        return bool(
            distance <= self._landing_gps_preacquire_range
            and approaching
            and runway_clear
            and _mission_planning_segment_is_free(
                self.mission_map_yaml, self.p_d, cue_now))

    def _enter_landing_mpc(self, *, gps_preacquire=False):
        ready = (MissionManagerNode._gps_preacquire_entry_ready(self)
                 if gps_preacquire else self._landing_mpc_entry_ready())
        if not ready:
            return False
        moving_target_acquire = (
            getattr(self, '_landing_target_min_speed', 0.0) > 0.0)
        gps_velocity_acquire = bool(
            gps_preacquire or (
                moving_target_acquire
                and getattr(self, 'state', None) != 'TAKEOFF'))
        self._return_staging = False
        self._return_staging_arrived = False
        self._return_intercept_lock_active = False
        self._gps_preacquire_active = gps_velocity_acquire
        self._landing_failure_hold = None
        self._landing_bias_locked = False
        self._landing_mpc.reset()
        self._landing_reference.reset()
        self._landing_solve_t = None
        self._landing_last_solve_t = None
        self._landing_hold_z = float(self.p_d[2])
        self._last_mpc_setpoint = self.p_d.copy()
        self._publish_planned_path(None)
        self._set_state(
            'LANDING_ACQUIRE',
            ('(GPS velocity pre-acquire; hold altitude for ArUco)'
             if gps_velocity_acquire
             else ('(direct moving-target ACQUIRE; hold altitude for ArUco)'
                   if moving_target_acquire
                   else '(3 KF-accepted ArUco fixes within 0.5 s)')))
        return True

    def _recover_precland(self):
        """Reclaim Offboard at the current altitude and reacquire ArUco."""
        self._landing_mpc.reset()
        self._landing_reference.reset()
        self._landing_solve_t = None
        self._landing_last_solve_t = None
        # Warm up Offboard at the current state.  ACQUIRE keeps this altitude
        # while restoring the vision entry qualification and XY alignment.
        self._landing_hold_z = float(self.p_d[2])
        self._hold_pos = self.p_d.copy()
        self._native_precland_accepted = False
        self._last_offboard_cmd = None
        self._landing_recovery_since = self._now()
        self._precland_commit_until = None
        # A staging-started attempt keeps its runway/speed/clearance contract
        # across PRECLAND recovery.  Only an explicit restage or DONE clears it.
        self._landing_failure_hold = None
        self._set_state(
            'LANDING_ACQUIRE', '(ArUco lost; reclaim Offboard and reacquire)')

    def _precland_terminal_latched(self):
        """Keep PX4 in charge once a centred low-altitude commit has begun."""
        position = getattr(self, 'p_d', None)
        cue = getattr(self, 'cue', None)
        if (position is None or cue is None
                or getattr(self, '_precland_commit_until', None) is None):
            return False
        target = MissionManagerNode._cue_at_now(self) + getattr(
            self, '_bias', np.zeros(3))
        height = float(position[2] - target[2])
        return 0.0 <= height <= self._precland_commit_height

    def _run_landing_mpc(self):
        """Run wang LandingMPC; PX4 still owns physical contact and disarm."""
        if not self._landing_target_fresh():
            self._landing_reference.reset()
            self._landing_mpc.reset()
            self._landing_solve_t = None
            if getattr(self, '_landing_failure_hold', None) is None:
                self._landing_failure_hold = self.p_d.copy()
            self._hold_pos = self._landing_failure_hold.copy()
            self._send_goto(self._hold_pos)
            return

        target, target_v = self._target()
        p_rel = self.p_d - target
        v_rel = self.v_d - target_v
        gps_preacquire = bool(getattr(
            self, '_gps_preacquire_active', False))
        gps_speed_limit = _GPS_PREACQUIRE_SPEED_LIMIT_M_S
        if gps_preacquire:
            runway_clear, eta_s = MissionManagerNode._terminal_runway_status(
                self, p_rel[2])
            if not runway_clear:
                self._landing_mpc.reset()
                self._landing_reference.reset()
                self._landing_solve_t = None
                self._landing_last_solve_t = None
                self._landing_hold_z = None
                self._gps_preacquire_active = False
                self._landing_failure_hold = None
                if hasattr(self, '_bias'):
                    self._bias[:2] = 0.0
                self._return_staging = getattr(
                    self, '_return_staging_enabled', True)
                self._return_staging_arrived = False
                if not self._return_staging:
                    # The current intercept has just lost its certified
                    # runway.  After braking, force a fresh moving-target
                    # route instead of relocking the expired endpoint.
                    self._return_intercept_deadline_t = None
                    self._return_intercept_lock_active = False
                    self._last_return_plan_t = None
                if getattr(self, '_path_failure_hold', None) is None:
                    self._path_failure_hold = self.p_d.copy()
                self._hold_pos = self._path_failure_hold.copy()
                self._send_goto(self._hold_pos)
                recovery = ('reuse staging path' if self._return_staging
                            else 'resume dynamic pursuit')
                self._set_state(
                    'RETURN',
                    f'(GPS acquire runway expired; stop and {recovery})')
                self.get_logger().warn(
                    'GPS acquire stopped: insufficient runway '
                    f'(endpoint ETA {eta_s:.1f} s)')
                return
        entry_vision_ready = self._vision_correction_converged()
        vision_measurement_fresh = self._vision_measurement_fresh()
        acquiring = self.state == 'LANDING_ACQUIRE'
        vision_ready = (
            entry_vision_ready if acquiring else vision_measurement_fresh)
        vision_coasting = (
            not acquiring and not vision_measurement_fresh
            and MissionManagerNode._vision_track_usable(self))
        # ArUco can qualify entry only 0.5 s after the route brake begins.
        # Keep the jerk limit, but give ACQUIRE the route controller's braking
        # authority; DESCEND returns to the conservative landing limit.
        self._landing_mpc.a_max = (
            self._path_mpc_a_max if acquiring else self._landing_mpc_a_max)

        if acquiring or not vision_ready:
            # At acquisition, or on a vision dropout, align horizontally while
            # holding the altitude at which this attempt began.  Disabling the
            # cone and retaining that exact height prevents both descent and a
            # low-altitude recovery climb.
            self._landing_mpc.cone_k = 0.0
            self._landing_mpc.z_ref = float(p_rel[2])
            if self._landing_hold_z is None:
                self._landing_hold_z = float(self.p_d[2])
        else:
            self._landing_mpc.cone_k = self._landing_mpc_cone
            # Aim inside the handoff gate instead of asymptotically at its
            # boundary.  Two MPC steps leave a deterministic crossing margin
            # without relaxing the 0.65 m PRECLAND safety predicate.
            self._landing_mpc.z_ref = max(
                0.0, self._precland_commit_height
                - 2.0 * self._landing_mpc_vz_max * self.mpc_dt)
            self._landing_hold_z = None

        now = self._now()
        last_solve = getattr(self, '_landing_last_solve_t', None)
        solve_due = (
            last_solve is None or now < last_solve
            or now - last_solve >= self.mpc_dt - 1.0e-6)
        if solve_due:
            prior_reference_ready = (
                self._landing_reference.ready()
                and self._landing_solve_t is not None
                and now - self._landing_solve_t
                + self._landing_reference.lead
                < self._landing_reference.T - 1.0e-6)
            self._landing_last_solve_t = now
            target_p, target_v_h, target_a = predict_const_vel(
                target, target_v, self.mpc_dt, self._landing_mpc.N)
            output_step = min(
                int(self._landing_reference.lead / self.mpc_dt),
                self._landing_mpc.N - 1)
            solve_started = perf_counter()
            try:
                result = self._landing_mpc.solve(
                    p_rel, v_rel, target_p, target_v_h, target_a,
                    applied_acceleration=getattr(
                        self, '_last_sent_acceleration', np.zeros(3)),
                    output_step=output_step)
            finally:
                MissionManagerNode._record_mpc_solve(
                    self, perf_counter() - solve_started)
            absolute_prediction = target_p + result.pred_rel_pos
            absolute_velocity = target_v_h + result.pred_rel_vel
            result_well_formed = (
                result.success
                and np.all(np.isfinite(np.column_stack((
                    absolute_prediction, result.pred_rel_vel,
                    result.pred_rel_acc)))))
            speed_safe = (
                result_well_formed
                and (not gps_preacquire
                     or _mpc_horizontal_speed_is_safe(
                         absolute_velocity, gps_speed_limit)))
            accepted = (
                speed_safe and _mpc_prediction_is_safe(
                    self.mission_map_yaml, self.p_d,
                    absolute_prediction, planning=gps_preacquire))
            if accepted:
                self._landing_reference.set_plan(
                    p_rel, v_rel, result.pred_rel_pos,
                    result.pred_rel_vel, result.pred_rel_acc,
                    self.mpc_dt, target, target_v, np.zeros(3))
                self._landing_solve_t = now
            elif (gps_preacquire and result_well_formed and not speed_safe
                  and prior_reference_ready):
                self.get_logger().warn(
                    'LandingMPC speed recovery: keep prior certified PVA',
                    throttle_duration_sec=2.0)
            else:
                self._landing_reference.reset()
                self._landing_mpc.reset()
                self._landing_solve_t = None
                if getattr(self, '_landing_failure_hold', None) is None:
                    self._landing_failure_hold = self.p_d.copy()
                self._hold_pos = self._landing_failure_hold.copy()
                self.get_logger().warn(
                    'LandingMPC rejected; hold without stale control',
                    throttle_duration_sec=2.0)

        if (self._landing_reference.ready()
                and self._landing_solve_t is not None):
            pos, vel, acc = self._landing_reference.sample(
                self._now() - self._landing_solve_t)
            if self._landing_hold_z is not None:
                pos = np.asarray(pos, float).copy()
                vel = np.asarray(vel, float).copy()
                acc = np.asarray(acc, float).copy()
                pos[2] = self._landing_hold_z
                vel[2] = 0.0
                acc[2] = 0.0
            segment_is_free = (_mission_planning_segment_is_free
                               if getattr(
                                   self, '_gps_preacquire_active', False)
                               else _mission_segment_is_free)
            if (segment_is_free(self.mission_map_yaml, self.p_d, pos)
                    and np.all(np.isfinite(np.r_[pos, vel, acc]))
                    and (not gps_preacquire
                         or _mpc_stream_speed_is_safe(
                             vel, gps_speed_limit))):
                self._last_mpc_setpoint = np.asarray(pos, float).copy()
                last_acceleration = getattr(
                    self, '_last_sent_acceleration', np.zeros(3))
                last_time = getattr(self, '_last_sent_acceleration_t', None)
                elapsed = (getattr(self, 'dt', self.mpc_dt)
                           if last_time is None else max(
                               0.0, self._now() - last_time))
                streamed_acceleration = _limit_acceleration_slew(
                    last_acceleration, acc, self._landing_mpc.j_max, elapsed)
                if self._landing_hold_z is not None:
                    streamed_acceleration[2] = 0.0
                self._landing_failure_hold = None
                self._send(pos, vel, streamed_acceleration)
            else:
                self._landing_reference.reset()
                self._landing_mpc.reset()
                if getattr(self, '_landing_failure_hold', None) is None:
                    self._landing_failure_hold = self.p_d.copy()
                self._hold_pos = self._landing_failure_hold.copy()
                self._send_goto(self._hold_pos)
                return
        else:
            if getattr(self, '_landing_failure_hold', None) is None:
                self._landing_failure_hold = self.p_d.copy()
            self._hold_pos = self._landing_failure_hold.copy()
            self._send_goto(self._hold_pos)
            return

        horizontal = float(np.linalg.norm(p_rel[:2]))
        relative_speed = float(np.linalg.norm(v_rel[:2]))
        target_speed_ready = (
            float(np.linalg.norm(target_v[:2])) + 1.0e-3
            >= getattr(self, '_landing_target_min_speed', 0.0))
        if (acquiring and entry_vision_ready
                and target_speed_ready
                and horizontal <= self._landing_xy_tol
                and relative_speed <= self._landing_v_tol
                and p_rel[2] >= 0.0):
            runway_clear, eta_s = MissionManagerNode._terminal_runway_status(
                self, p_rel[2])
            if not runway_clear:
                self.get_logger().warn(
                    'ArUco aligned but descent held: insufficient runway '
                    f'(endpoint ETA {eta_s:.1f} s)',
                    throttle_duration_sec=2.0)
                return
            if p_rel[2] <= self._precland_commit_height:
                # Reacquisition below the terminal gate must not pass through
                # DESCEND, whose normal z_ref is the higher commit height.
                self._last_safe_goto = self.p_d.copy()
                self._enter_precland(horizontal)
                return
            # Keep the certified horizontal P/V/A continuous across the mode
            # switch.  The next solve may replace it with a descent plan; if
            # the absolute-speed guard needs a recovery tick, the old plan is
            # still a safe altitude-hold instead of a zero-velocity Goto.
            self._landing_hold_z = None
            # Freeze the independently qualified visual/GPS calibration before
            # camera geometry changes during descent.  Reacquisition may use
            # the latched value, but it cannot learn a new landing point low.
            self._landing_bias_locked = True
            # Keep the staging runway contract armed.  A transient ArUco coast
            # can return DESCEND to ACQUIRE, and that retry must still stop at
            # the route endpoint instead of chasing through the next turn.
            self._landing_failure_hold = None
            self._set_state(
                'LANDING_DESCEND',
                f'(ArUco aligned, d={horizontal:.2f} m, '
                f'dv={relative_speed:.2f} m/s)')
            return

        if (not acquiring and not vision_ready):
            if vision_coasting:
                self.get_logger().warn(
                    'ArUco entry window lost; hold altitude on bounded KF '
                    'coast', throttle_duration_sec=2.0)
                return
            # Keep streaming the certified horizontal plan while reacquiring.
            # `_landing_hold_z` below independently freezes z/vz/az.
            self._landing_hold_z = float(self.p_d[2])
            self._set_state(
                'LANDING_ACQUIRE',
                '(ArUco track invalid/stale; hold altitude and reacquire)')
            return

        runway_clear, _ = MissionManagerNode._terminal_runway_status(self)
        if (not acquiring and entry_vision_ready
                and 0.0 <= p_rel[2] <= self._precland_commit_height
                and horizontal <= self._landing_xy_tol
                and relative_speed <= self._landing_v_tol
                and runway_clear):
            self._last_safe_goto = self._last_mpc_setpoint.copy()
            self._enter_precland(horizontal)

    def _publish_landing_target(self):
        """Publish one live target measurement; never a flight setpoint."""
        if self.p_d is None:
            reason = 'vehicle position unavailable'
        elif not self._landing_target_fresh():
            reason = 'trailer cue source stale'
        elif not MissionManagerNode._precland_target_allowed(self):
            reason = 'vision/alignment blind-commit gate closed'
        else:
            reason = None
        if reason is not None:
            self.get_logger().warn(
                f'landing target blocked: {reason}',
                throttle_duration_sec=2.0)
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
            previous = self.state
            self.get_logger().info(f'{previous} -> {s}  {why}')
            if s == 'PRECLAND':
                self._precland_attempts = getattr(
                    self, '_precland_attempts', 0) + 1
            if previous == 'PRECLAND' and s == 'LANDING_ACQUIRE':
                self._precland_recoveries = getattr(
                    self, '_precland_recoveries', 0) + 1
            if previous == 'LANDING_DESCEND' and s == 'LANDING_ACQUIRE':
                self._landing_descend_recoveries = getattr(
                    self, '_landing_descend_recoveries', 0) + 1
            self.state = s
            if s == 'HOVER':
                self._hover_since = self._now()
            elif previous == 'HOVER':
                self._hover_since = None
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
            if s == 'DONE':
                self._gps_preacquire_active = False
                self._log_experiment_metrics()

    def _return_intercept_is_viable(self):
        """Keep an intercept that the drone can reach before the trailer."""
        deadline = getattr(self, '_return_intercept_deadline_t', None)
        arc = getattr(self, '_mission_arc_m', None)
        path = getattr(self, '_mission_path', None)
        if deadline is None or arc is None or path is None:
            return False
        now = self._now()
        target_eta = float(deadline) - now
        progress = float(getattr(self, '_mission_progress_m', 0.0))
        remaining = max(0.0, float(arc[-1]) - progress)
        cruise = min(
            float(getattr(self, '_path_mpc_speed', 0.0)),
            _PATH_HARD_SPEED_LIMIT_M_S)
        acceleration = float(getattr(self, '_path_mpc_a_max', 0.0))
        if (not math.isfinite(target_eta) or target_eta <= 0.0
                or cruise <= 0.0 or acceleration <= 0.0):
            return False

        segment = int(np.clip(
            np.searchsorted(arc, progress, side='right') - 1,
            0, len(path) - 2))
        tangent = np.asarray(path[segment + 1], float) - np.asarray(
            path[segment], float)
        tangent_norm = float(np.linalg.norm(tangent[:2]))
        along_speed = 0.0
        if tangent_norm > 1.0e-9:
            along_speed = max(
                0.0, float(np.asarray(self.v_d, float)[:2]
                           @ (tangent[:2] / tangent_norm)))
        along_speed = min(along_speed, cruise)
        acceleration_distance = max(
            0.0, (cruise ** 2 - along_speed ** 2)
            / (2.0 * acceleration))
        if remaining <= acceleration_distance:
            drone_eta = (
                math.sqrt(along_speed ** 2
                          + 2.0 * acceleration * remaining)
                - along_speed) / acceleration
        else:
            drone_eta = ((cruise - along_speed) / acceleration
                         + (remaining - acceleration_distance) / cruise)
        margin = (0.0 if remaining <= getattr(
            self, 'mission_tolerance', 0.7)
                  else float(getattr(
                      self, '_return_replan_min_period', 2.0)))
        self._return_intercept_drone_eta_s = drone_eta
        self._return_intercept_target_eta_s = target_eta
        return drone_eta + margin <= target_eta

    def _start_global_plan(self, goal_local_enu, *, return_route):
        """Start one serialized map-goal or trailer A*/B-spline leg."""
        if self._planner_pool is None or self.p_d is None:
            raise RuntimeError('global planner is unavailable')
        rolling_return = (
            return_route and self.state == 'RETURN'
            and self._mission_path is not None)
        if return_route and not rolling_return:
            self._return_intercept_deadline_t = None
            self._return_intercept_lock_active = False
        current = self.p_d.copy()
        start = current.copy()
        # READY is the YAML spawn/deck origin.  Do not let centimetre-scale
        # EKF noise select a different outbound A* tie on every run.
        if not return_route and self.state == 'READY':
            start[:2] = 0.0
        self._hold_pos = current
        goal = (None if goal_local_enu is None
                else np.asarray(goal_local_enu, float).copy())
        if goal is not None and not np.all(np.isfinite(goal)):
            raise ValueError('global plan goal must be finite')
        self._plan_goal_lead_s = 0.0
        self._plan_endpoint_residual_m = math.nan
        if (return_route and goal is not None
                and not getattr(self, '_return_staging', False)
                and getattr(self, '_terminal_waypoints_local', None) is not None
                and getattr(self, 'cue_v', None) is not None):
            velocity = np.asarray(self.cue_v, float)
            if velocity.shape == (3,) and np.all(np.isfinite(velocity)):
                latency_lead = max(0.0, float(getattr(
                    self, '_return_plan_lead_s', 0.0)))
                target_speed = float(np.linalg.norm(velocity[:2]))
                if target_speed > 1.0e-6:
                    # Keep the certified finite path's zero-speed endpoint;
                    # place it far enough ahead to stop if retargeting stops.
                    path_speed = min(
                        self._path_mpc_speed, _PATH_HARD_SPEED_LIMIT_M_S)
                    stop_distance = _s_curve_stop_distance(
                        path_speed, self._path_speed_profile_a_max,
                        getattr(getattr(self, '_path_mpc', None),
                                'j_max', 2.0))
                    # Keep the endpoint ahead until the *next* rolling plan
                    # can finish: current worker latency + cooldown + next
                    # worker latency, followed by the finite-path stop reserve.
                    requested_lead = (
                        2.0 * latency_lead
                        + self._return_replan_min_period
                        + stop_distance / target_speed)
                    route_goal = None
                    if getattr(self, '_terminal_route_cyclic', False):
                        route_goal = _forward_cyclic_route_position(
                            goal[:2], velocity[:2],
                            self._terminal_waypoints_local, requested_lead,
                            self._terminal_waypoint_tolerance,
                            self._terminal_min_cruise_speed,
                            self._terminal_route_match_tolerance)
                    if route_goal is not None:
                        goal[:2] = route_goal
                        lead = requested_lead
                    else:
                        endpoint_eta = _forward_endpoint_eta_s(
                            goal[:2], velocity[:2],
                            self._terminal_waypoints_local,
                            self._terminal_waypoint_tolerance,
                            self._terminal_min_cruise_speed,
                            self._terminal_route_match_tolerance)
                        lead = min(requested_lead, endpoint_eta)
                        goal[:2] += velocity[:2] * lead
                else:
                    lead = 0.0
                self._plan_goal_lead_s = lead
        if not rolling_return:
            self._publish_planned_path(None)
        if self._plan_future is not None:
            self._plan_future.cancel()
        self._plan_future = self._planner_pool.submit(
            _plan_global_path, self.mission_map_yaml, list(start),
            None if goal is None else goal.tolist(), include_diagnostics=True)
        self._plan_start = start
        self._plan_goal = goal
        self._plan_started_t = self._now()
        if return_route:
            self._last_return_plan_t = self._plan_started_t
        if not rolling_return:
            self._mission_arc_m = None
            self._mission_path = None
            self._mission_progress_m = 0.0
        if return_route:
            destination = ('landing staging point' if getattr(
                self, '_return_staging', False) else 'moving trailer')
            state = 'RETURN_PLAN'
        else:
            state, destination = 'MISSION_PLAN', 'mission goal'
        self._set_state(state, f'(A* + geometry B-spline -> {destination})')

    def _publish_planned_path(self, path):
        """Publish only geometry accepted by the flight authority."""
        message = PoseArray()
        stamp = self.get_clock().now().to_msg()
        message.header.stamp = stamp
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
        if path is None and hasattr(self, 'sfc_pub'):
            self._publish_sfc(None)
        if (path is None and hasattr(self, 'active_plan_pub')
                and getattr(self, 'mission_map_yaml', '')):
            self.active_plan_pub.publish(_active_plan_marker_message(
                self.mission_map_yaml, None, None, 0, stamp))
            self._active_sfc_diagnostics = None

    def _publish_sfc(self, diagnostics):
        """Publish the certified active-path SFC on the legacy UI topic."""
        message = Float32MultiArray()
        if diagnostics is not None:
            boxes_min = np.asarray(
                diagnostics['sfc_boxes_min_map'], float)
            boxes_max = np.asarray(
                diagnostics['sfc_boxes_max_map'], float)
            if (boxes_min.ndim != 2 or boxes_min.shape[1] != 3
                    or boxes_max.shape != boxes_min.shape
                    or not np.all(np.isfinite(np.r_[boxes_min, boxes_max]))
                    or np.any(boxes_max < boxes_min)):
                raise ValueError('SFC boxes must be finite ordered Nx3 arrays')
            # The live YAML view is top-down, so publish xmin,ymin,xmax,ymax.
            message.data = np.column_stack((
                boxes_min[:, 0], boxes_min[:, 1],
                boxes_max[:, 0], boxes_max[:, 1])).astype(
                    np.float32).ravel().tolist()
        self.sfc_pub.publish(message)

    def _commit_active_path(
            self, arc_m, path, progress_m, diagnostics, *, reset_mpc=False):
        """Atomically replace controller geometry and the live UI snapshot."""
        sequence = getattr(self, '_active_plan_seq', 0) + 1
        snapshot = None
        if hasattr(self, 'active_plan_pub'):
            stamp = self.get_clock().now().to_msg()
            snapshot = _active_plan_marker_message(
                self.mission_map_yaml, path, diagnostics, sequence, stamp)
        arc = np.asarray(arc_m, float)
        points = np.asarray(path, float)
        if (len(arc) != len(points) or len(points) < 2
                or not np.all(np.isfinite(np.column_stack((arc, points))))
                or not np.all(np.diff(arc) > 0.0)):
            raise ValueError('active path arc contract failed')

        self._mission_arc_m = arc
        self._mission_path = points
        self._mission_progress_m = float(progress_m)
        self._active_plan_seq = sequence
        self._active_sfc_diagnostics = diagnostics
        self._path_terminal_goto = None
        if reset_mpc and getattr(self, '_path_mpc', None) is not None:
            self._path_mpc.reset()
            self._path_reference.reset()
            self._path_solve_t = None
            self._path_last_solve_t = None
        if snapshot is not None:
            self.active_plan_pub.publish(snapshot)
        self._publish_planned_path(points)
        if hasattr(self, 'sfc_pub'):
            self._publish_sfc(diagnostics)

    def _target(self):
        """Return the live cue plus a fresh, horizontal ArUco correction."""
        cue = (MissionManagerNode._cue_at_now(self)
               if self.cue is not None else self.p_d)
        if (not getattr(self, '_landing_bias_locked', False)
                and getattr(self, 'state', 'LANDING_ACQUIRE')
                == 'LANDING_ACQUIRE'
                and MissionManagerNode._vision_measurement_fresh(self)
                and MissionManagerNode._vision_velocity_converged(self)):
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
        diagnostics_pub = getattr(self, 'landing_diagnostics_pub', None)
        if diagnostics_pub is not None:
            diagnostics_pub.publish(String(
                data=MissionManagerNode._landing_diagnostics(self)))
        if self.state == 'DONE' or self.p_d is None:
            return
        if (self.state in ('LANDING_ACQUIRE', 'LANDING_DESCEND', 'PRECLAND')
                and self.landed is True and self.armed is False):
            self._set_state('DONE', '(PX4 landed and auto-disarmed)')
            return
        if (self.state in ('LANDING_ACQUIRE', 'LANDING_DESCEND', 'PRECLAND')
                and getattr(self, '_ground_contact_seen', False)):
            # Contact is a one-way terminal boundary.  Keep feeding PX4's
            # native PRECLAND target when possible, but never reclaim Offboard
            # or command an acquire climb after contact.
            if self.state == 'PRECLAND':
                self._publish_landing_target()
            return
        if (self.state in ('LANDING_ACQUIRE', 'LANDING_DESCEND')
                and (getattr(self, '_touchdown_metric_candidate', None)
                     is not None
                     or getattr(self, '_landing_contact_confirmed', False))):
            # Never reclaim Offboard or command a climb once contact has begun.
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

        if (self.state in (
                'RETURN_PLAN', 'RETURN',
                'LANDING_ACQUIRE', 'LANDING_DESCEND')
                and not self._cue_fresh()):
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
                direct_landing = (
                    self.auto_start
                    and getattr(self, '_landing_target_min_speed', 0.0) > 0.0)
                if direct_landing and self._cue_fresh():
                    self._enter_landing_mpc()
                elif self.auto_start and self._planner_pool is not None:
                    self._start_global_plan(None, return_route=False)
                elif self.auto_start and self._cue_fresh():
                    distance = float(np.linalg.norm(
                        self.p_d[:2] - self.cue[:2]))
                    self._enter_precland(distance)
                else:
                    self._hold_pos = self.p_d.copy()
                    self._set_state('READY', f'(PX4 takeoff at {self.p_d[2]:.1f} m)')
            return

        if self.state == 'READY':
            self._send(self._hold_pos)
            return

        if self.state == 'HOVER':
            # HOVER is a mission/command state, not a controller handoff.
            # Keep the terminal TrackingMPC P/V/A reference so its braking
            # feed-forward is not replaced by a position-only setpoint.
            if not self._follow_path():
                if getattr(self, '_path_failure_hold', None) is None:
                    self._path_failure_hold = self.p_d.copy()
                self._hold_pos = self._path_failure_hold.copy()
                self._send_goto(self._hold_pos)
            return

        if self.state in ('MISSION_PLAN', 'RETURN_PLAN'):
            return_route = self.state == 'RETURN_PLAN'
            rolling_path = (
                return_route
                and getattr(self, '_mission_path', None) is not None)
            rolling_path_safe = False
            if (return_route
                    and not getattr(self, '_return_staging', False)
                    and self._landing_mpc_entry_ready()
                    and self._enter_landing_mpc()):
                self._plan_future.cancel()
                self._plan_future = None
                return
            if rolling_path:
                rolling_path_safe = self._follow_path()
                if not rolling_path_safe:
                    if getattr(self, '_path_failure_hold', None) is None:
                        self._path_failure_hold = self.p_d.copy()
                    self._hold_pos = self._path_failure_hold.copy()
                    self._send_goto(self._hold_pos)
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
            # The two-second rolling cooldown starts when a worker finishes,
            # not when it started.  At 5x the worker itself takes 4--11 sim-s;
            # start-based timing otherwise launches the next plan immediately.
            plan_completed_t = self._now()
            if return_route:
                self._last_return_plan_t = plan_completed_t
            try:
                plan_result = self._plan_future.result()
                if len(plan_result) == 4:
                    arc_m, path, expanded, diagnostics = plan_result
                else:  # Compatibility with focused tests and old workers.
                    arc_m, path, expanded = plan_result
                    diagnostics = None
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
            plan_age = max(
                0.0, plan_completed_t
                - float(getattr(
                    self, '_plan_started_t', plan_completed_t)))
            # The outbound city plan is the first measured worker latency.
            # Keep the worst observation so a short lucky solve cannot shrink
            # the next moving-target lead below the 5x compute requirement.
            self._return_plan_lead_s = max(
                float(getattr(self, '_return_plan_lead_s', 0.0)),
                plan_age)
            intercept_deadline = None
            if return_route and not getattr(self, '_return_staging', False):
                # Learn the actual planner latency.  At accelerated simulation
                # rates a valid city plan can consume far more than 3.5 s of
                # simulation time; recompute the remaining closed-route lead
                # so a waypoint turn does not invalidate the same prediction.
                planned_goal = getattr(self, '_plan_goal', None)
                cue_velocity = getattr(self, 'cue_v', None)
                if (planned_goal is not None and self.cue is not None
                        and cue_velocity is not None):
                    remaining_lead = max(
                        0.0, float(getattr(
                            self, '_plan_goal_lead_s', 0.0)) - plan_age)
                    intercept_deadline = plan_completed_t + remaining_lead
                    expected_endpoint = None
                    if getattr(self, '_terminal_route_cyclic', False):
                        expected_endpoint = _forward_cyclic_route_position(
                            np.asarray(self.cue, float)[:2],
                            np.asarray(cue_velocity, float)[:2],
                            self._terminal_waypoints_local, remaining_lead,
                            self._terminal_waypoint_tolerance,
                            self._terminal_min_cruise_speed,
                            self._terminal_route_match_tolerance)
                    if expected_endpoint is None:
                        expected_endpoint = (
                            np.asarray(self.cue, float)[:2]
                            + np.asarray(cue_velocity, float)[:2]
                            * remaining_lead)
                    endpoint_residual = float(np.linalg.norm(
                        expected_endpoint
                        - np.asarray(planned_goal, float)[:2]))
                    self._plan_endpoint_residual_m = endpoint_residual
                    cue_speed = float(np.linalg.norm(
                        np.asarray(cue_velocity, float)[:2]))
                    residual_limit = max(2.0, 0.75 * cue_speed)
                    if endpoint_residual > residual_limit:
                        self.get_logger().warn(
                            'stale moving-target plan discarded: endpoint '
                            f'residual {endpoint_residual:.2f} m')
                        if rolling_path:
                            if rolling_path_safe:
                                self._set_state(
                                    'RETURN',
                                    '(keep prior route; stale result)')
                            else:
                                self._hold_pos = self.p_d.copy()
                                self._set_state(
                                    'ABORT',
                                    '(stale result and no safe prior route)')
                        else:
                            self._start_global_plan(
                                self.cue, return_route=True)
                        return
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
            try:
                # A rolling replacement includes a new current-to-route splice,
                # so the worker's original corridor cannot certify it.  Legacy
                # three-tuple workers likewise need an active-path corridor.
                if rolling_path or diagnostics is None:
                    arc_m, path, diagnostics = _active_path_sfc(
                        self.mission_map_yaml, path)
                prepared = None
                if rolling_path and getattr(self, '_path_mpc', None) is not None:
                    prepared = MissionManagerNode._shadow_tracking_plan(
                        self, arc_m, path, replacement_progress)
                    if prepared is None:
                        self.get_logger().warn(
                            'rolling replacement rejected by shadow '
                            'TrackingMPC/planning-clearance check')
                        if rolling_path_safe:
                            self._set_state(
                                'RETURN', '(keep prior route; unsafe swap)')
                        else:
                            self._hold_pos = self.p_d.copy()
                            self._set_state(
                                'ABORT', '(no dynamically safe replacement)')
                        return
                MissionManagerNode._commit_active_path(
                    self, arc_m, path, replacement_progress, diagnostics,
                    reset_mpc=not rolling_path)
                if (return_route
                        and not getattr(self, '_return_staging', False)):
                    self._return_intercept_deadline_t = intercept_deadline
                    self._return_intercept_lock_active = False
                if prepared is not None:
                    self._path_mpc, result = prepared
                    zeros = np.zeros(3)
                    self._path_reference.set_plan(
                        self.p_d, self.v_d,
                        result.predicted_pos, result.predicted_vel,
                        result.predicted_acc, self.mpc_dt,
                        zeros, zeros, zeros)
                    self._path_solve_t = self._now()
                    self._path_last_solve_t = self._path_solve_t
            except (KeyError, RuntimeError, ValueError) as exc:
                self.get_logger().error(
                    f'global active-path SFC rejected: {exc}')
                if rolling_path and rolling_path_safe:
                    self._set_state(
                        'RETURN', '(SFC failure; keep prior safe route)')
                    return
                fallback = 'ABORT' if return_route else 'READY'
                self._hold_pos = self.p_d.copy()
                self._set_state(fallback, '(active-path SFC failure; hold)')
                return
            drift_text = ''
            planned_goal = getattr(self, '_plan_goal', None)
            if (return_route and not getattr(self, '_return_staging', False)
                    and planned_goal is not None and self.cue is not None):
                goal_drift = float(getattr(
                    self, '_plan_endpoint_residual_m', math.nan))
                drift_text = f', target drift {goal_drift:.2f} m'
            sfc_time_ms = float(diagnostics.get(
                'sfc_generation_time_ms', math.nan))
            sfc_text = (f', SFC {sfc_time_ms:.3f} ms'
                        if math.isfinite(sfc_time_ms) else '')
            self.get_logger().info(
                f'global A*/B-spline: {len(path)} samples, '
                f'{arc_m[-1]:.1f} m, {expanded} A* expansions, '
                f'{plan_age:.2f} s{drift_text}{sfc_text}')
            # Diagnostic snapshot only: lets path-only runs retain cumulative
            # MPC timing without waiting for touchdown or changing control.
            if hasattr(self, '_mpc_solve_count'):
                MissionManagerNode._log_experiment_metrics(
                    self, final=False)
            next_state = 'RETURN' if return_route else 'MISSION'
            self._set_state(
                next_state, '(validated geometry B-spline -> TrackingMPC)')
            return

        if self.state in ('MISSION', 'RETURN'):
            return_route = self.state == 'RETURN'
            failure_hold = getattr(self, '_path_failure_hold', None)
            if failure_hold is not None:
                self._hold_pos = np.asarray(failure_hold, float).copy()
                self._send_goto(self._hold_pos)
                if not _mission_segment_is_free(
                        self.mission_map_yaml, self.p_d, self.p_d):
                    self._set_state('ABORT', '(left hard clearance; hold)')
                    return
                if float(np.linalg.norm(self.v_d)) > self.settle_v_tol:
                    return
                # Retry the same certified route from rest.  If the vehicle is
                # inside only the planning reserve, _follow_path uses its
                # hard-safe forward rejoin instead of launching an impossible
                # A* from a planning-blocked start.
                self._path_failure_hold = None
            if return_route and getattr(self, '_return_staging', False):
                if (MissionManagerNode._gps_preacquire_entry_ready(self)
                        and self._enter_landing_mpc(gps_preacquire=True)):
                    return
            elif (return_route and self._landing_mpc_entry_ready()
                  and self._enter_landing_mpc()):
                return
            if not self._follow_path():
                hold = getattr(self, '_path_failure_hold', None)
                if hold is None:
                    hold = self.p_d.copy()
                    self._path_failure_hold = hold.copy()
                else:
                    hold = np.asarray(hold, float).copy()
                hold[2] = self.takeoff_alt
                self._hold_pos = hold.copy()
                self._send_goto(hold)
                if not _mission_segment_is_free(
                        self.mission_map_yaml, self.p_d, self.p_d):
                    self._set_state(
                        'ABORT', '(left hard clearance; hold)')
                return
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
                intercept_locked = (
                    retarget_due
                    and MissionManagerNode._return_intercept_is_viable(self))
                was_locked = bool(getattr(
                    self, '_return_intercept_lock_active', False))
                if intercept_locked != was_locked:
                    self._return_intercept_lock_active = intercept_locked
                    if intercept_locked:
                        self.get_logger().info(
                            'moving-target intercept locked: '
                            f'drone ETA '
                            f'{self._return_intercept_drone_eta_s:.1f}s, '
                            f'trailer ETA '
                            f'{self._return_intercept_target_eta_s:.1f}s')
                    else:
                        self.get_logger().info(
                            'moving-target intercept released; replan')
            if (return_route and not getattr(
                    self, '_return_staging', False) and retarget_due
                    and not intercept_locked
                    and _mission_planning_segment_is_free(
                        self.mission_map_yaml, self.p_d, self.p_d)):
                # The paper pipeline reruns all three global stages for every
                # moving-target update.  Keep the prior certified path active
                # in RETURN_PLAN while the process worker computes the latest
                # A* -> optimizer-SFC -> B-spline replacement; commit path and
                # active-path SFC together only after the exact checks pass.
                self.get_logger().info(
                    'moving-target update: schedule rolling '
                    'A* -> SFC -> B-spline')
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
                        'HOVER',
                        '(B-spline mission goal, '
                        f'altitude {self._mission_path[-1][2]:.1f} m)')
                elif getattr(self, '_return_staging', False):
                    self._return_staging_arrived = True
            return

        if self.state in ('LANDING_ACQUIRE', 'LANDING_DESCEND'):
            if nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                now = self._now()
                if self._landing_recovery_since is None:
                    self._landing_recovery_since = now
                self._send(self.p_d, self.v_d, np.zeros(3))
                if (now - self._landing_recovery_since >= 1.0
                        and (self._last_offboard_cmd is None
                             or now - self._last_offboard_cmd >= 1.0)):
                    self._cmd(
                        VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self._last_offboard_cmd = now
                return
            if self._landing_recovery_since is not None:
                # PRECLAND may have descended during the Offboard warm-up.
                # Re-anchor the no-descent floor to the accepted switch state.
                self._landing_hold_z = float(self.p_d[2])
            self._landing_recovery_since = None
            self._run_landing_mpc()
            return

        if self.state == 'PRECLAND':
            now = self._now()
            if getattr(self, '_landing_contact_confirmed', False):
                return
            published = self._publish_landing_target()
            if not published:
                if MissionManagerNode._precland_terminal_latched(self):
                    self.get_logger().warn(
                        'PRECLAND target unavailable below terminal latch; '
                        'keep PX4 authority (no Offboard climb)',
                        throttle_duration_sec=2.0)
                    return
                self._recover_precland()
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
        self._log_experiment_metrics()
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
