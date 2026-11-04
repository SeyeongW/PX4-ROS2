"""ROS-free CJU route generation for the hardware MAVROS mission.

Only geometry lives here: site-frame transforms, A* -> SFC -> B-spline, and a
collision-checked spatial carrot.  The module does not import ROS, MAVROS,
``px4_msgs`` or Gazebo.  Flight authority stays in ``aruco_landing_node``.

The route is horizontal.  Its z value is used only to query the 3-D planning
world; the real vehicle keeps the takeoff-relative altitude chosen by the
hardware mission node.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .astar import AStarPlanner3D
from .bspline_optimizer import BsplineOptimizer
from .sfc import SafeFlightCorridor
from .uniform_bspline import UniformBspline
from .world_model import WorldModel


@dataclass(frozen=True)
class RouteMapInfo:
    """Hardware-facing metadata from one immutable map snapshot."""

    name: str
    origin_lat: float
    origin_lon: float
    heading_deg_enu: float
    horizontal_accuracy: str
    hardware_flight_approved: bool
    vehicle_clearance_m: float
    mission_goal_xy: tuple[float, float]
    # True for an obstacle-free, drone-relative field map: the map origin is the
    # vehicle's own local ENU origin (0, 0), so no WGS84 survey is needed and the
    # hardware node anchors the map on the drone rather than on a fixed site.
    drone_relative: bool = False


@dataclass(frozen=True)
class RoutePlan:
    """One atomically accepted local-ENU horizontal route."""

    arc_m: np.ndarray
    path_local_xy: np.ndarray
    expanded_nodes: int
    # Observation metadata only; it never participates in route acceptance or
    # control. NaN preserves callers that construct a spliced RoutePlan.
    astar_plan_time_s: float = math.nan
    sfc_generation_time_s: float = math.nan
    sfc_boxes_min: np.ndarray | None = None
    sfc_boxes_max: np.ndarray | None = None


def _finite_vector(name: str, values, size: int) -> np.ndarray:
    vector = np.asarray(values, float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must contain {size} finite values')
    return vector


def _positive(name: str, value) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return number


@lru_cache(maxsize=8)
def _document(map_yaml: str) -> dict:
    path = Path(map_yaml).expanduser().resolve()
    document = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict):
        raise ValueError('route map must be a YAML mapping')
    if int(document.get('schema_version', 0)) != 1:
        raise ValueError('route map schema_version must be 1')
    return document


def rotation_for_heading(heading_deg_enu: float) -> np.ndarray:
    """Row-vector ENU -> site rotation (a passive rotation by ``-heading``)."""
    heading = math.radians(float(heading_deg_enu))
    if not math.isfinite(heading):
        raise ValueError('site heading must be finite')
    return np.array([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)],
    ])


def local_to_map(local_xy, site_origin_local_xy, rotation) -> np.ndarray:
    """Convert MAVROS local ENU XY to the track-aligned site frame."""
    points = np.asarray(local_xy, float)
    origin = _finite_vector('site_origin_local_xy', site_origin_local_xy, 2)
    matrix = np.asarray(rotation, float)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError('local_xy must contain finite XY points')
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError('rotation must be a finite 2x2 matrix')
    return (points - origin) @ matrix


def map_to_local(map_xy, site_origin_local_xy, rotation) -> np.ndarray:
    """Convert track-aligned site XY back to MAVROS local ENU."""
    points = np.asarray(map_xy, float)
    origin = _finite_vector('site_origin_local_xy', site_origin_local_xy, 2)
    matrix = np.asarray(rotation, float)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError('map_xy must contain finite XY points')
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError('rotation must be a finite 2x2 matrix')
    return points @ matrix.T + origin


@lru_cache(maxsize=8)
def route_map_info(map_yaml: str) -> RouteMapInfo:
    document = _document(str(map_yaml))
    site = document['site']
    mission = document['mission']
    # A drone-relative map has no fixed WGS84 site: the map origin is wherever
    # the vehicle launches, expressed as (0, 0) in its own local ENU frame. The
    # origin_wgs84 field is then optional and, if absent, defaults to (0, 0) so
    # that origin_lat/origin_lon stay finite for any caller that reads them.
    drone_relative = site.get('drone_relative', False) is True
    if drone_relative and 'origin_wgs84' not in site:
        origin = np.zeros(2)
    else:
        origin = _finite_vector('site.origin_wgs84', site['origin_wgs84'], 2)
        if not (-90.0 <= origin[0] <= 90.0
                and -180.0 <= origin[1] <= 180.0):
            raise ValueError(
                'site.origin_wgs84 is outside latitude/longitude bounds')
    heading = float(site['heading_deg_enu'])
    rotation_for_heading(heading)
    goal = _finite_vector('mission.goal_m', mission['goal_m'], 2)
    return RouteMapInfo(
        name=str(site.get('name', 'unnamed site')),
        origin_lat=float(origin[0]),
        origin_lon=float(origin[1]),
        heading_deg_enu=heading,
        horizontal_accuracy=str(site.get('horizontal_accuracy', 'unknown')),
        # Only a literal YAML boolean can open the props-on gate. In
        # particular, quoted strings such as "false" are truthy in Python.
        hardware_flight_approved=(
            site.get('hardware_flight_approved', False) is True),
        vehicle_clearance_m=_positive(
            'mission.vehicle_clearance_xy_m',
            mission['vehicle_clearance_xy_m']),
        mission_goal_xy=(float(goal[0]), float(goal[1])),
        drone_relative=drone_relative,
    )


@lru_cache(maxsize=24)
def _world(map_yaml: str, *, z_half_width: float) \
        -> tuple[dict, np.ndarray, float, WorldModel]:
    document = _document(str(map_yaml))
    mission = document['mission']
    terrain = document['terrain']
    if terrain['coordinate_frame'] != site_frame(document):
        raise ValueError('mission and terrain must use the site frame')

    altitude = _positive('mission.cruise_altitude_m',
                         mission['cruise_altitude_m'])
    clearance = _positive('mission.vehicle_clearance_xy_m',
                          mission['vehicle_clearance_xy_m'])

    lows, highs = [], []
    for obstacle in mission.get('obstacles', []):
        center = _finite_vector(
            f"obstacle {obstacle.get('name', '?')} center_m",
            obstacle['center_m'], 3)
        size = _finite_vector(
            f"obstacle {obstacle.get('name', '?')} size_m",
            obstacle['size_m'], 3)
        if np.any(size <= 0.0):
            raise ValueError('obstacle sizes must be positive')
        half = 0.5 * size
        low, high = center - half, center + half
        # No overflight: these barriers are lateral keep-outs for this mission.
        low[2], high[2] = -1.0e4, 1.0e4
        lows.append(low)
        highs.append(high)

    center = _finite_vector('terrain.center_m', terrain['center_m'], 2)
    size = _finite_vector('terrain.size_m', terrain['size_m'], 2)
    if np.any(size <= 0.0):
        raise ValueError('terrain.size_m must be positive')
    half = 0.5 * size
    if not math.isfinite(z_half_width) or z_half_width < 0.0:
        raise ValueError('z_half_width must be finite and non-negative')
    bounds_min_xy = center - half + clearance
    bounds_max_xy = center + half - clearance
    if np.any(bounds_min_xy >= bounds_max_xy):
        raise ValueError('terrain is too small for the configured clearance')
    world = WorldModel.from_boxes(
        lows, highs,
        [*bounds_min_xy, altitude - z_half_width],
        [*bounds_max_xy, altitude + z_half_width],
        xy_clearance_m=clearance,
    )
    rotation = rotation_for_heading(
        float(document['site']['heading_deg_enu']))
    return mission, rotation, altitude, world


def site_frame(document: dict) -> str:
    frame = str(document['site']['coordinate_frame'])
    if str(document['mission']['coordinate_frame']) != frame:
        raise ValueError('site and mission coordinate frames differ')
    return frame


def plan_route(map_yaml: str, start_local_xy, goal_local_xy,
               site_origin_local_xy) -> RoutePlan:
    """Generate one exact-safe A* -> geometry B-spline local-ENU route."""
    start_local = _finite_vector('start_local_xy', start_local_xy, 2)
    goal_local = _finite_vector('goal_local_xy', goal_local_xy, 2)
    origin_local = _finite_vector(
        'site_origin_local_xy', site_origin_local_xy, 2)
    mission, rotation, altitude, planner_world = _world(
        str(map_yaml), z_half_width=0.0)
    _, _, _, spline_world = _world(
        str(map_yaml), z_half_width=0.5)

    start = np.r_[local_to_map(start_local, origin_local, rotation), altitude]
    goal = np.r_[local_to_map(goal_local, origin_local, rotation), altitude]
    if np.linalg.norm(goal[:2] - start[:2]) <= 1.0e-6:
        raise RuntimeError('route start and goal have no spatial separation')
    if not bool(planner_world.is_free(start)[0]):
        raise RuntimeError('route exact start is blocked or outside the map')
    if not bool(planner_world.is_free(goal)[0]):
        raise RuntimeError('route exact goal is blocked or outside the map')

    resolution = _positive(
        'mission.planner_resolution_m', mission['planner_resolution_m'])
    planner = AStarPlanner3D(
        planner_world,
        resolution_m=resolution,
        # The one hard vehicle clearance is the complete geometry contract.
        # Do not add a second soft path-clearance preference here.
        clearance_weight=0.0,
        clearance_pref_m=0.0,
        altitude_pref_m=altitude,
        heuristic_weight=1.0,
        exact_edges=True,
    )
    astar_started = time.perf_counter()
    result = planner.plan(start, goal)
    # A FREE POINT IN A BLOCKED CELL IS A GRID ARTEFACT, NOT AN OBSTACLE. The
    # exact start was proved free above, but A* tests the CELL it falls in, and
    # at planner_resolution_m 1.0 with vehicle_clearance_xy_m 1.0 a point with
    # a few tens of centimetres of room sits in a cell whose sample is inside
    # the grown obstacle. The vehicle is then told there is no route out of a
    # position it is legally in — 18% of starts within half a metre of an
    # obstacle, which is exactly where it most needs one.
    #
    # `nearest_free_point` cannot help: it moves a point that is itself
    # blocked, and this one is not, so it returns the same point unmoved. Reach
    # the grid through a relay whose whole cell is clear instead. The exact
    # start stays the first waypoint below, so the chord to the relay is
    # certified by the same `segment_is_free_exact` loop as every other chord.
    if not result.success and 'start cell blocked' in result.message:
        relay = _cell_free_relay(planner_world, start, resolution)
        if relay is not None:
            result = planner.plan(relay, goal)
    astar_plan_time_s = time.perf_counter() - astar_started
    if not result.success:
        raise RuntimeError(f'route A* failed: {result.message}')

    points = [start]
    for point in result.waypoints_m:
        if not np.allclose(points[-1], point, atol=1.0e-9, rtol=0.0):
            points.append(np.asarray(point, float))
    if np.allclose(points[-1], goal, atol=1.0e-9, rtol=0.0):
        points[-1] = goal
    else:
        points.append(goal)
    waypoints = np.asarray(points, float)
    if not all(planner_world.segment_is_free_exact(a, b)
               for a, b in zip(waypoints[:-1], waypoints[1:])):
        raise RuntimeError('route A* returned an unsafe exact-endpoint chord')

    control_spacing = _positive(
        'mission.bspline_control_spacing_m',
        mission['bspline_control_spacing_m'])
    optimizer = BsplineOptimizer(
        spline_world,
        cruise_speed_m_s=None,
        ctrl_spacing_m=control_spacing,
        max_acc=3.0,
        lambda_smooth=1.0,
        lambda_dist=0.5,
        lambda_feas=1.0,
        lambda_fit=0.2,
        strict_validation=True,
    )
    optimized = optimizer.optimize(waypoints)
    # SMOOTHING IS AN IMPROVEMENT, NOT A PRECONDITION. The A* chords above are
    # already certified one by one with `segment_is_free_exact`, so a rejected
    # spline means the smoothed curve would have cut a corner the polyline does
    # not — a reason to fly the polyline, not to abandon a route that was
    # proved safe a dozen lines earlier. Refusing outright failed 35% of starts
    # within half a metre of an obstacle, exactly where the optimiser has least
    # room to bulge and exactly where the vehicle most needs a way out.
    #
    # The fallback is not a relaxation: `positions` still goes through the same
    # continuous collision check below, and the corners cost comfort rather
    # than safety — TrackingMPC tracks it under the same jerk bound either way.
    sample_spacing = _positive(
        'mission.bspline_sample_spacing_m',
        mission['bspline_sample_spacing_m'])
    if sample_spacing > 0.25:
        raise ValueError('mission.bspline_sample_spacing_m must be <= 0.25')
    guide_length = float(np.linalg.norm(
        np.diff(waypoints, axis=0), axis=1).sum())

    if optimized.accepted:
        control_points = optimized.spline.q.copy()
        control_points[:, 2] = altitude
        spline = UniformBspline(control_points, optimized.spline.ts)
        dense_count = max(200, int(math.ceil(
            guide_length / sample_spacing)) * 4)
        _, dense, _, _ = spline.sample(dense_count)
        source = 'B-spline'
    else:
        # SMOOTHING IS AN IMPROVEMENT, NOT A PRECONDITION. The A* chords above
        # are each certified with `segment_is_free_exact`, so a rejected spline
        # means the SMOOTHED curve would cut a corner the polyline does not —
        # a reason to fly the polyline, not to discard a route proved safe a
        # dozen lines earlier. Refusing outright failed 35% of starts within
        # half a metre of an obstacle: exactly where the optimiser has least
        # room to bulge, and exactly where the vehicle most needs a way out.
        #
        # This is not a relaxation. The polyline goes through the same
        # continuous collision check below and the same corridor cover; the
        # corners cost comfort, not clearance, and TrackingMPC tracks it under
        # the same jerk bound either way.
        dense = np.asarray(waypoints, float).copy()
        source = 'A* polyline (B-spline rejected)'
    dense[:, 2] = altitude
    dense_arc = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(dense, axis=0), axis=1))]
    keep = np.r_[True, np.diff(dense_arc) > 1.0e-9]
    dense_arc, dense = dense_arc[keep], dense[keep]
    if len(dense_arc) < 2 or dense_arc[-1] <= 0.0:
        raise RuntimeError(f'route {source} has no spatial extent')
    sample_arc = np.r_[np.arange(0.0, dense_arc[-1], sample_spacing),
                       dense_arc[-1]]
    positions = np.column_stack([
        np.interp(sample_arc, dense_arc, dense[:, axis])
        for axis in range(3)])
    positions[0], positions[-1] = start, goal
    positions[:, 2] = altitude
    if not all(spline_world.segment_is_free_exact(a, b)
               for a, b in zip(positions[:-1], positions[1:])):
        raise RuntimeError(
            f'route {source} failed continuous collision checking')

    positions, _corridor = SafeFlightCorridor(
        spline_world).cover_polyline(positions)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(positions[:, :2], axis=0), axis=1))]
    local_path = map_to_local(positions[:, :2], origin_local, rotation)
    if (len(local_path) < 2 or not np.all(np.isfinite(local_path))
            or not np.all(np.diff(arc) > 0.0)
            or not np.allclose(local_path[0], start_local, atol=1.0e-6)
            or not np.allclose(local_path[-1], goal_local, atol=1.0e-6)):
        raise RuntimeError('route spatial contract failed')
    return RoutePlan(
        arc, local_path, int(result.expanded), astar_plan_time_s,
        optimized.sfc_generation_time_s,
        optimized.corridor.boxes_min, optimized.corridor.boxes_max)


def _cell_free_relay(world, start, resolution_m: float):
    """Nearest point A* can start from, reachable in a straight line.

    Only for a start that is itself free but lands in a blocked grid cell. The
    relay must clear the obstacles by more than the cell's own half-diagonal,
    so that whatever cell it falls in is clear however the grid is phased, and
    the chord from the true start must be free — otherwise the route would
    begin with a segment nothing certified.

    Returns None when nothing nearby qualifies, which leaves A*'s own refusal
    to stand rather than inventing a start.
    """
    start = np.asarray(start, float)
    needed = float(resolution_m) * math.sqrt(3.0) / 2.0
    for radius in np.arange(resolution_m, 4.0 * resolution_m + 1e-9,
                            0.5 * resolution_m):
        best = None
        for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            candidate = start + np.array([
                radius * math.cos(angle), radius * math.sin(angle), 0.0])
            if world.clearance(candidate) <= needed:
                continue
            if not world.segment_is_free_exact(start, candidate):
                continue
            score = world.clearance(candidate)
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None:
            return best[1]
    return None


def segment_is_free(map_yaml: str, site_origin_local_xy, start_local_xy,
                    goal_local_xy) -> bool:
    """Check one local-ENU horizontal chord against the immutable map."""
    try:
        origin = _finite_vector(
            'site_origin_local_xy', site_origin_local_xy, 2)
        points = np.vstack((
            _finite_vector('start_local_xy', start_local_xy, 2),
            _finite_vector('goal_local_xy', goal_local_xy, 2)))
        _, rotation, altitude, world = _world(
            str(map_yaml), z_half_width=0.0)
        mapped = local_to_map(points, origin, rotation)
        return world.segment_is_free_exact(
            [*mapped[0], altitude], [*mapped[1], altitude])
    except (KeyError, TypeError, ValueError):
        return False


@lru_cache(maxsize=8)
def obstacle_boxes(map_yaml: str) -> tuple[tuple[str, tuple[float, float, float],
                                                tuple[float, float, float]], ...]:
    """The mission's obstacles as (name, centre_xyz, size_xyz) in the SITE frame.

    Read from the document rather than from `_world`, because `_world` stretches
    every obstacle to +-10 km in z to forbid overflight. That is the right box
    to plan against and the wrong one to draw: a viewer that showed it would
    render 25 walls to the stratosphere instead of the 10 m poles the field is
    supposed to represent.
    """
    document = _document(str(map_yaml))
    out = []
    for index, obstacle in enumerate(document['mission'].get('obstacles', [])):
        name = str(obstacle.get('name', f'obstacle_{index}'))
        centre = _finite_vector(f'obstacle {name} center_m',
                                obstacle['center_m'], 3)
        size = _finite_vector(f'obstacle {name} size_m', obstacle['size_m'], 3)
        if np.any(size <= 0.0):
            raise ValueError('obstacle sizes must be positive')
        out.append((name, tuple(map(float, centre)), tuple(map(float, size))))
    return tuple(out)


def clearance(map_yaml: str, site_origin_local_xy, local_xy) -> float:
    """Distance from one local-ENU point to the nearest obstacle, in metres.

    Measured BEYOND the map's own `vehicle_clearance_xy_m`, because that is
    what `WorldModel` subtracts: 0 here means the vehicle is exactly on the
    hard margin the route was certified against, not touching a wall.

    Fails to 0.0 — the most conservative answer — for the same reason
    `segment_is_free` fails to False: a caller that cannot measure the
    clearance must behave as though there is none. `_world` is lru_cached, so
    this is cheap enough to ask on every control tick.
    """
    try:
        origin = _finite_vector('site_origin_local_xy', site_origin_local_xy, 2)
        point = _finite_vector('local_xy', local_xy, 2)
        _, rotation, altitude, world = _world(str(map_yaml), z_half_width=0.0)
        mapped = local_to_map(point[None, :], origin, rotation)[0]
        return float(world.clearance([*mapped, altitude]))
    except (KeyError, TypeError, ValueError, OSError):
        # OSError too, unlike `segment_is_free`: this one is asked on every
        # control tick, and a control loop is the wrong place to raise from.
        return 0.0


def nearest_free_point(
        map_yaml: str, site_origin_local_xy, local_xy, *,
        max_radius_m: float = 5.0, ring_step_m: float = 0.25,
        angular_samples: int = 24,
) -> np.ndarray | None:
    """The nearest collision-free local-ENU point to ``local_xy``, or None.

    Returns ``local_xy`` unchanged when it is already free. Otherwise it searches
    outward on rings and returns the closest free point — the point a moving
    target that has wandered into a (virtual) keep-out can still be approached
    to. None means no free point within ``max_radius_m``; the caller keeps its
    own fallback (hold / reject) rather than inventing a goal.

    Uses the same ``segment_is_free`` point test as the endpoint gate, so a goal
    this returns is one ``plan_route`` and ``_route_endpoint_reason`` accept. All
    failure modes collapse to None: a caller that cannot project safely must not
    fly at an unprojected blocked point.
    """
    try:
        origin = _finite_vector('site_origin_local_xy', site_origin_local_xy, 2)
        point = _finite_vector('local_xy', local_xy, 2)
    except (TypeError, ValueError):
        return None
    if segment_is_free(map_yaml, origin, point, point):
        return point.copy()
    if not (math.isfinite(max_radius_m) and max_radius_m > 0.0
            and math.isfinite(ring_step_m) and ring_step_m > 0.0
            and int(angular_samples) >= 4):
        return None
    radius = ring_step_m
    while radius <= max_radius_m + 1.0e-9:
        for k in range(int(angular_samples)):
            angle = 2.0 * math.pi * k / int(angular_samples)
            candidate = point + radius * np.array(
                [math.cos(angle), math.sin(angle)])
            if segment_is_free(map_yaml, origin, candidate, candidate):
                return candidate
        radius += ring_step_m
    return None


def _path_position(arc_m, path_xy, distance_m: float) -> np.ndarray:
    arc = np.asarray(arc_m, float)
    path = np.asarray(path_xy, float)
    distance = float(np.clip(distance_m, arc[0], arc[-1]))
    return np.array([
        np.interp(distance, arc, path[:, axis]) for axis in range(2)])


def _spatial_target(arc_m, path_xy, position_xy, progress_m: float,
                    lookahead_m: float, cross_track_limit_m: float):
    arc = np.asarray(arc_m, float)
    path = np.asarray(path_xy, float)
    current = _finite_vector('position_xy', position_xy, 2)
    if (path.ndim != 2 or path.shape[1] != 2 or len(path) != len(arc)
            or len(path) < 2 or not np.all(np.isfinite(path))
            or not np.all(np.isfinite(arc)) or not np.all(np.diff(arc) > 0.0)):
        raise ValueError('route must be a finite Nx2 path with increasing arc')
    lookahead = _positive('lookahead_m', lookahead_m)
    cross_limit = _positive('cross_track_limit_m', cross_track_limit_m)
    progress = float(np.clip(progress_m, 0.0, arc[-1]))
    if progress >= arc[-1]:
        return progress, path[-1].copy(), float(np.linalg.norm(
            current - path[-1]))

    window = max(2.0 * lookahead, 2.0 * cross_limit)
    first = max(0, int(np.searchsorted(arc, progress, side='right')) - 1)
    last = min(len(arc) - 1, int(np.searchsorted(
        arc, min(arc[-1], progress + window), side='right')))
    best_distance = math.inf
    candidate = progress
    for index in range(first, last):
        a, b = path[index], path[index + 1]
        delta = b - a
        length2 = float(delta @ delta)
        if length2 <= 0.0:
            continue
        fraction = float(np.clip(
            (current - a) @ delta / length2, 0.0, 1.0))
        projection = a + fraction * delta
        distance = float(np.linalg.norm(current - projection))
        if distance < best_distance:
            best_distance = distance
            candidate = max(
                progress,
                float(arc[index] + fraction * (arc[index + 1] - arc[index])))
    progress = candidate
    tracking_fraction = float(np.clip(
        1.0 - best_distance / cross_limit, 0.0, 1.0))
    target_s = min(progress + lookahead * tracking_fraction, float(arc[-1]))
    return progress, _path_position(arc, path, target_s), best_distance


def safe_route_target(
        map_yaml: str, site_origin_local_xy, arc_m, path_local_xy,
        position_local_xy, progress_m: float, lookahead_m: float,
        cross_track_limit_m: float,
) -> tuple[float, np.ndarray | None, float]:
    """Return the furthest collision-checked carrot on an accepted route."""
    progress, target, cross_track = _spatial_target(
        arc_m, path_local_xy, position_local_xy, progress_m,
        lookahead_m, cross_track_limit_m)

    def checker(point):
        return segment_is_free(
            map_yaml, site_origin_local_xy, position_local_xy, point)
    if checker(target):
        return progress, target, cross_track
    for offset in np.linspace(float(lookahead_m), 0.0, 21)[1:]:
        candidate = _path_position(
            arc_m, path_local_xy,
            min(progress + offset, float(np.asarray(arc_m)[-1])))
        if checker(candidate):
            return progress, candidate, cross_track
    return progress, None, cross_track


def splice_route_from_current(
        arc_m, path_local_xy, current_local_xy, lookahead_m: float,
        segment_check: Callable[[np.ndarray, np.ndarray], bool],
) -> RoutePlan | None:
    """Join a completed plan without commanding back to its old start."""
    arc = np.asarray(arc_m, float)
    path = np.asarray(path_local_xy, float)
    current = _finite_vector('current_local_xy', current_local_xy, 2)
    projection, _, _ = _spatial_target(
        arc, path, current, 0.0, float(arc[-1]), 1.0)
    join_s = min(projection + 0.5 * float(lookahead_m), float(arc[-1]))
    join = _path_position(arc, path, join_s)
    if not segment_check(current, join):
        return None
    tail = path[arc > join_s + 1.0e-9]
    joined = np.vstack((current, join, tail))
    keep = np.r_[True, np.linalg.norm(
        np.diff(joined, axis=0), axis=1) > 1.0e-9]
    joined = joined[keep]
    if len(joined) < 2:
        return None
    joined_arc = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(joined, axis=0), axis=1))]
    if (not np.all(np.isfinite(joined))
            or not np.all(np.diff(joined_arc) > 0.0)):
        return None
    return RoutePlan(joined_arc, joined, 0)
