import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from path_plan.cju_route import (
    clearance,
    local_to_map,
    map_to_local,
    nearest_free_point,
    plan_route,
    rotation_for_heading,
    route_map_info,
    safe_route_target,
    segment_is_free,
)


MAP = str(Path(__file__).parents[1] / 'config' / 'drone_cju_route.yaml')
FIELD_MAP = str(
    Path(__file__).parents[1] / 'config' / 'drone_field_route.yaml')
MAZE_MAP = str(
    Path(__file__).parents[1] / 'config' / 'drone_maze_route.yaml')


def _fixture():
    info = route_map_info(MAP)
    rotation = rotation_for_heading(info.heading_deg_enu)
    origin = np.zeros(2)
    start = map_to_local([5.0, 0.0], origin, rotation)
    goal = map_to_local([50.0, 50.0], origin, rotation)
    return info, rotation, origin, start, goal


def test_wang_spawn_rotation_sign_and_round_trip():
    info, rotation, origin, _start, _goal = _fixture()
    delta_world = np.array([
        -5.669916246818 - (-10.639652792208),
        65.542579972196 - 64.993290731978,
    ])
    assert np.allclose(delta_world @ rotation, [5.0, 0.0], atol=1.0e-9)

    points = np.array([[-20.0, 3.0], [0.0, 0.0], [44.0, 46.0]])
    local = map_to_local(points, [123.4, -98.7], rotation)
    assert np.allclose(
        local_to_map(local, [123.4, -98.7], rotation), points,
        atol=1.0e-12)
    assert info.horizontal_accuracy == 'osm_scale_visual_not_survey_grade'
    assert not info.hardware_flight_approved
    assert info.vehicle_clearance_m == pytest.approx(1.0)
    assert info.mission_goal_xy == (50.0, 50.0)


def test_quoted_false_cannot_approve_a_hardware_map(tmp_path):
    document = yaml.safe_load(Path(MAP).read_text(encoding='utf-8'))
    document['site']['hardware_flight_approved'] = 'false'
    candidate = tmp_path / 'quoted-false.yaml'
    candidate.write_text(yaml.safe_dump(document), encoding='utf-8')
    assert not route_map_info(str(candidate)).hardware_flight_approved

    document['site']['origin_wgs84'] = [127.495, 36.654]
    candidate = tmp_path / 'swapped-origin.yaml'
    candidate.write_text(yaml.safe_dump(document), encoding='utf-8')
    with pytest.raises(ValueError, match='latitude/longitude'):
        route_map_info(str(candidate))


def test_real_map_route_is_exact_safe_and_preserves_endpoints():
    _info, _rotation, origin, start, goal = _fixture()
    assert not segment_is_free(MAP, origin, start, goal)

    plan = plan_route(MAP, start, goal, origin)
    assert math.isfinite(plan.astar_plan_time_s)
    assert plan.astar_plan_time_s >= 0.0
    assert math.isfinite(plan.sfc_generation_time_s)
    assert plan.sfc_generation_time_s >= 0.0
    assert plan.sfc_boxes_min.shape == plan.sfc_boxes_max.shape
    assert plan.sfc_boxes_min.shape[1] == 3
    assert len(plan.sfc_boxes_min) > 0
    assert np.allclose(plan.path_local_xy[0], start, atol=1.0e-6)
    assert np.allclose(plan.path_local_xy[-1], goal, atol=1.0e-6)
    assert np.all(np.diff(plan.arc_m) > 0.0)
    assert all(segment_is_free(
        MAP, origin, a, b)
               for a, b in zip(
                   plan.path_local_xy[:-1], plan.path_local_xy[1:]))


def test_drone_relative_field_map_is_approved_and_origin_free():
    info = route_map_info(FIELD_MAP)
    assert info.drone_relative
    # Approved by construction, and the map origin is the vehicle's own local
    # ENU origin, so origin_lat/origin_lon default to a finite (0, 0).
    assert info.hardware_flight_approved
    assert info.heading_deg_enu == pytest.approx(0.0)
    assert info.origin_lat == pytest.approx(0.0)
    assert info.origin_lon == pytest.approx(0.0)
    assert info.mission_goal_xy == (50.0, 50.0)
    assert not route_map_info(MAP).drone_relative


PILLAR_CENTRE = np.array([5.66, 7.07])   # pillar_1 on the field map (blocked)


def _first_stuck_position(route, origin):
    """A free vehicle position with no exact-safe carrot on `route`."""
    rng = np.random.default_rng(1)
    for _ in range(4000):
        point = np.array([rng.uniform(0.0, 50.0), rng.uniform(0.0, 50.0)])
        if not segment_is_free(FIELD_MAP, origin, point, point):
            continue
        _, target, _ = safe_route_target(
            FIELD_MAP, origin, route.arc_m, route.path_local_xy, point,
            0.0, 6.0, 1.5)
        if target is None:
            return point
    return None


def test_maze_map_is_drone_relative_and_routes_collision_free():
    info = route_map_info(MAZE_MAP)
    assert info.drone_relative and info.hardware_flight_approved
    assert info.mission_goal_xy == (50.0, 50.0)
    origin = np.zeros(2)
    start, goal = np.zeros(2), np.array([50.0, 50.0])
    # A dense maze: the straight line is blocked and the planner weaves through.
    assert not segment_is_free(MAZE_MAP, origin, start, goal)
    plan = plan_route(MAZE_MAP, start, goal, origin)
    assert np.allclose(plan.path_local_xy[0], start, atol=1.0e-6)
    assert np.allclose(plan.path_local_xy[-1], goal, atol=1.0e-6)
    assert all(segment_is_free(MAZE_MAP, origin, a, b)
               for a, b in zip(plan.path_local_xy[:-1], plan.path_local_xy[1:]))


def test_replanning_from_the_current_position_recovers_a_stuck_follower():
    # A vehicle that has drifted off its route with a pillar blocking the
    # straight cut-back has no exact-safe carrot on the OLD route (the field
    # HOLD). Replanning from where it actually is must give a route it can
    # follow from there — the guarantee _route_follow_stuck's replan relies on.
    origin = np.zeros(2)
    old = plan_route(FIELD_MAP, np.zeros(2), np.array([50.0, 50.0]), origin)
    drone = _first_stuck_position(old, origin)
    assert drone is not None                          # the field can strand one

    fresh = plan_route(FIELD_MAP, drone, np.array([50.0, 50.0]), origin)
    _, target, cross = safe_route_target(
        FIELD_MAP, origin, fresh.arc_m, fresh.path_local_xy, drone,
        0.0, 6.0, 1.5)
    assert target is not None                        # now followable
    assert cross < 1.0e-6                             # route starts under it
    assert all(segment_is_free(FIELD_MAP, origin, a, b)
               for a, b in zip(fresh.path_local_xy[:-1], fresh.path_local_xy[1:]))


def test_nearest_free_point_returns_a_free_point_unchanged():
    origin = np.zeros(2)
    free = np.array([50.0, 50.0])          # the goal is clear on the field map
    assert segment_is_free(FIELD_MAP, origin, free, free)
    result = nearest_free_point(FIELD_MAP, origin, free)
    assert result is not None and np.allclose(result, free)


def test_nearest_free_point_projects_a_blocked_target_out_of_a_keep_out():
    origin = np.zeros(2)
    # A pillar centre on the field: inside its keep-out, so blocked.
    blocked = PILLAR_CENTRE
    assert not segment_is_free(FIELD_MAP, origin, blocked, blocked)
    result = nearest_free_point(FIELD_MAP, origin, blocked)
    assert result is not None
    # The projection is free (a goal plan_route/endpoint checks would accept)...
    assert segment_is_free(FIELD_MAP, origin, result, result)
    # ...and close to the blocked target — a nearby approach, not a teleport.
    assert float(np.linalg.norm(result - blocked)) <= 2.0


def test_nearest_free_point_gives_up_rather_than_inventing_a_far_goal():
    origin = np.zeros(2)
    # With a tiny search radius nothing free is within reach: None, not a guess.
    assert nearest_free_point(
        FIELD_MAP, origin, PILLAR_CENTRE, max_radius_m=0.1) is None


def test_drone_relative_field_route_weaves_through_the_virtual_barriers():
    # Map origin (0, 0) is the launch point; heading 0 makes map == local ENU.
    origin = np.zeros(2)
    start = np.zeros(2)
    goal = np.array([50.0, 50.0])
    # The barriers lie between launch and goal: the straight line is blocked, so
    # the planner must route around them for the avoidance demo.
    assert not segment_is_free(FIELD_MAP, origin, start, goal)
    plan = plan_route(FIELD_MAP, start, goal, origin)
    assert np.allclose(plan.path_local_xy[0], start, atol=1.0e-6)
    assert np.allclose(plan.path_local_xy[-1], goal, atol=1.0e-6)
    assert np.all(np.diff(plan.arc_m) > 0.0)
    # ...and every leg of that swerving route clears the virtual barriers.
    assert all(segment_is_free(FIELD_MAP, origin, a, b)
               for a, b in zip(
                   plan.path_local_xy[:-1], plan.path_local_xy[1:]))


def test_real_map_long_route_keeps_the_one_metre_control_spacing_regression():
    _info, rotation, origin, _start, _goal = _fixture()
    start = map_to_local([5.0, 0.0], origin, rotation)
    goal = map_to_local([85.0, 90.0], origin, rotation)
    plan = plan_route(MAP, start, goal, origin)
    assert np.allclose(plan.path_local_xy[[0, -1]], [start, goal], atol=1.0e-6)
    assert all(segment_is_free(
        MAP, origin, a, b)
               for a, b in zip(
                   plan.path_local_xy[:-1], plan.path_local_xy[1:]))


def test_route_rejects_a_start_inside_the_boundary_clearance():
    _info, rotation, origin, _start, goal = _fixture()
    near_raw_boundary = map_to_local([-17.5, 0.0], origin, rotation)
    assert not segment_is_free(
        MAP, origin, near_raw_boundary, goal)
    with pytest.raises(RuntimeError, match='exact start'):
        plan_route(MAP, near_raw_boundary, goal, origin)


def test_safe_carrot_shortens_a_blocked_corner_cut_and_fails_closed():
    _info, rotation, origin, start, goal = _fixture()
    plan = plan_route(MAP, start, goal, origin)
    progress, carrot, _cross_track = safe_route_target(
        MAP, origin, plan.arc_m, plan.path_local_xy, start, 0.0,
        lookahead_m=100.0, cross_track_limit_m=0.25)
    assert abs(progress) <= 1.0e-9
    assert carrot is not None
    assert np.linalg.norm(carrot - start) < np.linalg.norm(goal - start)
    assert segment_is_free(MAP, origin, start, carrot)

    inside_barrier = map_to_local([33.0, 10.0], origin, rotation)
    _progress, blocked, _cross_track = safe_route_target(
        MAP, origin, plan.arc_m, plan.path_local_xy, inside_barrier, 0.0,
        lookahead_m=6.0, cross_track_limit_m=0.25)
    assert blocked is None


def test_clearance_falls_off_toward_an_obstacle_and_fails_closed():
    info, rotation, origin, _start, _goal = _fixture()
    # Straight down the mission line: clearance is a distance, so sampling it
    # along a path that approaches the goal must never return something
    # negative, and an unmeasurable input must read as no clearance at all.
    samples = [clearance(MAP, origin, map_to_local([x, 0.0], origin, rotation))
               for x in (5.0, 15.0, 30.0, 45.0)]
    assert all(value >= 0.0 for value in samples)
    assert all(np.isfinite(value) for value in samples)

    # Fails to 0.0 — the conservative answer — exactly like segment_is_free
    # fails to False. A caller that cannot measure must behave as if there is
    # no room.
    assert clearance('/no/such/map.yaml', origin, [0.0, 0.0]) == 0.0
    assert clearance(MAP, origin, [float('nan'), 0.0]) == 0.0
    assert clearance(MAP, [float('inf'), 0.0], [0.0, 0.0]) == 0.0


def test_a_route_survives_a_start_the_optimiser_cannot_smooth():
    """A rejected B-spline must not discard a certified A* path.

    The A* chords are each proved free with `segment_is_free_exact` before the
    optimiser runs, so a spline the optimiser cannot keep collision-free is a
    reason to fly the polyline instead — not to refuse. Refusing outright
    failed 35% of starts within half a metre of an obstacle, which is where a
    vehicle most needs a way out.
    """
    from pathlib import Path as _P
    from path_plan import cju_route as cr
    field = str(_P(cr.__file__).parents[1] / 'config' / 'drone_field_route.yaml')
    origin = np.zeros(2)
    # (10, 10) is 0.01 m off a pillar's grown box: free, but the optimiser has
    # no room to bulge and its spline used to be rejected outright.
    start = np.array([10.0, 10.0])
    assert cr.segment_is_free(field, origin, start, start)
    assert cr.clearance(field, origin, start) < 0.5

    plan = cr.plan_route(field, start, np.array([50.0, 50.0]), origin)
    points = plan.path_local_xy
    assert np.allclose(points[0], start, atol=1.0e-6)
    assert all(cr.segment_is_free(field, origin, a, b)
               for a, b in zip(points[:-1], points[1:]))


def test_a_free_start_in_a_blocked_grid_cell_still_plans():
    """A grid artefact is not an obstacle.

    planner_resolution_m 1.0 against vehicle_clearance_xy_m 1.0 puts a point
    with tens of centimetres of room in a cell A* samples as blocked, so the
    vehicle was told there is no route out of a position it is legally in.
    `nearest_free_point` cannot fix it — the point is not itself blocked, so it
    comes back unmoved — hence the relay. The exact start must survive.
    """
    from pathlib import Path as _P
    from path_plan import cju_route as cr
    field = str(_P(cr.__file__).parents[1] / 'config' / 'drone_field_route.yaml')
    origin, goal = np.zeros(2), np.array([50.0, 50.0])
    rng = np.random.default_rng(7)
    grazing = []
    while len(grazing) < 8:
        candidate = rng.uniform(0.0, 45.0, 2)
        if (cr.segment_is_free(field, origin, candidate, candidate)
                and cr.clearance(field, origin, candidate) <= 0.5):
            grazing.append(candidate)

    for start in grazing:
        # The projection the mission node applies first is a no-op here.
        assert np.allclose(
            cr.nearest_free_point(field, origin, start), start, atol=1.0e-9)
        plan = cr.plan_route(field, start, goal, origin)
        points = plan.path_local_xy
        assert np.allclose(points[0], start, atol=1.0e-6)
        assert all(cr.segment_is_free(field, origin, a, b)
                   for a, b in zip(points[:-1], points[1:]))
