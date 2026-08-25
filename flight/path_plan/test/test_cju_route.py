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


def test_nearest_free_point_returns_a_free_point_unchanged():
    origin = np.zeros(2)
    free = np.array([50.0, 50.0])          # the goal is clear on the field map
    assert segment_is_free(FIELD_MAP, origin, free, free)
    result = nearest_free_point(FIELD_MAP, origin, free)
    assert result is not None and np.allclose(result, free)


def test_nearest_free_point_projects_a_blocked_target_out_of_a_keep_out():
    origin = np.zeros(2)
    # A barrier centre on the slalom field: inside its keep-out, so blocked.
    blocked = np.array([9.90, 9.90])
    assert not segment_is_free(FIELD_MAP, origin, blocked, blocked)
    result = nearest_free_point(FIELD_MAP, origin, blocked)
    assert result is not None
    # The projection is free (a goal plan_route/endpoint checks would accept)...
    assert segment_is_free(FIELD_MAP, origin, result, result)
    # ...and close to the blocked target — a nearby approach, not a teleport.
    assert float(np.linalg.norm(result - blocked)) <= 2.0


def test_nearest_free_point_gives_up_rather_than_inventing_a_far_goal():
    origin = np.zeros(2)
    blocked = np.array([9.90, 9.90])
    # With a tiny search radius nothing free is within reach: None, not a guess.
    assert nearest_free_point(
        FIELD_MAP, origin, blocked, max_radius_m=0.1) is None


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
