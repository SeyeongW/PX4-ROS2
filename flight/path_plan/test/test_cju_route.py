from pathlib import Path

import numpy as np
import pytest
import yaml

from path_plan.cju_route import (
    local_to_map,
    map_to_local,
    plan_route,
    rotation_for_heading,
    route_map_info,
    safe_route_target,
    segment_is_free,
)


MAP = str(Path(__file__).parents[1] / 'config' / 'drone_cju_route.yaml')


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
    assert np.allclose(plan.path_local_xy[0], start, atol=1.0e-6)
    assert np.allclose(plan.path_local_xy[-1], goal, atol=1.0e-6)
    assert np.all(np.diff(plan.arc_m) > 0.0)
    assert all(segment_is_free(
        MAP, origin, a, b)
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
