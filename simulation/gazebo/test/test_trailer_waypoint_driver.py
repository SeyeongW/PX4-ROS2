import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import yaml


GAZEBO = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    'trailer_waypoint_driver', GAZEBO / 'trailer_waypoint_driver.py')
DRIVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DRIVER
SPEC.loader.exec_module(DRIVER)


def test_stop_speed_requires_both_pose_and_command_to_settle():
    assert DRIVER.conservative_stop_speed(None, 1.0) == 1.0
    assert DRIVER.conservative_stop_speed(0.0, 1.0) == 1.0
    assert DRIVER.conservative_stop_speed(1.0, 0.0) == 1.0
    assert DRIVER.conservative_stop_speed(0.1, 0.15) == 0.15


def test_city_launcher_starts_the_trailer_only_after_takeoff():
    launcher = (GAZEBO / 'run_px4_map.sh').read_text(encoding='utf-8')
    gate = "ros2 topic echo --filter 'm.landed_state in (2, 3)' --once"

    assert '"$MAP" == "city" && -z "${TRAILER_START_FILE:-}"' in launcher
    assert 'city trailer takeoff gate needs START_MAVROS=1' in launcher
    assert gate in launcher
    assert launcher.index(gate) < launcher.index(
        'exec python3 -u "$SCRIPT_DIR/trailer_waypoint_driver.py"')


def test_linear_shuttle_repeats_fifty_metres_forward_then_reverse():
    document = yaml.safe_load(
        (GAZEBO / 'maps/drone_cju_track.yaml').read_text(encoding='utf-8'))
    trailer = document['trailer']
    endpoints = trailer['shuttle_endpoints_enu_m']
    route = DRIVER.LinearShuttleRoute(
        endpoints,
        trailer['shuttle_leg_length_m'],
        trailer['turnaround_creep_speed_m_s'],
        trailer['turnaround_brake_margin_m'],
    )

    assert math.isclose(route.leg_length_m, 50.0, abs_tol=1.0e-6)
    frame = document['frames']['stadium_endpoint']
    heading = math.radians(frame['heading_deg_enu'])
    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])
    origin = np.asarray(frame['origin_enu_m'][:2], float)
    map_endpoints = (np.asarray(endpoints) - origin) @ rotation
    assert np.allclose(map_endpoints, [[5.0, 0.0], [5.0, 50.0]],
                       atol=1.0e-6)
    assert np.allclose(map_endpoints, np.round(map_endpoints), atol=1.0e-9)

    speed = trailer['cruise_speed_m_s']
    forward = route.command(*route.start, speed, trailer['acceleration_m_s2'])
    forward_body = DRIVER.world_to_model_xy(
        *forward, trailer['spawn_pose_enu']['yaw'])
    assert np.allclose(forward_body, [1.0, 0.0], atol=1.0e-6)

    reverse = route.command(*route.end, speed, trailer['acceleration_m_s2'])
    reverse_body = DRIVER.world_to_model_xy(
        *reverse, trailer['spawn_pose_enu']['yaw'])
    assert np.allclose(reverse_body, [-1.0, 0.0], atol=1.0e-6)
    assert route.completed_legs == 1 and route.completed_loops == 0

    route.command(*route.start, speed, trailer['acceleration_m_s2'])
    assert route.completed_legs == 2 and route.completed_loops == 1


def test_city_patrol_repeats_the_full_black_road_with_stop_turns():
    document = yaml.safe_load(
        (GAZEBO / 'maps/city_coordinates_uav.yaml').read_text(encoding='utf-8'))
    trailer = document['trailer']
    expected = [
        [-150.0, 507.0], [-191.0, 511.5], [-202.3, 483.6],
        [-272.1, 307.1], [-305.8, 171.8], [-312.1, 159.7],
        [-378.8, 168.0], [-467.1, 168.6], [-550.9, 159.1],
        [-527.4, 55.6], [-516.0, -47.3], [-513.5, -510.3],
        [-321.7, -510.9], [-320.4, -322.3], [512.2, -318.5],
        [512.2, -96.2], [505.5, 49.0], [261.3, 53.7],
        [54.9, 48.6], [57.5, 117.8], [70.8, 232.8],
        [116.5, 438.0], [16.8, 458.8],
    ]

    assert trailer['model_uri'] == 'model://moving_platform_aruco_velocity'
    assert trailer['route_type'] == 'waypoints'
    assert trailer['patrol_mode'] == 'repeat'
    assert trailer['cruise_speed_m_s'] == 9.0
    assert trailer['acceleration_m_s2'] == 9.0
    assert trailer['command_rate_hz'] == 50.0
    assert trailer['stop_at_waypoints'] is True
    assert trailer['turn_speed_tolerance_m_s'] == 0.2
    assert trailer['stop_waypoint_indices'] == [1, 5, 8, 11, 12, 13, 14, 16, 18, 21]
    assert 'corner_radius_m' not in trailer
    assert trailer['waypoints_enu_m'] == expected
    assert [trailer['spawn_pose_enu'][axis] for axis in ('x', 'y')] == expected[0]

    closed_patrol = expected + [expected[0]]
    patrol_length = sum(
        math.dist(first, second)
        for first, second in zip(closed_patrol, closed_patrol[1:]))
    assert math.isclose(patrol_length, 4028.839, abs_tol=1.0e-3)


def test_stop_turn_route_brakes_stops_then_accelerates_on_the_next_straight():
    route = DRIVER.StopTurnWaypointRoute(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        waypoint_tolerance_m=0.5,
        turn_speed_tolerance_m_s=0.2,
    )

    # At 9 m/s and 9 m/s^2, braking starts 4.5 m before the tolerance edge.
    assert np.allclose(route.command(0.0, 0.0, 0.0, 9.0, 9.0), [9.0, 0.0])
    assert np.allclose(route.command(5.0, 0.0, 9.0, 9.0, 9.0), [9.0, 0.0])
    braking = route.command(8.0, 0.0, 9.0, 9.0, 9.0)
    assert 0.0 < braking[0] < 9.0 and braking[1] == 0.0

    # Reaching a corner while moving commands zero and does not turn early.
    assert route.command(10.0, 0.0, 1.0, 9.0, 9.0) == (0.0, 0.0)
    assert route.target_index == 1

    # Once almost stopped, one full zero cycle precedes the next straight.
    assert route.command(10.0, 0.0, 0.1, 9.0, 9.0) == (0.0, 0.0)
    assert np.allclose(route.command(10.0, 0.0, 0.0, 9.0, 9.0), [0.0, 9.0])
    assert route.target_index == 2
    for _ in range(3):
        route.command(10.0, 10.0, 0.0, 9.0, 9.0)
    assert route.target_index == 0
    for _ in range(3):
        route.command(0.0, 0.0, 0.0, 9.0, 9.0)
    assert route.completed_loops == 1


def test_stop_turn_route_latches_a_crossed_corner_and_counts_from_start_index():
    route = DRIVER.StopTurnWaypointRoute(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        waypoint_tolerance_m=0.5,
        turn_speed_tolerance_m_s=0.2,
        start_index=1,
        stop_waypoint_indices=[0, 1, 2],
    )

    # A delayed pose has crossed target 2. STOPPING remains latched even as
    # Euclidean distance grows on the far side, so it never commands reverse.
    assert route.command(10.0, 10.7, 9.0, 9.0, 9.0) == (0.0, 0.0)
    assert route.command(10.0, 12.0, 1.0, 9.0, 9.0) == (0.0, 0.0)
    assert route.target_index == 2
    assert route.command(10.0, 12.0, 0.0, 9.0, 9.0) == (0.0, 0.0)
    route.command(10.0, 12.0, 0.0, 9.0, 9.0)
    assert route.target_index == 0
    assert route.completed_loops == 0

    for _ in range(3):
        route.command(0.0, 0.0, 0.0, 9.0, 9.0)
    for _ in range(3):
        route.command(10.0, 0.0, 0.0, 9.0, 9.0)
    assert route.completed_loops == 1


def test_stop_turn_route_passes_shape_points_without_stopping():
    route = DRIVER.StopTurnWaypointRoute(
        [(0.0, 0.0), (10.0, 0.0), (20.0, 1.0), (20.0, 10.0)],
        waypoint_tolerance_m=0.5,
        turn_speed_tolerance_m_s=0.2,
        stop_waypoint_indices=[1, 3],
    )

    # Complete the real corner at index 1.
    route.command(10.0, 0.0, 0.0, 9.0, 9.0)
    route.command(10.0, 0.0, 0.0, 9.0, 9.0)
    command = route.command(10.0, 0.0, 0.0, 9.0, 9.0)
    assert route.target_index == 2 and math.isclose(np.linalg.norm(command), 9.0)

    # Index 2 only shapes the road centreline, so it advances at cruise speed.
    command = route.command(20.0, 1.0, 9.0, 9.0, 9.0)
    assert route.target_index == 3
    assert math.isclose(np.linalg.norm(command), 9.0)


def test_city_stop_turn_route_completes_one_acceleration_limited_loop():
    document = yaml.safe_load(
        (GAZEBO / 'maps/city_coordinates_uav.yaml').read_text(encoding='utf-8'))
    trailer = document['trailer']
    waypoints = [tuple(point) for point in trailer['waypoints_enu_m']]
    stops = set(trailer['stop_waypoint_indices'])
    route = DRIVER.StopTurnWaypointRoute(
        waypoints,
        trailer['waypoint_tolerance_m'],
        trailer['turn_speed_tolerance_m_s'],
        stop_waypoint_indices=trailer['stop_waypoint_indices'],
    )
    position = np.asarray(waypoints[0], dtype=float)
    velocity = np.zeros(2, dtype=float)
    dt = 1.0 / trailer['command_rate_hz']
    acceleration = trailer['acceleration_m_s2']

    for _ in range(30_000):
        old_target = route.target_index
        desired = np.asarray(route.command(
            *position,
            float(np.linalg.norm(velocity)),
            trailer['cruise_speed_m_s'],
            acceleration,
        ))
        if route.target_index != old_target and old_target in stops:
            assert np.linalg.norm(velocity) <= trailer['turn_speed_tolerance_m_s']
        delta = desired - velocity
        delta_norm = float(np.linalg.norm(delta))
        limit = acceleration * dt
        if delta_norm > limit:
            delta *= limit / delta_norm
        assert np.linalg.norm(delta) / dt <= acceleration + 1.0e-9
        velocity += delta
        position += velocity * dt
        if route.completed_loops == 1:
            break

    assert route.completed_loops == 1
    assert route.reached_waypoints == len(waypoints)
