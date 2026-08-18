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
