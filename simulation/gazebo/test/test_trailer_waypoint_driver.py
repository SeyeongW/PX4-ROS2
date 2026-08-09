import importlib.util
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


def test_world_velocity_is_published_in_model_axes():
    document = yaml.safe_load(
        (GAZEBO / 'maps/drone_cju_track.yaml').read_text(encoding='utf-8'))
    trailer = document['trailer']
    spawn = trailer['spawn_pose_enu']
    route = DRIVER.StadiumRoute(
        trailer['stadium_center_enu_m'],
        trailer['stadium_straight_length_m'],
        trailer['stadium_curve_radius_m'],
        trailer['stadium_heading_deg'],
        trailer['stadium_direction'],
    )
    world_velocity = np.asarray(route.direction(spawn['x'], spawn['y'])) * 3.0
    body_velocity = DRIVER.world_to_model_xy(
        *world_velocity, spawn['yaw'])
    assert np.allclose(body_velocity, [3.0, 0.0], atol=2.0e-6)

    local_offset = route._vector_to_world(
        0.0, -route.curve_radius_m - 2.0)
    route.direction(route.center[0] + local_offset[0],
                    route.center[1] + local_offset[1])
    assert np.isclose(route.max_cross_track_error_m, 2.0)
