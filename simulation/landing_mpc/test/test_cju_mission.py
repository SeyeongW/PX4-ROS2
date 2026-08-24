import importlib.util
import math
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from geometry_msgs.msg import PointStamped, Pose, PoseArray, Vector3Stamped
from rclpy.clock import Clock
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from px4_msgs.msg import (LandingTargetPose, VehicleCommand,
                          VehicleLocalPosition,
                          VehicleLandDetected, VehicleStatus)

from path_plan.astar import AStarPlanner3D
from path_plan.bspline_optimizer import BsplineOptimizer
from path_plan.world_model import WorldModel
from landing_mpc.frame import LOCAL_ENU_FRAME_ID
import landing_mpc.mission_manager_node as mission_module
from landing_mpc.mission_manager_node import (MissionManagerNode,
                                              _forward_cyclic_route_position,
                                              _forward_endpoint_eta_s,
                                              _mpc_prediction_is_safe,
                                              _mission_segment_is_free,
                                              _path_reference_horizon,
                                              _plan_global_path,
                                              _splice_path_from_current,
                                              _spatial_path_target)
from landing_mpc.marker_kf_node import MarkerKfNode
from landing_mpc.trailer_cue_node import TrailerCueNode


GAZEBO = Path(__file__).parents[2] / 'gazebo'
MAP = GAZEBO / 'maps/drone_cju_track.yaml'
CITY_MAP = GAZEBO / 'maps/city_coordinates_uav.yaml'
QUICK_CITY_MAP = GAZEBO / 'maps/city_coordinates_uav_quick_landing.yaml'
WORLD = GAZEBO / 'worlds/drone_cju.world'
TRACK_MODEL = (GAZEBO / 'models/drone_cju_track_running_track/model.sdf')
STADIUM_MODEL = GAZEBO / 'models/drone_cju_track_stadium/model.sdf'
TRACK_MESH = TRACK_MODEL.parent / 'meshes/running_track_surface.obj'
TRACK_MTL = TRACK_MODEL.parent / 'meshes/running_track_surface.mtl'
MISSION_LAUNCHER = GAZEBO / 'run_gimbal.sh'
MAP_LAUNCHER = GAZEBO / 'run_px4_map.sh'
UI_SPEC = importlib.util.spec_from_file_location(
    'cju_mission_ui', GAZEBO / 'tools/cju_mission_ui.py')
UI = importlib.util.module_from_spec(UI_SPEC)
UI_SPEC.loader.exec_module(UI)


def _mission_xy(path, document):
    mission = document['mission']
    frame = document['frames'][mission['coordinate_frame']]
    spawn_pose = document['spawn']['gazebo_spawn_pose_enu']
    heading = math.radians(frame['heading_deg_enu'])
    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])
    origin = np.asarray(frame['origin_enu_m'][:2])
    spawn = np.asarray([spawn_pose['x'], spawn_pose['y']])
    return (path[:, :2] + spawn - origin) @ rotation


def _endpoint_pose_to_enu(pose, document):
    frame = document['frames']['stadium_endpoint']
    heading = math.radians(frame['heading_deg_enu'])
    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])
    resolved = np.asarray(pose, float).copy()
    resolved[:2] = (resolved[:2] @ rotation.T
                    + np.asarray(frame['origin_enu_m'][:2]))
    resolved[5] += heading
    return resolved


def _blocked(points, mission):
    hit = np.zeros(len(points), dtype=bool)
    clearance = mission['vehicle_clearance_xy_m']
    for obstacle in mission['obstacles']:
        centre = np.asarray(obstacle['center_m'][:2])
        half = 0.5 * np.asarray(obstacle['size_m'][:2])
        gap = np.maximum(np.abs(points - centre) - half, 0.0)
        hit |= np.linalg.norm(gap, axis=1) <= clearance
    return hit


def _segment_samples(path):
    samples = []
    for start, end in zip(path[:-1], path[1:]):
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / 0.1)) + 1)
        samples.append(start + np.linspace(0.0, 1.0, count)[:, None]
                       * (end - start))
    return np.vstack(samples)


def _obj_vertices(path):
    return np.asarray([
        [float(value) for value in line.split()[1:4]]
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.startswith('v ')
    ])


def _wgs84_to_enu(point, origin):
    semi_major = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)

    def ecef(coordinate):
        latitude, longitude, height = coordinate
        latitude = math.radians(latitude)
        longitude = math.radians(longitude)
        radius = semi_major / math.sqrt(
            1.0 - eccentricity_sq * math.sin(latitude) ** 2)
        return np.array([
            (radius + height) * math.cos(latitude) * math.cos(longitude),
            (radius + height) * math.cos(latitude) * math.sin(longitude),
            (radius * (1.0 - eccentricity_sq) + height)
            * math.sin(latitude),
        ])

    latitude = math.radians(origin[0])
    longitude = math.radians(origin[1])
    rotation = np.array([
        [-math.sin(longitude), math.cos(longitude), 0.0],
        [-math.sin(latitude) * math.cos(longitude),
         -math.sin(latitude) * math.sin(longitude), math.cos(latitude)],
        [math.cos(latitude) * math.cos(longitude),
         math.cos(latitude) * math.sin(longitude), math.sin(latitude)],
    ])
    return rotation @ (ecef(point) - ecef(origin))


def test_segment_collision_cannot_hide_between_sampling_points():
    world = WorldModel.from_boxes(
        [[0.49, 0.49, -0.1]], [[0.51, 0.51, 0.1]],
        [-1.0, -1.0, -1.0], [2.0, 2.0, 1.0])
    assert not world.segment_is_free(
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], step_m=0.5)
    result = AStarPlanner3D(world, resolution_m=1.0).plan(
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    assert result.success and len(result.waypoints_m) >= 3
    assert all(world.segment_is_free(a, b)
               for a, b in zip(result.waypoints_m[:-1],
                               result.waypoints_m[1:]))


def test_cju_clearance_uses_configured_xy_radius():
    _, _, _, altitude, world = mission_module._mission_collision_contract(
        str(MAP))
    _, _, _, _, planning_world = mission_module._mission_collision_contract(
        str(MAP), True)
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    barrier = document['mission']['obstacles'][4]
    centre = np.asarray(barrier['center_m'], float)
    half = 0.5 * np.asarray(barrier['size_m'], float)
    clearance = document['mission']['vehicle_clearance_xy_m']
    # A square expansion would block this corner point, while its true radial
    # distance is sqrt(2) * 0.75 times the configured clearance.
    rounded_corner = centre.copy()
    rounded_corner[:2] += half[:2] + 0.75 * clearance
    assert world.is_free(rounded_corner)[0]

    tangent = centre.copy()
    tangent[0] += half[0] + clearance
    assert not world.is_free(tangent)[0]
    planning_only = centre.copy()
    margin = document['mission']['bspline_clearance_margin_m']
    planning_only[0] += half[0] + clearance + 0.5 * margin
    assert world.is_free(planning_only)[0]
    assert not planning_world.is_free(planning_only)[0]


def test_city_mission_profile_uses_gazebo_enu_buildings_and_map_bounds():
    document = yaml.safe_load(CITY_MAP.read_text(encoding='utf-8'))
    rotation, spawn, origin, altitude, world = (
        mission_module._mission_collision_contract(str(CITY_MAP)))
    _, _, _, _, planning_world = mission_module._mission_collision_contract(
        str(CITY_MAP), True)
    first = document['obstacles']['buildings'][0]

    assert document['mission']['obstacle_source'] == 'city_buildings'
    assert np.allclose(rotation, np.eye(2))
    assert np.allclose(spawn, [587.0, 580.0])
    assert np.allclose(origin, [0.0, 0.0])
    assert altitude == 10.0
    assert world.boxes_min.shape == world.boxes_max.shape == (205, 3)
    assert np.allclose(world.bounds_min[:2], [-650.0, -650.0])
    assert np.allclose(world.bounds_max[:2], [650.0, 650.0])
    assert world.xy_clearance_m == 1.0
    assert planning_world.xy_clearance_m == 1.0
    assert np.allclose(world.boxes_min[0, :2], first['aabb_xy_m']['min'])
    assert np.allclose(world.boxes_max[0, :2], first['aabb_xy_m']['max'])
    centre = 0.5 * (world.boxes_min[0] + world.boxes_max[0])
    centre[2] = altitude
    assert not world.is_free(centre)[0]
    assert world.is_free([[587.0, 580.0, altitude],
                          [-165.0, 0.0, altitude]]).all()
    assert document['mission']['goal_m'] == [-165.0, 0.0]


def test_city_obstacle_profile_runs_astar_sfc_and_bspline(tmp_path):
    city = tmp_path / 'small_city.yaml'
    city.write_text(
        """
map:
  bounds_enu_m: {x: [-5, 15], y: [-5, 5]}
frames:
  gazebo_world: {convention: ENU}
terrain:
  collision_geometry: {top_z_m: 0}
spawn:
  gazebo_spawn_pose_enu: {x: 0, y: 0, z: 0}
mission:
  coordinate_frame: gazebo_world
  obstacle_source: city_buildings
  cruise_altitude_m: 2
  goal_m: [10, 0]
  planner_resolution_m: 1
  vehicle_clearance_xy_m: 0.5
  bspline_clearance_margin_m: 0.25
  bspline_control_spacing_m: 1
  bspline_sample_spacing_m: 0.2
obstacles:
  buildings:
    - footprint:
        outer: [[4, -0.75], [6, -0.75], [6, 0.75], [4, 0.75]]
      foundation_z_m: 0
      roof_z_m: 3
""",
        encoding='utf-8')

    arc, path, expanded, diagnostics = _plan_global_path(
        city, include_diagnostics=True)
    rotation, spawn, origin, altitude, world = (
        mission_module._mission_collision_contract(str(city), True))
    map_path = np.column_stack((
        (path[:, :2] + spawn - origin) @ rotation,
        np.full(len(path), altitude)))

    assert expanded > 0
    assert np.allclose(path[0], [0.0, 0.0, 2.0])
    assert np.allclose(path[-1], [10.0, 0.0, 2.0])
    assert np.all(np.diff(arc) > 0.0)
    assert all(world.segment_is_free(a, b)
               for a, b in zip(map_path[:-1], map_path[1:]))
    assert len(diagnostics['sfc_boxes_min_map']) > 0
    assert math.isfinite(diagnostics['sfc_generation_time_ms'])
    assert diagnostics['sfc_generation_time_ms'] >= 0.0


def test_city_runtime_cospawn_plan_accepts_a_small_real_takeoff_offset(
        tmp_path):
    document = yaml.safe_load(CITY_MAP.read_text(encoding='utf-8'))
    trailer = document['trailer']['spawn_pose_enu']
    spawn = document['spawn']['gazebo_spawn_pose_enu']
    base_link_offset = (
        document['frames']['px4_local']['origin_enu_m'][2] - spawn['z'])
    spawn.update(x=trailer['x'], y=trailer['y'], z=0.0)
    origin = [trailer['x'], trailer['y'], base_link_offset]
    document['frames']['px4_local']['origin_enu_m'] = origin
    document['frames']['mavros_local']['origin_enu_m'] = origin
    runtime_map = tmp_path / 'runtime_city.yaml'
    runtime_map.write_text(yaml.safe_dump(document), encoding='utf-8')

    arc, path, expanded, diagnostics = _plan_global_path(
        runtime_map, [0.08, 0.0, 10.0], include_diagnostics=True)

    assert expanded > 0
    assert np.allclose(path[0], [0.08, 0.0, 10.0])
    assert np.allclose(path[-1], [-15.0, -507.0, 10.0])
    assert 500.0 < arc[-1] < 550.0
    assert len(diagnostics['sfc_boxes_min_map']) > 0


def test_cju_yaml_astar_avoids_configured_barriers_outbound_and_return():
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    outbound_arc, outbound, expanded_out, diagnostics = _plan_global_path(
        MAP, include_diagnostics=True)
    boxes_min = diagnostics['sfc_boxes_min_map']
    boxes_max = diagnostics['sfc_boxes_max_map']
    assert boxes_min.shape == boxes_max.shape
    assert boxes_min.ndim == 2 and boxes_min.shape[1] == 3
    assert len(boxes_min) > 0
    assert np.all(np.isfinite(np.r_[boxes_min, boxes_max]))
    assert np.all(boxes_max > boxes_min)
    rotation, spawn, origin, altitude, planning_world = (
        mission_module._mission_collision_contract(str(MAP), True))
    planning_world = WorldModel.from_boxes(
        planning_world.boxes_min, planning_world.boxes_max,
        [*planning_world.bounds_min[:2], altitude - 0.5],
        [*planning_world.bounds_max[:2], altitude + 0.5],
        xy_clearance_m=planning_world.xy_clearance_m)
    outbound_map = np.column_stack((
        (outbound[:, :2] + spawn - origin) @ rotation,
        np.full(len(outbound), altitude)))
    assert all(planning_world.box_is_free(low, high)
               for low, high in zip(boxes_min, boxes_max))
    assert all(np.any(np.all(
        (a >= boxes_min - 1.0e-9) & (a <= boxes_max + 1.0e-9)
        & (b >= boxes_min - 1.0e-9) & (b <= boxes_max + 1.0e-9), axis=1))
        for a, b in zip(outbound_map[:-1], outbound_map[1:]))
    # A real controller reaches a waypoint with tolerance rather than landing
    # on its exact grid coordinate. This offset reproduces that dynamic start.
    reached_goal = outbound[-1] + np.array([1.1, -0.3, 0.0])
    inbound_arc, inbound, expanded_back = _plan_global_path(
        MAP, start_local_enu=reached_goal, goal_local_enu=outbound[0])
    moving_cue = np.array([0.37, 0.61, 0.0])
    moving_arc, moving_path, _ = _plan_global_path(
        MAP, start_local_enu=outbound[-1], goal_local_enu=moving_cue)
    # Exact samples from a flight replan that exposed floor-rounded control
    # spacing: the configured 2 m must remain an upper bound.
    regression_start = np.array([39.290245, 54.666210, 4.9920654])
    regression_goal = np.array([-2.99999801, 26.9825885, -0.24])
    regression_arc, regression_path, _ = _plan_global_path(
        MAP, start_local_enu=regression_start,
        goal_local_enu=regression_goal)
    spawn_pose = document['spawn']['gazebo_spawn_pose_enu']
    spawn_xy = np.asarray([spawn_pose['x'], spawn_pose['y']], float)
    trailer_return_paths = []
    for trailer_y in (25.0, 50.0):
        trailer_world = _endpoint_pose_to_enu(
            [5.0, trailer_y, 0.0, 0.0, 0.0, 0.0], document)[:2]
        trailer_local = np.r_[trailer_world - spawn_xy, 0.0]
        trailer_arc, trailer_path, _ = _plan_global_path(
            MAP, start_local_enu=outbound[-1],
            goal_local_enu=trailer_local)
        assert np.allclose(trailer_path[-1, :2], trailer_local[:2])
        trailer_return_paths.append((trailer_arc, trailer_path))
    outbound_stadium = _mission_xy(outbound, document)
    inbound_stadium = _mission_xy(inbound, document)
    assert np.allclose(outbound_stadium[[0, -1]],
                       [[5.0, 0.0], [50.0, 50.0]])
    assert np.allclose(inbound[0], reached_goal)
    assert np.allclose(inbound[-1], outbound[0])
    assert np.allclose(moving_path[0], outbound[-1])
    assert np.allclose(moving_path[-1], [0.37, 0.61, 5.0])
    assert np.allclose(regression_path[0, :2], regression_start[:2])
    assert np.allclose(regression_path[-1, :2], regression_goal[:2])
    assert not _mission_segment_is_free(MAP, outbound[0], outbound[-1])
    trajectories = [
        (outbound_arc, outbound),
        (inbound_arc, inbound),
        (moving_arc, moving_path),
        (regression_arc, regression_path),
        *trailer_return_paths,
    ]
    for arc_m, path in trajectories:
        assert len(arc_m) == len(path) > 100 and arc_m[0] == 0.0
        assert np.all(np.diff(arc_m) > 0.0)
        assert np.isfinite(np.column_stack((arc_m, path))).all()
        expected_arc = np.r_[0.0, np.cumsum(np.linalg.norm(
            np.diff(path, axis=0), axis=1))]
        assert np.allclose(arc_m, expected_arc)
        assert all(_mission_segment_is_free(MAP, a, b)
                   for a, b in zip(path[:-1], path[1:]))
        map_path = _mission_xy(path, document)
        assert not _blocked(_segment_samples(map_path),
                            document['mission']).any()

    direct = outbound_stadium[0] + np.linspace(0.0, 1.0, 1000)[:, None] * (
        outbound_stadium[-1] - outbound_stadium[0])
    assert _blocked(direct, document['mission']).any()
    outbound_samples = _segment_samples(outbound_stadium)
    inbound_samples = _segment_samples(inbound_stadium)
    assert not _blocked(outbound_samples, document['mission']).any()
    assert not _blocked(inbound_samples, document['mission']).any()
    assert expanded_out > 0 and expanded_back > 0
    assert len(outbound) > 100 and len(inbound) > 100
    assert np.linalg.norm(outbound[-1, :2] - outbound[0, :2]) > 50.0
    terrain_centre = np.asarray(document['terrain']['center_m'])
    half_terrain = 0.5 * np.asarray(document['terrain']['size_m'])
    for path in (outbound_stadium, inbound_stadium):
        assert np.all(path >= terrain_centre - half_terrain - 1.0e-9)
        assert np.all(path <= terrain_centre + half_terrain + 1.0e-9)
    obstacle_centres = np.asarray([
        obstacle['center_m'] for obstacle in document['mission']['obstacles']
    ])
    assert np.equal(obstacle_centres, np.round(obstacle_centres)).all()
    assert obstacle_centres[:, :2].tolist() == [
        [33, 10], [18, 39], [21, 25], [49, 41], [31, 36],
        [45, 12], [39, 21], [42, 51], [17, 18], [22, 33],
        [44, 33], [24, 14], [39, 0], [30, 23], [49, 4],
        [21, 3], [15, 34], [42, 41], [35, 49], [28, 46],
        [12, 48], [50, 23], [12, 8], [30, 1], [12, 25]]
    assert len({tuple(xy) for xy in obstacle_centres[:, :2]}) == 25
    route = (np.asarray(document['mission']['goal_m'], float)
             - outbound_stadium[0, :2])
    relative = obstacle_centres[:, :2] - outbound_stadium[0, :2]
    lateral_offsets = (
        route[0] * relative[:, 1] - route[1] * relative[:, 0]
    ) / np.linalg.norm(route)
    assert (lateral_offsets > 0).any()
    assert (lateral_offsets < 0).any()
    field = document['facilities']['stadium_field']
    field_low = (np.asarray(field['center_m'], float)
                 - 0.5 * np.asarray(field['size_m'], float))
    field_high = (np.asarray(field['center_m'], float)
                  + 0.5 * np.asarray(field['size_m'], float))
    inflated_half = np.array([0.45 / 2.0, 0.35 / 2.0]) \
        + document['mission']['vehicle_clearance_xy_m']
    assert np.all(obstacle_centres[:, :2] - inflated_half >= field_low)
    assert np.all(obstacle_centres[:, :2] + inflated_half <= field_high)
    generation_bounds = document['mission']['obstacle_generation_bounds_m']
    assert document['mission']['obstacle_layout_seed'] == 5053
    assert generation_bounds == {'x': [0, 50], 'y': [0, 51]}
    assert np.all((obstacle_centres[:, 0] >= generation_bounds['x'][0])
                  & (obstacle_centres[:, 0] <= generation_bounds['x'][1]))
    assert np.all((obstacle_centres[:, 1] >= generation_bounds['y'][0])
                  & (obstacle_centres[:, 1] <= generation_bounds['y'][1]))
    obstacle_sizes = np.asarray([
        obstacle['size_m'] for obstacle in document['mission']['obstacles']
    ], float)
    physical_world = WorldModel.from_boxes(
        obstacle_centres - obstacle_sizes / 2.0,
        obstacle_centres + obstacle_sizes / 2.0,
        [-100.0, -100.0, -100.0], [200.0, 200.0, 100.0])
    assert not physical_world.segment_is_free(
        [*outbound_stadium[0], 5.0], [*outbound_stadium[-1], 5.0])

    # The running track has no collision geometry, so lock the infield
    # contract against the actual inner edge of its generated visual mesh.
    track_vertices = _obj_vertices(TRACK_MESH)
    track_surface = track_vertices[np.isclose(track_vertices[:, 2], 0.016)]
    assert len(track_surface) % 2 == 0
    inner_track = track_surface[::2, :2]
    corners = (obstacle_centres[:, None, :2]
               + np.array([[[-0.225, -0.175], [-0.225, 0.175],
                            [0.225, -0.175], [0.225, 0.175]]]))
    edges = np.roll(inner_track, -1, axis=0) - inner_track
    offsets = corners[:, :, None, :] - inner_track[None, None, :, :]
    cross = (edges[None, None, :, 0] * offsets[:, :, :, 1]
             - edges[None, None, :, 1] * offsets[:, :, :, 0])
    assert np.all(np.all(cross >= -1.0e-9, axis=2)
                  | np.all(cross <= 1.0e-9, axis=2))
    assert all(obstacle['size_m'] == [0.45, 0.35, 10]
               for obstacle in document['mission']['obstacles'])
    assert document['mission']['goal_m'] == [50, 50]
    assert 'takeoff_speed_m_s' not in document['mission']
    assert 'obstacle_clearance_m' not in document['mission']
    assert document['mission']['vehicle_clearance_xy_m'] == 1.0
    assert document['mission']['bspline_clearance_margin_m'] == 0.5
    assert document['mission']['bspline_control_spacing_m'] == 2.0
    assert document['mission']['bspline_sample_spacing_m'] == 0.1
    assert document['mission']['mpc_path_lookahead_m'] == 6.0
    assert document['mission']['mpc_path_cross_track_m'] == 0.25
    assert 'precland_handoff_m' not in document['mission']
    assert 'return_replan_distance_m' not in document['mission']
    assert document['mission']['return_replan_min_period_s'] == 2.0
    assert 'cruise_speed_m_s' not in document['mission']
    assert 'bspline_sample_period_s' not in document['mission']
    assert 'bspline_acceleration_limit_m_s2' not in document['mission']
    px4 = document['px4_vehicle']['sitl_parameter_overrides']
    assert px4['MIS_TAKEOFF_ALT'] == 5.0
    assert px4['MPC_TKO_SPEED'] == 0.5
    assert px4['MPC_XY_CRUISE'] == 3.0
    assert px4['MPC_XY_VEL_MAX'] == 10.0
    assert px4['MPC_ACC_HOR'] == 3.0
    assert px4['MPC_JERK_AUTO'] == 4.0
    assert px4['MPC_LAND_SPEED'] == 0.7
    assert px4['MPC_LAND_CRWL'] == 0.1
    assert px4['LNDMC_Z_VEL_MAX'] == 0.08
    assert px4['LNDMC_XY_VEL_MAX'] == 1.5
    assert px4['COM_DISARM_LAND'] == 2.0
    assert px4['PLD_BTOUT'] == 0.5
    assert px4['PLD_HACC_RAD'] == 0.5
    assert px4['PLD_VEL_THR'] == 0.3
    assert px4['PLD_FAPPR_ALT'] == 0.5
    assert px4['PLD_SRCH_ALT'] == 5.0
    assert px4['PLD_SRCH_TOUT'] == 10.0
    assert px4['PLD_MAX_SRCH'] == 3
    assert np.allclose(outbound[:, 2], 5.0)
    assert np.allclose(inbound[:, 2], 5.0)


def test_geometry_only_bspline_is_independent_of_speed_limits():
    world = WorldModel.from_boxes(
        [[50.0, 50.0, -1.0]], [[51.0, 51.0, 1.0]],
        [-10.0, -10.0, -2.0], [60.0, 60.0, 2.0])
    guide = np.array([
        [0.0, 0.0, 0.0], [8.0, 3.0, 0.0], [16.0, 0.0, 0.0]])
    slow_limits = BsplineOptimizer(
        world, cruise_speed_m_s=None, ctrl_spacing_m=2.0,
        max_vel=0.01, max_acc=0.01, lambda_feas=1.0e6).optimize(guide)
    fast_limits = BsplineOptimizer(
        world, cruise_speed_m_s=None, ctrl_spacing_m=2.0,
        max_vel=100.0, max_acc=100.0, lambda_feas=0.0).optimize(guide)
    assert slow_limits.spline.ts == fast_limits.spline.ts == 1.0
    assert np.allclose(slow_limits.spline.q, fast_limits.spline.q)


def test_planned_path_is_latched_geometry_only_local_enu():
    published = []
    state = SimpleNamespace(
        get_clock=lambda: Clock(),
        planned_path_pub=SimpleNamespace(publish=published.append),
    )
    MissionManagerNode._publish_planned_path(state, None)
    MissionManagerNode._publish_planned_path(
        state, np.array([[1.0, 2.0, 5.0], [4.0, 6.0, 5.0]]))

    assert published[0].header.frame_id == LOCAL_ENU_FRAME_ID
    assert published[0].poses == []
    assert [[pose.position.x, pose.position.y, pose.position.z]
            for pose in published[1].poses] == [[1.0, 2.0, 5.0],
                                                [4.0, 6.0, 5.0]]
    assert all(pose.orientation.w == 1.0 for pose in published[1].poses)
    qos = mission_module._planned_path_qos()
    assert qos.depth == 1
    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL

    cleared = []
    state.state = 'RETURN'
    state.p_d = np.array([1.0, 2.0, 5.0])
    state.get_logger = lambda: SimpleNamespace(info=lambda *_: None)
    state._publish_planned_path = cleared.append
    MissionManagerNode._set_state(state, 'ABORT', '(cue stale)')
    assert cleared == [None]


def test_optimizer_sfc_is_latched_and_decoded_by_the_yaml_ui():
    published = []
    state = SimpleNamespace(
        sfc_pub=SimpleNamespace(publish=published.append))
    diagnostics = {
        'sfc_boxes_min_map': np.array([
            [1.0, 2.0, 4.5], [5.0, 6.0, 4.5]]),
        'sfc_boxes_max_map': np.array([
            [3.0, 4.0, 5.5], [8.0, 9.0, 5.5]]),
    }

    MissionManagerNode._publish_sfc(state, diagnostics)
    vertices = UI._sfc_message_vertices(published[-1])

    assert vertices.shape == (2, 4, 2)
    assert np.allclose(vertices[0], [
        [1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])
    assert np.allclose(vertices[1], [
        [5.0, 6.0], [8.0, 6.0], [8.0, 9.0], [5.0, 9.0]])

    MissionManagerNode._publish_sfc(state, None)
    assert UI._sfc_message_vertices(published[-1]).shape == (0, 4, 2)
    with pytest.raises(ValueError, match='divisible by four'):
        UI._sfc_message_vertices(Float32MultiArray(data=[1.0, 2.0, 3.0]))


def test_active_path_and_sfc_commit_as_one_plan_number():
    active_messages, legacy_paths, legacy_sfcs = [], [], []
    input_path = np.array([
        [0.0, 0.0, 5.0], [10.0, 0.0, 5.0],
    ])
    arc, path, diagnostics = mission_module._active_path_sfc(MAP, input_path)
    assert math.isfinite(diagnostics['sfc_generation_time_ms'])
    assert diagnostics['sfc_generation_time_ms'] >= 0.0
    state = SimpleNamespace(
        mission_map_yaml=str(MAP),
        _active_plan_seq=0,
        active_plan_pub=SimpleNamespace(publish=active_messages.append),
        get_clock=lambda: Clock(),
        _publish_planned_path=lambda value: legacy_paths.append(
            np.asarray(value).copy()),
        sfc_pub=object(),
        _publish_sfc=legacy_sfcs.append,
    )

    MissionManagerNode._commit_active_path(
        state, arc, path, 0.0, diagnostics)

    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    frame = document['mission']['coordinate_frame']
    sequence, path_map, sfc_vertices = UI._active_plan_marker_snapshot(
        active_messages[-1], frame)
    rotation, origin, spawn = UI._frame_contract(document)
    assert active_messages[-1].markers[0].action == Marker.DELETEALL
    assert sequence == state._active_plan_seq == 1
    assert np.allclose(path_map[:, :2], UI._local_to_map(
        path, rotation, origin, spawn))
    assert len(sfc_vertices) == len(diagnostics['sfc_boxes_min_map'])
    assert np.array_equal(state._mission_path, path)
    assert len(legacy_paths) == len(legacy_sfcs) == 1

    old_path = state._mission_path.copy()
    bad = {
        'sfc_boxes_min_map': np.zeros((1, 3)),
        'sfc_boxes_max_map': np.zeros((1, 3)),
    }
    with pytest.raises(ValueError, match='positive SFC'):
        MissionManagerNode._commit_active_path(
            state, arc, path, 0.0, bad)
    assert state._active_plan_seq == 1
    assert np.array_equal(state._mission_path, old_path)
    assert len(active_messages) == 1


def test_active_plan_ui_rejects_mixed_or_malformed_snapshot():
    stamp = Clock().now().to_msg()
    clear_only = mission_module._active_plan_marker_message(
        MAP, None, None, 0, stamp)
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    frame = document['mission']['coordinate_frame']
    sequence, path, sfc = UI._active_plan_marker_snapshot(clear_only, frame)
    assert sequence is None and path.shape == (0, 3)
    assert sfc.shape == (0, 4, 2)

    arc, path_local, diagnostics = mission_module._active_path_sfc(
        MAP, np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]]))
    del arc
    message = mission_module._active_plan_marker_message(
        MAP, path_local, diagnostics, 3, stamp)
    message.markers[-1].header.frame_id = 'wrong_generation'
    with pytest.raises(ValueError, match='frame/stamp mismatch'):
        UI._active_plan_marker_snapshot(message, frame)


def test_live_ui_keeps_only_superseded_return_plans_in_history():
    empty_sfc = np.empty((0, 4, 2))
    mission_path = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]])
    first_return = np.array([[1.0, 0.0, 10.0], [2.0, 1.0, 10.0]])
    latest_return = np.array([[1.5, 0.5, 10.0], [3.0, 2.0, 10.0]])
    data = {
        'state': 'MISSION',
        'active_plan': (None, np.empty((0, 3)), empty_sfc),
        'active_plan_is_return': False,
        'return_path_history': [],
        'dynamic_replans': 0,
    }

    assert UI._update_active_plan_history(
        data, (1, mission_path, empty_sfc))
    data['state'] = 'RETURN_PLAN'
    assert UI._update_active_plan_history(
        data, (2, first_return, empty_sfc))
    assert data['return_path_history'] == []

    data['state'] = 'RETURN'
    assert UI._update_active_plan_history(
        data, (3, latest_return, empty_sfc))
    assert len(data['return_path_history']) == 1
    assert data['dynamic_replans'] == 1
    assert np.array_equal(data['return_path_history'][0], first_return)
    assert not UI._update_active_plan_history(
        data, (3, latest_return + 10.0, empty_sfc))
    assert len(data['return_path_history']) == 1

    assert UI._update_active_plan_history(
        data, (None, np.empty((0, 3)), empty_sfc))
    assert len(data['return_path_history']) == 2
    assert data['dynamic_replans'] == 1
    assert np.array_equal(data['return_path_history'][-1], latest_return)
    assert data['active_plan'][0] is None


def test_path_reference_horizon_hard_caps_straight_flight_at_twelve():
    arc = np.array([0.0, 200.0])
    path = np.array([[0.0, 0.0, 5.0], [200.0, 0.0, 5.0]])

    positions, velocities = _path_reference_horizon(
        arc, path, 0.0, 0.1, 5, 20.0, 3.0, 2.0)

    assert np.allclose(np.linalg.norm(velocities, axis=1), 12.0)
    assert np.allclose(positions[:, 0], 1.2 * np.arange(1.0, 6.0))


def test_path_reference_horizon_obeys_tight_curve_acceleration_limit():
    arc = np.array([0.0, 10.0, 20.0, 110.0])
    path = np.array([[0.0, 0.0, 5.0],
                     [10.0, 0.0, 5.0],
                     [10.0, 10.0, 5.0],
                     [10.0, 100.0, 5.0]])
    curvature = math.sqrt(2.0) / 10.0

    _, velocities = _path_reference_horizon(
        arc, path, 10.0, 0.01, 1, 10.0, 3.0, 2.0)

    assert np.linalg.norm(velocities[0]) <= math.sqrt(
        3.0 / curvature) + 1.0e-12


def test_path_speed_envelope_brakes_backward_and_accelerates_forward():
    path = np.array([[0.0, 0.0, 5.0],
                     [5.0, 0.0, 5.0],
                     [5.0, 1.0, 5.0],
                     [5.0, 6.0, 5.0],
                     [5.0, 106.0, 5.0]])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(path, axis=0), axis=1))]
    acceleration = 2.0

    speeds = mission_module._path_speed_envelope(
        arc, path, 10.0, acceleration)
    intervals = np.diff(arc)

    assert speeds[0] < 10.0       # braking propagated before the corner
    assert speeds[2] < 10.0       # acceleration propagated after the corner
    assert np.all(speeds[1:] ** 2 - speeds[:-1] ** 2
                  <= 2.0 * acceleration * intervals + 1.0e-12)
    assert np.all(speeds[:-1] ** 2 - speeds[1:] ** 2
                  <= 2.0 * acceleration * intervals + 1.0e-12)


def test_path_reference_horizon_brakes_at_the_bspline_endpoint():
    arc = np.array([0.0, 5.0, 10.0])
    path = np.column_stack((arc, np.zeros(3), np.ones(3) * 5.0))

    positions, velocities = _path_reference_horizon(
        arc, path, 9.7, 0.1, 20, 3.0, 3.0, 2.0)

    assert positions.shape == velocities.shape == (20, 3)
    assert np.all(np.isfinite(np.column_stack((positions, velocities))))
    assert np.all(np.diff(positions[:, 0]) >= -1.0e-12)
    assert np.all(positions[:, 0] <= 10.0)
    assert np.all(np.linalg.norm(velocities, axis=1) <= 3.0 + 1.0e-12)
    assert np.allclose(positions[-1], [10.0, 0.0, 5.0])
    assert np.allclose(velocities[-1], 0.0)

    # A 3 m/s triangular S-curve with j=2 needs 3.674 m, not the
    # constant-acceleration formula's optimistic 1.5 m.
    stop_distance = 3.0 ** 1.5 / math.sqrt(2.0)
    assert np.isclose(
        mission_module._s_curve_stop_speed(stop_distance, 3.0, 2.0), 3.0)
    assert mission_module._s_curve_stop_speed(1.5, 3.0, 2.0) < 3.0


@pytest.mark.parametrize(
    ('speed', 'expected_distance'), [
        (0.25, 0.25 ** 1.5 / math.sqrt(2.0)),
        (10.0, 52.5),
    ])
def test_s_curve_stop_speed_and_distance_are_inverse(
        speed, expected_distance):
    distance = mission_module._s_curve_stop_distance(speed, 1.0, 2.0)

    assert distance == pytest.approx(expected_distance)
    assert mission_module._s_curve_stop_speed(
        distance, 1.0, 2.0) == pytest.approx(speed)


def test_return_relative_braking_is_identity_outside_ten_metres():
    arc = np.array([0.0, 20.0])
    path = np.array([[0.0, 0.0, 5.0], [20.0, 0.0, 5.0]])
    legacy = _path_reference_horizon(
        arc, path, 0.0, 0.1, 20, 3.0, 3.0, 2.0)
    outside = _path_reference_horizon(
        arc, path, 0.0, 0.1, 20, 3.0, 3.0, 2.0,
        target_velocity_xy=np.array([1.0, 0.0]),
        target_range_xy_m=10.0 + 1.0e-6,
        relative_brake_start_m=10.0)

    assert np.array_equal(outside[0], legacy[0])
    assert np.array_equal(outside[1], legacy[1])


def test_far_moving_target_pursuit_keeps_curvature_braking():
    arc = np.array([0.0, 10.0, 20.0, 120.0])
    path = np.array([[0.0, 0.0, 5.0],
                     [10.0, 0.0, 5.0],
                     [10.0, 10.0, 5.0],
                     [10.0, 110.0, 5.0]])

    _, velocities = _path_reference_horizon(
        arc, path, 9.5, 0.1, 10, 12.0, 2.0, 2.0,
        target_velocity_xy=np.array([9.0, 0.0]),
        target_range_xy_m=50.0,
        relative_brake_start_m=10.0)

    speeds = np.linalg.norm(velocities[:, :2], axis=1)
    assert np.any(speeds < 9.0)
    assert np.all(speeds <= 12.0 + 1.0e-12)


def test_far_moving_target_pursuit_uses_twelve_on_a_straight():
    arc = np.array([0.0, 200.0])
    path = np.array([[0.0, 0.0, 5.0], [200.0, 0.0, 5.0]])

    _, velocities = _path_reference_horizon(
        arc, path, 0.0, 0.1, 10, 12.0, 3.0, 4.0,
        target_velocity_xy=np.array([9.0, 0.0]),
        target_range_xy_m=50.0,
        relative_brake_start_m=10.0)

    assert np.allclose(np.linalg.norm(velocities[:, :2], axis=1), 12.0)


def test_return_relative_braking_slows_on_the_bspline_consistently():
    arc = np.array([0.0, 30.0])
    path = np.array([[0.0, 0.0, 5.0], [30.0, 0.0, 5.0]])
    positions, velocities = _path_reference_horizon(
        arc, path, 0.0, 0.1, 20, 3.0, 3.0, 2.0,
        target_velocity_xy=np.array([1.0, 0.0]),
        target_range_xy_m=8.0,
        relative_brake_start_m=10.0)

    relative_speed = np.linalg.norm(
        velocities[:, :2] - np.array([1.0, 0.0]), axis=1)
    assert np.all(relative_speed <= 0.3 + 1.0e-12)
    position_steps = np.diff(np.vstack(([0.0, 0.0, 5.0], positions)), axis=0)
    assert np.allclose(position_steps / 0.1, velocities, atol=1.0e-12)


def test_streamed_acceleration_is_slew_limited_at_the_control_rate():
    acceleration = np.zeros(3)
    desired = np.array([0.2, -0.2, 0.1])

    samples = []
    for _ in range(5):
        acceleration = mission_module._limit_acceleration_slew(
            acceleration, desired, jerk_m_s3=2.0, elapsed_s=0.02)
        samples.append(acceleration.copy())

    samples = np.asarray(samples)
    assert np.all(np.abs(np.diff(
        np.vstack((np.zeros(3), samples)), axis=0)) <= 0.04 + 1.0e-12)
    assert np.allclose(samples[-1], desired)


def test_mpc_prediction_checks_every_horizon_chord(monkeypatch):
    checked = []

    def segment_is_free(_map, start, goal):
        checked.append((np.asarray(start).copy(), np.asarray(goal).copy()))
        return float(goal[0]) < 2.0

    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', segment_is_free)
    prediction = np.array([[1.0, 0.0, 5.0], [2.0, 0.0, 5.0]])

    assert not _mpc_prediction_is_safe(
        MAP, np.array([0.0, 0.0, 5.0]), prediction)
    assert len(checked) == 2
    assert np.allclose(checked[0][0], [0.0, 0.0, 5.0])
    assert np.allclose(checked[1][0], prediction[0])


def test_mapless_landing_checks_only_finite_segments():
    start = np.array([0.0, 0.0, 6.0])
    goal = np.array([1.0, 0.0, 6.0])

    assert _mission_segment_is_free('', start, goal)
    assert mission_module._mission_planning_segment_is_free('', start, goal)
    assert not _mission_segment_is_free('', start, [math.nan, 0.0, 6.0])


def test_mpc_horizontal_speed_requires_the_absolute_xy_cap_for_whole_horizon():
    check = mission_module._mpc_horizontal_speed_is_safe

    assert check(np.array([[6.0, 8.0, 0.0], [10.0, 0.0, 1.0]]), 10.0)
    landing_limit = mission_module._GPS_PREACQUIRE_SPEED_LIMIT_M_S
    assert landing_limit == 12.0
    assert check(np.array([[12.0, 0.0, 0.0]]), landing_limit)
    assert not check(np.array([[12.01, 0.0, 0.0]]), landing_limit)
    assert not check(np.array([[10.0, 0.01, 0.0]]), 10.0)
    assert not check(
        np.array([[10.1, 0.0, 0.0], [9.9, 0.0, 0.0]]), 10.0)
    assert not check(
        np.array([[10.3, 0.0, 0.0], [9.9, 0.0, 0.0]]), 10.0)
    assert not check(
        np.array([[10.1, 0.0, 0.0], [10.05, 0.0, 0.0]]), 10.0)
    assert not check(
        np.array([[9.9, 0.0, 0.0], [10.19, 0.0, 0.0],
                  [9.9, 0.0, 0.0]]), 10.0)
    assert not check(np.array([[math.nan, 0.0, 0.0]]), 10.0)
    assert not check(np.array([1.0, 0.0, 0.0]), 10.0)


def test_city_speed_layers_keep_their_separate_limits():
    launcher = MISSION_LAUNCHER.read_text(encoding='utf-8')
    city = yaml.safe_load(CITY_MAP.read_text(encoding='utf-8'))
    px4 = city['px4_vehicle']['sitl_parameter_overrides']

    assert '-p path_mpc_speed_m_s:=12.0' in launcher
    assert '-p path_mpc_v_max_m_s:=12.0' in launcher
    assert '-p path_mpc_a_max_m_s2:=3.0' in launcher
    assert '-p path_mpc_jerk_m_s3:=4.0' in launcher
    assert '-p path_speed_profile_a_max_m_s2:=3.0' in launcher
    assert '-p landing_mpc_v_max_m_s:=5.0' in launcher
    assert '-p entry_fix_window_s:="${CITY_ENTRY_FIX_WINDOW_S:-2.0}"' in launcher
    assert '-p bias_max_m:=1.0' in launcher
    assert px4['MIS_TAKEOFF_ALT'] == 10.0
    assert px4['MPC_XY_CRUISE'] == 12.0
    assert px4['MPC_XY_VEL_MAX'] == 12.0
    assert px4['MPC_ACC_HOR'] == 3.0
    assert px4['MPC_JERK_AUTO'] == 4.0
    assert px4['LNDMC_XY_VEL_MAX'] == 12.0
    assert city['mission']['cruise_altitude_m'] == 10.0
    assert city['mission']['hover_duration_s'] == 3.0
    assert city['mission']['return_staging_enabled'] is False
    assert city['trailer']['cruise_speed_m_s'] == 7.0
    assert city['trailer']['cruise_speed_m_s'] + 5.0 == 12.0
    assert mission_module._GPS_PREACQUIRE_SPEED_LIMIT_M_S == 12.0
    assert mission_module._PATH_HARD_SPEED_LIMIT_M_S == 12.0


def test_mpc_stream_speed_never_starts_a_new_overspeed():
    check = mission_module._mpc_stream_speed_is_safe

    assert check([6.0, 8.0, 0.0], 10.0)
    assert not check([10.01, 0.0, 0.0], 10.0)
    assert not check([10.1, 0.0, 0.0], 10.0)
    assert not check([10.22, 0.0, 0.0], 10.0)


def test_tracking_mpc_recovery_stream_cannot_ratchet_above_its_xy_cap(
        monkeypatch):
    class RecoveringMpc:
        N = 4
        j_max = 2.0
        v_max = 9.0

        def __init__(self):
            self.calls = 0
            self.reset_count = 0

        def solve(self, *_args, **_kwargs):
            self.calls += 1
            direction = np.array([0.6, 0.8, 0.0])
            speeds = ([8.60, 8.58, 8.56, 8.54] if self.calls == 1
                      else [10.4, 9.6, 9.2, self.v_max])
            return SimpleNamespace(
                success=True,
                predicted_pos=np.array([
                    [1.0, 0.0, 5.0], [2.0, 0.0, 5.0],
                    [3.0, 0.0, 5.0], [4.0, 0.0, 5.0]]),
                predicted_vel=np.asarray(speeds)[:, None] * direction,
                predicted_acc=np.tile(-0.2 * direction, (self.N, 1)))

        def reset(self):
            self.reset_count += 1

    clock = [0.0]
    sent = []
    controller = RecoveringMpc()
    state = SimpleNamespace(
        state='MISSION',
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=8.62 * np.array([0.6, 0.8, 0.0]),
        _mission_arc_m=np.array([0.0, 4.0]),
        _mission_path=np.array([[0.0, 0.0, 5.0], [4.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=1.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _path_mpc=controller,
        _path_reference=mission_module.HorizonReference(lead_s=0.1),
        _path_solve_t=None,
        _path_failure_hold=None,
        _last_sent_acceleration=np.zeros(3),
        _last_sent_acceleration_t=None,
        mpc_dt=0.1,
        dt=0.02,
        _path_mpc_speed=8.0,
        _path_mpc_a_max=2.0,
        _path_speed_profile_a_max=1.0,
        _now=lambda: clock[0],
        _send=lambda pos, vel, acc: sent.append((
            np.asarray(pos).copy(), np.asarray(vel).copy(),
            np.asarray(acc).copy())),
        _send_goto=lambda *_: pytest.fail(
            'unavoidable overspeed recovery must remain continuous MPC PVA'),
        get_logger=lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mission_module, '_safe_spatial_path_target',
        lambda *_args, **_kwargs: (0.0, np.array([1.0, 0.0, 5.0])))
    monkeypatch.setattr(
        MissionManagerNode, '_tracking_path_reference',
        lambda _self, *_args, **_kwargs: (
            np.zeros((controller.N, 3)),
            np.zeros((controller.N, 3))))
    monkeypatch.setattr(
        mission_module, '_mpc_prediction_is_safe',
        lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)

    assert MissionManagerNode._follow_path(state)
    expected = state._path_reference.sample(0.1)
    clock[0] = 0.1
    state.v_d = 10.5 * np.array([0.6, 0.8, 0.0])
    assert MissionManagerNode._follow_path(state)

    assert controller.calls >= 2
    assert controller.reset_count == 0
    assert state._path_solve_t == 0.0
    assert len(sent) == 2
    velocities = np.asarray([command[1][:2] for command in sent])
    speeds = np.linalg.norm(velocities, axis=1)
    assert np.all(speeds <= controller.v_max + 1.0e-6)
    assert np.allclose(sent[-1][0], expected[0])
    assert np.allclose(sent[-1][1], expected[1])
    assert speeds[1] <= speeds[0] + 1.0e-6
    assert all(command[2][0] < 0.0 and command[2][1] < 0.0
               for command in sent)


def test_collision_contracts_are_prewarmed_with_runtime_cache_keys(
        monkeypatch):
    calls = []
    monkeypatch.setattr(
        mission_module, '_mission_collision_contract',
        lambda *args: calls.append(args))

    mission_module._prewarm_mission_collision_contracts(MAP)

    assert calls == [(str(MAP),), (str(MAP), True)]


def test_tracking_prediction_can_require_the_planning_reserve(monkeypatch):
    hard_calls = []
    planning_calls = []
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free',
        lambda *_: hard_calls.append(True) or True)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda *_: planning_calls.append(True) or True)

    assert _mpc_prediction_is_safe(
        MAP, np.array([0.0, 0.0, 5.0]),
        np.array([[1.0, 0.0, 5.0], [2.0, 0.0, 5.0]]), planning=True)
    assert planning_calls == [True, True]
    assert hard_calls == []


def test_ui_keeps_physical_obstacles_raw_and_clearance_on_the_drone():
    import matplotlib.pyplot as plt

    mission = {
        'vehicle_clearance_xy_m': 1.0,
        'obstacles': [{
            'center_m': [0.0, 0.0, 5.0],
            'size_m': [0.45, 0.35, 10.0],
        }],
    }
    fig, ax = plt.subplots()
    UI._draw_obstacles(ax, mission)
    assert len(ax.patches) == 1
    physical = ax.patches[0]
    half_size = np.array([0.225, 0.175])
    assert np.allclose(physical.get_xy(), -half_size)
    assert np.allclose([physical.get_width(), physical.get_height()],
                       2.0 * half_size)

    safety = UI._draw_vehicle_radius(ax, 1.0)
    safety.center = (2.0, 3.0)
    assert safety.radius == 1.0
    assert safety.center == (2.0, 3.0)
    plt.close(fig)


def test_city_ui_uses_waypoint_route_and_raw_building_boxes():
    document = yaml.safe_load(CITY_MAP.read_text(encoding='utf-8'))
    rotation, origin, spawn = UI._frame_contract(document)
    route = UI._trailer_route(document, rotation, origin)
    obstacles = UI._physical_obstacles(document)

    assert route.shape == (24, 2)
    assert np.allclose(route[:-1], document['trailer']['waypoints_enu_m'])
    assert np.allclose(route[-1], route[0])
    speed = float(document['trailer']['cruise_speed_m_s'])
    first_leg_s = np.linalg.norm(route[1] - route[0]) / speed
    loop_s = np.linalg.norm(np.diff(route, axis=0), axis=1).sum() / speed
    assert np.allclose(
        UI._trailer_position(route, speed, first_leg_s), route[1])
    assert np.allclose(UI._trailer_position(route, speed, loop_s), route[0])

    quick = yaml.safe_load(QUICK_CITY_MAP.read_text(encoding='utf-8'))
    rotation, origin, _spawn = UI._frame_contract(quick)
    quick_route = UI._trailer_route(quick, rotation, origin)
    start = quick['trailer']['start_index']
    assert quick_route.shape == (24, 2)
    assert np.allclose(quick_route[0], quick['trailer']['waypoints_enu_m'][start])
    assert np.allclose(quick_route[-1], quick_route[0])
    assert len(obstacles) == 205
    assert np.allclose(
        UI._local_to_map([[0.0, 0.0, 10.0]], rotation, origin, spawn)[0],
        [587.0, 580.0])


def test_ui_places_diagnostics_to_the_right_of_the_map():
    import matplotlib.pyplot as plt

    fig, map_ax, info_ax = UI._figure_layout()
    fig.canvas.draw()
    assert map_ax.get_position().x1 < info_ax.get_position().x0
    map_ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    UI._expand_axes_to_points(map_ax, [[-2.0, 3.0]], pad_m=1.0)
    assert map_ax.get_xlim() == pytest.approx((-3.0, 1.0))
    assert map_ax.get_ylim() == pytest.approx((0.0, 4.0))
    plt.close(fig)


def test_live_ui_rejects_bad_frames_and_publishes_valid_ned_as_local_enu():
    point = PointStamped()
    point.header.frame_id = LOCAL_ENU_FRAME_ID
    point.point.x, point.point.y, point.point.z = 7.0, 4.0, 5.0
    assert np.allclose(UI._point_message_to_local(point), [7.0, 4.0, 5.0])
    assert UI._point_message_sample(point) is None
    point.header.stamp.sec = 2
    point.header.stamp.nanosec = 500_000_000
    stamp_s, sampled = UI._point_message_sample(point)
    assert stamp_s == 2.5
    assert np.allclose(sampled, [7.0, 4.0, 5.0])
    point.header.frame_id = 'map'
    assert UI._point_message_to_local(point) is None
    point.header.frame_id = LOCAL_ENU_FRAME_ID
    point.point.x = float('nan')
    assert UI._point_message_to_local(point) is None

    path = PoseArray()
    path.header.frame_id = LOCAL_ENU_FRAME_ID
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = 1.0, 2.0, 5.0
    path.poses.append(pose)
    assert np.allclose(UI._path_message_to_local(path), [[1.0, 2.0, 5.0]])
    path.poses.clear()
    assert UI._path_message_to_local(path).shape == (0, 3)
    bad_pose = Pose()
    bad_pose.position.x = float('inf')
    path.poses.append(bad_pose)
    assert UI._path_message_to_local(path) is None
    path.poses.clear()
    path.header.frame_id = 'map'
    assert UI._path_message_to_local(path) is None

    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    rotation, origin, spawn = UI._frame_contract(document)
    assert np.allclose(
        UI._local_to_map([[0.0, 0.0, 5.0]], rotation, origin, spawn),
        [[5.0, 0.0]])

    published = []
    state = SimpleNamespace(
        _now=lambda: 1.0,
        get_clock=lambda: Clock(),
        vehicle_position_pub=SimpleNamespace(publish=published.append),
    )
    position = VehicleLocalPosition()
    position.x, position.y, position.z = 4.0, 7.0, -5.0
    position.vx = position.vy = position.vz = 0.0
    position.xy_valid = position.z_valid = True
    position.v_xy_valid = position.v_z_valid = True
    MissionManagerNode._on_pos(state, position)
    assert len(published) == 1
    assert published[0].header.frame_id == LOCAL_ENU_FRAME_ID
    assert np.allclose(
        [published[0].point.x, published[0].point.y,
         published[0].point.z], [7.0, 4.0, 5.0])
    position.xy_valid = False
    MissionManagerNode._on_pos(state, position)
    assert len(published) == 1


def test_world_and_yaml_share_stadium_obstacles_and_spawns():
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    assert document['mission']['command_sequence'] == [
        'takeoff', 'mission', 'land']
    assert document['mission']['coordinate_frame'] == 'stadium_endpoint'
    assert document['terrain']['coordinate_frame'] == 'stadium_endpoint'
    launcher = MISSION_LAUNCHER.read_text(encoding='utf-8')
    map_launcher = MAP_LAUNCHER.read_text(encoding='utf-8')
    assert 'commands=(takeoff mission land)' in launcher
    assert 'wait_for_states READY 120' in launcher
    takeoff_case = launcher.split('      takeoff)', 1)[1].split(
        '      mission)', 1)[0]
    mission_case = launcher.split('      mission)', 1)[1].split(
        '      land)', 1)[0]
    assert takeoff_case.count('touch "$TRAILER_START_FILE"') == 1
    assert takeoff_case.index('wait_for_states READY 120') < takeoff_case.index(
        'touch "$TRAILER_START_FILE"')
    assert 'touch "$TRAILER_START_FILE"' not in mission_case
    assert 'waiting for takeoff clearance' in map_launcher
    assert "send_until_state mission 'MISSION_PLAN|MISSION|HOVER'" in launcher
    assert "wait_for_states 'MISSION|HOVER' 120" in launcher
    assert 'DEFAULT_MISSION_HOVER_TIMEOUT_S=180' in launcher
    assert 'DEFAULT_MISSION_HOVER_TIMEOUT_S=300' in launcher
    assert 'MISSION_HOVER_TIMEOUT_S:-$DEFAULT_MISSION_HOVER_TIMEOUT_S' in launcher
    assert ("send_until_state land 'RETURN_PLAN|RETURN|LANDING_ACQUIRE|"
            "LANDING_DESCEND|PRECLAND|DONE'" in launcher)
    assert 'DEFAULT_LANDING_DONE_TIMEOUT_S=180' in launcher
    assert 'DEFAULT_LANDING_DONE_TIMEOUT_S=600' in launcher
    assert 'LANDING_DONE_TIMEOUT_S:-$DEFAULT_LANDING_DONE_TIMEOUT_S' in launcher
    assert '반복 순찰' not in launcher
    assert 'approach_alt' not in launcher
    assert 'acquire_xy' not in launcher
    assert 'touchdown_height' not in launcher
    assert 'z_floor' not in launcher
    assert ("printf 'flight_control_owner\\t%s\\n' "
            "'mission_manager_mpc_then_px4_precland'" in launcher)
    assert 'PX4 NAV_PRECLAND' in launcher
    assert "printf -v LANDING_TARGET_SPEED_ARG '%.9f'" in launcher
    assert '-p landing_target_min_speed_m_s:="$LANDING_TARGET_SPEED_ARG"' in launcher
    assert 'TAKEOFF -> LANDING_ACQUIRE ' in launcher
    assert 'RESET_TRAILER_FROM_YAML="$RESET_TRAILER_FOR_RUN"' in launcher
    assert 'TRAILER_START_INDEX="${TRAILER_START_INDEX:-0}"' in launcher
    assert 'LANDING_TAKEOFF_ALT=25.0' not in launcher
    assert 'moving_trailer_marker_surface' in launcher
    assert '-p landing_gps_preacquire_range_m:=35.0' in launcher
    assert '-p landing_mpc_v_max_m_s:=5.0' in launcher
    assert '-p path_mpc_a_max_m_s2:=3.0' in launcher
    assert '-p path_mpc_jerk_m_s3:=4.0' in launcher
    assert '-p path_speed_profile_a_max_m_s2:=3.0' in launcher
    assert 'GIMBAL_AIM_START_RANGE_M:-60.0' in launcher
    assert 'GIMBAL_AIM_FULL_RANGE_M:-35.0' in launcher
    assert 'aim_start_range_m:="$GIMBAL_AIM_START"' in launcher
    assert 'aim_full_range_m:="$GIMBAL_AIM_FULL"' in launcher
    assert 'prefer_cue_aim:=true' in launcher
    assert 'CITY_MIN_MARKER_PX:-20.0' in launcher
    assert 'min_marker_px:="${CITY_MIN_MARKER_PX:-20.0}"' in launcher
    assert 'debug_dir:="$PX4_MAP_RUNTIME_DIR/aruco_debug"' in launcher
    assert 'gimbal_attitude_source:=camera_imu' in launcher
    assert '--live --map "$LANDING_COORDINATES"' in launcher
    assert '${MISSION_VIEW:-1}' in launcher
    assert "retry_command 'geometry B-spline/TrackingMPC 상태 확인'" in launcher
    assert '"MPC_XY_CRUISE":' in map_launcher
    assert '"MPC_XY_VEL_MAX":' in map_launcher
    assert '"MPC_ACC_HOR":' in map_launcher
    assert '"MPC_JERK_AUTO":' in map_launcher
    assert 'print(f"param set {name} {value}")' in map_launcher
    assert 'Gazebo ▶/clock 확인 후 같은 명령 재입력' in launcher
    assert 'SIM_PID=$!' in launcher
    assert 'required_alive' in launcher
    assert 'required component exited:' in launcher
    assert 'trailer_odometry.jsonl' in launcher
    assert 'flight.ulg' in launcher
    assert 'XDG_STATE_HOME' in launcher
    assert 'manifest.tsv' in launcher
    assert 'pkill' not in launcher
    assert 'MAX_PAIR_DISAGREEMENT_M:-1.0' in launcher
    assert 'MAX_PAIR_DISAGREEMENT_M:-1000000' not in launcher
    world = ET.parse(WORLD).getroot().find('world')
    assert world is not None and world.attrib['name'] == document['map'][
        'gazebo_world_name']
    assert all(include.findtext('uri') != 'model://drone_cju_campus'
               for include in world.findall('include'))

    ground = world.find("model[@name='stadium_ground']")
    ground_size = [float(v) for v in ground.findtext(
        "./link/collision[@name='collision']/geometry/box/size").split()]
    ground_collision_pose = [float(v) for v in ground.findtext(
        "./link/collision[@name='collision']/pose").split()]
    ground_model_pose = [float(v) for v in ground.findtext('pose').split()]
    assert np.allclose(ground_size[:2], document['terrain']['size_m'])
    assert ground.find('pose').attrib['relative_to'] == 'stadium_endpoint'
    assert np.allclose(ground_model_pose, 0.0)
    assert np.allclose(ground_collision_pose[:2],
                       document['terrain']['center_m'])
    ground_visual = ground.find(
        "./link/visual[@name='stadium_ground_surface']")
    assert ground_visual is not None
    assert np.allclose([float(v) for v in ground_visual.findtext(
        'geometry/box/size').split()][:2], document['terrain']['size_m'])
    assert float(ground_visual.findtext('pose').split()[2]) < 0.0

    model = world.find("model[@name='mission_obstacles']")
    model_pose = [float(v) for v in model.findtext('pose').split()]
    assert model.find('pose').attrib['relative_to'] == 'stadium_endpoint'
    assert np.allclose(model_pose, 0.0)
    expected_names = {
        obstacle['name'] for obstacle in document['mission']['obstacles']}
    collision_names = {
        collision.attrib['name'].removesuffix('_collision')
        for collision in model.findall('./link/collision')}
    visual_names = {
        visual.attrib['name'] for visual in model.findall('./link/visual')}
    assert collision_names == expected_names
    assert visual_names == expected_names
    for obstacle in document['mission']['obstacles']:
        assert obstacle['center_m'][2] == 5.0
        assert obstacle['size_m'][2] == 10.0
        collision = model.find(
            f"./link/collision[@name='{obstacle['name']}_collision']")
        pose = [float(v) for v in collision.findtext('pose').split()]
        size = [float(v) for v in collision.findtext(
            'geometry/box/size').split()]
        assert np.allclose(pose[:3], obstacle['center_m'])
        assert np.allclose(size, obstacle['size_m'])
        visual = model.find(
            f"./link/visual[@name='{obstacle['name']}']")
        assert np.allclose([float(v) for v in visual.findtext('pose').split()],
                           [*obstacle['center_m'], 0.0, 0.0, 0.0])
        assert np.allclose([float(v) for v in visual.findtext(
            'geometry/box/size').split()], obstacle['size_m'])

    spawn = document['spawn']['gazebo_spawn_pose_enu']
    spawn_element = world.find("frame[@name='drone_spawn']/pose")
    assert spawn_element.attrib['relative_to'] == 'stadium_endpoint'
    spawn_local = [float(v) for v in spawn_element.text.split()]
    assert np.allclose(spawn_local,
                       [5.0, 0.0, 2.051, 0.0, 0.0, math.pi / 2.0])
    assert np.equal(spawn_local[:2], np.round(spawn_local[:2])).all()
    spawn_frame = _endpoint_pose_to_enu(spawn_local, document)
    assert np.allclose(spawn_frame, [spawn[key] for key in
                                     ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
    trailer = document['trailer']
    trailer_include = next(include for include in world.findall('include')
                           if include.findtext('name') == 'trailer')
    trailer_pose_element = trailer_include.find('pose')
    assert trailer_pose_element.attrib['relative_to'] == 'stadium_endpoint'
    trailer_local = [float(v) for v in trailer_pose_element.text.split()]
    assert np.allclose(trailer_local,
                       [5.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2.0])
    trailer_frame = world.find("frame[@name='trailer_spawn']/pose")
    assert trailer_frame.attrib['relative_to'] == 'stadium_endpoint'
    assert np.allclose([float(v) for v in trailer_frame.text.split()],
                       trailer_local)
    trailer_pose = _endpoint_pose_to_enu(trailer_local, document)
    configured_pose = trailer['spawn_pose_enu']
    assert np.allclose(trailer_pose, [configured_pose[key] for key in
                                      ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
    assert np.allclose([spawn['x'], spawn['y']],
                       [configured_pose['x'], configured_pose['y']])
    assert trailer['route_type'] == 'linear_shuttle'
    endpoints = np.asarray(trailer['shuttle_endpoints_enu_m'], float)
    frame = document['frames']['stadium_endpoint']
    heading = math.radians(frame['heading_deg_enu'])
    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])
    map_endpoints = (endpoints - frame['origin_enu_m'][:2]) @ rotation
    assert np.allclose(map_endpoints, [[5.0, 0.0], [5.0, 50.0]])
    assert np.allclose(map_endpoints, np.round(map_endpoints), atol=1.0e-9)
    footprint_half = 0.5 * np.asarray(trailer['body_footprint_m'], float)
    swept_low = map_endpoints.min(axis=0) - footprint_half
    swept_high = map_endpoints.max(axis=0) + footprint_half
    for obstacle in document['mission']['obstacles']:
        centre = np.asarray(obstacle['center_m'][:2], float)
        half = 0.5 * np.asarray(obstacle['size_m'][:2], float)
        gap = np.maximum.reduce((
            centre - half - swept_high,
            swept_low - (centre + half),
            np.zeros(2),
        ))
        assert np.linalg.norm(gap) >= 1.0
    assert math.isclose(np.linalg.norm(endpoints[1] - endpoints[0]), 50.0,
                        abs_tol=1.0e-6)
    assert trailer['shuttle_leg_length_m'] == 50
    assert trailer['route_length_m'] == 100
    assert trailer['patrol_mode'] == 'repeat'
    assert trailer['command_rate_hz'] == 50.0
    assert trailer['cruise_speed_m_s'] == 1.0
    assert 'TRAILER_SPEED_FOR_RUN=3.0' not in launcher
    assert 'LANDING_TRAILER_SPEED="${LANDING_CONFIG[4]}"' in launcher
    assert 'LANDING_CUE_RATE="${LANDING_CONFIG[5]}"' in launcher
    assert '-p rate_hz:="$LANDING_CUE_RATE"' in launcher

    track = ET.parse(TRACK_MODEL).getroot()
    assert track.find(".//visual[@name='continuous_red_surface']") is not None
    materials = TRACK_MTL.read_text(encoding='utf-8')
    assert 'newmtl cju_track_red' in materials
    assert 'Kd 0.540000 0.180000 0.130000' in materials
    assert 'newmtl cju_lane_white' in materials


def test_osm_georeference_and_generated_mesh_bounds_are_one_contract():
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    reference = document['map']['real_world_reference']
    world = ET.parse(WORLD).getroot().find('world')
    spherical = world.find('spherical_coordinates')
    origin = [
        float(spherical.findtext('latitude_deg')),
        float(spherical.findtext('longitude_deg')),
        float(spherical.findtext('elevation')),
    ]
    assert np.allclose(origin, reference['spherical_origin_wgs84'])
    endpoint = document['frames']['stadium_endpoint']
    assert endpoint['origin_wgs84'][:2] == [
        36.653960886920, 127.495466874950]
    endpoint_enu = _wgs84_to_enu(endpoint['origin_wgs84'], origin)
    assert np.allclose(endpoint_enu[:2], endpoint['origin_enu_m'][:2],
                       atol=2.0e-6)
    heading = math.radians(reference['model_heading_deg_enu'])
    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])
    assert math.isclose(reference['model_heading_deg_enu'],
                        endpoint['heading_deg_enu'])
    shuttle = np.asarray(document['trailer']['shuttle_endpoints_enu_m'], float)
    shuttle_heading = math.degrees(math.atan2(
        shuttle[1, 1] - shuttle[0, 1], shuttle[1, 0] - shuttle[0, 0])) % 360.0
    assert math.isclose(
        (reference['model_heading_deg_enu'] + 90.0) % 360.0,
        shuttle_heading,
    )
    assert math.isclose(reference['stadium_long_axis_heading_deg_enu'],
                        shuttle_heading)
    configured_centre = (
        np.asarray(endpoint['track_center_m']) @ rotation.T
        + np.asarray(endpoint['origin_enu_m'][:2])
    )
    source_centre = _wgs84_to_enu(
        [*reference['source_track_center_wgs84'],
         reference['elevation_m_approx']], origin)
    assert np.linalg.norm(source_centre[:2] - configured_centre) < 1.0

    endpoint_pose = [float(value) for value in world.find(
        "frame[@name='stadium_endpoint']").findtext('pose').split()]
    assert np.allclose(endpoint_pose[:3], endpoint['origin_enu_m'])
    assert math.isclose(endpoint_pose[5], heading, abs_tol=1.0e-12)
    centre_frame = world.find("frame[@name='stadium_center']/pose")
    assert centre_frame.attrib['relative_to'] == 'stadium_endpoint'
    centre_local = [float(value) for value in centre_frame.text.split()]
    assert np.allclose(centre_local[:3],
                       document['frames']['stadium_center']['origin_m'])
    resolved_centre = (np.asarray(centre_local[:2]) @ rotation.T
                       + np.asarray(endpoint['origin_enu_m'][:2]))
    assert np.allclose(resolved_centre, configured_centre, atol=2.0e-6)
    for name in ('drone_cju_track_stadium',
                 'drone_cju_track_running_track'):
        include = next(item for item in world.findall('include')
                       if item.findtext('name') == name)
        pose = include.find('pose')
        assert pose.attrib['relative_to'] == 'stadium_endpoint'
        assert np.allclose([float(value) for value in pose.text.split()], 0.0)

    terrain_centre = np.asarray(document['terrain']['center_m'], float)
    terrain_half = 0.5 * np.asarray(document['terrain']['size_m'], float)
    terrain_corners = np.asarray([
        terrain_centre + [x_sign * terrain_half[0],
                          y_sign * terrain_half[1]]
        for x_sign in (-1, 1) for y_sign in (-1, 1)
    ])
    terrain_enu = (terrain_corners @ rotation.T
                   + np.asarray(endpoint['origin_enu_m'][:2]))
    bounds = document['map']['bounds_enu_m']
    assert np.allclose(
        [terrain_enu[:, 0].min(), terrain_enu[:, 0].max()], bounds['x'],
        atol=2.0e-6)
    assert np.allclose(
        [terrain_enu[:, 1].min(), terrain_enu[:, 1].max()], bounds['y'],
        atol=2.0e-6)

    track_vertices = _obj_vertices(TRACK_MESH)
    assert len(track_vertices) >= 1000
    assert np.isfinite(track_vertices).all()
    assert np.allclose(track_vertices[1, :2], [88.0, 0.0])
    assert math.isclose(track_vertices[:, 0].min(), 0.6, abs_tol=2.0e-6)
    track_size = (track_vertices[:, :2].max(axis=0)
                  - track_vertices[:, :2].min(axis=0))
    assert np.allclose(track_size, reference['track_footprint_m'][::-1],
                       atol=2.0e-6)
    assert track_vertices[:, 2].min() > 0.014
    assert np.allclose(
        [track_vertices[:, 0].min(), track_vertices[:, 0].max()],
        reference['track_bounds_stadium_endpoint_m']['x'], atol=2.0e-6)
    assert np.allclose(
        [track_vertices[:, 1].min(), track_vertices[:, 1].max()],
        reference['track_bounds_stadium_endpoint_m']['y'], atol=2.0e-6)
    mesh_text = TRACK_MESH.read_text(encoding='utf-8')
    assert mesh_text.count('\no lane_boundary_') == 9
    assert '\no continuous_red_surface\n' in mesh_text
    track_model = ET.parse(TRACK_MODEL).getroot()
    assert [visual.attrib['name']
            for visual in track_model.findall('.//visual')] == [
                'continuous_red_surface']
    assert track_model.find('.//collision') is None


def test_cju_map_contains_only_requested_facilities():
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    facilities = document['facilities'].copy()
    assert facilities.pop('coordinate_frame') == 'stadium_endpoint'
    assert set(facilities) == {
        'stadium_field', 'basketball_court', 'jokgu_court'}

    stadium = ET.parse(STADIUM_MODEL).getroot()
    visuals = {visual.attrib['name']: visual
               for visual in stadium.findall('.//visual')}
    assert set(facilities).issubset(visuals)
    assert all(name in facilities or '_line_' in name for name in visuals)
    assert stadium.find('.//collision') is None
    for name, configured in facilities.items():
        centre = np.asarray(configured['center_m'], float)
        size = np.asarray(configured['size_m'], float)
        assert np.equal(centre, np.round(centre)).all()
        pose = np.asarray([
            float(value) for value in visuals[name].findtext('pose').split()
        ])
        visual_size = np.asarray([
            float(value) for value in visuals[name].findtext(
                'geometry/box/size').split()
        ])
        assert np.allclose(pose[:2], centre)
        assert np.allclose(visual_size[:2], size)
        assert np.equal(pose[:3], np.round(pose[:3])).all()

    assert facilities['stadium_field']['center_m'] == [44, 48]
    assert facilities['basketball_court']['center_m'] == [44, 113]
    assert facilities['jokgu_court']['center_m'] == [44, -18]
    assert facilities['stadium_field']['size_m'] == [68, 105]
    assert facilities['basketball_court']['size_m'] == [28, 15]
    assert facilities['jokgu_court']['size_m'] == [10, 20]
    assert (facilities['jokgu_court']['size_m'][1]
            > facilities['jokgu_court']['size_m'][0])

    model_text = STADIUM_MODEL.read_text(encoding='utf-8')
    for removed in ('goal_', 'stand', 'royal', 'canopy', 'rail', 'tree_'):
        assert removed not in model_text
    assert not (STADIUM_MODEL.parent / 'meshes/grandstand_roof.obj').exists()
    assert not (STADIUM_MODEL.parent
                / 'meshes/stadium_plaza_surface.obj').exists()


def test_land_uses_astar_until_observation_based_landing_mpc_is_ready(
        monkeypatch):
    warnings = []
    staging_goal = np.array([4.0, 5.0, 3.0])
    logger = SimpleNamespace(
        warn=lambda message: warnings.append(message),
        error=lambda *_: None,
    )
    state = SimpleNamespace(
        auto_start=False,
        mission_command_topic='/mission/command',
        _takeoff_requested=False,
        _planner_pool=object(),
        _plan_future=None,
        p_d=np.array([0.0, 0.0, 3.0]),
        cue=np.zeros(3),
        _return_staging_goal=staging_goal,
        _return_staging=False,
        mission_map_yaml=str(MAP),
        _t_solve=1.0,
        state='PRECHECK',
        get_logger=lambda: logger,
        _cue_fresh=lambda: True,
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    landing_ready = [False]
    state._landing_mpc_entry_ready = lambda: landing_ready[0]
    state._enter_landing_mpc = lambda: (
        state._set_state('LANDING_ACQUIRE') or True)
    plans = []

    def start_plan(goal, *, return_route):
        plans.append((None if goal is None else np.asarray(goal).copy(),
                      return_route))
        state._set_state('RETURN_PLAN' if return_route else 'MISSION_PLAN')

    state._start_global_plan = start_plan
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: False)

    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'PRECHECK' and warnings
    MissionManagerNode._on_command(state, SimpleNamespace(data='takeoff'))
    assert state._takeoff_requested

    state.state = 'READY'
    MissionManagerNode._on_command(state, SimpleNamespace(data='mission'))
    assert state.state == 'MISSION_PLAN'
    assert len(plans) == 1 and plans[-1] == (None, False)
    plans.clear()

    state.state = 'HOVER'
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'RETURN_PLAN'
    assert len(plans) == 1 and plans[-1][1] is True
    assert np.allclose(plans[-1][0], staging_goal)
    assert state._return_staging
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert len(plans) == 1

    plans.clear()
    state.state = 'HOVER'
    landing_ready[0] = True
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'RETURN_PLAN'
    assert len(plans) == 1
    assert np.allclose(plans[-1][0], staging_goal)

    # Profiles without a fixed staging contract retain direct vision entry.
    plans.clear()
    state.state = 'HOVER'
    state._return_staging_goal = None
    state._return_staging = False
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'LANDING_ACQUIRE'
    assert plans == []

    # Disabling staging in YAML must win even if a stale/precomputed staging
    # coordinate exists on the node.
    plans.clear()
    state.state = 'HOVER'
    state._return_staging_goal = staging_goal
    state._return_staging_enabled = False
    state._return_staging = False
    state.cue = np.array([7.0, 8.0, 0.0])
    landing_ready[0] = False
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'RETURN_PLAN'
    assert len(plans) == 1 and plans[-1][1] is True
    assert np.allclose(plans[-1][0], state.cue)
    assert not state._return_staging


def test_land_waits_for_the_configured_three_second_hover():
    now = [12.99]
    state = SimpleNamespace(
        auto_start=False,
        mission_command_topic='/mission/command',
        _takeoff_requested=False,
        state='HOVER',
        _mission_hover_duration=3.0,
        _hover_since=10.0,
        _now=lambda: now[0],
        _return_staging_goal=None,
        _return_staging=False,
        _cue_fresh=lambda: True,
        _landing_mpc_entry_ready=lambda: True,
        get_logger=lambda: SimpleNamespace(warn=lambda *_: None),
    )
    state._enter_landing_mpc = lambda: setattr(
        state, 'state', 'LANDING_ACQUIRE') or True

    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'HOVER'

    now[0] = 13.0
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'LANDING_ACQUIRE'


def test_nonfinite_trailer_velocity_never_reaches_setpoints():
    warnings = []
    state = SimpleNamespace(
        cue_v=np.ones(3),
        _t_cue_v=1.0,
        _cue_stamp=lambda _: 1.0,
        get_logger=lambda: SimpleNamespace(
            warn=lambda *args, **kwargs: warnings.append(args[0])),
    )
    message = SimpleNamespace(
        vector=SimpleNamespace(x=float('nan'), y=1.0, z=0.0))
    MissionManagerNode._on_cue_v(state, message)
    assert np.allclose(state.cue_v, 0.0)
    assert warnings


def test_trailer_cue_requires_fresh_px4_local_enu_position_and_velocity():
    warnings = []
    state = SimpleNamespace(
        cue=None,
        cue_v=np.zeros(3),
        _t_cue=None,
        _t_cue_v=None,
        cue_timeout_s=1.0,
        _now=lambda: 10.5,
        get_logger=lambda: SimpleNamespace(
            warn=lambda *args, **kwargs: warnings.append(args[0])),
    )
    state._cue_stamp = lambda message: MissionManagerNode._cue_stamp(
        state, message)

    position = PointStamped()
    position.header.frame_id = LOCAL_ENU_FRAME_ID
    position.header.stamp.sec = 10
    position.point.x, position.point.y, position.point.z = 1.0, 2.0, 3.0
    velocity = Vector3Stamped()
    velocity.header = position.header
    velocity.vector.x = 1.0
    MissionManagerNode._on_cue(state, position)
    assert not MissionManagerNode._cue_fresh(state)
    MissionManagerNode._on_cue_v(state, velocity)
    assert MissionManagerNode._cue_fresh(state)

    # Separate DDS callbacks must not create a one-tick false stale verdict.
    position.header.stamp.nanosec = 100_000_000
    MissionManagerNode._on_cue(state, position)
    assert MissionManagerNode._cue_fresh(state)

    velocity.header.frame_id = 'map'
    MissionManagerNode._on_cue_v(state, velocity)
    assert not MissionManagerNode._cue_fresh(state)
    assert np.allclose(state.cue_v, 0.0)
    assert warnings


def test_landing_cue_projects_five_hz_sample_to_now_without_extrapolating_stale():
    now = [10.18]
    state = SimpleNamespace(
        cue=np.array([1.0, -2.0, 1.811]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        _t_cue_source=10.0,
        cue_timeout_s=1.0,
        _precland_target_timeout=0.5,
        _now=lambda: now[0],
    )

    assert np.allclose(
        MissionManagerNode._cue_at_now(state), [2.62, -2.0, 1.811])

    now[0] = 9.99
    assert np.allclose(
        MissionManagerNode._cue_at_now(state), state.cue)

    now[0] = 11.0
    assert np.allclose(
        MissionManagerNode._cue_at_now(state), [5.5, -2.0, 1.811])


def test_planning_state_keeps_offboard_heartbeat_and_hold_setpoint():
    counts = {'state': 0, 'ocm': 0, 'send': 0}
    pending = SimpleNamespace(done=lambda: False)
    node = SimpleNamespace(
        k=0,
        state='MISSION_PLAN',
        state_pub=SimpleNamespace(
            publish=lambda _: counts.__setitem__('state', counts['state'] + 1)),
        _ocm=lambda: counts.__setitem__('ocm', counts['ocm'] + 1),
        _send=lambda _: counts.__setitem__('send', counts['send'] + 1),
        p_d=np.zeros(3),
        _hold_pos=np.array([0.0, 0.0, 3.0]),
        _plan_future=pending,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        _now=lambda: 0.0,
    )
    for _ in range(50):
        MissionManagerNode._tick(node)
    assert counts == {'state': 50, 'ocm': 50, 'send': 50}


def test_initial_return_plan_enters_landing_before_the_worker_finishes():
    class PendingPlan:
        cancelled = False

        @staticmethod
        def done():
            return False

        @classmethod
        def cancel(cls):
            cls.cancelled = True

    entered = []
    state = SimpleNamespace(
        k=0,
        state='RETURN_PLAN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _send=lambda *_: (_ for _ in ()).throw(
            AssertionError('landing-ready RETURN_PLAN must not hold')),
        p_d=np.array([0.0, 0.0, 5.0]),
        cue=np.array([1.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        _mission_path=None,
        _plan_future=PendingPlan(),
        _landing_mpc_entry_ready=lambda: True,
        _enter_landing_mpc=lambda: entered.append(True) or True,
    )

    MissionManagerNode._tick(state)

    assert entered == [True]
    assert PendingPlan.cancelled
    assert state._plan_future is None


@pytest.mark.parametrize(
    ('raw_goal', 'expected_goal', 'expected_lead'), [
        ([10.0, 1.42, 0.0], [100.0, 1.42, 0.0], 10.0),
        ([95.0, 0.0, 0.0], [100.0, 0.0, 0.0], 5.0 / 9.0),
    ])
def test_noncyclic_return_plan_caps_zero_terminal_goal_at_waypoint(
        raw_goal, expected_goal, expected_lead):
    submissions = []
    published = []
    goal = np.asarray(raw_goal, float)
    state = SimpleNamespace(
        state='HOVER',
        p_d=np.array([0.0, 0.0, 5.0]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        _terminal_waypoints_local=np.array([[0.0, 0.0], [100.0, 0.0]]),
        _terminal_route_cyclic=False,
        _terminal_waypoint_tolerance=0.5,
        _terminal_route_match_tolerance=2.0,
        _terminal_min_cruise_speed=8.8,
        _return_plan_lead_s=2.0,
        _return_replan_min_period=2.0,
        _path_mpc_speed=10.0,
        _path_speed_profile_a_max=1.0,
        _path_mpc=SimpleNamespace(j_max=2.0),
        _mission_path=None,
        _plan_future=None,
        _planner_pool=SimpleNamespace(submit=lambda *args, **_kwargs: (
            submissions.append(args) or SimpleNamespace(cancel=lambda: None))),
        _publish_planned_path=published.append,
        mission_map_yaml=str(MAP),
        _now=lambda: 12.0,
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)

    MissionManagerNode._start_global_plan(state, goal, return_route=True)

    assert np.allclose(goal, raw_goal)
    assert np.allclose(submissions[0][3], expected_goal)
    assert np.allclose(state._plan_goal, submissions[0][3])
    assert state._plan_goal_lead_s == pytest.approx(expected_lead)
    assert state._last_return_plan_t == 12.0
    assert published == [None]


def test_cyclic_return_plan_projects_lead_through_the_next_leg():
    submissions = []
    goal = np.array([95.0, 1.42, 0.0])
    state = SimpleNamespace(
        state='HOVER',
        p_d=np.array([0.0, 0.0, 5.0]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        _terminal_waypoints_local=np.array([
            [0.0, 0.0], [100.0, 0.0],
            [100.0, 100.0], [0.0, 100.0],
        ]),
        _terminal_route_cyclic=True,
        _terminal_waypoint_tolerance=0.5,
        _terminal_route_match_tolerance=2.0,
        _terminal_min_cruise_speed=8.8,
        _return_plan_lead_s=2.0,
        _return_replan_min_period=2.0,
        _path_mpc_speed=10.0,
        _path_speed_profile_a_max=1.0,
        _path_mpc=SimpleNamespace(j_max=2.0),
        _mission_path=None,
        _plan_future=None,
        _planner_pool=SimpleNamespace(submit=lambda *args, **_kwargs: (
            submissions.append(args) or SimpleNamespace(cancel=lambda: None))),
        _publish_planned_path=lambda *_: None,
        mission_map_yaml=str(MAP),
        _now=lambda: 12.0,
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)

    MissionManagerNode._start_global_plan(state, goal, return_route=True)

    assert state._plan_goal_lead_s == pytest.approx(71.0 / 6.0)
    assert np.allclose(submissions[0][3], [98.5, 100.0, 0.0])
    assert np.allclose(goal, [95.0, 1.42, 0.0])


def test_outbound_plan_anchors_xy_to_the_yaml_spawn_not_ekf_noise():
    submissions = []
    measured = np.array([0.03, -0.04, 9.9])
    state = SimpleNamespace(
        state='READY',
        p_d=measured.copy(),
        _mission_path=None,
        _plan_future=None,
        _planner_pool=SimpleNamespace(submit=lambda *args, **_kwargs: (
            submissions.append(args) or SimpleNamespace(cancel=lambda: None))),
        _publish_planned_path=lambda *_: None,
        mission_map_yaml=str(MAP),
        _now=lambda: 12.0,
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)

    MissionManagerNode._start_global_plan(
        state, None, return_route=False)

    assert np.allclose(submissions[0][2], [0.0, 0.0, measured[2]])
    assert np.allclose(state._plan_start, [0.0, 0.0, measured[2]])
    assert np.allclose(state._hold_pos, measured)


def test_staging_return_plan_skips_the_moving_goal_stale_gate(monkeypatch):
    class CompletedPlan:
        @staticmethod
        def done():
            return True

        @staticmethod
        def result():
            return (
                np.array([0.0, 10.0]),
                np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]]),
                7,
                {},
            )

    commits = []
    retries = []
    state = SimpleNamespace(
        k=0,
        state='RETURN_PLAN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _send=lambda *_: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([0.0, 0.0, 5.0]),
        cue=np.zeros(3),
        cue_v=np.array([9.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _landing_mpc_entry_ready=lambda: False,
        _mission_path=None,
        _hold_pos=np.array([0.0, 0.0, 5.0]),
        _plan_future=CompletedPlan(),
        _plan_start=np.array([0.0, 0.0, 5.0]),
        _plan_goal=np.array([500.0, 0.0, 5.0]),
        _plan_goal_lead_s=0.0,
        _plan_endpoint_residual_m=math.nan,
        _plan_started_t=0.0,
        _return_plan_lead_s=2.0,
        _return_replan_min_period=2.0,
        _return_staging=True,
        mission_tolerance=0.7,
        mission_map_yaml=str(MAP),
        _now=lambda: 3.0,
        _start_global_plan=lambda *args, **kwargs: retries.append(
            (args, kwargs)),
        get_logger=lambda: SimpleNamespace(
            info=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    monkeypatch.setattr(
        MissionManagerNode, '_commit_active_path',
        lambda _self, *args, **kwargs: commits.append((args, kwargs)))

    MissionManagerNode._tick(state)

    assert state.state == 'RETURN'
    assert state._return_plan_lead_s == 3.0
    assert math.isnan(state._plan_endpoint_residual_m)
    assert len(commits) == 1
    assert retries == []


def test_rolling_return_plan_follows_old_path_then_atomically_swaps(monkeypatch):
    class PendingPlan:
        @staticmethod
        def done():
            return False

    class CompletedPlan:
        @staticmethod
        def done():
            return True

        @staticmethod
        def result():
            return (
                np.array([0.0, 10.0]),
                np.array([[0.0, 1.0, 5.0], [10.0, 1.0, 5.0]]),
                7,
            )

    gotos = []
    holds = []
    published = []
    submissions = []
    now = [0.0]
    old_path = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
    state = SimpleNamespace(
        k=0,
        state='RETURN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _send=lambda target: holds.append(np.asarray(target).copy()),
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        cue=np.array([20.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _last_safe_goto=np.array([2.0, 0.0, 5.0]),
        _planner_pool=SimpleNamespace(submit=lambda *args, **_kwargs: (
            submissions.append(args) or PendingPlan())),
        _plan_future=None,
        _publish_planned_path=published.append,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        _mission_arc_m=np.array([0.0, 10.0]),
        _mission_path=old_path.copy(),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        _mission_sample_spacing=0.1,
        mission_tolerance=0.7,
        mission_map_yaml=str(MAP),
        _now=lambda: now[0],
        get_logger=lambda: SimpleNamespace(
            info=lambda *_: None, error=lambda *_: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._landing_mpc_entry_ready = lambda: False
    state._follow_path = lambda: MissionManagerNode._follow_path(state)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)

    MissionManagerNode._start_global_plan(
        state, state.cue, return_route=True)
    assert state.state == 'RETURN_PLAN'
    assert published == []
    assert np.array_equal(state._mission_path, old_path)
    assert np.allclose(submissions[0][2], state._hold_pos)

    for x in (0.5, 1.0, 1.5):
        state.p_d = np.array([x, 0.0, 5.0])
        MissionManagerNode._tick(state)
    assert holds == []
    assert len(gotos) == 3
    assert gotos[0][0] < gotos[1][0] < gotos[2][0]

    state.p_d = np.array([2.0, 0.0, 4.87])
    now[0] = 5.0
    state._plan_future = CompletedPlan()
    MissionManagerNode._tick(state)
    assert state.state == 'RETURN'
    assert state._last_return_plan_t == 5.0
    assert len(gotos) == 4
    assert len(published) == 1
    assert np.allclose(state._mission_path[0, :2], state.p_d[:2])
    assert np.isclose(state._mission_path[0, 2], 5.0)
    assert np.allclose(state._mission_path[-1], CompletedPlan.result()[1][-1])
    assert np.all(np.diff(state._mission_arc_m) > 0.0)

    events = []
    state._send_goto = lambda *_: events.append('goto')
    state._start_global_plan = lambda goal, *, return_route: events.append(
        ('plan', np.asarray(goal).copy(), return_route))
    state._return_replan_min_period = 2.0
    state.cue = np.array([21.0, 0.0, 0.0])
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)
    now[0] = 6.999
    MissionManagerNode._tick(state)
    assert events == ['goto']

    now[0] = 7.0
    MissionManagerNode._tick(state)
    assert events[1] == 'goto'
    assert events[2][0] == 'plan' and events[2][2]
    assert np.allclose(events[2][1], [21.0, 0.0, 0.0])
    assert len(published) == 1
    assert np.allclose(
        state._mission_path[-1], CompletedPlan.result()[1][-1])


def test_rolling_return_rejects_an_unreachable_replacement(monkeypatch):
    class CompletedPlan:
        @staticmethod
        def done():
            return True

        @staticmethod
        def result():
            return (
                np.array([0.0, 10.0]),
                np.array([[0.0, 1.0, 5.0], [10.0, 1.0, 5.0]]),
                7,
            )

    old_path = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
    gotos = []
    published = []
    state = SimpleNamespace(
        k=0,
        state='RETURN_PLAN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _send=lambda *_: None,
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([1.0, 0.0, 5.0]),
        cue=np.array([20.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _plan_future=CompletedPlan(),
        _mission_arc_m=np.array([0.0, 10.0]),
        _mission_path=old_path,
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _publish_planned_path=published.append,
        _now=lambda: 0.0,
        get_logger=lambda: SimpleNamespace(
            info=lambda *_: None, error=lambda *_: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._landing_mpc_entry_ready = lambda: False
    state._follow_path = lambda: MissionManagerNode._follow_path(state)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free',
        lambda _map, _start, target: np.isclose(target[1], 0.0))
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda _map, _start, target: np.isclose(target[1], 0.0))

    MissionManagerNode._tick(state)

    assert state.state == 'RETURN'
    assert state._mission_path is old_path
    assert len(gotos) == 1
    assert published == []


@pytest.mark.parametrize('shadow_accepts', [False, True])
def test_rolling_return_shadow_mpc_controls_atomic_swap(
        monkeypatch, shadow_accepts):
    class CompletedPlan:
        @staticmethod
        def done():
            return True

        @staticmethod
        def result():
            return (
                np.array([0.0, 10.0]),
                np.array([[0.0, 1.0, 5.0], [10.0, 1.0, 5.0]]),
                7,
            )

    old_path = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
    published = []
    transitions = []
    reference_plans = []
    candidate_mpc = object()
    shadow_result = SimpleNamespace(
        predicted_pos=np.array([[2.0, 0.5, 5.0]]),
        predicted_vel=np.array([[1.0, 0.0, 0.0]]),
        predicted_acc=np.zeros((1, 3)),
    )
    prepared = (candidate_mpc, shadow_result) if shadow_accepts else None
    state = SimpleNamespace(
        k=0,
        state='RETURN_PLAN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _send=lambda *_: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([1.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        cue=np.array([28.0, 0.0, 0.0]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _plan_future=CompletedPlan(),
        _plan_goal=np.array([80.5, 0.0, 0.0]),
        _plan_goal_lead_s=47.0 / 6.0,
        _plan_started_t=0.0,
        _return_replan_min_period=2.0,
        _mission_arc_m=np.array([0.0, 10.0]),
        _mission_path=old_path,
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        _path_mpc=object(),
        _path_reference=SimpleNamespace(
            set_plan=lambda *args: reference_plans.append(args)),
        _path_solve_t=None,
        mpc_dt=0.1,
        mission_map_yaml=str(MAP),
        _publish_planned_path=published.append,
        _now=lambda: 2.0,
        get_logger=lambda: SimpleNamespace(
            info=lambda *_args, **_kwargs: None,
            warn=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None),
    )
    state._set_state = lambda new, why='': (
        transitions.append((new, why)), setattr(state, 'state', new))
    state._landing_mpc_entry_ready = lambda: False
    state._follow_path = lambda: True
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)
    monkeypatch.setattr(
        mission_module, '_active_path_sfc',
        lambda _map, path: (
            np.r_[0.0, np.cumsum(np.linalg.norm(
                np.diff(path, axis=0), axis=1))], path, {}))
    monkeypatch.setattr(
        MissionManagerNode, '_shadow_tracking_plan', lambda *_: prepared)

    MissionManagerNode._tick(state)

    assert state.state == 'RETURN'
    assert state._plan_endpoint_residual_m == pytest.approx(0.0)
    if shadow_accepts:
        assert state._mission_path is not old_path
        assert state._path_mpc is candidate_mpc
        assert len(reference_plans) == len(published) == 1
        assert transitions[-1][1] == (
            '(validated geometry B-spline -> TrackingMPC)')
    else:
        assert state._mission_path is old_path
        assert reference_plans == published == []
        assert transitions[-1][1] == '(keep prior route; unsafe swap)'


@pytest.mark.parametrize(
    ('prior_safe', 'expected_state', 'expected_reason'), [
        (True, 'RETURN', '(keep prior route; stale result)'),
        (False, 'ABORT', '(stale result and no safe prior route)'),
    ])
def test_stale_rolling_result_keeps_the_old_path_sfc_pair(
        prior_safe, expected_state, expected_reason):
    class CompletedPlan:
        @staticmethod
        def done():
            return True

        @staticmethod
        def result():
            return np.array([0.0, 1.0]), np.array([
                [0.0, 1.0, 5.0], [1.0, 1.0, 5.0]]), 1

    old_path = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
    old_sfc = {'contract': 'old'}
    replans = []
    published = []
    gotos = []
    transitions = []
    state = SimpleNamespace(
        k=0,
        state='RETURN_PLAN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _send=lambda *_: None,
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([1.0, 0.0, 5.0]),
        cue=np.array([20.0, 0.0, 0.0]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _landing_mpc_entry_ready=lambda: False,
        _follow_path=lambda: prior_safe,
        _plan_future=CompletedPlan(),
        _plan_goal=np.zeros(3),
        _plan_started_t=1.0,
        _return_replan_min_period=2.0,
        _mission_path=old_path,
        _active_sfc_diagnostics=old_sfc,
        _publish_planned_path=published.append,
        _start_global_plan=lambda goal, *, return_route: replans.append(
            (np.asarray(goal).copy(), return_route)),
        _now=lambda: 13.0,
        get_logger=lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None),
    )
    state._set_state = lambda new, why='': (
        transitions.append((new, why)), setattr(state, 'state', new))

    MissionManagerNode._tick(state)

    assert state.state == expected_state
    assert transitions[-1] == (expected_state, expected_reason)
    assert state._mission_path is old_path
    assert state._active_sfc_diagnostics is old_sfc
    assert state._plan_future is None
    assert state._plan_endpoint_residual_m == pytest.approx(20.0)
    assert state._return_plan_lead_s == 12.0
    assert replans == published == []
    assert len(gotos) == int(not prior_safe)


def test_stale_initial_return_retries_with_the_measured_planner_lead():
    class CompletedPlan:
        @staticmethod
        def done():
            return True

        @staticmethod
        def result():
            return np.array([0.0, 1.0]), np.array([
                [0.0, 1.0, 5.0], [1.0, 1.0, 5.0]]), 1

    class PendingPlan:
        @staticmethod
        def cancel():
            return None

    submissions = []
    published = []
    state = SimpleNamespace(
        k=0,
        state='RETURN_PLAN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _send=lambda *_: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([1.0, 0.0, 5.0]),
        _hold_pos=np.array([1.0, 0.0, 5.0]),
        cue=np.array([20.0, 0.0, 0.0]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _landing_mpc_entry_ready=lambda: False,
        _plan_future=CompletedPlan(),
        _plan_goal=np.zeros(3),
        _plan_started_t=1.0,
        _return_replan_min_period=2.0,
        _mission_path=None,
        _terminal_waypoints_local=np.array([[0.0, 0.0], [100.0, 0.0]]),
        _terminal_waypoint_tolerance=0.5,
        _terminal_route_match_tolerance=2.0,
        _terminal_min_cruise_speed=8.8,
        _path_mpc_speed=10.0,
        _path_speed_profile_a_max=1.0,
        _path_mpc=SimpleNamespace(j_max=2.0),
        _planner_pool=SimpleNamespace(submit=lambda *args, **_kwargs: (
            submissions.append(args) or PendingPlan())),
        _publish_planned_path=published.append,
        mission_map_yaml=str(MAP),
        mission_tolerance=0.7,
        _now=lambda: 13.0,
        get_logger=lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._start_global_plan = lambda goal, *, return_route: (
        MissionManagerNode._start_global_plan(
            state, goal, return_route=return_route))

    MissionManagerNode._tick(state)

    assert state.state == 'RETURN_PLAN'
    assert state._return_plan_lead_s == 12.0
    assert state._plan_goal_lead_s == pytest.approx(80.0 / 9.0)
    assert np.allclose(submissions[0][3], [100.0, 0.0, 0.0])
    assert published == [None]
    assert isinstance(state._plan_future, PendingPlan)


def test_failed_return_plan_holds_in_abort_without_retry(monkeypatch):
    class FailedPlan:
        @staticmethod
        def done():
            return True

        @staticmethod
        def result():
            raise RuntimeError('rejected geometry')

    retries = []
    state = SimpleNamespace(
        k=0,
        state='RETURN_PLAN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _send=lambda _: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        _cue_fresh=lambda: True,
        cue=np.array([4.0, 5.0, 0.0]),
        p_d=np.array([1.0, 2.0, 5.0]),
        _hold_pos=np.zeros(3),
        _plan_future=FailedPlan(),
        _now=lambda: 1.0,
        takeoff_alt=5.0,
        _planner_pool=object(),
        _last_return_plan_t=1.0,
        _return_replan_min_period=2.0,
        get_logger=lambda: SimpleNamespace(error=lambda *_: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._landing_mpc_entry_ready = lambda: False
    state._follow_path = lambda: MissionManagerNode._follow_path(state)
    state._start_global_plan = lambda goal, *, return_route: retries.append(
        (np.asarray(goal).copy(), return_route))

    MissionManagerNode._tick(state)

    assert state.state == 'ABORT'
    assert state._plan_future is None
    assert np.allclose(state._hold_pos, state.p_d)

    sent = []
    state.state = 'RETURN_PLAN'
    state._plan_future = FailedPlan()
    state._mission_arc_m = np.array([0.0, 10.0])
    state._mission_path = np.array([
        [1.0, 2.0, 5.0], [11.0, 2.0, 5.0]])
    state._mission_progress_m = 0.0
    state._mission_lookahead = 2.0
    state._mission_cross_track = 0.25
    state._last_safe_goto = np.array([3.0, 2.0, 5.0])
    state.cue = np.array([20.0, 2.0, 0.0])
    state._send_goto = lambda target: sent.append(np.asarray(target).copy())
    state.mission_map_yaml = str(MAP)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda *_: True)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free',
        lambda *_: True)
    MissionManagerNode._tick(state)
    assert state.state == 'RETURN'
    assert len(sent) == 1

    state.state = 'ABORT'
    MissionManagerNode._tick(state)
    assert retries == []

    MissionManagerNode._tick(state)
    assert retries == []


def test_phase0_precheck_is_fail_closed():
    status = VehicleStatus()
    status.arming_state = VehicleStatus.ARMING_STATE_DISARMED
    status.pre_flight_checks_pass = True
    status.failsafe = False
    status.failure_detector_status = VehicleStatus.FAILURE_NONE
    state = SimpleNamespace(
        p_d=np.zeros(3),
        _t_position=10.0,
        _local_valid=True,
        _ref_alt=70.0,
        _status=status,
        _t_status=10.0,
        precheck_timeout=1.0,
        engaged=False,
        cue=np.zeros(3),
        cue_v=np.zeros(3),
        _t_cue=10.0,
        _t_cue_v=10.0,
        cue_timeout_s=2.0,
        auto_start=False,
        _planner_pool=object(),
        _pending=[],
        _now=lambda: 10.5,
    )
    state._cue_fresh = lambda: MissionManagerNode._cue_fresh(state)

    assert MissionManagerNode._precheck_issues(state) == []
    state._local_valid = False
    assert 'local position invalid/stale' in MissionManagerNode._precheck_issues(
        state)
    state._local_valid = True
    status.failsafe = True
    assert 'PX4 failsafe active' in MissionManagerNode._precheck_issues(state)
    status.failsafe = False
    status.arming_state = 0
    assert 'vehicle is not confirmed disarmed' in (
        MissionManagerNode._precheck_issues(state))
    status.arming_state = VehicleStatus.ARMING_STATE_DISARMED
    state._t_status = 11.0
    assert 'vehicle status unavailable/stale' in (
        MissionManagerNode._precheck_issues(state))


def test_native_takeoff_sends_only_px4_nav_takeoff():
    commands = []
    state = SimpleNamespace(
        _ref_alt=70.0,
        takeoff_alt=5.0,
        _cmd=lambda *args, **kwargs: commands.append((args, kwargs)),
    )

    assert MissionManagerNode._send_takeoff(state)

    assert commands == [(
        (VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,), {'p7': 75.0})]

    state._ref_alt = None
    assert not MissionManagerNode._send_takeoff(state)
    assert len(commands) == 1


def test_phase1_observes_px4_takeoff_without_offboard_setpoints():
    forbidden = lambda *_: (_ for _ in ()).throw(
        AssertionError('application flight setpoint used during PX4 takeoff'))
    state = SimpleNamespace(
        k=0,
        state='TAKEOFF',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=forbidden,
        _send=forbidden,
        _send_goto=forbidden,
        p_d=np.array([1.0, 2.0, 3.0]),
        v_d=np.zeros(3),
        takeoff_alt=5.0,
        armed=True,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF),
        _now=lambda: 1.0,
        auto_start=False,
        _planner_pool=object(),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)

    MissionManagerNode._tick(state)

    assert state.state == 'TAKEOFF'

    state.p_d[2] = state.takeoff_alt
    MissionManagerNode._tick(state)
    assert state.state == 'READY'
    assert np.allclose(state._hold_pos, state.p_d)

    entered = []
    state.state = 'TAKEOFF'
    state.auto_start = True
    state._planner_pool = object()
    state._landing_target_min_speed = 9.0
    state._cue_fresh = lambda: True
    state._enter_landing_mpc = lambda: entered.append(True)
    state._enter_precland = forbidden
    state._start_global_plan = forbidden
    MissionManagerNode._tick(state)
    assert entered == [True]
    assert state.state == 'TAKEOFF'


def test_native_handoff_never_reclaims_control_after_mode_exit():
    forbidden = lambda *_: (_ for _ in ()).throw(
        AssertionError('application reclaimed control after PX4 handoff'))

    takeoff = SimpleNamespace(
        k=0,
        state='TAKEOFF',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _native_takeoff_accepted=True,
        _native_precland_accepted=False,
        _ocm=forbidden,
        _send=forbidden,
        _send_takeoff=forbidden,
        p_d=np.array([0.0, 0.0, 3.0]),
        v_d=np.zeros(3),
        takeoff_alt=5.0,
        armed=True,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_POSCTL),
        _now=lambda: 1.0,
    )
    MissionManagerNode._tick(takeoff)
    assert takeoff.state == 'TAKEOFF'

    precland = SimpleNamespace(
        k=0,
        state='PRECLAND',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _native_takeoff_accepted=False,
        _native_precland_accepted=True,
        _ocm=forbidden,
        _send=forbidden,
        _send_goto=forbidden,
        _cmd=forbidden,
        _publish_landing_target=lambda: True,
        p_d=np.zeros(3),
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_POSCTL),
        _now=lambda: 1.0,
        landed=False,
        armed=True,
    )
    MissionManagerNode._tick(precland)
    assert precland.state == 'PRECLAND'


def test_phase2_goal_completion_waits_until_slow_then_holds():
    replans = []
    waypoint = np.array([1.0, 0.0, 5.0])
    state = SimpleNamespace(
        k=0,
        state='MISSION',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=waypoint.copy(),
        v_d=np.array([0.21, 0.0, 0.0]),
        _mission_arc_m=np.array([0.0, 1.0]),
        _mission_path=np.array([[0.0, 0.0, 5.0], waypoint]),
        _mission_progress_m=1.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        _mission_sample_spacing=0.1,
        mission_tolerance=0.7,
        settle_v_tol=0.2,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _send_goto=lambda *_: None,
        _start_global_plan=lambda *args, **kwargs: replans.append((args, kwargs)),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._follow_path = lambda: MissionManagerNode._follow_path(state)

    MissionManagerNode._tick(state)
    assert state.state == 'MISSION'

    state.v_d[:] = 0.0
    MissionManagerNode._tick(state)
    assert state.state == 'HOVER'
    assert np.allclose(state._hold_pos, waypoint)
    assert replans == []


def test_hover_keeps_terminal_tracking_mpc_reference_and_holds_on_failure():
    sends = []
    gotos = []
    follows = []
    state = SimpleNamespace(
        k=0,
        state='HOVER',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([1.0, 2.0, 5.0]),
        _hold_pos=np.array([1.0, 2.0, 5.0]),
        _follow_path=lambda: follows.append(True) or True,
        _send=lambda *args: sends.append(args),
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
    )

    MissionManagerNode._tick(state)
    assert follows == [True]
    assert sends == []

    state.p_d = np.array([1.1, 2.2, 5.0])
    state._follow_path = lambda: False
    MissionManagerNode._tick(state)
    assert np.allclose(state._hold_pos, state.p_d)
    assert sends == []
    assert np.allclose(gotos, [state.p_d])


def test_phase3_replans_until_observation_based_landing_mpc_is_ready(
        monkeypatch):
    replans = []
    transitions = []
    waypoint = np.array([10.0, 0.0, 5.0])
    now = 20.0
    state = SimpleNamespace(
        k=0,
        state='RETURN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=waypoint.copy(),
        v_d=np.zeros(3),
        cue=np.array([14.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _now=lambda: now,
        _mission_arc_m=np.array([0.0, 1.0]),
        _mission_path=np.array([[9.0, 0.0, 5.0], waypoint]),
        _mission_progress_m=1.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        _mission_sample_spacing=0.1,
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _last_return_plan_t=18.0,
        _return_replan_min_period=2.0,
        _send_goto=lambda *_: None,
        _plan_future=None,
        _publish_planned_path=lambda *_: None,
        get_logger=lambda: SimpleNamespace(
            info=lambda *_: None, warn=lambda *_args, **_kwargs: None),
    )
    landing_ready = [False]
    state._landing_mpc_entry_ready = lambda: landing_ready[0]
    state._enter_landing_mpc = lambda: (
        setattr(state, 'state', 'LANDING_ACQUIRE') or True)
    state._follow_path = lambda: MissionManagerNode._follow_path(state)
    state._start_global_plan = lambda goal, *, return_route: replans.append(
        (np.asarray(goal).copy(), return_route))

    def set_state(new, why=''):
        transitions.append((new, why))
        state.state = new

    state._set_state = set_state
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free',
        lambda _map, _start, goal: not np.allclose(goal, state.cue))
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)
    MissionManagerNode._tick(state)
    assert len(replans) == 1 and replans[0][1]

    replans.clear()
    state.state = 'RETURN'
    state.cue = np.array([14.0, 0.0, 0.0])
    landing_ready[0] = True
    MissionManagerNode._tick(state)
    assert state.state == 'LANDING_ACQUIRE'
    assert replans == []


def test_staging_return_enters_gps_preacquire_only_after_all_gates(
        monkeypatch):
    replans = []
    published = []
    fresh = [False]
    runway_clear = [False]
    chord_clear = [False]
    state = SimpleNamespace(
        k=0,
        state='RETURN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        cue=np.array([20.0, 0.0, 0.0]),
        cue_v=np.array([-9.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _landing_target_fresh=lambda: fresh[0],
        mission_map_yaml=str(MAP),
        _mission_arc_m=np.array([0.0, 10.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [10.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        _last_return_plan_t=0.0,
        _return_replan_min_period=2.0,
        _return_staging=True,
        _return_staging_arrived=False,
        _return_staging_goal=np.array([10.0, 0.0, 5.0]),
        _landing_gps_preacquire_range=35.0,
        _now=lambda: 10.0,
        _follow_path=lambda: True,
        _start_global_plan=lambda *args, **kwargs: replans.append(
            (args, kwargs)),
        # A camera track may already exist at the wider city aim range, but
        # fixed staging must still use the protected GPS pre-acquire branch.
        _landing_mpc_entry_ready=lambda: True,
        _landing_mpc=SimpleNamespace(reset=lambda: None),
        _landing_reference=SimpleNamespace(reset=lambda: None),
        _publish_planned_path=published.append,
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._enter_landing_mpc = lambda **kwargs: (
        MissionManagerNode._enter_landing_mpc(state, **kwargs))
    monkeypatch.setattr(
        MissionManagerNode, '_terminal_runway_status',
        lambda *_args, **_kwargs: (runway_clear[0], 60.0))
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda *_: chord_clear[0])

    MissionManagerNode._tick(state)
    assert state.state == 'RETURN'
    assert replans == []

    state._return_staging_arrived = True
    MissionManagerNode._tick(state)
    assert state.state == 'RETURN'

    fresh[0] = True
    state.cue[0] = 36.0
    MissionManagerNode._tick(state)
    assert state.state == 'RETURN'

    state.cue[0] = 20.0
    state.cue_v[0] = 9.0
    MissionManagerNode._tick(state)
    assert state.state == 'RETURN'

    state.cue_v[0] = -9.0
    MissionManagerNode._tick(state)
    assert state.state == 'RETURN'

    runway_clear[0] = True
    MissionManagerNode._tick(state)
    assert state.state == 'RETURN'

    chord_clear[0] = True
    MissionManagerNode._tick(state)

    assert state.state == 'LANDING_ACQUIRE'
    assert not state._return_staging
    assert not state._return_staging_arrived
    assert state._gps_preacquire_active
    assert state._landing_hold_z == 5.0
    assert replans == []
    assert published == [None]


def test_return_schedules_paper_pipeline_from_latest_cue_every_two_seconds(
        monkeypatch):
    sent = []
    replans = []
    now = [11.999]
    state = SimpleNamespace(
        k=0,
        state='RETURN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        cue=np.array([31.9, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _now=lambda: now[0],
        _mission_arc_m=np.array([0.0, 10.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [10.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        _mission_sample_spacing=0.1,
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _last_return_plan_t=10.0,
        _return_replan_min_period=2.0,
        _send_goto=lambda target: sent.append(np.asarray(target).copy()),
        get_logger=lambda: SimpleNamespace(
            info=lambda *_: None, warn=lambda *_: None),
    )
    state._landing_mpc_entry_ready = lambda: False
    state._follow_path = lambda: MissionManagerNode._follow_path(state)

    def start_global_plan(goal, *, return_route):
        replans.append((np.asarray(goal).copy(), return_route))
        state._last_return_plan_t = now[0]

    state._start_global_plan = start_global_plan
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)

    MissionManagerNode._tick(state)
    assert replans == [] and len(sent) == 1

    state.cue = np.array([32.0, 0.0, 0.0])
    now[0] = 12.0
    MissionManagerNode._tick(state)
    assert len(sent) == 2
    assert len(replans) == 1 and replans[-1][1]
    assert np.allclose(replans[-1][0], [32.0, 0.0, 0.0])
    # Scheduling never mutates the active path/SFC before the worker result
    # passes A* -> SFC -> B-spline and the atomic commit gate.
    assert np.allclose(state._mission_path[-1], [10.0, 0.0, 5.0])

    now[0] = 13.999
    MissionManagerNode._tick(state)
    assert len(replans) == 1 and len(sent) == 3

    state.cue = np.array([33.0, 0.0, 0.0])
    state.v_d = np.array([6.0, 0.0, 0.0])
    state._path_mpc_speed = 12.0
    state._path_mpc_a_max = 3.0
    state._return_intercept_deadline_t = 30.0
    now[0] = 14.0
    MissionManagerNode._tick(state)
    assert len(replans) == 1 and len(sent) == 4
    assert state._return_intercept_lock_active

    state._return_intercept_deadline_t = 14.5
    MissionManagerNode._tick(state)
    assert len(replans) == 2 and len(sent) == 5
    assert np.allclose(replans[-1][0], [33.0, 0.0, 0.0])
    assert not state._return_intercept_lock_active


def test_landing_target_pose_maps_fresh_local_enu_to_px4_ned():
    published = []
    now = [10.2]
    state = SimpleNamespace(
        cue=np.array([10.0, 20.0, 1.5]),
        cue_v=np.array([3.0, 4.0, 0.0]),
        p_d=np.array([7.0, 15.0, 6.5]),
        v_d=np.array([1.0, 1.5, -0.5]),
        _cue_fresh=lambda: True,
        _t_cue_source=10.0,
        _t_cue_v_source=10.0,
        _last_landing_source_t=None,
        _precland_target_timeout=0.5,
        dt=0.02,
        vis_entry_valid=True,
        vis_valid=True,
        vis=np.array([10.0, 20.0, 1.5]),
        _bias=np.zeros(3),
        _t_vis=10.1,
        vis_fresh=0.5,
        _landing_xy_tol=0.5,
        _landing_v_tol=0.3,
        _precland_commit_height=0.65,
        _precland_commit_grace=8.0,
        _precland_commit_until=None,
        _touchdown_metric_candidate=None,
        _now=lambda: now[0],
        _target=lambda: (
            np.array([10.0, 20.0, 1.5]),
            np.array([3.0, 4.0, 0.0])),
        landing_target_pub=SimpleNamespace(publish=published.append),
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    state._landing_target_fresh = lambda: (
        MissionManagerNode._landing_target_fresh(state))
    state._vision_measurement_fresh = lambda: (
        MissionManagerNode._vision_measurement_fresh(state))

    MissionManagerNode._publish_landing_target(state)

    assert len(published) == 1
    message = published[0]
    assert isinstance(message, LandingTargetPose)
    assert message.timestamp == 0 and not message.is_static
    assert message.abs_pos_valid and message.rel_pos_valid
    assert message.rel_vel_valid
    assert np.allclose(
        [message.x_abs, message.y_abs, message.z_abs], [20.0, 10.0, -1.5])
    assert np.allclose(
        [message.x_rel, message.y_rel, message.z_rel], [5.0, 3.0, 5.0])
    assert np.allclose([message.vx_rel, message.vy_rel], [2.5, 2.0])

    now[0] = 10.3
    MissionManagerNode._publish_landing_target(state)
    assert len(published) == 1

    now[0] = 10.6
    MissionManagerNode._publish_landing_target(state)
    assert len(published) == 1

    state._t_cue_source = state._t_cue_v_source = 10.6
    state.vis_entry_valid = False
    MissionManagerNode._publish_landing_target(state)
    assert len(published) == 1


def test_landing_target_fresh_tolerates_only_one_control_tick_future_skew():
    now = [10.0]
    state = SimpleNamespace(
        _cue_fresh=lambda: True,
        _t_cue_source=10.01,
        _t_cue_v_source=10.01,
        _precland_target_timeout=0.5,
        dt=0.02,
        _now=lambda: now[0],
    )

    assert MissionManagerNode._landing_target_fresh(state)
    state._t_cue_source = state._t_cue_v_source = 10.021
    assert not MissionManagerNode._landing_target_fresh(state)
    state._t_cue_source = state._t_cue_v_source = 9.499
    assert not MissionManagerNode._landing_target_fresh(state)


def test_precland_runway_waits_for_endpoint_reversal_to_finish():
    endpoints = np.array([[0.0, 0.0], [0.0, 50.0]])

    # Imminent endpoint reversal and its low-speed transition are unsafe.
    assert _forward_endpoint_eta_s(
        [0.0, 49.5], [0.0, 1.0], endpoints, 0.2, 0.7) == pytest.approx(0.5)
    assert _forward_endpoint_eta_s(
        [0.0, 50.0], [0.0, -0.2], endpoints, 0.2, 0.7) == 0.0

    # Once the reverse leg is established, the full 50 m runway is available.
    assert _forward_endpoint_eta_s(
        [0.0, 49.5], [0.0, -1.0], endpoints, 0.2, 0.7) == pytest.approx(49.5)

    # ACQUIRE reserves the descent plus PX4's terminal landing time.  Fifteen
    # seconds is enough for the final handoff alone, but not from five metres.
    state = SimpleNamespace(
        cue=np.array([0.0, 35.0, 0.0]),
        cue_v=np.array([0.0, 1.0, 0.0]),
        _terminal_waypoints_local=endpoints,
        _terminal_waypoint_tolerance=0.2,
        _terminal_route_match_tolerance=0.2,
        _terminal_min_cruise_speed=0.7,
        _precland_runway_required_s=10.5,
        _precland_commit_height=0.65,
        _landing_mpc_vz_max=0.6,
    )
    clear, eta = MissionManagerNode._terminal_runway_status(state)
    assert clear and eta == pytest.approx(15.0)
    clear, eta = MissionManagerNode._terminal_runway_status(
        state, preparation_s=4.5)
    assert clear and eta == pytest.approx(15.0)
    clear, eta = MissionManagerNode._terminal_runway_status(
        state, preparation_s=4.51)
    assert not clear and eta == pytest.approx(15.0)
    clear, eta = MissionManagerNode._terminal_runway_status(state, 5.0)
    assert not clear and eta == pytest.approx(15.0)


def test_cyclic_route_projection_crosses_turns_and_fails_closed():
    route = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 0.0],
        [10.0, 10.0], [0.0, 10.0],
    ])

    projected = _forward_cyclic_route_position(
        [9.0, 0.0], [2.0, 0.0], route, 2.0, 0.2, 1.0, 0.5)

    assert np.allclose(projected, [10.0, 3.0])
    assert _forward_cyclic_route_position(
        [0.0, 0.0], [1.0, 0.0], np.zeros((2, 2)),
        1.0, 0.2, 0.5) is None
    assert _forward_cyclic_route_position(
        [0.0, 0.0], [1.0, 0.0], route,
        math.nan, 0.2, 0.5) is None


def _city_terminal_waypoints_local():
    document = yaml.safe_load(CITY_MAP.read_text(encoding='utf-8'))
    spawn = document['spawn']['gazebo_spawn_pose_enu']
    spawn_xy = np.array([spawn['x'], spawn['y']])
    waypoints = np.asarray(document['trailer']['waypoints_enu_m'], float)
    return document, waypoints - spawn_xy


def _city_terminal_min_speed(document):
    speed = float(document['trailer']['cruise_speed_m_s'])
    tolerance = float(
        document['px4_vehicle']['sitl_parameter_overrides']['PLD_VEL_THR'])
    return speed - tolerance


def test_city_landing_staging_uses_longest_leg_and_required_runway():
    document, waypoints = _city_terminal_waypoints_local()
    speed = document['trailer']['cruise_speed_m_s']
    acquire_altitude = document['mission']['cruise_altitude_m']
    stop_distance = mission_module._s_curve_stop_distance(
        speed, 2.0, 2.0)
    speed_change_s = 2.0 * stop_distance / speed
    entry_window_s = mission_module.DEFAULT_ENTRY_FIX_WINDOW_S
    alignment_margin_s = document['mission']['landing_alignment_margin_s']
    required_s = (
        10.5 + (acquire_altitude - 0.65) / 0.6
        + speed_change_s + entry_window_s
        + alignment_margin_s)

    staging = mission_module._landing_staging_point(
        waypoints, speed, required_s)
    longest_leg = waypoints[14] - waypoints[13]
    direction = longest_leg / np.linalg.norm(longest_leg)

    assert np.allclose(
        staging, waypoints[14] - speed * required_s * direction)
    assert np.linalg.norm(waypoints[14] - staging) == pytest.approx(
        speed * required_s)
    assert speed_change_s == pytest.approx(4.5)
    assert entry_window_s == pytest.approx(0.5)
    assert alignment_margin_s == pytest.approx(20.0)
    assert acquire_altitude == pytest.approx(10.0)
    assert np.linalg.norm(waypoints[14] - staging) == pytest.approx(
        357.5833333333)


    # The GPS cue uses the marker/deck height in PX4-local ENU.  At the 35 m
    # pre-acquire boundary the corrected staging point leaves 33.0 s for
    # velocity matching before the complete descent/PRECLAND runway expires.
    deck_z = (
        document['trailer']['spawn_pose_enu']['z']
        + document['trailer']['marker_surface_height_m']
        - document['frames']['mavros_local']['origin_enu_m'][2])
    landing_required_s = (
        10.5 + (acquire_altitude - deck_z - 0.65) / 0.6)
    entry_eta_s = (
        np.linalg.norm(waypoints[14] - staging) + 35.0) / speed
    assert deck_z == pytest.approx(1.811)
    assert entry_eta_s - landing_required_s == pytest.approx(
        33.0183333333)
    assert mission_module._landing_staging_point(
        [[0.0, 0.0], [1.0, 0.0]], speed, required_s) is None


def _city_terminal_state(waypoints, position, velocity):
    document = yaml.safe_load(CITY_MAP.read_text(encoding='utf-8'))
    return SimpleNamespace(
        cue=np.r_[position, 0.0],
        cue_v=np.r_[velocity, 0.0],
        _terminal_waypoints_local=waypoints,
        _terminal_waypoint_tolerance=0.5,
        _terminal_route_match_tolerance=2.0,
        _terminal_min_cruise_speed=_city_terminal_min_speed(document),
        _precland_runway_required_s=10.5,
        _precland_commit_height=0.65,
        _landing_mpc_vz_max=0.6,
    )


def test_city_terminal_eta_uses_the_current_straight_leg_only():
    document, waypoints = _city_terminal_waypoints_local()
    start, end = waypoints[13:15]
    leg = end - start
    direction = leg / np.linalg.norm(leg)
    position = start + 0.25 * leg
    velocity = document['trailer']['cruise_speed_m_s'] * direction
    speed = document['trailer']['cruise_speed_m_s']

    eta = _forward_endpoint_eta_s(
        position, velocity, waypoints, 0.5,
        _city_terminal_min_speed(document))

    assert eta == pytest.approx(0.75 * np.linalg.norm(leg) / speed)
    assert np.allclose(waypoints[0], [-737.0, -73.0])


def test_city_route_match_tolerance_accepts_142cm_gps_cross_track():
    document, waypoints = _city_terminal_waypoints_local()
    start, end = waypoints[13:15]
    leg = end - start
    direction = leg / np.linalg.norm(leg)
    normal = np.array([-direction[1], direction[0]])
    position = start + 0.25 * leg + 1.42 * normal
    speed = document['trailer']['cruise_speed_m_s']
    velocity = speed * direction
    route_tolerance = document['mission'][
        'terminal_route_match_tolerance_m']

    accepted = _forward_endpoint_eta_s(
        position, velocity, waypoints, 0.5,
        _city_terminal_min_speed(document), route_tolerance)
    rejected = _forward_endpoint_eta_s(
        position, velocity, waypoints, 0.5,
        _city_terminal_min_speed(document), 0.5)

    assert route_tolerance == 2.0
    assert accepted == pytest.approx(0.75 * np.linalg.norm(leg) / speed)
    assert rejected == 0.0


def test_city_long_leg_eta_tolerates_small_gps_velocity_angle_noise():
    document, waypoints = _city_terminal_waypoints_local()
    leg = waypoints[14] - waypoints[13]
    direction = leg / np.linalg.norm(leg)
    normal = np.array([-direction[1], direction[0]])
    speed = document['trailer']['cruise_speed_m_s']
    noisy_velocity = speed * direction + 0.02 * normal

    eta = _forward_endpoint_eta_s(
        waypoints[13], noisy_velocity, waypoints, 0.5,
        _city_terminal_min_speed(document))

    assert eta == pytest.approx(np.linalg.norm(leg)
                                / np.linalg.norm(noisy_velocity))


def test_city_terminal_gate_holds_for_an_imminent_turn_or_stop():
    document, waypoints = _city_terminal_waypoints_local()
    leg = waypoints[14] - waypoints[13]
    direction = leg / np.linalg.norm(leg)
    state = _city_terminal_state(
        waypoints, waypoints[14] - 2.0 * direction,
        document['trailer']['cruise_speed_m_s'] * direction)

    clear, eta = MissionManagerNode._terminal_runway_status(state)
    assert not clear and eta == pytest.approx(
        2.0 / document['trailer']['cruise_speed_m_s'])

    state.cue_v[:2] = (
        _city_terminal_min_speed(document) - 0.01) * direction
    clear, eta = MissionManagerNode._terminal_runway_status(state)
    assert not clear and eta == 0.0


def test_city_terminal_gate_rejects_short_runway_from_25_metres():
    document, waypoints = _city_terminal_waypoints_local()
    leg = waypoints[12] - waypoints[11]
    direction = leg / np.linalg.norm(leg)
    speed = document['trailer']['cruise_speed_m_s']
    state = _city_terminal_state(waypoints, waypoints[11], speed * direction)

    clear, eta = MissionManagerNode._terminal_runway_status(state, 25.0)

    assert not clear
    assert eta == pytest.approx(np.linalg.norm(leg) / speed)


def test_city_terminal_gate_accepts_a_long_clear_leg_from_25_metres():
    document, waypoints = _city_terminal_waypoints_local()
    leg = waypoints[14] - waypoints[13]
    direction = leg / np.linalg.norm(leg)
    speed = document['trailer']['cruise_speed_m_s']
    state = _city_terminal_state(waypoints, waypoints[13], speed * direction)

    clear, eta = MissionManagerNode._terminal_runway_status(state, 25.0)

    assert clear
    assert eta == pytest.approx(np.linalg.norm(leg) / speed)


def test_precland_latches_a_bounded_aligned_camera_blind_commit():
    now = [10.0]
    state = SimpleNamespace(
        cue=np.zeros(3),
        cue_v=np.array([1.0, 0.0, 0.0]),
        p_d=np.array([0.1, 0.0, 0.64]),
        v_d=np.array([1.0, 0.0, 0.0]),
        _bias=np.zeros(3),
        _landing_xy_tol=0.5,
        _landing_v_tol=0.3,
        _precland_commit_height=0.65,
        _precland_commit_grace=8.0,
        _precland_commit_until=None,
        _touchdown_metric_candidate=None,
        _now=lambda: now[0],
        _vision_measurement_fresh=lambda: True,
        state='LANDING_DESCEND',
        get_logger=lambda: SimpleNamespace(info=lambda *_: None),
    )

    assert MissionManagerNode._precland_target_allowed(state)
    assert np.isclose(state._precland_commit_until, 18.0)
    MissionManagerNode._set_state(state, 'PRECLAND')
    assert np.isclose(state._precland_commit_until, 18.0)

    state._vision_measurement_fresh = lambda: False
    now[0] = 10.36
    state.p_d[2] = 0.66
    assert MissionManagerNode._precland_target_allowed(state)
    # A small vertical estimator rebound must not command a reacquisition
    # climb after the qualified low-altitude handoff.
    state.v_d[0] = 1.31
    assert not MissionManagerNode._precland_target_allowed(state)
    state.v_d[0] = 1.0
    state.p_d[0] = 0.51
    assert not MissionManagerNode._precland_target_allowed(state)
    state.p_d[0] = 0.1
    now[0] = 18.01
    assert not MissionManagerNode._precland_target_allowed(state)

    state._precland_commit_until = None
    now[0] = 11.0
    assert not MissionManagerNode._precland_target_allowed(state)


def test_precland_loss_reclaims_offboard_from_the_current_state():
    reset = []
    state = SimpleNamespace(
        state='PRECLAND',
        p_d=np.array([2.0, 3.0, 0.2]),
        cue=np.array([2.0, 3.0, 0.0]),
        _landing_mpc_handoff_height=1.5,
        _landing_mpc=SimpleNamespace(reset=lambda: reset.append('mpc')),
        _landing_reference=SimpleNamespace(reset=lambda: reset.append('ref')),
        _landing_solve_t=1.0,
        _landing_last_solve_t=1.0,
        _native_precland_accepted=True,
        _last_offboard_cmd=1.0,
        _precland_commit_until=11.0,
        _gps_preacquire_active=True,
        _now=lambda: 10.0,
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)

    MissionManagerNode._recover_precland(state)

    assert reset == ['mpc', 'ref']
    assert state.state == 'LANDING_ACQUIRE'
    assert np.isclose(state._landing_hold_z, 0.2)
    assert np.allclose(state._hold_pos, [2.0, 3.0, 0.2])
    assert not state._native_precland_accepted
    assert state._precland_commit_until is None
    assert state._gps_preacquire_active

    state.get_logger = lambda: SimpleNamespace(info=lambda *_: None)
    state._log_experiment_metrics = lambda: None
    MissionManagerNode._set_state(state, 'DONE')
    assert not state._gps_preacquire_active


def test_landing_mpc_entry_uses_observation_and_safety_not_distance(
        monkeypatch):
    safety = [True]
    runway_clear = [True]
    state = SimpleNamespace(
        p_d=np.array([0.0, 0.0, 5.0]),
        cue=np.array([3.58, 0.0, 0.0]),
        v_d=np.array([3.84, 0.0, 0.0]),
        cue_v=np.zeros(3),
        vis_entry_valid=True,
        mission_map_yaml=str(MAP),
        _landing_gps_preacquire_range=35.0,
        _landing_target_min_speed=0.0,
        _landing_target_fresh=lambda: True,
        _vision_measurement_fresh=lambda: True,
        _vision_correction_converged=lambda: True,
    )
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda *_: safety[0])
    monkeypatch.setattr(
        MissionManagerNode, '_terminal_runway_status',
        lambda *_args, **_kwargs: (runway_clear[0], 60.0))

    assert MissionManagerNode._landing_mpc_entry_ready(state)
    state.cue[0] = 80.0
    assert MissionManagerNode._landing_mpc_entry_ready(state)
    safety[0] = False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    safety[0] = True
    state.vis_entry_valid = False
    state._vision_correction_converged = lambda: False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    # Moving-deck acquisition may begin before vision, but only after the
    # route controller has closed to the configured GPS handoff range.
    state._landing_target_min_speed = 9.0
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    # A fresh visual track may extend entry beyond the GPS range, but it must
    # never bypass the same moving-route runway gate.
    state.vis_entry_valid = True
    state._vision_measurement_fresh = lambda: True
    state._vision_correction_converged = lambda: True
    runway_clear[0] = False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    runway_clear[0] = True
    assert MissionManagerNode._landing_mpc_entry_ready(state)
    state.vis_entry_valid = False
    state._vision_correction_converged = lambda: False
    state.cue[0] = 35.0
    assert MissionManagerNode._landing_mpc_entry_ready(state)
    state.v_d[0] = 0.0
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    state.v_d[0] = 3.84
    runway_clear[0] = False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    runway_clear[0] = True
    state._landing_gps_preacquire_range = 0.0
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    state._landing_gps_preacquire_range = 35.0
    safety[0] = False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    safety[0] = True
    state._landing_target_fresh = lambda: False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    state._landing_target_fresh = lambda: True
    state._landing_target_min_speed = 0.0
    state.vis_entry_valid = True
    state._vision_measurement_fresh = lambda: False
    state._vision_correction_converged = lambda: False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)


def test_direct_moving_target_entry_enables_gps_safety_mode(monkeypatch):
    resets = []
    published = []
    transitions = []
    state = SimpleNamespace(
        p_d=np.array([0.0, 0.0, 10.0]),
        v_d=np.array([12.0, 0.0, 0.0]),
        cue=np.array([20.0, 0.0, 0.0]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        vis_entry_valid=False,
        mission_map_yaml=str(MAP),
        _landing_gps_preacquire_range=35.0,
        _landing_target_min_speed=9.0,
        _landing_target_fresh=lambda: True,
        _vision_measurement_fresh=lambda: False,
        _vision_correction_converged=lambda: False,
        state='RETURN',
        _return_staging=True,
        _return_staging_arrived=True,
        _landing_mpc=SimpleNamespace(
            reset=lambda: resets.append('mpc')),
        _landing_reference=SimpleNamespace(
            reset=lambda: resets.append('reference')),
        _publish_planned_path=published.append,
    )
    state._landing_mpc_entry_ready = lambda: (
        MissionManagerNode._landing_mpc_entry_ready(state))
    state._set_state = lambda new, why='': transitions.append((new, why))
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)
    monkeypatch.setattr(
        MissionManagerNode, '_terminal_runway_status',
        lambda *_args, **_kwargs: (True, 60.0))

    assert MissionManagerNode._enter_landing_mpc(state)

    assert state._gps_preacquire_active
    assert not state._return_staging
    assert not state._return_staging_arrived
    assert state._landing_hold_z == 10.0
    assert resets == ['mpc', 'reference']
    assert published == [None]
    assert transitions == [(
        'LANDING_ACQUIRE',
        '(GPS velocity pre-acquire; hold altitude for ArUco)')]


def test_takeoff_direct_entry_preserves_stationary_trailer_release(monkeypatch):
    transitions = []
    state = SimpleNamespace(
        state='TAKEOFF',
        auto_start=True,
        p_d=np.array([0.0, 0.0, 10.0]),
        v_d=np.zeros(3),
        cue=np.array([0.0, 0.0, 0.0]),
        cue_v=np.zeros(3),
        vis_entry_valid=False,
        mission_map_yaml=str(MAP),
        _landing_gps_preacquire_range=35.0,
        _landing_target_min_speed=9.0,
        _landing_target_fresh=lambda: True,
        _vision_measurement_fresh=lambda: False,
        _vision_correction_converged=lambda: False,
        _return_staging=False,
        _return_staging_arrived=False,
        _landing_mpc=SimpleNamespace(reset=lambda: None),
        _landing_reference=SimpleNamespace(reset=lambda: None),
        _publish_planned_path=lambda *_: None,
    )
    state._landing_mpc_entry_ready = lambda: (
        MissionManagerNode._landing_mpc_entry_ready(state))
    state._set_state = lambda new, why='': transitions.append((new, why))
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)
    monkeypatch.setattr(
        MissionManagerNode, '_terminal_runway_status',
        lambda *_args, **_kwargs: (False, 0.0))

    assert MissionManagerNode._enter_landing_mpc(state)

    assert not state._gps_preacquire_active
    assert transitions == [(
        'LANDING_ACQUIRE',
        '(direct moving-target ACQUIRE; hold altitude for ArUco)')]


def test_fresh_vision_needs_no_depression_angle():
    state = SimpleNamespace(
        vis_entry_valid=True,
        vis_valid=True,
        vis=np.array([80.0, 0.0, 0.0]),
        cue=np.zeros(3),
        _t_vis=10.0,
        vis_fresh=0.5,
        _now=lambda: 10.1,
    )

    assert MissionManagerNode._vision_measurement_fresh(state)
    state.vis_entry_valid = False
    assert not MissionManagerNode._vision_measurement_fresh(state)
    assert MissionManagerNode._vision_track_usable(state)


def test_vision_bias_waits_for_kf_velocity_convergence():
    state = SimpleNamespace(
        cue=np.array([10.0, 20.0, 0.0]),
        cue_v=np.array([7.0, 0.0, 0.0]),
        vis=np.array([10.6, 20.0, 0.0]),
        vis_v=np.zeros(3),
        vis_valid=True,
        vis_entry_valid=True,
        _t_vis=1.0,
        vis_fresh=0.5,
        _now=lambda: 1.0,
        _landing_v_tol=0.3,
        _bias=np.zeros(3),
        bias_max=5.0,
        bias_tau=1.5,
        bias_rate=0.3,
        dt=0.02,
        _t_cue_source=None,
        state='LANDING_ACQUIRE',
        _landing_bias_locked=False,
    )

    assert not MissionManagerNode._vision_velocity_converged(state)
    target, _ = MissionManagerNode._target(state)
    assert np.allclose(target, state.cue)
    assert np.allclose(state._bias, 0.0)

    state.vis_v = state.cue_v.copy()
    assert MissionManagerNode._vision_velocity_converged(state)
    target, _ = MissionManagerNode._target(state)
    assert np.allclose(target[:2], [10.006, 20.0])
    assert np.allclose(state._bias[:2], [0.006, 0.0])

    state._landing_bias_locked = True
    state.vis = np.array([10.9, 20.0, 0.0])
    target, _ = MissionManagerNode._target(state)
    assert np.allclose(target[:2], [10.006, 20.0])
    assert np.allclose(state._bias[:2], [0.006, 0.0])


def test_gps_preacquire_holds_altitude_and_matches_velocity_until_vision_aligns(
        monkeypatch):
    class FakeMpc:
        N = 2
        j_max = 2.0
        cone_k = 2.0
        z_ref = 1.5
        a_max = 1.0

        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def solve(self, p_rel, v_rel, *_, **_kwargs):
            return SimpleNamespace(
                success=True,
                pred_rel_pos=np.tile(np.asarray(p_rel, float), (2, 1)),
                pred_rel_vel=np.tile(np.asarray(v_rel, float), (2, 1)),
                pred_rel_acc=np.zeros((2, 3)),
            )

    sent = []
    gotos = []
    prediction_modes = []
    planning_streams = []
    now = [10.0]
    status = VehicleStatus()
    status.arming_state = VehicleStatus.ARMING_STATE_ARMED
    status.nav_state = VehicleStatus.NAVIGATION_STATE_OFFBOARD
    status.failsafe = False
    status.failure_detector_status = VehicleStatus.FAILURE_NONE
    state = SimpleNamespace(
        state='LANDING_ACQUIRE',
        p_d=np.array([0.0, 0.0, 25.0]),
        v_d=np.array([9.0, 0.0, 0.0]),
        cue=np.array([0.0, 0.0, 2.0]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        mission_map_yaml=str(MAP),
        _landing_mpc=FakeMpc(),
        _landing_reference=mission_module.HorizonReference(lead_s=0.1),
        _landing_solve_t=None,
        _landing_last_solve_t=None,
        _landing_failure_hold=None,
        _gps_preacquire_active=True,
        _landing_hold_z=25.0,
        _landing_mpc_cone=2.0,
        _path_mpc_a_max=3.0,
        # Route tracking is capped below the terminal GPS matching speed.
        _path_mpc_v_max=8.0,
        _landing_mpc_a_max=1.0,
        _landing_mpc_vz_max=0.6,
        _landing_xy_tol=0.5,
        _landing_v_tol=0.3,
        _landing_target_min_speed=9.0,
        _precland_commit_height=0.65,
        _local_valid=True,
        _t_position=now[0],
        _status=status,
        _t_status=now[0],
        armed=True,
        mpc_dt=0.1,
        dt=0.02,
        _last_sent_acceleration=np.zeros(3),
        _last_sent_acceleration_t=9.98,
        vis_entry_valid=False,
        _landing_target_fresh=lambda: True,
        _vision_correction_converged=lambda: state.vis_entry_valid,
        _vision_measurement_fresh=lambda: state.vis_entry_valid,
        _target=lambda: (state.cue.copy(), state.cue_v.copy()),
        _now=lambda: now[0],
        _send=lambda pos, vel=None, acc=None: sent.append(
            (np.asarray(pos).copy(), np.asarray(vel).copy(),
             np.asarray(acc).copy())),
        _send_goto=lambda pos: gotos.append(np.asarray(pos).copy()),
        _enter_precland=lambda *_: pytest.fail(
            'GPS pre-acquire must not skip to PRECLAND'),
        get_logger=lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    monkeypatch.setattr(
        mission_module, '_mpc_prediction_is_safe',
        lambda *_args, **kwargs: (
            prediction_modes.append(kwargs.get('planning', False)) or True))
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda *_: planning_streams.append(True) or True)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free',
        lambda *_: pytest.fail(
            'GPS pre-acquire used hard instead of planning clearance'))
    monkeypatch.setattr(
        MissionManagerNode, '_terminal_runway_status',
        lambda *_args, **_kwargs: (True, 60.0))

    MissionManagerNode._run_landing_mpc(state)

    assert state.state == 'LANDING_ACQUIRE'
    assert sent[-1][0][2] == 25.0
    assert sent[-1][1] == pytest.approx([9.0, 0.0, 0.0])
    assert sent[-1][2][2] == 0.0

    state.vis_entry_valid = True
    state.p_d[0] = 1.0
    now[0] += 0.1
    state._t_position = now[0]
    MissionManagerNode._run_landing_mpc(state)
    assert state.state == 'LANDING_ACQUIRE'

    state.p_d[0] = 0.0
    state.cue_v[0] = state.v_d[0] = 8.8
    now[0] += 0.1
    state._t_position = now[0]
    MissionManagerNode._run_landing_mpc(state)
    assert state.state == 'LANDING_ACQUIRE'

    state.cue_v[0] = state.v_d[0] = 9.0
    now[0] += 0.1
    state._t_position = now[0]
    MissionManagerNode._run_landing_mpc(state)
    assert state.state == 'LANDING_DESCEND'
    assert state._gps_preacquire_active
    transition_reference = state._landing_reference
    transition_solve_t = state._landing_solve_t
    assert transition_reference.ready()
    assert transition_solve_t == now[0]
    assert prediction_modes == [True, True, True, True]
    assert planning_streams == [True, True, True, True]

    # A one-frame correction residual must stop vertical motion without
    # throwing away the certified horizontal P/V/A and falling back to Goto.
    monkeypatch.setattr(
        MissionManagerNode, '_vision_track_usable', lambda *_: False)
    state.vis_entry_valid = False
    now[0] += 0.02
    state._t_position = now[0]
    expected_pos, expected_vel, expected_acc = transition_reference.sample(
        now[0] - transition_solve_t)
    MissionManagerNode._run_landing_mpc(state)

    assert state.state == 'LANDING_ACQUIRE'
    assert state._landing_reference is transition_reference
    assert state._landing_reference.ready()
    assert state._landing_solve_t == transition_solve_t
    assert state._landing_mpc.reset_count == 0
    assert gotos == []
    assert np.allclose(sent[-1][0][:2], expected_pos[:2])
    assert np.allclose(sent[-1][1][:2], expected_vel[:2])
    assert np.allclose(sent[-1][2][:2], expected_acc[:2])
    assert sent[-1][0][2] == state.p_d[2]
    assert sent[-1][1][2] == 0.0
    assert sent[-1][2][2] == 0.0

    state.state = 'LANDING_ACQUIRE'
    state._gps_preacquire_active = True
    state._landing_reference.reset()
    state._landing_solve_t = None
    state._landing_last_solve_t = None
    state._landing_failure_hold = None
    state.cue_v[0] = state.v_d[0] = 12.01
    first_hold = state.p_d.copy()
    now[0] += 0.1
    MissionManagerNode._run_landing_mpc(state)
    state.p_d = np.array([1.0, 1.0, 24.0])
    now[0] += 0.1
    MissionManagerNode._run_landing_mpc(state)

    assert np.allclose(gotos, [first_hold, first_hold])
    assert np.allclose(state._landing_failure_hold, first_hold)
    assert not state._landing_reference.ready()
    assert state._landing_solve_t is None


@pytest.mark.parametrize('staging_enabled', [True, False])
def test_gps_preacquire_runway_expiry_honours_return_mode(
        monkeypatch, staging_enabled):
    gotos = []
    path = np.array([[0.0, 0.0, 25.0], [10.0, 0.0, 25.0]])
    state = SimpleNamespace(
        state='LANDING_ACQUIRE',
        p_d=np.array([3.0, 0.0, 25.0]),
        v_d=np.array([7.0, 0.0, 0.0]),
        cue=np.array([5.0, 0.0, 2.0]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        _landing_target_fresh=lambda: True,
        _target=lambda: (state.cue.copy(), state.cue_v.copy()),
        _gps_preacquire_active=True,
        _landing_mpc=SimpleNamespace(reset=lambda: None),
        _landing_reference=SimpleNamespace(reset=lambda: None),
        _landing_solve_t=1.0,
        _landing_last_solve_t=1.0,
        _landing_hold_z=25.0,
        _landing_failure_hold=None,
        _bias=np.array([1.0, -1.0, 0.0]),
        _path_failure_hold=None,
        _return_staging_enabled=staging_enabled,
        _return_staging=False,
        _return_staging_arrived=False,
        _return_intercept_deadline_t=12.0,
        _return_intercept_lock_active=True,
        _last_return_plan_t=8.0,
        _mission_path=path,
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    monkeypatch.setattr(
        MissionManagerNode, '_terminal_runway_status',
        lambda *_args, **_kwargs: (False, 4.0))

    MissionManagerNode._run_landing_mpc(state)

    assert state.state == 'RETURN'
    assert state._return_staging is staging_enabled
    assert not state._return_staging_arrived
    assert not state._gps_preacquire_active
    assert np.allclose(state._bias, 0.0)
    assert np.allclose(state._path_failure_hold, [3.0, 0.0, 25.0])
    assert np.allclose(gotos, [[3.0, 0.0, 25.0]])
    assert state._mission_path is path
    if staging_enabled:
        assert state._return_intercept_deadline_t == 12.0
        assert state._return_intercept_lock_active
        assert state._last_return_plan_t == 8.0
    else:
        assert state._return_intercept_deadline_t is None
        assert not state._return_intercept_lock_active
        assert state._last_return_plan_t is None


def test_staging_descent_coast_reacquire_expires_to_fixed_restage(
        monkeypatch):
    class FakeMpc:
        N = 2
        j_max = 2.0
        cone_k = 2.0
        z_ref = 0.65
        a_max = 1.0

        def reset(self):
            pass

        def solve(self, p_rel, v_rel, *_, **_kwargs):
            return SimpleNamespace(
                success=True,
                pred_rel_pos=np.tile(np.asarray(p_rel, float), (2, 1)),
                pred_rel_vel=np.tile(np.asarray(v_rel, float), (2, 1)),
                pred_rel_acc=np.zeros((2, 3)),
            )

    sent = []
    gotos = []
    runway_clear = [True]
    path = np.array([[0.0, 0.0, 25.0], [10.0, 0.0, 25.0]])
    state = SimpleNamespace(
        state='LANDING_DESCEND',
        p_d=np.array([3.0, 0.0, 20.0]),
        v_d=np.array([9.0, 0.0, 0.0]),
        cue=np.array([3.0, 0.0, 2.0]),
        cue_v=np.array([9.0, 0.0, 0.0]),
        mission_map_yaml=str(MAP),
        _landing_target_fresh=lambda: True,
        _target=lambda: (state.cue.copy(), state.cue_v.copy()),
        _vision_correction_converged=lambda: False,
        _vision_measurement_fresh=lambda: False,
        _gps_preacquire_active=True,
        _landing_mpc=FakeMpc(),
        _landing_reference=mission_module.HorizonReference(lead_s=0.1),
        _landing_solve_t=None,
        _landing_last_solve_t=None,
        _landing_hold_z=None,
        _landing_failure_hold=None,
        _path_failure_hold=None,
        _return_staging=False,
        _return_staging_arrived=False,
        _mission_path=path,
        _landing_mpc_cone=2.0,
        _path_mpc_a_max=3.0,
        _landing_mpc_a_max=1.0,
        _landing_xy_tol=0.5,
        _landing_v_tol=0.3,
        _precland_commit_height=0.65,
        mpc_dt=0.1,
        dt=0.02,
        _last_sent_acceleration=np.zeros(3),
        _last_sent_acceleration_t=9.98,
        _now=lambda: 10.0,
        _send=lambda pos, vel=None, acc=None: sent.append(
            (np.asarray(pos).copy(), np.asarray(vel).copy(),
             np.asarray(acc).copy())),
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        get_logger=lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    monkeypatch.setattr(
        MissionManagerNode, '_terminal_runway_status',
        lambda *_args, **_kwargs: (runway_clear[0], 60.0))
    monkeypatch.setattr(
        MissionManagerNode, '_vision_track_usable', lambda *_: False)
    monkeypatch.setattr(
        mission_module, '_mpc_prediction_is_safe',
        lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)

    MissionManagerNode._run_landing_mpc(state)

    assert state.state == 'LANDING_ACQUIRE'
    assert state._gps_preacquire_active
    assert len(sent) == 1 and gotos == []

    runway_clear[0] = False
    first_hold = state.p_d.copy()
    MissionManagerNode._run_landing_mpc(state)

    assert state.state == 'RETURN'
    assert state._return_staging and not state._return_staging_arrived
    assert not state._gps_preacquire_active
    assert len(sent) == 1
    assert np.allclose(gotos, [first_hold])
    assert np.allclose(state._path_failure_hold, first_hold)
    assert state._mission_path is path


def test_landing_mpc_acquires_without_descent_then_hands_low_altitude_to_px4(
        monkeypatch):
    class FakeMpc:
        N = 2
        j_max = 2.0
        cone_k = 2.0
        z_ref = 1.5
        a_max = 1.0

        def __init__(self):
            self.solve_count = 0

        def reset(self):
            pass

        def solve(self, p_rel, v_rel, *_, **_kwargs):
            self.solve_count += 1
            return SimpleNamespace(
                success=True,
                pred_rel_pos=np.tile(np.asarray(p_rel, float), (2, 1)),
                pred_rel_vel=np.tile(np.asarray(v_rel, float), (2, 1)),
                pred_rel_acc=np.zeros((2, 3)),
            )

    class FakeReference:
        def __init__(self):
            self._ready = False
            self.lead = 0.1
            self.T = 0.2

        def ready(self):
            return self._ready

        def reset(self):
            self._ready = False

        def set_plan(self, *_):
            self._ready = True

        def sample(self, _elapsed):
            return (state.p_d.copy(), np.zeros(3),
                    np.array([0.0, 0.0, -0.2]))

    sent = []
    gotos = []
    precland = []
    runway_clear = [False]
    runway_heights = []
    correction_ready = [True]
    measurement_fresh = [True]
    track_usable = [True]
    now = [10.0]
    state = SimpleNamespace(
        k=0,
        state='LANDING_ACQUIRE',
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        cue=np.array([0.0, 0.0, 0.0]),
        cue_v=np.zeros(3),
        mission_map_yaml=str(MAP),
        _landing_mpc=FakeMpc(),
        _landing_reference=FakeReference(),
        _landing_solve_t=None,
        _landing_last_solve_t=None,
        _landing_hold_z=5.0,
        _landing_mpc_handoff_height=1.5,
        _precland_commit_height=0.65,
        _landing_mpc_cone=2.0,
        _path_mpc_a_max=3.0,
        _landing_mpc_a_max=1.0,
        _landing_mpc_vz_max=0.6,
        _landing_xy_tol=0.5,
        _landing_v_tol=0.3,
        _mpc_solve_every=5,
        mpc_dt=0.1,
        dt=0.02,
        _last_sent_acceleration=np.zeros(3),
        _last_sent_acceleration_t=9.98,
        vis_entry_valid=True,
        _landing_target_fresh=lambda: True,
        _target=lambda: (state.cue.copy(), state.cue_v.copy()),
        _vision_correction_converged=lambda: correction_ready[0],
        _vision_measurement_fresh=lambda: measurement_fresh[0],
        _now=lambda: now[0],
        _send=lambda pos, vel=None, acc=None: sent.append(
            (np.asarray(pos).copy(), np.asarray(vel).copy(),
             np.asarray(acc).copy())),
        _send_goto=lambda pos: gotos.append(np.asarray(pos).copy()),
        _enter_precland=lambda distance: precland.append(distance) or True,
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)

    def runway_status(_self, descent_height_m=None):
        runway_heights.append(descent_height_m)
        return runway_clear[0], 15.0

    monkeypatch.setattr(
        MissionManagerNode, '_terminal_runway_status', runway_status)
    monkeypatch.setattr(
        MissionManagerNode, '_vision_track_usable',
        lambda _self: track_usable[0])

    MissionManagerNode._run_landing_mpc(state)
    assert state.state == 'LANDING_ACQUIRE'
    assert state._landing_mpc.a_max == 3.0
    assert sent and sent[-1][0][2] == 5.0
    assert sent[-1][1][2] == 0.0 and sent[-1][2][2] == 0.0
    assert precland == []
    assert runway_heights == [5.0]

    runway_clear[0] = True
    now[0] = 10.02
    MissionManagerNode._run_landing_mpc(state)
    assert state.state == 'LANDING_DESCEND'
    assert runway_heights[-1] == 5.0

    # Once DESCEND is entered, a fresh ArUco measurement keeps descent live
    # even while its slowly learned cue correction crosses the entry bound.
    correction_ready[0] = False
    state._landing_reference.reset()
    now[0] = 10.04
    MissionManagerNode._run_landing_mpc(state)
    assert state._landing_mpc.a_max == 1.0
    assert state._landing_mpc.z_ref == pytest.approx(0.53)
    assert state._landing_mpc.solve_count == 1
    assert precland == []

    state.vis_entry_valid = False
    measurement_fresh[0] = False
    state.p_d[2] = 1.0
    state._last_sent_acceleration[2] = -0.4
    state._last_sent_acceleration_t = 10.08
    now[0] = 10.1
    MissionManagerNode._run_landing_mpc(state)
    assert state._landing_mpc.solve_count == 2
    assert state.state == 'LANDING_DESCEND'
    assert sent[-1][0][2] == pytest.approx(1.0)
    assert sent[-1][1][2] == pytest.approx(0.0)
    assert sent[-1][2][2] == pytest.approx(0.0)

    track_usable[0] = False
    now[0] = 10.2
    MissionManagerNode._run_landing_mpc(state)
    assert state._landing_mpc.solve_count == 3
    assert state.state == 'LANDING_ACQUIRE'
    assert state._landing_mpc.z_ref == pytest.approx(1.0)
    assert sent[-1][0][2] == pytest.approx(1.0)
    assert sent[-1][1][2] == pytest.approx(0.0)
    assert sent[-1][2][2] == pytest.approx(0.0)

    now[0] = 10.3
    MissionManagerNode._run_landing_mpc(state)
    assert state._landing_mpc.solve_count == 4
    assert state.state == 'LANDING_ACQUIRE'
    assert state._landing_mpc.z_ref == pytest.approx(1.0)
    assert sent[-1][0][2] == pytest.approx(1.0)

    state.vis_entry_valid = True
    correction_ready[0] = True
    measurement_fresh[0] = True
    now[0] = 10.4
    MissionManagerNode._run_landing_mpc(state)
    assert state._landing_mpc.solve_count == 5
    assert state.state == 'LANDING_DESCEND'

    state.p_d[2] = 1.5
    now[0] = 10.5
    MissionManagerNode._run_landing_mpc(state)
    assert state._landing_mpc.solve_count == 6
    assert precland == []

    state.p_d[2] = 0.65
    now[0] = 10.6
    MissionManagerNode._run_landing_mpc(state)
    assert state._landing_mpc.solve_count == 7
    assert precland == [0.0]

    # A low-altitude reacquisition hands off directly: one DESCEND tick would
    # otherwise raise z_ref from 0.30 m to the 0.65 m commit height.
    precland.clear()
    state.state = 'LANDING_ACQUIRE'
    state.p_d[2] = 0.3
    state._landing_hold_z = 0.3
    state._landing_reference.reset()
    now[0] = 10.7
    MissionManagerNode._run_landing_mpc(state)
    assert precland == [0.0]
    assert np.allclose(state._last_safe_goto, [0.0, 0.0, 0.3])


def test_precland_terminal_latch_never_reclaims_into_a_climb():
    warnings = []
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError('terminal PRECLAND reclaimed Offboard'))
    state = SimpleNamespace(
        k=0,
        state='PRECLAND',
        state_pub=SimpleNamespace(publish=lambda _: None),
        p_d=np.array([2.0, 3.0, 0.3]),
        cue=np.array([2.0, 3.0, 0.0]),
        _bias=np.zeros(3),
        _precland_commit_height=0.65,
        _precland_commit_until=9.0,
        _landing_contact_confirmed=False,
        _native_precland_accepted=True,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND),
        landed=False,
        armed=True,
        _publish_landing_target=lambda: False,
        _recover_precland=forbidden,
        _ocm=forbidden,
        _now=lambda: 10.0,
        get_logger=lambda: SimpleNamespace(
            warn=lambda message, **_kwargs: warnings.append(message)),
    )

    MissionManagerNode._tick(state)

    assert state.state == 'PRECLAND'
    assert warnings and 'no Offboard climb' in warnings[-1]


def test_precland_handoff_sends_one_native_px4_command(monkeypatch):
    commands = []
    targets = []
    goto = []
    segment_safe = [True]
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free',
        lambda *_: segment_safe[0])
    state = SimpleNamespace(
        state='RETURN',
        p_d=np.array([0.0, 0.0, 5.0]),
        _last_safe_goto=np.array([0.5, 0.0, 5.0]),
        _gps_preacquire_active=False,
        mission_map_yaml=str(MAP),
        _cmd=lambda *args: commands.append(args),
        _publish_landing_target=lambda: targets.append(True) or True,
        _publish_planned_path=lambda path: targets.append(path),
    )

    def set_state(new, why=''):
        state.state = new
        state._precland_since = 10.0
        state._last_precland_cmd = None

    state._set_state = set_state
    assert MissionManagerNode._enter_precland(state, 4.0)

    assert state.state == 'PRECLAND'
    assert targets == [True, None] and commands == []

    handoff = {'ocm': 0, 'hold': []}
    forbidden = lambda *_: (_ for _ in ()).throw(
        AssertionError('application flight control used after PRECLAND handoff'))
    state.k = 0
    state.state_pub = SimpleNamespace(publish=lambda _: None)
    state.p_d = np.array([0.0, 0.0, 5.0])
    state._status = SimpleNamespace(
        nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD)
    state._ocm = lambda: handoff.__setitem__('ocm', handoff['ocm'] + 1)
    state._send = lambda target: handoff['hold'].append(
        np.asarray(target).copy())
    state._send_goto = lambda target: goto.append(np.asarray(target).copy())
    state._now = lambda: 10.2
    state.landed = False
    state.armed = True
    MissionManagerNode._tick(state)
    assert commands == [(VehicleCommand.VEHICLE_CMD_NAV_PRECLAND,)]
    assert handoff == {'ocm': 1, 'hold': []}
    assert len(goto) == 1
    assert np.allclose(goto[0], state._last_safe_goto)

    segment_safe[0] = False
    MissionManagerNode._tick(state)
    assert handoff['ocm'] == 2 and len(handoff['hold']) == 1
    assert np.allclose(handoff['hold'][-1], state.p_d)
    assert len(goto) == 1 and state._precland_goto is None

    segment_safe[0] = True
    state._precland_goto = state._last_safe_goto.copy()
    recovered = []
    state._recover_precland = lambda: recovered.append(True)
    state._publish_landing_target = lambda: False
    MissionManagerNode._tick(state)
    assert recovered == [True]

    state._status.nav_state = VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND
    state._publish_landing_target = lambda: True
    state._ocm = state._send = state._send_goto = state._cmd = forbidden
    MissionManagerNode._tick(state)
    assert commands == [(VehicleCommand.VEHICLE_CMD_NAV_PRECLAND,)]


def test_precland_recovery_streams_then_requests_offboard():
    sent = []
    commands = []
    heartbeat = []
    state = SimpleNamespace(
        k=0,
        state='LANDING_ACQUIRE',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _native_takeoff_accepted=False,
        _native_precland_accepted=False,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND),
        landed=False,
        armed=True,
        p_d=np.array([0.0, 0.0, 0.3]),
        v_d=np.array([0.2, 0.8, 0.0]),
        cue=np.array([1.0, 2.0, 0.0]),
        cue_v=np.array([0.0, 1.0, 0.0]),
        _bias=np.zeros(3),
        _landing_hold_z=1.5,
        _landing_recovery_since=10.0,
        _last_offboard_cmd=None,
        _ocm=lambda: heartbeat.append(True),
        _cue_fresh=lambda: True,
        _send=lambda p, v=None, a=None: sent.append((p, v, a)),
        _cmd=lambda *args: commands.append(args),
        _now=lambda: 11.0,
    )

    MissionManagerNode._tick(state)

    assert heartbeat == [True]
    assert np.allclose(sent[0][0], state.p_d)
    assert np.allclose(sent[0][1], state.v_d)
    assert commands == [(
        VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)]

    ran_mpc = []
    state._status.nav_state = VehicleStatus.NAVIGATION_STATE_OFFBOARD
    state.p_d[2] = 0.2
    state._run_landing_mpc = lambda: ran_mpc.append(True)
    MissionManagerNode._tick(state)
    assert ran_mpc == [True]
    assert np.isclose(state._landing_hold_z, 0.2)
    assert state._landing_recovery_since is None

    state._touchdown_metric_candidate = object()
    sent.clear()
    commands.clear()
    heartbeat.clear()
    MissionManagerNode._tick(state)
    assert sent == [] and commands == [] and heartbeat == []

    state.landed = True
    state.armed = False
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    MissionManagerNode._tick(state)
    assert state.state == 'DONE'


def test_mission_follows_bspline_by_space_and_sends_goto_only():
    sent_goto = []
    state = SimpleNamespace(
        k=0,
        state='MISSION',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([1.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 2.0, 4.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [4.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _send=lambda *_: (_ for _ in ()).throw(
            AssertionError('TrajectorySetpoint used in MISSION')),
        _send_goto=lambda target: sent_goto.append(np.asarray(target).copy()),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._follow_path = lambda: MissionManagerNode._follow_path(state)

    MissionManagerNode._tick(state)
    assert np.allclose(sent_goto[-1], [3.0, 0.0, 5.0])
    assert np.isclose(state._mission_progress_m, 1.0)

    sent_goto.clear()
    state.p_d = np.array([1.0, 1.0, 5.0])
    MissionManagerNode._tick(state)
    assert np.isclose(state._mission_progress_m, 1.0)
    assert np.allclose(sent_goto[-1], [1.0, 0.0, 5.0])


def test_static_path_hands_terminal_braking_to_one_fixed_goto(monkeypatch):
    class Controller:
        j_max = 2.0

        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    controller = Controller()
    reference = mission_module.HorizonReference(lead_s=0.1)
    gotos = []
    state = SimpleNamespace(
        state='MISSION',
        p_d=np.array([30.0, 0.0, 5.0]),
        v_d=np.array([2.5, 0.0, 0.0]),
        _mission_arc_m=np.array([0.0, 20.0, 40.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [20.0, 0.0, 5.0], [40.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=15.0,
        _mission_cross_track=1.0,
        mission_tolerance=0.7,
        mission_map_yaml=str(MAP),
        _path_mpc=controller,
        _path_mpc_a_max=2.0,
        _path_mpc_speed=6.0,
        _path_speed_profile_a_max=1.0,
        _path_reference=reference,
        _path_solve_t=1.0,
        _path_failure_hold=np.array([1.0, 1.0, 5.0]),
        _path_terminal_goto=None,
        _path_terminal_goto_enabled=True,
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        get_logger=lambda: SimpleNamespace(info=lambda *_args: None),
    )
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)

    assert MissionManagerNode._follow_path(state)
    assert np.allclose(state._path_terminal_goto, [40.0, 0.0, 5.0])
    assert np.allclose(gotos[-1], state._path_terminal_goto)
    assert controller.resets == 1
    assert state._path_solve_t is None
    assert state._path_failure_hold is None

    state.p_d = np.array([32.0, 0.0, 5.0])
    assert MissionManagerNode._follow_path(state)
    assert np.allclose(gotos[-1], [40.0, 0.0, 5.0])
    assert controller.resets == 1


def test_moving_return_never_latches_a_static_terminal_goto(monkeypatch):
    gotos = []
    state = SimpleNamespace(
        state='RETURN',
        _return_staging=False,
        p_d=np.array([15.0, 0.0, 5.0]),
        v_d=np.array([6.0, 0.0, 0.0]),
        _mission_arc_m=np.array([0.0, 20.0, 40.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [20.0, 0.0, 5.0], [40.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=15.0,
        _mission_cross_track=1.0,
        mission_map_yaml=str(MAP),
        _path_mpc=None,
        _path_reference=None,
        _path_terminal_goto=None,
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)

    assert MissionManagerNode._follow_path(state)
    assert state._path_terminal_goto is None
    assert gotos and not np.allclose(gotos[-1], state._mission_path[-1])


def test_return_tracks_the_bspline_with_relative_braking_when_available(
        monkeypatch):
    sent = []
    reference_calls = []
    state = SimpleNamespace(
        k=0,
        state='RETURN',
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        cue=np.array([8.0, 0.0, 5.0]),
        cue_v=np.array([1.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _path_relative_brake_distance=10.0,
        _path_target_relative_speed=0.3,
        _mission_arc_m=np.array([0.0, 5.0, 10.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [10.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _path_mpc=mission_module.TrackingMPC(
            dt_s=0.1, horizon=10, v_max=5.0, a_max=3.0),
        _path_reference=mission_module.HorizonReference(lead_s=0.1),
        _path_solve_t=None,
        _mpc_solve_every=5,
        mpc_dt=0.1,
        _path_mpc_speed=3.0,
        _path_mpc_a_max=3.0,
        _now=lambda: 0.0,
        _send=lambda pos, vel, acc: sent.append((
            np.asarray(pos).copy(), np.asarray(vel).copy(),
            np.asarray(acc).copy())),
        _send_goto=lambda *_: (_ for _ in ()).throw(
            AssertionError('TrackingMPC unexpectedly fell back to Goto')),
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    planning_checks = []
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda *_: planning_checks.append(True) or True)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free',
        lambda *_: (_ for _ in ()).throw(
            AssertionError(
                'TrackingMPC used hard instead of planning clearance')))
    original_reference = mission_module._path_reference_horizon

    def capture_reference(*args, **kwargs):
        reference_calls.append(kwargs.copy())
        return original_reference(*args, **kwargs)

    monkeypatch.setattr(
        mission_module, '_path_reference_horizon', capture_reference)

    assert MissionManagerNode._follow_path(state)
    assert len(sent) == 1
    assert np.allclose(reference_calls[0]['target_velocity_xy'], [1.0, 0.0])
    assert np.isclose(reference_calls[0]['target_range_xy_m'], 8.0)
    assert np.isclose(reference_calls[0]['relative_brake_start_m'], 10.0)
    assert np.isclose(reference_calls[0]['target_relative_speed_m_s'], 0.3)
    assert planning_checks
    assert sent[0][0][0] > 0.0
    assert np.all(np.isfinite(np.concatenate(sent[0])))


def test_tracking_mpc_uses_continuous_hard_horizon_as_dynamic_reserve(
        monkeypatch):
    sent = []
    modes = []
    state = SimpleNamespace(
        state='MISSION',
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 5.0, 10.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [10.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _path_mpc=mission_module.TrackingMPC(
            dt_s=0.1, horizon=10, v_max=5.0, a_max=3.0),
        _path_reference=mission_module.HorizonReference(lead_s=0.1),
        _path_solve_t=None,
        mpc_dt=0.1,
        _path_mpc_speed=3.0,
        _path_mpc_a_max=3.0,
        _now=lambda: 0.0,
        _send=lambda pos, vel, acc: sent.append((pos, vel, acc)),
        _send_goto=lambda *_: pytest.fail(
            'hard-safe MPC recovery must not switch to Goto'),
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mission_module, '_safe_spatial_path_target',
        lambda *_args, **_kwargs: (0.0, np.array([2.0, 0.0, 5.0])))

    def prediction(_map, _current, _positions, planning=False):
        modes.append(planning)
        return not planning

    monkeypatch.setattr(mission_module, '_mpc_prediction_is_safe', prediction)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda *_: pytest.fail('stream must switch to hard clearance'))

    assert MissionManagerNode._follow_path(state)
    assert modes == [True, False]
    assert len(sent) == 1
    assert np.all(np.isfinite(np.concatenate(sent[0])))


def test_tracking_mpc_retries_once_with_a_lower_speed_reference(monkeypatch):
    class SpeedLimitedRetryMpc:
        N = 2
        j_max = 2.0
        v_max = 10.0

        def __init__(self):
            self.calls = 0

        def solve(self, *_args, **_kwargs):
            self.calls += 1
            speed = 10.1 if self.calls == 1 else 8.0
            return SimpleNamespace(
                success=True,
                predicted_pos=np.array([
                    [1.0, 0.0, 5.0], [2.0, 0.0, 5.0]]),
                predicted_vel=np.array([
                    [speed, 0.0, 0.0], [speed, 0.0, 0.0]]),
                predicted_acc=np.zeros((2, 3)))

        def reset(self):
            pass

    references = []
    sent = []
    controller = SpeedLimitedRetryMpc()
    state = SimpleNamespace(
        state='MISSION',
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 2.0]),
        _mission_path=np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=1.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _path_mpc=controller,
        _path_reference=mission_module.HorizonReference(lead_s=0.1),
        _path_solve_t=None,
        mpc_dt=0.1,
        _path_mpc_speed=8.5,
        _path_mpc_a_max=2.0,
        _now=lambda: 0.0,
        _send=lambda pos, vel, acc: sent.append((pos, vel, acc)),
        _send_goto=lambda *_: pytest.fail('safe retry switched to Goto'),
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mission_module, '_safe_spatial_path_target',
        lambda *_args, **_kwargs: (0.0, np.array([1.0, 0.0, 5.0])))
    monkeypatch.setattr(
        MissionManagerNode, '_tracking_path_reference',
        lambda _self, *_args, speed_limit_m_s=None, **_kwargs: (
            references.append(speed_limit_m_s)
            or (np.zeros((2, 3)), np.zeros((2, 3)))))
    monkeypatch.setattr(
        mission_module, '_mpc_prediction_is_safe', lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)

    assert MissionManagerNode._follow_path(state)
    assert controller.calls == 2
    assert references[0] is None
    assert 0.5 <= references[1] < state._path_mpc_speed
    assert len(sent) == 1


def test_tracking_mpc_retries_slower_when_fast_horizon_cuts_an_obstacle(
        monkeypatch):
    class ClearanceRetryMpc:
        N = 2
        j_max = 2.0
        v_max = 12.0

        def __init__(self):
            self.calls = 0

        def solve(self, *_args, **_kwargs):
            self.calls += 1
            end_x = 3.0 if self.calls == 1 else 2.0
            return SimpleNamespace(
                success=True,
                predicted_pos=np.array([
                    [1.0, 0.0, 5.0], [end_x, 0.0, 5.0]]),
                predicted_vel=np.array([
                    [8.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
                predicted_acc=np.zeros((2, 3)))

        def reset(self):
            pass

    references = []
    sent = []
    controller = ClearanceRetryMpc()
    state = SimpleNamespace(
        state='MISSION',
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 3.0]),
        _mission_path=np.array([[0.0, 0.0, 5.0], [3.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=1.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _path_mpc=controller,
        _path_reference=mission_module.HorizonReference(lead_s=0.1),
        _path_solve_t=None,
        mpc_dt=0.1,
        _path_mpc_speed=12.0,
        _path_mpc_a_max=2.0,
        _now=lambda: 0.0,
        _send=lambda pos, vel, acc: sent.append((pos, vel, acc)),
        _send_goto=lambda *_: pytest.fail('safe retry switched to Goto'),
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mission_module, '_safe_spatial_path_target',
        lambda *_args, **_kwargs: (0.0, np.array([1.0, 0.0, 5.0])))
    monkeypatch.setattr(
        MissionManagerNode, '_tracking_path_reference',
        lambda _self, *_args, speed_limit_m_s=None, **_kwargs: (
            references.append(speed_limit_m_s)
            or (np.zeros((2, 3)), np.zeros((2, 3)))))
    monkeypatch.setattr(
        mission_module, '_mpc_prediction_is_safe',
        lambda _map, _current, predicted, **_kwargs:
            float(np.asarray(predicted)[-1, 0]) <= 2.0)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)

    assert MissionManagerNode._follow_path(state)
    assert controller.calls == 2
    assert references == [None, pytest.approx(6.0)]
    assert len(sent) == 1


def test_shadow_tracking_mpc_retries_slower_before_rejecting_a_safe_swap(
        monkeypatch):
    calls = []

    class ClearanceRetryMpc:
        N = 2
        j_max = 2.0
        v_max = 12.0

        def solve(self, *_args, **_kwargs):
            calls.append(True)
            end_x = 3.0 if len(calls) == 1 else 2.0
            return SimpleNamespace(
                success=True,
                predicted_pos=np.array([
                    [1.0, 0.0, 5.0], [end_x, 0.0, 5.0]]),
                predicted_vel=np.array([
                    [8.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
                predicted_acc=np.zeros((2, 3)))

    references = []
    state = SimpleNamespace(
        state='RETURN',
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        _path_mpc=ClearanceRetryMpc(),
        _path_reference=SimpleNamespace(lead=0.1),
        _last_sent_acceleration=np.zeros(3),
        mpc_dt=0.1,
        _path_mpc_speed=12.0,
        mission_map_yaml=str(MAP),
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    arc = np.array([0.0, 3.0])
    path = np.array([[0.0, 0.0, 5.0], [3.0, 0.0, 5.0]])
    monkeypatch.setattr(
        MissionManagerNode, '_tracking_path_reference',
        lambda _self, *_args, speed_limit_m_s=None, **_kwargs: (
            references.append(speed_limit_m_s)
            or (np.zeros((2, 3)), np.zeros((2, 3)))))
    monkeypatch.setattr(
        mission_module, '_mpc_prediction_is_safe',
        lambda _map, _current, predicted, planning=False:
            planning and float(np.asarray(predicted)[-1, 0]) <= 2.0)

    prepared = MissionManagerNode._shadow_tracking_plan(
        state, arc, path, 0.0)

    assert prepared is not None
    assert len(calls) == 2
    assert references == [None, pytest.approx(6.0)]


def test_clearance_rejection_streams_the_still_live_certified_pva(
        monkeypatch):
    class UnsafeMpc:
        N = 4
        j_max = 2.0
        v_max = 12.0

        def __init__(self):
            self.calls = 0
            self.reset_count = 0

        def solve(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                success=True,
                predicted_pos=np.tile([20.0, 0.0, 5.0], (self.N, 1)),
                predicted_vel=np.zeros((self.N, 3)),
                predicted_acc=np.zeros((self.N, 3)))

        def reset(self):
            self.reset_count += 1

    sent = []
    now = [0.1]
    controller = UnsafeMpc()
    reference = mission_module.HorizonReference(lead_s=0.1)
    prior_positions = np.array([
        [0.1, 0.0, 5.0], [0.2, 0.0, 5.0],
        [0.3, 0.0, 5.0], [0.4, 0.0, 5.0]])
    reference.set_plan(
        np.array([0.0, 0.0, 5.0]), np.zeros(3),
        prior_positions, np.zeros((4, 3)), np.zeros((4, 3)), 0.1,
        np.zeros(3), np.zeros(3), np.zeros(3))
    state = SimpleNamespace(
        state='RETURN',
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 20.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [20.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _path_mpc=controller,
        _path_reference=reference,
        _path_solve_t=0.0,
        _path_failure_hold=None,
        _last_sent_acceleration=np.zeros(3),
        _last_sent_acceleration_t=0.0,
        mpc_dt=0.1,
        dt=0.02,
        _path_mpc_speed=12.0,
        _path_mpc_a_max=3.0,
        _now=lambda: now[0],
        _send=lambda pos, vel, acc: sent.append((pos, vel, acc)),
        _send_goto=lambda *_: pytest.fail('live prior PVA switched to Goto'),
        get_logger=lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mission_module, '_safe_spatial_path_target',
        lambda *_args, **_kwargs: (0.0, np.array([2.0, 0.0, 5.0])))
    monkeypatch.setattr(
        MissionManagerNode, '_tracking_path_reference',
        lambda *_args, **_kwargs: (
            np.zeros((4, 3)), np.zeros((4, 3))))
    monkeypatch.setattr(
        mission_module, '_mpc_prediction_is_safe', lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)

    assert MissionManagerNode._follow_path(state)

    assert controller.calls == 2
    assert controller.reset_count == 0
    assert reference.ready()
    assert state._path_failure_hold is None
    assert len(sent) == 1
    assert np.all(np.isfinite(np.concatenate(sent[0])))

    # A rejected replacement must not turn the 50 Hz stream into a 50 Hz
    # double-solve loop; retry again only at the configured 10 Hz MPC cadence.
    now[0] = 0.12
    assert MissionManagerNode._follow_path(state)
    assert controller.calls == 2
    assert len(sent) == 2

    now[0] = 0.2
    assert MissionManagerNode._follow_path(state)
    assert controller.calls == 4
    assert len(sent) == 3


def test_rejected_tracking_mpc_uses_the_planning_safe_path_carrot(
        monkeypatch):
    class RejectingMPC:
        N = 2
        j_max = 2.0

        def __init__(self):
            self.reset_count = 0

        def solve(self, *_args, **_kwargs):
            zeros = np.zeros((self.N, 3))
            return SimpleNamespace(
                success=False,
                predicted_pos=zeros.copy(),
                predicted_vel=zeros.copy(),
                predicted_acc=zeros.copy())

        def reset(self):
            self.reset_count += 1

    sent = []
    gotos = []
    position = np.array([1.0, 0.0, 5.0])
    reference = mission_module.HorizonReference(lead_s=0.1)
    zeros = np.zeros((2, 3))
    reference.set_plan(
        position, np.zeros(3), zeros, zeros, zeros, 0.1,
        np.zeros(3), np.zeros(3), np.zeros(3))
    mpc = RejectingMPC()
    state = SimpleNamespace(
        state='MISSION',
        p_d=position.copy(),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 2.0, 4.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [4.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _path_mpc=mpc,
        _path_reference=reference,
        _path_solve_t=0.0,
        _path_failure_hold=None,
        mpc_dt=0.1,
        _path_mpc_speed=3.0,
        _path_mpc_a_max=3.0,
        _path_speed_profile_a_max=3.0,
        _now=lambda: 1.0,
        _send=lambda *args: sent.append(args),
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        get_logger=lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)

    assert reference.ready()
    assert MissionManagerNode._follow_path(state)

    assert not reference.ready()
    assert state._path_solve_t is None
    assert mpc.reset_count == 1
    assert sent == []
    assert len(gotos) == 1
    assert state._path_failure_hold is None


def test_return_plan_failure_holds_the_first_position_across_ticks():
    class PendingPlan:
        @staticmethod
        def done():
            return False

    gotos = []
    first_position = np.array([1.0, 2.0, 5.0])
    state = SimpleNamespace(
        k=0,
        state='RETURN_PLAN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=first_position.copy(),
        cue=np.array([20.0, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _landing_mpc_entry_ready=lambda: False,
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [10.0, 0.0, 5.0]]),
        _plan_future=PendingPlan(),
        _path_failure_hold=None,
        _follow_path=lambda: False,
        _send=lambda *_: (_ for _ in ()).throw(
            AssertionError('RETURN_PLAN failure streamed stale PVA')),
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        _now=lambda: 0.0,
    )

    MissionManagerNode._tick(state)
    state.p_d = np.array([4.0, 5.0, 4.5])
    MissionManagerNode._tick(state)

    assert len(gotos) == 2
    assert np.allclose(gotos, [first_position, first_position])
    assert np.allclose(state._path_failure_hold, first_position)
    assert np.allclose(state._hold_pos, first_position)


def test_spatial_carrot_is_continuous_at_cross_track_threshold():
    arc = np.array([0.0, 20.0])
    path = np.array([[0.0, 0.0, 5.0], [20.0, 0.0, 5.0]])

    on_path = _spatial_path_target(
        arc, path, np.array([10.0, 0.0, 5.0]), 10.0, 6.0, 0.25)
    halfway = _spatial_path_target(
        arc, path, np.array([10.0, 0.125, 5.0]), 10.0, 6.0, 0.25)
    inside = _spatial_path_target(
        arc, path, np.array([10.0, 0.249, 5.0]), 10.0, 6.0, 0.25)
    boundary = _spatial_path_target(
        arc, path, np.array([10.0, 0.250, 5.0]), 10.0, 6.0, 0.25)
    outside = _spatial_path_target(
        arc, path, np.array([10.0, 0.251, 5.0]), 10.0, 6.0, 0.25)

    assert np.allclose(on_path[1], [16.0, 0.0, 5.0])
    assert np.allclose(halfway[1], [13.0, 0.0, 5.0])
    assert np.allclose(inside[1], [10.024, 0.0, 5.0])
    assert np.allclose(boundary[1], [10.0, 0.0, 5.0])
    assert np.allclose(outside[1], boundary[1])


def test_completed_rolling_path_splices_from_the_current_position():
    arc = np.array([0.0, 5.0, 10.0])
    path = np.array([
        [0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
    current = np.array([2.0, 1.0, 5.0])

    replacement = _splice_path_from_current(
        lambda *_: True, arc, path, current, 6.0)

    assert replacement is not None
    joined_arc, joined = replacement
    assert np.allclose(joined[0], current)
    assert np.allclose(joined[1], [10.0, 0.0, 5.0])
    assert np.allclose(joined[-1], path[-1])
    assert np.all(np.diff(joined_arc) > 0.0)

    # Normal altitude tracking error does not invalidate a fixed-altitude
    # rolling plan; the splice keeps measured XY and uses the route plane Z.
    _, altitude_projected = _splice_path_from_current(
        lambda *_: True, arc, path, np.array([2.0, 1.0, 4.87]), 6.0)
    assert np.allclose(altitude_projected[0], [2.0, 1.0, 5.0])
    assert np.allclose(altitude_projected[:, 2], 5.0)

    # The projection search covers the complete replacement, not only its
    # first lookahead window.
    _, late_joined = _splice_path_from_current(
        lambda *_: True, arc, path, np.array([8.0, 1.0, 5.0]), 2.0)
    assert np.allclose(late_joined, [[8.0, 1.0, 5.0],
                                    [10.0, 0.0, 5.0]])

    # If the long chord is blocked, retain the old half-lookahead connector.
    _, fallback = _splice_path_from_current(
        lambda _start, target: target[0] <= 5.0,
        arc, path, current, 6.0)
    assert np.allclose(fallback[1], [5.0, 0.0, 5.0])


def test_completed_rolling_path_rejects_an_unsafe_splice():
    arc = np.array([0.0, 5.0, 10.0])
    path = np.array([
        [0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [10.0, 0.0, 5.0]])

    assert _splice_path_from_current(
        lambda *_: False, arc, path,
        np.array([2.0, 1.0, 5.0]), 6.0) is None


def test_blocked_current_position_aborts_without_impossible_replan(monkeypatch):
    sent_goto = []
    replans = []
    state = SimpleNamespace(
        k=0,
        state='MISSION',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([1.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 2.0, 4.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [4.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _send_goto=lambda target: sent_goto.append(np.asarray(target).copy()),
        _start_global_plan=lambda goal, *, return_route: replans.append(
            (goal, return_route)),
    )
    transitions = []
    state._set_state = lambda new, why='': (
        transitions.append((new, why)), setattr(state, 'state', new))
    state._follow_path = lambda: MissionManagerNode._follow_path(state)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: False)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: False)

    MissionManagerNode._tick(state)

    assert state._mission_progress_m == 0.0
    assert np.allclose(sent_goto[-1], state.p_d)
    assert replans == []
    assert transitions[-1] == (
        'ABORT', '(left hard clearance; hold)')


def test_planning_reserve_loss_stops_then_retries_the_certified_path(
        monkeypatch):
    gotos = []
    replans = []
    follows = []
    first_position = np.array([1.0, 0.0, 5.0])
    state = SimpleNamespace(
        k=0,
        state='MISSION',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=first_position.copy(),
        v_d=np.array([1.0, 0.0, 0.0]),
        _mission_arc_m=np.array([0.0, 2.0, 4.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [4.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _path_failure_hold=None,
        _send_goto=lambda target: gotos.append(np.asarray(target).copy()),
        _start_global_plan=lambda *args, **kwargs: replans.append(
            (args, kwargs)),
    )

    def follow():
        follows.append(state.p_d.copy())
        if len(follows) == 1:
            state._path_failure_hold = state.p_d.copy()
            return False
        state._path_failure_hold = None
        return True

    state._follow_path = follow
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)

    MissionManagerNode._tick(state)
    state.p_d = np.array([1.4, 0.0, 5.0])
    MissionManagerNode._tick(state)
    state.v_d[:] = 0.0
    MissionManagerNode._tick(state)

    assert len(follows) == 2
    assert replans == []
    assert state.state == 'MISSION'
    assert len(gotos) == 3
    assert np.allclose(gotos, [first_position] * 3)


def test_planning_reserve_loss_rejoins_forward_hard_safe_path(monkeypatch):
    sent_goto = []
    state = SimpleNamespace(
        p_d=np.array([1.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 2.0, 4.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [4.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _path_mpc=None,
        _path_reference=None,
        _send_goto=lambda target: sent_goto.append(np.asarray(target).copy()),
        get_logger=lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: False)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)

    assert MissionManagerNode._follow_path(state)
    assert np.allclose(sent_goto, [[2.0, 0.0, 5.0]])
    assert state._mission_progress_m == 1.0


def test_planning_reserve_recovery_uses_hard_mpc_pva_not_goto(monkeypatch):
    sent = []
    planning_checks = []
    hard_checks = []
    prediction_modes = []
    state = SimpleNamespace(
        state='MISSION',
        p_d=np.array([1.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 2.0, 4.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [4.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_map_yaml=str(MAP),
        _path_mpc=mission_module.TrackingMPC(
            dt_s=0.1, horizon=10, v_max=5.0, a_max=3.0),
        _path_reference=mission_module.HorizonReference(lead_s=0.1),
        _path_solve_t=None,
        mpc_dt=0.1,
        _path_mpc_speed=3.0,
        _path_mpc_a_max=3.0,
        _now=lambda: 0.0,
        _send=lambda pos, vel, acc: sent.append((
            np.asarray(pos).copy(), np.asarray(vel).copy(),
            np.asarray(acc).copy())),
        _send_goto=lambda *_: (_ for _ in ()).throw(
            AssertionError('hard-safe TrackingMPC recovery used Goto')),
        get_logger=lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda *_: planning_checks.append(True) or False)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free',
        lambda *_: hard_checks.append(True) or True)
    original_prediction_check = mission_module._mpc_prediction_is_safe

    def capture_prediction_mode(*args, **kwargs):
        prediction_modes.append(kwargs.get('planning', False))
        return original_prediction_check(*args, **kwargs)

    monkeypatch.setattr(
        mission_module, '_mpc_prediction_is_safe', capture_prediction_mode)

    assert MissionManagerNode._follow_path(state)
    assert prediction_modes == [False]
    assert planning_checks and hard_checks
    assert len(sent) == 1
    assert np.all(np.isfinite(np.concatenate(sent[0])))


def test_spatial_carrot_shortens_to_the_farthest_safe_target(monkeypatch):
    sent_goto = []
    replans = []
    state = SimpleNamespace(
        k=0,
        state='MISSION',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        # Deliberately outside the 0.25 m tracking gate: a shorter exact-safe
        # carrot must be tried before an expensive full replan.
        p_d=np.array([1.0, 0.3, 5.0]),
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 5.0, 10.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [10.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=6.0,
        _mission_cross_track=0.25,
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _send_goto=lambda target: sent_goto.append(np.asarray(target).copy()),
        _start_global_plan=lambda goal, *, return_route: replans.append(
            (goal, return_route)),
    )
    state._follow_path = lambda: MissionManagerNode._follow_path(state)
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda _map, _start, goal: float(goal[0]) <= 2.5)

    MissionManagerNode._tick(state)

    assert np.isclose(state._mission_progress_m, 1.0)
    assert 1.0 <= sent_goto[-1][0] <= 2.5
    assert replans == []


def test_spatial_progress_does_not_jump_to_a_distant_branch():
    arc = np.arange(0.0, 22.0)
    path = np.column_stack((
        np.r_[np.arange(0.0, 11.0), np.arange(10.0, -1.0, -1.0)],
        np.r_[np.zeros(11), np.ones(11) * 0.1],
        np.ones(22) * 5.0))
    progress, _, cross_track = _spatial_path_target(
        arc, path, np.array([1.0, 0.1, 5.0]), 0.0, 2.0, 0.25)
    assert progress < 3.0
    assert cross_track <= 0.1 + 1.0e-9


def test_mission_manager_uses_separate_mpcs_but_never_forces_disarm():
    source = Path(mission_module.__file__).read_text(encoding='utf-8')

    assert 'TrackingMPC(' in source
    assert 'LandingMPC(' in source
    assert 'w_vxy=20.0' in source
    assert "p('precland_blind_commit_grace_s', 8.0).value" in source
    assert "'LANDING_ACQUIRE'" in source
    assert "'LANDING_DESCEND'" in source
    assert "'PRECLAND'" in source
    assert "'TOUCHDOWN'" not in source
    assert '21196' not in source
    assert not hasattr(MissionManagerNode, '_touchdown_geometry')


def test_geometry_goto_keeps_the_same_horizontal_speed_cap():
    published = []
    state = SimpleNamespace(
        get_clock=Clock,
        goto_pub=SimpleNamespace(publish=published.append),
        _path_mpc_v_max=9.0,
    )

    MissionManagerNode._send_goto(state, np.array([1.0, 2.0, 5.0]))

    message = published[0]
    assert np.allclose(message.position, [2.0, 1.0, -5.0])
    assert message.flag_set_max_horizontal_speed
    assert not message.flag_set_max_vertical_speed
    assert message.max_horizontal_speed == 9.0
    assert np.isnan(message.max_vertical_speed)
    assert np.allclose(state._last_sent_acceleration, 0.0)


def test_trajectory_send_remembers_the_acceleration_it_published():
    published = []
    state = SimpleNamespace(
        get_clock=Clock,
        sp_pub=SimpleNamespace(publish=published.append),
    )

    acceleration = np.array([0.2, -0.1, 0.05])
    MissionManagerNode._send(
        state, np.array([1.0, 2.0, 5.0]), np.zeros(3), acceleration)

    assert np.allclose(state._last_sent_acceleration, acceleration)
    assert np.allclose(published[0].acceleration, [-0.1, 0.2, -0.05])


def test_precland_only_publishes_target_and_observes_px4_landed_disarmed():
    targets = []
    forbidden = lambda *_: (_ for _ in ()).throw(
        AssertionError('application flight control used after PRECLAND handoff'))
    state = SimpleNamespace(
        k=0,
        state='PRECLAND',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=forbidden,
        _send=forbidden,
        _send_goto=forbidden,
        _cmd=forbidden,
        _publish_landing_target=lambda: targets.append(True) or True,
        _cue_fresh=lambda: True,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND),
        p_d=np.zeros(3),
        _now=lambda: 10.0,
        landed=False,
        armed=True,
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)

    MissionManagerNode._tick(state)
    assert state.state == 'PRECLAND' and len(targets) == 1

    land = VehicleLandDetected()
    land.ground_contact = True
    land.maybe_landed = True
    MissionManagerNode._on_land(state, land)
    state.armed = False
    MissionManagerNode._tick(state)
    assert state.state == 'PRECLAND'

    land.landed = True
    MissionManagerNode._on_land(state, land)
    state.armed = True
    MissionManagerNode._tick(state)
    assert state.state == 'PRECLAND'

    state.armed = False
    MissionManagerNode._tick(state)
    assert state.state == 'DONE'


def test_ground_contact_is_a_one_way_boundary_even_without_a_metric_snapshot():
    published = []
    forbidden = lambda *_: (_ for _ in ()).throw(
        AssertionError('flight or reacquisition command issued after contact'))
    state = SimpleNamespace(
        k=0,
        state='PRECLAND',
        state_pub=SimpleNamespace(publish=lambda _: None),
        p_d=np.zeros(3),
        cue=None,
        landed=False,
        armed=True,
        _touchdown_metric_recorded=False,
        _touchdown_metric_candidate=None,
        _ground_contact_seen=False,
        _publish_landing_target=lambda: published.append(True) or False,
        _recover_precland=forbidden,
        _ocm=forbidden,
        _send=forbidden,
        _send_goto=forbidden,
        _cmd=forbidden,
    )

    contact = VehicleLandDetected()
    contact.ground_contact = True
    MissionManagerNode._on_land(state, contact)
    assert state._ground_contact_seen
    assert state._touchdown_metric_candidate is None

    contact.ground_contact = False
    MissionManagerNode._on_land(state, contact)
    assert state._ground_contact_seen
    MissionManagerNode._tick(state)
    assert published == [True]
    assert state.state == 'PRECLAND'


def test_experiment_metrics_use_first_confirmed_ground_contact_segment():
    state = SimpleNamespace(
        state='RETURN',
        _marker_metric_frames=0,
        _marker_metric_hits=0,
        _touchdown_metric_recorded=False,
        _touchdown_metric_candidate=None,
        landed=False,
        cue=np.array([1.0, 2.0, 0.0]),
        cue_v=np.array([1.0, 0.0, 0.0]),
        _bias=np.array([0.1, -0.1, 0.0]),
        p_d=np.array([1.4, 2.3, 0.5]),
        v_d=np.array([1.1, -0.2, -0.3]),
        _mpc_solve_count=0,
        _mpc_solve_total_s=0.0,
        _mpc_solve_max_s=0.0,
    )
    MissionManagerNode._on_marker_detection(
        state, SimpleNamespace(data=True))
    assert state._marker_metric_frames == 0

    state.state = 'LANDING_ACQUIRE'
    MissionManagerNode._on_marker_detection(
        state, SimpleNamespace(data=False))
    MissionManagerNode._on_marker_detection(
        state, SimpleNamespace(data=True))
    assert (state._marker_metric_hits, state._marker_metric_frames) == (1, 2)

    # A contact that begins while PRECLAND recovery owns Offboard must also
    # inhibit a climb and use the same confirmed-contact metric contract.
    state.state = 'LANDING_ACQUIRE'
    contact = VehicleLandDetected()
    contact.ground_contact = True
    MissionManagerNode._on_land(state, contact)
    assert not state._touchdown_metric_recorded

    # A transient contact must be discarded rather than reported as touchdown.
    contact.ground_contact = False
    MissionManagerNode._on_land(state, contact)
    assert state._touchdown_metric_candidate is None

    contact.ground_contact = True
    MissionManagerNode._on_land(state, contact)
    state.p_d[:] = 99.0  # confirmation must retain the segment's first sample
    contact.maybe_landed = True
    MissionManagerNode._on_land(state, contact)
    assert state._touchdown_metric_recorded
    assert np.isclose(state._landing_xy_error_m, 0.5)
    assert np.isclose(state._landing_error_3d_m, 0.5 ** 0.5)
    assert np.isclose(
        state._touchdown_relative_speed_3d_m_s,
        (0.1 ** 2 + 0.2 ** 2 + 0.3 ** 2) ** 0.5)
    assert np.isclose(state._touchdown_relative_vertical_speed_m_s, 0.3)

    MissionManagerNode._on_marker_detection(
        state, SimpleNamespace(data=True))
    assert state._marker_metric_frames == 2

    MissionManagerNode._record_mpc_solve(state, 0.002)
    MissionManagerNode._record_mpc_solve(state, 0.003)
    assert state._mpc_solve_count == 2
    assert np.isclose(state._mpc_solve_total_s, 0.005)
    assert np.isclose(state._mpc_solve_max_s, 0.003)

    logs = []
    state.get_logger = lambda: SimpleNamespace(info=logs.append)
    MissionManagerNode._log_experiment_metrics(state, final=False)
    assert 'mpc_count=2' in logs[-1]
    assert 'mpc_total_ms=5' in logs[-1]
    assert not hasattr(state, '_experiment_metrics_logged')
    MissionManagerNode._log_experiment_metrics(state)
    MissionManagerNode._log_experiment_metrics(state)
    assert len(logs) == 2


def test_trailer_cue_publishes_each_gazebo_update_once():
    published = {'position': [], 'velocity': []}
    state = SimpleNamespace(
        _lock=threading.Lock(),
        _updated=True,
        _pos=np.array([1.0, 2.0]),
        _vel=np.array([3.0, 0.0]),
        _source_ns=1_000_000_000,
        _last_report_t=float('inf'),
        deck_z=1.811,
        get_clock=Clock,
        pos_pub=SimpleNamespace(
            publish=lambda message: published['position'].append(message)),
        vel_pub=SimpleNamespace(
            publish=lambda message: published['velocity'].append(message)),
    )

    TrailerCueNode._tick(state)
    TrailerCueNode._tick(state)

    assert len(published['position']) == len(published['velocity']) == 1
    position = published['position'][0]
    velocity = published['velocity'][0]
    assert position.header.frame_id == velocity.header.frame_id \
        == LOCAL_ENU_FRAME_ID
    assert position.header.stamp == velocity.header.stamp


def test_expired_marker_filter_recovers_from_new_fix():
    messages = []
    state = SimpleNamespace(
        x=np.array([100.0, 100.0, 3.0, 0.0]),
        P=np.eye(4),
        sigma_m=0.06,
        max_coast=3.0,
        _t_last=1.0,
        _t_meas=1.0,
        entry_fix_count=3,
        entry_window=0.5,
        _entry_fix_ns=[1_000_000_000],
        _now=lambda: 10.0,
        get_logger=lambda: SimpleNamespace(info=messages.append),
    )
    state._initialise = lambda z, t: MarkerKfNode._initialise(state, z, t)
    state._record_entry_fix = lambda stamp, reset=False: (
        MarkerKfNode._record_entry_fix(state, stamp, reset))
    fix = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=10, nanosec=0)),
        point=SimpleNamespace(x=4.0, y=-2.0))

    MarkerKfNode._on_meas(state, fix)

    assert np.allclose(state.x, [4.0, -2.0, 0.0, 0.0])
    assert state._t_last == state._t_meas == 10.0
    assert state._entry_fix_ns == [10_000_000_000]
    assert messages == ['expired marker track recovered from a fresh fix']


def test_marker_entry_needs_three_distinct_accepted_fixes_in_half_second():
    now = [10.5]
    state = SimpleNamespace(
        entry_fix_count=3,
        entry_window=0.5,
        rate=50.0,
        _entry_fix_ns=[],
    )

    def record(stamp):
        MarkerKfNode._record_entry_fix(state, stamp)

    def confirmed():
        return MarkerKfNode._entry_confirmed(state, now[0])

    record(10_000_000_000)
    record(10_000_000_000)  # duplicate capture is not a second fix
    record(10_200_000_000)
    assert not confirmed()
    record(10_500_000_000)
    assert confirmed()

    now[0] = 11.000001
    assert not confirmed()
