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
    assert "send_until_state mission 'MISSION_PLAN|MISSION|HOVER'" in launcher
    assert "wait_for_states 'MISSION|HOVER' 120" in launcher
    assert 'wait_for_states HOVER 180' in launcher
    assert ("send_until_state land 'RETURN_PLAN|RETURN|LANDING_ACQUIRE|"
            "LANDING_DESCEND|PRECLAND|DONE'" in launcher)
    assert 'wait_for_states DONE 180' in launcher
    assert '반복 순찰' not in launcher
    assert 'approach_alt' not in launcher
    assert 'acquire_xy' not in launcher
    assert 'touchdown_height' not in launcher
    assert 'z_floor' not in launcher
    assert ("printf 'flight_control_owner\\t%s\\n' "
            "'mission_manager_mpc_then_px4_precland'" in launcher)
    assert 'PX4 NAV_PRECLAND' in launcher
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
    assert np.allclose(plans[-1][0], state.cue)
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert len(plans) == 1

    plans.clear()
    state.state = 'HOVER'
    landing_ready[0] = True
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'LANDING_ACQUIRE'
    assert plans == []


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
    state._plan_future = CompletedPlan()
    MissionManagerNode._tick(state)
    assert state.state == 'RETURN'
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
    state._last_return_plan_t = 0.0
    state._return_replan_min_period = 2.0
    state.cue = np.array([21.0, 0.0, 0.0])
    now[0] = 2.0
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free', lambda *_: True)
    MissionManagerNode._tick(state)
    assert events[0] == 'goto'
    assert events[1][0] == 'plan' and events[1][2]
    assert np.allclose(events[1][1], [21.0, 0.0, 0.0])
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
        mission_module, '_mission_segment_is_free', lambda *_: True)
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
    )

    MissionManagerNode._tick(state)
    assert follows == [True]
    assert sends == []

    state.p_d = np.array([1.1, 2.2, 5.0])
    state._follow_path = lambda: False
    MissionManagerNode._tick(state)
    assert np.allclose(state._hold_pos, state.p_d)
    assert len(sends) == 1 and np.allclose(sends[0][0], state.p_d)


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
            info=lambda *_: None, warn=lambda *_: None),
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
        mission_module, '_mission_planning_segment_is_free', lambda *_: False)
    MissionManagerNode._tick(state)
    assert len(replans) == 1 and replans[0][1]

    replans.clear()
    state.state = 'RETURN'
    state.cue = np.array([14.0, 0.0, 0.0])
    landing_ready[0] = True
    MissionManagerNode._tick(state)
    assert state.state == 'LANDING_ACQUIRE'
    assert replans == []


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
    now[0] = 14.0
    MissionManagerNode._tick(state)
    assert len(replans) == 2 and len(sent) == 4
    assert np.allclose(replans[-1][0], [33.0, 0.0, 0.0])


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
        _shuttle_endpoints_local=endpoints,
        _shuttle_endpoint_tolerance=0.2,
        _shuttle_min_terminal_speed=0.7,
        _precland_runway_required_s=10.5,
        _precland_commit_height=0.65,
        _landing_mpc_vz_max=0.6,
    )
    clear, eta = MissionManagerNode._terminal_runway_status(state)
    assert clear and eta == pytest.approx(15.0)
    clear, eta = MissionManagerNode._terminal_runway_status(state, 5.0)
    assert not clear and eta == pytest.approx(15.0)


def test_precland_allows_only_a_bounded_aligned_camera_blind_commit():
    now = [10.0]
    state = SimpleNamespace(
        cue=np.zeros(3),
        cue_v=np.array([1.0, 0.0, 0.0]),
        p_d=np.array([0.1, 0.0, 0.3]),
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
    now[0] = 11.0
    assert MissionManagerNode._precland_target_allowed(state)
    state.v_d[0] = 1.31
    assert not MissionManagerNode._precland_target_allowed(state)
    state.v_d[0] = 1.0
    now[0] = 18.01
    assert not MissionManagerNode._precland_target_allowed(state)


def test_precland_loss_reclaims_offboard_from_the_current_state():
    reset = []
    state = SimpleNamespace(
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


def test_landing_mpc_entry_uses_observation_and_safety_not_distance(
        monkeypatch):
    safety = [True]
    state = SimpleNamespace(
        p_d=np.array([0.0, 0.0, 5.0]),
        cue=np.array([3.58, 0.0, 0.0]),
        v_d=np.array([3.84, 0.0, 0.0]),
        cue_v=np.zeros(3),
        vis_entry_valid=True,
        mission_map_yaml=str(MAP),
        _landing_target_fresh=lambda: True,
        _vision_measurement_fresh=lambda: True,
    )
    monkeypatch.setattr(
        mission_module, '_mission_planning_segment_is_free',
        lambda *_: safety[0])

    assert MissionManagerNode._landing_mpc_entry_ready(state)
    state.cue[0] = 80.0
    assert MissionManagerNode._landing_mpc_entry_ready(state)
    safety[0] = False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    safety[0] = True
    state.vis_entry_valid = False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)
    state.vis_entry_valid = True
    state._vision_measurement_fresh = lambda: False
    assert not MissionManagerNode._landing_mpc_entry_ready(state)


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
    precland = []
    runway_clear = [False]
    runway_heights = []
    track_converged = [True]
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
        _vision_correction_converged=lambda: state.vis_entry_valid,
        _now=lambda: now[0],
        _send=lambda pos, vel=None, acc=None: sent.append(
            (np.asarray(pos).copy(), np.asarray(vel).copy(),
             np.asarray(acc).copy())),
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
        MissionManagerNode, '_vision_track_converged',
        lambda _self: track_converged[0])

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

    state._landing_reference.reset()
    now[0] = 10.04
    MissionManagerNode._run_landing_mpc(state)
    assert state._landing_mpc.a_max == 1.0
    assert state._landing_mpc.z_ref == 0.65
    assert state._landing_mpc.solve_count == 1
    assert precland == []

    state.vis_entry_valid = False
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

    track_converged[0] = False
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
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)
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
    assert sent[0][0][0] > 0.0
    assert np.all(np.isfinite(np.concatenate(sent[0])))


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
    assert np.allclose(joined[1], [5.0, 0.0, 5.0])
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
    assert np.allclose(late_joined, [[8.0, 1.0, 5.0], [9.0, 0.0, 5.0],
                                    [10.0, 0.0, 5.0]])


def test_completed_rolling_path_rejects_an_unsafe_splice():
    arc = np.array([0.0, 5.0, 10.0])
    path = np.array([
        [0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [10.0, 0.0, 5.0]])

    assert _splice_path_from_current(
        lambda *_: False, arc, path,
        np.array([2.0, 1.0, 5.0]), 6.0) is None


def test_blocked_spatial_carrot_replans_without_committing(monkeypatch):
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
    state._follow_path = lambda: MissionManagerNode._follow_path(state)
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: False)

    MissionManagerNode._tick(state)

    assert state._mission_progress_m == 0.0
    assert np.allclose(sent_goto[-1], state.p_d)
    assert replans == [(None, False)]


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
        mission_module, '_mission_segment_is_free',
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


def test_geometry_goto_contains_position_without_speed_constraints():
    published = []
    state = SimpleNamespace(
        get_clock=Clock,
        goto_pub=SimpleNamespace(publish=published.append),
    )

    MissionManagerNode._send_goto(state, np.array([1.0, 2.0, 5.0]))

    message = published[0]
    assert np.allclose(message.position, [2.0, 1.0, -5.0])
    assert not message.flag_set_max_horizontal_speed
    assert not message.flag_set_max_vertical_speed
    assert np.isnan(message.max_horizontal_speed)
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
