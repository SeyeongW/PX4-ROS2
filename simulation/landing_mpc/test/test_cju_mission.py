import importlib.util
import math
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from geometry_msgs.msg import PointStamped, Pose, PoseArray, Vector3Stamped
from rclpy.clock import Clock
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from px4_msgs.msg import (LandingTargetPose, VehicleCommand,
                          VehicleLocalPosition,
                          VehicleLandDetected, VehicleStatus)

from path_plan.astar import AStarPlanner3D
from path_plan.bspline_optimizer import BsplineOptimizer
from path_plan.world_model import WorldModel
from landing_mpc.frame import LOCAL_ENU_FRAME_ID
import landing_mpc.mission_manager_node as mission_module
from landing_mpc.mission_manager_node import (MissionManagerNode,
                                              _mission_segment_is_free,
                                              _plan_global_path,
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
    clearance = mission['obstacle_clearance_m']
    for obstacle in mission['obstacles']:
        centre = np.asarray(obstacle['center_m'][:2])
        half = 0.5 * np.asarray(obstacle['size_m'][:2]) + clearance
        hit |= np.all(np.abs(points - centre) <= half, axis=1)
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


def test_cju_yaml_astar_avoids_configured_barriers_outbound_and_return():
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    outbound_arc, outbound, expanded_out = _plan_global_path(MAP)
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
        [25, 24], [33, 37], [15, 7], [14, 14], [39, 30],
        [21, 11], [28, 27], [32, 17], [17, 40], [33, 24],
        [32, 6], [26, 17], [23, 5], [24, 25], [15, 34],
        [18, 22], [27, 38], [32, 30], [28, 36], [23, 34]]
    assert len({tuple(xy) for xy in obstacle_centres[:, :2]}) == 20
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
        + document['mission']['obstacle_clearance_m']
    assert np.all(obstacle_centres[:, :2] - inflated_half >= field_low)
    assert np.all(obstacle_centres[:, :2] + inflated_half <= field_high)
    assert np.all((obstacle_centres[:, :2] >= 0)
                  & (obstacle_centres[:, :2] <= 40))
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
    assert document['mission']['obstacle_clearance_m'] == 2
    assert document['mission']['bspline_clearance_margin_m'] == 0.5
    assert document['mission']['bspline_control_spacing_m'] == 2.0
    assert document['mission']['bspline_sample_spacing_m'] == 0.1
    assert document['mission']['mpc_path_lookahead_m'] == 6.0
    assert document['mission']['mpc_path_cross_track_m'] == 0.25
    assert document['mission']['precland_handoff_m'] == 6.0
    assert document['mission']['return_replan_distance_m'] == 3.0
    assert document['mission']['return_replan_min_period_s'] == 1.0
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
    assert px4['MPC_LAND_CRWL'] == 0.3
    assert px4['LNDMC_Z_VEL_MAX'] == 0.25
    assert px4['LNDMC_XY_VEL_MAX'] == 1.5
    assert px4['COM_DISARM_LAND'] == 2.0
    assert px4['PLD_BTOUT'] == 0.5
    assert px4['PLD_HACC_RAD'] == 0.5
    assert px4['PLD_VEL_THR'] == 0.3
    assert px4['PLD_FAPPR_ALT'] == 0.1
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
    assert document['mission']['command_sequence'] == ['takeoff', 'land']
    assert document['mission']['coordinate_frame'] == 'stadium_endpoint'
    assert document['terrain']['coordinate_frame'] == 'stadium_endpoint'
    launcher = MISSION_LAUNCHER.read_text(encoding='utf-8')
    map_launcher = MAP_LAUNCHER.read_text(encoding='utf-8')
    assert 'commands=(takeoff land)' in launcher
    assert "wait_for_states 'MISSION|HOVER' 120" in launcher
    assert 'wait_for_states HOVER 180' in launcher
    assert "send_until_state land 'RETURN_PLAN|RETURN|PRECLAND|DONE'" in launcher
    assert 'wait_for_states DONE 180' in launcher
    assert '반복 순찰' not in launcher
    assert 'approach_alt' not in launcher
    assert 'acquire_xy' not in launcher
    assert 'touchdown_height' not in launcher
    assert 'z_floor' not in launcher
    assert "printf 'flight_control_owner\\t%s\\n' 'px4_native'" in launcher
    assert 'PX4 NAV_PRECLAND' in launcher
    assert '--live --map "$LANDING_COORDINATES"' in launcher
    assert '${MISSION_VIEW:-1}' in launcher
    assert "retry_command 'geometry B-spline/PX4 Goto 상태 확인'" in launcher
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
    assert trailer['cruise_speed_m_s'] == 1.0
    assert 'TRAILER_SPEED_FOR_RUN=3.0' not in launcher
    assert 'LANDING_TRAILER_SPEED="${LANDING_CONFIG[4]}"' in launcher

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


def test_land_uses_astar_only_when_blocked_then_hands_direct_route_to_px4(
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
        _precland_handoff=15.0,
        _t_solve=1.0,
        state='PRECHECK',
        get_logger=lambda: logger,
        _cue_fresh=lambda: True,
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._enter_precland = lambda *args: state._set_state('PRECLAND')
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

    state.state = 'HOVER'
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'RETURN_PLAN'
    assert len(plans) == 1 and plans[-1][1] is True
    assert np.allclose(plans[-1][0], state.cue)
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert len(plans) == 1

    plans.clear()
    state.state = 'HOVER'
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert state.state == 'PRECLAND'
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


def test_failed_return_plan_holds_in_abort_for_automatic_retry():
    class FailedPlan:
        @staticmethod
        def done():
            return True

        @staticmethod
        def result():
            raise RuntimeError('rejected geometry')

    now = [1.0]
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
        _now=lambda: now[0],
        takeoff_alt=5.0,
        _planner_pool=object(),
        _last_return_plan_t=1.0,
        _return_replan_min_period=1.0,
        get_logger=lambda: SimpleNamespace(error=lambda *_: None),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._start_global_plan = lambda goal, *, return_route: retries.append(
        (np.asarray(goal).copy(), return_route))

    MissionManagerNode._tick(state)

    assert state.state == 'ABORT'
    assert state._plan_future is None
    assert np.allclose(state._hold_pos, state.p_d)

    now[0] = 1.99
    MissionManagerNode._tick(state)
    assert retries == []

    now[0] = 2.0
    MissionManagerNode._tick(state)
    assert len(retries) == 1 and retries[0][1]


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
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)

    MissionManagerNode._tick(state)

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


def test_phase2_goal_completion_holds_instead_of_reverse_patrol():
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
        v_d=np.zeros(3),
        _mission_arc_m=np.array([0.0, 1.0]),
        _mission_path=np.array([[0.0, 0.0, 5.0], waypoint]),
        _mission_progress_m=1.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _send_goto=lambda *_: None,
        _start_global_plan=lambda *args, **kwargs: replans.append((args, kwargs)),
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)

    MissionManagerNode._tick(state)
    assert state.state == 'HOVER'
    assert np.allclose(state._hold_pos, waypoint)
    assert replans == []


def test_phase3_replans_only_while_live_trailer_route_is_blocked(monkeypatch):
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
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _precland_handoff=15.0,
        _return_plan_goal=np.array([14.0, 0.0, 0.0]),
        _return_replan_distance=3.0,
        _send_goto=lambda *_: None,
        _plan_future=None,
    )
    state._start_global_plan = lambda goal, *, return_route: replans.append(
        (np.asarray(goal).copy(), return_route))
    state._enter_precland = lambda *args: setattr(state, 'state', 'PRECLAND')

    def set_state(new, why=''):
        transitions.append((new, why))
        state.state = new

    state._set_state = set_state
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: False)
    MissionManagerNode._tick(state)
    assert len(replans) == 1 and replans[0][1]

    replans.clear()
    state.state = 'RETURN'
    state.cue = np.array([14.0, 0.0, 0.0])
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)
    MissionManagerNode._tick(state)
    assert state.state == 'PRECLAND'
    assert replans == []


def test_return_replans_from_latest_gps_after_three_metres(monkeypatch):
    replans = []
    sent = []
    state = SimpleNamespace(
        k=0,
        state='RETURN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        _status=SimpleNamespace(
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD),
        p_d=np.array([0.0, 0.0, 5.0]),
        v_d=np.zeros(3),
        cue=np.array([32.9, 0.0, 0.0]),
        _cue_fresh=lambda: True,
        _mission_arc_m=np.array([0.0, 10.0]),
        _mission_path=np.array([
            [0.0, 0.0, 5.0], [10.0, 0.0, 5.0]]),
        _mission_progress_m=0.0,
        _mission_lookahead=2.0,
        _mission_cross_track=0.25,
        mission_tolerance=0.7,
        settle_v_tol=0.3,
        takeoff_alt=5.0,
        mission_map_yaml=str(MAP),
        _precland_handoff=15.0,
        _return_plan_goal=np.array([30.0, 0.0, 0.0]),
        _return_replan_distance=3.0,
        _send_goto=lambda target: sent.append(np.asarray(target).copy()),
    )
    state._start_global_plan = lambda goal, *, return_route: replans.append(
        (np.asarray(goal).copy(), return_route))
    monkeypatch.setattr(
        mission_module, '_mission_segment_is_free', lambda *_: True)

    MissionManagerNode._tick(state)
    assert replans == [] and len(sent) == 1

    state.cue = np.array([33.0, 0.0, 0.0])
    MissionManagerNode._tick(state)
    assert len(replans) == 1 and replans[0][1]
    assert np.allclose(replans[0][0], state.cue)


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
        _now=lambda: now[0],
        _target=lambda: (
            np.array([10.0, 20.0, 1.5]),
            np.array([3.0, 4.0, 0.0])),
        landing_target_pub=SimpleNamespace(publish=published.append),
    )
    state._landing_target_fresh = lambda: (
        MissionManagerNode._landing_target_fresh(state))

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


def test_precland_handoff_sends_one_native_px4_command():
    commands = []
    targets = []
    state = SimpleNamespace(
        state='RETURN',
        p_d=np.zeros(3),
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

    handoff = {'ocm': 0, 'hold': 0}
    forbidden = lambda *_: (_ for _ in ()).throw(
        AssertionError('Goto used during PRECLAND handoff'))
    state.k = 0
    state.state_pub = SimpleNamespace(publish=lambda _: None)
    state.p_d = np.zeros(3)
    state._status = SimpleNamespace(
        nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD)
    state._ocm = lambda: handoff.__setitem__('ocm', handoff['ocm'] + 1)
    state._send = lambda *_: handoff.__setitem__('hold', handoff['hold'] + 1)
    state._send_goto = forbidden
    state._now = lambda: 10.2
    state.landed = False
    state.armed = True
    MissionManagerNode._tick(state)
    assert commands == [(VehicleCommand.VEHICLE_CMD_NAV_PRECLAND,)]
    assert handoff == {'ocm': 1, 'hold': 1}

    state._status.nav_state = VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND
    state._ocm = state._send = state._send_goto = state._cmd = forbidden
    MissionManagerNode._tick(state)
    assert commands == [(VehicleCommand.VEHICLE_CMD_NAV_PRECLAND,)]


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

    MissionManagerNode._tick(state)
    assert np.allclose(sent_goto[-1], [3.0, 0.0, 5.0])
    assert np.isclose(state._mission_progress_m, 1.0)

    sent_goto.clear()
    state.p_d = np.array([1.0, 1.0, 5.0])
    MissionManagerNode._tick(state)
    assert np.isclose(state._mission_progress_m, 1.0)
    assert np.allclose(sent_goto[-1], [3.0, 0.0, 5.0])


def test_spatial_carrot_is_continuous_at_cross_track_threshold():
    arc = np.array([0.0, 20.0])
    path = np.array([[0.0, 0.0, 5.0], [20.0, 0.0, 5.0]])

    inside = _spatial_path_target(
        arc, path, np.array([10.0, 0.250, 5.0]), 10.0, 6.0, 0.25)
    outside = _spatial_path_target(
        arc, path, np.array([10.0, 0.251, 5.0]), 10.0, 6.0, 0.25)

    assert np.allclose(inside[1], [16.0, 0.0, 5.0])
    assert np.allclose(outside[1], inside[1])


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


def test_mission_manager_has_no_custom_landing_or_forced_disarm_contract():
    source = Path(mission_module.__file__).read_text(encoding='utf-8')

    assert 'LandingMPC' not in source
    assert "'DESCEND'" not in source
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
        _publish_landing_target=lambda: targets.append(True),
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
        _now=lambda: 10.0,
        get_logger=lambda: SimpleNamespace(info=messages.append),
    )
    state._initialise = lambda z, t: MarkerKfNode._initialise(state, z, t)
    fix = SimpleNamespace(point=SimpleNamespace(x=4.0, y=-2.0))

    MarkerKfNode._on_meas(state, fix)

    assert np.allclose(state.x, [4.0, -2.0, 0.0, 0.0])
    assert state._t_last == state._t_meas == 10.0
    assert messages == ['expired marker track recovered from a fresh fix']
