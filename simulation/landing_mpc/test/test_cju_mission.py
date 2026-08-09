import math
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from rclpy.clock import Clock

from landing_mpc.mission_manager_node import (MissionManagerNode,
                                              _plan_global_path)
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


def test_cju_dynamic_astar_patrol_avoids_all_four_barriers_both_ways():
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    outbound, expanded_out, speed = _plan_global_path(MAP)
    # A real controller reaches a waypoint with tolerance rather than landing
    # on its exact grid coordinate. This offset reproduces that dynamic start.
    reached_goal = outbound[-1] + np.array([1.1, -0.3, 0.0])
    inbound, expanded_back, _ = _plan_global_path(
        MAP, start_local_enu=reached_goal, goal_local_enu=outbound[0])
    outbound_stadium = _mission_xy(outbound, document)
    inbound_stadium = _mission_xy(inbound, document)
    assert np.allclose(outbound_stadium, np.round(outbound_stadium))
    assert np.allclose(inbound_stadium, np.round(inbound_stadium))

    direct = outbound_stadium[0] + np.linspace(0.0, 1.0, 1000)[:, None] * (
        outbound_stadium[-1] - outbound_stadium[0])
    assert _blocked(direct, document['mission']).any()
    outbound_samples = _segment_samples(outbound_stadium)
    inbound_samples = _segment_samples(inbound_stadium)
    assert not _blocked(outbound_samples, document['mission']).any()
    assert not _blocked(inbound_samples, document['mission']).any()
    assert expanded_out > 0 and expanded_back > 0
    assert len(outbound) >= 3 and len(inbound) >= 3
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
        [-47, 2], [-41, 12], [-47, 22], [-41, 32]]
    centre_x = document['mission']['goal_m'][0]
    for samples in (outbound_samples, inbound_samples):
        for obstacle in obstacle_centres:
            nearest = samples[np.argmin(np.abs(samples[:, 1] - obstacle[1]))]
            assert ((nearest[0] - obstacle[0])
                    * (obstacle[0] - centre_x) < 0.0)
    assert speed == 2.0
    assert np.allclose(outbound[:, 2], 5.0)
    assert np.allclose(inbound[:, 2], 5.0)


def test_world_and_yaml_share_stadium_obstacles_and_spawns():
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    assert document['mission']['command_sequence'] == [
        'takeoff', 'mission', 'land']
    assert document['mission']['coordinate_frame'] == 'stadium_endpoint'
    assert document['terrain']['coordinate_frame'] == 'stadium_endpoint'
    launcher = MISSION_LAUNCHER.read_text(encoding='utf-8')
    assert 'commands=(takeoff mission land)' in launcher
    assert "send_until_state mission 'MISSION'" in launcher
    assert '-p approach_alt:="${CJU_APPROACH_ALT_M:-8.0}"' in launcher
    assert ('-p touchdown_height_m:="${CJU_TOUCHDOWN_HEIGHT_M:-0.40}"'
            in launcher)
    assert '-p z_floor_margin_m:="${CJU_Z_FLOOR_MARGIN_M:-0.22}"' in launcher
    assert "retry_command 'A* 상태 확인'" in launcher
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
    spawn_frame = _endpoint_pose_to_enu(
        [float(v) for v in spawn_element.text.split()], document)
    assert np.allclose(spawn_frame, [spawn[key] for key in
                                     ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
    trailer = document['trailer']
    trailer_include = next(include for include in world.findall('include')
                           if include.findtext('name') == 'trailer')
    trailer_pose_element = trailer_include.find('pose')
    assert trailer_pose_element.attrib['relative_to'] == 'stadium_endpoint'
    trailer_pose = _endpoint_pose_to_enu(
        [float(v) for v in trailer_pose_element.text.split()], document)
    configured_pose = trailer['spawn_pose_enu']
    assert np.allclose(trailer_pose, [configured_pose[key] for key in
                                      ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
    assert trailer['cruise_speed_m_s'] == 3.0

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
    assert endpoint['origin_wgs84'][:2] == [36.6540480, 127.4964451]
    endpoint_enu = _wgs84_to_enu(endpoint['origin_wgs84'], origin)
    assert np.allclose(endpoint_enu[:2], endpoint['origin_enu_m'][:2],
                       atol=2.0e-6)
    heading = math.radians(reference['model_heading_deg_enu'])
    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])
    assert math.isclose(reference['model_heading_deg_enu'],
                        endpoint['heading_deg_enu'])
    assert math.isclose(
        (reference['model_heading_deg_enu'] + 90.0) % 360.0,
        document['trailer']['stadium_heading_deg'] % 360.0,
    )
    assert math.isclose(reference['stadium_long_axis_heading_deg_enu'],
                        document['trailer']['stadium_heading_deg'])
    configured_centre = (
        np.asarray(endpoint['track_center_m']) @ rotation.T
        + np.asarray(endpoint['origin_enu_m'][:2])
    )
    assert np.allclose(configured_centre,
                       document['trailer']['stadium_center_enu_m'],
                       atol=2.0e-6)
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
    assert np.allclose(track_vertices[1, :2], [0.0, 0.0])
    assert np.linalg.norm(track_vertices[:, :2], axis=1).min() < 2.0e-6
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


def test_three_english_commands_and_landing_interrupt():
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
        _patrol_origin=None,
        _planning_to_map_goal=False,
        p_d=np.array([0.0, 0.0, 3.0]),
        cue=np.zeros(3),
        _t_solve=1.0,
        state='IDLE',
        get_logger=lambda: logger,
        _cue_fresh=lambda: True,
    )
    state._set_state = lambda new, why='': setattr(state, 'state', new)
    state._start_mission_plan = lambda goal: state._set_state('MISSION_PLAN')

    MissionManagerNode._on_command(state, SimpleNamespace(data='mission'))
    assert state.state == 'IDLE' and warnings
    MissionManagerNode._on_command(state, SimpleNamespace(data='takeoff'))
    assert state._takeoff_requested
    state.state = 'READY'
    MissionManagerNode._on_command(state, SimpleNamespace(data='patrol'))
    assert state.state == 'READY'
    MissionManagerNode._on_command(state, SimpleNamespace(data='mission'))
    assert state.state == 'MISSION_PLAN'

    future = SimpleNamespace(cancelled=False)
    future.cancel = lambda: setattr(future, 'cancelled', True)
    state._plan_future = future
    MissionManagerNode._on_command(state, SimpleNamespace(data='land'))
    assert future.cancelled and state._plan_future is None
    assert state.state == 'APPROACH'


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
    )
    for _ in range(50):
        MissionManagerNode._tick(node)
    assert counts == {'state': 50, 'ocm': 50, 'send': 50}


def test_touchdown_requires_low_relative_speed():
    state = SimpleNamespace(
        p_d=np.array([0.0, 0.0, 0.20]),
        v_d=np.array([0.0, 0.0, 0.8]),
        touch_h=0.25,
        touch_xy=0.6,
        touch_v=0.4,
        _xy_to=lambda _: 0.1,
    )
    target = np.zeros(3)
    target_v = np.zeros(3)

    ready, _, _ = MissionManagerNode._touchdown_geometry(
        state, target, target_v)
    assert not ready

    state.v_d[:] = 0.1
    ready, _, _ = MissionManagerNode._touchdown_geometry(
        state, target, target_v)
    assert ready

    state.p_d[2] = 0.28
    state.v_d[:] = [0.45, 0.0, 0.0]
    state._xy_to = lambda _: 0.75
    ready, _, _ = MissionManagerNode._touchdown_geometry(
        state, target, target_v)
    assert not ready
    ready, _, _ = MissionManagerNode._touchdown_geometry(
        state, target, target_v, 1.5)
    assert ready


def test_z_floor_removes_downward_feed_forward():
    published = []
    state = SimpleNamespace(
        z_floor=0.0,
        get_clock=Clock,
        sp_pub=SimpleNamespace(publish=published.append),
    )
    pos = np.array([1.0, 2.0, -0.2])
    vel = np.array([3.0, 4.0, -1.0])
    acc = np.array([5.0, 6.0, -2.0])

    MissionManagerNode._send(state, pos, vel, acc)

    message = published[0]
    assert np.allclose(message.position, [2.0, 1.0, 0.0])
    assert np.allclose(message.velocity, [4.0, 3.0, 0.0])
    assert np.allclose(message.acceleration, [6.0, 5.0, 0.0])
    assert np.allclose(pos, [1.0, 2.0, -0.2])
    assert np.allclose(vel, [3.0, 4.0, -1.0])
    assert np.allclose(acc, [5.0, 6.0, -2.0])

    MissionManagerNode._send(
        state, np.array([1.0, 2.0, 0.2]), vel, acc)
    assert np.allclose(published[1].position, [2.0, 1.0, -0.2])
    assert np.allclose(published[1].velocity, [4.0, 3.0, 1.0])
    assert np.allclose(published[1].acceleration, [6.0, 5.0, 2.0])

    MissionManagerNode._send(
        state, np.array([1.0, 2.0, 0.0]), vel, acc)
    assert np.allclose(published[2].position, [2.0, 1.0, 0.0])
    assert np.allclose(published[2].velocity, [4.0, 3.0, 0.0])
    assert np.allclose(published[2].acceleration, [6.0, 5.0, 0.0])


def test_touchdown_never_transitions_back_to_flight():
    transitions = []
    commands = []
    now = 10.0
    state = SimpleNamespace(
        k=9,
        state='TOUCHDOWN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        p_d=np.array([2.0, 0.0, 0.20]),
        v_d=np.zeros(3),
        _t_touch=1.0,
        _t_touch_ok=None,
        _now=lambda: now,
        _target=lambda: (np.zeros(3), np.zeros(3)),
        _send=lambda *_: None,
        z_floor=0.05,
        _touchdown_geometry=lambda *_: (True, 0.1, 0.1),
        touch_dwell=2.0,
        _disarm_every=10,
        contact=False,
        _cmd=lambda *args: commands.append(args),
        armed=True,
        _cue_fresh=lambda: True,
        _set_state=lambda new, why='': transitions.append((new, why)),
    )

    MissionManagerNode._tick(state)

    assert transitions == []
    assert commands[0][1:] == (0.0, 0.0)

    state._touchdown_geometry = lambda *_: (False, 2.0, 1.0)
    MissionManagerNode._tick(state)
    assert transitions == []


def test_trailer_cue_publishes_each_gazebo_update_once():
    published = {'position': [], 'velocity': []}
    state = SimpleNamespace(
        _lock=threading.Lock(),
        _updated=True,
        _pos=np.array([1.0, 2.0]),
        _vel=np.array([3.0, 0.0]),
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


def test_stale_cue_aborts_without_flying_to_the_old_position():
    sent = []
    state = SimpleNamespace(
        k=0,
        state='APPROACH',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: None,
        p_d=np.array([1.0, 2.0, 5.0]),
        cue=np.array([50.0, 60.0, 1.811]),
        _t_cue=1.0,
        cue_timeout_s=2.0,
        _now=lambda: 10.0,
        approach_alt=16.0,
        _set_state=lambda new, why='': setattr(state, 'state', new),
        _send=lambda position, *_: sent.append(position.copy()),
    )
    state._cue_fresh = lambda: MissionManagerNode._cue_fresh(state)

    MissionManagerNode._tick(state)

    assert state.state == 'ABORT'
    assert np.allclose(sent[-1], [1.0, 2.0, 16.0])


def test_touchdown_without_status_or_fresh_cue_stays_fail_closed():
    counts = {'ocm': 0, 'send': 0}
    commands = []
    transitions = []
    state = SimpleNamespace(
        k=98,
        state='TOUCHDOWN',
        state_pub=SimpleNamespace(publish=lambda _: None),
        _ocm=lambda: counts.__setitem__('ocm', counts['ocm'] + 1),
        p_d=np.array([0.0, 0.0, 0.20]),
        cue=np.array([50.0, 60.0, 1.811]),
        _t_touch=1.0,
        _t_touch_ok=1.0,
        _t_cue=1.0,
        cue_timeout_s=2.0,
        _now=lambda: 10.0,
        _target=lambda: (_ for _ in ()).throw(AssertionError('stale cue used')),
        _send=lambda *_: counts.__setitem__('send', counts['send'] + 1),
        z_floor=0.05,
        touch_dwell=2.0,
        _disarm_every=10,
        contact=False,
        _cmd=lambda *args: commands.append(args),
        armed=None,
        get_logger=lambda: SimpleNamespace(error=lambda *_: None),
        _set_state=lambda new, why='': transitions.append((new, why)),
    )
    state._cue_fresh = lambda: MissionManagerNode._cue_fresh(state)

    MissionManagerNode._tick(state)
    MissionManagerNode._tick(state)

    assert transitions == []
    assert counts == {'ocm': 2, 'send': 2}
    assert len(commands) == 1  # 5 Hz normal retry; no stale-cue force disarm

    commands.clear()
    state.k = 108
    state.contact = True
    MissionManagerNode._tick(state)
    MissionManagerNode._tick(state)
    assert len(commands) == 1
    assert commands[0][1:] == (0.0, 21196.0)


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
