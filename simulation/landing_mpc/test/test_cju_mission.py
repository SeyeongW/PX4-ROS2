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
MISSION_LAUNCHER = GAZEBO / 'run_gimbal.sh'


def _stadium_xy(path, document):
    trailer = document['trailer']
    spawn_pose = document['spawn']['gazebo_spawn_pose_enu']
    heading = math.radians(trailer['stadium_heading_deg'])
    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])
    centre = np.asarray(trailer['stadium_center_enu_m'])
    spawn = np.asarray([spawn_pose['x'], spawn_pose['y']])
    return (path[:, :2] + spawn - centre) @ rotation


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


def test_cju_dynamic_astar_patrol_avoids_all_four_barriers_both_ways():
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    outbound, expanded_out, speed = _plan_global_path(MAP)
    # A real controller reaches a waypoint with tolerance rather than landing
    # on its exact grid coordinate. This offset reproduces that dynamic start.
    reached_goal = outbound[-1] + np.array([1.1, -0.3, 0.0])
    inbound, expanded_back, _ = _plan_global_path(
        MAP, start_local_enu=reached_goal, goal_local_enu=outbound[0])
    outbound_stadium = _stadium_xy(outbound, document)
    inbound_stadium = _stadium_xy(inbound, document)

    direct = outbound_stadium[0] + np.linspace(0.0, 1.0, 1000)[:, None] * (
        outbound_stadium[-1] - outbound_stadium[0])
    assert _blocked(direct, document['mission']).any()
    assert not _blocked(_segment_samples(outbound_stadium),
                        document['mission']).any()
    assert not _blocked(_segment_samples(inbound_stadium),
                        document['mission']).any()
    assert expanded_out > 0 and expanded_back > 0
    assert len(outbound) >= 3 and len(inbound) >= 3
    assert np.linalg.norm(outbound[-1, :2] - outbound[0, :2]) > 50.0
    half_terrain = 0.5 * np.asarray(document['terrain']['size_m'])
    assert np.all(np.abs(outbound_stadium) <= half_terrain + 1.0e-9)
    assert np.all(np.abs(inbound_stadium) <= half_terrain + 1.0e-9)
    assert speed == 2.0
    assert np.allclose(outbound[:, 2], 5.0)
    assert np.allclose(inbound[:, 2], 5.0)


def test_world_and_yaml_share_stadium_obstacles_and_spawns():
    document = yaml.safe_load(MAP.read_text(encoding='utf-8'))
    assert document['mission']['command_sequence'] == [
        'takeoff', 'mission', 'land']
    launcher = MISSION_LAUNCHER.read_text(encoding='utf-8')
    assert 'commands=(takeoff mission land)' in launcher
    assert "send_until_state mission 'MISSION'" in launcher
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
    assert np.allclose(ground_size[:2], document['terrain']['size_m'])
    assert ground.find('./link/visual') is None

    stadium = ET.parse(STADIUM_MODEL).getroot()
    plaza_uri = stadium.findtext(
        ".//visual[@name='stadium_plaza']/geometry/mesh/uri")
    assert plaza_uri == (
        'model://drone_cju_track_stadium/meshes/stadium_plaza_surface.obj')
    plaza_mesh = STADIUM_MODEL.parent / 'meshes/stadium_plaza_surface.obj'
    assert plaza_mesh.is_file()

    model = world.find("model[@name='mission_obstacles']")
    model_pose = [float(v) for v in model.findtext('pose').split()]
    assert np.allclose(model_pose[:2],
                       document['trailer']['stadium_center_enu_m'])
    assert math.isclose(model_pose[5], math.radians(
        document['trailer']['stadium_heading_deg']))
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

    spawn = document['spawn']['gazebo_spawn_pose_enu']
    spawn_frame = [float(v) for v in world.find(
        "frame[@name='drone_spawn']").findtext('pose').split()]
    assert np.allclose(spawn_frame, [spawn[key] for key in
                                     ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
    trailer = document['trailer']
    trailer_include = next(include for include in world.findall('include')
                           if include.findtext('name') == 'trailer')
    trailer_pose = [float(v) for v in trailer_include.findtext('pose').split()]
    configured_pose = trailer['spawn_pose_enu']
    assert np.allclose(trailer_pose, [configured_pose[key] for key in
                                      ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
    assert trailer['cruise_speed_m_s'] == 3.0

    track = ET.parse(TRACK_MODEL).getroot()
    surface = track.find(".//visual[@name='continuous_red_surface']")
    assert surface is not None
    colour = [float(value) for value in surface.findtext(
        'material/diffuse').split()]
    assert np.allclose(colour, [0.54, 0.18, 0.13, 1.0])


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


def test_touchdown_never_transitions_back_to_flight():
    transitions = []
    now = 10.0
    state = SimpleNamespace(
        k=1,
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
        _touchdown_geometry=lambda *_: (False, 2.0, 0.0),
        touch_dwell=2.0,
        contact=False,
        _cmd=lambda *_: None,
        armed=True,
        _cue_fresh=lambda: True,
        _set_state=lambda new, why='': transitions.append((new, why)),
    )

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
    assert len(commands) == 2  # normal disarm only; no stale-cue force disarm


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
