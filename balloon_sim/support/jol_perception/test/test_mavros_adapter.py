"""Exercise the real adapter methods without connecting ROS or a flight controller."""
import ast
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jol_perception import mavros_adapter


def command(position=(1., 2., -3.), velocity=(.3, .4, -.5), acceleration=(.01, .02, -.03)):
    xyz = lambda value:NS(x=value[0], y=value[1], z=value[2])
    return NS(header=NS(frame_id='jol/flight_ned', stamp=NS(sec=50, nanosec=0)), points=[NS(
        transforms=[NS(translation=xyz(position))], velocities=[NS(linear=xyz(velocity))],
        accelerations=[NS(linear=xyz(acceleration))])])


@pytest.fixture
def adapter():
    tree = ast.parse(Path(mavros_adapter.__file__).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    cls = next(node for node in main.body if isinstance(node, ast.ClassDef)); cls.bases = []
    namespace = dict(vars(mavros_adapter))
    namespace['SetMode'] = namespace['CommandBool'] = NS(Request=lambda:NS())
    namespace['ExtendedState'] = NS(LANDED_STATE_ON_GROUND=1)
    namespace['CommandVtolTransition']=NS(Request=lambda:NS())
    class PositionTarget(NS):
        IGNORE_YAW=1024;IGNORE_YAW_RATE=2048;FRAME_LOCAL_NED=1
        def __init__(self):super().__init__(header=NS(),position=NS(),velocity=NS(),acceleration_or_force=NS())
    namespace['PositionTarget']=PositionTarget
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 mavros_adapter.__file__, 'exec'), namespace)
    obj = namespace['Adapter'].__new__(namespace['Adapter'])
    obj.now = lambda:50.
    obj.pose = dict(stamp=49.99, position=np.zeros(3), rotation=Rotation.identity())
    obj.velocity = dict(stamp=49.99, velocity=np.zeros(3))
    obj.fcu = dict(connected=True, armed=False, mode='POSCTL', system_status=3, received=49.9)
    obj.extended = dict(landed=True, landed_state=1, received=49.9)
    obj.reference = dict(origin_world=[-5., -6., .1], initial_local_enu=[0., 0., 0.], north_yaw_world_rad=np.pi/2)
    obj.reference_samples = deque(maxlen=100)
    obj.ground_reference_revision=0;obj.ground_reference_pose_stamp=None;obj.ground_reference_source=None
    obj.home = np.array([-5., -6., .1]); obj.home_yaw = 0.
    obj.horizontal=65.;obj.height=20.
    obj.aircraft_type='iris_proxy';obj.profile={};obj.flight={}
    obj.vtol_pending=None;obj.vtol_requested_state=None;obj.vtol_requested_at=None
    obj.route_goal=None;obj.route_yaw=0.;obj.airspeed=None
    obj.launch_home=None;obj.return_stage=None;obj.return_requested=None;obj.return_plan=None
    obj.return_start=obj.return_stage_start=obj.return_dwell=None;obj.return_hold=None
    obj.return_setpoint=None
    obj.return_home_reached=False;obj.return_landed_at_home=None;obj.route_obstacles=[];obj.route_clearance=.75
    obj.vtol_client=NS(service_is_ready=lambda:True)
    obj.phase = 'standby'; obj.reason = ''; obj.command = None; obj.clock_fault = False
    obj.stream_start = 48.; obj.last_now = 49.98; obj.start_time = obj.armed_time = None
    obj.last_setpoint_sent=49.98
    obj.arm_pending = obj.mode_pending = None; obj.mode_ack = obj.arm_requested = False
    obj.stop_requested = None; obj.last_control_ack = -1.
    obj.mode_client = obj.arm_client = NS(service_is_ready=lambda:True)
    obj.mode_calls = []; obj.arm_calls = []
    obj.request_mode = lambda mode:obj.mode_calls.append(mode) or True
    obj.request_arm = lambda armed:obj.arm_calls.append(armed) or True
    obj.publish_setpoint = lambda now:None; obj.publish_feedback = lambda now:None
    obj.get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:NS(sec=50,nanosec=0)))
    obj.setpoints=[];obj.setpoint_pub=NS(publish=obj.setpoints.append)
    return obj


def reference_adapter(adapter):
    adapter.aircraft_type='tailsitter20cm_reference'
    adapter.profile=mavros_adapter.aircraft_profile(adapter.aircraft_type,
        Path(__file__).resolve().parents[2]/'jol_sim/config/tailsitter_20cm_reference.json')
    adapter.flight=adapter.profile['flight_parameters']
    adapter.phase='tracking';adapter.fcu.update(armed=True,mode='OFFBOARD')
    adapter.extended.update(landed=False,landed_state=2,vtol_state=3)
    adapter.command=dict(stamp=50.);adapter.armed_time=49.9
    adapter.vtol_calls=[];adapter.vtol_reply=Future()
    def call(request):
        adapter.vtol_calls.append(request.state)
        return adapter.vtol_reply
    adapter.vtol_client=NS(service_is_ready=lambda:True,call_async=call)
    return adapter


def advance_reference(adapter,now,state):
    adapter.now=lambda:now
    adapter.pose['stamp']=adapter.velocity['stamp']=now-.01
    adapter.fcu['received']=now-.01
    adapter.extended.update(vtol_state=state,received=now-.01)
    adapter.command=dict(stamp=now)


def test_reference_profile_is_explicit_and_enables_distinct_command_bounds():
    with pytest.raises(ValueError,match='reference_physics_profile_required'):
        mavros_adapter.aircraft_profile('tailsitter20cm_reference')
    profile=mavros_adapter.aircraft_profile('tailsitter20cm_reference',
        Path(__file__).resolve().parents[2]/'jol_sim/config/tailsitter_20cm_reference.json')
    assert profile['data_status']=='reference_scaled_not_measured'
    msg=command(velocity=(8.,0.,0.),acceleration=(3.5,0.,0.))
    with pytest.raises(ValueError,match='flight_envelope'):
        mavros_adapter.read_command(msg,50.,[-5.,-6.,.1],65.,20.)
    packet=mavros_adapter.read_command(msg,50.,[-5.,-6.,.1],65.,20.,9.,4.)
    np.testing.assert_allclose(packet['velocity'],[0.,8.,0.])


@pytest.mark.parametrize('key,value', [
    ('heading_turn_min_height_m',0.),('heading_turn_min_height_m',3.1),
    ('heading_max_rate_deg_s',0.),('heading_max_rate_deg_s',90.1),
    ('heading_max_rate_deg_s',float('nan')),('heading_max_rate_deg_s',True)])
def test_reference_heading_limits_reject_invalid_configuration(tmp_path, key, value):
    import json
    profile=json.loads((Path(__file__).resolve().parents[2]/'jol_sim/config/tailsitter_20cm_reference.json').read_text())
    profile['flight_parameters'][key]=value
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='invalid_reference_flight_parameter_'+key):
        mavros_adapter.aircraft_profile('tailsitter20cm_reference',path)


def test_vtol_command_ack_does_not_claim_observed_fixed_wing_state(adapter):
    reference_adapter(adapter)
    reply=adapter.transition_service(4,NS())
    assert reply.success and adapter.vtol_calls==[4]
    adapter.vtol_reply.set_result(NS(success=True,result=0))
    adapter.tick()
    assert adapter.vtol_state(50.)==3 and adapter.vtol_requested_state==4
    advance_reference(adapter,50.2,1);adapter.tick()
    assert adapter.vtol_requested_state==4
    advance_reference(adapter,50.4,4);adapter.tick()
    assert adapter.vtol_state(50.4)==4 and adapter.vtol_requested_state is None


def test_reference_stop_requests_mc_and_waits_for_observed_mc_before_auto_land(adapter):
    reference_adapter(adapter);adapter.extended['vtol_state']=4
    adapter.begin_land('user_stop')
    assert adapter.vtol_calls==[3] and adapter.mode_calls==[]
    adapter.vtol_reply.set_result(NS(success=True,result=0))
    advance_reference(adapter,50.2,2);adapter.tick()
    assert adapter.mode_calls==[]
    advance_reference(adapter,50.4,3);adapter.tick()
    assert adapter.mode_calls==['AUTO.LAND'] and adapter.vtol_requested_state is None


def test_return_records_actual_launch_and_lands_only_after_measured_home_arrival(adapter):
    adapter.pose['position']=np.array([.15,-.10,.02])
    assert adapter.start(None,NS()).success
    np.testing.assert_allclose(adapter.launch_home,[.15,-.10,.02])
    adapter.pose['position']=np.array([20.,5.,5.])
    reference_adapter(adapter)
    response=adapter.return_home(None,NS())
    assert response.success and adapter.phase=='returning'
    assert adapter.return_home(None,NS()).success  # A mission can attach to the watchdog's same return.
    adapter.tick()
    assert adapter.return_stage=='braking' and adapter.mode_calls==['OFFBOARD']
    advance_reference(adapter,50.1,3);adapter.tick()
    advance_reference(adapter,50.7,3);adapter.tick()
    assert adapter.return_stage=='returning' and adapter.return_plan['motion_validation']['passed']
    np.testing.assert_allclose(adapter.return_requested,[.15,-.10,5.])
    assert 'AUTO.LAND' not in adapter.mode_calls
    adapter.publish_setpoint=type(adapter).publish_setpoint.__get__(adapter,type(adapter))
    adapter.publish_setpoint(51.)
    assert adapter.setpoints[-1].position.x>adapter.launch_home[0]
    sent=adapter.setpoints[-1]
    np.testing.assert_allclose(adapter.return_setpoint['position'],
        np.array(adapter.reference['origin_world'])+[sent.position.x,sent.position.y,sent.position.z])
    np.testing.assert_allclose(adapter.return_setpoint['velocity'],[sent.velocity.x,sent.velocity.y,sent.velocity.z])
    assert adapter.return_setpoint['stamp']==51.
    adapter.pose['position']=adapter.return_requested.copy()
    advance_reference(adapter,51.,3);adapter.tick()
    assert 'AUTO.LAND' not in adapter.mode_calls
    advance_reference(adapter,52.1,3);adapter.tick()
    assert adapter.return_home_reached and adapter.return_stage=='landing'
    assert adapter.phase=='landing' and adapter.mode_calls[-1]=='AUTO.LAND'
    adapter.extended.update(landed=True,landed_state=1);adapter.fcu['armed']=False
    advance_reference(adapter,53.,3);adapter.tick()
    assert adapter.return_stage=='complete' and adapter.return_landed_at_home is True


def test_return_landing_requires_tight_home_alignment_and_low_actual_speed(adapter):
    reference_adapter(adapter);adapter.launch_home=np.zeros(3)
    adapter.phase='returning';adapter.return_stage='returning';adapter.return_start=49.
    adapter.return_stage_start=49.;adapter.return_requested=np.array([0.,0.,5.])
    adapter.pose['position']=np.array([.2,0.,5.]);adapter.velocity['velocity']=np.zeros(3)
    adapter.tick()
    assert adapter.return_dwell is None and adapter.mode_calls==[]
    adapter.pose['position']=np.array([.08,0.,5.]);adapter.velocity['velocity']=np.array([.13,0.,0.])
    advance_reference(adapter,50.5,3);adapter.tick()
    assert adapter.return_dwell is None and adapter.mode_calls==[]
    adapter.velocity['velocity']=np.array([.11,0.,0.])
    advance_reference(adapter,51.,3);adapter.tick()
    assert adapter.return_dwell==51. and adapter.mode_calls==[]
    advance_reference(adapter,52.1,3);adapter.tick()
    assert adapter.return_home_reached and adapter.mode_calls[-1]=='AUTO.LAND'
    # Final landed accuracy is assessed again; hover alignment cannot hide ground drift.
    adapter.pose['position']=np.array([.36,0.,0.])
    adapter.extended.update(landed=True,landed_state=1);adapter.fcu['armed']=False
    advance_reference(adapter,53.,3);adapter.tick()
    assert adapter.return_landed_at_home is False and adapter.return_stage=='landed_away_from_takeoff'


def test_reference_return_waits_for_mc_then_brakes_before_translating(adapter):
    reference_adapter(adapter);adapter.launch_home=np.zeros(3)
    adapter.extended['vtol_state']=4;adapter.pose['position']=np.array([30.,10.,8.])
    adapter.velocity['velocity']=np.array([8.,0.,0.])
    assert adapter.return_home(None,NS()).success
    assert adapter.vtol_calls==[3] and adapter.mode_calls==[]
    adapter.tick();assert adapter.return_stage=='transition_mc'
    adapter.vtol_reply.set_result(NS(success=True,result=0))
    advance_reference(adapter,50.2,2);adapter.tick()
    assert adapter.return_stage=='transition_mc'
    advance_reference(adapter,50.4,3);adapter.tick()
    assert adapter.return_stage=='braking' and adapter.return_plan is None
    advance_reference(adapter,51.,3);adapter.tick()
    assert adapter.return_stage=='braking' and adapter.mode_calls==[]


def test_return_rejects_missing_home_and_feedback_loss_uses_explicit_emergency_land(adapter):
    reference_adapter(adapter)
    assert not adapter.return_home(None,NS()).success
    adapter.launch_home=np.zeros(3)
    assert adapter.return_home(None,NS()).success
    adapter.pose['stamp']=49.
    adapter.tick()
    assert adapter.phase=='landing' and adapter.return_stage=='failed_emergency_landing'
    assert adapter.reason=='return_feedback_or_OFFBOARD_lost'
    assert adapter.mode_calls and all(mode=='AUTO.LAND' for mode in adapter.mode_calls)
    assert not adapter.return_home_reached


def test_offboard_loss_streams_hold_and_requires_real_ack_and_new_mode_feedback(adapter):
    reference_adapter(adapter);adapter.launch_home=np.zeros(3)
    adapter.pose['position']=np.array([30.,10.,8.]);adapter.extended['vtol_state']=4
    adapter.fcu['mode']='AUTO.LOITER';adapter.tick()
    assert adapter.phase=='returning' and adapter.return_stage=='reacquire_offboard'
    assert adapter.return_home(None,NS()).success and adapter.vtol_calls==[]
    adapter.publish_setpoint=type(adapter).publish_setpoint.__get__(adapter,type(adapter))
    adapter.publish_setpoint(50.)
    np.testing.assert_allclose([adapter.setpoints[-1].position.x,adapter.setpoints[-1].position.y,
                               adapter.setpoints[-1].position.z],[30.,10.,8.])
    reply=Future();requests=[]
    adapter.mode_client=NS(service_is_ready=lambda:True,call_async=lambda request:requests.append(request) or reply)
    adapter.request_mode=type(adapter).request_mode.__get__(adapter,type(adapter))
    advance_reference(adapter,50.5,4);adapter.tick();assert requests==[]
    advance_reference(adapter,51.1,4);adapter.tick()
    assert [request.custom_mode for request in requests]==['OFFBOARD'] and not adapter.mode_ack
    reply.set_result(NS(mode_sent=True))
    advance_reference(adapter,51.2,4);adapter.tick()
    assert adapter.return_stage=='reacquire_offboard' and adapter.vtol_calls==[]
    adapter.fcu['mode']='OFFBOARD';adapter.fcu['received']=51.1;adapter.tick()
    assert adapter.return_stage=='reacquire_offboard'  # Old mode feedback cannot confirm the ACK.
    advance_reference(adapter,51.4,4);adapter.tick()
    assert adapter.return_stage=='transition_mc' and adapter.vtol_calls==[3]


def test_failed_offboard_reacquisition_has_deadline_and_explicit_local_landing(adapter):
    reference_adapter(adapter);adapter.launch_home=np.zeros(3);adapter.fcu['mode']='AUTO.LOITER'
    adapter.tick()
    advance_reference(adapter,51.1,3);adapter.tick()
    assert adapter.mode_calls==['OFFBOARD'] and adapter.return_stage=='reacquire_offboard'
    advance_reference(adapter,56.1,3);adapter.tick()
    assert adapter.phase=='landing' and adapter.return_stage=='failed_emergency_landing'
    assert adapter.reason=='return_OFFBOARD_reacquisition_failed_emergency_local_landing'
    assert 'AUTO.LAND' in adapter.mode_calls and not adapter.return_home_reached


def test_home_return_geometry_preserves_height_and_certifies_obstacles():
    plan,goal=mavros_adapter.home_return_plan([40.,10.,8.],[-2.,0.,.1],[-2.,0.,.1],65.,20.)
    np.testing.assert_allclose(goal,[-2.,0.,8.])
    assert plan['motion_validation']['passed']
    with pytest.raises(ValueError,match='controller_envelope'):
        mavros_adapter.home_return_plan([40.,10.,21.],[-2.,0.,.1],[-2.,0.,.1],65.,20.)
    with pytest.raises(ValueError,match='obstacle'):
        mavros_adapter.home_return_plan([40.,10.,8.],[-2.,0.,.1],[-2.,0.,.1],65.,20.,
            obstacles=[dict(min=[18.,3.,7.],max=[22.,7.,9.])])


def test_reference_pva_sets_ros_enu_route_yaw_and_keeps_legacy_yaw_ignored(adapter):
    adapter.publish_setpoint=type(adapter).publish_setpoint.__get__(adapter,type(adapter))
    adapter.publish_setpoint(50.)
    assert adapter.setpoints[-1].type_mask & 1024
    reference_adapter(adapter);adapter.route_goal=np.array([3.,0.,8.])
    adapter.launch_home=np.zeros(3);adapter.pose['position']=np.array([0.,0.,1.])
    adapter.command=dict(stamp=50.,position=np.zeros(3),velocity=np.zeros(3),acceleration=np.zeros(3))
    adapter.publish_setpoint(50.)
    msg=adapter.setpoints[-1]
    assert msg.type_mask==2048
    assert msg.yaw==pytest.approx(0.)  # No elapsed send time, so no yaw step.
    adapter.command['velocity']=np.array([3.,4.,0.])
    adapter.publish_setpoint(50.02)
    assert adapter.setpoints[-1].yaw==pytest.approx(np.deg2rad(45.)*.02)


def test_reference_preview_and_arming_preserve_measured_ground_heading(adapter):
    reference_adapter(adapter)
    adapter.publish_setpoint=type(adapter).publish_setpoint.__get__(adapter,type(adapter))
    adapter.phase='standby';adapter.fcu['armed']=False
    adapter.extended.update(landed=True,landed_state=1)
    adapter.pose['rotation']=Rotation.from_euler('z',.37)
    adapter.route_goal=np.array([0.,-40.,8.])
    adapter.publish_setpoint(50.)
    assert adapter.setpoints[-1].yaw==pytest.approx(.37)
    assert adapter.start(None,NS()).success
    adapter.pose['rotation']=Rotation.from_euler('z',.4)
    adapter.publish_setpoint(50.02)
    assert adapter.phase=='arming' and adapter.setpoints[-1].yaw==pytest.approx(.37)
    adapter.phase='landing';adapter.publish_setpoint(50.04)
    assert adapter.setpoints[-1].yaw==pytest.approx(.4)


@pytest.mark.parametrize('blocked', ['low','landed','unknown','stale','future','unarmed','no_home','future_pose'])
def test_reference_heading_turn_requires_actual_fresh_airborne_clearance(adapter, blocked):
    reference_adapter(adapter)
    adapter.publish_setpoint=type(adapter).publish_setpoint.__get__(adapter,type(adapter))
    adapter.launch_home=np.zeros(3);adapter.pose['position']=np.array([0.,0.,.7])
    adapter.route_goal=np.array([0.,-40.,8.]);adapter.route_yaw=.3
    adapter.command=dict(stamp=50.,position=np.zeros(3),velocity=np.array([0.,0.,1.]),acceleration=np.zeros(3))
    if blocked=='low':adapter.pose['position'][2]=.01  # A premature IN_AIR report is insufficient.
    elif blocked=='landed':adapter.extended.update(landed=True,landed_state=1)
    elif blocked=='unknown':adapter.extended['landed_state']=0
    elif blocked=='stale':adapter.extended['received']=47.
    elif blocked=='future':adapter.extended['received']=50.001
    elif blocked=='unarmed':adapter.fcu['armed']=False
    elif blocked=='no_home':adapter.launch_home=None
    elif blocked=='future_pose':adapter.pose['stamp']=50.001
    adapter.publish_setpoint(50.)
    assert adapter.setpoints[-1].yaw==pytest.approx(.3)


@pytest.mark.parametrize('phase', ['tracking','returning'])
def test_reference_turns_gradually_during_vertical_flight_without_changing_pva(adapter, phase):
    reference_adapter(adapter);adapter.phase=phase
    adapter.publish_setpoint=type(adapter).publish_setpoint.__get__(adapter,type(adapter))
    adapter.launch_home=np.zeros(3);adapter.pose['position']=np.array([0.,0.,.6])
    adapter.route_goal=np.array([-5.,-46.,8.]);adapter.route_yaw=0.
    adapter.command=dict(stamp=50.,position=np.array([0.,0.,.6]),velocity=np.array([0.,0.,1.]),acceleration=np.zeros(3))
    adapter.last_setpoint_sent=50.
    for i in range(1,6):
        now=50.+i*.02
        adapter.pose['stamp']=adapter.velocity['stamp']=now
        adapter.fcu['received']=adapter.extended['received']=now
        adapter.publish_setpoint(now)
        sent=adapter.setpoints[-1]
        assert sent.yaw==pytest.approx(-np.deg2rad(45.)*i*.02)
        assert sent.velocity.x==0. and sent.velocity.y==0.
        if phase=='tracking':assert sent.velocity.z==1.


@pytest.mark.parametrize('sign', [-1.,1.])
def test_reference_yaw_slew_uses_shortest_wraparound_and_clamps_elapsed_time(adapter, sign):
    reference_adapter(adapter)
    adapter.publish_setpoint=type(adapter).publish_setpoint.__get__(adapter,type(adapter))
    adapter.launch_home=np.zeros(3);adapter.pose['position']=np.array([0.,0.,1.])
    adapter.route_yaw=sign*np.deg2rad(179.)
    desired=-sign*np.deg2rad(179.)
    direction=np.array([np.cos(desired),np.sin(desired),0.])
    adapter.command=dict(stamp=50.,position=np.zeros(3),velocity=direction,acceleration=np.zeros(3))
    adapter.last_setpoint_sent=49.98
    adapter.publish_setpoint(50.)
    delta=np.arctan2(np.sin(adapter.route_yaw-sign*np.deg2rad(179.)),
                     np.cos(adapter.route_yaw-sign*np.deg2rad(179.)))
    assert delta==pytest.approx(sign*np.deg2rad(.9))
    adapter.route_yaw=0.;adapter.last_setpoint_sent=40.
    adapter.publish_setpoint(50.)
    assert adapter.route_yaw==pytest.approx(-sign*np.deg2rad(4.5))


@pytest.mark.parametrize('change', ['legacy','unarmed','mode','grounded','stale'])
def test_reference_transition_service_requires_real_fresh_offboard_airborne_state(adapter,change):
    reference_adapter(adapter)
    if change=='legacy':adapter.aircraft_type='iris_proxy'
    elif change=='unarmed':adapter.fcu['armed']=False
    elif change=='mode':adapter.fcu['mode']='POSCTL'
    elif change=='grounded':adapter.extended['landed']=True
    elif change=='stale':adapter.pose['stamp']=49.
    assert not adapter.transition_service(4,NS()).success
    assert adapter.vtol_calls==[]


def test_ned_pva_is_converted_once_to_mavros_ros_enu():
    result = mavros_adapter.read_command(command(), 50., [-5., -6., .1], 65., 20.)
    np.testing.assert_allclose(result['position'], [2., 1., 3.])
    np.testing.assert_allclose(result['velocity'], [.4, .3, .5])
    np.testing.assert_allclose(result['acceleration'], [.02, .01, .03])
    # The MAVROS local_cb ENU->NED transform then restores the original desired NED.
    np.testing.assert_allclose(mavros_adapter.ned_to_enu(result['position']), [1., 2., -3.])


def test_reference_uses_surveyed_spawn_and_ekf_translation_without_rotating_world():
    samples = [dict(stamp=i*.1, position=[2., 3., -.2], velocity=[0., 0., 0.], yaw=.3) for i in range(11)]
    result = mavros_adapter.reference_from_samples(samples, [-5., -6., .1], .1)
    np.testing.assert_allclose(result['origin_world'], [-7., -9., .3])
    assert result['north_yaw_world_rad'] == np.pi/2
    assert result['initial_yaw_error_rad'] == pytest.approx(.2)
    samples[-1]['velocity'] = [.11, 0., 0.]
    with pytest.raises(ValueError, match='stationary_reference_required'):
        mavros_adapter.reference_from_samples(samples, [-5., -6., .1], .1)


@pytest.mark.parametrize('mutation,reason', [
    (lambda msg:setattr(msg.header, 'frame_id', 'world'), 'single_NED'),
    (lambda msg:setattr(msg.header.stamp, 'sec', 49), 'fresh_setpoint'),
    (lambda msg:setattr(msg.points[0], 'accelerations', []), 'complete_PVA'),
    (lambda msg:setattr(msg.points[0].transforms[0].translation, 'x', float('nan')), 'finite_XYZ'),
    (lambda msg:setattr(msg.points[0].transforms[0].translation, 'x', 80.), 'flight_envelope'),
])
def test_command_input_guards(mutation, reason):
    msg = command(); mutation(msg)
    with pytest.raises(ValueError, match=reason):
        mavros_adapter.read_command(msg, 50., [-5., -6., .1], 65., 20.)


def test_startup_only_prestreams_and_never_arms(adapter):
    adapter.tick()
    assert adapter.phase == 'standby' and adapter.ready(50.)
    assert adapter.mode_calls == [] and adapter.arm_calls == []


def test_fresh_ekf_cannot_enable_start_before_px4_preflight_health_pass(adapter):
    adapter.receive_state(NS(connected=True,armed=False,mode='POSCTL',system_status=0))
    assert adapter.fresh(50.) and not adapter.ready(50.)
    assert not adapter.start(None,NS()).success
    assert adapter.arm_calls==[] and adapter.mode_calls==[]
    adapter.receive_state(NS(connected=True,armed=False,mode='POSCTL',system_status=3))
    assert adapter.ready(50.)


@pytest.mark.parametrize('system_status', [0, 2])
def test_initial_reference_discards_preflight_samples_and_restarts_healthy_warmup(adapter, system_status):
    adapter.reference = None
    adapter.reference_samples.extend(dict(stamp=48.+i*.1, position=[2.,3.,-.2],
        velocity=[0.,0.,0.], yaw=.3) for i in range(11))
    adapter.fcu['system_status'] = system_status
    adapter.tick()
    assert adapter.reference is None and not adapter.reference_samples
    assert adapter.arm_calls == [] and adapter.mode_calls == []
    adapter.fcu['system_status'] = 3
    for i in range(11):
        now = 50.+i*.1
        adapter.now = lambda now=now:now
        adapter.pose.update(stamp=now, position=np.array([2.,3.,-.2]),
                            rotation=Rotation.from_euler('z',.3))
        adapter.velocity['stamp'] = now
        adapter.fcu['received'] = adapter.extended['received'] = now
        adapter.tick()
        if i < 10:assert adapter.reference is None
    assert adapter.reference['reference_sample_count'] == 11
    np.testing.assert_allclose(adapter.reference['origin_world'], [-7.,-9.,.3])
    assert adapter.reference['north_yaw_world_rad'] == np.pi/2
    assert adapter.reference['initial_yaw_error_rad'] == pytest.approx(.3)


def test_initial_reference_warmup_requires_continuous_causal_feedback(adapter):
    adapter.reference = None
    for i in range(6):
        now = 50.+i*.1
        adapter.now = lambda now=now:now
        adapter.pose['stamp'] = adapter.velocity['stamp'] = now
        adapter.fcu['received'] = adapter.extended['received'] = now
        adapter.tick()
    assert adapter.reference is None and len(adapter.reference_samples) == 6
    adapter.pose['stamp'] = 50.501  # A future pose cannot enter the calibration window.
    adapter.tick()
    assert adapter.reference is None and not adapter.reference_samples
    adapter.pose['stamp'] = 50.5
    adapter.tick()
    assert len(adapter.reference_samples) == 1
    assert adapter.reference_samples[0]['stamp'] == 50.5


def test_preflight_reference_corrects_only_ground_z_with_current_stationary_pose(adapter):
    original = dict(adapter.reference)
    for i in range(11):
        now = 50.+i*.1
        adapter.now = lambda now=now:now
        adapter.pose.update(stamp=now, position=np.array([.02,-.03,-.08]))
        adapter.velocity['stamp'] = now
        adapter.fcu['received'] = adapter.extended['received'] = now
        adapter.tick()
    assert adapter.reference == original  # Rolling samples alone never move the frame.
    adapter.now = lambda:51.1
    adapter.pose.update(stamp=51.1,position=np.array([.02,-.03,-.09]))
    adapter.velocity['stamp'] = 51.1
    adapter.command = dict(stamp=51.)
    reply = adapter.prepare_reference(None,NS())
    assert reply.success and adapter.reference_samples[-1]['stamp'] == 51.1
    np.testing.assert_allclose(adapter.reference['origin_world'][:2],original['origin_world'][:2])
    assert adapter.reference['north_yaw_world_rad'] == original['north_yaw_world_rad']
    expected_z = adapter.home[2]-np.mean([s['position'][2] for s in adapter.reference_samples])
    assert adapter.reference['origin_world'][2] == pytest.approx(expected_z)
    assert .05 <= adapter.reference['origin_world'][2]+adapter.pose['position'][2] <= .15
    np.testing.assert_allclose(adapter.reference['initial_local_enu'],adapter.pose['position'])
    assert adapter.ground_reference_revision == 1 and adapter.ground_reference_pose_stamp == 51.1
    assert adapter.ground_reference_source == 'declared_flat_simulation_ground_and_stationary_EKF'
    assert adapter.command is None and adapter.arm_calls == [] and adapter.mode_calls == []


@pytest.mark.parametrize('blocked', ['armed','stale','moving','busy','pending'])
def test_preflight_reference_rejects_unsafe_updates_without_changing_frame(adapter, blocked):
    adapter.reference_samples.extend(dict(stamp=48.9+i*.1,position=np.zeros(3),
        velocity=np.zeros(3),yaw=0.) for i in range(11))
    original = dict(adapter.reference)
    if blocked == 'armed':adapter.fcu['armed'] = True
    elif blocked == 'stale':adapter.pose['stamp'] = 49.
    elif blocked == 'moving':adapter.velocity['velocity'] = np.array([.11,0.,0.])
    elif blocked == 'busy':adapter.phase = 'arming'
    elif blocked == 'pending':adapter.mode_pending = Future()
    assert not adapter.prepare_reference(None,NS()).success
    assert adapter.reference == original and adapter.ground_reference_revision == 0
    assert adapter.arm_calls == [] and adapter.mode_calls == []


def test_preflight_frame_remains_fixed_after_arming_and_in_flight(adapter):
    adapter.reference_samples.extend(dict(stamp=48.9+i*.1,position=np.zeros(3),
        velocity=np.zeros(3),yaw=0.) for i in range(11))
    assert adapter.prepare_reference(None,NS()).success
    original = dict(adapter.reference)
    reference_adapter(adapter)
    adapter.pose['position'] = np.array([3.,4.,8.])
    advance_reference(adapter,50.2,3)
    adapter.tick()
    assert not adapter.reference_samples
    assert not adapter.prepare_reference(None,NS()).success
    assert adapter.reference == original and adapter.ground_reference_revision == 1


def test_explicit_start_requires_fresh_grounded_feedback_and_pre_stream(adapter):
    adapter.stream_start = 49.5
    response = adapter.start(None, NS())
    assert not response.success and adapter.arm_calls == [] and adapter.mode_calls == []
    adapter.stream_start = 48.
    response = adapter.start(None, NS())
    assert response.success and adapter.phase == 'arming'
    assert adapter.mode_calls == ['OFFBOARD'] and adapter.arm_calls == []
    assert not adapter.start(None, NS()).success
    adapter.fcu['mode'] = 'OFFBOARD'; adapter.mode_ack = True
    adapter.tick()
    assert adapter.arm_calls == [True]
    assert adapter.phase == 'arming'  # Command ACK is not proof of actual arming.
    adapter.fcu['armed'] = True; adapter.tick()
    assert adapter.phase == 'tracking'


def test_stale_trajectory_requests_px4_autonomous_landing(adapter):
    adapter.phase = 'tracking'; adapter.fcu.update(armed=True, mode='OFFBOARD')
    adapter.extended['landed'] = False; adapter.armed_time = 48.
    adapter.command = dict(stamp=49.5)
    adapter.tick()
    assert adapter.phase == 'landing' and adapter.reason == 'trajectory_setpoint_watchdog'
    assert 'AUTO.LAND' in adapter.mode_calls and True not in adapter.arm_calls


def test_stop_and_late_arming_never_restart_tracking(adapter):
    adapter.phase = 'arming'; adapter.arm_pending = object()
    adapter.stop(None, NS()); adapter.tick()
    assert adapter.phase == 'landing'
    adapter.arm_pending = None; adapter.tick()
    assert adapter.phase == 'landing'  # Old disarmed heartbeat predates Stop.
    adapter.now = lambda:51.; adapter.fcu['received'] = 50.9
    adapter.extended['received'] = 50.9
    adapter.pose['stamp'] = adapter.velocity['stamp'] = 50.99
    adapter.tick()
    assert adapter.phase == 'standby'
    adapter.fcu['armed'] = True; adapter.extended['landed'] = False
    adapter.tick()
    assert adapter.phase == 'landing' and 'AUTO.LAND' in adapter.mode_calls
    assert True not in adapter.arm_calls


def test_landing_keeps_uninterrupted_hold_stream_ready(adapter):
    adapter.phase = 'landing'
    adapter.tick()
    assert adapter.phase == 'standby' and adapter.stream_start == 48.
    assert adapter.ready(50.)


def test_short_future_feedback_tick_does_not_publish_or_erase_actual_stream_warmup(adapter):
    adapter.publish_setpoint=type(adapter).publish_setpoint.__get__(adapter,type(adapter))
    adapter.publish_setpoint(50.)
    count=len(adapter.setpoints);adapter.pose['stamp']=50.001
    adapter.tick()
    assert len(adapter.setpoints)==count and not adapter.ready(50.)
    assert adapter.stream_start==48. and adapter.last_setpoint_sent==50.
    adapter.now=lambda:50.02;adapter.pose['stamp']=adapter.velocity['stamp']=50.01
    adapter.tick()
    assert adapter.stream_start==48. and adapter.ready(50.02)
    assert adapter.last_setpoint_sent==50.02


def test_real_setpoint_stream_gap_requires_new_prestream_warmup(adapter):
    adapter.publish_setpoint=type(adapter).publish_setpoint.__get__(adapter,type(adapter))
    adapter.publish_setpoint(50.)
    adapter.now=lambda:50.3;adapter.pose['stamp']=50.301
    adapter.tick()
    assert adapter.stream_start is None and adapter.last_setpoint_sent==50.
    adapter.now=lambda:50.32;adapter.pose['stamp']=adapter.velocity['stamp']=50.31
    adapter.tick()
    assert adapter.stream_start==50.32 and not adapter.ready(50.32)


def test_landing_completion_waits_for_heartbeat_after_last_control_ack(adapter):
    adapter.begin_land('user_stop'); adapter.last_control_ack = 50.5
    adapter.now = lambda:51.; adapter.fcu['received'] = 50.4
    adapter.extended['received'] = 50.9
    adapter.pose['stamp'] = adapter.velocity['stamp'] = 50.99
    adapter.tick()
    assert adapter.phase == 'landing'
    adapter.fcu['received'] = 50.9
    adapter.tick()
    assert adapter.phase == 'standby' and adapter.ready(51.)


def test_feedback_staleness_and_clock_rewind_remove_readiness(adapter):
    adapter.pose['stamp'] = 49.
    assert not adapter.ready(50.)
    adapter.pose['stamp'] = 49.99; adapter.last_now = 51.
    adapter.tick()
    assert adapter.clock_fault and not adapter.ready(50.)


def test_actual_landed_state_is_preserved_and_ground_disarm_is_not_arming(adapter):
    adapter.receive_extended(NS(landed_state=2))
    assert adapter.extended['landed_state'] == 2 and not adapter.landed(50.)
    adapter.receive_extended(NS(landed_state=1)); adapter.fcu['armed'] = True
    adapter.begin_land('user_stop')
    assert adapter.arm_calls == [False]
    source = Path(mavros_adapter.__file__).read_text()
    assert 'landed_state=self.extended' in source
    assert '/gazebo/set_entity_state' not in source and '/gazebo/model_states' not in source
