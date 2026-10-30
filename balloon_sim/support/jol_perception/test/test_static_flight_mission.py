"""Exercise the real mission methods without ROS/Gazebo or a second simulator."""
import ast
from collections import deque
from concurrent.futures import Future
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jol_perception import static_flight
from jol_perception.static_flight_core import build_static_plan


@pytest.fixture
def mission(tmp_path):
    tree = ast.parse(Path(static_flight.__file__).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    cls = next(node for node in main.body if isinstance(node, ast.ClassDef))
    cls.bases = []
    namespace = dict(vars(static_flight))
    namespace['String'] = lambda **values: SimpleNamespace(**values)
    namespace['Trigger'] = SimpleNamespace(Request=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 static_flight.__file__, 'exec'), namespace)
    obj = namespace['Mission'].__new__(namespace['Mission'])
    obj.target_source='camera_bbox_lidar';obj.backend='native';obj.start_client=None;obj.stop_client=None
    obj.return_client=None;obj.return_future=None;obj.checkpoint=dict(outcome='pending');obj.gate_feedback=None
    obj.aircraft_type='iris_proxy';obj.physics_profile={};obj.reference_flight={}
    obj.vtol_stage='multicopter';obj.vtol_future=None;obj.vtol_stage_start=None;obj.fw_observed=False
    obj.fw_client=None;obj.mc_client=None
    obj.arm_future=None;obj.arm_requested=None;obj.autopilot_land_start=None
    obj.transfer_start=None;obj.takeoff_settle_dwell=None
    own = dict(stamp=49.99, position=[-5., -6., .1], velocity=[0., 0., 0.])
    controller = dict(stamp=49.99, controller_ready=True, setpoint_frame='jol/flight_ned',
        origin_world=[-5., -6., .12], north_yaw_world_rad=np.pi/2,
        horizontal_limit_m=65., height_limit_m=20., position=own['position'],
        velocity=own['velocity'], acceleration_world=[0., 0., 0.],
        acceleration_source='native_velocity_finite_difference')
    estimate = dict(stamp=49.99, valid=True, frame_id='world', source='camera_bbox_lidar',
        target_id=1, position=[37.1, 13.5, 8.], covariance=(np.eye(3)*.0225).tolist(),
        evidence=dict(gimbal_centered=True))
    obj.now = lambda: 50.
    obj.own = own; obj.own_history = deque([own])
    obj.controller = controller; obj.controller_history = deque([controller])
    obj.locked = estimate; obj.estimates = deque([estimate])
    obj.plan = build_static_plan(own['position'], estimate['position'])
    obj.lock_frame = dict(origin_world=controller['origin_world'],
                          north_yaw_world_rad=controller['north_yaw_world_rad'])
    obj.options = dict(path_kind='spline', standoff=1., max_speed=1.5, max_acceleration=.6, max_jerk=1.2)
    obj.scene = dict(obstacles=[]); obj.output = tmp_path
    obj.phase = 'preview_ready'; obj.reason = ''; obj.last_now = 49.9
    obj.start = 0.; obj.landing = None; obj.dwell = None
    obj.last_save = 50.; obj.samples = []; obj.flown = []; obj.events = []
    obj.flight_launched = False; obj.saved_samples = 0
    obj.commands = []; obj.command = obj.commands.append
    obj.status_messages = []
    obj.status_pub = SimpleNamespace(publish=obj.status_messages.append)
    obj.flown_pub = object();obj.path_pub=object(); obj.published_paths = []
    obj.plan_messages=[];obj.plan_pub=SimpleNamespace(publish=obj.plan_messages.append)
    obj.publish_path = lambda pub, positions: obj.published_paths.append((pub, positions))
    obj.publish_plan = lambda: None
    obj.save = lambda: None
    def transition(phase, reason=''):
        obj.phase, obj.reason, obj.dwell = phase, reason, None
        obj.events.append(dict(stamp=obj.now(), phase=phase, reason=reason))
    obj.transition = transition
    return obj


@pytest.fixture
def mavros_mission(mission):
    """Use asynchronous service futures and genuine FCU state acknowledgements."""
    mission.backend = 'mavros'; mission.start = None
    mission.controller.update(armed=False, mode='POSCTL', connected=True, landed_state=1,
                              own_pose_source='mavros_px4_ekf', adapter_phase='standby',
                              pending_control_command=False,fcu_state_stamp=49.9)
    mission.start_requests = []; mission.stop_requests = []
    future = Future()
    def request_start(request):
        mission.start_requests.append(request)
        return future
    mission.start_client = SimpleNamespace(service_is_ready=lambda: True, call_async=request_start)
    mission.stop_client = SimpleNamespace(service_is_ready=lambda: True,
                                          call_async=mission.stop_requests.append)
    mission.start_reply = future
    return mission


def advance(mission, stamp):
    mission.now = lambda: stamp
    mission.own['stamp'] = stamp-.01
    mission.controller['stamp'] = stamp-.01
    if 'fcu_state_stamp' in mission.controller:mission.controller['fcu_state_stamp']=stamp-.01


def launch(mission):
    return mission.action('launch', SimpleNamespace(success=None, message=''))


def test_mavros_lock_waits_for_reference_ack_revision_and_new_frame_odometry(mavros_mission):
    mission=mavros_mission;future=Future();requests=[]
    mission.reference_client=SimpleNamespace(service_is_ready=lambda:True,
        call_async=lambda request:(requests.append(request),future)[1])
    mission.controller['ground_reference_revision']=2
    mission.locked=None;mission.plan=None;mission.phase='coordinate_ready'
    response=mission.action('lock',SimpleNamespace(success=None,message=''))
    assert response.success and mission.phase=='locked' and mission.plan is None
    assert mission.reason=='ground_reference_preparation_pending'
    assert len(requests)==1 and mission.start_requests==[] and not mission.can_launch(50.)
    # Repeated preview cannot issue a second refresh or start flight.
    assert mission.action('preview',SimpleNamespace(success=None,message='')).success
    assert len(requests)==1
    future.set_result(SimpleNamespace(success=True,message='ground_reference_prepared'))
    mission.tick()
    assert mission.plan is None  # ACK alone is not a new coordinate frame.
    mission.controller.update(ground_reference_revision=3,ground_reference_pose_stamp=49.99)
    mission.tick()
    assert mission.plan is None  # This odometry still predates the refreshed frame.
    advance(mission,50.1);mission.tick()
    assert mission.phase=='preview_ready' and mission.plan is not None
    assert mission.start_requests==[] and mission.commands==[] and not mission.flight_launched


@pytest.mark.parametrize('failure',['unavailable','rejected','timeout'])
def test_mavros_reference_preparation_failure_never_previews_or_launches(mavros_mission,failure):
    mission=mavros_mission;future=Future()
    mission.reference_client=SimpleNamespace(service_is_ready=lambda:failure!='unavailable',
        call_async=lambda request:future)
    mission.locked=None;mission.plan=None;mission.phase='coordinate_ready'
    response=mission.action('lock',SimpleNamespace(success=None,message=''))
    assert response.success and mission.phase=='locked'
    if failure=='rejected':
        future.set_result(SimpleNamespace(success=False,message='ground_motion_or_state_invalid'))
        mission.tick()
    elif failure=='timeout':
        advance(mission,55.1);mission.tick()
    expected={'unavailable':'MAVROS_prepare_reference_service_required',
              'rejected':'ground_reference_preparation_rejected',
              'timeout':'ground_reference_preparation_timeout'}[failure]
    assert expected in mission.reason and mission.plan is None
    assert not mission.can_launch(mission.now()) and mission.start_requests==[]
    assert mission.commands==[] and not mission.flight_launched


@pytest.mark.parametrize('message',['stationary_reference_required','stationary_reference_warmup_required'])
def test_mavros_reference_transient_stationarity_retry_preserves_handshake_and_never_arms(mavros_mission,message):
    mission=mavros_mission;futures=[]
    def request_reference(request):
        future=Future();futures.append(future);return future
    mission.reference_client=SimpleNamespace(service_is_ready=lambda:True,call_async=request_reference)
    mission.controller['ground_reference_revision']=2
    mission.action('preview',SimpleNamespace())
    futures[0].set_result(SimpleNamespace(success=False,message=message))
    mission.tick()
    advance(mission,50.2);mission.tick()
    assert len(futures)==1 and mission.reason=='ground_reference_preparation_pending'
    advance(mission,50.4);mission.tick()
    assert len(futures)==2 and mission.reference_requested==50. and mission.reference_requested_revision==2
    futures[1].set_result(SimpleNamespace(success=True,message='prepared'))
    mission.controller.update(ground_reference_revision=3,ground_reference_pose_stamp=50.39)
    mission.tick()
    assert mission.plan is None
    advance(mission,50.5);mission.tick()
    assert mission.phase=='preview_ready' and mission.plan is not None
    assert mission.start_requests==[] and mission.commands==[] and not mission.flight_launched


def test_mavros_reference_stationarity_retries_share_original_timeout(mavros_mission):
    mission=mavros_mission;requests=[]
    def request_reference(request):
        requests.append(request);future=Future()
        future.set_result(SimpleNamespace(success=False,message='stationary_reference_required'))
        return future
    mission.reference_client=SimpleNamespace(service_is_ready=lambda:True,call_async=request_reference)
    mission.action('preview',SimpleNamespace())
    for stamp in np.arange(50.,55.21,.1):
        advance(mission,float(stamp));mission.estimates[-1]['stamp']=float(stamp)-.01;mission.tick()
    assert len(requests)>1 and mission.reference_requested==50.
    assert mission.phase=='locked' and mission.plan is None and 'preparation_timeout' in mission.reason
    assert not mission.can_launch(mission.now()) and mission.start_requests==[] and mission.commands==[]


def test_mavros_unavailable_reference_repreview_invalidates_previous_launchable_plan(mavros_mission):
    mission=mavros_mission
    assert mission.can_launch(50.)
    mission.reference_client=SimpleNamespace(service_is_ready=lambda:False)
    response=mission.action('preview',SimpleNamespace())
    assert not response.success and mission.phase=='locked' and mission.plan is None
    assert 'MAVROS_prepare_reference_service_required' in mission.reason
    assert not mission.can_launch(50.) and mission.start_requests==[] and mission.commands==[]


def test_mavros_launch_waits_for_service_reply_and_armed_offboard_feedback(mavros_mission):
    mission = mavros_mission
    assert launch(mission).success is True
    assert mission.phase == 'arming' and mission.start is None
    mission.tick()
    assert mission.commands == [] and mission.samples == []
    assert len(mission.start_requests) == 1

    # Neither an accepted service request nor one FCU acknowledgement is flight.
    mission.start_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(armed=False, mode='OFFBOARD')
    advance(mission, 50.2); mission.tick()
    assert mission.phase == 'arming' and mission.start is None and mission.commands == []
    mission.controller.update(armed=True, mode='POSCTL')
    advance(mission, 50.4); mission.tick()
    assert mission.phase == 'arming' and mission.start is None and mission.commands == []

    mission.controller['mode'] = 'OFFBOARD'
    advance(mission, 50.6); mission.tick()
    assert mission.phase == 'takeoff' and mission.start == 50.6
    assert len(mission.commands) == 1 and len(mission.samples) == 1
    assert np.allclose(mission.commands[0]['position'], mission.plan['home'])
    assert not any(event.get('phase') == 'arrived' for event in mission.events)


def test_reset_does_not_relabel_recorded_gate_flight_as_local_landing(mission):
    mission.record_plan=build_static_plan([-2.,0.,.1],[37.,13.,8.],
        controller_profile='tailsitter20cm_reference',max_speed=9.,max_acceleration=3.5,max_jerk=12.,
        target_is_virtual_waypoint=True,target_is_pass_through_gate=True,standoff=0.,body_radius=.16)
    mission.plan=None;mission.phase='coordinate_ready';mission.checkpoint=dict(outcome='passed')
    mission.events=[dict(stamp=30.,phase='arrived')]
    type(mission).save(mission)
    report=json.loads((mission.output/'report.json').read_text())
    assert report['checkpoint']['outcome']=='passed'
    assert report['success_behavior']=='hold_behind_target'
    assert report['failure_behavior']=='return_to_takeoff_then_land'


def test_mavros_duplicate_launch_does_not_restart_arming_or_clear_flight(mavros_mission):
    mission = mavros_mission
    assert launch(mission).success is True
    mission.samples = [dict(stamp=50., phase='arming')]
    reply = launch(mission)
    assert reply.success is False and 'duplicate_launch' in reply.message
    assert mission.phase == 'arming' and mission.start is None
    assert len(mission.start_requests) == 1 and len(mission.samples) == 1


@pytest.mark.parametrize('service_error', [False, True])
def test_mavros_rejected_start_requests_stop_and_never_reports_arrival(mavros_mission, service_error):
    mission = mavros_mission
    assert launch(mission).success is True
    if service_error:mission.start_reply.set_exception(RuntimeError('FCU unavailable'))
    else:mission.start_reply.set_result(SimpleNamespace(success=False))
    mission.tick()
    assert mission.phase == 'autopilot_landing' and mission.reason == 'PX4_start_rejected'
    assert len(mission.stop_requests) == 1 and mission.commands == []
    assert mission.start is None and mission.samples == []
    assert not any(event.get('phase') == 'arrived' for event in mission.events)


def test_mavros_abort_during_arming_requires_landed_and_disarmed_confirmation(mavros_mission):
    mission = mavros_mission
    assert launch(mission).success is True
    assert mission.action('abort', SimpleNamespace(success=None, message='')).success is True
    assert mission.phase == 'autopilot_landing' and len(mission.stop_requests) == 1
    assert not launch(mission).success
    mission.controller.update(armed=True, mode='AUTO.LAND', landed_state=1,
                              adapter_phase='landing')
    mission.tick()
    assert mission.phase == 'autopilot_landing' and mission.commands == []
    mission.controller.update(armed=False, landed_state=2)
    advance(mission, 50.2); mission.tick()
    assert mission.phase == 'autopilot_landing'
    mission.controller['landed_state'] = 1
    advance(mission, 50.4); mission.tick()
    assert mission.phase == 'autopilot_landing'
    # A late PX4 arm response must drain before reporting grounded completion.
    mission.controller.update(adapter_phase='standby', pending_control_command=True)
    advance(mission, 50.6); mission.tick()
    assert mission.phase == 'autopilot_landing'
    mission.controller.pop('pending_control_command')
    advance(mission, 50.8); mission.tick()
    assert mission.phase == 'autopilot_landing'
    mission.controller['pending_control_command'] = False
    advance(mission, 51.); mission.tick()
    assert mission.phase == 'aborted' and mission.reason == 'user_abort'
    assert not any(event.get('phase') == 'arrived' for event in mission.events)


@pytest.mark.parametrize('armed,mode', [(False, 'OFFBOARD'), (True, 'AUTO.LAND')])
def test_mavros_flight_stops_commands_when_armed_offboard_feedback_is_lost(mavros_mission, armed, mode):
    mission = mavros_mission
    assert launch(mission).success
    mission.start_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(armed=True, mode='OFFBOARD')
    mission.tick()
    assert mission.phase == 'takeoff'
    mission.commands.clear()
    mission.controller.update(armed=armed, mode=mode)
    advance(mission, 50.2); mission.tick()
    assert mission.phase == 'autopilot_landing'
    assert len(mission.stop_requests) == 1 and mission.commands == []


def test_mavros_ned_reference_change_waits_for_actual_landing_before_aborting(mavros_mission):
    mission = mavros_mission
    assert launch(mission).success
    mission.controller['origin_world'] = [-4., -6., .12]
    mission.tick()
    assert mission.phase == 'autopilot_landing' and 'NED_reference_changed' in mission.reason
    assert len(mission.stop_requests) == 1 and mission.commands == []
    mission.controller.update(armed=True,mode='AUTO.LAND',adapter_phase='landing',landed_state=2)
    advance(mission,50.2);mission.tick()
    assert mission.phase=='autopilot_landing' and len(mission.stop_requests)==1
    mission.controller.update(armed=False,adapter_phase='standby',landed_state=1)
    advance(mission,50.4);mission.tick()
    assert mission.phase=='aborted' and 'NED_reference_changed' in mission.reason


def test_mavros_feedback_loss_keeps_landing_pending_until_fresh_landed_disarmed_confirmation(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    state=dict(mission.controller);mission.own_history.clear();mission.controller_history.clear()
    before=len(mission.commands);mission.tick()
    assert mission.phase=='autopilot_landing' and mission.reason=='own_or_controller_lost_watchdog_descent'
    assert mission.return_requests==[] and len(mission.stop_requests)==1 and len(mission.commands)==before
    started=mission.autopilot_land_start
    mission.now=lambda:51.;mission.tick()
    assert mission.phase=='autopilot_landing' and len(mission.stop_requests)==1
    assert mission.autopilot_land_start==started and mission.controller is None
    # An armed aircraft at Z2.87 must never be reported terminal/landed.
    state.update(stamp=51.,fcu_state_stamp=51.,armed=True,mode='AUTO.LAND',adapter_phase='landing',landed_state=2)
    mission.controller_history.append(state);mission.own_history.append(dict(stamp=51.,position=[-2.,0.,2.87],velocity=[0.,0.,0.]))
    mission.tick();assert mission.phase=='autopilot_landing'
    # A fresh mission packet with stale FCU state still cannot prove disarm.
    state.update(armed=False,adapter_phase='standby',landed_state=1,fcu_state_stamp=48.)
    mission.tick();assert mission.phase=='autopilot_landing'
    state['fcu_state_stamp']=51.;mission.tick()
    assert mission.phase=='aborted' and mission.reason=='own_or_controller_lost_watchdog_descent'
    assert len(mission.stop_requests)==1 and not any(e.get('phase')=='arrived' for e in mission.events)


def test_feedback_loss_cancels_pending_home_request_without_resetting_landing_deadline(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    mission.begin_landing('user_abort');started=mission.autopilot_land_start
    mission.own_history.clear();mission.now=lambda:50.5;mission.tick()
    assert mission.phase=='autopilot_landing' and mission.reason=='user_abort'
    assert mission.return_reply.cancelled() and mission.return_future is None
    assert mission.emergency_local_landing and len(mission.stop_requests)==1
    assert mission.autopilot_land_start==started


def test_mavros_keeps_vertical_endpoint_until_measured_3m_height_and_speed_settle(mavros_mission):
    mission = mavros_mission
    assert launch(mission).success
    mission.start_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(armed=True, mode='OFFBOARD')
    mission.tick()
    duration = mission.plan['segments'][0]['duration_s']
    lift = np.array(mission.plan['segments'][0]['control_points'][-1])
    mission.own['position'] = (lift-[0., 0., .15]).tolist()
    stamp = mission.start+duration+1.
    advance(mission, stamp); mission.tick()
    assert mission.phase == 'takeoff' and mission.transfer_start is None
    assert mission.plan_time(stamp) == duration
    assert np.allclose(mission.commands[-1]['position'], lift)
    assert np.allclose(mission.commands[-1]['velocity'], 0.)
    assert np.allclose(mission.commands[-1]['acceleration'], 0.)

    mission.own['position'] = lift.tolist()
    advance(mission, stamp+.1); mission.tick()
    assert mission.takeoff_settle_dwell == stamp+.1
    advance(mission, stamp+.3); mission.tick()
    assert mission.phase == 'takeoff'
    # Being at the altitude with excessive velocity resets the dwell.
    mission.own['velocity'] = [0., 0., .13]
    advance(mission, stamp+.4); mission.tick()
    assert mission.phase == 'takeoff' and mission.takeoff_settle_dwell is None
    mission.own['velocity'] = [0., 0., 0.]
    advance(mission, stamp+.5); mission.tick()
    advance(mission, stamp+.91); mission.tick()
    assert mission.phase == 'spline_tracking'
    assert mission.transfer_start == stamp+.91
    assert mission.events[-1]['reason'] == '3m_feedback_settled'
    assert mission.events[-1]['plan_time_s'] == duration
    assert mission.events[-1]['takeoff_settle_extra_s'] == pytest.approx(1.91)
    assert mission.plan_time(stamp+1.01) == pytest.approx(duration+.1)
    assert np.allclose(mission.commands[-1]['position'], lift)
    assert np.allclose(mission.commands[-1]['velocity'], 0.)


def reference_mission(mission):
    mission.aircraft_type='tailsitter20cm_reference'
    mission.physics_profile=static_flight.aircraft_profile(mission.aircraft_type,
        Path(__file__).resolve().parents[2]/'jol_sim/config/tailsitter_20cm_reference.json')
    mission.reference_flight=mission.physics_profile['flight_parameters']
    mission.options['max_acceleration']=mission.reference_flight['max_acceleration']
    mission.controller.update(vtol_state=3,vtol_requested_state=None,pending_control_command=False)
    mission.vtol_calls=[];mission.vtol_reply=Future()
    def request(state):
        mission.vtol_calls.append(state)
        return mission.vtol_reply
    mission.fw_client=SimpleNamespace(service_is_ready=lambda:True,call_async=lambda req:request(4))
    mission.mc_client=SimpleNamespace(service_is_ready=lambda:True,call_async=lambda req:request(3))
    return mission


def pass_through_mission(mission):
    """Real sensor-derived gate plan and asynchronous return service."""
    reference_mission(mission)
    mission.reference_flight=dict(mission.reference_flight,progress_control=False)
    mission.options.update(continuous_takeoff=True,takeoff_join_speed=1.,takeoff_min_duration=4.,
        target_is_virtual_waypoint=True,target_is_pass_through_gate=True,standoff=0.,
        controller_profile='tailsitter20cm_reference',max_speed=9.,max_acceleration=3.5,
        max_jerk=12.,body_radius=.16,gate_width=1.2,gate_height=1.2,gate_exit_distance=20.)
    mission.own.update(position=[-2.,0.,.1],velocity=[0.,0.,0.])
    mission.controller['origin_world']=[-2.,0.,.12]
    mission.return_requests=[];mission.return_reply=Future()
    def request_return(request):
        mission.return_requests.append(request)
        return mission.return_reply
    mission.return_client=SimpleNamespace(service_is_ready=lambda:True,call_async=request_return)
    mission.prepare_plan(50.)
    return mission


def armed_gate_mission(mission):
    pass_through_mission(mission)
    assert launch(mission).success
    mission.start_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(armed=True,mode='OFFBOARD')
    mission.tick()
    mission.phase='spline_tracking';mission.vtol_stage='fixed_wing';mission.fw_observed=True
    mission.controller['vtol_state']=4
    return mission


def gate_feedback_sample(mission,offset_time=0.,lateral_offset=0.):
    gate=mission.plan['gate'];time_s=gate['planned_crossing_time_s']+offset_time
    sample=static_flight.sample_plan(mission.plan,time_s)
    mission.own.update(position=(np.array(sample['position'])+
        lateral_offset*np.array(gate['right_world'])).tolist(),velocity=sample['velocity'])
    advance(mission,mission.start+time_s)
    return sample


def test_gate_pass_does_not_claim_arrival_or_brake_before_plane(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    mission.reference_flight=dict(mission.reference_flight,back_transition_after_gate_when_slow=False)
    gate_feedback_sample(mission,-.05);mission.tick()
    assert mission.checkpoint['outcome']=='pending' and mission.vtol_calls==[]
    gate_feedback_sample(mission,.05);mission.tick()
    assert mission.checkpoint['outcome']=='passed'
    assert mission.checkpoint['source']=='mavros_px4_ekf_segment_plane_intersection'
    assert mission.phase=='spline_tracking' and mission.vtol_stage=='fixed_wing'
    assert mission.vtol_calls==[] and mission.return_requests==[] and mission.stop_requests==[]
    assert np.linalg.norm(mission.commands[-1]['velocity'])>1.
    assert not any(event.get('phase')=='arrived' for event in mission.events)
    # Only the subsequent stopping region behind a passed gate may request MC.
    normal=np.array(mission.plan['gate']['normal_world'])
    # Stay inside the trigger region while retaining the configured jerk and
    # transition stopping reserve; a longer transition must not waive it.
    duration=mission.reference_flight['back_transition_duration_s']
    remaining=5*duration+25/(2*mission.options['max_acceleration'])+1.5
    mission.update_vtol(np.array(mission.plan['goal'])-remaining*normal,5*normal,mission.now())
    assert mission.vtol_calls==[3] and mission.vtol_stage=='transition_to_mc'


def test_slow_after_gate_transitions_before_reserve_is_consumed(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    mission.reference_flight=dict(mission.reference_flight,back_transition_after_gate_when_slow=True)
    normal=np.asarray(mission.plan['gate']['normal_world']);goal=np.asarray(mission.plan['goal'])
    mission.checkpoint=dict(outcome='pending')
    mission.update_vtol(goal-20.*normal,7.2*normal,50.)
    assert mission.vtol_calls==[]
    mission.checkpoint=dict(outcome='passed')
    mission.update_vtol(goal-20.*normal,7.2*normal,50.1)
    assert mission.vtol_calls==[3] and mission.vtol_stage=='transition_to_mc'
    assert mission.events[-2]['back_transition_trigger']=='passed_gate_and_entry_speed_ready'


def test_gate_back_transition_waits_for_measured_entry_speed_then_requests_mc(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    mission.reference_flight.update(back_entry_ground_speed_m_s=7.2,back_transition_duration_s=.55)
    mission.checkpoint=dict(outcome='passed')
    normal=np.asarray(mission.plan['gate']['normal_world']);goal=np.asarray(mission.plan['goal'])
    mission.update_vtol(goal-16.5*normal,8.4*normal,50.)
    assert mission.vtol_calls==[] and mission.vtol_stage=='fixed_wing' and mission.return_requests==[]
    assert mission.events[-1]['back_transition_guard']=='waiting_for_measured_speed'
    mission.update_vtol(goal-13.*normal,7.2*normal,50.1)
    assert mission.vtol_calls==[3] and mission.vtol_stage=='transition_to_mc'
    assert mission.events[-2]['entry_speed_limit_m_s']==7.2
    assert mission.events[-2]['speed_m_s']==pytest.approx(7.2)


def test_gate_back_transition_without_braking_reserve_returns_without_forcing_mc(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    mission.reference_flight.update(back_entry_ground_speed_m_s=7.2,back_transition_duration_s=.55)
    mission.checkpoint=dict(outcome='passed')
    normal=np.asarray(mission.plan['gate']['normal_world']);goal=np.asarray(mission.plan['goal'])
    mission.update_vtol(goal-16.*normal,8.4*normal,50.)
    assert mission.vtol_calls==[] and len(mission.return_requests)==1
    assert mission.phase=='autopilot_landing' and mission.reason=='back_transition_entry_speed_or_distance_infeasible'
    assert mission.events[-3]['back_transition_guard']=='infeasible'


def test_missed_gate_requests_return_and_stops_outbound_commands(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    gate_feedback_sample(mission,-.05,lateral_offset=.6);mission.tick()
    mission.commands.clear()
    gate_feedback_sample(mission,.05,lateral_offset=.6);mission.tick()
    assert mission.checkpoint['outcome']=='missed'
    assert mission.phase=='autopilot_landing' and mission.reason=='checkpoint_missed_return_to_takeoff'
    assert len(mission.return_requests)==1 and mission.stop_requests==[] and mission.commands==[]
    assert any(event.get('action')=='return_to_takeoff' for event in mission.events)
    assert not launch(mission).success
    assert not mission.action('abort',SimpleNamespace(success=None,message='')).success
    assert len(mission.return_requests)==1


@pytest.mark.parametrize('reason',['tracking_error_exceeded','arrival_timeout','user_abort'])
def test_gate_normal_flight_failures_return_to_takeoff(mavros_mission,reason):
    mission=armed_gate_mission(mavros_mission)
    mission.begin_landing(reason)
    assert mission.phase=='autopilot_landing' and mission.reason==reason
    assert len(mission.return_requests)==1 and mission.stop_requests==[]
    assert mission.return_future is mission.return_reply


def test_selected_reference_plan_checks_dynamics_and_retains_comparison_diagnostics(mavros_mission):
    mission=pass_through_mission(mavros_mission)
    assert mission.plan['dynamics_validation']['passed']
    alternate=json.loads((mission.output/'alternate_plan.json').read_text())
    assert 'checks' in alternate['dynamics_validation']
    mission.physics_profile=json.loads(json.dumps(mission.physics_profile))
    mission.physics_profile['mass_kg']*=10.
    with pytest.raises(ValueError,match='tailsitter_dynamics_'):mission.prepare_plan(50.)


def test_calibration_change_invalidates_preview_until_explicit_unlock_and_relock(mission):
    mission.locked['calibration_revision']='revision_a'
    changed=json.loads(json.dumps(mission.locked));changed['calibration_revision']='revision_b'
    mission.estimates.append(changed)
    assert mission.launch_block_reason(50.)=='calibration_changed_unlock_and_relock_required'
    mission.tick()
    assert mission.phase=='locked' and mission.plan is None and mission.lock_revision_changed
    assert not mission.action('lock',SimpleNamespace()).success
    # A transient return to the old revision cannot silently re-enable launch.
    mission.estimates.append(dict(changed,calibration_revision='revision_a'))
    assert not mission.coordinate_matches(50.)
    assert not mission.action('preview',SimpleNamespace()).success
    mission.estimates.append(changed)
    assert mission.action('unlock',SimpleNamespace()).success
    assert mission.action('lock',SimpleNamespace()).success
    assert mission.locked['calibration_revision']=='revision_b' and mission.phase=='preview_ready'


@pytest.mark.parametrize('phase',['takeoff','spline_tracking','arrived'])
def test_airborne_revision_change_returns_immediately_including_post_gate_hold(mavros_mission,phase):
    mission=armed_gate_mission(mavros_mission);mission.phase=phase
    mission.locked['calibration_revision']='revision_a'
    mission.estimates.append(dict(mission.locked,calibration_revision='revision_b'))
    before=len(mission.commands);mission.tick()
    assert mission.phase=='autopilot_landing' and mission.reason=='calibration_changed_return_to_takeoff'
    assert len(mission.return_requests)==1 and mission.stop_requests==[] and len(mission.commands)==before


def test_lost_hold_coordinate_has_one_second_grace_then_returns_without_fake_coordinate(mavros_mission):
    mission=armed_gate_mission(mavros_mission);mission.phase='arrived'
    mission.own['position']=mission.plan['goal'];mission.own['velocity']=[0.,0.,0.]
    mission.estimates.clear();mission.tick()
    assert mission.phase=='arrived' and mission.sensor_unavailable_since==50.
    advance(mission,50.9);mission.tick()
    assert mission.phase=='arrived' and not mission.return_requests
    advance(mission,51.01);before=len(mission.commands);mission.tick()
    assert mission.phase=='autopilot_landing' and mission.reason=='sensor_coordinate_unavailable'
    assert len(mission.return_requests)==1 and len(mission.commands)==before


def test_coordinate_recovery_resets_missing_grace_and_revision_loss_is_not_accepted(mavros_mission):
    mission=armed_gate_mission(mavros_mission);mission.phase='arrived'
    mission.own['position']=mission.plan['goal'];mission.own['velocity']=[0.,0.,0.]
    mission.locked['calibration_revision']='revision_a'
    estimate=dict(mission.locked)
    mission.estimates.clear();mission.tick()
    advance(mission,50.8);mission.estimates.append(dict(estimate,stamp=50.8));mission.tick()
    assert mission.phase=='arrived' and mission.sensor_unavailable_since is None
    # Missing calibration identity cannot silently match the locked revision.
    advance(mission,51.);mission.estimates.append(dict(estimate,stamp=51.,calibration_revision=None));mission.tick()
    assert not mission.coordinate_matches(51.) and mission.sensor_unavailable_since==51.
    advance(mission,51.9);mission.tick();assert mission.phase=='arrived'
    advance(mission,52.01);mission.tick()
    assert mission.phase=='autopilot_landing' and mission.reason=='sensor_coordinate_unavailable'


def test_mode_loss_attaches_to_adapter_return_instead_of_requesting_local_land(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    mission.controller.update(mode='AUTO.LOITER',adapter_phase='returning',return_stage='reacquire_offboard')
    mission.tick()
    assert mission.phase=='autopilot_landing' and mission.reason=='PX4_armed_or_OFFBOARD_lost'
    assert len(mission.return_requests)==1 and mission.stop_requests==[]


def test_gate_arming_failure_uses_ground_stop_instead_of_airborne_return(mavros_mission):
    mission=pass_through_mission(mavros_mission)
    assert launch(mission).success
    mission.start_reply.set_result(SimpleNamespace(success=False))
    mission.tick()
    assert mission.phase=='autopilot_landing' and mission.reason=='PX4_start_rejected'
    assert len(mission.stop_requests)==1 and mission.return_requests==[]
    assert mission.start is None and mission.commands==[]


@pytest.mark.parametrize('service_error',[False,True])
def test_rejected_return_falls_back_once_and_never_reports_success(mavros_mission,service_error):
    mission=armed_gate_mission(mavros_mission)
    mission.begin_landing('checkpoint_missed_return_to_takeoff')
    if service_error:mission.return_reply.set_exception(RuntimeError('MAVROS unavailable'))
    else:mission.return_reply.set_result(SimpleNamespace(success=False))
    mission.tick()
    assert len(mission.stop_requests)==1 and mission.return_future is None
    assert mission.reason.endswith('return_rejected_emergency_local_landing')
    advance(mission,50.2);mission.tick()
    assert len(mission.stop_requests)==1 and mission.phase=='autopilot_landing'
    assert not any(event.get('phase')=='arrived' for event in mission.events)


def test_return_service_ack_does_not_finish_until_real_home_landing_and_disarm(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    mission.begin_landing('checkpoint_missed_return_to_takeoff')
    mission.return_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(adapter_phase='returning',return_stage='returning',landed_state=2)
    mission.tick()
    assert mission.phase=='autopilot_landing' and mission.return_future is None
    assert mission.stop_requests==[]
    mission.controller.update(adapter_phase='landing',return_stage='landing',return_home_reached=True,
        landed_state=1,armed=True)
    advance(mission,50.2);mission.tick()
    assert mission.phase=='autopilot_landing'
    mission.controller.update(adapter_phase='standby',return_stage='complete',armed=False,
        pending_control_command=True)
    advance(mission,50.4);mission.tick()
    assert mission.phase=='autopilot_landing'
    mission.controller['pending_control_command']=False
    advance(mission,50.6);mission.tick()
    assert mission.phase=='aborted' and mission.reason=='checkpoint_missed_return_to_takeoff'
    assert not any(event.get('phase')=='arrived' for event in mission.events)


def test_exact_gate_plane_sample_does_not_lose_the_previous_front_side(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    gate=mission.plan['gate'];center=np.array(gate['center_world']);normal=np.array(gate['normal_world'])
    for stamp,position in ((50.1,center-.1*normal),(50.2,center)):
        mission.own['position']=position.tolist();advance(mission,stamp);mission.check_gate(stamp)
        assert mission.checkpoint['outcome']=='pending'
    mission.own['position']=(center+.1*normal).tolist();advance(mission,50.3)
    mission.check_gate(50.3)
    assert mission.checkpoint['outcome']=='passed' and mission.phase=='spline_tracking'
    assert mission.return_requests==[] and mission.stop_requests==[]


def test_pending_return_service_ack_blocks_grounded_completion(mavros_mission):
    mission=armed_gate_mission(mavros_mission)
    mission.begin_landing('checkpoint_missed_return_to_takeoff')
    mission.controller.update(landed_state=1,armed=False,adapter_phase='standby',
        return_stage='complete',pending_control_command=False)
    mission.tick()
    assert mission.phase=='autopilot_landing' and not mission.return_reply.done()
    mission.return_reply.set_result(SimpleNamespace(success=True))
    advance(mission,50.2);mission.tick()
    assert mission.phase=='aborted' and mission.return_future is None


def test_reference_mission_back_transition_uses_measured_stopping_distance(mavros_mission):
    mission=reference_mission(mavros_mission)
    mission.vtol_stage='fixed_wing';mission.controller['vtol_state']=4
    goal=np.array(mission.plan['goal'])
    mission.update_vtol(goal-[25.,0.,0.],np.array([8.,0.,0.]),50.)
    assert mission.vtol_calls==[]
    mission.update_vtol(goal-[10.,0.,0.],np.array([8.,0.,0.]),50.1)
    assert mission.vtol_calls==[3] and mission.vtol_stage=='transition_to_mc'
    mission.vtol_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(vtol_state=3,pending_control_command=True)
    mission.update_vtol(goal,np.zeros(3),50.2)
    assert mission.vtol_stage=='transition_to_mc'
    mission.controller['pending_control_command']=False
    mission.update_vtol(goal,np.zeros(3),50.3)
    assert mission.vtol_stage=='multicopter'


def test_reference_mission_accelerates_in_mc_before_fw_and_checks_available_distance(mavros_mission):
    mission=reference_mission(mavros_mission);mission.vtol_stage='accelerating_for_fw'
    goal=np.array(mission.plan['goal']);far=goal-[30.,0.,0.]
    mission.update_vtol(far,np.zeros(3),50.,[6.,0.,0.])
    assert mission.vtol_calls==[] and mission.vtol_stage=='accelerating_for_fw'
    mission.update_vtol(far,np.array([6.,0.,0.]),50.1,[3.,0.,0.])
    assert mission.vtol_calls==[]
    mission.update_vtol(far,np.array([6.,0.,0.]),50.2,[6.,0.,0.])
    assert mission.vtol_calls==[4] and mission.vtol_stage=='transition_to_fw'
    assert not mission.fw_observed
    mission.controller['vtol_state']=4
    mission.update_vtol(far,np.array([6.,0.,0.]),50.3,[6.,0.,0.])
    assert mission.vtol_stage=='fixed_wing' and mission.fw_observed
    mission.vtol_stage='accelerating_for_fw';mission.vtol_calls.clear()
    mission.controller['vtol_state']=3;mission.fw_observed=False
    mission.update_vtol(goal-[10.,0.,0.],np.array([6.,0.,0.]),50.4,[6.,0.,0.])
    assert mission.vtol_calls==[] and mission.vtol_stage=='skipped_no_transition_space'


def test_reference_mission_measured_overspeed_brakes_even_with_distant_goal(mavros_mission):
    mission=reference_mission(mavros_mission)
    mission.vtol_stage='fixed_wing';mission.controller['vtol_state']=4
    speed=mission.reference_flight['max_speed']+.1
    mission.update_vtol(np.array(mission.plan['goal'])-[60.,0.,0.],np.array([speed,0.,0.]),50.)
    assert mission.vtol_calls==[3]
    assert mission.events[-2]['back_transition_trigger']=='measured_speed_limit'


def test_reference_mission_never_reports_fw_arrival_when_transition_was_skipped(mavros_mission):
    mission=reference_mission(mavros_mission)
    assert launch(mission).success
    mission.start_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(armed=True,mode='OFFBOARD')
    mission.tick()
    mission.phase='spline_tracking';mission.vtol_stage='skipped_no_transition_space'
    mission.transfer_start=50.-mission.plan['duration_s']
    mission.own['position']=mission.plan['goal']
    mission.tick()
    assert mission.phase=='autopilot_landing'
    assert mission.reason=='FW_transition_not_observed_skipped_no_transition_space'
    assert not any(event.get('phase')=='arrived' for event in mission.events)


def test_reference_front_transition_enters_early_but_ack_is_not_fw_observation(mavros_mission):
    mission=reference_mission(mavros_mission);mission.vtol_stage='accelerating_for_fw'
    far=np.array(mission.plan['goal'])-[30.,0.,0.]
    entry=mission.reference_flight['front_entry_ground_speed_m_s']
    planned=mission.reference_flight['front_entry_planned_speed_m_s']+.1
    mission.update_vtol(far,np.array([entry-.1,0.,0.]),50.,[planned,0.,0.])
    assert mission.vtol_calls==[]
    mission.update_vtol(far,np.array([entry+.1,0.,0.]),50.1,[planned,0.,0.])
    assert mission.vtol_calls==[4] and not mission.fw_observed
    mission.vtol_reply.set_result(SimpleNamespace(success=True))
    mission.update_vtol(far,np.array([entry+.2,0.,0.]),50.2,[planned,0.,0.])
    assert mission.vtol_stage=='transition_to_fw' and not mission.fw_observed


def test_mission_honors_adapter_not_ready_even_with_fresh_odometry(mavros_mission):
    mission=mavros_mission
    mission.controller['controller_ready']=False
    assert not mission.ready(50.) and not mission.can_launch(50.)
    assert mission.launch_block_reason(50.)=='NED_controller_or_own_odometry_not_ready'


def test_continuous_virtual_mission_hands_off_at_3m_with_nonzero_velocity(mavros_mission):
    mission=reference_mission(mavros_mission)
    mission.options.update(continuous_takeoff=True,takeoff_join_speed=1.,takeoff_min_duration=4.,
        target_is_virtual_waypoint=True,standoff=0.,controller_profile='tailsitter20cm_reference',
        max_speed=9.,max_acceleration=3.5,max_jerk=12.,body_radius=.16)
    mission.own.update(position=[-2.,0.,.1],velocity=[0.,0.,0.])
    mission.prepare_plan(50.)
    np.testing.assert_allclose(mission.plan['goal'],mission.locked['position'])
    assert launch(mission).success
    mission.start_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(armed=True,mode='OFFBOARD')
    mission.tick()
    advance(mission,54.05)
    mission.own.update(position=mission.plan['geometry']['lift'],velocity=[0.,0.,1.])
    mission.tick()
    assert mission.phase=='spline_tracking'
    assert mission.commands[-1]['velocity'][2]>.8
    event=next(e for e in mission.events if e.get('reason')=='3m_continuous_vertical_handoff')
    assert event['stopped_at_join'] is False and event['measured_join_velocity_world'][2]==1.
    advance(mission,54.1)
    mission.own['position']=np.array(mission.own['position'])+[0.,0.,.05]
    mission.tick()
    assert mission.phase=='spline_tracking' and mission.commands[-1]['velocity'][2]>.8


def test_published_join_guidance_ramps_lead_and_caps_initial_projection_advance(mavros_mission):
    mission=pass_through_mission(mavros_mission)
    mission.reference_flight=dict(mission.reference_flight,progress_control=True,path_lookahead_s=.2)
    mission.prepare_plan(50.)
    assert launch(mission).success
    mission.start_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(armed=True,mode='OFFBOARD')
    mission.tick()
    join=54.05
    advance(mission,join)
    mission.estimates[-1]['stamp']=join-.01
    mission.own.update(position=mission.plan['geometry']['lift'],velocity=[0.,0.,1.])
    mission.tick()
    assert mission.phase=='spline_tracking'
    previous=np.array(mission.commands[-1]['position'])
    previous_clock=mission.reference_history[-1][1]
    for index in range(1,21):
        age=index*.05
        advance(mission,join+age)
        mission.estimates[-1]['stamp']=join+age-.01
        # An EKF observation ahead of the command must not make the nominal
        # trajectory jump several samples during the handoff ramp.
        mission.own['position']=(np.array(mission.plan['geometry']['lift'])+[0.,0.,age+.15]).tolist()
        mission.tick()
        assert mission.phase=='spline_tracking'
        sample=mission.commands[-1]
        clock=mission.reference_history[-1][1]
        assert 0. <= clock-previous_clock < .072
        assert np.linalg.norm(np.array(sample['position'])-previous) < .08
        assert .9 < np.linalg.norm(sample['velocity']) < 1.5
        assert np.linalg.norm(sample['acceleration']) < 1.
        if index==1:
            assert clock-previous_clock < .051  # Old fixed lead advanced ~.25s.
            assert np.linalg.norm(np.array(sample['position'])-previous) < .052
        previous=np.array(sample['position']);previous_clock=clock


def test_reference_mission_cannot_arrive_before_actual_mc_and_drained_transition(mavros_mission):
    mission=reference_mission(mavros_mission)
    assert launch(mission).success
    mission.start_reply.set_result(SimpleNamespace(success=True))
    mission.controller.update(armed=True,mode='OFFBOARD')
    mission.tick()
    mission.vtol_stage='transition_to_mc';mission.vtol_stage_start=50.
    mission.fw_observed=True
    mission.vtol_future=mission.vtol_reply
    mission.transfer_start=50.-mission.plan['duration_s']
    mission.own['position']=mission.plan['goal']
    mission.controller.update(vtol_state=4,pending_control_command=True)
    mission.phase='spline_tracking';mission.tick()
    assert mission.phase=='spline_tracking' and mission.dwell is None
    mission.vtol_reply.set_result(SimpleNamespace(success=True))
    mission.controller['vtol_state']=3
    advance(mission,50.2);mission.tick()
    assert mission.phase=='spline_tracking' and mission.dwell is None
    mission.controller['pending_control_command']=False
    advance(mission,50.4);mission.tick()
    assert mission.vtol_stage=='multicopter' and mission.phase=='spline_tracking'
    advance(mission,51.6);mission.tick()
    assert mission.phase=='arrived'


def test_failed_braking_during_tick_stops_commands_without_crashing(mission):
    mission.phase = 'spline_tracking'
    # General controller freshness passes .3s, but braking requires .1s.
    mission.controller['stamp'] = 49.8
    mission.tick()
    assert mission.phase == 'aborted'
    assert 'fresh_measured_acceleration_required' in mission.reason
    assert mission.commands == []
    status = json.loads(mission.status_messages[-1].data)
    assert status['phase'] == 'aborted' and 'tracking_error_m' not in status


def test_preview_and_launch_require_grounded_stationary_vehicle(mission):
    assert mission.grounded(50.) and mission.can_launch(50.)
    mission.own['position'] = [-5., -6., 3.1]
    # A preview matching the airborne pose must still not permit launch.
    mission.plan['home'] = mission.own['position']
    assert not mission.grounded(50.)
    assert mission.launch_block_reason(50.) == 'grounded_stationary_vehicle_required'
    with pytest.raises(ValueError, match='grounded_stationary_vehicle_required'):
        mission.prepare_plan(50.)
    mission.own['position'] = [-5., -6., .1]
    mission.own['velocity'] = [.2, 0., 0.]
    assert not mission.grounded(50.)
    assert mission.launch_block_reason(50.) == 'grounded_stationary_vehicle_required'


def test_locked_preview_recovers_after_temporary_ground_height_dip(mission):
    mission.phase='coordinate_ready';mission.locked=None;mission.plan=None
    mission.own['position'][2]=.04262146
    reply=mission.action('lock',SimpleNamespace())
    assert reply.success and mission.phase=='locked' and mission.plan is None
    assert mission.reason=='preview_failed: ground_reference_below_controller_floor'
    assert not mission.can_launch(50.) and mission.commands==[]
    mission.tick()
    assert mission.phase=='locked' and mission.plan is None
    # Fresh measured height re-enters the unchanged command envelope.
    advance(mission,50.1);mission.estimates[-1]['stamp']=50.09
    mission.own['position'][2]=.06262146
    mission.tick()
    assert mission.phase=='preview_ready' and mission.can_launch(50.1)
    assert mission.plan['home'][2]==pytest.approx(.06262146)
    assert mission.commands==[]  # A recovered preview never starts a flight.


def test_real_envelope_failure_is_not_automatically_retried(mission):
    mission.phase='locked';mission.plan=None
    mission.reason='preview_failed: path_outside_controller_envelope'
    calls=[];mission.prepare_plan=lambda now:calls.append(now)
    mission.tick()
    assert mission.phase=='locked' and calls==[] and not mission.can_launch(50.)


def test_braking_plan_outside_native_controller_envelope_is_rejected(mission):
    mission.own['position'] = [59.4, -6., 8.]
    mission.own['velocity'] = [1.3, 0., 0.]
    mission.controller.update(position=mission.own['position'], velocity=mission.own['velocity'])
    # Keep the sensed surface well beyond the stop so envelope is the failure.
    mission.plan['target_world'] = [150., -6., 8.]
    mission.plan['target_surface_safety'].update(direction_world=[1., 0., 0.],
        required_plane_separation_m=.8)
    mission.begin_landing('user_abort')
    assert mission.phase == 'aborted'
    assert 'path_outside_controller_envelope' in mission.reason
    assert mission.commands == []


def test_new_preview_archives_previous_flight_and_clears_arrival_report(mission):
    mission.save = type(mission).save.__get__(mission, type(mission))
    mission.flight_launched = True; mission.phase = 'coordinate_ready'
    mission.events = [dict(stamp=30., phase='arrived'), dict(stamp=40., phase='coordinate_ready')]
    mission.samples = [dict(stamp=30., own_stamp=29.99, phase='arrived')]
    mission.flown = [[-5., -6., 8.]]; mission.landing = dict(previous='landing')
    (mission.output/'landing_plan.json').write_text(json.dumps(mission.landing))
    (mission.output/'plan.json').write_text(json.dumps(dict(previous='plan')))
    mission.save()
    prior_telemetry = (mission.output/'telemetry.jsonl').read_text()
    assert json.loads((mission.output/'report.json').read_text())['arrived'] is True

    mission.prepare_plan(50.)  # Calls the real begin_new_recording and save.
    mission.save()
    archives = list((mission.output/'flights').iterdir())
    assert len(archives) == 1
    assert json.loads((archives[0]/'report.json').read_text())['arrived'] is True
    assert (archives[0]/'telemetry.jsonl').read_text() == prior_telemetry
    assert json.loads((archives[0]/'landing_plan.json').read_text()) == dict(previous='landing')
    report = json.loads((mission.output/'report.json').read_text())
    assert report['arrived'] is False and report['samples'] == 0
    assert report['phase'] == 'preview_ready'
    assert (mission.output/'telemetry.jsonl').read_text() == ''
    assert not (mission.output/'landing_plan.json').exists()
    assert mission.flight_launched is False and mission.start is None and mission.landing is None
    assert mission.published_paths[-1] == (mission.flown_pub, [])


def test_lock_waits_for_safe_surface_uncertainty_before_accepting_coordinate(mission):
    mission.locked = None; mission.plan = None; mission.phase = 'coordinate_ready'
    estimate = mission.estimates[-1]
    estimate['covariance'] = (np.eye(3)*.08).tolist()
    # General estimate validity passes, but body+clearance+2sigma exceeds 1m.
    assert mission.estimate(50.) is not None
    response = mission.action('lock', SimpleNamespace(success=None, message=''))
    assert response.success is False and mission.locked is None
    assert response.message == 'standoff_too_small_for_vehicle_and_sensor_uncertainty'
    status = json.loads(mission.status_messages[-1].data)
    assert status['coordinate_valid'] is True and status['can_lock'] is False

    estimate['covariance'] = (np.eye(3)*.02).tolist()
    assert mission.lock_block_reason(50.) == ''
    mission.publish_status(50.)
    assert json.loads(mission.status_messages[-1].data)['can_lock'] is True


def test_lock_freezes_recent_safe_observation_only_with_current_valid_alignment(mission):
    mission.locked = None; mission.plan = None; mission.phase = 'coordinate_ready'
    safe = json.loads(json.dumps(mission.estimates[-1]))
    safe['stamp'] = 49.95
    latest = json.loads(json.dumps(safe))
    latest.update(stamp=49.99, covariance=(np.eye(3)*.08).tolist())
    mission.estimates = deque([safe, latest])
    before = json.dumps([safe, latest], sort_keys=True)
    assert mission.lock_estimate(50.) is safe
    assert mission.lock_block_reason(50.) == ''
    assert json.dumps([safe, latest], sort_keys=True) == before

    for change_latest, field, value in (
        (False, 'stamp', 49.1),
        (True, 'valid', False),
        (True, 'evidence', dict(gimbal_centered=False)),
        (False, 'target_id', 2),
        (False, 'position', [38.2, 13.5, 8.]),
    ):
        candidate, current = json.loads(before)
        (current if change_latest else candidate)[field] = value
        mission.estimates = deque([candidate, current])
        with pytest.raises(ValueError):
            mission.lock_estimate(50.)
        assert mission.lock_block_reason(50.) != ''


def test_camera_only_mission_accepts_only_its_configured_coordinate_source(mission):
    mission.target_source='camera_monocular_size'
    assert mission.estimate(50.) is None
    camera=dict(mission.estimates[-1],source='camera_monocular_size')
    mission.estimates.append(camera)
    assert mission.estimate(50.) is camera
    assert mission.lock_estimate(50.) is camera
    assert mission.estimate(51.) is None
