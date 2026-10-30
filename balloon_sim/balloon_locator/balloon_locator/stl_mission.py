# SPDX-License-Identifier: Apache-2.0
"""STL-only static waypoint terminal policy on the preserved NUB mission.

Source: ros2_nub/src/jol_perception/jol_perception/static_flight.py
Source SHA256: 1a5a5425bbe202fcc9538ddcb7b9a96af1396134fce3a4694d3909bda1b0fe44
The upstream Mission class is nested inside main(), with no extension hook for
its sensor-loss watchdog. This local source snapshot changes that watchdog and
adds honest terminal-coordinate metadata; shared planning/control helpers remain
imports. Do not interpret the terminal policy as tracking a moving target.
"""
from collections import deque
import json
import math
from pathlib import Path
import shutil
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from jol_perception.observation_path import sample_plan
from jol_perception.static_braking import build_braking_landing, sample_braking_landing
from jol_perception.static_flight_core import build_static_plan, valid_estimate, flight_frame, to_ned, stationary_sample, validate_controller_envelope, validate_tailsitter_plan_dynamics, target_surface_safety, path_progress, checkpoint_crossing
from jol_perception.observer_flight import causal_odometry
from jol_perception.mavros_adapter import aircraft_profile


def terminal_static_hold_allowed(*, now, phase, aircraft_type, options, locked,
                                 plan, own, loss_since, loss_position, latest_estimate):
    """Use only an already locked, static point during bounded final occlusion.

    This does not make a missing camera measurement fresh. FCU readiness, NED
    reference, calibration and tracking-error watchdogs still run in tick().
    Loss must begin within the final metre; later proximity cannot erase an
    earlier sensor failure. Arrival permits a stationary hold at the locked goal.
    """
    try:
        if (aircraft_type != 'stl_pacer_quad' or phase not in ('spline_tracking', 'arrived')
                or options.get('target_is_virtual_waypoint') is not True
                or options.get('target_is_pass_through_gate') or not plan or plan.get('gate')
                or not locked or locked.get('valid') is not True
                or locked.get('source') != 'camera_monocular_size'
                or locked.get('frame_id') != 'world'
                or locked.get('evidence', {}).get('target_is_static') is not True
                or not own):
            return False
        for value in (now, loss_since, own['stamp']):
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
        if not 0 <= now-own['stamp'] <= .2 or not 0 <= now-loss_since:
            return False
        position, velocity, initial, goal, target = [np.asarray(v, dtype=float) for v in
            (own['position'], own['velocity'], loss_position, plan['goal'], locked['position'])]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in
               (position, velocity, initial, goal, target)):
            return False
        if np.linalg.norm(goal-target) > 1e-6 or np.linalg.norm(initial-goal) > 1.:
            return False
        if latest_estimate:
            stamp = latest_estimate.get('stamp')
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp > now:
                return False
            if now-stamp <= .8:
                # Any fresh valid coordinate is handled by coordinate_matches;
                # a different valid position/ID must not become "occlusion".
                if (latest_estimate.get('valid') is True
                        or latest_estimate.get('frame_id') != 'world'
                        or latest_estimate.get('source') != 'camera_monocular_size'
                        or latest_estimate.get('target_id', 1) != locked.get('target_id', 1)):
                    return False
                revision = latest_estimate.get('calibration_revision')
                if revision and revision != locked.get('calibration_revision'):
                    return False
                reason = str(latest_estimate.get('reason', '')).lower()
                if any(word in reason for word in
                       ('calib', 'frame', 'tf', 'clock', 'reset', 'reference', 'waiting', 'start')):
                    return False
        distance = float(np.linalg.norm(position-goal))
        if phase == 'arrived':
            return distance <= .5 and np.linalg.norm(velocity) <= .2
        return distance <= 1. and now-loss_since <= 15.
    except (KeyError, TypeError, ValueError):
        return False


def archive_records(output):
    """Preserve each flight before reusing the flat latest-flight output."""
    files = [output/name for name in ('plan.json', 'alternate_plan.json', 'source_snapshot.json',
             'landing_plan.json', 'return_plan.json', 'events.json', 'telemetry.jsonl', 'report.json') if (output/name).is_file()]
    if not files:return None
    (output/'flights').mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix='flight_', dir=output/'flights'))
    for path in files:shutil.copy2(path, archive/path.name)
    # Delete only after every copy succeeds. Unrelated user files stay put.
    for path in files:path.unlink()
    return archive


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from nav_msgs.msg import Odometry, Path as RosPath
    from geometry_msgs.msg import PoseStamped, Transform, Twist
    from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

    class Mission(Node):
        def __init__(self):
            super().__init__('jol_static_flight')
            if not self.get_parameter('use_sim_time').value: raise ValueError('simulation_time_required')
            self.output=Path(self.declare_parameter('output_dir','/tmp/jol_static_flight').value)
            self.output.mkdir(parents=True,exist_ok=True)
            archive_records(self.output)
            self.backend=self.declare_parameter('flight_backend','native').value
            self.target_source=self.declare_parameter('target_source','camera_bbox_lidar').value
            if self.backend not in ('native','mavros'):raise ValueError('invalid_flight_backend')
            self.aircraft_type=self.declare_parameter('aircraft_type','iris_proxy').value
            self.physics_profile=aircraft_profile(self.aircraft_type,self.declare_parameter('physics_profile_path','').value)
            self.reference_flight=self.physics_profile.get('flight_parameters',{})
            if self.reference_flight and self.backend!='mavros':raise ValueError('reference_tailsitter_requires_MAVROS')
            self.options={key:self.declare_parameter(key,value).value for key,value in dict(
                path_kind='spline',takeoff_height=3.,standoff=1.,vertical_handle=2.,approach_handle=8.,
                min_duration=0.,max_speed=1.5,max_acceleration=.6,max_jerk=1.2,
                body_radius=.39 if self.backend=='mavros' else .36,takeoff_min_duration=0.,
                continuous_takeoff=False,takeoff_join_speed=.8,target_is_virtual_waypoint=False,
                target_is_pass_through_gate=False,gate_width=1.2,gate_height=1.2,gate_exit_distance=20.).items()}
            if self.backend=='mavros' and self.options['takeoff_min_duration']==0.:
                self.options['takeoff_min_duration']=12.
            if self.reference_flight:
                for key in ('max_speed','max_acceleration','max_jerk','body_radius','takeoff_min_duration',
                            'continuous_takeoff','takeoff_join_speed','target_is_virtual_waypoint','standoff',
                            'target_is_pass_through_gate','gate_width','gate_height','gate_exit_distance',
                            'gate_crossing_speed','gate_crossing_tangential_acceleration'):
                    if key in self.reference_flight:self.options[key]=self.reference_flight[key]
                self.options['controller_profile']='tailsitter20cm_reference'
            self.scene=json.loads(Path(self.declare_parameter('scene_file','').value).read_text())
            self.estimates=deque(maxlen=30);self.own_history=deque(maxlen=50);self.controller_history=deque(maxlen=50)
            self.controller=None;self.own=None;self.locked=None;self.plan=None;self.landing=None
            self.record_plan=None
            self.phase='detecting';self.reason='waiting_for_sensor_coordinate';self.last_now=None
            self.start=None;self.dwell=None;self.events=[];self.samples=[];self.flown=[];self.last_save=-1.;self.saved_samples=0
            self.flight_launched=False
            self.arm_future=None;self.arm_requested=None;self.autopilot_land_start=None
            self.arm_retry_count=0;self.arm_retry_at=None
            self.transfer_start=None;self.takeoff_settle_dwell=None
            self.progress_time=0.;self.reference_history=deque(maxlen=4000)
            self.vtol_stage='multicopter';self.vtol_future=None;self.vtol_stage_start=None;self.fw_observed=False
            self.checkpoint=dict(outcome='pending');self.gate_feedback=None;self.return_future=None
            self.lock_revision_changed=False;self.sensor_unavailable_since=None
            self.terminal_loss_position=None;self.terminal_loss_eligible=None;self.coordinate_mode='unavailable'
            self.emergency_local_landing=False
            self.reference_future=None;self.reference_requested=None;self.reference_requested_revision=None;self.reference_retry_at=None
            self.start_client=self.create_client(Trigger,'/jol/mavros/start') if self.backend=='mavros' else None
            self.reference_client=self.create_client(Trigger,'/jol/mavros/prepare_reference') if self.backend=='mavros' else None
            self.stop_client=self.create_client(Trigger,'/jol/mavros/stop') if self.backend=='mavros' else None
            self.return_client=self.create_client(Trigger,'/jol/mavros/return_home') if self.backend=='mavros' else None
            self.fw_client=self.create_client(Trigger,'/jol/mavros/transition_fw') if self.reference_flight else None
            self.mc_client=self.create_client(Trigger,'/jol/mavros/transition_mc') if self.reference_flight else None
            self.command_pub=self.create_publisher(MultiDOFJointTrajectory,'/jol/sim_observer/setpoint',1)
            self.status_pub=self.create_publisher(String,'/jol/static_flight/status',10)
            self.plan_pub=self.create_publisher(String,'/jol/static_flight/plan',10)
            self.path_pub=self.create_publisher(RosPath,'/jol/observer/flight_path',1)
            self.flown_pub=self.create_publisher(RosPath,'/jol/observer/flown_path',1)
            self.create_subscription(String,'/jol/static_target/estimate',self.receive_estimate,10)
            self.create_subscription(String,'/jol/sim_observer/state',self.receive_controller,10)
            self.create_subscription(Odometry,'/jol/observer/odometry',self.receive_own,10)
            for action in ('lock','unlock','preview','launch','abort','reset'):
                self.create_service(Trigger,'/jol/static_flight/'+action,lambda req,res,action=action:self.action(action,res))
            self.create_timer(.05,self.tick)

        def now(self):return self.get_clock().now().nanoseconds*1e-9
        def receive_estimate(self,msg):
            try:
                packet=json.loads(msg.data)
                if not isinstance(packet,dict) or type(packet.get('stamp')) not in (int,float) or not math.isfinite(packet['stamp']):
                    raise ValueError('invalid_estimate_stamp')
                self.estimates.append(packet)
            except (ValueError,TypeError):self.estimates.clear()
        def receive_controller(self,msg):
            try:
                packet=json.loads(msg.data);flight_frame(packet)
                if (type(packet.get('stamp')) not in (int,float) or not math.isfinite(packet['stamp'])
                        or not all(math.isfinite(float(packet[key])) and float(packet[key])>0
                                   for key in ('horizontal_limit_m','height_limit_m'))):
                    raise ValueError('invalid_controller_status')
                self.controller_history.append(packet)
            except (ValueError,TypeError,KeyError):self.controller=None;self.controller_history.clear()
        def receive_own(self,msg):
            if msg.header.frame_id!='world' or msg.child_frame_id!='jol/observer_body':return
            p,q,v=msg.pose.pose.position,msg.pose.pose.orientation,msg.twist.twist.linear
            values=np.array([p.x,p.y,p.z,q.x,q.y,q.z,q.w,v.x,v.y,v.z])
            if not np.isfinite(values).all() or np.linalg.norm(values[3:7])<1e-6:return
            velocity=Rotation.from_quat(values[3:7]).apply(values[7:])
            self.own_history.append(dict(stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                position=values[:3].tolist(),velocity=velocity.tolist()))
        def estimate(self,now):
            latest=next((p for p in reversed(self.estimates) if 0<=now-p.get('stamp',-1)<=.8),{})
            return latest if valid_estimate(latest,now) and latest.get('source')==self.target_source else None
        def ready(self,now):
            self.controller=next((p for p in reversed(self.controller_history) if 0<=now-p['stamp']<=.3),None)
            return bool(self.controller and 0<=now-self.controller['stamp']<=.3
                        and self.controller.get('controller_ready') is True
                        and causal_odometry(self.own_history,now) is not None)
        def active(self):return self.phase in ('arming','takeoff','spline_tracking','arrived','abort_braking','abort_landing','autopilot_landing')
        def grounded(self,now):
            own=causal_odometry(self.own_history,now)
            return bool(own and own['position'][2]<=.25 and np.linalg.norm(own['velocity'])<=.15)
        def lock_estimate(self,now):
            latest=self.estimate(now)
            if latest is None:raise ValueError('no_valid_sensor_coordinate')
            if latest.get('evidence',{}).get('gimbal_centered') is not True:
                raise ValueError('camera_gimbal_alignment_required')
            own=causal_odometry(self.own_history,now)
            if own is None:raise ValueError('NED_controller_or_own_odometry_not_ready')
            reason='fresh_coordinate_for_locked_target_required'
            # Freeze an actual, still-fresh safe observation. A single noisy
            # newer covariance must not invalidate the UI's ready snapshot.
            for candidate in sorted(self.estimates,key=lambda item:item['stamp'],reverse=True):
                if (not valid_estimate(candidate,now)
                        or candidate.get('source')!=self.target_source
                        or candidate.get('target_id',1)!=latest.get('target_id',1)
                        or (candidate.get('calibration_revision') or None)!=(latest.get('calibration_revision') or None)
                        or candidate.get('evidence',{}).get('gimbal_centered') is not True
                        or np.linalg.norm(np.array(candidate['position'])-latest['position'])>=1.):continue
                try:
                    if not self.options.get('target_is_virtual_waypoint'):
                        target_surface_safety(own['position'],candidate['position'],candidate['covariance'],self.options['standoff'],self.options.get('body_radius',.36))
                except (ValueError,KeyError,TypeError) as error:reason=str(error);continue
                return candidate
            raise ValueError(reason)
        def lock_block_reason(self,now):
            if self.active():return 'flight_active_abort_before_unlock_or_relock'
            if self.locked and (getattr(self,'lock_revision_changed',False) or self.calibration_changed(now)):
                return 'calibration_changed_unlock_and_relock_required'
            estimate=self.estimate(now)
            if estimate is None:return 'no_valid_sensor_coordinate'
            if estimate.get('evidence',{}).get('gimbal_centered') is not True:return 'camera_gimbal_alignment_required'
            if not self.ready(now):return 'NED_controller_or_own_odometry_not_ready'
            if not self.grounded(now):return 'grounded_stationary_vehicle_required'
            try:
                self.lock_estimate(now)
            except (ValueError,KeyError,TypeError) as error:return str(error)
            return ''
        def coordinate_matches(self,now):
            estimate=self.estimate(now)
            return bool(not getattr(self,'lock_revision_changed',False) and estimate and self.locked
                        and (estimate.get('calibration_revision') or None)==(self.locked.get('calibration_revision') or None)
                        and estimate.get('target_id',1)==self.locked.get('target_id',1)
                        and np.linalg.norm(np.array(estimate['position'])-self.locked['position'])<1.)
        def calibration_changed(self,now):
            latest=next((packet for packet in reversed(self.estimates) if 0<=now-packet.get('stamp',-1)<=.8),{})
            revision=latest.get('calibration_revision')
            return bool(self.locked and revision and revision!=(self.locked.get('calibration_revision') or None))
        def can_launch(self,now):
            return not self.launch_block_reason(now)
        def launch_block_reason(self,now):
            own=causal_odometry(self.own_history,now)
            if getattr(self,'lock_revision_changed',False) or self.calibration_changed(now):
                return 'calibration_changed_unlock_and_relock_required'
            if self.phase!='preview_ready' or not self.plan:return self.reason or 'valid_preview_required'
            if not self.ready(now):return 'NED_controller_or_own_odometry_not_ready'
            if not self.grounded(now):return 'grounded_stationary_vehicle_required'
            if not self.coordinate_matches(now):return 'fresh_coordinate_for_locked_target_required'
            if self.estimate(now).get('evidence',{}).get('gimbal_centered') is not True:
                return 'camera_gimbal_alignment_required'
            if not own or np.linalg.norm(np.array(own['position'])-self.plan['home'])>=.25:
                return 'vehicle_moved_regenerate_preview'
            if self.backend=='mavros' and (not self.start_client.service_is_ready() or self.controller.get('armed')):
                return 'MAVROS_start_service_or_disarmed_vehicle_required'
            if self.options.get('target_is_pass_through_gate') and self.backend=='mavros' and not self.return_client.service_is_ready():
                return 'MAVROS_return_home_service_required'
            if self.reference_flight and (self.controller.get('vtol_state')!=3
                    or not self.fw_client.service_is_ready() or not self.mc_client.service_is_ready()):
                return 'MC_state_and_VTOL_transition_services_required'
            return ''
        def transition(self,phase,reason=''):
            self.phase=phase;self.reason=reason;self.dwell=None
            self.events.append(dict(stamp=self.now(),phase=phase,reason=reason))
            self.get_logger().info('Static flight '+phase+': '+reason)
            self.save()
        def prepare_plan(self,now):
            if getattr(self,'lock_revision_changed',False) or self.calibration_changed(now):
                raise ValueError('calibration_changed_unlock_and_relock_required')
            if not self.coordinate_matches(now):raise ValueError('lock_required_with_fresh_coordinate')
            if not self.ready(now):raise ValueError('controller_not_ready')
            if not self.grounded(now):raise ValueError('grounded_stationary_vehicle_required')
            home=self.own['position'];target=self.locked['position']
            if home[2] < .05:
                raise ValueError('ground_reference_below_controller_floor')
            options=dict(self.options,target_covariance=self.locked['covariance'])
            plan=build_static_plan(home,target,obstacles=self.scene['obstacles'],**options)
            alternate=build_static_plan(home,target,obstacles=self.scene['obstacles'],**dict(options,path_kind='right_angle' if self.options['path_kind']=='spline' else 'spline'))
            if self.reference_flight:
                validate_tailsitter_plan_dynamics(plan,self.physics_profile)
                validate_tailsitter_plan_dynamics(alternate,self.physics_profile,strict=False)
            origin,rotation=flight_frame(self.controller)
            for candidate in (plan,alternate):validate_controller_envelope(candidate,self.controller)
            plan.update(stamp=now,source=self.locked['source'],target_id=self.locked.get('target_id',1),
                reference_clock_policy=('measured_EKF_path_progress' if self.reference_flight.get('progress_control')
                                        else 'nominal_time_with_takeoff_settle'),
                goal_ned=to_ned(plan['goal'],self.controller).tolist(),
                ned_reference=dict(origin_world=origin.tolist(),rotation_world_to_ned=rotation.tolist(),
                                   north_yaw_world_rad=self.controller['north_yaw_world_rad']),
                comparison={p['path_kind']:{key:p[key] for key in ('duration_s','sampled_length_m','sampled_max_curvature_m_inv','motion_validation')} for p in (plan,alternate)})
            self.begin_new_recording()
            self.plan=plan;self.lock_frame=dict(origin_world=origin.tolist(),north_yaw_world_rad=self.controller['north_yaw_world_rad'])
            (self.output/'source_snapshot.json').write_text(json.dumps(dict(estimate=self.locked,own=self.own,controller=self.controller),indent=2))
            (self.output/'plan.json').write_text(json.dumps(plan,indent=2))
            (self.output/'alternate_plan.json').write_text(json.dumps(alternate,indent=2))
            self.publish_plan();self.transition('preview_ready')
        def retry_ground_reference_preview(self,now):
            # A temporary EKF ground-height dip must not latch a failed preview.
            # Other geometry/controller failures still require operator review.
            if (self.phase!='locked' or self.reason!='preview_failed: ground_reference_below_controller_floor'
                    or self.own is None or self.own['position'][2]<.05
                    or not self.ready(now) or not self.grounded(now) or not self.coordinate_matches(now)):
                return
            try:self.prepare_plan(now)
            except (ValueError,KeyError,TypeError) as error:
                self.invalidate_plan();self.transition('locked','preview_failed: '+str(error))
        def begin_preview(self,now):
            client=getattr(self,'reference_client',None)
            if self.backend!='mavros' or client is None:
                self.prepare_plan(now);return
            if getattr(self,'reference_future',None) is not None:return
            self.invalidate_plan();self.transition('locked','ground_reference_preparation_pending')
            if not self.coordinate_matches(now):raise ValueError('lock_required_with_fresh_coordinate')
            if not self.ready(now):raise ValueError('controller_not_ready')
            if not self.grounded(now) or self.controller.get('armed') is not False:
                raise ValueError('grounded_disarmed_vehicle_required_for_reference')
            if not client.service_is_ready():raise ValueError('MAVROS_prepare_reference_service_required')
            revision=self.controller.get('ground_reference_revision',0)
            if type(revision) is not int or revision<0:raise ValueError('invalid_ground_reference_revision')
            self.reference_requested=now;self.reference_requested_revision=revision;self.reference_retry_at=None
            self.reference_future=client.call_async(Trigger.Request())
        def advance_preview_reference(self,now):
            future=getattr(self,'reference_future',None)
            if future is None:return
            if self.phase!='locked':self.reference_future=None;return
            try:
                if now-self.reference_requested>5.:raise ValueError('ground_reference_preparation_timeout')
                retry_at=getattr(self,'reference_retry_at',None)
                if retry_at is not None:
                    if now<retry_at:return
                    if (not self.ready(now) or not self.grounded(now)
                            or self.controller.get('armed') is not False or not self.coordinate_matches(now)):return
                    if not self.reference_client.service_is_ready():
                        raise ValueError('MAVROS_prepare_reference_service_required')
                    self.reference_future=self.reference_client.call_async(Trigger.Request())
                    self.reference_retry_at=None;return
                if not future.done():return
                response=future.result()
                if response is None or not response.success:
                    if response is not None and response.message in ('stationary_reference_required','stationary_reference_warmup_required'):
                        self.reference_retry_at=now+.35;return
                    raise ValueError('ground_reference_preparation_rejected'+(': '+response.message if response else ''))
                if not self.ready(now):return
                revision=self.controller.get('ground_reference_revision')
                pose_stamp=self.controller.get('ground_reference_pose_stamp')
                if (type(revision) is not int or revision<=self.reference_requested_revision
                        or type(pose_stamp) not in (int,float) or not math.isfinite(pose_stamp)
                        or not 0<=pose_stamp<=self.controller['stamp']
                        or self.own is None or self.own['stamp']<=pose_stamp):return
                self.reference_future=None
                self.prepare_plan(now)
            except (ValueError,KeyError,TypeError,RuntimeError,OSError) as error:
                self.reference_future=None;self.invalidate_plan()
                self.transition('locked','preview_failed: '+str(error))
        def begin_new_recording(self):
            if not self.flight_launched:return
            self.save();archive_records(self.output)
            self.events=[];self.samples=[];self.flown=[];self.saved_samples=0
            self.start=None;self.landing=None;self.flight_launched=False
            self.record_plan=None
            self.checkpoint=dict(outcome='pending');self.gate_feedback=None;self.return_future=None
            self.publish_path(self.flown_pub,[])
        def action(self,action,response):
            now=self.now();self.own=causal_odometry(self.own_history,now)
            try:
                if action=='lock':
                    reason=self.lock_block_reason(now)
                    if reason:raise ValueError(reason)
                    estimate=self.lock_estimate(now)
                    self.begin_new_recording()
                    self.locked=json.loads(json.dumps(estimate));self.plan=None
                    self.lock_revision_changed=False;self.sensor_unavailable_since=None
                    self.transition('locked')
                    try:self.begin_preview(now)
                    except (ValueError,KeyError,TypeError) as error:
                        self.invalidate_plan();self.transition('locked','preview_failed: '+str(error))
                elif action=='unlock':
                    if self.active():raise ValueError('flight_active_abort_first')
                    self.locked=None;self.invalidate_plan();self.transition('coordinate_ready' if self.estimate(now) else 'detecting')
                    self.lock_revision_changed=False;self.sensor_unavailable_since=None
                elif action=='preview':
                    if self.active():raise ValueError('flight_active')
                    try:self.begin_preview(now)
                    except (ValueError,KeyError,TypeError) as error:
                        self.invalidate_plan();self.transition('locked','preview_failed: '+str(error));raise
                elif action=='launch':
                    if not self.can_launch(now):raise ValueError('launch_not_ready_or_duplicate_launch')
                    self.flight_launched=True;self.samples=[];self.flown=[];self.saved_samples=0
                    self.record_plan=self.plan
                    self.transfer_start=None;self.takeoff_settle_dwell=None
                    self.progress_time=0.;self.reference_history=deque(maxlen=4000)
                    self.vtol_stage='multicopter';self.vtol_future=None;self.vtol_stage_start=None;self.fw_observed=False
                    self.checkpoint=dict(outcome='pending');self.gate_feedback=None;self.return_future=None
                    self.sensor_unavailable_since=None
                    self.emergency_local_landing=False
                    (self.output/'telemetry.jsonl').write_text('')
                    if self.backend=='mavros':
                        self.arm_retry_count=0;self.arm_retry_at=None
                        self.arm_future=self.start_client.call_async(Trigger.Request());self.arm_requested=now
                        self.transition('arming','waiting_for_PX4_OFFBOARD_and_arming_ack')
                    else:self.start=now;self.transition('takeoff')
                elif action=='abort':
                    if self.phase not in ('arming','takeoff','spline_tracking','arrived'):raise ValueError('no_active_flight')
                    self.begin_landing('user_abort')
                elif action=='reset':
                    if self.active():raise ValueError('abort_and_land_before_reset')
                    if self.own is None or np.linalg.norm(self.own['velocity'])>.15 or self.own['position'][2]>.25:
                        raise ValueError('reset_requires_grounded_vehicle_no_pose_teleport')
                    self.locked=None;self.invalidate_plan();self.start=None;self.landing=None
                    self.lock_revision_changed=False;self.sensor_unavailable_since=None
                    self.transition('coordinate_ready' if self.estimate(now) else 'detecting')
                response.success=True;response.message=self.phase
            except (ValueError,KeyError,TypeError,OSError) as error:
                response.success=False;response.message=str(error)
            self.events.append(dict(stamp=now,action=action,success=response.success,message=response.message))
            self.publish_status(now);self.save();return response
        def invalidate_plan(self):
            self.plan=None;self.plan_pub.publish(String(data=json.dumps(dict(stamp=self.now(),valid=False))))
            self.publish_path(self.path_pub,[])
        def publish_path(self,publisher,positions):
            msg=RosPath();msg.header.frame_id='world';msg.header.stamp=self.get_clock().now().to_msg()
            for position in positions:
                pose=PoseStamped();pose.header=msg.header;pose.pose.orientation.w=1.
                pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=map(float,position);msg.poses.append(pose)
            publisher.publish(msg)
        def publish_plan(self):
            if not self.plan:return
            self.plan_pub.publish(String(data=json.dumps(dict(self.plan,stamp=self.now()))))
            self.publish_path(self.path_pub,[p['position'] for p in self.plan['samples']])
        def command(self,sample):
            msg=MultiDOFJointTrajectory();msg.header.frame_id='jol/flight_ned';msg.header.stamp=self.get_clock().now().to_msg()
            msg.joint_names=['observer_body'];point=MultiDOFJointTrajectoryPoint();transform=Transform();transform.rotation.w=1.
            transform.translation.x,transform.translation.y,transform.translation.z=map(float,to_ned(sample['position'],self.controller))
            v,a=Twist(),Twist()
            v.linear.x,v.linear.y,v.linear.z=map(float,to_ned(sample['velocity'],self.controller,True))
            a.linear.x,a.linear.y,a.linear.z=map(float,to_ned(sample['acceleration'],self.controller,True))
            point.transforms=[transform];point.velocities=[v];point.accelerations=[a];msg.points=[point]
            self.command_pub.publish(msg)
        def plan_time(self,stamp):
            elapsed=max(0.,stamp-self.start)
            first=self.plan['segments'][0]['duration_s']
            history=getattr(self,'reference_history',())
            if self.plan.get('reference_clock_policy')=='measured_EKF_path_progress' and history:
                return float(np.interp(stamp,[r[0] for r in history],[r[1] for r in history]))
            if self.plan.get('continuous_takeoff'):return elapsed
            if self.backend=='mavros':
                if self.transfer_start is not None and stamp>=self.transfer_start:
                    return first+stamp-self.transfer_start
                return min(first,elapsed)
            return elapsed
        def request_vtol_stage(self,state,now):
            client=self.fw_client if state==4 else self.mc_client
            if not client or not client.service_is_ready():
                self.begin_landing('VTOL_transition_service_unavailable');return
            self.vtol_future=client.call_async(Trigger.Request());self.vtol_stage_start=now
            self.vtol_stage='transition_to_fw' if state==4 else 'transition_to_mc'
            self.events.append(dict(stamp=now,vtol_stage=self.vtol_stage,vtol_requested_state=state))
        def update_vtol(self,position,velocity,now,planned_velocity=None):
            state=self.controller.get('vtol_state',0)
            if self.vtol_future and self.vtol_future.done():
                if self.vtol_future.exception() or not self.vtol_future.result().success:
                    self.begin_landing('VTOL_transition_request_rejected');return
            if 'VTOL_transition_rejected' in self.controller.get('reason','') or state==0:
                self.begin_landing('VTOL_observed_state_or_command_invalid');return
            speed=float(np.linalg.norm(velocity[:2]))
            remaining=float(np.linalg.norm((np.array(self.plan['goal'])-position)[:2]))
            if self.vtol_stage=='accelerating_for_fw':
                planned_speed=float(np.linalg.norm(np.asarray(planned_velocity)[:2])) if planned_velocity is not None else 0.
                minimum=self.reference_flight['fw_min_airspeed_m_s']
                # Front transition accelerates toward the measured airspeed
                # requirement. Waiting for that speed in MC mode caused the
                # timed curve to outrun the vehicle before transition began.
                entry=self.reference_flight.get('front_entry_ground_speed_m_s',.8*minimum)
                planned_entry=self.reference_flight.get('front_entry_planned_speed_m_s',.7*minimum)
                if speed>=entry and planned_speed>=planned_entry:
                    reserve=(speed*(self.reference_flight['back_transition_duration_s']+
                        self.reference_flight['front_transition_duration_s'])+
                        speed*speed/(2*self.options['max_acceleration'])+.3*minimum)
                    if remaining>reserve:self.request_vtol_stage(4,now)
                    else:
                        self.vtol_stage='skipped_no_transition_space';self.reason=self.vtol_stage
                        self.events.append(dict(stamp=now,vtol_stage=self.vtol_stage,
                            remaining_m=remaining,required_transition_reserve_m=reserve))
            if self.vtol_stage=='transition_to_fw':
                if state==4:
                    self.vtol_stage='fixed_wing';self.fw_observed=True
                    self.events.append(dict(stamp=now,vtol_stage='fixed_wing',vtol_state=4))
                elif now-self.vtol_stage_start>self.reference_flight['front_transition_duration_s']+8.:
                    self.begin_landing('VTOL_front_transition_timeout');return
            if self.vtol_stage in ('fixed_wing','transition_to_fw'):
                stopping=speed*self.reference_flight['back_transition_duration_s']+speed*speed/(2*self.options['max_acceleration'])+2.
                early_back=(self.vtol_stage=='fixed_wing'
                    and self.reference_flight.get('back_transition_after_gate_when_slow',False)
                    and self.plan.get('gate') and self.checkpoint['outcome']=='passed'
                    and speed<=self.reference_flight.get('back_entry_ground_speed_m_s',0.))
                # Transition dynamics can outrun even a speed-controlled plan;
                # retain the measured overspeed and remaining-distance guards.
                if self.plan.get('gate') and self.vtol_stage=='fixed_wing' and speed>=self.reference_flight['max_speed']:
                    self.begin_landing('measured_speed_limit_exceeded');return
                if early_back or (remaining<=stopping and (not self.plan.get('gate') or self.checkpoint['outcome']=='passed')) or (
                        not self.plan.get('gate') and self.vtol_stage=='fixed_wing'
                        and speed>=self.reference_flight['max_speed']):
                    entry_limit=self.reference_flight.get('back_entry_ground_speed_m_s')
                    if self.plan.get('gate') and entry_limit is not None:
                        acceleration=self.options['max_acceleration'];jerk=self.options['max_jerk']
                        braking_time=(2*math.sqrt(speed/jerk) if speed<=acceleration**2/jerk
                                      else speed/acceleration+acceleration/jerk)
                        reserve=(speed*self.reference_flight['back_transition_duration_s']
                                 +speed*braking_time/2.+self.options.get('body_radius',0.))
                        if remaining<=reserve:
                            self.events.append(dict(stamp=now,back_transition_guard='infeasible',speed_m_s=speed,
                                entry_speed_limit_m_s=entry_limit,remaining_m=remaining,required_braking_reserve_m=reserve))
                            self.begin_landing('back_transition_entry_speed_or_distance_infeasible');return
                        if speed>entry_limit:
                            self.events.append(dict(stamp=now,back_transition_guard='waiting_for_measured_speed',
                                speed_m_s=speed,entry_speed_limit_m_s=entry_limit,remaining_m=remaining,
                                required_braking_reserve_m=reserve))
                            return
                    self.events.append(dict(stamp=now,back_transition_trigger=(
                        'passed_gate_and_entry_speed_ready' if early_back else
                        'measured_speed_limit' if speed>=self.reference_flight['max_speed']
                        else 'measured_stopping_distance'),speed_m_s=speed,remaining_m=remaining,
                        entry_speed_limit_m_s=entry_limit))
                    self.request_vtol_stage(3,now)
            elif self.vtol_stage=='transition_to_mc':
                if (state==3 and not self.controller.get('pending_control_command',True)
                        and self.controller.get('vtol_requested_state') is None):
                    self.vtol_stage='multicopter'
                    self.events.append(dict(stamp=now,vtol_stage='multicopter',vtol_state=3))
                elif now-self.vtol_stage_start>self.reference_flight['back_transition_duration_s']+8.:
                    self.begin_landing('VTOL_back_transition_timeout')
        def begin_landing(self,reason):
            if self.backend=='mavros':
                returning=bool(self.plan and self.plan.get('gate') and self.start is not None
                    and self.controller and self.controller.get('armed') is True
                    and self.controller.get('connected',True) and self.own
                    and 0<=self.now()-self.own['stamp']<=.3
                    and self.return_client and self.return_client.service_is_ready())
                if returning:
                    self.return_future=self.return_client.call_async(Trigger.Request())
                    self.events.append(dict(stamp=self.now(),action='return_to_takeoff',reason=reason,
                        requested_home_world=self.plan['home']))
                else:self.autopilot_stop()
                self.autopilot_land_start=self.now()
                self.transition('autopilot_landing',reason);return
            if self.own is None:self.transition('aborted','own_odometry_missing_watchdog_descent');return
            now=self.now();state=self.controller or {}
            try:
                if (not 0<=now-state.get('stamp',-1)<=.1
                        or state.get('acceleration_source')!='native_velocity_finite_difference'):
                    raise ValueError('fresh_measured_acceleration_required')
                self.landing=build_braking_landing(state['position'],state['velocity'],state['acceleration_world'],
                    landing_z=max(.10,min(self.plan['home'][2],.15)),
                    max_speed=self.options['max_speed'],max_acceleration=self.options['max_acceleration'],
                    # Leave jerk authority for feedback and the attitude lag;
                    # using the controller's full .8 limit caused stop overshoot.
                    max_jerk=min(.35,self.options['max_jerk']),obstacles=self.scene['obstacles'],clearance=.75,
                    surface_boundary=dict(target_world=self.plan['target_world'],**self.plan['target_surface_safety']))
                validate_controller_envelope(self.landing,state)
            except (ValueError,TypeError,KeyError) as error:
                # Do not label an unsafe or discontinuous fallback as a planned
                # landing. Command watchdog still limits force/jerk and descends.
                self.transition('aborted','braking_plan_failed_watchdog_descent: '+str(error));return
            self.landing_start=now;self.landing['start_stamp']=now
            (self.output/'landing_plan.json').write_text(json.dumps(self.landing,indent=2))
            self.transition('abort_braking',reason)
        def autopilot_stop(self):
            if self.stop_client and self.stop_client.service_is_ready():self.stop_client.call_async(Trigger.Request())
        def begin_emergency_landing(self,reason):
            if self.backend!='mavros':
                self.autopilot_stop();self.transition('aborted',reason);return
            if getattr(self,'emergency_local_landing',False):return
            pending=self.return_future
            if pending is not None and not pending.done():pending.cancel()
            self.return_future=None;self.emergency_local_landing=True
            self.autopilot_stop()
            # Feedback loss cannot establish a home route. Keep any earlier
            # landing deadline and cause; wait for actual PX4 completion.
            cause=self.reason if self.phase=='autopilot_landing' and self.reason else reason
            if self.phase!='autopilot_landing' or self.autopilot_land_start is None:self.autopilot_land_start=self.now()
            self.events.append(dict(stamp=self.now(),action='emergency_local_landing',reason=reason,
                                    home_return_requested=False))
            self.transition('autopilot_landing',cause)
        def retry_transient_start(self,now,message):
            """Retry only an explicit pre-ARM readiness refusal for this launch.

            Never retry accepted requests, FCU command rejections or exceptions.
            The original 15s deadline is unchanged; these retries end after 3s.
            """
            state=self.controller or {}
            if (self.aircraft_type!='stl_pacer_quad'
                    or message!='fresh_landed_EKF_reference_and_prestream_required'
                    or self.arm_requested is None or not 0<=now-self.arm_requested<3.
                    or self.arm_retry_count>=3
                    or state.get('armed') is not False or state.get('fcu_system_status')!=3
                    or state.get('landed_state')!=1 or not self.ready(now) or not self.grounded(now)
                    or not self.start_client.service_is_ready()):return False
            if self.arm_retry_at is None:
                self.arm_retry_at=now+.35
            elif now>=self.arm_retry_at:
                try:self.arm_future=self.start_client.call_async(Trigger.Request())
                except Exception as error:
                    self.begin_landing('PX4_start_retry_service_error: '+str(error));return True
                self.arm_retry_count+=1;self.arm_retry_at=None
                self.events.append(dict(stamp=now,action='retry_start_prearm_readiness',
                    retry=self.arm_retry_count,reason=message))
            return True
        def autopilot_landing_complete(self,now):
            state=next((packet for packet in reversed(self.controller_history) if 0<=now-packet['stamp']<=.3),None)
            return bool(state and state.get('connected') is True
                and type(state.get('fcu_state_stamp')) in (int,float) and 0<=now-state['fcu_state_stamp']<=2.
                and state.get('landed_state')==1 and state.get('armed') is False
                and state.get('adapter_phase')=='standby' and self.return_future is None
                and state.get('pending_control_command',True) is False)
        def check_gate(self,now):
            gate=self.plan.get('gate') if self.plan else None
            if not gate or self.checkpoint['outcome']!='pending' or not self.own:return
            current=self.own
            previous=self.gate_feedback
            if previous and current['stamp']<=previous['stamp']:return
            on_plane=abs(float((np.asarray(current['position'])-gate['center_world'])@np.asarray(gate['normal_world'])))<1e-10
            if not on_plane or previous is None:
                self.gate_feedback=dict(stamp=current['stamp'],position=list(current['position']))
            if previous is None:return
            if not 0<current['stamp']-previous['stamp']<=.3:
                self.begin_landing('checkpoint_feedback_gap');return
            crossing=checkpoint_crossing(gate,previous['position'],current['position'])
            if not crossing['valid']:
                self.begin_landing('checkpoint_'+crossing['reason']);return
            if crossing['crossed']:
                stamp=previous['stamp']+crossing['fraction']*(current['stamp']-previous['stamp'])
                self.checkpoint=dict(crossing,outcome='passed' if crossing['passed'] else 'missed',stamp=stamp,
                    source='mavros_px4_ekf_segment_plane_intersection')
                self.events.append(dict(stamp=now,checkpoint=self.checkpoint))
                if crossing['missed']:self.begin_landing('checkpoint_missed_return_to_takeoff')
            elif crossing['current_plane_distance_m']>0:
                self.checkpoint=dict(outcome='unobserved',stamp=current['stamp'])
                self.begin_landing('checkpoint_crossing_unobserved_return_to_takeoff')
        def backend_metadata(self):
            state=self.controller or {}
            return dict(flight_backend='mavros_px4' if self.backend=='mavros' else 'native',
                own_pose_source=state.get('own_pose_source','mavros_px4_ekf' if self.backend=='mavros' else 'ideal_gazebo_self_odometry'),
                feedback_controller=state.get('feedback_controller','MAVROS_PX4_Gazebo_motor_dynamics' if self.backend=='mavros' else 'Gazebo_force_torque_NED_position_attitude_loop'),
                armed=state.get('armed',False),autopilot_mode=state.get('mode',''),
                autopilot_connected=state.get('connected',False),landed_state=state.get('landed_state'),
                controller_error=state.get('error',state.get('reason','')),
                return_stage=state.get('return_stage'),launch_home_world=state.get('launch_home_world'),
                return_home_error_m=state.get('return_home_error_m'),return_home_reached=state.get('return_home_reached',False),
                return_landed_at_home=state.get('return_landed_at_home'))
        def aircraft_metadata(self):
            state=self.controller or {}
            return dict(aircraft_type=self.aircraft_type,
                physics_data_status=self.physics_profile.get('data_status','Iris_proxy'),
                vtol_stage=self.vtol_stage,vtol_state=state.get('vtol_state'),fixed_wing_observed=self.fw_observed,
                attitude_quaternion_world=state.get('attitude_quaternion_world'),nose_axis_world=state.get('nose_axis_world'),
                airspeed_m_s=state.get('airspeed_m_s'))
        def publish_status(self,now,sample=None):
            estimate=self.estimate(now);target=self.locked or estimate
            lock_reason=self.lock_block_reason(now)
            matching=estimate if estimate and self.locked and estimate.get('target_id',1)==self.locked.get('target_id',1) and self.coordinate_matches(now) else None
            packet=dict(stamp=now,phase=self.phase,reason=self.reason,coordinate_valid=bool(estimate),
                coordinate_mode=self.coordinate_mode,
                locked=bool(self.locked),plan_ready=bool(self.plan),controller_ready=self.ready(now),
                can_lock=not self.locked and not lock_reason,lock_block_reason=lock_reason,
                can_launch=self.can_launch(now),launch_block_reason=self.launch_block_reason(now),
                camera_centered=bool(estimate and estimate.get('evidence',{}).get('gimbal_centered') is True),
                camera_alignment_error_deg=estimate.get('evidence',{}).get('camera_alignment_error_deg') if estimate else None,
                gimbal_tracking=bool(estimate and estimate.get('evidence',{}).get('gimbal_tracking') is True),
                target_id=target.get('target_id',1) if target else None,
                lock_bbox=matching.get('bbox') if matching else None,
                lock_bbox_stamp=matching['stamp'] if matching else None,
                detection_image_size=matching.get('evidence',{}).get('image_size') if matching else None,
                target_world=target['position'] if target else None,
                target_ned=to_ned(target['position'],self.controller).tolist() if target and self.controller else None,
                goal_world=self.plan['goal'] if self.plan else None,
                standoff_m=self.plan['standoff_m'] if self.plan else self.options['standoff'],
                target_is_virtual_waypoint=self.options.get('target_is_virtual_waypoint',False),
                takeoff_height=3.,target_source=self.target_source,simulation_only=True,
                checkpoint=self.checkpoint if self.plan and self.plan.get('gate') else None,
                **self.backend_metadata(),**self.aircraft_metadata())
            if self.own:
                p=np.array(self.own['position']);packet.update(position_world=p.tolist(),velocity_world=self.own['velocity'],
                    own_stamp=self.own['stamp'],odometry_age_s=now-self.own['stamp'],
                    takeoff_height_m=float(p[2]-self.plan['home'][2]) if self.plan else None)
                if sample and self.phase not in ('arming','autopilot_landing'):
                    reference=(sample_plan(self.plan,self.plan_time(self.own['stamp']))
                               if self.phase in ('takeoff','spline_tracking','arrived') else
                               sample_braking_landing(self.landing,max(0.,self.own['stamp']-self.landing_start)))
                    packet['tracking_error_m']=float(np.linalg.norm(p-reference['position']))
                if self.plan:packet['arrival_error_m']=float(np.linalg.norm(p-self.plan['goal']))
                if target:packet['target_surface_distance_m']=float(np.linalg.norm(p-target['position']))
            self.status_pub.publish(String(data=json.dumps(packet)))
        def save(self):
            record_plan=self.plan or getattr(self,'record_plan',None)
            (self.output/'events.json').write_text(json.dumps(self.events,indent=2))
            with (self.output/'telemetry.jsonl').open('a') as stream:
                for row in self.samples[self.saved_samples:]:stream.write(json.dumps(row)+'\n')
            self.saved_samples=len(self.samples)
            (self.output/'report.json').write_text(json.dumps(dict(phase=self.phase,reason=self.reason,arrived=any(e.get('phase')=='arrived' for e in self.events),
                checkpoint=self.checkpoint if record_plan and record_plan.get('gate') else None,
                success_behavior='hold_behind_target' if record_plan and record_plan.get('gate') else 'hold_at_goal',
                failure_behavior='return_to_takeoff_then_land' if record_plan and record_plan.get('gate') else 'land',
                simulation_only=True,target_source=self.target_source,**self.backend_metadata(),**self.aircraft_metadata(),
                native_setpoint_frame='jol/flight_ned',samples=len(self.samples)),indent=2))
        def tick(self):
            now=self.now()
            if now<=0:return
            if self.last_now is not None and now<self.last_now:
                self.estimates.clear();self.own_history.clear();self.controller_history.clear();self.controller=None;self.locked=None;self.invalidate_plan()
                self.transition('aborted','simulation_clock_reset')
            self.last_now=now;self.own=causal_odometry(self.own_history,now)
            self.advance_preview_reference(now)
            self.retry_ground_reference_preview(now)
            sample=None
            self.coordinate_mode='live' if self.estimate(now) else 'unavailable'
            if (self.locked and self.phase in ('locked','preview_ready')
                    and not getattr(self,'lock_revision_changed',False) and self.calibration_changed(now)):
                self.lock_revision_changed=True;self.invalidate_plan()
                self.transition('locked','calibration_changed_unlock_and_relock_required')
            if self.phase in ('takeoff','spline_tracking','arrived'):
                if self.calibration_changed(now):self.begin_landing('calibration_changed_return_to_takeoff')
                elif self.coordinate_matches(now):
                    self.sensor_unavailable_since=None;self.terminal_loss_position=None;self.terminal_loss_eligible=None
                else:
                    self.sensor_unavailable_since=getattr(self,'sensor_unavailable_since',None)
                    if self.sensor_unavailable_since is None:
                        self.sensor_unavailable_since=now
                        self.terminal_loss_position=self.own['position'] if self.own else None
                        self.terminal_loss_eligible=None
                    terminal=terminal_static_hold_allowed(now=now,phase=self.phase,
                        aircraft_type=self.aircraft_type,options=self.options,locked=self.locked,
                        plan=self.plan,own=self.own,loss_since=self.sensor_unavailable_since,
                        loss_position=self.terminal_loss_position,
                        latest_estimate=self.estimates[-1] if self.estimates else None)
                    # A fresh conflict or early loss permanently disqualifies
                    # this outage. It cannot become "occlusion" by aging away.
                    self.terminal_loss_eligible=terminal and self.terminal_loss_eligible is not False
                    if self.terminal_loss_eligible:self.coordinate_mode='locked_static_terminal'
                    elif now-self.sensor_unavailable_since>=1.:self.begin_landing('sensor_coordinate_unavailable')
            if self.phase in ('detecting','coordinate_ready'):
                self.phase='coordinate_ready' if self.estimate(now) else 'detecting'
                self.reason='' if self.phase=='coordinate_ready' else 'waiting_for_sensor_coordinate'
            if self.active():
                if not self.ready(now):
                    self.begin_emergency_landing('own_or_controller_lost_watchdog_descent')
                    if self.backend=='mavros':
                        if self.autopilot_landing_complete(now):self.transition('aborted',self.reason)
                        elif now-self.autopilot_land_start>90.:self.transition('aborted','PX4_landing_timeout_unverified')
                elif self.phase!='autopilot_landing' and (self.controller['origin_world']!=self.lock_frame['origin_world'] or abs(self.controller['north_yaw_world_rad']-self.lock_frame['north_yaw_world_rad'])>1e-6):
                    self.begin_emergency_landing('NED_reference_changed_watchdog_descent')
                elif (self.backend=='mavros' and self.phase in ('takeoff','spline_tracking','arrived')
                      and not (self.controller.get('armed') is True and self.controller.get('mode')=='OFFBOARD')):
                    self.begin_landing('PX4_armed_or_OFFBOARD_lost')
                else:
                    p=np.array(self.own['position']);v=np.array(self.own['velocity'])
                    if self.phase=='arming':
                        start_done=self.arm_future.done()
                        start_error=self.arm_future.exception() if start_done else None
                        start_reply=self.arm_future.result() if start_done and not start_error else None
                        if start_done and (start_error or start_reply is None or not start_reply.success):
                            message=str(start_error) if start_error else (
                                start_reply.message if start_reply is not None else 'empty_start_response')
                            if start_error or not self.retry_transient_start(now,message):
                                self.begin_landing('PX4_start_rejected: '+message)
                        elif (self.arm_future.done() and self.controller.get('armed') is True
                              and self.controller.get('mode')=='OFFBOARD'):
                            self.start=now;self.transition('takeoff','PX4_OFFBOARD_and_armed_confirmed')
                            sample=sample_plan(self.plan,0.)
                        elif now-self.arm_requested>15.:self.begin_landing('PX4_arming_timeout')
                    elif self.phase=='autopilot_landing':
                        if self.controller.get('return_stage'):
                            row=dict(stamp=now,own_stamp=self.own['stamp'],phase=self.phase,position=p.tolist(),velocity=v.tolist(),
                                **self.backend_metadata())
                            for field in ('position','velocity','acceleration'):
                                desired=self.controller.get('return_desired_'+field+'_world')
                                if desired is not None:row['desired_'+field]=desired
                            row['return_setpoint_stamp']=self.controller.get('return_setpoint_stamp')
                            self.samples.append(row)
                            if len(self.samples)%4==0:self.flown.append(p.tolist());self.publish_path(self.flown_pub,self.flown)
                            if self.controller.get('return_plan') and not (self.output/'return_plan.json').exists():
                                (self.output/'return_plan.json').write_text(json.dumps(self.controller['return_plan'],indent=2))
                        if self.return_future and self.return_future.done():
                            if self.return_future.exception() or not self.return_future.result().success:
                                self.autopilot_stop();self.reason+='; return_rejected_emergency_local_landing'
                            self.return_future=None
                        if self.autopilot_landing_complete(now):
                            self.transition('aborted',self.reason)
                        elif now-self.autopilot_land_start>(150. if self.controller.get('return_stage') else 90.):
                            self.autopilot_stop();self.transition('aborted','PX4_landing_timeout')
                    elif self.phase in ('takeoff','spline_tracking'):
                        if self.plan.get('gate'):self.check_gate(now)
                        if self.phase=='autopilot_landing':
                            self.publish_status(now);self.save();return
                        elapsed=self.plan_time(now);first=self.plan['segments'][0]
                        if self.plan.get('continuous_takeoff'):
                            if self.phase=='takeoff':elapsed=min(now-self.start,first['duration_s'])
                            elif self.reference_flight.get('progress_control'):
                                step=max(.01,now-self.reference_history[-1][0]) if self.reference_history else .05
                                # Introduce the guidance lead over one second;
                                # applying it at once jumps the position command
                                # despite the underlying curve's C2 join.
                                join_age=max(0.,now-self.transfer_start) if self.transfer_start is not None else 1.
                                ramp=min(1.,join_age)
                                lead_fraction=ramp**3*(10.-15.*ramp+6.*ramp**2)
                                progress=path_progress(self.plan,p,previous_time=self.progress_time,
                                    max_advance=step if join_age<=1. else 5*step)
                                self.progress_time=progress['trajectory_time_s']
                                elapsed=min(self.plan['duration_s'],self.progress_time+
                                    self.reference_flight.get('path_lookahead_s',.2)*lead_fraction)
                            self.reference_history.append((now,elapsed))
                        sample=sample_plan(self.plan,elapsed)
                        if elapsed>=first['duration_s'] and self.phase=='takeoff':
                            if self.plan.get('continuous_takeoff'):
                                # Keep the nonzero upward velocity at the join:
                                # no hover/settling dwell between the two spans.
                                lift=np.array(first['control_points'][-1])
                                if p[2]>=lift[2]:
                                    self.transfer_start=now;self.progress_time=first['duration_s']
                                    self.transition('spline_tracking','3m_continuous_vertical_handoff')
                                    self.events[-1].update(plan_time_s=first['duration_s'],
                                        measured_join_velocity_world=v.tolist(),stopped_at_join=False)
                                    if self.reference_flight:self.vtol_stage='accelerating_for_fw'
                                    self.save()
                                elif now-self.start>first['duration_s']+10.:
                                    self.begin_landing('continuous_3m_takeoff_timeout')
                            elif self.backend=='mavros':
                                # PX4's motor/takeoff ramp can lag a time-only
                                # plan. Hold the 3m endpoint until measured
                                # position AND speed settle, then start the
                                # zero-v/a cubic transfer clock.
                                lift=np.array(first['control_points'][-1])
                                sample=stationary_sample(lift)
                                if np.linalg.norm(p-lift)<.05 and np.linalg.norm(v)<.12:
                                    if self.takeoff_settle_dwell is None:self.takeoff_settle_dwell=now
                                    if now-self.takeoff_settle_dwell>=.4:
                                        self.transfer_start=now;self.transition('spline_tracking','3m_feedback_settled')
                                        self.events[-1].update(plan_time_s=first['duration_s'],
                                            takeoff_settle_extra_s=now-self.start-first['duration_s'])
                                        if self.reference_flight:self.vtol_stage='accelerating_for_fw'
                                        self.save()
                                else:self.takeoff_settle_dwell=None
                                if now-self.start>first['duration_s']+20.:self.begin_landing('3m_takeoff_settle_timeout')
                            else:self.transition('spline_tracking')
                        if self.reference_flight and self.phase=='spline_tracking':self.update_vtol(p,v,now,sample['velocity'])
                        if elapsed>=self.plan['duration_s'] and self.phase in ('takeoff','spline_tracking'):
                            if self.reference_flight and not self.fw_observed:
                                self.begin_landing('FW_transition_not_observed_'+self.vtol_stage)
                            elif (not self.plan.get('gate') or self.checkpoint['outcome']=='passed') and (
                                    np.linalg.norm(p-self.plan['goal'])<.25 and np.linalg.norm(v)<.2
                                    and (not self.reference_flight or self.controller.get('vtol_state')==3
                                        and self.vtol_stage=='multicopter'
                                        and not self.controller.get('pending_control_command',True))):
                                if self.dwell is None:self.dwell=now
                                if now-self.dwell>1.:self.transition('arrived')
                            else:self.dwell=None
                            if elapsed>self.plan['duration_s']+15 or now-self.start>self.plan['duration_s']+60:
                                self.begin_landing('arrival_timeout')
                        if self.phase in ('takeoff','spline_tracking') and np.linalg.norm(p-sample['position'])>2.:self.begin_landing('tracking_error_exceeded')
                        if self.plan.get('continuous_takeoff') and self.phase in ('takeoff','spline_tracking') and now-self.start>self.plan['duration_s']+60:
                            self.begin_landing('continuous_path_progress_timeout')
                        if self.phase in ('abort_braking','abort_landing'):sample=sample_braking_landing(self.landing,0.)
                        if self.phase=='autopilot_landing':sample=None
                    elif self.phase=='arrived':sample=stationary_sample(self.plan['goal'])
                    elif self.phase in ('abort_braking','abort_landing'):
                        elapsed=now-self.landing_start;sample=sample_braking_landing(self.landing,elapsed)
                        if sample['phase']=='abort_landing' and self.phase=='abort_braking':
                            self.transition('abort_landing',self.reason)
                        if elapsed>=self.landing['duration_s'] and p[2]<.19 and np.linalg.norm(v)<.15:
                            if self.dwell is None:self.dwell=now
                            if now-self.dwell>=1:self.transition('aborted',self.reason)
                        elif elapsed>self.landing['duration_s']+15:self.transition('aborted','landing_timeout')
                    if not self.active():sample=None
                    if sample:
                        self.command(sample)
                        reference=(sample_plan(self.plan,self.plan_time(self.own['stamp']))
                                   if self.phase in ('takeoff','spline_tracking','arrived') else
                                   sample_braking_landing(self.landing,max(0.,self.own['stamp']-self.landing_start)))
                        row=dict(stamp=now,own_stamp=self.own['stamp'],phase=self.phase,position=p.tolist(),velocity=v.tolist(),
                            trajectory_time_s=self.plan_time(now) if self.phase in ('takeoff','spline_tracking','arrived') else None,
                            reference_time_s=self.plan_time(now) if self.phase in ('takeoff','spline_tracking','arrived') else None,
                            desired_position=sample['position'],desired_velocity=sample['velocity'],desired_acceleration=sample['acceleration'],
                            desired_position_at_feedback_stamp=reference['position'],
                            command_ned=to_ned(sample['position'],self.controller).tolist(),
                            tracking_error_m=float(np.linalg.norm(p-reference['position'])),
                            coordinate_fresh=bool(self.estimate(now)),takeoff_height_m=float(p[2]-self.plan['home'][2]),
                            coordinate_mode=self.coordinate_mode,
                            **self.aircraft_metadata())
                        self.samples.append(row)
                        if len(self.samples)%4==0:self.flown.append(p.tolist());self.publish_path(self.flown_pub,self.flown)
            self.publish_status(now,sample)
            if now-self.last_save>2:
                self.save();self.publish_plan();self.last_save=now
    rclpy.init(args=args);node=Mission()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.save();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
