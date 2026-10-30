"""Gazebo-only PX4/MAVROS bridge. MAVROS converts ROS ENU setpoints to MAVLink NED."""
from collections import deque
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .observation_path import _segment, _finish_plan, sample_plan


def vector(values):
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('finite_XYZ_required')
    return result


def ned_to_enu(values):
    north, east, down = vector(values)
    return np.array([east, north, -down])


def aircraft_profile(aircraft_type, path=''):
    if aircraft_type not in ('iris_proxy','tailsitter20cm_reference'):
        raise ValueError('invalid_aircraft_type')
    if aircraft_type=='iris_proxy':return {}
    if not path:raise ValueError('reference_physics_profile_required')
    profile=json.loads(Path(path).read_text())
    flight=profile['flight_parameters']
    for key in ('max_speed','max_acceleration','max_jerk','front_transition_duration_s',
                'back_transition_duration_s','fw_min_airspeed_m_s','transition_airspeed_m_s'):
        if type(flight.get(key)) not in (int,float) or not math.isfinite(flight[key]) or flight[key]<=0:
            raise ValueError('invalid_reference_flight_parameter_'+key)
    for key in ('front_entry_ground_speed_m_s','front_entry_planned_speed_m_s','back_entry_ground_speed_m_s',
                'gate_crossing_speed'):
        if key in flight and (type(flight[key]) not in (int,float)
                or not math.isfinite(flight[key]) or not 0<flight[key]<=flight['max_speed']):
            raise ValueError('invalid_reference_flight_parameter_'+key)
    if 'gate_crossing_tangential_acceleration' in flight:
        value=flight['gate_crossing_tangential_acceleration']
        if type(value) not in (int,float) or not math.isfinite(value) or abs(value)>flight['max_acceleration']:
            raise ValueError('invalid_reference_flight_parameter_gate_crossing_tangential_acceleration')
    for key in ('continuous_takeoff','progress_control','target_is_virtual_waypoint','target_is_pass_through_gate',
                'back_transition_after_gate_when_slow'):
        if key in flight and not isinstance(flight[key],bool):raise ValueError('invalid_reference_boolean_'+key)
    if 'path_lookahead_s' in flight and (type(flight['path_lookahead_s']) not in (int,float)
            or not math.isfinite(flight['path_lookahead_s']) or not 0<flight['path_lookahead_s']<=.5):
        raise ValueError('invalid_reference_path_lookahead_s')
    for key, maximum in (('heading_turn_min_height_m',3.),('heading_max_rate_deg_s',90.)):
        if key in flight and (type(flight[key]) not in (int,float)
                or not math.isfinite(flight[key]) or not 0<flight[key]<=maximum):
            raise ValueError('invalid_reference_flight_parameter_'+key)
    return profile


def reference_from_samples(samples, home_world, home_yaw_world):
    """Surveyed spawn establishes translation; GPS plugin fixes North to world +Y."""
    positions = np.asarray([s['position'] for s in samples], dtype=float)
    if len(samples) < 10 or samples[-1]['stamp']-samples[0]['stamp'] < 1.:
        raise ValueError('stationary_reference_warmup_required')
    if (not np.isfinite(positions).all() or np.max(np.linalg.norm(positions-positions.mean(axis=0), axis=1)) > .05
            or any(np.linalg.norm(vector(s['velocity'])) > .1 for s in samples)):
        raise ValueError('stationary_reference_required')
    yaw = float(samples[-1]['yaw'])
    yaw_error = math.atan2(math.sin(yaw-home_yaw_world), math.cos(yaw-home_yaw_world))
    return dict(origin_world=(vector(home_world)-positions.mean(axis=0)).tolist(),
                north_yaw_world_rad=math.pi/2, initial_local_enu=positions.mean(axis=0).tolist(),
                calibration_source='declared_spawn_and_stationary_PX4_EKF',
                initial_mavros_yaw_enu_rad=yaw, declared_spawn_yaw_world_rad=float(home_yaw_world),
                initial_yaw_error_rad=yaw_error, reference_sample_count=len(samples))


def read_command(msg, now, origin, horizontal_limit, height_limit, max_speed=3., max_acceleration=3.):
    if msg.header.frame_id != 'jol/flight_ned' or len(msg.points) != 1:
        raise ValueError('single_NED_PVA_point_required')
    stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    if not math.isfinite(stamp) or not -.02 <= now-stamp <= .3:
        raise ValueError('fresh_setpoint_required')
    point = msg.points[0]
    if len(point.transforms) != 1 or len(point.velocities) != 1 or len(point.accelerations) != 1:
        raise ValueError('complete_PVA_required')
    p, v, a = point.transforms[0].translation, point.velocities[0].linear, point.accelerations[0].linear
    p, v, a = [ned_to_enu([value.x, value.y, value.z]) for value in (p, v, a)]
    world = vector(origin)+p
    if (np.linalg.norm(p[:2]) > horizontal_limit or not .05 <= world[2] <= height_limit
            or np.linalg.norm(v) > max_speed or np.linalg.norm(a) > max_acceleration):
        raise ValueError('setpoint_outside_flight_envelope')
    return dict(stamp=stamp, position=p, velocity=v, acceleration=a)


def home_return_plan(position, home, origin, horizontal_limit, height_limit, *,
                     obstacles=(), clearance=.75, max_speed=3., max_acceleration=1., max_jerk=2.):
    """Stopped MC -> recorded launch XY at current height -> native PX4 landing."""
    position, home, origin = map(vector, (position, home, origin))
    goal = home.copy(); goal[2] = max(position[2], home[2]+3.)
    if (not math.isfinite(horizontal_limit) or not math.isfinite(height_limit)
            or any(np.linalg.norm(p[:2]-origin[:2]) > horizontal_limit-.5
                   or not .05 <= p[2] <= height_limit-.5 for p in (position, goal))):
        raise ValueError('return_path_outside_controller_envelope')
    delta = goal-position
    if np.linalg.norm(delta) < 1e-6:
        return None, goal
    segment = _segment([position, position+delta/3., position+2*delta/3., goal],
                       0., 'return_home', max_speed, max_acceleration, max_jerk, composed_bounds=True)
    plan = _finish_plan(position, goal, 0., [segment], max_speed, max_acceleration,
                        max_jerk, .05, obstacles, clearance)
    return plan, goal


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
    from trajectory_msgs.msg import MultiDOFJointTrajectory
    from mavros_msgs.msg import PositionTarget, State, ExtendedState, VfrHud
    from mavros_msgs.srv import CommandBool, SetMode, CommandVtolTransition
    from tf2_ros import TransformBroadcaster

    class Adapter(Node):
        def __init__(self):
            super().__init__('jol_mavros_adapter')
            if not self.get_parameter('use_sim_time').value:
                raise ValueError('Gazebo_simulation_time_required')
            self.aircraft_type=self.declare_parameter('aircraft_type','iris_proxy').value
            self.profile=aircraft_profile(self.aircraft_type,self.declare_parameter('physics_profile_path','').value)
            self.flight=self.profile.get('flight_parameters',{})
            self.home = vector(self.declare_parameter('home_world', [-5., -6., .1]).value)
            self.home_yaw = float(self.declare_parameter('home_yaw_world', 0.).value)
            self.horizontal = float(self.declare_parameter('horizontal_limit_m', 65.).value)
            self.height = float(self.declare_parameter('height_limit_m', 20.).value)
            if not math.isfinite(self.home_yaw) or min(self.horizontal, self.height) <= 0:
                raise ValueError('invalid_flight_reference_or_envelope')
            self.pose = self.velocity = self.fcu = self.extended = self.reference = None
            self.reference_samples = deque(maxlen=100)
            self.ground_reference_revision = 0; self.ground_reference_pose_stamp = None
            self.ground_reference_source = None
            self.previous_velocity = None; self.acceleration = np.zeros(3)
            self.command = None; self.last_now = None; self.stream_start = None; self.last_odom_stamp = -1.
            self.last_setpoint_sent=None
            self.phase = 'standby'; self.reason = ''; self.start_time = self.armed_time = None
            self.mode_pending = self.arm_pending = None; self.mode_ack = False; self.arm_requested = False
            self.stop_requested = None; self.last_control_ack = -1.
            self.vtol_pending=None;self.vtol_requested_state=None;self.vtol_requested_at=None
            self.route_goal=None;self.route_yaw=self.home_yaw;self.airspeed=None
            self.launch_home=None;self.return_stage=None;self.return_requested=None;self.return_plan=None
            self.return_start=self.return_stage_start=self.return_dwell=None;self.return_hold=None
            self.return_setpoint=None
            self.return_reacquire_requested_at=None
            self.return_home_reached=False;self.return_landed_at_home=None;self.route_obstacles=[];self.route_clearance=.75
            self.clock_fault = False
            self.setpoint_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
            self.state_pub = self.create_publisher(String, '/jol/sim_observer/state', 10)
            self.odom_pub = self.create_publisher(Odometry, '/jol/observer/odometry', 10)
            self.tf = TransformBroadcaster(self)
            self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.receive_pose, qos_profile_sensor_data)
            self.create_subscription(TwistStamped, '/mavros/local_position/velocity_local', self.receive_velocity, qos_profile_sensor_data)
            self.create_subscription(State, '/mavros/state', self.receive_state, 10)
            self.create_subscription(ExtendedState, '/mavros/extended_state', self.receive_extended, 10)
            self.create_subscription(VfrHud,'/mavros/vfr_hud',self.receive_airspeed,qos_profile_sensor_data)
            self.create_subscription(String,'/jol/static_flight/plan',self.receive_plan,10)
            self.create_subscription(MultiDOFJointTrajectory, '/jol/sim_observer/setpoint', self.receive_command, 10)
            self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
            self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
            self.vtol_client=self.create_client(CommandVtolTransition,'/mavros/cmd/vtol_transition')
            self.create_service(Trigger, '/jol/mavros/start', self.start)
            self.create_service(Trigger, '/jol/mavros/prepare_reference', self.prepare_reference)
            self.create_service(Trigger, '/jol/mavros/stop', self.stop)
            self.create_service(Trigger, '/jol/mavros/return_home', self.return_home)
            for state,name in ((4,'transition_fw'),(3,'transition_mc')):
                self.create_service(Trigger,'/jol/mavros/'+name,
                    lambda request,response,state=state:self.transition_service(state,response))
            self.create_timer(.02, self.tick)

        def now(self):return self.get_clock().now().nanoseconds*1e-9
        @staticmethod
        def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        def receive_state(self, msg):
            self.fcu = dict(connected=msg.connected, armed=msg.armed, mode=msg.mode,
                            system_status=int(msg.system_status), received=self.now())
        def receive_extended(self, msg):
            self.extended = dict(landed_state=int(msg.landed_state),
                landed=msg.landed_state == ExtendedState.LANDED_STATE_ON_GROUND,
                vtol_state=int(getattr(msg,'vtol_state',0)),received=self.now())
        def receive_airspeed(self,msg):
            if math.isfinite(msg.airspeed) and msg.airspeed>=0:
                self.airspeed=dict(value=float(msg.airspeed),received=self.now())
        def receive_plan(self,msg):
            try:
                packet=json.loads(msg.data)
                if packet.get('valid') is True and packet.get('frame_id')=='world':
                    self.route_goal=vector(packet['goal'])
                    self.route_obstacles=packet.get('obstacles',[])
                    self.route_clearance=float(packet.get('clearance_m',.75))
                elif packet.get('valid') is False:self.route_goal=None
            except (ValueError,KeyError,TypeError):self.route_goal=None
        def receive_pose(self, msg):
            p, q = msg.pose.position, msg.pose.orientation
            try:
                position = vector([p.x, p.y, p.z]); quaternion = np.array([q.x, q.y, q.z, q.w])
                if not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-6:raise ValueError('invalid_attitude')
                self.pose = dict(stamp=self.stamp(msg), position=position, rotation=Rotation.from_quat(quaternion))
            except ValueError:self.pose = None
        def receive_velocity(self, msg):
            v = msg.twist.linear
            try:
                value = dict(stamp=self.stamp(msg), velocity=vector([v.x, v.y, v.z]))
                if self.previous_velocity:
                    dt = value['stamp']-self.previous_velocity['stamp']
                    if .001 <= dt <= .3:
                        self.acceleration = (value['velocity']-self.previous_velocity['velocity'])/dt
                self.previous_velocity = self.velocity = value
            except ValueError:self.velocity = None
        def receive_command(self, msg):
            if not self.reference:return
            try:self.command = read_command(msg, self.now(), self.reference['origin_world'], self.horizontal, self.height,
                self.flight['max_speed']*1.5 if self.flight else 3.,
                self.flight['max_acceleration']*1.5 if self.flight else 3.)
            except (ValueError, AttributeError, TypeError) as error:self.reason = str(error)
        def landed(self, now):
            return bool(self.extended and self.extended['landed'] and 0 <= now-self.extended['received'] <= 2.)
        def fresh(self, now):
            return bool(self.fcu and self.fcu['connected'] and 0 <= now-self.fcu['received'] <= 2.
                        and self.pose and self.velocity and 0 <= now-self.pose['stamp'] <= .3
                        and 0 <= now-self.velocity['stamp'] <= .3 and abs(self.pose['stamp']-self.velocity['stamp']) <= .1)
        def ready(self, now):
            return bool(self.reference and not self.clock_fault and self.fresh(now)
                        # Local PX4 HEARTBEAT reports STANDBY only after its
                        # preflight health checks pass; fresh EKF alone is insufficient.
                        and self.fcu.get('system_status') in ((3,4) if self.fcu['armed'] else (3,))
                        and self.stream_start is not None and now-self.stream_start >= 1.)
        def append_reference_sample(self):
            if not self.reference_samples or self.pose['stamp'] > self.reference_samples[-1]['stamp']:
                self.reference_samples.append(dict(stamp=self.pose['stamp'], position=self.pose['position'],
                    velocity=self.velocity['velocity'], yaw=float(self.pose['rotation'].as_euler('xyz')[2])))
        def prepare_reference(self, request, response):
            now = self.now()
            if (not self.reference or self.clock_fault or self.phase != 'standby' or not self.fresh(now)
                    or not self.landed(now) or self.fcu['armed'] or self.fcu.get('system_status') != 3
                    or self.arm_pending is not None or self.mode_pending is not None
                    or self.vtol_pending is not None or self.vtol_requested_state is not None):
                response.success = False
                response.message = 'fresh_healthy_disarmed_grounded_standby_reference_required'
                return response
            self.append_reference_sample()
            try:
                if self.reference_samples[-1]['stamp'] > self.pose['stamp']:
                    raise ValueError('causal_ground_reference_required')
                if np.linalg.norm(self.velocity['velocity']) > .1:
                    raise ValueError('stationary_reference_required')
                candidate = reference_from_samples(self.reference_samples, self.home, self.home_yaw)
            except ValueError as error:
                response.success = False; response.message = str(error)
                return response
            # Gazebo's declared flat spawn height anchors only Z. Preserve the
            # surveyed horizontal frame and never re-anchor an active flight.
            origin = vector(self.reference['origin_world']).copy()
            origin[2] = candidate['origin_world'][2]
            self.reference = dict(self.reference, origin_world=origin.tolist(),
                                  initial_local_enu=self.pose['position'].tolist())
            self.ground_reference_revision += 1
            self.ground_reference_pose_stamp = float(self.pose['stamp'])
            self.ground_reference_source = 'declared_flat_simulation_ground_and_stationary_EKF'
            self.command = None
            response.success = True
            response.message = 'ground_Z_reference_prepared_revision_'+str(self.ground_reference_revision)
            return response
        def vtol_state(self,now):
            return self.extended.get('vtol_state',0) if self.extended and 0<=now-self.extended['received']<=2. else 0
        def start(self, request, response):
            now = self.now()
            if not self.ready(now) or not self.landed(now):
                response.success = False; response.message = 'fresh_landed_EKF_reference_and_prestream_required'
            elif self.fcu['armed'] or self.phase in ('arming', 'tracking', 'landing'):
                response.success = False; response.message = 'duplicate_start_or_airborne_vehicle'
            elif not self.mode_client.service_is_ready() or not self.arm_client.service_is_ready():
                response.success = False; response.message = 'MAVROS_mode_and_arming_services_required'
            elif self.aircraft_type=='tailsitter20cm_reference' and (self.vtol_state(now)!=3 or not self.vtol_client.service_is_ready()):
                response.success=False;response.message='observed_MC_and_VTOL_service_required'
            else:
                self.phase = 'arming'; self.reason = ''; self.start_time = now
                self.mode_ack = self.arm_requested = False; self.command = None
                # Record the actual EKF position at this launch, not the configured spawn.
                self.launch_home=self.pose['position'].copy();self.return_stage=None
                self.route_yaw=float(self.pose['rotation'].as_euler('xyz')[2])
                self.return_home_reached=False;self.return_landed_at_home=None;self.return_plan=self.return_requested=None
                self.return_setpoint=None
                self.request_mode('OFFBOARD')
                response.success = True; response.message = 'OFFBOARD_arming_requested_wait_for_armed_state'
            return response
        def transition_service(self,state,response):
            now=self.now()
            if self.aircraft_type!='tailsitter20cm_reference':
                response.success=False;response.message='reference_tailsitter_required'
            elif (not self.ready(now) or not self.fcu['armed'] or self.fcu['mode']!='OFFBOARD'
                  or self.landed(now) or self.vtol_state(now)==0):
                response.success=False;response.message='fresh_armed_OFFBOARD_airborne_VTOL_required'
            elif not self.request_vtol(state):
                response.success=False;response.message='VTOL_service_unavailable_or_command_pending'
            else:
                response.success=True;response.message='VTOL_command_requested_wait_for_observed_state_'+str(state)
            return response
        def request_vtol(self,state):
            if self.vtol_pending is not None or not self.vtol_client.service_is_ready():return False
            if self.vtol_requested_state==state and self.vtol_requested_at is not None and self.now()-self.vtol_requested_at<2.:
                return False
            request=CommandVtolTransition.Request();request.state=state
            future=self.vtol_client.call_async(request);self.vtol_pending=future
            self.vtol_requested_state=state;self.vtol_requested_at=self.now()
            def done(result):
                self.vtol_pending=None;self.last_control_ack=self.now()
                try:
                    reply=result.result()
                    if not reply.success:self.reason='PX4_VTOL_transition_rejected_'+str(state)+'_'+str(reply.result)
                except Exception as error:self.reason='MAVROS_VTOL_error: '+str(error)
            future.add_done_callback(done);return True
        def stop(self, request, response):
            self.begin_land('user_stop')
            response.success = True; response.message = 'landing_or_ground_disarm_requested'
            return response
        def return_home(self, request, response):
            now=self.now()
            response.success=self.begin_return(now)
            response.message=('return_to_recorded_takeoff_active_wait_for_actual_home_landing' if response.success
                else 'fresh_armed_airborne_EKF_recorded_launch_home_and_mode_service_required')
            return response
        def begin_return(self,now,reason='return_to_recorded_takeoff_then_land'):
            # Mission and adapter watchdog can observe the same mode loss in
            # either order. Attaching must not cancel an already safe return.
            if self.phase=='returning':return True
            if self.phase=='landing':return bool(self.return_home_reached and self.return_stage=='landing')
            if (self.launch_home is None or not self.reference or not self.fresh(now)
                    or not self.fcu['armed'] or self.landed(now)
                    or self.clock_fault or not self.mode_client.service_is_ready()):return False
            self.phase='returning';self.reason=reason;self.command=None
            self.return_start=self.return_stage_start=now
            self.return_stage='transition_mc' if self.fcu['mode']=='OFFBOARD' else 'reacquire_offboard'
            self.return_dwell=None;self.return_hold=self.pose['position'].copy();self.return_plan=None
            self.return_setpoint=None;self.return_reacquire_requested_at=None
            self.return_requested=self.launch_home.copy();self.return_home_reached=False;self.return_landed_at_home=None
            if self.return_stage=='reacquire_offboard':self.mode_ack=False
            elif self.aircraft_type=='tailsitter20cm_reference' and self.vtol_state(now)!=3:self.request_vtol(3)
            return True
        def request_mode(self, mode):
            if self.mode_pending is not None or not self.mode_client.service_is_ready():return False
            request = SetMode.Request(); request.custom_mode = mode
            future = self.mode_client.call_async(request); self.mode_pending = future
            def done(result):
                self.mode_pending = None
                self.last_control_ack = self.now()
                try:
                    accepted = result.result().mode_sent
                    if mode == 'OFFBOARD':self.mode_ack = bool(accepted)
                    if not accepted:self.reason = 'PX4_mode_rejected_'+mode
                except Exception as error:self.reason = 'MAVROS_mode_error: '+str(error)
            future.add_done_callback(done); return True
        def request_arm(self, armed):
            if self.arm_pending is not None or not self.arm_client.service_is_ready():return False
            request = CommandBool.Request(); request.value = armed
            future = self.arm_client.call_async(request); self.arm_pending = future
            def done(result):
                self.arm_pending = None
                self.last_control_ack = self.now()
                try:
                    reply = result.result()
                    if not reply.success:self.reason = 'PX4_arming_rejected_'+str(reply.result)
                except Exception as error:self.reason = 'MAVROS_arming_error: '+str(error)
            future.add_done_callback(done); return True
        def begin_land(self, reason):
            if self.phase=='returning' and reason!='return_home_arrived':self.return_stage='failed_emergency_landing'
            self.phase = 'landing'; self.reason = reason; self.command = None
            self.stop_requested = self.now()
            if self.fcu and self.fcu['armed']:
                if self.landed(self.now()):self.request_arm(False)
                elif self.aircraft_type=='tailsitter20cm_reference' and self.vtol_state(self.now())!=3:
                    self.request_vtol(3)
                else:self.request_mode('AUTO.LAND')
        def publish_setpoint(self, now):
            if not self.reference or not self.pose:return
            position = self.pose['position'].copy(); velocity = np.zeros(3); acceleration = np.zeros(3)
            if self.phase == 'arming':position = np.asarray(self.reference['initial_local_enu'])
            elif self.phase == 'tracking' and self.command:
                age = min(.1, max(0., now-self.command['stamp']))
                position = self.command['position']+self.command['velocity']*age+self.command['acceleration']*.5*age*age
                velocity = self.command['velocity']+self.command['acceleration']*age
                acceleration = self.command['acceleration']
            elif self.phase=='returning' and self.return_hold is not None:
                position=self.return_hold.copy()
                if self.return_stage=='returning' and self.return_plan:
                    sample=sample_plan(self.return_plan,max(0.,now-self.return_stage_start))
                    position=vector(sample['position'])-vector(self.reference['origin_world'])
                    velocity=vector(sample['velocity']);acceleration=vector(sample['acceleration'])
            msg = PositionTarget(); msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = 'map'
            # Despite this MAVLink frame label, the MAVROS ROS API accepts ENU P/V/A.
            msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            msg.type_mask = PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE
            if self.aircraft_type=='tailsitter20cm_reference':
                if self.phase in ('standby','landing'):
                    self.route_yaw=float(self.pose['rotation'].as_euler('xyz')[2])
                elif (self.phase in ('tracking','returning') and self.fresh(now) and self.fcu['armed']
                        and self.extended and self.extended.get('landed_state') == 2
                        and 0<=now-self.extended['received']<=2. and self.launch_home is not None
                        and self.pose['position'][2]-self.launch_home[2]
                            >= self.flight.get('heading_turn_min_height_m',.5)):
                    direction=(velocity if np.linalg.norm(velocity[:2])>.2 else
                        self.route_goal-(np.asarray(self.reference['origin_world'])+self.pose['position'])
                        if self.route_goal is not None else np.zeros(3))
                    if np.linalg.norm(direction[:2])>.2:
                        desired=math.atan2(direction[1],direction[0])
                        error=math.atan2(math.sin(desired-self.route_yaw),math.cos(desired-self.route_yaw))
                        dt=0. if self.last_setpoint_sent is None else min(.1,max(0.,now-self.last_setpoint_sent))
                        limit=math.radians(self.flight.get('heading_max_rate_deg_s',45.))*dt
                        yaw=self.route_yaw+float(np.clip(error,-limit,limit))
                        self.route_yaw=math.atan2(math.sin(yaw),math.cos(yaw))
                msg.type_mask=PositionTarget.IGNORE_YAW_RATE;msg.yaw=self.route_yaw
            msg.position.x, msg.position.y, msg.position.z = map(float, position)
            msg.velocity.x, msg.velocity.y, msg.velocity.z = map(float, velocity)
            msg.acceleration_or_force.x, msg.acceleration_or_force.y, msg.acceleration_or_force.z = map(float, acceleration)
            # Record the PVA actually sent; AUTO.LAND's internal setpoints are not observed here.
            self.return_setpoint=(dict(stamp=now,position=(vector(self.reference['origin_world'])+position).tolist(),
                velocity=velocity.tolist(),acceleration=acceleration.tolist()) if self.phase=='returning' else None)
            self.setpoint_pub.publish(msg)
            if self.last_setpoint_sent is None or now-self.last_setpoint_sent>.2:self.stream_start=now
            self.last_setpoint_sent=now
        def publish_feedback(self, now):
            if not self.reference or not self.fresh(now):return
            position = np.asarray(self.reference['origin_world'])+self.pose['position']
            velocity = self.velocity['velocity']; stamp = self.pose['stamp']
            odom = Odometry(); odom.header.frame_id = 'world'; odom.child_frame_id = 'jol/observer_body'
            odom.header.stamp.sec = int(stamp); odom.header.stamp.nanosec = int((stamp-int(stamp))*1e9)
            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, position)
            q = self.pose['rotation'].as_quat()
            odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = map(float, q)
            body = self.pose['rotation'].inv().apply(velocity)
            odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, body)
            if stamp > self.last_odom_stamp:
                self.odom_pub.publish(odom); self.last_odom_stamp = stamp
                transform = TransformStamped(); transform.header = odom.header; transform.child_frame_id = odom.child_frame_id
                transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, position)
                transform.transform.rotation = odom.pose.pose.orientation; self.tf.sendTransform(transform)
            packet = dict(stamp=stamp, frame_id='world', setpoint_frame='jol/flight_ned', controller_ready=self.ready(now),
                horizontal_limit_m=self.horizontal, height_limit_m=self.height, position=position.tolist(), velocity=velocity.tolist(),
                acceleration_world=self.acceleration.tolist(), acceleration_source='mavros_velocity_finite_difference',
                acceleration_stamp=self.velocity['stamp'], connected=self.fcu['connected'], armed=self.fcu['armed'],
                mode=self.fcu['mode'], landed=self.landed(now),
                fcu_system_status=self.fcu.get('system_status'),
                landed_state=self.extended['landed_state'] if self.extended else 0,
                vtol_state=self.vtol_state(now),vtol_requested_state=self.vtol_requested_state,
                attitude_quaternion_world=q.tolist(),nose_axis_world=self.pose['rotation'].apply([0.,0.,1.]).tolist(),
                airspeed_m_s=self.airspeed['value'] if self.airspeed and 0<=now-self.airspeed['received']<=2. else None,
                aircraft_type=self.aircraft_type,physics_data_status=self.profile.get('data_status','Iris_proxy'),
                adapter_phase=self.phase, reason=self.reason,
                pending_control_command=self.arm_pending is not None or self.mode_pending is not None or self.vtol_pending is not None,
                fcu_state_stamp=self.fcu['received'], stop_requested_stamp=self.stop_requested,
                command_stale=not self.command or now-self.command['stamp'] > .4,
                flight_backend='mavros_px4', feedback_controller='MAVROS_PX4_Gazebo_motor_dynamics',
                own_pose_source='mavros_px4_ekf', feedback_source='mavros_px4_ekf', simulation_only=True, **self.reference)
            packet.update(ground_reference_revision=self.ground_reference_revision,
                ground_reference_pose_stamp=self.ground_reference_pose_stamp,
                ground_reference_source=self.ground_reference_source)
            packet.update(return_stage=self.return_stage,return_home_reached=self.return_home_reached,
                return_landed_at_home=self.return_landed_at_home,
                launch_home_world=(vector(self.reference['origin_world'])+self.launch_home).tolist() if self.launch_home is not None else None,
                return_requested_world=(vector(self.reference['origin_world'])+self.return_requested).tolist() if self.return_requested is not None else None,
                return_home_error_m=float(np.linalg.norm((self.pose['position']-self.launch_home)[:2])) if self.launch_home is not None else None,
                return_desired_position_world=self.return_setpoint['position'] if self.return_setpoint else None,
                return_desired_velocity_world=self.return_setpoint['velocity'] if self.return_setpoint else None,
                return_desired_acceleration_world=self.return_setpoint['acceleration'] if self.return_setpoint else None,
                return_setpoint_stamp=self.return_setpoint['stamp'] if self.return_setpoint else None,
                return_plan={key:self.return_plan[key] for key in ('model','frame_id','goal','duration_s','segments','limits','motion_validation')}
                    if self.return_plan else None)
            self.state_pub.publish(String(data=json.dumps(packet)))
        def tick_return(self,now):
            if not self.fresh(now) or not self.fcu['armed']:
                self.begin_land('return_feedback_or_OFFBOARD_lost');return
            if now-self.return_start>120.:
                self.begin_land('return_to_takeoff_timeout');return
            if self.return_stage=='reacquire_offboard':
                # Continue the real 50Hz current-position hold stream. A mode
                # service ACK alone cannot prove PX4 accepted OFFBOARD.
                requested=getattr(self,'return_reacquire_requested_at',None)
                if now-self.return_stage_start>6.:
                    self.begin_land('return_OFFBOARD_reacquisition_failed_emergency_local_landing');return
                if requested is None and now-self.return_stage_start>=1.:
                    if not self.request_mode('OFFBOARD'):
                        self.begin_land('return_OFFBOARD_service_unavailable_emergency_local_landing');return
                    self.return_reacquire_requested_at=now
                elif (requested is not None and self.mode_ack and self.fcu['mode']=='OFFBOARD'
                        and self.fcu['received']>max(requested,self.last_control_ack)):
                    self.return_stage='transition_mc';self.return_stage_start=now
                    if self.aircraft_type=='tailsitter20cm_reference' and self.vtol_state(now)!=3:self.request_vtol(3)
                return
            if self.fcu['mode']!='OFFBOARD':
                self.begin_land('return_OFFBOARD_lost_after_reacquisition');return
            if self.return_stage=='transition_mc':
                if self.aircraft_type=='tailsitter20cm_reference' and (self.vtol_state(now)!=3 or self.vtol_requested_state==4):
                    self.request_vtol(3)
                elif self.vtol_pending is None and self.vtol_requested_state is None:
                    self.return_hold=self.pose['position'].copy();self.return_stage='braking'
                    self.return_stage_start=now
                if now-self.return_start>12.:self.begin_land('return_observed_MC_timeout')
            elif self.return_stage=='braking':
                # PX4's position controller brakes physically at a fixed measured point.
                settled=(np.linalg.norm(self.velocity['velocity'])<=.2
                         and np.linalg.norm(self.pose['position']-self.return_hold)<=.6)
                self.return_dwell=(self.return_dwell if self.return_dwell is not None else now) if settled else None
                if self.return_dwell is not None and now-self.return_dwell>=.5:
                    try:
                        origin=vector(self.reference['origin_world'])
                        self.return_plan,goal=home_return_plan(origin+self.pose['position'],origin+self.launch_home,
                            origin,self.horizontal,self.height,obstacles=self.route_obstacles,clearance=self.route_clearance,
                            max_speed=min(3.,.7*self.flight.get('max_speed',3.)),
                            max_acceleration=min(1.,.7*self.flight.get('max_acceleration',1.)),
                            max_jerk=min(2.,.7*self.flight.get('max_jerk',2.)))
                    except (ValueError,KeyError,TypeError) as error:
                        self.begin_land('return_plan_rejected: '+str(error));return
                    self.return_requested=goal-origin;self.return_hold=self.return_requested.copy()
                    self.return_stage='returning';self.return_stage_start=now;self.return_dwell=None
                elif now-self.return_stage_start>30.:self.begin_land('return_MC_braking_timeout')
            elif self.return_stage=='returning':
                arrived=(np.linalg.norm((self.pose['position']-self.launch_home)[:2])<=.10
                         and np.linalg.norm(self.pose['position']-self.return_requested)<=.6
                         and np.linalg.norm(self.velocity['velocity'])<=.12)
                self.return_dwell=(self.return_dwell if self.return_dwell is not None else now) if arrived else None
                if self.return_dwell is not None and now-self.return_dwell>=1.:
                    self.return_home_reached=True;self.return_stage='landing';self.begin_land('return_home_arrived')
                elif self.return_plan and now-self.return_stage_start>self.return_plan['duration_s']+30.:
                    self.begin_land('return_home_not_reached_before_deadline')
        def tick(self):
            now = self.now()
            if self.last_now is not None and now < self.last_now:
                self.clock_fault = True; self.begin_land('simulation_clock_reset_restart_required')
            self.last_now = now
            if (self.vtol_requested_state is not None and self.vtol_pending is None
                    and self.vtol_state(now)==self.vtol_requested_state
                    and self.extended['received']>self.last_control_ack):
                self.vtol_requested_state=None;self.vtol_requested_at=None
            # Keep a recent stationary window for an explicit preflight Z
            # survey; a healthy startup alone does not end EKF ground drift.
            if (not self.clock_fault and self.phase == 'standby' and self.fresh(now) and self.landed(now)
                    and not self.fcu['armed'] and self.fcu.get('system_status') == 3):
                self.append_reference_sample()
                if not self.reference:
                    try:self.reference = reference_from_samples(self.reference_samples, self.home, self.home_yaw)
                    except ValueError:pass
            else:self.reference_samples.clear()
            if self.phase == 'arming':
                if not self.fresh(now) or now-self.start_time > 10.:
                    self.begin_land('PX4_arming_timeout_or_feedback_lost')
                elif self.fcu['armed'] and self.fcu['mode'] == 'OFFBOARD':
                    self.phase = 'tracking'; self.armed_time = now
                elif self.mode_ack and self.fcu['mode'] == 'OFFBOARD' and not self.arm_requested:
                    self.arm_requested = self.request_arm(True)
            elif self.phase == 'tracking':
                if not self.fresh(now) or not self.fcu['armed']:
                    self.begin_land('PX4_feedback_or_OFFBOARD_lost')
                elif self.fcu['mode']!='OFFBOARD':
                    if not self.begin_return(now,'PX4_OFFBOARD_lost_return_to_takeoff'):
                        self.begin_land('PX4_OFFBOARD_lost_home_return_unavailable_emergency_local_landing')
                elif now-self.armed_time > 1. and (not self.command or now-self.command['stamp'] > .4):
                    self.begin_land('trajectory_setpoint_watchdog')
            elif self.phase=='returning':self.tick_return(now)
            if self.phase == 'landing' and self.fcu:
                if self.fcu['armed']:
                    if self.landed(now):self.request_arm(False)
                    elif self.aircraft_type=='tailsitter20cm_reference' and (self.vtol_state(now)!=3 or self.vtol_requested_state==4):
                        self.request_vtol(3)
                    elif self.vtol_pending is not None or self.vtol_requested_state is not None:pass
                    elif self.fcu['mode'] != 'AUTO.LAND':self.request_mode('AUTO.LAND')
                elif (self.landed(now) and self.arm_pending is None and self.mode_pending is None and self.vtol_pending is None
                      and (self.stop_requested is None or self.fcu['received'] > max(self.stop_requested, self.last_control_ack))):
                    self.phase = 'standby'
                    if self.return_home_reached:
                        self.return_landed_at_home=bool(self.pose and np.linalg.norm((self.pose['position']-self.launch_home)[:2])<=.35)
                        self.return_stage='complete' if self.return_landed_at_home else 'landed_away_from_takeoff'
            # A late arming acknowledgement after Stop must never revive flight.
            if self.phase == 'standby' and self.fcu and self.fcu['armed']:
                self.begin_land('unexpected_armed_state_after_stop_or_startup')
            if self.fresh(now):self.publish_setpoint(now)
            elif self.last_setpoint_sent is None or now-self.last_setpoint_sent>.2:self.stream_start=None
            self.publish_feedback(now)

    rclpy.init(args=args)
    node = Adapter()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__ == '__main__':main()
