"""LiDAR로 대상을 고르고 센서·팔의 조준 시각별 융합 상태를 제어에 연결한다."""
from collections import deque
from itertools import islice
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .fusion_core import transform_matrix
from .tf_guard import guarded_tf
from .pointing_core import (ARM_LOWER, ARM_UPPER, TargetSelector, choose_turret_state, feedback_step_interval, forecast,
                            joint_slew_lower_bound, limit_step, patrol_step, ray_error, solve_arm, solve_turret,
                            turret_geometry)

NAMES = {'turret': ['yaw_joint', 'pitch_joint'], 'arm': ['joint1', 'joint2', 'joint3', 'joint4']}


def seconds(stamp):
    return stamp.sec+stamp.nanosec*1e-9


class Pointing(Node):
    def __init__(self):
        super().__init__('jol_pointing')
        self.horizon = float(self.declare_parameter('prediction_horizon', .5).value)
        self.turret_lead = float(self.declare_parameter('turret_lead', 0.).value)
        target_id = int(self.declare_parameter('target_id', 0).value)
        acquisition_speed = float(self.declare_parameter('acquisition_min_speed', 0.).value)
        self.max_age = float(self.declare_parameter('max_track_age', .5).value)
        self.max_coast = float(self.declare_parameter('max_coast', 1.).value)
        self.max_fusion_age = float(self.declare_parameter('max_fusion_age', .35).value)
        self.max_camera_evidence_age = float(self.declare_parameter('max_camera_evidence_age', .8).value)
        self.fusion_enabled = bool(self.declare_parameter('coordinate_fusion_enabled', True).value)
        self.joint_age = float(self.declare_parameter('max_joint_age', .3).value)
        self.arm_enabled = bool(self.declare_parameter('arm_enabled', True).value)
        self.arm_rate = float(self.declare_parameter('arm_rate_limit', 1.5).value)
        self.yaw_rate = float(self.declare_parameter('turret_yaw_rate_limit', 1.2).value)
        self.pitch_rate = float(self.declare_parameter('turret_pitch_rate_limit', .8).value)
        self.patrol_elevation = float(self.declare_parameter('patrol_elevation_deg', 20.).value)
        self.patrol_period = float(self.declare_parameter('patrol_period', 8.).value)
        self.rod_length = float(self.declare_parameter('rod_length', .30).value)
        self.camera_mount = str(self.declare_parameter('camera_mount', 'turret').value)
        self.lidar_mount_mode = str(self.declare_parameter('lidar_mount_mode', 'payload').value)
        turret_geometry(self.camera_mount, self.lidar_mount_mode)
        values = [self.horizon, self.turret_lead, acquisition_speed, self.max_age, self.max_coast,
                  self.joint_age, self.arm_rate, self.yaw_rate, self.pitch_rate, self.rod_length,
                  self.max_fusion_age, self.max_camera_evidence_age]
        if (not np.isfinite(values).all() or not 0 <= self.turret_lead <= self.horizon <= 3
                or target_id < 0 or acquisition_speed < 0 or min(values[3:]) <= 0
                or self.arm_rate > 1.5 or self.yaw_rate > 1.2 or self.pitch_rate > .8):
            raise ValueError('Invalid pointing horizon, freshness, or simulated joint-rate parameters')
        patrol_step([0., 0.], 0., self.patrol_elevation, self.patrol_period, self.yaw_rate, self.pitch_rate)
        self.selector = TargetSelector(target_id, acquisition_speed)
        self.packet, self.background_ready, self.active, self.last_tick = None, False, False, None
        self.fused_packet = None
        self.auto_acquisition_enabled = True
        self.track_min_stamp = -math.inf
        self.fusion_min_stamp = -math.inf
        self.mode, self.patrol_arm_hold = 'background', None
        self.event_seq = 0
        self.last_event = dict(seq=0, action='startup', stamp=None, previous_target_id=None)
        # 200 Hz 관절값이 TF보다 먼저 도착해도 0.3초 구간을 보존한다.
        self.states = {key: deque(maxlen=100) for key in NAMES}
        self.last_commands = {}
        self.selection_changed_stamp = None
        self.tf, self.listener = guarded_tf(self, cache_time=Duration(seconds=15.))
        self.command_publishers = {name: self.create_publisher(JointState, '/jol/'+name+'/command', 10) for name in NAMES}
        self.status_pub = self.create_publisher(String, '/jol/pointing/status', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/jol/pointing/markers', 10)
        self.create_subscription(String, '/jol/tracks', self.receive_tracks, 10)
        self.create_subscription(String, '/jol/fused_tracks', self.receive_fusion, 10)
        self.create_subscription(Int32, '/jol/pointing/target_id', self.receive_selection, 10)
        self.create_service(Trigger, '/jol/pointing/stop_tracking', self.stop_tracking)
        self.create_service(Trigger, '/jol/pointing/resume_auto', self.resume_auto)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/jol/background_ready', self.receive_background, latched)
        for key in NAMES:
            self.create_subscription(JointState, '/jol/'+key+'/joint_states',
                                     lambda msg, key=key: self.receive_joints(key, msg), qos_profile_sensor_data)
        self.create_timer(.05, self.tick)

    def receive_background(self, msg):
        self.background_ready = bool(msg.data)
        if not self.background_ready:
            self.fused_packet = None

    def receive_selection(self, msg):
        try:
            self.selector.request(int(msg.data))
            self.fused_packet = None
        except ValueError as error:
            self.get_logger().warning(str(error))

    def receive_tracks(self, msg):
        try:
            packet = json.loads(msg.data)
            if (not isinstance(packet, dict) or not np.isfinite(float(packet['stamp']))
                    or not isinstance(packet['tracks'], list) or len(packet['tracks']) > 256):
                raise ValueError('Invalid tracks packet')
            packet['stamp'] = float(packet['stamp'])
            now = self.get_clock().now().nanoseconds*1e-9
            # 시간 초기화 전 패킷은 버리고 짧은 /clock 전달 지연은 후속 검사로 넘긴다.
            if (packet['stamp'] > now+self.max_age or packet['stamp'] <= self.track_min_stamp
                    or (self.packet is not None and packet['stamp'] < self.packet['stamp'])):
                return
            self.packet = packet
        except (ValueError, TypeError, KeyError) as error:
            self.get_logger().warning('Rejected pointing tracks: '+str(error), throttle_duration_sec=5)

    def receive_joints(self, key, msg):
        if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
            return
        positions = dict(zip(msg.name, msg.position))
        if not all(name in positions and np.isfinite(positions[name]) for name in NAMES[key]):
            return
        stamp = seconds(msg.header.stamp)
        now = self.get_clock().now().nanoseconds*1e-9
        if (stamp <= 0 or stamp > now+self.joint_age
                or (self.states[key] and stamp <= seconds(self.states[key][-1].header.stamp))):
            return
        self.states[key].append(msg)

    def receive_fusion(self, msg):
        if not self.auto_acquisition_enabled:
            return
        try:
            packet = json.loads(msg.data)
            if not isinstance(packet, dict):
                raise ValueError('Invalid fusion packet')
            stamp = float(packet['stamp'])
            state_stamp = float(packet.get('current_state_stamp', stamp))
            revision = int(packet.get('fusion_revision', 0))
            if (not np.isfinite(stamp) or stamp <= 0 or not isinstance(packet['tracks'], list)
                    or not np.isfinite(state_stamp) or state_stamp < stamp or revision < 0
                    or revision != packet.get('fusion_revision', 0) or len(packet['tracks']) > 256):
                raise ValueError('Invalid fusion packet')
            now = self.get_clock().now().nanoseconds*1e-9
            previous = ((self.fused_packet['stamp'], self.fused_packet.get('fusion_revision', 0),
                         self.fused_packet.get('current_state_stamp', self.fused_packet['stamp']))
                        if self.fused_packet is not None else (-math.inf, -math.inf, -math.inf))
            if (state_stamp > now+1e-6 or stamp <= self.fusion_min_stamp
                    or (stamp, revision, state_stamp) <= previous):
                return
            packet['stamp'] = stamp
            self.fused_packet = packet
        except (ValueError, TypeError, KeyError) as error:
            self.get_logger().warning('Rejected pointing fusion: '+str(error), throttle_duration_sec=5)

    def base_status(self, now, state):
        changed = getattr(self, 'selection_changed_stamp', None)
        return dict(stamp=now, frame_id='world', command_stamp=None, command_success=False,
                    arm_enabled=getattr(self, 'arm_enabled', True),
                    camera_mount=getattr(self, 'camera_mount', 'turret'),
                    lidar_mount_mode=getattr(self, 'lidar_mount_mode', 'payload'),
                    target_id=self.selector.selected_id, requested_target_id=self.selector.requested_id,
                    background_ready=self.background_ready, state=state, mode=self.mode,
                    auto_acquisition_enabled=self.auto_acquisition_enabled,
                    patrol_elevation_deg=self.patrol_elevation, patrol_period_s=self.patrol_period,
                    event_seq=self.event_seq, event_action=self.last_event['action'],
                    event_stamp=self.last_event['stamp'], last_event=self.last_event.copy(),
                    selection_changed_stamp=changed,
                    selection_age_s=now-changed if changed is not None else None)

    def acquisition_event(self, enabled, action):
        stamp = self.get_clock().now().to_msg()
        now = seconds(stamp)
        previous = self.selector.selected_id
        self.auto_acquisition_enabled = enabled
        self.selector.request(0)
        self.selection_changed_stamp = None
        # 정지·재개 이후에는 이전 영상·LiDAR 결과로 재획득하지 않는다.
        self.packet = self.fused_packet = None
        self.track_min_stamp = now
        self.fusion_min_stamp = now
        self.patrol_arm_hold = None
        self.event_seq += 1
        self.last_event = dict(seq=self.event_seq, action=action, stamp=now, previous_target_id=previous)
        self.mode = 'patrol' if self.background_ready else 'background'
        self.hold_patrol_arm(now, stamp)
        status = self.base_status(now, action+'_requested')
        status['command_mode'] = 'pending_patrol'
        self.markers_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
        return status

    def stop_tracking(self, request, response):
        self.acquisition_event(False, 'stop_tracking')
        response.success, response.message = True, 'Tracking stopped; patrol active, automatic acquisition disabled'
        return response

    def resume_auto(self, request, response):
        self.acquisition_event(True, 'resume_auto')
        response.success, response.message = True, 'Automatic LiDAR acquisition resumed during patrol'
        return response

    def latest_joints(self, key, now):
        if self.states[key]:
            msg = self.states[key][-1]
            if 0 <= now-seconds(msg.header.stamp) <= self.joint_age:
                positions = dict(zip(msg.name, msg.position))
                return np.array([positions[name] for name in NAMES[key]])
        return None

    def command_reference(self, key, now, measured):
        # 기하 평가는 같은 시각의 TF, 이동폭 제한은 가장 최신 관절값을 쓴다.
        joints = self.latest_joints(key, now)
        if joints is not None:
            return joints, seconds(self.states[key][-1].header.stamp)
        return np.asarray(measured['actual_joints']), measured.get('joint_state_stamp', now)

    def hold_patrol_arm(self, now, stamp):
        if not getattr(self, 'arm_enabled', True):
            return
        # 순찰 진입 때 실측 자세를 저장해 이전 목표로 계속 움직이지 않게 한다.
        if self.patrol_arm_hold is None:
            self.patrol_arm_hold = self.latest_joints('arm', now)
        if self.patrol_arm_hold is not None:
            self.command('arm', self.patrol_arm_hold, stamp)
            self.active = True

    def patrol(self, status, stamp, dt):
        now = status['stamp']
        if self.mode not in ('patrol', 'background'):
            self.patrol_arm_hold = None
        self.mode = 'patrol' if self.background_ready else 'background'
        self.selector.selected_id, self.selector.last_seen = None, -math.inf
        self.selection_changed_stamp = None
        self.fused_packet = None
        status.update(mode=self.mode, command_mode='patrol', target_id=None,
                      selection_changed_stamp=None, selection_age_s=None)
        if 'selection' in status:
            status['selection'].update(selected_id=None, distance_m=None)
        self.hold_patrol_arm(now, stamp)
        joints = self.latest_joints('turret', now)
        if joints is None:
            status.update(state='waiting_for_patrol_joint_state', command_success=False)
        else:
            command = patrol_step(joints, dt, self.patrol_elevation, self.patrol_period,
                                  self.yaw_rate, self.pitch_rate)
            self.command('turret', command, stamp)
            self.active = True
            status.update(command_stamp=now, command_success=True,
                          turret=dict(command_joints=command.tolist(), actual_joints=joints.tolist(),
                                      desired_elevation_deg=self.patrol_elevation))
            try:
                measured, _, _ = self.measured('turret', now)
                direction = measured['actual_direction']
                measured['measured_elevation_deg'] = math.degrees(math.atan2(
                    direction[2], math.hypot(direction[0], direction[1])))
                measured['command_joints'] = command.tolist()
                status['turret'].update(measured)
            except ValueError as error:
                status['turret']['measurement_reason'] = str(error)
        if self.patrol_arm_hold is not None:
            status['arm'] = dict(command_joints=self.patrol_arm_hold.tolist(), command_mode='hold')
        self.markers_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))

    def measured(self, key, now):
        geometry = turret_geometry(getattr(self, 'camera_mount', 'turret'),
                                   getattr(self, 'lidar_mount_mode', 'payload'))
        frames = ([geometry['frame'], 'jol/base'] if key == 'turret'
                  else ['jol/rod_base', 'jol/rod_tip', 'jol/arm_base'])
        # 관절값과 정확히 같은 시각의 TF가 있는 최근 표본만 선택한다.
        for message in reversed(self.states[key]):
            stamp = seconds(message.header.stamp)
            if not 0 <= now-stamp <= self.joint_age:
                continue
            try:
                matrices = {}
                for frame in frames:
                    tf = self.tf.lookup_transform('world', frame, Time.from_msg(message.header.stamp)).transform
                    p, q = tf.translation, tf.rotation
                    matrices[frame] = transform_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
                values = dict(zip(message.name, message.position))
                positions = np.array([values[name] for name in NAMES[key]])
                axis = geometry['axis'] if key == 'turret' else 0
                origin, direction = matrices[frames[0]][:3, 3], matrices[frames[0]][:3, axis]
                result = dict(joint_state_stamp=stamp, actual_tf_stamp=stamp, actual_joints=positions.tolist(),
                              actual_frame=frames[0], actual_axis_index=axis,
                              actual_origin=origin.tolist(), actual_direction=direction.tolist(),
                              base_frame=frames[-1], base_tf_stamp=stamp,
                              base_world_transform=matrices[frames[-1]].tolist())
                if key == 'arm':
                    result['rod_tip'] = matrices['jol/rod_tip'][:3, 3].tolist()
                return result, matrices[frames[-1]], positions
            except Exception:
                continue
        raise ValueError(key+'_joint_state_or_exact_tf_unavailable')

    def selection_origin(self, source_stamp):
        """LiDAR 상태와 같은 시각의 시스템 중심을 구한다."""
        try:
            transform = self.tf.lookup_transform('world', 'jol/base',
                Time(nanoseconds=round(source_stamp*1e9))).transform
            p = transform.translation
            origin = [p.x, p.y, p.z]
            if not np.isfinite(origin).all():
                raise ValueError('Non-finite system origin')
            return origin
        except Exception as error:
            raise ValueError('selection_origin_exact_tf_unavailable') from error

    def command(self, key, values, stamp):
        if key == 'arm' and not getattr(self, 'arm_enabled', True):
            return
        msg = JointState()
        msg.header.stamp, msg.header.frame_id = stamp, 'world'
        msg.name, msg.position = NAMES[key], [float(v) for v in values]
        self.command_publishers[key].publish(msg)
        self.last_commands[key] = msg.position

    def hold_diagnostics(self):
        # 이 노드가 실제 받은 TF 기록만 제한된 크기로 진단한다.
        limit = 32
        with self.tf.guard_lock:
            children = {child: dict(earliest_stamp=times[0]*1e-9 if times else None,
                                   latest_stamp=times[-1]*1e-9 if times else None,
                                   sample_count=len(times))
                        for child, times in islice(self.tf.dynamic_times.items(), limit)}
            guard = dict(dynamic_children=children,
                         dynamic_children_total=len(self.tf.dynamic_times),
                         dynamic_children_limit=limit,
                         dynamic_children_truncated=len(self.tf.dynamic_times) > limit,
                         guard_rejections=self.tf.guard_rejections,
                         future_rejections=self.tf.future_rejections)
        return dict(latest_joint_stamp={key: seconds(messages[-1].header.stamp) if messages else None
                                       for key, messages in self.states.items()}, tf_guard=guard)

    def hold(self, status, stamp):
        self.mode = status['mode'] = 'hold'
        if self.active:
            for key in NAMES:
                if self.states[key]:
                    msg = self.states[key][-1]
                    if 0 <= status['stamp']-seconds(msg.header.stamp) <= self.joint_age:
                        values = dict(zip(msg.name, msg.position))
                        self.command(key, [values[n] for n in NAMES[key]], stamp)
                        continue
                if key in self.last_commands:
                    self.command(key, self.last_commands[key], stamp)
            status['command_stamp'] = status['stamp']
        status['command_success'] = False
        status['command_mode'] = 'hold' if self.active else 'preserve_background_survey'
        self.markers_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))

    def tick(self):
        started = time.perf_counter()
        stamp = self.get_clock().now().to_msg()
        now = seconds(stamp)
        # 같은 /clock 값의 중복 콜백이 이동 명령을 정지 명령으로 덮지 않게 한다.
        if self.last_tick is not None and now == self.last_tick:
            return
        dt = .05 if self.last_tick is None else min(.1, max(0., now-self.last_tick))
        if self.last_tick is not None and now < self.last_tick:
            self.packet, self.active, self.background_ready = None, False, False
            self.fused_packet = None
            self.track_min_stamp = -math.inf
            self.fusion_min_stamp = -math.inf
            self.patrol_arm_hold = None
            self.mode = 'background'
            self.last_commands.clear()
            self.event_seq += 1
            self.last_event = dict(seq=self.event_seq, action='clock_reset', stamp=now,
                                   previous_target_id=self.selector.selected_id)
            self.selector.request(self.selector.requested_id)
            self.selection_changed_stamp = None
            for queue in self.states.values():
                queue.clear()
        self.last_tick = now
        status = self.base_status(now, 'background_patrol')
        if not self.background_ready:
            self.patrol(status, stamp, dt)
            return
        if not self.auto_acquisition_enabled:
            status['state'] = 'patrol_auto_disabled'
            self.patrol(status, stamp, dt)
            return
        try:
            previous_id = self.selector.selected_id
            origin, origin_stamp = None, None
            if self.packet is not None and 0 <= now-float(self.packet['stamp']) <= self.max_age:
                origin_stamp = float(self.packet['stamp'])
                origin = self.selection_origin(origin_stamp)
            track, state = self.selector.choose(self.packet, now, self.max_age, self.max_coast, origin)
            status.update(state=state, target_id=self.selector.selected_id,
                          selection=dict(self.selector.detail, origin_frame='jol/base',
                                         origin_tf_stamp=origin_stamp))
            if track is None:
                if (self.selector.selected_id is not None
                        and 0 <= now-self.selector.last_seen <= self.max_coast):
                    self.hold(status, stamp)
                else:
                    self.selector.selected_id = None
                    self.patrol(status, stamp, dt)
                return
            if previous_id != self.selector.selected_id:
                self.fused_packet = None
                self.fusion_min_stamp = now
                self.selection_changed_stamp = now
            status.update(selection_changed_stamp=self.selection_changed_stamp,
                          selection_age_s=now-self.selection_changed_stamp)
            self.mode = status['mode'] = 'tracking'
            self.patrol_arm_hold = None
            source_stamp = float(self.packet['stamp'])
            # ID는 한 번만 고른다. 같은 대상의 현재점과 미래점은 조준 시각만 다르다.
            goals = {}
            for name, horizon in (('turret', self.turret_lead), ('arm', self.horizon)):
                selected, state_stamp, detail = choose_turret_state(
                    track, source_stamp, self.fused_packet, now, self.max_fusion_age, self.max_age,
                    enabled=self.fusion_enabled, selection_horizon=horizon,
                    max_exposure_age=getattr(self, 'max_camera_evidence_age', .8))
                goals[name] = dict(forecast(selected, state_stamp, now, horizon),
                                   target_id=int(track['id']), frame_id='world')
                status[name+'_state'] = detail
            prediction, turret_prediction = goals['arm'], goals['turret']
            status['prediction_state'] = status['arm_state']
            if not getattr(self, 'arm_enabled', True):
                status.pop('arm_state')
            status.update(source_stamp=source_stamp, packet_age_s=now-source_stamp,
                          measurement_age_s=now-float(track['last_seen']), observed=bool(track['observed']),
                          track_position=track['position'], track_velocity=track['velocity'],
                          position_covariance=track['position_covariance'], prediction=prediction,
                          turret_prediction=turret_prediction, target_id=int(track['id']))
            turret, turret_base, turret_q = self.measured('turret', now)
            turret_ik = solve_turret(turret_prediction['position'], turret_base,
                                    getattr(self, 'camera_mount', 'turret'),
                                    getattr(self, 'lidar_mount_mode', 'payload'))
            turret_reference, turret_reference_stamp = self.command_reference('turret', now, turret)
            turret_interval = feedback_step_interval(dt, now, turret_reference_stamp)
            turret_command = (limit_step(turret_reference, turret_ik['joints'], [self.yaw_rate, self.pitch_rate], turret_interval,
                              [-math.pi, -math.pi/6], [math.pi, math.pi/6], continuous_yaw=True)
                              if turret_ik['success'] else turret_reference)
            mechanisms = [('turret', turret, turret_ik, turret_prediction['position'], turret_command,
                           turret_reference, turret_reference_stamp, [self.yaw_rate, self.pitch_rate], turret_interval)]
            if getattr(self, 'arm_enabled', True):
                arm, arm_base, arm_q = self.measured('arm', now)
                arm_ik = solve_arm(prediction['position'], arm_base, self.rod_length)
                arm_reference, arm_reference_stamp = self.command_reference('arm', now, arm)
                arm_interval = feedback_step_interval(dt, now, arm_reference_stamp)
                arm_command = (limit_step(arm_reference, arm_ik['joints'], self.arm_rate, arm_interval, ARM_LOWER, ARM_UPPER)
                               if arm_ik['success'] else arm_reference)
                mechanisms.append(('arm', arm, arm_ik, prediction['position'], arm_command,
                                   arm_reference, arm_reference_stamp, [self.arm_rate]*4, arm_interval))
            for name, measured, solution, target, command, reference, reference_stamp, rates, interval in mechanisms:
                error = ray_error(measured['actual_origin'], measured['actual_direction'], target)
                measured.update(desired_joints=solution['joints'], command_joints=command.tolist(),
                                target_position=target, ik_success=solution['success'], ik_reason=solution['reason'],
                                ideal_origin=solution['origin'], ideal_direction=solution['direction'],
                                ideal_angle_error_deg=solution['angle_error_deg'],
                                measured_angle_error_deg=error.pop('angle_error_deg'), **error)
                measured.update(command_reference_joints=reference.tolist(),
                                command_reference_joint_stamp=reference_stamp,
                                command_feedback_age_s=now-reference_stamp,
                                command_age_compensation_s=interval-dt,
                                command_step_interval_s=interval,
                                joint_rate_limits_rad_s=rates,
                                joint_slew_lower_bound_s=joint_slew_lower_bound(
                                    measured['actual_joints'], solution['joints'], rates,
                                    continuous_yaw=name == 'turret') if solution['success'] else None,
                                joint_slew_assumption='fixed_target_and_base_velocity_limits_only')
                measured['alignment_tolerance_deg'] = 2.0
                measured['aligned'] = bool(solution['success'] and measured['ray_depth_m'] > 0
                                           and measured['measured_angle_error_deg'] <= 2.0)
                measured['alignment_state'] = 'aligned' if measured['aligned'] else 'slewing'
                status[name] = measured
            if getattr(self, 'arm_enabled', True):
                arm['hardstop_unwinding'] = bool(abs(arm_ik['joints'][0]-arm_q[0]) > math.pi)
                if arm['hardstop_unwinding']:
                    arm['alignment_state'] = 'hardstop_unwinding'
                arm['ideal_rod_tip'] = arm_ik['tip']
            # 이동 중 실측 오차와 명령 성공은 별도로 기록한다.
            if any(item[2]['success'] for item in mechanisms):
                for name, _, _, _, command, *_ in mechanisms:
                    self.command(name, command, stamp)
                self.active = True
            status.update(command_stamp=now if self.active else None,
                          command_success=all(item[2]['success'] for item in mechanisms),
                          alignment_success=all(item[1]['aligned'] for item in mechanisms),
                          command_mode='follow', processing_ms=1000*(time.perf_counter()-started))
            if not status['command_success']:
                status['state'] = 'joint_limit_or_ik_partial_hold'
            elif status.get('arm', {}).get('hardstop_unwinding'):
                status['state'] = 'arm_hardstop_unwinding'
            self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
            self.publish_markers(stamp, status)
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
            status['state'], status['error'] = 'hold_invalid_state_or_tf', str(error)
            status['hold_diagnostics'] = self.hold_diagnostics()
            if 'selection' not in status:
                status['selection'] = dict(
                    policy='manual_id' if self.selector.requested_id else 'nearest_recent_3d',
                    reference_frame='world', origin_frame='jol/base', origin=None,
                    origin_tf_stamp=None, source_stamp=None, evaluated_stamp=now,
                    selected_id=None, distance_m=None, candidate_distances_m=[],
                    reason='selection_origin_unavailable_or_invalid_packet')
            self.hold(status, stamp)

    def publish_markers(self, stamp, status):
        markers = [Marker(action=Marker.DELETEALL)]

        def line(identity, points, color, width=.08):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = 'world', stamp
            marker.ns, marker.id, marker.type = 'pointing_geometry', identity, Marker.LINE_STRIP
            marker.action, marker.pose.orientation.w, marker.scale.x = Marker.ADD, 1., width
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
            marker.lifetime = Duration(seconds=.3).to_msg()
            markers.append(marker)

        target = np.array(status['prediction']['position'])
        if 'arm' in status:
            arm = status['arm']
            origin, tip, direction = map(np.asarray, (arm['actual_origin'], arm['rod_tip'], arm['actual_direction']))
            line(1, [origin, tip], (1., .65, .1, 1.), .07)
            length = min(80., max(2., float(np.linalg.norm(target-tip))))
            line(2, [tip, tip+direction*length], (.2, 1., .85, .8))
        line(3, [status['track_position'], target], (.85, .3, 1., 1.), .12)
        sphere = Marker()
        sphere.header.frame_id, sphere.header.stamp = 'world', stamp
        sphere.ns, sphere.id, sphere.type, sphere.action = 'prediction', 4, Marker.SPHERE, Marker.ADD
        sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = map(float, target)
        sphere.pose.orientation.w = 1.
        sphere.scale.x = sphere.scale.y = sphere.scale.z = .65
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = .9, .2, 1., .9
        sphere.lifetime = Duration(seconds=.3).to_msg()
        markers.append(sphere)
        label = Marker()
        label.header.frame_id, label.header.stamp = 'world', stamp
        label.ns, label.id, label.type, label.action = 'prediction', 5, Marker.TEXT_VIEW_FACING, Marker.ADD
        label.pose.position.x, label.pose.position.y, label.pose.position.z = map(float, target+[0., 0., 12.0])
        label.pose.orientation.w, label.scale.z = 1., 1.1
        label.color.r, label.color.g, label.color.b, label.color.a = .95, .6, 1., 1.
        label.text = (f"FOLLOW ID {status['target_id']} +{status['prediction']['horizon_s']:.1f}s\n"
                      f"{status['state']} | camera {status['turret']['measured_angle_error_deg']:.1f} deg")
        label.lifetime = Duration(seconds=.3).to_msg()
        markers.append(label)
        self.markers_pub.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    node = Pointing()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # Humble may invalidate a pending DDS take when SIGINT shuts its context.
        # Live callback/runtime failures must still terminate with an error.
        if rclpy.ok():
            raise
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
