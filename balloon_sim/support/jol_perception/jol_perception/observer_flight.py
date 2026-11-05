"""융합 관측에서 만든 제한된 관찰 경로를 같은 Gazebo 월드에서 1회 비행한다.

표적을 쫓는 재계획기는 아니다. 유효한 융합 snapshot에서 관찰점을 정한 뒤
12m 이내의 짧은 접근·정지·착륙을 수행한다. Gazebo 자기 상태는 이상적이다.
기존 observation 미리보기와 PX4 고정 fixture의 실행 정책은 바꾸지 않는다.
"""
from collections import deque
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .observation_node import observation_inputs
from .observation_path import build_plan, build_corner_plan, sample_plan


def make_flight_plan(status, tracks, own, scene, now):
    """기존 시각/ID/공분산 검사를 통과한 실제 camera_lidar 상태만 출발점으로 쓴다."""
    evidence = status.get('prediction_state', status.get('arm_state', {}))
    if evidence.get('control_state_source') != 'camera_lidar':
        raise ValueError('waiting_for_accepted_camera_lidar_control_state')
    data = observation_inputs(status, tracks, own, scene, now, standoff=10.)
    home, goal = data['home'], data['goal'].copy()
    horizontal = goal[:2]-home[:2]
    distance = np.linalg.norm(horizontal)
    if distance < 1.:
        raise ValueError('observation_goal_too_close_to_home')
    goal[:2] = home[:2]+horizontal*min(1., 12./distance)
    goal[2] = min(goal[2], 8.)
    plan = build_corner_plan(home, goal, max_speed=1.8, max_acceleration=.7, max_jerk=1.4,
                             obstacles=data['obstacles'], clearance=.75)
    # 자동 착륙 통로도 동일한 정적 장애물/표적 snapshot에서 확인한다.
    build_plan([goal[0], goal[1], home[2]], goal, obstacles=data['obstacles'], clearance=.75)
    for segment in plan['segments']:
        controls = np.asarray(segment['control_points'])
        if (np.any(controls < data['lower']) or np.any(controls > data['upper'])
                or np.any(np.linalg.norm(controls[:, :2]-home[:2], axis=1) > 12.+1e-8)):
            raise ValueError('path_leaves_bounded_flight_area')
    if plan['duration_s'] > 90:
        raise ValueError('flight_curve_too_long')
    plan.update(vehicle_limits=dict(speed_m_s=2., acceleration_m_s2=1., jerk_m_s3=2.),
                stamp=now, target_id=data['identity'], source_stamp=status['source_stamp'], planned_at=now,
                target_current_position=data['current']['position'],
                target_forecast_position=data['future']['position'],
                fusion_evidence=evidence, source='camera_lidar',
                standoff_m=10., raw_observation_goal=data['goal'].tolist(),
                static_collision_checked=True, dynamic_collision_guaranteed=False,
                execution_policy='gazebo_classic_bounded_observation',
                own_pose_source='ideal_gazebo_self_odometry',
                limitations=['Snapshot observation waypoint, not pursuit of a moving target.',
                             'Known-track proximity monitor cannot detect unobserved obstacles.',
                             'Force/torque rigid-body simulation, no PX4 or rotor dynamics.'])
    return plan


def bounded_landing_start(position, home):
    """Keep abort setpoints inside the C++ command envelope even after overshoot."""
    position, home = np.asarray(position, dtype=float).copy(), np.asarray(home, dtype=float)
    delta = position[:2]-home[:2]
    distance = np.linalg.norm(delta)
    if distance > 12.: position[:2] = home[:2]+delta*(12./distance)
    position[2] = np.clip(position[2], home[2], 8.5)
    return position


def causal_odometry(history, now, max_age=.2):
    # /clock and odometry have independent DDS delivery; defer future samples.
    return next((item for item in reversed(history) if 0 <= now-item['stamp'] <= max_age), None)


def nearest_approach(own_position, own_velocity, packet, now, horizon=1.5):
    """새로운 알려진 추적 객체와 등속도 최근접 거리를 검사한다. 정답 입력 없음."""
    closest = math.inf
    if not packet or not 0 <= now-packet['stamp'] <= .5:
        return closest
    for track in packet['tracks']:
        if not 0 <= now-float(track['last_seen']) <= 1.:
            continue
        target = np.asarray(track['position'])+(now-packet['stamp'])*np.asarray(track['velocity'])
        relative = target-own_position
        velocity = np.asarray(track['velocity'])-own_velocity
        t = np.clip(-relative@velocity/max(float(velocity@velocity), 1e-9), 0, horizon)
        closest = min(closest, float(np.linalg.norm(relative+t*velocity)))
    return closest


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry, Path as RosPath
    from geometry_msgs.msg import PoseStamped, Transform, Twist
    from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
    from std_msgs.msg import String

    class Flight(Node):
        def __init__(self):
            super().__init__('jol_observer_flight')
            if not self.get_parameter('use_sim_time').value:
                raise ValueError('Gazebo simulation time is required')
            self.output = Path(self.declare_parameter('output_dir', '/tmp/jol_observer_flight').value)
            self.output.mkdir(parents=True, exist_ok=True)
            self.scene = json.loads(Path(self.declare_parameter('scene_file', '').value).read_text())
            self.history, self.latest, self.status, self.own = deque(maxlen=40), None, None, None
            self.plan, self.phase, self.reason = None, 'waiting_for_fusion', ''
            self.phase_start = self.last_now = self.hold_start = None
            self.samples, self.events, self.path_poses = [], [], []
            self.own_history = deque(maxlen=50)
            self.command_pub = self.create_publisher(MultiDOFJointTrajectory, '/jol/sim_observer/setpoint', 1)
            self.status_pub = self.create_publisher(String, '/jol/observer/flight_status', 10)
            self.plan_pub = self.create_publisher(String, '/jol/observer/flight_plan', 10)
            self.path_pub = self.create_publisher(RosPath, '/jol/observer/flight_path', 1)
            self.flown_pub = self.create_publisher(RosPath, '/jol/observer/flown_path', 1)
            self.create_subscription(Odometry, '/jol/observer/odometry', self.odometry, 10)
            self.create_subscription(String, '/jol/tracks', self.tracks, 10)
            self.create_subscription(String, '/jol/pointing/status', self.pointing, 10)
            self.create_timer(.05, self.tick)

        def pointing(self, msg):
            try: self.status = json.loads(msg.data)
            except (ValueError, TypeError): self.status = None

        def tracks(self, msg):
            try:
                packet = json.loads(msg.data)
                if packet['frame_id'] != 'world': return
                self.latest = packet
                self.history.append(packet)
            except (ValueError, KeyError, TypeError):
                self.latest = None

        def odometry(self, msg):
            if msg.header.frame_id != 'world' or msg.child_frame_id != 'jol/observer_body': return
            p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
            values = np.array([p.x,p.y,p.z,q.x,q.y,q.z,q.w,v.x,v.y,v.z])
            if not np.isfinite(values).all() or np.linalg.norm(values[3:7]) < 1e-6: return
            velocity = Rotation.from_quat(values[3:7]).apply(values[7:])
            self.own = dict(stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                            frame_id='world', position=values[:3].tolist(), velocity=velocity.tolist())
            self.own_history.append(self.own)

        def transition(self, phase, now, reason=''):
            self.phase, self.phase_start, self.hold_start = phase, now, None
            self.reason = reason
            self.events.append(dict(stamp=now, phase=phase, reason=reason))
            self.get_logger().info('Flight '+phase+(': '+reason if reason else ''))
            self.save()

        def save(self):
            report = dict(phase=self.phase, reason=self.reason, events=self.events,
                          completed=self.phase == 'complete', simulation_only=True,
                          sensor_source='camera_lidar', own_pose_source='ideal_gazebo_self_odometry',
                          dynamic_collision_guaranteed=False, samples=len(self.samples))
            if self.plan:
                errors = [s['tracking_error_m'] for s in self.samples if s['phase'] == 'transfer']
                report.update(target_id=self.plan['target_id'], goal=self.plan['goal'],
                              transfer_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None)
            (self.output/'report.json').write_text(json.dumps(report, indent=2))
            (self.output/'telemetry.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in self.samples))

        def hold_ready(self, condition, now, duration):
            if not condition:
                self.hold_start = None
                return False
            if self.hold_start is None: self.hold_start = now
            return now-self.hold_start >= duration

        def publish_path(self, field, samples):
            path = RosPath()
            path.header.frame_id = 'world'; path.header.stamp = self.get_clock().now().to_msg()
            for position in samples:
                pose = PoseStamped(); pose.header = path.header
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, position)
                pose.pose.orientation.w = 1.
                path.poses.append(pose)
            field.publish(path)

        def command(self, sample):
            msg = MultiDOFJointTrajectory(); msg.header.frame_id = 'world'
            msg.header.stamp = self.get_clock().now().to_msg(); msg.joint_names = ['observer_body']
            point = MultiDOFJointTrajectoryPoint(); transform = Transform(); transform.rotation.w = 1.
            transform.translation.x, transform.translation.y, transform.translation.z = map(float, sample['position'])
            velocity, acceleration = Twist(), Twist()
            velocity.linear.x, velocity.linear.y, velocity.linear.z = map(float, sample['velocity'])
            acceleration.linear.x, acceleration.linear.y, acceleration.linear.z = map(float, sample['acceleration'])
            point.transforms=[transform]; point.velocities=[velocity]; point.accelerations=[acceleration]
            msg.points=[point]; self.command_pub.publish(msg)

        def tick(self):
            now = self.get_clock().now().nanoseconds*1e-9
            if now <= 0: return
            if self.last_now is not None and now < self.last_now:
                self.plan = None; self.history.clear(); self.own_history.clear(); self.status = None
                self.transition('aborted', now, 'simulation_clock_reset')
            self.last_now = now
            if self.phase in ('complete', 'aborted'):
                self.status_pub.publish(String(data=json.dumps(dict(stamp=now,phase=self.phase,reason=self.reason,
                    target_id=self.plan['target_id'] if self.plan else None,command_active=False))))
                return
            self.own = causal_odometry(self.own_history, now)
            if self.own is None:
                self.reason = 'waiting_for_fresh_own_odometry'
                if self.plan is not None:
                    self.transition('aborted', now, 'own_odometry_lost_watchdog_descent')
                return  # Gazebo command watchdog descends if flight was active.
            own_p, own_v = np.array(self.own['position']), np.array(self.own['velocity'])
            if self.plan is None:
                try:
                    if self.status is None: raise ValueError('waiting_for_pointing_status')
                    packet = next((p for p in reversed(self.history)
                                   if abs(p['stamp']-self.status.get('source_stamp', -1)) < 1e-6), None)
                    if packet is None: raise ValueError('waiting_for_matching_track_packet')
                    self.plan = make_flight_plan(self.status, packet, self.own, self.scene, now)
                    (self.output/'source_snapshot.json').write_text(json.dumps(
                        dict(status=self.status, tracks=packet, own=self.own, scene=self.scene), indent=2))
                    (self.output/'plan.json').write_text(json.dumps(self.plan, indent=2))
                    self.plan_pub.publish(String(data=json.dumps(self.plan)))
                    self.transition('takeoff', now)
                except (ValueError, TypeError, KeyError, StopIteration) as error:
                    self.reason = str(error)
                    self.status_pub.publish(String(data=json.dumps(dict(stamp=now,phase=self.phase,reason=self.reason))))
                    return
            elapsed = now-self.phase_start
            approach = nearest_approach(own_p, own_v, self.latest, now)
            if self.phase not in ('landing','abort_landing') and (approach < 10. or
                    np.linalg.norm(own_p[:2]-np.array(self.plan['home'][:2])) > 13.5 or own_p[2] > 9.):
                self.landing_start = bounded_landing_start(own_p, self.plan['home'])
                self.transition('abort_landing', now, 'known_track_proximity_or_flight_boundary')
                elapsed = 0.
            zero = [0.,0.,0.]
            if self.phase == 'takeoff':
                takeoff = self.plan['segments'][0]
                goal = np.array(takeoff['control_points'][-1])
                sample = sample_plan(self.plan, min(elapsed, takeoff['duration_s']))
                if self.hold_ready(elapsed >= takeoff['duration_s'] and np.linalg.norm(own_p-goal)<.18 and np.linalg.norm(own_v)<.2, now, 1.):
                    self.transition('transfer', now)
                elif elapsed > takeoff['duration_s']+10:
                    self.landing_start = bounded_landing_start(own_p, self.plan['home']); self.transition('abort_landing',now,'takeoff_timeout')
            elif self.phase == 'transfer':
                path_time = self.plan['segments'][0]['duration_s']+elapsed
                sample = sample_plan(self.plan,path_time)
                if path_time >= self.plan['duration_s']: self.transition('hold',now)
            elif self.phase == 'hold':
                sample = dict(position=self.plan['goal'],velocity=zero,acceleration=zero)
                if self.hold_ready(np.linalg.norm(own_p-self.plan['goal'])<.25 and np.linalg.norm(own_v)<.2,now,3.):
                    self.landing_start=np.array(self.plan['goal']);self.transition('landing',now)
                elif elapsed>20:
                    self.landing_start=bounded_landing_start(own_p, self.plan['home']);self.transition('abort_landing',now,'arrival_timeout')
            else:
                target=self.landing_start.copy(); target[2]=max(self.plan['home'][2],target[2]-.35*elapsed)
                sample=dict(position=target.tolist(),velocity=[0.,0.,-.35 if target[2]>self.plan['home'][2] else 0.],acceleration=zero)
                if self.hold_ready(own_p[2]<self.plan['home'][2]+.055 and np.linalg.norm(own_v)<.15,now,1.):
                    self.transition('complete' if self.phase=='landing' else 'aborted',now,self.reason)
                elif elapsed>35: self.transition('aborted',now,'landing_timeout')
            self.command(sample)
            record=dict(stamp=now,phase=self.phase,position=own_p.tolist(),velocity=own_v.tolist(),
                        desired_position=sample['position'],tracking_error_m=float(np.linalg.norm(own_p-sample['position'])),
                        known_track_closest_approach_m=approach if math.isfinite(approach) else None,
                        track_monitor_fresh=bool(self.latest and 0 <= now-self.latest['stamp'] <= .5),
                        target_id=self.plan['target_id'],reason=self.reason)
            self.samples.append(record); self.status_pub.publish(String(data=json.dumps(record)))
            if len(self.samples)%10==0:
                self.path_poses.append(own_p.tolist())
                self.publish_path(self.path_pub,[s['position'] for s in self.plan['samples']])
                self.publish_path(self.flown_pub,self.path_poses)
            if len(self.samples)%100==0 or self.phase in ('complete','aborted'): self.save()

    rclpy.init(args=args)
    node = Flight()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.save(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__':
    main()
