"""공통 추적 ID로 이격 관찰 경로를 제안한다. 이 노드는 비행 명령을 발행하지 않는다."""
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from .observation_path import build_plan, build_corner_plan, _xyz, _positive
from .pointing_core import forecast


class PlanningError(ValueError):
    """UI가 문장 해석 없이 차단 이유를 표시하는 작은 입력 오류 계약."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _vector(value):
    return _xyz(value, 'coordinate')


def _fresh(stamp, now, limit=.5):
    if type(stamp) not in (int, float) or not 0 <= stamp <= now or now-stamp > limit:
        raise PlanningError('input_stale', 'Missing, stale or future state timestamp')


def _covariance(value, size):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or any(isinstance(v, (bool, np.bool_))
            for v in np.asarray(value, dtype=object).flat):
        raise PlanningError('track_invalid', 'Covariance must contain numeric values')
    covariance = np.asarray(value, dtype=float)
    if (covariance.shape != (size, size) or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-7)):
        raise PlanningError('track_invalid', 'Expected finite symmetric position covariance')
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() < -1e-7:
        raise PlanningError('track_invalid', 'Position covariance is not positive semidefinite')
    return max(0., float(eigenvalues.max()))


def _padding(value):
    return 3*math.sqrt(_covariance(value, 3))


def _track_snapshot(packet, now=None):
    # 시각만 갱신된 동일 정지 상태는 유지하되, 목록·모델·공분산 변화는 무시하지 않는다.
    if not isinstance(packet, dict) or not isinstance(packet.get('tracks'), list):
        raise PlanningError('track_invalid', 'Invalid track inventory')
    stamp = _positive(packet.get('stamp'), 'track packet stamp', allow_zero=True)
    now = stamp if now is None else now
    _fresh(stamp, now)
    for track in packet['tracks']:
        if not isinstance(track, dict):
            raise PlanningError('track_invalid', 'Invalid track entry')
        _fresh(track.get('last_seen'), now, 1.)
        if track['last_seen'] > stamp:
            raise PlanningError('track_invalid', 'Observation timestamp exceeds track state timestamp')
    rows = [{key: track.get(key) for key in ('id', 'position', 'velocity',
            'state_covariance', 'prediction', 'confirmed')} for track in packet['tracks']
            if isinstance(track, dict)]
    if len(rows) != len(packet['tracks']):
        raise PlanningError('track_invalid', 'Invalid track entry')
    for row in rows:
        if isinstance(row['prediction'], dict):
            # 새 스캔의 예측 시각만 제외한다. 모델·Q·평균·공분산은 계속 비교한다.
            row['prediction'] = {key: value for key, value in row['prediction'].items() if key != 'stamp'}
    return json.dumps([packet.get('frame_id'), rows], sort_keys=True, allow_nan=False)


def observation_inputs(status, tracks, own, scene, now, standoff=10., fixed_scene=False):
    """현재·미래 표적 주위를 보호하고 상승부터 목적지까지 정적 경로를 검사한다.

    움직이는 물체의 미래 점유 공간은 증명하지 않는다. 실시간 입력의 결과는
    항상 미리보기이며, 별도 고정 장면 fixture만 격리 SITL 시험에 사용할 수 있다.
    """
    now = _positive(now, 'planning time', allow_zero=True)
    standoff = _positive(standoff, 'standoff')
    if standoff < 2.:
        raise ValueError('Simulation standoff must be finite and at least 2m')
    if type(fixed_scene) is not bool or not isinstance(scene, dict):
        raise ValueError('Explicit scene object and boolean fixed_scene are required')
    for packet in (status, tracks, own):
        if not isinstance(packet, dict):
            raise PlanningError('input_missing', 'Missing planning input')
        _fresh(packet.get('stamp'), now)
    if any(packet.get('frame_id') != 'world' for packet in (status, tracks, own, scene)):
        raise PlanningError('frame_mismatch', 'Explicit common world frame is required')
    _fresh(status.get('source_stamp'), now)
    if status['source_stamp'] > status['stamp'] or abs(status['source_stamp']-tracks['stamp']) > 1e-6:
        raise PlanningError('source_mismatch', 'Pointing status must reference this exact track packet')
    if status.get('command_mode') != 'follow':
        raise PlanningError('target_unavailable', 'No active shared target')
    identity = status.get('target_id')
    if type(identity) is not int or identity <= 0:
        raise PlanningError('target_unavailable', 'Invalid shared target ID')
    future = status.get('prediction', {})
    current = status.get('turret_prediction', {})
    for goal in (future, current):
        if not isinstance(goal, dict):
            raise ValueError('Missing pointing goal')
        if type(goal.get('target_id')) is not int or goal['target_id'] != identity or goal.get('frame_id') != 'world':
            raise ValueError('Pointing goals must share the selected ID and world frame')
        if type(goal.get('stamp')) not in (int, float) or not math.isfinite(goal['stamp']):
            raise ValueError('Missing pointing goal timestamp')
        _vector(goal.get('position'))
        _padding(goal.get('position_covariance'))
        _fresh(goal.get('state_stamp'), status['stamp'])
        propagation = goal.get('propagation_s')
        if (type(propagation) not in (int, float) or not math.isfinite(propagation)
                or propagation < 0 or abs(goal['stamp']-goal['state_stamp']-propagation) > 1e-6):
            raise PlanningError('source_mismatch', 'Inconsistent pointing state propagation time')
    horizon = future.get('horizon_s')
    if (type(horizon) not in (int, float) or not 0 <= horizon <= 3
            or type(current.get('horizon_s')) not in (int, float) or current['horizon_s'] != 0
            or abs(future['stamp']-status['stamp']-horizon) > 1e-6
            or abs(current['stamp']-status['stamp']) > 1e-6):
        raise ValueError('Inconsistent current/future pointing time')
    if any(key in own and own[key] is not True for key in ('pose_valid', 'velocity_valid')):
        raise PlanningError('own_pose_invalid', 'Observer estimator marks pose or velocity invalid')
    for key in ('pose_covariance', 'velocity_covariance'):
        if key in own:
            try:
                _covariance(own[key], 6)
            except (ValueError, TypeError, OverflowError) as error:
                raise PlanningError('own_pose_invalid', 'Observer covariance is invalid') from error
    home = _vector(own.get('position'))
    if np.linalg.norm(_vector(own.get('velocity'))) > .1:
        raise PlanningError('own_pose_moving', 'A resting observer pose is required; do not splice a launch path in flight')
    bounds = scene.get('bounds', {})
    if not isinstance(bounds, dict):
        raise ValueError('Missing flight bounds')
    lower, upper = _vector(bounds.get('min')), _vector(bounds.get('max'))
    if np.any(lower >= upper):
        raise ValueError('Invalid explicit flight bounds')
    source_obstacles = scene.get('obstacles')
    if not isinstance(source_obstacles, list):
        raise ValueError('Explicit static obstacle inventory is required')
    obstacles = list(source_obstacles)
    for obstacle in obstacles:
        if (not isinstance(obstacle, dict)
                or np.any(_vector(obstacle.get('min')) > _vector(obstacle.get('max')))):
            raise PlanningError('scene_invalid', 'Invalid static obstacle AABB')
    items = tracks.get('tracks')
    if not isinstance(items, list) or not items or len(items) > 256:
        raise ValueError('Invalid track inventory')
    target = _vector(future['position'])
    ids, selected_padding = set(), None
    for track in items:
        if not isinstance(track, dict) or not isinstance(track.get('prediction'), dict):
            raise ValueError('Missing track state or motion model')
        ident = track.get('id')
        if type(ident) is not int or ident <= 0 or ident in ids:
            raise ValueError('Missing or duplicate track ID')
        ids.add(ident)
        _fresh(track.get('last_seen'), now, 1.)
        if track['last_seen'] > tracks['stamp']:
            raise ValueError('Observation timestamp exceeds track state timestamp')
        _vector(track.get('position'))
        _vector(track.get('velocity'))
        if track.get('confirmed') is not True:
            raise ValueError('Unconfirmed obstacle state')
        if fixed_scene and np.linalg.norm(_vector(track.get('velocity'))) > 1e-9:
            raise ValueError('Fixed-scene SITL fixture cannot contain moving targets')
        estimate = forecast(track, tracks['stamp'], now, 0.)
        # 3σ는 통계적 여유이며 실제 장애물이 반드시 이 안에 있다는 보장은 아니다.
        padding = _padding(estimate['position_covariance'])
        center = _vector(estimate['position'])
        end = target if ident == identity else center
        radius = standoff if ident == identity else .7
        lower_center, upper_center = np.minimum(center, end), np.maximum(center, end)
        if ident == identity:
            # 현재/미래 융합점도 각자의 공분산만큼 보호한다. 원시 현재 P만 쓰면 과소평가된다.
            padding = max(padding, _padding(current['position_covariance']),
                          _padding(future['position_covariance']))
            if not np.allclose(_vector(status.get('track_position')), track['position'], rtol=0, atol=1e-6):
                raise ValueError('Selected source position differs from pointing input')
            if fixed_scene and (not np.allclose(target, center, rtol=0, atol=1e-6)
                    or not np.allclose(current['position'], center, rtol=0, atol=1e-6)):
                raise ValueError('Fixed-scene fixture contains inconsistent target positions')
            # 영상 융합 현재점도 보호 공간에 포함한다. 원시 LiDAR 추정과 같다고 가정하지 않는다.
            lower_center = np.minimum(lower_center, current['position'])
            upper_center = np.maximum(upper_center, current['position'])
        obstacles.append(dict(name='track_'+str(ident),
            min=(lower_center-radius-padding).tolist(), max=(upper_center+radius+padding).tolist()))
        if ident == identity:
            selected_padding = padding
    if selected_padding is None:
        raise PlanningError('target_unavailable', 'Selected ID absent from current obstacle inventory')
    # 보호 상자를 향해 들어가지 않도록 가까운 면 바깥에 관찰점을 둔다.
    direction = home[:2]-target[:2]
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise ValueError('No horizontal observation direction')
    direction /= length
    clearance = .75
    distance = (standoff+selected_padding+clearance+.1)/float(np.max(abs(direction)))
    goal = target.copy()
    goal[:2] += direction*distance
    # own의 시각만 바뀐 경우는 위치/속도 변화가 아니므로 별도로 제외한다.
    snapshot = json.dumps([identity, [current['position'], current['position_covariance']],
        [future['position'], future['position_covariance']], _track_snapshot(tracks),
        {key: value for key, value in own.items() if key != 'stamp'}, scene],
        sort_keys=True, allow_nan=False)
    return dict(home=home, goal=goal, obstacles=obstacles, lower=lower, upper=upper,
                snapshot=snapshot, identity=identity, target=target, current=current,
                future=future, bounds=bounds, clearance=clearance)


def prepare_observation(status, tracks, own, scene, now, standoff=10., fixed_scene=False):
    """신선한 공통 입력으로 정적 곡선을 검사한다. 라이브 결과는 항상 미리보기다."""
    data = observation_inputs(status, tracks, own, scene, now, standoff, fixed_scene)
    # ponytail: 단일 곡선이 막히면 거부한다. 동적 우회는 실시간 지도·재계획이 필요하다.
    planner = build_plan if fixed_scene else build_corner_plan
    plan = planner(data['home'], data['goal'], obstacles=data['obstacles'], clearance=data['clearance'])
    # 제어점의 볼록껍질로 샘플 사이를 포함한 참조점 경계 준수를 확인한다.
    for segment in plan['segments']:
        controls = np.asarray(segment['control_points'], dtype=float)
        if np.any(controls < data['lower']) or np.any(controls > data['upper']):
            raise PlanningError('path_rejected', 'Curve leaves configured flight bounds')
    identity, current, future = data['identity'], data['current'], data['future']
    target, home, goal, bounds = data['target'], data['home'], data['goal'], data['bounds']
    plan.update(target_id=identity, source_stamp=status['source_stamp'], stamp=now, planned_at=now,
        target_current_position=current['position'], target_forecast_position=target.tolist(),
        target_forecast_stamp=future['stamp'], standoff_m=standoff,
        observation_goal=goal.tolist(), home=home.tolist(), flight_bounds=bounds,
        scene_is_fixed=bool(fixed_scene), static_collision_checked=True,
        dynamic_collision_checked=False, execution_allowed=False, own_pose_valid=True,
        own_pose_validation='finite_fresh_resting_snapshot_not_independent_estimator_verification',
        execution_policy='isolated_fixed_scene_sitl' if fixed_scene else 'preview_only_requires_live_monitor',
        limitations=['Protected boxes use a fixed snapshot and a statistical covariance margin.',
                     'Moving/unobserved obstacles and actual vehicle tracking error require live validation.',
                     'The 1m ascent and flight bounds apply to the vehicle reference point.'])
    return plan


def main(args=None):
    # ROS 입출력은 이 경계에만 둬 순수 계산 함수를 별도로 검사할 수 있게 한다.
    import rclpy
    from rclpy.node import Node
    from rclpy.clock import Clock, ClockType
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry, Path as RosPath
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    class Observation(Node):
        def __init__(self):
            super().__init__('jol_observation_proposal')
            self.standoff = float(self.declare_parameter('standoff_m', 10.).value)
            scene_file = self.declare_parameter('scene_file', '').value
            if not scene_file:
                raise ValueError('scene_file with explicit bounds and obstacles is required')
            self.scene = json.loads(Path(scene_file).read_text())
            self.status, self.tracks, self.own = None, None, None
            self.track_history = deque(maxlen=16)
            self.plan, self.snapshot = None, None
            self.received_at = {}
            self.clock_stamp = self.get_clock().now().nanoseconds*1e-9
            self.clock_advanced_at = time.monotonic()
            self.plan_created_at = None
            self.plan_pub = self.create_publisher(String, '/jol/observer/plan', 10)
            self.path_pub = self.create_publisher(RosPath, '/jol/observer/path', 10)
            self.status_pub = self.create_publisher(String, '/jol/observer/status', 10)
            self.create_subscription(String, '/jol/pointing/status', lambda m: self.packet('status', m), 10)
            self.create_subscription(String, '/jol/tracks', lambda m: self.packet('tracks', m), 10)
            self.create_subscription(Odometry, '/jol/observer/odometry', self.odometry, qos_profile_sensor_data)
            self.create_service(Trigger, '/jol/observer/prepare', self.prepare)
            # /clock가 멈춰도 미리보기와 UI 준비 상태가 벽시계 기준으로 만료되어야 한다.
            self.create_timer(.1, self.expire, clock=Clock(clock_type=ClockType.STEADY_TIME))

        def packet(self, field, message):
            try:
                packet = json.loads(message.data, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                if not isinstance(packet, dict):
                    raise ValueError('Packet must be an object')
            except (ValueError, TypeError):
                packet = None
            setattr(self, field, packet)
            self.received_at[field] = time.monotonic()
            if field == 'tracks':
                if packet is None:
                    self.track_history.clear()
                else:
                    self.track_history.append(packet)

        def odometry(self, message):
            p, v = message.pose.pose.position, message.twist.twist.linear
            self.own = dict(stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9,
                frame_id=message.header.frame_id, position=[p.x, p.y, p.z], velocity=[v.x, v.y, v.z],
                provenance='nav_msgs/Odometry', estimator_validity='not_provided',
                pose_covariance_known=any(message.pose.covariance),
                pose_covariance=np.asarray(message.pose.covariance).reshape(6, 6).tolist(),
                velocity_covariance=np.asarray(message.twist.covariance).reshape(6, 6).tolist())
            self.received_at['own'] = time.monotonic()

        def freshness(self, now):
            wall = time.monotonic()
            result = {}
            for field in ('status', 'tracks', 'own'):
                packet = getattr(self, field)
                stamp = packet.get('stamp') if isinstance(packet, dict) else None
                age = now-stamp if type(stamp) in (int, float) and 0 <= stamp <= 1e15 else None
                result[field] = dict(ros_age_s=age,
                    wall_age_s=wall-self.received_at[field] if field in self.received_at else None)
            return result

        def inspect(self, for_prepare=False):
            now, wall = self.get_clock().now().nanoseconds*1e-9, time.monotonic()
            if now < self.clock_stamp:
                self.clock_stamp, self.clock_advanced_at = now, wall
                self.status, self.tracks, self.own = None, None, None
                self.track_history.clear()
                raise PlanningError('clock_reset', 'ROS clock reset; fresh inputs are required')
            if now > self.clock_stamp:
                self.clock_stamp, self.clock_advanced_at = now, wall
            # 시작할 때는 시계보다 누락된 입력을 먼저 알려 사용자가 필요한 토픽을 알 수 있게 한다.
            for field in ('status', 'tracks', 'own'):
                if not isinstance(getattr(self, field), dict):
                    raise PlanningError('input_missing', 'Missing or malformed planning input: '+field)
            if wall-self.clock_advanced_at > .5:
                raise PlanningError('clock_stopped', 'ROS clock stopped; snapshot preview is unavailable')
            if any(age['wall_age_s'] is None or age['wall_age_s'] > .5
                   for age in self.freshness(now).values()):
                raise PlanningError('input_stale', 'No fresh input received within 0.5 wall-clock seconds')
            if self.plan is not None and not for_prepare:
                if self.status.get('target_id') != self.plan['target_id'] or self.status.get('command_mode') != 'follow':
                    raise PlanningError('target_changed', 'Shared target changed or stopped')
                if now > self.plan['preview_expires_at'] or wall-self.plan_created_at > .5:
                    raise PlanningError('preview_expired', 'Snapshot preview expired')
            _fresh(self.tracks.get('stamp'), now)
            source = self.status.get('source_stamp')
            tracks = next((p for p in reversed(self.track_history) if p.get('stamp') == source), None)
            # 최신 목록이 달라졌으면 과거 source 패킷만으로 장애물을 누락시키지 않는다.
            if tracks is not None and _track_snapshot(tracks, now) != _track_snapshot(self.tracks, now):
                raise PlanningError('source_mismatch', 'Latest obstacle state changed; wait for matching pointing status')
            data = observation_inputs(self.status, tracks, self.own, self.scene, now, self.standoff)
            if self.plan is not None and not for_prepare and data['snapshot'] != self.snapshot:
                raise PlanningError('snapshot_changed', 'Input geometry, covariance, pose or scene changed')
            return now, tracks, data

        def emit_status(self, state, reason='', code='', ready=False):
            now = self.get_clock().now().nanoseconds*1e-9
            self.status_pub.publish(String(data=json.dumps(dict(stamp=now, state=state, reason=reason,
                reason_code=code, ready=ready, input_freshness=self.freshness(now),
                execution_allowed=False, dynamic_collision_checked=False), allow_nan=False)))

        def clear(self, reason, code):
            self.plan, self.snapshot = None, None
            stamp = self.get_clock().now().to_msg()
            self.plan_pub.publish(String(data=json.dumps(dict(stamp=stamp.sec+stamp.nanosec*1e-9,
                frame_id='world', valid=False, reason=reason, reason_code=code,
                execution_allowed=False, dynamic_collision_checked=False,
                execution_policy='preview_only_requires_live_monitor'))))
            path = RosPath()
            path.header.frame_id, path.header.stamp = 'world', stamp
            self.path_pub.publish(path)
            self.emit_status('invalidated', reason, code)

        def prepare(self, request, response):
            try:
                now, tracks, data = self.inspect(for_prepare=True)
                plan = prepare_observation(self.status, tracks, self.own, self.scene, now, self.standoff)
                self.plan, self.snapshot = plan, data['snapshot']
                self.plan_created_at = time.monotonic()
                plan.update(valid=True, preview_expires_at=now+.5)
                self.plan_pub.publish(String(data=json.dumps(plan, allow_nan=False)))
                path = RosPath()
                path.header.frame_id = 'world'
                path.header.stamp = self.get_clock().now().to_msg()
                for sample in plan['samples']:
                    pose = PoseStamped()
                    pose.header = path.header
                    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = sample['position']
                    pose.pose.orientation.w = 1.
                    path.poses.append(pose)
                self.path_pub.publish(path)
                self.emit_status('preview_ready', code='preview_ready', ready=True)
                response.success, response.message = True, 'Collision-checked snapshot preview; flight is not enabled'
            except (ValueError, TypeError, KeyError, OverflowError, np.linalg.LinAlgError) as error:
                self.clear(str(error), getattr(error, 'code', 'path_rejected'))
                response.success, response.message = False, str(error)
            return response

        def expire(self):
            try:
                self.inspect()
                state = 'preview_ready' if self.plan is not None else 'ready'
                self.emit_status(state, code=state, ready=True)
            except (ValueError, TypeError, KeyError, OverflowError, np.linalg.LinAlgError) as error:
                code = getattr(error, 'code', 'input_invalid')
                if self.plan is not None:
                    self.clear(str(error), code)
                else:
                    self.emit_status('blocked', str(error), code)

    rclpy.init(args=args)
    node = None
    try:
        node = Observation()
        rclpy.spin(node)
    except KeyboardInterrupt:
        # launch의 정상 SIGINT 종료는 오류 traceback을 남기지 않는다.
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
