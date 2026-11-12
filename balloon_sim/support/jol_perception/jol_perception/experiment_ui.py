"""ROS 상태·영상·관절값을 보여 주고 CSV/JSONL을 저장하는 Qt 실험 화면이다.

기존 모드는 추적 서비스와 관찰 경로 미리보기만 요청한다.
static_flight_mode에서는 별도 락·미리보기·발사 서비스로 Gazebo 시험 비행을 요청한다.
원시 점군·영상·TF 전체 저장은 별도의 rosbag 녹화가 담당한다."""
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import uuid

from PyQt5 import QtCore, QtGui, QtWidgets
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger


TOPICS = {'pointing': '/jol/pointing/status', 'tracks': '/jol/tracks',
          'fused': '/jol/fused_tracks', 'camera': '/jol/camera/status',
          'detections': '/jol/camera/detections', 'metrics': '/jol/metrics',
          'observer': '/jol/observer/status', 'observer_plan': '/jol/observer/plan',
          'flight': '/jol/observer/flight_status', 'flight_plan': '/jol/observer/flight_plan',
          'static_estimate': '/jol/static_target/estimate',
          'static_sensor': '/jol/static_target/status',
          'static_status': '/jol/static_flight/status', 'static_plan': '/jol/static_flight/plan'}
STATIC_PHASES = {'detecting': '물체 검출 대기', 'coordinate_ready': '좌표 확인 · 락 가능',
                 'locked': '좌표 락 · 경로 미리보기 대기', 'preview_ready': '경로 준비 · 발사 대기',
                 'arming': 'PX4 OFFBOARD · 시동 확인 중',
                 'takeoff': '3m 수직 이륙', 'spline_tracking': '스플라인 추종',
                 'arrived': '도달 · 목표 주변 정지', 'abort_braking': '비행 중단 · 감속', 'abort_landing': '비행 중단 · 착륙',
                 'autopilot_landing': 'PX4 자동 착륙',
                 'aborted': '비행 중단 완료'}
STATIC_ACTIVE = {'arming', 'takeoff', 'spline_tracking', 'arrived', 'abort_braking', 'abort_landing', 'autopilot_landing'}
CSV_FIELDS = ['wall_time', 'monotonic_ns', 'sim_stamp', 'status_stamp', 'track_stamp',
              'mode', 'state', 'auto_acquisition_enabled', 'background_ready',
              'target_id', 'x_m', 'y_m', 'z_m', 'speed_m_s', 'nearest_surface_m',
              'control_state_source', 'arm_control_state_source', 'joint_stamp', 'yaw_deg', 'elevation_deg', 'yolo_ready',
              'inference_ms', 'track_count', 'pointing_rx_age_s', 'tracks_rx_age_s',
              'position_stamp', 'position_source_age_s', 'track_source_age_s',
              'control_status_age_s', 'tracks_communication_fresh',
              'position_observed', 'last_observation_stamp', 'last_observation_age_s',
              'surface_stamp', 'surface_age_s', 'image_stamp', 'image_exposure_age_s', 'image_rx_age_s',
              'detection_stamp', 'detection_age_s', 'detection_count', 'fusion_stamp',
              'fusion_exposure_stamp', 'fusion_exposure_age_s',
              'fusion_geometry_status', 'fusion_accepted_count', 'selected_fusion_accepted',
              'selected_fusion_current_valid', 'lidar_minus_fused_stamp_s', 'parse_error_count',
              'static_phase', 'ned_x_m', 'ned_y_m', 'ned_z_m', 'takeoff_height_m',
              'flight_tracking_error_m', 'drone_x_m', 'drone_y_m', 'drone_z_m',
              'camera_centered', 'camera_alignment_error_deg', 'gimbal_tracking']


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def number(value, digits=2):
    return f'{float(value):.{digits}f}' if finite(value) else '—'


def static_autopilot_text(status):
    mode = status.get('autopilot_mode')
    mode = mode if isinstance(mode, str) and mode else '모드 미수신'
    armed = status.get('armed')
    arm_text = '무장' if armed is True else '비무장' if armed is False else '무장 미확인'
    connection = '' if status.get('autopilot_connected') is True else '연결 두절 · ' \
        if status.get('autopilot_connected') is False else '연결 미확인 · '
    vtol = ''
    if status.get('aircraft_type') == 'tailsitter20cm_reference':
        state = status.get('vtol_state')
        label = {1: '전진 전환 중', 2: '호버 복귀 중', 3: '호버', 4: '전진 비행'}.get(state) if type(state) is int else None
        vtol = ' · '+(label or 'VTOL 미수신')
    return 'PX4 · '+connection+mode+' · '+arm_text+vtol


def vector(values):
    return (isinstance(values, (tuple, list)) and len(values) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and finite(v) for v in values))


def bbox(values):
    return (isinstance(values, (tuple, list)) and len(values) == 4
            and all(type(v) in (int, float) and finite(v) for v in values)
            and values[0] >= 0 and values[1] >= 0 and values[2] > 0 and values[3] > 0)


def source_age(now, stamp):
    return float(now)-float(stamp) if finite(stamp) else None


def reject_json_constant(value):
    raise ValueError('nonfinite JSON constant '+value)


def measurement_age_text(age, kind='측정'):
    if not finite(age):
        return '시각 미확인'
    if age < -.05:
        return f'미래 시각 {abs(age):.2f}s'
    return f'{max(0., age):.2f}s 전 {kind}'


def surface_distance(track, communication_fresh):
    """선택 패널과 표의 표면 거리에 같은 최신성 기준을 적용한다."""
    value = track.get('nearest_distance')
    return value if communication_fresh and track.get('observed') is True and finite(value) else None


def fused_state(track, packet):
    """영상 노출값과 현재로 예측한 융합값의 좌표·시각을 함께 유지한다."""
    exposure = track.get('source_exposure_stamp', packet.get('source_exposure_stamp', packet.get('stamp')))
    current = track.get('current_state', {})
    valid = (track.get('temporal_fusion', {}).get('valid') is True
             and vector(current.get('position')) and finite(track.get('current_state_stamp')))
    return (current['position'], track['current_state_stamp'], exposure, True) if valid else (
        track.get('position'), track.get('stamp', exposure), exposure, False)


class SessionRecorder:
    """새 세션에만 기록하고 디스크 오류는 화면에 남긴다."""
    def __init__(self):
        self.directory = None
        self.handles = {}
        self.error = ''
        self.rows = 0

    @property
    def active(self):
        return bool(self.handles)

    def start(self, root, use_sim_time, context=None):
        self.close()
        self.error, self.rows = '', 0
        try:
            root = Path(root).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            session = datetime.now(timezone.utc).strftime('ui_%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:6]
            self.directory = root / session
            self.directory.mkdir(exist_ok=False)
            for name in ('telemetry.jsonl', 'events.jsonl', 'selected_target.csv'):
                self.handles[name] = (self.directory/name).open('x', encoding='utf-8', newline='', buffering=1)
            self.csv = csv.DictWriter(self.handles['selected_target.csv'], fieldnames=CSV_FIELDS)
            self.csv.writeheader()
            self.manifest = dict(format_version=3, created_utc=utc_now(), use_sim_time=use_sim_time,
                    clocks={'sim_stamp': 'ROS clock seconds; simulation time only if use_sim_time=true',
                            'stamp': 'source message acquisition/status stamp in ROS seconds',
                            'monotonic_ns': 'local monotonic receive/write clock, not a ROS stamp',
                            'wall_time': 'UTC ISO 8601 wall clock'},
                    csv_columns=CSV_FIELDS,
                    freshness={'tracks_communication_fresh': 'wall receive age <= stale_timeout; not measurement freshness',
                               'position_source_age_s': 'ROS now minus displayed state stamp; a prediction is not a new observation',
                               'last_observation_age_s': 'ROS now minus last_seen for the displayed state; null when unavailable',
                               'track_source_age_s': 'ROS now minus latest LiDAR state packet stamp',
                               'surface_age_s': 'ROS now minus surface_stamp; only observed, communication-fresh tracks',
                               'image_exposure_age_s': 'ROS now minus displayed annotated Image header stamp',
                               'image_rx_age_s': 'wall age since annotated Image receipt; distinct from exposure age',
                               'lidar_minus_fused_stamp_s': 'displayed LiDAR state stamp minus displayed fusion state stamp; exposure and valid replay states retain their own stamps',
                               'control_status_age_s': 'ROS now minus pointing status stamp'},
                    raw_bag_included=False, csv_hz_max=10,
                    telemetry_topics=TOPICS, pointcloud_subscription=False, context=context or {})
            with (self.directory/'manifest.json').open('x', encoding='utf-8') as stream:
                json.dump(self.manifest, stream, ensure_ascii=False, allow_nan=False, indent=2)
            return True
        except OSError as error:
            self.fail(error)
            return False

    def update_context(self, context):
        if not self.active or context == self.manifest['context']:
            return
        try:
            self.manifest['context'] = context
            temporary = self.directory/'manifest.pending'
            with temporary.open('w', encoding='utf-8') as stream:
                json.dump(self.manifest, stream, ensure_ascii=False, allow_nan=False, indent=2)
            temporary.replace(self.directory/'manifest.json')
        except (OSError, ValueError, TypeError) as error:
            self.fail(error)

    def fail(self, error):
        self.error = str(error)
        self.close()

    def write(self, filename, row):
        if not self.active:
            return
        try:
            if filename.endswith('.csv'):
                self.csv.writerow(row)
            else:
                self.handles[filename].write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            self.rows += 1
        except (OSError, ValueError, TypeError) as error:
            self.fail(error)

    def close(self):
        for stream in self.handles.values():
            try:
                stream.close()
            except OSError as error:
                self.error = str(error)
        self.handles.clear()


class ExperimentNode(Node):
    def __init__(self, **kwargs):
        super().__init__('jol_experiment_ui', **kwargs)
        self.record_dir = str(self.declare_parameter('record_dir', str(Path.home()/'ros2_nub/results/ui')).value)
        self.auto_record = bool(self.declare_parameter('auto_record', False).value)
        self.raw_bag_uri = str(self.declare_parameter('raw_bag_uri', '').value)
        self.stale_timeout = float(self.declare_parameter('stale_timeout', 2.).value)
        self.service_timeout = float(self.declare_parameter('service_timeout', 3.).value)
        self.static_flight_mode = bool(self.declare_parameter('static_flight_mode', False).value)
        if not .2 <= self.stale_timeout <= 60 or not .1 <= self.service_timeout <= 60:
            raise ValueError('Invalid UI stale/service timeout')
        self.packets, self.received = {}, {}
        self.latest_image = None
        self.image_history = deque(maxlen=20)
        self.image_sequence = 0
        self.background_ready = None
        self.joints = {}
        self.joint_stamp = None
        self.pending = None
        self.command_tone = 'neutral'
        self.command_message = '명령 대기 · 추적 종료 후 자동 재획득은 재개 버튼으로 허용'
        self.events = deque(maxlen=5000)
        self.recorder = SessionRecorder()
        self.history = {}
        self.last_track_stamp = None
        self.last_ros_stamp = None
        self.last_observer_plan = None
        self.static_launch_latched = False
        self.static_refresh_after_ns = 0
        self.static_actual_path = deque(maxlen=4000)
        self.last_error = ''
        self.parse_errors = {}
        try:
            source = Path(__file__).resolve()
            self.source_info = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        except OSError as error:
            self.source_info = dict(path=__file__, sha256=None, error=str(error))
        self.control_clients = {action: self.create_client(Trigger, '/jol/pointing/'+action)
                        for action in ('stop_tracking', 'resume_auto')}
        self.control_clients['prepare_observation'] = self.create_client(Trigger, '/jol/observer/prepare')
        if self.static_flight_mode:
            self.control_clients.update({f'static_{action}': self.create_client(Trigger, '/jol/static_flight/'+action)
                                         for action in ('lock', 'unlock', 'preview', 'launch', 'abort', 'reset')})
            self.control_clients['static_start_acquisition'] = self.create_client(Trigger, '/jol/static_target/start_acquisition')
            self.control_clients['static_reset_acquisition'] = self.create_client(Trigger, '/jol/static_target/reset_acquisition')
            self.command_message = '탐지 시작 → 카메라 탐색 → 영상 정렬 → 크기 기반 좌표 → 락 → 발사'
        self.target_pub = self.create_publisher(Int32, '/jol/pointing/target_id', 1)
        for key, topic in TOPICS.items():
            self.create_subscription(String, topic, lambda msg, key=key: self.receive_packet(key, msg), 5)
        sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/jol/camera/annotated', self.receive_image, sensor)
        self.create_subscription(JointState, '/jol/gimbal/joint_states' if self.static_flight_mode else
                                 '/jol/turret/joint_states', self.receive_joints, sensor)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/jol/background_ready', self.receive_background, latched)
        if self.auto_record:
            self.start_recording(self.record_dir, method='auto_record')

    def clocks(self):
        return dict(wall_time=utc_now(), monotonic_ns=time.monotonic_ns(),
                    sim_stamp=self.get_clock().now().nanoseconds*1e-9)

    def age(self, key):
        received = self.received.get(key)
        return (time.monotonic_ns()-received)*1e-9 if received is not None else math.inf

    def fresh(self, key):
        return self.age(key) <= self.stale_timeout

    def state_fresh(self, key):
        """재수신은 원본 측정 시각을 새롭게 만들지 않는다."""
        age = source_age(self.clocks()['sim_stamp'], self.packets.get(key, {}).get('stamp'))
        return self.fresh(key) and finite(age) and -.05 <= age <= self.stale_timeout

    def fusion_packet(self):
        camera = self.packets.get('camera', {})
        if (not self.fresh('fused') or (camera and not self.state_fresh('camera'))
                or camera.get('inference_ready') is False
                or camera.get('fusion_ready') is False):
            return {}
        return self.packets.get('fused', {})

    def check_clock(self):
        now = self.clocks()['sim_stamp']
        if self.last_ros_stamp is not None and now < self.last_ros_stamp:
            self.event('clock_reset', 'observed', method='ros_clock',
                       previous_stamp=self.last_ros_stamp, current_stamp=now)
            self.packets.clear()
            self.received.clear()
            self.history.clear()
            self.last_track_stamp = None
            self.latest_image = None
            self.image_history.clear()
            self.image_sequence += 1
            self.joints.clear()
            self.joint_stamp = None
            self.background_ready = None
            self.last_observer_plan = None
            self.static_launch_latched = False
            self.static_refresh_after_ns = 0
            self.static_actual_path.clear()
        self.last_ros_stamp = now

    def receive_packet(self, key, msg):
        self.check_clock()
        try:
            packet = json.loads(msg.data, parse_constant=reject_json_constant)
            if (not isinstance(packet, dict) or not isinstance(packet.get('stamp'), (int, float))
                    or isinstance(packet.get('stamp'), bool) or not finite(packet['stamp'])):
                raise ValueError('stamp or object invalid')
            # 빈 검출/융합 대기 패킷에는 변환할 좌표가 없고 frame_id도 비어 있을 수 있다.
            if (key in ('tracks', 'fused', 'pointing') and (key == 'pointing' or packet.get('tracks'))
                    and packet.get('frame_id', 'world') != 'world'):
                raise ValueError('World 화면에는 world 좌표가 필요합니다')
            if key in ('tracks', 'fused'):
                tracks = packet.get('tracks')
                if not isinstance(tracks, list) or len(tracks) > 256:
                    raise ValueError('tracks invalid')
                for track in tracks:
                    if (not isinstance(track, dict) or not isinstance(track.get('id'), int)
                            or isinstance(track.get('id'), bool) or track['id'] <= 0
                            or not vector(track.get('position')) or not vector(track.get('velocity'))):
                        raise ValueError('track coordinates invalid')
                    if 'class_name' in track and not isinstance(track['class_name'], str):
                        raise ValueError('track class label invalid')
                    if 'fusion_update' in track and not isinstance(track['fusion_update'], dict):
                        raise ValueError('fusion_update invalid')
                    if any(field in track and not isinstance(track[field], dict)
                           for field in ('current_state', 'temporal_fusion')):
                        raise ValueError('replayed fusion state invalid')
                    if 'last_seen' in track and not finite(track['last_seen']):
                        raise ValueError('last_seen invalid')
            if key == 'pointing':
                if any(not isinstance(packet.get(field, {}), dict)
                       for field in ('turret_state', 'arm_state', 'selection', 'turret', 'arm',
                                     'prediction', 'turret_prediction')):
                    raise ValueError('control state invalid')
                identity = packet.get('target_id')
                if identity is not None and (not isinstance(identity, int) or isinstance(identity, bool) or identity <= 0):
                    raise ValueError('control target ID invalid')
                if any(field in packet and not vector(packet[field]) for field in ('track_position', 'track_velocity')):
                    raise ValueError('control coordinates invalid')
            if key == 'flight':
                if (packet.get('phase') not in ('waiting_for_fusion','takeoff','transfer','hold','landing',
                                               'abort_landing','complete','aborted')
                        or not isinstance(packet.get('reason', ''), str)
                        or any(field in packet and not vector(packet[field]) for field in ('position','velocity','desired_position'))):
                    raise ValueError('비행 상태 형식 오류')
            if key == 'flight_plan':
                if (packet.get('frame_id') != 'world' or packet.get('execution_policy') != 'gazebo_classic_bounded_observation'
                        or not vector(packet.get('home')) or not vector(packet.get('goal'))
                        or not isinstance(packet.get('samples'), list) or len(packet['samples']) > 20000
                        or any(not isinstance(row, dict) or not vector(row.get('position')) for row in packet['samples'])):
                    raise ValueError('Gazebo 비행 경로 형식 오류')
            if key == 'static_status':
                if (packet.get('phase') not in STATIC_PHASES
                        or not isinstance(packet.get('reason', ''), str)
                        or any(type(packet.get(field)) is not bool for field in
                               ('coordinate_valid', 'locked', 'plan_ready', 'controller_ready', 'can_launch'))
                        or any(field in packet and packet[field] is not None and not vector(packet[field])
                               for field in ('target_world', 'target_ned', 'position_world', 'velocity_world'))
                        or packet['coordinate_valid'] and not vector(packet.get('target_world'))):
                    raise ValueError('정적 비행 상태·좌표 형식 오류')
                if packet.get('lock_bbox') is not None and (not bbox(packet['lock_bbox'])
                        or type(packet.get('target_id')) is not int or packet['target_id'] <= 0
                        or type(packet.get('lock_bbox_stamp')) not in (int, float)
                        or not finite(packet['lock_bbox_stamp'])):
                    raise ValueError('락한 카메라 검출의 박스·ID·노출 시각 형식 오류')
                size = packet.get('detection_image_size')
                if size is not None and (not isinstance(size, (list, tuple)) or len(size) != 2
                        or any(type(value) is not int or value <= 0 for value in size)):
                    raise ValueError('검출 영상 크기 형식 오류')
                if any(packet.get(field) is not None and type(packet[field]) is not bool
                       for field in ('camera_centered', 'gimbal_tracking')):
                    raise ValueError('짐벌 영상 정렬 상태 형식 오류')
                if ('can_lock' in packet and type(packet['can_lock']) is not bool
                        or 'lock_block_reason' in packet and not isinstance(packet['lock_block_reason'], str)):
                    raise ValueError('좌표 락 준비 상태·사유 형식 오류')
                if ('flight_backend' in packet and packet['flight_backend'] not in ('native', 'mavros_px4')
                        or any(field in packet and not isinstance(packet[field], str)
                               for field in ('feedback_controller', 'own_pose_source'))):
                    raise ValueError('비행 제어·측위 출처 형식 오류')
                error = packet.get('camera_alignment_error_deg')
                if error is not None and (type(error) not in (int, float) or not finite(error) or not 0 <= error <= 180):
                    raise ValueError('짐벌 영상 중심 잔차 형식 오류')
            if key == 'static_plan':
                if type(packet.get('valid')) is not bool or packet['valid'] and (
                        packet.get('frame_id') != 'world'
                        or not vector(packet.get('home')) or not vector(packet.get('goal'))
                        or not isinstance(packet.get('samples'), list) or len(packet['samples']) > 20000
                        or any(not isinstance(row, dict) or not vector(row.get('position')) for row in packet['samples'])):
                    raise ValueError('정적 비행 경로 형식 오류')
            if key == 'observer':
                if (not isinstance(packet.get('state'), str)
                        or not isinstance(packet.get('reason', ''), str)
                        or not isinstance(packet.get('reason_code', ''), str)
                        or type(packet.get('ready', False)) is not bool
                        or not isinstance(packet.get('input_freshness', {}), dict)
                        or packet.get('execution_allowed') is not False
                        or packet.get('dynamic_collision_checked') is not False):
                    raise ValueError('관찰 상태 계약 오류: 미리보기 전용 상태가 필요합니다')
            if key == 'observer_plan':
                if type(packet.get('valid')) is not bool or not isinstance(packet.get('reason', ''), str):
                    raise ValueError('관찰 계획 유효성 표시가 없습니다')
                if packet['valid'] and (packet.get('frame_id') != 'world'
                        or not vector(packet.get('home')) or not vector(packet.get('goal'))
                        or any(type(packet.get(field)) not in (int, float) or not finite(packet[field])
                               for field in ('planned_at', 'preview_expires_at'))
                        or not packet['planned_at'] <= packet['preview_expires_at']
                        or packet.get('execution_allowed') is not False
                        or packet.get('dynamic_collision_checked') is not False):
                    raise ValueError('관찰 계획의 좌표·만료 시각·실행 불허·충돌 검사 상태가 올바르지 않습니다')
                samples = packet.get('samples', [])
                if packet['valid'] and (not isinstance(samples, list) or len(samples) > 10000
                        or any(not isinstance(sample, dict) or not vector(sample.get('position')) for sample in samples)):
                    raise ValueError('관찰 경로 표본의 XYZ 좌표가 올바르지 않습니다')
            if key == 'camera' and 'pipeline' in packet:
                pipeline = packet['pipeline']
                if (not isinstance(pipeline, dict) or not isinstance(pipeline.get('counters', {}), dict)
                        or any(type(value) is not int or value < 0
                               for value in pipeline.get('counters', {}).values())):
                    raise ValueError('camera pipeline counters invalid')
            if key == 'detections' and (not isinstance(packet.get('detections'), list)
                    or len(packet['detections']) > 4096
                    or any(not isinstance(item, dict) for item in packet['detections'])):
                raise ValueError('detections invalid')
            self.packets[key] = packet
            self.received[key] = time.monotonic_ns()
            if key == 'static_status':
                if packet['phase'] != 'preview_ready':
                    self.static_launch_latched = False
                if packet['phase'] in STATIC_ACTIVE and vector(packet.get('position_world')):
                    self.static_actual_path.append(packet['position_world'])
            if key == 'observer_plan' and packet['valid']:
                self.last_observer_plan = packet
            if key in ('camera', 'fused') and not self.fusion_packet():
                # 처리 실패 뒤 늦게 도착한 융합값도 다시 표시하지 않는다.
                self.packets.pop('fused', None)
                self.received.pop('fused', None)
            if key == 'tracks':
                stamp = float(packet['stamp'])
                if self.last_track_stamp is not None and stamp < self.last_track_stamp:
                    self.history.clear()
                self.last_track_stamp = stamp
                for track in packet['tracks']:
                    points = self.history.setdefault(track['id'], deque(maxlen=150))
                    points.append((stamp, *track['position'][:2]))
                    while points and stamp-points[0][0] > 15.:
                        points.popleft()
                self.history = {k: v for k, v in self.history.items() if v and stamp-v[-1][0] <= 15.}
            self.recorder.write('telemetry.jsonl', dict(self.clocks(), topic=TOPICS[key], data=packet))
            if key in ('camera', 'fused', 'detections'):
                self.recorder.update_context(self.recording_context())
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            if key in ('observer', 'observer_plan', 'static_status', 'static_plan'):
                self.packets.pop(key, None)
                self.received.pop(key, None)
            self.last_error = f'{key}: {error}'
            self.parse_errors[key] = self.parse_errors.get(key, 0)+1
            self.event('input_parse_error', 'error', method='ros_subscription', topic=TOPICS[key],
                       error=str(error), count=self.parse_errors[key], success=False)

    def receive_image(self, msg):
        self.check_clock()
        self.latest_image = msg
        self.image_history.append(msg)
        self.image_sequence += 1
        self.received['image'] = time.monotonic_ns()

    def static_lock_annotation(self):
        """백엔드가 현재 락에 대응시킨 실제 검출만 사용하며 과거 박스를 투영하지 않는다."""
        if not self.static_flight_mode or not self.state_fresh('static_status'):
            return None
        status = self.packets['static_status']
        age = source_age(self.clocks()['sim_stamp'], status.get('lock_bbox_stamp'))
        if (status.get('locked') is not True or status.get('coordinate_valid') is not True
                or not bbox(status.get('lock_bbox')) or not finite(age) or not -.05 <= age <= self.stale_timeout):
            return None
        return status

    def display_image(self):
        annotation = self.static_lock_annotation()
        if annotation:
            for image in reversed(self.image_history):
                stamp = image.header.stamp.sec+image.header.stamp.nanosec*1e-9
                if abs(stamp-annotation['lock_bbox_stamp']) <= 1e-6:
                    return image
        return self.latest_image

    def receive_joints(self, msg):
        self.check_clock()
        if len(msg.name) == len(msg.position):
            values = dict(zip(msg.name, msg.position))
            if self.static_flight_mode:
                values = {legacy: values.get(actual) for legacy, actual in
                          (('yaw_joint', 'gimbal_yaw_joint'), ('pitch_joint', 'gimbal_pitch_joint'))}
            if all(finite(values.get(k)) for k in ('yaw_joint', 'pitch_joint')):
                self.joints = values
                self.joint_stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                self.received['joints'] = time.monotonic_ns()

    def receive_background(self, msg):
        self.check_clock()
        self.background_ready = bool(msg.data)

    def event(self, action, phase, method='qt_button', **details):
        row = dict(self.clocks(), action=action, phase=phase, method=method,
                   observed_event_seq=self.packets.get('pointing', {}).get('event_seq'), **details)
        self.events.append(row)
        self.recorder.write('events.jsonl', row)
        return row

    def start_recording(self, directory, method='qt_button'):
        ok = self.recorder.start(directory, self.get_parameter('use_sim_time').value, self.recording_context())
        if ok:
            self.record_dir = str(directory)
            self.event('recording', 'start', method=method, success=True)
        return ok

    def recording_context(self):
        camera = self.packets.get('camera', {})
        fusion = self.packets.get('fused', {})
        detections = self.packets.get('detections', {})
        return dict(raw_bag_uri=self.raw_bag_uri or None, raw_bag_uri_verified=False,
                    ui_source=self.source_info,
                    reported_model=camera.get('model') or detections.get('model'),
                    reported_requested_provider=camera.get('requested_provider'),
                    reported_active_providers=camera.get('active_providers'),
                    reported_subtype_validation=camera.get('subtype_validation'),
                    reported_calibration_valid=fusion.get('calibration_valid', camera.get('calibrated')),
                    reported_calibration_source=fusion.get('calibration_source'),
                    metadata_origin='ROS camera/status, camera/detections and fused_tracks; model hash and calibration contents unverified')

    def stop_recording(self, method='qt_button'):
        self.event('recording', 'stop', method=method, success=True)
        self.recorder.close()

    def request_command(self, action):
        if self.pending is not None:
            return
        if action.startswith('static_') and not self.static_action_allowed(action):
            self.command_message = '요청 차단 · 최신 좌표·락·경로·제어 상태를 확인하세요'
            self.command_tone = 'warning'
            self.event(action, 'blocked', success=False)
            return
        self.send_command(action)

    def send_command(self, action):
        """Send an allowed UI command or the sensor reset after Mission accepts reset."""
        request_id = uuid.uuid4().hex
        self.event(action, 'request', request_id=request_id, success=None)
        client = self.control_clients[action]
        if not client.service_is_ready():
            self.command_message = ('미리보기 요청 실패 · 관찰 서비스 연결 안 됨' if action == 'prepare_observation'
                                    else '명령 실패 · 제어 서비스 연결 안 됨')
            self.command_tone = 'error'
            self.event(action, 'unavailable', request_id=request_id, success=False)
            return
        try:
            future = client.call_async(Trigger.Request())
            self.pending = dict(action=action, request_id=request_id, future=future,
                                deadline=time.monotonic()+self.service_timeout)
            if action == 'static_launch':
                # 응답 수락과 상태 전이 사이에도 같은 계획을 두 번 발사하지 않는다.
                self.static_launch_latched = True
            self.command_message = '경로 미리보기 응답 대기 중…' if action == 'prepare_observation' else '제어 서비스 응답 대기 중…'
            self.command_tone = 'warning'
        except Exception as error:
            self.command_message = '명령 전송 실패: '+str(error)
            self.command_tone = 'error'
            self.event(action, 'error', request_id=request_id, success=False, error=str(error))

    def poll_command(self):
        if self.pending is None:
            return
        request = self.pending
        future = request['future']
        if future.done():
            self.pending = None
            try:
                response = future.result()
                if response is None:
                    raise RuntimeError('empty service response')
                prefix = ('미리보기 요청 수락 · 계획 상태 확인' if response.success else '미리보기 요청 실패') \
                    if request['action'] == 'prepare_observation' else ('제어 응답 성공' if response.success else '제어 응답 실패')
                self.command_message = prefix+' · '+response.message
                self.command_tone = 'neutral' if response.success else 'error'
                if request['action'].startswith('static_') and response.success:
                    # 응답 직전 상태로 다음 버튼을 활성화하지 않고 새 상태 발행을 기다린다.
                    self.static_refresh_after_ns = time.monotonic_ns()
                if request['action'] == 'static_launch' and not response.success:
                    self.static_launch_latched = False
                if request['action'] == 'static_launch' and response.success:
                    self.static_actual_path.clear()
                if request['action'] in ('static_reset', 'static_unlock') and response.success:
                    self.static_launch_latched = False
                    if request['action'] == 'static_reset':
                        self.static_actual_path.clear()
                self.event(request['action'], 'response', request_id=request['request_id'],
                           success=bool(response.success), message=response.message)
                if request['action'] == 'static_reset' and response.success:
                    self.send_command('static_reset_acquisition')
            except Exception as error:
                self.command_message = '제어 응답 오류: '+str(error)
                self.command_tone = 'error'
                self.event(request['action'], 'error', request_id=request['request_id'], success=False, error=str(error))
        elif time.monotonic() >= request['deadline']:
            self.pending = None
            self.control_clients[request['action']].remove_pending_request(future)
            future.cancel()
            self.command_message = '응답 시간 초과 · 적용 여부 미확인, 실제 상태를 확인하세요'
            self.command_tone = 'warning'
            self.event(request['action'], 'timeout', request_id=request['request_id'], success=None)

    def static_action_allowed(self, action):
        """Qt 버튼·키보드·직접 요청에 같은 최신 상태 검사를 적용한다."""
        if (not self.static_flight_mode or self.pending is not None or not self.state_fresh('static_status')
                or self.received.get('static_status', 0) < self.static_refresh_after_ns):
            return False
        status = self.packets['static_status']
        phase = status['phase']
        if action == 'static_start_acquisition':
            sensor=self.packets.get('static_sensor',{})
            return (self.state_fresh('static_sensor') and sensor.get('calibration_valid') is True
                    and sensor.get('acquisition',{}).get('stage')=='waiting_start'
                    and not status.get('locked') and phase not in STATIC_ACTIVE)
        if action == 'static_launch':
            return (phase == 'preview_ready' and status['can_launch'] and status['coordinate_valid']
                    and status['locked'] and status['plan_ready'] and status['controller_ready']
                    and status.get('camera_centered') is True
                    and vector(status.get('target_world')) and vector(status.get('target_ned'))
                    and not self.static_launch_latched)
        if action == 'static_lock':
            return (status['coordinate_valid'] and status['controller_ready'] and vector(status.get('target_ned'))
                    and status.get('camera_centered') is True
                    and status.get('can_lock', True) is True
                    and not status['locked'] and phase in ('coordinate_ready', 'detecting'))
        if action == 'static_unlock':
            return status['locked'] and phase not in STATIC_ACTIVE
        if action == 'static_preview':
            return status['coordinate_valid'] and status['locked'] and phase in ('locked', 'preview_ready')
        if action == 'static_abort':
            return phase in STATIC_ACTIVE and phase not in ('abort_braking', 'abort_landing', 'autopilot_landing')
        if action != 'static_reset' or phase in STATIC_ACTIVE:
            return False
        position, velocity = status.get('position_world'), status.get('velocity_world')
        own_stamp = status.get('own_stamp')
        age = source_age(self.clocks()['sim_stamp'], own_stamp) if type(own_stamp) in (int, float) else None
        reported_age = status.get('odometry_age_s')
        if ('own_stamp' not in status and type(reported_age) in (int, float)
                and finite(reported_age) and reported_age >= 0):
            age = source_age(self.clocks()['sim_stamp'], status['stamp'])+reported_age
        # Match Mission's causal_odometry (.2 s), ground height and speed guard.
        if (type(age) not in (int, float) or not finite(age) or not 0 <= age <= .2
                or not vector(position) or not vector(velocity)
                or position[2] > .25 or math.hypot(*velocity) > .15):
            return False
        return (status.get('flight_backend') != 'mavros_px4' or
                status.get('autopilot_connected') is True and status.get('armed') is False
                and type(status.get('landed_state')) is int and status['landed_state'] == 1)

    def static_launch_block_reason(self):
        if not self.state_fresh('static_status'):
            return '비행 상태 수신 / 원본 시각 확인'
        status = self.packets['static_status']
        if self.pending is not None:
            return '명령 응답 대기'
        if self.received.get('static_status', 0) < self.static_refresh_after_ns:
            return '응답 후 실제 상태 확인 중'
        if status['phase'] in STATIC_ACTIVE:
            return STATIC_PHASES[status['phase']]
        if self.static_launch_latched:
            return '발사 처리 중 · 중복 발사 차단'
        if not status['locked'] and status.get('can_lock') is False:
            detail = status.get('lock_block_reason') or status.get('reason')
            return '센서 좌표·안전 여유 확인 대기'+(' · '+detail if detail else '')
        for field, text in (('coordinate_valid', '카메라 단안 좌표 미확인'),
                            ('camera_centered', '짐벌 영상 중심 정렬 대기'),
                            ('locked', '좌표 락 필요'), ('plan_ready', '경로 준비 필요'),
                            ('controller_ready', '비행 제어기 준비 대기'), ('can_launch', '발사 조건 미충족')):
            if status.get(field) is not True:
                detail = status.get('launch_block_reason') or status.get('reason')
                return text+(' · '+detail if detail else '')
        if not self.control_clients['static_launch'].service_is_ready():
            return '발사 서비스 연결 대기'
        return ''

    def select_target(self, identity):
        if self.pending is not None:
            return
        self.target_pub.publish(Int32(data=identity))
        self.event('select_target', 'request', target_id=identity, success=None)
        self.command_message = f'ID {identity} 선택 요청 전송 · 실제 선택은 상태 표시에서 확인'
        self.command_tone = 'neutral'

    def snapshot(self):
        self.check_clock()
        clocks = self.clocks()
        # 재수신된 과거 관절값을 현재 각도로 표시하지 않는다. 시계 허용오차는 다른 입력과 같다.
        joint_age = source_age(clocks['sim_stamp'], self.joint_stamp)
        joint_current = (self.fresh('joints') and finite(joint_age)
                         and -.05 <= joint_age <= self.stale_timeout)
        status = self.packets.get('pointing', {})
        packet = self.packets.get('tracks', {})
        camera = self.packets.get('camera', {})
        identity = status.get('target_id') if self.state_fresh('pointing') else None
        tracks = packet.get('tracks', [])
        target = next((t for t in tracks if t['id'] == identity), {})
        position = status.get('track_position', target.get('position', []))
        position_stamp = status.get('source_stamp') if 'track_position' in status else packet.get('stamp')
        observed = status.get('observed') if 'track_position' in status else target.get('observed')
        # 예측 메시지가 갱신돼도 마지막 실측 나이를 초기화하지 않는다.
        last_seen = (float(status['stamp'])-float(status['measurement_age_s'])
                     if 'track_position' in status and finite(status.get('measurement_age_s'))
                     else target.get('last_seen') if position_stamp == packet.get('stamp') else None)
        velocity = status.get('track_velocity', target.get('velocity', []))
        source = status.get('turret_state', {}).get('control_state_source', '')
        distance = surface_distance(target, self.fresh('tracks'))
        displayed_image = self.display_image()
        image_stamp = (displayed_image.header.stamp.sec+displayed_image.header.stamp.nanosec*1e-9
                       if displayed_image is not None else None)
        detection = self.packets.get('detections', {})
        fusion = self.fusion_packet()
        selected_fusion = next((t for t in fusion.get('tracks', []) if t['id'] == identity), {})
        _, fused_stamp, exposure_stamp, current_valid = fused_state(selected_fusion, fusion)
        row = dict(clocks, status_stamp=status.get('stamp'), track_stamp=packet.get('stamp'),
                   mode=status.get('mode', ''), state=status.get('state', ''),
                   auto_acquisition_enabled=status.get('auto_acquisition_enabled'),
                   background_ready=self.background_ready, target_id=identity,
                   x_m=position[0] if vector(position) else None,
                   y_m=position[1] if vector(position) else None,
                   z_m=position[2] if vector(position) else None,
                   speed_m_s=math.sqrt(sum(float(v)**2 for v in velocity)) if vector(velocity) else None,
                   nearest_surface_m=distance,
                   control_state_source=source,
                   arm_control_state_source=status.get('arm_state', {}).get('control_state_source', ''),
                   joint_stamp=self.joint_stamp,
                   yaw_deg=math.degrees(self.joints['yaw_joint']) if joint_current else None,
                   elevation_deg=-math.degrees(self.joints['pitch_joint']) if joint_current else None,
                   yolo_ready=bool(camera.get('ready')) if self.state_fresh('camera') else None,
                   inference_ms=camera.get('inference_ms'), track_count=len(tracks),
                   pointing_rx_age_s=self.age('pointing') if finite(self.age('pointing')) else None,
                   tracks_rx_age_s=self.age('tracks') if finite(self.age('tracks')) else None)
        row.update(position_stamp=position_stamp if vector(position) else None,
                   position_source_age_s=source_age(row['sim_stamp'], position_stamp) if vector(position) else None,
                   track_source_age_s=source_age(row['sim_stamp'], packet.get('stamp')),
                   control_status_age_s=source_age(row['sim_stamp'], status.get('stamp')),
                   tracks_communication_fresh=self.fresh('tracks'),
                   position_observed=observed, last_observation_stamp=last_seen,
                   last_observation_age_s=source_age(row['sim_stamp'], last_seen),
                   surface_stamp=packet.get('stamp') if distance is not None else None,
                   surface_age_s=source_age(row['sim_stamp'], packet.get('stamp')) if distance is not None else None,
                   image_stamp=image_stamp, image_exposure_age_s=source_age(row['sim_stamp'], image_stamp),
                   image_rx_age_s=self.age('image') if finite(self.age('image')) else None,
                   detection_stamp=detection.get('stamp'),
                   detection_age_s=source_age(row['sim_stamp'], detection.get('stamp')),
                   detection_count=len(detection.get('detections', [])) if self.fresh('detections') else None,
                   fusion_stamp=fused_stamp, fusion_geometry_status=fusion.get('fusion_reason'),
                   fusion_exposure_stamp=exposure_stamp,
                   fusion_exposure_age_s=source_age(row['sim_stamp'], exposure_stamp),
                   fusion_accepted_count=sum(t.get('fusion_update', {}).get('accepted') is True
                                             for t in fusion.get('tracks', [])) if fusion else None,
                   selected_fusion_accepted=selected_fusion.get('fusion_update', {}).get('accepted'),
                   selected_fusion_current_valid=current_valid if selected_fusion else None,
                   lidar_minus_fused_stamp_s=source_age(position_stamp, fused_stamp)
                                             if finite(position_stamp) and selected_fusion else None,
                   parse_error_count=sum(self.parse_errors.values()))
        if not self.state_fresh('pointing'):
            for key in ('x_m', 'y_m', 'z_m', 'speed_m_s', 'nearest_surface_m', 'control_state_source',
                        'position_stamp', 'position_source_age_s', 'position_observed',
                        'last_observation_stamp', 'last_observation_age_s', 'surface_stamp', 'surface_age_s',
                        'selected_fusion_accepted', 'selected_fusion_current_valid',
                        'lidar_minus_fused_stamp_s', 'arm_control_state_source'):
                row[key] = None
        if self.static_flight_mode:
            mission = self.packets.get('static_status', {}) if self.state_fresh('static_status') else {}
            estimate = self.packets.get('static_estimate', {}) if self.state_fresh('static_estimate') else {}
            coordinate = mission.get('target_world') if mission.get('coordinate_valid') else None
            ned = mission.get('target_ned') if mission.get('coordinate_valid') else None
            position = mission.get('position_world')
            row.update(static_phase=mission.get('phase'), target_id=1 if vector(coordinate) else None,
                       state=mission.get('phase'), mode='static_flight',
                       status_stamp=mission.get('stamp'), takeoff_height_m=mission.get('takeoff_height_m'),
                       flight_tracking_error_m=mission.get('tracking_error_m'),
                       camera_centered=mission.get('camera_centered'),
                       camera_alignment_error_deg=mission.get('camera_alignment_error_deg'),
                       gimbal_tracking=mission.get('gimbal_tracking'))
            row.update(nearest_surface_m=estimate.get('nearest_distance_m') if estimate.get('valid') is True else None,
                       surface_stamp=estimate.get('stamp') if estimate.get('valid') is True else None,
                       surface_age_s=source_age(row['sim_stamp'],estimate.get('stamp')) if estimate.get('valid') is True else None)
            for values, keys in ((coordinate, ('x_m', 'y_m', 'z_m')),
                                 (ned, ('ned_x_m', 'ned_y_m', 'ned_z_m')),
                                 (position, ('drone_x_m', 'drone_y_m', 'drone_z_m'))):
                row.update({key: values[index] if vector(values) else None for index, key in enumerate(keys)})
        return row


class WorldPlot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 135)
        self.setAccessibleName('월드 좌표 XY 궤적, 20·40·60미터 거리 눈금')
        self.history, self.tracks, self.selected = {}, [], None
        self.yaw = None
        self.prediction, self.arm = None, {}
        self.observation_path, self.observation_goal = [], None
        self.flight_path_display = False
        self.static_flight_display = False
        self.actual_path = []
        self.setToolTip('황색 점선: 계획 경로 · 마름모: 목표점.\n'
                        'XY 투영만 표시하므로 수직 상승·고도·충돌 안전을 판단할 수 없습니다. '
                        '만료된 경로는 지우며 비행 명령은 보내지 않습니다.')
        self.stale = True

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#202833'))
        width, height = self.width(), self.height()
        center = QtCore.QPointF(width/2, height/2)
        scale = min(width-55, height-48)/140
        def point(x, y):
            return QtCore.QPointF(center.x()+x*scale, center.y()-y*scale)
        painter.setPen(QtGui.QPen(QtGui.QColor('#354252'), 1, QtCore.Qt.DotLine))
        for coordinate in range(-60, 61, 20):
            painter.drawLine(point(-65, coordinate), point(65, coordinate))
            painter.drawLine(point(coordinate, -65), point(coordinate, 65))
        painter.setPen(QtGui.QPen(QtGui.QColor('#8293a8'), 1))
        # 작은 화면에서는 눈금 글자만 줄이고 월드 범위는 유지한다.
        for coordinate in ((-40, 40) if scale < 1.2 else (-60, -40, -20, 20, 40, 60)):
            painter.drawText(point(coordinate, 0)+QtCore.QPointF(2, 14), str(coordinate))
            painter.drawText(point(0, coordinate)+QtCore.QPointF(4, -2), str(coordinate))
        painter.drawLine(point(-65, 0), point(65, 0))
        painter.drawLine(point(0, -65), point(0, 65))
        painter.setPen(QtGui.QColor('#a9b7c6'))
        painter.drawText(12, 20, 'World XY   /   계획 · 실제 비행' if self.static_flight_display else 'World XY   /   최근 15초')
        painter.drawText(width-36, int(center.y()-7), '+X')
        painter.drawText(int(center.x()+7), 18, '+Y')
        for identity, samples in self.history.items():
            color = QtGui.QColor('#69b1ff' if identity == self.selected else '#a9b7c6')
            color.setAlpha(80 if self.stale else 170)
            painter.setPen(QtGui.QPen(color, 2))
            path = QtGui.QPainterPath()
            for index, sample in enumerate(samples):
                (path.moveTo if index == 0 else path.lineTo)(point(sample[1], sample[2]))
            painter.drawPath(path)
        for track in self.tracks:
            p = point(*track['position'][:2])
            color = QtGui.QColor('#69b1ff') if track['id'] == self.selected else QtGui.QColor('#a9b7c6')
            painter.setPen(QtGui.QPen(color, 2))
            painter.setBrush(color if track.get('observed') and not self.stale else QtCore.Qt.NoBrush)
            painter.drawEllipse(p, 5, 5)
            painter.drawText(p+QtCore.QPointF(9, -9), f'ID {track["id"]}')
        painter.setPen(QtGui.QPen(QtGui.QColor('#e6edf3'), 2))
        painter.setBrush(QtGui.QColor('#e6edf3'))
        painter.drawEllipse(center, 4, 4)
        if finite(self.yaw):
            angle = math.radians(self.yaw)
            painter.drawLine(center, point(18*math.cos(angle), 18*math.sin(angle)))
        if vector(self.prediction) and not self.stale:
            p = point(*self.prediction[:2])
            painter.setPen(QtGui.QPen(QtGui.QColor('#f0a96b'), 2))
            painter.drawLine(p+QtCore.QPointF(-5, -5), p+QtCore.QPointF(5, 5))
            painter.drawLine(p+QtCore.QPointF(-5, 5), p+QtCore.QPointF(5, -5))
        origin, direction = self.arm.get('actual_origin'), self.arm.get('actual_direction')
        if vector(origin) and vector(direction) and not self.stale:
            painter.setPen(QtGui.QPen(QtGui.QColor('#f0a96b'), 2, QtCore.Qt.DashLine))
            painter.drawLine(point(*origin[:2]), point(origin[0]+20*direction[0], origin[1]+20*direction[1]))
        if self.observation_path:
            painter.setPen(QtGui.QPen(QtGui.QColor('#edbd63'), 2, QtCore.Qt.DashLine))
            painter.setBrush(QtCore.Qt.NoBrush)
            path = QtGui.QPainterPath()
            for index, position in enumerate(self.observation_path):
                (path.moveTo if index == 0 else path.lineTo)(point(*position[:2]))
            painter.drawPath(path)
            p = point(*self.observation_goal[:2])
            painter.drawPolygon(QtGui.QPolygonF([p+QtCore.QPointF(x, y)
                                                for x, y in ((0, -6), (6, 0), (0, 6), (-6, 0))]))
            painter.drawText(12, height-30, '황색 점선 / ◇ Gazebo 비행 계획' if self.flight_path_display
                             else '황색 점선 / ◇ 관찰 미리보기 · 실행 불가')
        if self.actual_path:
            painter.setPen(QtGui.QPen(QtGui.QColor('#69b1ff'), 2))
            path = QtGui.QPainterPath()
            for index, position in enumerate(self.actual_path):
                (path.moveTo if index == 0 else path.lineTo)(point(*position[:2]))
            painter.drawPath(path)
        if self.stale:
            painter.setPen(QtGui.QColor('#f0c674'))
            painter.drawText(12, height-12, '미수신 / 오래된 상태 · 과거 궤적')
        else:
            painter.setPen(QtGui.QColor('#a9b7c6'))
            painter.drawText(12, height-12, '● 센서 목표   ━ 실제 비행' if self.static_flight_display else
                             '● 관측  ○ 예측   × 미래점'+('   주황 점선: 막대 방향' if self.arm else ''))


class FlightPath3D(QtWidgets.QWidget):
    """World XYZ display only; interaction never publishes a flight command."""
    def __init__(self):
        super().__init__()
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.setObjectName('flightPath3D')
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 4), facecolor='#202833'))
        self.canvas.setMinimumSize(220, 140)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.axes = self.canvas.figure.add_subplot(111, projection='3d', facecolor='#202833')
        self.axes.set(xlabel='World X [m]', ylabel='World Y [m]', zlabel='World Z [m]')
        self.axes.tick_params(colors='#a9b7c6', labelsize=8)
        for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
            axis.label.set_color('#c8d4e1')
            axis.pane.set_facecolor('#252e3a')
            axis.pane.set_edgecolor('#48586b')
        self.plan_line, = self.axes.plot([], [], [], '--', color='#edbd63', label='Planned path', linewidth=1.8)
        self.actual_line, = self.axes.plot([], [], [], color='#69b1ff', label='Actual flight', linewidth=2)
        self.takeoff_line, = self.axes.plot([], [], [], ':', color='#72d6aa', label='3 m vertical takeoff')
        self.target_marker, = self.axes.plot([], [], [], 'x', color='#ff8c91', label='Sensor surface', markersize=7)
        self.goal_marker, = self.axes.plot([], [], [], 'D', color='#72d6aa', label='Planned goal', markersize=5)
        legend = self.axes.legend(loc='upper center', bbox_to_anchor=(.5, 1.04), ncol=3, fontsize=7.5,
                                  facecolor='#202833', edgecolor='#48586b', columnspacing=1.)
        for text in legend.get_texts():
            text.set_color('#e6edf3')
        self.canvas.figure.subplots_adjust(left=.03, right=.90, bottom=.08, top=.98)
        self.planned, self.actual, self.target, self.home, self.goal = [], [], None, None, None
        self._plan_signature = None
        self._draw_signature = None
        self._last_draw = -math.inf
        self._needs_fit = True
        self.summary = QtWidgets.QLabel('World XYZ · 좌표와 경로 수신 대기')
        self.summary.setProperty('role', 'muted')
        self.summary.setWordWrap(True)
        self.reset_button = QtWidgets.QPushButton('시점 초기화')
        self.reset_button.setObjectName('reset3DViewButton')
        self.reset_button.clicked.connect(self.reset_view)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel('드래그: 회전 · 휠: 확대/축소'), 1)
        controls.addWidget(self.reset_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(controls)
        layout.addWidget(self.summary)
        self.canvas.mpl_connect('scroll_event', self.zoom)
        self.reset_view()

    def set_path(self, plan, actual, target, stale=False):
        had_target = self.target is not None
        self.planned = [row['position'] for row in plan.get('samples', [])]
        self.actual = list(actual)
        self.target = target if vector(target) else None
        self.home, self.goal = plan.get('home'), plan.get('goal')
        signature = tuple(tuple(point) for point in self.planned)
        self._needs_fit |= signature != self._plan_signature or self.target is not None and not had_target
        self._plan_signature = signature
        self.summary.setText('World XYZ [m] · 계획 '+str(len(self.planned))+'점 / 실제 '+str(len(self.actual))+'점'+
                             (' · 오래된 상태 / 과거 궤적' if stale else
                              ' · 센서 추정 가상 목표점' if plan.get('target_is_virtual_waypoint') else ' · 센서 표면 / 비접촉 도달점'))
        tone(self.summary, 'warning' if stale else 'neutral')
        self.redraw()

    @staticmethod
    def line_points(line, points):
        # Keep endpoints while bounding canvas work; stored coordinates stay intact.
        step = max(1, math.ceil(len(points)/1000))
        displayed = points[::step]
        if points and (not displayed or displayed[-1] != points[-1]):
            displayed.append(points[-1])
        line.set_data_3d(*zip(*displayed) if displayed else ([], [], []))

    def fit_view(self):
        points = self.planned+self.actual+[point for point in (self.home, self.goal, self.target) if vector(point)]
        if not points:
            points = [[-5., -5., 0.], [5., 5., 10.]]
        ranges = [(min(point[axis] for point in points), max(point[axis] for point in points)) for axis in range(3)]
        for limits, setter in zip(ranges, (self.axes.set_xlim3d, self.axes.set_ylim3d, self.axes.set_zlim3d)):
            low, high = limits
            margin = max(.5, (high-low)*.08)
            setter(low-margin, high+margin)
        self.axes.set_box_aspect((4., 3., 2.5))
        self._needs_fit = False

    def redraw(self, force=False):
        now = time.monotonic()
        if not force and (not self.isVisible() or now-self._last_draw < .2):
            return
        signature = self.draw_signature()
        if not force and not self._needs_fit and signature == self._draw_signature:
            return
        self.line_points(self.plan_line, self.planned)
        self.line_points(self.actual_line, self.actual)
        self.line_points(self.target_marker, [self.target] if self.target else [])
        self.line_points(self.goal_marker, [self.goal] if vector(self.goal) else [])
        vertical = [self.home, [self.home[0], self.home[1], self.home[2]+3.]] if vector(self.home) else []
        self.line_points(self.takeoff_line, vertical)
        if self._needs_fit:
            self.fit_view()
        self.canvas.draw_idle()
        self._draw_signature = self.draw_signature()
        self._last_draw = now

    def draw_signature(self):
        return (self._plan_signature, tuple(tuple(point) for point in self.actual),
                tuple(self.target or ()), tuple(self.home or ()), tuple(self.goal or ()),
                self.axes.elev, self.axes.azim, getattr(self.axes, 'roll', 0.),
                tuple(self.axes.get_xlim3d()), tuple(self.axes.get_ylim3d()), tuple(self.axes.get_zlim3d()),
                self.canvas.width(), self.canvas.height())

    def reset_view(self):
        self.axes.view_init(elev=25., azim=-60.)
        self.fit_view()
        self.redraw(force=True)

    def zoom(self, event):
        if event.inaxes is not self.axes or not event.step:
            return
        factor = .85 if event.step > 0 else 1/.85
        for getter, setter in ((self.axes.get_xlim3d, self.axes.set_xlim3d),
                               (self.axes.get_ylim3d, self.axes.set_ylim3d),
                               (self.axes.get_zlim3d, self.axes.set_zlim3d)):
            low, high = getter(); center = (low+high)/2
            setter(center+(low-center)*factor, center+(high-center)*factor)
        self.canvas.draw_idle()


def tone(widget, value):
    """색 상태가 바뀔 때만 스타일을 갱신한다."""
    if widget.property('tone') != value:
        widget.setProperty('tone', value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class MetricCard(QtWidgets.QFrame):
    def __init__(self, title):
        super().__init__()
        self.setProperty('role', 'metric')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(3)
        label = QtWidgets.QLabel(title)
        label.setProperty('role', 'muted')
        self.value = QtWidgets.QLabel('—')
        self.value.setProperty('role', 'metricValue')
        self.value.setAccessibleName(title)
        self.detail = QtWidgets.QLabel('수신 대기')
        self.detail.setProperty('role', 'muted')
        layout.addWidget(label)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def update_value(self, value, detail, state='neutral'):
        self.value.setText(value)
        self.detail.setText(detail)
        tone(self.value, state)


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, node, start_timer=True):
        super().__init__()
        self.node = node
        self.executor = SingleThreadedExecutor(context=node.context)
        self.executor.add_node(node)
        self.last_image_sequence = -1
        self.rendered_image = None
        self.lock_box_drawn = False
        self.last_csv_write = -math.inf
        self.pixmap = None
        self.setWindowTitle('NUB · 카메라 단독 비행 실험' if node.static_flight_mode else 'JOL · LiDAR / YOLO 실험 모니터')
        self.setObjectName('jolExperimentDashboard')
        self.resize(1360, 920)
        self.setMinimumSize(960, 700)
        # QGroundControl's role-based dark/enabled/disabled palette informs these local Qt colors.
        # https://github.com/mavlink/qgroundcontrol/blob/master/src/QmlControls/QGCPalette.cc
        self.setStyleSheet('''
            QWidget { background:#171c23; color:#e6edf3;
                font-family:"Noto Sans CJK KR"; font-size:13px; }
            QLabel { background:transparent; }
            QLabel[role="muted"], QLabel[tone="neutral"] { color:#a9b7c6; }
            QLabel[role="heading"] { font-size:12px; font-weight:600; }
            QLabel[role="mono"] { font-family:"DejaVu Sans Mono"; color:#c8d4e1; }
            QLabel[role="badge"] { padding:2px 8px; }
            QLabel[tone="good"] { color:#72d6aa; }
            QLabel[tone="warning"] { color:#f0c674; }
            QLabel[tone="error"] { color:#ff8c91; }
            QLabel#titleLabel { font-size:15px; font-weight:600; }
            QLabel#modeLabel { font-size:12px; font-weight:600; padding:2px 8px;
                border-left:1px solid #394553; }
            QFrame[role="panel"], QFrame#controlBar, QFrame#metricsBar, QFrame#feedbackBar {
                background:#202833; border:1px solid #394553; border-radius:2px; }
            QFrame[role="metric"] { background:transparent; border:0; }
            QLabel[role="metricValue"] { font-family:"DejaVu Sans Mono";
                font-size:19px; font-weight:600; color:#e6edf3; }
            QPushButton { background:#202833; border:1px solid #48586b; border-radius:3px;
                padding:5px 10px; min-height:20px; }
            QPushButton:hover { background:#2b394a; border-color:#6d91bd; }
            QPushButton:pressed { background:#33485f; }
            QPushButton:focus, QSpinBox:focus, QLineEdit:focus { border:2px solid #82baff; }
            QPushButton:disabled { color:#8793a3; background:#252c36; border-color:#36404d; }
            QPushButton#resumeAutoButton { background:#2864a6; border-color:#20588f; color:white; }
            QPushButton#resumeAutoButton:hover { background:#20588f; }
            QPushButton#resumeAutoButton:disabled { background:#252c36; color:#8793a3; border-color:#36404d; }
            QPushButton#recordButton[tone="warning"] { color:#ff8c91; border-color:#ff8c91; }
            QPushButton#staticLockButton { background:#234a73; border-color:#5e8bbb; }
            QPushButton#staticLockButton:hover { background:#2d5b88; }
            QPushButton#staticLaunchButton { background:#217a5b; border-color:#51b68c; font-weight:600; }
            QPushButton#staticLaunchButton:hover { background:#278b68; }
            QPushButton#staticAbortButton:enabled { color:#ff8c91; border-color:#bc6269; }
            QPushButton#staticLaunchButton:disabled, QPushButton#staticLockButton:disabled {
                background:#252c36; color:#8793a3; border-color:#36404d; }
            QLineEdit, QSpinBox { background:#202833; border:1px solid #48586b;
                border-radius:2px; padding:4px; min-height:20px; selection-background-color:#345a83; }
            QTabWidget::pane { background:#202833; border:1px solid #394553; top:-1px; }
            QTabBar::tab { background:#252e3a; color:#acbbcc; padding:6px 16px;
                border:1px solid #394553; border-bottom:0; }
            QTabBar::tab:selected { background:#202833; color:#90c2ff; border-top:2px solid #2864a6; }
            QTabBar::tab:hover { background:#2d3948; }
            QTableWidget { background:#202833; alternate-background-color:#252e3a;
                border:0; gridline-color:#354252; selection-background-color:#294462; }
            QHeaderView::section { background:#2a3442; color:#c8d4e1; padding:6px;
                border:0; border-bottom:1px solid #394553; }
            QTableCornerButton::section { background:#2a3442; border:0; }
            QPlainTextEdit, QTextEdit { background:#171e27; color:#c8d4e1;
                border:1px solid #394553; selection-background-color:#345a83; }
            QScrollBar:vertical { background:#202833; width:11px; }
            QScrollBar::handle:vertical { background:#53657b; min-height:24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
            QScrollArea { border:0; background:#202833; }
            QScrollArea > QWidget > QWidget { background:#202833; }
            QSplitter::handle { background:#354252; }
            QSplitter::handle:hover { background:#6d91bd; }
            QToolTip { background:#202833; color:#e6edf3; border:1px solid #6d91bd; padding:5px; }
        ''')
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('NUB / 카메라 단독 비행' if node.static_flight_mode else 'JOL / 추적 실험')
        title.setObjectName('titleLabel')
        header.addWidget(title)
        header.addStretch()
        self.mode_label = QtWidgets.QLabel('연결 대기')
        self.mode_label.setObjectName('modeLabel')
        header.addWidget(self.mode_label)
        self.clock_label = QtWidgets.QLabel('ROS — s')
        self.clock_label.setProperty('role', 'mono')
        self.clock_label.setMinimumWidth(120)
        self.clock_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header.addWidget(self.clock_label)
        layout.addLayout(header)

        health = QtWidgets.QHBoxLayout()
        health.setSpacing(8)
        self.health_labels = {}
        for key in ('tracks', 'camera', 'image', 'background'):
            label = QtWidgets.QLabel('수신 대기')
            label.setProperty('role', 'badge')
            self.health_labels[key] = label
            health.addWidget(label)
        health.addStretch()
        self.record_badge = QtWidgets.QLabel('UI 기록 대기')
        self.record_badge.setProperty('role', 'badge')
        health.addWidget(self.record_badge)
        layout.addLayout(health)

        metrics_frame = QtWidgets.QFrame()
        metrics_frame.setObjectName('metricsBar')
        metrics = QtWidgets.QHBoxLayout(metrics_frame)
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setSpacing(0)
        self.target_card = MetricCard('선택 대상')
        self.speed_card = MetricCard('추정 속도')
        self.distance_card = MetricCard('센서 → 물체 표면')
        self.observation_card = MetricCard('마지막 LiDAR 실측')
        for card in (self.target_card, self.speed_card, self.distance_card, self.observation_card):
            metrics.addWidget(card, 1)
        layout.addWidget(metrics_frame)
        feedback_frame = QtWidgets.QFrame()
        feedback_frame.setObjectName('feedbackBar')
        feedback = QtWidgets.QHBoxLayout(feedback_frame)
        feedback.setContentsMargins(12, 5, 12, 5)
        self.arm_feedback = QtWidgets.QLabel('막대 방향: 미수신')
        self.arm_feedback.setObjectName('armMeasuredFeedback')
        self.turret_feedback = QtWidgets.QLabel('카메라 방향: 미수신')
        self.prediction_label = QtWidgets.QLabel('예측점 —')
        self.prediction_label.setProperty('role', 'mono')
        for label in (self.arm_feedback, self.turret_feedback, self.prediction_label):
            label.setWordWrap(True)
            feedback.addWidget(label, 1)
        layout.addWidget(feedback_frame)

        observer_frame = QtWidgets.QFrame()
        observer_frame.setProperty('role', 'panel')
        observer_row = QtWidgets.QHBoxLayout(observer_frame)
        observer_row.setContentsMargins(10, 5, 10, 5)
        self.preview_button = QtWidgets.QPushButton('경로 미리보기 요청')
        self.preview_button.setObjectName('observationPreviewButton')
        self.preview_button.setAccessibleName('관찰 경로 미리보기 요청, 비행 명령 없음')
        self.preview_button.setToolTip('관찰 서비스에 경로 계산만 요청합니다. Tab 이동 후 Space로 요청할 수 있습니다.')
        self.preview_button.clicked.connect(lambda: self.command('prepare_observation'))
        observer_row.addWidget(self.preview_button)
        self.observer_summary = QtWidgets.QLabel('관찰 입력 대기')
        self.observer_summary.setObjectName('observationSummary')
        self.observer_summary.setWordWrap(True)
        observer_row.addWidget(self.observer_summary, 1)
        observer_policy = QtWidgets.QLabel('비행 허용: 아니요\n동적 충돌: 미검증')
        self.observer_policy = observer_policy
        observer_policy.setObjectName('observationPolicy')
        tone(observer_policy, 'warning')
        observer_row.addWidget(observer_policy)
        layout.addWidget(observer_frame)

        def panel(title, subtitle):
            frame = QtWidgets.QFrame()
            frame.setProperty('role', 'panel')
            box = QtWidgets.QVBoxLayout(frame)
            box.setContentsMargins(10, 6, 10, 6)
            box.setSpacing(4)
            label = QtWidgets.QLabel(title)
            label.setProperty('role', 'heading')
            box.addWidget(label)
            description = QtWidgets.QLabel(subtitle)
            description.setProperty('role', 'muted')
            box.addWidget(description)
            return frame, box

        self.content_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.content_splitter.setChildrenCollapsible(False)
        panels = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        panels.setChildrenCollapsible(False)
        camera_box, camera_layout = panel('짐벌 카메라 · 물체 검출' if node.static_flight_mode else '카메라 · YOLO',
                                         '원본 비율 / 노출 시각 기준')
        self.camera_label = QtWidgets.QLabel('영상 수신 대기\n카메라가 연결되면 영상이 표시됩니다.')
        self.camera_label.setObjectName('cameraPreview')
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setMinimumSize(400, 135)
        self.camera_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.camera_label.setStyleSheet('background:#252a31; color:#d6dde6;')
        camera_layout.addWidget(self.camera_label, 1)
        self.camera_summary = QtWidgets.QLabel('영상 수신 대기')
        self.camera_summary.setWordWrap(True)
        camera_layout.addWidget(self.camera_summary)
        panels.addWidget(camera_box)
        world_box, world_layout = panel('위치 · World XYZ (m)' if node.static_flight_mode else '위치 · World XY (m)',
                                       '황색: 계획 / 파랑: 실제 / 붉은 X: 센서 표면' if node.static_flight_mode else
                                       '실선: 최근 15초 궤적 / 3D 점군·박스: RViz')
        self.plot = WorldPlot()
        if node.static_flight_mode:
            self.flight_3d = FlightPath3D()
            self.world_views = QtWidgets.QStackedWidget()
            self.world_views.addWidget(self.flight_3d)
            self.world_views.addWidget(self.plot)
            selectors = QtWidgets.QHBoxLayout()
            for index, label in enumerate(('3D 궤적', 'XY 투영')):
                button = QtWidgets.QPushButton(label)
                button.setObjectName('world3DButton' if index == 0 else 'worldXYButton')
                button.clicked.connect(lambda checked=False, index=index: self.set_world_view(index))
                selectors.addWidget(button)
            selectors.addStretch()
            world_layout.addLayout(selectors)
            world_layout.addWidget(self.world_views, 1)
        else:
            world_layout.addWidget(self.plot, 1)
        self.world_summary = QtWidgets.QLabel('LiDAR 수신 대기')
        self.world_summary.setWordWrap(True)
        world_layout.addWidget(self.world_summary)
        panels.addWidget(world_box)
        panels.setSizes([720, 560])
        self.content_splitter.addWidget(panels)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName('detailTabs')
        self.tabs.setMinimumHeight(128)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setObjectName('trackTable')
        self.table.setHorizontalHeaderLabels(['ID', '관측 상태', 'X (m)', 'Y (m)', 'Z (m)', '속도 (m/s)', '표면 거리 (m)', '카메라 식별'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(7, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.tabs.addTab(self.table, '트랙 목록')

        def details_tab(title):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            content = QtWidgets.QWidget()
            box = QtWidgets.QVBoxLayout(content)
            box.setContentsMargins(16, 12, 16, 12)
            box.setAlignment(QtCore.Qt.AlignTop)
            scroll.setWidget(content)
            self.tabs.addTab(scroll, title)
            return box

        self.target_label = QtWidgets.QLabel('선택 대상 없음')
        self.target_label.setObjectName('selectedTargetLabel')
        self.target_label.setWordWrap(True)
        self.target_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        details_tab('선택 대상 · 융합').addWidget(self.target_label)
        diagnostic_layout = details_tab('센서 · 시간 진단')
        self.status_detail = QtWidgets.QLabel('ROS 상태 미수신')
        self.status_detail.setWordWrap(True)
        self.status_detail.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.camera_status = QtWidgets.QLabel('YOLO 상태 대기')
        self.camera_status.setWordWrap(True)
        self.camera_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        diagnostic_layout.addWidget(self.status_detail)
        diagnostic_layout.addWidget(self.camera_status)
        self.observer_detail = QtWidgets.QLabel('관찰 노드와 자기 기체 위치 입력을 기다립니다.')
        self.observer_detail.setObjectName('observationDetail')
        self.observer_detail.setWordWrap(True)
        self.observer_detail.setTextFormat(QtCore.Qt.PlainText)
        self.observer_detail.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        details_tab('관찰 경로').addWidget(self.observer_detail)
        recording_layout = details_tab('실험 기록')
        recording_layout.addWidget(QtWidgets.QLabel('JSONL / CSV 저장 위치  ·  원시 점군·영상 rosbag은 별도 녹화'))
        recording = QtWidgets.QHBoxLayout()
        self.directory_input = QtWidgets.QLineEdit(node.record_dir)
        self.directory_input.setObjectName('recordDirectoryInput')
        self.directory_input.setAccessibleName('실험 기록 상위 폴더')
        self.directory_button = QtWidgets.QPushButton('폴더 선택')
        self.directory_button.clicked.connect(self.choose_directory)
        recording.addWidget(self.directory_input, 1)
        recording.addWidget(self.directory_button)
        recording_layout.addLayout(recording)
        self.record_label = QtWidgets.QLabel('기록 대기')
        self.record_label.setWordWrap(True)
        self.record_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        recording_layout.addWidget(self.record_label)
        self.content_splitter.addWidget(self.tabs)
        self.content_splitter.setSizes([430, 170])
        layout.addWidget(self.content_splitter, 1)

        control_bar = QtWidgets.QFrame()
        control_bar.setObjectName('controlBar')
        commands = QtWidgets.QHBoxLayout(control_bar)
        commands.setContentsMargins(8, 6, 8, 6)
        commands.setSpacing(8)
        self.stop_button = QtWidgets.QPushButton('추적 종료 · 순찰')
        self.stop_button.setObjectName('stopTrackingButton')
        self.stop_button.setToolTip('추적을 종료하고 상향 20°로 360° 순찰합니다. 자동 재획득은 재개 전까지 중지됩니다.')
        self.stop_button.clicked.connect(lambda: self.command('stop_tracking'))
        self.resume_button = QtWidgets.QPushButton('자동 추적 재개')
        self.resume_button.setObjectName('resumeAutoButton')
        self.resume_button.setToolTip('최근 1초 이내 실측한 확정 트랙의 칼만 위치 중 시스템 기준 3D 거리가 가장 가까운 대상을 추적합니다.')
        self.resume_button.clicked.connect(lambda: self.command('resume_auto'))
        self.target_input = QtWidgets.QSpinBox()
        self.target_input.setRange(0, 2147483647)
        self.target_input.setPrefix('ID ')
        self.target_input.setSpecialValueText('ID 자동 · 최근접')
        self.target_input.setObjectName('targetIdInput')
        self.target_input.setAccessibleName('요청할 대상 ID, 0은 최근접 자동 선택')
        self.target_input.setToolTip('0: 최근접 자동 선택. 양수 ID: 해당 대상 지정. 입력 후 선택 요청을 누르세요.')
        self.target_input.setMinimumWidth(145)
        self.target_input.setMaximumWidth(180)
        self.select_button = QtWidgets.QPushButton('ID 선택 요청')
        self.select_button.setObjectName('selectTargetButton')
        self.select_button.clicked.connect(lambda: self.node.select_target(self.target_input.value()))
        for widget in (self.stop_button, self.resume_button, self.target_input, self.select_button):
            commands.addWidget(widget)
        commands.addStretch()
        self.record_button = QtWidgets.QPushButton('UI 기록 시작')
        self.record_button.setObjectName('recordButton')
        self.record_button.setToolTip('이 화면의 CSV/JSONL 기록입니다. 원시 rosbag 녹화 상태는 별도로 확인하세요.')
        self.record_button.clicked.connect(self.toggle_record)
        commands.addWidget(self.record_button)
        layout.insertWidget(1, control_bar)
        self.command_label = QtWidgets.QLabel(node.command_message)
        self.command_label.setObjectName('commandResponseLabel')
        self.command_label.setProperty('role', 'muted')
        self.command_label.setTextFormat(QtCore.Qt.PlainText)
        self.command_label.setWordWrap(True)
        layout.addWidget(self.command_label)
        self.static_buttons = {}
        if node.static_flight_mode:
            title.setText('JOL / 정적 물체 · 3m 이륙 / 스플라인 시험')
            self.setWindowTitle('NUB · 30cm 풍선 / 카메라 단독 스플라인 비행')
            for widget in (self.stop_button, self.resume_button, self.target_input, self.select_button,
                           self.arm_feedback, self.prediction_label):
                widget.hide()
            self.preview_button.clicked.disconnect()
            self.preview_button.clicked.connect(lambda: self.command('static_preview'))
            self.preview_button.setText('경로 미리보기')
            self.preview_button.setAccessibleName('락 좌표로 경로 계산, 비행 시작 없음')
            self.preview_button.setToolTip('락한 센서 좌표로 경로를 계산합니다. 발사 버튼으로 비행을 시작합니다.')
            self.static_buttons['static_preview'] = self.preview_button
            for index, (action, label) in enumerate((('lock', '좌표 락'), ('unlock', '락 해제'),
                                                     ('launch', '발사'), ('abort', '비행 중단'),
                                                     ('reset', '초기화'))):
                button = QtWidgets.QPushButton(label)
                button.setObjectName('static'+action.title()+'Button')
                button.clicked.connect(lambda checked=False, action=action: self.command('static_'+action))
                commands.insertWidget(index, button)
                self.static_buttons['static_'+action] = button
            self.static_buttons['static_launch'].setToolTip('최신 좌표·락·계획·제어기 준비가 모두 확인되면 3m 이륙과 경로 추종을 시작합니다.')
            self.static_buttons['static_abort'].setToolTip('비행 제어기에 중단·착륙을 요청합니다. 실제 상태 전이로 완료를 확인합니다.')
            self.static_buttons['static_reset'].setToolTip('최신 자기 위치·속도로 지상 정지를 확인한 뒤 초기화합니다. PX4는 착륙·시동 해제도 확인합니다.')
            self.resume_button.clicked.disconnect()
            self.resume_button.clicked.connect(lambda: self.command('static_start_acquisition'))
            self.resume_button.setText('탐지 시작')
            self.resume_button.setToolTip('정면 초기 상태를 확인한 뒤 카메라 탐색과 실제 짐벌 선회를 시작합니다. 드론은 이륙하지 않습니다.')
            commands.removeWidget(self.resume_button)
            commands.insertWidget(0,self.resume_button)
            self.resume_button.show()
            self.static_buttons['static_start_acquisition']=self.resume_button
            self.plot.static_flight_display = True
            self.tabs.setTabText(1, '정적 물체 · World / NED')
            self.tabs.setTabText(3, '스플라인 비행')
            self.speed_card.layout().itemAt(0).widget().setText('계획 / 추종 오차')
            self.distance_card.layout().itemAt(0).widget().setText('드론 현재 상대 고도')
            self.observation_card.layout().itemAt(0).widget().setText('비행 제어 상태')
            self.observation_card.detail.setWordWrap(True)
        self.spin_timer = QtCore.QTimer(self)
        self.spin_timer.timeout.connect(self.pump_ros)
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)
        if start_timer:
            self.spin_timer.start(20)
            self.refresh_timer.start(100)
        self.refresh()

    def refresh_observation(self, now):
        """서비스 수락과 유효한 경로 수신을 구분하고 만료는 화면에서도 검사한다."""
        node = self.node
        if node.static_flight_mode:
            self.refresh_static_flight(now)
            return
        flight = node.packets.get('flight', {})
        flight_plan = node.packets.get('flight_plan', {})
        self.plot.flight_path_display = bool(flight)
        if flight:
            fresh = node.state_fresh('flight')
            phase = {'waiting_for_fusion':'융합 좌표 대기','takeoff':'수직 이륙','transfer':'곡선 전환·목표 접근',
                     'hold':'목표점 정지','landing':'착륙','abort_landing':'중단 후 착륙','complete':'비행 완료','aborted':'비행 중단'}.get(flight.get('phase'),'상태 확인')
            self.preview_button.setEnabled(False)
            self.observer_policy.setText('Gazebo 시험 비행\n동적 충돌: 미검증')
            self.observer_summary.setText(phase+('' if fresh else ' · 통신 / 원본 시각 확인')+
                f"\nID {flight.get('target_id', '—')} · 추종 오차 {number(flight.get('tracking_error_m'), 3)} m")
            tone(self.observer_summary, 'good' if fresh and flight.get('phase')=='complete' else 'neutral' if fresh else 'warning')
            valid = fresh and flight.get('phase') not in ('waiting_for_fusion','aborted') and bool(flight_plan)
            self.plot.observation_path = [row['position'] for row in flight_plan.get('samples', [])] if valid else []
            self.plot.observation_goal = flight_plan.get('goal') if valid else None
            self.observer_detail.setText(phase+f"\n상태 ROS {number(flight.get('stamp'))} · {flight.get('reason', '')}\n"+
                (f"계획 시간 {number(flight_plan.get('duration_s'))} s · world XYZ\n"
                 f"P₀ {flight_plan.get('corner_geometry', {}).get('P0', flight_plan.get('home'))}\n"
                 f"C {flight_plan.get('corner_geometry', {}).get('C')}\n"
                 f"Pg {flight_plan.get('goal')}\n" if flight_plan else '')+
                '경로 표시는 Path, 실제 비행 명령은 별도 setpoint 토픽입니다.\n'
                '자기 위치는 Gazebo의 이상적 피드백이며, 같은 위치로의 1회 관찰 비행입니다.')
            return
        self.observer_policy.setText('비행 허용: 아니요\n동적 충돌: 미검증')
        status, plan = node.packets.get('observer', {}), node.packets.get('observer_plan', {})
        service_ready = node.control_clients['prepare_observation'].service_is_ready()
        self.preview_button.setEnabled(node.pending is None and service_ready)
        reason = status.get('reason', '') or plan.get('reason', '')
        reasons = {
            'input_missing': '표적 상태·원본 트랙·자기 기체 위치 입력을 확인하세요.',
            'input_invalid': '입력 형식이나 시각이 올바르지 않습니다.',
            'input_stale': '입력이 오래되었거나 미래 시각입니다. 최신 측정과 ROS 시계를 확인하세요.',
            'frame_mismatch': '좌표 기준이 다릅니다. 입력은 공통 world 좌표가 필요합니다.',
            'source_mismatch': '공유 표적 상태와 같은 시각의 원본 트랙이 없습니다.',
            'target_unavailable': '추적 중인 공유 표적이 없습니다.',
            'own_pose_invalid': '자기 기체 위치·속도가 없거나 올바르지 않습니다.',
            'own_pose_moving': '자기 기체가 정지 상태가 아닙니다. 경로를 이어 붙일 수 없습니다.',
            'scene_invalid': '고정 장애물이나 비행 범위 설정을 확인하세요.',
            'track_invalid': '트랙의 확정 여부·실측 시각·추정 상태를 확인하세요.',
            'snapshot_changed': '표적·자기 기체·장애물 입력이 바뀌어 이전 계획을 폐기했습니다.',
            'clock_stopped': 'ROS 시계가 멈췄습니다. 이전 계획은 사용할 수 없습니다.',
            'clock_reset': 'ROS 시계가 초기화되어 이전 계획을 폐기했습니다.',
            'preview_expired': '미리보기 유효 시간이 끝났습니다. 최신 입력으로 다시 요청하세요.',
            'target_changed': '공유 표적이 바뀌거나 추적이 중지되어 계획을 폐기했습니다.',
            'path_rejected': '장애물·경계·입력 조건 때문에 경로를 만들지 못했습니다.',
        }
        message = reasons.get(status.get('reason_code'), reason or '입력 상태 확인 중')
        if status.get('reason_code') == 'input_missing' and status.get('input_freshness'):
            missing = [name for key, name in (('status', '제어 상태'), ('tracks', '원본 트랙'), ('own', '자기 기체 위치·속도'))
                       if status['input_freshness'].get(key) is None or
                       (isinstance(status['input_freshness'][key], dict) and
                        all(status['input_freshness'][key].get(age) is None for age in ('ros_age_s', 'wall_age_s')))]
            if missing:
                message = '입력 없음: '+' · '.join(missing)
        valid = (plan.get('valid') is True and node.state_fresh('observer')
                 and node.fresh('observer_plan') and status.get('state') == 'preview_ready'
                 and plan['planned_at'] <= now <= plan['preview_expires_at'])
        # 현재 계획만 그린다. 과거 계획 기록은 아래 읽기 전용 설명에만 남긴다.
        self.plot.observation_path = [s['position'] for s in plan.get('samples', [])] if valid else []
        self.plot.observation_goal = plan['goal'] if valid else None
        if not status:
            title, state = '관찰 상태 수신 대기', 'neutral'
            message = '경로 없음 · 관찰 노드와 자기 기체 위치 입력을 기다립니다.'
        elif not node.state_fresh('observer'):
            title, state = '관찰 통신 / 원본 시각 확인 필요', 'warning'
            message = '관찰 상태가 오래되었거나 미래 시각입니다. 이전 계획은 유효하지 않습니다.'
        elif valid:
            title, state = '관찰 경로 미리보기 · 실행 불가', 'neutral'
            message = f"ID {plan.get('target_id', '—')} · {max(0., plan['preview_expires_at']-now):.2f}s 후 만료"
        elif plan.get('valid') is True and now > plan['preview_expires_at']:
            title, state = '관찰 경로 만료', 'warning'
            message = reasons['preview_expired']
        elif status.get('ready') is True:
            title, state = '관찰 입력 준비 · 미리보기 요청 가능', 'neutral'
            message = '입력 확인 완료 · 경로를 요청하면 정적 장애물과 범위를 검사합니다.'
        else:
            title, state = '관찰 경로 없음 / 입력 확인', 'warning'
        self.observer_summary.setText(title+'\n'+message)
        tone(self.observer_summary, state)
        text = (title+'\n'+message+'\n\n비행 허용: 아니요 · 동적 충돌: 미검증\n'
                '움직이는 물체와 미관측 장애물의 미래 충돌은 확인하지 않았습니다.\n'
                f"미리보기 서비스: {'연결됨' if service_ready else '연결 대기'} · "
                f"상태 ROS {number(status.get('stamp'))} / {measurement_age_text(source_age(now, status.get('stamp')), '상태 갱신')}")
        if valid:
            text += (f"\n계획 ROS {number(plan['planned_at'])} → 만료 {number(plan['preview_expires_at'])} · "
                     f"계획 시간 {number(plan.get('duration_s'))}s\n"
                     f"자기 기체 시작 XYZ {', '.join(number(v) for v in plan['home'])} m\n"
                     f"관찰점 XYZ {', '.join(number(v) for v in plan['goal'])} m · 기준 world\n"
                     '상승 1m는 기체 기준점의 시작 높이로부터의 변화량입니다.')
        elif node.last_observer_plan:
            # 마지막 계획 한 개만 학습용으로 남긴다. 유효성 판단에는 사용하지 않는다.
            previous = node.last_observer_plan
            text += (f"\n\n최근 계획 기록 · 현재 유효하지 않음\nID {previous.get('target_id', '—')} · "
                     f"계획 ROS {number(previous['planned_at'])} · 계획 시간 {number(previous.get('duration_s'))}s\n"
                     f"과거 시작 XYZ {', '.join(number(v) for v in previous['home'])} m\n"
                     f"과거 관찰점 {', '.join(number(v) for v in previous['goal'])} m · 기준 world")
        if reason:
            text += '\n진단 원문: '+reason
        self.observer_detail.setText(text)
        tone(self.observer_detail, state)

    def refresh_static_flight(self, now):
        node = self.node
        fresh = node.state_fresh('static_status')
        status = node.packets.get('static_status', {}) if fresh else {}
        phase = status.get('phase')
        phase_text = STATIC_PHASES.get(phase, '비행 상태 미수신 / 원본 시각 확인')
        for action, button in self.static_buttons.items():
            button.setEnabled(node.static_action_allowed(action) and node.control_clients[action].service_is_ready())
        block_reason = node.static_launch_block_reason()
        self.observer_summary.setText(phase_text+'\n'+(
            '도달점 위치 유지 중 · 비행 중단으로 착륙' if phase == 'arrived' else
            '발사 버튼 잠금 · 비행 진행 상태 확인' if phase in STATIC_ACTIVE else
            '발사 대기 · '+block_reason if block_reason else
            '경로 준비 완료 · 발사 버튼으로 비행 시작'))
        sensor=node.packets.get('static_sensor',{}) if node.state_fresh('static_sensor') else {}
        acquisition=sensor.get('acquisition',{})
        sequence_text={'waiting_start':'정면 유지 · 탐지 시작 버튼을 누르세요',
            'resetting':'짐벌 초기 자세 복귀 중 · 실제 각도 확인',
            'camera_search':'카메라로 풍선 탐색 중', 'lidar_search':'LiDAR에서 물체 후보 탐색 중', 'slewing':'짐벌 선회·고각 이동 중',
            'yaw_slew':'짐벌 우측 선회 중 · 실제 각도 확인', 'pitch_slew':'선회 완료 · 표적 높이로 상향 중',
            'visual_centering':'영상 검출 · 중심 정렬 중','ready':'영상 정렬·카메라 좌표 확인'}.get(acquisition.get('stage'))
        if sequence_text and phase not in STATIC_ACTIVE and not status.get('locked'):
            self.observer_summary.setText(sequence_text+'\n'+phase_text)
        tone(self.observer_summary, 'good' if phase == 'arrived' else 'neutral' if fresh else 'warning')
        backend_kind = status.get('flight_backend')
        backend = ('PX4 · MAVROS · 20cm 테일시터(가정)' if status.get('aircraft_type') == 'tailsitter20cm_reference' else
                   'PX4 · MAVROS · Iris 대리 모델' if backend_kind == 'mavros_px4' else 'Gazebo · native 제어'
                   if backend_kind == 'native' or status.get('own_pose_source') == 'ideal_gazebo_self_odometry'
                   else 'Gazebo · 제어 상태 대기')
        operation = ('비행 상태: '+phase_text if phase in STATIC_ACTIVE else
                     '발사 준비: '+('완료' if self.static_buttons['static_launch'].isEnabled() else '대기'))
        self.observer_policy.setText(backend+'\n'+(
            static_autopilot_text(status)+'\n' if backend_kind == 'mavros_px4' else '')+operation)
        tone(self.observer_policy, 'good' if self.static_buttons['static_launch'].isEnabled() else 'neutral')
        plan = node.packets.get('static_plan', {})
        valid = fresh and status.get('plan_ready') is True and plan.get('valid') is True
        self.plot.flight_path_display = True
        self.plot.observation_path = [row['position'] for row in plan.get('samples', [])] if valid else []
        self.plot.observation_goal = plan.get('goal') if valid else None
        self.observer_detail.setText(phase_text+f"\n상태 ROS {number(status.get('stamp'))} · {status.get('reason', '')}\n"
            f"초기 수직 이륙 목표 {number(status.get('takeoff_height', 3.))} m · 현재 출발점 대비 고도 {number(status.get('takeoff_height_m'), 3)} m\n"
            f"계획 시간 {number(plan.get('duration_s')) if valid else '—'} s · 추종 오차 {number(status.get('tracking_error_m'), 3)} m\n"
            f"물체 표면까지 {number(status.get('target_surface_distance_m'), 3)} m · 도달 여유 {number(status.get('standoff_m'))} m\n"
            f"제어: {status.get('feedback_controller', '수신 대기')}\n측위: {status.get('own_pose_source', '수신 대기')}\n"
            '미리보기는 위치를 바꾸지 않습니다. 발사는 NED 제어기 경로 명령을 요청합니다.\n'
            '황색 점선: 계획 / 파란 실선: 실제 비행. 메인 3D 보기에서 회전·확대하거나 XY 투영으로 전환하세요.')

    def set_world_view(self, index):
        self.world_views.setCurrentIndex(index)
        if index == 0:
            self.flight_3d.redraw(force=True)

    def pump_ros(self):
        if not rclpy.ok(context=self.node.context):
            self.close()
            QtWidgets.QApplication.instance().quit()
            return
        deadline = time.monotonic()+.005
        for _ in range(8):
            self.executor.spin_once(timeout_sec=0.)
            if time.monotonic() >= deadline:
                break
        self.node.poll_command()

    def command(self, action):
        self.node.request_command(action)
        self.refresh()

    def choose_directory(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '실험 기록 상위 폴더 선택', self.directory_input.text())
        if path:
            self.directory_input.setText(path)

    def toggle_record(self):
        if self.node.recorder.active:
            self.node.stop_recording()
        else:
            self.node.start_recording(self.directory_input.text())
        self.refresh()

    def refresh(self):
        node = self.node
        node.poll_command()
        row = node.snapshot()
        status = node.packets.get('pointing', {})
        packet = node.packets.get('tracks', {})
        camera = node.packets.get('camera', {})
        active = node.state_fresh('pointing')
        mode = {'background': '배경 학습 · 상향 20° 회전', 'patrol': '360° 순찰 · 상향 20°',
                'tracking': '객체 추적', 'hold': '상태 대기 / 유지'}.get(status.get('mode'), '상태 확인 중')
        aligned = all(status.get(key, {}).get('aligned') is True for key in ('arm', 'turret'))
        if status.get('mode') == 'tracking':
            measured = all(finite(status.get(key, {}).get('measured_angle_error_deg')) for key in ('arm', 'turret'))
            mode += ' · '+('정렬 유지' if aligned else '선회 중' if measured else '정렬 확인 중')
        self.mode_label.setText(mode if active else '통신 / 원본 시각 확인 · 제어 상태 미확인')
        tone(self.mode_label, 'good' if active and status.get('mode') == 'tracking' and aligned else 'warning')
        self.clock_label.setText(f"ROS {number(row['sim_stamp'], 2)} s")
        image_current = (node.fresh('image') and finite(row['image_exposure_age_s'])
                         and -.05 <= row['image_exposure_age_s'] <= node.stale_timeout)
        health = {
            'tracks': ('LiDAR 통신 수신 · 상태 갱신' if node.state_fresh('tracks') else
                       'LiDAR 통신 수신 · 원본 시각 확인' if node.fresh('tracks') else 'LiDAR 수신 대기 / 두절', node.state_fresh('tracks')),
            'camera': ('YOLO 상태 대기' if not camera else 'YOLO 처리 준비' if row['yolo_ready'] else 'YOLO 준비 미확인 / 오류', bool(row['yolo_ready'])),
            'image': ('영상 수신 중' if image_current else '영상 수신 · 과거 노출' if node.fresh('image') else '영상 수신 대기 / 두절', image_current),
            'background': ('배경 준비' if node.background_ready else '배경 학습 대기', node.background_ready is True),
        }
        for key, (text, ready) in health.items():
            self.health_labels[key].setText(('●  ' if ready else '○  ')+text)
            tone(self.health_labels[key], 'good' if ready else 'neutral')
        enabled = status.get('auto_acquisition_enabled')
        auto_text = '허용' if enabled is True else '중지' if enabled is False else '미확인'
        self.status_detail.setText(f"자동 획득 {auto_text} · 배경 {'준비' if node.background_ready else '대기'} · "
            f"실제 yaw {number(row['yaw_deg'], 1)}° / 고각 {number(row['elevation_deg'], 1)}° · "
            f"상태 {status.get('state', '미수신')} · ROS {number(row['sim_stamp'], 2)} s\n"
            f"제어 상태 ROS {number(row['status_stamp'])} s / 나이 {number(row['control_status_age_s'])} s · "
            f"LiDAR 통신 {'수신 중' if row['tracks_communication_fresh'] else '대기 / 두절'} · "
            f"최신 LiDAR 상태 ROS {number(row['track_stamp'])} s / {measurement_age_text(row['track_source_age_s'], '갱신')}")
        pending = node.pending is not None
        for widget in (self.stop_button, self.resume_button, self.select_button, self.target_input):
            widget.setEnabled(not pending)
        self.command_label.setText(node.command_message+f" · 입력 파싱 오류 {row['parse_error_count']}건"+
                                   (' · 최근 '+node.last_error if node.last_error else ''))
        tone(self.command_label, 'error' if row['parse_error_count'] else node.command_tone)
        self.refresh_observation(row['sim_stamp'])
        target = row['target_id']
        observation = '관측' if row['position_observed'] is True else '예측' if row['position_observed'] is False else '미확인'
        state_tone = ('good' if row['position_observed'] is True and finite(row['last_observation_age_s'])
                      and -.05 <= row['last_observation_age_s'] <= node.stale_timeout else 'warning')
        self.target_card.update_value(f'ID {target}' if active and target is not None else '—',
            observation+(' · 최근접 자동' if status.get('selection', {}).get('policy') in ('nearest_observed_3d', 'nearest_recent_3d')
                         else ' · ID 지정' if status.get('requested_target_id', 0) else ' · 월드 좌표')
                if active and target is not None else '선택 대상 없음 / 미확인', state_tone)
        self.speed_card.update_value(number(row['speed_m_s'])+' m/s',
            'LiDAR 상태 추정 · '+observation, 'neutral' if row['speed_m_s'] is None else state_tone)
        self.distance_card.update_value(number(row['nearest_surface_m'])+' m',
            measurement_age_text(row['surface_age_s']) if finite(row['nearest_surface_m']) else '최근 관측된 표면 거리 없음')
        age = row['last_observation_age_s']
        self.observation_card.update_value(number(max(0., age))+' s' if finite(age) and age >= -.05 else '—',
            measurement_age_text(age, '실측'), state_tone)
        self.observation_card.setToolTip('예측 상태의 갱신 시각이 아닌 마지막 실제 LiDAR 관측부터의 경과 시간입니다.')
        self.distance_card.setToolTip('관측 중이며 트랙 통신이 최근 수신된 경우에만 센서에서 물체 표면까지의 거리를 표시합니다.')
        self.target_label.setText(f"선택 ID {target if target is not None else '없음'}  |  "
            f"LiDAR XYZ ({number(row['x_m'])}, {number(row['y_m'])}, {number(row['z_m'])}) m  |  "
            f"속도 {number(row['speed_m_s'])} m/s  |  표면 거리 {number(row['nearest_surface_m'])} m "
            f"(ROS {number(row['surface_stamp'])})\n"
            f"{observation} · XYZ 상태 ROS {number(row['position_stamp'])} / {measurement_age_text(row['position_source_age_s'], '갱신')} · "
            f"최종 실측 ROS {number(row['last_observation_stamp'])} / {measurement_age_text(row['last_observation_age_s'], '실측')} · "
            f"제어 출처: 포탑 {row['control_state_source'] or '대기'} / 팔 {row['arm_control_state_source'] or '미수신'}")
        self.target_label.setToolTip(f"표면 거리: {measurement_age_text(row['surface_age_s'])}. "
                                    '관측 중이고 트랙 통신이 최근 수신된 경우에만 표시합니다. '
                                    'LiDAR 상태 시각과 거리의 최신 트랙 시각은 다를 수 있습니다.')
        selection = status.get('selection', {}) if active else {}
        if selection:
            self.target_label.setText(self.target_label.text()+
                f"\n선택 정책: {'최근접 자동' if selection.get('policy') in ('nearest_observed_3d', 'nearest_recent_3d') else 'ID 지정'} · "
                f"시스템 기준 중심 거리 {number(selection.get('distance_m'))} m · "
                f"거리 시각 ROS {number(selection.get('distance_state_stamp', selection.get('source_stamp')))} · {selection.get('reason', '')}")
        for key, label, name in (('arm', self.arm_feedback, '막대'),
                                  ('turret', self.turret_feedback,
                                   '포탑·카메라' if status.get('camera_mount') == 'turret' else '포탑')):
            measured = status.get(key, {}) if active else {}
            error = measured.get('measured_angle_error_deg')
            state = ('정렬 완료' if measured.get('aligned') is True else
                     '관절 복귀 중' if measured.get('hardstop_unwinding') else '회전 중')
            label.setText(f'{name} 방향  {number(error, 1)}° · {state}' if finite(error)
                          else f'{name} 방향  — · 실측 대기')
            label.setToolTip('실제 TF 방향과 추정 목표점 사이 3D 각도. 2° 이내일 때 정렬 완료. 명령 수락과 별개입니다.\n'
                f"현재 ID 선택 후 {number(status.get('selection_age_s'))}s · "
                f"현재 IK 관절목표 도달 시간 하한 {number(measured.get('joint_slew_lower_bound_s'))}s.\n"
                '시간 하한은 목표·베이스가 고정됐을 때 관절 속도 제한만 계산한 값입니다. '
                '움직이는 목표의 정렬 완료 예상 시간이 아닙니다.')
            tone(label, 'good' if finite(error) and measured.get('aligned') is True else
                 'warning' if finite(error) else 'neutral')
        self.arm_feedback.setVisible(status.get('arm_enabled', True))
        predicted = status.get('prediction', {}).get('position') if active else None
        self.prediction_label.setText('예측 +'+number(status.get('prediction', {}).get('horizon_s'), 2)+'s 예측  '+', '.join(number(v, 1) for v in predicted)+' m'
                                      if vector(predicted) else '예측점  —')
        self.prediction_label.setToolTip('World XYZ. 칼만 예측 시간: +'+
            number(status.get('prediction', {}).get('horizon_s'), 2)+' s / 포탑 +'+
            number(status.get('turret_prediction', {}).get('horizon_s'), 2)+' s.')
        self.render_image()
        yolo = ('상태 대기' if not camera else '처리 준비' if row['yolo_ready'] else
                '대기 / 오류' if node.state_fresh('camera') else '통신 / 원본 시각 확인')
        self.camera_status.setText(f"YOLO {yolo} · NN 검출 {number(row['detection_count'], 0)}개 "
            f"(노출 ROS {number(row['detection_stamp'])}, 나이 {number(row['detection_age_s'])}s) · "
            f"추론 {number(camera.get('inference_ms'), 1)}ms\n"
            f"영상 노출 ROS {number(row['image_stamp'])} / {measurement_age_text(row['image_exposure_age_s'], '노출')} · "
            f"수신 후 {number(row['image_rx_age_s'])}s {'· 과거 영상' if not node.fresh('image') else ''}\n"
            f"기하 조건 {row['fusion_geometry_status'] or '미수신'} · 카메라 보정 수락 {number(row['fusion_accepted_count'], 0)}개 · "
            f"{camera.get('error') or camera.get('health_reason') or '처리 준비·기하 조건만으로 검출 성공을 뜻하지 않음'}\n"
            '기종 판정: 현재 장면에서 미검증 · NN 원시 라벨과 위치 추적은 별도 평가')
        if status.get('camera_mount') == 'arm':
            self.camera_status.setText(self.camera_status.text()+'\n팔 끝 고정 카메라 · 막대기와 같은 방향으로 회전')
        pipeline = camera.get('pipeline', {})
        if pipeline:
            counters = pipeline.get('counters', {})
            self.camera_status.setText(self.camera_status.text()+
                f"\n영상 처리 누계 · 노드 수신 {number(counters.get('image_received'), 0)} / "
                f"추론 제출 {number(counters.get('inference_submitted'), 0)} / "
                f"결과 발행 {number(counters.get('result_published'), 0)}\n"
                f"입력 거부 {number(counters.get('input_rejected'), 0)} / "
                f"대기 제외 {number(counters.get('queue_dropped'), 0)} / "
                f"추론 실패 {number(counters.get('inference_failed'), 0)} / "
                f"결과 폐기 {number(counters.get('result_discarded'), 0)}")
        self.camera_status.setToolTip('노드 수신은 카메라 처리 콜백에 도착한 영상 수입니다. '
            '원본 rosbag의 영상 수와 같다고 가정하지 않습니다. 거부·제외 사유별 계수는 원시 telemetry에 기록됩니다.')
        self.health_labels['camera'].setToolTip(
            f"상태: {camera.get('health_reason', '미수신')} · "
            f"영상 수신 후 {number(camera.get('image_wall_age_s'))}s / "
            f"추론 완료 후 {number(camera.get('result_wall_age_s'))}s. "
            '처리 준비는 검출 성공이나 기종 검증을 뜻하지 않습니다.')
        image_state = '영상 수신 중' if image_current else '과거 영상 / 수신 시각 확인'
        if self.pixmap is None:
            image_state = '영상 없음 / 수신 대기'
        self.camera_summary.setText(
            f"{image_state} · {measurement_age_text(row['image_exposure_age_s'], '노출')}\n"
            f"NN 검출 {number(row['detection_count'], 0)}개 · {number(camera.get('inference_ms'), 1)} ms 추론 "
            f"· 검출 나이 {number(row['detection_age_s'])} s")
        tone(self.camera_summary, 'neutral' if image_current and self.pixmap is not None else 'warning')
        tracks = packet.get('tracks', [])
        fusion = node.fusion_packet()
        fused_by_id = {t['id']: t for t in fusion.get('tracks', [])}
        selected_fusion = fused_by_id.get(target)
        if selected_fusion and selected_fusion.get('fusion_update', {}).get('accepted'):
            fused_position, _, _, current_valid = fused_state(selected_fusion, fusion)
            coordinates = ', '.join(number(v) for v in fused_position)
            kind = '현재' if current_valid else '노출 시각'
            self.target_label.setText(self.target_label.text()+
                f"\n{kind} 융합 XYZ ({coordinates}) m · 상태 ROS {number(row['fusion_stamp'])} / "
                f"영상 노출 ROS {number(row['fusion_exposure_stamp'])} · LiDAR−융합 Δt {number(row['lidar_minus_fused_stamp_s'], 3)}s "
                '(시각이 다르면 좌표 직접 비교 불가)')
        self.table.setRowCount(len(tracks))
        count = str(len(tracks)) if packet else '—'
        self.tabs.setTabText(0, f'트랙 목록  {count}')
        self.world_summary.setText(f"트랙 {count}개 · {measurement_age_text(row['track_source_age_s'], '상태 갱신')}\n"
            f"yaw {number(row['yaw_deg'], 1)}° / 고각 {number(row['elevation_deg'], 1)}° · 자동 획득 {auto_text if active else '미확인'}")
        tone(self.world_summary, 'neutral' if node.fresh('tracks') else 'warning')
        for index, track in enumerate(tracks):
            fused = fused_by_id.get(track['id'], {})
            label = fused.get('class_name', '—')
            if fused.get('subtype_ambiguous'):
                label += ' (기종 불확실)'
            elif label != '—' and fused.get('subtype_verified') is not True:
                label += ' (기종 미검증)'
            observed = track.get('observed', False)
            track_state = (('관측' if observed else '예측')+' · '+
                           measurement_age_text(source_age(row['sim_stamp'], track.get('last_seen')), '실측')+
                           '\n'+measurement_age_text(row['track_source_age_s'], '상태 갱신'))
            values = [str(track['id']), track_state if node.state_fresh('tracks') else
                      '통신 수신 · 원본 시각 확인' if node.fresh('tracks') else '통신 두절 · 과거 데이터',
                      *[number(v) for v in track['position']],
                      number(math.sqrt(sum(float(v)**2 for v in track['velocity']))),
                      number(surface_distance(track, node.fresh('tracks'))), label]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(value)
                item.setTextAlignment(QtCore.Qt.AlignVCenter | (QtCore.Qt.AlignLeft if column in (1, 7) else QtCore.Qt.AlignCenter))
                if column == 1:
                    item.setForeground(QtGui.QColor('#72d6aa' if observed and node.state_fresh('tracks') else '#f0c674'))
                if track['id'] == target:
                    item.setBackground(QtGui.QColor('#294462'))
                self.table.setItem(index, column, item)
            self.table.setRowHeight(index, 48)
        self.plot.history, self.plot.tracks, self.plot.selected = node.history, tracks, target
        self.plot.yaw, self.plot.stale = row['yaw_deg'], not node.state_fresh('tracks')
        self.plot.prediction = predicted
        self.plot.arm = status.get('arm', {}) if active else {}
        if node.static_flight_mode:
            self.refresh_static_readouts(row)
        self.plot.update()
        recorder = node.recorder
        self.record_button.setText('UI 기록 종료' if recorder.active else 'UI 기록 시작')
        tone(self.record_button, 'warning' if recorder.active else 'neutral')
        self.record_badge.setText('●  UI 기록 중' if recorder.active else '○  UI 기록 대기')
        self.record_badge.setToolTip('UI CSV/JSONL 기록입니다. 원시 rosbag 녹화와 별개입니다.\n'+
            (str(recorder.directory) if recorder.active else '실험 기록 탭에서 저장 위치를 설정합니다.'))
        tone(self.record_badge, 'good' if recorder.active else 'neutral')
        self.directory_input.setEnabled(not recorder.active)
        self.directory_button.setEnabled(not recorder.active)
        if recorder.error:
            self.record_badge.setText('UI 기록 오류 · 실험 기록 탭 확인')
            tone(self.record_badge, 'error')
            self.record_label.setText('기록 오류 · '+recorder.error)
            self.record_label.setStyleSheet('color:#ff8c91;')
        else:
            text = f'기록 중 · {recorder.directory} · {recorder.rows}행' if recorder.active else '기록 대기'
            self.record_label.setText(text+' · JSONL / CSV · 원시 rosbag은 별도 녹화')
            self.record_label.setStyleSheet('color:#c8d4e1;')
        now = row['monotonic_ns']
        if now-self.last_csv_write >= 100_000_000:
            recorder.write('selected_target.csv', row)
            self.last_csv_write = now

    def refresh_static_readouts(self, row):
        """단일 정적 센서 좌표와 실제 드론 궤적만 표시한다. 정답 좌표는 구독하지 않는다."""
        node = self.node
        fresh = node.state_fresh('static_status')
        mission = node.packets.get('static_status', {}) if fresh else {}
        estimate = node.packets.get('static_estimate', {}) if node.state_fresh('static_estimate') else {}
        world = mission.get('target_world') if mission.get('coordinate_valid') else None
        ned = mission.get('target_ned') if mission.get('coordinate_valid') else None
        phase = mission.get('phase')
        text = STATIC_PHASES.get(phase, '비행 상태 미수신 / 원본 시각 확인')
        self.mode_label.setText(text)
        tone(self.mode_label, 'good' if phase == 'arrived' else 'neutral' if fresh else 'warning')
        self.target_card.update_value('정적 물체 1' if vector(world) else '—',
                                     '좌표 락' if mission.get('locked') else '센서 좌표 확인' if vector(world) else '검출 / 좌표 대기')
        self.speed_card.update_value(number(mission.get('tracking_error_m'), 3)+' m', '계획 대비 실제 위치 오차')
        self.distance_card.update_value(number(mission.get('takeoff_height_m'), 3)+' m',
                                        '출발 높이 기준 · 초기 수직 이륙 '+number(mission.get('takeoff_height', 3.))+' m 후 곡선 접근')
        control_state = ('도달 유지' if phase == 'arrived' else '시동 중' if phase == 'arming' else
                         '착륙 중' if phase in ('abort_braking', 'abort_landing', 'autopilot_landing') else
                         '비행 중' if phase in STATIC_ACTIVE else
                         '준비' if self.static_buttons['static_launch'].isEnabled() else '대기')
        control_detail = (static_autopilot_text(mission) if mission.get('flight_backend') == 'mavros_px4' else
            '좌표 '+('유효' if vector(world) else '없음')+' / 제어기 '+('준비' if mission.get('controller_ready') else '대기'))
        self.observation_card.update_value(control_state, control_detail)
        self.distance_card.setToolTip('현재 고도의 출발점 대비 차이입니다. 처음 3m를 수직 이륙한 뒤 곡선에서 목표 높이까지 더 상승합니다.')
        self.observation_card.setToolTip('발사는 유효 좌표·락·계획·제어기와 최신 상태가 모두 준비됐을 때 가능합니다.')
        coordinates = lambda values: '('+', '.join(number(v, 3) for v in values)+')' if vector(values) else '—'
        self.target_label.setText('센서 추정 World XYZ '+coordinates(world)+' m\n제어 기준 NED XYZ '+coordinates(ned)+' m\n'+
            '드론 도달점 '+coordinates(mission.get('goal_world'))+' m · '+
            ('가상 목표 좌표' if mission.get('target_is_virtual_waypoint') else '표면 앞 '+number(mission.get('standoff_m'),1)+' m')+'\n'+
            '드론→목표 중심 '+number(mission.get('target_surface_distance_m'),2)+' m / 도달점 오차 '+number(mission.get('arrival_error_m'),3)+' m\n'+
            f"락 {'확정' if mission.get('locked') else '해제'} · 측정 ROS {number(estimate.get('stamp'))} s")
        self.target_label.setToolTip('유효성은 최신 비행 상태의 coordinate_valid로 판정합니다. 락은 현재 센서 추정 좌표를 확정합니다.')
        self.arm_feedback.hide()
        self.prediction_label.hide()
        alignment_error = mission.get('camera_alignment_error_deg')
        tracking, centered = mission.get('gimbal_tracking'), mission.get('camera_centered')
        alignment_state = ('정렬 완료' if centered is True else '정렬 중' if centered is False else '정렬 미확인') \
            if tracking is True and finite(alignment_error) else '비전 추적 대기' if tracking is False else '정렬 미확인'
        self.turret_feedback.setText(f"짐벌 yaw {number(row['yaw_deg'], 1)}° / 고각 {number(row['elevation_deg'], 1)}° · "
                                    f"중심 잔차 {number(alignment_error, 2)}° · {alignment_state}")
        self.turret_feedback.setToolTip('yaw·고각은 실제 관절 피드백입니다. 중심 잔차와 정렬 상태는 최신 카메라 검출 기반입니다.\n'
                                      '짐벌 명령각이나 Gazebo 정답 위치로 정렬 완료를 판단하지 않습니다.')
        tone(self.turret_feedback, 'good' if alignment_state == '정렬 완료' else
             'warning' if alignment_state == '정렬 중' else 'neutral')
        self.status_detail.setText(f"정적 물체 1개 / 드론 1대 · 포탑 구동 없음\n"
            f"짐벌 yaw {number(row['yaw_deg'], 1)}° / 고각 {number(row['elevation_deg'], 1)}°\n"
            f"영상 중심 잔차 {number(alignment_error, 2)}° · {alignment_state}\n"
            f"상태 ROS {number(mission.get('stamp'))} · 좌표 {mission.get('coordinate_valid', '미수신')} · "
            f"락 {mission.get('locked', '미수신')} · 경로 {mission.get('plan_ready', '미수신')} · 제어기 {mission.get('controller_ready', '미수신')}"+
            ('\n'+static_autopilot_text(mission) if mission.get('flight_backend') == 'mavros_px4' else ''))
        for key, label, ready in (('tracks', '카메라 크기 기반 좌표 수신', estimate.get('valid') is True),
                                  ('camera', '물체 검출·센서 대응', vector(world)),
                                  ('background', '센서 좌표 유효', vector(world))):
            self.health_labels[key].setText(('●  ' if ready else '○  ')+label+('' if ready else ' · 대기'))
            tone(self.health_labels[key], 'good' if ready else 'neutral')
        self.camera_status.setText('짐벌 카메라 단독 · 알려진 지름 30cm\n'+
            f"좌표 ROS {number(estimate.get('stamp'))} · {estimate.get('reason', '센서 좌표 수신 대기')}\n"
            '영상 노출 시각·카메라 보정·짐벌 TF를 반영합니다.\n'
            f"지름 {number(estimate.get('observed_diameter_px'),1)} px / 위치 최대 σ {number(estimate.get('maximum_position_std_m'),3)} m\n"
            '정확한 구형 지름 30cm 가정으로 중심을 추정합니다. 실물 크기·보정 검증 필요.')
        self.camera_summary.setText('영상 '+('수신 중' if node.fresh('image') else '수신 대기 / 두절')+' · '+
            measurement_age_text(row['image_exposure_age_s'], '노출')+'\n'+
            ('풍선 검출·카메라 좌표 유효' if vector(world) else '물체 검출 / 센서 좌표 대기'))
        self.world_summary.setText('정적 목표 '+coordinates(world)+' m\n황색: 계획 · 파란색: 실제 드론 궤적 / World XYZ [m]')
        self.table.setRowCount(1 if vector(world) else 0)
        self.tabs.setTabText(0, '정적 물체  1' if vector(world) else '정적 물체  —')
        if vector(world):
            values = ['1', '락 확정' if mission.get('locked') else '좌표 확인',
                      *[number(v, 3) for v in world], '정적', number(row['nearest_surface_m'],2), '센서 대응']
            for column, value in enumerate(values):
                self.table.setItem(0, column, QtWidgets.QTableWidgetItem(value))
        self.plot.history, self.plot.arm, self.plot.prediction = {}, {}, None
        self.plot.tracks = [dict(id=1, position=world, observed=True)] if vector(world) else []
        self.plot.selected, self.plot.stale = 1, not fresh
        self.plot.actual_path = list(node.static_actual_path)
        plan = node.packets.get('static_plan', {})
        valid_plan = mission.get('plan_ready') is True and plan.get('valid') is True
        self.flight_3d.set_path(plan if valid_plan else {}, node.static_actual_path, world, stale=not fresh)
        self.flight_3d.summary.setToolTip('실제 궤적의 측위 출처: '+mission.get('own_pose_source', '미수신')+
                                        '\n좌표축 눈금은 m이며 축별 표시 비율은 다릅니다.')
        self.plot.setToolTip('황색 점선: 계획 / 파란 실선: 실제 드론 궤적 / 점: 센서 추정 목표.\n'
                             'XY 투영입니다. 수직 상승과 고도는 메인 3D 보기 또는 RViz에서 확인하세요.')

    def render_image(self):
        node = self.node
        self.lock_box_drawn = False
        msg = node.display_image()
        if msg is None and self.last_image_sequence != node.image_sequence:
            self.pixmap = None
            self.last_image_sequence = node.image_sequence
            self.camera_label.setText('카메라 영상 수신 대기')
        if msg is not None and (self.last_image_sequence != node.image_sequence or self.rendered_image is not msg):
            channels = {'bgr8': 3, 'rgb8': 3, 'mono8': 1}
            try:
                if (msg.encoding not in channels or not 0 < msg.width <= 8192 or not 0 < msg.height <= 8192
                        or msg.step < msg.width*channels[msg.encoding] or len(msg.data) < msg.step*msg.height):
                    raise ValueError('unsupported or malformed image')
                fmt = QtGui.QImage.Format_Grayscale8 if msg.encoding == 'mono8' else QtGui.QImage.Format_RGB888
                image = QtGui.QImage(bytes(msg.data), msg.width, msg.height, msg.step, fmt).copy()
                if msg.encoding == 'bgr8':
                    image = image.rgbSwapped()
                self.pixmap = QtGui.QPixmap.fromImage(image)
                self.last_image_sequence = node.image_sequence
                self.rendered_image = msg
            except (ValueError, TypeError) as error:
                self.camera_label.setText('영상 형식 오류: '+str(error))
                self.pixmap = None
        if self.pixmap is not None:
            scaled = self.pixmap.scaled(self.camera_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            annotation = node.static_lock_annotation()
            image_stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9 if msg is not None else None
            if annotation and finite(image_stamp) and abs(image_stamp-annotation['lock_bbox_stamp']) <= 1e-6:
                x, y, w, h = annotation['lock_bbox']
                size = annotation.get('detection_image_size')
                if x+w <= msg.width and y+h <= msg.height and (size is None or size == [msg.width, msg.height]):
                    # 박스는 원본 영상과 같은 노출이다. 선은 축소 후 그려 작은 표적도 보이게 한다.
                    sx, sy = scaled.width()/msg.width, scaled.height()/msg.height
                    painter = QtGui.QPainter(scaled)
                    painter.setPen(QtGui.QPen(QtGui.QColor('#ffca28'), 2.5))
                    painter.drawRect(QtCore.QRectF(x*sx, y*sy, w*sx, h*sy))
                    painter.setFont(QtGui.QFont('DejaVu Sans', 10, QtGui.QFont.Bold))
                    text = 'LOCKED '+str(annotation['target_id'])
                    metrics = painter.fontMetrics()
                    tx = max(2., min(x*sx-4, scaled.width()-metrics.horizontalAdvance(text)-10))
                    ty = max(metrics.height()+4., y*sy-7)
                    painter.fillRect(QtCore.QRectF(tx-4, ty-metrics.ascent()-3,
                        metrics.horizontalAdvance(text)+8, metrics.height()+4), QtGui.QColor('#171c23'))
                    painter.drawText(QtCore.QPointF(tx,ty),text)
                    painter.end()
                    self.lock_box_drawn = True
            self.camera_label.setPixmap(scaled)

    def closeEvent(self, event):
        self.spin_timer.stop()
        self.refresh_timer.stop()
        self.executor.remove_node(self.node)
        self.executor.shutdown(timeout_sec=0.)
        if self.node.recorder.active:
            self.node.stop_recording(method='window_close')
        # 화면 종료는 센서와 제어기의 동작을 바꾸지 않는다.
        super().closeEvent(event)


def main(args=None):
    # Qt 콜백에서 KeyboardInterrupt가 발생하거나 ROS가 먼저 닫히지 않도록 종료를 한 경로로 모은다.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
    node = ExperimentNode()
    window = Dashboard(node)
    window.show()
    previous_handlers = {sig: signal.signal(sig, lambda signum, frame: app.quit())
                         for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        app.exec_()
    finally:
        window.close()
        node.recorder.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
