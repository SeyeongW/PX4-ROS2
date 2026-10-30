"""명시한 외부 카메라의 영상과 수동 ROI만 발행한다."""
from array import array
import json
import math
import os
from pathlib import Path
import time

import cv2
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .fusion_core import image_bgr


V4L_DEV = Path('/dev/v4l')
V4L_SYS = Path('/sys/class/video4linux')


def require_external_camera(device):
    """번호 자동 선택을 금지하고 알려진 노트북 카메라를 차단한다."""
    available = '\n'.join(
        f'  /dev/{entry.name}: {(entry / "name").read_text().strip()}'
        for entry in sorted(V4L_SYS.glob('video*')) if (entry / 'name').exists())
    stable = '\n'.join(f'  {entry} -> {entry.resolve()}'
                       for entry in sorted(V4L_DEV.glob('by-*/*')))

    def reject(reason):
        raise ValueError(
            reason + '\n팔에 장착한 별도 카메라를 연결하고 렌즈를 가려 장치를 확인한 뒤 '
            'device:=/dev/v4l/by-id/... 또는 /dev/v4l/by-path/... 를 지정하세요.\n'
            'JOL_ARM_CAMERA 환경변수로 같은 경로를 지정할 수도 있습니다.\n'
            f'현재 영상 장치:\n{available or "  없음"}\n안정 경로:\n{stable or "  없음"}')

    path = Path(device) if device else None
    if path is None or path.parent not in (V4L_DEV / 'by-id', V4L_DEV / 'by-path'):
        reject('외부 카메라의 고정 장치 경로를 명시해야 합니다. 자동 선택은 하지 않습니다.')
    if not path.is_char_device():
        reject(f'카메라 장치가 없거나 영상 장치가 아닙니다: {path}')
    entry = V4L_SYS / path.resolve().name
    if not (entry / 'name').exists():
        reject(f'V4L2 장치 정보를 확인할 수 없습니다: {path}')
    name = (entry / 'name').read_text().strip()
    for parent in entry.resolve().parents:
        if (parent / 'idVendor').exists() and (parent / 'idProduct').exists():
            usb_id = ((parent / 'idVendor').read_text().strip().lower(),
                      (parent / 'idProduct').read_text().strip().lower())
            if usb_id == ('30c9', '00d9'):
                reject(f'노트북 내장 HP 카메라는 사용할 수 없습니다: {name}')
            break
    if 'hp true vision' in name.lower():
        reject(f'노트북 내장 HP 카메라는 사용할 수 없습니다: {name}')
    return str(path)


def target_message(stamp, width, height, bbox=None, selection_id=0):
    """영상 밖이거나 유효하지 않은 추적 영역은 폐기한다."""
    valid = False
    try:
        x, y, w, h = map(float, bbox)
        valid = (all(math.isfinite(v) for v in (x, y, w, h))
                 and width > 0 and height > 0 and x >= 0 and y >= 0
                 and w >= 2 and h >= 2 and x+w <= width and y+h <= height)
    except (TypeError, ValueError):
        pass
    return dict(stamp=stamp, width=width, height=height, frame_id='real_camera_optical',
                valid=valid, source='manual_roi_csrt', selection_id=selection_id,
                bbox=[x, y, w, h] if valid else None,
                error_normalized=[2*(x+w/2)/width-1, 2*(y+h/2)/height-1] if valid else None)


class RealCamera(Node):
    def __init__(self):
        super().__init__('jol_real_camera')
        if self.get_parameter('use_sim_time').value:
            raise ValueError('Real camera requires use_sim_time=false')
        self.add_on_set_parameters_callback(self.validate_parameters)
        device = require_external_camera(self.declare_parameter('device', '').value)
        self.width = int(self.declare_parameter('width', 640).value)
        self.height = int(self.declare_parameter('height', 480).value)
        fps = float(self.declare_parameter('fps', 10.).value)
        self.gui = bool(self.declare_parameter('gui', True).value)
        self.seconds = float(self.declare_parameter('seconds', 0.).value)
        self.initial_roi = self.declare_parameter('initial_roi', '').value
        if not (0 < self.width <= 4096 and 0 < self.height <= 2160
                and math.isfinite(fps) and 0 < fps <= 60
                and math.isfinite(self.seconds) and self.seconds >= 0):
            raise ValueError('Invalid camera dimensions, fps, or duration')
        if self.initial_roi:
            self.initial_roi = json.loads(self.initial_roi)
            if not target_message(0., self.width, self.height, self.initial_roi)['valid']:
                raise ValueError('initial_roi must be [x,y,width,height] inside the image')
        self.wall_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        # 로컬 원본 영상은 재전송을 허용해 큰 ROS 메시지의 누락을 줄인다.
        image_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.image_pub = self.create_publisher(Image, '/jol/camera/image_raw', image_qos)
        self.info_pub = self.create_publisher(CameraInfo, '/jol/camera/camera_info', qos_profile_sensor_data)
        self.target_pub = self.create_publisher(String, '/jol/real_camera/target', 10)
        self.create_subscription(Image, '/jol/camera/annotated', self.receive_overlay, qos_profile_sensor_data)
        self.create_subscription(String, '/jol/real_arm/status', lambda msg: self.receive_status('arm', msg), 10)
        self.create_subscription(String, '/jol/real_tracking/status', lambda msg: self.receive_status('tracking', msg), 10)
        self.status, self.overlay = {}, None
        self.tracker = self.bbox = None
        self.selection_id = self.failures = 0
        self.done, self.started = False, time.monotonic()
        # 카메라 응답 대기는 1초로 제한한다.
        os.environ.setdefault('OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT', '1')
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError('Cannot open camera: '+device)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.gui:
            cv2.namedWindow('JOL real camera', cv2.WINDOW_NORMAL)
        self.create_timer(1/fps, self.tick, clock=self.steady_clock)
        self.get_logger().info('Camera uncalibrated (K[0]=0); NO MOTOR COMMANDS. R: select ROI, C: clear, Q: quit.')

    @staticmethod
    def validate_parameters(parameters):
        forbidden = any(p.name == 'use_sim_time' and p.value for p in parameters)
        return SetParametersResult(successful=not forbidden, reason='Real camera requires wall time' if forbidden else '')

    def receive_overlay(self, msg):
        if msg.header.frame_id != 'real_camera_optical':
            return
        try:
            self.overlay = (image_bgr(msg), msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9)
        except (ValueError, cv2.error):
            self.overlay = None

    def receive_status(self, key, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.status[key] = (data, time.monotonic())
        except (ValueError, TypeError):
            pass

    def publish_target(self, stamp=None):
        target = target_message(self.wall_clock.now().nanoseconds*1e-9 if stamp is None else stamp,
                                self.width, self.height, self.bbox, self.selection_id)
        self.target_pub.publish(String(data=json.dumps(target, allow_nan=False)))
        return target

    def clear(self):
        self.tracker = self.bbox = None
        self.selection_id += 1
        self.publish_target()

    def select(self, frame, bbox):
        self.clear()
        if not target_message(0., self.width, self.height, bbox)['valid']:
            return
        tracker = cv2.TrackerCSRT_create()
        try:
            if tracker.init(frame, tuple(int(v) for v in bbox)) is False:
                return
            self.tracker, self.bbox = tracker, bbox
        except cv2.error as error:
            self.get_logger().warning('ROI selection failed: '+str(error))

    def tick(self):
        if self.done:
            return
        if self.seconds and time.monotonic()-self.started >= self.seconds:
            self.done = True
            self.clear()
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.clear()
            self.failures += 1
            if self.failures >= 5:
                self.get_logger().error('Camera lost after 5 failed reads; exiting')
                self.done = True
            return
        self.failures = 0
        self.height, self.width = frame.shape[:2]
        stamp = self.wall_clock.now().to_msg()
        msg = Image()
        msg.header.stamp, msg.header.frame_id = stamp, 'real_camera_optical'
        msg.height, msg.width, msg.encoding, msg.step = self.height, self.width, 'bgr8', self.width*3
        msg.data = array('B', frame.tobytes())
        self.image_pub.publish(msg)
        info = CameraInfo()  # K[0]=0은 미보정 카메라를 뜻한다.
        info.header, info.height, info.width = msg.header, self.height, self.width
        self.info_pub.publish(info)
        if self.initial_roi:
            self.select(frame, self.initial_roi)
            self.initial_roi = ''
        elif self.tracker is not None:
            try:
                tracked, bbox = self.tracker.update(frame)
                self.bbox = bbox if tracked else None
                if not target_message(0., self.width, self.height, self.bbox)['valid']:
                    self.clear()
            except cv2.error:
                self.clear()
        target = self.publish_target(stamp.sec+stamp.nanosec*1e-9)
        if self.gui:
            self.show(frame, target)

    def show(self, frame, target):
        display = frame.copy()
        if target['valid']:
            x, y, w, h = map(int, target['bbox'])
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        lines = ['NO MOTOR COMMANDS | UNCALIBRATED', 'R: select ROI | C: clear | Q: quit',
                 'ROI: '+('TRACKING' if target['valid'] else 'NOT SELECTED / LOST')]
        for key in ('arm', 'tracking'):
            value, seen = self.status.get(key, ({'status': 'waiting'}, 0.))
            age = time.monotonic()-seen
            state = str(value.get('state', value.get('status', 'waiting')))
            lines.append(f'{key}: '+('FRESH ' if age < 2 else 'STALE ')+state[:70])
            if key == 'arm':
                voltages = [motor.get('voltage_v') for motor in value.get('motors', [])
                            if isinstance(motor, dict) and isinstance(motor.get('voltage_v'), (int, float))]
                if voltages:
                    lines.append(f'Motor voltage: {min(voltages):.1f} .. {max(voltages):.1f} V')
                lines.extend('FAULT: '+str(fault)[:85] for fault in value.get('faults', [])[:2])
        for index, line in enumerate(lines):
            color = (0, 0, 255) if 'FAULT' in line else (0, 255, 255)
            cv2.putText(display, line, (10, 22+index*23), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1)
        cv2.imshow('JOL real camera', display)
        if self.overlay is not None:
            overlay, stamp = self.overlay
            age = self.wall_clock.now().nanoseconds*1e-9-stamp
            view = overlay.copy()
            fresh = 0 <= age <= .8
            label = f'NN overlay: {"FRESH" if fresh else "STALE"} ({age:.2f}s)'
            cv2.putText(view, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (0, 255, 0) if fresh else (0, 0, 255), 2)
            cv2.imshow('JOL NN overlay', view)
        key = cv2.waitKey(1) & 0xff
        if key in (ord('r'), ord('R')):
            self.clear()  # 영역 선택을 기다리는 동안 기존 목표를 무효화한다.
            bbox = cv2.selectROI('JOL select target (Enter: accept, Esc: cancel)', frame, False, False)
            cv2.destroyWindow('JOL select target (Enter: accept, Esc: cancel)')
            self.select(frame, bbox)
        elif key in (ord('c'), ord('C')):
            self.clear()
        # GTK 창 상태값 대신 Q/Esc로 종료한다.
        elif key in (ord('q'), ord('Q'), 27):
            self.done = True
            self.clear()

    def close(self):
        if rclpy.ok():
            self.clear()
        self.capture.release()
        if self.gui:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RealCamera()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
