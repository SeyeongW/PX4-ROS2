"""Passive camera views; image/status subscriptions never issue flight commands."""
import json
import math
import time


def image_stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def current_frame(message, received, now, timeout):
    """Both exposure time and receipt time must be current, including paused sim."""
    return (message is not None and received is not None
            and -.05 <= now - image_stamp(message) <= timeout
            and 0 <= time.monotonic_ns() - received <= timeout * 1e9)


def camera_node(base):
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    class CameraNode(base):
        def __init__(self, **kwargs):
            self.onboard_frames = {}
            self.onboard_observation = {}
            self.onboard_observation_received = None
            super().__init__(**kwargs)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            for kind, suffix in (('raw', 'image_raw'), ('annotated', 'annotated')):
                self.create_subscription(Image, '/jol/onboard_camera/' + suffix,
                    lambda msg, kind=kind: self.receive_onboard_image(kind, msg), qos)
            self.create_subscription(String, '/jol/onboard_camera/observation',
                                     self.receive_onboard_observation, 1)

        def check_clock(self):
            previous = self.last_ros_stamp
            super().check_clock()
            if previous is not None and self.last_ros_stamp < previous:
                self.onboard_frames.clear()
                self.onboard_observation = {}
                self.onboard_observation_received = None

        def receive_onboard_image(self, kind, message):
            self.check_clock()
            # One reference per source; no frame queue or copy in ROS callbacks.
            self.onboard_frames[kind] = (message, time.monotonic_ns())

        def receive_onboard_observation(self, message):
            self.check_clock()
            try:
                packet = json.loads(message.data)
                stamp = packet.get('observation_stamp') if isinstance(packet, dict) else None
                if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                        or not math.isfinite(stamp)):
                    raise ValueError('invalid_observation_stamp')
                self.onboard_observation = packet
                self.onboard_observation_received = time.monotonic_ns()
            except (ValueError, TypeError):
                self.onboard_observation = {}
                self.onboard_observation_received = None

        def onboard_display(self):
            now = self.clocks()['sim_stamp']
            raw, raw_received = self.onboard_frames.get('raw', (None, None))
            for kind in ('annotated', 'raw'):
                message, received = self.onboard_frames.get(kind, (None, None))
                if current_frame(message, received, now, self.stale_timeout):
                    # Prefer live raw video when annotated processing falls >250ms behind.
                    if (kind == 'annotated'
                            and current_frame(raw, raw_received, now, self.stale_timeout)
                            and image_stamp(raw) - image_stamp(message) > .25):
                        continue
                    observation = self.onboard_observation
                    age = time.monotonic_ns() - (self.onboard_observation_received or 0)
                    if not (kind == 'annotated' and observation.get('frame_fresh') is True
                            and abs(image_stamp(message) - observation.get('observation_stamp', -math.inf)) < 1e-6
                            and 0 <= age <= self.stale_timeout * 1e9):
                        observation = {}
                    return message, kind, observation
            return None, 'stale' if self.onboard_frames else 'waiting', {}

    return CameraNode


def camera_dashboard(base):
    from PyQt5 import QtCore, QtGui, QtWidgets

    class CameraPanel(QtWidgets.QFrame):
        def __init__(self):
            super().__init__()
            self.setObjectName('onboardCameraPanel')
            self.setProperty('role', 'panel')
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(10, 6, 10, 6)
            layout.setSpacing(4)
            title = QtWidgets.QLabel('드론 탑재 카메라 · 영상 모니터')
            title.setProperty('role', 'heading')
            layout.addWidget(title)
            subtitle = QtWidgets.QLabel('고정 기수 카메라 / 원본 비율')
            subtitle.setProperty('role', 'muted')
            layout.addWidget(subtitle)
            self.image = QtWidgets.QLabel()
            self.image.setObjectName('onboardCameraPreview')
            self.image.setAlignment(QtCore.Qt.AlignCenter)
            self.image.setMinimumSize(300, 135)
            self.image.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
            self.image.setStyleSheet('background:#252a31; color:#d6dde6;')
            layout.addWidget(self.image, 1)
            self.summary = QtWidgets.QLabel()
            self.summary.setObjectName('onboardCameraSummary')
            self.summary.setTextFormat(QtCore.Qt.PlainText)
            self.summary.setWordWrap(True)
            layout.addWidget(self.summary)
            self.message = self.pixmap = None

        def update_feed(self, message, kind, observation):
            if message is None:
                self.message = self.pixmap = None
                self.image.setText('영상 수신 대기' if kind == 'waiting' else '영상 두절 / 과거 프레임 숨김')
                self.summary.setText('검출 상태 미확인 · 영상 모니터 전용')
                return
            if message is not self.message:
                try:
                    channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}
                    if (message.encoding not in channels or not 0 < message.width <= 8192
                            or not 0 < message.height <= 8192
                            or message.step < message.width * channels[message.encoding]
                            or len(message.data) < message.step * message.height):
                        raise ValueError('unsupported or malformed image')
                    fmt = QtGui.QImage.Format_Grayscale8 if message.encoding == 'mono8' else QtGui.QImage.Format_RGB888
                    image = QtGui.QImage(bytes(message.data), message.width, message.height, message.step, fmt).copy()
                    if message.encoding == 'bgr8':
                        image = image.rgbSwapped()
                    self.pixmap = QtGui.QPixmap.fromImage(image)
                    self.message = message
                except (ValueError, TypeError):
                    self.message = self.pixmap = None
                    self.image.setText('영상 형식 오류')
                    self.summary.setText('검출 상태 미확인 · 영상 모니터 전용')
                    return
            self.image.setPixmap(self.pixmap.scaled(self.image.size(), QtCore.Qt.KeepAspectRatio,
                                                    QtCore.Qt.FastTransformation))
            detected = ('풍선 후보 검출' if observation.get('detected') is True else
                        '풍선 미검출' if observation.get('detected') is False else '검출 상태 미확인')
            source = '기존 검출 주석' if kind == 'annotated' else '원본 영상 · 검출 주석 대기'
            self.summary.setText(f'{source} · 노출 {image_stamp(message):.2f}s\n{detected} · 영상 모니터 전용')

    class CameraDashboard(base):
        def __init__(self, node, start_timer=True):
            super().__init__(node, start_timer=start_timer)
            self.camera_label.setMinimumWidth(300)
            self.onboard_panel = CameraPanel()
            panels = self.content_splitter.widget(0)
            panels.insertWidget(1, self.onboard_panel)
            panels.setSizes([400, 400, 480])
            self.refresh()

        def refresh(self):
            super().refresh()
            if not hasattr(self, 'onboard_panel'):
                return
            self.onboard_panel.update_feed(*self.node.onboard_display())
            if not current_frame(self.node.display_image(), self.node.received.get('image'),
                                 self.node.clocks()['sim_stamp'], self.node.stale_timeout):
                self.camera_label.setText('영상 수신 대기 / 과거 프레임 숨김')
                self.camera_summary.setText('짐벌 영상 미수신 / 두절 · 검출 상태 미확인')

    return CameraDashboard
