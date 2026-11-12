#!/usr/bin/env python3
"""지상 짐벌 관측 카메라(좌측) + 드론 온보드 노즈콘 카메라(우측) + 출격 승인 트리거 통합 뷰어.

조작 키 (뷰어 창 클릭 후 입력):
  - [SPACE]/[ENTER]/[y]: 유효한 좌표 송신 승인 요청
  - [c]/[ESC]: 좌표 송신 승인 취소 요청 (FC 비행 중단 명령이 아님)
"""
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from vision_msgs.msg import Detection2DArray

from .ros_utils import image_to_bgr

class DetectionViewerNode(Node):
    def __init__(self):
        super().__init__('detection_viewer_node')
        self.declare_parameter('image_topic', '/jol/onboard_camera/image_raw')
        self.declare_parameter('detection_topic', '/balloon/onboard/detections')
        self.declare_parameter('ground_image_topic', '/jol/camera/image_raw')
        self.declare_parameter('ground_detection_topic', '/balloon/detections')
        self.declare_parameter('ukf_status_topic', '/balloon/status')
        self.declare_parameter('bridge_status_topic', '/balloon/bridge_status')
        self.declare_parameter('max_age_sec', 1.0)
        self.declare_parameter('simulation_mode', False)
        self.declare_parameter('simulation_approval_file', '')
        self.max_age = self.get_parameter('max_age_sec').value
        if not math.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError('max_age_sec must be finite and positive')
        self.simulation_mode = bool(self.get_parameter('simulation_mode').value)
        flag_path = self.get_parameter('simulation_approval_file').value
        self.simulation_flag = Path(flag_path) if self.simulation_mode and flag_path else None

        self.latest_detections = []
        self.ground_detections = []
        self.ground_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.onboard_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.ukf_status = {}
        self.frame_count = 0
        self.ukf_rx_count = 0
        self.ukf_hz = 0.0
        self._last_ukf_wall = None
        self.approval_requested = False
        self.cancel_requested = False
        self.bridge_status = {}
        self.t_bridge = None
        self.t_ground = self.t_onboard = None
        approval_topic = '/mission/approve_launch' if self.simulation_mode else '/balloon/launch_approved'
        self.approve_pub = self.create_publisher(Bool, approval_topic, 10)

        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            Detection2DArray,
            self.get_parameter('detection_topic').value,
            self._on_detection, 10)
        self.create_subscription(
            Image,
            self.get_parameter('ground_image_topic').value,
            self._on_ground_image, qos_profile_sensor_data)
        self.create_subscription(
            Detection2DArray,
            self.get_parameter('ground_detection_topic').value,
            self._on_ground_detection, 10)
        self.create_subscription(
            String,
            self.get_parameter('ukf_status_topic').value,
            self._on_ukf_status, 10)
        self.create_subscription(Bool, approval_topic, self._on_approve_msg, 10)
        self.create_subscription(String, self.get_parameter('bridge_status_topic').value,
                                 self._on_bridge_status, 10)
        self.create_timer(0.05, self._render_dashboard)

    def _on_approve_msg(self, msg):
        # This is a request, never evidence that the bridge accepted or FC flew.
        self.approval_requested = bool(msg.data)
        self.cancel_requested = not msg.data

    def _on_bridge_status(self, msg):
        try:
            packet = json.loads(msg.data)
            if not isinstance(packet, dict):
                raise ValueError('bridge status must be an object')
            self.bridge_status = packet
            self.t_bridge = time.monotonic()
        except (ValueError, TypeError):
            self.bridge_status = {}
            self.t_bridge = None

    def _bridge_fresh(self):
        stamp = self.bridge_status.get('stamp')
        now = self.get_clock().now().nanoseconds * 1e-9
        return (self.t_bridge is not None and 0 <= time.monotonic() - self.t_bridge <= self.max_age
                and type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= now - stamp <= self.max_age)

    def _coordinate_ready(self):
        if not self.simulation_mode:
            return self._bridge_fresh() and self.bridge_status.get('coordinate_valid') is True
        stamp = self.ukf_status.get('measurement_stamp')
        current = self.get_clock().now().nanoseconds * 1e-9
        return (self.ukf_status.get('valid') is True and self._last_ukf_wall is not None
                and 0 <= time.monotonic() - self._last_ukf_wall <= self.max_age
                and type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= current - stamp <= self.max_age)

    def _request_approval(self, approve):
        if approve and (self.approval_requested or not self._coordinate_ready()):
            return False
        if self.simulation_flag is not None:
            try:
                if approve:
                    self.simulation_flag.write_text('approved\n')
                else:
                    self.simulation_flag.unlink(missing_ok=True)
            except OSError as exc:
                self.get_logger().error(f'Simulation approval file failed: {exc}')
                return False
        self.approve_pub.publish(Bool(data=approve))
        self.approval_requested = approve
        self.cancel_requested = not approve
        self.get_logger().info('Coordinate approval requested' if approve else
                               'Coordinate approval cancellation requested; FC flight stop is NOT confirmed')
        return True

    def _status_lines(self):
        fc_line = 'FC STATUS UNKNOWN | Flight / stop not confirmed'
        prefix = 'SIMULATION ' if self.simulation_mode else ''
        state = self.bridge_status.get('approval_state') if self._bridge_fresh() else None
        if self.simulation_mode:
            action = 'CANCEL REQUEST SENT' if self.cancel_requested else ('APPROVAL REQUEST SENT' if self.approval_requested else 'NO REQUEST')
            return prefix + action + ' | Simulation flight / stop not confirmed', fc_line + ' | SPACE: request, C/ESC: cancel request'
        if self.cancel_requested and state != 'cancelled':
            line = 'CANCEL REQUEST SENT | Awaiting bridge confirmation; FC flight stop not confirmed'
        elif self.approval_requested and state in (None, 'waiting'):
            line = 'APPROVAL REQUEST SENT | Awaiting bridge confirmation'
        elif state is not None:
            sending = 'TRANSMITTING (local send only)' if self.bridge_status.get('broadcasting') is True else 'PAUSED'
            line = f'APPROVAL {str(state).upper()} | COORDINATE BROADCAST {sending}'
        else:
            line = 'NO CURRENT BRIDGE CONFIRMATION'
        if self._bridge_fresh():
            fc = self.bridge_status.get('fc')
            if isinstance(fc, dict) and fc.get('heartbeat_fresh') is True and type(fc.get('armed')) is bool:
                armed = 'ARMED' if fc['armed'] else 'DISARMED'
                fc_line = f'FC HEARTBEAT: {armed} | Flight / stop not confirmed'
        return prefix + line, fc_line + ' | SPACE: request, C/ESC: cancel coordinate broadcast'

    def _on_detection(self, msg):
        self.latest_detections = msg.detections

    def _on_ground_detection(self, msg):
        self.ground_detections = msg.detections

    def _on_ukf_status(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                raise ValueError('status must be an object')
            self.ukf_status = data
            now_w = time.monotonic()
            if self._last_ukf_wall is not None and now_w > self._last_ukf_wall:
                inst_hz = 1.0 / (now_w - self._last_ukf_wall)
                self.ukf_hz = 0.8 * self.ukf_hz + 0.2 * inst_hz if self.ukf_hz > 0 else inst_hz
            self._last_ukf_wall = now_w
            if data.get('ned') is not None:
                self.ukf_rx_count += 1
        except (ValueError, TypeError):
            self.ukf_status = {}
            self._last_ukf_wall = None

    def _on_ground_image(self, msg):
        try:
            g_raw = image_to_bgr(msg)
        except ValueError:
            return
        h_raw, w_raw = g_raw.shape[:2]
        g_vis = cv2.resize(g_raw, (640, 360))
        sx = 640.0 / max(w_raw, 1)
        sy = 360.0 / max(h_raw, 1)

        # 중앙 조준선(Crosshair) 표시
        cv2.drawMarker(g_vis, (320, 180), (140, 140, 140), cv2.MARKER_CROSS, 28, 1)

        for det in self.ground_detections:
            cx = int(det.bbox.center.position.x * sx)
            cy = int(det.bbox.center.position.y * sy)
            diam_orig = (det.bbox.size_x + det.bbox.size_y) * 0.5
            r = max(6, int(diam_orig * sx * 0.5))
            score = det.results[0].hypothesis.score if det.results else 0.0
            cv2.circle(g_vis, (cx, cy), r + 2, (0, 255, 255), 2)
            cv2.drawMarker(g_vis, (cx, cy), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 1)
            cv2.putText(
                g_vis,
                f'BALLOON VERIFIED (score={score:.2f}, d={diam_orig:.1f}px)',
                (max(10, cx - 110), max(22, cy - r - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

        cv2.putText(g_vis, '1. GROUND GIMBAL CAMERA (12deg Telephoto + 6-DOF UKF)',
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2)

        ned = self.ukf_status.get('ned')
        sigma = self.ukf_status.get('sigma_m')
        ukf_n = self.ukf_status.get('n', 0)
        ukf_valid = self._coordinate_ready()
        if (isinstance(ned, list) and len(ned) == 3
                and all(type(v) in (int, float) and math.isfinite(v) for v in ned)
                and type(sigma) in (int, float) and math.isfinite(sigma)):
            wx, wy, wz = ned[1], ned[0], -ned[2]
            sig_str = f'{sigma:.2f}m' if sigma is not None else '?'
            tag = 'LOCKED' if ukf_valid else 'CONVERGING'
            col = (60, 255, 60) if ukf_valid else (0, 215, 255)
            cv2.putText(
                g_vis,
                f'UKF[{tag}] World=({wx:.1f}, {wy:.1f}, {wz:.1f})m | n={ukf_n} | s={sig_str}',
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
            cv2.putText(
                g_vis,
                f'Local estimator input: {self.ukf_hz:.1f} Hz (rx={self.ukf_rx_count} pkts)',
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 255, 220), 1)
        else:
            cv2.putText(
                g_vis,
                f'Scanning sky sector for balloon... (n={ukf_n})',
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 200, 255), 1)

        self.ground_frame = g_vis
        self.t_ground = time.monotonic()

    def _on_image(self, msg):
        try:
            frame = image_to_bgr(msg)
        except ValueError:
            return

        if frame.shape[1] != 640 or frame.shape[0] != 360:
            frame = cv2.resize(frame, (640, 360))

        cv2.drawMarker(frame, (320, 180), (120, 120, 120), cv2.MARKER_CROSS, 24, 1)

        for det in self.latest_detections:
            cx = int(det.bbox.center.position.x)
            cy = int(det.bbox.center.position.y)
            r = max(4, int(det.bbox.size_x / 2))
            score = det.results[0].hypothesis.score if det.results else 0.0
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            cv2.putText(frame, f'LOS LOCK {score:.2f} (d={det.bbox.size_x:.0f}px)',
                        (max(8, cx - r - 20), max(20, cy - r - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1)

        self.frame_count += 1
        n = len(self.latest_detections)
        cv2.putText(frame, '2. DRONE ONBOARD NOSE-CONE CAMERA (66deg Wide)',
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 120), 2)
        cv2.putText(frame, f'Onboard Detections={n}  Frame={self.frame_count}',
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)

        self.onboard_frame = frame
        self.t_onboard = time.monotonic()
        self._render_dashboard()

    def _render_dashboard(self):
        now = time.monotonic()
        frames = []
        for frame, received in ((self.ground_frame, self.t_ground), (self.onboard_frame, self.t_onboard)):
            if received is None or not 0 <= now - received <= self.max_age:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, 'IMAGE UNAVAILABLE / STALE', (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (180, 200, 255), 1)
            frames.append(frame)
        top = np.hstack(frames)
        banner = np.zeros((68, 1280, 3), dtype=np.uint8)
        banner[:] = (45, 35, 25)
        for i, line in enumerate(self._status_lines()):
            cv2.putText(banner, line, (14, 25 + i * 27),
                        cv2.FONT_HERSHEY_SIMPLEX, .53, (200, 210, 255), 1)
        cv2.imshow('onboard', np.vstack((top, banner)))
        key = cv2.waitKey(1) & 0xFF
        if key in (32, 13, 10, ord('y'), ord('Y')):
            self._request_approval(True)
        elif key in (27, ord('c'), ord('C')):
            self._request_approval(False)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
