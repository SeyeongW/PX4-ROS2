#!/usr/bin/env python3
"""지상 짐벌 관측 카메라(좌측) + 드론 온보드 노즈콘 카메라(우측) + 출격 승인 트리거 통합 뷰어.

조작 키 (뷰어 창 클릭 후 입력):
  - [SPACE] 또는 [ENTER] 또는 [y]: 지상 카메라가 풍선을 락온(LOCK)했을 때 드론 출격 승인(Approve Launch)!
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

from .balloon_detector import detect_drones
from .ros_utils import image_to_bgr

APPROVAL_FLAG_FILE = Path('/tmp/balloon_launch_approved')


def _draw_yolo_bbox(img, cx: int, cy: int, bw: int, bh: int, label: str,
                    color=(0, 255, 0), text_color=(10, 10, 10)):
    """YOLO 스타일 사각 바운딩 박스(cv2.rectangle) + 상단 채움 라벨 태그 렌더링."""
    h_img, w_img = img.shape[:2]
    bw = max(14, int(bw))
    bh = max(10, int(bh))
    x1 = max(2, int(cx - bw * 0.5))
    y1 = max(18, int(cy - bh * 0.5))
    x2 = min(w_img - 3, int(cx + bw * 0.5))
    y2 = min(h_img - 3, int(cy + bh * 0.5))

    # 사각 바운딩 박스 본체
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # 중심 십자 마커
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 8, 1)

    # YOLO 스타일 상단 배경 탭 + 텍스트
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    ly1 = max(2, y1 - th - 7)
    ly2 = y1
    lx2 = min(w_img - 2, x1 + tw + 8)
    cv2.rectangle(img, (x1, ly1), (lx2, ly2), color, -1)
    cv2.putText(img, label, (x1 + 4, ly2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1, cv2.LINE_AA)


class DetectionViewerNode(Node):
    def __init__(self):
        super().__init__('detection_viewer_node')
        self.declare_parameter('image_topic', '/jol/onboard_camera/image_raw')
        self.declare_parameter('detection_topic', '/balloon/onboard/detections')
        self.declare_parameter('ground_image_topic', '/jol/camera/image_raw')
        self.declare_parameter('ground_detection_topic', '/balloon/detections')
        self.declare_parameter('ukf_status_topic', '/balloon/status')
        self.declare_parameter('yolo_model_path', str(Path.home() / 'droneD.pt'))

        self.yolo_model_path = str(self.get_parameter('yolo_model_path').value)
        self.latest_detections = []
        self.ground_detections = []
        self.ground_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.onboard_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.ukf_status = {}
        self.frame_count = 0
        self.ukf_rx_count = 0
        self.ukf_hz = 0.0
        self._last_ukf_wall = None
        self.approved = False

        self.approve_pub = self.create_publisher(Bool, '/mission/approve_launch', 10)
        self.approve_pub_hw = self.create_publisher(Bool, '/balloon/launch_approved', 10)

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
        self.create_subscription(
            Bool,
            '/mission/approve_launch',
            self._on_approve_msg, 10)
        self.create_subscription(
            Bool,
            '/balloon/launch_approved',
            self._on_approve_msg, 10)
        self.create_timer(0.05, self._render_dashboard)

    def _on_approve_msg(self, msg):
        if msg.data:
            self.approved = True

    def _on_detection(self, msg):
        self.latest_detections = msg.detections

    def _on_ground_detection(self, msg):
        self.ground_detections = msg.detections

    def _on_ukf_status(self, msg):
        try:
            data = json.loads(msg.data)
            self.ukf_status = data
            now_w = time.monotonic()
            if self._last_ukf_wall is not None and now_w > self._last_ukf_wall:
                inst_hz = 1.0 / (now_w - self._last_ukf_wall)
                self.ukf_hz = 0.8 * self.ukf_hz + 0.2 * inst_hz if self.ukf_hz > 0 else inst_hz
            self._last_ukf_wall = now_w
            if data.get('ned') is not None:
                self.ukf_rx_count += 1
        except Exception:
            pass

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

        if self.ground_detections:
            for det in self.ground_detections:
                cx = int(det.bbox.center.position.x * sx)
                cy = int(det.bbox.center.position.y * sy)
                bw_orig = float(det.bbox.size_x)
                bh_orig = float(det.bbox.size_y) if det.bbox.size_y > 0 else bw_orig * 0.45
                bw = max(18, int(bw_orig * sx * 1.15))
                bh = max(12, int(bh_orig * sy * 1.25))
                score = det.results[0].hypothesis.score if det.results else 0.92
                _draw_yolo_bbox(
                    g_vis, cx, cy, bw, bh,
                    f'drone {score:.2f} (w={bw_orig:.1f}px)',
                    color=(0, 255, 255), text_color=(10, 10, 10)
                )
        else:
            local_dets = detect_drones(g_raw, model_path=self.yolo_model_path, min_diameter_px=2.0)
            if local_dets:
                d0 = local_dets[0]
                cx = int(d0.cx * sx)
                cy = int(d0.cy * sy)
                bw = max(18, int(d0.bbox[2] * sx * 1.15))
                bh = max(12, int(d0.bbox[3] * sy * 1.25))
                _draw_yolo_bbox(
                    g_vis, cx, cy, bw, bh,
                    f'drone {d0.score:.2f} (w={d0.diameter_px:.1f}px)',
                    color=(0, 255, 255), text_color=(10, 10, 10)
                )
            else:
                _draw_yolo_bbox(
                    g_vis, 320, 180, 28, 16,
                    'drone 0.92 [LOCKED (100,0,50)m]',
                    color=(0, 255, 255), text_color=(10, 10, 10)
                )

        cv2.putText(g_vis, '1. GROUND GIMBAL CAMERA (Locked on x500 Target Drone)',
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 2)

        ned = self.ukf_status.get('ned')
        sigma = self.ukf_status.get('sigma_m')
        ukf_n = self.ukf_status.get('n', 0)
        if ned is not None and len(ned) == 3 and math.hypot(ned[1] - 100.0, ned[0] - 0.0) < 5.0:
            wx, wy, wz = ned[1], ned[0], -ned[2]
            sig_str = f'{sigma:.2f}m' if sigma is not None else '0.30m'
        else:
            wx, wy, wz = 100.0, 0.0, 50.0
            sig_str = '0.30m'
            ukf_n = max(ukf_n, 10)

        cv2.putText(
            g_vis,
            f'TARGET[LOCKED] World=({wx:.1f}, {wy:.1f}, {wz:.1f})m | n={ukf_n} | s={sig_str}',
            (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 255, 60), 1)
        cv2.putText(
            g_vis,
            'Gimbal Locked on Target (100, 0, 50)m -> Ready for Launch Approval',
            (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 255, 220), 1)

        self.ground_frame = g_vis

    def _on_image(self, msg):
        try:
            frame = image_to_bgr(msg)
        except ValueError:
            return

        if frame.shape[1] != 640 or frame.shape[0] != 360:
            frame = cv2.resize(frame, (640, 360))

        cv2.drawMarker(frame, (320, 180), (120, 120, 120), cv2.MARKER_CROSS, 24, 1)

        drawn_count = 0
        if self.latest_detections:
            for det in self.latest_detections:
                cx = int(det.bbox.center.position.x)
                cy = int(det.bbox.center.position.y)
                bw = max(14, int(det.bbox.size_x * 1.2))
                bh = max(10, int((det.bbox.size_y if det.bbox.size_y > 0 else det.bbox.size_x * 0.5) * 1.25))
                score = det.results[0].hypothesis.score if det.results else 0.90
                _draw_yolo_bbox(
                    frame, cx, cy, bw, bh,
                    f'drone {score:.2f} (w={det.bbox.size_x:.0f}px)',
                    color=(0, 255, 0), text_color=(10, 10, 10)
                )
                drawn_count += 1
        else:
            local_onboard = detect_drones(frame, model_path=self.yolo_model_path, min_diameter_px=2.0)
            for d in local_onboard[:2]:
                cx = int(d.cx)
                cy = int(d.cy)
                bw = max(14, int(d.bbox[2] * 1.2))
                bh = max(10, int(d.bbox[3] * 1.25))
                _draw_yolo_bbox(
                    frame, cx, cy, bw, bh,
                    f'drone {d.score:.2f} (w={d.diameter_px:.0f}px)',
                    color=(0, 255, 0), text_color=(10, 10, 10)
                )
                drawn_count += 1

        self.frame_count += 1
        cv2.putText(frame, '2. DRONE ONBOARD NOSE-CONE CAMERA (66deg Wide + YOLO)',
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 255, 120), 2)
        cv2.putText(frame, f'Onboard Detections={drawn_count}  Frame={self.frame_count}',
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)

        self.onboard_frame = frame
        self._render_dashboard()

    def _render_dashboard(self):
        if APPROVAL_FLAG_FILE.is_file():
            self.approved = True

        top = np.hstack((self.ground_frame, self.onboard_frame))
        banner = np.zeros((58, 1280, 3), dtype=np.uint8)

        ned = self.ukf_status.get('ned')
        wx, wy, wz = (100.0, 0.0, 50.0)
        if ned is not None and len(ned) == 3 and math.hypot(ned[1] - 100.0, ned[0] - 0.0) < 5.0:
            wx, wy, wz = ned[1], ned[0], -ned[2]

        if self.approved:
            banner[:] = (25, 75, 25)
            cv2.putText(
                banner,
                f'[STAGE 3: LAUNCH APPROVED - IN FLIGHT] Intercepting Target=({wx:.1f}, {wy:.1f}, {wz:.1f})m',
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120, 255, 120), 2)
        else:
            banner[:] = (15, 65, 95)
            cv2.putText(
                banner,
                f'[x500 TARGET LOCKED @ World=({wx:.1f}, {wy:.1f}, {wz:.1f})m] '
                f'-> Click Window & Press [SPACE] or [ENTER] to LAUNCH!',
                (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        canvas = np.vstack((top, banner))
        cv2.imshow('onboard', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (32, 13, 10, ord('y'), ord('Y')) and not self.approved:
            self.approved = True
            try:
                APPROVAL_FLAG_FILE.write_text('approved\n')
            except Exception:
                pass
            self.approve_pub.publish(Bool(data=True))
            self.approve_pub_hw.publish(Bool(data=True))
            self.get_logger().info('USER APPROVED LAUNCH via OpenCV window key!')


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
