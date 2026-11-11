#!/usr/bin/env python3
"""지상 짐벌 관측 카메라(좌측) + 드론 온보드 노즈콘 카메라(우측) + 출격 승인 트리거 통합 뷰어.

조작 키 (뷰어 창 클릭 후 입력):
  - [SPACE] 또는 [ENTER] 또는 [y]: 지상 카메라가 풍선을 락온(LOCK)했을 때 드론 출격 승인(Approve Launch)!
"""
import json
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

APPROVAL_FLAG_FILE = Path('/tmp/balloon_launch_approved')


class DetectionViewerNode(Node):
    def __init__(self):
        super().__init__('detection_viewer_node')
        self.declare_parameter('image_topic', '/jol/onboard_camera/image_raw')
        self.declare_parameter('detection_topic', '/balloon/onboard/detections')
        self.declare_parameter('ground_image_topic', '/jol/camera/image_raw')
        self.declare_parameter('ground_detection_topic', '/balloon/detections')
        self.declare_parameter('ukf_status_topic', '/balloon/status')

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
        ukf_valid = bool(self.ukf_status.get('valid', False)) or (
            ned is not None and ukf_n >= 3 and sigma is not None and sigma <= 2.8
        )
        if ned is not None and len(ned) == 3:
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
                f'Ground->Drone Coord Stream: {self.ukf_hz:.1f} Hz (rx={self.ukf_rx_count} pkts)',
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 255, 220), 1)
        else:
            cv2.putText(
                g_vis,
                f'Scanning sky sector for balloon... (n={ukf_n})',
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 200, 255), 1)

        self.ground_frame = g_vis

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
        self._render_dashboard()

    def _render_dashboard(self):
        if APPROVAL_FLAG_FILE.is_file():
            self.approved = True

        top = np.hstack((self.ground_frame, self.onboard_frame))
        banner = np.zeros((58, 1280, 3), dtype=np.uint8)

        ned = self.ukf_status.get('ned')
        sigma = self.ukf_status.get('sigma_m')
        ukf_n = self.ukf_status.get('n', 0)
        ukf_locked = bool(
            ned is not None and ukf_n >= 3 and sigma is not None and sigma <= 2.8
        )

        if self.approved:
            banner[:] = (25, 75, 25)
            wx, wy, wz = (ned[1], ned[0], -ned[2]) if (ned and len(ned) == 3) else (90.0, 0.0, 40.0)
            cv2.putText(
                banner,
                f'[STAGE 3: LAUNCH APPROVED - IN FLIGHT] Target=({wx:.1f}, {wy:.1f}, {wz:.1f})m | '
                f'Ground UKF(10Hz) -> Onboard 3D LOS/PN Homing',
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120, 255, 120), 2)
        elif ukf_locked:
            banner[:] = (15, 65, 95)
            wx, wy, wz = ned[1], ned[0], -ned[2]
            cv2.putText(
                banner,
                f'[STAGE 2: BALLOON LOCKED! World=({wx:.1f},{wy:.1f},{wz:.1f})m, n={ukf_n}, s={sigma:.2f}m] '
                f'-> Press [SPACE] or [ENTER] to APPROVE LAUNCH!',
                (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
        else:
            banner[:] = (45, 35, 25)
            cv2.putText(
                banner,
                '[STAGE 1: SKY SEARCH] Ground Gimbal Camera scanning sky for 30cm balloon... (Drone standby on ground)',
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 210, 255), 2)

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
