import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class StreamTestNode(Node):
    """YOLO 없이 카메라(또는 테스트 패턴) 영상에 상태 오버레이를 입혀
    '영상이 잘 송출되고 있다'를 시각적으로 확인하기 위한 노드."""

    def __init__(self):
        super().__init__("stream_test_node")

        self.declare_parameter("device_id", 0)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("timer_period", 0.033)  # ~30 fps
        self.declare_parameter("debug_scale", 0.5)

        self.device_id = self.get_parameter("device_id").value
        self.frame_width = self.get_parameter("frame_width").value
        self.frame_height = self.get_parameter("frame_height").value
        self.jpeg_quality = self.get_parameter("jpeg_quality").value
        self.timer_period = self.get_parameter("timer_period").value
        self.debug_scale = self.get_parameter("debug_scale").value

        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.camera_ok = self.cap.isOpened()
        if not self.camera_ok:
            self.get_logger().warn(
                f"Camera {self.device_id} open failed -> sending synthetic TEST PATTERN."
            )

        # image_transport 규칙대로 <base>/compressed 에 발행 -> rqt에서 base만 골라도 됨
        self.image_pub = self.create_publisher(CompressedImage, "/perception/yolo_result/compressed", 10)
        self.raw_pub = self.create_publisher(CompressedImage, "/perception/raw_image/compressed", 10)

        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.fps = 0.0

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info("Stream-Test Node started. Publishing dressed-up test video.")

    def _make_test_pattern(self):
        """SMPTE 풍 컬러바 + 그라데이션. 프레임이 멈추면 바로 티가 나게."""
        h, w = self.frame_height, self.frame_width
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # BGR
        bars = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
                (255, 0, 255), (0, 0, 255), (255, 0, 0)]
        bw = max(1, w // len(bars))
        for i, c in enumerate(bars):
            frame[:, i * bw:(i + 1) * bw] = c
        grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h // 4, 1))
        frame[-h // 4:, :, 0] = grad
        frame[-h // 4:, :, 1] = grad
        frame[-h // 4:, :, 2] = grad
        return frame

    def timer_callback(self):
        t_capture = time.time()
        if self.camera_ok:
            ret, frame = self.cap.read()
            if not ret:
                frame = self._make_test_pattern()
        else:
            frame = self._make_test_pattern()

        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            self.fps = inst_fps if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst_fps
        self.frame_count += 1

        # 깨끗한 축소본(raw) + 오버레이 입힌 버전(annotated)
        raw_small = cv2.resize(frame, (0, 0), fx=self.debug_scale, fy=self.debug_scale)
        annotated = raw_small.copy()
        self._draw_overlay(annotated, now, t_capture)

        self._publish(self.image_pub, annotated, now)
        self._publish(self.raw_pub, raw_small, now)

    def _draw_overlay(self, frame, now, t_capture):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 반투명 상단 바
        bar_h = max(46, h // 6)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # 점멸 LIVE LED (멈춤 감지용)
        blink = (self.frame_count // 8) % 2 == 0
        dot_color = (0, 0, 255) if blink else (0, 0, 110)
        cv2.circle(frame, (16, 18), 7, dot_color, -1)
        cv2.putText(frame, "LIVE - STREAM OK", (30, 24), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 송신 시각 (수신측 화면 시계와 비교하면 지연 눈으로 확인 가능)
        wall = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
        cv2.putText(frame, f"sent {wall}", (10, 42), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 하단 상태 줄
        proc_ms = (now - t_capture) * 1000.0
        elapsed = now - self.start_time
        line1 = f"f#{self.frame_count}  fps {self.fps:4.1f}  proc {proc_ms:4.1f}ms"
        line2 = (f"{'CAM' if self.camera_ok else 'TEST'}  {w}x{h}  "
                 f"JPEG Q{self.jpeg_quality}  up {elapsed:5.1f}s")
        cv2.putText(frame, line1, (10, h - 26), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (10, h - 8), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 좌->우로 흐르는 막대 (프레임이 살아있음을 증명)
        x = int((now * 100) % w)
        cv2.line(frame, (x, bar_h), (x, h - 32), (0, 255, 0), 1)

    def _publish(self, pub, frame, now):
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        msg = CompressedImage()
        # 수신측이 end-to-end 지연을 계산할 수 있도록 타임스탬프 기록
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StreamTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap is not None:
            node.cap.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
