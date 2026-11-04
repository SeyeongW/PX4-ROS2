#!/usr/bin/env python3
"""SIYI RTSP 영상을 sensor_msgs/Image(bgr8)로 발행. 스트림 지연은 latency_sec로 stamp에 반영."""
import threading

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from .ros_utils import bgr_to_image


class SiyiCameraNode(Node):
    def __init__(self):
        super().__init__('siyi_camera_node')
        self.declare_parameter('rtsp_url', 'rtsp://192.168.144.25:8554/main.264')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('latency_sec', 0.0)   # 영상이 실제보다 늦게 도착하는 시간(실측해서 넣을 것)
        self.url = str(self.get_parameter('rtsp_url').value)
        self.latency = float(self.get_parameter('latency_sec').value)
        self.pub = self.create_publisher(Image, str(self.get_parameter('image_topic').value), 2)
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        cap = None
        while self._run and rclpy.ok():
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    self.get_logger().warn('RTSP 열기 실패, 재시도', throttle_duration_sec=3.0)
                    cv2.waitKey(1000)
                    continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                continue
            t = self.get_clock().now().nanoseconds - int(self.latency * 1e9)
            self.pub.publish(bgr_to_image(frame, rclpy.time.Time(nanoseconds=t).to_msg(), 'camera'))
        if cap is not None:
            cap.release()

    def stop(self):
        self._run = False


def main(args=None):
    rclpy.init(args=args)
    node = SiyiCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
