#!/usr/bin/env python3
"""온보드 카메라 + 디텍션 오버레이 뷰어. OpenCV 창으로 표시."""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from .ros_utils import image_to_bgr


class DetectionViewerNode(Node):
    def __init__(self):
        super().__init__('detection_viewer_node')
        self.declare_parameter('image_topic', '/jol/onboard_camera/image_raw')
        self.declare_parameter('detection_topic', '/balloon/onboard/detections')

        self.latest_detections = []
        self.frame_count = 0

        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            Detection2DArray,
            self.get_parameter('detection_topic').value,
            self._on_detection, 10)

    def _on_detection(self, msg):
        self.latest_detections = msg.detections

    def _on_image(self, msg):
        try:
            frame = image_to_bgr(msg)
        except ValueError:
            return

        for det in self.latest_detections:
            cx = int(det.bbox.center.position.x)
            cy = int(det.bbox.center.position.y)
            r = int(det.bbox.size_x / 2)
            score = det.results[0].hypothesis.score if det.results else 0
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (cx - r, cy - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.frame_count += 1
        n = len(self.latest_detections)
        cv2.putText(frame, f'det={n}  frame={self.frame_count}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('onboard', frame)
        cv2.waitKey(1)


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
