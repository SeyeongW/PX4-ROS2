#!/usr/bin/env python3
"""영상 -> 풍선 검출 -> vision_msgs/Detection2DArray. bbox 크기는 지름(px)."""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .balloon_detector import detect_balloons, parse_hsv_ranges
from .ros_utils import image_to_bgr


class BalloonDetectorNode(Node):
    def __init__(self):
        super().__init__('balloon_detector_node')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('detections_topic', '/balloon/detections')
        # 기본값: 빨강 풍선 (색상이 0/180에서 갈라지므로 범위 2개). 풍선 색에 맞게 바꿀 것.
        self.declare_parameter('hsv_ranges', [0, 120, 70, 10, 255, 255, 170, 120, 70, 180, 255, 255])
        self.declare_parameter('min_diameter_px', 6.0)
        self.declare_parameter('max_diameter_px', 1000.0)
        self.declare_parameter('min_circularity', 0.6)
        self.declare_parameter('min_fill', 0.55)
        self.declare_parameter('process_scale', 0.5)   # 4K 같은 큰 영상은 줄여서 처리
        self.declare_parameter('max_detections', 3)

        self.ranges = parse_hsv_ranges(self.get_parameter('hsv_ranges').value)
        self.pub = self.create_publisher(Detection2DArray, str(self.get_parameter('detections_topic').value), 10)
        self.create_subscription(Image, str(self.get_parameter('image_topic').value), self._on_image, qos_profile_sensor_data)

    def _on_image(self, msg: Image):
        p = self.get_parameter
        try:
            frame = image_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        dets = detect_balloons(
            frame, self.ranges, min_diameter_px=p('min_diameter_px').value,
            max_diameter_px=p('max_diameter_px').value, min_circularity=p('min_circularity').value,
            min_fill=p('min_fill').value, scale=p('process_scale').value)
        out = Detection2DArray()
        out.header = msg.header
        for d in dets[:int(p('max_detections').value)]:
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = d.cx
            det.bbox.center.position.y = d.cy
            det.bbox.size_x = det.bbox.size_y = d.diameter_px
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'balloon'
            hyp.hypothesis.score = float(d.score)
            det.results.append(hyp)
            out.detections.append(det)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = BalloonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__ == '__main__':
    main()
