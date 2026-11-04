#!/usr/bin/env python3
"""검출된 풍선을 화면 중앙에 두도록 짐벌 속도 명령(/gimbal/cmd_rate)을 낸다. P 제어 + 데드밴드."""
import time

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2DArray


class GimbalTrackerNode(Node):
    def __init__(self):
        super().__init__('gimbal_tracker_node')
        self.declare_parameter('image_width', 1920.0)
        self.declare_parameter('image_height', 1080.0)
        self.declare_parameter('kp_yaw', 60.0)         # 정규화 오차(-1..1) 1.0일 때 속도 명령
        self.declare_parameter('kp_pitch', 60.0)
        self.declare_parameter('yaw_sign', 1.0)        # 풍선이 오른쪽일 때 + 속도가 오른쪽으로 도는지 실기 확인
        self.declare_parameter('pitch_sign', -1.0)     # 풍선이 아래일 때 + 속도가 아래로 도는지 실기 확인
        self.declare_parameter('deadband', 0.03)
        self.declare_parameter('max_rate', 60.0)
        self.declare_parameter('lost_timeout_sec', 0.5)
        self.declare_parameter('zoom_gain', 1.0)       # 줌이 클수록 속도를 줄임: rate / zoom^zoom_gain

        self.zoom = 1.0
        self.last_det = None
        self.last_time = 0.0
        self.pub = self.create_publisher(Vector3, '/gimbal/cmd_rate', 10)
        self.create_subscription(Detection2DArray, '/balloon/detections', self._on_det, 10)
        self.create_subscription(Float32, '/gimbal/zoom', lambda m: setattr(self, 'zoom', max(1.0, m.data)), 10)
        self.create_timer(0.05, self._tick)

    def _on_det(self, msg: Detection2DArray):
        if msg.detections:
            c = msg.detections[0].bbox.center.position
            self.last_det = (c.x, c.y)
            self.last_time = time.monotonic()

    def _tick(self):
        p = self.get_parameter
        if self.last_det is None or time.monotonic() - self.last_time > p('lost_timeout_sec').value:
            return   # 명령을 멈추면 짐벌 노드의 cmd_timeout으로 정지
        w, h = p('image_width').value, p('image_height').value
        ex = (self.last_det[0] - w / 2.0) / (w / 2.0)
        ey = (self.last_det[1] - h / 2.0) / (h / 2.0)
        db = p('deadband').value
        ex = 0.0 if abs(ex) < db else ex
        ey = 0.0 if abs(ey) < db else ey
        s = 1.0 / (self.zoom ** p('zoom_gain').value)
        lim = p('max_rate').value
        cmd = Vector3()
        cmd.x = max(-lim, min(lim, p('yaw_sign').value * p('kp_yaw').value * ex * s))
        cmd.y = max(-lim, min(lim, p('pitch_sign').value * p('kp_pitch').value * ey * s))
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GimbalTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
