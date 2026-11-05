#!/usr/bin/env python3
"""검출된 풍선을 화면 중앙에 두도록 짐벌 속도 명령(/gimbal/cmd_rate)을 낸다. P 제어 + 데드밴드."""
import math
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
        self.last_stamp = None
        self.last_clock = None
        self.active = False
        p = lambda name: self.get_parameter(name).value
        names = ('image_width', 'image_height', 'kp_yaw', 'kp_pitch', 'yaw_sign',
                 'pitch_sign', 'deadband', 'max_rate', 'lost_timeout_sec', 'zoom_gain')
        if (not all(math.isfinite(p(name)) for name in names)
                or min(p('image_width'), p('image_height'), p('lost_timeout_sec')) <= 0
                or min(p('kp_yaw'), p('kp_pitch'), p('zoom_gain')) < 0
                or p('yaw_sign') not in (-1., 1.) or p('pitch_sign') not in (-1., 1.)
                or not 0 <= p('deadband') < 1 or not 0 < p('max_rate') <= 100):
            raise ValueError('invalid_tracker_configuration')
        self.pub = self.create_publisher(Vector3, '/gimbal/cmd_rate', 10)
        self.create_subscription(Detection2DArray, '/balloon/detections', self._on_det, 10)
        self.create_subscription(Float32, '/gimbal/zoom', self._on_zoom, 10)
        self.create_timer(0.05, self._tick)

    def _now(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_clock is not None and now < self.last_clock:
            self.last_det = self.last_stamp = None
        self.last_clock = now
        return now

    def _on_zoom(self, msg):
        if math.isfinite(msg.data) and msg.data >= 1.:
            self.zoom = float(msg.data)

    def _on_det(self, msg: Detection2DArray):
        now = self._now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if (not 0 < stamp <= now + .1
                or now - stamp > self.get_parameter('lost_timeout_sec').value
                or self.last_stamp is not None and stamp <= self.last_stamp):
            return
        self.last_stamp = stamp
        self.last_det = None
        if msg.detections:
            c = msg.detections[0].bbox.center.position
            if (not math.isfinite(c.x) or not math.isfinite(c.y)
                    or not 0 <= c.x < self.get_parameter('image_width').value
                    or not 0 <= c.y < self.get_parameter('image_height').value):
                return
            self.last_det = (c.x, c.y)
            self.last_time = time.monotonic()

    def _tick(self):
        p = self.get_parameter
        now = self._now()
        if (self.last_det is None or self.last_stamp is None
                or not 0 <= now - self.last_stamp <= p('lost_timeout_sec').value
                or time.monotonic() - self.last_time > p('lost_timeout_sec').value):
            if self.active:
                self.pub.publish(Vector3())
                self.active = False
            return
        w, h = p('image_width').value, p('image_height').value
        ex = (self.last_det[0] - w / 2.0) / (w / 2.0)
        ey = (self.last_det[1] - h / 2.0) / (h / 2.0)
        db = p('deadband').value
        ex = 0.0 if abs(ex) < db else ex
        ey = 0.0 if abs(ey) < db else ey
        s = math.exp(-p('zoom_gain').value * math.log(self.zoom))
        lim = p('max_rate').value
        cmd = Vector3()
        cmd.x = max(-lim, min(lim, p('yaw_sign').value * p('kp_yaw').value * ex * s))
        cmd.y = max(-lim, min(lim, p('pitch_sign').value * p('kp_pitch').value * ey * s))
        self.pub.publish(cmd)
        self.active = True


def main(args=None):
    rclpy.init(args=args)
    node = GimbalTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__ == '__main__':
    main()
