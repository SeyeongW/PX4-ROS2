#!/usr/bin/env python3
"""검출 + 짐벌 자세 + 줌 -> 정지 풍선의 NED 위치 추정.

발행: position_topic (기본 /balloon/position_ned, PointStamped, 첫 샘플부터 계속 발행),
      /balloon/estimate (PoseWithCovarianceStamped), /balloon/status (String, JSON: n, sigma_m 등)
서비스: /balloon/reset (Trigger)
NED 원점 = offset_ned 기준 (offset_ned = 카메라 회전중심의 NED 위치, 원점이 카메라면 0,0,0)
"""
import bisect
import json
import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2DArray

from .geometry import gimbal_to_azel, locate_balloon
from .static_estimator import StaticTargetEstimator


class BalloonLocatorNode(Node):
    def __init__(self):
        super().__init__('balloon_locator_node')
        d = self.declare_parameter
        d('image_width', 1920.0)
        d('image_height', 1080.0)
        d('hfov_deg_at_1x', 80.0)            # 임시값. 실제 캘리브레이션/데이터시트 값으로 교체할 것
        d('balloon_diameter_m', 0.30)        # 임시값. 실제 풍선 지름으로 교체할 것
        d('base_heading_deg', 0.0)           # 짐벌 yaw=0이 가리키는 방위각(북 기준 시계방향)
        d('yaw_sign', 1.0)
        d('yaw_zero_deg', 0.0)
        d('pitch_sign', 1.0)
        d('pitch_zero_deg', 0.0)
        d('offset_ned', [0.0, 0.0, 0.0])
        d('pixel_sigma_px', 1.0)
        d('diameter_rel_sigma', 0.05)
        d('angle_sigma_deg', 0.2)
        d('max_range_m', 300.0)
        d('min_diameter_px', 8.0)
        d('max_attitude_gap_sec', 0.3)       # 영상 시각 앞뒤 자세 샘플 간격이 이보다 크면 버림
        d('forget', 1.0)
        d('position_topic', '/balloon/position_ned')
        d('frame_id', 'local_ned')

        p = self.get_parameter
        self.est = StaticTargetEstimator(forget=p('forget').value)
        self.att = deque(maxlen=200)         # (t[s], yaw, pitch)
        self.zoom = 1.0

        self.pub_pt = self.create_publisher(PointStamped, str(p('position_topic').value), 10)
        self.pub_est = self.create_publisher(PoseWithCovarianceStamped, '/balloon/estimate', 10)
        self.pub_status = self.create_publisher(String, '/balloon/status', 10)
        self.create_subscription(Detection2DArray, '/balloon/detections', self._on_det, 10)
        self.create_subscription(Vector3Stamped, '/gimbal/attitude', self._on_att, 50)
        self.create_subscription(Float32, '/gimbal/zoom', lambda m: setattr(self, 'zoom', max(1.0, m.data)), 10)
        self.create_service(Trigger, '/balloon/reset', self._on_reset)
        self.create_timer(0.2, self._publish)
        self.skipped = {'no_att': 0, 'range': 0, 'small': 0}

    def _on_reset(self, _req, resp):
        self.est.reset()
        resp.success = True
        resp.message = 'reset'
        return resp

    def _on_att(self, msg: Vector3Stamped):
        self.att.append((Time.from_msg(msg.header.stamp).nanoseconds * 1e-9, msg.vector.x, msg.vector.y))

    def _attitude_at(self, t):
        """t 시각의 (yaw, pitch)를 앞뒤 샘플 선형 보간. yaw는 ±180 경계를 고려."""
        if len(self.att) < 2:
            return None
        ts = [a[0] for a in self.att]
        i = bisect.bisect_left(ts, t)
        if i == 0 or i >= len(ts):
            return None
        (t0, y0, p0), (t1, y1, p1) = self.att[i - 1], self.att[i]
        if t1 - t0 > self.get_parameter('max_attitude_gap_sec').value:
            return None
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        dy = (y1 - y0 + 180.0) % 360.0 - 180.0
        return y0 + a * dy, p0 + a * (p1 - p0)

    def _on_det(self, msg: Detection2DArray):
        if not msg.detections:
            return
        p = self.get_parameter
        det = msg.detections[0]
        diameter = det.bbox.size_x
        if diameter < p('min_diameter_px').value:
            self.skipped['small'] += 1
            return
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        ya = self._attitude_at(t)
        if ya is None:
            self.skipped['no_att'] += 1
            return
        az, el = gimbal_to_azel(ya[0], ya[1], p('base_heading_deg').value, p('yaw_sign').value,
                                p('yaw_zero_deg').value, p('pitch_sign').value, p('pitch_zero_deg').value)
        pos, cov, rng = locate_balloon(
            det.bbox.center.position.x, det.bbox.center.position.y, diameter,
            width_px=p('image_width').value, height_px=p('image_height').value,
            hfov_deg_at_1x=p('hfov_deg_at_1x').value, zoom=self.zoom, az_deg=az, el_deg=el,
            balloon_diameter_m=p('balloon_diameter_m').value, offset_ned=p('offset_ned').value,
            pixel_sigma_px=p('pixel_sigma_px').value, diameter_rel_sigma=p('diameter_rel_sigma').value,
            angle_sigma_deg=p('angle_sigma_deg').value)
        if rng > p('max_range_m').value:
            self.skipped['range'] += 1
            return
        self.est.update(pos, cov)

    def _publish(self):
        if self.est.n == 0:
            return
        p = self.get_parameter
        now = self.get_clock().now().to_msg()
        mean, cov = self.est.mean, self.est.cov
        pt = PointStamped()
        pt.header.stamp, pt.header.frame_id = now, p('frame_id').value
        pt.point.x, pt.point.y, pt.point.z = (float(v) for v in mean)
        self.pub_pt.publish(pt)

        pc = PoseWithCovarianceStamped()
        pc.header = pt.header
        pc.pose.pose.position = pt.point
        pc.pose.pose.orientation.w = 1.0
        full = np.zeros((6, 6))
        full[:3, :3] = cov
        pc.pose.covariance = full.flatten().tolist()
        self.pub_est.publish(pc)

        sigma = self.est.sigma
        self.pub_status.publish(String(data=json.dumps({
            'n': self.est.n, 'sigma_m': None if not math.isfinite(sigma) else round(sigma, 3),
            'rejects': self.est.rejects, 'skipped': self.skipped,
            'zoom': self.zoom, 'ned': [round(float(v), 3) for v in mean]})))


def main(args=None):
    rclpy.init(args=args)
    node = BalloonLocatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
