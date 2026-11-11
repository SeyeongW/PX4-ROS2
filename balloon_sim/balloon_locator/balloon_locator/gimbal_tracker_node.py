#!/usr/bin/env python3
"""SIYI A8 mini 짐벌 탐색(Sky Search) 및 비주얼 서보잉 락온(Visual Servoing Lock-On) 노드.

- enable_search=True 일 때:
  1) SEARCHING: 풍선 미검출 시 지정된 방위각/고각 범위 내에서 부드러운 탐색 스윕(/gimbal/cmd_rate) 수행
  2) LOCKED: 연속 lock_confirm_frames 프레임 이상 풍선 검증 시 즉시 락온하여 화면 중앙 정렬(P 제어 + 데드밴드)
- enable_search=False (기본값, 단위 테스트 호환):
  검출된 풍선이 있을 때만 P 제어 속도 명령을 내고, 소실 시 즉시 정지(0, 0) 명령 발행.
"""
import math
import time

import rclpy
from geometry_msgs.msg import Vector3, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
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

        # Sky Search -> Lock-On 파라미터 (실기체 SIYI A8 mini 탐색용)
        self.declare_parameter('enable_search', False)
        self.declare_parameter('lock_confirm_frames', 3)
        self.declare_parameter('search_resume_timeout_sec', 2.0)
        self.declare_parameter('search_yaw_rate', 18.0)
        self.declare_parameter('search_pitch_rate', 8.0)
        self.declare_parameter('search_period_sec', 11.0)
        self.declare_parameter('search_yaw_limit_deg', 40.0)
        self.declare_parameter('search_pitch_min_deg', 5.0)
        self.declare_parameter('search_pitch_max_deg', 45.0)

        self.zoom = 1.0
        self.last_det = None
        self.last_time = 0.0
        self.last_stamp = None
        self.last_clock = None
        self.active = False
        self.locked = False
        self.det_streak = 0
        self.search_t0 = time.monotonic()
        self.yaw_dir = 1.0
        self.pitch_dir = 1.0
        self.current_att = None  # (yaw_deg, pitch_deg, roll_deg)

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
        self.pub_locked = self.create_publisher(Bool, '/balloon/tracking_locked', 10)
        self.create_subscription(Detection2DArray, '/balloon/detections', self._on_det, 10)
        self.create_subscription(Float32, '/gimbal/zoom', self._on_zoom, 10)
        self.create_subscription(Vector3Stamped, '/gimbal/attitude', self._on_att, 20)
        self.create_timer(0.05, self._tick)

    def _now(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_clock is not None and now < self.last_clock:
            self.last_det = self.last_stamp = None
            self.det_streak = 0
            self.locked = False
        self.last_clock = now
        return now

    def _on_zoom(self, msg):
        if math.isfinite(msg.data) and msg.data >= 1.:
            self.zoom = float(msg.data)

    def _on_att(self, msg: Vector3Stamped):
        if all(math.isfinite(v) for v in (msg.vector.x, msg.vector.y, msg.vector.z)):
            self.current_att = (float(msg.vector.x), float(msg.vector.y), float(msg.vector.z))

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
            self.det_streak += 1
            confirm = max(1, int(self.get_parameter('lock_confirm_frames').value))
            if not self.get_parameter('enable_search').value or self.det_streak >= confirm:
                if not self.locked and self.get_parameter('enable_search').value:
                    self.get_logger().info(
                        f'BALLOON LOCKED! (streak={self.det_streak}, center=({c.x:.1f}, {c.y:.1f}))'
                    )
                self.locked = True
        else:
            self.det_streak = 0

    def _tick(self):
        p = self.get_parameter
        now = self._now()
        enable_search = bool(p('enable_search').value)
        lost_timeout = float(p('lost_timeout_sec').value)
        det_valid = (
            self.last_det is not None
            and self.last_stamp is not None
            and 0 <= now - self.last_stamp <= lost_timeout
            and time.monotonic() - self.last_time <= lost_timeout
        )

        if not det_valid:
            if enable_search:
                resume_sec = float(p('search_resume_timeout_sec').value)
                if self.locked and (time.monotonic() - self.last_time) <= resume_sec:
                    # 일시적 프레임 드랍 시 잠시 정지 대기
                    if self.active:
                        self.pub.publish(Vector3())
                        self.active = False
                    self.pub_locked.publish(Bool(data=True))
                    return
                if self.locked:
                    self.get_logger().info('Balloon lost -> Resuming SKY SEARCH...')
                    self.locked = False
                    self.det_streak = 0
                    self.search_t0 = time.monotonic()
                # SKY SEARCH 스윕 속도 명령 생성
                t_s = time.monotonic() - self.search_t0
                period = max(2.0, float(p('search_period_sec').value))
                omega = 2.0 * math.pi / period
                yaw_rate = float(p('search_yaw_rate').value) * math.cos(omega * t_s)
                pitch_rate = float(p('search_pitch_rate').value) * math.sin(2.0 * omega * t_s)
                if self.current_att is not None:
                    cur_yaw, cur_pitch, _ = self.current_att
                    yaw_lim = float(p('search_yaw_limit_deg').value)
                    p_min = float(p('search_pitch_min_deg').value)
                    p_max = float(p('search_pitch_max_deg').value)
                    if abs(cur_yaw) > yaw_lim and (cur_yaw * yaw_rate * p('yaw_sign').value) > 0:
                        yaw_rate = -yaw_rate
                    if cur_pitch < p_min and (pitch_rate * p('pitch_sign').value) < 0:
                        pitch_rate = abs(pitch_rate) * p('pitch_sign').value
                    elif cur_pitch > p_max and (pitch_rate * p('pitch_sign').value) > 0:
                        pitch_rate = -abs(pitch_rate) * p('pitch_sign').value
                lim = float(p('max_rate').value)
                cmd = Vector3()
                cmd.x = max(-lim, min(lim, yaw_rate))
                cmd.y = max(-lim, min(lim, pitch_rate))
                self.pub.publish(cmd)
                self.pub_locked.publish(Bool(data=False))
                self.active = True
                return
            else:
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
        if enable_search:
            self.pub_locked.publish(Bool(data=bool(self.locked)))
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
