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
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from vision_msgs.msg import Detection2DArray

from .gimbal_configuration import TRACKER_DEFAULTS, tracker_configuration_errors


class GimbalTrackerNode(Node):
    def __init__(self):
        super().__init__('gimbal_tracker_node')
        for name, default in TRACKER_DEFAULTS.items():
            self.declare_parameter(name, default)
        errors = tracker_configuration_errors(
            {name: self.get_parameter(name).value for name in TRACKER_DEFAULTS})
        if errors:
            raise ValueError('; '.join(errors))
        self.add_on_set_parameters_callback(self._validate_parameters)

        self.zoom = 1.0
        self.last_det = None
        self.last_time = 0.0
        self.last_stamp = None
        self.last_frame_time = None
        self.last_att_stamp = None
        self.last_att_time = 0.0
        self.last_clock = None
        self.active = False
        self.locked = False
        self.det_streak = 0
        self.search_t0 = time.monotonic()
        self.current_att = None  # (yaw_deg, pitch_deg, roll_deg)

        self.pub = self.create_publisher(Vector3, '/gimbal/cmd_rate', 10)
        self.pub_locked = self.create_publisher(Bool, '/balloon/tracking_locked', 10)
        self.create_subscription(Detection2DArray, '/balloon/detections', self._on_det, 10)
        self.create_subscription(Float32, '/gimbal/zoom', self._on_zoom, 10)
        self.create_subscription(Vector3Stamped, '/gimbal/attitude', self._on_att, 20)
        self.create_timer(0.05, self._tick)

    def _validate_parameters(self, parameters):
        values = {name: self.get_parameter(name).value for name in TRACKER_DEFAULTS}
        values.update({parameter.name: parameter.value for parameter in parameters})
        errors = tracker_configuration_errors(values)
        return SetParametersResult(successful=not errors, reason='; '.join(errors))

    def _now(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_clock is not None and now < self.last_clock:
            self.last_det = self.last_stamp = None
            self.current_att = self.last_att_stamp = None
            self.last_frame_time = None
            self.last_att_time = 0.0
            self.det_streak = 0
            self.locked = False
        self.last_clock = now
        return now

    def _on_zoom(self, msg):
        if math.isfinite(msg.data) and msg.data >= 1.:
            self.zoom = float(msg.data)

    def _on_att(self, msg: Vector3Stamped):
        now = self._now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if (not 0 < stamp <= now + .1
                or now - stamp > self.get_parameter('attitude_timeout_sec').value
                or self.last_att_stamp is not None and stamp <= self.last_att_stamp):
            return
        self.last_att_stamp = stamp
        self.current_att = None
        if all(math.isfinite(v) for v in (msg.vector.x, msg.vector.y, msg.vector.z)):
            self.current_att = (float(msg.vector.x), float(msg.vector.y), float(msg.vector.z))
            self.last_att_time = time.monotonic()

    def _on_det(self, msg: Detection2DArray):
        now = self._now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if (not 0 < stamp <= now + .1
                or now - stamp > self.get_parameter('lost_timeout_sec').value
                or self.last_stamp is not None and stamp <= self.last_stamp):
            return
        self.last_stamp = stamp
        self.last_det = None
        self.last_frame_time = None
        if msg.detections:
            c = msg.detections[0].bbox.center.position
            if (not math.isfinite(c.x) or not math.isfinite(c.y)
                    or not 0 <= c.x < self.get_parameter('image_width').value
                    or not 0 <= c.y < self.get_parameter('image_height').value):
                self.det_streak = 0
                return
            self.last_det = (c.x, c.y)
            self.last_time = time.monotonic()
            self.det_streak += 1
            confirm = self.get_parameter('lock_confirm_frames').value
            if not self.get_parameter('enable_search').value or self.det_streak >= confirm:
                if not self.locked and self.get_parameter('enable_search').value:
                    self.get_logger().info(
                        f'BALLOON LOCKED! (streak={self.det_streak}, center=({c.x:.1f}, {c.y:.1f}))'
                    )
                self.locked = True
        else:
            self.det_streak = 0
        # Empty arrays are normal processed frames, unlike a failed/stalled stream.
        self.last_frame_time = time.monotonic()

    def _tick(self):
        p = self.get_parameter
        now = self._now()
        enable_search = bool(p('enable_search').value)
        lost_timeout = float(p('lost_timeout_sec').value)
        wall_now = time.monotonic()
        if enable_search:
            att_timeout = p('attitude_timeout_sec').value
            stream_fresh = (self.last_stamp is not None and self.last_frame_time is not None
                and 0 <= now - self.last_stamp <= lost_timeout
                and 0 <= wall_now - self.last_frame_time <= lost_timeout)
            attitude_fresh = (self.current_att is not None and self.last_att_stamp is not None
                and 0 <= now - self.last_att_stamp <= att_timeout
                and 0 <= wall_now - self.last_att_time <= att_timeout)
            if not stream_fresh or not attitude_fresh:
                self.pub.publish(Vector3())
                self.pub_locked.publish(Bool(data=False))
                self.active = self.locked = False
                self.det_streak = 0
                self.search_t0 = wall_now
                return
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
                self.locked = False
                self.pub_locked.publish(Bool(data=False))
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
