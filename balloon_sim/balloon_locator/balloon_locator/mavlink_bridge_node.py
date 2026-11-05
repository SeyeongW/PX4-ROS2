#!/usr/bin/env python3
"""지상 PC: 풍선 NED 추정 -> 위경도 + 홈 기준 상대고도 -> MAVLink COMMAND_INT -> 텔레메트리 모뎀.

구독: topic (기본 /balloon/position_ned, PointStamped), status_topic (기본 /balloon/status, JSON)
송신: COMMAND_INT  command=MAV_CMD_USER_1, frame=GLOBAL_RELATIVE_ALT_INT,
      x=lat*1e7, y=lon*1e7, z=드론 홈 지면 기준 상대고도[m], param1=n, param2=sigma_m
      수신 측은 onboard/offboard_follower.py
전제: balloon_locator_node의 offset_ned = [0,0,0] (NED 원점 = 카메라 회전중심).
게이트: /balloon/status의 n >= min_samples 이고 sigma_m <= max_sigma_m 일 때만 송신.
"""
import json
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped
from pymavlink import mavutil
from rclpy.node import Node
from std_msgs.msg import String

from .geo import ned_to_global

M = mavutil.mavlink


class MavlinkBridgeNode(Node):
    def __init__(self):
        super().__init__('mavlink_bridge_node')
        d = self.declare_parameter
        d('topic', '/balloon/position_ned')
        d('status_topic', '/balloon/status')
        d('expect_frame_id', 'local_ned')
        d('min_samples', 30)
        d('max_sigma_m', 2.0)
        d('max_age_sec', 1.0)                         # 위치/상태가 이보다 오래되면 송신 중지
        d('cam_lat_deg', float('nan'))                # 카메라 회전중심 위도 (현장 실측 필수)
        d('cam_lon_deg', float('nan'))                # 카메라 회전중심 경도 (현장 실측 필수)
        d('cam_height_above_home_m', float('nan'))    # 카메라 회전중심 높이 - 드론 이륙지점 지면 높이
        d('declination_deg', 0.0)                     # base_heading_deg를 자북으로 쟀으면 편각(동+, 한국 약 -8.5)
        d('device', '/dev/ttyUSB0')                   # PC 쪽 SiK 모뎀
        d('baud', 57600)
        d('rate_hz', 5.0)                             # balloon_locator_node 발행 주기와 같게
        d('target_system', 1)
        d('target_component', 191)                    # 온보드 컴퓨터 (offboard_follower)

        p = lambda n: self.get_parameter(n).value
        self.cam = (p('cam_lat_deg'), p('cam_lon_deg'), p('cam_height_above_home_m'))
        if not all(math.isfinite(v) for v in self.cam):
            raise RuntimeError('cam_lat_deg, cam_lon_deg, cam_height_above_home_m 를 설정해야 한다')
        self.decl = p('declination_deg')
        self.expect_frame = p('expect_frame_id')
        self.min_samples, self.max_sigma, self.max_age = p('min_samples'), p('max_sigma_m'), p('max_age_sec')
        self.target = (p('target_system'), p('target_component'))
        # Reject bad survey/transport settings before opening a real device.
        if (not -90.<self.cam[0]<90. or not -180.<=self.cam[1]<=180.
                or not math.isfinite(self.decl) or not self.expect_frame
                or any(not math.isfinite(v) or v<=0 for v in (self.max_sigma,self.max_age,p('rate_hz')))
                or self.min_samples<1 or p('baud')<=0 or any(not 1<=v<=255 for v in self.target)):
            raise RuntimeError('invalid_camera_reference_or_mavlink_bridge_configuration')

        self.mav = mavutil.mavlink_connection(p('device'), baud=p('baud'),
                                              source_system=250, source_component=191)
        self.point = self.status = None
        self.t_point = self.t_status = 0.0
        self.point_stamp = None
        self.create_subscription(PointStamped, p('topic'), self._on_point, 10)
        self.create_subscription(String, p('status_topic'), self._on_status, 10)
        self.create_timer(1.0 / p('rate_hz'), self._send)
        self.create_timer(1.0, self._heartbeat)
        self.get_logger().info(f"{p('topic')} -> {p('device')} @ {p('rate_hz')} Hz, camera {self.cam}")

    def _heartbeat(self):
        self.mav.mav.heartbeat_send(M.MAV_TYPE_GCS, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def _on_point(self, msg: PointStamped):
        if msg.header.frame_id != self.expect_frame:
            self.get_logger().warn(f'frame_id {msg.header.frame_id!r} != {self.expect_frame!r}, 무시',
                                   throttle_duration_sec=5.0)
            return
        point = (msg.point.x, msg.point.y, msg.point.z)
        stamp = (msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec)*1e-9
        current = self.get_clock().now().nanoseconds*1e-9
        if (not all(math.isfinite(v) for v in point) or not 0 < stamp <= current
                or current-stamp > self.max_age
                or self.point_stamp is not None and stamp <= self.point_stamp):
            return
        self.point, self.t_point, self.point_stamp = point, time.monotonic(), stamp

    def _on_status(self, msg: String):
        try:
            packet = json.loads(msg.data)
            if not isinstance(packet, dict):raise ValueError('status must be an object')
            self.status, self.t_status = packet, time.monotonic()
        except (ValueError,TypeError):
            self.status = None

    def _send(self):
        now = time.monotonic()
        if self.point is None or self.status is None:
            return
        if now - self.t_point > self.max_age or now - self.t_status > self.max_age:
            return
        current = self.get_clock().now().nanoseconds*1e-9
        stamp = self.status.get('measurement_stamp')
        wall_age = self.status.get('measurement_wall_age')
        if (self.status.get('valid') is not True or self.status.get('frame_id') != self.expect_frame
                or self.status.get('pose_source') != 'gimbal'
                or self.status.get('calibration_valid') is not True
                or self.status.get('station_offset_ned') != [0.,0.,0.]
                or type(stamp) not in (int,float) or not math.isfinite(stamp)
                or not 0 <= current-stamp <= self.max_age
                or self.point_stamp is None or abs(self.point_stamp-stamp)>1e-6
                or type(wall_age) not in (int,float) or not math.isfinite(wall_age)
                or not 0 <= wall_age+now-self.t_status <= self.max_age):
            return
        n, sigma = self.status.get('n', 0), self.status.get('sigma_m')
        if (type(n) not in (int,float) or not math.isfinite(n) or n < self.min_samples
                or type(sigma) not in (int,float) or not math.isfinite(sigma) or not 0 <= sigma <= self.max_sigma):
            self.get_logger().info(f'수렴 대기: n={n}, sigma_m={sigma}', throttle_duration_sec=2.0)
            return
        lat, lon, alt = ned_to_global(*self.point, *self.cam, self.decl)
        self.mav.mav.command_int_send(*self.target, M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, M.MAV_CMD_USER_1,
                                      0, 0, float(n), float(sigma), 0, 0,
                                      int(round(lat * 1e7)), int(round(lon * 1e7)), float(alt))


def main(args=None):
    rclpy.init(args=args)
    node = MavlinkBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mav.close()
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__ == '__main__':
    main()
