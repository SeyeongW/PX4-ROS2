#!/usr/bin/env python3
"""지상 PC: 풍선 6-DOF UKF NED 추정 -> 위경도 + 홈 기준 상대고도 + 추정속도 -> MAVLink COMMAND_INT -> SiK 텔레메트리 모뎀.

구독:
  - topic (기본 /balloon/position_ned, PointStamped)
  - status_topic (기본 /balloon/status, JSON)
  - launch_approval_topic (기본 /balloon/launch_approved, Bool)
송신:
  - COMMAND_INT command=MAV_CMD_USER_1, frame=GLOBAL_RELATIVE_ALT_INT,
    x=lat*1e7, y=lon*1e7, z=드론 홈 지면 기준 상대고도[m],
    param1=n (샘플 수), param2=sigma_m (불확실도[m]),
    param3=vn_mps (UKF 북쪽 속도[m/s]), param4=ve_mps (UKF 동쪽 속도[m/s])
  - 수신 측: onboard/tailsitter_onboard.py (또는 legacy interceptor.py / offboard_follower.py)
전제: balloon_locator_node의 offset_ned = [0,0,0] (NED 원점 = 카메라 회전중심).
게이트:
  1) /balloon/status의 n >= min_samples 이고 sigma_m <= max_sigma_m
  2) require_launch_approval=True 일 때 /balloon/launch_approved == True 승인 완료 후 송신
"""
import json
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped
from pymavlink import mavutil
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .geo import ned_to_global

M = mavutil.mavlink


class MavlinkBridgeNode(Node):
    def __init__(self):
        super().__init__('mavlink_bridge_node')
        d = self.declare_parameter
        d('topic', '/balloon/position_ned')
        d('status_topic', '/balloon/status')
        d('launch_approval_topic', '/balloon/launch_approved')
        d('require_launch_approval', True)
        d('approval_timeout_sec', 30.0)
        d('bridge_status_topic', '/balloon/bridge_status')
        d('fc_component', 1)
        d('fc_heartbeat_timeout_sec', 3.0)
        d('expect_frame_id', 'local_ned')
        d('min_samples', 30)
        d('max_sigma_m', 2.0)
        d('max_age_sec', 1.0)                         # 위치/상태가 이보다 오래되면 송신 중지
        d('cam_lat_deg', float('nan'))                # 카메라 회전중심 위도 (현장 실측 필수)
        d('cam_lon_deg', float('nan'))                # 카메라 회전중심 경도 (현장 실측 필수)
        d('cam_height_above_home_m', float('nan'))    # 카메라 회전중심 높이 - 드론 이륙지점 지면 높이
        d('declination_deg', 0.0)                     # base_heading_deg를 자북으로 쟀으면 편각(동+, 한국 약 -8.5)
        d('device', '/dev/ttyUSB0')                   # PC 쪽 SiK 모뎀 (또는 udpout:...)
        d('baud', 57600)
        d('rate_hz', 10.0)                            # 6-DOF UKF 발행 주기(10 Hz)와 동기화
        d('target_system', 1)
        d('target_component', 191)                    # 온보드 컴퓨터 (tailsitter_onboard)

        p = lambda n: self.get_parameter(n).value
        self.cam = (p('cam_lat_deg'), p('cam_lon_deg'), p('cam_height_above_home_m'))
        if not all(math.isfinite(v) for v in self.cam):
            raise RuntimeError('cam_lat_deg, cam_lon_deg, cam_height_above_home_m 를 설정해야 한다')
        self.decl = p('declination_deg')
        self.expect_frame = p('expect_frame_id')
        self.min_samples, self.max_sigma, self.max_age = p('min_samples'), p('max_sigma_m'), p('max_age_sec')
        self.require_approval = bool(p('require_launch_approval'))
        self.target = (p('target_system'), p('target_component'))
        self.approval_timeout = p('approval_timeout_sec')
        self.fc_component = p('fc_component')
        self.fc_timeout = p('fc_heartbeat_timeout_sec')
        # Reject bad survey/transport settings before opening a real device.
        if (not -90. < self.cam[0] < 90. or not -180. <= self.cam[1] <= 180.
                or not math.isfinite(self.decl) or not self.expect_frame
                or any(not math.isfinite(v) or v <= 0 for v in (self.max_sigma, self.max_age, p('rate_hz'), self.approval_timeout, self.fc_timeout))
                or self.min_samples < 1 or p('baud') <= 0 or any(not 1 <= v <= 255 for v in (*self.target, self.fc_component))):
            raise RuntimeError('invalid_camera_reference_or_mavlink_bridge_configuration')

        self.mav = mavutil.mavlink_connection(p('device'), baud=p('baud'),
                                              source_system=250, source_component=191)
        self.point = self.status = None
        self.t_point = self.t_status = 0.0
        self.point_stamp = None
        self.launch_approved = False
        self.approval_deadline = 0.0
        self.approval_requested = False
        self.approval_state = 'waiting' if self.require_approval else 'not_required'
        self.fc_heartbeat_time = None
        self.fc_armed = None
        self.pub_status = self.create_publisher(String, p('bridge_status_topic'), 10)
        self.create_subscription(PointStamped, p('topic'), self._on_point, 10)
        self.create_subscription(String, p('status_topic'), self._on_status, 10)
        self.create_subscription(Bool, p('launch_approval_topic'), self._on_approval, 10)
        self.create_timer(1.0 / p('rate_hz'), self._send)
        self.create_timer(1.0, self._heartbeat)
        self.get_logger().info(
            f"{p('topic')} -> {p('device')} @ {p('rate_hz')} Hz, camera {self.cam}, "
            f"require_launch_approval={self.require_approval}"
        )

    def _heartbeat(self):
        try:
            self.mav.mav.heartbeat_send(M.MAV_TYPE_GCS, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        except (OSError, ValueError) as exc:
            self._revoke('transport_error')
            self.get_logger().error(f'MAVLink heartbeat send failed: {exc}', throttle_duration_sec=2.)

    def _revoke(self, reason):
        self.launch_approved = False
        self.approval_deadline = 0.0
        self.approval_state = reason

    def _on_approval(self, msg: Bool):
        if not msg.data:
            self.approval_requested = False
            self._revoke('cancelled')
            return
        # A held/repeated True cannot refresh the lease or re-arm after expiry.
        if self.approval_requested:
            return
        self.approval_requested = True
        now = time.monotonic()
        if not self._coordinates_valid(now):
            self._revoke('rejected_coordinates')
            return
        self.launch_approved = True
        self.approval_deadline = now + self.approval_timeout
        self.approval_state = 'approved'
        self.get_logger().info('Coordinate broadcast approved; FC arming/flight is not confirmed.')

    def _on_point(self, msg: PointStamped):
        if msg.header.frame_id != self.expect_frame:
            self.get_logger().warn(f'frame_id {msg.header.frame_id!r} != {self.expect_frame!r}, 무시',
                                   throttle_duration_sec=5.0)
            return
        point = (msg.point.x, msg.point.y, msg.point.z)
        stamp = (msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec) * 1e-9
        current = self.get_clock().now().nanoseconds * 1e-9
        if (not all(math.isfinite(v) for v in point) or not 0 < stamp <= current
                or current - stamp > self.max_age
                or self.point_stamp is not None and stamp <= self.point_stamp):
            return
        self.point, self.t_point, self.point_stamp = point, time.monotonic(), stamp

    def _on_status(self, msg: String):
        try:
            packet = json.loads(msg.data)
            if not isinstance(packet, dict):
                raise ValueError('status must be an object')
            self.status, self.t_status = packet, time.monotonic()
        except (ValueError, TypeError):
            self.status = None

    def _coordinates_valid(self, now):
        return self._coordinate_state(now) == 'valid'

    def _coordinate_state(self, now):
        if self.point is None or self.status is None:
            return 'invalid'
        if not 0 <= now - self.t_point <= self.max_age or not 0 <= now - self.t_status <= self.max_age:
            return 'invalid'
        current = self.get_clock().now().nanoseconds * 1e-9
        stamp = self.status.get('measurement_stamp')
        wall_age = self.status.get('measurement_wall_age')
        if (self.status.get('valid') is not True or self.status.get('frame_id') != self.expect_frame
                or self.status.get('pose_source') != 'gimbal'
                or self.status.get('calibration_valid') is not True
                or self.status.get('station_offset_ned') != [0., 0., 0.]
                or type(stamp) not in (int, float) or not math.isfinite(stamp)
                or not 0 <= current - stamp <= self.max_age
                or self.point_stamp is None or not 0 <= current - self.point_stamp <= self.max_age
                or type(wall_age) not in (int, float) or not math.isfinite(wall_age)
                or not 0 <= wall_age + now - self.t_status <= self.max_age):
            return 'invalid'
        n, sigma = self.status.get('n', 0), self.status.get('sigma_m')
        if not (type(n) in (int, float) and math.isfinite(n) and n >= self.min_samples
                and type(sigma) in (int, float) and math.isfinite(sigma) and 0 <= sigma <= self.max_sigma):
            return 'invalid'
        # Point/status arrive on separate ROS subscriptions. A fresh unmatched
        # pair pauses only this send tick; it must not consume the user's lease.
        return 'valid' if abs(self.point_stamp - stamp) <= 1e-6 else 'waiting_pair'

    def _poll_fc(self, now):
        try:
            # Bound each poll even when unrelated telemetry arrives continuously.
            for _ in range(8):
                msg = self.mav.recv_match(type='HEARTBEAT', blocking=False, timeout=.002)
                if msg is None:
                    break
                if (msg.get_srcSystem() == self.target[0] and msg.get_srcComponent() == self.fc_component
                        and msg.autopilot != M.MAV_AUTOPILOT_INVALID and msg.type != M.MAV_TYPE_GCS):
                    self.fc_heartbeat_time = now
                    self.fc_armed = bool(msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
        except (OSError, ValueError):
            self.fc_heartbeat_time = None
            self._revoke('transport_error')

    def _send(self):
        now = time.monotonic()
        self._poll_fc(now)
        coordinate_state = self._coordinate_state(now)
        valid = coordinate_state == 'valid'
        if self.launch_approved and now >= self.approval_deadline:
            self._revoke('expired')
        elif self.launch_approved and coordinate_state == 'invalid':
            self._revoke('coordinates_invalid')
        broadcasting = False
        if valid and (self.launch_approved or (not self.require_approval and self.approval_state == 'not_required')):
            vel_ned = self.status.get('velocity_ned')
            vn, ve = 0.0, 0.0
            if (isinstance(vel_ned, (list, tuple)) and len(vel_ned) >= 2
                    and all(type(v) in (int, float) and math.isfinite(v) for v in vel_ned[:2])):
                vn, ve = map(float, vel_ned[:2])
            lat, lon, alt = ned_to_global(*self.point, *self.cam, self.decl)
            try:
                self.mav.mav.command_int_send(*self.target, M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, M.MAV_CMD_USER_1,
                    0, 0, float(self.status['n']), float(self.status['sigma_m']), vn, ve,
                    int(round(lat * 1e7)), int(round(lon * 1e7)), float(alt))
                broadcasting = True  # local send only, not onboard receipt or flight confirmation
            except (OSError, ValueError) as exc:
                self._revoke('transport_error')
                self.get_logger().error(f'Coordinate send failed: {exc}', throttle_duration_sec=2.)
        fc_age = None if self.fc_heartbeat_time is None else now - self.fc_heartbeat_time
        fc_fresh = fc_age is not None and 0 <= fc_age <= self.fc_timeout
        self.pub_status.publish(String(data=json.dumps(dict(
            stamp=self.get_clock().now().nanoseconds * 1e-9,
            approval_state=self.approval_state, approval_required=self.require_approval,
            approval_remaining_sec=max(0., self.approval_deadline - now),
            coordinate_valid=valid, coordinate_state=coordinate_state, broadcasting=broadcasting,
            fc=dict(system=self.target[0], component=self.fc_component, heartbeat_fresh=fc_fresh,
                    heartbeat_age_sec=fc_age, armed=self.fc_armed if fc_fresh else None)), allow_nan=False)))


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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
