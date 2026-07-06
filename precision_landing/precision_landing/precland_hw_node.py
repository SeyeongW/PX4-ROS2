#!/usr/bin/env python3
"""ArUco 정밀착륙 제어기 — 실기체 하드웨어용 (ArduPilot / MAVROS, GUIDED).

SITL 의 precision_landing_node 를 실기체에 맞게 단순화·안전화한 버전입니다.

SITL 대비 달라진 점
  1. 마커 측정: 정규화 오프셋 + HFOV 근사 대신, aruco_pose_node 가 캘리브레이션으로
     푼 **실측 3D 위치(m)** (/perception/marker_pose, 카메라 프레임)를 그대로 사용.
     마커 윗면까지의 높이도 카메라 pose(tvec.z)에서 직접 얻어 FC 고도 오차와 무관.
  2. 이동 마커(cue/APPROACH/moving_marker)와 자동이륙을 기본 제거 — 정지 마커를 보고
     내려앉는 단순·안전 동작. (auto_takeoff=false: 조종사가 이륙·GUIDED 전환 후 인계)
  3. 터치다운: 공중 강제 disarm 대신 **ArduPilot LAND 모드로 인계** — 저고도까지
     정렬·하강한 뒤 LAND 로 전환하면 FC 가 접지 감지·자동 disarm 을 처리(평지 안전).
  4. 조종사 오버라이드 = 즉시 중단: GUIDED 가 아니거나 disarm 되면 셋포인트 송출을
     멈추고 IDLE 로 복귀. 조종사가 모드 스위치로 언제든 회수 가능.

상태 기계
  (TAKEOFF) : auto_takeoff=true 일 때만. GUIDED→arm→flight_alt 까지 이륙.
  IDLE      : armed + GUIDED + 마커 감지 될 때까지 제자리(속도 0) 대기.
  ALIGN     : flight_alt 호버하며 마커 중심으로 수평 정렬(속도 서보).
  DESCEND   : 깔때기(funnel) 허용오차를 따라 정렬하며 동시에 하강.
  LAND      : 저고도·중심 정렬 상태에서 LAND 모드로 전환, FC 에 착륙 인계.
  DONE      : disarm 확인 후 노드 종료.

토픽 계약
  in  /perception/marker_pose    geometry_msgs/PoseStamped (카메라 프레임 tvec)
  in  /perception/aruco_detected std_msgs/Bool
  in  /mavros/state              mavros_msgs/State
  in  /mavros/local_position/pose geometry_msgs/PoseStamped (BEST_EFFORT)
  out /mavros/setpoint_raw/local mavros_msgs/PositionTarget (속도 + yaw)
  out /precision_landing/debug   std_msgs/String
"""

import math
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode


class Stage(Enum):
    TAKEOFF = 0
    IDLE = 1
    ALIGN = 2
    DESCEND = 3
    LAND = 4
    DONE = 5


DT = 0.05  # 50 ms (20 Hz) 제어 주기


class PreclandHwNode(Node):
    def __init__(self):
        super().__init__('precland_hw_node')

        # --- 파라미터 -------------------------------------------------------
        self.flight_alt = self.declare_parameter('flight_alt', 4.0).value
        # 하강 깔때기: 허용 수평오차가 고도에 비례해 넓어짐(높이 관대, 지면 근처 land_
        # align_radius 로 좁아짐). funnel_radius(h) = max(land_align_radius, descend_cone·h).
        self.descend_cone = self.declare_parameter('descend_cone', 0.35).value
        self.descend_min_scale = self.declare_parameter('descend_min_scale', 0.3).value
        self.descend_rate = self.declare_parameter('descend_rate', 0.25).value  # m/s 하강
        # 착륙 확정 수평오차 게이트(m). 깔때기 바닥 반경.
        self.land_align_radius = self.declare_parameter('land_align_radius', 0.15).value
        # 마커 윗면 위 이 높이 아래로 내려오고 + 중심 정렬되면 LAND 모드로 인계.
        # 저고도에선 마커가 화면을 넘쳐 검출이 끊기므로, 여기서 FC 에 넘긴다.
        self.land_switch_alt = self.declare_parameter('land_switch_alt', 1.0).value
        # 속도 서보: v = v_gain·error, vel_max 로 clamp. 1차 응답(오버슈트 없는 수렴).
        # 실기체는 보수적으로 느리게(SITL 보다 낮은 상한).
        self.vel_gain = self.declare_parameter('vel_gain', 0.6).value       # 1/s
        self.vel_max = self.declare_parameter('vel_max', 0.8).value         # m/s
        # 정지 마커 등속 칼만 필터(평활화 + 프레임 누락 시 coast).
        self.kf_accel_std = self.declare_parameter('kf_accel_std', 0.1).value
        self.kf_meas_std = self.declare_parameter('kf_meas_std', 0.05).value  # pose 는 정밀
        self.coast_ticks = self.declare_parameter('coast_ticks', 20).value   # ~1.0 s
        # 카메라 마운트 의존 이미지→기체 매핑. 정렬 중 엉뚱한 방향으로 가면 튜닝:
        #   lat_swap      : 카메라 x/y 축 교환(마커를 못 잡고 빙글빙글)
        #   lat_sign_fwd  : 전방(+X) 보정 부호 ±1
        #   lat_sign_left : 좌측(+Y) 보정 부호 ±1
        self.lat_swap = self.declare_parameter('lat_swap', False).value
        self.lat_sign_fwd = self.declare_parameter('lat_sign_fwd', 1.0).value
        self.lat_sign_left = self.declare_parameter('lat_sign_left', 1.0).value
        # auto_takeoff=false(기본): 조종사가 이륙·GUIDED 전환 후 IDLE 에서 인계.
        # true 면 노드가 스스로 GUIDED→arm→이륙(SITL/무인 테스트용, 실비행 주의).
        self.auto_takeoff = self.declare_parameter('auto_takeoff', False).value
        # 안전: GUIDED 가 아니면 셋포인트를 절대 송출하지 않음. 조종사가 모드 스위치로
        # 언제든 회수(RC 오버라이드) 가능. false 로 두면 이 게이트를 끔(권장 안 함).
        self.require_guided = self.declare_parameter('require_guided', True).value

        self.stage = Stage.TAKEOFF if self.auto_takeoff else Stage.IDLE

        # --- 상태 -----------------------------------------------------------
        self.mav_state = State()
        self.pos = [0.0, 0.0, 0.0]     # 로컬 ENU 위치
        self.yaw = 0.0                 # 현재 헤딩(ENU, rad)
        self.hold_yaw = 0.0            # 정렬 중 유지할 헤딩
        self.tvec = None               # 최신 마커 위치(카메라 프레임 m): (right, down, dist)
        self.marker_h = self.flight_alt  # 마커 윗면 위 높이(tvec.z 기반, coast 시 유지)
        self.fresh = 0                 # 마커 신선도 카운트다운
        self.req_ticks = 0
        self.dbg_tick = 0
        self.land_sent = 0
        # 칼만 상태 x=[E, N, vE, vN]
        self.kf_x = np.zeros(4)
        self.kf_P = np.eye(4)
        self.kf_init = False
        self.kf_miss = 0
        self.cmd_vel = [0.0, 0.0, 0.0]  # 명령 ENU 속도 (E, N, Up)

        # --- Pub/Sub --------------------------------------------------------
        self.raw_pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.debug_pub = self.create_publisher(String, '/precision_landing/debug', 10)

        self.create_subscription(State, '/mavros/state', self._state_cb, 10)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._pose_cb,
            qos_profile_sensor_data)
        self.create_subscription(
            PoseStamped, '/perception/marker_pose', self._marker_cb, 10)
        self.create_subscription(Bool, '/perception/aruco_detected', self._detected_cb, 10)

        # --- 서비스 클라이언트 ----------------------------------------------
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_cli = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # 조종사 안내 로그용 엣지 추적
        self._prev_connected = False
        self._prev_armed = False
        self._prev_guided = False
        self._prev_marker = False
        self._guide_tick = 0

        self.create_timer(DT, self.tick)
        self._log('============ ArUco 정밀착륙 (실기체) ============')
        if self.auto_takeoff:
            self._log(f'자동이륙 ON — 노드가 GUIDED/시동/이륙({self.flight_alt:.1f} m)까지 스스로 합니다.')
        else:
            self._log('진행 순서:  ① 시동(ARM)  →  ② 이륙 후 GUIDED 전환  →  '
                      '③ 마커 위로 이동  →  자동 정렬·하강·착륙')
            self._log('* 언제든 모드 스위치로 회수 가능 (GUIDED 벗어나면 즉시 제어 중단).')
        self._log('FC(MAVROS) 연결을 기다리는 중')

    # -----------------------------------------------------------------------
    # 콜백
    def _state_cb(self, msg):
        self.mav_state = msg

    def _pose_cb(self, msg):
        self.pos[0] = msg.pose.position.x
        self.pos[1] = msg.pose.position.y
        self.pos[2] = msg.pose.position.z
        q = msg.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _marker_cb(self, msg):
        # 카메라 광학 프레임 tvec: x=오른쪽, y=아래, z=마커까지 거리(≈높이).
        self.tvec = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def _detected_cb(self, msg):
        if msg.data:
            self.fresh = 10   # 10 × 50 ms = 500 ms 신선도 창

    # -----------------------------------------------------------------------
    def tick(self):
        if self.fresh > 0:
            self.fresh -= 1
        marker_ok = self.fresh > 0 and self.tvec is not None

        if not self.mav_state.connected:
            self._prev_connected = False
            return
        if not self._prev_connected:
            self._prev_connected = True
            self._log('FC 연결됨 (MAVROS).')
            if not self.auto_takeoff:
                self._log('다음 할 일: 시동(ARM)을 걸어주세요.')

        armed = self.mav_state.armed
        guided = self.mav_state.mode == 'GUIDED'

        # 안전 게이트: 비행 중 조종사가 GUIDED 를 벗어나거나 disarm 하면 즉시 중단.
        if self.stage in (Stage.ALIGN, Stage.DESCEND):
            if not armed:
                self._log('시동 꺼짐 — 정밀착륙 중단, 대기 상태로.  -> IDLE')
                self.stage = Stage.IDLE
            elif self.require_guided and not guided:
                self._log('GUIDED 이탈(조종사 회수) — 제어 중단, 대기 상태로.  -> IDLE')
                self.stage = Stage.IDLE

        if self.stage == Stage.TAKEOFF:
            # ArduCopter GUIDED 는 착륙 상태에서 셋포인트만으론 안 뜸 → 명시적 takeoff.
            self.req_ticks += 1
            send = (self.req_ticks % 40 == 1)      # ~2 s 마다
            if not guided:
                if send:
                    self._log('TAKEOFF: set GUIDED'); self._set_mode('GUIDED')
            elif not armed:
                if send:
                    self._log('TAKEOFF: arming'); self._arm(True)
            elif self.pos[2] < self.flight_alt - 0.5:
                if send:
                    self._log(f'TAKEOFF: cmd takeoff -> {self.flight_alt:.1f} m')
                    self._takeoff(self.flight_alt)
            else:
                self._log('-> IDLE (reached flight_alt)')
                self.stage = Stage.IDLE
            return

        elif self.stage == Stage.IDLE:
            self.cmd_vel = [0.0, 0.0, 0.0]
            self.hold_yaw = self.yaw      # 정렬 시작 전까지 현재 헤딩 추종
            self._idle_guide(armed, guided, marker_ok)
            if armed and guided and marker_ok:
                self._log('마커 감지 — 정밀착륙을 시작합니다!  → 정렬(ALIGN)')
                self._kf_reset()
                self.stage = Stage.ALIGN

        elif self.stage == Stage.ALIGN:
            err = self._track(marker_ok)
            self.cmd_vel[2] = 0.0                 # 고도 유지
            if self.kf_miss > self.coast_ticks:
                self._log('마커를 놓쳤습니다 — 대기 상태로 복귀.  → IDLE')
                self.stage = Stage.IDLE
            elif err is not None and marker_ok and err < self._funnel_radius():
                self._log('정렬 완료 — 하강을 시작합니다.  → 하강(DESCEND)')
                self.stage = Stage.DESCEND

        elif self.stage == Stage.DESCEND:
            err = self._track(marker_ok)
            h = self.marker_h
            lateral = err if err is not None else 999.0
            if h < self.land_switch_alt and lateral < self.land_align_radius:
                # 저고도·중심 정렬: 여기서부터는 마커가 화면을 넘쳐 검출이 불안정하므로
                # FC 의 LAND 모드로 인계 → 접지 감지·자동 disarm.
                self._log(f'중심 정렬({lateral:.2f} m) & 저고도(h={h:.2f} m) — '
                          f'FC에 착륙을 인계합니다.  → 착륙(LAND)')
                self.stage = Stage.LAND
                return
            if self.kf_miss > self.coast_ticks:
                # 비전 상실: 하강 멈추고 재정렬(정지 마커라 추종할 cue 없음).
                self._log('하강 중 마커 상실 — 고도 유지하며 재정렬.  → 정렬(ALIGN)')
                self.cmd_vel[2] = 0.0
                self.stage = Stage.ALIGN
            elif self.kf_init:
                funnel = self._funnel_radius()
                if lateral <= funnel:
                    # 깔때기 안: 항상 야금야금 하강(중심에 가까울수록 빠르게).
                    scale = self._clamp(1.0 - lateral / funnel,
                                        self.descend_min_scale, 1.0)
                    self.cmd_vel[2] = -self.descend_rate * scale
                else:
                    self.cmd_vel[2] = 0.0         # 이 고도엔 너무 벗어남 → 재정렬 먼저
            else:
                self.cmd_vel[2] = 0.0

        elif self.stage == Stage.LAND:
            # LAND 모드로 전환하고 셋포인트 송출을 멈춘다(오프보드 간섭 방지).
            # /mavros/state 로 확인될 때까지 ~1 s 마다 재전송. disarm 되면 착륙 완료.
            self.land_sent += 1
            if not armed:
                self._log('접지·시동 꺼짐 확인 — 착륙 완료! 수고하셨습니다. (노드 종료)')
                self.stage = Stage.DONE
                rclpy.shutdown()
                return
            if self.mav_state.mode != 'LAND' and self.land_sent % 20 == 1:
                self._log('LAND 모드로 전환 요청 — 접지하면 자동으로 시동이 꺼집니다.')
                self._set_mode('LAND')
            return                                 # 셋포인트 송출 안 함

        elif self.stage == Stage.DONE:
            return

        self._publish_velocity()

    # -----------------------------------------------------------------------
    # 조종사용 단계별 안내(IDLE 대기 중). 상태 변화(엣지)마다 한 번씩 + 3초마다 지금
    # 필요한 다음 동작을 다시 알려준다. 로그만 보고도 순서를 따라올 수 있게.
    def _idle_guide(self, armed, guided, marker_ok):
        if armed and not self._prev_armed:
            self._log('시동 걸림 (ARMED).')
            self._log('다음 할 일: 이륙한 뒤 GUIDED 모드로 전환하세요.')
        elif not armed and self._prev_armed:
            self._log('시동 꺼짐 (DISARMED). 다시 시동을 걸어주세요.')
        if guided and not self._prev_guided:
            self._log('GUIDED 모드 확인.')
            self._log('다음 할 일: 드론을 마커 위로 이동하세요 — 하방 카메라가 '
                      '마커를 잡으면 자동으로 정렬·하강·착륙합니다.')
        elif not guided and self._prev_guided:
            self._log('GUIDED 해제(수동 조종). 다시 GUIDED로 전환하면 이어집니다.')
        if marker_ok and not self._prev_marker:
            self._log('마커 보임.')
        elif not marker_ok and self._prev_marker:
            self._log('마커를 놓쳤습니다 — 카메라 시야에 마커가 들어오게 이동하세요.')
        self._prev_armed, self._prev_guided, self._prev_marker = armed, guided, marker_ok

        self._guide_tick += 1
        if self._guide_tick % 60 == 0:          # 3초마다 현재 필요한 동작 리마인드
            if not armed:
                self._log('대기 중 — 시동(ARM)을 걸어주세요.')
            elif not guided:
                self._log('대기 중 — 이륙 후 GUIDED 모드로 전환하세요.')
            elif not marker_ok:
                self._log('대기 중 — 드론을 마커 위로 (하방 카메라가 마커를 봐야 시작).')

    # 마커를 등속 칼만 필터로 추종하고 추정 위치로 속도 서보. 드론↔마커 수평거리(m) 반환.
    def _track(self, marker_ok):
        self._kf_predict()
        if marker_ok:
            zE, zN = self._measure_marker_world()
            self._kf_update(zE, zN)
            self.marker_h = max(self.tvec[2], 0.05)   # 카메라가 측정한 마커 위 높이
            self.kf_miss = 0
        else:
            self.kf_miss += 1

        if not self.kf_init:
            self.cmd_vel[0] = self.cmd_vel[1] = 0.0
            return None

        eE = self.kf_x[0] - self.pos[0]
        eN = self.kf_x[1] - self.pos[1]
        # 정지 마커: 속도 피드포워드(kf_x[2:4])는 ≈0 이라 사실상 순수 비례 서보.
        self._servo_to(self.kf_x[0], self.kf_x[1], self.kf_x[2], self.kf_x[3])

        self.dbg_tick += 1
        if self.dbg_tick % 20 == 0:
            self._log(f'h={self.marker_h:.2f} err=({eE:+.2f},{eN:+.2f}) '
                      f'cmd=({self.cmd_vel[0]:+.2f},{self.cmd_vel[1]:+.2f}) '
                      f'miss={self.kf_miss}')
        return math.hypot(eE, eN)

    def _funnel_radius(self):
        return max(self.land_align_radius, self.descend_cone * self.marker_h)

    def _servo_to(self, tE, tN, vffE=0.0, vffN=0.0):
        eE = tE - self.pos[0]
        eN = tN - self.pos[1]
        self.cmd_vel[0] = self._clamp(vffE + self.vel_gain * eE,
                                      -self.vel_max, self.vel_max)
        self.cmd_vel[1] = self._clamp(vffN + self.vel_gain * eN,
                                      -self.vel_max, self.vel_max)
        return math.hypot(eE, eN)

    # 카메라 프레임 tvec(m) → 마커 월드(E,N) 위치. tvec.x=이미지오른쪽, tvec.y=이미지
    # 아래(m). 이미지→기체(마운트 부호), 기체→월드(현재 yaw 회전), 드론위치 가산.
    def _measure_marker_world(self):
        gx, gy = self.tvec[0], self.tvec[1]        # 오른쪽 m, 아래 m
        if self.lat_swap:
            gx, gy = gy, gx
        fwd = self.lat_sign_fwd * (-gy)            # 기체 전방(+X)
        left = self.lat_sign_left * (-gx)          # 기체 좌측(+Y)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        de = c * fwd - s * left
        dn = s * fwd + c * left
        return self.pos[0] + de, self.pos[1] + dn

    # --- 칼만 필터(등속, 상태 [E, N, vE, vN]) -------------------------------
    def _kf_reset(self):
        self.kf_init = False
        self.kf_miss = 0

    def _kf_predict(self):
        if not self.kf_init:
            return
        dt = DT
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        sa2 = self.kf_accel_std ** 2
        Q = sa2 * np.array([[dt**4 / 4, 0, dt**3 / 2, 0],
                            [0, dt**4 / 4, 0, dt**3 / 2],
                            [dt**3 / 2, 0, dt**2, 0],
                            [0, dt**3 / 2, 0, dt**2]], dtype=float)
        self.kf_x = F @ self.kf_x
        self.kf_P = F @ self.kf_P @ F.T + Q

    def _kf_update(self, zE, zN):
        r2 = self.kf_meas_std ** 2
        if not self.kf_init:
            self.kf_x = np.array([zE, zN, 0.0, 0.0])
            self.kf_P = np.diag([r2, r2, 1.0, 1.0])
            self.kf_init = True
            return
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = r2 * np.eye(2)
        z = np.array([zE, zN])
        y = z - H @ self.kf_x
        S = H @ self.kf_P @ H.T + R
        K = self.kf_P @ H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(4) - K @ H) @ self.kf_P

    def _publish_velocity(self):
        pt = PositionTarget()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED   # mavros ENU→NED 변환
        pt.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                        PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                        PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                        PositionTarget.IGNORE_YAW_RATE)
        pt.velocity.x = self.cmd_vel[0]   # East
        pt.velocity.y = self.cmd_vel[1]   # North
        pt.velocity.z = self.cmd_vel[2]   # Up
        pt.yaw = self.hold_yaw            # 헤딩 고정(카메라 매핑 유효성 유지)
        self.raw_pub.publish(pt)

    # -----------------------------------------------------------------------
    def _set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        self.set_mode_cli.call_async(req)

    def _arm(self, value):
        req = CommandBool.Request()
        req.value = value
        self.arming_cli.call_async(req)

    def _takeoff(self, alt):
        req = CommandTOL.Request()
        req.altitude = float(alt)
        self.takeoff_cli.call_async(req)

    def _log(self, text):
        self.get_logger().info(text)
        self.debug_pub.publish(String(data=text))

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))


def main(args=None):
    rclpy.init(args=args)
    node = PreclandHwNode()
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
