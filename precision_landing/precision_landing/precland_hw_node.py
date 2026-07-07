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
  IDLE      : armed 대기. 시동 순간 위치를 홈으로 캡처(마커 위에서 시동 권장).
              GUIDED 되면 GOHOME 으로(return_home_on_guided=true).
  GOHOME    : 홈(시동 지점=마커 위)으로 flight_alt 유지하며 자동 복귀. 마커가
              시야에 들어오면 ALIGN 으로 인계. (모드 전환 직후 수동 조종 회피)
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
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Range
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
    GOHOME = 6


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
        # ============================ PID 속도 서보 ============================
        # 수평 정렬 제어식:  v = Kp·e + Ki·∫e dt + Kd·de/dt (+ 피드포워드),  vel_max clamp
        #   e = 마커 월드위치 − 드론위치 (축별 E, N).  출력 v = FC 로 보내는 속도 명령.
        #   Kp(vel_gain): 위치오차 비례. 응답 속도. 크면 빠르지만 진동/오버슈트.
        #   Ki(vel_ki)  : 적분. 바람 등 지속 외란의 정상상태 치우침 제거. 과하면 헌팅.
        #   Kd(vel_kd)  : 미분. 감쇠(진동·오버슈트 억제). 과하면 노이즈에 떨림.
        #
        # ── 튜닝 절차 (지상/저고도에서, 마커 위 호버로) ──────────────────────
        #  1) Ki=0, Kd=0 으로 시작. Kp 를 낮은 값(0.3)부터 조금씩 ↑ —
        #     마커로 신속히 붙되 '진동 직전'에서 멈춘다.
        #  2) 진동(마커 위에서 좌우로 흔들)이 보이면 Kd 를 조금씩 ↑ 해 눌러준다.
        #     (Kd 과하면 모터가 미세하게 떨림 → 그 직전까지만.)
        #  3) 중심에서 한쪽으로 살짝 치우쳐 호버(바람/무게중심)하면 Ki 를 조금씩 ↑ —
        #     남은 오프셋이 서서히 0 으로. (Ki 과하면 느린 왕복 헌팅 → 줄인다.)
        #  4) i_vel_max 는 I 항이 낼 수 있는 최대 속도. 강풍이면 ↑, 폭주 걱정되면 ↓.
        #
        # ── 증상 → 처방 ─────────────────────────────────────────────────────
        #   빠르게 갔다가 지나쳐 되돌아옴(오버슈트)      : Kp↓  또는  Kd↑
        #   마커 위에서 지속 진동/흔들림                 : Kd↑ (안되면 Kp↓)
        #   중심 못 맞추고 옆으로 치우쳐 정지            : Ki↑
        #   중심 부근을 느리게 왕복(헌팅)               : Ki↓
        #   반응이 굼떠 마커 추종이 느림                 : Kp↑
        #   모터가 고주파로 미세 진동                    : Kd↓ (미분이 노이즈 증폭)
        #
        # ── 디버그 로그 관찰 (약 1초마다) ───────────────────────────────────
        #   h=1.20(cam) err=(+0.05,-0.03) i=(+0.12,-0.08) cmd=(+0.18,-0.14) miss=0
        #     err : 수평 오차(m, E/N).  0 으로 수렴해야 정렬 완료.
        #     i   : 적분 누적. 계속 커지기만 하면 Ki 과함/anti-windup 확인.
        #     cmd : 최종 속도 명령(m/s). 떨리면 Kd↓/Kp↓.  vel_max 에 자주 붙으면 상한 고려.
        #
        # 주의: 실기체는 보수적으로(낮은 Kp·vel_max). 게인은 축(E/N) 공통 적용.
        # 라이브 튜닝: 비행 중 재시작 없이 즉시 반영 —
        #   ros2 param set /precland_hw_node vel_kd 0.12
        # 좋은 값을 찾으면 위 기본값에 반영해 저장(다음 실행에도 유지).
        self.vel_gain = self.declare_parameter('vel_gain', 0.6).value       # Kp (1/s)
        self.vel_max = self.declare_parameter('vel_max', 0.8).value         # m/s (출력 상한)
        self.vel_ki = self.declare_parameter('vel_ki', 0.15).value          # Ki (1/s^2)
        self.vel_kd = self.declare_parameter('vel_kd', 0.08).value          # Kd (무차원)
        # I 항이 낼 수 있는 최대 속도(m/s). anti-windup 상한.
        self.i_vel_max = self.declare_parameter('i_vel_max', 0.3).value
        # ======================================================================
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

        # 하방 라이다(레인지파인더)로 높이 측정. 마커 검출과 무관하게 연속·정밀한
        # 수직 거리를 주므로, 저고도에서 마커가 화면을 넘쳐 검출이 끊겨도 고도 판단이
        # 유지됨(하강률·LAND 인계). 수평 정렬(x,y)은 여전히 카메라 마커로 함.
        # ArduPilot RNGFND → MAVROS 는 보통 sensor_msgs/Range 로 발행.
        self.use_lidar_height = self.declare_parameter('use_lidar_height', False).value
        self.lidar_topic = self.declare_parameter(
            'lidar_topic', '/mavros/rangefinder/rangefinder').value
        # 유효 측정 범위(m). 이 밖의 값(0/음수/최대 초과)은 무시하고 폴백.
        self.lidar_min = self.declare_parameter('lidar_min', 0.1).value
        self.lidar_max = self.declare_parameter('lidar_max', 40.0).value
        # 센서 장착면과 착륙 기준면(다리 접지)의 오프셋(m). 라이다가 기체 배 밑에
        # 있으면 실제 다리 접지까지 거리는 range - offset. 필요 없으면 0.
        self.lidar_offset = self.declare_parameter('lidar_offset', 0.0).value
        # 기울어졌을 때 slant range → 수직 높이 보정(range × cosθ). 정렬 중엔 거의
        # 수평이라 영향 작지만, 켜두면 안전.
        self.lidar_tilt_comp = self.declare_parameter('lidar_tilt_comp', True).value

        # --- 홈 복귀 동작 (이 파일에서 직접 튜닝 — 런치에 없음) --------------
        # GUIDED 전환 시: 조종사가 마커 위로 수동 이동하는 대신, 시동(ARM)했던
        # 위치=홈으로 지정 고도(flight_alt) 유지하며 자동 복귀 → 거기서 마커
        # 추적·착륙. 마커 위에서 시동을 걸면 홈=마커 위가 된다. 모드 전환 직후
        # 수동 조종이 어려운 문제를 피함.
        # ★ 이 4개는 런치에서 안 넘기므로 여기 기본값이 곧 실제값. 숫자만 바꾸면 됨.
        self.return_home_on_guided = self.declare_parameter(
            'return_home_on_guided', True).value
        # 홈 도착 판정 반경(m). 이 안 + 고도 맞으면 '홈 도착'.
        self.home_radius = self.declare_parameter('home_radius', 0.6).value
        # 지정 고도 허용오차(m).
        self.alt_tol = self.declare_parameter('alt_tol', 0.4).value
        # 복귀 중 수직 속도 상한(m/s) — 지정 고도까지 오르내림.
        self.climb_rate = self.declare_parameter('climb_rate', 0.5).value

        self.stage = Stage.TAKEOFF if self.auto_takeoff else Stage.IDLE

        # --- 상태 -----------------------------------------------------------
        self.mav_state = State()
        self.pos = [0.0, 0.0, 0.0]     # 로컬 ENU 위치
        self.yaw = 0.0                 # 현재 헤딩(ENU, rad)
        self.hold_yaw = 0.0            # 정렬 중 유지할 헤딩
        self.tvec = None               # 최신 마커 위치(카메라 프레임 m): (right, down, dist)
        self.marker_h = self.flight_alt  # 지면(마커면) 위 높이. 라이다 우선, 없으면 tvec.z
        self.lidar_h = None            # 최신 라이다 수직 높이(m, 보정·오프셋 반영)
        self.lidar_fresh = 0           # 라이다 신선도 카운트다운
        self.cos_tilt = 1.0            # 수직 대비 기체 기울기 cos (R22), 틸트 보정용
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
        self.home_e = 0.0             # 시동 시점 위치(홈, ENU E)
        self.home_n = 0.0             # 시동 시점 위치(홈, ENU N)
        self.home_captured = False    # 시동 시 홈 좌표 캡처 여부
        # PID 속도 서보 상태(축별 E, N)
        self.pid_int = [0.0, 0.0]     # 적분 누적(∫e dt)
        self.pid_prev_e = [0.0, 0.0]  # 직전 오차(미분용)
        self.pid_primed = False       # 리셋 직후 첫 샘플 미분 킥 방지

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
        if self.use_lidar_height:
            self.create_subscription(
                Range, self.lidar_topic, self._range_cb, qos_profile_sensor_data)

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

        # 라이브 튜닝: 비행 중 재시작 없이 게인·임계값을 즉시 반영.
        #   ros2 param set /precland_hw_node vel_kd 0.12
        #   ros2 param get /precland_hw_node vel_kd
        # declare_parameter 를 전부 마친 뒤 등록해야 선언 시점 콜백을 피함.
        self._live_params = {
            'vel_gain', 'vel_ki', 'vel_kd', 'i_vel_max', 'vel_max',
            'descend_rate', 'descend_cone', 'descend_min_scale',
            'land_align_radius', 'land_switch_alt',
            'flight_alt', 'home_radius', 'alt_tol', 'climb_rate',
            'lat_swap', 'lat_sign_fwd', 'lat_sign_left',
            'lidar_min', 'lidar_max', 'lidar_offset', 'lidar_tilt_comp',
        }
        self.add_on_set_parameters_callback(self._on_set_params)

        self.create_timer(DT, self.tick)
        self._log('============ ArUco 정밀착륙 (실기체) ============')
        if self.auto_takeoff:
            self._log(f'자동이륙 ON — 노드가 GUIDED/시동/이륙({self.flight_alt:.1f} m)까지 스스로 합니다.')
        else:
            self._log('진행 순서:  ① 마커 위에서 시동(ARM) → 그 지점이 홈  →  '
                      '② 수동 이륙·비행  →  ③ GUIDED 전환 → 홈(마커 위)으로 자동 '
                      '복귀·정렬·하강·착륙')
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
        # 기체 z축(위)의 월드 수직 성분 = R22. 수평이면 1, 기울면 <1.
        # 라이다 slant range → 수직 높이 = range × cos(tilt) = range × R22.
        self.cos_tilt = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)

    def _marker_cb(self, msg):
        # 카메라 광학 프레임 tvec: x=오른쪽, y=아래, z=마커까지 거리(≈높이).
        self.tvec = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def _range_cb(self, msg: Range):
        # ArduPilot RNGFND → MAVROS. 유효 범위 밖(0/음수/최대 초과)이면 버리고 폴백.
        r = float(msg.range)
        if not (self.lidar_min <= r <= self.lidar_max):
            return
        if self.lidar_tilt_comp and self.cos_tilt > 0.1:
            r *= self.cos_tilt          # slant → 수직 높이
        self.lidar_h = max(r - self.lidar_offset, 0.0)
        self.lidar_fresh = 20           # 20 × 50 ms = 1.0 s 신선도 창

    def _detected_cb(self, msg):
        if msg.data:
            self.fresh = 10   # 10 × 50 ms = 500 ms 신선도 창

    # -----------------------------------------------------------------------
    def tick(self):
        if self.fresh > 0:
            self.fresh -= 1
        if self.lidar_fresh > 0:
            self.lidar_fresh -= 1
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

        # 시동(ARM) 순간의 위치를 홈으로 캡처. 마커 위에서 시동 → 홈=마커 위.
        # disarm 되면 리셋해 다음 시동에서 다시 캡처.
        if armed and not self.home_captured:
            self.home_e, self.home_n = self.pos[0], self.pos[1]
            self.home_captured = True
            self._log(f'홈 위치 캡처(시동 지점): E={self.home_e:+.2f} N={self.home_n:+.2f} m '
                      '(여기가 마커 위여야 함)')
        elif not armed:
            self.home_captured = False

        # 안전 게이트: 비행 중 조종사가 GUIDED 를 벗어나거나 disarm 하면 즉시 중단.
        if self.stage in (Stage.GOHOME, Stage.ALIGN, Stage.DESCEND):
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
            self.hold_yaw = self.yaw      # 복귀/정렬 시작 전까지 현재 헤딩 고정
            self._idle_guide(armed, guided, marker_ok)
            if armed and guided:
                if self.return_home_on_guided and self.home_captured:
                    self._log('GUIDED 전환 — 홈(시동 지점=마커 위)으로 지정 고도 '
                              f'{self.flight_alt:.1f} m 유지하며 복귀합니다.  → 복귀(GOHOME)')
                    self._pid_reset()
                    self.stage = Stage.GOHOME
                elif marker_ok:
                    self._log('마커 감지 — 정밀착륙을 시작합니다!  → 정렬(ALIGN)')
                    self._kf_reset()
                    self.stage = Stage.ALIGN

        elif self.stage == Stage.GOHOME:
            # 홈(시동 지점=마커 위)으로 수평 복귀 + 지정 고도로 상승/하강.
            # 마커가 시야에 들어오면 즉시 정밀 정렬로 인계.
            self._servo_to(self.home_e, self.home_n)
            ez = self.flight_alt - self.pos[2]
            self.cmd_vel[2] = self._clamp(self.vel_gain * ez,
                                          -self.climb_rate, self.climb_rate)
            dist = math.hypot(self.home_e - self.pos[0], self.home_n - self.pos[1])
            self.dbg_tick += 1
            if self.dbg_tick % 20 == 0:
                self._log(f'복귀중 dist={dist:.2f} m  alt={self.pos[2]:.2f}/'
                          f'{self.flight_alt:.1f} m')
            if marker_ok:
                self._log('마커 감지 — 정밀착륙을 시작합니다!  → 정렬(ALIGN)')
                self._kf_reset()
                self.stage = Stage.ALIGN
            elif dist < self.home_radius and abs(ez) < self.alt_tol:
                if self.dbg_tick % 40 == 0:
                    self._log('홈 도착 — 마커 탐색 중(하방 카메라에 마커가 들어와야 착륙 시작).')

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
            self._log('GUIDED 모드 확인 — 홈(마커 위)으로 자동 복귀를 시작합니다.')
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
                self._log('대기 중 — 마커 위에서 시동(ARM)을 걸어주세요(그 지점이 홈).')
            elif not guided:
                self._log('대기 중 — 이륙 후 GUIDED 로 전환하면 홈으로 복귀합니다.')

    # 마커를 등속 칼만 필터로 추종하고 추정 위치로 속도 서보. 드론↔마커 수평거리(m) 반환.
    def _track(self, marker_ok):
        self._kf_predict()
        if marker_ok:
            zE, zN = self._measure_marker_world()
            self._kf_update(zE, zN)
            self.kf_miss = 0
        else:
            self.kf_miss += 1

        # 높이: 라이다 우선(마커 검출과 무관하게 연속), 라이다가 끊기면 카메라 tvec.z
        # 폴백, 둘 다 없으면 직전 marker_h 유지(coast).
        if self.use_lidar_height and self.lidar_fresh > 0 and self.lidar_h is not None:
            self.marker_h = max(self.lidar_h, 0.05)
        elif marker_ok:
            self.marker_h = max(self.tvec[2], 0.05)

        if not self.kf_init:
            self.cmd_vel[0] = self.cmd_vel[1] = 0.0
            return None

        eE = self.kf_x[0] - self.pos[0]
        eN = self.kf_x[1] - self.pos[1]
        # 정지 마커: 속도 피드포워드(kf_x[2:4])는 ≈0 이라 사실상 순수 비례 서보.
        self._servo_to(self.kf_x[0], self.kf_x[1], self.kf_x[2], self.kf_x[3])

        self.dbg_tick += 1
        if self.dbg_tick % 20 == 0:
            hsrc = 'lidar' if (self.use_lidar_height and self.lidar_fresh > 0
                               and self.lidar_h is not None) else 'cam'
            self._log(f'h={self.marker_h:.2f}({hsrc}) err=({eE:+.2f},{eN:+.2f}) '
                      f'i=({self.pid_int[0]:+.2f},{self.pid_int[1]:+.2f}) '
                      f'cmd=({self.cmd_vel[0]:+.2f},{self.cmd_vel[1]:+.2f}) '
                      f'miss={self.kf_miss}')
        return math.hypot(eE, eN)

    def _funnel_radius(self):
        return max(self.land_align_radius, self.descend_cone * self.marker_h)

    def _servo_to(self, tE, tN, vffE=0.0, vffN=0.0):
        # PID 속도 서보(축별 E, N). 출력=속도 명령, vel_max 로 clamp.
        e = (tE - self.pos[0], tN - self.pos[1])
        vff = (vffE, vffN)
        for i in (0, 1):
            # D: 오차 미분(마커 정지 → ≈ -드론속도). 리셋 직후 첫 샘플은 0(킥 방지).
            de = (e[i] - self.pid_prev_e[i]) / DT if self.pid_primed else 0.0
            i_term = self._clamp(self.vel_ki * self.pid_int[i],
                                 -self.i_vel_max, self.i_vel_max)
            u = vff[i] + self.vel_gain * e[i] + i_term + self.vel_kd * de
            u_sat = self._clamp(u, -self.vel_max, self.vel_max)
            # anti-windup(조건부 적분): 출력이 포화 중이고 오차가 더 밀어붙이는
            # 방향이면 적분을 멈춰 누적 폭주를 막는다.
            if not (u != u_sat and u * e[i] > 0.0):
                self.pid_int[i] += e[i] * DT
            self.cmd_vel[i] = u_sat
        self.pid_prev_e = list(e)
        self.pid_primed = True
        return math.hypot(e[0], e[1])

    def _pid_reset(self):
        self.pid_int = [0.0, 0.0]
        self.pid_prev_e = [0.0, 0.0]
        self.pid_primed = False

    # 라이브 튜닝 콜백: `ros2 param set` 이 오면 해당 self.* 를 즉시 갱신 →
    # 다음 제어 tick 부터 새 값 적용(노드 재시작 불필요).
    def _on_set_params(self, params):
        for p in params:
            if p.name in self._live_params:
                setattr(self, p.name, p.value)
                self._log(f'[param] {p.name} = {p.value}')
        return SetParametersResult(successful=True)

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
        self._pid_reset()

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
