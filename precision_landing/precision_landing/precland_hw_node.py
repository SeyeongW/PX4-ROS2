#!/usr/bin/env python3
"""ArUco 정밀착륙 제어기 — 실기체 하드웨어용 (ArduPilot / MAVROS, GUIDED).

SITL 의 precision_landing_node 를 실기체에 맞게 단순화·안전화한 버전입니다.

SITL 대비 달라진 점
  1. 마커 측정: 정규화 오프셋 + HFOV 근사 대신, aruco_pose_node 가 캘리브레이션으로
     푼 **실측 3D 위치(m)** (/perception/marker_pose, 카메라 프레임)를 그대로 사용.
     마커 윗면까지의 높이도 카메라 pose(tvec.z)에서 직접 얻어 FC 고도 오차와 무관.
  2. 이동 마커(cue/APPROACH/moving_marker) 제거 — 정지 마커를 보고 내려앉는 단순 동작.
  3. 터치다운: 공중 강제 disarm 대신 **ArduPilot LAND 모드로 인계** — 저고도까지
     정렬·하강한 뒤 LAND 로 전환하면 FC 가 접지 감지·자동 disarm 을 처리(평지 안전).
  4. 조종사 오버라이드 = 즉시 중단: 비행 제어 단계에서 GUIDED 가 아니거나 disarm 되면
     셋포인트 송출을 멈추고 IDLE 로 복귀. 조종사가 모드 스위치로 언제든 회수 가능.

비행 시나리오 (실비행 검증 반영)
  시동(ARM) 순간 → 노드가 GUIDED 전환 + flight_alt(=5 m) 자동 이륙 → 그 고도에서
  search_time(=5 s) 호버하며 하방 카메라로 마커 탐색. 마커를 보면 즉시 정밀 정렬·
  하강·착륙. 정해진 시간 안에 못 보면 RTL(자동복귀)로 홈에 돌아가 착륙. 단, RTL 로
  내려가는 도중 마커가 시야에 들어오면 GUIDED 로 회수해 그 자리에서 정밀 정렬·착륙.

상태 기계
  IDLE      : 연결 후 시동(ARM) 대기. 시동되면 곧바로 TAKEOFF.
  TAKEOFF   : GUIDED 전환 → flight_alt 까지 자동 이륙(CommandTOL). 도달하면 SEARCH.
  SEARCH    : flight_alt 호버하며 search_time 동안 마커 탐색. 마커 보이면 ALIGN,
              시간 초과면 RTL 요청 후 RTL_WAIT.
  ALIGN     : flight_alt 호버하며 마커 중심으로 수평 정렬(속도 서보).
  DESCEND   : 깔때기(funnel) 허용오차를 따라 정렬하며 동시에 하강.
  LAND      : 저고도·중심 정렬 상태에서 LAND 모드로 전환, FC 에 착륙 인계.
  RTL_WAIT  : FC 의 RTL 로 홈 복귀·착륙에 맡김(셋포인트 미송출). 도중 마커가 보이면
              GUIDED 회수(RTL_INTERCEPT). 단 비전 착륙 시도가 max_land_attempts 회를
              넘으면 마커가 보여도 가로채지 않고 홈 착륙(진동 루프 방지). 마커 없이
              RTL 이 disarm 까지 가면 미션 종료.
  RTL_INTERCEPT : GUIDED 회수 요청 → 모드 확인되면 ALIGN 으로 인계.
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
    IDLE = 0
    TAKEOFF = 1
    SEARCH = 2
    ALIGN = 3
    DESCEND = 4
    LAND = 5
    RTL_WAIT = 6
    RTL_INTERCEPT = 7
    DONE = 8


DT = 0.05  # 50 ms (20 Hz) 제어 주기


class PreclandHwNode(Node):
    def __init__(self):
        super().__init__('precland_hw_node')

        # --- 파라미터 -------------------------------------------------------
        # 자동 이륙 목표 고도 = 마커 탐색 호버 고도(m). 실비행 검증값 5 m.
        self.flight_alt = self.declare_parameter('flight_alt', 5.0).value
        # 이륙 후 이 고도에서 마커를 탐색하는 시간(s). 지나면 RTL 로 복귀.
        self.search_time = self.declare_parameter('search_time', 5.0).value
        # 비전 착륙 시도 상한. SEARCH 가 마커를 못 찾고 RTL 로 떨어질 때마다 1회 소진.
        # 이 횟수를 넘으면 그 다음 RTL 은 마커가 보여도 가로채지 않고 홈에 착륙(진동
        # 루프 방지). 0 이하 = 무제한(항상 가로채기 시도).
        self.max_land_attempts = self.declare_parameter('max_land_attempts', 3).value
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
        # 안전: 비행 제어 단계(SEARCH/ALIGN/DESCEND)에서 GUIDED 가 아니면 셋포인트를
        # 절대 송출하지 않음. 조종사가 모드 스위치로 언제든 회수(RC 오버라이드) 가능.
        # false 로 두면 이 게이트를 끔(권장 안 함).
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

        # --- 이륙/호버 수직 제어 (이 파일에서 직접 튜닝 — 런치에 없음) -------
        # 이륙 도달·호버 고도 허용오차(m).
        self.alt_tol = self.declare_parameter('alt_tol', 0.4).value
        # 이륙 '완료' 판정 고도(m). 이륙 명령은 계속 flight_alt(5 m)로 내리되, 이 고도만
        # 넘으면 이륙됐다고 보고 SEARCH 로 넘어간다. flight_alt 근처에서 간당간당하며
        # 재이륙 명령이 반복되는 것을 막기 위해 flight_alt 보다 넉넉히 낮게 둔다.
        # (flight_alt=5.0 기준 4.0 → 1 m 여유.) flight_alt 보다 낮게 설정할 것.
        self.takeoff_done_alt = self.declare_parameter('takeoff_done_alt', 4.0).value
        # SEARCH 호버 중 고도 유지 수직 속도 상한(m/s).
        self.climb_rate = self.declare_parameter('climb_rate', 0.5).value

        self.stage = Stage.IDLE

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
        self.search_deadline = None    # SEARCH 마커 탐색 마감 시각(초). None=미시작
        # 이번 시동(ARM) 사이클에 자동 이륙을 이미 시작했는지. 조종사가 비행 중
        # 수동 회수(모드 스위치)하면 IDLE 로 가는데, 여기서 곧바로 재이륙하지 않도록
        # 래치. disarm→재arm 하면 리셋되어 다시 이륙 가능.
        self.takeoff_latched = False
        # 이번 비행에서 GUIDED 가 한 번이라도 확립됐는지. 이륙 초기 GUIDED 핸드셰이크
        # (아직 GUIDED 아님)와, 확립 후 조종사 회수(GUIDED 이탈)를 구분하기 위함.
        self.guided_seen = False
        # 비전 착륙 시도 소진 카운트(SEARCH 실패 → RTL 마다 +1). disarm 시 리셋.
        self.land_attempts = 0
        self._final_rtl_logged = False  # '더 이상 가로채지 않음' 안내 1회만
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
            'flight_alt', 'search_time', 'max_land_attempts',
            'alt_tol', 'takeoff_done_alt', 'climb_rate',
            'lat_swap', 'lat_sign_fwd', 'lat_sign_left',
            'lidar_min', 'lidar_max', 'lidar_offset', 'lidar_tilt_comp',
        }
        self.add_on_set_parameters_callback(self._on_set_params)

        self.create_timer(DT, self.tick)
        self._log('============ ArUco 정밀착륙 (실기체) ============')
        self._log(f'진행 순서:  ① 시동(ARM) → 노드가 GUIDED 전환 + {self.flight_alt:.1f} m 자동 이륙  '
                  f'→  ② {self.search_time:.0f} s 호버하며 마커 탐색  →  ③ 마커 보이면 정렬·'
                  '하강·착륙 / 못 보면 RTL 복귀(RTL 중 마커 보이면 회수해 착륙)')
        self._log('* 비행 중 언제든 모드 스위치로 회수 가능 (GUIDED 벗어나면 즉시 제어 중단).')
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
            self._log('다음 할 일: 시동(ARM)을 걸어주세요. 시동 즉시 GUIDED 전환 + '
                      f'{self.flight_alt:.1f} m 자동 이륙합니다.')

        armed = self.mav_state.armed
        guided = self.mav_state.mode == 'GUIDED'
        # 시동 사이클/GUIDED 확립 추적. disarm 하면 다음 시동에서 다시 이륙 가능하게 리셋.
        if not armed:
            self.takeoff_latched = False
            self.guided_seen = False
            self.land_attempts = 0
            self._final_rtl_logged = False
        elif guided:
            self.guided_seen = True

        # 안전 게이트: 노드가 셋포인트로 비행을 지휘하는 단계에서 조종사가 GUIDED 를
        # 벗어나거나 disarm 하면 즉시 중단하고 대기로. (RTL_WAIT/RTL_INTERCEPT 는 일부러
        # GUIDED 가 아니므로 제외 — 각 단계가 스스로 처리.)
        if self.stage in (Stage.TAKEOFF, Stage.SEARCH, Stage.ALIGN, Stage.DESCEND):
            if not armed:
                self._log('시동 꺼짐 — 정밀착륙 중단, 대기 상태로.  -> IDLE')
                self.stage = Stage.IDLE
            elif (self.require_guided and not guided
                  and (self.stage != Stage.TAKEOFF or self.guided_seen)):
                # 조종사 회수(모드 스위치). 단 TAKEOFF 초기 GUIDED 핸드셰이크 중
                # (아직 GUIDED 확립 전)에는 걸지 않는다. 확립 후 이탈하면 즉시 중단.
                self._log('GUIDED 이탈(조종사 회수) — 제어 중단, 대기 상태로.  -> IDLE')
                self.stage = Stage.IDLE

        if self.stage == Stage.IDLE:
            self.cmd_vel = [0.0, 0.0, 0.0]
            self.hold_yaw = self.yaw      # 이륙/정렬 시작 전까지 현재 헤딩 고정
            self._idle_guide(armed, guided, marker_ok)
            # 시동 순간 자동 이륙 — 단, 이번 시동 사이클에 아직 이륙을 안 걸었을 때만
            # (조종사가 수동 회수했다가 다시 오는 경우 재이륙 폭주 방지; disarm 하면 리셋).
            if armed and not self.takeoff_latched:
                self._log(f'시동 감지 — GUIDED 전환 후 {self.flight_alt:.1f} m 자동 '
                          '이륙합니다.  → 이륙(TAKEOFF)')
                self.req_ticks = 0
                self.takeoff_latched = True
                self.stage = Stage.TAKEOFF
            return

        elif self.stage == Stage.TAKEOFF:
            # 시동 직후: GUIDED 전환 → CommandTOL 로 flight_alt 까지 자동 이륙.
            # ArduCopter GUIDED 는 착륙 상태에서 셋포인트만으론 안 뜸 → 명시적 takeoff.
            self.req_ticks += 1
            send = (self.req_ticks % 40 == 1)      # ~2 s 마다 재요청
            if not guided:
                if send:
                    self._log('이륙: GUIDED 전환 요청'); self._set_mode('GUIDED')
            elif self.pos[2] < self.takeoff_done_alt:
                # 명령 목표는 계속 flight_alt(5 m) — 완료 판정만 takeoff_done_alt(4 m).
                if send:
                    self._log(f'이륙: {self.flight_alt:.1f} m 로 상승 명령 '
                              f'(현재 {self.pos[2]:.1f} m)')
                    self._takeoff(self.flight_alt)
            else:
                self._log(f'이륙 완료(≥{self.takeoff_done_alt:.1f} m, 현재 {self.pos[2]:.1f} m) '
                          f'— {self.search_time:.0f} s 동안 마커 탐색.  → 탐색(SEARCH)')
                self.search_deadline = self._now() + self.search_time
                self._pid_reset()
                self.stage = Stage.SEARCH
            return

        elif self.stage == Stage.SEARCH:
            # flight_alt 호버(수평 정지 + 고도 유지)하며 하방 카메라로 마커 탐색.
            self.cmd_vel[0] = self.cmd_vel[1] = 0.0
            ez = self.flight_alt - self.pos[2]
            self.cmd_vel[2] = self._clamp(self.vel_gain * ez,
                                          -self.climb_rate, self.climb_rate)
            self.hold_yaw = self.yaw
            remain = (self.search_deadline or self._now()) - self._now()
            self.dbg_tick += 1
            if self.dbg_tick % 20 == 0:
                self._log(f'마커 탐색중 남은시간={max(remain, 0.0):.1f} s  '
                          f'alt={self.pos[2]:.2f}/{self.flight_alt:.1f} m')
            if marker_ok:
                self._log('마커 발견 — 정밀착륙을 시작합니다!  → 정렬(ALIGN)')
                self._kf_reset()
                self.stage = Stage.ALIGN
            elif remain <= 0.0:
                self.land_attempts += 1
                unlimited = self.max_land_attempts <= 0
                left = (self.max_land_attempts - self.land_attempts) if not unlimited else -1
                if unlimited or left > 0:
                    self._log(f'{self.search_time:.0f} s 내 마커 미발견 — RTL 로 홈 복귀합니다. '
                              f'(비전 착륙 시도 {self.land_attempts}'
                              f'{"" if unlimited else "/" + str(self.max_land_attempts)}, '
                              'RTL 중 마커 보이면 재가로채기)  → RTL(RTL_WAIT)')
                else:
                    self._log(f'{self.search_time:.0f} s 내 마커 미발견 — 비전 착륙 시도 '
                              f'{self.land_attempts}/{self.max_land_attempts} 소진. 이후로는 '
                              'RTL 홈 착륙에 맡깁니다(재가로채기 안 함).  → RTL(RTL_WAIT)')
                self._set_mode('RTL')
                self.req_ticks = 0
                self.stage = Stage.RTL_WAIT
                return

        elif self.stage == Stage.RTL_WAIT:
            # FC 의 RTL 에 홈 복귀·하강·착륙을 맡긴다(셋포인트 미송출로 간섭 방지).
            # 내려가는 도중 하방 카메라에 마커가 들어오면 GUIDED 로 회수해 정밀 착륙.
            self.dbg_tick += 1
            if not armed:
                self._log('RTL 착륙·시동 꺼짐 확인 — 미션 종료. 수고하셨습니다. (노드 종료)')
                self.stage = Stage.DONE
                rclpy.shutdown()
                return
            # 시도 상한 초과: 마커가 보여도 가로채지 않고 FC 의 RTL 홈 착륙에 맡긴다.
            attempts_left = (self.max_land_attempts <= 0
                             or self.land_attempts < self.max_land_attempts)
            if marker_ok and attempts_left:
                self._log('RTL 복귀 중 마커 포착 — GUIDED 로 회수해 정밀 정렬합니다.  '
                          '→ 회수(RTL_INTERCEPT)')
                self.req_ticks = 0
                self.stage = Stage.RTL_INTERCEPT
            elif marker_ok and not attempts_left:
                if not self._final_rtl_logged:
                    self._final_rtl_logged = True
                    self._log(f'마커 보이지만 비전 착륙 시도 {self.max_land_attempts}회 소진 — '
                              '재가로채기 없이 RTL 홈 착륙을 계속합니다.')
            elif self.dbg_tick % 40 == 0:
                self._log(f'RTL 복귀 중(mode={self.mav_state.mode}) — 마커 탐색 지속.')
            return                                 # 셋포인트 송출 안 함

        elif self.stage == Stage.RTL_INTERCEPT:
            # RTL → GUIDED 회수. 모드가 실제로 GUIDED 로 바뀔 때까지 재요청하고,
            # 확인되면 ALIGN 으로 인계(그 전엔 셋포인트를 보내지 않아 안전 게이트 회피).
            self.req_ticks += 1
            if not armed:
                self._log('시동 꺼짐 — 미션 종료. (노드 종료)')
                self.stage = Stage.DONE
                rclpy.shutdown()
                return
            if guided:
                self._log('GUIDED 회수 완료 — 정밀 정렬을 시작합니다.  → 정렬(ALIGN)')
                self._kf_reset()
                self.stage = Stage.ALIGN
            elif self.req_ticks % 20 == 1:
                self._log('GUIDED 회수 요청 중...'); self._set_mode('GUIDED')
            return                                 # GUIDED 확인 전엔 송출 안 함

        elif self.stage == Stage.ALIGN:
            err = self._track(marker_ok)
            self.cmd_vel[2] = 0.0                 # 고도 유지
            if self.kf_miss > self.coast_ticks:
                # 마커 상실: 대기로 죽지 말고 flight_alt 로 올라가 재탐색(다시 못 찾으면 RTL).
                self._log(f'마커를 놓쳤습니다 — {self.flight_alt:.1f} m 로 올라가 재탐색.  → 탐색(SEARCH)')
                self.search_deadline = self._now() + self.search_time
                self.stage = Stage.SEARCH
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
    # 조종사용 안내(IDLE 대기 중). 시동만 걸면 노드가 GUIDED 전환·이륙부터 착륙까지
    # 자동으로 진행하므로, 여기서는 '시동 대기' 리마인드만.
    def _idle_guide(self, armed, guided, marker_ok):
        if not armed and self._prev_armed:
            self._log('시동 꺼짐 (DISARMED). 다시 시동을 걸면 자동 이륙합니다.')
        self._prev_armed = armed

        self._guide_tick += 1
        if self._guide_tick % 60 == 0:          # 3초마다 현재 필요한 동작 리마인드
            if not armed:
                self._log(f'대기 중 — 시동(ARM)을 걸어주세요. 즉시 GUIDED + '
                          f'{self.flight_alt:.1f} m 이륙 후 마커를 탐색합니다.')

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

    def _now(self):
        # 단조 증가 시각(초). SEARCH 탐색 마감 계산용.
        return self.get_clock().now().nanoseconds * 1e-9

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
