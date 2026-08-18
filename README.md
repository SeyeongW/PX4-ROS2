# PX4-ROS2 — CJU 이동 트레일러 자율 미션

청주대학교 운동장용 `wang` 프로파일을 `jo` 브랜치에서 개발·검증한
ROS 2/PX4 스택이다. 드론은 장애물을 피해 목표점까지 이동한 뒤, 움직이는
트레일러의 GPS cue로 전역 경로를 갱신하고 ArUco/PX4 PRECLAND로 착륙한다.

> 현재 판정 — 2026-08-18
>
> - Gazebo GPS/MAVLink-in-the-loop 전체 미션: `DONE`, 자동 착륙·disarm 성공
> - 경로 planner/SFC/ABORT/failsafe 실패: 0
> - 실기체 props-on: **NO-GO** — 아래 P0 항목을 먼저 완료해야 한다
> - 최신 검증은 dirty worktree에서 수행했으므로 커밋 후 clean 재실행이 필요하다

## 8월 17~18일 인수인계 핵심

| 영역 | 반영 내용 |
|---|---|
| 맵 | 정수 중심좌표 장애물 25개, YAML/Gazebo collision/visual/test 동기화. `barrier_19`는 `(35,49,5)`로 복구 |
| 안전반경 | raw 물리 장애물 + 드론 중심 hard 반경 1.0 m + 추종 reserve 0.5 m = 계획반경 1.5 m |
| 전역계획 | 최소 2초 주기의 최신 cue마다 별도 프로세스에서 A* → optimizer SFC → B-spline 전체 재수행 |
| SFC/UI | 승인 경로의 모든 선분을 free box로 덮고, 동일 plan 번호의 path/SFC를 한 `MarkerArray`로 원자 갱신 |
| 계획시간 | exact 충돌판정의 broad-phase/scalar 연산 최적화. 알고리즘·가중치·해상도는 변경하지 않음 |
| 착륙 | 하강 전 runway gate, KF bounded coast, 재획득 시 고도 유지, 저고도 재상승·재이륙 방지 |
| GPS | 실기 `wang`의 MAVLink 수신 흐름을 독립 `trailer_link` 패키지로 이식하고 기존 `/marker/cue*` ABI에 연결 |
| Gazebo GPS | 트레일러 odometry를 WGS84/MAVLink로 변환해 PTY serial reader 전체 경로를 시험하는 emulator 추가 |
| 짐벌 | `land` 진입 시 30 deg/s로 완만하게 nadir 이동하고, `DONE` 뒤 명령을 중단 |

논문과 동일한 핵심 전역 구조는 다음과 같다.

```text
latest trailer cue
        ↓
A* topology → optimizer SFC cost → geometry B-spline
        ↓ exact 1.5 m validation
active path + active-path SFC atomic commit
        ↓
TrackingMPC → ArUco/KF alignment → LandingMPC → PX4 PRECLAND
```

계산 중이거나 새 계획이 실패하면 기존에 승인된 path/SFC를 계속 사용한다.

## 안전·SFC 계약

- YAML/Gazebo 장애물 크기는 물리 AABB 그대로다. 장애물마다 별도 1 m halo를
  저장하지 않는다.
- `vehicle_clearance_xy_m: 1.0`은 물리 장애물 표면부터 드론 중심까지의
  runtime hard 반경이다.
- `bspline_clearance_margin_m: 0.5`를 더해 A*/optimizer SFC/B-spline/final
  path는 총 1.5 m로 계획한다. 이중 팽창은 없다.
- optimizer SFC는 B-spline 목적함수에 들어가 경로 형상에 영향을 준다.
- UI의 파란 `active-path SFC`는 최종 승인 경로의 각 선분이 적어도 하나의
  free convex box에 완전히 포함됨을 인증한다. 박스끼리 겹치는 것은 정상이다.
- path와 SFC는 `/mission/active_plan_markers` 한 메시지에서 같은 plan 번호로
  교체된다. UI는 서로 다른 세대의 path/SFC를 조합하지 않는다.

현재 TrackingMPC의 runtime 검사는 hard 1.0 m를 기준으로 한다. 따라서 계획
경로는 1.5 m를 지켜도 실제 기체가 0.5 m reserve 일부를 사용할 수 있다.
실기체에서 실제 1.5 m까지 강제하려면 위치추정 오차를 포함한 obstacle-aware
MPC/guard가 별도 필요하다.

## GPS 좌표 구조

GPS는 양방향 위치 교환이 아니다. 트레일러가 자기 GPS를 방송하고, 드론은
자기 MAVROS global/local 위치와 결합해 트레일러의 PX4-local ENU 좌표를 만든다.

```text
Trailer FCU GLOBAL_POSITION_INT
  → trailer_gps_node
  → /trailer/fix + /trailer/velocity_enu

Drone MAVROS global fix + local ENU pose
  + trailer fix/velocity
  → trailer_target_node
  → /marker/cue + /marker/cue_velocity (px4_local_enu)
  → MissionManager / planner / landing
```

핵심 변환은 다음과 같다.

- ROS local: `x=East`, `y=North`, `z=Up`
- PX4/MAVLink velocity: `vx=North`, `vy=East`, `vz=Down`
- 수신 변환: `ENU = (vy, vx, -vz) × 0.01`
- target: `vehicle_local_ENU + ENU(vehicle_fix → trailer_fix)`
- 별도 stadium heading 회전을 GPS adapter에 다시 적용하지 않는다.
- GNSS altitude는 AMSL/ellipsoid datum 차이 때문에 착륙 제어에 쓰지 않는다.
  `TRAILER_DECK_Z_M`에는 현장에서 측정한 PX4 local-ENU deck 높이를 넣는다.

position/velocity는 같은 수신 epoch와 `px4_local_enu` frame으로 발행한다.
invalid fix, 비유한 값, source regression/freeze, 입력 skew, stale data, 4 Hz 미만
수신은 fail-closed로 cue를 발행하지 않는다.

### GPS 인수인계 핵심 4개

- [`trailer_gps_node.py`](flight/trailer_link/trailer_link/trailer_gps_node.py):
  트레일러 FCU의 serial MAVLink를 받아 `/trailer/fix`와
  `/trailer/velocity_enu`를 같은 수신시각으로 발행한다.
- [`geodesy.py`](flight/trailer_link/trailer_link/geodesy.py): WGS84 두 fix의
  차이를 East/North로 변환한다.
- [`trailer_target_node.py`](flight/trailer_link/trailer_link/trailer_target_node.py):
  드론 MAVROS global/local과 트레일러 fix/velocity를 검증·결합해
  `/marker/cue`와 `/marker/cue_velocity`를 발행한다.
- [`run_gps_cue.sh`](flight/trailer_link/run_gps_cue.sh): 실기 reader/adapter를
  단일 publisher와 source-rate gate로 기동한다.

YAML과 UI가 GPS를 직접 변환하는 구조는 아니다. 위 adapter가 기존
`/marker/cue*` ABI로 변환하고 MissionManager와 live UI가 이를 구독한다.
YAML은 맵·장애물·좌표계 계약을 유지하며, 정적 UI는 YAML만 사용한다.
Gazebo 재현에서는
[`trailer_mavlink_emulator.py`](simulation/gazebo/tools/trailer_mavlink_emulator.py)가
odometry를 WGS84/MAVLink PTY 입력으로 바꿔 같은 네 파일의 경로를 통과시킨다.

## 빌드

ROS 2 Humble, PX4 v1.17 호환 `px4_msgs`, Micro XRCE-DDS Agent가 필요하다.

```bash
cd ~/PX4-ROS2-wang
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
rosdep install --from-paths flight simulation/landing_mpc \
  --ignore-src --rosdistro humble -r -y \
  --skip-keys "px4_msgs ament_python"
colcon build --symlink-install
source install/setup.bash
```

## 실행

### 권장: Gazebo GPS/MAVLink-in-the-loop

이 모드는 Gazebo pose를 `/marker/cue`로 직접 전달하지 않는다. 트레일러
odometry를 WGS84/MAVLink로 만든 뒤 PTY serial, 실제 GPS parser와 target
adapter를 통과시킨다.

```bash
cd ~/PX4-ROS2-wang
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
source install/setup.bash

LANDING_MAP=cju-track TRAILER_CUE_SOURCE=gps TRAILER_LINK=sim GPS_SIM_SEED=4 \
  ./simulation/gazebo/run_gimbal.sh mission
```

기본 emulator 조건은 위치 잡음 0.03 m, 속도 잡음 0.02 m/s, 고정 지연
0.08 s, dropout 0이다. `GPS_SIM_POSITION_NOISE_M`,
`GPS_SIM_VELOCITY_NOISE_M_S`, `GPS_SIM_DELAY_S`, `GPS_SIM_DROPOUT`,
`GPS_SIM_SEED`로 고정한다.

터미널 입력 순서:

1. `takeoff` → `READY`
2. `mission` → 목표 `(50,50)` 도달 후 `HOVER`
3. `land`
4. `DONE`과 PX4 자동 disarm 확인
5. `Ctrl-C`로 ULog/CSV/manifest 정리

### 비교용 direct Gazebo cue

```bash
TRAILER_CUE_SOURCE=gazebo ./simulation/gazebo/run_gimbal.sh mission
```

### 실기 GPS cue component

MAVROS가 드론의 global fix와 local ENU pose를 이미 발행하고 있어야 한다.

```bash
cd ~/PX4-ROS2-wang
read -rp "Trailer serial device [/dev/ttyUSB0]: " TRAILER_DEV
TRAILER_DEV="${TRAILER_DEV:-/dev/ttyUSB0}"
read -rp "Measured trailer deck z in PX4 local ENU [m]: " TRAILER_DECK_Z_M
export TRAILER_DEV TRAILER_DECK_Z_M

TRAILER_LINK=1 TRAILER_BAUD=57600 TRAILER_SYSID=1 \
  ./flight/trailer_link/run_gps_cue.sh
```

이 명령은 cue adapter만 실행한다. 전체 실기체 자동미션 launcher가 아니며,
아래 P0를 해결하기 전 props-on 비행에 사용하지 않는다.

### UI/맵 검사

```bash
# planner·좌표·YAML 계약 검사
python3 simulation/gazebo/tools/cju_mission_ui.py --check

# 정적 이미지
python3 simulation/gazebo/tools/cju_mission_ui.py --save /tmp/cju_path.png

# 실행 중 ROS graph의 원자 path/SFC 구독
python3 simulation/gazebo/tools/cju_mission_ui.py --live
```

UI는 비행 명령을 발행하지 않는다. 정적 프리뷰는 YAML 기반이고, live 모드의
트레일러 위치와 동적 path/SFC만 선택한 cue source를 따른다.

## 최신 전체 실험 결과

기준 artifact: `20260818T173325Z.LSsmQm` (`GPS_SIM_SEED=4`).

| 항목 | 결과 |
|---|---:|
| 전체 상태 | `DONE`, landed, auto-disarm |
| 초기 68 m 계획 | 1.28 s — 이전 동일 workload 12.97 s |
| RETURN 계획 | 8/8 승인, 평균 0.229 s, 최대 0.42 s |
| 움직이는 목표 drift | 최대 0.46 m — 이전 7.64 m |
| planner / SFC / ABORT / failsafe | 0 / 0 / 0 / 0 |
| TrackingMPC 계산 | 평균 5.10 ms, 최대 34.64 ms |
| EKF 기준 추종 RMSE / 최대 | 0.214 / 0.346 m |
| land 입력→DONE | 44.78 s — 이전 82.54 s |
| DESCEND / PRECLAND recovery | 0 / 0 |
| 착륙 XY 오차 | 0.281 m |
| 접촉 후 상승·재무장 | 0 / 0 |
| ArUco 착륙구간 검출률 | 67.3% |

계획 B-spline의 최소 장애물 거리는 1.5959 m였다. Gazebo ground-truth 실제
최소거리는 `barrier_14`에서 1.4528 m였다. hard 1.0 m는 지켰지만 계획선
1.5 m보다 4.7 cm 안쪽이므로, 0.5 m 추종 reserve의 약 9.4%를 사용했다.
planner/SFC 충돌이 아니라 추종·위치추정 편차다. `barrier_19`의 실제
최소거리는 3.017 m로, y=49 복구 후 이전 근접 문제는 제거됐다.

최신 실행의 quality는 기능 실패가 아니라 `dirty_tree` 때문에 `WARN`이다.
아래 커밋을 clean checkout에서 같은 조건으로 재실행해야 발표용 재현 근거가 된다.

## 논문과의 관계

일치하는 부분:

- 최소 2초 주기의 moving-target 요청마다 A* → SFC → B-spline 전체 전역계획
- SFC corridor 비용이 B-spline 형상에 영향
- ArUco 정렬 → 하강 → 착지, constant-velocity KF

실기 적용을 위해 바꾼 부분:

- 논문 추종기는 2D unicycle MPC지만 현재는 PX4 P/V/A interface용 3D
  double-integrator TrackingMPC다.
- 논문은 marker KF velocity feed-forward를 기술하지만 현재 LandingMPC는
  GPS/Gazebo `/marker/cue_velocity`를 사용하고 ArUco는 위치 bias 보정에 쓴다.
- GPS/MAVLink, gimbal, runway gate, bounded KF coast, PX4 PRECLAND와
  접촉 후 재상승 금지는 논문 외 실기 안전 확장이다.

따라서 발표에서는 “논문 전역계획 구조를 재현하고 PX4 실기 인터페이스에 맞게
추종·착륙 제어를 확장했다”고 설명해야 한다. 논문 구현을 수치까지 그대로
복제했다고 표현하면 안 된다.

## 실기체 적용 전 P0

1. **GPS 시간 무결성**: `GLOBAL_POSITION_INT.time_boot_ms`를 ROS 시각에
   매핑하고 reboot/wrap을 처리해야 한다. 현재 수신시각 restamp는 RF/queue
   지연 패킷을 fresh로 보이게 할 수 있다.
2. **RTK 품질 gate**: 드론·트레일러의 RTK fixed, hacc/eph, correction age,
   covariance와 동일 datum을 control gate로 강제해야 한다.
3. **실측 기준점**: GNSS antenna/EKF origin→landing marker lever arm·yaw,
   `TRAILER_DECK_Z_M`, 현장 장애물 좌표를 측량해야 한다.
4. **PX4/px4_msgs 고정**: 현재 성공은 저장소 밖의 수정된 PX4 PRECLAND와
   matching `px4_msgs`에 의존한다. patch, firmware, revision을 저장소에서
   재현 가능하게 고정해야 한다.
5. **실기 perception/actuator**: SIYI A8 calibration·TF, ArUco topic chain,
   실제 gimbal feedback과 top-level hardware launcher가 완결되지 않았다.
6. **HIL/props-off**: delay/dropout/fix-loss/reboot, camera loss, planner failure,
   RC takeover/kill, 접촉·미끄러짐·auto-disarm을 고장주입으로 통과해야 한다.

추가로 ArUco 검출률 67.3%, LANDING_DESCEND 최대 하강속도 약 0.885 m/s,
짐벌 slew saturation이 남아 있다. 이 조건 전에는 UI/topic replay,
GPS bench, props-off와 HIL까지만 허용한다.

## 검증

```bash
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
source install/setup.bash

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest flight/path_plan/test -q
python3 -m pytest flight/trailer_link/test -q
python3 -m pytest simulation/landing_mpc/test -q
python3 -m pytest simulation/gazebo/test -q

python3 simulation/gazebo/tools/cju_mission_ui.py --check
bash -n simulation/gazebo/run_gimbal.sh flight/trailer_link/run_gps_cue.sh
```

8월 18일 커밋 전 마지막 결과는 관련 Python 회귀시험 159개 통과
(`path_plan 22 + trailer_link 13 + landing_mpc 104 + gazebo 20`)였다.

## 주요 파일

| 경로 | 역할 |
|---|---|
| [`flight/path_plan`](flight/path_plan) | A*, optimizer SFC, B-spline, active-path SFC, exact collision |
| [`flight/trailer_link`](flight/trailer_link) | trailer MAVLink GPS 수신과 PX4-local ENU cue |
| [`simulation/landing_mpc`](simulation/landing_mpc) | 미션 상태기계, Tracking/Landing MPC, gimbal/ArUco 결합 |
| [`simulation/gazebo`](simulation/gazebo) | CJU 맵·월드, launcher, GPS emulator, UI, exporter |
| [`camera/aruco_landing`](camera/aruco_landing) | ArUco 검출·품질 gate |
| [`camera/siyi_gimbal`](camera/siyi_gimbal) | SIYI 제어·상태 |

실행 artifact는 기본적으로 다음 경로에 저장된다.

```text
${XDG_STATE_HOME:-$HOME/.local/state}/px4-ros2-wang/cju/<run-id>/
```

`DONE` 뒤 `Ctrl-C`는 ULog/CSV/manifest를 정리하는 정상 종료 절차다.
