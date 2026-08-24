# PX4-ROS2 — JO 도심 고속환경

`applepark_city_uav` 도심맵에서 7 m/s 트레일러 순찰과
`A* → SFC → B-spline → MPC` 경로계획을 검증하는 ROS 2 / PX4 연구
workspace다. 청주대 운동장 실기 적용 프로파일은 `PX4-ROS2-wang`에서
관리하며, 이 workspace의 운영 대상은 도심맵이다.

## 현재 범위

| 항목 | JO 계약 |
|---|---|
| map | `simulation/gazebo/maps/city_coordinates_uav.yaml` |
| world | `applepark_city_uav` |
| drone spawn | trailer WP0 deck, Gazebo ENU `(-150, 507, 2.051)` |
| takeoff / mission altitude | fixed 10 m |
| integrated drone speed | mission / TrackingMPC / hard / PX4 cap 12 m/s |
| trailer | 23-waypoint stop-turn patrol, 7 m/s |
| trailer acceleration / command rate | 9 m/s² / 50 Hz |
| landing straight-run alignment reserve | 20 s |
| planning clearance | raw building AABB + 1.0 m |
| runtime vehicle/safety-marker radius | 1.0 m |

트레일러 waypoint, stop index, city world/texture와 driver는 JO 고유 계약이다.
Wang 패치를 이식할 때 이 값들은 변경하지 않는다.

## 2026-08-18 공통 패치 이식

Wang의 다음 공통 기능을 JO 코드에 병합했다.

- raw AABB + rounded XY clearance 계약
- exact segment broad-phase와 scalar 거리 연산 최적화
- free convex box로 최종 polyline을 덮는 active-path SFC helper
- `flight/trailer_link` GPS/MAVLink 수신·좌표변환·fail-closed adapter
- Gazebo odometry→WGS84→MAVLink→PTY emulator와 회귀시험
  (도심 one-command의 기본 GPS 입력 경로)
- MissionManager의 비동기 전체 A*→optimizer SFC→B-spline 재계획,
  path/SFC 원자 교체, LandingMPC runway/coast/저고도 terminal latch
- JO의 기존 `ground_contact` 단방향 래치와 접촉 후 재상승 금지 유지
- JO 트레일러는 `takeoff` clearance에서 출발하는 기존 gate 유지

비활성 CJU fixture는 공통 MissionManager 회귀시험용으로만 Wang과
동기화했다. JO 운영 맵이나 23개 waypoint를 대체하지 않는다.

## 자동 전체 미션 실행판 두 개

두 실행판은 표준 `city_coordinates_uav.yaml`의 WP0 데크 `(-150, 507)`에
드론과 트레일러를 함께 배치한다. 드론이 10 m까지 이륙한 뒤 트레일러가
7 m/s 폐루프 순찰을 시작한다. 미션점은 요청 위치와 가장 가까우면서 건물
미션 좌표는 건물 사이의 안전한 도심 지점 `(-165, 0)`이며, 다음 순서를 자동
실행한다.

```text
TAKEOFF → READY → MISSION_PLAN → MISSION → HOVER(3 s)
        → RETURN_PLAN ↔ RETURN → LANDING_ACQUIRE → PRECLAND → DONE
```

귀환은 고정 staging point를 사용하지 않는다. 최신 GPS cue를 향해 2초마다
`A* → SFC → B-spline`을 다시 만들고, 검증된 새 경로만 원자적으로
교체하면서 트레일러를 추종한다. mission/RETURN 기준속도, TrackingMPC 상한,
내부 hard cap, PX4 cruise/velocity cap은 모두 12 m/s다. YAML 23개 waypoint는
실제 driver와 지도 모두 WP22→WP0까지 연결된
4028.839 m 폐루프로 취급한다.

실제 화면을 보면서 확인하는 1.00× 기준판:

```bash
cd ~/PX4-ROS2-jo && ./simulation/gazebo/run_city_landing.sh visual
```

Gazebo GUI, ArUco 검출 화면, live mission map이 열린다. `DONE`과 자동 무장해제를
확인한 뒤 `Ctrl-C`로 종료한다.

YAML이 배속과 화면/종료 정책을 소유하는 5.00× 실험판:

```bash
cd ~/PX4-ROS2-jo && ./simulation/gazebo/run_city_landing.sh fast
```

이 판은 Gazebo와 ArUco 화면을 숨기고 YAML 경로 live map만 연다. 최신 승인
경로는 초록색, 교체된 복귀 경로는 보라색으로 누적 표시하며,
`PRECLAND -> DONE` 뒤 자동 종료한다. 배속은
`simulation/gazebo/profiles/city_landing_fast.yaml`의
`simulation_speed_factor` 한 곳에서 바꾼다. 배속은 물리 속도 7/12 m/s를
곱하지 않고 Gazebo/PX4의 simulation-time 진행률만 바꾼다. 5.00×에서는
2초 sim-time 요청 주기가 wall-time 0.4초지만 단일 planner가 직렬화하므로,
승인 경로는 실제 계산이 끝나는 약 1.7~3.3 wall-time 간격으로 교체된다.
fast 프로필은 이 부하에 맞춰 SITL GPS를 10 Hz로 올리고 cue timeout을 2초로
확장한다. 물리 기준 성공 판정은 1.00× 시각화판으로 수행한다.

## 도심 GPS 수동 전체 미션

빌드가 끝난 workspace에서 다음 한 줄로 실행한다.

```bash
cd ~/PX4-ROS2-jo && LANDING_MAP=city ./simulation/gazebo/run_gimbal.sh mission
```

`city` 프로파일은 별도 환경변수 없이 GPS cue와 Gazebo MAVLink PTY emulator를
선택한다. 통합 launcher와 PX4의 수평 속도 계층은 모두 12 m/s다. 트레일러는 YAML의
WP0에서 7 m/s로 실행된다. 드론은 같은 WP0 데크 위에서 시작하고, 10 m 이륙이
끝난 뒤 트레일러가 출발한다. UI도 함께 열리며 명령 prompt에는 아래 순서대로
입력한다.

```text
명령> takeoff   # 트레일러 WP0 데크에서 이륙 후 10 m READY까지 대기
명령> mission   # YAML goal까지 10 m로 계획·추종 후 3초 HOVER
명령> land      # 최신 GPS cue로 복귀·추종·착륙 후 DONE까지 대기
```

도심 비행 authority는 standalone `/path_plan/cmd_vel`이 아니라 MissionManager에
통합된 다음 경로다.

```text
GPS local ENU cue + vehicle local ENU
  → MissionManager current-start/latest-goal snapshot
  → city Gazebo-world ENU 변환
  → A* → optimizer SFC → cubic B-spline
  → exact collision + active-path SFC 검증
  → path/SFC 원자 교체 → 3-D TrackingMPC → PX4 OFFBOARD
  → ArUco/LandingMPC → PX4 PRECLAND
```

`mission`은 YAML goal을 사용하고, `land`의 이동 표적 복귀는 최신 GPS cue를
목표로 비동기 전체 A*→SFC→B-spline을 재계획한다. 새 path와 SFC가 모두
검증된 경우에만 같은 plan 번호로 교체하며 실패하면 기존 안전 경로를 유지한다.
standalone `ros2 launch path_plan path_plan.launch.py`는 별도 planner demo이고
이 one-command 비행 authority가 아니다.

## City YAML 오프라인 동적경로 10회 실험

논문 그림의 `city-yaml`은 Gazebo/PX4 비행이 아니라
`city_uav_trailer_loop.yaml`을 사용하는 결정론적 오프라인 롤아웃이다. 드론
속도 상한과 MPC 기준속도는 12 m/s, 트레일러 속도는 10 m/s로 고정된다.
A* 동적 재계획, optimizer SFC, B-spline, MPC horizon과 최종 실행경로를 10회
반복하고 각 실행 결과를 독립 디렉터리에 보존한다.

```bash
cd ~/PX4-ROS2-jo
./simulation/gazebo/run_experiments_10x.sh city-yaml 10
```

기본 CSV 위치는 다음과 같다.

```text
~/.local/state/px4-ros2-jo/batches/city-yaml_<UTC>/
├── batch_runs.csv                 # 10회 성공/실패와 개별 파일 위치
├── batch_summary.csv              # 성공한 실행별 핵심지표 1행
├── batch_statistics.csv           # 10회 mean/std/min/max
└── run_01/ ... run_10/
    ├── data/offline_timeseries_10hz.csv
    ├── data/path_points.csv
    ├── tables/offline_plan_attempts.csv
    ├── tables/offline_replan_metrics.csv
    ├── tables/offline_sfc_boxes.csv
    ├── tables/summary_metrics.csv
    └── figures/{01_pipeline_four_panels,02_pipeline_overlay_four_paths}.png
```

저장 위치를 고정하려면 `BATCH_ROOT`를 지정한다.

```bash
cd ~/PX4-ROS2-jo
BATCH_ROOT="$HOME/Gazebo_filght_planned_10x" \
  ./simulation/gazebo/run_experiments_10x.sh city-yaml 10
```

동일 YAML에는 난수나 외란 주입이 없어 10회 경로 형상은 반복되는 것이
정상이며 A*/B-spline/SFC/MPC wall-clock 계산시간이 주된 반복 통계다.
`sfc_violation_count`는 optimizer control-point box만으로 차량의 active-path
이탈을 판정할 수 없어 `N/A`로 기록한다. 퇴화 seed fallback은 raw 최소폭에서
제외하지 않고 별도 개수·비율과 비퇴화 최소폭/p05를 함께 저장한다. ArUco,
착륙오차와 touchdown 지표도 이 데이터셋에는 telemetry가 없어 `N/A`다.
Gazebo/PX4 경로추종 10회 모드는
`city-path`, 착륙까지 포함한 도심 10회 모드는 `city`를 사용한다.

## GPS cue

실기 수신 경로는 Wang과 같다. 이는 양방향 GPS 교환이 아니라, 트레일러가
자기 위치를 보내고 드론이 자기 MAVROS 위치와 결합하는 단방향 target cue다.

```text
Trailer FCU GLOBAL_POSITION_INT
  → trailer_gps_node
  → /trailer/fix + /trailer/velocity_enu

Drone MAVROS global fix + local ENU pose
  + trailer fix/velocity
  → trailer_target_node
  → /marker/cue + /marker/cue_velocity
```

- cue frame: `px4_local_enu`, x=East, y=North, z=Up
- horizontal velocity: MAVLink `(vx North, vy East)` → `(vy, vx) × 0.01`
- GNSS 수직속도는 제어 cue에서 사용하지 않는다.
- deck z는 GNSS altitude가 아니라 실측 `TRAILER_DECK_Z_M`을 사용한다.
- JO city는 드론 spawn에서 최대 약 1.55 km 떨어지므로 field runner의
  `GPS_MAX_DISTANCE_M` 기본값은 2000 m다.

### 인수인계용 GPS 핵심 4개

| 파일 | 역할 |
|---|---|
| `flight/trailer_link/trailer_link/trailer_gps_node.py` | 트레일러 FCU의 `GLOBAL_POSITION_INT`를 받아 `/trailer/fix`와 ENU `/trailer/velocity_enu`를 같은 수신시각으로 발행한다. |
| `flight/trailer_link/trailer_link/geodesy.py` | WGS84 두 fix의 차이를 ENU로 바꾸고 `drone local ENU + 상대 ENU`로 트레일러 local 좌표를 계산한다. |
| `flight/trailer_link/trailer_link/trailer_target_node.py` | 드론 MAVROS global/local과 트레일러 fix/velocity를 검증·결합해 `/marker/cue*`를 발행한다. |
| `flight/trailer_link/run_gps_cue.sh` | 실기용 reader/adapter를 단일 publisher와 source-rate gate로 기동한다. 전체 hardware mission launcher는 아니다. |

YAML과 UI가 GPS를 직접 해석하지는 않는다. GPS adapter가 기존 ABI인
`/marker/cue`와 `/marker/cue_velocity`로 변환하고, MissionManager와 UI가 이를
구독한다. YAML은 맵·장애물·좌표계 계약만 보관한다.

### 실기 GPS cue 실행

MAVROS가 실행 중이고 deck z를 측정한 뒤 cue adapter만 구동한다.

```bash
cd ~/PX4-ROS2-jo
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
source install/setup.bash

read -rp "Measured trailer deck z in PX4 local ENU [m]: " TRAILER_DECK_Z_M
export TRAILER_DECK_Z_M
TRAILER_DEV=/dev/ttyUSB0 \
TRAILER_LINK=1 \
TRAILER_BAUD=57600 \
TRAILER_SYSID=1 \
GPS_MAX_DISTANCE_M=2000 \
  ./flight/trailer_link/run_gps_cue.sh
```

`run_gps_cue.sh`는 전체 hardware mission launcher가 아니다. 또한 현재 GPS
cue는 PX4-local ENU `PointStamped`이며 MissionManager가 city YAML의 spawn,
origin, heading 계약으로 Gazebo-world ENU planner 좌표로 명시적으로 변환한다.
외부에서 두 좌표를 섞거나 단순 remap하지 않는다.

### 도심 Gazebo GPS/MAVLink-in-the-loop 내부 경로

위 one-command는 다음 software wiring을 자동으로 구성한다.

```text
/model/trailer/odometry
  → simulation/gazebo/tools/trailer_mavlink_emulator.py
  → WGS84 GLOBAL_POSITION_INT over PTY
  → trailer_gps_node → /trailer/fix + /trailer/velocity_enu
  → trailer_target_node → /marker/cue + /marker/cue_velocity
  → MissionManager + live UI
```

이는 실기와 같은 reader/adapter 인터페이스를 통과하는 도심 GPS-SITL 시험이지만
RF·RTK 링크 성능시험은 아니다.

### SITL 전용 착륙 파라미터

city YAML의 `LNDMC_XY_VEL_MAX=12`는 비행 명령 상한이 아니라 PX4 land
detector가 이동 deck의 7 m/s world-frame 속도를 허용하는 접촉판정 임계값이다.
`PLD_FAPPR_ALT=0`도 SITL 전용이다. 둘 다 hardware-ready 설정이 아니며 실기체에
그대로 복사하지 않는다. 실기는 deck-relative 접촉 판정과 속도/고도 전환을
별도로 검증해야 한다.

## GPS 실기 전 P0

- `GLOBAL_POSITION_INT.time_boot_ms`↔ROS clock mapping과 RF queue latency
- drone/trailer RTK fixed, hacc/eph, correction age, covariance, 동일 datum gate
- antenna/EKF origin→landing marker lever arm·yaw와 deck z 측량
- 실제 radio dropout/fix-loss/reboot/wrap fault injection
- 외부 PX4 PRECLAND patch와 matching `px4_msgs` revision 고정
- props-off/HIL, RC takeover/kill, single flight authority 검증

Gazebo MAVLink emulator는 software wiring 시험이지 RF·RTK 성능 시험이 아니다.

## 빌드와 회귀시험

```bash
cd ~/PX4-ROS2-jo
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"

rosdep install --from-paths flight simulation/landing_mpc \
  --ignore-src --rosdistro humble -r -y \
  --skip-keys "px4_msgs ament_python"
colcon build --symlink-install --packages-select \
  path_plan trailer_link landing_mpc
source install/setup.bash

python3 -m pytest \
  flight/path_plan/test \
  flight/trailer_link/test \
  simulation/landing_mpc/test \
  simulation/gazebo/test -q
python3 simulation/gazebo/tools/validate_city_uav_expansion.py
python3 simulation/gazebo/tools/validate_self_contained_maps.py
```

## 주요 파일

- 도심 map: `simulation/gazebo/maps/city_coordinates_uav.yaml`
- 착륙 집중 map: `simulation/gazebo/maps/city_coordinates_uav_quick_landing.yaml`
- 시각화/배속 profile: `simulation/gazebo/profiles/city_landing_{visual,fast}.yaml`
- 두 profile launcher: `simulation/gazebo/run_city_landing.sh`
- 도심 route: `simulation/gazebo/trailer_waypoint_driver.py`
- 도심 planner config: `flight/path_plan/config/city_uav.yaml`
- collision model: `flight/path_plan/path_plan/world_model.py`
- MAVLink emulator: `simulation/gazebo/tools/trailer_mavlink_emulator.py`
- landing state machine: `simulation/landing_mpc/landing_mpc/mission_manager_node.py`
