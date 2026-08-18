# PX4-ROS2 — JO 도심 고속환경

`applepark_city_uav` 도심맵에서 9 m/s 트레일러 순찰과
`A* → SFC → B-spline → MPC` 경로계획을 검증하는 ROS 2 / PX4 연구
workspace다. 청주대 운동장 실기 적용 프로파일은 `PX4-ROS2-wang`에서
관리하며, 이 workspace의 운영 대상은 도심맵이다.

## 현재 범위

| 항목 | JO 계약 |
|---|---|
| map | `simulation/gazebo/maps/city_coordinates_uav.yaml` |
| world | `applepark_city_uav` |
| drone spawn | Gazebo ENU `(587, 580, 0)` |
| cruise band | 20–30 m, nominal 25 m |
| drone reference / max speed | 12 / 20 m/s |
| trailer | 23-waypoint stop-turn patrol, 9 m/s |
| trailer acceleration / command rate | 9 m/s² / 50 Hz |
| planning clearance | raw building AABB + 1.5 m |
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
  (현재 one-command 통합은 CJU 회귀 fixture)
- CJU 회귀 MissionManager의 비동기 전체 A*→optimizer SFC→B-spline 재계획,
  path/SFC 원자 교체, LandingMPC runway/coast/저고도 terminal latch
- JO의 기존 `ground_contact` 단방향 래치와 접촉 후 재상승 금지 유지
- JO 트레일러는 `takeoff` clearance에서 출발하는 기존 gate 유지

비활성 CJU fixture는 공통 MissionManager 회귀시험용으로만 Wang과
동기화했다. JO 운영 맵이나 23개 waypoint를 대체하지 않는다.

## 도심 경로계획

```text
city map YAML
  → AStarNode
  → optimizer SFC + cubic B-spline
  → unicycle tracking MPC
  → /path_plan/cmd_vel
```

```bash
cd ~/PX4-ROS2-jo
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
source install/setup.bash
ros2 launch path_plan path_plan.launch.py
```

현재 `/path_plan/cmd_vel`을 PX4 OFFBOARD setpoint로 바꾸는 bridge는 아직
완성되지 않았다. 따라서 이 pipeline을 실기 비행 authority라고 표현하면 안
된다.

도심 world와 9 m/s 순찰만 실행할 때는 기존 launcher를 사용한다.

```bash
START_MAVROS=1 DRIVE_TRAILER=1 TRAILER_SPEED_M_S=9 \
  ./simulation/gazebo/run_px4_map.sh city
```

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
cue는 PX4-local ENU `PointStamped`이고 도심 A* goal은 Gazebo-world ENU
`PoseStamped`다. 두 좌표를 섞거나 단순 remap하지 않는다.

### Gazebo GPS/MAVLink-in-the-loop 실행

다음은 비활성 CJU fixture에서 Gazebo odometry를 MAVLink serial로 변환한 뒤
실제 GPS reader와 adapter를 통과시키는 software wiring 시험이다. JO 도심 GPS
추종 실험이나 RF·RTK 성능시험은 아니다.

```bash
cd ~/PX4-ROS2-jo
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
source install/setup.bash

LANDING_MAP=cju-track TRAILER_CUE_SOURCE=gps TRAILER_LINK=sim GPS_SIM_SEED=4 \
  ./simulation/gazebo/run_gimbal.sh mission
```

### 아직 연결하지 않은 이유

현재 도심 A* 1회는 같은 map에서 약 28 s가 걸릴 수 있다. 9 m/s 표적은 그
동안 약 250 m 이동하므로 Wang의 2초 full-replan 구조를 그대로 연결하는 것은
안전하지 않다. 도심 GPS chase에는 다음이 먼저 필요하다.

1. PX4-local cue→Gazebo-world ENU 변환과 current-start/goal 원자 snapshot
2. 최신 목표 coalescing과 stale-result discard
3. 도심 planner 계산시간을 2초 cadence 안으로 줄인 뒤 비동기 적용
4. `/path_plan/cmd_vel`→PX4 OFFBOARD bridge와 failure HOLD

이 네 조건 전에는 “GPS로 도심 트레일러를 실시간 추종한다”고 주장하지 않는다.

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
- 도심 route: `simulation/gazebo/trailer_waypoint_driver.py`
- 도심 planner config: `flight/path_plan/config/city_uav.yaml`
- collision model: `flight/path_plan/path_plan/world_model.py`
- MAVLink emulator: `simulation/gazebo/tools/trailer_mavlink_emulator.py`
- landing state machine: `simulation/landing_mpc/landing_mpc/mission_manager_node.py`
