# PX4 내장 L1-style / B-spline 교차검증

## 결론과 적용 범위

X500 멀티콥터에서 말하는 이 실험의 `L1`은 고정익의 구형 L1 controller가 아니다.
PX4 v1.17 멀티콥터 `AUTO.MISSION`이 사용하는
`FlightTaskAuto -> PositionSmoothing::_getL1Point()`의 **L1-style waypoint crossing**이다.
B-spline/SFC/MPC 코드는 삭제하지 않으며, 같은 3D A*/SFC 결과를 아래 두 분기로 나눠 A/B
비교한다.

```text
동일 map/occupancy + 동일 3D A*/SFC centerline
              |
              +-- A: constrained cubic B-spline + local MPC -> ROS 2 offboard
              |
              +-- B: checked waypoint triplets -> MAVLink mission -> PX4 AUTO.MISSION
```

두 분기를 동시에 실행하면 안 된다. L1 adapter는 `/fmu/in/trajectory_setpoint`를 publish하지
않으며, 실행 전에 해당 topic의 publisher 수가 0인지 확인한다. MAVROS, city autonomy node,
다른 mission node가 보이면 fail-closed하고, 프로세스 authority lock도 별도로 잡는다.

L1 분기는 정적 A*/SFC 경로 추종 비교용이다. Depth 기반 동적 장애물 회피, jerk-input MPC,
움직이는 트레일러 인지/정밀착륙은 B-spline/MPC 분기의 책임으로 남긴다. 따라서 L1 실험 중
기존 `mission_cli land`를 동시에 연결하면 안 된다. L1 실험을 종료하고 B-spline/MPC
controller에 명시적으로 authority를 넘긴 뒤 트레일러 복귀 시퀀스를 시험한다.

## PX4 v1.17 소스 감사 근거

감사한 PX4는 `/home/xogus/PX4-Autopilot`의 `v1.17.0`, commit
`d6f12ad1c4f70ad3230afd7d86e971421e02fef4`이다.

- `src/modules/flight_mode_manager/tasks/Auto/FlightTaskAuto.cpp:162-181`:
  `_prev_wp`, 현재 position setpoint, `_next_wp`를 3점 배열로 만들어
  `PositionSmoothing.generateSetpoints()`에 넣는다.
- `src/lib/motion_planning/PositionSmoothing.cpp:65-78`: 속도와 target 방향 차이가 약 10도
  이상(`cos_align < 0.98`)일 때 turning으로 판정한다.
- `src/lib/motion_planning/PositionSmoothing.cpp:142-175`: turning이면 `_getL1Point()`를
  호출하고 `max(target_acceptance_radius, 5 m)` look-ahead를 이전 waypoint-current
  waypoint 선상에 만든다.
- `src/modules/mavlink/mavlink_mission.cpp:1395-1433`: 이 PX4 버전의 mission position은
  global/global-relative frame만 파싱한다. `MAV_FRAME_LOCAL_NED` mission item은 지원하지
  않는다.
- `src/modules/commander/Commander.cpp:1127-1145`: `MAV_CMD_MISSION_START` 수락 시
  `AUTO_MISSION`으로 바꾸고 arm한다. 그래서 실행에는 별도의 강한 확인 문자열이 필요하다.
- 고정익은 `src/modules/fw_mode_manager`와 `src/lib/npfg`의 NPFG를 사용한다. 이 X500
  결과를 고정익 L1 결과라고 표현하면 안 된다.

## 안전 계약

`px4_l1_export.py`는 UAV 도시 좌표 YAML을 읽어 다음을 자동 수행한다.

1. 최소 1 m x 1 m yaw-invariant UAV envelope와 YAML의 분해된 hard radius를 사용한다.
2. 26-neighbor hierarchical 3D A*와 3D SFC를 실행한다.
3. 모든 segment를 0.25 m 이하 간격으로 collision check한다.
4. PX4가 실제로 turning으로 판정할 수 있는 waypoint triplet은 단일 convex SFC box의
   **수평 5 m inset** 안에 세 점이 모두 들어가는지 증명한다. 이 값은 along-track
   look-ahead이므로 가상의 수직 5 m 반경으로 적용하지 않는다.
5. 직선 triplet에는 5 m를 장애물 hard inflation으로 잘못 적용하지 않는다. 5 m는 firmware
   look-ahead이지 기체 반경이 아니기 때문이다.
6. 지면/건물 표면 위 하방 10 m clearance를 3-D occupancy의 수직 팽창으로 강제한다.
7. map hash, occupancy hash, planner run ID, hard radius, surface clearance, corridor,
   triplet certificate를 YAML에
   기록하고 전체 문서 SHA-256을 붙인 뒤 consumer로 round-trip 검증한다.

5 m inset 증명이 안 되면 `L1_SFC_CERTIFICATE_INFEASIBLE`로 L1 분기만 거부한다. 이때
B-spline 분기의 정상 SFC를 축소하거나 안전 조건을 낮추지 않는다.

MAVLink mission은 Gazebo-world ENU를 그대로 보낼 수 없으므로 다음 경계를 사용한다.

- plan의 `world_home_enu_m`: PX4가 부팅해 home을 잡은 Gazebo spawn 좌표
- 실시간 `HOME_POSITION`: WGS-84 latitude/longitude/AMSL
- world waypoint와 world home의 East/North 차이: WGS-84 local tangent 변환
- world Z 차이: `MAV_FRAME_GLOBAL_RELATIVE_ALT_INT`의 relative altitude

업로드 전에 첫 waypoint의 XY가 world home에서 2 m 이내인지, vehicle이 disarmed인지,
PX4 quadrotor heartbeat인지, local NED origin 오차가 2 m 이내인지 확인한다. 업로드 뒤에는
mission 전체를 다시 download하여 좌표를 대조한다. 실행 뒤에는 armed `AUTO.MISSION`
heartbeat와 유효한 `MISSION_CURRENT`를 모두 받아야 성공으로 본다.

## fresh clone 실행 순서

먼저 패키지를 빌드하고 PX4/Gazebo SITL을 **MAVROS 없이** 실행한다. 같은 시점에 기존
B-spline autonomy node도 실행하지 않는다.

```bash
cd ~/PX4-ROS2
./gazebo/setup_autonomy_deps.sh
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
colcon build --symlink-install --packages-select autonomy_planner
source install/setup.bash
python3 -c 'from pymavlink import mavutil'
```

공통 3D A*/SFC plan을 생성한다. 요청 시작/목표 center z는 11.5 m이며, planner는
하방 10 m surface-clearance와 기체 수직 envelope를 만족하는 3-D voxel 중심으로 위로
snap할 수 있다. 목표 `(200, -128)`까지 간 다음 같은 안전 경로로 돌아오는 finite
patrol이다.

```bash
ros2 run autonomy_planner export_px4_l1_plan \
  --repository-root "$HOME/PX4-ROS2" \
  --start -120 115 11.5 \
  --goal 200 -128 11.5 \
  --world-home -120 115 0.24 \
  --surface-clearance-m 10 \
  --output /tmp/px4_l1_city_plan.yaml

ros2 run autonomy_planner px4_l1_mission validate \
  --plan /tmp/px4_l1_city_plan.yaml
```

출력된 plan의 두 식별자를 읽는다.

```bash
PLAN_SHA=$(python3 -c 'import yaml; print(yaml.safe_load(open("/tmp/px4_l1_city_plan.yaml"))["plan_sha256"])')
MAP_REV=$(python3 -c 'import yaml; print(yaml.safe_load(open("/tmp/px4_l1_city_plan.yaml"))["map_revision"])')
```

arm하지 않고 mission protocol과 readback만 먼저 검증한다. PX4 SITL의 GCS MAVLink가 다른
port이면 `--connection`만 바꾼다. 기본값은 `udpin:0.0.0.0:14550`이다.

```bash
ros2 run autonomy_planner px4_l1_mission upload \
  --plan /tmp/px4_l1_city_plan.yaml \
  --expect-map-revision "$MAP_REV" \
  --expect-plan-sha256 "$PLAN_SHA" \
  --authorize-plan-replacement REPLACE_PX4_MISSION
```

실행 직전 exclusive 조건을 눈으로도 확인한다.

```bash
ros2 topic info /fmu/in/trajectory_setpoint --verbose
ros2 node list
```

`Publisher count: 0`이고 MAVROS/city autonomy mission controller가 없을 때만 실행한다.
아래 `MISSION_START`는 PX4를 arm하므로 실제 회전부·프로펠러 안전을 확보한 뒤 사용한다.

```bash
ros2 run autonomy_planner px4_l1_mission execute \
  --plan /tmp/px4_l1_city_plan.yaml \
  --patrol-cycles 1 \
  --expect-map-revision "$MAP_REV" \
  --expect-plan-sha256 "$PLAN_SHA" \
  --authorize-plan-replacement REPLACE_PX4_MISSION \
  --execute-auto-mission AUTO_MISSION_WILL_ARM
```

## A/B metric 계약

`l1_metrics.py`는 두 방식에 동일한 interface를 적용한다. 비교 전에 다음 값이 완전히 같아야
한다.

- map revision 및 occupancy SHA-256
- 공통 A*/SFC centerline route SHA-256 및 planner run ID
- PX4 firmware git revision
- Gazebo scenario seed, wind profile, payload profile
- hard radius, minimum surface clearance 및 초기 ENU 위치(0.25 m 이내)

두 실행의 world-ENU timestamp/position/velocity와 동일 occupancy에서 얻은 clearance trace로
다음을 계산한다.

- 성공 여부, duration, reference/flown path length, final error
- RMS/p95/max cross-track error와 backtracking distance
- minimum clearance와 hard-clearance violation sample 수
- max/p95 speed, acceleration, jerk

모든 paired trial은 먼저 collision-free와 hard-clearance violation 0을 만족해야 한다. 그 뒤
경로 길이, 시간, cross-track, acceleration/jerk를 비교한다. 안전 위반이 있는 더 빠른 경로는
최적 결과로 채택하지 않는다.

## 2026-07-13 로컬 smoke 결과

실제 `city_coordinates_uav.yaml`에 위 exporter를 실행한 결과는 다음과 같다.

- map revision: `city_uav:dc90d7d81edee111`
- hard radius: 1.450 m
- minimum surface clearance: 10.0 m (기체 vertical envelope 포함 occupancy 적용)
- 요청 z: 11.5 m, 검증된 SFC centerline z: 15.0 m
- common SFC centerline / PX4 waypoint 수: 195
- out-and-back 1 cycle mission item 수: 389 (첫 item은 TAKEOFF)
- exporter 결과를 consumer로 다시 validate: 통과
- ENU/global 변환, plan tamper 검출, proof flag, 5 m turning certificate, authority guard,
  MAVLink upload/readback/start mock, A/B metric 계약 집중 테스트: 24 passed

이 smoke는 exporter와 protocol contract 검증이다. 실제 Gazebo에서 arm/비행한 성공 횟수로
간주하면 안 된다. Gazebo 통합 trial은 mode, `MISSION_CURRENT`, vehicle local position,
clearance trace와 함께 별도 결과 파일로 남겨야 한다.
