# PX4 UAV 전역·지역 경로계획 통합 아키텍처

## 1. 문서의 성격

이 문서는 city waypoint 비행과 이동 트레일러 복귀·정밀착륙을 하나의 PX4 native
ROS 2 DDS 제어권 아래 통합하기 위한 설계 계약이다. 현재 저장소의 구현 상태는
`reports/uav_trailer_landing_refactor/baseline_inventory.md`를 따른다.

아래에서 “필수”, “한다”라고 쓴 항목은 최종 구현이 만족해야 하는 규범이며, 현재
코드가 이미 충족한다는 뜻이 아니다. 특히 현재 planner는 2D이고, 현재 moving landing은
MAVROS/ArduPilot legacy 경로이므로 3D 또는 PX4 통합 완료로 간주하지 않는다.

## 2. 임무 계약

최종 임무는 다음 순서를 수행한다.

1. city ENU spawn에서 AGL 10 m까지 수직 이륙한다.
2. 고정 waypoint 목록을 반복 비행한다. 최초 일반 목표는 `(200,-125)`다.
3. static 3D map으로 A* path, 3D SFC, constrained/time-scaled B-spline을 만든다.
4. 전방 depth local map과 jerk-input MPC로 미등록 장애물을 회피한다.
5. map surface preview, down LiDAR, down depth를 융합해 구조물 상단과 약 10 m를
   유지한다.
6. 터미널 `land` 명령은 현재 위치 착륙이 아니라 waypoint 임무 중지와 이동
   트레일러 복귀를 요청한다.
7. 트레일러의 예측 위치로 전역 목표를 제한된 주기로 갱신한다.
8. 근접 후 front ArUco search/acquire, front→down handoff, relative velocity match,
   precision descent를 수행한다.
9. touchdown을 여러 독립 신호로 연속 확인한 뒤에만 정상 disarm한다.

비상 착륙은 `land`와 별도 명령이어야 한다.

## 3. 계층 구조와 단일 제어권

```text
static city map + trailer state + vehicle state
                   │
                   ▼
          Mission Coordinator / FSM
          ├─ Global 3D A*
          ├─ Static 3D SFC
          ├─ Constrained B-spline
          ├─ AGL reference manager
          ├─ Local depth map + jerk MPC
          ├─ Trailer intercept predictor
          ├─ ArUco estimator
          └─ Precision landing controller
                   │ typed p,v,a,yaw reference
                   ▼
        Setpoint authority/mux (한 인스턴스)
                   │ ENU→NED boundary conversion
                   ▼
 /fmu/in/offboard_control_mode
 /fmu/in/trajectory_setpoint
 /fmu/in/vehicle_command
```

핵심 불변식은 다음과 같다.

- 최종 `/fmu/in/trajectory_setpoint` publisher는 정확히 하나다.
- planner, local MPC, landing controller는 최종 PX4 topic을 직접 publish하지 않는다.
- controller 전환 시 직전 reference의 `p,v,a,yaw,yaw_rate`를 다음 controller 초기값으로
  넘긴다.
- 새 controller가 첫 유효 reference를 확정하기 전에는 마지막 안전 reference를 유지한다.
- solver thread와 무관한 heartbeat/setpoint publisher가 20–50 Hz를 유지한다.
- MAVROS/ArduPilot legacy launch는 PX4 Gazebo native launch에 포함하지 않는다.

현재 `mission_node.py`가 native final publisher 하나를 갖는 구조는 유지할 수 있지만,
planner/landing 기능을 이 노드 내부 API 또는 typed reference로 통합해야 한다. legacy
`truck_mission.launch.py`처럼 publisher를 두 개 만들고 boolean gate로 송신 시점만 나누는
방식은 publisher-count 계약을 만족하지 않는다.

## 4. 내부 reference 자료형

모듈 간 reference는 최소 다음 필드를 가져야 한다. 초기 구현에서는 immutable Python
dataclass로 coordinator process 내부에 둘 수 있고, process를 분리할 때 같은 의미의
ROS interface를 정의한다.

```text
TrajectoryReference
  stamp                 # reference가 유효한 ROS time
  valid_until           # stale command 재사용 방지
  source                # GLOBAL, LOCAL_MPC, LANDING, HOLD, FAILSAFE
  frame_id = map_enu
  position_enu_m[3]
  velocity_enu_mps[3]
  acceleration_enu_mps2[3]
  jerk_enu_mps3[3]      # validation/log; PX4 jerk actuator 입력으로 간주하지 않음
  yaw_enu_rad
  yaw_rate_enu_radps
  trajectory_id
  sequence
  safety_status
```

authority는 FSM 상태와 reference freshness가 모두 일치할 때만 선택한다. 선택 규칙은
우선순위를 숨은 숫자로 두지 말고 `(state, allowed_source)` 표로 고정한다. solver 실패 시
이전 command를 무기한 재사용하지 않는다.

## 5. 좌표계 계약

### 5.1 표준 frame

```text
map / Gazebo world : ENU  (x East, y North, z Up)
base_link           : FLU  (x Forward, y Left, z Up)
camera optical      : RDF  (x Right, y Down, z Forward)
PX4 local           : NED  (x North, y East, z Down)
PX4 body            : FRD  (x Forward, y Right, z Down)
```

모든 planner, SFC, trajectory, local map, trailer estimator는 `map_enu`를 사용하고 PX4
topic 직전에만 NED로 바꾼다.

city origin을 `o_E=[x0,y0,z0]`라 하면 위치 변환은 다음과 같다.

```text
p_N = [p_E.y-y0, p_E.x-x0, -(p_E.z-z0)]
p_E = o_E + [p_N.y, p_N.x, -p_N.z]
yaw_N = wrap(pi/2 - yaw_E)
```

vector는 origin translation 없이 축만 바꾼다.

```text
v_N = [v_E.y, v_E.x, -v_E.z]
a_N = [a_E.y, a_E.x, -a_E.z]
j_N = [j_E.y, j_E.x, -j_E.z]
```

FLU↔FRD는 `[x,y,z] -> [x,-y,-z]`다. optical→FLU는 SDF mount rotation과 optical
convention을 합성한 정규화 quaternion/rotation matrix 하나로 제공한다. 카메라별로
부호 수식을 복사하지 않는다.

### 5.2 TF chain

최소 TF chain:

```text
map_enu
├─ vehicle/base_link
│  ├─ vehicle/front_camera_link
│  │  └─ vehicle/front_camera_optical
│  ├─ vehicle/front_depth_optical
│  ├─ vehicle/down_camera_link
│  │  └─ vehicle/down_camera_optical
│  ├─ vehicle/down_depth_optical
│  └─ vehicle/down_lidar_link
└─ trailer/odom
   └─ trailer/base_link
      ├─ trailer/landing_deck
      └─ trailer/aruco_marker
```

SDF `gz_frame_id` 문자열만으로 TF 연결이 생겼다고 가정하지 않는다. launch 후 TF lookup,
축 방향, timestamp를 각각 시험한다.

## 6. 외부 topic 계약

### 6.1 PX4 native DDS

실제 설치된 `px4_msgs`의 suffix를 launch 시 확인한다. 현재 repository의 PX4 v1.17
경로는 local position/status에 `_v1` suffix를 사용한다.

필수 입력:

```text
/fmu/out/vehicle_local_position_v1
/fmu/out/vehicle_status_v1
/fmu/out/vehicle_odometry            # 추가 필요
/fmu/out/vehicle_command_ack
/fmu/out/vehicle_land_detected 또는 설치 버전의 동등 topic
```

유일한 최종 출력:

```text
/fmu/in/offboard_control_mode
/fmu/in/trajectory_setpoint
/fmu/in/vehicle_command
```

### 6.2 sensor

```text
/front_camera/image
/front_camera/camera_info
/front_depth/image
/front_depth/points
/down_camera/image
/down_camera/camera_info
/down_depth/image
/down_depth/points
/down_lidar
/down_lidar/points
```

현재 model/bridge의 depth 이름은 `/depth_camera*`와 `/camera_info`이므로 표준 topic으로
명시적으로 remap하거나 모든 consumer가 하나의 parameter contract를 사용해야 한다.
현재 하방 depth는 없으므로 추가 전에는 해당 기능을 enabled로 보고하지 않는다.

### 6.3 trailer 및 command

```text
/trailer/odometry
/trailer/pose
/trailer/velocity
/mission/command                 # enum을 담은 typed topic/action/service
/mission/state
/mission/events
```

권장 CLI:

```bash
ros2 run autonomy_planner mission_cli land
ros2 run autonomy_planner mission_cli emergency-land
```

`land`는 `RETURN_TO_TRAILER_AND_LAND` enum으로 변환된다. 문자열을 각 노드가 따로
파싱하지 않는다. 명령은 request ID와 수락/거부 응답을 가져 중복 입력을 구분한다.

## 7. 정적 3D map

city YAML의 274 building footprint와 `[foundation_z,roof_z]`를 prism으로 사용한다.
최소 API는 다음과 같다.

```text
is_inside_map(p_xyz)
is_occupied(p_xyz)
is_inflated_occupied(p_xyz)
distance_to_obstacle(p_xyz)
highest_surface_below(x, y, z_limit)
segment_is_free(p0, p1)
```

초기 구현은 2.5D terrain/roof height layer + building prism + 3D voxel inflation이어도
된다. 단, `(x,y,z)` collision query가 실제로 건물 높이를 반영해야 한다.

inflation은 다음 항의 합이며 horizontal/up/down을 분리한다.

```text
r_xy = drone_radius + localization_uncertainty + tracking_margin + safety_margin
r_up = upper_extent + vertical_uncertainty_up + safety_margin_up
r_dn = lower_extent + vertical_uncertainty_down + safety_margin_down
```

map boundary, minimum/maximum altitude, no-fly prism, geofence도 같은 query 또는 명시적
constraint로 검사한다.

## 8. 전역 계획 파이프라인

### 8.1 3D A*

3D voxel `(i,j,k)`, 기본 26-neighbor, diagonal supercover collision check를 쓴다.
edge cost는 길이, clearance, 상승 비용을 분리한다.

```text
c = w_len sqrt(dx²+dy²+(w_z dz)²)
  + w_clear phi(d_clear)
  + w_climb max(dz,0)
```

heuristic은 non-negative 추가 비용을 제외한 weighted Euclidean lower bound를 사용한다.
상세 계약은 `docs/astar_3d_planner_explanation.md`에 정의한다.

### 8.2 3D SFC

각 corridor cell은 convex polyhedron으로 표현한다.

```text
C_i = { p in R^3 | A_i p <= b_i }
```

필수 검증:

- inflated occupancy와 교차하지 않음
- 해당 A* segment와 시작/끝을 포함
- 인접 cell overlap이 drone diameter + tracking margin보다 큼
- corridor gap 없음
- control point/curve segment에 cell assignment 존재
- dense continuous collision post-check 통과

생성 실패 시 segment 분할, margin-aware A* 재계획, 마지막에는 `HOLD_REPLAN`으로 간다.

### 8.3 constrained cubic B-spline

```text
p(t) = sum_i N_i,3(t) Q_i
```

목적함수:

```text
J = w_ref sum ||Q_i-Q_i_ref||²
  + w_a integral ||p''(t)||² dt
  + w_j integral ||p'''(t)||² dt
  + w_T T
  + w_goal ||p(T)-p_goal||²
```

제약:

```text
p(0)=p0, v(0)=v0, a(0)=a0
p(T)=p_goal
v(T)=v_goal       # moving intercept/landing approach일 때 trailer velocity
Q_i in assigned C_i
||v|| <= v_max, ||a|| <= a_max, ||j|| <= j_max
z_min <= z <= z_max, geofence satisfied
```

B-spline derivative control point bound와 dense post-validation을 함께 쓴다. 동역학
제약 위반 시 control point를 장애물 방향으로 미는 대신 knot interval/total duration을
늘리고 다시 푼다.

## 9. 10 m vertical clearance manager

world z=10 고정이 아니라 최고 surface 위 10 m를 목표로 한다.

```text
z_map_ref(s) = highest_surface_below(x(s), y(s)) + h_clear
h_lidar      = filtered down range
h_depth      = z_vehicle - robust_highest_surface_z_in_future_footprint
```

센서 융합은 시간 정렬과 validity gate 후 보수적으로 더 높은 surface를 선택한다.

```text
z_surface_fused = max(z_map_preview, z_lidar_surface, z_down_depth_surface)
z_ref_raw       = z_surface_fused + 10 m
```

reference에는 vertical jerk-limited filter를 적용한다.

```text
|v_z_ref| <= vz_max
|a_z_ref| <= az_max
|j_z_ref| <= jz_max
```

structure edge 전방을 preview하지 못하거나 down sensor가 stale하면 자유공간으로 보지
않고 감속/HOLD한다. precision landing 진입 후에는 10 m manager의 authority를 끄고
landing controller가 deck-relative vertical reference를 소유한다.

## 10. local depth map과 jerk MPC

전방 point cloud 처리:

```text
timestamp/intrinsics validation
→ NaN/Inf/range 제거
→ voxel downsample
→ optical→map TF
→ vehicle self-filter
→ ground/known-static 분리
→ rolling occupancy/ESDF update
→ inflation
→ TTL expiry
```

MPC state와 input:

```text
x = [p, v, a] in map_enu
u = jerk
```

discrete dynamics:

```text
p+ = p + v dt + 0.5 a dt² + (1/6) j dt³
v+ = v + a dt + 0.5 j dt²
a+ = a + j dt
```

목적함수:

```text
J = sum(
      ||p-p_ref||²_Qp + ||v-v_ref||²_Qv + ||a-a_ref||²_Qa
    + ||j||²_Rj + ||j-j_prev||²_Rdj
    + rho ||slack||²
    ) + terminal_cost
```

hard constraint:

```text
||v|| <= v_max
||a|| <= a_max
||j|| <= j_max
distance(p, obstacle) >= local_margin
p inside local convex free corridor and global geofence
```

매 주기 첫 control만 실행한다. obstacle 소멸 뒤 global B-spline으로 reference blending을
통해 재합류한다. infeasible/timeout은 감속→HOLD, 반복 blocked는 global replan 요청으로
간다. heartbeat thread는 optimizer와 분리한다.

## 11. 통합 상태기계 계약

아래 표의 timeout 값은 YAML parameter여야 하며 표의 수치는 정책 예시다. 모든 전환은
`event_id, from, to, reason, source_stamp`를 log한다.

| 상태 | entry action / 제어권 | 필수 입력 | 성공 전환 | timeout·실패 전환 |
|---|---|---|---|---|
| `IDLE` | setpoint 없음, command 대기 | vehicle status | start→`PREFLIGHT` | 잘못된 command 거부 log |
| `PREFLIGHT` | map/frame/sensor freshness, geofence, publisher count 검사 | PX4 status/odom, TF, map, sensors | 모두 정상→`OFFBOARD_PRESTREAM` | 실패→`FAILSAFE_HOLD` 또는 IDLE |
| `OFFBOARD_PRESTREAM` | 현재 p,v,a에서 연속 hold stream | odom/status | 충분한 stream→`ARMING` | stale→`FAILSAFE_HOLD` |
| `ARMING` | OFFBOARD/arm request; 이 상태에서만 자동 arm | status/ack | armed+OFFBOARD→`VERTICAL_TAKEOFF` | reject/timeout→`FAILSAFE_HOLD` |
| `VERTICAL_TAKEOFF` | 현재 xy 고정, AGL 10 m jerk-limited ascent | odom, down range/map | tolerance+dwell→`GLOBAL_PLAN` | sensor stale→`FAILSAFE_HOLD`; obstacle→`ABORT_CLIMB` |
| `GLOBAL_PLAN` | 3D A*/SFC/spline planning; hover authority | 3D map, current state, waypoint | valid trajectory→`GLOBAL_TRACK` | no path/infeasible→`HOLD_REPLAN` |
| `GLOBAL_TRACK` | global spline reference | odom, trajectory, down sensors, depth | waypoint→`WAYPOINT_HOLD`; unknown obstacle→`LOCAL_AVOIDANCE`; land command→`RETURN_REQUESTED` | stale/limit→`FAILSAFE_HOLD` |
| `LOCAL_AVOIDANCE` | local jerk MPC reference | odom, local ESDF, global ref | obstacle clear+dwell→`GLOBAL_TRACK` | infeasible→`HOLD_REPLAN`; imminent collision→`ABORT_CLIMB` |
| `WAYPOINT_HOLD` | continuous p,v=0,a=0 hold, 다음 waypoint 선택 | odom, command | dwell/next→`GLOBAL_PLAN`; land→`RETURN_REQUESTED` | stale→`FAILSAFE_HOLD` |
| `RETURN_REQUESTED` | waypoint mission freeze, trailer freshness 확인 | typed command, trailer odom | fresh→`TRAILER_INTERCEPT_PLAN` | stale→`FAILSAFE_HOLD` |
| `TRAILER_INTERCEPT_PLAN` | predicted target 반복 A*/SFC/spline | trailer estimate, map, UAV state | valid→`TRAILER_INTERCEPT_TRACK` | no path→`HOLD_REPLAN` |
| `TRAILER_INTERCEPT_TRACK` | moving-target global trajectory; bounded replan | trailer/vehicle odom, depth | visual search envelope→`MARKER_SEARCH_FRONT` | trailer stale→`FAILSAFE_HOLD`; blocked→`HOLD_REPLAN` |
| `MARKER_SEARCH_FRONT` | 고도/속도 유지, bounded yaw scan | front RGB, trailer prediction | qualified marker→`MARKER_ACQUIRE_FRONT` | timeout→intercept/search retry or hold |
| `MARKER_ACQUIRE_FRONT` | front pose estimator dwell/quality gate | front pose+quality | stable→`CAMERA_HANDOFF` | dropout→`MARKER_SEARCH_FRONT` |
| `CAMERA_HANDOFF` | front/down estimate overlap consistency 검사 | both cameras, TF | consistent dwell→`MARKER_TRACK_DOWN` | disagreement→`MARKER_ACQUIRE_FRONT` |
| `MARKER_TRACK_DOWN` | down pose estimator active, approach deck | down marker/depth, trailer odom | covariance/visibility gate→`VELOCITY_MATCH` | dropout→search/climb per altitude |
| `VELOCITY_MATCH` | trailer velocity feed-forward + relative MPC | relative state | relative horizontal speed gate→`PRECISION_ALIGN` | stale/outlier→`MARKER_TRACK_DOWN`/hold |
| `PRECISION_ALIGN` | deck altitude 유지, lateral/yaw convergence | down pose, depth/lidar, trailer state | funnel gate+dwell→`PRECISION_DESCENT` | loss→`MARKER_TRACK_DOWN` or `ABORT_CLIMB` |
| `PRECISION_DESCENT` | jerk-limited relative descent | all terminal sensors | low altitude gate→`FINAL_APPROACH` | gate loss→align/hold; long loss→climb |
| `FINAL_APPROACH` | 저속·저jerk deck-relative tracking | range/depth/relative state | contact candidates→`TOUCHDOWN_CONFIRM` | bounce/diverge→`ABORT_CLIMB` if possible |
| `TOUCHDOWN_CONFIRM` | thrust-safe hold; 연속 landed/contact 검증 | PX4 landed, range, vz, accel/contact | dwell 통과→`DISARM` | 조건 해제→`FINAL_APPROACH` |
| `DISARM` | 정상 disarm 1회 요청 및 ack 확인 | touchdown latch, status/ack | disarmed→`IDLE` | reject→hold/log, force-disarm 금지 |
| `HOLD_REPLAN` | 현재 연속 hover, bounded replan/backoff | odom, map/local status | valid plan→호출 상태 | retry limit→`FAILSAFE_HOLD` |
| `ABORT_CLIMB` | horizontal brake + collision-checked vertical escape | odom, up/down clearance | safe→`HOLD_REPLAN` | climb 불가→`EMERGENCY_LAND`/PX4 failsafe |
| `FAILSAFE_HOLD` | 마지막 유효 위치에서 감속/hold | 최소 odom/status | 회복+dwell→명시된 resume state | 지속 stale→`EMERGENCY_LAND` |
| `EMERGENCY_LAND` | PX4 emergency/auto land; trailer return 의미 아님 | PX4 status 가능 시 | disarmed→`IDLE` | FC authority로 이관 |

unexpected disarm 또는 OFFBOARD exit에는 자동 재arm/재진입하지 않는다. 이미 PX4
failsafe가 권한을 가져갔다면 coordinator는 명령 경쟁을 중단한다.

## 12. timing, timestamp, concurrency

권장 rate:

| loop | rate | clock/dt 계약 |
|---|---:|---|
| PX4 heartbeat/setpoint | 20–50 Hz | monotonic scheduler, deadline miss 계측 |
| mission FSM | 20 Hz | message source stamp freshness |
| local map | sensor rate, 최대 30 Hz | sensor stamp + TF stamp |
| jerk MPC | 10–20 Hz | measured solve-start state와 실제 horizon dt |
| static global plan | event-driven | control tick마다 금지 |
| moving intercept replan | 0.5–2 Hz 또는 goal 이동 threshold | hysteresis 적용 |
| ArUco estimator | camera rate | frame timestamp 기반 prediction/update |

ROS time, wall monotonic time, PX4 microsecond timestamp의 용도를 구분한다. simulation
sensor alignment은 ROS `/clock`/message stamp를 사용하고 watchdog/deadline은 wall
monotonic time을 사용한다. fixed timer period를 실제 dt로 간주하지 않는다.

## 13. diagnostics와 기록

최소 기록 필드:

```text
time, mission state/event, selected authority
vehicle/reference p,v,a,j in ENU and NED
setpoint deadline miss and publisher count
A* status/time/expanded/path length/min clearance
SFC cell count/min overlap
B-spline solve/time scaling/max v,a,j
local map age/nearest obstacle
MPC status/iterations/solve time/predicted clearance
map/down-lidar/down-depth surface and fused AGL
trailer p,v,a,predicted intercept, freshness
front/down ArUco ID/pose/reprojection/covariance/dropout
relative p,v,a, landing gates, touchdown evidence
PX4 nav/arming/failsafe/landed and command ack
```

계획 실패와 sensor stale은 자유공간 또는 성공으로 기록하지 않고 명시적 enum reason을
남긴다.

## 14. 검증 게이트

문서 또는 offline plot만으로 다음 단계로 넘어가지 않는다.

1. frame/TF와 single publisher contract
2. 3D map query와 3D A*/SFC/spline 단위시험
3. jerk MPC dynamics/constraint/fallback 시험
4. sensor stale/NaN/TF failure fault injection
5. 정적 city waypoint Gazebo 시험
6. unmodelled obstacle 회피 시험
7. trailer route/odometry 단독 시험
8. terminal `land` return/intercept 시험
9. front→down handoff 및 touchdown 시험
10. 서로 다른 land 시각/noise seed의 최소 10회 회귀시험

각 단계는 collision, geofence, OFFBOARD dropout, in-air disarm, v/a/j limit, min clearance,
solver deadline을 수치로 남긴다. simulator 성공은 실차 aerodynamic, camera latency,
lighting, GNSS/EKF reset, deck contact 검증을 대체하지 않는다.

## 15. 현재 구현과 목표의 경계

현재 구현에서 재사용 가능한 기반:

- PX4 native DDS final publisher와 별도 heartbeat timer
- city ENU↔PX4 NED origin/reset 처리
- 274 footprint YAML parser와 conservative 2D raster
- 2D A* corner-cut prevention과 checked fallback
- native Gazebo depth/down-LiDAR adapter
- calibrated ArUco `solvePnP`의 기초
- selective trailer model/texture

목표 달성을 위해 반드시 추가/교체할 부분:

- 3D occupancy/A*/SFC/constrained time-scaled spline
- down depth/RGB-D와 공통 TF utility
- rolling local map와 jerk-input constrained MPC
- typed `land` command와 moving intercept planner
- timestamp 기반 trailer/marker estimator
- PX4 native precision landing controller와 touchdown gate
- force-disarm 제거, legacy MAVROS profile 격리
- 통합 launch 및 10회 Gazebo validation evidence
