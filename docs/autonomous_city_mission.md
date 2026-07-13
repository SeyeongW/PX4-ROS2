# 도시맵 PX4 자율경로 임무

이 임무는 도시맵 ENU 좌표 `(-120, 115)`에서 출발하여 먼저 제자리에서
10 m 수직 이륙한 다음, 목표 `(200, -125)`까지 다음 순서로 비행한다.

```text
도시 YAML 장애물
  -> 점유격자와 안전 팽창
  -> 8방향 A*
  -> LOS 경로 축약
  -> SFC 정적 안전회랑
  -> cubic B-spline
  -> PX4 OFFBOARD + receding-horizon MPC
                     ^
전방 depth 임시회피 --+-- 하방 lidar 10 m 간격 보정
```

구현은 `autonomy_planner` ROS 2 패키지에 있으며 PX4 펌웨어는 수정하지
않는다. 비행 명령과 상태는 PX4 v1.17 `px4_msgs` DDS 토픽을 사용한다.
전방 depth와 하방 lidar는 Gazebo Harmonic Transport에서 직접 읽으므로,
Gazebo 버전이 다른 `ros_gz_bridge`가 설치돼 있어도 이 임무에는 영향을 주지
않는다.

## 좌표계

전역 계획은 Gazebo world ENU에서 수행한다.

- 모델-root 스폰 ENU: `(-120, 115, 0)`
- PX4 로컬 원점(`base_link`) ENU: `(-120, 115, 0.24)`
- 목표 ENU: `(200, -125)`
- 지도상 변위 ENU: `(+320, -240)`
- PX4 local NED 목표: `(-240, +320, -10)`

변환식은 다음과 같다.

```text
north = y_enu - origin_y
east  = x_enu - origin_x
down  = -(z_enu - origin_z)
```

따라서 `(200, -125)`를 PX4 setpoint에 그대로 넣으면 안 된다.
`FrameTransform`만 좌표 변환을 담당하도록 하여 중복 변환을 방지한다.
순정 `x500_base` SDF가 모델 root에서 `base_link`를 +0.24 m에 두기 때문에
PX4 NED 원점은 지면 `z=0`이 아니다. 맵 지표면과 장애물 고도 datum은
계속 ENU `z=0`을 유지한다.

## 상태기계

1. PX4 local position과 두 센서가 유효해질 때까지 대기한다.
2. 현재 위치 setpoint를 2초간 전송하여 PX4 OFFBOARD 입력 조건을 만든다.
3. `VEHICLE_CMD_DO_SET_MODE(OFFBOARD)`와 arm 명령을 보낸다.
4. 시작 XY를 고정하고 NED z를 10 m 감소시켜 수직 이륙한다.
5. 높이 오차 0.4 m 이내를 1.5초 유지하면 전역경로 추종을 시작한다.
6. A*/SFC/B-spline 참조를 MPC로 추종한다.
7. 미등록 전방 장애물을 보면 임시 측방 참조를 만들고 MPC로 우회한 뒤
   원래 B-spline에 재합류한다.
8. 목표 1.5 m 이내에 도착하면 착륙하지 않고 목표점에서 호버한다.

PX4가 failsafe를 보고하면 새 경로 진행을 중단하고 현재 위치를 유지한다.

## A* 핵심 알고리즘

### 1. 점유격자 생성

`city_coordinates.yaml`의 274개 건물 `outer` polygon을 셀과의 실제 교차로
rasterize한다. 단순히 셀 중심이 polygon 안인지 검사하는 방식과 달리, 작은
건물이나 모서리가 셀 사이로 빠지는 것을 막는다. 기본 온라인 격자는 1 m이고
건물 점유 셀을 5 m 팽창시킨다. 이 5 m에는 기체 크기, 위치추종 오차, 안전
여유가 함께 포함된다.

SciPy Euclidean distance transform으로 모든 자유 셀에서 가장 가까운 점유
셀까지의 clearance도 계산한다. 이 값은 경로가 건물 경계에 붙지 않도록 이동
비용과 SFC 폭 계산에 사용된다.

### 2. 이웃과 이동비용

각 셀에서 상하좌우와 대각선, 총 8개 이웃을 검사한다.

```text
직선 이동 기본비용 = resolution
대각 이동 기본비용 = sqrt(2) * resolution
```

대각 이동에서는 맞닿은 두 직교 셀 중 하나라도 점유되어 있으면 이동을
금지한다. 이것이 `corner cutting` 방지 조건이다. 이 조건이 없으면 경로의
선분이 두 건물 모서리 사이를 수학적으로 가로질러 실제 기체가 충돌할 수 있다.

실제 누적 비용은 다음 형태다.

```text
g_new = g_current
      + move_distance * (1 + clearance_weight / max(clearance, resolution))
```

따라서 같은 길이라면 장애물에서 더 먼 셀이 선택된다. `clearance_weight=0`이면
순수 최단거리 A*가 된다.

### 3. 휴리스틱과 optimality

휴리스틱은 현재 셀 중심에서 목표 셀 중심까지의 Euclidean 거리다.

```text
h(n) = sqrt((x_goal-x_n)^2 + (y_goal-y_n)^2)
f(n) = g(n) + h(n)
```

모든 edge 비용은 실제 이동거리 이상이므로 Euclidean `h`는 남은 실제 비용을
과대평가하지 않는 admissible heuristic이다. 우선순위 큐에서 가장 작은 `f`를
가진 노드를 꺼내며, goal이 pop될 때 현재 비용 정의에 대한 최적 경로가
확정된다. `came_from`을 goal부터 start까지 역추적해 좌표 경로를 만든다.

핵심 의사코드는 다음과 같다.

```python
open_heap = [(h(start), 0, start)]
g[start] = 0

while open_heap:
    _, current_g, current = heappop(open_heap)
    if current == goal:
        return reconstruct(came_from)

    for neighbor, move_distance in valid_8_neighbors(current):
        if diagonal_corner_is_blocked(current, neighbor):
            continue
        clearance = distance_to_nearest_obstacle(neighbor)
        edge = move_distance * (1 + weight / max(clearance, resolution))
        candidate = current_g + edge
        if candidate < g.get(neighbor, infinity):
            came_from[neighbor] = current
            g[neighbor] = candidate
            heappush(open_heap, (candidate + h(neighbor), candidate, neighbor))
```

현재 도시 임무 기본값의 오프라인 결과는 다음과 같다.

- A* grid path: 565점
- LOS 축약 중심선: 3점
- 검증된 cubic B-spline: 1,126점, 0.5 m 간격
- 경로 길이: 약 562.29 m
- 목표 PX4 local NED: `(-240, 320, -10)`

## SFC 정적 안전회랑

A*의 모든 계단형 셀을 그대로 추종하지 않고, 먼저 현재 점에서 충돌 없이
보이는 가장 먼 점을 반복 선택하는 line-of-sight 축약을 한다. 축약된 각 선분
주변에 장애물 clearance보다 작은 convex capsule을 만든다. 연속 capsule의
합집합을 10 m 비행층으로 extrusion한 것이 이 구현의 정적 SFC다.

각 capsule의 반경은 중심선 전체에서 측정한 최소 clearance에서 셀 반대각선
오차를 뺀 값이다. capsule 내부를 다시 촘촘히 점유격자와 대조하므로 회랑이
건물 팽창 영역에 닿으면 반경을 줄이거나 계획을 실패시킨다.

## B-spline

LOS 중심선을 cubic B-spline control path로 사용한다. SciPy `splprep`로 곡선을
만든 뒤 다음 조건을 모두 만족하는 경우에만 채택한다.

- 모든 dense sample이 팽창 점유격자의 자유 셀이다.
- sample 사이의 선분도 모두 충돌이 없다.
- 모든 sample이 적어도 하나의 SFC capsule 안에 있다.
- 시작점과 목표점이 원래 좌표와 정확히 같다.

평활화 강도를 차례로 낮춰 재시도하며, 안전성 검사를 하나라도 통과하지 못하면
정적 회랑 중심선으로 되돌아간다. 즉 곡선을 얻기 위해 충돌 안전성을 희생하지
않는다. 현재 `(200,-125)` 임무에서는 cubic B-spline이 정상 채택된다.

## Depth 지역회피와 MPC

전방 OAK-D depth point cloud에서 기체 진행 envelope 안의 3-percentile 거리를
사용한다. 단일 이상 픽셀에는 반응하지 않으면서 실제 장애물에는 빠르게
반응하기 위한 값이다.

- 10 m 안: point가 적은 좌/우 방향 중 정적 지도에서도 자유인 방향으로 7 m
  임시 측방 참조를 생성한다.
- 4 m 안: 수평 진행을 정지하고 3 m 상승 참조를 만든다.
- 장애물이 사라지면 2초 TTL 후 offset을 감쇠시키며 기존 B-spline에 합류한다.

MPC 모델은 3축 double integrator다.

```text
p[k+1] = p[k] + dt*v[k] + 0.5*dt^2*a[k]
v[k+1] = v[k] + dt*a[k]
```

8단계, `dt=0.2 s` horizon에서 다음 비용을 최소화한다.

```text
J = 8.0 * sum(||p-p_ref||^2)
  + 2.0 * sum(||v-v_ref||^2)
  + 0.15 * sum(||a||^2)
  + 0.10 * sum(||a[k]-a[k-1]||^2)
```

각 축 가속도는 `±3 m/s²`로 제한하며 L-BFGS-B로 매 제어 주기 재계산한다.
최적화가 실패해도 setpoint가 끊기지 않도록 동일 제한을 갖는 bounded PD가
fallback으로 동작한다. 위치·속도·가속도 참조는 PX4 `TrajectorySetpoint`에
feed-forward로 전달된다.

## 하방 10 m 간격

하방 lidar는 50 Hz, 0.1~100 m 범위의 단일 ray다. 현재 world 고도에서 측정
거리를 빼 바로 아래 표면 고도를 추정한다.

```text
surface_z = current_world_z - lidar_range
altitude_ref = surface_z + 10 m
```

고도 참조는 `2 m/s` 이하로 slew-limit한다. 라이다 거리가 4 m보다 작아지면
즉시 상승 참조가 우선한다. 도시의 등록 건물은 A*가 XY로 우회하므로 이 센서는
미등록 구조물이나 바로 아래 표면에 대한 보조 안전장치다. 단일 ray로 구조물의
면적과 가장자리를 알 수는 없으므로, 하부 전체 형상 인식이 필요하면 별도의
하방 depth camera가 추가되어야 한다.

## 실행

새 PC에서는 한 번만 다음을 실행한다.

```bash
cd ~/PX4-ROS2
./gazebo/setup_px4_sitl.sh
./gazebo/setup_autonomy_deps.sh
```

비행 없이 경로만 검사하고 그림을 생성하려면:

```bash
./gazebo/run_autonomous_city_mission.sh --plan-only
# ~/city_astar_sfc_bspline.png 생성
```

PX4와 도시맵을 실행하여 실제 임무를 시작하려면:

```bash
./gazebo/run_autonomous_city_mission.sh
```

RTX가 없는 환경이나 CI에서는 다음처럼 서버 전용으로 실행할 수 있다.

```bash
HEADLESS=1 USE_NVIDIA=0 ./gazebo/run_autonomous_city_mission.sh
```

목표에 도착한 뒤에도 프로세스는 호버를 유지한다. `Ctrl-C`를 누르면 임무,
PX4, Gazebo, XRCE Agent를 같은 실행 세션 범위에서 종료한다.
