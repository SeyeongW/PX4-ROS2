# PX4-ROS2 — CJU 이동 트레일러 자율 미션 (`jo`)

청주대학교 종합운동장 Gazebo 환경에서 장애물을 피해 목표점까지 비행한 뒤,
움직이는 트레일러를 실시간으로 재계획·추종하고 PX4 정밀착륙으로 인계하는
ROS 2 / PX4 연구 스택이다.

> **현재 판정 (2026-08-17 KST)**
>
> - **Gazebo SITL:** 전체 미션 `DONE`, PX4 자동 착륙·자동 disarm 확인
> - **실기체 props-on:** **NO-GO**
> - ROS 2의 분리된 MPC가 B-spline 추종과 ArUco 정렬·저고도 하강을 담당하고,
>   PX4는 저수준 비행제어와 최종 PRECLAND 접촉판정·자동 disarm을 담당한다.
> - 현재 성공한 이동표적 PRECLAND는 이 저장소 밖의 수정된 PX4 소스에
>   의존한다. `jo`만 clone해서는 같은 착륙이 재현되지 않는다.

## 현재 상태 요약

| 항목 | 현재 상태 |
|---|---|
| 운영 맵 | [`drone_cju_track.yaml`](simulation/gazebo/maps/drone_cju_track.yaml) |
| 출발 / 목표 | 맵 `(5, 0)` / `(50, 50)`, 순항고도 5 m |
| 장애물 | `[0,50] × [0,50]` 범위의 seed 5053 기반 정수 중심좌표 20개, `barrier_2` 수동 조정 |
| 트레일러 | 맵 `(5, 0) ↔ (5, 50)`, 1 m/s 왕복 |
| 출항 경로 | A* → geometry-only B-spline → TrackingMPC |
| 복귀 경로 | 최초 A*/B-spline 후 2초마다 검증된 tail만 최신 cue로 교체 |
| 착륙 인계 | 짐벌 수평 GPS/cue 거리 10→9 m 연속 조준, KF 승인 ArUco 3개/0.5 s, live cue 직선 1.5 m-safe → LandingMPC 정렬·하강 → 저고도 PX4 PRECLAND |
| 실시간 UI | 승인 경로·실제 기체·트레일러·미션 상태를 읽기 전용 표시 |
| 현재 브랜치 | `jo` |

### 현재 1 m + 분리 MPC Gazebo 검증

현재 hard clearance 1 m, TrackingMPC, LandingMPC, land 전 짐벌 무명령과
ArUco 3개/0.5 s gate는 `20260817T133221Z.wgkdQS` GUI 전체 실행으로 확인했다.
PRECLAND는 첫 시도에 완료됐고 PX4가 자동 무장해제했다. artifact의 map
SHA-256은 실행 YAML과 일치하지만 작업트리가 dirty였으므로 커밋 단위 재현
증거는 아니다.

| 지표 | 결과 |
|---|---:|
| 상태 전이 | `TAKEOFF → READY → MISSION → HOVER → RETURN → LANDING_ACQUIRE → LANDING_DESCEND → PRECLAND → DONE` |
| 전체 A*/B-spline | 2회 (출항 1회 + 최초 RETURN 1회) |
| 2초 tail retarget | 8회 승인 / 0회 거절 |
| TrackingMPC / LandingMPC reject | 0 / 0 |
| planner failure / ABORT / PX4 failsafe | 0 / 0 / 0 |
| PRECLAND 시도 / 복귀 | 1 / 0 |
| 경로 추종 RMSE / 최대오차 | 0.223 / 0.321 m |
| MPC 평균 / 최대 계산시간 | 4.663 / 38.860 ms |
| 최대 body-rate | 43.47 deg/s (90 deg/s 경고선 미만) |
| LandingMPC 구간의 잘못된 position-only HOLD | 0회 |
| Gazebo ground-truth 연속 최소거리 | 1.619 m (hard 1 m 바깥 0.619 m) |
| 실제 데크 접촉 중심오차 / GT 하강속도 | 0.131 m / 0.101 m/s |
| PX4 착륙판정 / 자동 disarm | 성공 / 성공 |
| 관련 회귀시험 | 134 passed (path_plan + landing_mpc + exporter) |
| quality v3 | `WARN (dirty_tree)` |

이 실행에서 다미터 추월·재접근, PRECLAND 재획득, 접촉 뒤 재상승은 재현되지
않았다. land 전에는 실제 gimbal command가 0건이었고 encoder도 초기 상태를
유지했다. land 후에는 현재 실측각부터 약 1.2 s 동안 nadir로 이동했다. 다만
10→9 m 전환 aim lag가 최대 56.3°였고, 현재 runway 요구 10.5 s는
실측 PRECLAND→disarm 11.51 s보다 짧다. 또한 CSV의 기본 touchdown 속도는 실제
접촉보다 약 4.68 s 늦은 PX4 `ground_contact` 기준이므로 물리 접촉 지표와
구분해야 한다.

### 변경 전 2 m Gazebo 검증

아래 전체 실행은 당시 seed 5053 맵과 50 Hz cue, 2초 tail retarget을 반영한
`20260812T170141Z.dyaF0u`이다. 실행 시각은 2026-08-13 02:01 KST이며,
실행 당시 YAML과 artifact의 `map.yaml` SHA-256이 일치했다. 현재 YAML은
안전거리를 1 m로 바꾼 후이므로 hash가 다르다. 또한 이 실행은 커밋 전 dirty
worktree에서 수행했으므로 아래 결과는 이번 커밋 자체의 clean 재현 증거는 아니다.

| 지표 | 결과 |
|---|---:|
| 상태 전이 | `TAKEOFF → READY → MISSION → HOVER → RETURN → PRECLAND → DONE` |
| 전체 A*/B-spline | 2회 (출항 1회 + 최초 RETURN 1회) |
| 2초 tail retarget | 22회 승인 / 0회 거절 |
| planner failure / ABORT | 0 / 0 |
| PX4 failsafe / ULog dropout | 0 / 0 |
| tail 교체 간격 | 1.978–2.026 s (중앙값 2.003 s) |
| tail endpoint와 동시 트레일러 위치 최대 오차 | 0.066 m |
| 최대 수평속도 / 인접 속도 변화 | 3.224 m/s / 0.030 m/s |
| 최대 수평가속도 / body-rate | 3.466 m/s² / 44.969 deg/s |
| estimator 기준 장애물 표면 최소거리 | 2.260 m (2 m 바깥 0.260 m) |
| Gazebo ground-truth 연속 검산 최소거리 | 2.086 m (2 m 바깥 0.086 m) |
| PX4 착륙판정 / 자동 disarm | 성공 / 성공 |
| quality v3 | `WARN (low_obstacle_reserve, dirty_tree)` |
| 전체 Python 회귀시험 | 129 passed |

quality v3는 실행 실패, failsafe, sample gap, speed jump, ULog dropout,
planner failure, ABORT와 장애물 침범을 `FAIL`로 판정한다. 이번 실행에는 해당
실패가 없었지만 estimator 기준 추가 여유가 0.5 m보다 작고 작업트리가 dirty여서
`WARN`이다. 위 ground-truth 값은 원시 Gazebo 궤적의 연속 선분을 별도로 검산한
값이며, 현재 exporter 값보다 실제 여유가 작다. 이 실행은 변경 전 2 m 기준의
기록이므로 현재 1 m 설정의 근거로 재사용하지 않는다. 이 판정은 실기 안전
인증이 아니며 저장소 밖 PX4 dirty patch도 아직 재현 가능한 형태로 고정되지
않았다. 현재 1 m 설정의 검증은 바로 위 `wgkdQS` 결과를 기준으로 한다.

## 2026-08-12 13:00 KST 이후 변경

이 시각 이후 변경은 이번 `jo` 커밋 전까지 dirty worktree에서 개발·검증했다.

- 장애물 생성 범위를 `[0,50] × [0,50]`으로 넓히고 seed 5053의 정수 좌표
  20개로 재배치했다. `barrier_2`는 수동으로 조정했으며 Gazebo world와 YAML,
  테스트의 좌표를 동기화했다.
- 장애물 안전거리를 축별 사각 팽창이 아니라 물리 AABB 표면에서의 정확한
  XY 유클리드 반경으로 통일했다. 현재 hard/runtime/UI/report 기준은 1.0 m이고,
  A*/B-spline 및 tail 연결은 추종오차 여유 0.5 m를 더한 1.5 m로 계획한다.
- 명령 순서를 `takeoff → READY → mission → HOVER → land`로 분리했다.
  수동 모드에서는 이륙 직후 미션이 자동으로 시작되지 않는다.
- 트레일러 cue 발행률을 YAML의 50 Hz 설정과 launcher에 맞췄다.
- RETURN 전체 A*/B-spline은 최초 1회만 계산한다. 이후 2초마다 기존 경로의
  진행 가능한 앞부분을 유지하고, 최신 트레일러까지 1.5 m-safe인 tail만
  검증 후 원자적으로 교체한다. 연결할 수 없을 때만 전체 planner를 실행하며,
  계산·실패 중에는 기존 승인 경로를 계속 추종한다.
- YAML/live UI의 안전영역을 같은 rounded 1 m 형상으로 표시하고, RETURN
  경로는 누적 이력 대신 현재 활성 경로 하나만 보여준다.
- wang 계열의 TrackingMPC와 LandingMPC를 서로 다른 제어기로 미션 매니저
  내부에 연결했다. TrackingMPC는 B-spline 공간경로를 추종하고 LandingMPC는
  fresh ArUco 관측 이후 고정고도 정렬과 하강만 담당한다. 저고도 접촉과 자동
  disarm은 PX4 PRECLAND가 계속 담당한다.
- 근거 없는 6 m 착륙 인계와 60° 각도 gate를 제거했다. `land` 전에는 짐벌에
  위치 명령을 보내지 않고 현재 관절 상태를 그대로 둔다. `land` 후 수평
  GPS/cue 거리 10 m 밖에서는 현재 측정각부터 속도 제한을 적용해 관절 yaw
  0°/pitch -90°로 이동·유지하고, 10→9 m에서 연속 전환한 뒤 9 m 안에서
  트레일러를 직접 조준한다. 착륙·자동 무장해제가 확인되어 미션이 `DONE`이
  되면 다시 위치 명령을 끊고 마지막 관절 상태를 그대로 둔다.
  서로 다른 KF 승인 ArUco 3개가 0.5 s 안에 들어오고 live segment가 1.5 m
  planning-safe일 때만 착륙 MPC에 진입한다.
- Gazebo cue 원본 시각이 `/clock`보다 한 50 Hz tick 앞서는 callback 순서차를
  허용해 이동 중 position-only HOLD가 끼는 문제를 제거했다. TrackingMPC에는
  최적화 horizon과 50 Hz 실제 송신값 모두에 축별 2 m/s³ acceleration-slew
  제한을 적용했다. `T7a5e5`의 raw setpoint bag에서 축별 최대 2.00002 m/s³
  이내를 확인했다. XY 합성 jerk/가속도 상한은 별도 실기체 계약 항목이다.
- MAVROS parameter service의 첫 응답 discovery race를 제한시간 내 재시도로
  처리하고, exporter의 장애물 residual도 유클리드 1 m 계약으로 맞췄다.
- `t5Jwr1`에서 전체 planner 2회, tail 교체 12회, 두 MPC reject 0, `DONE`,
  자동 disarm과 planner failure·ABORT·failsafe·dropout 0을 확인했다.

## 빠른 실행 — 현재 개발 PC

### 1. 빌드

현재 launcher는 ROS 2 Humble, PX4 v1.17에 맞는 `px4_msgs` workspace와
`MicroXRCEAgent`가 이미 설치되어 있다고 가정한다.

~~~bash
cd /path/to/PX4-ROS2
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
colcon build --symlink-install
source install/setup.bash
~~~

### 2. YAML 프리뷰만 실행

다음 명령은 YAML을 읽어 장애물·안전영역·예상 경로를 표시할 뿐,
Gazebo·PX4·기체 제어를 시작하지 않는다.

~~~bash
cd /path/to/PX4-ROS2
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"

# 창으로 정적/애니메이션 프리뷰
python3 simulation/gazebo/tools/cju_mission_ui.py

# 창 없이 YAML·planner·좌표 계약 검사
python3 simulation/gazebo/tools/cju_mission_ui.py --check

# 이미지로 저장
python3 simulation/gazebo/tools/cju_mission_ui.py --save /tmp/cju_path.png
~~~

`--live`는 YAML-only 모드가 아니라 이미 실행 중인 ROS graph를 구독하는
옵션이다.

### 3. Gazebo 전체 미션 실험

~~~bash
cd /path/to/PX4-ROS2
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
source install/setup.bash
./simulation/gazebo/run_gimbal.sh mission
~~~

터미널 입력 순서는 다음과 같다.

1. `명령>` 프롬프트에서 `takeoff` 입력 후 `READY` 확인
2. `mission` 입력 후 `(50,50)` `HOVER` 확인
3. `land` 입력
4. `DONE`과 PX4 자동 disarm 확인
5. `Ctrl-C`로 launcher를 종료해 ULog·CSV·manifest 정리 완료

GUI 실행에서는 live mission map이 자동으로 열린다.

~~~bash
# Gazebo server만 실행하고 모든 viewer 비활성화
HEADLESS=1 ./simulation/gazebo/run_gimbal.sh mission

# live mission map만 비활성화
MISSION_VIEW=0 ./simulation/gazebo/run_gimbal.sh mission

# ArUco viewer와 mission map 모두 비활성화
ARUCO_VIEW=0 MISSION_VIEW=0 ./simulation/gazebo/run_gimbal.sh mission
~~~

비교용 실행은 다음 두 개만 유지한다.

~~~bash
./simulation/gazebo/run_gimbal.sh gimbal
./simulation/gazebo/run_gimbal.sh baseline
~~~

## 최초 환경 구축

~~~bash
git clone --branch jo --single-branch https://github.com/SeyeongW/PX4-ROS2.git
cd PX4-ROS2

sudo bash simulation/gazebo/install_apt_deps.sh
./simulation/gazebo/setup_px4_sitl.sh
./simulation/gazebo/link_px4_model.sh

source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
colcon build --symlink-install
source install/setup.bash
~~~

다만 이 명령만으로 현재 결과를 완전히 복원할 수는 없다.

- `$HOME/px4_ros2_ws`의 PX4 v1.17 호환 `px4_msgs`가 별도로 필요하다.
- `Micro-XRCE-DDS-Agent v2.4.3` 설치 자동화가 이 저장소에 없다.
- `setup_px4_sitl.sh`는 stock PX4 v1.17.0을 준비하며, 아래의 이동표적
  PRECLAND 패치를 적용하지 않는다.
- 현재 PC의 수정된 PX4 checkout과 실기체 firmware는 이 저장소에 포함되지 않는다.

따라서 fresh PC에서는 build와 일반 SITL 점검까지만 기대해야 하며, 수정된
PRECLAND를 별도로 고정하기 전에는 최신 이동 트레일러 착륙을 재현 완료로
판정하면 안 된다.

## 작동 원리

기본 원칙은 **ROS 2가 안전한 공간 경로와 두 MPC의 단계 전환을 정하고,
PX4가 저수준 비행제어와 최종 접촉판정·자동 disarm을 소유하는 것**이다.

~~~mermaid
flowchart LR
    A[Phase 0<br/>PRECHECK] --> B[Phase 1<br/>PX4 NAV_TAKEOFF]
    B --> R[READY<br/>mission 명령 대기]
    R --> C[A* topology]
    C --> D[Geometry-only B-spline]
    D --> E[Phase 2<br/>TrackingMPC]
    E --> F[HOVER at 50,50]
    F --> G[Phase 3<br/>Rolling RETURN replan]
    G --> H[3 KF-accepted ArUco fixes / 0.5 s<br/>1.5 m-safe segment]
    H --> I[LandingMPC<br/>ACQUIRE then DESCEND]
    I --> J[Low-alt PX4 NAV_PRECLAND]
    J --> K[PX4 landed + auto-disarm]
~~~

| 단계 | 상태 | 역할 |
|---|---|---|
| Phase 0 | `PRECHECK` | PX4 위치·상태, 고도 기준, cue, planner를 검사한다. 실패하면 arm하지 않는다. |
| Phase 1 | `TAKEOFF` | `NAV_TAKEOFF`를 요청한다. 상승 프로파일과 자세는 PX4가 만든다. |
| Phase 1 | `READY` | 5 m 호버를 유지하고 `mission` 명령을 기다린다. |
| Phase 2 | `MISSION_PLAN → MISSION → HOVER` | 현재 위치에서 목표 `(50,50)`까지 계획하고 TrackingMPC로 추종한다. |
| Phase 3 | `RETURN_PLAN → RETURN → LANDING_ACQUIRE → LANDING_DESCEND → PRECLAND → DONE` | 2초마다 검증된 tail을 최신 트레일러 위치로 교체한다. 관측·1.5 m 계획안전 조건을 통과하면 LandingMPC가 고정고도 정렬 후 하강하고, fresh ArUco·정렬·다음 반전까지의 runway를 다시 확인한 0.65 m 이하에서만 PX4에 최종 착륙을 인계한다. |

### A*와 B-spline

A*는 YAML 장애물을 기준으로 우회 topology를 찾는다. 현재 설정은 1 m 격자와
장애물 표면 기준 유클리드 반경 1 m clearance이며, PX4 추종오차를 위한
B-spline 계획 여유 0.5 m를 둔다. 실제 충돌 합격 기준·UI·리포트는 여전히
1 m이다. 장애물은 높이 10 m이므로 5 m 순항고도에서 위로 넘어가지 않는다.

CJU 미션의 B-spline은 A* 경로의 모서리를 부드럽게 보강하는
**geometry-only 후처리기**다. 속도나 시간표를 만들지 않으며, 최종 곡선은
누적거리 기준 0.1 m 간격으로 재표본화된다.

다음 조건을 모두 통과한 경로만 채택한다.

- A* 시작점·끝점과 모든 edge가 자유공간일 것
- SciPy solver가 성공하고 결과가 모두 finite일 것
- B-spline 출력의 모든 chord가 장애물 표면과 유클리드 1.5 m 이상 떨어질 것
- 반환 경로의 시작점·끝점이 실제 요청점과 일치할 것

하나라도 실패하면 경로를 발행하지 않고 현재 위치를 유지한 채 재시도한다.
범용 [`bspline_node.py`](flight/path_plan/path_plan/bspline_node.py)는 아직
legacy 시간·속도 trajectory 계약이므로 CJU의 geometry-only 경로와 혼용하지
않는다.

### 공간 경로 추종과 PX4 소유권

미션 매니저는 실제 위치를 경로에 투영하고 약 6 m 앞의 공간점만
추종 대상으로 제한한다. TrackingMPC는 이 승인 경로에서 제동 가능한 P/V/A
참조를 만들고 10 Hz로 다시 풀며, 50 Hz로 보간해 PX4에 보낸다. 예측 horizon의
모든 선분은 YAML hard clearance로 다시 검사한다.
TrackingMPC의 가속도는 2 m/s³ jerk 제한으로 이어져 계획 대기 HOLD에서 첫
P/V/A 명령으로 넘어갈 때도 한 번에 최대 가속도로 뛰지 않는다.

계산 실패·비정상값·안전검사 실패 시 MPC 결과를 사용하지 않고, 기존에 검증된
공간 target의 PX4 Goto로만 제한적으로 fallback한다. PX4는 MPC setpoint 아래의
위치·속도·자세·모터 제어를 계속 담당한다.

| PX4 파라미터 | 값 | 의미 |
|---|---:|---|
| `MPC_XY_CRUISE` | 3.0 m/s | 경로 순항속도 |
| `MPC_XY_VEL_MAX` | 10.0 m/s | 목표가 아닌 절대 상한 |
| `MPC_ACC_HOR` | 3.0 m/s² | 수평 가속도 |
| `MPC_JERK_AUTO` | 4.0 m/s³ | 자동비행 jerk |
| `MPC_LAND_SPEED` | 0.7 m/s | 일반 착륙 하강속도 |
| `MPC_LAND_CRWL` | 0.1 m/s | 유효 HAGL 하의 지면 근처 crawl |
| `LNDMC_Z_VEL_MAX` | 0.08 m/s | PX4 착륙판정 수직속도 임계값 |
| `COM_DISARM_LAND` | 2.0 s | PX4 착륙 후 자동 disarm |

### 이동 트레일러 재계획

Gazebo의 [`trailer_cue_node.py`](simulation/landing_mpc/landing_mpc/trailer_cue_node.py)는
트레일러 ground-truth를 다음 공통 인터페이스로 변환한다.

| 토픽 | 의미 |
|---|---|
| `/marker/cue` | 트레일러 위치, `px4_local_enu` |
| `/marker/cue_velocity` | 트레일러 속도, `px4_local_enu` |
| `/mission/planned_path` | 현재 승인된 공간 경로 |
| `/mission/vehicle_position` | 실제 기체 위치 |
| `/mission/state` | 현재 Phase/상태 |

`land` 입력 시 고정 트랙 경유점으로 가지 않고 **현재 기체 위치에서 최신
트레일러 cue까지** 계획한다. Gazebo cue는 YAML의 50 Hz command rate로
전달한다. 최초 RETURN만 A*와 B-spline으로 만들고, 이후에는 기존 경로의
앞부분을 유지한 채 2초마다 1.5 m-safe인 tail만 최신 cue로 한 번에 교체한다.
tail 연결이 불가능할 때만 전체 플래너를 실행하며, 계산·실패 중에도 기존
승인 경로를 계속 추종한다.

짐벌은 cue 기준 수평거리 10 m 밖에서 nadir를 유지하고, 10→9 m에서 트레일러
방향으로 연속 전환하며, 9 m 안에서 직접 조준한다. 서로 다른 촬영시각의 KF 승인
ArUco 측정 3개가 0.5 s 안에 들어오고 live cue까지의 직선이 1.5 m
planning-safe일 때 `LANDING_ACQUIRE`로 전환한다. LandingMPC는 먼저 고도를
고정한 채 위치·상대속도와
비전 보정을 수렴시키고, 검증 후 `LANDING_DESCEND`로 하강한다. LandingMPC가
0.65 m까지 직접 하강하고, fresh ArUco·정렬 조건과 다음 셔틀 반전 전 runway를
다시 통과했을 때만 `LandingTargetPose`와 `NAV_PRECLAND`를 PX4에
인계한다. PX4가 `AUTO_PRECLAND`를 수락하면 ROS Offboard setpoint를 중단하며,
최종 접촉판정과 자동 disarm은 PX4가 담당한다.

ArUco는 장거리 항법 제어기가 아니다. 가까운 거리에서 fresh하고 유효할 때
GPS/cue의 수평 bias를 보정하는 보조 관측이다.

### 좌표계

- YAML 장애물과 목표: `stadium_endpoint`
- ROS 미션 내부 기체·cue·경로: `px4_local_enu`
- PX4 메시지: NED

변환은 다음 계약을 따른다.

~~~text
NED x = ENU y
NED y = ENU x
NED z = -ENU z
~~~

실기용 GPS 어댑터도 두 GPS를 같은 측량 원점·heading·고도 기준의
`px4_local_enu`로 변환하고 source timestamp·정확도·freshness를 함께
검증해야 한다.

## YAML 프리뷰와 실시간 UI

UI는 flight-control 명령을 발행하지 않는 읽기 전용 도구다.

~~~bash
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"

# YAML 기반 정적/애니메이션 프리뷰
python3 simulation/gazebo/tools/cju_mission_ui.py

# 창 없이 planner와 좌표 계약 확인
python3 simulation/gazebo/tools/cju_mission_ui.py --check

# PNG 저장
python3 simulation/gazebo/tools/cju_mission_ui.py --save /tmp/cju_path.png

# 이미 실행 중인 Gazebo/실기 ROS graph 구독
python3 simulation/gazebo/tools/cju_mission_ui.py --live
~~~

`--live`는 UI 안에서 경로를 새로 계산하지 않고 미션 매니저가 실제 승인한
`/mission/planned_path`를 표시한다. 기체와 cue도 실제 토픽을 사용한다.
기존 YAML 프리뷰는 그대로 유지된다.

## 로그와 재현 자료

기본 artifact 경로는 다음과 같다.

~~~text
${XDG_STATE_HOME:-$HOME/.local/state}/px4-ros2-wang/cju/<run-id>/
~~~

다른 디스크를
쓰려면 실행 전에 `CJU_LOG_ROOT=/data/cju`처럼 지정한다.

| 파일 | 내용 |
|---|---|
| `manifest.tsv` | git 상태, 명령, 종료상태와 실행 메타데이터 |
| `map.yaml` | 실행 시점의 immutable 맵 snapshot |
| `gimbal_mission.log` | 상태 전이·planner·착륙 로그 |
| `gimbal_mission_view.log` | live UI 로그 |
| `flight.ulg` | PX4 ULog |
| `trailer_odometry.jsonl` | 트레일러 pose/velocity |
| `flight_1hz.csv` | 논문/그래프용 1 Hz 데이터 |
| `flight_summary.csv` | 자동 요약과 v3 quality gate 판정 |

`DONE` 뒤 `Ctrl-C`는 정상 정리 절차다. 이때 launcher 종료코드가 130일 수
있지만, 비행 성공 여부는 `DONE`, PX4 `landed`, auto-disarm과 artifact를
함께 확인한다.

## 테스트

~~~bash
source /opt/ros/humble/setup.bash
source "$HOME/px4_ros2_ws/install/setup.bash"
source install/setup.bash

export PYTHONPATH="$PWD/camera/siyi_gimbal:$PYTHONPATH"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest flight/*/test camera/*/test simulation/*/test -q

python3 simulation/gazebo/tools/validate_self_contained_maps.py
python3 simulation/gazebo/tools/cju_mission_ui.py --check
bash -n simulation/gazebo/run_gimbal.sh simulation/gazebo/run_px4_map.sh
~~~

현재 전체 Python 결과는 `143 passed`다.

## 저장소 구조

| 경로 | 역할 |
|---|---|
| [`flight/path_plan`](flight/path_plan) | A*, SFC, B-spline과 exact collision 계약 |
| [`simulation/landing_mpc`](simulation/landing_mpc) | CJU Phase 0–3 미션, cue, gimbal/marker 결합 |
| [`simulation/gazebo`](simulation/gazebo) | YAML 맵, launcher, artifact exporter, live UI |
| [`simulation/px4_models`](simulation/px4_models) | PX4 SITL 기체·하향 LiDAR 모델 |
| [`camera/rtsp_bridge`](camera/rtsp_bridge) | SIYI A8 RTSP → ROS Image/CameraInfo |
| [`camera/siyi_gimbal`](camera/siyi_gimbal) | SIYI UDP 제어·상태 |
| [`camera/aruco_landing`](camera/aruco_landing) | 실기용 ArUco 검출·품질 gate |
| [`simulation/gz_bridge`](simulation/gz_bridge) | Gazebo transport → ROS 2 bridge |

## 남은 핵심 문제 — 실기체 NO-GO 사유

### P0. 수정된 PX4가 저장소 밖에 있음

최신 성공은 `$HOME/PX4-Autopilot`의 PX4 v1.17.0 dirty checkout에 의존한다.
외부 변경은 대략 다음 범위다.

- uXRCE-DDS `/fmu/in/landing_target_pose` 입력 추가
- PRECLAND 이동표적 속도 feed-forward
- 위치와 상대속도의 연속 정렬 gate
- FlightTaskAuto의 moving-position velocity 연결

이 변경은 `jo`에 포함되지 않았고 실기체용 `.px4` firmware도 고정·빌드·HIL
검증되지 않았다. 먼저 PX4 patch와 `px4_msgs` revision을 저장소에서
재현 가능하게 고정해야 한다.

### P0. 실제 두 GPS가 아직 연결되지 않음

Gazebo의 cue는 GPS가 아니라 완벽한 simulator ground-truth다. 저장소에는
드론 GPS와 트레일러 GPS를 공통 ENU로 만드는 NavSatFix/RTK adapter가 없다.
현재 YAML 좌표도 `not_survey_grade`이므로 현장 RTK 원점·heading·장애물 측량
없이 실물 collision authority로 사용하면 안 된다.

### P0. 실기 A8 인식·bringup이 완결되지 않음

A8 camera calibration, optical/body TF, RTSP→ArUco→미션 topic 연결과
top-level hardware launcher가 완결되지 않았다. placeholder intrinsic으로
props-on 비행하면 안 된다.

### P1. 비행 중 feedback watchdog 부족

위치·VehicleStatus freshness와 PX4 failsafe 검사는 PRECHECK에는 있지만,
이륙 후 모든 airborne state에 공통 적용되는 feedback-loss guard가 부족하다.
현재는 상당 부분 PX4 내부 failsafe에 의존한다.

### P1. 실제 접촉·착륙 성능 미검증

최신 Gazebo 실행에서는 `accel_spike`나 접촉 뒤 상승·미끄러짐이 재현되지 않았고,
PX4 자동 disarm까지 정상 완료됐다. 다만 Gazebo 접촉 모델은 실제 랜딩기어·갑판
충격을 대신하지 못하므로, 실기 적용 전에는 HIL과 props-off 단계시험으로 접촉
하중, 미끄러짐과 자동 disarm을 다시 검증해야 한다.

## 2026-08-03 이후 변경 이력

- **2026-08-04 — `f5dfc08`**
  - CJU 종합운동장 월드와 YAML 좌표 계약 추가
  - 현재 위치 기반 A*와 이동 ArUco 트레일러 착륙 최초 통합
- **2026-08-09 — `b812caf`**
  - 경기장 모델과 자산을 self-contained 구조로 정리
  - 좌표계, 다중 marker, gimbal 추종, launcher와 로그 보존 안정화
- **2026-08-11 — `eda3d3a`**
  - geometry-only CJU B-spline과 exact fail-closed collision 검사
  - PX4-native takeoff/Goto/PRECLAND/auto-disarm로 책임 분리
  - live cue 3 m rolling RETURN 재계획
  - 정수 장애물 재배치, 하향 LiDAR/HAGL, live UI와 artifact exporter
  - 전체 Python 회귀시험 110개 통과
- **2026-08-11 — `b7d4844`**
  - rolling RETURN 재계획 중 exact-safe Goto를 유지해 PX4 smoother reset 제거
  - 착륙 crawl 0.1 m/s와 B-spline 추가 margin 1.0 m 적용
  - body-rate·접촉 충격·장애물 여유·planner/ABORT를 판정하는 quality v3 추가
  - clean Gazebo 전체 미션과 PX4 자동 disarm, 전체 회귀시험 114개 통과

## 실기체 적용 전 최소 완료 조건

- PX4 PRECLAND/DDS patch와 matching `px4_msgs`를 pin하고 실기 target
  firmware를 reproducible build할 것
- 두 GPS의 공통 RTK ENU adapter와 source timestamp/accuracy/freshness gate를
  구현할 것
- 현장 원점·heading·장애물을 측량하고 YAML 변환을 검증할 것
- SIYI A8 calibration·TF·ArUco topic chain을 props-off에서 검증할 것
- 비행 중 feedback-loss/failsafe watchdog과 단일 control authority를 검증할 것
- HIL에서 moving-target 재계획, cue loss, camera loss, PRECLAND fallback,
  RC takeover/kill을 통과할 것

이 조건 전에는 **UI 확인, topic replay, props-off bench와 HIL까지만 허용**하고
실기체 props-on 자동미션에는 사용하지 않는다.
