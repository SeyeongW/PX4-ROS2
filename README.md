# PX4-ROS2 — MPC 정밀착륙

무빙 ArUco 표적에 대한 **MPC 정밀착륙** 스택. 저장소는 단 하나의 기준으로 나뉩니다:

> **이 코드가 진짜 기체에서 도는가?**

```
flight/       비행 임무·제어 (실기체)
camera/       카메라·짐벌·인식 전부
simulation/   시뮬레이터 전용
```

경계에 걸치는 패키지는 두지 않습니다. 갈라지면 쪼갭니다.

---

## flight/ — 실기체

| 패키지 | 역할 |
|---|---|
| `path_plan` | A*/SFC/B-spline 전역 경로. CJU의 B-spline은 A* 경로를 geometry-only로 보강하며, mission manager는 공간 Goto 목표만 PX4에 전달 |

## camera/ — 카메라·짐벌·인식

| 패키지 | 역할 |
|---|---|
| **`rtsp_bridge`** | **A8 mini RTSP → ROS Image + CameraInfo** (GStreamer). 실비행 인식의 입력 |
| **`siyi_gimbal`** | SIYI A8 mini 제어 — 시동 걸리면 직하방 조준, 유지. 프로토콜 표는 `siyi_commands.py` |
| **`aruco_landing`** | 실기체 ArUco 인식 (보정 solvePnP, 품질 게이트) |
| `gimbal_camera` | 짐벌 카메라 시뮬 모델 — gz-sim + Gazebo Classic 양쪽 |

### PX4/MAVROS 실기체 연결

```bash
ros2 launch mavros px4.launch fcu_url:=/dev/ttyTHS1:921600
```

이 저장소는 PX4용 MAVROS를 유지합니다. 현재 `flight/`에는 PX4 실기체에서
arm·mode 전환·setpoint·착륙까지 책임지는 검증 완료 미션 노드가 없습니다.
`path_plan`의 `/path_plan/cmd_vel`을 바로 기체에 연결하지 마세요. PX4/MAVROS
명령 브리지와 운용자 승인·failsafe를 먼저 구현하고 props-off bench와 HIL을
통과해야 합니다.

카메라 체인은 별도로 띄웁니다:

```bash
ros2 launch rtsp_bridge rtsp_bridge.launch.py        # RTSP → /gimbal_camera/image
ros2 run aruco_landing aruco_pose_node               # → /perception/down/marker_pose
```

> ### ⚠️ 카메라 캘리브레이션이 필요합니다
> `rtsp_bridge`는 `camera_info_file`이 비어 있으면 **PLACEHOLDER intrinsic**을
> 발행하고 경고를 반복합니다. solvePnP는 그래도 "그럴듯한" 자세를 내놓지만
> **거리 스케일이 틀립니다** — 실기체 A8 mini로 체커보드 캘리브레이션을 한 뒤
> 그 파일을 지정하세요.
>
> 짐벌 IP `192.168.144.25`는 **제어(UDP 37260)** 와 **영상(RTSP 8554)** 에 모두
> 쓰이지만 서로 다른 채널입니다.

---

## simulation/ — 시뮬레이터 전용

| 패키지 | 역할 |
|---|---|
| **`landing_mpc`** | 인식 체인 + 짐벌 조준 + A*/B-spline 미션 상태기계. 비행 제어와 착륙 판정은 PX4가 담당 → [`docs/ROLES.md`](simulation/landing_mpc/docs/ROLES.md) |
| `gazebo` | 월드·모델·실행 스크립트 → [`MAPS.md`](simulation/gazebo/MAPS.md) |
| `px4_models` | PX4 SITL 기체 (`link_px4_model.sh`가 PX4 트리에 심링크) |
| `gz_bridge` | gz-transport → ROS 2 센서/clock 브리지 |

기본 착륙 실험은 실제 비율의 청주대학교 종합운동장이며, 입력은
`takeoff → land` 두 단어입니다. Phase 0 `PRECHECK`가 PX4·큐·planner를
fail-closed로 검사하고, Phase 1 `TAKEOFF`는 PX4 `NAV_TAKEOFF`가 5 m까지
이륙합니다. Phase 2는 YAML A*가 장애물 topology를 만들고 geometry-only
B-spline이 그 공간 경로를 보강한 뒤, PX4 Goto로 `(50,50)`까지 이동해
`HOVER`합니다. 속도·가속도·jerk는 B-spline이 아니라 PX4
Goto/PositionSmoothing의 `MPC_XY_CRUISE=3`, `MPC_ACC_HOR`,
`MPC_JERK_AUTO`가 만들고 `MPC_XY_VEL_MAX=10`이 절대 상한을 맡습니다.
6 m lookahead는 시간표가 아니라 곡선상의 공간 목표이며, 장애물 근처에서는
exact 선분 검사로 안전한 거리까지 자동 축소됩니다.

Phase 3에서 `land`를 입력하면 현재 기체에서 live trailer cue까지의
직선이 YAML 장애물 기준으로 안전한지 먼저 검사합니다. 막혔으면
A*→geometry-only B-spline `RETURN`을 계획하고, 6 m 이내의 안전한
직선이 확보되면 `LandingTargetPose`를 보내 `NAV_PRECLAND`로 인계합니다.
그 뒤 속도·가속·자세·하강·접촉판정·자동 무장해제는 PX4만 담당하고,
ArUco는 landing target의 수평 위치 보정에만 사용됩니다.
트레일러는 맵 `(5,0) ↔ (5,50)` 직선을 `1 m/s`로 왕복합니다.

```bash
./simulation/gazebo/run_gimbal.sh mission     # 전체 미션
./simulation/gazebo/run_gimbal.sh gimbal      # 짐벌 + 인식만
./simulation/gazebo/run_gimbal.sh baseline    # 고정 카메라 (비교군)
HEADLESS=1 ./simulation/gazebo/run_gimbal.sh mission
```

---

## `jo` 브랜치 변경 이력·검증 상태 (2026-08-03 이후)

### 주요 변경

- **8월 4일 (`f5dfc08`)** — CJU 종합운동장 월드·정수 장애물·이동
  ArUco 트레일러와 첫 장거리 A* 착륙 미션을 통합했습니다.
- **8월 9일 (`b812caf`)** — 월드/모델 생성기를 정리하고 실제 비율 트랙,
  3-marker 인식, 짐벌 추종, 트레일러 왕복 구동과 미션 회귀시험을
  `jo` 브랜치 기준으로 통합했습니다.
- **8월 11일 (이번 `jo` 업데이트)**
  - Phase 0 `PRECHECK`, Phase 1 PX4 `NAV_TAKEOFF`, Phase 2
    A*→geometry-only B-spline→PX4 Goto, Phase 3 rolling RETURN→PX4
    `NAV_PRECLAND`로 책임을 분리했습니다.
  - B-spline에서 CJU 경로의 시간·속도·가속도 목적을 제거하고, solver 상태,
    finite 결과, 모든 출력 선분의 exact AABB 충돌검사를 통과한 경로만
    발행하도록 fail-closed 처리했습니다.
  - 트레일러 cue가 3 m 이동할 때마다 최신 위치로 RETURN을 재계획하며,
    6 m 안에서 live 직선이 안전할 때만 PRECLAND로 넘깁니다.
  - 장애물 중심을 정수 좌표와 `[0,40] × [0,40]` 범위로 정리하고,
    장애물 7/14/15/19를 재배치했습니다. 물리 장애물, YAML, world 및
    회귀시험 좌표는 동일합니다.
  - 하향 LiDAR의 방향·자가차폐를 수정해 PX4 HAGL과 1 m 이하
    `MPC_LAND_CRWL=0.3 m/s`가 동작하도록 했습니다.
  - 기존 YAML 사전 프리뷰를 유지하면서 실제 승인 경로, 기체, 트레일러,
    미션 상태를 읽기 전용으로 표시하는 live UI와 ULog 1 Hz artifact
    exporter를 추가했습니다.

### 최신 Gazebo 검증

2026-08-11 run `20260811T171441Z.rq1hDz`에서
`TAKEOFF → MISSION → HOVER → RETURN → PRECLAND → DONE`을 완료했습니다.
Artifact는 `${XDG_STATE_HOME:-$HOME/.local/state}/px4-ros2-wang/cju/<run-id>/`
아래에 저장됩니다. 이 run의 `exit_status=130`은 DONE 뒤 런처를 종료한 값입니다.

| 항목 | 결과 |
|---|---:|
| A*/B-spline 승인 | 7/7, planner failure·ABORT 0 |
| PX4 failsafe / ULog dropout | 0 / 0 |
| 실제 장애물 표면 최소거리 | 3.04 m |
| 설정된 2 m 위험영역 바깥 최소여유 | 0.271 m |
| PX4 landed 시 중심 오차 / 상대 XY 속도 | 0.053 m / 0.008 m/s |
| PX4 착륙판정 / 자동 무장해제 | 성공 / 성공 |
| 전체 Python 회귀시험 | 110 passed |

이 run의 CSV `quality=PASS`는 failsafe·로그 연속성·속도 step 중심의 제한된
판정입니다. 접촉 시 3D 가속도 29.2 m/s²와 RETURN 재계획 경계의 최대 body
rate 112.6 deg/s는 별도로 남아 있으므로 이를 실기체 안전 승인으로 해석하면
안 됩니다. 커밋 전 dirty tree에서 실행되어 `paper_reproducible=0`이기도 합니다.

### 현재 적용 범위

**Gazebo SITL은 재현 가능한 실험 경로가 완성됐지만, 실기체 props-on은 아직
NO-GO입니다.** 현재 성공은 저장소 밖 `~/PX4-Autopilot`의 미커밋
PX4 v1.17 PRECLAND/DDS 수정에 의존하며 실기체용 firmware가 고정·빌드되지
않았습니다. `/marker/cue`도 아직 Gazebo ground truth이고, 두 GPS를 공통
측량 ENU로 변환하는 adapter와 실기 A8 calibration/TF, 비행 중 feedback
watchdog이 없습니다. 또한 `flight/path_plan`의 범용 ROS B-spline 노드는
여전히 legacy 시간·속도 출력을 사용하므로 CJU geometry-only 경로와 혼용하지
마세요.

---

## 비행 인터페이스 규칙

### 1. 비행제어기는 PX4만 사용합니다

저장소의 기체 펌웨어 기준은 PX4입니다. 실기체 연결은 PX4용 MAVROS를 사용하며,
시뮬레이션 정밀착륙 스택은 PX4 uXRCE-DDS `/fmu/*` 토픽을 직접 사용합니다.

### 2. 한 기체에는 하나의 명령 권한만 둡니다

MAVROS와 uXRCE-DDS를 같은 PX4에 연결할 수는 있지만, 동시에 둘 이상의 노드가
arm/mode/setpoint를 발행하면 안 됩니다. 각 실행 스크립트가 선택한 인터페이스와
실제 setpoint 발행자를 시작 전에 확인하세요.

---

## 빌드 · 테스트

```bash
colcon build && source install/setup.bash

PYTHONPATH="$PWD/camera/siyi_gimbal:$PYTHONPATH" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest flight/*/test camera/*/test simulation/*/test -q
```

---

## 환경 구축 (최초 1회)

<details>
<summary>ROS 2 Humble + PX4용 MAVROS</summary>

```bash
sudo apt-get install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep
sudo apt-get install -y ros-humble-mavros ros-humble-mavros-msgs ros-humble-mavros-extras
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```
</details>

<details>
<summary>PX4 SITL + Gazebo Harmonic (시뮬레이션용)</summary>

```bash
sudo bash simulation/gazebo/install_apt_deps.sh   # sudo 필요
./simulation/gazebo/setup_px4_sitl.sh             # PX4 트리 준비
./simulation/gazebo/link_px4_model.sh             # px4_models를 PX4에 심링크
python3 simulation/gazebo/gen_aruco_model.py      # 마커 텍스처 생성
```

`px4_models/`를 옮기거나 저장소를 다른 경로로 옮기면 `link_px4_model.sh`를
다시 실행해야 합니다 — PX4 트리의 심링크가 절대경로입니다.
</details>

<details>
<summary>Gazebo Classic (짐벌 카메라 Classic 버전용)</summary>

```bash
sudo apt install gazebo11 ros-humble-gazebo-ros-pkgs
cmake -S camera/gimbal_camera/plugins -B camera/gimbal_camera/plugins/build
cmake --build camera/gimbal_camera/plugins/build
```
</details>

<details>
<summary>PC ↔ Jetson 네트워크 (CycloneDDS + Tailscale)</summary>

현재는 `rmw_fastrtps_cpp`를 씁니다. Cyclone 설정은 `config/`에 남아 있습니다.

```bash
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
export CYCLONEDDS_URI=$PWD/config/cyclonedds_pc.xml     # 필요할 때만
```
</details>

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| 짐벌이 안 움직임 | `/siyi_gimbal_node/status`의 `bad_rx`와 자세 피드백 확인. IP·기체 네트워크 |
| PX4용 MAVROS 연결 실패 | `px4.launch`, `fcu_url`, 시리얼 권한 `sudo usermod -aG dialout $USER` 확인 |
| SITL에서 월드를 못 찾음 | 저장소를 옮겼다면 `link_px4_model.sh` 재실행 |
| Gazebo에 모델이 안 뜸 | `run_gimbal.sh` / `run_px4_map.sh`를 쓰세요 (경로를 직접 설정합니다) |
| 토픽이 안 보임 (두 머신) | `tailscale status`, RMW 구현 일치 확인 |
