# PX4-ROS2 — 실기체 trailer 통합 임무

무빙 트레일러까지 **TrackingMPC 경로 추종 후 ArUco P 정밀착륙**하는 스택.
저장소는 단 하나의 기준으로 나뉩니다:

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
| **`mpc_landing`** | **최종 실기체 미션.** 고정 CJU 목표 → trailer GPS 복귀 → ArUco P 착륙. 두 장거리 구간은 A*/SFC/B-spline + TrackingMPC이고 PX4 I/O는 MAVROS 단일 노드가 담당 |
| `path_plan` | A*/SFC/B-spline 전역 + MPC 지역 경로 |
| `offboard` | C++ MAVROS offboard 제어 모음 |

## camera/ — 카메라·짐벌·인식

| 패키지 | 역할 |
|---|---|
| **`siyi_gimbal`** | SIYI A8 mini 제어 — 시동 걸리면 직하방 조준, 유지. **시리얼/UDP 양쪽 지원**(기본 `/dev/ttyTHS1`). 프로토콜 표는 `siyi_commands.py` |
| **`aruco_landing`** | 실기체 ArUco 인식 (보정 solvePnP, 품질 게이트) |
| `gimbal_camera` | 짐벌 카메라 시뮬 모델 — gz-sim + Gazebo Classic 양쪽 |

### 실비행 실행

```bash
cd flight/mpc_landing
./run_px4 trailer
```

이 명령 하나가 MAVROS, 900 MHz trailer GPS, 카메라, ArUco 검출기, SIYI
짐벌과 미션 노드를 시작합니다. PRECHECK 완료 후 같은 터미널에 아래 세 단어를
순서대로 입력합니다.

```
PRECHECK ─TAKEOFF─► ARM/5 m 이륙 ─► READY
                                      │ MISSION
                                      ▼
              MISSION_PLAN → A*/SFC/B-spline + TrackingMPC
                                      └────────► CJU (50,50) HOVER
                                                        │ LAND
                                                        ▼
              RETURN_PLAN → 최신 trailer GPS 반복 재계획/복귀
                                      └────────► ArUco 획득
                                                        ▼
                                                기존 P 제어 착륙/DONE
```

`TAKEOFF`, `MISSION`, `LAND`가 아닌 입력과 순서가 맞지 않는 입력은 거부됩니다.
`ABORT`는 어느 단계에서든 현재 위치 착륙을 요청합니다. 다른 터미널에서는 현재
대기 중인 명령을 `approve`로 실행하거나 명령 토픽에 정확한 단어를 보낼 수 있습니다.

```bash
ros2 topic echo /aruco_landing_node/state
ros2 run mpc_landing approve aruco_landing_node
ros2 topic pub --once /aruco_landing_node/command std_msgs/msg/String \
  "{data: MISSION}"
ros2 run mpc_landing abort aruco_landing_node
```

기본 장치가 다를 때만 환경변수로 바꿉니다.

```bash
FCU_URL=/dev/ttyACM0:57600 TRAILER_DEV=/dev/ttyUSB1 ./run_px4 trailer
```

청주대 지도는 현장 측량 전까지 `hardware_flight_approved: false`이므로 ARM 승인이
차단됩니다. `ROUTE_MAP_APPROVED=1`은 측량을 대신하지 않으며 실제 비행에서 임의로
사용하면 안 됩니다.

### CJU `(50,50)`을 실제 운동장에 맞추는 방법

`(50,50)`은 WGS84 위경도가 아니라 `stadium_endpoint` 현장 좌표계의 미터 단위
XY입니다. 노드는 동기화된 드론 WGS84/Local pose로 현장 원점을 MAVROS Local ENU에
놓고, 지도 heading을 적용해 `(50,50)`을 Local ENU로 변환합니다. Local ENU에
`(50,50)`을 그대로 넣지 않습니다.

현재 OSM 정합값으로 계산한 예상점은 약
`36.654458, 127.495961`이지만 유효 정밀도가 없는 참고값이므로 비행 목표로 확정하면
안 됩니다. 실제 적용 전에는 다음 네 가지를 RTK Fixed 또는 측량급 장비로 확인합니다.

1. 다시 찾을 수 있는 현장 `(0,0)` 기준점을 설치하고 WGS84를 반복 측정해
   `origin_wgs84`와 `horizontal_accuracy`를 갱신합니다.
2. 기준점과 +x 방향의 50~100 m 장기선 두 점으로
   `heading_deg_enu = degrees(atan2(North, East))`를 계산합니다. 이는 +East에서
   반시계 방향으로 잰 각도입니다.
3. 보정된 좌표로 `(50,50)`을 stake-out한 뒤 독립 검측점으로 정·역변환 오차를
   확인합니다. 현재 지도 중심 후보 `(44,46)`과 `(50,50)`은 약 7.21 m 떨어진
   서로 다른 점이므로 원하는 임무 지점도 이때 확정해야 합니다.
4. 지형/비행구역, 이륙점·목표·전체 경로, 실제 장애물 외곽과 5 m 고도의
   전선·조명탑 등을 측정합니다. `terrain.center_m/size_m`를 실제 허용 구역으로
   갱신하고, 각 장애물 외곽을 `site_xy = ENU(origin→point) @ R`로 변환한 뒤
   회전 물체·전선까지 감싸는 현장 좌표계 축정렬 AABB의 `center_m/size_m`로
   barrier를 교체해야 합니다. planner는 모든 barrier를 전고도 비행금지
   기둥으로 취급합니다. 이 검증 후에만 `hardware_flight_approved: true`로
   바꿉니다.

목표가 원점에서 약 70.7 m 떨어져 있어 heading 오차 1°는 목표를 약 1.23 m 옮깁니다.
따라서 휴대전화 GPS나 한 번의 일반 GNSS 측정만으로는 이 지도를 승인할 수 없습니다.

---

## simulation/ — 시뮬레이터 전용

| 패키지 | 역할 |
|---|---|
| **`landing_mpc`** | **MPC 착륙 스택 본체.** 인식 체인 + 짐벌 조준 + 미션 상태기계 → [`docs/ROLES.md`](simulation/landing_mpc/docs/ROLES.md) |
| `gazebo` | 월드·모델·실행 스크립트 → [`MAPS.md`](simulation/gazebo/MAPS.md) |
| `px4_models` | PX4 SITL 기체 (`link_px4_model.sh`가 PX4 트리에 심링크) |
| `gz_bridge` | gz-transport → ROS 2 센서/clock 브리지 |

```bash
./simulation/gazebo/run_gimbal.sh mission     # 전체 미션
./simulation/gazebo/run_gimbal.sh gimbal      # 짐벌 + 인식만
./simulation/gazebo/run_gimbal.sh baseline    # 고정 카메라 (비교군)
HEADLESS=1 ./simulation/gazebo/run_gimbal.sh mission
```

---

## 최종 실기체 제어 계약

- 경로 생성: A* → SFC → B-spline
- 경로 추종: Wang과 같은 모델·비용함수의 `TrackingMPC`
- 착륙: 기존 실기체 검증 ArUco P 제어. 현재 속도 추정은 위치 차분 저역통과이며
  시뮬레이션의 KF는 실기체 검증 경로에 임의 이식하지 않음
- 장애물 팽창: `vehicle_clearance_xy_m: 1.0`만 적용
- PX4 명령: `/mavros/setpoint_raw/local` 단일 authority
- 상태 흐름: PRECHECK → TAKEOFF/READY → MISSION_PLAN/MISSION/HOVER →
  RETURN_PLAN/RETURN → ArUco P → DONE
- 운영 진입점: `./run_px4 trailer` 하나

---

## 빌드 · 테스트

```bash
colcon build && source install/setup.bash

PYTHONPATH="$PWD/camera/siyi_gimbal:$PWD/flight/mpc_landing:$PYTHONPATH" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest flight/*/test camera/*/test simulation/*/test -q
```

`flight_logs/`는 ArduPilot dataflash `.BIN`입니다. colcon이 만드는 `log/`와는
다른 것이라 이름을 분리해 두었습니다.

---

## 환경 구축 (최초 1회)

<details>
<summary>ROS 2 Humble + MAVROS</summary>

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
| `mpc_landing`가 시동을 거부 | 프리체크 로그를 보세요. 어떤 항목이 왜 FAIL인지 한 줄씩 찍습니다 |
| `approve` 호출이 거부됨 | 게이트가 아닌 단계입니다. 조기 승인은 삼키지 않고 거부합니다 |
| 짐벌이 안 움직임 | `/siyi_gimbal_node/status`의 `bad_rx`와 자세 피드백 확인. IP·기체 네트워크 |
| MAVROS 연결 실패 | `fcu_url` 확인, 시리얼 권한 `sudo usermod -aG dialout $USER` |
| SITL에서 월드를 못 찾음 | 저장소를 옮겼다면 `link_px4_model.sh` 재실행 |
| Gazebo에 모델이 안 뜸 | `run_gimbal.sh` / `run_px4_map.sh`를 쓰세요 (경로를 직접 설정합니다) |
| 토픽이 안 보임 (두 머신) | `tailscale status`, RMW 구현 일치 확인 |
