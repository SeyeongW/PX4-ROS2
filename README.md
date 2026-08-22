# PX4-ROS2 — 실기체 trailer 통합 임무

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
| **`mpc_landing`** | **최종 실기체 미션.** trailer GPS → A*/SFC/B-spline → TrackingMPC 순항 → ArUco P 착륙. PX4 I/O는 MAVROS 단일 노드가 담당 |
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
짐벌과 미션 노드를 시작합니다. ARM 전 승인만 조종자가 직접 합니다.

```
프리체크 ─승인─► ARM/이륙 ─► A*/SFC/B-spline ─► TrackingMPC 순항
                                                    │ ArUco 획득
                                                    ▼
                                              기존 P 제어 착륙
```

```bash
ros2 topic echo /aruco_landing_node/state
ros2 run mpc_landing approve aruco_landing_node
ros2 run mpc_landing abort aruco_landing_node
```

기본 장치가 다를 때만 환경변수로 바꿉니다.

```bash
FCU_URL=/dev/ttyACM0:57600 TRAILER_DEV=/dev/ttyUSB1 ./run_px4 trailer
```

청주대 지도는 현장 측량 전까지 `hardware_flight_approved: false`이므로 ARM 승인이
차단됩니다. `ROUTE_MAP_APPROVED=1`은 측량을 대신하지 않으며 실제 비행에서 임의로
사용하면 안 됩니다.

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
- 착륙: 기존 실기체 검증 ArUco P 제어
- 장애물 팽창: `vehicle_clearance_xy_m: 1.0`만 적용
- PX4 명령: `/mavros/setpoint_raw/local` 단일 authority
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
