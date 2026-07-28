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
| **`mpc_landing`** | **게이트형 MPC 정밀착륙 미션.** 5 m 이륙 → 마커 탐색 → MPC 하강. 단계마다 조종자 승인 |
| `path_plan` | A*/SFC/B-spline 전역 + MPC 지역 경로 |
| `offboard` | C++ MAVROS offboard 제어 모음 |

## camera/ — 카메라·짐벌·인식

| 패키지 | 역할 |
|---|---|
| **`rtsp_bridge`** | **A8 mini RTSP → ROS Image + CameraInfo** (GStreamer). 실비행 인식의 입력 |
| **`siyi_gimbal`** | SIYI A8 mini 제어 — 시동 걸리면 직하방 조준, 유지. 프로토콜 표는 `siyi_commands.py` |
| **`aruco_landing`** | 실기체 ArUco 인식 (보정 solvePnP, 품질 게이트) |
| `gimbal_camera` | 짐벌 카메라 시뮬 모델 — gz-sim + Gazebo Classic 양쪽 |

### 실비행 실행

```bash
ros2 launch mavros apm.launch fcu_url:=/dev/ttyTHS1:921600   # 별도로 먼저
ros2 launch mpc_landing flight_bringup.launch.py             # 짐벌 + 미션
```

미션은 **단계마다 멈춰서 승인을 기다립니다.**

```
프리체크 PASS ─승인─► 시동 ─승인─► 이륙(5 m) ─승인─► 탐색
                                                        │ 마커 발견 (자동)
                                                        ▼
                                                   MPC 하강 → 착륙
```

```bash
ros2 topic echo /mpc_landing_node/state                          # 지금 뭘 기다리는지
ros2 service call /mpc_landing_node/approve std_srvs/srv/Trigger # 게이트 해제
ros2 service call /mpc_landing_node/abort   std_srvs/srv/Trigger # 중단·착륙
```

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

## 두 가지 규칙

### 1. MPC는 한 벌만 존재합니다

`mpc_landing`는 자체 MPC를 갖지 않고 `landing_mpc.mpc.LandingMPC`를 **그대로
import**합니다. 실비행의 목적이 MPC 검증인데, 비슷하게 다시 짠 것을 날리면
그 사본이 검증될 뿐이기 때문입니다.

- **가중치**는 SITL 값 그대로 (`w_xy=6.0`, `w_terminal=40.0` …)
- **한계값**만 실기체용으로 낮춤 (`v_max` 3.5 → 0.8, `vz_max` → 0.35, `a_max` → 0.6)

### 2. 파라미터는 노드에만

`flight/`의 런치 파일은 **파라미터를 하나도 넘기지 않습니다.** 모든 튜닝값은 노드의
`_declare()` 안에, 이유를 적은 주석과 함께 있습니다. 런치와 소스로 나뉘면
소스가 "지금 뭘 날리고 있나"의 답이 아니게 됩니다.

일회성 실험은 편집 없이:
```bash
ros2 run mpc_landing mpc_landing_node --ros-args -p takeoff_alt_m:=3.0
```

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
