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
| `path_plan` | A*/SFC/B-spline 전역 + MPC 지역 경로. PX4 명령 브리지는 아직 구현 전 |

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
| **`landing_mpc`** | **MPC 착륙 스택 본체.** 인식 체인 + 짐벌 조준 + 미션 상태기계 → [`docs/ROLES.md`](simulation/landing_mpc/docs/ROLES.md) |
| `gazebo` | 월드·모델·실행 스크립트 → [`MAPS.md`](simulation/gazebo/MAPS.md) |
| `px4_models` | PX4 SITL 기체 (`link_px4_model.sh`가 PX4 트리에 심링크) |
| `gz_bridge` | gz-transport → ROS 2 센서/clock 브리지 |

기본 착륙 실험은 장애물 없는 `300 × 100 m` 직선 셔틀 맵이며, 트레일러가
`3 m/s`로 300 m 전진한 뒤 차체를 돌리지 않고 300 m 후진합니다.

```bash
./simulation/gazebo/run_gimbal.sh mission     # 전체 미션
./simulation/gazebo/run_gimbal.sh gimbal      # 짐벌 + 인식만
./simulation/gazebo/run_gimbal.sh baseline    # 고정 카메라 (비교군)
HEADLESS=1 ./simulation/gazebo/run_gimbal.sh mission
```

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
