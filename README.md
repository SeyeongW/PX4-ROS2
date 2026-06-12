# ArduPilot ROS 2 오프보드 제어 + YOLO 추적

ArduPilot 기반 드론을 ROS 2로 Offboard 제어하며, 하방 카메라 + YOLO로 지상 표적을 추적하는 시스템입니다.

## 프로젝트 구성

```
PX4-ROS2/
├── offboard/               # C++ MAVROS 오프보드 제어 노드
│   └── src/
│       ├── offboard_control.cpp          # 기본 웨이포인트 비행
│       ├── offboard_sim_waypoints.cpp    # 시뮬레이션용 호버 추적
│       └── offboard_tracking_control.cpp # YOLO 연동 실시간 추적
├── camera_detection/       # Python YOLO 인식 노드
│   └── camera_detection/
│       ├── yolo_processor_node.py        # 실기체용 (TensorRT 엔진)
│       ├── yolo_processor_sim_node.py    # 시뮬레이션용 (.pt 모델)
│       ├── yolo_processor_depth.py       # RealSense 뎁스 카메라용
│       └── commander.py                  # 추적 타겟 ID 지정 CLI
├── precision_landing/      # ArUco 정밀착륙 (Python)
│   ├── precision_landing/precision_landing_node.py  # 착륙 상태기계 제어
│   ├── precision_landing/moving_marker_node.py      # 마커 이동 + ENU 좌표 송출
│   └── launch/precision_landing.launch.py           # 풀 bringup 런치
├── gazebo/                 # Gazebo Harmonic 시뮬레이션 자산
│   ├── models/iris_with_down_camera/     # 하방 카메라 장착 Iris 모델
│   ├── worlds/iris_down_camera_runway.sdf
│   ├── launch/camera_bridge.launch.py    # Gazebo → ROS 2 카메라 브리지
│   ├── install_apt_deps.sh
│   └── run_sim.sh
└── config/                 # CycloneDDS 네트워크 설정 (PC ↔ Jetson)
    ├── cyclonedds_pc.xml
    └── cyclonedds_jetson.xml
```

---

## 시스템 요구사항

| 항목 | 사양 |
|------|------|
| OS | Ubuntu 22.04 LTS (Jammy) |
| ROS 2 | Humble Hawksbill |
| 비행 제어기 소프트웨어 | ArduPilot (MAVROS 기반) |
| 시뮬레이터 | Gazebo Harmonic (gz-sim 8) |
| GPU (실기체) | NVIDIA Jetson (TensorRT 추론용) |

---

## 1단계: ROS 2 Humble 설치

```bash
sudo apt-get update && sudo apt-get install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt-get install -y software-properties-common
sudo add-apt-repository universe

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt-get update
sudo apt-get install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep

# 매 터미널마다 자동 source되도록 설정
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 2단계: MAVROS 설치

MAVROS는 ROS 2와 ArduPilot(MAVLink) 사이의 미들웨어입니다.

```bash
sudo apt-get install -y \
  ros-humble-mavros \
  ros-humble-mavros-msgs \
  ros-humble-mavros-extras

# GeographicLib 데이터셋 (필수)
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
```

---

## 3단계: Python 의존성 설치

```bash
pip3 install ultralytics opencv-python numpy

# RealSense 뎁스 카메라를 사용하는 경우
pip3 install pyrealsense2
```

---

## 4단계: 워크스페이스 설정 및 빌드

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone -b wang https://github.com/SeyeongW/PX4-ROS2.git
cd ~/ros2_ws

# 의존성 자동 설치
rosdep init  # (최초 1회)
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 빌드
colcon build --symlink-install
source install/setup.bash

# 매 터미널마다 자동 source되도록 설정
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

---

## 5단계: ArduPilot SITL + Gazebo 시뮬레이션 환경 구성

시뮬레이션을 사용하지 않고 실기체만 사용한다면 이 단계를 건너뛰어도 됩니다.

### 5-1. apt 의존성 일괄 설치 (sudo 필요)

```bash
sudo bash ~/ros2_ws/src/PX4-ROS2/gazebo/install_apt_deps.sh
```

이 스크립트는 다음을 설치합니다: `gz-harmonic`, `libgz-sim8-dev`, `cmake`, `build-essential`, `rapidjson-dev`

### 5-2. ArduPilot 소스 클론 및 SITL 빌드

```bash
cd ~
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot

# 의존성 설치 (Python, 빌드 도구, MAVProxy 등)
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile   # sim_vehicle.py를 PATH에 등록

# SITL 빌드 (시간 소요)
./waf configure --board sitl
./waf copter
```

### 5-3. ardupilot_gazebo 플러그인 빌드

ArduPilot과 Gazebo를 연결하는 플러그인입니다.

```bash
cd ~
git clone https://github.com/ArduPilot/ardupilot_gazebo.git
cd ardupilot_gazebo
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
```

### 5-4. ros_gz 브리지 빌드 (Gazebo Harmonic ↔ ROS 2)

> Humble 바이너리 패키지는 Gazebo Fortress 기준이므로, Harmonic과 쓰려면 소스 빌드가 필요합니다.

```bash
export GZ_VERSION=harmonic
mkdir -p ~/ros_gz_ws/src && cd ~/ros_gz_ws/src
git clone https://github.com/gazebosim/ros_gz.git -b humble
cd ~/ros_gz_ws
GZ_VERSION=harmonic colcon build --merge-install --packages-up-to ros_gz_bridge

echo "source ~/ros_gz_ws/install/setup.bash" >> ~/.bashrc
```

---

## 시뮬레이션 실행 (ArduPilot SITL + Gazebo)

4개의 터미널이 필요합니다.

### 터미널 1 — Gazebo 실행

```bash
cd ~/ros2_ws/PX4-ROS2/gazebo
./run_sim.sh
```

Gazebo에 하방 카메라가 장착된 Iris 모델이 로드됩니다.

### 터미널 2 — ArduPilot SITL 실행

```bash
cd ~/ardupilot
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console
```

MAVProxy 콘솔에서 기체를 제어할 수 있습니다:

```
mode guided
arm throttle
takeoff 10
```

### 터미널 3 — MAVROS 실행

SITL과 ROS 2를 연결합니다.

```bash
ros2 launch mavros apm.launch fcu_url:=udp://:14550@
```

> MAVProxy(`sim_vehicle.py`)는 `127.0.0.1:14550` 으로만 MAVLink를 내보냅니다.
> 그래서 MAVROS는 **14550 에 bind**하고 remote(`@` 뒤)는 비워서 상대 주소를 자동 학습시킵니다.
> `@14555` 처럼 빈 포트로 송신하게 두면 `connection refused`(UDP closed)가 납니다.

### YOLO 시뮬레이션 노드 실행 (선택)

> 카메라 영상 브리지는 아래 **ArUco 정밀착륙 런치에 포함**되어 자동 실행됩니다.
> YOLO만 단독으로 쓸 때는 카메라 브리지를 별도로 띄우세요:
> `source ~/ros_gz_ws/install/setup.bash && ros2 launch ~/ros2_ws/PX4-ROS2/gazebo/launch/camera_bridge.launch.py`
> ( `ros2 topic hz /down_camera/image` 로 ~30 Hz 확인 )

```bash
ros2 run camera_detection yolo_processor_sim_node \
  --ros-args -p image_topic:=/down_camera/image -p model_path:=yolo11s.pt
```

### 오프보드 추적 제어 실행

```bash
# 웨이포인트 시뮬레이션
ros2 run offboard offboard_sim_waypoints

# 또는 YOLO 연동 추적 제어
ros2 run offboard offboard_tracking_control
```

---

## ArUco 마커 정밀착륙

하방 카메라로 ArUco 마커(DICT_4X4_50, ID 0)를 검출해 그 위로 정밀착륙합니다.
마커는 **움직이는 플랫폼**(차량) 위에 있을 수 있으며, 드론은 외부에서 송출되는
마커 좌표(cue)로 먼저 접근하다가 카메라가 마커를 잡으면 비전 서보로 인계받아
플랫폼 속도를 맞춰가며 착륙합니다.

`precision_landing.launch.py` **하나가 ROS 쪽을 전부 띄웁니다** — Gazebo와
ArduPilot SITL만 따로 켜면 되고, MAVROS·카메라 브리지를 따로 수동 실행할 필요가 없습니다.

### 런치파일 구성 (`precision_landing.launch.py`)

런치 하나가 다음 **4개 노드를 자동 기동**하고 SITL에 연결합니다:

| 노드 | 패키지 | 역할 |
|------|--------|------|
| MAVROS | `mavros` | ROS 2 ↔ ArduPilot MAVLink 브리지 (`fcu_url`로 SITL 연결) |
| 카메라 브리지 | `ros_gz_bridge` | Gazebo 하방 카메라 → `/down_camera/image` 토픽 |
| ArUco 검출 | `camera_detection` | 마커 검출 → `/perception/aruco_offset`, `/perception/aruco_detected` |
| 정밀착륙 제어 | `precision_landing` | 상태기계(이륙→접근→정렬→하강→착륙), 속도 셋포인트 발행 |

> 마커 이동 + 좌표 송출(`moving_marker_node`)은 월드의 일부라
> `gazebo/run_sim.sh`가 Gazebo와 함께 띄웁니다(런치에서 중복 실행하지 않음).

주요 런치 인자 (`이름:=값` 으로 변경):

| 인자 | 기본값 | 역할 |
|------|--------|------|
| `fcu_url` | `udp://:14550@` | MAVROS↔SITL 연결 (14550 bind, remote 자동학습) |
| `flight_alt` | `5.0` | 이륙/접근 호버 고도 (m) |
| `auto_takeoff` | `true` | 노드가 스스로 GUIDED→시동→이륙 |
| `use_cue` | `true` | 비전 인식 전 cue 좌표로 먼저 접근 |
| `platform_height` | `1.0` | 마커가 올라앉은 플랫폼 높이 (m, 평면 지면이면 0) |
| `land_clearance` | `0.2` | 마커 윗면 위 이 높이에서 강제 disarm |
| `vel_gain` / `vel_max` | `0.4` / `5.0` | 정밀 수평 속도 게인 / 상한 |
| `approach_vel_max` / `approach_decel_s` | `10.0` / `5.0` | 접근 최대 속도 / ETA 감속 시작(s) |
| `lat_swap` / `lat_sign_fwd` / `lat_sign_left` | `false` / `1` / `1` | 카메라 마운트 이미지→기체 매핑 보정 |
| `yaw_track` | `true` | APPROACH 중 진행방향으로 기수 정렬 (false면 헤딩 고정) |

> 전체 파라미터 · 코드 함수 · 디버그 로그 읽는 법 · 튜닝 가이드는
> **[`docs/precision_landing.md`](docs/precision_landing.md)** 를 참고하세요.

### 사전 준비 (최초 1회)

```bash
# 마커 텍스처 생성 (안 하면 마커가 검은 박스로 보임)
python3 ~/ros2_ws/PX4-ROS2/gazebo/gen_aruco_model.py

# 빌드
cd ~/ros2_ws/PX4-ROS2
colcon build --symlink-install --packages-select camera_detection precision_landing
source install/setup.bash
```

### 실행 (터미널 3개)

```bash
# 터미널 1 — Gazebo
cd ~/ros2_ws/PX4-ROS2/gazebo && ./run_sim.sh

# 터미널 2 — ArduPilot SITL
cd ~/ardupilot && sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console --map

# 터미널 3 — 서버 일괄 (MAVROS + 카메라 브리지 + 검출 + 정밀착륙[Python])
source ~/ros2_ws/PX4-ROS2/install/setup.bash
source ~/ros_gz_ws/install/setup.bash
ros2 launch precision_landing precision_landing.launch.py
```

### 착륙 동작

기본값 `auto_takeoff:=true` 면 `precision_landing` 노드가 **스스로 GUIDED 전환 → 시동 → `flight_alt`까지 이륙**한 뒤,
cue 접근 → 마커 감지 → 정렬 → 하강 → 플랫폼 윗면 `land_clearance` 이내에서 **강제 disarm**으로 안착합니다.
(지면까지 내려가는 `LAND` 모드는 쓰지 않습니다 — 플랫폼을 무시하므로.)

수동으로 띄우려면 `auto_takeoff:=false` 로 실행하고 터미널 2(MAVProxy)에서 직접 이륙시키세요.
그러면 `armed + GUIDED + (마커 감지 또는 유효 cue)` 상태가 됐을 때 자동으로 인계받습니다.

```
mode guided
arm throttle
takeoff 5
```

### 모니터링

```bash
ros2 topic echo /precision_landing/debug                        # 단계 전환 로그
ros2 run rqt_image_view rqt_image_view /perception/aruco_debug/compressed
```

> **마운트 매핑 튜닝:** 기체가 마커 반대 방향으로 미끄러지면 런치 인자
> `lat_swap` / `lat_sign_fwd` / `lat_sign_left` 를 뒤집으세요. 디버그 로그의
> `off`(이미지 오프셋)가 `cmd`(명령 속도)에 의해 줄어드는지로 매핑이 맞는지 판단합니다
> (`--symlink-install` 빌드면 재빌드 불필요). 자세한 내용은 `docs/precision_landing.md`.

---

## 실기체 운용

### 연결 구성

```
[Pixhawk] --USB/UART--> [Jetson] --MAVROS--> [ROS 2 노드]
                            |
                       [USB 카메라 or RealSense]
```

### MAVROS 실행 (Pixhawk 연결)

```bash
# USB 시리얼 연결 (포트는 환경에 따라 변경)
ros2 launch mavros apm.launch fcu_url:=serial:///dev/ttyACM0:921600

# 또는 UDP (GCS 포워딩)
ros2 launch mavros apm.launch fcu_url:=udp://192.168.1.1:14550@14555
```

### YOLO 실기체 노드 실행 (TensorRT)

```bash
# yolo11n.engine 파일을 프로젝트 루트에 위치시키거나 경로를 지정
ros2 run camera_detection yolo_processor_node \
  --ros-args -p model_path:=/path/to/yolo11n.engine
```

### RealSense 뎁스 카메라 노드 실행

```bash
ros2 run camera_detection yolo_processor_depth
```

### 추적 타겟 ID 지정 (Commander)

```bash
ros2 run camera_detection commander
# Target ID >> 1   (숫자 입력 후 Enter → YOLO ID 1번 객체를 추적)
# Target ID >> q   (종료)
```

### 오프보드 추적 제어 실행

```bash
ros2 run offboard offboard_tracking_control
```

---

## PC ↔ Jetson 네트워크 설정 (Tailscale)

두 머신이 서로 다른 네트워크에 있을 때 ROS 2 토픽을 공유하려면 CycloneDDS + Tailscale VPN을 사용합니다.

### Tailscale 설치

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

### CycloneDDS 설정 적용

**PC에서:**

```bash
# config/cyclonedds_pc.xml 의 <Peer address> 를 Jetson의 Tailscale IP로 수정
export CYCLONEDDS_URI=~/ros2_ws/src/PX4-ROS2/config/cyclonedds_pc.xml
```

**Jetson에서:**

```bash
# config/cyclonedds_jetson.xml 의 <Peer address> 를 PC의 Tailscale IP로 수정
export CYCLONEDDS_URI=~/ros2_ws/src/PX4-ROS2/config/cyclonedds_jetson.xml
```

매 터미널마다 자동 적용하려면:

```bash
echo 'export CYCLONEDDS_URI=~/ros2_ws/src/PX4-ROS2/config/cyclonedds_pc.xml' >> ~/.bashrc
```

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `ros2 run offboard` 실행 후 아무 반응 없음 | MAVROS가 실행 중인지 확인: `ros2 topic echo /mavros/state` |
| MAVROS 연결 실패 (`timeout`) | `fcu_url` 포트/경로 확인, Pixhawk 시리얼 권한: `sudo usermod -aG dialout $USER` 후 재로그인 |
| Gazebo에 모델이 안 뜸 | `GZ_SIM_RESOURCE_PATH` 누락 → `run_sim.sh` 사용 |
| `ArduPilotPlugin` 로드 실패 | `~/ardupilot_gazebo/build` 미생성 → 5-3 재빌드 |
| SITL이 Gazebo와 연결 안 됨 | Gazebo를 **먼저** 실행 후 `sim_vehicle.py` 실행, 포트 9002 점유 확인 |
| `ros2 topic`에 카메라 없음 | `ros_gz_bridge` 빌드/`source` 여부, `GZ_VERSION=harmonic` 확인 |
| YOLO 모델 로드 실패 | `model_path` 파라미터로 절대 경로 지정, TensorRT 엔진은 Jetson에서만 동작 |
| 두 머신 간 토픽 안 보임 | CycloneDDS IP 설정 확인, `tailscale status`로 VPN 연결 확인 |
