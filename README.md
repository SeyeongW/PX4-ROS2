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
cd ~/ros2_ws/src/PX4-ROS2/gazebo
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
ros2 launch mavros apm.launch fcu_url:=udp://127.0.0.1:14550@14555
```

### 터미널 4 — 카메라 브리지 실행

Gazebo 카메라 영상을 ROS 2 토픽으로 전달합니다.

```bash
source ~/ros_gz_ws/install/setup.bash
ros2 launch ~/ros2_ws/src/PX4-ROS2/gazebo/launch/camera_bridge.launch.py
```

### 카메라 토픽 확인

```bash
ros2 topic list | grep down_camera
# /down_camera/image
# /down_camera/camera_info

ros2 topic hz /down_camera/image   # ~30 Hz 확인
```

### YOLO 시뮬레이션 노드 실행 (선택)

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
