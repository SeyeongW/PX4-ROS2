# ArduPilot ROS 2 Offboard 제어 (Docker 기반 개발 가이드)

이 프로젝트는 ArduPilot 기반 기체를 ROS 2에서 Offboard 제어하기 위한 환경을 제공합니다. 모든 개발 환경은 **도커(Docker)**를 통해 통일되어 있어, 복잡한 설정 없이 즉시 시작할 수 있습니다.

---

## 빠른 시작 가이드

팀원들과 함께 다음 단계를 순서대로 진행하여 환경을 구축하십시오.

### 1. 도커 설치 (최초 1회)
Ubuntu 시스템에 도커와 도커 컴포즈를 설치합니다. (이미 설치되어 있다면 생략)

```bash
# 도커 설치 스크립트 실행
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 docker 그룹에 추가 (로그아웃 후 재로그인 필요)
sudo usermod -aG docker $USER

# 도커 컴포즈 설치
sudo apt-get update
sudo apt-get install -y docker-compose
```
> [!IMPORTANT]
> `usermod` 명령 실행 후, 시스템을 **로그아웃 했다가 다시 로그인**해야 `sudo` 없이 도커 명령어를 사용할 수 있습니다.

### 2. 프로젝트 클론
워크스페이스 폴더를 만들고 프로젝트를 클론합니다.

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone -b wang https://github.com/SeyeongW/PX4-ROS2.git
cd PX4-ROS2
```

### 3. 도커 컨테이너 실행

상황에 맞는 방법을 선택하십시오.

#### 방법 A: 팀원용 (이미 빌드된 이미지 사용 - 권장 👍)
의존성 설치나 데이터셋 다운로드를 기다릴 필요 없이 즉시 시작할 수 있습니다.
1. `docker-compose.yml` 설정이 `image: seyeongw/...`로 되어 있는지 확인합니다. (기본값)
2. 실행 (이미지가 없으면 자동으로 다운로드합니다):
   ```bash
   docker-compose up
   ```

#### 방법 B: 관리자용 (이미지 수정 및 배포)
환경 설정(`Dockerfile`)을 직접 수정해야 할 때 사용합니다.
1. `docker-compose.yml`에서 `image:` 줄을 주석 처리하고, `build:` 섹션 주석을 해제합니다.
2. 빌드 및 실행:
   ```bash
   docker-compose up --build
   ```
3. 수정 완료 후 팀원들에게 배포:
   ```bash
   ./push_image.sh
   ```

---

### 4. 컨테이너 내부 접속 및 작업
컨테이너가 실행 중인 상태에서 **새로운 터미널**을 열어 접속합니다.

```bash
# 1. 단축 명령어로 접속
chmod +x in.sh  # (최초 1회 실행)
./in.sh

# 2. 코드 빌드 및 실행
cd /ros2_ws
colcon build --symlink-install --packages-select offboard
source install/setup.bash
ros2 run offboard offboard_control
```

```bash
# 1. 컨테이너 내부 접속 (언제든 필요할 때 실행)
docker exec -it px4_ros2_offboard bash

# 2. 코드 빌드 및 실행
cd /ros2_ws
colcon build --symlink-install --packages-select offboard
source install/setup.bash
ros2 run offboard offboard_control
```

### 5. 상태 확인 및 로그 보기
백그라운드에서 돌아가는 서버의 상태를 확인하고 싶을 때 사용합니다.

```bash
# 실행 중인 컨테이너 상태 확인
docker ps

# 실시간 로그 확인 (디버깅 시 유용)
docker logs -f px4_ros2_offboard

# 서버 종료
docker-compose down
```

---

## 💡 개발 팁

### 실시간 코드 반영
로컬 컴퓨터(Host)에서 VS Code 등으로 코드를 수정하면, 컨테이너 내부의 코드도 **즉시 반영**됩니다. 이미지를 다시 빌드할 필요 없이 컨테이너 내부 터미널에서 `colcon build`만 다시 해주면 됩니다.

### 하드웨어 연결 (FCU)
이 도커 설정은 `privileged: true` 및 `network_mode: host`를 사용하므로, USB로 연결된 픽스호크(Pixhawk) 등의 하드웨어를 별도 설정 없이 컨테이너 내부에서 인식할 수 있습니다.

### GUI 도구 사용 (RViz2 등)
X11 포워딩 설정이 포함되어 있어, 호스트의 디스플레이를 통해 컨테이너 내부의 RViz2나 Gazebo 창을 띄울 수 있습니다.

---

## 수동 설치 가이드 (도커 미사용 시)
*도커를 사용하지 않는 환경은 기존 리드미 내용을 참조하십시오... (생략)*

---

# ArduPilot SITL + Gazebo 하방 카메라 시뮬레이션

ArduPilot SITL과 Gazebo(gz-sim Harmonic)를 연동해, **기존 Iris 쿼드로터에 하방(아래)을 바라보는 RGB 카메라를 부착한 모델**을 시뮬레이션에 띄우는 환경입니다. 카메라 영상은 `ros_gz_bridge`를 통해 ROS 2 토픽으로 전달되어 기존 YOLO 검출 노드와 연동할 수 있습니다.

> **검증 환경:** Ubuntu 22.04 (jammy) · ROS 2 Humble · Gazebo Harmonic (gz-sim 8) · ArduPilot 4.6 beta

## 1. 구성 자산 (이 저장소)

```
gazebo/
├── models/iris_with_down_camera/   # iris + 하방 카메라 모델 (SDF)
│   ├── model.config
│   └── model.sdf
├── worlds/iris_down_camera_runway.sdf  # 카메라 검증용 월드 (지면 + 마커)
├── launch/camera_bridge.launch.py  # gz 카메라 → ROS 2 브리지
├── install_apt_deps.sh            # sudo 필요한 apt 설치 일괄 스크립트
└── run_sim.sh                      # 리소스 경로 설정 + Gazebo 실행 헬퍼
```

> 아래 환경은 실제로 빌드·실행하여 검증되었습니다: Gazebo에서 iris+하방 카메라가 로드되고
> `/down_camera/image`(640×480 rgb8)가 ROS 2 토픽으로 정상 발행됨을 확인했습니다.

- `iris_with_down_camera` 모델은 `ardupilot_gazebo`의 `iris_with_standoffs` 에어프레임을 그대로 include 하고, `base_link` 아래에 **고정(fixed) 조인트**로 카메라 링크를 부착합니다. 카메라 링크를 Y축으로 +90° 피치시켜 광학 +X축이 정확히 지면(-Z)을 향하도록 했습니다.
- 비행에 필요한 lift-drag / motor / `ArduPilotPlugin` 설정은 `iris_with_ardupilot`과 동일하므로, 별도 수정 없이 SITL과 바로 연동됩니다.

## 2. 사전 설치 (드라이버 / 패키지)

> 아래 `apt` 명령들은 `sudo`가 필요합니다. 터미널에서 직접 실행하십시오.

> **빠른 길:** `sudo`가 필요한 apt 설치(2-1, 2-3의 헤더, 빌드 도구)는 한 번에 처리하는 스크립트를 제공합니다.
> ```bash
> sudo bash gazebo/install_apt_deps.sh
> ```
> 이후 2-2 / 2-3 / 2-4의 **빌드 단계만** sudo 없이 진행하면 됩니다. 아래는 개별 설치 설명입니다.

### 2-1. Gazebo Harmonic 설치 (gz-sim 8)

```bash
sudo apt-get update
sudo apt-get install -y curl lsb-release gnupg

sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt-get update
sudo apt-get install -y gz-harmonic
gz sim --version   # Gazebo Sim, version 8.x 확인
```

### 2-2. ArduPilot 소스 클론 & SITL 빌드

```bash
cd ~
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot

# 의존성 자동 설치 (Python, 빌드 도구, MAVProxy 등)
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile          # sim_vehicle.py 등을 PATH에 등록

# SITL(소프트웨어 시뮬레이션) 빌드
./waf configure --board sitl
./waf copter
```

빌드가 끝나면 `sim_vehicle.py`로 ArduCopter SITL을 실행할 수 있습니다.

### 2-3. ardupilot_gazebo 플러그인 빌드 (Harmonic용)

`ArduPilotPlugin`과 모터/센서 시스템을 제공하는 Gazebo 플러그인입니다.

```bash
# 빌드 의존성 (gz-sim8 = Harmonic)
sudo apt-get install -y libgz-sim8-dev rapidjson-dev cmake build-essential

cd ~
git clone https://github.com/ArduPilot/ardupilot_gazebo.git
cd ardupilot_gazebo
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
```

빌드 산출물은 `~/ardupilot_gazebo/build`에 생성됩니다. (`run_sim.sh`가 이 경로를 자동으로 `GZ_SIM_SYSTEM_PLUGIN_PATH`에 등록합니다.)

### 2-4. ros_gz 브리지 (ROS 2 Humble + Gazebo Harmonic)

> Humble용 `ros-humble-ros-gz` 바이너리는 Gazebo **Fortress** 기준이므로, Harmonic과 쓰려면 **소스 빌드**가 필요합니다.

```bash
sudo apt-get install -y python3-rosdep python3-colcon-common-extensions
source /opt/ros/humble/setup.bash

export GZ_VERSION=harmonic
mkdir -p ~/ros_gz_ws/src && cd ~/ros_gz_ws/src
git clone https://github.com/gazebosim/ros_gz.git -b humble
cd ~/ros_gz_ws

# 카메라 브리지만 필요하면 해당 패키지까지만 빌드(빠름)
GZ_VERSION=harmonic colcon build --merge-install --packages-up-to ros_gz_bridge
# 사용 시: source ~/ros_gz_ws/install/setup.bash
```

## 3. 시뮬레이션 실행

3개의 터미널을 사용합니다.

**터미널 1 — Gazebo (하방 카메라 + Iris 월드)**

```bash
cd ~/ros2_ws/src/PX4-ROS2/gazebo   # (경로는 환경에 맞게)
./run_sim.sh
```

`run_sim.sh`가 `GZ_SIM_RESOURCE_PATH`(우리 모델/월드 + `ardupilot_gazebo` 모델)와 `GZ_SIM_SYSTEM_PLUGIN_PATH`(플러그인 빌드 경로)를 설정한 뒤 `iris_down_camera_runway.sdf`를 실행합니다.

**터미널 2 — ArduPilot SITL**

```bash
cd ~/ardupilot
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console
```

SITL은 `JSON` 백엔드로 9002 포트의 `ArduPilotPlugin`과 통신합니다. MAVProxy 콘솔에서 시동/이륙 테스트:

```
mode guided
arm throttle
takeoff 10
```

**터미널 3 — 카메라 ROS 2 브리지**

```bash
source /opt/ros/humble/setup.bash
source ~/ros_gz_ws/install/setup.bash
ros2 launch ~/ros2_ws/src/PX4-ROS2/gazebo/launch/camera_bridge.launch.py
```

## 4. 카메라 토픽 확인

```bash
ros2 topic list | grep down_camera
#   /down_camera/image
#   /down_camera/camera_info

ros2 topic hz /down_camera/image          # ~30 Hz
ros2 run rqt_image_view rqt_image_view /down_camera/image
```

월드에는 지면과 빨강/파랑 마커 박스를 배치해 두어, 하방 카메라가 정상 동작하면 영상에서 마커들을 바로 확인할 수 있습니다. 이 토픽을 `camera_detection` 패키지의 YOLO 노드 입력으로 연결하면 시뮬레이션 영상에서 객체 검출이 가능합니다.

## 5. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| Gazebo에 모델이 안 뜸 | `GZ_SIM_RESOURCE_PATH`에 모델 경로 누락 → `run_sim.sh` 사용 |
| `ArduPilotPlugin` 로드 실패 | `~/ardupilot_gazebo/build` 미생성/경로 누락 → 2-3 재빌드 |
| 카메라 영상이 검정/없음 | 월드에 `gz-sim-sensors-system` 필요(본 월드엔 포함됨), GPU/OGRE2 렌더 환경 확인 |
| SITL이 연결 안 됨 | Gazebo를 **먼저** 실행 후 `sim_vehicle.py` 실행, 포트 9002 점유 확인 |
| `ros2 topic`에 카메라 없음 | `ros_gz_bridge`(2-4) 소스 빌드/`source` 여부, `GZ_VERSION=harmonic` 확인 |
