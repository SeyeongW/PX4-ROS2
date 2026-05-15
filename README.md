# ArduPilot ROS 2 Offboard 제어 (Docker 기반 개발 가이드)

이 프로젝트는 ArduPilot 기반 기체를 ROS 2에서 Offboard 제어하기 위한 환경을 제공합니다. 모든 개발 환경은 **도커(Docker)**를 통해 통일되어 있어, 복잡한 설정 없이 즉시 시작할 수 있습니다.

---

## 🚀 빠른 시작 가이드 (A to Z)

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

### 3. 도커 컨테이너 빌드 및 실행
프로젝트 루트 디렉토리(`PX4-ROS2`)에서 다음 명령을 실행합니다.

```bash
docker-compose up --build
```
- **빌드(`--build`)**: `Dockerfile`을 바탕으로 ROS 2, MAVROS, 데이터셋 등이 설치된 이미지를 만듭니다. (최초 1회만 오래 걸림)
- **실행(`up`)**: 가상 컴퓨터(컨테이너)를 켭니다. 로컬의 소스 코드가 컨테이너 내부와 연결됩니다.

### 4. 컨테이너 내부에서 코드 빌드 및 실행
컨테이너가 실행 중인 상태에서 **새로운 터미널**을 열고 다음 명령을 통해 컨테이너 내부로 접속합니다.

```bash
# 1. 컨테이너 내부 접속
docker exec -it px4_ros2_offboard bash

# 2. 코드 빌드 (수정 시마다 실행)
cd /ros2_ws
colcon build --symlink-install --packages-select offboard
source install/setup.bash

# 3. 노드 실행 (MAVROS가 실행 중이어야 함)
ros2 run offboard offboard_control
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