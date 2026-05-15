# ArduPilot ROS 2 Offboard 제어

이 저장소는 **MAVROS**를 사용하여 ArduPilot 기반 기체(Copter/Rover/Plane)를 ROS 2에서 Offboard 제어하기 위한 구현을 제공합니다.

## 사전 요구 사항

이 패키지를 사용하기 전에 시스템에 ROS 2(Humble 권장)가 설치되어 있어야 합니다.

### 1. MAVROS 및 의존성 설치
먼저 MAVROS 패키지와 메시지 정의를 설치합니다:
```bash
sudo apt-get update
sudo apt-get install ros-humble-mavros ros-humble-mavros-msgs
```

### 2. GeographicLib 데이터셋 설치
MAVROS는 좌표 변환(예: GPS에서 로컬 프레임으로)을 위해 GeographicLib 데이터셋이 필요합니다. 다음 스크립트를 실행하여 설치하십시오:
```bash
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
```

## 설정 및 설치

다음 단계에 따라 ROS 2 워크스페이스에 저장소를 클론하고 빌드하십시오.

### 1. 워크스페이스 생성 (이미 있는 경우 생략)
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2. 저장소 클론
워크스페이스의 `src` 폴더에 저장소를 클론합니다:
```bash
git clone -b wang https://github.com/SeyeongW/PX4-ROS2.git
```

### 3. 패키지 의존성 설치
`rosdep`을 사용하여 나머지 의존성을 자동으로 설치합니다:
```bash
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### 4. 워크스페이스 빌드
```bash
colcon build --packages-select offboard
source install/setup.bash
```

## Offboard 제어 노드 실행

비행 제어기(SITL 또는 실제 하드웨어)가 MAVROS를 통해 연결되어 있는지 확인하십시오.

### 1. MAVROS 실행
MAVROS 노드를 실행합니다 (SITL 예시):
```bash
ros2 launch mavros apm_sitl.launch
```

### 2. 제어 노드 실행
새 터미널에서 다음을 실행합니다:
```bash
ros2 run offboard offboard_control
```

## 노드 로직 설명

`offboard_control` 노드는 다음과 같은 비행 단계를 처리하기 위해 상태 머신(State Machine)을 구현합니다:
- **Warmup (준비)**: 모드를 `GUIDED`로 설정하고 기체를 Arming 합니다.
- **Takeoff (이륙)**: 목표 고도(기본값: 50m)까지 고도를 높입니다.
- **Hold (대기)**: 미션을 시작하기 전 잠시 대기합니다.
- **Move (이동)**: 시작 위치와 헤딩을 기준으로 론모어(Lawnmower) 경로를 실행합니다.
- **Landing (RTL/복귀)**: 미션이 완료되면 `RTL`(Return to Launch) 모드로 전환하여 복귀합니다.

모든 좌표는 ROS에서 사용하는 **ENU**(East-North-Up) 표준을 따릅니다.