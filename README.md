## Docker를 이용한 개발 (권장)

팀원 간의 동일한 개발 환경을 유지하기 위해 도커 사용을 권장합니다. 로컬 소스 코드가 컨테이너 내부와 공유되므로, 코드를 수정해도 이미지를 다시 빌드할 필요가 없습니다.

### 1. 도커 환경 실행
프로젝트 루트 디렉토리에서 다음 명령을 실행합니다:
```bash
docker-compose up --build
```
이 명령은 이미지를 빌드하고, 로컬 코드를 마운트한 뒤, 의존성 설치 및 빌드(`colcon build`)까지 자동으로 수행합니다.

### 2. 컨테이너 내부에서 작업
컨테이너가 실행 중인 상태에서 새로운 터미널을 열어 접속할 수 있습니다:
```bash
docker exec -it px4_ros2_offboard bash
```

---

## 수동 설치 및 설정 (도커 미사용 시)

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