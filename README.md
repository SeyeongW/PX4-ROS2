## Docker를 이용한 개발 (권장)

팀원 간의 동일한 개발 환경을 유지하고 복잡한 의존성 설치 과정을 생략하기 위해 도커 사용을 강력히 권장합니다. 

### 도커 사용의 장점
- **환경 통일**: 모든 팀원이 동일한 ROS 2, MAVROS, GeographicLib 버전을 사용하여 "환경 문제"를 방지합니다.
- **실시간 코드 반영**: 로컬 컴퓨터의 소스 코드가 컨테이너 내부와 **볼륨 마운트**로 연결되어 있어, 로컬에서 코드를 수정하면 컨테이너에 즉시 반영됩니다. (이미지 재빌드 불필요)
- **간편한 설정**: `docker-compose` 명령 하나로 모든 의존성 설치와 빌드 환경 준비가 완료됩니다.

### 1. 도커 환경 실행
프로젝트 루트 디렉토리에서 다음 명령을 실행합니다:
```bash
docker-compose up --build
```
- 처음 실행 시에는 이미지를 빌드하고 데이터셋을 다운로드하는 데 시간이 다소 소요될 수 있습니다.
- 이 명령은 로컬 코드를 마운트하고, 자동으로 `colcon build`까지 수행합니다.

### 2. 코드 수정 및 재빌드
1. 로컬 환경(예: VS Code)에서 코드를 자유롭게 수정합니다.
2. 수정된 코드를 반영하려면 실행 중인 컨테이너 터미널에서 다음 명령을 실행하여 다시 컴파일합니다:
   ```bash
   colcon build --symlink-install --packages-select offboard
   ```

### 3. 컨테이너 내부 터미널 접속
새로운 터미널 창에서 실행 중인 컨테이너에 접속하려면 다음 명령을 사용합니다:
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