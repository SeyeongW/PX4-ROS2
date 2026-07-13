# PX4 커스텀 Gazebo 모델

이 폴더에는 PX4 SITL이 스폰하는 커스텀 기체 모델이 들어 있습니다. PX4는 자기
소스 트리(`~/PX4-Autopilot/Tools/simulation/gz/models/`) 안에서 모델을 찾기
때문에, 이 폴더의 모델을 **각자 컴퓨터의 PX4-Autopilot에 복사(또는 심링크)**
해야 시뮬레이션에서 사용할 수 있습니다.

## 모델 목록

### `x500_depth_lidar`
PX4 기본 쿼드콥터 `x500` + **전방 뎁스카메라(OAK-D Lite)** + **하방 라이다(LW20)**.

- 전방 뎁스카메라: gz `/depth_camera`(Image), `/depth_camera/points`, `/camera_info`
- 전방 RGB카메라: OAK-D Lite IMX214 (월드 경로 토픽)
- 하방 라이다: `.../lidar_sensor_link/sensor/lidar/scan` — PX4 gz 브릿지가 이
  기본 경로를 구독해 `distance_sensor`(하방, `ROTATION_DOWNWARD_FACING`)로
  발행합니다. **이 라이다 센서에 `<topic>` 오버라이드를 넣지 마세요.** 넣으면
  PX4가 distance_sensor를 못 잡습니다.

## 설치 방법 (각자 컴퓨터에서 1회)

PX4-Autopilot 경로는 환경변수 `PX4_DIR`로 바꿀 수 있습니다(기본 `~/PX4-Autopilot`).

### 방법 A — 심링크 (권장: 저장소 업데이트가 바로 반영됨)
```bash
REPO="$(cd "$(dirname "$0")/.." && pwd)"   # 또는 PX4-ROS2 레포 경로
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
ln -s "$REPO/px4_models/x500_depth_lidar" \
      "$PX4_DIR/Tools/simulation/gz/models/x500_depth_lidar"
```

### 방법 B — 복사
```bash
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
cp -r px4_models/x500_depth_lidar \
      "$PX4_DIR/Tools/simulation/gz/models/"
```

### 설치 확인
```bash
ls "$HOME/PX4-Autopilot/Tools/simulation/gz/models/x500_depth_lidar/model.sdf"
```

## 사용

설치 후에는 별도 PX4 재빌드가 필요 없습니다(순정 airframe `4001`을 재사용하고,
모델은 `PX4_SIM_MODEL=gz_x500_depth_lidar`로 선택). 맵 런처가 자동으로 스폰합니다:

```bash
./gazebo/run_px4_map.sh mountain   # 또는 city
```

`run_px4_map.sh`는 시작 시 이 모델(`x500_depth_lidar`)이 PX4 트리에 있는지
확인하며, 없으면 위 설치 안내와 함께 종료합니다.
