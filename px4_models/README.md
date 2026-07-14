# PX4 커스텀 Gazebo 모델

이 폴더에는 PX4 SITL이 스폰하는 커스텀 기체 모델이 들어 있습니다. PX4는 자기
소스 트리(`~/PX4-Autopilot/Tools/simulation/gz/models/`) 안에서 모델을 찾으므로,
`gazebo/link_px4_model.sh`가 이 저장소 원본을 **심링크**로 연결합니다.

## 모델 목록

### `x500_city_rgbd_lidar`
PX4 기본 쿼드콥터 `x500`에 이 저장소가 직접 정의한 전방 RGB/depth, 하방
RGB/depth 및 단일빔 라이다 payload를 장착한 모델입니다. 외부 OAK-D/LW20
모델 include에 의존하지 않습니다.

- 전방 RGB: `/front_camera/image`, `/front_camera/camera_info`
- 전방 depth: `/front_depth/image`(32FC1), `/front_depth/preview`(mono8 표시용), `/front_depth/points`, `/front_depth/camera_info`
- 하방 RGB: `/down_camera/image`, `/down_camera/camera_info`
- 하방 depth: `/down_depth/image`(32FC1), `/down_depth/preview`(mono8 표시용), `/down_depth/points`, `/down_depth/camera_info`
- 하방 라이다: `/down_lidar`, `/down_lidar/points`; Gazebo 원본
  `.../lidar_sensor_link/sensor/lidar/scan`은 PX4 gz 브릿지가 구독해
  `distance_sensor`(하방, `ROTATION_DOWNWARD_FACING`)로
  발행합니다. **이 라이다 센서에 `<topic>` 오버라이드를 넣지 마세요.** 넣으면
  PX4가 distance_sensor를 못 잡습니다.

## 설치 방법

PX4-Autopilot 경로는 환경변수 `PX4_DIR`로 바꿀 수 있습니다(기본 `~/PX4-Autopilot`).

```bash
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
./gazebo/link_px4_model.sh "$PX4_DIR"
```

실행 스크립트도 매번 이 연결을 검증하고, 다른 심링크라면 원자적으로
교체합니다. 같은 이름의 실제 파일/디렉터리가 존재하면 사용자 자산을
삭제하지 않고 명확한 오류로 중단합니다.

### 설치 확인
```bash
readlink -f "$PX4_DIR/Tools/simulation/gz/models/x500_city_rgbd_lidar"
# 출력: <PX4-ROS2>/px4_models/x500_city_rgbd_lidar
```

## 사용

설치 후에는 별도 PX4 재빌드가 필요 없습니다(순정 airframe `4001`을 재사용하고,
모델은 `PX4_SIM_MODEL=gz_x500_city_rgbd_lidar`로 선택). 맵 런처가 자동으로 스폰합니다:

```bash
./gazebo/run_px4_map.sh mountain   # 또는 city
```

`run_px4_map.sh`는 시작 시 `x500_city_rgbd_lidar`가 현재 clone을 가리키는
심링크인지 강제 검증합니다. PX4 기본 `x500_depth_lidar`와 이름이
겹치지 않으므로 fresh clone과 기존 PX4 checkout에서 같은 자산을 로드합니다.
