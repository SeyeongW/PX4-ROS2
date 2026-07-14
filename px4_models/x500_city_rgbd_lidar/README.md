# x500_city_rgbd_lidar — 기체 사양

PX4 기본 쿼드콥터 **Holybro X500**에 전방/하방 센서를 추가한 커스텀 기체입니다.
`run_px4_map.sh mountain|city`로 스폰되며, 순정 airframe `4001`을 재사용하므로
PX4 재빌드가 필요 없습니다(`PX4_SIM_MODEL=gz_x500_city_rgbd_lidar`로 모델 선택).

설치 방법은 상위 [`../README.md`](../README.md) 참고.

## 센서 구성

| 위치 | 센서 | 종류 | ROS 2 토픽 | 해상도/사양 |
|------|------|------|-----------|------------|
| 전방 | RGB 카메라(수평) | camera | `/front_camera/image`, `/front_camera/camera_info` | 640×360 @ 15 Hz |
| 전방 | Depth 카메라 | depth camera | `/front_depth/image`, `/front_depth/points`, `/front_depth/camera_info` | 320×240, 32FC1 @ 12 Hz |
| 하방 | RGB 카메라 | camera | `/down_camera/image`, `/down_camera/camera_info` | 640×480 @ 20 Hz |
| 하방 | Depth 카메라 | depth camera | `/down_depth/image`, `/down_depth/points`, `/down_depth/camera_info` | 320×240, 32FC1 @ 15 Hz |
| 하방 | LW20 라이다 | gpu_lidar (단일 빔) | `/down_lidar`, `/down_lidar/points` | 0.1–100 m |

- ROS 2 토픽은 `gazebo/launch/sensor_bridge.launch.py`의 gz-sim8 네이티브 브리지로
  전달되며 `run_px4_map.sh`가 자동 실행합니다. 센서별 Gazebo 구독은 ROS 구독자가
  있을 때만 활성화되어 비사용 RGB/point cloud의 CPU 복사를 피합니다.
- **하방 라이다는 PX4 `distance_sensor`로도 연동**됩니다(하방 인식, orientation=25).
  PX4 gz 브릿지가 라이다의 기본 gz 경로를 구독하므로, 라이다 센서에는 `<topic>`
  오버라이드를 넣지 마세요.
- 전방/하방 RGB와 Depth는 각각 별도 link, sensor, Gazebo topic, optical frame을
  사용합니다. 하방 거리 추정은 단안 영상의 크기 환산이 아니라 실제
  `depth_camera` 영상/point cloud와 단일빔 LiDAR를 함께 사용합니다.
- 전방 Depth의 Gazebo native cloud `/front_depth/image/points`는 PX4 임무의 경량
  native adapter가 직접 구독하고, ROS bridge에서는 `/front_depth/points`로 remap합니다.
- 320×240와 제한된 update rate는 software/headless renderer에서도 sensor freshness를
  유지하기 위한 값입니다. adapter는 2×2 spatial stride 뒤에도 중앙 ROI의 3-percentile
  장애물 검출을 유지합니다.

## 카메라와 센서 확인

먼저 한 터미널에서 도시맵, PX4, MAVROS 및 센서 브리지를 함께 실행합니다.

```bash
cd ~/PX4-ROS2
./gazebo/run_px4_map.sh city
```

`PX4 SITL instance 0 is already running`이 나오면 다른 터미널의 동일 런처가 아직
실행 중인 것입니다. 그 터미널에서 `Ctrl-C`로 정상 종료하고 다시 실행하세요.
런처는 중복 PX4와 UDP 14540/14580 충돌을 Gazebo 시작 전에 차단합니다.

전방과 하방 RGB를 동시에 보려면 각각 별도 터미널에서 실행합니다.

```bash
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view --clear-config /front_camera/image
```

```bash
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /down_camera/image
```

- 실제 raw 토픽 이름은 `/front_camera/image`와 `/down_camera/image`이며
  `_raw` 접미사가 없습니다.
- 브리지는 구독자 요구 기반이므로 토픽을 선택한 뒤 첫 영상까지 1~3초 걸릴 수
  있습니다. 인자 없이 rqt를 실행하면 이전에 저장된 토픽이 다시 선택될 수 있습니다.
- `/front_depth/image` 또는 `/down_depth/image`는 `32FC1` 거리 영상입니다.
  rqt의 **Dynamic range**를 켜야 검은 화면처럼 보이지 않습니다.

전체 payload가 실제 메시지를 발행하는지 빠르게 확인하려면 다음을 사용합니다.

```bash
for topic in \
  /front_camera/image /front_camera/camera_info \
  /front_depth/image /front_depth/points /front_depth/camera_info \
  /down_camera/image /down_camera/camera_info \
  /down_depth/image /down_depth/points /down_depth/camera_info \
  /down_lidar /down_lidar/points
do
  echo "===== $topic"
  timeout 10 ros2 topic echo "$topic" --once \
    --qos-profile sensor_data --no-arr
done
```

MAVROS/PX4 기본 센서 연결은 아래 값들이 메시지를 내고 `connected: true`인지로
확인합니다. 기본 실행은 MAVROS 전용이므로 `/fmu/*` 토픽이 없는 것이 정상입니다.

```bash
ros2 topic echo /mavros/state --once
timeout 10 ros2 topic hz /mavros/imu/data --wall-time
timeout 10 ros2 topic hz /mavros/local_position/pose --wall-time
timeout 10 ros2 topic hz /mavros/global_position/global --wall-time
```

## 물리 제원 (시뮬레이션 모델)

| 항목 | 값 |
|------|-----|
| 기체 질량 | base_link 2.0 kg (+ 로터 ~0.06, 센서 ~0.16 → 총 ≈ 2.2 kg) |
| 관성 | Ixx = Iyy = 0.0217, Izz = 0.04 kg·m² |
| 로터 | 4개, 위치 X ±0.13 m / Y ±0.22 m (휠베이스 ≈ 0.5 m) |
| 호버 스로틀 | 0.6 (`MPC_THR_HOVER`) |

## 비행 성능 한계 (PX4 MPC 파라미터 기본값)

| 항목 | 파라미터 | 값 |
|------|----------|-----|
| **최대 수평속도** | `MPC_XY_VEL_MAX` | **12 m/s** (≈ 43 km/h) |
| Position 모드 수평속도 | `MPC_VEL_MANUAL` | 10 m/s |
| 자동비행 순항속도 | `MPC_XY_CRUISE` | 5 m/s |
| 최대 상승속도 | `MPC_Z_VEL_MAX_UP` | 3 m/s |
| 최대 하강속도 | `MPC_Z_VEL_MAX_DN` | 1.5 m/s |
| 최대 기울기 | `MPC_TILTMAX_AIR` | 45° |
| 최대 수평 가속도 | `MPC_ACC_HOR_MAX` | 5 m/s² |
| 이륙 속도 | `MPC_TKO_SPEED` | 1.5 m/s |
| 착륙 속도 | `MPC_LAND_SPEED` | 0.7 m/s |

속도 한계는 파라미터라 런타임에 바꿀 수 있습니다(기체 추력 한계 내):

```bash
# PX4 콘솔(pxh>)에서
param set MPC_XY_VEL_MAX 20
```

기본 세팅에서 실제 최고속도는 45° 기울기의 추력-항력 균형으로 결정되며,
사실상 MPC 한계인 12 m/s가 상한입니다.
