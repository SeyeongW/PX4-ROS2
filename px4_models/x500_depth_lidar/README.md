# x500_depth_lidar — 기체 사양

PX4 기본 쿼드콥터 **Holybro X500**에 전방/하방 센서를 추가한 커스텀 기체입니다.
`run_px4_map.sh mountain|city`로 스폰되며, 순정 airframe `4001`을 재사용하므로
PX4 재빌드가 필요 없습니다(`PX4_SIM_MODEL=gz_x500_depth_lidar`로 모델 선택).

설치 방법은 상위 [`../README.md`](../README.md) 참고.

## 센서 구성

| 위치 | 센서 | 종류 | ROS 2 토픽 | 해상도/사양 |
|------|------|------|-----------|------------|
| 전방 | RGB 카메라 | camera | `/front_camera/image`, `/front_camera/camera_info` | 640×360 @ 15 Hz |
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
