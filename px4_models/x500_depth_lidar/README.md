# x500_depth_lidar — 기체 사양

PX4 기본 쿼드콥터 **Holybro X500**에 전방/하방 센서를 추가한 커스텀 기체입니다.
`run_px4_map.sh mountain|city`로 스폰되며, 순정 airframe `4001`을 재사용하므로
PX4 재빌드가 필요 없습니다(`PX4_SIM_MODEL=gz_x500_depth_lidar`로 모델 선택).

설치 방법은 상위 [`../README.md`](../README.md) 참고.

## 센서 구성

| 위치 | 센서 | 종류 | ROS 2 토픽 | 해상도/사양 |
|------|------|------|-----------|------------|
| 전방 | OAK-D Lite 뎁스 | depth camera | `/depth_camera`, `/depth_camera/points`, `/camera_info` | 640×480, 32FC1 |
| 전방 | OAK-D Lite RGB | camera (IMX214) | `/front_camera/image`, `/front_camera/camera_info` | 1920×1080 |
| 하방 | 모노 카메라 | camera | `/down_camera/image`, `/down_camera/camera_info` | 1280×960, rgb8 |
| 하방 | LW20 라이다 | gpu_lidar (단일 빔) | `/down_lidar`, `/down_lidar/points` | 0.1–100 m |

- ROS 2 토픽은 `gazebo/launch/sensor_bridge.launch.py`(ros_gz_bridge)로 브릿지되며
  `run_px4_map.sh`가 자동 실행합니다.
- **하방 라이다는 PX4 `distance_sensor`로도 연동**됩니다(하방 인식, orientation=25).
  PX4 gz 브릿지가 라이다의 기본 gz 경로를 구독하므로, 라이다 센서에는 `<topic>`
  오버라이드를 넣지 마세요.
- 하방 카메라는 `<topic>down_camera/image</topic>`를 써서 `camera_info`가
  `/down_camera/camera_info`로 분리됩니다(전방 뎁스캠의 `/camera_info`와 충돌 방지).

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
