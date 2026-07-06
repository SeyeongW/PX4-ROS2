# ArUco 정밀착륙 — 실기체 하드웨어 버전

SITL 정밀착륙(`precision_landing_node` + `aruco_detector_node`)을 **실기체(Jetson +
CSI/USB 카메라, ArduPilot/MAVROS)** 에 맞게 옮긴 버전입니다. 정지 ArUco 마커를
하방 카메라로 보고 그 위에 정밀 착륙합니다.

- 검출: `camera_detection/aruco_pose_node.py` — 카메라 캘리브레이션 기반 pose 추정
- 제어: `precision_landing/precland_hw_node.py` — GUIDED 속도 서보 + LAND 인계
- 런치: `precision_landing/launch/precland_hw.launch.py` — 카메라+MAVROS+검출+제어

---

## 1. SITL 버전과 무엇이 다른가

| 항목 | SITL (`precision_landing_node`) | 실기체 (`precland_hw_node`) |
|------|--------------------------------|-----------------------------|
| 마커 측정 | 정규화 오프셋 + `cam_hfov` 근사 | **캘리브레이션 pose(solvePnP) 실측 3D(m)** |
| 마커 위 높이 | FC 고도 `pos.z − platform_height` | **카메라 pose `tvec.z`**(FC 고도 오차 무관) |
| 이동 마커(cue/APPROACH) | 있음(moving_marker) | **제거**(정지 마커 전용, 단순·안전) |
| 이륙 | `auto_takeoff=true` 기본 | **`auto_takeoff=false` 기본**(조종사 이륙 후 인계) |
| 터치다운 | 공중 강제 disarm(플랫폼 위) | **LAND 모드 인계**(평지, FC 가 접지·disarm) |
| 조종사 오버라이드 | — | **GUIDED 이탈/disarm 시 즉시 중단**(RC 회수) |

카메라 광학 프레임(OpenCV) 규약: `tvec = (오른쪽 m, 아래 m, 마커까지 거리 m)`.
하방 카메라를 정하방으로 달면 `tvec.z ≈ 마커 윗면까지의 높이`.

---

## 2. 상태 기계

```
(TAKEOFF) → IDLE → ALIGN → DESCEND → LAND → DONE
```

| 단계 | 하는 일 |
|------|---------|
| **TAKEOFF** | `auto_takeoff=true` 일 때만. GUIDED→arm→`flight_alt` 이륙. |
| **IDLE** | armed + GUIDED + 마커 감지 될 때까지 제자리(속도 0) 대기. |
| **ALIGN** | `flight_alt` 호버하며 마커 중심으로 수평 정렬(속도 서보). |
| **DESCEND** | 깔때기(funnel) 허용오차 안에서 정렬하며 동시에 하강. |
| **LAND** | `land_switch_alt` 아래 + 중심 정렬 시 **LAND 모드로 전환**, 셋포인트 중단, FC 인계. |
| **DONE** | disarm 확인 후 노드 종료. |

**안전:** ALIGN/DESCEND 중 GUIDED 를 벗어나거나(조종사 모드 스위치) disarm 되면 즉시
셋포인트 송출을 멈추고 IDLE 로 복귀. 조종사가 언제든 회수 가능.

---

## 3. 하드웨어 준비 (실비행 전 체크리스트)

1. **카메라 캘리브레이션** (가장 중요 — pose 정확도의 핵심):
   ```bash
   ros2 run camera_calibration cameracalibrator \
       --size 8x6 --square 0.025 \
       image:=/down_camera/image_raw camera:=/down_camera
   ```
   CALIBRATE → SAVE 후 `ost.yaml` 의 `camera_matrix`/`distortion_coefficients` 를
   `camera_detection/config/down_camera.yaml` 형식으로 옮겨 저장(또는 그 경로를
   `calib_file:=` 로 지정). **샘플 값 그대로 쓰면 거리/정렬이 틀립니다.**

2. **마커 실측 크기**: 인쇄한 마커 한 변을 자로 재서 `marker_size:=`(m)에 정확히.

3. **카메라 마운트 매핑 확정**(지상 테스트): 기체를 손으로 들고 마커 위에서
   움직이며 `/precision_landing/debug` 의 `err`/`cmd` 부호를 확인:
   - 마커를 못 잡고 빙글빙글 → `lat_swap:=true`
   - 하강 중 전방으로 발산 → `lat_sign_fwd:=-1.0`
   - 가로로 직선 발산 → `lat_sign_left:=-1.0`

4. **ArduPilot 파라미터**: `GUID_OPTIONS` 로 속도 셋포인트 허용, EKF 원점/고도 정상,
   LAND 관련 `LAND_SPEED` 확인.

---

## 4. 실행

```bash
source ~/ros2_ws/PX4-ROS2/install/setup.bash

# Jetson CSI 카메라(GStreamer)
ros2 launch precision_landing precland_hw.launch.py \
    camera_driver:=gscam \
    fcu_url:=/dev/ttyTHS1:921600 \
    marker_size:=0.20 \
    calib_file:=/abs/경로/down_camera.yaml

# USB 카메라
ros2 launch precision_landing precland_hw.launch.py \
    camera_driver:=v4l2 video_device:=/dev/video0 \
    fcu_url:=/dev/ttyTHS1:921600 marker_size:=0.20

# 이미 카메라 노드가 따로 돈다면
ros2 launch precision_landing precland_hw.launch.py \
    camera_driver:=none \
    image_topic:=/내/카메라/image_raw camera_info_topic:=/내/카메라/camera_info
```

비행 순서: 조종사가 **직접 이륙 → GUIDED 전환** → 마커가 하방 카메라에 들어오면
노드가 자동으로 ALIGN→DESCEND→LAND 인계. 이상 시 **모드 스위치로 즉시 회수**.

---

## 5. 토픽 인터페이스

| 방향 | 토픽 | 타입 | 의미 |
|------|------|------|------|
| in | `/perception/marker_pose` | `geometry_msgs/PoseStamped` | 마커 pose(카메라 프레임 tvec) |
| in | `/perception/aruco_detected` | `std_msgs/Bool` | 마커 검출 여부 |
| in | `/mavros/state` | `mavros_msgs/State` | 연결/시동/모드 |
| in | `/mavros/local_position/pose` | `geometry_msgs/PoseStamped` | 로컬 ENU 위치(BEST_EFFORT) |
| out | `/mavros/setpoint_raw/local` | `mavros_msgs/PositionTarget` | 속도 + yaw 셋포인트 |
| out | `/precision_landing/debug` | `std_msgs/String` | 단계 전환/정렬 디버그 |
| (검출) out | `/perception/aruco_debug/compressed` | `sensor_msgs/CompressedImage` | 디버그 영상(축·거리 오버레이) |

---

## 6. 주요 파라미터 (`precland_hw_node`)

| 파라미터 | 기본 | 역할 |
|----------|------|------|
| `auto_takeoff` | `false` | true 면 노드가 스스로 GUIDED→arm→이륙(무인 테스트용). |
| `require_guided` | `true` | GUIDED 아니면 셋포인트 미송출(조종사 회수 안전). |
| `flight_alt` | `4.0` | 이륙/정렬 호버 고도(m). |
| `land_switch_alt` | `1.0` | 마커 위 이 높이 아래 + 중심 정렬 시 LAND 모드 인계. |
| `land_align_radius` | `0.15` | 착륙 확정 수평오차 게이트(m) = 깔때기 바닥. |
| `descend_cone` | `0.35` | 깔때기 반각 기울기(m오차/m고도). |
| `descend_rate` | `0.25` | 하강 속도(m/s). |
| `vel_gain` | `0.6` | 수평 속도 서보 게인(1/s). 진동↓ / 굼뜸↑. |
| `vel_max` | `0.8` | 수평 속도 상한(m/s). 실기체는 보수적으로. |
| `kf_meas_std` | `0.05` | pose 측정 잡음(m). pose 는 정밀하므로 작게. |
| `coast_ticks` | `20` | 검출 없이 예측만으로 버티는 최대 틱(~1 s). |
| `lat_swap`/`lat_sign_fwd`/`lat_sign_left` | `false`/`1`/`1` | 카메라 마운트 이미지→기체 매핑(3장). |

검출(`aruco_pose_node`): `marker_size`, `marker_id`(-1=아무 마커), `aruco_dict`,
`camera_info_topic`, `calib_file`.

---

## 7. 튜닝 가이드 (증상 → 대응)

| 증상 | 대응 |
|------|------|
| 좌우 진동 | `vel_gain` ↓, `vel_max` ↓ |
| 정렬이 굼뜸 | `vel_gain` ↑ |
| 마커 못 잡고 빙글빙글 | `lat_swap:=true` |
| 한쪽으로 직선 발산 | 해당 축 `lat_sign_*` 부호 반전 |
| 거리(tvec.z)가 이상 | 캘리브레이션·`marker_size` 재확인 |
| 착륙 충격 | ArduPilot `LAND_SPEED` ↓, `land_switch_alt` ↓ |
| 검출 자주 끊김 | 마커 크기↑/조명 개선, `coast_ticks` ↑ |
| 너무 일찍/늦게 하강 | `descend_cone` 조정 |

---

> **경고:** 첫 실비행은 넓은 공터에서, 조종사가 항상 RC 로 즉시 회수할 준비를 한
> 상태로 진행하세요. 카메라 마운트 매핑(`lat_*`)은 반드시 지상에서 먼저 확정하고,
> 저고도 LAND 인계 고도(`land_switch_alt`)는 마커가 화면에 남아 있는 높이로 두세요.
