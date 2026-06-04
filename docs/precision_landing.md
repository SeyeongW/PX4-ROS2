# ArUco 정밀착륙 시스템 — 파라미터 & 코드 기능 보고서

이동하는 ArUco 마커(플랫폼) 위로 드론을 정밀착륙시키는 시스템의 동작 원리,
런치 파라미터, 노드 파라미터, 핵심 함수의 역할을 정리한 문서입니다.

대상 패키지: `precision_landing` (+ 인식은 `camera_detection/aruco_detector_node`)
제어 노드: `precision_landing/precision_landing/precision_landing_node.py`
마커 이동/좌표 송출: `precision_landing/precision_landing/moving_marker_node.py`

---

## 1. 전체 동작 개요 (상태 기계)

제어기는 50 ms(20 Hz) 주기로 도는 상태 기계입니다.

```
TAKEOFF → IDLE → APPROACH → ALIGN → DESCEND → DONE
```

| 단계 | 하는 일 |
|------|---------|
| **TAKEOFF** | GUIDED 전환 → 시동(arm) → `flight_alt`까지 이륙. `auto_takeoff=false`면 건너뛰고 IDLE에서 사람이 띄우길 대기. |
| **IDLE** | armed + GUIDED + (마커가 보이거나 유효한 cue 수신) 될 때까지 제자리 호버. |
| **APPROACH** | 카메라가 마커를 보기 전, 외부에서 송출되는 ENU 좌표(cue)를 향해 `flight_alt` 고도로 빠르게 접근. 움직이는 표적도 계속 추종. |
| **ALIGN** | 카메라가 마커를 잡으면 호버하며 비전 서보로 좌우 정렬. 접근 속도(approach_vel_max)에서 정밀 속도(vel_max)로 부드럽게 감속. |
| **DESCEND** | 깔때기(funnel) 모양 허용오차를 따라 정렬하면서 동시에 하강. 마커가 플랫폼(높이 `platform_height`) 위에 있으므로 모든 고도 계산은 "마커 윗면까지의 높이" 기준. |
| **DONE** | 플랫폼 윗면 `land_clearance` 이내 + 중심 정렬 상태에서 **강제 disarm**(모터 차단)으로 안착. LAND 모드를 안 쓰는 이유는 그게 지면까지 내려가 플랫폼을 무시하기 때문. |

**Cue → 비전 핸드오프:** 외부 좌표 송출원(`moving_marker_node`)이 마커의 로컬 ENU
위치를 계속 보냄. 드론은 그 cue를 향해 블라인드로 접근(APPROACH)하다가, 하방
카메라가 마커를 보는 순간 비전 서보(ALIGN/DESCEND)로 인계. 비전이 끊기면 다시 cue로 복귀.

---

## 2. 토픽 인터페이스

| 방향 | 토픽 | 타입 | 의미 |
|------|------|------|------|
| in | `/perception/aruco_offset` | `geometry_msgs/Point` | 정규화된 마커 이미지 오프셋 [-1, 1] |
| in | `/perception/aruco_detected` | `std_msgs/Bool` | 마커 검출 여부 |
| in | `/marker/position` | `geometry_msgs/PointStamped` | ENU cue (x=East, y=North) |
| in | `/marker/velocity` | `geometry_msgs/Vector3Stamped` | 마커 속도 ENU (피드포워드용, 있으면 우선) |
| in | `/mavros/state` | `mavros_msgs/State` | 연결/시동/모드 상태 |
| in | `/mavros/local_position/pose` | `geometry_msgs/PoseStamped` | 로컬 ENU 위치 (BEST_EFFORT QoS) |
| out | `/mavros/setpoint_raw/local` | `mavros_msgs/PositionTarget` | 속도 + yaw 셋포인트 |
| out | `/precision_landing/debug` | `std_msgs/String` | 단계 전환 / 정렬 디버그 로그 |

> 주의: `/mavros/local_position/pose`는 BEST_EFFORT(sensor) QoS로 발행되므로
> sensor QoS로 구독해야 메시지가 들어옵니다.

---

## 3. 런치 파라미터 (`precision_landing.launch.py`)

런치에서 `이름:=값` 으로 바꿀 수 있는 인자들입니다.

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `fcu_url` | `udp://:14550@` | MAVROS↔SITL MAVLink 연결. MAVProxy가 127.0.0.1:14550으로만 송신하므로 **14550에 bind**하고 remote(`@` 뒤)는 비워 상대 주소 자동 학습. |
| `image_topic` | `/down_camera/image` | ArUco 검출이 구독할 카메라 영상 토픽. |
| `flight_alt` | `5.0` | 이륙/접근 호버 고도 (m). |
| `auto_takeoff` | `true` | 노드가 스스로 GUIDED→시동→이륙. false면 사람이 직접 이륙시키고 IDLE에서 인계. |
| `lat_swap` | `false` | 이미지 x/y 축 교환. 드론이 마커를 못 잡고 빙글빙글 돌면(축 90° 회전) true. |
| `lat_sign_fwd` | `1.0` | 전방(+X) 보정 부호 ±1. 하강 중 전방으로 발산하면 부호 반전. |
| `lat_sign_left` | `1.0` | 좌측(+Y) 보정 부호 ±1. 가로로 직선 발산하면 부호 반전. |
| `vel_gain` | `0.4` | 수평 속도 서보 게인 (1/s). 진동하면 낮추고 굼뜨면 올림. |
| `vel_max` | `5.0` | 정밀(ALIGN/DESCEND) 수평 속도 상한 (m/s). 이동 플랫폼을 따라잡으려면 충분히 커야 함. 정지/저속 마커면 0.5로 낮춰도 됨. |
| `approach_vel_max` | `10.0` | APPROACH 단계 최대 속도 (m/s). 플랫폼까지 빠르게 접근. |
| `approach_decel_s` | `5.0` | 도착 예상 ETA가 이 값(초) 이하로 떨어지면 속도 상한을 `approach_vel_max`→`vel_max`로 선형 감속. |
| `use_cue` | `true` | cue 추종(APPROACH) 사용. false면 순수 비전 동작(기존). |
| `yaw_track` | `true` | APPROACH 중 진행방향(명령 속도)으로 기수 정렬. false면 헤딩 고정. |
| `yaw_track_min_speed` | `0.5` | 이 속도(m/s) 이하에선 노이즈로 안 돌게 헤딩 유지. |
| `platform_height` | `1.0` | 마커가 올라앉은 플랫폼(차량) 높이 (m). 모든 고도 로직이 `pos.z − platform_height` 기준. 평면 지면 마커면 0. |
| `land_clearance` | `0.2` | 마커 윗면 위 이 높이에서 강제 disarm. 다리가 박힌 채 동력 상태로 닿으면 전복되므로 닿기 직전 모터를 끊어 "속도 맞춘 자유낙하"로 안착. 너무 높으면 낙하 충격 ↑. |

---

## 4. 노드 내부 파라미터 (`precision_landing_node.py`)

런치에 노출되진 않았지만 `ros2 param set`으로 실시간 튜닝 가능한 값들입니다.

### 4-1. 하강 깔때기 (funnel)

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `descend_cone` | `0.35` | 깔때기 반각 기울기 (m오차/m고도). 고도가 높을수록 허용오차 넓고, 낮아질수록 `land_align_radius`로 좁아짐. `funnel_radius(alt) = max(land_align_radius, descend_cone·alt)`. |
| `descend_min_scale` | `0.3` | 깔때기 안에서 정렬이 덜 됐어도 보장하는 최소 하강률(전체 `descend_rate`의 비율). 즉 완벽 정렬 전에도 계속 야금야금 내려감. |
| `descend_rate` | `0.3` | 기본 하강 속도 (m/s, 아래 방향). |
| `land_align_radius` | `0.20` | 착륙 확정 수평오차 게이트(m). 깔때기의 바닥(최소 반경)이기도 함. 이보다 멀면 재정렬 먼저. |
| `final_descent_h` | `0.6` | 마커 윗면 위 이 높이 아래로는 마커(~0.8 m)가 화면을 넘쳐 비전이 끊김 → cue(플랫폼 중심)/KF 코스트로 **open-loop** 마무리 하강. |

### 4-2. 칼만 필터 (등속도 모델, 상태 [E, N, vE, vN])

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `kf_accel_std` | `0.1` | 프로세스 잡음 (m/s²). 표적이 얼마나 기동할 수 있는지. |
| `kf_meas_std` | `0.15` | 검출 잡음 (m). |
| `coast_ticks` | `30` | 측정 없이 예측만으로 버티는 최대 틱 수(~1.5 s). 프레임 누락 시 추정값으로 활주(coast). |

### 4-3. 속도 서보

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `vel_gain` | `0.8`* | 비전 서보 게인. `v = v_marker + vel_gain·error`. 1차 응답(오버슈트 없는 지수 수렴). |
| `vel_max` | `1.0`* | 정밀 속도 상한. |
| `approach_vel_max` | `10.0` | 접근 속도 상한. |
| `approach_decel_s` | `5.0` | ETA 기반 감속 시작 시점. |
| `approach_ramp_s` | `2.0` | ALIGN 진입 후 `eff_vel_max`를 접근속도→정밀속도로 줄이는 시간(s). 급격한 저크 방지. |

\* 노드 기본값. 런치에서는 `vel_gain=0.4`, `vel_max=5.0`으로 덮어씀.

### 4-4. 카메라/매핑

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `cam_hfov` | `1.20` | 하방 카메라 수평 화각(rad). 픽셀→미터 변환에 사용. |
| `cam_aspect` | `4/3` | 영상 가로/세로 비율. |
| `lat_swap` / `lat_sign_fwd` / `lat_sign_left` | `false`/`1`/`1` | 카메라 마운트 의존 이미지→기체 매핑 보정(3장 참조). |

### 4-5. cue 추종

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `use_cue` | `true` | cue 추종 사용 여부. |
| `cue_timeout` | `1.0` | 마지막 cue 메시지 후 이 시간(s)이 지나면 stale 처리. 죽은 송출원 좌표를 계속 쫓지 않게. |

### 4-6. 헤딩 제어

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `yaw_track` | `true` | **APPROACH 동안** 명령 수평속도 방향(`atan2(vN, vE)`, ENU yaw=0이 East)으로 기수를 돌려 진행방향을 봄. ALIGN/DESCEND는 헤딩 고정(왕복 플랫폼 반전 시 180° 회전 방지 + 비전 서보 안정). false면 전 구간 헤딩 고정. |
| `yaw_track_min_speed` | `0.5` | 이 속도(m/s) 이하에선 헤딩을 갱신하지 않고 마지막 값 유지(저속 노이즈로 인한 회전 방지). |

> 카메라 매핑(`_measure_marker_world`)은 고정 setpoint가 아니라 **실제 현재 헤딩
> `self.yaw`** 로 body→world 회전을 하므로, 기수가 도는 중에도 마커 월드 위치
> 추정이 정확합니다.

---

## 5. 핵심 함수 역할

| 함수 | 역할 |
|------|------|
| `tick()` | 50 ms 메인 루프. 단계별 분기 + 마지막에 `_publish_velocity()` 호출. |
| `_track(marker_ok, vmax_override)` | KF 예측/갱신 후 추정 위치로 속도 서보. 드론↔마커 수평거리 반환. ALIGN/DESCEND의 핵심. |
| `_servo_to(E, N, vffE, vffN, vmax)` | 절대 ENU 점을 향한 1차 속도 서보. `v = v_ff + kp·e` (clamp). 피드포워드로 이동표적의 정상상태 지연 제거. |
| `_measure_marker_world()` | 정규화 이미지 오프셋 → 마커 월드(E,N) 위치. 고도로 픽셀→미터 스케일, 이미지→기체(마운트 부호), 기체→월드(yaw 회전). |
| `_height_above_marker()` | `pos.z − platform_height`. 카메라 투영·깔때기·착륙 게이트가 쓰는 "마커 윗면 위 높이". |
| `_funnel_radius()` | 현재 고도에서 허용 수평오차(깔때기 반경). 실시간 파라미터 반영. |
| `_kf_predict()` / `_kf_update()` / `_kf_reset()` | 등속도 칼만 필터. `_kf_reset(seed_vel=...)`은 핸드오프 시 cue 속도를 초기 속도로 심어 이동표적을 첫 틱부터 추종(정지로 오인해 표적을 놓치는 한계주기 방지). |
| `_cue_cb()` / `_vel_cb()` | cue 위치/속도 수신. 속도는 명시적 `/marker/velocity` 우선, 없으면 위치 차분으로 폴백. |
| `_publish_velocity()` | `PositionTarget`(속도 E,N,Up + yaw) 발행. mavros가 ENU→NED 변환. yaw 고정으로 카메라 매핑 유효성 유지(자전 방지). |
| `_force_disarm()` | `MAV_CMD_COMPONENT_ARM_DISARM`(400), param2=21196 매직으로 ArduCopter의 공중 disarm 거부를 우회. |

---

## 6. 디버그 로그 읽는 법

DESCEND/ALIGN 중 ~1 Hz로 찍히는 줄:

```
off=(+0.00,+0.00) yaw=+90 err=(-0.02,+0.00) cmd=(-3.02,+0.00)
```

| 필드 | 의미 |
|------|------|
| `off` | 이미지 안 마커 위치(정규화). 0이면 화면 정중앙. |
| `yaw` | 현재 기수 방위(도, ENU). 정렬 중엔 고정. |
| `err` | 월드 위치오차 `(eE, eN)` = 마커추정 − 드론. 0이면 정확히 위에 있음. |
| `cmd` | 명령 수평속도 `(E, N)`. |

**올바른 매핑의 판정:** 속도(`cmd`)가 이미지 오프셋(`off`)을 줄이는 방향이어야 함.
오프셋이 커지면 swap/부호가 틀린 것 → `lat_swap`/`lat_sign_*` 조정.

**이동표적 정상 추종의 모습:** `off≈0`, `err≈0` 인데 `cmd`가 0이 아니라 플랫폼
속도와 같음(예: 서쪽 3 m/s 이동 → `cmd=(-3.0, 0)`). 이는 속도 피드포워드가
플랫폼 속도를 그대로 실어 위치오차 없이 따라가는 정상 동작.

---

## 7. 튜닝 가이드 (증상 → 대응)

| 증상 | 대응 |
|------|------|
| 좌우로 진동(overshoot) | `vel_gain` ↓ (0.4 → 0.3) |
| 정렬이 너무 굼뜸 | `vel_gain` ↑, `vel_max` ↑ |
| 마커를 못 잡고 빙글빙글 | `lat_swap:=true` |
| 한쪽으로 직선 발산 | 해당 축 `lat_sign_*` 부호 반전 |
| 이동 플랫폼을 못 따라잡음 | `vel_max` ↑ (플랫폼 속도 + 위치보정 여유 이상으로) |
| 접근이 느림 | `approach_vel_max` ↑ |
| 도착 시 과속/오버슈트 | `approach_decel_s` ↑ |
| 너무 일찍/늦게 하강 | `descend_cone` 조정(작을수록 더 정렬돼야 하강) |
| 착륙 충격이 큼 | `land_clearance` ↓ |
| 플랫폼을 비껴 착륙 | `land_align_radius` ↓ (착륙 게이트 강화) |

---

## 8. 이동 마커 노드 (`moving_marker_node.py`)

Gazebo 안에서 마커 모델을 움직이고, **같은 틱에** 그 위치를 ENU `PointStamped`로
송출(cue)합니다. `gazebo/run_sim.sh`가 Gazebo와 함께 띄웁니다(정밀착륙 런치는
중복 실행하지 않음).

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `world` | `iris_down_camera_runway` | gz `set_pose` 대상 월드 이름. |
| `model` | `aruco_marker_0` | 움직일 모델 이름. |
| `marker_topic` | `/marker/position` | 위치 cue 발행 토픽. |
| `vel_topic` | `/marker/velocity` | 속도 발행 토픽(피드포워드). |
| `rate` | `50.0` | 발행/이동 주기(Hz). 높을수록 고속에서 텔레포트 스텝이 작음. |
| `pattern` | `line` | 궤적: `static` / `line`(등속 직선 왕복) / `circle`. |
| `center_e`, `center_n` | `1.0`, `0.0` | 궤적 중심 (ENU). |
| `amplitude` | `1.5` | 진폭(m). line은 ±amplitude 왕복(직선 1레그 = 2·amplitude). |
| `speed` | `0.3` | 경로 속도(m/s). |
| `z` | `0.002` | 마커 지면 높이(m). |
| `move_model` | `true` | false면 모델은 그대로 두고 cue만 발행(외부 무버/정적 마커용). |

> **기하 메모:** 드론이 월드 원점에서 스폰되고 MAVROS의 로컬(EKF) 원점도 거기에
> 맞춰지므로, 이 월드에선 로컬 ENU = Gazebo 월드 XY 평면(E=world x, N=world y).
> 따라서 마커 위치를 그대로 ENU cue로 송출.
>
> 속도는 line 패턴을 등속 **삼각파**(사인 아님)로 만들어 크기가 일정 →
> 드론의 속도 피드포워드와 정확히 일치 → 깔끔한 착륙.

궤적/속도는 `run_sim.sh`의 `MARKER_*` 환경변수로 조정합니다.
