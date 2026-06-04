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

## 7. 속도 피드포워드 — 왜 필요한가

이동하는 플랫폼 위에 착륙하는 핵심 제어 개념입니다.

### 피드백 vs 피드포워드

- **피드백(feedback):** 결과(오차)를 보고 사후에 고치는 방식.
  비례 서보 `v = vel_gain · e` 가 여기 해당 (`e` = 마커위치 − 드론위치).
  오차가 생겨야만 반응합니다.
- **피드포워드(feed-forward):** 미리 아는 정보를 제어 입력에 선제적으로 더하는 방식.
  여기서는 **마커(플랫폼)의 속도**를 그대로 더합니다.

제어 법칙 (`_servo_to`):

```
v = v_ff        +  vel_gain · e
    └ 피드포워드      └ 피드백
    (플랫폼 속도)     (위치오차 보정)
```

### 왜 이동 표적엔 피드백만으론 부족한가

정지 마커면 피드백만으로 충분합니다(`v_ff≈0`). 그러나 플랫폼이 예컨대 서쪽
3 m/s로 움직이면, **피드백만** 쓸 경우 드론이 3 m/s를 내려면
`vel_gain · e = 3` 이어야 하므로 **항상 위치오차가 남습니다**:

```
e_정상상태 = 3 / vel_gain = 3 / 0.4 ≈ 7.5 m   (마커보다 7.5 m 뒤처짐 = lag)
```

이러면 `land_align_radius`(0.20 m) 착륙 게이트를 절대 통과하지 못합니다.
**피드포워드를 더하면** 플랫폼 속도를 `v_ff`가 직접 담당하므로 `e=0`이어도
이미 3 m/s가 나와 **위치오차 0으로 수렴**합니다.

### 디버그 로그에서 보이는 모습

```
err=(+0.00,+0.00)  cmd=(-3.02,+0.00)
```

**오차는 0인데 명령 속도는 −3** — 피드포워드(플랫폼 속도)가 −3을 만들고
피드백(`0.4·e`)이 미세 보정만 하는 정상 상태. 피드포워드가 없으면 이 `err`은
0이 될 수 없습니다.

### v_ff(마커 속도)는 어디서 얻나

| 출처 | 사용 단계 | 비고 |
|------|-----------|------|
| `/marker/velocity` (명시적) | APPROACH | `moving_marker_node`가 직접 송출 → 가장 정확(지연·잡음 없음). 있으면 우선 사용. |
| 위치 cue 시간 차분 | APPROACH | `/marker/velocity`가 없을 때 폴백(`_cue_cb`). 다소 느리고 노이즈 있음. |
| 칼만 필터 속도 추정 `kf_x[2:4]` | ALIGN / DESCEND | 비전으로 추정한 마커 속도를 피드포워드. 정지 마커면 ≈0이라 기존 동작 보존. |

> 한 줄 요약: **피드백 = 틀린 만큼 고친다(사후). 피드포워드 = 아는 만큼 먼저
> 넣는다(선제).** 둘을 합쳐야 움직이는 표적을 지연 없이 따라가 그 위에 착륙 가능.

---

## 8. 칼만 필터 — 개념과 게인

### 8-1. 왜 쓰나

하방 카메라의 마커 검출은 **노이즈가 있고(흔들림), 가끔 끊깁니다(프레임 누락).**
이를 그대로 제어에 쓰면 드론이 따라 떨립니다. 칼만 필터(KF)는

1. **노이즈를 평활화**해 부드러운 추정값을 만들고,
2. 검출이 끊긴 동안 **예측만으로 활주(coast)** 하며 버티고,
3. 마커의 **속도를 추정**해 피드포워드(7장)로 넘겨줍니다.

### 8-2. 상태와 모델 (등속도)

상태 벡터는 마커의 월드 위치 + 속도 4개입니다:

```
x = [E, N, vE, vN]      (E=동쪽 위치, N=북쪽 위치, vE/vN=속도)
```

**등속도(constant-velocity) 모델**을 씁니다 — "마커는 대체로 일정한 속도로
움직인다"고 가정. 플랫폼이 직선 등속으로 다니므로 잘 맞고, 추정한 속도를
그대로 피드포워드로 쓸 수 있습니다. 가속(방향 전환 등)은 "프로세스 잡음"으로
흡수합니다.

### 8-3. 두 단계: 예측(predict) ↔ 보정(update)

매 틱(50 ms) 다음을 반복합니다.

**① 예측 (`_kf_predict`, 매 틱)** — 모델로 한 스텝 전진:

```
x ← F·x          위치 += 속도·dt   (등속 전진)
P ← F·P·Fᵀ + Q   불확실성 증가
```
- `F`: 상태 천이 행렬 (위치에 속도×dt를 더함)
- `Q`: **프로세스 잡음** — 모델이 틀릴 수 있는 정도(가속 가능성). `kf_accel_std`로 결정.

**② 보정 (`_kf_update`, 측정이 있을 때만)** — 검출값 `z=(zE,zN)`으로 교정:

```
y = z − H·x              혁신(innovation): 측정 − 예측
K = P·Hᵀ·(H·P·Hᵀ+R)⁻¹   칼만 게인
x ← x + K·y              게인만큼 측정 쪽으로 당김
P ← (I − K·H)·P          불확실성 감소
```
- `H`: 관측 행렬 (위치만 측정, 속도는 직접 못 봄 → 위치 변화로 간접 추정)
- `R`: **측정 잡음** — 검출이 얼마나 못 믿을지. `kf_meas_std`로 결정.
- `K`(칼만 게인): **측정과 예측을 섞는 가중치.** 측정을 믿을수록(R 작음) K가 커져 측정 쪽으로, 모델을 믿을수록(Q 작음) K가 작아져 예측 쪽으로.

검출이 없으면 ②를 건너뛰고 ①만 → **활주(coast)**. `kf_miss`가 `coast_ticks`를
넘으면 "마커 잃음"으로 처리.

### 8-4. 게인값(튜닝 노브)의 의미

| 게인 | 의미 | 키우면 | 줄이면 |
|------|------|--------|--------|
| `kf_accel_std` | 프로세스 잡음 σₐ (표적이 얼마나 기동할 수 있나, m/s²) | 측정을 더 신뢰 → **민첩**(빠른 반응), 노이즈 ↑ | 모델을 더 신뢰 → **부드러움**, 반응 lag ↑ |
| `kf_meas_std` | 측정 잡음 σ_z (검출 노이즈, m) | 측정을 덜 신뢰 → **부드러움**, lag ↑ | 측정을 더 신뢰 → **민첩**, 노이즈 ↑ |
| `coast_ticks` | 측정 없이 예측만으로 버티는 최대 틱 | 긴 드롭아웃도 견딤(위험: 오래된 추정 신뢰) | 금방 "잃음" 판정(안전하지만 잘 끊김) |

> **핵심은 두 값의 비율** `kf_accel_std / kf_meas_std`:
> 비율 ↑ = 민첩하지만 떨림, 비율 ↓ = 부드럽지만 느림.
> 빠른 플랫폼이면 `kf_accel_std`를 0.1 → 0.5로 올려 추종력을 키웁니다.

초기화(`_kf_update` 첫 측정): 위치는 검출값으로, 속도는 핸드오프 시 cue 속도를
`seed_vel`로 심습니다(8-5). 공분산 `P=diag([r², r², 1, 1])` — 위치는 측정만큼
확신, 속도는 느슨하게 둬서 비전이 곧 교정하게.

### 8-5. 속도 시드(seed)로 한계주기 방지

APPROACH→ALIGN 핸드오프 때 KF 속도를 cue 속도로 **미리 심어둡니다**
(`_kf_reset(seed_vel=…)`). 안 그러면 KF가 v=0에서 시작 → 드론이 마커를 정지로
오인하고 감속 → 표적이 빠져나가 비전 상실 → APPROACH로 튕김(빠른 획득/상실/재접근
한계주기). 시드 덕에 첫 틱부터 이동 표적을 매칭합니다.

---

## 9. 제어 로직 & 알고리즘 개념

### 9-1. 속도 제어 (위치 타겟이 아니라)

ALIGN/DESCEND는 위치가 아니라 **속도 셋포인트**를 보냅니다
(`/mavros/setpoint_raw/local`). 위치 타겟은 중첩 위치 루프 + 지연으로 좌우
진동했기 때문입니다. 제어 법칙은 1차(first-order):

```
v = vel_gain · e          (e = 추정 마커위치 − 드론위치)
⇒ ė = −vel_gain · e       → e가 지수적으로 0에 수렴, 오버슈트 없음
```

- 시간상수 `τ = 1 / vel_gain` (예: 0.4 → τ≈2.5 s). 게인 ↑ = 빠르지만 진동 위험.
- 여기에 7장의 **피드포워드**(`v_ff`)를 더해 이동 표적의 정상상태 lag를 제거.
- `vel_max`로 속도를 clamp(포화).

### 9-2. ETA 기반 접근 감속 (APPROACH)

플랫폼까지 멀면 빠르게, 도착 직전엔 정밀속도로 줄여 오버슈트를 막습니다:

```
ETA = 남은거리 / approach_vel_max
ETA ≥ approach_decel_s  → 속도상한 = approach_vel_max          (전속)
ETA <  approach_decel_s → 상한을 approach_vel_max→vel_max 로 선형 감속
```

ALIGN 진입 후엔 `approach_ramp_s` 동안 상한을 정밀속도까지 부드럽게 더 내려
급격한 저크를 방지합니다.

### 9-3. 하강 깔때기 (funnel)

"완벽히 정렬되면 내려간다"가 아니라 **정렬하면서 동시에 하강**합니다.
허용 수평오차가 고도에 따라 변하는 원뿔(깔때기):

```
funnel_radius(alt) = max(land_align_radius, descend_cone · alt)
```

- 고도 높음 → 반경 넓음(관대), 낮아질수록 `land_align_radius`로 좁아짐.
- 깔때기 **안**: 항상 하강(최소 `descend_min_scale`×`descend_rate` 보장),
  중심에 가까울수록 빠르게. **밖**: 하강 멈추고 재정렬.
- 깔때기가 고도와 함께 좁아지므로 **내려가면서 자연히 중심으로 수렴**.

### 9-4. 픽셀 → 미터 투영 (`_measure_marker_world`)

정규화 이미지 오프셋(−1~1)을 마커의 월드 위치로 변환:

```
half_w = alt · tan(hfov/2),  half_h = half_w / aspect   (고도 의존 스케일)
gx = ix · half_w,  gy = iy · half_h                      (이미지 미터)
fwd = sign_fwd·(−gy),  left = sign_left·(−gx)            (이미지→기체, 마운트 부호)
(de, dn) = R(yaw) · (fwd, left)                          (기체→월드, 실제 yaw 회전)
마커월드 = 드론위치 + (de, dn)
```

- **고도 의존**: 같은 픽셀 오차라도 저고도에선 더 작은 미터 → 하강할수록 자동으로 정밀.
- `alt`는 AGL이 아니라 **마커 윗면 위 높이**(`pos.z − platform_height`).
- 회전엔 고정 setpoint가 아니라 **실제 현재 yaw**(`self.yaw`)를 써서 기수가 도는 중에도 정확.

### 9-5. 최종 하강 & 착륙

- **open-loop 마무리** (`final_descent_h` 아래): 마커(~0.8 m)가 화면을 넘쳐 비전이
  끊기므로, 가장 믿을 수 있는 수평 기준인 **cue(플랫폼 중심)** 로 곧장 하강. 중심
  정렬(`land_align_radius`) 전엔 하강 보류 → 비껴 착륙 방지.
- **강제 disarm** (`land_clearance` 이내): 플랫폼 윗면 직전에서 모터 차단
  (`MAV_CMD_COMPONENT_ARM_DISARM`, param2=21196 매직으로 공중 disarm 거부 우회).
  LAND 모드를 안 쓰는 이유는 그게 플랫폼을 무시하고 지면까지 내려가기 때문.
  "속도 맞춘 자유낙하"로 안착해 끌림·전복 방지(플랫폼 마찰도 mu=1로 낮춤).

### 9-6. 헤딩 제어 (진행방향 정렬)

APPROACH에서 명령 속도 방향으로 기수를 돌립니다:

```
hold_yaw = atan2(vN, vE)   (ENU yaw: 0=East, CCW +),  단 |v| ≥ yaw_track_min_speed 일 때만
```

- 저속에선 갱신 안 함(노이즈 회전 방지). ALIGN/DESCEND는 헤딩 **고정**(왕복 플랫폼
  반전 시 180° 회전·비전 불안정 방지).
- 카메라 매핑이 **실제 yaw**를 쓰므로(9-4) 기수가 돌아도 추정은 정확.

---

## 10. 튜닝 가이드 (증상 → 대응)

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
| 추정이 떨림/노이즈 | `kf_meas_std` ↑ 또는 `kf_accel_std` ↓ (부드럽게, lag↑) |
| 이동표적 반응이 느림/뒤처짐 | `kf_accel_std` ↑ (0.1→0.5, 민첩하게) |
| 검출 끊김에 자주 "잃음" | `coast_ticks` ↑ (활주 시간 연장) |

---

## 11. 이동 마커 노드 (`moving_marker_node.py`)

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
