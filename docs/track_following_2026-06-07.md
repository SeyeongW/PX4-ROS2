# 트랙 라인추종 트럭 — 개발 보고서 (2026-06-07)

400 m 육상 트랙(좌·우 흰 경계선) 위에서, **전방 카메라로 차선 사이 중앙을 추종**하며
주행하는 플랫폼(트럭)을 구현했다. 트럭 위에는 기존 정밀착륙 드론이 탑재되어 함께
이동한다. 추종은 영상→지면 역투영(IPM)→**가상 중앙선 곡선 적합**→2차 제어로 이루어지며,
모든 게인은 한 YAML 파일에서 조절한다.

대상: `precision_landing/precision_landing/track_follower_node.py`
관련 문서: [정밀착륙 보고서](precision_landing.md)

---

## 1. 구성 요소

| 파일 | 역할 |
|---|---|
| `gazebo/gen_track_model.py` | 400 m 트랙 모델 생성기(직선 84.39 m×2 + 반원 R=36.8 m×2). 붉은 노면 띠 + 좌·우 흰 경계선(차선폭 2·`LANE_HALF`=2.0 m). |
| `gazebo/models/running_track/` | 위 생성기가 만든 트랙 SDF(중앙선은 월드에 안 그림 — 노드가 수식으로 만든다). |
| `gazebo/models/aruco_platform/model.sdf` | 트럭. 앞-위(0.6, 0, 0.85)에 35° 하향 **전방 카메라**(`front_camera/image`) 추가. 기존 ArUco 마커·DetachableJoint·VelocityControl 유지. |
| `precision_landing/.../track_follower_node.py` | 라인 인식 + 가상 중앙선 적합 + 2차 조향 + 동적 속도. `moving_marker_node` 대체(같은 cmd_vel/cue/detach 계약). |
| `gazebo/launch/camera_bridge.launch.py` | 전방·하방 카메라 gz→ROS 브리지. |
| `gazebo/launch/track_view.launch.py` | `rqt_image_view`로 오버레이 보기. |
| `gazebo/config/track_follower.yaml` | **모든 튜닝 파라미터(게인 포함) 단일 파일**. |
| `gazebo/run_sim.sh` | Gazebo + 브리지 + 뷰어 + 추종노드 한 번에 기동(YAML 로드). |

월드에서 잔디 지면(`ground_plane`)·활주로(`runway`)는 제거(요청). 트럭은 kinematic,
드론은 그 위에 타므로 충돌용 지면 불필요.

---

## 2. 추종 알고리즘 (매 프레임)

1. **흰 라인 검출**: ROI(영상 높이 45~95%)를 회색조 임계화(`white_thresh`) → `n_bands`개
   가로 스트립으로 분할 → 각 스트립의 흰 픽셀을 클러스터링(`line_gap_px`).
2. **IPM (역투영)**: 카메라 장착 기하(높이 0.85, 피치 0.61, FOV 1.6, 전방 0.6)로 핀홀
   광선을 지면(z=0)과 교차 → 각 픽셀을 **미터 좌표 (X 전방, Y 좌측)** 로 변환. 원근
   왜곡이 펴져 곡선에서도 편향이 없다.
3. **밴드별 차선 중심점**: 두 클러스터(양쪽 라인)면 **같은 행=같은 X의 두 점 중점**(기하학적
   정확). 한쪽만 보이면 **알려진 차선 반폭**(`half_lane`)으로 `중심 = Y_line ∓ half_lane`.
4. **가상 중앙선 적합**: 중심점들 `(X,Y)`에 최소제곱 다항 적합 `Y = P(X)` (`fit_degree` 2=곡선).
   이 적합 곡선이 “수식으로 만든 가상의 중앙선”이며 추종 대상이다.
5. **제어량 추출**: 차량 근접점 `X0`에서
   - 횡오차 `e_y = P(X0)` [m], 헤딩 `ψ = atan(P'(X0))` [rad], 곡률 `κ = P''/(1+P'²)^1.5` [1/m].
6. **2차 조향 + 곡률 피드포워드** (아래 §3).
7. **동적 속도** (아래 §4) → 차체속도 `v`, yaw rate `ω`를 **gz VelocityControl에 body-frame
   Twist**(`linear.x=v`, `angular.z=ω`)로 발행.

또한 트럭 실제 (E,N)을 `/marker/position`으로, 무장 시 DetachableJoint를 분리(드론 발진).

---

## 3. 조향 제어 — 2차 폐루프 설계

차선 유지 루프를 2차계로 모델링한다:

```
ÿ + 2ζωₙ ẏ + ωₙ² y = 0          (y = 횡오차 [m])
운동학 ẏ = v·ψ, ψ̇ = ω 에 제어식
  ω = kappa_ff·v·κ  +  k_ψ·ψ  +  k_y·e_y
를 넣으면  ωₙ² = v·k_y,  2ζωₙ = k_ψ
⇒  k_y = ωₙ²/v_cruise ,  k_ψ = 2ζωₙ
```

- **`k_y·e_y` = P** (중심 복귀), **`k_ψ·ψ` = D**(감쇠; 헤딩이 횡오차 변화율 ∝ 이므로 미분항),
  **`v·κ` = 곡률 피드포워드**(일정 곡률 구간 정상상태 오차 0).
- `ctrl_bw`(=ωₙ)가 P/응답속도, `zeta`(=ζ)가 감쇠를 정한다. **ζ로 진동↔민첩성**을 조절.
- 참고(Bode): ζ=0.707이면 −3 dB 대역폭 = ωₙ인 maximally-flat 지점. 다만 실루프 지연
  때문에 기본값은 과감쇠 **ζ=1.2**.
- `k_y`는 **순항속도 기준 고정**(현재속도로 다시 계산하면 저속에서 폭증→불안정).

---

## 4. 동적 속도 (고정값 아님)

명시적 속도 대신 경로에서 목표를 만들고 가·감속 한계로 부드럽게 추종한다.

```
v_target = min( forward_speed(cap),  sqrt(lat_accel_max / |κ|) )   # 곡률 한계
v ← v + clamp(v_target − v, −decel_max·dt, +accel_max·dt)          # 가·감속 램프
```
- 곡선이 급할수록(횡가속 예산) 자동 감속, 출발 시 0→cap 부드러운 램프.
- 차량(비홀로노믹)은 **추종오차로 감속하지 않는다**(느리면 횡제어력↓ → 진동). 드론과 다른 점.

---

## 5. 파라미터 (`gazebo/config/track_follower.yaml`)

게인 한 줄 고치고 `run_sim.sh` 재시작하면 적용. `FOLLOW_SPEED`/`FOLLOW_BW`/`FOLLOW_ZETA`
env는 설정 시에만 파일을 덮어쓴다.

| 파라미터 | 기본 | 의미 |
|---|---|---|
| `ctrl_bw` | 1.2 | ωₙ [rad/s], 루프 대역폭(≈P) |
| `zeta` | 1.2 | ζ 감쇠비(↑=진동 억제, ↓=민첩) |
| `kappa_ff` | 1.0 | 곡률 피드포워드 배율 |
| `yaw_rate_max` | 0.8 | 조향 명령 클램프 [rad/s] |
| `ctrl_lpf_alpha` | 1.0 | e_y/ψ/κ EMA(1.0=off; <1 평활, 지연↑) |
| `forward_speed` | 3.0 | 순항 속도 상한 [m/s] |
| `lat_accel_max` | 2.5 | 곡선 속도 한계 √(a/κ) [m/s²] |
| `accel_max`/`decel_max` | 1.5/2.5 | 가/감속 램프 [m/s²] |
| `min_speed` | 1.0 | 추종 중 최저 속도(조향력 확보) |
| `white_thresh` | 180 | 흰 라인 임계 |
| `roi_top`/`roi_bot` | 0.45/0.95 | ROI 상/하단(높이 비율) |
| `n_bands` | 12 | ROI 가로 분할 수 |
| `band_min_px` | 6 | 밴드 유효 최소 흰 픽셀 |
| `line_gap_px` | 20 | 두 라인 클러스터 분리 간격 [px] |
| `min_cluster_px` | 2 | 클러스터 최소 픽셀 |
| `fit_degree` | 2 | 2=곡선, 1=직선 적합 |
| `half_lane` | 1.0 | 차선 반폭 [m](단일 라인 보정) |
| `lost_hold_frames` | 8 | 라인 분실 시 관성 유지 프레임 |
| `cam_fov`/`cam_height`/`cam_pitch`/`cam_x` | 1.6/0.85/0.61/0.6 | IPM용 카메라 기하(SDF와 일치) |

---

## 6. 실행 / 보기

```bash
cd ~/ros2_ws/PX4-ROS2/gazebo
./run_sim.sh            # Gazebo + 카메라 브리지 + 오버레이 뷰어 + 추종노드 자동
```
- 오버레이 창(`/perception/track_debug`): 초록=검출 라인, 주황=라인 위치, 노랑=차선중심,
  마젠타=적합 곡선, 노랑 화살표=조향, 좌상단 HUD=`e_y[m]/ψ[deg]/κ/ωₙ·ζ/yaw/speed`.
- 토픽만: `rqt_image_view /perception/track_debug` (또는 `/front_camera/image` 원본).
- 빠른 튜닝: `FOLLOW_ZETA=1.6 ./run_sim.sh`, `FOLLOW_SPEED=2 ./run_sim.sh` 등.
- 헤드리스(서버만): `HEADLESS=1 ./run_sim.sh`. 브리지/뷰어 끄기: `BRIDGE=0`/`VIEW=0`.

---

## 7. 디버깅 일지 — 오늘 해결한 핵심 이슈

진동·이탈을 잡는 과정에서 밝혀진 원인들(재발 방지용 기록):

1. **VelocityControl은 body 프레임**: cmd_vel의 linear/angular는 `LinearVelocityCmd`/
   `AngularVelocityCmd`(차체 프레임). 월드 프레임 linear를 보내면 회전 시 옆으로 미끄러져
   라인을 못 지킴 → `linear.x=v`(전진) + `angular.z=ω`로 수정.
2. **저속 게인 폭증**: `k_y=ωₙ²/v`를 현재 속도로 계산하면 저속에서 발산 → **순항속도 고정**.
3. **차량에 “오차 시 감속” 금지**: 드론과 달리 비홀로노믹이라 느려지면 횡제어력이 줄어 진동.
   곡률 기반 감속만 사용.
4. **곡선 코너컷(인식 편향)**: 영상 중점은 perspective로 곡선에서 안쪽으로 ~0.85 m 편향
   → **IPM 미터평면 + 알려진 차선폭** 보정으로 제거.
5. **가상 중앙선 — 실패한 두 방법**: ① 좌·우 라인을 각각 적합 후 평균 → 동심원 sqrt 평균이라
   곡선 편향(2.2 m 이탈), Y=P(X)가 반원 정점에서 퇴화. ② 2라인 밴드만 사용 → 점이 적고 X에
   몰려 2차 적합 불안정(8 m 이탈). → **밴드별 중점(같은 X)→전체 점 적합**이 정답.
6. **직선 진동의 진짜 원인 = QoS**: 노드가 카메라를 **BEST_EFFORT**로 구독했는데 브리지는
   **RELIABLE**로 0.9 MB 이미지를 발행 → BE가 큰 프레임을 불규칙 드랍 → 제어 주기가
   들쭉날쭉 → **직선에서도 좌우 요동**(GUI는 부하↑로 더 심함). → 이미지 구독을
   **RELIABLE, depth 1**로 변경. (`ros2 topic hz`는 RELIABLE 구독이라 데이터가 보여 오진하기 쉬움.)
   ※ `camera_detection/aruco_detector_node`도 동일 패턴 — 정밀착륙이 떨면 같이 수정.

---

## 8. 검증 (헤드리스 SITL, 한 바퀴)

| 구간 | 결과 |
|---|---|
| 직선 횡오차 | **std 0.001 m**(반전 1/660) — 사실상 무진동 |
| 곡선 반지름 오차 | ±0.0~0.4 m (차선 ±1.0 m 내), std 0.18~0.23 |
| 속도 | 0→cap 부드러운 램프, 동적 |
| `/marker/position` | RELIABLE 후 규칙적 수신(이전 드랍) |

직선에서 좌우 간격 유지하며 주행, 곡선도 차선 안에서 매끄럽게 통과.

---

## 9. 남은 일 / 한계

- 직선 정상상태 오프셋이 구간에 따라 0.3~0.5 m(차선 내). 단일 라인 known-width 보정의
  잔차 — IPM 외부 캘리브/X0 위치로 더 줄일 여지.
- IPM은 평지·강체 카메라 가정. 노면 요철/서스펜션 있으면 보정 필요.
- 미세 튜닝은 `config/track_follower.yaml`의 `ctrl_bw`(P)·`zeta`(D)로.
