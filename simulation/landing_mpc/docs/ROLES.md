# landing_mpc — 노드/모듈 역할 한눈에 보기

`docs/nodes.md`가 **왜 그렇게 설계했는가**(수식·유도·실측)를 다룬다면, 이 문서는
**무엇이 무엇에게 무엇을 주는가**만 다룹니다. "예측 노드가 여러 개인 것 같다"는
인상은 실제로는 **역할이 다른 네 단계가 각각 이름을 갖고 있어서**입니다 — 아래
표의 "혼동 포인트"를 먼저 보세요.

---

## 0. 한 장 요약

```
카메라 ─► aruco_detector_node ─► marker_tf_node ─► marker_kf_node ─┐
          (픽셀→착륙점)          (카메라→ENU)      (평활·coast)    │
                                                                  ▼
gz 트레일러 ─► trailer_cue_node ────────────────────────► mission_manager_node
               (원거리 큐)                                   (단계 시퀀싱)
                                                                  │
   ┌──────────────────────────────────────────────────────────────┤
   ▼                        ▼                    ▼                ▼
predictor              mpc(+model)          reference        PX4 setpoint
(표적의 미래)          (최적 가속도)        (50 Hz 보간)
                                                                  │
gimbal_control_node ◄─────────────────────────────────────────────┘
(렌즈 조준 + 관절 엔코더)
```

**노드는 6개, 라이브러리는 5개.** 실행 파일로 뜨는 것은 노드 6개뿐입니다.

---

## 1. 혼동 포인트 — "예측"이라는 이름이 붙은 네 가지

이름이 비슷해서 중복처럼 보이지만, 넷은 **묻는 질문이 다릅니다.**

| 모듈 | 질문 | 대상 | 시간축 |
|---|---|---|---|
| `marker_kf_node` | "표적이 **지금** 어디 있나?" | 표적 | 과거 측정 융합 |
| `predictor.py` | "표적이 **앞으로** 어디 갈까?" | 표적 | 미래 외삽 |
| `mpc.py` | "**내가** 뭘 해야 하나?" | 자기 기체 | 미래 최적화 |
| `reference.py` | "그 계획을 **지금 이 순간** 어떻게 쏘나?" | 자기 기체 | 계획 내부 보간 |

즉 **표적 추정(KF) → 표적 예측(predictor) → 자기 계획(MPC) → 계획 재생(reference)**
의 직렬 파이프라인이고, 경쟁하는 중복이 아닙니다. KF가 없으면 예측할 값이 없고,
predictor가 없으면 MPC가 움직이는 표적을 놓치고, reference가 없으면 10 Hz 계획을
50 Hz로 쏠 수 없습니다.

---

## 2. 노드 (실행 파일 6개)

### 2.1 `aruco_detector_node` — 픽셀에서 착륙점으로
- **입력** `/gimbal_camera/image`, `/gimbal_camera/camera_info`
- **출력** `/aruco/pose_cam` (카메라 광학 프레임), `/aruco/detected`,
  `/aruco/center_error`, `/aruco/debug_image/compressed`
- 드론·좌표계·착륙을 모르는 순수 검출기. 마커 **사다리**(큰 것 2 + 중앙 작은 것 1)를
  전부 풀고 **하나의 착륙점**으로 환산해 내보내므로, 어느 마커를 썼는지는 이 노드
  위로 새어 나가지 않습니다.

### 2.2 `marker_tf_node` — 카메라 프레임 → 로컬 ENU
- **입력** `/aruco/pose_cam` + 기체 자세/위치 + `/gimbal/joint_state`
- **출력** `/marker/measured`
- 이미지 **촬영 시각**으로 기체 상태를 보간해서 변환합니다. "지금"으로 변환하면
  이동 표적에서 그대로 위치 오차가 됩니다.

### 2.3 `marker_kf_node` — 간헐적 관측 → 연속 추정
- **입력** `/marker/measured`
- **출력** `/marker/position`, `/marker/velocity`, `/marker/valid`
- 등속 KF + coast(최대 3 s). 검출이 끊겨도 추정을 이어갑니다.

### 2.4 `trailer_cue_node` — 원거리 큐 (SITL 전용)
- **입력** (gz) 트레일러 pose
- **출력** `/marker/cue`, `/marker/cue_velocity`
- 90 m에서는 카메라가 마커를 못 봅니다. 실기체에서는 트럭의 GPS 텔레메트리가
  맡을 자리를 SITL에서 대신합니다. **이 노드만이 시뮬레이터 전용입니다.**

### 2.5 `gimbal_control_node` — 렌즈 조준
- **입력** `/marker/cue`, `/marker/position`, 기체 자세
- **출력** (gz) 관절 명령, `/gimbal/joint_state`, `/gimbal/aim_error_deg`, TF
- 기체 자세와 카메라 지향을 분리합니다. 관절 엔코더를 함께 내보내는 것이
  `marker_tf_node`가 프레임 오프셋에 면역이 되는 이유입니다.

### 2.6 `mission_manager_node` — 단계 시퀀싱 + 유일한 setpoint 권한
- **입력** `/marker/cue*`, `/marker/position`, `/marker/valid`, PX4 상태
- **출력** `/fmu/in/trajectory_setpoint`, `/fmu/in/offboard_control_mode`,
  `/fmu/in/vehicle_command`, `/mission/state`
- 단계: `IDLE → TAKEOFF → APPROACH → ACQUIRE → DESCEND → TOUCHDOWN → DONE`
  (실패 시 `ABORT`)
- **다른 setpoint 발행자와 절대 같이 띄우지 마세요.**
- 원칙: **큐가 날고, 마커가 중심을 잡는다.** 미션 전체는 `/marker/cue`를 추종하고,
  비전은 거기에 **천천히 필터링되는 수평 보정**으로만 들어갑니다.

---

## 3. 라이브러리 (실행 파일 아님)

| 모듈 | 역할 |
|---|---|
| `model.py` | 상대좌표 이중적분기 + condensing 행렬. MPC의 예측 모델 |
| `predictor.py` | 표적 미래 궤적 (등속/등가속/폴리핏) |
| `mpc.py` | 비용·제약 정의, 2단 볼록 QP, 안전콘 |
| `reference.py` | MPC 계획 → 50 Hz 연속 setpoint 보간 |
| `frame.py` | ENU↔NED, 카메라/짐벌 회전 체인. 단위 테스트 있음 |

`python3 -m landing_mpc.frame` 으로 프레임 변환 자체 검증이 가능합니다.

---

## 4. 실행

```bash
./simulation/gazebo/run_gimbal.sh mission     # 전체 미션
./simulation/gazebo/run_gimbal.sh gimbal      # 짐벌+인식만
HEADLESS=1 ./simulation/gazebo/run_gimbal.sh mission
```

---

## 5. 이 스택은 시뮬레이션 전용입니다

`trailer_cue_node`가 gz에서 표적 위치를 읽고, `mission_manager_node`가 PX4
uXRCE 토픽으로 직접 setpoint를 씁니다. 따라서 이 미션을 실기체에 그대로
실행하면 안 됩니다. PX4/MAVROS 실기체 미션은 아직 구현·검증되지 않았으며,
운용자 승인·좌표변환·상태 감시·failsafe를 포함한 별도 어댑터와 bench/HIL
검증이 먼저 필요합니다.
