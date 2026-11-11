# 📐 Mathematical Algorithms & Control Architecture (`balloon_sim`)

본 문서는 `balloon_sim` 패키지에서 구현된 **지상-공중 협력형 쿼드 테일시터(Quad-Tailsitter) 자율 추적 및 폐루프 유도 제어 시스템**의 수학적 모델과 알고리즘을 상세히 기술합니다.

---

## 목차 (Table of Contents)

1. [다중 루프 캐스케이드 폐루프 아키텍처 (Multi-Loop Cascaded Closed-Loop)](#1-다중-루프-캐스케이드-폐루프-아키텍처-multi-loop-cascaded-closed-loop)
2. [비전 기반 풍선 형상 검증 및 타원 피팅 (Balloon Detection & Shape Verification)](#2-비전-기반-풍선-형상-검증-및-타원-피팅-balloon-detection--shape-verification)
3. [지상 팬-틸트 짐벌 비주얼 서보잉 및 핀홀 역투영 (Ground Gimbal Visual Servoing & Back-Projection)](#3-지상-팬-틸트-짐벌-비주얼-서보잉-및-핀홀-역투영-ground-gimbal-visual-servoing--back-projection)
4. [6-DOF 무향 칼만 필터 (Unscented Kalman Filter, UKF)](#4-6-dof-무향-칼만-필터-unscented-kalman-filter-ukf)
5. [쿼드 테일시터 공기역학 및 Wings-Level $\text{SO}(3)$ 자세 제어](#5-쿼드-테일시터-공기역학-및-wings-level-textso3-자세-제어)
6. [종말 온보드 3차원 시선각(LOS) 및 비례항법(PN) 유도 법칙](#6-종말-온보드-3차원-시선각los-및-비례항법pn-유도-법칙)

---

## 1. 다중 루프 캐스케이드 폐루프 아키텍처 (Multi-Loop Cascaded Closed-Loop)

전체 시스템은 외부 지상 관측 루프, 중기 유도 루프, 종말 온보드 시선각 유도 루프가 결합된 **3중 캐스케이드 폐루프(Multi-Loop Cascaded Closed-Loop)** 구조를 이룹니다.

```mermaid
flowchart TD
    subgraph Ground["Loop 1: 지상 관측 및 6-DOF UKF 추정 루프 (30 Hz -> 10 Hz)"]
        G1["지상 2축 짐벌 카메라 (1280x720, HFOV 12 deg)"] --> G2["HSV 색상 분할 + 원형도/타원 피팅 검증"]
        G2 -->|픽셀 오차 (du, dv)| G3["팬-틸트 비주얼 서보잉 (Visual Servoing)"]
        G3 -->|Joint Command| G1
        G2 -->|"(u, v, d_px) + TF(t_exposure)"| G4["6-DOF UKF (위치 + 속도 6차원 상태 추정)"]
    end

    subgraph Trigger["운용자 발진 승인 게이트 (Human-in-the-Loop Trigger)"]
        G4 -->|"n >= 3, s < 5.0 m"| T1["풍선 락온 완료 (STAGE 2: BALLOON LOCKED)"]
        T1 -->|"SPACE / ENTER 또는 /balloon/launch_approved"| T2["발진 승인 (STAGE 3: LAUNCH APPROVED)"]
    end

    subgraph MidCourse["Loop 2: 중기 유도 루프 (Mid-Course Guidance, 50 Hz)"]
        T2 --> M1["수직 이륙(TAKEOFF) -> 피치다운 천이(TRANSITION)"]
        G4 -->|"10 Hz /balloon/estimated_position"| M2["추정 좌표 지향 P+PN 유도 (TS_UKF_APPROACH)"]
        M1 --> M2
    end

    subgraph Terminal["Loop 3: 종말 온보드 비전 유도 루프 (Terminal Homing, 50 Hz)"]
        M2 -->|"거리 <= 40m & 온보드 카메라 Lock"| E1["온보드 3D LOS 광선 복원 + 광학 거리 R_opt 융합"]
        E1 --> E2["3차원 비례항법 (3D PN) + 공력 양력 보상"]
        E2 --> E3["통과 판정 (Cond 1 AND Cond 2) -> 역천이 제동 -> RTL"]
    end
```

---

## 2. 비전 기반 풍선 형상 검증 및 타원 피팅 (Balloon Detection & Shape Verification)

- **구현 파일**: [`balloon_detector.py`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/balloon_locator/balloon_locator/balloon_detector.py)

단순 색상 검출로 인해 주변 구조물이나 노이즈를 풍선으로 오인하는 것을 방지하기 위해, **3단계 기하학적 형상 검증**을 수행합니다.

### 2.1 HSV 이중 대역 분할 및 모폴로지 연산
적색 풍선 표면의 조명 반사(Highlight)와 음영(Shading)을 모두 포함하도록 HSV 색공간에서 두 개의 적색 대역을 결합합니다:

$$
M_{\text{raw}}(u,v) = \mathbb{I}\!\left(H \in [0, 12] \cup [168, 180],\; S \in [80, 255],\; V \in [60, 255]\right)
$$

이후 타원형 커널 $K_{5\times 5}$를 이용한 모폴로지 열림(Opening) 및 닫힘(Closing) 연산으로 고주파 배경 노이즈를 제거합니다.

### 2.2 원형도(Circularity) 및 최소외접원 충전율(Fill Ratio) 검증
검출된 외곽선(Contour)의 면적을 $A$, 둘레를 $P$, 최소외접원(`cv2.minEnclosingCircle`)의 반지름을 $r_{\text{enc}}$라 할 때:

$$
\text{Circularity} = \frac{4\pi A}{P^2} \ge 0.55, \qquad \text{Fill Ratio} = \frac{A}{\pi r_{\text{enc}}^2} \ge 0.55
$$

두 조건을 동시에 만족하는 후보만 구형(Spherical) 표적으로 인정합니다.

### 2.3 서브픽셀 타원 피팅 (`cv2.fitEllipse`)
단순 정수형 바운딩 박스($\max(w, h)$)는 1픽셀 양자화 오차로 인해 원거리($98\text{ m}$)에서 약 $2.5\text{ m}$의 거리 점프를 유발합니다. 이를 해결하기 위해 외곽선 점들에 최소자승 타원 피팅을 적용하여 서브픽셀 중심 $(u_c, v_c)$와 조화 평균 직경 $d_{\text{px}}$를 추출합니다:

$$
d_{\text{px}} = \sqrt{d_{\text{major}} \cdot d_{\text{minor}}}
$$

또한, 원거리에서 Gazebo 24각 폴리곤 구체 가장자리의 림 쉐이딩(Limb Shading)으로 인해 발생하는 서브픽셀 직경 축소 현상을 보상하기 위해 유효 보정 계수($+0.90\text{ px}$)를 적용합니다.

---

## 3. 지상 팬-틸트 짐벌 비주얼 서보잉 및 핀홀 역투영 (Ground Gimbal Visual Servoing & Back-Projection)

- **구현 파일**: [`gazebo_bridge.py`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/balloon_locator/balloon_locator/gazebo_bridge.py), [`geometry.py`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/balloon_locator/balloon_locator/geometry.py)

### 3.1 탐색(Sky Search) 및 락온 비주얼 서보잉(Visual Servoing)
1. **탐색 모드 (`SEARCHING`)**: 초기 오프타겟 방향($\psi_0 = -0.36\text{ rad}$, $\theta_0 = -0.14\text{ rad}$)에서 출발하여 상공을 리사주/정현파 패턴으로 스캔합니다:
   $$\psi_{\text{cmd}}(t) = -0.36 \cos(0.55\, t), \qquad \theta_{\text{cmd}}(t) = -0.14 - 0.2725 \sin(0.55\, t)$$
2. **락온 서보잉 모드 (`LOCKED`)**: 풍선이 3프레임 연속 검증되면 픽셀 중심 오차 $(e_u, e_v) = (u_c - c_x, v_c - c_y)$를 각도 오차로 변환하여 짐벌 팬/틸트 명령을 갱신합니다:
   $$\Delta \psi = -K_p \frac{u_c - c_x}{f_x}, \qquad \Delta \theta = -K_p \frac{v_c - c_y}{f_y} \qquad \left(f_x = f_y = \frac{W}{2\tan(\text{HFOV}/2)} = 6089.2\text{ px}\right)$$

### 3.2 구면 핀홀 역투영 (Exact Spherical Pinhole Back-Projection)
직경 $D = 0.30\text{ m}$인 구체가 광축으로부터 각도 $\alpha$만큼 떨어진 위치에 투영될 때, 광학 중심으로부터 구체 중심까지의 유클리드 거리 $r$과 카메라 광학 좌표계(`jol/camera_optical`, $Z$-전방, $X$-우측, $Y$-하방)에서의 3차원 위치 $\mathbf{p}_{\text{cam}}$은 다음과 같습니다:

$$
\mathbf{u}_{\text{ray}} = \frac{1}{\sqrt{\tilde{x}^2 + \tilde{y}^2 + 1}} \begin{bmatrix} \tilde{x} \\ \tilde{y} \\ 1 \end{bmatrix}, \qquad \tilde{x} = \frac{u_c - c_x}{f_x},\quad \tilde{y} = \frac{v_c - c_y}{f_y}
$$

$$
r = \frac{D / 2}{\sin(\beta / 2)} \approx \frac{f_x D}{d_{\text{px}}}, \qquad \mathbf{p}_{\text{cam}} = r\,\mathbf{u}_{\text{ray}}
$$

### 3.3 노출 시각(`stamp`) 동기화 TF 변환 및 비등방성 측정 공분산
짐벌이 회전하는 동안 프레임 지연으로 인한 각도 오차를 제거하기 위해, 이미지 헤더의 노출 타임스탬프 $t_{\text{img}}$에 정확히 일치하는 TF 변환 $(\mathbf{R}_{\text{world}\leftarrow\text{cam}}(t_{\text{img}}), \mathbf{t}_{\text{cam}}(t_{\text{img}}))$을 조회합니다:

$$
\mathbf{z}_{\text{world}} = \mathbf{R}_{\text{world}\leftarrow\text{cam}}(t_{\text{img}})\,\mathbf{p}_{\text{cam}} + \mathbf{t}_{\text{cam}}(t_{\text{img}})
$$

단안 거리 측정의 불확실성은 시선 방향(깊이 $r$)으로 거리 제곱에 비례($\sigma_r \propto r^2 / (f_x D)\,\sigma_d$)하고, 시선 수직 방향(각도)으로는 거리에 선형 비례($\sigma_\perp \propto r \sigma_{uv} / f_x$)합니다. 따라서 자코비안 $\mathbf{J}_{\text{cam}} = \frac{\partial \mathbf{p}_{\text{cam}}}{\partial (u, v, d_{\text{px}})}$를 통해 타원체 형태의 **비등방성 측정 공분산(Anisotropic Measurement Covariance)**을 구성합니다:

$$
\mathbf{R}_{\text{world}} = \mathbf{R}_{\text{world}\leftarrow\text{cam}} \left( \mathbf{J}_{\text{cam}} \operatorname{diag}(\sigma_u^2, \sigma_v^2, \sigma_d^2) \mathbf{J}_{\text{cam}}^T \right) \mathbf{R}_{\text{world}\leftarrow\text{cam}}^T + \sigma_{\text{tf}}^2 \mathbf{I}_3
$$

---

## 4. 6-DOF 무향 칼만 필터 (Unscented Kalman Filter, UKF)

- **구현 파일**: [`static_estimator.py`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/balloon_locator/balloon_locator/static_estimator.py) (`UKFTargetEstimator`)

정지된 풍선뿐만 아니라 바람에 의해 흔들리는(복원력을 받는 평균 회귀 운동) 풍선의 위치와 속도를 동시에 정밀 추정하기 위해 **6자유도(6-DOF) 상태 공간 무향 칼만 필터(UKF)**를 사용합니다.

### 4.1 상태 벡터 및 시그마 포인트 생성 (Merwe Scaled Sigma Points)
상태 벡터 $\mathbf{x} \in \mathbb{R}^6$는 월드 좌표계(ENU)에서의 3차원 위치와 3차원 속도로 정의됩니다 ($L = 6$):

$$
\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \end{bmatrix} = \begin{bmatrix} p_x & p_y & p_z & v_x & v_y & v_z \end{bmatrix}^T
$$

파라미터 $\alpha_{\text{ukf}} = 0.3$, $\beta_{\text{ukf}} = 2.0$, $\kappa_{\text{ukf}} = 0$, $\lambda = \alpha_{\text{ukf}}^2 (L + \kappa_{\text{ukf}}) - L$에 대해 $2L + 1 = 13$개의 시그마 포인트 $\mathcal{X}_i$를 생성합니다:

$$
\mathcal{X}_0 = \hat{\mathbf{x}}, \qquad \mathcal{X}_i = \hat{\mathbf{x}} + \left(\sqrt{(L + \lambda)\mathbf{P}}\right)_i, \qquad \mathcal{X}_{i+L} = \hat{\mathbf{x}} - \left(\sqrt{(L + \lambda)\mathbf{P}}\right)_i \quad (i = 1, \dots, L)
$$

여기서 $\sqrt{(L+\lambda)\mathbf{P}}$는 수치적 안정성을 위해 대칭 행렬 보정 후 촐레스키 분해(Cholesky Decomposition)로 계산합니다.

### 4.2 무향 예측 단계 (Unscented Prediction with Velocity Damping & Forgetting)
계류된 풍선의 공기저항 및 계류줄 감쇠 특성을 반영하여 속도 감쇠 계수 $\gamma = e^{-\eta \Delta t}$를 갖는 상태 전이 모델 $f(\mathbf{x}, \Delta t)$를 통해 각 시그마 포인트를 전파합니다:

$$
\mathcal{X}_i^{-} = f(\mathcal{X}_i, \Delta t) = \begin{bmatrix} \mathbf{p}_i + \mathbf{v}_i \Delta t \\ \gamma \mathbf{v}_i \end{bmatrix}
$$

예측 평균 $\hat{\mathbf{x}}^{-}$와 예측 공분산 $\mathbf{P}^{-}$는 지수 망각 인자(Exponential Forgetting Factor, $\tau_{\text{forget}}$)와 연속 시간 백색 가속도 프로세스 노이즈 $\mathbf{Q}(\Delta t)$를 결합하여 산출합니다:

$$
\hat{\mathbf{x}}^{-} = \sum_{i=0}^{2L} W_i^{(m)} \mathcal{X}_i^{-}, \qquad \mathbf{P}^{-} = e^{\Delta t / \tau_{\text{forget}}} \sum_{i=0}^{2L} W_i^{(c)} (\mathcal{X}_i^{-} - \hat{\mathbf{x}}^{-})(\mathcal{X}_i^{-} - \hat{\mathbf{x}}^{-})^T + \mathbf{Q}(\Delta t)
$$

### 4.3 마할라노비스($\chi^2$) 게이팅 및 무향 갱신 (Unscented Update)
관측 모델 $\mathcal{Z}_i = h(\mathcal{X}_i^{-}) = \mathbf{p}_i^{-}$로부터 예측 관측치 $\hat{\mathbf{z}}^{-}$와 혁신 공분산(Innovation Covariance) $\mathbf{S}$, 교차 공분산 $\mathbf{P}_{xz}$를 계산합니다:

$$
\mathbf{S} = \sum_{i=0}^{2L} W_i^{(c)} (\mathcal{Z}_i - \hat{\mathbf{z}}^{-})(\mathcal{Z}_i - \hat{\mathbf{z}}^{-})^T + \mathbf{R}_{\text{world}}, \qquad \mathbf{P}_{xz} = \sum_{i=0}^{2L} W_i^{(c)} (\mathcal{X}_i^{-} - \hat{\mathbf{x}}^{-})(\mathcal{Z}_i - \hat{\mathbf{z}}^{-})^T
$$

혁신 벡터 $\mathbf{y} = \mathbf{z}_{\text{world}} - \hat{\mathbf{z}}^{-}$에 대해 마할라노비스 거리 제곱 $d_M^2 = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y}$를 평가하여 이상치(Outlier)를 제거한 뒤 상태를 갱신합니다:

$$
\mathbf{K} = \mathbf{P}_{xz} \mathbf{S}^{-1}, \qquad \hat{\mathbf{x}} = \hat{\mathbf{x}}^{-} + \mathbf{K}\mathbf{y}, \qquad \mathbf{P} = \mathbf{P}^{-} - \mathbf{K}\mathbf{S}\mathbf{K}^T
$$

추정된 위치 공분산 블록 $\mathbf{P}_{pp} \in \mathbb{R}^{3\times 3}$의 최대 고유값 제곱근 $\sigma_{\max} = \sqrt{\lambda_{\max}(\mathbf{P}_{pp})}$을 뷰어의 불확실성 지표 `s`로 출력하며, 지상 관측 노드는 이를 **10 Hz** 주기로 드론에 브로드캐스트합니다.

---

## 5. 쿼드 테일시터 공기역학 및 Wings-Level $\text{SO}(3)$ 자세 제어

- **구현 파일**: [`tailsitter_guidance.py`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/tools/tailsitter_guidance.py), [`jol.sdf`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/balloon_locator/models/jol/jol.sdf)

### 5.1 쿼드 테일시터 좌표계 및 공력 특성 (`SYS_AUTOSTART=1045`, `CA_AIRFRAME=4`)
쿼드 테일시터는 기체 프레임(PX4 FRD 기준)에서 **4개의 로터 추력축과 온보드 카메라 광축이 모두 기체 코 방향($-\mathbf{z}_b = [0, 0, -1]^T$)으로 정렬**되어 있습니다.
- **수직 호버링(VTOL Hover)**: 기체 $-\mathbf{z}_b$ 축이 하늘(NED Up, $[0, 0, -1]^T$)을 향함 ($\text{Pitch} = 0^\circ$).
- **전진 수평 비행(Forward Flight)**: 기체가 앞으로 숙여(Pitch Down, $\theta \in [-50^\circ, -75^\circ]$) $-\mathbf{z}_b$ 축이 전방 비행 방향을 향하며, 주익(`libLiftDragPlugin.so`, 좌/우 날개 면적 각 $0.09\text{ m}^2$, $C_{L0}=0.15$, $C_{L\alpha}=4.2$)이 양력을 발생시킵니다.

### 5.2 Wings-Level ($\text{Roll}=0^\circ$) 정규직교 자세 행렬 및 쿼터니언 생성 (`nose_dir_to_quat_frd`)
기체의 코($-\mathbf{z}_b$)를 임의의 3차원 목표 단위벡터 $\mathbf{n}_{\text{des}} \in S^2$ (NED)로 지향할 때, 오일러 각 변환을 사용하면 $\theta = -90^\circ$ 부근에서 짐벌락(Gimbal Lock) 특이점이 발생합니다. 이를 원천 배제하기 위해 $\text{SO}(3)$ 회전 행렬을 직접 구성합니다:

1. 기체 $\mathbf{z}_b$ 축 결정:
   $$\mathbf{z}_b = -\frac{\mathbf{n}_{\text{des}}}{\|\mathbf{n}_{\text{des}}\|}$$
2. 주익 수평 유지(Wings-Level, $\text{Roll}=0^\circ$)를 위한 수평 진행 방위각 $\psi = \operatorname{atan2}(n_E, n_N)$ 및 우측 날개 축 $\mathbf{y}_b$ 결정:
   $$\mathbf{h} = \begin{bmatrix} \cos\psi \\ \sin\psi \\ 0 \end{bmatrix}, \qquad \mathbf{y}_b = \frac{\mathbf{z}_b \times \mathbf{h}}{\|\mathbf{z}_b \times \mathbf{h}\|}$$
3. 기체 배면(Belly) 축 $\mathbf{x}_b$ 완성 및 회전 행렬 $\mathbf{R}_{\text{NED}\leftarrow\text{FRD}} \in \text{SO}(3)$ 구성:
   $$\mathbf{x}_b = \mathbf{y}_b \times \mathbf{z}_b, \qquad \mathbf{R}_{\text{NED}\leftarrow\text{FRD}} = \begin{bmatrix} \mathbf{x}_b & \mathbf{y}_b & \mathbf{z}_b \end{bmatrix}$$

이 회전 행렬을 Shepperd 알고리즘(`rotmat_to_quat`)을 통해 단위 쿼터니언 $\mathbf{q}_{\text{des}} = [q_w, q_x, q_y, q_z]^T$로 변환하여 MAVLink `SET_ATTITUDE_TARGET` 메시지로 **50 Hz** 전송합니다.

### 5.3 자세-추력 분리 제어 및 고속 주익 양력 보상 (`compute_tailsitter_cmd`)
전진 비행 중 요구 가속도 벡터 $\mathbf{a}_{\text{cmd}} = [a_N, a_E, a_D]^T$가 주어졌을 때, 기체 코 방향과 로터 추력을 단순 합벡터 $\mathbf{f}_{\text{tot}} = \mathbf{a}_{\text{cmd}} - \mathbf{g}$로 결합하면 원거리 접근 시 과도한 추력으로 인해 고도가 솟구치는(Climb Overshoot) 문제가 발생합니다. 이를 해결하기 위해 **코 지향 벡터($\mathbf{n}_{\text{des}}$)와 스칼라 추력($T_{\text{norm}}$)을 분리(Decouple)**하고, 대기속도 $V = \|\mathbf{v}\|$에 따른 주익 공기역학적 양력 가속도 $a_{\text{lift}}(V)$를 중력에서 차감합니다:

$$
g_{\text{eff}}(V) = 9.81 \cdot \operatorname{clip}\!\left(1.0 - 0.042\max(0, V - 4.0),\; 0.35,\; 1.0\right)
$$

$$
\mathbf{f}_{\text{dir}} = \begin{bmatrix} a_N \\ a_E \\ a_D - g_{\text{eff}}(V) \end{bmatrix}, \qquad \mathbf{n}_{\text{des}} = \frac{\mathbf{f}_{\text{dir}}}{\|\mathbf{f}_{\text{dir}}\|}
$$

---

## 6. 종말 온보드 3차원 시선각(LOS) 및 비례항법(PN) 유도 법칙

- **구현 파일**: [`tailsitter_guidance.py`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/tools/tailsitter_guidance.py)

### 6.1 중기 UKF 유도에서 종말 온보드 비전 유도로의 핸드오버 (Handover)
- **중기 유도 (`TS_UKF_APPROACH`)**: 지상 카메라의 10 Hz UKF 추정 좌표 $\hat{\mathbf{p}}_{\text{ukf}}$와 추정 속도 $\hat{\mathbf{v}}_{\text{ukf}}$를 수신하여 비례 제어 + 속도 감쇠 + 3D 비례항법(PN) 결합 가속도를 생성합니다.
- **종말 유도 전환 조건 (`TS_LOS_PURSUIT`)**: 표적까지의 거리가 $R \le 40.0\text{ m}$ 이내로 진입하고, 드론 코에 장착된 광각 카메라(`1280×720`, `HFOV=90°`, $F_X = 640\text{ px}$)가 풍선을 연속 2프레임 이상 포착하면 즉시 온보드 카메라 직결 폐루프로 전환합니다.

### 6.2 온보드 3차원 시선 단위벡터($\mathbf{u}_{\text{los}}$) 및 광학 거리($R_{\text{opt}}$) 복원 (`pixel_to_los_ned`)
온보드 카메라 이미지에서 검출된 풍선 픽셀 중심 $(u, v)$와 픽셀 직경 $d_{\text{px}}$로부터 기체 FRD 프레임의 시선 광선을 생성하고, 현재 드론의 실시간 자세 회전 행렬 $\mathbf{R}_{\text{NED}\leftarrow\text{FRD}}(\phi, \theta, \psi)$을 곱해 NED 월드 시선 단위벡터 $\mathbf{u}_{\text{los}}$를 복원합니다:

$$
\mathbf{v}_{\text{frd}} = \begin{bmatrix} -\frac{v - c_y}{F_Y} \\ \frac{u - c_x}{F_X} \\ -1.0 \end{bmatrix}, \qquad \mathbf{u}_{\text{los}} = \frac{\mathbf{R}_{\text{NED}\leftarrow\text{FRD}}\,\mathbf{v}_{\text{frd}}}{\|\mathbf{R}_{\text{NED}\leftarrow\text{FRD}}\,\mathbf{v}_{\text{frd}}\|}
$$

근접 구간($d_{\text{px}} \ge 12\text{ px}$)에서는 온보드 카메라의 겉보기 직경으로부터 광학 거리 $R_{\text{opt}}$를 직접 산출하여, 지상 카메라 원거리 깊이의 잔여 바이어스까지 실시간으로 보정합니다:

$$
R_{\text{opt}} = \frac{F_X \cdot D_{\text{balloon}}}{d_{\text{px}}}, \qquad \hat{\mathbf{p}}_{\text{target}} \leftarrow (1 - \alpha)\hat{\mathbf{p}}_{\text{target}} + \alpha \left(\mathbf{p}_{\text{drone}} + R_{\text{opt}}\mathbf{u}_{\text{los}}\right)
$$

### 6.3 3차원 비례항법 (3D Proportional Navigation, PN) 및 시선 정렬 유도
드론에 대한 표적의 상대 위치 $\mathbf{r} = \hat{\mathbf{p}}_{\text{target}} - \mathbf{p}_{\text{drone}}$과 상대 속도 $\mathbf{v}_{\text{rel}} = \hat{\mathbf{v}}_{\text{target}} - \mathbf{v}_{\text{drone}}$으로부터 3차원 시선 각속도 벡터 $\boldsymbol{\Omega}_{\text{los}}$와 접근 속도 $V_c$를 계산합니다:

$$
\boldsymbol{\Omega}_{\text{los}} = \frac{\mathbf{r} \times \mathbf{v}_{\text{rel}}}{\|\mathbf{r}\|^2}, \qquad V_c = -\frac{\mathbf{r} \cdot \mathbf{v}_{\text{rel}}}{\|\mathbf{r}\|}
$$

유효 항법 상수 $N' = 4.0$을 적용한 3차원 비례항법 가속도 $\mathbf{a}_{\text{PN}}$과 시선 방향 속도 정렬 제어기를 결합하여 종말 가속도 명령을 생성합니다:

$$
\mathbf{a}_{\text{PN}} = N' V_c \left(\boldsymbol{\Omega}_{\text{los}} \times \mathbf{u}_{\text{los}}\right), \qquad \mathbf{a}_{\text{cmd}} = K_v \left(V_{\text{des}}\mathbf{u}_{\text{los}} - \mathbf{v}_{\text{drone}}\right) + \mathbf{a}_{\text{PN}}
$$

종말 단계에서는 기체 코의 피치다운 하한 제한을 해제(`allow_dive=True`)하여 기체 코($-\mathbf{z}_b$)가 풍선 중심 아래 방향까지 자유롭게 숙여지도록 보장합니다.

### 6.4 동시 만족 통과 판정 (Simultaneous Pass-Through Condition) 및 역천이 복귀
원거리에서 일시적인 그림자나 짐벌 가림으로 인해 조기에 통과로 오판하는 것을 방지하기 위해, 다음 **두 가지 물리적 조건이 동시에(AND) 만족**될 때만 풍선 관통 완료(`TS_BRAKE_TO_HOVER`)로 판정합니다:

1. **Condition 1 (3차원 유클리드 최근접 진입)**:
   $$\|\mathbf{p}_{\text{drone}} - \hat{\mathbf{p}}_{\text{target}}\|_2 < 2.2\text{ m}$$
2. **Condition 2 (목표 평면 통과 — 내적 부호 반전 또는 근접 브레이크어웨이)**:
   $$(\mathbf{p}_{\text{drone}} - \hat{\mathbf{p}}_{\text{target}}) \cdot \hat{\mathbf{u}}_{\text{approach}} > 0.3\text{ m} \quad \text{or} \quad \left(\text{Frame Loss after } d_{\text{px,max}} \ge 25\text{ px}\right)$$

통과 직후 기수 피치를 $+15^\circ \sim +25^\circ$로 들어 올리는 **역천이 공기역학적 제동(Back-Transition Aerodynamic Braking)**을 2.5초간 수행하여 수평 속도를 감속한 뒤, PX4 `AUTO.RTL` 모드를 트리거하여 이륙 원점으로 안전하게 복귀 및 수직 착륙합니다.
