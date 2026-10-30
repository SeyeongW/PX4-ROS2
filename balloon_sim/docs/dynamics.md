# 카메라 기반 정적 풍선 접근: 좌표 추정·경로·비행 동역학

이 문서는 현재 기본 `stl_pacer_quad` 시뮬레이션의 코드와 설정을 설명한다. 기체와 배터리를 합한 **총질량 2 kg**, 사용자 STL 10개, PACER 모터 4개를 사용하는 멀티콥터 모델이다. 카메라가 추정한 정적 목표 좌표에 접근한 뒤 정지 대기한다. LiDAR, 기수 수평 고정익 전환, 게이트 통과는 기본 실행에 적용하지 않는다.

공간 경로는 **3차 Bézier**, 경로를 따라가는 시간함수는 **5차 다항식**이다. ROS가 위치·속도·가속도 목표를 전달하고 PX4가 모터를 제어하며 Gazebo가 힘과 토크를 적분한다. 기체의 위치를 직접 갱신하는 방식으로 비행하지 않는다.

실행 방법은 [실행 안내](run.md), 전체 구성은 [저장소 README](../../README.md)를 참고한다. 이 문서의 수식은 현재 구현 설명이며 실기체 물성 식별이나 비행 인증 결과가 아니다.

## 1. 기본 실험과 적용 범위

| 항목 | 현재 설정 | 근거 수준 |
|---|---|---|
| 기체 총질량 | 2.0 kg, 기체·배터리 포함 | 사용자 지정; 실측 없음 |
| 무게중심 | 기체 중심 `[0, 0, 0]` m | 사용자 방향에 따른 모델 가정 |
| 관성 | `diag(0.02, 0.02, 0.01)` kg·m² | 등가 균일 원통 가정 |
| 기체 외형 | STL 10개를 하나의 몸체에 조립 | 원본 형상 유지, mm 단위 가정 |
| 모터·프로펠러 | T-MOTOR PACER P2207 V3 1950KV, 6S, P49436-3 | 제조사 정적 추력표 참고 |
| 목표 | 지름 0.30 m 정적 구, 세계좌표 `[0, -40, 10]` m | 시뮬레이션 배치 |
| 드론 초기 위치 | `[-2, 0, 0.2045]` m | 시뮬레이션 배치 |
| 센서 | 짐벌 카메라, 1280×960, 수평 화각 12° | 시뮬레이션 광학 설정 |
| 이륙 | 출발 기준 3 m 수직 상승 | 계획·실제 실행 점검 |
| 목표 동작 | 센서 목표 좌표에서 정지 대기 | `target_is_virtual_waypoint=true`, `standoff=0` |
| 게이트 통과 | 기본 비활성 | `target_is_pass_through_gate=false` |

40 m는 시스템 기준 **수평 거리**다. 높이와 카메라 장착 위치를 포함한 시선 거리는 이보다 길다. 기체와 목표 표식 사이의 접촉은 가상 목표 설정에 따라 제외하지만, 이것이 일반 장애물 회피를 의미하지는 않는다.

설정 근거: [기체 프로필](../balloon_locator/config/stl_pacer_quad.json), [Gazebo launch](../balloon_locator/launch/gazebo.launch.py), [실험 장면](../balloon_locator/balloon_locator/gazebo_scene.py).

## 2. 좌표계와 센서 위치 추정

### 2.1 좌표계 정의

- 세계좌표 $W$: Gazebo ENU, $x$=동, $y$=북, $z$=위.
- 기체좌표 $B$: Gazebo FLU, $x$=앞, $y$=왼쪽, $z$=위. STL 기수 길이 방향은 변환 후 $+z_B$이므로 이 모델의 외형 기수와 일반 항공기 $+x_B$를 혼동하지 않는다.
- 광학좌표 $C$: $x$=영상 오른쪽, $y$=영상 아래, $z$=광학 전방.
- 관제 센서 NED: 원점은 세계 `[0,0,0]`, 북은 세계 $+y_W$.
- FCU 로컬 NED: PX4 EKF의 원점에 대응하는 별도 좌표계. 센서 NED와 원점이 같다고 가정하지 않는다.
- PX4 기체 FRD: FLU에서 $D=\operatorname{diag}(1,-1,-1)$로 축을 변환한다.

$R_{AB}$는 B좌표 벡터를 A좌표로 변환하는 회전행렬로 표기한다.

### 2.2 픽셀·알려진 크기·TF로 좌표 산출

왜곡 보정된 CameraInfo에서 $f_x,f_y,c_x,c_y$를 읽는다. 검출 중심이 $(u,v)$, 검출 지름이 $d_{px}$, 알려진 풍선 지름이 $D_b=0.30\,m$일 때:

$$
f=\frac{f_x+f_y}{2},\qquad
r=\frac{D_b f}{d_{px}}.
$$

이는 알려진 크기를 사용하는 **작은 시야각 근사**다. 코드에는 구의 정확한 접선 투영 역산이 구현되어 있지 않다. `exact_sphere`라는 시뮬레이션 보정 이름은 정확한 크기의 구를 장면에 배치했다는 뜻이며, 거리 역산식이 근사 없이 정확하다는 뜻이 아니다.

$$
\widetilde b_C=
\begin{bmatrix}(u-c_x)/f_x\\(v-c_y)/f_y\\1\end{bmatrix},
\qquad \widehat b_C=\frac{\widetilde b_C}{\|\widetilde b_C\|}.
$$

노출 시각 $t_e$의 카메라 TF를 사용한다. 회전 $R_{WC}(t_e)$와 카메라 광학 원점 $t_{WC}(t_e)$로:

$$
p_W=t_{WC}(t_e)+rR_{WC}(t_e)\widehat b_C.
$$

따라서 짐벌이 회전해도 동일한 세계좌표계로 결과를 해석한다. 노출 시각의 TF가 없거나 보정·프레임·측정 신선도가 유효하지 않으면 좌표를 제어 입력으로 승인하지 않는다. Gazebo 정답 목표 좌표는 위 식의 입력으로 사용하지 않는다.

센서 NED 변환은:

$$
S=\begin{bmatrix}0&1&0\\1&0&0\\0&0&-1\end{bmatrix},
\qquad p_{N,station}=Sp_W.
$$

출처: [geometry.py](../balloon_locator/balloon_locator/geometry.py)의 `optical_ray`, `range_from_diameter`; [balloon_locator_node.py](../balloon_locator/balloon_locator/balloon_locator_node.py)의 관측 시각 TF 변환; [gazebo_bridge.py](../balloon_locator/balloon_locator/gazebo_bridge.py)의 `mission_estimate`.

### 2.3 정적 좌표 추정과 불확실성

현재 정적 추정기는 속도 상태를 가진 등속 칼만필터가 아니다. 각 관측 $z_k$, 독립 측정 공분산 $R_k$, 정보행렬 $\Lambda_k$, 정보벡터 $\eta_k$에 대해:

$$
\Lambda_k=\lambda\Lambda_{k-1}+R_k^{-1},\qquad
\eta_k=\lambda\eta_{k-1}+R_k^{-1}z_k,
\qquad \widehat p_k=\Lambda_k^{-1}\eta_k.
$$

기본 망각계수는 $\lambda=1$이다. 반복 관측으로 줄어드는 무작위 오차와, 풍선 크기·초점거리·짐벌 영점·장착 위치에서 오는 공통 오차를 분리한다:

$$
P_k=\Lambda_k^{-1}+P_{common,k}.
$$

구현은 각 공통 공분산의 차이에서 양의 고유값 성분을 누적해 기존 오차 하한을 낮추지 않는다. 따라서 같은 영상을 여러 번 본다는 이유로 보정 오차가 0에 가까워졌다고 판단하지 않는다. 시선 단위벡터 $\widehat b$에 대한 기본 공분산 형태는:

$$
R=\sigma_r^2\widehat b\widehat b^T+
\sigma_\perp^2(I-\widehat b\widehat b^T),
$$

$$
\sigma_r=r\sqrt{(\sigma_{px}/d_{px})^2+\sigma_D^2},\qquad
\sigma_\perp=r\sqrt{(\sigma_{px}/f)^2+\sigma_\theta^2}.
$$

실제 노드는 픽셀 무작위 성분, 공통 크기/초점 오차, 장착 오차, TF 시각 오차를 나누어 위 형태에 추가한다. 시각 불확실성은 20 ms 간격의 TF 차분으로 얻은 시선 끝점 변화율 $\dot p$에 대해 $\sigma_t^2\dot p\dot p^T$로 근사한다. 단일 영상·알려진 크기를 사용하는 방법이므로 실제 풍선 변형이나 픽셀 편향은 반드시 실측해야 한다.

출처: [static_estimator.py](../balloon_locator/balloon_locator/static_estimator.py), [geometry.py](../balloon_locator/balloon_locator/geometry.py), [balloon_locator_node.py](../balloon_locator/balloon_locator/balloon_locator_node.py).

### 2.4 FCU 원점과 NED 변환

FCU 북방향을 세계좌표에서 $\psi_N$, FCU 로컬 원점의 세계좌표를 $o_W$라 두면:

$$
R_{NW}=\begin{bmatrix}
\cos\psi_N&\sin\psi_N&0\\
\sin\psi_N&-\cos\psi_N&0\\
0&0&-1
\end{bmatrix}.
$$

$$
p_N=R_{NW}(p_W-o_W),\qquad
v_N=R_{NW}v_W,\qquad a_N=R_{NW}a_W.
$$

현재 Gazebo GPS 기준은 $\psi_N=\pi/2$이다. 원점은 선언한 초기 위치 $H_W$와 지상 정지 상태의 MAVROS 로컬 ENU 위치 평균으로 구한다:

$$
o_W=H_W-\frac{1}{n}\sum_{k=1}^n p_{ENU,k}.
$$

최소 표본 수·기간·정지 조건을 검사하고, 미리보기 준비 시 지상 높이 기준을 다시 확인한다. 실제 장비에는 측량·북방향·HOME/EKF 재설정 검증이 별도로 필요하다.

**MAVROS API 주의:** mission 내부 메시지는 `jol/flight_ned`이지만, MAVROS의 ROS `PositionTarget` 입력 P/V/A는 ENU이다. `coordinate_frame=FRAME_LOCAL_NED`라는 MAVLink 프레임 라벨 때문에 NED 값을 그대로 ROS 입력에 넣으면 축을 두 번 잘못 해석할 수 있다. 현재 adapter가 NED→ENU로 돌려 보내고 MAVROS가 MAVLink NED로 변환한다.

출처: [static_flight_core.py](../support/jol_perception/jol_perception/static_flight_core.py)의 `flight_frame`, `to_ned`; [mavros_adapter.py](../support/jol_perception/jol_perception/mavros_adapter.py)의 `reference_from_samples`, `read_command`, `publish_setpoint`.

## 3. 3 m 수직 이륙과 3차 Bézier 경로

### 3.1 공간 제어점

세계좌표에서 출발 위치 $H$, 센서 락 목표 $G$, 수직 이륙 끝점 $L$을 사용한다:

$$
L=H+3e_z,\qquad e_z=[0,0,1]^T.
$$

첫 구간은 제어점 $[H,H+e_z,H+2e_z,L]$인 직선 Bézier다. x·y 위치를 유지하며 초기 3 m를 상승한다. 이후 곡선의 제어점은:

$$
h=\min\left(4,\frac{G_z-L_z}{2}\right),\qquad
P_0=L,\quad P_1=L+he_z,\quad P_2=L+2he_z,\quad P_3=G.
$$

4 m는 현재 launch의 `vertical_handle` 기본값이다. 목표는 이륙 끝점보다 최소 0.2 m 높아야 한다. `approach_handle`은 정지 접속용 일반 경로 옵션으로 남아 있지만, 현재 **연속 접속** 제어점 구성에는 사용하지 않는다.

$$
B(u)=(1-u)^3P_0+3(1-u)^2uP_1+3(1-u)u^2P_2+u^3P_3,
\qquad 0\leq u\leq1.
$$

$$
B'(u)=3\left[(1-u)^2(P_1-P_0)+2(1-u)u(P_2-P_1)+u^2(P_3-P_2)\right],
$$

$$
B''(u)=6\left[(1-u)(P_2-2P_1+P_0)+u(P_3-2P_2+P_1)\right],
\qquad B'''(u)=6(P_3-3P_2+3P_1-P_0).
$$

초기 곡선 접선과 2차 미분은:

$$
B'(0)=3he_z,\qquad B''(0)=0.
$$

수직 구간 끝과 곡선 시작의 속도 방향이 일치하고, 비영속도 접속에서 횡방향 가속도 점프가 생기지 않도록 구성했다. 현재 목표 높이에서는 $2h=G_z-L_z$가 되어 끝 접선은 수평이지만, 임의로 더 높은 목표를 지정하면 항상 수평 끝 접선인 것은 아니다.

출처: [static_flight_core.py](../support/jol_perception/jol_perception/static_flight_core.py)의 `build_static_plan`; [Gazebo launch](../balloon_locator/launch/gazebo.launch.py).

### 3.2 5차 시간함수와 C2 접속

각 구간의 시작 시각 $t_s$, 지속시간 $T$, 정규화 시간 $\tau=(t-t_s)/T$에 대해:

$$
u(\tau)=\sum_{j=0}^{5}\beta_j{5\choose j}(1-\tau)^{5-j}\tau^j.
$$

현재 비게이트 연속 경로에서 양 끝의 매개변수 가속도는 0이며:

$$
\dot u_0=\frac{v_{entry}}{3\|P_1-P_0\|},\qquad
\dot u_1=\frac{v_{exit}}{3\|P_3-P_2\|},\qquad
m_0=T\dot u_0,\quad m_1=T\dot u_1,
$$

$$
[\beta_0,\ldots,\beta_5]=
[0,m_0/5,2m_0/5,1-2m_1/5,1-m_1/5,1].
$$

수직 구간은 $v_{entry}=0,\;v_{exit}=0.35\,m/s$, 곡선 구간은 $v_{entry}=0.35\,m/s,\;v_{exit}=0$다. 수직 직선과 곡선 시작 모두 $B''=0$, 시간함수 양 끝 $\ddot u=0$이므로 계획상 연결부는:

$$
p_{takeoff}(T)=B(0)=L,\qquad
v_{takeoff}(T)=v_{curve}(0)=0.35e_z,\qquad
a_{takeoff}(T)=a_{curve}(0)=0.
$$

즉 위치·속도·가속도에 대한 C2 연결이다. 저크의 접속 연속성(C3)까지 보장한다는 뜻은 아니다. 정지 접속 모드에서는 $m_0=m_1=0$인 $u=10\tau^3-15\tau^4+6\tau^5$를 사용한다.

시간 미분을 포함한 운동량은:

$$
p(t)=B(u(t)),\qquad v(t)=B'(u)\dot u,
$$

$$
a(t)=B''(u)\dot u^2+B'(u)\ddot u,
$$

$$
j(t)=B'''(u)\dot u^3+3B''(u)\dot u\ddot u+B'(u)\dddot u.
$$

곡률은 미분이 0이 아닌 곳에서:

$$
\kappa(u)=\frac{\|B'(u)\times B''(u)\|}{\|B'(u)\|^3}.
$$

실제 mission은 측정 높이가 3 m에 도달했는지 확인한 뒤 다음 상태로 넘어간다. 접속 시각 제한, 제어 주기, EKF·모터·기체 응답이 있으므로 **계획의 C2를 실제 명령/기체 운동의 완전한 C2로 간주하지 않는다**.

출처: [static_flight_core.py](../support/jol_perception/jol_perception/static_flight_core.py)의 `_moving_segment`; [observation_path.py](../support/jol_perception/jol_perception/observation_path.py)의 `_evaluate`; [stl_mission.py](../balloon_locator/balloon_locator/stl_mission.py)의 `plan_time`, `tick`.

### 3.3 운동 제한과 시간 결정

현재 프로필 제한은 속도 1.5 m/s, 가속도 0.6 m/s², 저크 1.2 m/s³이다. 계획에는 피드백 여유를 둔다:

$$
v_{plan,max}=0.9\times1.5=1.35\;m/s,\qquad
a_{plan,max}=0.7\times0.6=0.42\;m/s^2,
\qquad j_{plan,max}=1.2\;m/s^3.
$$

3차 공간 곡선과 5차 시간함수의 합성 위치는 최대 15차다. 구현은 Bernstein 미분 제어점의 convex-hull을 6회 이분해 구간 전체의 속도·가속도·저크 상계를 계산한다. 시각화 샘플만 검사하는 방식이 아니다. 이 상계가 각 제한을 만족하고 시간 제어점이 단조 증가할 때까지 $T$를 늘린다. 이륙 최소 시간은 6 s다. 지나치게 큰 접속 속도와 제한이 양립하지 않으면 계획을 거부한다.

직각 비교 경로는 동일 출발·목표·제한에서 $H\rightarrow L\rightarrow C\rightarrow G$, $C=[H_x,H_y,G_z]^T$로 구성하며 꺾이는 접속점에서 정지한다. 현재 실측 비교는 Bézier만 수행했고 직각은 계획시간 비교다.

출처: [observation_path.py](../support/jol_perception/jol_perception/observation_path.py)의 `_composed_derivative_bounds`, `_segment`; [static_flight_core.py](../support/jol_perception/jol_perception/static_flight_core.py).

## 4. 질량·관성·모터와 Gazebo 동역학

### 4.1 기체 조립과 관성 가정

원본 STL의 공통 조립좌표를 mm→m로 축소하고, 원본 `[0,1013.5,0]`을 중심으로 $R_x(+90^\circ)$ 변환한다. 몸체 한 개에 10개 visual을 붙이며, 실제 접촉은 반경 약 0.136 m·길이 0.409 m의 보수적 원통으로 근사한다. 프로펠러 접촉은 별도 모델링하지 않는다.

총질량과 관성은:

$$
m=2\;kg,\quad c=[0,0,0]^T,\quad
J=\operatorname{diag}(0.02,0.02,0.01)\;kg\,m^2.
$$

위 관성은 반지름 $r_c=0.10\,m$, 높이 $h_c=0.30\,m$인 등가 균일 원통:

$$
J_{xx}=J_{yy}=\frac{m}{12}(3r_c^2+h_c^2),\qquad J_{zz}=\frac12mr_c^2
$$

에서 정한 **미측정 가정값**이다. STL 외피 부피·재료·배터리 배치로 식별한 값이 아니다. 총질량은 몸체 1.982 kg, 로터 4×0.004 kg, GPS 수치 분할 0.002 kg로 나누고 평행축 정리를 이용해 전체 CG·관성을 다시 맞춘다:

$$
J_{total}=\sum_i\left[J_i+m_i(\|d_i\|^2I-d_id_i^T)\right].
$$

출처: [stl_airframe.py](../balloon_locator/balloon_locator/stl_airframe.py), [기체 프로필](../balloon_locator/config/stl_pacer_quad.json).

### 4.2 정적 추력 피팅과 응답

제조사 P2207 V3 1950KV·6S·P49436-3 시험값:

| RPM | 추력(gf) | 전압(V) | 전류(A) |
|---:|---:|---:|---:|
| 13110 | 259.3 | 25.2 | 2.6 |
| 19960 | 634.4 | 25.2 | 9.2 |
| 24042 | 928.1 | 25.1 | 16.9 |
| 28976 | 1378.8 | 24.9 | 28.1 |
| 33011 | 1760.1 | 24.8 | 43.8 |

$$
\Omega=\mathrm{RPM}\frac{\pi}{30},\qquad T=k_T\Omega^2,\qquad Q=k_Q\Omega^2,
$$

$$
k_T=\frac{\sum_j\Omega_j^2T_j}{\sum_j\Omega_j^4}
=1.4477561466\times10^{-6}\;\frac{N}{(rad/s)^2}.
$$

추력 피팅 RMSE는 약 0.1336 N이다. 반작용 토크 자료가 없어 $Q/T=0.01\,m$를 **가정**했다:

$$
k_Q=0.01k_T=1.4477561466\times10^{-8}\;\frac{Nm}{(rad/s)^2}.
$$

모터 속도는 1차 응답으로 갱신한다:

$$
\dot\Omega=\frac{\Omega_{cmd}-\Omega}{\tau},\qquad
\Omega_{k+1}=e^{-\Delta t/\tau}\Omega_k+
(1-e^{-\Delta t/\tau})\Omega_{cmd,k}.
$$

가속 시 $\tau=0.025\,s$, 감속 시 $\tau=0.05\,s$이며 모두 가정값이다. 최대 33011 RPM에서 피팅한 모터 4개의 정적 추력/중량비는 약 3.5284다. 이것은 정적 제조사 조건의 수치이고 배터리 전압강하·열·장착 간섭·실제 비행 중 추력 보장을 의미하지 않는다.

$$
u_{hover}=\frac{mg}{4k_T\Omega_{max}^2}\simeq0.2834135861.
$$

이 값은 프로필의 PX4 hover thrust 설정에 사용된다. Gazebo 로터 조인트에는 `rotorVelocitySlowdownSim=50`이 적용되므로, 표시/관측된 조인트 각속도를 실제 모터 각속도로 해석할 때 50배 보정이 필요하다.

출처: [T-MOTOR 공식 제품](https://www.tmotorhobby.com/goods-1181-T-MOTORHOBBY%2BP2207%2BV3.html), [공식 추력표](https://www.tmotorhobby.com/images/20220411/1649663656357709.jpg), [motor_fit 구현](../support/jol_sim/launch/tailsitter_physics.py), [Gazebo 모터 구현](https://github.com/PX4/PX4-SITL_gazebo-classic/blob/6697ab169ceab512dc706acea63df4c882662c60/src/gazebo_motor_model.cpp), [1차 필터 구현](https://github.com/PX4/PX4-SITL_gazebo-classic/blob/6697ab169ceab512dc706acea63df4c882662c60/include/common.h).

### 4.3 강체 운동과 로터 배치

FLU 기준 로터 위치는 다음과 같고 모든 추력축은 $a_i=[0,0,1]^T$다:

$$
r_0=[0,-0.12,-0.015]^T,\quad r_1=[0,0.12,-0.015]^T,
$$

$$
r_2=[0.12,0,-0.015]^T,\quad r_3=[-0.12,0,-0.015]^T\quad(m).
$$

회전 방향은 `[ccw, ccw, cw, cw]`. 양의 $Q_i$에 대한 몸체 반작용 부호는 $s_i=[-1,-1,+1,+1]$이다. $\omega_B$는 기체 각속도이며 모터 각속도 $\Omega_i$와 구분한다.

$$
\dot p_W=v_W,\qquad
m\dot v_W=mg_W+R_{WB}\sum_iT_i a_i+F_{contact},
\qquad g_W=[0,0,-9.80665]^T,
$$

$$
J\dot\omega_B+\omega_B\times(J\omega_B)
=\sum_i(r_i-c)\times(T_i a_i)+\sum_i s_iQ_i a_i+\tau_{contact},
$$

$$
\dot R_{WB}=R_{WB}[\omega_B]_\times.
$$

이 식은 링크별 Gazebo 동역학의 기체 수준 요약이다. 실제 모델에는 회전 링크와 접촉 제약이 있으며 Python이 이 식을 직접 적분하는 것은 아니다. 기체 항력·날개 양력·실속·프로펠러 후류는 식별하지 않아 새 모델에 추가하지 않았다. 모터 플러그인의 rotor drag와 rolling moment도 0으로 둔다.

추력별 토크 계수 $b_i$와 할당행렬 $A$는:

$$
b_i=(r_i-c)\times a_i+s_i\frac{k_Q}{k_T}a_i,
\qquad
A=\begin{bmatrix}a_{0z}&a_{1z}&a_{2z}&a_{3z}\\b_0&b_1&b_2&b_3\end{bmatrix},
$$

$$
\begin{bmatrix}F_z\\\tau_x\\\tau_y\\\tau_z\end{bmatrix}
=A\begin{bmatrix}T_0\\T_1\\T_2\\T_3\end{bmatrix}.
$$

여기서 $b_i$는 3×1 열벡터여서 $A$는 4×4다. 생성 시 rank=4를 검사한다. PX4 `CA_ROTOR*_P*`, `A*`에는 FLU→FRD로 변환한 위치/축을 넣고, `KM`은 PX4 규약에 맞춰 ccw 양수·cw 음수로 전달한다. FLU 반작용 토크 부호와 PX4 파라미터 부호를 같은 좌표계 값처럼 비교하면 안 된다.

출처: [stl_airframe.py](../balloon_locator/balloon_locator/stl_airframe.py)의 `build_model`; [Gazebo 모터 힘·토크 구현](https://github.com/PX4/PX4-SITL_gazebo-classic/blob/6697ab169ceab512dc706acea63df4c882662c60/src/gazebo_motor_model.cpp).

## 5. PX4 경로 추종과 자세 제어

흐름은 `Bézier PVA → ROS mission → MAVROS → PX4 → 로터 힘/토크 → Gazebo → IMU/GPS → EKF 피드백`이다. `SYS_AUTOSTART=10015`와 Iris 센서/플러그인 템플릿을 이용하지만, Iris 외형·질량을 새 기체 물성으로 쓰지는 않는다.

다음은 PX4 제어기의 **포화·anti-windup 처리를 생략한 설명용 요약식**이다. 프로젝트 Python 코드가 아래 PID를 새로 구현하거나 SI 단위 토크를 직접 보내는 것은 아니다.

$$
v_{sp}=v_{ff}+K_p(p_{sp}-p),
$$

$$
a_{sp}=a_{ff}+K_v(v_{sp}-v)+I_v-K_d\dot v.
$$

PX4는 제한된 가속도/추력벡터에서 목표 roll/pitch를 구한다. NED의 수직 가속도를 포함하는 기본 형태는 $b_{z,d}\propto[-a_x,-a_y,g-a_z]^T$이며, 실제 코드는 수평/수직 가속도 분리 옵션, 기울기 제한, 최소/최대 추력 및 수직 우선 할당을 추가한다. 추력벡터와 yaw 목표를 목표 쿼터니언 $q_d$로 바꾼다.

$$
q_e=q^{-1}\otimes q_d,\qquad
e_q=2\operatorname{Im}(\operatorname{canonical}(q_e)),\qquad
\omega_{sp}=K_qe_q+\omega_{yaw,ff}.
$$

실제 PX4 자세제어는 추력축 우선의 reduced attitude 및 yaw 가중치를 적용한 $q_d$로 계산한다. 각속도 제어는:

$$
\tau_{cmd}=K_\omega(\omega_{sp}-\omega)+I_\omega
-K_{\dot\omega}\dot\omega+K_{ff}\omega_{sp}.
$$

이 결과는 PX4 제어할당의 정규화된 토크/추력 입력으로 처리되므로 이 요약의 $\tau_{cmd}$를 보정된 실물 Nm 계측값으로 보아서는 안 된다. 모터 포화·적분 제한·이륙 ramp도 실제 응답에 영향을 준다.

**현재 yaw 제한:** STL 멀티콥터 분기는 MAVROS PVA의 yaw와 yaw rate를 모두 `IGNORE`한다. 경로 접선 방향으로 yaw를 계속 맞추는 코드는 기존 `tailsitter20cm_reference` 분기에만 있다. 새 기체가 물리적으로 회전하고 PX4가 자세를 안정화하는 것과, 기수 방향을 경로 접선에 명시적으로 정렬하는 것은 별개다. 현재 STL 모델에 후자가 적용되었다고 주장하지 않는다. 또한 외형 기수가 위로 서 있는 새 형상을 수평으로 전환하는 고정익 비행은 구현 범위 밖이다.

출처: [mavros_adapter.py](../support/jol_perception/jol_perception/mavros_adapter.py); 로컬에서 대조한 PX4 commit `d6f12ad1c4f70ad3230afd7d86e971421e02fef4`의 [위치/속도 제어](https://github.com/PX4/PX4-Autopilot/blob/d6f12ad1c4f70ad3230afd7d86e971421e02fef4/src/modules/mc_pos_control/PositionControl/PositionControl.cpp), [자세 제어](https://github.com/PX4/PX4-Autopilot/blob/d6f12ad1c4f70ad3230afd7d86e971421e02fef4/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp), [각속도 제어](https://github.com/PX4/PX4-Autopilot/blob/d6f12ad1c4f70ad3230afd7d86e971421e02fef4/src/lib/rate_control/rate_control.cpp).

## 6. 락·도달 판정·근접 검출 유실

기본 동작은 `탐지 시작 → 짐벌 탐색/중심 정렬 → 유효 좌표 락 → 자동 경로 미리보기 → 발사 → 3 m 수직 상승 → 곡선 추종 → 도달 대기`다. 카메라는 미검출 시 yaw를 순회하고 약 13° 위를 보며, 검출 후 영상 중심 정렬을 수행한다. LiDAR 토픽이나 거리값을 이용하지 않는다.

기본 도달 판정은 계획 종료 뒤 EKF 위치 오차 $\|p-G\|<0.25\,m$, 속도 $\|v\|<0.2\,m/s$를 1 s 넘게 유지하는 것이다. 이는 EKF 기반 상태 판정이며 Gazebo 물리 위치가 실제로 0.25 m 안에 있다는 별도 검증을 대체하지 않는다.

새 STL 정적 목표는 마지막 1 m 안에서 시작한 검출 유실에 한해 기존 락 좌표를 최대 15 s 유지하며 접근할 수 있다. 다른 좌표·보정 변경·TF/시계 오류·먼 거리에서 시작한 유실은 이 예외로 처리하지 않는다. 도달 상태에서는 락 좌표에서 0.5 m 이내·속도 0.2 m/s 이내의 대기 조건을 적용한다. UI는 이 상태를 `락 좌표 유지(가림)`로 표시하고 현재 관측이 유효한 것처럼 꾸미지 않는다. FCU·자기위치·좌표계·추종오차 가드는 계속 적용된다.

기본 point 임무의 중단은 현재 위치 착륙이다. 게이트 임무의 실패 시 복귀 정책과 혼동하지 않는다.

출처: [stl_mission.py](../balloon_locator/balloon_locator/stl_mission.py)의 `terminal_static_hold_allowed`, `begin_landing`, `tick`; [stl_runtime.py](../balloon_locator/balloon_locator/stl_runtime.py); [gazebo_bridge.py](../balloon_locator/balloon_locator/gazebo_bridge.py).

## 7. 기존 실행 근거와 남은 검증

아래는 게시 구조 변경 전에 수행한 `20261030_stl_pacer_05`의 기록이다. 이번 Markdown 작성 중 새 비행 실험을 수행한 결과가 아니다. 해당 소스 기록의 manifest SHA-256은 `d038e9ed622a062e0574d05adfe47273d24a4baa84187ef625f9f50b28a7313d`이며, 게시용 폴더 전체의 해시와는 다르다. 임시 원시 로그는 저장 정책에 따라 삭제되어 이 저장소에 원시 시계열이 포함되어 있지 않다. 아래 값은 기존 실행 기록에서 인용한 요약값이다.

| 항목 | 기존 실행값 | 해석 |
|---|---:|---|
| 수직→곡선 접속 상승 높이 | 3.009216 m | 출발 기준 상승량 |
| 접속 상승속도 | 0.355185 m/s | 명목 0.35 m/s와 구분 |
| EKF 최대 추종오차 | 0.288294 m | EKF와 계획 비교 |
| 대기 종료 EKF→락 목표 오차 | 0.107289 m | 도달 상태 판정용 좌표 |
| Gazebo 실제 위치→락 목표 오차 | 0.290782 m | 0.25 m 정밀도 조건 미달 |
| 락 좌표→표적 정답 오차 | 0.047672 m | 시뮬레이션 단일 결과 |
| Gazebo 실제 최대속도 | 1.263094 m/s | 물리 상태 속도 |
| Gazebo 속도 차분 최대가속도 | 0.809259 m/s² | 설정 0.6 m/s² 초과 |
| EKF 속도 차분 최대가속도 | 0.814393 m/s² | 상태추정 잡음 포함 가능 |
| Bézier 계획시간 | 68.361939 s | 전체 계획시간 |
| 직각 계획시간 | 71.772860 s | 동일 조건 계획 비교만 수행 |

당시 UI 락·자동 미리보기·발사·3 m 상승·곡선 추종·도달 대기·중단·착륙/비무장을 확인했고, 빌드·SDF 검사·250개 테스트가 통과했다. 현재 게시용 경로 변경의 검증 결과는 저장소 README의 별도 확인 내용과 구분한다. 로그 행 수는 독립 반복 실험 수가 아니며 한 번의 도달을 일반적인 성공률로 해석하지 않는다.

남은 항목:

- 실제 가속도 상한 준수와 EKF–세계좌표 정합 개선. 계획 미분 상계 통과가 실제 운동 제한 통과를 뜻하지 않는다.
- 질량 분포·CG·관성, 모터 반작용 토크·응답, 배터리·열·공력의 실측.
- 공식 모터 외경 28.1 mm와 STL pod 내경 약 27 mm의 장착 간섭 해결. 형상 배치가 실물 조립 가능성을 보장하지 않는다.
- 풍선 실제 크기/변형·고정 줌·초점거리·픽셀 편향·짐벌 장착/시각 동기 보정. 시뮬레이션의 정확한 구·12° 화각 결과를 실물 40 m 성능으로 전용하지 않는다.
- 새 STL의 명시적 경로 yaw 정렬, 수평 기수 전환, 고정익 공력은 현재 적용되지 않았다.
- 마지막 접근 검출 유실의 영상 인과 확인. 가림 가능성은 있으나 당시 확보한 프레임만으로 원인을 확정하지 못했다.
- 실기체 비행과 새 물성의 HIL 검증은 수행하지 않았다.

## 부록 A. 게이트 통과 수식 — 현재 기본 모드에서는 비활성

게이트 중심 $G$, 출발 $H$, 수평 단위방향 $n$, 수평거리 $R$, 뒤쪽 대기거리 $D$를:

$$
R=\|G_{xy}-H_{xy}\|,\qquad
n=\frac{[G_x-H_x,G_y-H_y,0]^T}{R},\qquad F=G+Dn
$$

로 둔다. 게이트 방향은 출발→목표 수평방향으로 정한 실험 가정이며 영상에서 실제 게이트의 면 자세를 복원한 것이 아니다. 끝점을 $P_3=F$, 안쪽 제어점을 $P_1=L+he_z,P_2=L+2he_z$로 두면 수평 이동이 $(R+D)u^3$가 되어:

$$
u_g=\left(\frac{R}{R+D}\right)^{1/3},\qquad
h=\frac{(G_z-L_z)(1-u_g^3)}{3(u_g-u_g^3)}
$$

로 계산할 때 $B(u_g)=G$다. 목표는 멈추는 끝점이 아니라 곡선 내부의 통과점이 된다. 선택형 게이트 속도 조건은 같은 cubic을 de Casteljau로 나누고 양쪽 시간함수의 속도·접선 가속도를 맞춘다. 이 경우 매개변수 가속도는:

$$
\ddot u=\frac{a_t-\dot u^2(B'\cdot B'')/\|B'\|}{\|B'\|}
$$

이며, $\beta_2$와 $\beta_3$에 각각 양 끝 $T^2\ddot u/20$ 항을 더한다. 이는 기본 point 임무에서 사용하지 않는 분기다.

### 실제 게이트 통과 판정

연속하는 유효 EKF 위치 $p_0,p_1$의 게이트면 signed distance:

$$
d_0=(p_0-G)\cdot n,\qquad d_1=(p_1-G)\cdot n.
$$

앞→뒤 통과인 $d_0<0<d_1$일 때:

$$
\alpha=\frac{-d_0}{d_1-d_0},\qquad p_\times=p_0+\alpha(p_1-p_0).
$$

게이트 오른쪽/위 단위축을 $r_g,u_g^{axis}$, 폭/높이를 $w,h_g$, 기체 보호반경을 $r_b$라고 하면:

$$
|(p_\times-G)\cdot r_g|+r_b\leq\frac w2,\qquad
|(p_\times-G)\cdot u_g^{axis}|+r_b\leq\frac {h_g}2
$$

를 모두 만족해야 성공이다. 위치 구간 길이가 2 m를 넘는 점프, 역방향 이동, 단지 목표에 가까운 표본은 성공으로 처리하지 않는다. 호출부는 측정 시각의 신선도를 추가 검사한다. 기본 STL point 임무가 이 판정이나 실패 복귀를 수행한다고 설명해서는 안 된다.

출처: [static_flight_core.py](../support/jol_perception/jol_perception/static_flight_core.py)의 `build_static_plan`, `_split_cubic`, `_moving_segment`, `checkpoint_crossing`; [stl_mission.py](../balloon_locator/balloon_locator/stl_mission.py)의 `check_gate`, `begin_landing`.
