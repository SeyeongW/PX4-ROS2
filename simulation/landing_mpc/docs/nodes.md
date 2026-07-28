# landing_mpc — 노드별 역할과 수식

움직이는 아루코 표적에 대한 정밀착륙 스택. 설계 근거는 `LANDING_MPC_SPEC.md`,
개념 설명은 `MPC_동적착륙_설계문서.docx`.

> 수식은 LaTeX(`$...$`, `$$...$$`)입니다. GitHub은 그대로 렌더링하고, VSCode는
> 내장 미리보기(`Ctrl+Shift+V`)에서 보이지 않으면 **Markdown+Math** 또는
> **Markdown Preview Enhanced** 확장을 설치하세요.

핵심 개념: 정밀착륙 = **상대좌표 랑데부(도킹)**. 표적에 붙은 좌표계에서
상대위치·상대속도를 동시에 0으로 몬다.

---

## 0. 파이프라인

```
[카메라]
   │ /down_camera/image
   ▼
aruco_detector_node ──► /aruco/pose_cam      (카메라 광학 프레임)
   │
   ▼
marker_tf_node      ──► /marker/measured     (로컬 ENU, 검출 시에만)
   │   ▲ 기체 자세·위치 (PX4)
   ▼
marker_kf_node      ──► /marker/position     (연속 추정 + coast)
   │                    /marker/velocity
   │                    /marker/valid
   ▼
mission_manager_node ─ predictor ─ mpc ─ reference ─► /fmu/in/trajectory_setpoint
                        (표적 미래)  (최적 입력)  (조밀 보간)
```

**추정(KF)과 제어(MPC)의 분업**

| | KF | MPC |
|---|---|---|
| 정체 | 추정기 | 제어기 |
| 질문 | "지금 어디 있나?" | "뭘 해야 하나?" |
| 시간 | 과거 측정 융합 | 미래 입력 최적화 |
| 다룸 | 노이즈·결측 | 비용·제약 |

둘 다 동역학 모델을 쓰지만 목적이 다르고, **경쟁이 아니라 직렬 연결**된다.

---

# 1. 제어 계층

## 1.1 model.py — 드론 동역학 (예측 모델)

**한 줄**: 드론을 상대좌표 이중적분기로 두고, N스텝 미래를 행렬 한 번에 계산한다.

상대 상태 $x=[p,\ v]^\top$ (표적 대비), 입력은 드론 가속도:

$$\dot p = v, \qquad \dot v = a_d - a_t$$

| 기호 | 정의 | 단위 |
|---|---|---|
| $p$ | 상대 위치 $p_{drone}-p_{target}$ | m |
| $v$ | 상대 속도 | m/s |
| $a_d$ | 드론 가속도 (**제어 입력**) | m/s² |
| $a_t$ | 표적 가속도 (**기지의 피드포워드**) | m/s² |

이중적분기 + 입력 계단유지(ZOH)이므로 **등가속 정확 이산화**가 성립한다
(이 경우 RK4와 Euler가 동일):

$$p_{k+1} = p_k + v_k\,\Delta t + \tfrac12 a_k\,\Delta t^2,
\qquad v_{k+1} = v_k + a_k\,\Delta t$$

### Condensing — 미래 전체를 행렬곱으로

$$P = F_p\,x_0 + G_p\,A, \qquad V = F_v\,x_0 + G_v\,A$$

$$[F_p]_{k,:} = (1,\ k\,\Delta t), \quad [F_v]_{k,:} = (0,\ 1)$$

$$[G_p]_{k,j} = \Delta t^2\!\left(\tfrac12 + (k-1-j)\right),\qquad
[G_v]_{k,j} = \Delta t \quad (j<k)$$

| 기호 | 정의 |
|---|---|
| $N$ | 예측 지평 길이(스텝 수) |
| $\Delta t$ | MPC 스텝 |
| $x_0=[p_0,v_0]^\top$ | 현재 상대 상태 |
| $A=[a_0,\dots,a_{N-1}]^\top$ | 구간별 순가속도 열 |
| $P,V\in\mathbb R^{N}$ | 미래 위치·속도 (스텝 $1..N$) |
| $F_p,F_v$ | 자유응답(입력 0) 사상 |
| $G_p,G_v$ | 입력→상태 사상 |

$[G_p]_{k,j}$의 의미: $a_j$는 준 순간 반스텝($\tfrac12\Delta t^2$)만 위치에 넣고,
그 뒤로는 속도가 되어 매 스텝 $\Delta t^2$씩 위치를 민다.

**MPC가 후보 입력을 넣을 때마다 for문이 아니라 $G_p A$ 한 번으로 미래가 나온다.**

---

## 1.2 predictor.py — 표적의 미래

**한 줄**: 표적은 조종할 수 없으므로 관측으로 외삽한다. $\tau=k\Delta t$.

**① 등속** (직선 표적)

$$p_t(\tau)=p_t+\tau\,v_t,\qquad v_t(\tau)=v_t,\qquad a_t(\tau)=0$$

**② 등가속** (완만한 곡선)

$$p_t(\tau)=p_t+\tau v_t+\tfrac12\tau^2 a_t,\qquad
v_t(\tau)=v_t+\tau a_t,\qquad a_t(\tau)=a_t$$

**③ 폴리핏** (급한 곡선) — 최근 궤적에 최소제곱 다항식 $c$를 적합하고 미분:

$$p_t(\tau)\approx\sum_{i=0}^{d} c_i\,\tau^{\,d-i},\qquad
v_t=\frac{dp_t}{d\tau},\qquad a_t=\frac{d^2p_t}{d\tau^2}$$

예측된 $a_t$가 **곡선표적 lag를 없애는 피드포워드**다.

---

## 1.3 mpc.py — 제어기 본체

**한 줄**: 상대좌표 비용을 최소화하는 가속도 시퀀스를 매 틱 QP로 풀고 첫 스텝만 쓴다.

### 비용함수

$$J=\sum_{k=1}^{N}\Big[
w_{xy}\lVert p_{xy,k}\rVert^2
+ w_z\,(p_{z,k}-z_{\mathrm{ref},k})^2
+ w_{vxy}\lVert v_{xy,k}\rVert^2
+ w_{vz}\,v_{z,k}^2
+ w_a\lVert a_k\rVert^2\Big]$$
$$\quad+\ w_f\big(\lVert p_N\rVert^2+\lVert v_N\rVert^2\big)
\ +\ w_j\sum_{k=0}^{N-1}\lVert a_k-a_{k-1}\rVert^2$$

| 기호 | 정의 | 기본값 |
|---|---|---|
| $w_{xy},w_z$ | 수평·수직 위치 오차 가중 | 6.0, 3.0 |
| $w_{vxy},w_{vz}$ | 상대속도 가중 (속도 정합) | 1.5, 1.5 |
| $w_a$ | 입력 크기 가중 (에너지) | 0.05 |
| $w_f$ | **터미널** 가중 — 착륙 성패를 가름 | 40.0 |
| $w_j$ | 저크 소프트 가중 | 0.5 |
| $z_{\mathrm{ref},k}$ | 접근 corridor 높이(아래 §안전콘) | — |
| $a_{-1}$ | **직전에 실제 적용한** 가속도 (재계획 연속성) | — |

### 표적 가속도를 자유응답에 접기

순가속도가 $a_d-a_t$이므로:

$$p_{\mathrm{free}} = F_p x_0 - G_p A_t,\qquad
v_{\mathrm{free}} = F_v x_0 - G_v A_t$$

$$P = p_{\mathrm{free}} + G_p u,\qquad V = v_{\mathrm{free}} + G_v u$$

여기서 $u$가 결정변수(드론 가속도 열), $A_t$는 예측된 표적 가속도 열.

### $U$에 대한 2차식으로 축약

$P,V$를 대입하면 $J$는 $u$의 2차식이 된다:

$$J(u)=\tfrac12 u^\top H u + f^\top u + \mathrm{const}$$

$$H = 2\Big(w_p G_p^\top G_p + w_v G_v^\top G_v + w_a I + w_j D^\top D\Big)
 + 2w_f\big(g_p g_p^\top + g_v g_v^\top\big)$$

$$f = 2\Big(w_p G_p^\top (p_{\mathrm{free}}-p_{\mathrm{ref}})
 + w_v G_v^\top v_{\mathrm{free}}\Big)
 + 2w_f\big(g_p\,\delta_p + g_v\,\delta_v\big)
 - 2w_j D^\top e_1\,a_{-1}$$

| 기호 | 정의 |
|---|---|
| $g_p=[G_p]_{N,:},\ g_v=[G_v]_{N,:}$ | 지평 끝(터미널) 행 |
| $\delta_p = p_{\mathrm{free},N}-p_{\mathrm{ref},N}$, $\delta_v=v_{\mathrm{free},N}$ | 터미널 잔차 |
| $D$ | 차분 행렬, $(Du)_k=u_k-u_{k-1}$ |
| $e_1$ | 첫 성분만 1인 단위벡터 |

> $\sum(G_p u)^2$을 전개하면 $u^\top(G_p^\top G_p)u$ — "제곱합"이 행렬로는 $G_p^\top G_p$.

### 최적성 조건

비용이 볼록(제곱합)이므로 최소점은 유일하고

$$\nabla J = Hu+f = 0$$

제약이 없으면 $Hu=-f$ 선형 풀이 한 번(뉴턴 1스텝). 제약이 있어 QP 솔버를 쓰되
가짜 골짜기가 없다.

### 제약 — 전부 $u$의 선형 부등식

**① 박스(추력 대용)**

$$-a_{\max}\le a_k\le a_{\max}$$

**② 저크(슬루레이트) — 점진적 가속·감속**

$$\lvert a_k-a_{k-1}\rvert \le j_{\max}\Delta t$$

0번 행은 $a_{-1}$ 기준이라 **재계획 경계에서도 가속 프로파일이 이어진다.**

측정된 무릎: $j_{\max}\Delta t = 0.2\ \mathrm{m/s^2}$/스텝

| $j_{\max}\Delta t$ | 정상상태 $\lVert v_{xy}\rVert$ | corridor 침범 | 결과 |
|---|---|---|---|
| 0.05 | 1.487 | 288 | 착륙 실패 |
| 0.10 | 0.241 | 30 | 추종 악화 |
| **0.20** | **0.055** | **0** | **최적** |
| 0.80 | 0.032 | 0 | 기체 쏠림 |

**③ 속도 — 저크를 고려한 실현가능 포락선**

단순히 $\lvert v\rvert\le v_{\max}$로 두면, 현재 속도가 이미 한계를 넘었을 때
1스텝 만에 되돌리라는 요구가 되어 **QP가 매번 실패**한다(SITL에서 50% 실패 관측).
저크 제한 하에서 **실제 도달 가능한** 속도로 완화한다.

가장 강한 감속($s=-1$) / 가속($s=+1$)은 가속도를 $j_{\max}$로 램프시켜 $\pm a_{\max}$에 포화:

$$t_1=\frac{a_{\max}-s\,a_{-1}}{j_{\max}}$$

$$v^{s}(t)=\begin{cases}
v_0 + a_{-1}t + s\,\dfrac{j_{\max}}{2}t^2, & t\le t_1\\[2mm]
v^{s}(t_1) + s\,a_{\max}(t-t_1), & t> t_1
\end{cases}$$

$$\bar v_k=\max\big(v_{hi},\,v^{-}(t_k)\big),\qquad
\underline v_k=\min\big(v_{lo},\,v^{+}(t_k)\big)$$

$$\underline v_k \le v_{\mathrm{free},k} + (G_v u)_k \le \bar v_k$$

### 안전콘 — 2단 풀이 + 소프트 레퍼런스

콘 $p_z\ge k_c\lVert p_{xy}\rVert$는 xy와 z가 얽힌 비선형(SOC)이라 통째로는
실시간이 깨진다. **xy를 먼저 풀어 $\lVert p_{xy}\rVert$를 상수로 굳힌 뒤** z를 푼다:

$$z_{\mathrm{ref},k}=\min\big(k_c\,\lVert \hat p_{xy,k}\rVert,\ z_c^{\max}\big)$$

**왜 하드 제약이 아니라 레퍼런스인가** — 하한으로 넣으면 장거리에서 실현 불가:

$$\underbrace{k_c\lVert p_{xy}\rVert}_{0.25\times 70\approx 18\,\mathrm m}
\ \gg\
\underbrace{p_{z,0}+\tfrac12 a_{\max}T^2}_{3.9+\frac12\cdot4\cdot2^2\approx 11.9\,\mathrm m}$$

→ SITL에서 `qp_fail=128/128`(전량 실패) 관측. 레퍼런스로 두면 z 비용이
corridor로 끌어올릴 뿐이라 **항상 해가 있고**, $\lVert p_{xy}\rVert\to0$이면
corridor도 0으로 내려가 하강이 열린다(게이팅 효과 동일).

### 실패 폴백 — 속도 포화 PD

QP가 실패해도 **낡은 해를 재사용하지 않는다**(그러면 하던 방향으로 계속 가속해 발산):

$$v_{\mathrm{des}}=\mathrm{clip}\big(-k_p\,p_0,\ v_{lo},\ v_{hi}\big),\qquad
a_{fb}=\mathrm{clip}\big(k_d(v_{\mathrm{des}}-v_0),\ \pm a_{\max}\big)$$

$k_d=2\sqrt{k_p}$면 이중적분기에 대해 임계감쇠. 속도 포화가 없으면 70 m 오차에서
$a_{\max}$로 수 초간 포화되어 21 m/s까지 가속(실제로 트레일러 충돌).

---

## 1.4 reference.py — 고주파 참조 스트리밍

**한 줄**: MPC 계획을 버리지 않고 경과시간으로 보간해 50 Hz로 흘린다.

### 구간내 해가 정확히 2차식인 이유

플랜트가 이중적분기 + 가속도 ZOH이므로, 스텝 $k$ 구간에서 $a$가 상수. 정확히 적분하면
$s=t-t_k\in[0,\Delta t]$에 대해

$$a(t)=a_k \qquad\text{(구간별 상수)}$$
$$v(t)=v_k+a_k\,s \qquad\text{(구간별 1차)}$$
$$p(t)=p_k+v_k\,s+\tfrac12 a_k\,s^2 \qquad\text{(구간별 2차)}$$

$s=\Delta t$를 넣으면 condensing이 쓴 등가속 갱신과 **완전히 일치**한다. 즉 이
2차식은 근사가 아니라 **MPC 자기 모델의 정확한 연속시간 해**이며, 계획면은 $C^1$
2차 스플라인이다. 따라서 $(p,v,a)$가 서로 **정확히 일관**(속도는 위치의 도함수,
가속도는 속도의 도함수)하고, 이것이 PX4가 매끄럽게 추종하는 조건이다.

### 상수 선행거리 $L$ — 진동 방지

$$t_{\mathrm{sample}}=\mathrm{clip}\big(\tau+L,\ 0,\ N\Delta t\big),\qquad L=\Delta t$$

| 방식 | 한 solve 주기 동안 선행거리 | 증상 |
|---|---|---|
| 한 점만 고정 발행 | $0.2\to0$ 감소 후 점프(톱니) | 뚝뚝 끊김 |
| 보간, $L=0$ | 항상 0 (명령=현재위치) | 전진력 없음 → 흔들림 |
| 보간, $L=\Delta t$ | 항상 일정 | 안정 |

### 상대 → 절대

$$p_{\mathrm{cmd}}(\tau)=p_t(\tau)+p_{\mathrm{rel}}(\tau),\quad
v_{\mathrm{cmd}}(\tau)=v_t(\tau)+v_{\mathrm{rel}}(\tau),\quad
a_{\mathrm{cmd}}(\tau)=a_t(\tau)+a_{\mathrm{rel}}(\tau)$$

| 기호 | 정의 |
|---|---|
| $\tau$ | 마지막 solve 이후 경과시간 |
| $L$ | 상수 선행시간(look-ahead) |
| $j$ | 활성 스텝 $j=\lfloor \tau/\Delta t\rfloor$ |
| $p_k,v_k,a_k$ | 상대 계획의 knot / 구간 가속도 |

**두 주파수 분리**: MPC solve 10 Hz(무거운 QP), 참조 스트림 50 Hz(가벼운 표본).

---

# 2. 인식 계층

## 2.1 frame.py — 좌표 변환 (전담)

**ENU ↔ NED** (자기역함수):

$$v_{NED}=S\,v_{ENU},\qquad
S=\begin{pmatrix}0&1&0\\1&0&0\\0&0&-1\end{pmatrix}$$

**하방 카메라 → 월드**. solvePnP는 마커를 **카메라 광학 프레임**(REP-103:
$+X$ 오른쪽, $+Y$ 아래, $+Z$ 전방)으로 준다. 카메라가 피치 $+90^\circ$로 장착돼
정확히 아래를 보므로, 광학 좌표 $(u,v,w)$에 대해

$$p_{FLU}=(-v,\ -u,\ -w)$$

$$p_{ENU}=S\,R(q)\,D\,p_{FLU},\qquad D=\mathrm{diag}(1,-1,-1)$$

$$p_{\mathrm{marker}}=p_{\mathrm{drone}}^{ENU}+p_{ENU}$$

| 기호 | 정의 |
|---|---|
| $(u,v,w)$ | 마커의 카메라 광학 좌표 (solvePnP $t$) |
| $D$ | FLU→FRD |
| $R(q)$ | PX4 쿼터니언 $q=[w,x,y,z]$의 회전행렬 (FRD→NED) |
| $S$ | NED→ENU |

쿼터니언 회전행렬:

$$R(q)=\begin{pmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy)\\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx)\\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{pmatrix}$$

**$R(q)$를 빼먹으면**(자세 무시) 기체가 기울 때마다 마커가 흔들려 보인다 —
착륙 제어가 절대 봐선 안 되는 신호.

검증: `python3 -m landing_mpc.frame`

---

## 2.2 aruco_detector_node — 마커 검출

**한 줄**: 픽셀 → 카메라 광학 프레임 마커 자세. 드론도 프레임도 착륙도 모른다.

핀홀 투영 + PnP:

$$s\begin{pmatrix}u\\v\\1\end{pmatrix}
= K\,[\,R\mid t\,]\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix},\qquad
K=\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}$$

마커 평면 물체점(코너 순서 TL,TR,BR,BL), 한 변 $s_m$:

$$\left(\mp\tfrac{s_m}{2},\ \pm\tfrac{s_m}{2},\ 0\right)$$

솔버는 평면 사각형 전용 `SOLVEPNP_IPPE_SQUARE`, 코너는 서브픽셀 정제.

**마커 크기는 실측해야 한다** — $t$(따라서 거리)가 $s_m$에 **선형 비례**하므로
틀리면 전체가 스케일 오차를 먹는다. 텍스처 측정:

$$s_m = 1.95\,\mathrm m\times\frac{400\ \mathrm{px}}{520\ \mathrm{px}} = 1.5000\ \mathrm m$$

즉 모델의 1.95 m 평면은 quiet zone 포함이고 **검은 코드는 1.5 m**가 맞다.

### 비전 가시영역은 구가 아니라 **원뿔**이다 (유도)

하방 고정 카메라는 바로 아래만 본다. 고도 $h$(덱 기준)에서 마커 전체가 화면에
들어오는 **수평 오프셋 한계**는

$$\boxed{\ r(h)=h\tan\frac{\mathrm{vfov}}{2}-\frac{s_m}{2}\ }$$

| 고도 $h$ | FOV 반경 | 마커 여유 $r(h)$ |
|---|---|---|
| 8 m | 5.03 m | **4.28 m** |
| 4 m | 2.52 m | 1.77 m |
| 2 m | 1.26 m | 0.51 m |
| **1.19 m** | 0.75 m | **0** |

**따라서 "30 m 밖에서 비전 인수인계" 같은 건 불가능하다.** 접근고도 8 m에서도
수평 4.3 m 안에 들어와야 마커가 보인다(SITL 실측 인수인계 4.5 m, 3.4 m, 1.7 m).
미션 매니저는 이 식으로 인수인계 반경을 **고도에서 유도**한다.

$r(h)=0$ 인 지점이 곧 아래의 블라인드 임계라, **두 제약은 같은 기하의 양 끝**이다.

### FOV 블라인드 구간 (유도)

수직 화각과 지상 커버리지:

$$\mathrm{vfov}=2\arctan\!\left(\tan\frac{\mathrm{hfov}}{2}\cdot\frac{H_{px}}{W_{px}}\right),
\qquad w(h)=2h\tan\frac{\mathrm{vfov}}{2}$$

마커가 전부 들어오려면 $w(h)\ge s_m$, 즉

$$\boxed{\ h_{\min}=\frac{s_m}{2\tan(\mathrm{vfov}/2)}\ }$$

수치($\mathrm{hfov}=1.396$, $640\times480$, $s_m=1.5$):

$$\tan\frac{\mathrm{vfov}}{2}=0.8391\times0.75=0.6293
\ \Rightarrow\ h_{\min}=\frac{1.5}{2\times0.6293}=1.19\ \mathrm m$$

**SITL 실측과 일치**: 1.60 m에서 마지막 검출, 1.13 m부터 0건.

---

## 2.3 marker_tf_node — 좌표 변환 노드

**한 줄**: 카메라 프레임 마커 자세 + **같은 시각의** 기체 상태 → 로컬 ENU 위치.
이미지도 ArUco도 모르는 순수 기하 변환.

### 시간 정합이 핵심

이미지는 도착 시점에 이미 낡았다. **현재** 기체 상태로 변환하면 마커가 진행방향으로
밀린다(대각 접근에서 1.17 m 편향 실측). 기체 상태를 버퍼링해 **촬영 시각으로 보간**:

$$\hat s(t_{\mathrm{img}})=(1-\alpha)\,s_i+\alpha\,s_{i+1},\qquad
\alpha=\frac{t_{\mathrm{img}}-t_i}{t_{i+1}-t_i}$$

적용 후 편향 1.17 → 0.66 m.

> **시도했다 되돌린 것**: PX4 자체 타임스탬프로 키를 바꾸고 오프셋을 running-min으로
> 추정 → 오히려 악화(0.66 → 1.08 m, 지연 편향 재발). 오프셋 추정 자체가 편향됨.
> 수신 시각 기준이 우세.

### 실측 정확도 (정지 호버링, 진실값 기지)

| 고도 | 표본 | 평균오차 | 표준편차 |
|---|---|---|---|
| 8 m | 67 | 0.068 m | 0.086 |
| 6 m | 83 | **0.024 m** | 0.004 |
| 4 m | 88 | 0.059 m | 0.033 |
| ≤3 m | 0 | 검출 없음(블라인드) | — |

편향 ≈ 0 → **검출기와 변환 체인은 정확하다.** 기동 중 오차 0.66 m는 순수 산포.

---

## 2.4 marker_kf_node — 추정 + coast

**한 줄**: 노이즈·결측 있는 검출을 연속 상태로 만들고, **끊기면 관성 주행**한다.

수평 등속 모델(z는 기지의 덱 높이라 추정 안 함), 상태
$x=(p_x,\ p_y,\ v_x,\ v_y)^\top$:

$$F=\begin{pmatrix}1&0&\Delta t&0\\0&1&0&\Delta t\\0&0&1&0\\0&0&0&1\end{pmatrix},
\qquad
H=\begin{pmatrix}1&0&0&0\\0&1&0&0\end{pmatrix}$$

백색 가속도 잡음 $\sigma_a$가 만드는 프로세스 잡음:

$$Q=\sigma_a^2\begin{pmatrix}
\frac{\Delta t^4}{4}&0&\frac{\Delta t^3}{2}&0\\
0&\frac{\Delta t^4}{4}&0&\frac{\Delta t^3}{2}\\
\frac{\Delta t^3}{2}&0&\Delta t^2&0\\
0&\frac{\Delta t^3}{2}&0&\Delta t^2
\end{pmatrix},
\qquad R=\sigma_m^2 I_2$$

**예측(coast)**

$$x^- = F x,\qquad P^- = F P F^\top + Q$$

**보정(측정 도착 시)**

$$y = z - Hx^-,\qquad S = HP^-H^\top + R$$
$$K = P^-H^\top S^{-1},\qquad
x = x^- + Ky,\qquad P=(I-KH)P^-$$

**이상치 게이팅** (마할라노비스 거리):

$$d^2 = y^\top S^{-1} y > \gamma^2 \ \Rightarrow\ \text{측정 기각}$$

| 기호 | 정의 | 기본값 |
|---|---|---|
| $\sigma_m$ | 측정 잡음 표준편차 | **0.06 m** (§2.3 실측값) |
| $\sigma_a$ | 표적 기동성(백색 가속도) | 1.5 m/s² |
| $\gamma$ | 게이팅 문턱 | 5 |
| $z$ | 측정 위치 (`/marker/measured`) | — |
| $y$ | 혁신(innovation) | — |
| $K$ | 칼만 이득 | — |

$\sigma_m$은 추측이 아니라 **§2.3에서 실제로 잰 값**이다.

### coast 동작 (SITL 실측)

측정이 끊겨도 predict만 계속 → 연속 추정 유지. `max_coast` 초과 시 `/marker/valid`가
False로 바뀌어 "지금부터 추측 중"임을 상위에 알린다.

```
h=1.60 m  fixes=1  KF_err=0.183  valid=True   ← 마지막 검출
h=1.13 m  fixes=0  KF_err=0.183  valid=True   ← 블라인드, coast 시작
h=0.19 m  fixes=0  KF_err=0.183  valid=True
h=0.12 m  fixes=0  KF_err=0.183  valid=False  ← 3 s 초과, 자동 무효
```

---

## 2.5 mission_manager_node — 단계 시퀀싱 + 목표 소스 선택

**한 줄**: 어느 목표를 믿을지 결정하고 단계를 밟는다. **유일한 Offboard 권한자.**

이 노드가 존재하는 이유는 §2.2의 두 제약이다: 마커는 **~30 m 밖에서 해상되지 않고**,
**원뿔 $r(h)$ 밖이면 화면에 없다.** 그래서 접근은 표적이 스스로 보고한 좌표
(`/marker/cue`, 실기에선 트럭 텔레메트리)로 날고, 비전은 가까워진 뒤에만 쓴다.

| 단계 | 목표 소스 | 하는 일 |
|---|---|---|
| `IDLE` | — | 큐 대기 → arm + Offboard |
| `TAKEOFF` | — | 이륙 |
| `APPROACH` | **큐** | 접근고도에서 큐로 순항(속도 피드포워드), 하강 안 함 |
| `DESCEND` | **비전(KF)** | MPC 상대좌표 랑데부 + corridor 하강 |
| `TOUCHDOWN` | — | 접지 판정 → disarm |
| `ABORT` | 큐 | 비전 소실 → 상승·복귀 후 재시도 |

**인수인계 조건** (히스테리시스): $d<0.7\,r(h)$ **이고** `/marker/valid`.
복귀는 더 넓은 반경에서만 → 경계에서 채터링 방지.

**커밋 고도**: $h\le h_{commit}$ 아래에선 원뿔이 닫혀 비전 소실이 **정상**이므로
abort하지 않고 KF coast로 밀어붙인다.

### 실측 (3 m/s 원운동 트레일러, 실제 ArUco 인식)

```
IDLE → TAKEOFF → APPROACH
APPROACH → DESCEND (vision acquired at 1.0 m, cone r=2.2 m)
DESCEND → APPROACH (drifted outside the vision cone)   ← 헌팅 2회
APPROACH → DESCEND (1.6 m, cone r=1.7 m)
DESCEND → APPROACH
APPROACH → DESCEND (1.7 m, cone r=1.9 m)
DESCEND → TOUCHDOWN (xy 0.51 m, |v_rel| 1.71 m/s)
TOUCHDOWN → DONE (disarmed)
```

**전 구간 성공하지만 품질은 아직 낮다**: 완벽한 큐(gz 치트)로는 xy 0.24 m /
0.10 m/s였는데, 실제 인식으로는 **xy 0.51 m / 1.71 m/s**. 원인은 좁은 원뿔 안에서
검출이 간헐적이라 종말 구간 목표 추정이 나쁜 것. 남은 과제다.

---

## 2.6 미해결 문제 — 카메라 지향과 기체 자세의 결합

전 파이프라인이 동작하는데도 **이동 표적 착륙이 재현되지 않는다.** 원인을 추측이
아니라 계측으로 특정했으므로, 그 과정과 **기각된 가설들**을 남긴다.

### 계측 방법

`scratchpad/vis_diag.py` 가 카메라 광축을 덱 평면에 투영해 **틸트를 반영한**
지상 발자국 중심을 구하고, 거기서 실제 트레일러까지 거리를 가시반경 $r(h)$ 와
비교한다. 즉 **"기하적으로 보여야 하는가"** 와 **"실제로 검출됐는가"** 를 대조한다.

```
기하 OK + 검출 없음  →  검출기/렌더링 문제
기하 NOT OK          →  비행이 카메라를 엉뚱한 곳에 겨눔
```

### 결과 (155초)

| 항목 | 값 |
|---|---|
| 기하 OK 일 때 검출 | **55/59초 (93%)** → 검출기는 정상 |
| 카메라가 표적을 **안** 겨눈 시간 | **62%** |
| 틸트 >10° | 41초, 최대 109° |

**병목은 검출이 아니라 지향이다.**

### 왜 기울어지는가 — 가속도와 틸트의 결합

멀티로터는 추력 벡터를 기울여 수평 가속을 만든다. 따라서

$$\tan\theta = \frac{a_h}{g}
\qquad\Longrightarrow\qquad
\Delta_{\text{시선}}(h) = h\tan\theta = \frac{h\,a_h}{g}$$

| $a_h$ | $\theta$ | $h{=}10$ m 시선 이동 | 가시반경 |
|---|---|---|---|
| 4.0 | 22.2° | 4.08 m | 5.54 m |
| 2.0 | 11.5° | 2.04 m | 5.54 m |

몸체 고정 카메라에서는 **기동하는 순간 표적을 잃는 구조**다.

### 기각된 가설 (전부 실험으로 반증)

| # | 가설 | 검증 | 결과 |
|---|---|---|---|
| 1 | 3 m/s 표적을 못 따라감 | 1.5 m/s 재실험 | **기각** — 동일 실패 |
| 2 | 반경 기반 이탈 판정 | 판정 제거 | 헌팅 15→**0**, 그러나 미착륙 |
| 3 | 핸드오프 시 속도 불일치 | 속도정합 조건 추가 | **기각** — dv 0.27에도 실패 |
| 4 | 우리 MPC의 `a_max` 과다 | 4.0→2.0 | **기각** — 겨냥 38→36%, 틸트>10° 41초 동일 |
| 5 | APPROACH를 MPC로 라우팅 | 구현 후 측정 | **기각** — 틸트 109→17°로 좋아졌으나 겨냥 36→**2%**, DESCEND 도달 0초 |

4번이 무효였던 이유가 중요하다: **APPROACH·ABORT 는 우리 MPC를 거치지 않는다.**
raw position setpoint 를 던지고 가속을 만드는 것은 **PX4 자신의 위치 제어기**다.

5번의 교훈: 부드러워졌지만 **표적에 도달하지 못하면 개선이 아니다.** 되돌렸다.

### 남은 선택지

1. **짐벌** — 지향을 기체 자세에서 분리. 위 트레이드오프 자체를 없애는 유일한 방법
   → **채택. 2.7절에서 구현·검증했다.**
2. **PX4 게인 하향** (`MPC_ACC_HOR_MAX`) — 단, 맵 런처가 특정 키만 읽으므로 주입 경로부터 필요
3. 광각 렌즈 / 중첩 소형 마커 — 원뿔 자체를 넓힘

---

## 2.7 gimbal_control_node — 지향을 자세에서 분리

2.6의 결론(“병목은 검출이 아니라 지향”)에 대한 답. 기체는 기울여야 가속하고,
기울이면 몸체 고정 카메라는 표적을 잃는다. **짐벌은 이 트레이드오프를 완화하는
게 아니라 없앤다.**

기체: `x500_gimbal_rgbd_lidar` (= 기존 `x500_city_rgbd_lidar` + 매단 3축 짐벌).
기존 센서·토픽을 **전부 포함하는 상위집합**이라 `GIMBAL=1` 하나로 기체만 바꿔
몸체고정 vs 짐벌을 **통제된 비교**로 돌릴 수 있다.

```
GIMBAL=1 ./gazebo/run_px4_map.sh mpc-landing-moving
ros2 launch landing_mpc gimbal_perception.launch.py
```

### 관절 체인 (유도)

`gimbal_down` 이 선언한 축은 yaw `(0,0,-1)`, roll `(-1,0,0)`, pitch `(0,1,0)`,
장착 yaw 는 $\pi$, 카메라 센서 자체도 yaw $\pi$ 를 갖는다. 이를 합성하면
카메라 프레임 → 동체 FLU 회전은

$$R = R_z(\pi - y)\,R_x(-r)\,R_y(p)\,R_z(\pi)$$

이고, 광축(카메라 +X)만 꺼내면

$$d_{\text{FLU}} = (\cos p\cos y,\; -\cos p\sin y,\; \sin p) \tag{5}$$

역으로 **원하는 방향을 겨누는 관절각**은

$$p = \arcsin(d_z), \qquad y = \operatorname{atan2}(-d_y,\, d_x) \tag{6}$$

roll 은 0 으로 고정한다 — 겨냥은 2자유도면 충분하고, roll 은 영상을 돌릴 뿐이다.

**짐벌 락은 버그가 아니라 기하다.** 천저($d=(0,0,-1)$)에서 (6)의 $y$ 는 정의되지
않는다. 그래서 $\lVert d_{xy}\rVert$ 가 작으면 직전 yaw 를 **유지**한다. 안 그러면
호버링 중 검출 노이즈만으로 짐벌이 계속 돌아간다.

### 제어 법칙

$$\text{표적점} \;\to\; d_{\text{ENU}} \;\xrightarrow{\text{eq. (2)-(4) 역}}\; d_{\text{FLU}} \;\xrightarrow{\text{eq. (6)}}\; (y, r, p)$$

핵심은 가운데 단계다. 기체 자세가 **전치(transpose)로** 들어가므로, 기체가 어떻게
기울든 그 기울기가 관절 명령에서 그대로 **차감**된다. 이것이 틸트 분리의 전부다.

표적 선택은 미션 매니저와 같은 규칙: vision 이 valid·fresh 면 vision, 아니면 cue,
둘 다 없으면 천저. **접근 구간에서 cue 를 겨누는 것**이 중요하다 — 마커가 아직
분해되지 않는 거리에서 미리 겨눠 놔야 vision 이 획득을 시작할 수 있다.

### PX4 게인이 아니라 PX4 프레임 — 하마터면 놓칠 뻔한 함정

카메라에는 IMU 가 달려 있고, 그게 **렌즈의 월드 자세**를 바로 준다. 당연히 그걸
쓰면 될 것 같다. **틀렸다.**

| 값 | 출처 | 기준 |
|---|---|---|
| gz 실제 기수방위 | Gazebo | 북에서 **90.00°** |
| PX4 보고 heading | EKF | 북에서 **117.33°** |
| **차이** | | **27.33°** |

원인: 월드의 `<magnetic_field>6.0e-6 2.3e-5 -4.2e-5</magnetic_field>` 자체가
편각 $\arctan(6.0/23.0)=14.6°$ 를 품고 있는데, PX4 EKF 가 그 위도·경도(산호세)의
WMM 편각을 **한 번 더** 적용한다. 둘이 더해져 27.33°.

즉 **Gazebo 월드 ENU 와 PX4 로컬 ENU 는 27.33° 어긋나 있다.** 카메라 IMU 는 전자,
드론 위치는 후자 — 섞으면 모든 마커 측정이 27.33° 회전한다. 20 m 거리에서 9 m 오차.
그리고 **조용히** 틀린다.

**해법: 프레임을 넘지 않는다.** 관절각은 *동체 상대* 량이므로 월드가 없다.

$$R_{\text{cam}\to\text{ENU}_{\text{PX4}}} = \underbrace{R(q_{\text{PX4}})}_{\text{기체 자세}} \circ \underbrace{R(y,r,p)}_{\text{관절, 동체상대}}$$

이러면 Gazebo 월드가 한 번도 등장하지 않아 편각 오프셋이 **원리적으로** 들어올 수
없다. 덤으로 이게 **실기체 그대로**다 — 짐벌 엔코더 + FC 자세.
그래서 `gimbal_down` 에 `JointStatePublisher` 를 넣고, `gimbal_control_node` 가
gz 엔코더를 읽어 `/gimbal/joint_state` 로 중계한다. 명령각이 아니라 **엔코더각**을
쓰는 이유는 서보의 적분항이 꺼져 있어(`i_max=0`) 중력 처짐이 남기 때문이다.

> 같은 함정이 `trailer_cue_node` 에도 있다. gz 월드 좌표에서 스폰을 뺀 값을
> PX4 로컬 좌표인 양 publish 한다. 짐벌과 무관한 **선재 이슈**이므로 여기서는
> 건드리지 않았지만, 이동표적 미션을 다시 볼 때 반드시 확인할 것.

### 실측 (SITL, 지상 정지 상태)

| 항목 | 값 |
|---|---|
| 관절 모델 (5) vs Gazebo 실측 광축, 5개 자세 (roll≠0 포함) | 처음 최대 0.09° → 수정 후 **0.0000°** |
| 그 0.09° 의 정체 | SDF 의 `3.14` vs $\pi$. $\pi-3.14=0.0016\,\text{rad}=0.0912°$ 와 **정확히** 일치했다 → SDF 를 정밀값으로 교체해 제거 |
| 서보 추종 (명령 vs 엔코더), 정상상태 | **0.00–0.01°** |
| 슬루 중 과도 최대 | 7.2° (90 °/s 제한) |
| 45° 기체 피치에서 보정 후 마커 위치 오차 | **0** (frame.py 셀프테스트) |

관절 모델을 Gazebo 실측과 맞춘 검증은 **월드 프레임 안에서만** 수행했다
(카메라 IMU vs base_link IMU + 관절각). PX4 를 끌어들이지 않았으므로 위의 27.33°
오프셋이 검증 자체를 오염시킬 수 없다.

### 비행 검증 (`scratchpad/gimbal_hover_test.py`)

호버링 상태에서 10 m 떨어진 **정지** 트레일러를 겨눈다. 진실값은 Gazebo pose 피드,
추정값은 영상+엔코더+PX4 자세 — **완전히 독립인 두 경로**라서 일치하면 의미가 있다.
큐는 gz 진실값을 PX4 프레임으로 회전해 만들되, 오프셋은 EKF 추정이라 드리프트하므로
**매번 실측**한다(실제로 +27.3°, +24.9°, +16.1°, +14.4°, +0.9° 로 크게 변했다).

| 기하 | 고도 | 검출률 | xy 오차 | 반경 잔차 |
|---|---|---|---|---|
| 사각(off-nadir 34°) | 15 m | **98.0 / 95.3%** | 0.44 m | +0.43 m (sd 0.035) |
| 사각(off-nadir 20°) | 28 m | **90.7%** | 0.65 m | +0.62 m (sd 0.048) |
| **천저 직상방** | 15 m | **100.0%** | **0.12 m** | +0.11 m (sd 0.011) |

**검출률이 답이다.** 2.6 의 몸체고정 카메라는 같은 종류의 상황에서 0.8~38% 였고,
3 m/s 추격 중에는 **0.3%** 였다. 짐벌은 **90~100%**.

오차는 **반경 방향(바깥쪽)** 이고 방위 성분은 작다(0.5°). 반경 잔차는 프레임
오프셋 추정과 무관하므로 이건 **진짜 인식 편향**이다. 그리고 직상방에서 0.12 m 로
떨어지는 것이 결정적 — 거리가 아니라 **비스듬히 보는 것**(단축 왜곡)이 원인이다.
직상방 0.12 m 는 몸체고정 정지 실측(8 m 에서 0.068 m)을 거리 비례로 늘린 값
(0.068 × 15/8 = 0.128 m)과 거의 정확히 같다 → **짐벌 인식 체인은 정상이다.**

### 부산물: `deck_z = 1.811` 의 출처를 확인했다

`use_deck_z:=false` 로 날려 raw z 를 재니 **1.686 m** 가 나왔다. 정답은
2.051 m(맵 yaml `marker_surface_height_m`, 모델에서 `platform_link` 가 모델 z 2.0
에 있고 마커가 그 위 0.051) 에서 x500_base 의 프레임 오프셋 0.24 m 를 뺀
**1.811 m**. 차이 0.125 m 는 **카메라 레버암**이다 — 렌즈는 base_link 보다 0.13 m
아래인데 `marker_tf_node` 는 카메라 오프셋을 렌즈가 아니라 `p_d` 에 더한다.
평소에는 z 를 `deck_z` 로 고정하므로 미션에 영향은 없지만, **알려진 0.13 m
계통오차**로 남긴다.

### tf2 트리

`gimbal_control_node` 가 **엔코더 각**으로 관절 체인을 브로드캐스트한다. 명령각이
아니라 엔코더각이므로 RViz 에 그려지는 건 “겨누라고 시킨 곳”이 아니라 **실제로
렌즈가 있는 곳**이다.

```
map ─► base_link ─► cgo3_mount_link ─► cgo3_vertical_arm_link
        └─► cgo3_horizontal_arm_link ─► gimbal_camera_link
              └─► gimbal_camera_optical_frame        (30 Hz)
```

리프 이름은 SDF 가 이미 `<optical_frame_id>` 로 선언한 것이고, 마지막 홉은
`sensor_bridge.launch.py` 의 고정 카메라와 같은 `(-π/2, 0, -π/2)` 규약이라
tf2 소비자 입장에서 두 카메라가 동일하게 보인다. `map→base_link` 도 여기서
쏜다 (`publish_base_tf`, 다른 노드가 소유하면 끄면 된다).

기하는 `frame.gimbal_tf_chain` 에 있고 **셀프테스트가 `gimbal_camera_rotation`
과 일치함을 강제**한다 — 그림과 수식이 갈라질 수 없다. SITL 실측 검증:
tf2 조회 vs `frame.py` 회전 **최대 0.029°** (n=9555, 잔차는 슬루 중 타임스탬프
차이), 렌즈 위치 base_link 기준 정확히 `(0, 0, -0.13) m`.

### 주행 중 검증 — 3 m/s 트레일러 추격 (180초)

| 항목 | 몸체고정 (2.6) | **짐벌** |
|---|---|---|
| 추격 중 검출률 | **0.3%** | **44.5%** (다른 런 35.6%) |
| 틸트 최대 | 109° | **43.6°** |
| DESCEND 이탈 | 15~16회 | **6회** |
| PX4/gz 요 오프셋 (런 중) | — | +1.7° (무해) |

**지향 문제는 해결됐다.** 검출률이 100배 이상 올랐고 틸트도 절반 이하다.
런 중 PX4/gz 오프셋이 +1.7° 로 수렴해 있어(지상에서는 27°) `trailer_cue_node`
의 프레임 버그가 이 결과를 오염시키지 않았다 — EKF 요각이 이륙 후 GPS 로
보정되기 때문이다.

**그런데 아직 착륙은 못 한다.** ABORT 사유는 전부 동일하다: `vision stale 3.1 s`
— KF 의 3초 coast 가 만료된다. 즉 검출이 **3초 넘게 연속으로** 끊긴다.

### 왜 끊기는가 — 짐벌 피치로 층화해 보면 명확하다

```
검출률 vs 짐벌 피치 (천저 = -90°)
  -90..-75° :  52.4%  (n=296) ####################
  -75..-60° :  80.6%  (n=155) ################################
  -60..-45° :  47.0%  (n=100) ##################
  -45..-30° :  29.6%  (n=135) ###########
  -30..-15° :  36.6%  (n=123) ##############
  -15..  0° :   1.2%  (n= 81)
    0.. 15° :   2.5%  (n= 40) #
```

**부각 15° 아래에서 검출이 붕괴한다.** 원인은 지향이 아니라 **표적 자신의
기하**다 — 마커는 바닥에 **누워 있으므로** 입사각이 얕아지면 투영 면적이
$\cos$ 로 줄고, 수평에 가까워지면 아예 보이지 않는다. 짐벌은 “어디를 보느냐”를
고쳤지만 “누운 평면을 옆에서 볼 수 없다”는 건 못 고친다.

로그의 `pitch=+4.3°`(수평 응시), `pitch=-25.9°` 구간이 정확히 8.9초·8.3초
검출 공백과 겹친다. 이건 드론이 트레일러에서 **멀어졌을 때** 생긴다.

**따라서 병목이 이동했다: “카메라 지향” → “표적 위에 머무르기”.**
남은 후보는 (1) DESCEND 중 속도 정합을 유지해 뒤처지지 않게 하기,
(2) `max_coast` 를 3초보다 늘려 얕은 각 구간을 타고 넘기,
(3) 마커를 세우거나 트레일러 측면에도 붙여 얕은 각에서도 보이게 하기.
2.5 의 원뿔 기반 handoff 로직도 재검토 대상이다 — 짐벌이 있으면 획득 영역이
더 이상 천저 원뿔이 아니다(이번 런은 `tan_vfov_half` 만 짐벌 카메라 값
0.7722 로 바꿔 돌렸다).

---

# 부록: 검증 명령

```bash
python3 -m landing_mpc.frame        # ENU/NED + 카메라 체인
python3 -m landing_mpc.reference    # 궤적 보간 연속성
```
