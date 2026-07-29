# Path Planning Pipeline: 수치적 기법 및 채택 알고리즘 분석

이 문서는 드론 자율 비행(Path Planning & Control) 파이프라인의 각 단계에서 사용된 **핵심 수학적/수치적 기법**들과 **해당 방식을 채택한 이유(Rationale)**를 명세합니다.

---

## 1. 전역 경로 탐색 (Global Path Planning)
### 🔹 알고리즘: 3D Grid A* (A-Star) Search
* **사용 기법**: 이산 그래프 탐색 (Discrete Graph Search), `f(n) = g(n) + h(n)` 휴리스틱 탐색
* **수치 미적분 사용 여부**: **X** (오직 사칙연산과 논리 연산만 사용)
* **채택 이유**:
  * **전역 최적성 (Global Optimality)**: 연속 공간에서 미분/최적화를 수행하는 방식(예: 인공 포텐셜 필드)은 건물 모서리나 ㄷ자 형태의 막힌 공간에서 '지역 최소점(Local Minima)'에 빠져 탈출하지 못하는 치명적인 단점이 있습니다.
  * A*는 공간을 이산화(격자화)하여 탐색하므로 맵 전체를 조망하여 **반드시 목적지까지 도달하는 충돌 없는 전역 경로**를 수학적으로 보장합니다.
  * 본 파이프라인에서는 단순 거리뿐만 아니라 벽면 이격 거리(Clearance), 비행 고도 유지(Altitude), 상승 벌점(Climb penalty) 등을 비용 함수(Cost function)에 합산하여 최적의 비행 경로를 도출하도록 설계되었습니다.

---

## 2. 안전 비행 공간 확보 (Safe Flight Corridor, SFC)
### 🔹 알고리즘: AABB (Axis-Aligned Bounding Box) Iterative Inflation
* **사용 기법**: 기하학적 공간 팽창 (Geometric Expansion) 및 충돌 체크(Collision Query)
* **채택 이유**:
  * 점으로 이루어진 A* 경로 주변에 드론이 비행할 수 있는 "여유 공간(터널)"을 수학적인 볼록 다면체(Convex Polyhedron)로 정의하기 위함입니다.
  * **B-Spline의 볼록 다각형 성질(Convex Hull Property)**과 결합하기 위한 필수 작업입니다. 스플라인 곡선의 제어점(Control Points)들을 이 SFC 박스 안에 가두기만 하면, **생성된 연속 곡선 전체가 절대 장애물과 충돌하지 않음이 수학적으로 완벽히 증명(Guaranteed-free)**됩니다.

---

## 3. 궤적 최적화 (Trajectory Optimization)
### 🔹 알고리즘: Uniform B-Spline 기반 L-BFGS 최적화 (Ego-Planner 방식)
* **사용 기법**: 준뉴턴법 (Quasi-Newton Method, L-BFGS 알고리즘) 및 해석적 미분 (Analytical Gradient)
* **채택 이유**:
  * A*가 만든 각진 경로를 부드러운 곡선으로 깎아내기 위해서는 목적 함수(스무딩, 장애물 척력, 속도/가속도 제한)를 최소화해야 합니다.
  * **경사하강법(Gradient Descent)의 한계**: 수렴 속도가 너무 느리고, 제어점들이 지그재그로 진동할 위험이 있습니다.
  * **순수 뉴턴법(Newton's Method)의 한계**: 2차 미분 행렬(Hessian)을 직접 계산하고 역행렬을 구하는 과정($O(N^3)$)이 너무 무거워 드론의 탑재 컴퓨터에서 실시간(ms) 처리가 불가능합니다.
  * **L-BFGS (Limited-memory BFGS)**: 과거의 1차 미분(기울기) 데이터 몇 개만 메모리에 저장하여 2차 미분 행렬을 '근사(Approximate)'합니다. 연산량이 경사하강법 수준으로 가벼우면서도 수렴 속도는 뉴턴법에 필적하므로, 0.1초 단위의 **초고속 실시간 궤적 재생성(Real-time Replanning)**에 가장 완벽한 타협점을 제공합니다.

---

## 4. 궤적 추종 제어기 (Trajectory Tracking Controller)
### 🔹 알고리즘: Unicycle MPC (Model Predictive Control)
* **사용 기법**: 
  1. **SLSQP (Sequential Least Squares Programming)** - 비선형 제약 조건 최적화
  2. **하이브리드 수치 적분 (Hybrid Numerical Integration)** - 4차 런지-쿠타(RK4) + 오일러(Euler)
* **채택 이유**:
  * **제약 조건 처리 (Hard Constraints)**: PID 등 고전 제어기와 달리, MPC는 드론의 물리적 한계치(`최대 가속도 3.0 m/s²`, `최대 각속도` 등)를 경계 조건(Bounds)으로 명시하여 물리 엔진을 뚫고 나가는 비현실적인 제어 명령을 원천 차단합니다. 이를 위해 제약 조건을 잘 다루는 **SLSQP 솔버**를 채택했습니다.
  * **RK4 수치 적분의 도입**: 곡률이 심한 궤적을 예측할 때, 단순 오일러(1차) 적분은 오차가 누적되어 코너링 시 궤적 예측이 밖으로 튕겨 나갑니다. 따라서 드론의 위치($x, y$)와 각도($\psi$) 등 물리 상태를 예측할 때는 정밀도가 매우 높은 **4차 런지-쿠타(RK4) 수치 적분**을 채택하여 예측 정확도를 끌어올렸습니다.
  * **하이브리드(Hybrid) 적분 구조 설계**: 
    모든 모델을 RK4로 돌리면 SLSQP 솔버가 오차(Error)의 기울기(Gradient)를 계산할 때 수식이 너무 복잡해져서 발산하거나 Local Minima에 빠지는 치명적인 버그(초기 160초간 드론이 정지하는 현상)가 발견되었습니다. 
    이를 해결하기 위해, **'물리 상태 예측'은 정밀한 RK4를 유지하고, 최적화 채점용 '경로 오차(CTE) 계산'은 단순한 Euler 투영법을 혼용하는 하이브리드 방식**을 고안하여 궤적 정밀도와 연산 안정성(즉각적인 제어 응답성)을 모두 획득했습니다.
