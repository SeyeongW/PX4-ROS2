# path_plan 진행 기록 (worklog)

> 이 파일은 **앞으로 하는 모든 작업을 누적 기록**하는 로그입니다. 새 작업/수정은
> 맨 위 "최근 작업"에 날짜와 함께 추가합니다. 알고리즘 상세는 `README.md`,
> 노드별 파라미터는 `docs/nodes.md` 참고.

## 현재 상태 (2026-07-14 기준)

- **완성**: A* → SFC → B-spline → MPC 파이프라인. 스폰(587,580)→트레일러(-587,-512)
  경로 생성 + 추종을 오프라인 시각화로 검증 완료 (충돌 0, 추종오차 순항 ~0.08 m).
- **미완(다음)**: 실제 Gazebo+PX4 sim 연결 — `/path_plan/odometry`, `/path_plan/depth`
  브리지 입력 + `/path_plan/cmd_vel` → PX4 OFFBOARD `TrajectorySetpoint` 브리지.
- **브랜치**: `wang`에 푸시 (jo 기반, jo는 팀원 mandu38의 52ddc82 위에 쌓음).

## 구성 요소

| 파일 | 역할 |
|------|------|
| `world_model.py` | 건물 AABB 장애물장 + free/clearance 쿼리 + city YAML 로더 |
| `astar.py` / `astar_node.py` | 3D 격자 A* (BigZaphod식 제네릭 코어 + 강화비용) + 경유점 `plan_through` |
| `sfc.py` / `sfc_node.py` | 자유 박스 회랑 (제어점별 보장된 free AABB) |
| `uniform_bspline.py` + `bspline_optimizer.py` / `bspline_node.py` | ego-planner식 코리도어 B-spline (L-BFGS + rebound) |
| `mpc_ros.py` / `mpc.py` / `mpc_node.py` | mpc_ros식 유니사이클 추종 MPC (+ 이중적분기 대안) |
| `tools/visualize_pipeline.py` | 오프라인 파이프라인 실행 + figures/ 생성 |

실행법: `README.md`의 Running 섹션. 경로/웨이포인트: `config/city_uav.yaml`.

---

## 최근 작업 (최신 위)

### 2026-07-14 — pursuit_sim에 visualize_pipeline 시각화 로직 이식 + 회랑박스 누락 수정
- **팔레트 통일**: `pursuit_sim.py`에 `visualize_pipeline.py`와 동일한 색 상수
  (`C_OBST/C_SFC/C_ASTAR/C_BSPL/C_MPC` + pursuit 전용 `C_TRAIL`) 도입. 드론 추종
  궤적을 파이프라인과 동일하게 마젠타(MPC), B-spline 초록, SFC 파랑으로 정렬
  (기존 빨강 B-spline/연두 드론 → 파이프라인 규약에 맞춤). animate 색도 통일.
- **SFC 회랑박스 채우기**: 맵 피규어(`_fig_topdown`)의 회랑을 `facecolor="none"`
  외곽선 → 파이프라인처럼 **옅은 파란 채우기**(alpha 0.12)로 변경.
- **회랑박스 누락 수정(사용자 요청)**: MPC 추종 피규어(`_fig_mpc`)가 드론 이동경로에
  회랑박스를 안 그리던 문제 해결 — 레퍼런스/추종 궤적 아래에 SFC 채움 박스 오버레이
  + 범례 추가.
- **3D 듀얼뷰 신규**(`_fig_3d` → `figures/8_pursuit_3d.png`): 파이프라인 `1_global_3d`
  스타일로 Perspective + Side 두 앵글. 건물 `bar3d`(원 지붕높이, `raw_footprints`
  이식), SFC 3D 와이어프레임, B-spline/드론 궤적은 순항고도, 트레일러 loop·경로는
  지면(z=0), 드론 시작/캡처 마커.
- **고도 밴드 파이프라인과 통일**: 시나리오(`city_uav_trailer_loop.yaml`)를 순항
  10–20 m + `overfly_allowed: true`로 변경(기존 20–30 m/lateral-only). 이제 드론이
  밴드 안에서 오르내리며 건물을 넘어감. mpc_ros의 분리형 고도홀드
  (`vz=z_kp*(z_ref−z)`)가 B-spline의 가변 z를 추종 → 코드 로직 변경 불필요, config만.
  3D 피규어(`_draw_pursuit_3d`)는 상수 cruise_z 대신 **실제 로그 z(`drone_z`)**를 그림.
- **비행고도 그래프 추가**: `_fig_profiles`(피규어 6)를 speed/accel 2단 → **speed/accel/
  altitude 3단**으로 확장. 고도 패널에 순항밴드(10–20 m) 음영 + 실제 `drone_z` 곡선.
- **tools/에서 직접 실행 지원**: `python3 tools/pursuit_sim.py`를 `tools/` 안에서 돌리면
  `ModuleNotFoundError: path_plan` 발생 → 스크립트 상단에서 패키지 루트(`parents[1]`)를
  `sys.path`에 삽입해 해결.
- **검증**: `python3 tools/pursuit_sim.py` 완주(캡처 t=175.2s, 17회 리플랜),
  5·6·7·8 피규어 정상 생성. 3D는 건물 위 순항+지면 추격이 한눈에 보임.

### 2026-07-14 — MPC/스플라인 "터짐" 버그 수정 (`UniformBspline.sample` 끝점)
- **증상**: visualize_pipeline에서 MPC 추종오차가 끝에서 375m로 폭발, 마젠타(MPC)가
  궤적 밖으로 발산. pursuit에서도 91m 스파이크.
- **원인**: `uniform_bspline.py` `sample()`이 `t0 + duration`을 평가하는데, **부동소수점
  오차로 도메인 끝 `knots[n]`을 아주 살짝 초과** → scipy `extrapolate=False`가 NaN →
  기존 `nan_to_num`이 **마지막 샘플을 (0,0,0)으로** 바꿈. 이 오염된 끝점이 MPC의
  레퍼런스/종료조건(`tp[-1]`)을 (0,0)으로 만들어, 드론이 실제 목표에서 종료 못하고
  오버슈트·발산 → 오차 폭발. `ts` 값에 따라 간헐적(트레일러 목표는 우연히 정상,
  (-300,-300)은 깨짐).
- **수정**: `sample()`에서 평가시각을 반열린 도메인 안으로 clip
  (`np.clip(..., t0, np.nextafter(knots[n], t0))`), `nan_to_num` 제거(버그 은폐 방지).
  스플라인 최적화 자체는 원래 정상이었음(목표 도달, free_frac 1.0) — 샘플링만 오염.
- **검증**: 끝점 목표 정확 도달, NaN 0, MPC 롤아웃 3135스텝에 정상 종료, 추종오차
  375m → 평균 0.53m/최대 1.62m. ROS 노드도 같은 sample을 쓰므로 함께 고쳐짐.

### 2026-07-14 — 완벽한 이착륙/순항 상태머신 분리 및 3D 시각화 개선
- **이착륙 및 순항 궤적 완전 분리**: B-Spline 하나로 바닥(0m)부터 순항(10m 이상)까지 한 번에 이으려 할 때 발생하던 하강(Dip)/코너-컷 현상을 근본적으로 해결하기 위해, 궤적 생성을 3단계(수직 이륙 → 10~20m 순항 → 수직 착륙) 상태머신 논리로 분리.
- **순항 모드 하한선(SFC) 강제**: 순항 중(`wp_cruise`) B-Spline 최적화기가 부드러움을 위해 아래로 파고드는 현상을 막고자, `WorldModel`의 `ground_clearance_m`을 10.0m로 설정. 이로 인해 10m 이하로 절대 내려가지 못하는 단단한 안전 비행 회랑(SFC) 바닥벽을 형성.
- **3D 맵 듀얼 뷰 렌더링**: 정지된 `1_global_3d.png` 이미지의 가독성(회전 불가) 한계를 해결하기 위해, `matplotlib` 3D Subplot을 분할하여 **측면 상단 뷰(Perspective)**와 건물을 뛰어넘는 고도 변화가 한눈에 보이는 **완전 측면 고도 뷰(Side)** 두 가지 앵글을 동시에 나란히 출력.
- **MPC 추종 오차(Spike) 및 발산 버그 완벽 수정**:
  - 격자(Grid) 기반 A* 알고리즘의 한계로 인해 시작 좌표가 인근 셀로 스냅(Snap)되어 발생하는 초기 추종 오차(약 2~3m) 문제 해결. 최적화기(`BsplineOptimizer`)에 궤적을 넘기기 직전, 양 끝점을 정확한 실수 좌표(`start_10`, `goal_10`)로 덮어씌워 0cm 오차로 출발하도록 강제.
  - `mpc_ros.py`의 다항식 피팅(Polynomial Fitting)에서 경로가 거의 일직선일 때 발생하는 다항식 진동 및 발산(RankWarning)을 막기 위해, 1차식(Linear) 강제 다운그레이드 임계치(`max_dev`)를 0.1mm(`1e-4`)에서 5cm(`5e-2`)로 대폭 완화하여 수치적 안정성 확보.

### 2026-07-14 — MPC 참조 궤적 피팅 최적화 및 경고 해결
- `mpc_ros.py`의 `_poly_fit`에서 참조 웨이포인트가 직선(Collinear)일 경우 발생하는 파이썬 `np.RankWarning` 문제를 해결.
- 무작정 경고를 숨기는 대신, 점들이 완벽한 직선에 가까우면(최대 직교 오차 < 1e-4) 피팅 차수를 3차에서 1차(Linear fit)로 강제 다운그레이드 하도록 수학적 예외 처리를 추가하여 안정성 개선.

### 2026-07-14 — 추적 시뮬레이션 시각화 및 로깅 고도화
- `pursuit_sim.py` 로그에 드론의 실제 제어 입력인 가속도(`drone_accel_mps2`)와 각속도(`drone_yawrate_rads`)를 추가 기록.
- 요약 피규어(Summary Figure)를 지도(Map) 뷰와 3단 상태 그래프(Graphs) 2개의 독립된 이미지 파일로 분리하여 가독성 대폭 향상.
- 맵 이미지에 안전 비행 회랑(SFC, 파란색 채우기), 최적화된 B-Spline(빨간 선), MPC 예측 궤적(노란 선)을 오버레이로 렌더링.
- `zorder` 레이어링을 엄격히 적용하여 드론의 실제 비행 궤적(초록 선) 아래에 스플라인과 회랑이 깔리도록 하여, 알고리즘 단계별 궤적 변천 과정을 완벽하게 시각화.

### 2026-07-14 — MPC 예측 모델 하이브리드 수치 적분 적용 (RK4 + Euler)
- 위치($x, y, \psi, v$) 등 물리적 상태의 궤적 예측은 정밀도가 높은 **4차 런지-쿠타(RK4)**를 적용하여 비선형 곡선을 정확히 추적.
- 하지만 경로 오차(`cte, epsi`)까지 RK4로 예측할 경우 최적화기(SLSQP)의 Gradient가 폭발하여 초기 160초간 멈추는 현상(Local Minimum)이 발생.
- 이를 해결하기 위해 물리 엔진은 RK4를 유지하고, 최적화 채점용 에러(Error) 평가는 Euler 투영 방식을 혼용하는 **하이브리드(Hybrid) 방식**으로 전환. RK4의 궤적 정밀도와 오일러의 연산 안정성(즉각적인 출발)을 모두 달성.

### 2026-07-14 — MPC 시각화 2건 수정
- **롤아웃 완주**: `visualize_pipeline.mpc_rollout`의 `max_steps`를 궤적 길이 기반
  `int(1.3*duration/dt)`로 자동 확장 → 이전엔 2500 스텝 상한에 걸려 경로 중간(62%)에서
  멈췄음. 이제 4014 스텝으로 전 구간 추종.
- **추종오차 지표 수정**: 오차를 레퍼런스 600샘플 최근접점으로 재던 것이 샘플간격
  아티팩트(~1.35 m 톱니)를 만들었음. 조밀 레퍼런스(6000점)+`cKDTree`로 교체 →
  **진짜 오차 노출: 순항 평균 0.68 m → 0.08 m**. max 3.23 m는 시작 정지→순항 과도상태.
- MPC 가중치도 튜닝(config·viz): `w_cte 2→6`, `w_omega 0.5→1`, `w_domega 2→6`,
  `w_da 0.1→0.5` (실제 위빙 감소용; 오차 주범은 아니었음).

### 2026-07-14 — 건물 전면 회피(overfly 금지) + 벽 여유 1 m
- `WorldModel.from_city_yaml`에 `overfly_allowed` 추가. False면 건물을 **순항밴드
  전체를 막는 기둥**으로 처리 → 낮은 건물도 넘지 않고 전부 측방 회피.
- 세 노드(astar/sfc/bspline)에 `overfly_allowed` 파라미터 연결.
- config: 세 노드 `inflation_xy_m: 1.0`(=벽 1 m 여유), `overfly_allowed: false`,
  A* `resolution_m: 4.0`(빡빡한 맵에서 깔끔 수렴에 필요).
- 검증: 스폰→트레일러 경로 건물 footprint 통과 0/800, free_frac=1.0.
- 배경: 새 맵 건물이 10–20 m라 25 m 순항 시 낮은 건물은 넘어갔었음(버그 아님, 3D 정상).
  사용자 요청으로 전면 측방회피로 전환.

### 2026-07-14 — 맵 갱신 대응 + 경유점 라우팅 + jo 푸시(da446b3)
- 팀원 `52ddc82`가 autonomy_planner 삭제 + 도심맵 전면 수정(스폰 587,580,
  건물 205개 전부 10–20 m). 로컬을 origin/jo에 정렬 후 path_plan을 그 위에 커밋.
- 경유점 지원: `AStarPlanner3D.plan_through` + `waypoints_enu_m` 파라미터 +
  시각화 `--waypoints` (경로를 각 경유점으로 강제 통과 → 슬라롬/난이도 경로).
- config start/goal을 스폰→트레일러로 설정. README에 Running/웨이포인트 섹션 추가.

### 2026-07-14 — ego-planner B-spline 백엔드 이식
- QingZhuanya/corridor_Bspline_optimization 참고. 기존 하드-프로젝션 `bspline.py` 삭제.
- `uniform_bspline.py`(균일 B-spline + parameterize) + `bspline_optimizer.py`
  (L-BFGS로 smoothness+corridor rebound+feasibility+fitness, `check_collision_and_rebound`).
- SFC 재작성: 제어점별 **보장된 free 박스**(자유점 seed) — 기존 "구간 AABB seed"가
  긴 구간에서 건물 포함 슬래브를 반환하던 버그 수정.

### 2026-07-14 — A* BigZaphod식 리팩터 + 강화 비용
- 제네릭 코어 `a_star_search(start,goal,neighbors,heuristic)` + 격자 어댑터.
- 강화 edge-cost: 거리×(1 + 이격부족 + 고도이탈) + 상승벌점 (`WorldModel.clearance` 추가).
- 벽 이격 0.00→0.70 m 개선(거리만 대비).

### 2026-07-14 — path_plan 패키지 신설, autonomy_planner 삭제
- 15k줄 autonomy_planner 제거, 깔끔한 path_plan ament_python 패키지로 재시작.
- A*/SFC/B-spline/MPC 4개 노드 + world_model + ros_msgs + launch + config + docs.
