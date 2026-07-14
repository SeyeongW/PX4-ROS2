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
