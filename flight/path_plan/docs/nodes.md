# path_plan — 노드별 알고리즘 · 수치 특성 · 파라미터 튜닝

파이프라인: `astar → sfc → bspline → mpc`
전역경로(A\*+SFC+B-spline)는 한 번(또는 재계획 시) 생성, MPC가 실시간 추종.

```
astar  --global_path-->  (sfc 시각화)
astar  --global_path-->  bspline --trajectory-->  mpc --cmd_vel--> PX4
```

파라미터 기본값은 `config/city_uav.yaml` 기준.

## 알고리즘 차수 요약

| 알고리즘 | 수학적 종류 | 핵심 차수 |
|----------|------------|-----------|
| A\* | 이산 그래프 최단경로 (Dijkstra + admissible heuristic) | 다항식 아님. 비용/휴리스틱 = 1차(선형) + L2거리 |
| SFC | 볼록 자유공간 분해 (반복 박스 팽창) | 1차(선형) 부등식 (AABB = 6 half-space) |
| B-spline | 균일 B-spline + 비선형 무제약 최적화(L-BFGS) | 곡선 3차(cubic), 비용 대부분 2차, 회랑항 3차→2차 |
| MPC | 비선형 MPC (receding horizon, SLSQP) | 모델 비선형(삼각함수), 레퍼런스 3차, 비용 2차 |

---

## 공통: 맵/밴드 파라미터
astar·sfc·bspline 세 노드가 각자 맵을 로드하므로 아래 planning 값은
**동일값 유지**. runtime MPC는 별도 hard radius 1.0 m를 사용한다.

| 파라미터 | 기본 | 의미 | 세팅 |
|----------|------|------|------|
| `vehicle_clearance_xy_m` | 1.5 | 1.0 m hard radius + 0.5 m tracking reserve인 planning 반경 | 안전반경↑ 하려면 ↑ (좁은 틈 막힐 위험) |
| `roof_clearance_m` | 10.0 | 지붕 위 수직 클리어런스 | "넘어가기 가능 높이" 결정 |
| `cruise_floor_m` / `cruise_ceiling_m` | 20 / 30 | 순항 고도 밴드 | 세 노드 동일값 필수 |

---

## 1. `astar_planner` — 전역 경로탐색 (프론트엔드)

- **정체**: 3D 격자 A\* (BigZaphod식 제네릭 콜백 코어 + 강화비용).
- **알고리즘**: best-first 그래프 탐색, `f=g+h`, 26-연결, 유클리드 휴리스틱,
  강화 edge-cost, 시야(LOS) shortcut.
- **수치 특성**: 이산 격자 · 전역 최적(admissible 휴리스틱) ·
  시간 ~ `O(확장수 × 26 × 장애물수)` · 실측 res=4m에서 ~2–4s, 확장 4–6천 ·
  한 번(또는 재계획 시)만 실행.
- **강화 비용**: `cost(n→m) = step·(1 + w_clear·이격부족 + w_alt·|Δ고도|) + w_climb·상승`

| 파라미터 | 기본 | 의미 | 튜닝 (↑ 효과) |
|----------|------|------|--------------|
| `resolution_m` | 2.0 | 격자 셀 크기 | ↓ 정밀·좁은틈 통과·느림 / ↑ 빠름·거친경로 |
| `max_expanded` | 400000 | 탐색 예산 | "budget exhausted" 실패 시 ↑ |
| `clearance_weight` | 0.4 | 벽 이격 보상 강도 | ↑ 골목 중앙(안전) / 과하면 우회·느림 |
| `clearance_pref_m` | 3.0 | 원하는 이격거리 | ↑ 더 벌어짐(밀집 도심선 실현 어려움) |
| `altitude_weight` | 0.05 | 순항고도 유지 강도 | ↑ 상하 방황 억제 |
| `altitude_pref_m` | 25.0 | 선호 고도(밴드 중앙) | |
| `climb_weight` | 0.5 | 상승 억제(에너지) | ↑ 평평 비행 선호 |

---

## 2. `sfc_builder` — 안전 회랑 (시각화/독립용)

- **정체**: 자유 AABB 박스 체인. *최적화기는 회랑을 자체 생성하므로 이 노드는 RViz 시각화용.*
- **알고리즘**: 자유점 seed → 면별 팽창(`box_is_free` 검증).
- **수치 특성**: 박스당 `O(6면 × extent/step × 장애물수)` · 모든 박스 무장애 보장 ·
  선형 부등식(축마다 lo≤p≤hi, 6개 half-space).

| 파라미터 | 기본 | 의미 | 튜닝 |
|----------|------|------|------|
| `step_m` | 0.5 | 팽창 스텝 | ↓ 정밀·느림 / ↑ 거친 박스 |
| `max_extent_m` | 8.0 | 박스 최대 반경 | ↑ 넓은 회랑 / 좁은 곳선 제한적 |
| `seed_spacing_m` | 3.0 | 박스 밀도(경로 샘플 간격) | ↓ 촘촘(연속성↑·느림) |

---

## 3. `bspline_optimizer` — 궤적 최적화 (백엔드)

- **정체**: ego-planner식 코리도어 3차 B-spline 최적화 (QingZhuanya/corridor_Bspline_optimization 이식). 비선형 무제약 최적화.
- **알고리즘**: parameterize 초기화 → **L-BFGS**(smoothness+corridor+feasibility+fitness,
  해석 그래디언트) → 충돌검사 → **rebound 반복**(`check_collision_and_rebound`).
- **수치 특성**: 제어점 `n ≈ 경로길이/ctrl_spacing`(예 83) · 변수 `3(n-6)` ·
  L-BFGS(maxiter 300) · 지역 최적(초기값 의존) · rebound ≤ max_rebound ·
  실측 0.1–0.2s, rebound 1회, 궤적 100% 무충돌.
- **비용**: `f = λ1·smooth(2차) + λ2·corridor(3차→2차) + λ3·feas(2차) + λ4·fitness(2차)`

| 파라미터 | 기본 | 의미 | 튜닝 (↑ 효과) |
|----------|------|------|--------------|
| `cruise_speed_m_s` | 4.0 | 시간배분(평균속도) | ↑ 빠른 궤적(속도·가속도↑) |
| `ctrl_spacing_m` | 5.0 | 제어점 간격 | ↓ 정밀·유연(느림) / ↑ 매끄럽·추종 느슨 |
| `max_vel_m_s` / `max_acc_m_s2` | 5 / 3 | 동역학 한계(feas 기준) | 기체 성능에 맞춤 |
| `lambda_smooth` | 1.0 | jerk 최소화 | ↑ 더 매끄럽·완만 |
| `lambda_dist` | 0.5 | 회랑 유지 | ↑ 박스 안으로 강하게(안전)/덜 매끄럼 (rebound 시 자동↑) |
| `lambda_feas` | 2.0 | v/a 한계 준수 | ↑ 속도초과 억제 (현재 5.7 낮추려면 ↑) |
| `lambda_fit` | 0.2 | A\* guide 추종 | ↑ 원경로 밀착(우회 억제) / ↓ 자유 최적화 |
| `demarcation_m` | 0.3 | 박스 벽 안쪽 여유 | ↑ 더 안쪽(안전마진) |
| `max_rebound` | 10 | 충돌 재최적화 최대 | 어려운 맵일수록 ↑ |
| `sample_count` | 300 | 발행 궤적 샘플 수 | 해상도 |

---

## 4. `tracking_mpc` — 추종 컨트롤러

- **정체**: mpc_ros식 비선형 유니사이클 MPC (Geonhee-LEE/mpc_ros 이식). receding horizon.
- **알고리즘**: 레퍼런스 3차 fit → cte/epsi → 유니사이클 롤아웃 최소화(**SLSQP**) →
  첫 입력 적용. 뎁스 회피(측방 시프트 + 감속). z축은 분리 P제어.
- **수치 특성**: `dt=0.1`(10Hz) · horizon `N=20` → 예측 2s · 변수 `2N=40` ·
  지역 최적/온라인 · solve 수 ms.
- **비용**: `J = Σ w_cte·cte² + w_epsi·epsi² + w_v(v−v_ref)² + w_ω·ω² + w_a·a² + w_dω·Δω² + w_da·Δa²`

| 파라미터 | 기본 | 의미 | 튜닝 (↑ 효과) |
|----------|------|------|--------------|
| `dt_s` | 0.1 | 제어주기/예측스텝 | ↓ 반응 빠름·계산↑ |
| `vehicle_clearance_xy_m` | 1.0 | runtime hard vehicle radius / RViz disk | planner 1.5 m와 역할을 구분 |
| `horizon` | 20 | 예측 스텝(N·dt=2s) | ↑ 안정·미리봄(느림) / ↓ 근시·빠름 |
| `v_ref_m_s` | 4.0 | 목표 순항속도 | |
| `v_max_m_s` / `a_max_m_s2` / `omega_max_rad_s` | 5 / 3 / 1.2 | 물리 한계 | |
| `z_kp` | 0.8 | 고도유지 P게인 | ↑ 고도추종 빠름(과하면 진동) |
| `vz_max_m_s` | 2.0 | 수직속도 한계 | |
| `w_cte` | 2.0 | 횡오차 벌점 | ↑ 경로에 딱 붙음 |
| `w_epsi` | 4.0 | 방위오차 벌점 | ↑ 기수 정렬 강함 |
| `w_v` | 1.0 | 속도추종 | |
| `w_omega` / `w_a` | 0.5 / 0.05 | 입력 사용 | ↑ 부드러운 조향/가속(덜 민첩) |
| `w_domega` / `w_da` | 2.0 / 0.1 | 입력 변화율(스무딩) | ↑ 떨림 억제 |
| `depth_trigger_m` / `depth_emergency_m` | 10 / 4 | 회피 시작 / 비상 거리 | |
| `avoid_lateral_m` | 7.0 | 회피 측방 이동량 | ↑ 크게 비켜감 |

---

## 튜닝 직관 (자주 만지는 것)

| 증상 | 조치 |
|------|------|
| 경로가 벽에 붙는다 | A\* `clearance_weight`↑ / `clearance_pref_m`↑ |
| 좁은 틈을 못 지난다 | A\* `resolution_m`↓ / `vehicle_clearance_xy_m`↓ |
| B-spline 속도초과(5.7) | `lambda_feas`↑ (또는 `cruise_speed`↓) |
| 궤적이 각지다/우회 | `lambda_smooth`↑ / `lambda_fit`↓ |
| 회랑 밖으로 샌다(충돌) | `lambda_dist`↑ / `demarcation_m`↑ / `max_rebound`↑ |
| MPC가 경로 이탈 | `w_cte`↑ / `w_epsi`↑ |
| MPC가 떨린다 | `w_domega`/`w_da`↑ / `horizon`↑ |

수식↔코드 대응은 `README.md`의 "Math ⇄ Code" 섹션 참고.
