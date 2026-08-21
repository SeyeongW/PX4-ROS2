# path_plan 진행 기록 (worklog)

> 이 파일은 **앞으로 하는 모든 작업을 누적 기록**하는 로그입니다. 새 작업/수정은
> 맨 위 "최근 작업"에 날짜와 함께 추가합니다. 알고리즘 상세는 `README.md`,
> 노드별 파라미터는 `docs/nodes.md` 참고.

## 현재 상태 (2026-07-16 기준)

- **완성**: A* → SFC → B-spline → **진짜 QP-MPC(`TrackingMPC`)** 파이프라인. 로컬 추종을
  pursuit 근사에서 홀로노믹 이중적분기 최적화 MPC로 교체하고 발산(속도>20, 가속~700) 수정.
  순항 10 m/s, |v|≤12 / |a|≤4. 이동 트레일러는 등속 인터셉트 + B-spline glide로 착륙.
- **미완(다음)**: 로컬 `pursuit:=true` 실비행으로 MPC 게인/재계획 주기, 착륙 인터셉트
  타이밍 튜닝. depth 로컬 회피 실측 검증.
- **브랜치**: `main`에 푸시.

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

### 2026-07-16 — 로컬 컨트롤러 진짜 QP-MPC 전환 + 발산 수정 + 이동착륙 인터셉트
- **배경**: 지금까지 "MPC"라던 로컬 추종은 실제로는 pursuit/unicycle 근사였음. 도심
  실비행에서 속도명령이 발산(속도 >20 m/s, 가속 ~700 m/s²)해 피치가 떨리는 문제 확인.
- **`mpc.py` / `mpc_node.py`**: 기본 컨트롤러를 **진짜 최적화 MPC(`TrackingMPC`)**로 교체.
  홀로노믹 3D 이중적분기 위의 condensed QP + SLSQP, |v|/|a| 하드제약 + terminal cost.
  `controller: mpc | pursuit | unicycle` 셀렉터 추가(기본 `mpc`, `holonomic` bool은 legacy로
  pursuit↔unicycle만 선택). strafing 멀티콥터에 홀로노믹이 맞아 unicycle의 nose-coupled
  제자리 스핀 없음.
- **발산 수정**: 원인은 속도 추정을 world-frame으로 못 먹여 매 tick MPC 최적점이 `a_max·dt`로
  붕괴한 것. (1) `/path_plan/odometry` 위치 **유한차분 + EMA**로 world 속도 추정, 스폰 스파이크
  게이트. (2) MPC integrator state를 **측정속도가 아닌 마지막 명령속도**로 두고
  **anti-windup 앵커**(`cmd_lead_max`)로 명령이 측정을 앞서는 양 제한. (3) 속도명령에 **jerk
  제한**(C1-연속) → 피치 attitude 매끈. (4) SLSQP warm-start를 box bound 안으로 clip +
  "Values in x were outside bounds" RuntimeWarning만 silence(제어율 로그 홍수 제거).
- **`config/city_uav.yaml`**: 순항 4→**10 m/s**, `max_vel` 5→**12**, `max_acc` 1.5→**4**
  (PX4 `MPC_XY_VEL_MAX`=12에 정렬). MPC cost 가중치 `mpc_q_pos=4 / q_vel=0.4 / r_acc=0.05 /
  q_terminal=20` 추가. B-spline `cruise_speed`가 MPC가 추종하는 속도프로파일의 실질 상한이라
  같이 상향.
- **`moving_land_node.py`(이동 트레일러 착륙 개선)**: `land_vel_gain`(0.45) 추가 — 최종단계는
  부드러운 게인으로 스냅 없이 안착. 데크 **등속 예측 인터셉트**(pos + vel·T_go)로 움직이는
  덱을 앞질러 조준하고 **B-spline glide**로 강하, `/path_plan/cmd_vel` 구독, 핸드오프 때 현재
  드론속도로 slew limiter 시딩. 하강 퍼널(`descend_cone`)은 인터셉트+글라이드 방식으로 대체.
- **`flight_logger.py`**: disarm 시 **Gazebo 실비행 figure 4종**(`gazebo_flight_topdown /
  _mpc / _profiles / _compute.png`) 저장 — visualize_pipeline과 동일 시각언어, 매 재계획의
  A*/B-spline 레퍼런스 전부 누적해 추종오차·compute 부하(A* plan time, MPC per-tick) 표시.
- **`gazebo/maps/city_uav_trailer_loop.yaml` + `run_px4_map.sh`**: 도심 실제 건물을 피하는
  **building-clear 폐루프 웨이포인트**(최소 이격 6.9 m, 둘레 ~4068 m) 추가 — gz 트레일러는
  실제 충돌하므로 analytic 정사각 대신 A*로 뽑은 개활 회랑을 위빙. `trailer_loop_driver`가 존재
  시 이를 추종, `tools/pursuit_sim.py`(무충돌)는 정사각 유지.
- **`visualize_pipeline.py`**: 폰트 크게+볼드, 장애물 색 밝게(경로 가림 방지), 기본값 갱신
  (`res` 8.0, goal=트레일러 SW `-587,-512`, `cruise_z` 35m=30~40m 밴드 중앙, full-height
  no-fly 컬럼, inflation 5.0). figures 1~8 + `pursuit.gif` 전면 재생성.
- **검증**: 본 환경엔 ROS2/Gazebo 없어 오프라인 시각화 + `py_compile` 기준. 다음: 로컬
  `pursuit:=true` 실비행으로 MPC 게인/재계획 주기, 착륙 인터셉트 타이밍 확인.

### 2026-07-15 — 이동 트레일러 추격 + 착륙 (pursuit_sim 실비행 이식, PR #34 main 병합)
- **배경**: 기존 도심 비행은 **정적 좌표**로만 갔음 — `mavros_static_path`가 고정 goal
  (트레일러 스폰)로 A*→SFC→B-spline→MPC를 **한 번** 계획하고 `AUTO.LAND`. `tools/
  pursuit_sim.py`가 오프라인으로 증명한 "이동 트레일러를 계속 재계획으로 추격" 거동을
  실제 PX4+Gazebo+MAVROS sim으로 옮기고, **움직이는 트레일러 위에 착륙**하도록 확장.
- **`trailer_loop_driver.py`(신규)**: 도심 트레일러(`trailer_aruco`)를 pursuit_sim과
  **동일한 정사각 루프**로 gz `cmd_vel` 구동(루프 지오메트리는 공용 `gazebo/maps/
  city_uav_trailer_loop.yaml`에서 로드), 실제 위치를 `/trailer/position` +
  `/trailer/velocity`(map ENU)로 방송. gz 미탑재 시 해석적 큐만 발행하도록 폴백.
- **`mavros_static_path.py`(수정)**: `pursuit_mode` 추가(기본 **false** → 정적 데모 불변).
  CRUISE에서 `replan_period_s`마다 트레일러 실시간 위치를 `/astar_planner/goal`로 재발행
  →파이프라인이 이동 표적으로 연속 재계획. `terminal_range_m` 이내 진입 시
  `/pursuit/land_enable` 래치 + **셋포인트 스트림 중단**으로 착륙 권한 이양(단일 권한).
  이동표적이라 `AUTO.LAND`(수직 하강)는 정적 데모에서만 사용.
- **`moving_land_node.py`(신규, 착륙 로직 단일 파일)**: 파라미터를 **전부 노드 내부**에
  두어 런치는 설정 없이 노드만 로드. 속도매칭 큐추종(`v = v_ff + kp·e`) + 하강 퍼널 +
  덱 위 force-disarm. 표적 소스를 **교체형 estimator** 뒤로 분리 — 지금은
  `CueVelocityEstimator`(속도기반), 나중에 `ArucoEstimator`(비전) 스텁으로 전환 시
  **이 파일만** 수정. 드론 위치는 `/path_plan/odometry`(map ENU)로 받아 큐와 동일 프레임.
- **월드**: 도심 트레일러 include를 cmd_vel 구동 가능한 `trailer_aruco`로 교체(미구동 시
  정지 → 정적 데모 영향 없음).
- **런치/등록**: `px4_mavros.launch.py`에 `pursuit:=true` 인자(참이면 드라이버+착륙노드
  스폰, 착륙노드는 파라미터 블록 없이), `setup.py`에 두 엔트리포인트 등록.
- **검증**: 본 환경엔 ROS2/Gazebo 없어 SITL 실행 불가 → 변경 파이썬 `py_compile` 통과 +
  월드 SDF XML well-formed만 확인. 실행 절차는 PR 본문 참조(`run_px4_map.sh city` →
  `ros2 launch path_plan px4_mavros.launch.py pursuit:=true`). 다음: 로컬에서
  `pursuit:=true` 실비행으로 타이밍/게인(재계획 주기, 착륙 vel_gain/퍼널) 확인.

### 2026-07-15 — 등속10 복귀 + 벽이격 5m설정 + flight_logger 3종 figure
- **가변속도 제거**: `mpc_node`에서 곡률 속도캡·`_path_curvature` 삭제, config
  `a_lat_max/v_min` 삭제 → 등속. **속도 10 고정**: config `v_ref_m_s=v_max_m_s=10`,
  launch `speed_from_fcu` 기본 false(FCU 12로 안 덮이게).
- **벽 이격 5m 설정**: inflation 10→5(3노드), clearance_pref 15→5, demarc 1.0→0.3.
  실측: **A* 경로 최소 11.9m, B-spline(실제 추종) 최소 6.1m**(스무딩 코너컷),
  collision_free=False(가장 좁은 코너에서 5m 버퍼를 살짝 침범—단 실제 벽까진 6.1m).
  ⇒ "12m"는 계획 경로 기준이고 실제 비행 최소는 ~6m. 하한(=날 수 있는 최소)은 5m.
- **flight_logger 전면 개편**: disarm(착륙) 시 visualize_pipeline과 같은 팔레트로
  3종 저장 — `gazebo_flight_topdown.png`(건물 폴리곤+A*주황+B-spline초록+실제비행
  마젠타, **실제 비행 최소 벽이격을 제목에 주석**), `gazebo_flight_profiles.png`
  (속도/가속도/고도 3단+순항밴드 음영), `gazebo_flight_mpc.png`(레퍼런스 vs 실제
  탑다운 + 추종오차 cKDTree, mean/max). odom 속도·A* global_path·B-spline 구독 추가.
  출력은 리포 `figures/`. 합성데이터로 3종 렌더 검증 OK.

### 2026-07-15 — A* 속도 25배↑ (weighted A* + 해상도 8m)
- **증상**: 실제 SITL A* 1회 계획이 ~77s(res 4.0, inflation 10, 1300m 맵, 파이썬
  shaped-cost, 96,681 노드 확장) → 이륙 후 HOLD에서 오래 대기.
- **원인**: 순수 파이썬 A* + 4m 격자로 1300m 맵(xy 셀 ~10만) + 노드마다
  `clearance()`(205 건물 거리) + 26이웃 `is_free`.
- **수정**: (1) `a_star_search`에 **weighted A*** 추가(`weight` 인자, f=g+weight·h;
  AStarPlanner3D `heuristic_weight` → astar_node 파라미터 → config). (2) config
  `resolution_m` 4→**8**, `heuristic_weight`=**1.5**.
- **실측**(inflation 10, clearance_pref 15, spawn→trailer):
  res4/w1=77s(96681노드) → res8/w1=21.9s → **res8/w1.5=3.0s**(2187노드) →
  res8/w2.0=0.8s(487노드). **벽 이격은 전부 ~26m로 동일**(맵이 트여서 weighted A*
  최적성 손해 사실상 0). 채택: **res8/w1.5 = 3초(25배↑)**. 더 빠르게=w2.0(0.8s).
- 검증: 컴파일·config·planner 생성 OK. 재빌드 불필요(.py/config 심링크).

### 2026-07-15 — 안전여유 10m + MPC 가변속도 + 랜딩 미작동 진단강화
- **벽 이격 3→10m**: config 세 노드(`inflation_xy_m`) 3.0→**10.0**, `clearance_pref_m`
  10→15, bspline `demarcation_m` 0.3→**1.0**. **오프라인 검증**: inflation 10m에서
  A* 여전히 성공(웨이포인트 13, expanded 96681, 77s), **B-spline/경로 실제 벽 이격
  최소 25m**(넓은 회랑 우회) → 10m는 충분히 실현가능하고 여유 큼. (기존 3m는 코너컷
  + 추종오차에 먹혀 벽을 긁었음.)
- **MPC 가변속도(등속 폐지)**: `mpc_node._holonomic_cmd`에 **측방가속도 한계 기반
  속도캡** 추가 — 앞쪽 look-ahead 구간의 최대 Menger 곡률 κ로 `v=sqrt(a_lat_max/κ)`
  계산, 직선=v_max·굽을수록 감속(`v_min` 바닥). config `/tracking_mpc`에
  `a_lat_max_m_s2: 4.0`, `v_min_m_s: 2.0`. **검증**: 곡률 κ=1/R 정확, 직선→12,
  R30→10.95, R15→7.75, R8→5.66 = √(a_lat·R) 정확. 목표근처 goal_slow 감속은 유지.
- **랜딩 미작동 진단강화(`mavros_static_path.py`)**: land 반경 8→**12m**(더 일찍
  트리거), CRUISE 중 **목표까지 거리 2s마다 로그**(`cruising: X m to goal`),
  `AUTO.LAND` 요청에 **결과 콜백**(`mode_sent=False`면 "PX4 rejected AUTO.LAND" 경고
  + `_ensure_landing`가 1Hz로 확정될 때까지 재시도). 이제 로그로 (a)목표 도달 여부와
  (b)PX4가 AUTO.LAND를 거부하는지 구분 가능. PX4 SITL landing custom_mode="AUTO.LAND".
- 재빌드 불필요(.py/config/launch 심링크). launch 파싱·임포트 OK.

### 2026-07-15 — RViz에 건물을 실제 3D 형상으로 표시(박스 아님)
- **새 노드 `building_markers.py`**: city YAML의 각 건물(`polygon_prism`)을 A*가
  쓰는 팽창 AABB가 아니라 **실제 `footprint.outer` 폴리곤(4~27각형)을
  foundation_z→roof_z로 압출한 3D 프리즘**으로 렌더. 좌표가 Gazebo 월드/계획경로와
  정확히 일치(map ENU). MarkerArray 3종: 벽(TRIANGLE_LIST, 높이별 콘크리트색 음영),
  지붕(TRIANGLE_LIST, **ear-clipping 삼각분할** — 오목 폴리곤도 정확), 수직 모서리선
  (LINE_LIST). latched(TRANSIENT_LOCAL)라 RViz 늦게 붙어도 수신. `/path_plan/buildings`.
- launch `px4_mavros.launch.py`에 `building_markers` 노드 추가(rviz 인자로 게이팅),
  rviz config에 MarkerArray 디스플레이(Buildings) 추가, setup.py 엔트리포인트 등록.
- 검증: `_triangulate`가 205개 건물 전부 n-2 삼각형으로 정확 분할(오류 0),
  colcon build OK, 노드 실행 시 `Publishing 205 buildings ...` + 토픽 발행 확인,
  launch 파싱 OK. **새 엔트리포인트라 colcon build 필요**(했음).

### 2026-07-15 — RViz 자동실행 + 목표 도달 자동착륙 + 이륙후 경로완료까지 호버
- **브리지 상태머신화(`mavros_static_path.py`)**: 기존 `_took_off` bool →
  `phase ∈ {CLIMB, HOLD, CRUISE, LAND}`.
  - `CLIMB`: `takeoff_alt_m`(35m)까지 **순수 수직**(vx=vy=0, vz=takeoff_vz).
  - `HOLD`: 이륙고도 도달 시 A* 트리거(`/astar_planner/start` 발행) 후 **제자리
    호버**(vz만 P제어로 고도유지, 측방 0). **`/path_plan/trajectory` 수신 =
    경로계산 완료 시에만 CRUISE로 전환** → "이륙하자마자 비스듬히 상승/이동"하던
    불안정 제거(사용자 요청: 경로 완료 후 출발).
  - `CRUISE`: MPC cmd_vel 전달. 목표(=trajectory 마지막 점) xy 반경
    `land_radius_m`(기본 8m) 안에 들면 LAND.
  - `LAND`: `set_mode(AUTO.LAND)`를 1Hz로 재요청(확인될 때까지), OFFBOARD 재요청은
    중단(착륙과 안 싸우게). 착륙 disarm 시 `flight_logger`가 결과 그림 저장.
- **RViz 자동실행**: `px4_mavros.launch.py`에 `rviz2` 노드(+`rviz:=false`로 끄기)
  + `rviz/path_plan.rviz`(fixed frame `map`, A* 경로=노랑, MPC preview=마젠타,
  드론 Odometry 화살표, TF). MAVROS는 기본적으로 TF를 안 쏘므로 **브리지가
  `map->base_link` TF를 직접 브로드캐스트**(RViz fixed frame `map` 존재 보장).
  RViz config는 절대경로 로드(재빌드 불필요), setup.py data_files에도 등록.
- **flight_logger 출력 경로 버그 수정**: `parents[3]`(=`~/ros2_ws/figures`, 리포
  밖) → `parents[2]`(=리포 `figures/`). 이제 `figures/gazebo_flight_2d.png` 저장.
- 검증: launch `--show-args` OK(takeoff 35 / land_radius 8 / rviz true), 브리지·
  flight_logger·astar 런타임 임포트 OK, 상태머신 헬퍼 6종 존재 확인.
- 참고: pursuit_sim gif는 코드정상이나 큰 맵+다수 리플랜으로 500s 타임아웃(EXIT
  124)에 미완 — `--animate` 라이브 창은 즉시 동작. 시나리오 파일은 복구됨.

### 2026-07-15 — 벽 충돌 근본원인 수정(이륙고도) + pursuit 애니메이션 복구
- **증상1: 드론이 벽에 부딪힘.** 진짜 원인은 **이륙고도 불일치**. 현재 맵
  `city_coordinates_uav.yaml`은 `a3d1d30(restore jo city)`에서 건물이 **20~50 m
  (평균 35, 최대 50)**로 바뀌었는데(예전 10~20 m 아님!), 순항밴드는 30~40 m인데도
  `px4_mavros.launch.py`의 `takeoff_alt_m` 기본값이 **20.0**이었음. 결과:
  (a) 이륙 후 A*에 넘기는 start의 z=20이 순항 바닥(30 m) 아래 → `_to_cell`이
  바운드 밖 셀로 스냅 → **"start cell blocked"로 A* 실패**(경로 없음), 또는
  (b) 20→30 m로 상승하며 **측방 이동 → 건물(20 m~) 사이를 긁고 지나감**. (이전
  세션이 "35 m 이륙" 코드라 설명했지만 런치 인자 기본값 20.0이 브리지 파라미터
  기본값 35.0을 덮어써서 실제론 20 m로 떴던 것 — 그래서 "안 됐던" 것.)
  → **수정: `takeoff_alt_m` 기본값 20.0 → 35.0**(순항밴드 중앙, 바닥≥floor).
  이제 스폰에서 순수 수직으로 35 m까지 오른 뒤에야 A* 계산·측방 순항 시작.
- **방어 로직**: `astar_node._replan`이 start/waypoint/goal의 z를
  `[cruise_floor, cruise_ceiling]`으로 클램프(`_clamp_to_band`) → odom/이륙고도가
  밴드를 1 m 벗어나도 "start cell blocked"로 죽지 않음.
- **증상2: pursuit_sim 애니메이션이 사라짐.** 진짜 원인은 코드가 아니라 **시나리오
  파일 부재**. `a3d1d30`이 `gazebo/maps/city_uav_trailer_loop.yaml`을 삭제 →
  `python3 tools/pursuit_sim.py`가 실행 즉시 `FileNotFoundError`로 크래시(애니는
  원래부터 `--animate`/`--gif` 플래그 필요, 코드는 멀쩡). → **git `c90cd31`에서
  시나리오 파일 복구**. 실행법: `python3 tools/pursuit_sim.py --animate`(라이브
  창) 또는 `--gif out.gif`(저장). 플래그 없으면 정적 피규어(5·6·7·8)만 생성.
- 빌드 불필요: install→build→소스가 전부 심링크(런치/config/py 즉시 반영).

### 2026-07-14 — PX4 연결(MAVROS OFFBOARD 브리지) — 정지목표 순항
- wang을 main에 병합(로컬 wang-우선, 이후 push)한 뒤, main의 city 맵 기준으로 PX4 연결 시작.
- **새 노드 `mavros_static_path.py`**: (1) `/mavros/local_position/odom`(local ENU)에
  스폰 오프셋 **+(587,580,0)** 를 더해 `/path_plan/odometry`(map ENU)로 재발행(속도/자세는
  평행이동이라 그대로), (2) MPC의 `/path_plan/cmd_vel`(TwistStamped)을
  `/mavros/setpoint_velocity/cmd_vel`로 전달, (3) setpoint 선-스트리밍 후
  `/mavros/set_mode`(OFFBOARD)+`/mavros/cmd/arming`으로 자동 arm. 이륙은 MPC vz(고도홀드,
  z_ref≈25m)로 상승. 좌표 근거: PX4 EKF local 원점=스폰=맵(587,580).
- **새 런치 `px4_mavros.launch.py`**: 기존 `path_plan.launch.py`(A*→SFC→Bspline→MPC,
  이미 /path_plan/odometry·/path_plan/cmd_vel로 remap)만 include + 브리지 노드 추가.
  world/PX4/MAVROS/gz-bridge는 **절대 재실행 안 함**(그건 `./gazebo/run_px4_map.sh city`가 담당).
- setup.py entry point + package.xml `mavros_msgs` 추가. `colcon build --symlink-install` 성공,
  executable/임포트 검증 완료. **실기(SITL) 비행 미검증**.
- 실행: 터미널1 `./gazebo/run_px4_map.sh city` → 터미널2
  `ros2 launch path_plan px4_mavros.launch.py`.
- depth→Range는 미연결(mpc는 inf 기본으로 동작).

#### SITL 1차 구동 디버깅 (같은 날)
- **증상**: arm 직후 `Disarmed by auto preflight disarming`(arm 후 이륙 안 하면 자동 disarm).
- **진짜 원인**: `astar_node`가 기동 즉시 크래시 → global_path/trajectory/cmd_vel 전무 →
  브리지가 zero setpoint만 스트림 → 이륙 못 함. 크래시 지점
  `wps = list(p("waypoints_enu_m", []).value)`: **rclpy는 빈 리스트 `[]` 기본값의
  파라미터 `.value`를 None으로 반환**(타입 추론 불가) → `list(None)` TypeError.
  config에도 `waypoints_enu_m: []`이 있어 항상 발동. → `.value or []`로 수정.
- **브리지 보강**: (1) odom 확보 후에만 arm, (2) **순수 수직 이륙 phase**(cruise_floor
  20m까지 vz만 상승 후 MPC 수평추종 인계) → preflight auto-disarm 회피 + 저고도 측방이동 방지.
  런치에 `auto_arm`/`takeoff_alt_m` 인자 노출(`auto_arm:=false`=arm 없는 안전 드라이런).
- **드라이런 검증(auto_arm:=false, sim 연결 상태)**: astar `A* path published: 6 wpts,
  expanded 65471`, bspline→trajectory(latched), **MPC cmd_vel 5.9Hz 실출력**
  (`linear x=0.28 y=-0.11 z=2.0`=상승중), 브리지 odom 30Hz·setpoint 20Hz. 체인 전체 정상.
- **남은 성능 이슈**: 초기 A* 1회 계획이 **~45s**(res 4.0, 1260m 맵, 파이썬 shaped-cost).
  이륙 phase가 그 동안 20m 호버로 버텨주긴 함. 필요시 resolution↑ 또는 계획 캐시.

#### SITL 2차: 제자리 회전/병진불가 → holonomic 전환 (armed 비행 검증됨)
- **증상**: 이륙·경로생성 OK인데 드론이 제자리에서 pitch/yaw만 까딱, 목표로 안 감.
- **진단(라이브)**: cmd_vel `angular.z=±1.2`(omega_max) 계속 포화, 실제 yaw가
  171°→-123°→-52°→0° 식으로 **빙글빙글**, 위치는 5초에 ~7m(정체). 원인:
  유니사이클 MPC는 속도가 기수방향에 묶여(nose-coupled) 기수를 경로로 돌려야 전진하는데,
  PX4 멀티콥터 yaw는 1.2보다 느리게(≈0.785 MPC_YAWRAUTO_MAX) 따라와 모델 불일치 →
  리밋사이클 → 기수가 계속 돌아 속도벡터도 회전 → 병진 0. **유니사이클이 멀티콥터에 부적합.**
- **수정**: `mpc_node`에 `holonomic`(기본 True) 모드 추가 — 멀티콥터는 옆으로도 날 수 있으니
  월드좌표 pure-pursuit로 look-ahead ref를 향해 직접 병진(속도=기수와 무관), yaw는
  진행방향으로 **부드럽게 분리**(yaw_kp/yaw_rate_max=0.6, 속도 게이팅 안 함), 목표 근처
  감속(goal_slow_radius). 유니사이클은 `holonomic:=false`로 보존.
- **검증(armed, OFFBOARD, 실비행)**: 드론이 (575,575,24)→(557,558,24)로 **SW ~4.1 m/s
  꾸준히 병진**, yaw는 -135°(진행방향)로 안정, 회전 소멸. 목표(-587,-512)로 정상 진행.
  ⇒ **PX4 SITL 실비행에서 A*→Bspline→MPC(holonomic)→OFFBOARD 파이프라인 동작 확인.**

### 2026-07-14 — 순항속도를 PX4 MPC_XY_VEL_MAX에서 자동 취득 (하드코딩 제거)
- 요구: "최대속도로 날되 12를 코드에 박고 싶진 않다."
- MAVROS Humble은 PX4 파라미터를 **표준 ROS2 파라미터**로 노출(`/mavros/param`,
  `ParamGet` srv는 deprecated). `MPC_XY_VEL_MAX=12.0`.
- **구현**: 브리지가 부팅 시 `/mavros/param/get_parameters`로 `MPC_XY_VEL_MAX`를 읽어
  (× `speed_scale`) `/tracking_mpc`의 `v_ref_m_s`·`v_max_m_s`를 `SetParameters`로 세팅.
  mpc_node엔 `add_on_set_parameters_callback` 추가해 라이브 반영(base_v_ref/mpc.v_max).
  런치 인자 `speed_from_fcu`(기본 true)·`speed_scale`(기본 1.0). 코드에 속도 숫자 없음.
- **검증(실비행)**: 로그 `MPC_XY_VEL_MAX=12.0 -> MPC cruise 12.0`, `/tracking_mpc`
  v_ref/v_max=12.0, 실측 |v_xy|≈**11.9~12.1 m/s**로 순항, 목표 근처 goal_slow 감속.
  여유 두려면 `speed_scale:=0.9`.

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
