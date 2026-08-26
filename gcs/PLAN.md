# drone_gcs — 계획서

도심 회피비행(`path_plan`) + 이동 트레일러 정밀착륙(`mpc_landing` / `landing_mpc`)을
하나의 지상국 GUI에서 보고 조작한다. QGroundControl 포크가 아니라 **독립 GUI**다.

## 0. 확정 사항

| 항목 | 결정 | 이유 |
|---|---|---|
| 스택 | PySide6 + rclpy | `path_plan.world_model` 등을 그대로 import, colcon 한 번에 빌드. 개발 속도 |
| 위치 | 이 저장소의 새 최상위 `gcs/` | README의 경계 규칙("실기체에서 도는가?") — GCS는 지상국에서 돈다 |
| 패키지 | `gcs/drone_gcs` (ament_python) | |
| 3D 기체 모델·자세 위젯 | 없음 | 요청 범위 밖. 맵 + 상태창 + 영상만 |

PySide6는 이 머신에 아직 없다(`pip install --user PySide6`). PyQt5 5.15 / PySide2 는
설치돼 있으므로 `drone_gcs/qt.py` import shim 한 겹을 두고 공통 API만 쓴다 → 폴백 가능.

## 1. 핵심 제약 — 맵은 언제든 교체된다

GUI 코드에 **맵 상수를 하나도 넣지 않는다.** 맵은 디렉터리 하나를 갈아끼우면 바뀐다.

```
gcs/drone_gcs/maps/<map_name>/
  map.yaml            # GUI가 읽는 유일한 파일 (디스크립터)
  basemap.png         # 배경 — 월드 지면 텍스처 또는 폴리곤 합성 렌더
  buildings.json      # 건물 링 + roof_z (벡터: 줌해도 선명, 클릭 조회)
  occupancy_z<z>.png  # (선택) 플래너가 보는 팽창 점유도
```

### map.yaml 스키마

```yaml
schema_version: 1
name: city_uav
source_world_yaml: simulation/gazebo/maps/city_coordinates_uav.yaml
bounds_enu_m: {x: [-650.0, 650.0], y: [-650.0, 650.0]}
basemap: {file: basemap.png, size_px: [2048, 2048]}   # m_per_px 는 bounds/size 에서 유도
occupancy: {file: occupancy_z25.png, cruise_z_m: 25.0}   # 선택
px4_local_origin_enu_m: [587.0, 580.0, 0.24]
spawn_enu_m: [587.0, 580.0]
cruise_band_m: [10.0, 20.0]
default_goal_enu_m: [200.0, -128.0, 15.0]
layers: [basemap, occupancy, buildings, geofence]
entities:
  - name: trailer_aruco          # gz 엔티티 이름 = 브리지의 키
    label: 트레일러
    footprint_m: [5.5, 3.0]
    color: "#f59f00"
    trail: true
    route_yaml: simulation/gazebo/maps/city_uav_trailer_loop.yaml
```

값은 전부 소스 YAML에서 유도한다. 하드코딩 금지 — 특히 `px4_local_origin_enu_m`은
`city_coordinates_uav.yaml`의 `frames.px4_local.origin_enu_m`에서 읽는다.

### 베이커

```bash
python3 tools/bake_map.py --world-yaml <city_coordinates_*.yaml> \
        [--ground-texture <png>] [--cruise-z 25] --out maps/<name>
```

- 지면 텍스처가 있으면 그것이 basemap. **georef는 월드 SDF에서 검증한다** —
  `applepark_uav.world`의 지면 visual이 `<heightmap><size>1300 1300 0.001</size>`
  이고 텍스처 `<size>1300</size>`(1300 m 당 1타일)이므로
  `road_surface_city_uav.png` 2048 px ↔ ENU ±650 m, 정확히 **0.634765625 m/px**.
- 텍스처가 없으면 건물 폴리곤에서 배경을 합성 렌더한다.
- `schema_version: 2` 계열 YAML이면 어떤 맵이든 통과 (city / mountain / mpc_landing_200m).
- 최소 구성도 허용: **이미지 + bounds 두 줄만 손으로 써도 뜬다** (외부 맵·위성사진용).
- 건물 링은 YAML footprint(구멍 포함)를 쓰고, `city_uav_building_vertices.csv`와
  개수·좌표를 교차 검증한다.

### 런타임 교체

GUI 상단 맵 드롭다운이 `maps/*/map.yaml`을 스캔한다. `--map <name|path>` CLI 인자도
받는다. 고르면 재시작 없이 basemap·벡터·투영을 갈아끼운다.

## 2. 맵 캔버스 (중심 위젯)

`map_view.py` — QPainter 단일 위젯. 좌표 변환은 **`MapProjection` 하나**만 존재하고
(ENU m ↔ 화면 px, pan/zoom) 나머지 그리기는 전부 ENU 미터로 한다.
`MapProjection`은 Qt에 의존하지 않는 순수 클래스 → 전부 pytest로 덮는다.

레이어 (아래→위):

1. basemap 픽스맵 — 줌 레벨별 캐시
2. 팽창 점유도 (선택, 반투명) — 경로가 왜 휘는지 보이게
3. 건물 폴리곤 — 높이별 음영. **현재 순항고도에서 넘을 수 있는 건물 / 못 넘는 건물을
   색으로 구분** (25 m 순항에서는 대부분 측면 회피이므로 이게 없으면 회피가 임의로 보인다)
4. geofence 경계
5. SFC 코리도 박스 → A\* 전역경로(점선) → B-spline 궤적(실선) → MPC preview(짧고 밝게)
6. 실제 비행 궤적 (ring buffer)
7. 동적 개체(트레일러 등) + 예정 루프 경로
8. 드론 아이콘(yaw 회전) + 헤딩선 + depth 콘, 목표·경유점 마커,
   드론–트레일러 상대선 + 거리 + 포획반경 원

상호작용: 휠 줌 / 드래그 팬 / `F` 드론 추적 / 클릭 = 목표 지정
(`/path_plan/astar_planner/goal`에 `PoseStamped` — 노드가 이미 실시간 재계획 지원) /
shift-클릭 = 경유점 추가 / 건물 클릭 = ID·높이 표시 / `R` 뷰 리셋.

## 3. 동적 개체 — 트레일러

트레일러 실제 pose는 **ROS에 없다.** `simulation/gazebo/trailer_waypoint_driver.py`가
gz-transport(`gz.transport13`, `gz.msgs10.Pose_V`)로 직접 받는다. 그래서 브리지가 필요하다.

새 노드 `simulation/gz_bridge/entity_pose_node.py`:

- gz `/world/<world>/dynamic_pose/info` (`Pose_V`) 구독 —
  `trailer_waypoint_driver.py`의 `PoseReceiver`를 재사용
- 파라미터 `entities: [trailer_aruco, x500_city_rgbd_lidar_0, ...]`
- 발행 `/sim/entity_poses` (`geometry_msgs/PoseArray`) + `/sim/entity_names`
  (`std_msgs/String`, JSON 배열, latched)

GCS는 `map.yaml`의 `entities:` 목록만 보고 레이어를 만든다 — 트레일러 전용 코드는 없다.
로버·다른 드론·새(bird)도 이름만 추가하면 뜬다. 실기체 단계에서는 같은 레이어에
다른 소스(MAVLink 등)를 꽂는다.

## 4. ROS 링크

`ros_link.py` — rclpy 노드를 **QThread에서 `MultiThreadedExecutor.spin`**, 위젯에는
Qt 시그널로 전달(queued connection). Falcon_QGC의 GUI 스레드 `spin_some` 방식과 일부러
다르게 간다 — 파이썬 GUI 스레드에 20 Hz 영상 DDS를 얹지 않는다.

| 토픽 | 타입 | 용도 |
|---|---|---|
| `/mavros/local_position/pose`, `.../velocity_local` | PoseStamped, TwistStamped | 드론 위치(맵 ENU로 오프셋)·속도·yaw |
| `/mavros/state`, `/mavros/extended_state`, `/mavros/battery` | State, ExtendedState, BatteryState | armed·모드·landed·배터리 |
| `/path_plan/global_path`, `/path_plan/trajectory_path` | Path (**TRANSIENT_LOCAL**) | A\*, B-spline |
| `/path_plan/corridor_markers` | MarkerArray (latched) | SFC 박스 |
| `/path_plan/tracking_mpc/mpc_preview` | Path | MPC 지평선 |
| `/path_plan/cmd_vel`, `/path_plan/depth` | TwistStamped, Range | 명령속도 화살표, 전방 최근접 |
| `/sim/entity_poses` | PoseArray | 트레일러 등 동적 개체 |
| `/front_camera/image`, `/down_camera/image`, `/gimbal_camera/image` | Image | 영상 |
| `/mission/state` 또는 `/mpc_landing_node/state` | String | 착륙 FSM 단계 |
| `/aruco/detected`, `/marker/position`, `/marker/valid` | Bool, PointStamped, Bool | 마커 인식 |

발행/서비스: `/path_plan/astar_planner/goal` (PoseStamped),
`/mpc_landing_node/approve`·`abort` (Trigger — 게이트 승인 버튼).

> QoS 함정: 플래너 토픽은 latched(TRANSIENT_LOCAL)다. 맞추지 않으면 아무것도 오지 않는다.
> 센서·영상은 BEST_EFFORT.

## 5. GUI 셸

- **중앙**: 맵 캔버스 (지배적)
- **우측 상태창**: 연결/armed/모드 칩, 고도·속도·배터리, ENU 좌표, 목표까지 거리·ETA,
  플래너 통계(경로길이 / 재계획 횟수 / 코리도 박스 수 / MPC 해결시간),
  전방 최근접 장애물 바
- **좌하단 영상**: 라이브 `sensor_msgs/Image` 토픽 드롭다운(rqt 스타일) + fps.
  rgb8/bgr8/mono8은 cv_bridge 없이 QImage 직결
- **좌상단 미션**: 목표 좌표 입력 + Plan, 경유점 리스트,
  정밀착륙 게이트 승인/중단 버튼 + 단계 표시(PRECHECK → … → TOUCHDOWN)

## 6. 없는 브리지 채우기

`flight/path_plan/PROGRESS.md`가 스스로 적어둔 TODO. `path_plan/px4_io_node.py` 하나로:

- `/mavros/local_position/pose` + `velocity_local` → `/path_plan/odometry`
  (MAVROS ENU + `px4_local.origin_enu_m` 오프셋 = 맵 ENU. NED 수학 재발명 금지)
- `/front_depth/image_raw` (32FC1) → `/path_plan/depth` (`Range`, 중앙 ROI 최소값,
  NaN/0 필터, max_range 클램프)
- `/path_plan/cmd_vel` → `/mavros/setpoint_raw/local` (`PositionTarget`, 속도 + yaw_rate 마스크)
- OFFBOARD 진입 순서: setpoint 스트림 ≥2 Hz **먼저** → 모드 전환 → arm
- 워치독: `cmd_vel` 0.5 s 끊기면 제자리 정지, 2 s면 AUTO.LOITER로 이탈

## 7. 브링업 + 테스트

- `gcs/run_city_mission.sh`: `run_px4_map.sh city` → MAVROS → sensor bridge →
  `entity_pose_node` → `path_plan.launch.py` → `px4_io_node` → GCS
- **오프라인 리플레이** `--replay <pursuit_sim CSV>`: `path_plan/tools/pursuit_sim.py`가
  이미 드론·트레일러·재계획을 CSV로 남긴다. Gazebo/PX4 없이 전부 움직인다 →
  GUI 개발과 시연의 주력 경로
- 헤드리스 pytest: `MapProjection` ENU↔px 왕복(코너·스폰 포함), 베이커 산출물(건물 205개,
  georef 값), 메시지→dataclass 변환, depth ROI 축약, ENU 오프셋 수학
- GUI 앱 자체는 에이전트가 이 환경에서 실행할 수 없다(Bash 툴에서 exit 144).
  화면 확인은 사람이 한다. 그래서 **paint 코드와 순수 로직을 분리**해 로직을 전부 테스트로 덮는다

## 작업 순서

1. 맵 팩 + 베이커
2. 맵 캔버스
3. ROS 링크(읽기 전용) + 리플레이
4. gz 엔티티 pose 브리지 (트레일러)
5. GUI 셸
6. PX4 IO 브리지
7. 브링업 + 문서

1+2만 되면 리플레이로 이미 볼 수 있다. 6이 "실제로 피해서 나는" 것을 만드는 조각이다.

## 리스크

- PySide6 pip ↔ 시스템 Qt5 충돌 → user site 설치 + `qt.py` shim으로 PyQt5 폴백 여지
- 2048 px 배경 + 폴리곤 매 프레임 → 줌별 픽스맵 캐시, 메시지마다 repaint 금지
  (20–30 Hz QTimer로 통합)
- A\*→SFC→B-spline 재계획은 수 초 걸린다 → "계획 중…" 상태 표시가 없으면 멈춘 것처럼 보인다
- 트레일러 pose가 gz-transport 전용 → 실기체에는 그 소스가 없다.
  개체 레이어를 소스 무관하게 설계해야 하는 이유
