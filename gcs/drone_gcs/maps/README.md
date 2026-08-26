# 맵 팩 — 맵 교체 방법

GCS는 맵을 **디렉터리 하나**로 본다. 여기 디렉터리를 추가하면 GUI 맵 목록에 나타난다.
GUI 코드에는 맵 상수가 없다.

```
maps/<이름>/
  map.yaml            # GUI가 읽는 유일한 파일
  basemap.png         # 위에서 본 배경. bounds_enu_m 전체에 정확히 펼쳐진다
  buildings.json      # 장애물 링 + 지붕 높이 (벡터 오버레이)
  occupancy_z<z>.png  # (선택) 순항고도에서 플래너가 막혔다고 보는 영역
```

래스터 규약: **row 0 = 최대 Y(북), column 0 = 최소 X(서)**.
`simulation/gazebo/tools/render_city_uav_astar_map.py`와 같은 규약이라 뒤집지 않고 겹친다.

## 1. 시뮬레이션 맵에서 굽기 (권장)

`schema_version: 2` 좌표 계약(`simulation/gazebo/maps/city_coordinates_*.yaml`)이 있으면
전부 자동으로 나온다.

```bash
cd gcs/drone_gcs
python3 tools/bake_map.py                     # 기본: city_uav
python3 tools/bake_map.py --world-yaml ../../simulation/gazebo/maps/<다른맵>.yaml
```

계약에서 가져오는 것: geofence, 건물 205개 링·지붕높이, PX4 로컬 원점, 스폰,
트레일러 엔티티 이름·footprint, 지면 텍스처, 고정 미션 좌표(목표·트레일러 목적지).

**점유도 레이어는 플래너 설정에서 온다** (`--planner-config`,
기본 `flight/path_plan/config/city_uav.yaml`). 벽 팽창·지붕 여유·overfly 정책을
A\*와 동일하게 맞추기 때문에, 지도가 A\*는 절대 지나가지 않는 경로를 열려 있는 것처럼
보여주지 않는다. city_uav는 `overfly_allowed: false`라 205개 전부가 전 고도 금지 기둥이다.

주요 옵션:

| 옵션 | 뜻 |
|---|---|
| `--no-ground-texture` | 텍스처 대신 건물 footprint로 배경을 합성 |
| `--ground-texture <png>` | 배경 이미지를 직접 지정 |
| `--basemap-px 2048` | 배경 해상도 |
| `--cruise-z 25` | 점유도를 계산할 고도 (기본: 순항밴드 중앙) |
| `--occupancy-res-m 1.0` | 점유도 해상도 |

지면 텍스처를 쓸 때 베이커는 월드 SDF의 heightmap `<size>`와 텍스처 `<size>`를 읽어
**한 타일이 지면 전체를 정확히 덮는지 검증**한다. 안 맞으면 실패한다 — 안 맞는 이미지를
배경으로 쓰면 모든 건물이 조용히 어긋난 자리에 그려지기 때문이다.

## 2. 손으로 만들기 (외부 맵·위성사진)

이미지와 bounds만 있으면 뜬다. 나머지는 전부 선택 항목이다.

```yaml
schema_version: 1
name: my_field
bounds_enu_m: {x: [-100.0, 100.0], y: [-50.0, 50.0]}
basemap: {file: basemap.png}      # size_px 생략하면 이미지에서 읽는다
```

이미지의 네 변이 `bounds_enu_m`의 네 변에 정확히 대응해야 한다. 그것만 맞으면
드론·경로·트레일러가 제 위치에 그려진다.

## 3. 선택 필드

| 필드 | 용도 |
|---|---|
| `px4_local_origin_enu_m` | MAVROS 로컬 ENU → 맵 ENU 평행이동. 틀리면 드론이 엉뚱한 곳에 뜬다 |
| `spawn_enu_m` | 초기 뷰 중심 |
| `cruise_band_m` | 순항 고도대. 건물 통과 가능 여부 색칠에 사용 |
| `overfly_allowed` | 플래너의 overfly 정책. false면 지붕 높이와 무관하게 전부 장애물 |
| `default_goal_enu_m` | 목표 입력창 기본값 |
| `markers` | 고정 관심점 (스폰·목표·트레일러 목적지) |
| `entities` | 동적 개체 (트레일러·로버·타 기체). `name`은 pose 소스가 보고하는 이름 |
| `layers` | 그릴 레이어와 순서 |

`entities[].name`은 Gazebo 엔티티 이름이다 (city_uav는 `trailer`).
`simulation/gz_bridge`의 엔티티 pose 브리지가 그 이름으로 pose를 발행한다.
맵 팩은 pose가 **어떻게** 오는지는 말하지 않으므로, 실기체에서는 같은 레이어에
다른 소스를 꽂을 수 있다.
