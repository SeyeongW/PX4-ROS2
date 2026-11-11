# PX4 쿼드 테일시터(`SYS_AUTOSTART=1045`) · 30cm 풍선 3D 비전 유도 시뮬레이션

수직 노즈콘 카메라(Raspberry Pi Camera Module 3 Standard, 66° HFOV, 640×360 @ 30Hz)를 탑재한 2 kg 쿼드 테일시터(`stl_pacer_quad`, `1045_gazebo-classic_quadtailsitter`)가 5m 수직 이륙 후 대각선 테일시터 자세로 전환하여 **30cm 풍선(`(90, 0, 40) m`)**을 온보드 비전 3D 시선(LOS)/비례항법(PN)으로 정밀 추적·통과하고 RTL로 자동 귀환·착륙하는 ROS 2 Humble + Gazebo Classic 11 + PX4 SITL 시뮬레이션 패키지입니다.

---

## 1. 주요 스크립트 및 파일 구조 (`balloon_sim/`)

| 경로 | 역할 |
| :--- | :--- |
| `tools/run_tailsitter_sim.sh` | Gazebo + PX4 테일시터 SITL + 온보드 검출기 + 비전 유도 제어기 통합 실행 (`run_vision_center.sh` 호환) |
| `tools/tailsitter_guidance.py` | 쿼드 테일시터 5m 수직 이륙 → 피치다운 전환 → 3D LOS/PN 비전 유도 → 통과 동시 판정 → 후면 전환 감속 및 RTL 착륙 |
| `tools/restart_flight.sh` | 실행 중인 시뮬레이션을 끄지 않고 테일시터 유도 비행만 즉시 재발사 (`refly.sh` 호환) |
| `tools/plot_tailsitter_log.py` | PX4 `.ulg` 바이너리 비행 로그를 파싱하여 6-패널 텔레메트리 대시보드 및 3D 궤적 그래프 생성 |
| `balloon_locator/balloon_locator/stl_airframe.py` | 10개 STL 조립 기체 + 4로터 차동 추력(`CA_AIRFRAME=4`) + 테일시터 주익/미익 공력(`libLiftDragPlugin.so`) SDF 생성 |
| `balloon_locator/config/stl_pacer_quad.json` | 기체 질량/관성/모터 추력 곡선 및 PX4 쿼드 테일시터 파라미터 정의 |
| `balloon_locator/launch/gazebo.launch.py` | ROS 2 + Gazebo Classic + PX4(`1045_gazebo-classic_quadtailsitter`) 런치 구성 |
| `plots/tailsitter_telemetry_dashboard.png` | 테일시터 비행 로그 6-패널 종합 텔레메트리 분석 그래프 |
| `plots/tailsitter_3d_trajectory.png` | 3D 공간 비행 궤적 및 평면(X-Y) 중심 정렬 정확도 그래프 |
| `plots/tailsitter_flight_summary.json` | 최근접 거리(CPA), 통과 속도, 위치 오차 요약 JSON |

---

## 2. 실행 및 비행 로그 분석 명령어

```bash
cd /home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim

# 1) 시뮬레이션 및 테일시터 비전 유도 비행 원클릭 실행
./tools/run_tailsitter_sim.sh

# 2) (선택) 시뮬레이터가 켜진 상태에서 드론만 즉시 재발사 (새 터미널)
./tools/restart_flight.sh

# 3) 비행 완료 후 PX4 ULog(.ulg)에서 분석 그래프 생성 (plots/ 폴더에 저장)
./tools/plot_tailsitter_log.py
```

---

## 3. 비행 로그 분석 결과 (`CPA = 0.65 m`, 횡방향 오차 `dY = -0.04 m`)

### 6-패널 텔레메트리 대시보드
![Tailsitter Telemetry Dashboard](plots/tailsitter_telemetry_dashboard.png)

### 3D 비행 궤적 및 평면(X-Y) 정렬 정확도
![Tailsitter 3D Trajectory](plots/tailsitter_3d_trajectory.png)
