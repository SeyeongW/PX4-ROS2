# balloon_sim 실행 안내

카메라 기반 레이싱 표식 추정과 PX4/MAVROS 경로 추종 패키지입니다. [수정 내용·검증 결과·남은 작업은 저장소 README](../README.md)에 정리했습니다.

기본 구성은 총 2kg 가정 STL 멀티콥터, 오른쪽 40m·높이 10m의 30cm 풍선, 3m 수직 이륙·3차 Bézier·가상 게이트 통과·5m 뒤 대기입니다. 사용자 락과 발사가 필요하며, 중단·실패 시 이륙점 복귀·착륙을 요청합니다.

## 현재 로컬 PC에서 실행

아래 경로는 검증에 사용한 기존 빌드입니다. 다른 PC에서는 실제 빌드 경로로 바꾸거나 `./tools/build_px4.sh`로 기본 `external/PX4-Autopilot` 빌드를 준비합니다.

```bash
cd /home/xogus/px4_ros2/.upstream/main/balloon_sim
export BALLOON_PX4_ROOT=/home/xogus/PX4-Autopilot
export BALLOON_PX4_PLUGIN_DIR=/home/xogus/ros2_rm/standalone/fixed_path_demo/.runtime_px4/gazebo_plugins
./tools/build_workspace.sh
./tools/run_tailsitter_sim.sh
```

UI에서 탐색 시작 → 좌표 확인 → 락·경로 미리보기 → 발사 순으로 사용합니다. UI/RViz 디자인과 두 카메라 관측은 유지했습니다. 영상 조향은 시간 동기화 검증 전까지 비활성화 상태입니다.

경로·운동 제한은 [`stl_pacer_quad.json`](balloon_locator/config/stl_pacer_quad.json)에서 설정합니다. 기본 게이트는 1.2m 사각 영역이며 30cm 풍선 자체와 구분합니다. 출력은 종료 시 삭제하는 임시 디렉터리를 사용합니다. `static_output:=/절대경로`를 명시하면 보존합니다.

## 별도 테일시터 실험

```bash
./tools/run_tailsitter_sim.sh mode:=pymav experimental_tailsitter:=true target_range:=90.0 target_height:=40.0 target_azimuth:=0.0 ui:=false rviz:=false
```

미측정 공력을 사용하는 이전 실험 경로입니다. 기본 멀티콥터의 검증 결과나 실기체 성능 근거로 사용하지 않습니다. 기존 `plots/` 그림도 과거 결과이며 이번 검증에서 새로 생성한 자료가 아닙니다.
