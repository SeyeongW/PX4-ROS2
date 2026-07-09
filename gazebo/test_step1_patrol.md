# Step 1 (정찰 웨이포인트) 테스트 명령어

새 터미널 창을 열 때마다, 그 터미널에 **딱 한 줄만** 복사해서 붙여넣으면 됩니다. 각 줄이
`docker compose exec sim bash -c "..."` 형태로 컨테이너 진입 + 명령 실행을 한 번에 합니다.

## 한눈에 보기 — 몇 번 터미널에 뭘?

| 터미널 | 역할 | 언제 열어야 하나 |
|---|---|---|
| **1** | Gazebo | 제일 먼저, 완전히 뜰 때까지 기다렸다가 2번 열기 |
| **2** | SITL(ArduPilot) | 1번이 뜬 뒤에 |
| **3** | 미션 launch | 1, 2번이 둘 다 뜬 뒤에 |
| (아무거나) | 빌드 / frame_fix / task_complete | 필요할 때 잠깐 쓰고 끝나는 명령이라 새 터미널 안 열어도 됨 — 위 1~3번 중 아무 데서나(단, 1·2·3번은 명령이 이미 실행 중이라 화면을 계속 차지하고 있으니, 실제로는 빌드 끝난 직후의 터미널을 재사용하거나 4번째 터미널을 하나 더 여는 걸 추천) |

---

## 터미널 1 — Gazebo

```bash
docker compose exec sim bash -c "cd ~/ros2_ws/PX4-ROS2 && WORLD=obstacle_field VIEW=0 ./gazebo/run_sim.sh"
```
Gazebo 창이 완전히 뜨고 안정될 때까지 기다린 후 다음 터미널로.

## 터미널 2 — SITL

```bash
docker compose exec sim bash -c "mkfifo /tmp/mav_cmd.fifo 2>/dev/null; ( while true; do cat /tmp/mav_cmd.fifo; done ) | sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --no-rebuild -w"
```
부팅 로그를 봐서 `AP: Frame: UNSUPPORTED`가 보이면, **새 터미널을 열지 말고** 이미 끝난
"빌드용" 터미널(또는 새로 4번째 터미널 하나)에서:
```bash
docker compose exec sim bash -c "echo 'param set FRAME_CLASS 1' > /tmp/mav_cmd.fifo; echo 'param set FRAME_TYPE 1' > /tmp/mav_cmd.fifo"
```

## 터미널 3 — 미션 launch

**정찰 모드**(신규 기능 — 6개 웨이포인트를 도는지 확인. `loiter_s`를 짧게 줘서 여러 바퀴
빨리 확인. `mission_area_e/n:=0.0`+`trigger_dist:=10.0`을 꼭 넣어야 함 — 안 넣으면 launch
기본값 `(120,40)`이 쓰이는데, obstacle_field 트레일러는 원점 반경 ~30m 안에서만 움직여서
`trigger_dist`에 영원히 안 들어가 드론이 이륙조차 못 함):
```bash
docker compose exec sim bash -c "cd ~/ros2_ws/PX4-ROS2 && ros2 launch precision_landing truck_mission.launch.py apf_enable:=true obstacle_map:=\$(pwd)/gazebo/config/obstacle_map.yaml patrol_route:=\$(pwd)/gazebo/config/patrol_route.yaml loiter_s:=5.0 mission_area_e:=0.0 mission_area_n:=0.0 trigger_dist:=10.0"
```

**레거시 회귀 테스트**(정찰모드 안 켜도 예전 단일 웨이포인트 동작이 그대로인지 확인할 때만
— 정찰 모드 테스트와 같은 세션에서 같이 할 필요는 없고, 나중에 따로 한 번 돌려도 됨).
**주의: 정찰 모드가 이미 돌고 있으면 먼저 터미널 3에서 `Ctrl+C`로 종료한 뒤** 같은
터미널에서 아래 명령을 실행할 것 — `mission_manager_node`는 한 번에 하나만 떠서 기체를
제어하므로, 정찰용 launch를 안 끄고 새 launch를 띄우면 두 노드가 동시에 setpoint를
쏴서 충돌함 (Gazebo·SITL은 그대로 둬도 됨):
```bash
docker compose exec sim bash -c "cd ~/ros2_ws/PX4-ROS2 && ros2 launch precision_landing truck_mission.launch.py apf_enable:=true obstacle_map:=\$(pwd)/gazebo/config/obstacle_map.yaml mission_area_e:=20.0 mission_area_n:=16.0"
```

이 터미널 화면에 `mission_manager_node`의 로그(`arrived at waypoint N`, `advance to
waypoint N`, `RETURN (...)`)가 그대로 찍히므로 순찰 상태를 보는 용도로는 이 터미널 하나면
충분합니다.

---

## 필요할 때만 잠깐 — 새 터미널 안 열어도 됨

**빌드** (소스/launch 파일을 수정했을 때 1번 터미널 열기 전에 먼저):
```bash
docker compose exec sim bash -c "cd ~/ros2_ws/PX4-ROS2 && colcon build --symlink-install --packages-select precision_landing"
```

**순찰 중 강제 종료 신호** (터미널 3이 정찰 모드로 돌고 있는 동안 아무 때나):
```bash
docker compose exec sim bash -c "ros2 topic pub /mission/task_complete std_msgs/msg/Bool '{data: true}' --once"
```
이 신호는 드론을 바로 착륙시키는 게 아니라 `mission_manager_node` → `precision_landing_node`로
제어권을 넘기는 신호일 뿐입니다(`RETURN (task complete) -> LANDING handoff` 로그 이후
`ALIGN → DESCEND → FINAL` 단계로 실제 하강/정렬을 시작, 수 초~수십 초 소요).

**⚠️ 신호를 보낸 직후 터미널 3에서 바로 `Ctrl+C` 하지 말 것.** `truck_mission.launch.py`
하나에 두 노드가 같이 떠 있어서, `precision_landing_node`가 실제 하강 속도를 명령하는
도중에 launch를 죽이면 두 노드가 동시에 죽어 setpoint 스트림이 끊기고, ArduPilot GUIDED가
마지막 하강 명령을 몇 초간 그대로 유지하면서 추락한 사례가 있음(2026-07-09 재현됨).
터미널 3 로그에 아래 줄(정상 disarm, 노드가 스스로 종료)이 뜰 때까지 기다린 뒤에만 끌 것:
```
disarmed onto platform -> shutting down node
```

**사고 위치 사후분석용 위치 로그** (필요할 때만, 별도 터미널 하나 열어서):
```bash
docker compose exec sim bash -c "ros2 topic echo /mavros/local_position/pose --field pose.position"
```
