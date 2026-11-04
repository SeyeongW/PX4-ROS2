# balloon_locator

SIYI A8 mini 짐벌 카메라로 **정지한 풍선**을 탐지하고, 지상국 기준 **NED 좌표(+불확실도)** 를 추정하는 ROS2 Humble 패키지.
기존 `pantilt_system`의 "탐지 -> 추적 -> 위치추정" 구조를 지상 고정 + SIYI 짐벌 + 정지 목표 가정으로 다시 짠 것이다.

```
siyi_camera_node ──/camera/image_raw──> balloon_detector_node ──/balloon/detections──┬─> gimbal_tracker_node ──/gimbal/cmd_rate──┐
siyi_gimbal_node ──/gimbal/attitude, /gimbal/zoom──────────────────────────────────┐ │                                           │
        ▲ UDP SDK                                                                    └─┴─> balloon_locator_node ──> /balloon/position_ned, /balloon/estimate, /balloon/status
        └────────────────────────────────────── (속도/줌 명령) ───────────────────────────────────────────────────────────────┘
```

## 범위
- 포함: 짐벌/카메라 드라이버(SIYI UDP+RTSP), HSV 풍선 검출, 짐벌 자동 추적, 정지 목표 위치 추정(지름 기반 거리 + 짐벌 각도).
- 드론 연동(실험): `mavlink_bridge_node`(지상 PC) -> 텔레메트리 모뎀 -> `onboard/offboard_follower.py`(드론 Pi) -> PX4 OFFBOARD. 아래 "드론 연동" 참고.
- LiDAR(Unitree L1)는 아직 연결하지 않았다. 풍선은 반사 점이 적을 가능성이 커서 카메라 지름 기반 거리가 주 소스.

## 좌표 규약 (이 패키지 전용, 레거시 pantilt_system의 부호 규약과 별개)
- NED: x=북, y=동, z=하(+). 방위각은 북 기준 시계방향 +, 고각은 위 +.
- 위치 = `offset_ned + 거리 × 시선벡터`. `offset_ned`는 카메라 회전중심의 NED 위치(원점을 카메라로 잡으면 0).
- 거리 = `풍선지름 × 초점거리(px) / 검출지름(px)`, 초점거리는 `hfov_deg_at_1x`와 줌 배율로 계산.

## 현장에서 반드시 교체할 값 (config/balloon_locator.yaml, 전부 TODO 표시)
| 값 | 이유 |
|---|---|
| `hfov_deg_at_1x` | 임시 80°. 거리 오차에 직접 비례. 데이터시트 또는 체커보드 캘리브레이션 |
| `balloon_diameter_m` | 임시 0.30 m. 실제 지름 |
| `base_heading_deg` | 짐벌 yaw=0이 가리키는 방위각. 컴퍼스/기준점으로 실측 |
| `yaw_sign`, `pitch_sign`, `*_zero_deg` | SIYI 보고값의 부호/영점은 실기에서 확인. 틀리면 좌우/상하가 뒤집힘 |
| `gimbal_tracker_node`의 `yaw_sign`, `pitch_sign` | 풍선이 오른쪽일 때 짐벌이 오른쪽으로 도는지 확인 |
| `latency_sec`, `attitude_latency_sec` | 영상/자세 지연. 짐벌이 움직일 때 좌표가 틀어지면 이 값을 조정 |
| `hsv_ranges` | 풍선 색 (기본 빨강) |

## SIYI 프로토콜 주의
`siyi_protocol.py`의 명령 ID/페이로드(0x0D 자세, 0x07 회전, 0x0F 절대 줌, 0x18 현재 줌, 0x08 센터)는 SDK 문서 기준 기억에 의존한다.
프레임/CRC는 SDK 문서의 예시 프레임으로 테스트했지만, **줌 관련 명령(0x0F, 0x18)은 A8 mini 펌웨어에서 실제로 동작하는지 첫 연결 때 확인**할 것.

## 빌드 / 실행 / 테스트
```bash
cd ~/colcon_ws/src && ln -s ~/workspace/balloon_locator .
cd ~/colcon_ws && colcon build --packages-select balloon_locator --symlink-install && source install/setup.bash
ros2 launch balloon_locator ground_station.launch.py
ros2 topic echo /balloon/status           # n, sigma_m
ros2 service call /balloon/reset std_srvs/srv/Trigger   # 풍선을 옮겼을 때

cd ~/workspace/balloon_locator && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q test
```
`/balloon/position_ned`는 첫 샘플부터 계속 발행된다(준비 상태 판정 없음). 신뢰도는 `/balloon/status`의 `n`, `sigma_m`으로 직접 판단한다.

## 추정기 동작
`static_estimator.py`: 측정마다 (거리 방향 / 직교 방향) 공분산을 분리해 가중 평균을 누적한다. 마할라노비스 게이팅으로 이상치를 거부하고,
연속 거부가 15회 이상이면 풍선이 움직인 것으로 보고 초기화한다. 바람에 천천히 떠내려가면 `forget`(<1)을 주면 최근 샘플 비중이 커진다.
평균은 무작위 오차만 줄인다. 방위각/지름/FOV 같은 **체계 오차는 줄지 않으므로** 위 표의 값이 정확도를 결정한다.

## 드론 연동
```
balloon_locator_node ─/balloon/position_ned, /balloon/status─> mavlink_bridge_node ─SiK─> (mavlink-router) ─> offboard_follower.py ─> PX4
```
- 왜 위경도로 보내나: 지상 NED 원점(카메라)과 PX4 local NED 원점(EKF 원점, 비행마다 바뀔 수 있음)이 다르다.
  bridge가 카메라 GPS 실측 위치로 위경도 + **드론 홈 지면 기준 상대고도**로 바꾸고, Pi는 `SET_POSITION_TARGET_GLOBAL_INT`(`GLOBAL_RELATIVE_ALT_INT`)로 보낸다.
  고도를 AMSL로 쓰지 않는 이유: GPS 두 대의 AMSL 차이가 수 m라서.
- 메시지: `COMMAND_INT`, `MAV_CMD_USER_1`, 대상 sys 1 / comp 191. x/y = 위경도×1e7, z = 상대고도[m], param1 = n, param2 = sigma_m.
- bridge 게이트: `/balloon/status`의 `n >= min_samples` 이고 `sigma_m <= max_sigma_m` 일 때만 송신. `frame_id != local_ned`면 무시.
- Pi 안전장치: 공중일 때만 OFFBOARD 요청(arm/이륙 안 함), 홈 기준 수평거리/고도 범위 밖 거부, 10 m 이상 점프는 10회 연속일 때만 채택,
  1 s 끊김 -> 그 자리 정지, 5 s 끊김 -> Hold, OFFBOARD에서 빠지면 종료(다시 뺏지 않음). 시간은 전부 `time.monotonic()`.
- 풍선 중심 그대로가 setpoint다. 풍선에 닿지 않으려면 `--alt-offset`(위로 m)을 준다.
- 모뎀이 FC TELEM 포트에 붙어 있으면 두 MAVLink 인스턴스 모두 `MAV_x_FORWARD=1`이어야 Pi까지 전달된다.
- bridge가 `/dev/ttyUSB0`를 점유하므로 같은 모뎀으로 QGC를 쓰려면 PC에서도 mavlink-router로 나눠야 한다.

```bash
pip install pymavlink                     # PC, Pi 둘 다
# 지상 PC (yaml의 cam_* 를 먼저 채울 것)
ros2 run balloon_locator mavlink_bridge_node --ros-args --params-file ~/workspace/balloon_locator/config/balloon_locator.yaml
# 드론 Pi (이 파일만 복사하면 됨, ROS 불필요)
python3 offboard_follower.py --max-dist 150 --min-alt 5 --max-alt 60 --alt-offset 3
```
검증 범위: 순수 함수 단위 테스트(`test/test_geo.py`)와 가짜 FCU 루프백(표적 수신 -> OFFBOARD 요청 -> 끊김 시 Hold -> 종료)까지.
**SITL(PX4 Gazebo)에서 `SET_POSITION_TARGET_GLOBAL_INT` 동작을 먼저 확인하고 실기로 갈 것.**

## 아직 검증하지 않은 것
- 실제 A8 mini와의 통신/영상 (이 환경에는 장비가 없음). 순수 로직(프로토콜, 기하, 추정기, 검출기)만 단위 테스트(22개)로 확인.
- ROS 노드의 실행(토픽 연결)은 import 확인까지만 했다.
- 자동 줌 제어는 없다 (`/gimbal/cmd_zoom`으로 수동 지정).
