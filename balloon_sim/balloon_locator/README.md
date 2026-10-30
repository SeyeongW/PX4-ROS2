# balloon_locator 실물 준비·운용

ROS2 Humble 센서 지상국 패키지다. 실물 경로는 카메라 → HSV 검출 → 짐벌 중심 정렬 →
알려진 풍선 크기와 보정된 카메라·자세를 이용한 위치 추정이다.
**현재 실물 지상국에는 UI 좌표 락·자동 탐색·3m Bézier 비행 제어가 연결되어 있지 않다.**
이 흐름은 Gazebo에서 별도로 검증했으며 실물 통합 완료로 표현하지 않는다.

## 빌드와 의존성

실물 컴퓨터에는 `balloon_locator/` 소스 폴더 전체를 가져간다. PC의 `build/`,
`install/`, NUB 설치본을 복사하지 않는다. 깨끗한 ROS Humble 터미널에서:

```bash
source /opt/ros/humble/setup.bash
# 현재 작업 디렉터리에 balloon_locator 소스 폴더가 있을 때
rosdep install --from-paths balloon_locator --ignore-src --rosdistro humble \
  --skip-keys 'jol_sim jol_perception rviz2'
colcon build --base-paths balloon_locator --packages-select balloon_locator --symlink-install
source install/local_setup.bash
```

세 skip-key는 선택형 로컬 Gazebo 의존성이다. 실물 센서 노드는 NUB를 import하지 않는다.
NumPy/OpenCV/PyYAML/pymavlink/ROS 메시지·tf2는 실제 사용 의존성이다.

## 장비 연결 전 설정 검사

```bash
ros2 run balloon_locator hardware_check --config balloon_locator/config/balloon_locator.yaml
ros2 run balloon_locator hardware_check --config balloon_locator/config/balloon_locator.yaml --bridge
```

검사는 카메라·시리얼·네트워크·ROS 노드를 열지 않는다. `configuration_valid`는 설정 구조,
`sensor_estimation_configured`는 보정/불확실도 준비 조건을 뜻한다. 장비 연결과 실제 교정,
비행 적합성은 인증하지 않는다. 기본 파일은 미보정이므로 종료코드2가 정상적인 결과다.
`--range-m 40`은 광학 예산을 위한 카메라→목표 거리 가정이다. 수평40m·상향 표적의
실제 사선거리보다 짧은 낙관적 하한이며 좌표 제어 입력이 아니다.

| 설정 | 실측·확인 내용 |
|---|---|
| `calibration_file` | 표준 ROS camera_calibration YAML의 K/D/R/P·영상크기. 실제 파일을 지정하며 예제행렬은 제공하지 않음 |
| `calibration_zoom` | 그 보정을 촬영한 고정 줌. 최근 `/gimbal/zoom`이 일치해야 보정 영상 발행 |
| `image_width/height` | locator·tracker·보정파일·실제 영상의 크기가 일치해야 함 |
| `latency_sec`, `attitude_latency_sec` | 영상 decode 완료/자세 수신과 실제 측정 시각의 차이. 기본0은 측정값이 아님 |
| `balloon_diameter_m`와 오차 | 실제 지름, 타원·변형·픽셀 지름 편향 포함. 공통 오차는 평균으로 줄지 않음 |
| `base_heading_deg`, 각 축 sign/zero | 북 기준 설치 방위와 SIYI 축 부호·영점. yaw_sign=+1은 기존 미측정값 |
| `calibration_valid` | 위 설치·광학·시각 검증 완료 후에만 true. 기본false 유지 |
| `cam_lat/lon`, `cam_height_above_home_m` | 카메라 회전중심의 실측 GPS, 드론 홈 지면 대비 높이. 기본NaN |

카메라는 단일 카메라·항등 R·plumb_bob/rational_polynomial을 지원한다. 왜곡보정한 영상과
유효 K·D=0의 CameraInfo를 동일 시각/프레임으로 발행한다. 출력 토픽 이름은 기존
`/camera/image_raw`를 유지하므로 보정파일 사용 시 이미 보정된 영상임을 주의한다.
줌 변경에 대한 다중 교정표는 지원하지 않는다. RTSP stamp는 **decode 완료−실측 latency의
추정값**이며 센서 노출시각을 직접 읽은 값이 아니다. FFmpeg 내부 버퍼 지연도 실측해야 한다.

기본1920px·80°·30cm·40m는 약8.58px 표적이며 현재 오차 모델의 공통 거리σ 하한은 약2.35m다.
좌표 허용σ0.5m를 만족하지 못한다. 게이트를 느슨하게 바꾸지 말고 줌별 교정과 크기/픽셀
편향을 측정해야 한다. `require_camera_info=true`로 명목 FOV 좌표 생성을 차단한다.

## 센서 지상국 실행

측정 설정을 준비하고 실제 센서 연결을 의도할 때 다음 명령을 사용한다.
이번 코드 검사에서는 이 실물 실행을 하지 않았다.

```bash
ros2 launch balloon_locator ground_station.launch.py config:=/절대/경로/balloon_locator.yaml
```

카메라·짐벌·검출·tracker·locator 다섯 노드만 실행한다. 설정오류는 노드 생성 전에 거부하고,
필수 노드 종료 시 지상국 전체를 종료한다. 미보정 `calibration_valid=false` 상태는 영상 벤치
확인을 허용하지만 좌표는 발행하지 않는다. 현재 실물 tracker는 **시야 안에서 검출한 표적의
중심 정렬**만 하며 시야 밖 표적의 자동 탐색은 구현하지 않았다.

`/balloon/position_ned`와 `/balloon/estimate`는 보정·표본수·불확실도·원본시각 조건을 통과한
좌표만 발행한다. `/balloon/status`에는 준비 여부와 거부 이유를 발행한다. 표적을 옮기거나
실험 조건을 바꾸면 `/balloon/reset`을 호출한다.

좌표는 NED(+북,+동,+아래)다. SIYI 회전은 312 순서 `Rz(yaw)Rx(roll)Ry(pitch)`다.
ArduPilot은 raw yaw를 반전하므로 설치 축 검증 없이 현재 부호를 실측값으로 사용하지 않는다.
카메라 NED와 PX4 EKF local NED의 원점/기준방향은 같다고 가정하지 않는다.

## 기체 탑재 스크립트 경계

`onboard/`는 원본 계열 실험 경로로 보존했다. 실행에 필요한 `common.py`, `guidance.py`,
검출 코드가 있으므로 파일 하나만 따로 복사하지 않는다. ROS 없이 사용하지만 numpy/OpenCV/
pymavlink 등 import하는 의존성은 필요하다.

- `offboard_follower.py`: 공중에서 수신 위경도/홈 상대고도 목표를 따르는 별도 방식.
- `interceptor.py`와 `run.sh`: 원본 PN/자동ARM·5m 이륙 실험. **UI의3m/Bézier 방식이 아니다.**
- `detector.py`의 UDP 소비자는 `interceptor.py`다. ROS detector와의 중복은 standalone 실행
  목적이 있으므로 미사용 파일로 삭제하지 않았다.

기본 실행은 인자검사 후 종료하며 장치/socket/로그를 열지 않는다. `--execute`를 명시하면
기존 실물 제어 경로가 활성화되고 interceptor는 유효 목표 수신 후 자동ARM할 수 있다.
이 옵션은 비행 준비 인증이 아니다. 자동실행 서비스로 등록하지 않는다.

```bash
python3 balloon_locator/onboard/offboard_follower.py --help
balloon_locator/onboard/run.sh  # 오프라인 인자검사만
```

FCU heartbeat 제한시간·주소/품질 검사·새 telemetry·setpoint 선행 스트림을 적용했다.
ACK 수락과 실제 상태 전환은 따로 확인한다. ACK는3초, ARM/모드는5초, 이륙 상태는30초
내에 확인하지 못하면 종료하며 같은 요청을 무한 재전송하지 않는다. interceptor는 유효한
FCU HOME_POSITION을 받기 전에는 ARM/이륙하지 않는다. 이륙 명령의 고도는
`HOME 해발고도 + 상대 상승고도`로 전송하고, 실제 상승 확인에는 홈 상대고도를 사용한다.
HOME에 유효한 시각이 있으면 중복·후퇴를 거부한다. 제어 시작 뒤 HOME 위경도/고도가
바뀌면 종료하므로, FCU가 ARM 시 HOME을 갱신하는 경우도 원점을 확인하고 재준비해야 한다.
기존16바이트 비전 UDP/COMMAND_INT에는 원본 노출시각·순번이 없으므로 네트워크
재전송을 완전히 검증하지 못한다. MAVLink1 HOME에는 확장 timestamp가 없을 수 있다.
watchdog 종료는 FCU 비무장·착륙 성공을 보장하지 않으며 failsafe/조종권 인계를 확인해야 한다.
새로운 실물 프로토콜/비행 미션 통합은 별도 검증이 필요하다.
명시한 `--log`만 보존하며 기본 legacy launcher 출력은 개인 임시 폴더에서 종료 후 제거한다.

## 검증과 출처

합성 영상·가짜 캡처/FCU로 오류·종료·좌표 계약을 검증한다. 실제 SIYI 연결, 모뎀,
GPS/시각 정합, 물성 식별과 실물 비행은 미검증이다. 최신 시험·Gazebo·미달 항목은
옵시디언의 `04 카메라 풍선 위치추정 - PX4 ROS2` 노트에서 확인한다.

- [ROS 카메라 YAML 형식](https://github.com/ros-perception/image_common/blob/humble/camera_calibration_parsers/src/parse_yml.cpp)
- [OpenCV 4.5.4 FFmpeg timeout](https://github.com/opencv/opencv/blob/4.5.4/modules/videoio/include/opencv2/videoio.hpp)
- [ROS Image 행 stride와 CameraInfo](https://github.com/ros2/common_interfaces/tree/humble/sensor_msgs/msg)
- [ArduPilot SIYI 312 구현](https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Mount/AP_Mount_Siyi.cpp)
- [PX4 v1.17 Offboard](https://docs.px4.io/v1.17/en/flight_modes/offboard)
- [MAVLink HOME_POSITION 고도 단위](https://mavlink.io/en/messages/common.html#HOME_POSITION)
- [PX4 v1.17 이륙 해발고도 처리](https://github.com/PX4/PX4-Autopilot/blob/v1.17.0/src/modules/navigator/navigator_main.cpp)
