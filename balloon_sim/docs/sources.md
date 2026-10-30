# 구성 요소·출처·배포 범위

## 포함한 소스

| 경로 | 역할 | 출처 / 라이선스 |
|---|---|---|
| `balloon_locator/` | 카메라 검출·좌표 추정·짐벌·STL 물리 모델·미션 | [PX4-ROS2 balloon_locator](https://github.com/SeyeongW/PX4-ROS2/tree/main/balloon_locator) 기반, 로컬 수정; MIT (아래 예외 확인) |
| `support/jol_perception/` | 기존 Qt UI·3D 경로·Bézier·MAVROS 어댑터 | 기존 로컬 ros2_nub 소스 발췌; 패키지 Apache-2.0 표기 유지 |
| `support/jol_sim/` | Gazebo 짐벌/평가 plugin·world 생성 helper | 기존 로컬 ros2_nub 소스 발췌; 패키지 Apache-2.0 표기 유지 |
| `balloon_locator/meshes/stl_airframe/` | 사용자 제공 기체 STL 10개 | 사용자 제공 자산. 코드 라이선스가 이 STL의 별도 재배포 권리를 자동 부여하지 않음 |
| `tools/` | 독립 빌드·실행·임시 출력 관리 | 본 통합 코드 |

원본 로컬 px4_ros2·ros2_nub 소스는 그대로 두고 공개 복사본의 경로·패키지 설치 구성만 조정했다. LiDAR·Intel D435·매니퓰레이터 외형, scan CSV, 과거 대용량 모델, 녹화·결과·빌드 바이너리는 이번 폴더에 포함하지 않는다. 공유 Python 코드에는 과거 기능 정의가 남아 있어도 기본 launch는 STL/PACER 카메라 단안 시나리오만 지원한다.

`balloon_locator/balloon_locator/stl_mission.py`는 Apache-2.0인 기존 NUB `static_flight.py`를 기반으로 한 수정본이므로 Apache-2.0 적용 대상이다. 이 파일은 MIT 패키지 폴더에 위치하지만 원래 라이선스·출처 표기를 유지한다.

코드 MIT 전문은 [LICENSE](../LICENSE), 지원 패키지 Apache-2.0 전문은 [LICENSE-APACHE-2.0](../LICENSE-APACHE-2.0)을 확인한다. 원본 저작권/라이선스 헤더를 유지했다.

## 외부 의존성 및 공식 참고

- [PX4 v1.17.0](https://github.com/PX4/PX4-Autopilot/tree/d6f12ad1c4f70ad3230afd7d86e971421e02fef4) — BSD-3-Clause, 제어기·EKF·SITL. 저장소에는 빌드 스크립트/고정 revision만 포함한다.
- [PX4 Gazebo Classic](https://github.com/PX4/PX4-SITL_gazebo-classic/tree/6697ab169ceab512dc706acea63df4c882662c60) — 모터 및 센서 plugin. 해당 저장소 라이선스 준수.
- [PX4 Classic 공식 설치/실행 문서](https://docs.px4.io/main/en/sim_gazebo_classic/) — 이번 통합은 Classic 11이며 새 Gazebo와 구별한다.
- [MAVROS ROS 2](https://github.com/mavlink/mavros) — ROS ENU/FLU ↔ MAVLink NED/FRD 변환 및 FCU 통신.
- [T-MOTOR PACER P2207 V3](https://www.tmotorhobby.com/goods-1181-T-MOTORHOBBY%2BP2207%2BV3.html), [공식 치수/질량표](https://www.tmotorhobby.com/images/20220411/1649663656669244.jpg), [공식 추력표](https://www.tmotorhobby.com/images/20220411/1649663656357709.jpg) — 1950 KV, 6S, P49436-3 조건. 추력 자료의 적합값과 가정 토크/시간상수를 구분한다.

외부 문서 조회는 Context7과 공식 소스 확인을 함께 사용했다. Context7 결과의 `main` 문서를 특정 버전의 실행 증거로 취급하지 않고, 설치된 소스와 고정 revision을 대조했다.

## 노즈 카메라·내부 배치 추가의 근거

- [Camera Module 3 표준형 공식 기구 도면](https://datasheets.raspberrypi.com/camera/camera-module-3-standard-mechanical-drawing.pdf): PCB·렌즈·커넥터 치수와 수평 66° 표기를 참고했다. [공식 제품 페이지](https://www.raspberrypi.com/products/camera-module-3/)의 대각 화각 75°와 수평 화각을 구별한다.
- 내부 전장품의 순서·위치는 사용자 제공 단면도를 참고했다. 어댑터 레일·배선·광학 중심·부품 상세 치수는 시뮬레이션 배치 가정이다. 원본 STL 10개는 변경하지 않았으며 공식 PDF·추가 이미지·렌더링 결과를 저장소에 복제하지 않았다.
- 640×360·15 Hz는 이 시뮬레이션 설정이며 제품 최대 성능을 뜻하지 않는다. 자동초점·왜곡·노출·롤링셔터와 기구 공차는 재현/검증하지 않았다.
