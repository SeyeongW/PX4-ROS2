# 설치·빌드·실행 명령

모든 명령은 `PX4-ROS2/balloon_sim`에서 실행한다. 루트의 다른 착륙 프로젝트와 별도 워크스페이스다. 기본 실행은 Gazebo/PX4 SITL이며 실기체 장치나 SIYI 카메라에 연결하지 않는다.

## 1. 환경

Ubuntu 22.04 / ROS 2 Humble / Gazebo **Classic 11**을 사용한다. Gazebo Harmonic의 `gz sim`과 다른 실행 환경이다. 로컬 확인 버전은 Gazebo 11.14.0, gazebo_ros 3.9.0, MAVROS 2.14.0이다. Classic은 현재 커뮤니티 지원 대상이며 새로운 Gazebo로의 이식은 이 게시 작업에 포함하지 않았다. [PX4 공식 Classic 안내](https://docs.px4.io/main/en/sim_gazebo_classic/).

ROS 2 Humble 설치가 끝난 Ubuntu 22.04에서 필요한 주요 패키지:

```bash
sudo apt update
sudo apt install gazebo libgazebo-dev ros-humble-gazebo-ros-pkgs \
  ros-humble-mavros ros-humble-mavros-extras ros-humble-mavros-msgs \
  ros-humble-vision-msgs ros-humble-tf2-ros ros-humble-cv-bridge \
  ros-humble-sensor-msgs-py ros-humble-rviz2 ros-humble-rosbag2-py \
  python3-colcon-common-extensions python3-rosdep python3-pytest \
  python3-numpy python3-scipy python3-opencv python3-yaml python3-jinja2 \
  python3-pyqt5 python3-matplotlib python3-pip \
  build-essential cmake ninja-build git pkg-config \
  libboost-system-dev libboost-thread-dev libboost-filesystem-dev \
  libopencv-dev libtinyxml-dev libprotobuf-dev protobuf-compiler libeigen3-dev \
  qtbase5-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
python3 -m pip install --user pymavlink
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
```

시스템 의존성 설치는 최초 한 번 인터넷이 필요하다. 위 명령은 설치 안내이며 이 문서 게시 작업에서 새 PC 설치를 실제 수행한 것은 아니다. 실행 중에는 네트워크 namespace를 분리하고 Gazebo 온라인 모델 다운로드를 끈다.

## 2. 코드와 PX4 준비

```bash
git clone --branch cho --single-branch https://github.com/SeyeongW/PX4-ROS2.git
cd PX4-ROS2/balloon_sim
mkdir -p external
git clone --branch v1.17.0 --single-branch \
  https://github.com/PX4/PX4-Autopilot.git external/PX4-Autopilot
bash external/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools
./tools/build_px4.sh
./tools/build_workspace.sh
```

PX4 개발 도구가 이미 준비되어 있으면 `mkdir`·수동 clone·`ubuntu.sh`를 생략하고 `./tools/build_px4.sh`부터 실행한다. 이 스크립트가 기본 위치에 PX4를 clone하고 필요한 submodule과 SITL/Classic plugin을 빌드한다. 기존 checkout을 강제로 reset하지 않으며 revision 불일치·기존 추적 파일 수정은 오류로 중단한다.

| 의존성 | 고정 revision |
|---|---|
| PX4 v1.17.0 | `d6f12ad1c4f70ad3230afd7d86e971421e02fef4` |
| sitl_gazebo-classic | `6697ab169ceab512dc706acea63df4c882662c60` |

고정 Classic CMake가 Qt의 `QT5_WRAP_CPP`를 항상 호출하므로 recipe는 `BUILD_GSTREAMER_PLUGIN=ON`으로 configure한다. Qt/GStreamer 개발 패키지가 필요하며 실제 빌드 target은 모터·센서·MAVLink plugin으로 제한한다. 이 configure 단계는 별도 임시 폴더에서 통과했다.

기본 빌드 병렬도는 2다. 메모리가 충분할 때만 `BALLOON_BUILD_JOBS=4 ./tools/build_px4.sh`처럼 변경한다. 외부 의존성은 `external/`, ROS 산출물은 `build/`, `install/`, `log/`에 생성되며 Git에 포함하지 않는다.

**새로운 clean upstream PX4 전체 빌드와 비행은 아직 검증하지 않았다.** 이번 공개 구조의 ROS 빌드·실행 점검은 기존에 설치된 PX4 바이너리/Classic plugin을 명시적으로 재사용했다. 이전 전체 비행은 사용자 수정이 포함된 바이너리를 사용했으므로 clean v1.17 결과와 동일하다고 간주하지 않는다.

## 3. 시뮬레이션 실행

```bash
./tools/run_gazebo.sh
```

이 명령이 Gazebo, PX4 SITL, MAVROS, 카메라 검출·좌표 추정, 미션, Qt UI, RViz를 실행한다. UI·RViz의 기존 디자인은 유지했다.

```bash
# GUI 없는 프로세스·센서 점검
./tools/run_gazebo.sh gui:=false ui:=false rviz:=false

# 목표 거리·높이 변경 예시 (기본값과 동일)
./tools/run_gazebo.sh target_range:=40 target_height:=10 target_azimuth:=-90
```

1. UI에서 탐지를 시작한다. 미검출 시 짐벌은 yaw 탐색을 한다.
2. 좌표/카메라 상태가 유효하면 락한다. 락이 자동 경로 미리보기를 요청한다.
3. FCU 자기 위치 기준 정합과 경로 검사가 끝나 발사 버튼이 활성화될 때 발사한다.
4. 기체는 자기 출발점에서 3 m 수직 상승하고 곡선을 따라 센서 락 목표 좌표에 도달해 대기한다.
5. 비행 중단은 현재 위치에서 착륙을 요청한다. 프로그램 종료는 `Ctrl+C`다.

락·발사 조건이 안 맞으면 UI 이유를 확인한다. 준비 검사를 우회하거나 Gazebo 정답 좌표를 대신 주입하지 않는다. 기본은 가상 목표점 도달이며 게이트 통과/이륙점 귀환 임무가 아니다.

### 기존 외부 빌드 재사용

소스 위치와 바이너리의 버전·수정 내역을 알고 있을 때만 명시한다. 개인 경로는 기본값에 내장하지 않았다.

```bash
export BALLOON_PX4_ROOT=/absolute/path/to/PX4-Autopilot
export BALLOON_PX4_BINARY=/absolute/path/to/px4
export BALLOON_PX4_PLUGIN_DIR=/absolute/path/to/gazebo_plugins
./tools/run_gazebo.sh
```

PX4 root에는 `build/px4_sitl_default/etc`와 Classic 모델 템플릿이 필요하다. plugin 디렉터리는 모터·IMU·GPS·MAVLink 등 동적 라이브러리를 포함해야 한다. 실행 스크립트는 `balloon_locator`, `jol_sim`, `jol_perception`이 이 폴더의 `install`에서 선택되었는지도 확인한다.

### 로그와 저장

기본 실행은 개인 임시 디렉터리에 데이터를 두고 프로세스 종료 후 제거한다. 기록 보존이 필요한 실험에만 직접 경로를 지정한다.

```bash
# 사용자가 기록 보존을 원하는 경우에만 실행
./tools/run_gazebo.sh static_output:="$PWD/saved_trial"
```

새 셸에서 같은 ROS_DOMAIN_ID만 지정하면 네트워크 namespace가 달라 실행 중 토픽이 보이지 않을 수 있다. `--exec`는 별도의 독립 점검 namespace를 만드는 기능이며 기존 실행에 접속하는 명령이 아니다. 실시간 상태는 함께 실행된 UI와 RViz에서 확인한다.

## 4. 검사

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$PWD/balloon_locator:$PYTHONPATH" \
  python3 -m pytest balloon_locator/test -q -p no:cacheprovider
bash -n tools/run_gazebo.sh tools/build_workspace.sh tools/build_px4.sh
```

외부 PX4가 없으면 해당 모델 생성 테스트는 skip될 수 있으므로 통과 숫자와 skip 숫자를 같이 확인한다. 테스트 통과는 비행 성공의 대체 증거가 아니다.

## 5. 문제 확인

- `Build first`: `./tools/build_workspace.sh` 실행 및 결과 확인.
- 다른 workspace 패키지가 선택됨: 새 터미널에서 실행하고 오래된 overlay source를 제거.
- `unshare: Operation not permitted`: 현재 Linux의 unprivileged user/network namespace 정책 확인. 격리를 자동 해제하지 않는다.
- 모델/plugin 없음: `./tools/build_px4.sh` 성공 여부 또는 세 환경변수 경로 확인.
- MAVROS GeographicLib 오류: 위 데이터 설치 스크립트 확인.
- 준비/락 이후 발사 불가: UI 사유, 좌표 freshness, FCU ready, landed/arming 상태 확인.
- offscreen UI: `QT_QPA_PLATFORM=offscreen`은 자동 점검용이며 화면을 띄우지 않는다.
