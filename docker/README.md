# Docker 시뮬레이션 환경

호스트가 Gazebo Classic 등 다른 버전을 쓰고 있어도 충돌 없이, 이 레포가 요구하는
**ROS 2 Humble + Gazebo Harmonic (gz-sim 8) + ArduPilot SITL** 스택을 컨테이너
안에서 그대로 재현합니다. 워크플로우는 README의 "4개 터미널" 방식을 그대로
유지하고, 터미널 대신 `docker exec` 셸을 여러 개 여는 것만 다릅니다.

## 사전 준비 (최초 1회)

```bash
# GUI(gz sim 창, rqt_image_view)를 컨테이너에서 띄우려면 X11 접근을 허용
xhost +local:docker
```

이 명령은 로그인마다(재부팅 시) 다시 실행해야 합니다. `~/.bashrc`에 추가해두면 편합니다.

## 빌드

```bash
cd ~/ros2_ws/PX4-ROS2
docker compose build   # ArduPilot SITL / ardupilot_gazebo / ros_gz 소스 빌드 포함 — 20~40분 소요
```

## 컨테이너 실행

```bash
docker compose up -d      # 백그라운드로 컨테이너 기동 (entrypoint는 bash 대기)
docker compose exec sim bash   # 터미널마다 이 명령으로 셸 진입
```

레포 코드는 호스트 `~/ros2_ws/PX4-ROS2`가 컨테이너 `/home/seo/ros2_ws/PX4-ROS2`에
그대로 바인드 마운트되므로, 호스트에서 코드를 수정하고 컨테이너 안에서
`colcon build`만 다시 돌리면 됩니다. `build/ install/ log/`는 `.gitignore`에 이미
포함되어 있고 컨테이너 안에서 생성돼도 호스트에 그대로 남습니다.

## 실행 순서 (README와 동일, 터미널 대신 `docker compose exec sim bash`)

```bash
# 셸 1 — Gazebo
docker compose exec sim bash -c "cd gazebo && ./run_sim.sh"

# 셸 2 — ArduPilot SITL
docker compose exec sim bash -c "sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console --map"

# 셸 3 — MAVROS + 카메라브리지 + 정밀착륙 (풀 bringup)
docker compose exec sim bash -c "ros2 launch precision_landing precision_landing.launch.py"
```

첫 실행 전에는 워크스페이스가 아직 빌드되어 있지 않으므로 셸 하나에서 먼저:

```bash
docker compose exec sim bash -c "
  python3 gazebo/gen_aruco_model.py &&
  colcon build --symlink-install
"
```

## 종료 / 정리

```bash
docker compose down          # 컨테이너 정지+삭제 (named volume은 유지됨)
docker compose down -v       # + ArduPilot/ros_gz named volume까지 삭제 (다음 up에서 이미지 내용으로 재생성)
```

## 구성 설명

| 항목 | 선택 | 이유 |
|------|------|------|
| 컨테이너 개수 | 단일 컨테이너 + `docker exec` 다중 셸 | 기존 4-터미널 워크플로우를 그대로 재현. gz-transport/MAVLink/DDS가 전부 컨테이너 내부 loopback으로 끝나서 컨테이너 간 네트워킹을 신경 쓸 필요 없음 |
| 네트워크 | `network_mode: host` | `run_sim.sh`가 이미 `GZ_IP=127.0.0.1`로 고정되어 있고, MAVProxy도 `127.0.0.1:14550`으로만 나가므로 호스트 네트워크를 그대로 쓰는 게 가장 단순 |
| 레포 마운트 | bind mount (`.:/home/seo/ros2_ws/PX4-ROS2`) | 호스트에서 에디터로 수정 → 컨테이너에서 즉시 빌드/실행 |
| ArduPilot/ardupilot_gazebo/ros_gz | 이미지 빌드 시 컴파일 + named volume | 이미지를 다시 빌드해도(Dockerfile 수정 시) 매번 20~40분씩 재컴파일하지 않도록 named volume에 보존. 완전히 새로 받고 싶으면 `docker compose down -v` |
| GPU | `deploy.resources.reservations.devices` (nvidia) | RTX 4060 패스스루로 gz-sim OGRE2 렌더링 + YOLO 추론 가속 |
| GUI | `/tmp/.X11-unix` 마운트 + `DISPLAY` 전달 | 호스트가 Wayland 세션이어도 `DISPLAY`는 XWayland 소켓을 가리키므로 그대로 동작 |

## GPU 가속 (선택사항, NVIDIA)

기본(`docker-compose.yml`)은 GPU 없이도 동작합니다 (gz sim이 소프트웨어 렌더링으로
자동 폴백 — 느리지만 동작은 함). NVIDIA GPU가 있으면:

```bash
cp docker-compose.gpu.yml.example docker-compose.override.yml
docker compose up -d --force-recreate
```

`docker-compose.override.yml`은 `docker compose`가 `docker-compose.yml`과 자동으로
합쳐 읽으므로 이후엔 `-f` 플래그 없이 평소처럼 `docker compose up -d`만 하면 됩니다.
(호스트에 `nvidia-container-toolkit` 설치되어 있어야 함. Intel+NVIDIA 하이브리드
노트북이면 이 파일의 `__GLX_VENDOR_LIBRARY_NAME` 등이 없으면 GPU가 있어도 안 씀.)

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `gz sim` 창이 안 뜸 / `cannot open display` | 호스트에서 `xhost +local:docker` 재실행 여부 확인 |
| GUI 창은 뜨는데 화면이 하얗게 나옴 | 호스트 README의 Wayland 관련 트러블슈팅과 동일 — 컨테이너 안에서 `RENDER_ENGINE=ogre ./run_sim.sh` 시도 |
| `docker compose build`에서 GPU 관련 에러 없이도 실행 시 `could not select device driver "nvidia"` | 호스트에 `nvidia-container-toolkit` 미설치/미설정 → `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| 컨테이너 안에서 `colcon build` 권한 오류 | 이미지의 UID/GID(기본 1000)가 호스트 사용자와 다르면 `docker compose build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)` |
| ArduPilot/ros_gz를 완전히 새로 빌드하고 싶음 | `docker compose down -v` 후 `docker compose build --no-cache` |
