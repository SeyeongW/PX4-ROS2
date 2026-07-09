# PX4-ROS2 시뮬레이션 환경 (ROS 2 Humble + Gazebo Harmonic + ArduPilot SITL)
#
# 호스트가 Gazebo Classic 등 다른 버전을 쓰고 있어도 충돌 없이 이 워크스페이스가
# 요구하는 Gazebo Harmonic(gz-sim 8) 스택을 그대로 재현하기 위한 이미지입니다.
#
# 레포(README.md) 5단계 설치 절차를 그대로 이미지 빌드 단계로 옮긴 것뿐이라,
# 실행 방법(4개 터미널 = 4개 `docker exec` 셸)은 README와 동일합니다.

FROM ubuntu:22.04

ARG USERNAME=seo
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=humble

# ---------------------------------------------------------------------------
# 1. 기본 도구 + locale
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      locales curl gnupg2 lsb-release software-properties-common \
      ca-certificates sudo git wget build-essential cmake ninja-build \
      rapidjson-dev python3-pip python3-dev tzdata \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. ROS 2 Humble
# ---------------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-base ros-humble-cv-bridge ros-humble-image-transport \
      python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/* \
    && rosdep init

# ---------------------------------------------------------------------------
# 3. MAVROS
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-mavros ros-humble-mavros-msgs ros-humble-mavros-extras \
      ros-humble-rqt-image-view \
    && rm -rf /var/lib/apt/lists/* \
    && /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

# ---------------------------------------------------------------------------
# 4. Gazebo Harmonic (OSRF 저장소)
# ---------------------------------------------------------------------------
RUN curl https://packages.osrfoundation.org/gazebo.gpg \
      --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
      http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends \
      gz-harmonic libgz-sim8-dev \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 5. Python 의존성 (시뮬레이션용 — TensorRT/RealSense는 실기체 전용이라 제외)
# ---------------------------------------------------------------------------
# ultralytics가 기본으로 끌어오는 CUDA 번들 torch(수 GB)는 디스크를 많이 먹으므로
# CPU-only wheel을 먼저 깔아 고정. GPU 추론이 필요하면 이 두 줄을 지우고
# ultralytics가 알아서 CUDA wheel을 받게 두면 됨(대신 이미지가 훨씬 커짐).
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir ultralytics opencv-python numpy

# ---------------------------------------------------------------------------
# 6. 비-root 사용자 (호스트 UID/GID와 맞춰 바인드 마운트 권한 문제 방지)
# ---------------------------------------------------------------------------
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
ENV HOME=/home/${USERNAME}
ENV USER=${USERNAME}
WORKDIR ${HOME}

RUN rosdep update

# ---------------------------------------------------------------------------
# 7. ArduPilot SITL 빌드 (--depth 1: 전체 git 히스토리 없이 소스만, 용량 절약)
# ---------------------------------------------------------------------------
# 주의: .git은 지우면 안 됨 — sim_vehicle.py가 기동할 때마다 waf를 다시 빌드하는데,
# 그 빌드가 버전 문자열 임베딩에 git rev-parse를 쓰기 때문에 이미지 빌드 후에도
# 계속 .git이 있어야 함(런타임 실패 원인이었음). install-prereqs가 까는
# arm-none-eabi 크로스컴파일러(~700MB)는 --board sitl에 안 쓰여서 그것만 지움.
RUN git clone --recurse-submodules --shallow-submodules --depth 1 \
      https://github.com/ArduPilot/ardupilot.git ${HOME}/ardupilot \
    && cd ${HOME}/ardupilot \
    && Tools/environment_install/install-prereqs-ubuntu.sh -y \
    && sudo rm -rf /opt/gcc-arm-none-eabi-10-2020-q4-major \
    && . ${HOME}/.profile \
    && ./waf configure --board sitl \
    && ./waf copter \
    && find build -name '*.o' -delete
WORKDIR ${HOME}/ardupilot

# ---------------------------------------------------------------------------
# 8. ardupilot_gazebo 플러그인
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/ArduPilot/ardupilot_gazebo.git ${HOME}/ardupilot_gazebo \
    && rm -rf ${HOME}/ardupilot_gazebo/.git
WORKDIR ${HOME}/ardupilot_gazebo
RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && make -j$(nproc) \
    && find . -name '*.o' -delete

# ---------------------------------------------------------------------------
# 9. ros_gz 브리지 (Harmonic 소스 빌드)
# ---------------------------------------------------------------------------
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
      ros-humble-actuator-msgs ros-humble-gps-msgs ros-humble-vision-msgs \
    && sudo rm -rf /var/lib/apt/lists/*
RUN mkdir -p ${HOME}/ros_gz_ws/src \
    && git clone --depth 1 -b humble https://github.com/gazebosim/ros_gz.git ${HOME}/ros_gz_ws/src/ros_gz \
    && rm -rf ${HOME}/ros_gz_ws/src/ros_gz/.git
WORKDIR ${HOME}/ros_gz_ws
SHELL ["/bin/bash", "-c"]
RUN . /opt/ros/humble/setup.bash \
    && GZ_VERSION=harmonic colcon build --merge-install --packages-up-to ros_gz_bridge \
    && rm -rf build log

WORKDIR ${HOME}

USER root
COPY env.sh /entrypoint-env.sh
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint-env.sh /entrypoint.sh
USER ${USERNAME}

# docker compose exec은 ENTRYPOINT를 안 거치므로, BASH_ENV(비대화형 bash -c)와
# ~/.bashrc(대화형 bash) 양쪽에 env.sh를 걸어서 exec으로 여는 모든 셸에 적용되게 함.
ENV BASH_ENV=/entrypoint-env.sh
RUN echo 'source /entrypoint-env.sh' >> ${HOME}/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
