#!/usr/bin/env bash
#
# sudo가 필요한 apt 설치만 모은 스크립트.
# 사용법:  sudo bash gazebo/install_apt_deps.sh
#
# 이 스크립트가 끝나면 sudo 없이 빌드(ArduPilot/플러그인/ros_gz)를 진행할 수 있습니다.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "이 스크립트는 sudo로 실행해야 합니다:  sudo bash $0" >&2
  exit 1
fi

echo "==> 기본 도구 설치"
apt-get update
apt-get install -y \
  curl lsb-release gnupg software-properties-common \
  build-essential cmake git git-lfs docker.io rapidjson-dev libopencv-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl \
  python3-colcon-common-extensions python3-rosdep python3-vcstool python3-pip \
  python3-opencv python3-numpy python3-scipy python3-matplotlib python3-pil python3-yaml \
  ros-humble-desktop ros-humble-navigation2 ros-humble-nav2-bringup \
  ros-humble-slam-toolbox ros-humble-rviz2 ros-humble-xacro \
  ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
  ros-humble-tf2-ros ros-humble-cv-bridge ros-humble-image-transport \
  ros-humble-vision-msgs ros-humble-pcl-ros \
  ros-humble-mavros ros-humble-mavros-msgs ros-humble-mavros-extras

# Ubuntu 22.04's stock Gazebo Classic and Gazebo Harmonic both normally own a
# binary named `gz`.  Install the official Open Robotics Gazebo 11 CLI variant
# first so Classic remains available as `gz11` while Harmonic owns `gz`.
if [[ "$(lsb_release -cs)" == "jammy" ]]; then
  echo "==> Gazebo Classic 11 / Harmonic 병행 설치 준비"
  add-apt-repository -y ppa:openrobotics/gazebo11-gz-cli
  apt-get update
  apt-get install -y gazebo11
fi

echo "==> Gazebo(OSRF) apt 저장소 등록"
curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  > /etc/apt/sources.list.d/gazebo-stable.list

echo "==> Gazebo Harmonic + 플러그인 빌드 헤더 설치"
apt-get update
apt-get install -y gz-harmonic libgz-sim8-dev

# The split Gazebo 11 package uses libgazebo11-dev as the compatible provider.
# Request it explicitly, then restore the ROS 2 Classic integration packages.
# --no-remove turns an unexpected future package conflict into a hard failure.
if [[ "$(lsb_release -cs)" == "jammy" ]] && dpkg-query -W gazebo11 >/dev/null 2>&1; then
  echo "==> ROS 2 Humble용 Gazebo Classic 플러그인 복원"
  apt-get install -y --no-remove \
    gazebo11 gz-harmonic libgz-sim8-dev libgazebo11-dev \
    ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
fi

echo "==> 완료. 설치 버전:"
gz sim --version || true
echo "이제 sudo 없이 빌드를 진행하면 됩니다."
