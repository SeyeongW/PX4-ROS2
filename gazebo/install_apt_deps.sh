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
apt-get install -y curl lsb-release gnupg cmake build-essential \
  rapidjson-dev python3-rosdep python3-colcon-common-extensions

echo "==> Gazebo(OSRF) apt 저장소 등록"
curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  > /etc/apt/sources.list.d/gazebo-stable.list

echo "==> Gazebo Harmonic + 플러그인 빌드 헤더 설치"
apt-get update
apt-get install -y gz-harmonic libgz-sim8-dev

echo "==> 완료. 설치 버전:"
gz sim --version || true
echo "이제 sudo 없이 빌드를 진행하면 됩니다."
