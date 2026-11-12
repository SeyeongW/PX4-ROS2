#!/usr/bin/env bash
# Ubuntu 22.04, ROS 2 Humble, Gazebo Classic 11. Does not install system packages.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /opt/ros/humble/setup.bash
export PYTHONDONTWRITEBYTECODE=1
colcon build --base-paths balloon_locator support --packages-up-to balloon_locator \
  --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release "$@"
