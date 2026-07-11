#!/usr/bin/env bash
# env.sh의 PX4 버전 -- docker/Dockerfile.px4/docker-compose.yml(sim-px4 서비스) 전용.
# ArduPilot 버전과 로드되는 파일이 달라야 해서(PATH가 서로 다른 SITL 도구를 가리킴)
# 별도 파일로 분리 -- 나머지(ROS2/ros_gz_ws/워크스페이스 source)는 동일.
[[ -n "$_PX4ROS2_ENV_LOADED" ]] && return 0
export _PX4ROS2_ENV_LOADED=1

source /opt/ros/humble/setup.bash

if [[ -f "$HOME/ros_gz_ws/install/setup.bash" ]]; then
  source "$HOME/ros_gz_ws/install/setup.bash"
fi

WS="$HOME/ros2_ws/PX4-ROS2"
if [[ -f "$WS/install/setup.bash" ]]; then
  source "$WS/install/setup.bash"
fi

export PATH="$HOME/.local/bin:$PATH"
export PX4_HOME="$HOME/PX4-Autopilot"
