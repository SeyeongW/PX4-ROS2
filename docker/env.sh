#!/usr/bin/env bash
# 컨테이너의 모든 셸(docker exec 포함)에서 공통으로 source되는 환경설정.
# BASH_ENV(비대화형 bash -c)와 ~/.bashrc(대화형 셸) 양쪽에서 이 파일을 가리키므로
# docker compose exec sim bash / docker compose exec sim bash -c "..." 둘 다 적용됨.
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

# sim_vehicle.py / MAVProxy 등 ArduPilot 도구 PATH 등록
export PATH="$HOME/ardupilot/Tools/autotest:$HOME/.local/bin:$PATH"
