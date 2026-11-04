#!/usr/bin/env bash
# Gazebo + PX4 SITL + onboard detector + pymavlink flight.
# One command: sim starts, PX4 boots, drone takes off and flies to target.
#
#   ./tools/run_vision_center.sh
#
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec ./tools/run_gazebo.sh \
    mode:=pymav \
    target_range:=90.0 \
    target_height:=40.0 \
    target_azimuth:=0.0 \
    gui:=true \
    ui:=false \
    rviz:=false \
    "$@"
