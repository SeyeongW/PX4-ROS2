#!/usr/bin/env bash
# Compatibility launcher: camera racing, STL 2kg MC, 3m takeoff and cubic spline.
# Legacy tailsitter requires mode:=pymav experimental_tailsitter:=true explicitly.
#
# 사용법:
#   ./tools/run_tailsitter_sim.sh
#
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec ./tools/run_gazebo.sh \
    mode:=mission \
    target_range:=40.0 \
    target_height:=10.0 \
    target_azimuth:=-90.0 \
    gui:=true \
    ui:=true \
    rviz:=true \
    "$@"
