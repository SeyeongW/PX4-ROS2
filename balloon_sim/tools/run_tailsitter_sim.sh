#!/usr/bin/env bash
# PX4 쿼드 테일시터(1045) + Gazebo 표적 드론(Target UAV) + 온보드 비전 검출기 + 3D LOS/PN 요격 유도 통합 실행 스크립트
#
# 사용법:
#   ./tools/run_tailsitter_sim.sh
#
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
./tools/build_workspace.sh
exec ./tools/run_gazebo.sh \
    mode:=pymav \
    target_type:=x500 \
    target_range:=100.0 \
    target_height:=50.0 \
    target_azimuth:=0.0 \
    gui:=true \
    ui:=false \
    rviz:=false \
    "$@"
