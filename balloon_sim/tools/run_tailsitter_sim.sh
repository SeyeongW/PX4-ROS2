#!/usr/bin/env bash
# PX4 쿼드 테일시터(1045) + Gazebo + 온보드 비전 검출기 + 3D LOS/PN 유도 비행 통합 실행 스크립트
#
# 사용법:
#   ./tools/run_tailsitter_sim.sh
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
