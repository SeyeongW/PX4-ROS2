#!/usr/bin/env bash
# 기존 Gazebo tailsitter_guidance.py 전용 파일 트리거. 실기체 브리지 승인과 무관함.
set -euo pipefail

echo "approved" > /tmp/balloon_launch_approved
echo "[시뮬레이션 승인 요청 기록] 실제 이륙·비행 상태는 확인하지 않습니다. 실기체에는 적용되지 않습니다."
