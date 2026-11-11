#!/usr/bin/env bash
# 지상 카메라 UKF가 풍선을 락온했을 때 드론 출격 승인(Approve Launch) 트리거 스크립트
set -euo pipefail

echo "approved" > /tmp/balloon_launch_approved
echo "★ [출격 승인 완료] 드론 이륙 및 폐루프 유도 비행을 시작합니다!"
