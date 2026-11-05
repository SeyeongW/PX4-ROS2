#!/usr/bin/env bash
# 이미 실행 중인 Gazebo + PX4 시뮬레이션을 재시작하지 않고
# 드론만 즉시 다시 이륙시켜 풍선으로 날아가게 하는 스크립트.
#
# 사용법 (새 터미널에서):
#   ./tools/refly.sh
#   FLY_TARGET_X=90 FLY_TARGET_Y=0 FLY_TARGET_Z=40 ./tools/refly.sh
#
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PX4_PID="$(pgrep -n -x px4 || true)"
if [[ -z "$PX4_PID" ]]; then
    echo "ERROR: 실행 중인 PX4 시뮬레이션을 찾을 수 없습니다. 먼저 ./tools/run_vision_center.sh 를 실행하세요." >&2
    exit 1
fi

# 기존 fly_to_target.py 프로세스가 남아있다면 종료
pkill -f "tools/fly_to_target.py" 2>/dev/null || true
sleep 0.3

source /opt/ros/humble/setup.bash
source install/local_setup.bash

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}"
export ROS_LOCALHOST_ONLY=1
export FLY_STARTUP_DELAY=0
export FLY_TARGET_X="${FLY_TARGET_X:-90.0}"
export FLY_TARGET_Y="${FLY_TARGET_Y:-0.0}"
export FLY_TARGET_Z="${FLY_TARGET_Z:-40.0}"
export PYTHONUNBUFFERED=1

echo "▶ 실행 중인 시뮬레이션(PX4 PID=$PX4_PID)에 접속하여 드론을 다시 발사합니다..."
exec nsenter --preserve-credentials --user --net -t "$PX4_PID" \
    python3 "$PWD/tools/fly_to_target.py" "$@"
