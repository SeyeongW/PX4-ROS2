#!/usr/bin/env bash
# Start only the PX4/MAVROS precision-landing simulation environment.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  cat <<EOF
Usage: $(basename "$0") [additional gz sim options]

Starts the bounded 200 x 200 m Gazebo world, dynamically spawns the JO city
x500 at ENU (30,30,0), and places the static JO ArUco trailer exactly 10 m
northeast at (37.071067811865476,37.071067811865476,0).

Defaults: MAVROS on, sensor bridge on, XRCE-DDS off, trailer motion forbidden.
Useful overrides: HEADLESS=1, PX4_DAEMON=1, START_MAVROS=0, START_BRIDGE=0.
EOF
  exit 0
fi

if [[ "${DRIVE_TRAILER:-0}" != "0" ]]; then
  echo "ERROR: DRIVE_TRAILER is forbidden in the stationary precision-landing baseline." >&2
  exit 2
fi

export START_MAVROS="${START_MAVROS:-1}"
export START_BRIDGE="${START_BRIDGE:-1}"
export START_XRCE="${START_XRCE:-0}"
export DRIVE_TRAILER=0
export MAVROS_FCU_URL="${MAVROS_FCU_URL:-udp://:14540@127.0.0.1:14580}"
export MAVROS_HEARTBEAT_TYPE="${MAVROS_HEARTBEAT_TYPE:-GCS}"
export MAVROS_HEARTBEAT_RATE="${MAVROS_HEARTBEAT_RATE:-10.0}"

exec "$SCRIPT_DIR/run_px4_map.sh" precision-landing "$@"
