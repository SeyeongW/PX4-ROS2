#!/usr/bin/env bash
# Start the 200 m PX4/MAVROS environment with its circular trailer route.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export START_MAVROS="${START_MAVROS:-1}"
export START_BRIDGE="${START_BRIDGE:-1}"
export START_XRCE="${START_XRCE:-0}"
export DRIVE_TRAILER="${DRIVE_TRAILER:-1}"
export MAVROS_FCU_URL="${MAVROS_FCU_URL:-udp://:14540@127.0.0.1:14580}"
export MAVROS_HEARTBEAT_TYPE="${MAVROS_HEARTBEAT_TYPE:-GCS}"
export MAVROS_HEARTBEAT_RATE="${MAVROS_HEARTBEAT_RATE:-10.0}"

exec "$SCRIPT_DIR/run_px4_map.sh" precision-landing-moving "$@"
