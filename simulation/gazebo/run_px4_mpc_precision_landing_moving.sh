#!/usr/bin/env bash
# Fixed Relative-OSQP mission with waypoint-driven trailer odometry.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_px4_precision_landing.sh" "$@"
