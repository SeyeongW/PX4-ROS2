#!/usr/bin/env bash
# Single PX4/MAVROS Relative-OSQP precision-landing entry point.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ $# -ne 0 ]]; then
  echo "Usage: $(basename "$0")" >&2
  exit 64
fi

VEHICLE_AUTHORITY_ID="${VEHICLE_AUTHORITY_ID:-vehicle_1}"
BASELINE_RUN_ID="${BASELINE_RUN_ID:-}"
CONTROL_STACK_NAME="px4_mavros_relative_osqp"
CONFIG_FILE="$REPO_DIR/precision_landing/config/mpc_precision_landing.yaml"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/control_authority.sh"
CONTROL_LOCK_FILE="$(resolve_control_authority_lock_file "$VEHICLE_AUTHORITY_ID")"

echo "CONTROL STACK        : $CONTROL_STACK_NAME"
echo "CONTROLLER           : relative_landing_mpc"
echo "MPC SOLVER           : osqp"
echo "AUTOPILOT            : PX4"
echo "COMMAND TRANSPORT    : mavros_position_target"
echo "COMMAND TOPICS       :"
echo "  - /mavros/setpoint_raw/local"
echo "  - /mavros/set_mode"
echo "  - /mavros/cmd/arming"
echo "CONFIG FILE          : $CONFIG_FILE"
echo "TRAILER STATE        : measured Gazebo odometry"
echo "PRODUCTION QUALIFIED : true"
echo "VEHICLE AUTHORITY ID : $VEHICLE_AUTHORITY_ID"
echo "CONTROL LOCK FILE    : $CONTROL_LOCK_FILE"
echo "BASELINE RUN ID      : ${BASELINE_RUN_ID:-<empty>}"
echo "ENVIRONMENT          : existing Gazebo/PX4/MAVROS runner"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: ros2 is unavailable; source ROS 2 and this workspace." >&2
  exit 69
fi

acquire_control_authority "$CONTROL_STACK_NAME" "$VEHICLE_AUTHORITY_ID"
echo "CONTROL AUTHORITY    : acquired"

# The estimator repeatedly solves small (4x4/8x8) systems. Multi-threaded
# BLAS adds no useful throughput there and its worker spin-wait can consume
# several cores, starving image callbacks until their capture stamps are
# stale. Keep the flight process deterministic unless an operator explicitly
# selects a different thread count for an offline experiment.
PRECISION_LANDING_BLAS_THREADS="${PRECISION_LANDING_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$PRECISION_LANDING_BLAS_THREADS"
export OMP_NUM_THREADS="$PRECISION_LANDING_BLAS_THREADS"
export MKL_NUM_THREADS="$PRECISION_LANDING_BLAS_THREADS"
echo "BLAS THREADS         : $PRECISION_LANDING_BLAS_THREADS"

exec ros2 launch precision_landing mpc_precision_landing.launch.py \
  "baseline_run_id:=$BASELINE_RUN_ID" \
  "vehicle_authority_id:=$VEHICLE_AUTHORITY_ID"
