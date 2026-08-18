#!/usr/bin/env bash
# Field GPS cue bringup for the current MissionManager.
# MAVROS must already publish the vehicle global fix and local ENU pose.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
TRAILER_LINK="${TRAILER_LINK:-1}"
TRAILER_DEV="${TRAILER_DEV:-/dev/ttyUSB0}"
TRAILER_BAUD="${TRAILER_BAUD:-57600}"
TRAILER_SYSID="${TRAILER_SYSID:-1}"
TRAILER_DECK_Z_M="${TRAILER_DECK_Z_M:-}"
GPS_INPUT_TIMEOUT_S="${GPS_INPUT_TIMEOUT_S:-1.0}"
GPS_MAX_INPUT_SKEW_S="${GPS_MAX_INPUT_SKEW_S:-0.25}"
GPS_MIN_RATE_HZ="${GPS_MIN_RATE_HZ:-4.0}"
GPS_READY_TIMEOUT_S="${GPS_READY_TIMEOUT_S:-15}"

if [[ -z "$TRAILER_DECK_Z_M" ]]; then
  echo "ERROR: set measured TRAILER_DECK_Z_M in PX4 local ENU." >&2
  exit 2
fi
if [[ "$TRAILER_LINK" != "0" && "$TRAILER_LINK" != "1" ]]; then
  echo "ERROR: TRAILER_LINK must be 0 or 1." >&2
  exit 2
fi
if ! [[ "$TRAILER_SYSID" =~ ^[0-9]+$ ]] \
    || (( TRAILER_SYSID < 1 || TRAILER_SYSID > 254 )); then
  echo "ERROR: TRAILER_SYSID must be 1..254." >&2
  exit 2
fi
python3 - "$TRAILER_DECK_Z_M" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value):
    raise SystemExit('TRAILER_DECK_Z_M must be finite')
PY
if [[ "$TRAILER_LINK" == "1" ]]; then
  TRAILER_CANON="$(readlink -f "$TRAILER_DEV" 2>/dev/null || true)"
  if [[ -z "$TRAILER_CANON" || ! -c "$TRAILER_CANON" ]]; then
    echo "ERROR: not a serial device: $TRAILER_DEV" >&2
    exit 2
  fi
fi

set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1091
source "$REPO_DIR/install/setup.bash"
set -u
TRAILER_PREFIX="$(ros2 pkg prefix trailer_link 2>/dev/null || true)"
if [[ "$(readlink -f "$TRAILER_PREFIX" 2>/dev/null || true)" \
    != "$(readlink -f "$REPO_DIR/install/trailer_link")" ]]; then
  echo "ERROR: build trailer_link in $REPO_DIR first." >&2
  exit 3
fi

publisher_count() {
  { ros2 topic info "$1" 2>/dev/null || true; } \
    | awk '/Publisher count:/ {print $3; found=1} END {if (!found) print 0}'
}
require_no_publisher() {
  local topic="$1" count
  count="$(publisher_count "$topic")"
  if [[ "$count" != "0" ]]; then
    echo "ERROR: $topic already has $count publisher(s)." >&2
    exit 4
  fi
}
require_one_publisher() {
  local topic="$1" count
  count="$(publisher_count "$topic")"
  if [[ "$count" != "1" ]]; then
    echo "ERROR: $topic needs exactly one publisher; found $count." >&2
    return 1
  fi
}
wait_one_publisher() {
  local topic="$1"
  for _ in $(seq 1 100); do
    [[ "$(publisher_count "$topic")" == "1" ]] && return 0
    sleep 0.1
  done
  require_one_publisher "$topic"
}

require_no_publisher /marker/cue
require_no_publisher /marker/cue_velocity
if [[ "$TRAILER_LINK" == "1" ]]; then
  require_no_publisher /trailer/fix
  require_no_publisher /trailer/velocity_enu
fi
wait_one_publisher /mavros/global_position/global || exit 5
wait_one_publisher /mavros/local_position/pose || exit 5

GPS_LOG_ROOT="${GPS_LOG_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/px4-ros2-wang/gps}"
mkdir -p "$GPS_LOG_ROOT"
RUN_DIR="$(mktemp -d "$GPS_LOG_ROOT/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  echo "GPS cue logs: $RUN_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ "$TRAILER_LINK" == "1" ]]; then
  setsid ros2 run trailer_link trailer_gps_node --ros-args \
    -p serial_device:="$TRAILER_DEV" -p baud:="$TRAILER_BAUD" \
    -p target_sysid:="$TRAILER_SYSID" \
    >"$RUN_DIR/trailer_gps.log" 2>&1 &
  PIDS+=("$!")
fi
setsid ros2 run trailer_link trailer_target_node --ros-args \
  -p deck_z_m:="$TRAILER_DECK_Z_M" \
  -p stale_after_s:="$GPS_INPUT_TIMEOUT_S" \
  -p max_input_skew_s:="$GPS_MAX_INPUT_SKEW_S" \
  -p min_source_rate_hz:="$GPS_MIN_RATE_HZ" \
  >"$RUN_DIR/trailer_target.log" 2>&1 &
PIDS+=("$!")

sleep 1
for topic in /trailer/fix /trailer/velocity_enu \
             /marker/cue /marker/cue_velocity; do
  wait_one_publisher "$topic" || exit 6
done
echo -n "waiting for coherent GPS cue (GPI >= ${GPS_MIN_RATE_HZ} Hz) ... "
if ! timeout "$GPS_READY_TIMEOUT_S" \
    ros2 topic echo /marker/cue --once >/dev/null 2>&1; then
  echo "FAILED" >&2
  echo "Run 'ros2 run trailer_link radio_probe --device $TRAILER_DEV'." >&2
  echo "TRAILER_LINK=0 requires an external reader that publishes both" >&2
  echo "/trailer/fix and /trailer/velocity_enu with one source stamp." >&2
  exit 7
fi
echo "READY"
echo "cue source: trailer GPS + vehicle MAVROS local ENU"
echo "logs      : $RUN_DIR"

while :; do
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || {
      echo "ERROR: GPS cue component exited; inspect $RUN_DIR" >&2
      exit 8
    }
  done
  sleep 1
done
