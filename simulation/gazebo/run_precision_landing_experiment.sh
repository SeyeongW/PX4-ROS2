#!/usr/bin/env bash
# Two-terminal fixed-profile ArUco precision-landing SITL runner.
set -Eeuo pipefail

SCRIPT_PATH="$(realpath -e "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

usage() {
  cat <<EOF
Usage:
  Terminal 1: $(basename "$0") environment
  Terminal 2: $(basename "$0") mission
  Moving Terminal 1: $(basename "$0") moving-environment
  Moving Terminal 2: $(basename "$0") moving-mission
  Camera Terminal: $(basename "$0") camera-view

environment starts the 200 x 200 m Gazebo GUI, PX4 SITL, MAVROS, the GCS
heartbeat, and the camera/depth/lidar bridge. mission performs an isolated
/tmp build, starts only the Relative-OSQP controller, verifies the
unarmed ON_GROUND preflight contract, and issues one mission start request.

The moving environment starts the trailer's circular route after PX4 spawns;
the mission approaches from measured position and estimates speed from ArUco.
camera-view waits for the mission's ArUco debug publisher and opens the
downward annotated image without rebuilding either ROS package.
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if [[ $# -ne 1 || ! "$1" =~ ^(environment|mission|moving-environment|moving-mission|camera-view)$ ]]; then
  echo "ERROR: invalid experiment subcommand" >&2
  usage >&2
  exit 64
fi

# Start from a clean process environment so a shell which previously sourced
# another checkout cannot redirect ROS package or Python resolution. Preserve
# only the desktop/session values required by the Gazebo GUI.
if [[ "${PX4_ROS22_EXPERIMENT_CLEAN_ENV:-0}" != "1" ]]; then
  exec env -i \
    HOME="$HOME" \
    USER="${USER:-$(id -un)}" \
    LOGNAME="${LOGNAME:-${USER:-$(id -un)}}" \
    SHELL="${SHELL:-/bin/bash}" \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG="${LANG:-C.UTF-8}" \
    LC_ALL="${LC_ALL:-}" \
    TERM="${TERM:-xterm-256color}" \
    DISPLAY="${DISPLAY:-}" \
    WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
    DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-}" \
    XAUTHORITY="${XAUTHORITY:-}" \
    PX4_ROS22_EXPERIMENT_CLEAN_ENV=1 \
    PX4_ROS22_EXPERIMENT_HEADLESS="${PX4_ROS22_EXPERIMENT_HEADLESS:-0}" \
    PX4_ROS22_EXPERIMENT_PX4_DAEMON="${PX4_ROS22_EXPERIMENT_PX4_DAEMON:-0}" \
    TRAILER_SPEED_M_S="${TRAILER_SPEED_M_S:-}" \
    PYTHONDONTWRITEBYTECODE=1 \
    bash "$SCRIPT_PATH" "$@"
fi

ROS_SETUP="/opt/ros/humble/setup.bash"
ROS_DOMAIN_ID_VALUE=42
ROS_LOCALHOST_ONLY_VALUE=1
GZ_PARTITION_VALUE="px4_ros22_precision_landing_${USER:-user}"
PX4_DIR_VALUE="$HOME/PX4-Autopilot"
if [[ "$1" == moving-* ]]; then
  EXPERIMENT_MODE="moving"
  ENVIRONMENT_RUNNER="$SCRIPT_DIR/run_precision_landing_200m_moving.sh"
  CONTROLLER_RUNNER="$SCRIPT_DIR/run_px4_mpc_precision_landing_moving.sh"
else
  EXPERIMENT_MODE="stationary"
  ENVIRONMENT_RUNNER="$SCRIPT_DIR/run_precision_landing_200m.sh"
  CONTROLLER_RUNNER="$SCRIPT_DIR/run_px4_mpc_precision_landing.sh"
fi
SUPERVISOR="$SCRIPT_DIR/tools/run_precision_landing_once.py"

[[ -f "$ROS_SETUP" ]] || {
  echo "ERROR: ROS 2 Humble setup is missing: $ROS_SETUP" >&2
  exit 69
}

if [[ "$1" == "environment" || "$1" == "moving-environment" ]]; then
  if [[ "$EXPERIMENT_MODE" == "moving" ]]; then
    DRIVE_TRAILER_VALUE=1
  else
    DRIVE_TRAILER_VALUE=0
  fi
  echo "$EXPERIMENT_MODE 트레일러 SITL 환경을 시작합니다. 이 터미널은 계속 유지하세요."
  echo "ROS_DOMAIN_ID    : $ROS_DOMAIN_ID_VALUE"
  echo "Gazebo partition : $GZ_PARTITION_VALUE"
  exec env \
    ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
    ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
    ROS_SETUP="$ROS_SETUP" \
    PX4_DIR="$PX4_DIR_VALUE" \
    GZ_PARTITION="$GZ_PARTITION_VALUE" \
    PX4_MAP_RUNTIME_DIR="/tmp/px4_ros22_precland_env_${USER:-user}" \
    HEADLESS="$PX4_ROS22_EXPERIMENT_HEADLESS" \
    PX4_DAEMON="$PX4_ROS22_EXPERIMENT_PX4_DAEMON" \
    START_MAVROS=1 \
    START_BRIDGE=1 \
    START_XRCE=0 \
    DRIVE_TRAILER="$DRIVE_TRAILER_VALUE" \
    FOLLOW_DRONE=0 \
    PHYSICS_ENGINE=gz-physics-dartsim-plugin \
    MAVROS_FCU_URL=udp://:14540@127.0.0.1:14580 \
    MAVROS_HEARTBEAT_TYPE=GCS \
    MAVROS_HEARTBEAT_RATE=10.0 \
    "$ENVIRONMENT_RUNNER"
fi

export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE"
export ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE"
export GZ_PARTITION="$GZ_PARTITION_VALUE"
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

ros_publisher_count() {
  local topic="${1:?missing topic}"
  local info
  info="$(ros2 topic info "$topic" 2>/dev/null || true)"
  awk '$1 == "Publisher" && $2 == "count:" {print $3}' <<<"$info"
}

if [[ "$1" == "camera-view" ]]; then
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "ERROR: camera-view requires a graphical desktop session." >&2
    exit 69
  fi
  for command in awk ros2 sleep; do
    command -v "$command" >/dev/null 2>&1 || {
      echo "ERROR: required camera-view command is missing: $command" >&2
      exit 69
    }
  done
  ros2 pkg prefix rqt_image_view >/dev/null 2>&1 || {
    echo "ERROR: ROS package rqt_image_view is not installed." >&2
    exit 69
  }

  RAW_TOPIC="/down_camera/image"
  DEBUG_TOPIC="/perception/down/debug"
  echo "하방 카메라 브리지를 기다립니다: $RAW_TOPIC"
  raw_ready=0
  for ((attempt = 1; attempt <= 30; attempt++)); do
    publisher_count="$(ros_publisher_count "$RAW_TOPIC")"
    if [[ "$publisher_count" == "1" ]]; then
      raw_ready=1
      break
    fi
    if [[ "$publisher_count" =~ ^[0-9]+$ ]] && \
        ((publisher_count > 1)); then
      echo "ERROR: $RAW_TOPIC has $publisher_count publishers." >&2
      echo "       Stop the stale environment; duplicate frames invalidate ArUco timestamps." >&2
      exit 73
    fi
    sleep 1
  done
  if [[ "$raw_ready" != "1" ]]; then
    echo "ERROR: $RAW_TOPIC publisher is absent; start environment first." >&2
    echo "Raw viewer after recovery: ros2 run rqt_image_view rqt_image_view --force-discover $RAW_TOPIC" >&2
    exit 75
  fi

  echo "ArUco 디버그 영상을 기다립니다: $DEBUG_TOPIC"
  echo "mission 또는 moving-mission을 실행하면 자동으로 열립니다."
  debug_ready=0
  for ((attempt = 1; attempt <= 120; attempt++)); do
    publisher_count="$(ros_publisher_count "$DEBUG_TOPIC")"
    if [[ "$publisher_count" == "1" ]]; then
      debug_ready=1
      break
    fi
    if [[ "$publisher_count" =~ ^[0-9]+$ ]] && \
        ((publisher_count > 1)); then
      echo "ERROR: $DEBUG_TOPIC has $publisher_count publishers." >&2
      echo "       Stop the duplicate precision-landing controller." >&2
      exit 73
    fi
    sleep 1
  done
  if [[ "$debug_ready" != "1" ]]; then
    echo "ERROR: ArUco debug publisher did not appear within 120 s." >&2
    echo "Confirm that mission is still running; the detector exits when the mission ends." >&2
    exit 75
  fi
  exec ros2 run rqt_image_view rqt_image_view \
    --force-discover "$DEBUG_TOPIC"
fi

for command in awk colcon python3 realpath ros2 setsid sha256sum timeout; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is missing: $command" >&2
    exit 69
  }
done
for required in "$ENVIRONMENT_RUNNER" "$CONTROLLER_RUNNER" "$SUPERVISOR"; do
  [[ -f "$required" ]] || {
    echo "ERROR: required experiment component is missing: $required" >&2
    exit 69
  }
done

# Never consume the checkout's contaminated build/install/log directories.
# Both packages are rebuilt from this checkout into one isolated /tmp overlay.
COLCON_ROOT="/tmp/px4_ros22_precision_landing_colcon_${USER:-user}"
BUILD_BASE="$COLCON_ROOT/build"
INSTALL_BASE="$COLCON_ROOT/install"
LOG_BASE="$COLCON_ROOT/log"
for candidate in "$BUILD_BASE" "$INSTALL_BASE" "$LOG_BASE"; do
  case "$(realpath -m -- "$candidate")" in
    /tmp/px4_ros22_precision_landing_colcon_*) ;;
    *)
      echo "ERROR: refusing non-isolated colcon path: $candidate" >&2
      exit 73
      ;;
  esac
done
mkdir -p "$BUILD_BASE" "$INSTALL_BASE" "$LOG_BASE"

BUILD_STAMP="$COLCON_ROOT/package-layout.sha256"
BUILD_SIGNATURE="$({
  sha256sum \
    "$REPO_DIR/flight/aruco_landing/setup.py" \
    "$REPO_DIR/flight/aruco_landing/setup.cfg" \
    "$REPO_DIR/flight/aruco_landing/package.xml" \
    "$REPO_DIR/flight/precision_landing/setup.py" \
    "$REPO_DIR/flight/precision_landing/setup.cfg" \
    "$REPO_DIR/flight/precision_landing/package.xml"
  find \
    "$REPO_DIR/flight/aruco_landing/config" \
    "$REPO_DIR/flight/aruco_landing/launch" \
    "$REPO_DIR/flight/precision_landing/config" \
    "$REPO_DIR/flight/precision_landing/launch" \
    -maxdepth 1 -type f -printf '%p\n' 2>/dev/null | sort
} | sha256sum | cut -d' ' -f1)"
OVERLAY_READY=0
if [[ -f "$INSTALL_BASE/setup.bash" && \
      -x "$INSTALL_BASE/precision_landing/lib/precision_landing/mpc_precision_landing_node" && \
      -x "$INSTALL_BASE/aruco_landing/lib/aruco_landing/aruco_pose_node" ]]; then
  if [[ ! -f "$BUILD_STAMP" || \
        "$(<"$BUILD_STAMP")" == "$BUILD_SIGNATURE" ]]; then
    OVERLAY_READY=1
    printf '%s\n' "$BUILD_SIGNATURE" >"$BUILD_STAMP"
  fi
fi

if [[ "$OVERLAY_READY" == "1" ]]; then
  echo "[1/4] 기존 /tmp overlay를 바로 사용합니다 (빌드 생략)."
else
  echo "[1/4] 최초 1회 /tmp 격리 overlay를 빌드합니다."
  colcon --log-base "$LOG_BASE" build \
    --base-paths "$REPO_DIR/flight/aruco_landing" "$REPO_DIR/flight/precision_landing" \
    --packages-select aruco_landing precision_landing \
    --build-base "$BUILD_BASE" \
    --install-base "$INSTALL_BASE" \
    --symlink-install \
    --event-handlers console_cohesion+
  printf '%s\n' "$BUILD_SIGNATURE" >"$BUILD_STAMP"
fi

set +u
# shellcheck disable=SC1090
source "$INSTALL_BASE/setup.bash"
set -u

expected_precision_prefix="$(realpath -m "$INSTALL_BASE/precision_landing")"
expected_camera_prefix="$(realpath -m "$INSTALL_BASE/aruco_landing")"
actual_precision_prefix="$(realpath -m "$(ros2 pkg prefix precision_landing)")"
actual_camera_prefix="$(realpath -m "$(ros2 pkg prefix aruco_landing)")"
if [[ "$actual_precision_prefix" != "$expected_precision_prefix" || \
      "$actual_camera_prefix" != "$expected_camera_prefix" ]]; then
  echo "ERROR: ROS package provenance is not the isolated PX4-ROS22 build." >&2
  echo "  precision_landing: $actual_precision_prefix" >&2
  echo "  aruco_landing    : $actual_camera_prefix" >&2
  exit 73
fi
if ! ros2 pkg executables precision_landing | \
    grep -Fxq 'precision_landing mpc_precision_landing_node'; then
  echo "ERROR: isolated build lacks mpc_precision_landing_node." >&2
  exit 69
fi

if timeout 3 ros2 service type /autonomy/start_precision_landing \
    2>/dev/null | grep -q .; then
  echo "ERROR: another precision-landing controller is already active." >&2
  echo "       Stop it before starting a new mission." >&2
  exit 73
fi

wait_for_topic() {
  local topic="${1:?missing topic}"
  local label="${2:?missing label}"
  if ! timeout 30 ros2 topic echo --once "$topic" >/dev/null 2>&1; then
    echo "ERROR: $label is not live on $topic." >&2
    echo "       Start Terminal 1 and wait for PX4 spawn confirmation." >&2
    return 1
  fi
  echo "  ready: $label"
}

echo "[2/4] PX4, MAVROS, 하방 카메라 입력을 확인합니다."
wait_for_topic /clock "Gazebo clock"
wait_for_topic /mavros/local_position/pose "PX4 local pose"
wait_for_topic /mavros/local_position/velocity_local "PX4 local velocity"
wait_for_topic /down_camera/image "downward RGB camera"
wait_for_topic /down_camera/camera_info "downward camera calibration"

require_single_publisher() {
  local topic="${1:?missing topic}"
  local label="${2:?missing label}"
  local count
  count="$(ros_publisher_count "$topic")"
  if [[ "$count" != "1" ]]; then
    echo "ERROR: $label must have exactly one publisher; found ${count:-0}." >&2
    echo "       Duplicate camera frames reuse timestamps and invalidate ArUco tracking." >&2
    echo "       Stop the stale environment and start Terminal 1 only once." >&2
    return 1
  fi
  echo "  ready: single $label publisher"
}

require_single_publisher /down_camera/image "downward RGB camera"
require_single_publisher /down_camera/camera_info "downward camera calibration"

MAVROS_PREFLIGHT_READY=0
for _ in {1..30}; do
  state="$(timeout 3 ros2 topic echo --once /mavros/state 2>/dev/null || true)"
  extended="$(
    timeout 3 ros2 topic echo --once /mavros/extended_state 2>/dev/null || true
  )"
  if grep -Eq '^connected: true$' <<<"$state" && \
      grep -Eq '^armed: false$' <<<"$state" && \
      grep -Eq '^landed_state: 1$' <<<"$extended"; then
    MAVROS_PREFLIGHT_READY=1
    break
  fi
  sleep 1
done
if [[ "$MAVROS_PREFLIGHT_READY" != "1" ]]; then
  echo "ERROR: PX4 must be connected, disarmed, and ON_GROUND." >&2
  exit 1
fi
echo "  ready: PX4 connected, disarmed, ON_GROUND"

RUN_ID="$EXPERIMENT_MODE-osqp-$(date -u '+%Y%m%dT%H%M%SZ')-$$"
RUN_DIR="/tmp/px4_ros22_precision_landing_runs/$RUN_ID"
mkdir -p "$RUN_DIR/ros_log"
CONTROLLER_LOG="$RUN_DIR/controller.log"
SUPERVISOR_LOG="$RUN_DIR/mission.log"
HOLD_LOG="$RUN_DIR/hold_service.log"
CONTROLLER_PGID=""
SUPERVISOR_PGID=""
CONTROLLER_MAY_BE_ACTIVE=0
CONTROLLER_STOP_ALLOWED=1
HOLD_REQUESTED=0

stop_group() {
  local pgid="${1:-}"
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -INT -- "-$pgid" 2>/dev/null || true
    for _ in {1..40}; do
      kill -0 -- "-$pgid" 2>/dev/null || break
      sleep 0.1
    done
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 -- "-$pgid" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  wait "$pgid" 2>/dev/null || true
}

request_hold() {
  HOLD_REQUESTED=1
  set +e
  timeout 7 ros2 service call \
    /autonomy/hold_precision_landing std_srvs/srv/Trigger '{}' \
    >"$HOLD_LOG" 2>&1
  local service_status=$?
  set -e
  if [[ "$service_status" == "0" ]] && \
      grep -Eq 'success=(True|true)' "$HOLD_LOG"; then
    echo "안전 위치 유지 요청이 확인되었습니다."
    return 0
  fi
  echo "WARN: 안전 위치 유지 응답을 확인하지 못했습니다: $HOLD_LOG" >&2
  return 1
}

environment_connected() {
  local state
  state="$(timeout 3 ros2 topic echo --once /mavros/state 2>/dev/null || true)"
  grep -Eq '^connected: true$' <<<"$state"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_group "$SUPERVISOR_PGID"
  if [[ "$CONTROLLER_STOP_ALLOWED" == "1" ]]; then
    stop_group "$CONTROLLER_PGID"
  elif [[ -n "$CONTROLLER_PGID" ]] && \
      kill -0 -- "-$CONTROLLER_PGID" 2>/dev/null; then
    echo "WARN: 비행 가능성이 있어 controller heartbeat를 유지합니다." >&2
    echo "      Terminal 1을 먼저 종료한 뒤 이 터미널에서 Ctrl-C를 누르세요." >&2
  fi
  exit "$status"
}

handle_interrupt() {
  if [[ "$CONTROLLER_MAY_BE_ACTIVE" != "1" ]]; then
    CONTROLLER_STOP_ALLOWED=1
    exit 130
  fi
  if [[ "$HOLD_REQUESTED" != "1" ]]; then
    request_hold || true
    echo "비행 중 종료 대신 hold를 유지합니다." >&2
    echo "Terminal 1을 Ctrl-C로 종료한 뒤 이 터미널에서 다시 Ctrl-C를 누르세요." >&2
    return
  fi
  if environment_connected; then
    echo "PX4가 아직 연결되어 있어 controller 종료를 거부합니다." >&2
    echo "Terminal 1을 먼저 Ctrl-C로 종료하세요." >&2
    return
  fi
  CONTROLLER_STOP_ALLOWED=1
  exit 130
}

trap cleanup EXIT
trap handle_interrupt INT TERM

echo "[3/4] $EXPERIMENT_MODE Relative-OSQP controller를 시작합니다."
echo "Run ID         : $RUN_ID"
echo "Controller log : $CONTROLLER_LOG"
setsid bash -c '
  set -Eeuo pipefail
  set +u
  source "$1"
  source "$2"
  set -u
  unset PRECISION_LANDING_CONFIG MPC_PRECISION_LANDING_CONFIG
  unset PRECISION_LANDING_VALIDATION_CONFIG PRECISION_LANDING_PROFILE
  export BASELINE_RUN_ID="$3"
  export ROS_LOG_DIR="$4"
  exec "$5"
' _ \
  "$ROS_SETUP" \
  "$INSTALL_BASE/setup.bash" \
  "$RUN_ID" \
  "$RUN_DIR/ros_log" \
  "$CONTROLLER_RUNNER" >"$CONTROLLER_LOG" 2>&1 &
CONTROLLER_PGID=$!

sleep 1
if ! kill -0 -- "-$CONTROLLER_PGID" 2>/dev/null; then
  echo "ERROR: controller exited during startup." >&2
  tail -80 "$CONTROLLER_LOG" >&2 || true
  exit 1
fi

echo "[4/4] 준비 계약을 재확인하고 임무를 한 번 시작합니다."
echo "Mission log    : $SUPERVISOR_LOG"
CONTROLLER_MAY_BE_ACTIVE=1
CONTROLLER_STOP_ALLOWED=0
setsid python3 -u "$SUPERVISOR" \
  --run-id "$RUN_ID" \
  --ready-timeout 120 \
  --mission-timeout 480 \
  > >(tee "$SUPERVISOR_LOG") 2>&1 &
SUPERVISOR_PGID=$!

set +e
wait "$SUPERVISOR_PGID"
MISSION_STATUS=$?
set -e
stop_group "$SUPERVISOR_PGID"
SUPERVISOR_PGID=""

if [[ "$MISSION_STATUS" == "0" ]]; then
  echo "실험 성공: PX4 landed + disarmed + ON_GROUND가 확인되었습니다."
  CONTROLLER_STOP_ALLOWED=1
  CONTROLLER_MAY_BE_ACTIVE=0
  stop_group "$CONTROLLER_PGID"
  CONTROLLER_PGID=""
  exit 0
fi

request_hold || true
echo "실험이 정상 완료되지 않았습니다 (status=$MISSION_STATUS)." >&2
echo "controller heartbeat와 hold는 의도적으로 유지합니다." >&2
echo "로그: $CONTROLLER_LOG" >&2
echo "먼저 Terminal 1을 Ctrl-C로 종료하고, 그 다음 여기서 Ctrl-C를 누르세요." >&2
while kill -0 -- "-$CONTROLLER_PGID" 2>/dev/null; do
  sleep 1
done
CONTROLLER_STOP_ALLOWED=1
CONTROLLER_PGID=""
exit "$MISSION_STATUS"
