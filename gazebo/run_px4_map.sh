#!/usr/bin/env bash
# Launch a checked-in map and dynamically spawn the real PX4 SITL vehicle.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MAP="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$MAP" in
  city)
    COORDINATES="$SCRIPT_DIR/maps/city_coordinates.yaml"
    ;;
  mountain)
    COORDINATES="$SCRIPT_DIR/maps/mountain_coordinates.yaml"
    ;;
  -h|--help|help|"")
    cat <<EOF
Usage: $(basename "$0") <city|mountain> [additional gz sim options]

Starts Gazebo Harmonic, waits for the selected world, then starts PX4 SITL.
PX4 itself creates the dynamic x500_depth_lidar vehicle (default x500 quad
with a forward depth camera and a downward lidar rangefinder).
The seo-branch flat_platform trailer is included in both maps.

Environment:
  PX4_DIR=~/PX4-Autopilot  Existing PX4 source/build (firmware is not changed)
  HEADLESS=1              Gazebo server only
  START_XRCE=0            Do not start Micro XRCE-DDS Agent
  START_MAVROS=0         Do not start MAVROS (MAVLink->ROS 2 bridge)
  MAVROS_FCU_URL=...     Override MAVROS fcu_url (default udp://:14540@127.0.0.1:14580)
  START_BRIDGE=0         Do not start the ros_gz camera/lidar bridge
  ROS_SETUP=...          ROS 2 setup.bash to source for MAVROS + bridge (default humble)
  FOLLOW_DRONE=1          Lock the gz camera to follow the drone (default: free camera)
  DRIVE_TRAILER=1         Drive the included trailer through YAML waypoints
  TRAILER_ROUTE_LOOPS=1   Stop the driver after one complete route (0=repeat)
  TRAILER_ROUTE=slope     Mountain-only terrain-follow safeguard test
  USE_NVIDIA=0            Disable NVIDIA PRIME render variables
  GZ_PARTITION=...        Gazebo transport partition (shared with PX4)
EOF
    exit 0
    ;;
  *)
    echo "ERROR: expected 'city' or 'mountain', got '$MAP'." >&2
    exit 2
    ;;
esac

for command in python3 gz; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is missing: $command" >&2
    exit 2
  }
done
python3 -c 'import yaml' >/dev/null 2>&1 || {
  echo "ERROR: python3-yaml is required (sudo apt install python3-yaml)." >&2
  exit 2
}
[[ -f "$COORDINATES" ]] || {
  echo "ERROR: coordinate contract is missing: $COORDINATES" >&2
  exit 2
}

mapfile -t MAP_CONFIG < <(
  python3 - "$COORDINATES" "$REPO_DIR" <<'PY'
import pathlib
import sys
import yaml

document = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pose = document["spawn"]["gazebo_spawn_pose_enu"]
keys = ("x", "y", "z", "roll", "pitch", "yaw")
print(document["map"]["gazebo_world_name"])
print(pathlib.Path(sys.argv[2], document["map"]["world_file"]).resolve())
print(",".join(str(pose[key]) for key in keys))
print(document["px4_vehicle"]["airframe_autostart_id"])
print(document["px4_vehicle"]["simulation_model"])
print(document["px4_vehicle"]["runtime_entity_name"])
PY
)
[[ ${#MAP_CONFIG[@]} -eq 6 ]] || {
  echo "ERROR: failed to read launch values from $COORDINATES" >&2
  exit 2
}
WORLD_NAME="${MAP_CONFIG[0]}"
WORLD_FILE="${MAP_CONFIG[1]}"
SPAWN_POSE="${MAP_CONFIG[2]}"
AUTOSTART_ID="${MAP_CONFIG[3]}"
SIM_MODEL="${MAP_CONFIG[4]}"
ENTITY_NAME="${MAP_CONFIG[5]}"

PX4_DIR="$(realpath -m "${PX4_DIR:-$HOME/PX4-Autopilot}")"
PX4_BUILD="$PX4_DIR/build/px4_sitl_default"
PX4_BIN="$PX4_BUILD/bin/px4"
PX4_ROOTFS="$PX4_BUILD/rootfs"
PX4_ENV="$PX4_ROOTFS/gz_env.sh"

if [[ ! -x "$PX4_BIN" || ! -f "$PX4_ENV" ]]; then
  cat >&2 <<EOF
ERROR: compatible PX4 SITL build was not found under:
       $PX4_DIR

This launcher never changes an existing firmware checkout. For a fresh PC,
run ./gazebo/setup_px4_sitl.sh once, then retry this command.
EOF
  exit 3
fi
CUSTOM_MODEL="${SIM_MODEL#gz_}"
for model in "$CUSTOM_MODEL" x500 x500_base OakD-Lite LW20; do
  [[ -f "$PX4_DIR/Tools/simulation/gz/models/$model/model.sdf" ]] || {
    echo "ERROR: PX4 Gazebo model asset is missing: $model" >&2
    if [[ "$model" == "$CUSTOM_MODEL" ]]; then
      cat >&2 <<EOF
       Install the checked-in custom model into your PX4 tree first:
         ln -s "$REPO_DIR/px4_models/$CUSTOM_MODEL" \\
               "$PX4_DIR/Tools/simulation/gz/models/$CUSTOM_MODEL"
       See px4_models/README.md for details.
EOF
    fi
    exit 3
  }
done
[[ -f "$WORLD_FILE" ]] || {
  echo "ERROR: world file is missing: $WORLD_FILE" >&2
  exit 3
}

# PX4's generated environment file appends these variables without guarding
# against `set -u`, so initialize them before sourcing it.
GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:-}"
GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
# shellcheck disable=SC1090
source "$PX4_ENV"
export GZ_SIM_RESOURCE_PATH="$SCRIPT_DIR/models:$SCRIPT_DIR/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
export GZ_PARTITION="${GZ_PARTITION:-px4_ros2_${USER:-user}}"

if [[ "${USE_NVIDIA:-1}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __VK_LAYER_NV_optimus=NVIDIA_only
  export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
fi
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi
if [[ "${HEADLESS:-0}" != "1" && "${FOLLOW_DRONE:-0}" == "1" ]]; then
  export PX4_GZ_FOLLOW=1
  export PX4_GZ_FOLLOW_OFFSET_X="${PX4_GZ_FOLLOW_OFFSET_X:--4.0}"
  export PX4_GZ_FOLLOW_OFFSET_Y="${PX4_GZ_FOLLOW_OFFSET_Y:--4.0}"
  export PX4_GZ_FOLLOW_OFFSET_Z="${PX4_GZ_FOLLOW_OFFSET_Z:-2.5}"
else
  unset PX4_GZ_FOLLOW
fi

RUNTIME_DIR="${PX4_MAP_RUNTIME_DIR:-/tmp/px4_ros2_map_${USER:-user}}"
mkdir -p "$RUNTIME_DIR"
GAZEBO_LOG="$RUNTIME_DIR/${MAP}_gazebo.log"
XRCE_LOG="$RUNTIME_DIR/${MAP}_xrce.log"
TRAILER_LOG="$RUNTIME_DIR/${MAP}_trailer.log"
MAVROS_LOG="$RUNTIME_DIR/${MAP}_mavros.log"
BRIDGE_LOG="$RUNTIME_DIR/${MAP}_sensor_bridge.log"
GAZEBO_PID=""
XRCE_PID=""
TRAILER_PID=""
MAVROS_PID=""
BRIDGE_PID=""

cleanup() {
  local status=$?
  local trailer_status=0
  local trailer_was_running=0
  trap - EXIT INT TERM
  if [[ -n "$TRAILER_PID" ]] && kill -0 "$TRAILER_PID" 2>/dev/null; then
    trailer_was_running=1
    kill "$TRAILER_PID" 2>/dev/null || true
  fi
  if [[ -n "$TRAILER_PID" ]]; then
    set +e
    wait "$TRAILER_PID" 2>/dev/null
    trailer_status=$?
    set -e
    if [[ "$trailer_was_running" == "0" && "$trailer_status" != "0" ]]; then
      echo "ERROR: trailer waypoint driver exited with status $trailer_status (log: $TRAILER_LOG)" >&2
      if [[ "$status" == "0" ]]; then
        status=6
      fi
    fi
  fi
  if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
    kill "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi
  if [[ -n "$MAVROS_PID" ]] && kill -0 "$MAVROS_PID" 2>/dev/null; then
    kill "$MAVROS_PID" 2>/dev/null || true
    wait "$MAVROS_PID" 2>/dev/null || true
  fi
  if [[ -n "$XRCE_PID" ]] && kill -0 "$XRCE_PID" 2>/dev/null; then
    kill "$XRCE_PID" 2>/dev/null || true
    wait "$XRCE_PID" 2>/dev/null || true
  fi
  if [[ -n "$GAZEBO_PID" ]] && kill -0 "$GAZEBO_PID" 2>/dev/null; then
    kill "$GAZEBO_PID" 2>/dev/null || true
    wait "$GAZEBO_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if gz topic -l 2>/dev/null | grep -q '^/world/.*/clock$'; then
  echo "ERROR: another Gazebo world is already active in GZ_PARTITION=$GZ_PARTITION." >&2
  echo "       Close it or choose a different GZ_PARTITION." >&2
  exit 4
fi

GZ_ARGS=(-v4 -r --physics-engine "${PHYSICS_ENGINE:-gz-physics-bullet-featherstone-plugin}")
if [[ "${HEADLESS:-0}" == "1" ]]; then
  GZ_ARGS+=(-s)
fi
GZ_ARGS+=("$@" "$WORLD_FILE")

echo "Map              : $MAP ($WORLD_NAME)"
echo "Coordinate YAML  : $COORDINATES"
echo "PX4 vehicle      : $SIM_MODEL / autostart $AUTOSTART_ID"
echo "Gazebo spawn ENU : $SPAWN_POSE"
echo "Expected entity  : $ENTITY_NAME"
echo "GZ_PARTITION     : $GZ_PARTITION"
echo "Gazebo log       : $GAZEBO_LOG"

gz sim "${GZ_ARGS[@]}" >"$GAZEBO_LOG" 2>&1 &
GAZEBO_PID=$!

ready=0
for _ in {1..90}; do
  if ! kill -0 "$GAZEBO_PID" 2>/dev/null; then
    echo "ERROR: Gazebo exited during startup. Last log lines:" >&2
    tail -80 "$GAZEBO_LOG" >&2 || true
    exit 5
  fi
  if gz service -i --service "/world/$WORLD_NAME/scene/info" 2>&1 | grep -q "Service providers"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  echo "ERROR: timed out waiting for Gazebo world '$WORLD_NAME'." >&2
  tail -80 "$GAZEBO_LOG" >&2 || true
  exit 5
fi

if [[ "${START_XRCE:-1}" == "1" ]]; then
  if pgrep -f 'MicroXRCEAgent.*udp4.*8888' >/dev/null 2>&1; then
    echo "Micro XRCE-DDS : reusing the existing UDP 8888 agent"
  elif [[ -x "$HOME/.local/bin/MicroXRCEAgent" ]]; then
    env LD_LIBRARY_PATH="$HOME/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$HOME/.local/bin/MicroXRCEAgent" udp4 -p 8888 >"$XRCE_LOG" 2>&1 &
    XRCE_PID=$!
    sleep 1
    if ! kill -0 "$XRCE_PID" 2>/dev/null; then
      echo "WARN: Micro XRCE-DDS Agent failed; PX4 still starts, but ROS 2 /fmu topics will be unavailable." >&2
      tail -20 "$XRCE_LOG" >&2 || true
      XRCE_PID=""
    else
      echo "Micro XRCE-DDS : UDP 8888 (log: $XRCE_LOG)"
    fi
  else
    echo "WARN: MicroXRCEAgent is not installed; PX4 flight works through MAVLink, but ROS 2 /fmu topics need the agent." >&2
  fi
fi

if [[ "${DRIVE_TRAILER:-0}" == "1" ]]; then
  python3 -c 'import gz.transport13, gz.msgs10.pose_v_pb2, yaml' >/dev/null 2>&1 || {
    echo "ERROR: DRIVE_TRAILER=1 needs Gazebo Harmonic Python transport bindings." >&2
    exit 6
  }
  TRAILER_ARGS=("$MAP" --loops "${TRAILER_ROUTE_LOOPS:-0}" --route "${TRAILER_ROUTE:-flat}")
  if [[ -n "${TRAILER_ROUTE_TIMEOUT:-}" ]]; then
    TRAILER_ARGS+=(--timeout "$TRAILER_ROUTE_TIMEOUT")
  fi
  python3 -u "$SCRIPT_DIR/trailer_waypoint_driver.py" "${TRAILER_ARGS[@]}" \
    > >(tee "$TRAILER_LOG") 2>&1 &
  TRAILER_PID=$!
  sleep 1
  if ! kill -0 "$TRAILER_PID" 2>/dev/null; then
    set +e
    wait "$TRAILER_PID"
    trailer_status=$?
    set -e
    TRAILER_PID=""
    echo "ERROR: trailer waypoint driver failed. Log:" >&2
    cat "$TRAILER_LOG" >&2 || true
    exit 6
  fi
  echo "Trailer route    : active (log: $TRAILER_LOG)"
else
  echo "Trailer route    : spawned, stationary (set DRIVE_TRAILER=1 to drive)"
fi

if [[ "${START_MAVROS:-1}" == "1" ]]; then
  MAVROS_FCU_URL="${MAVROS_FCU_URL:-udp://:14540@127.0.0.1:14580}"
  ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO:-humble}/setup.bash}"
  if [[ -f "$ROS_SETUP" ]]; then
    echo "MAVROS           : $MAVROS_FCU_URL (log: $MAVROS_LOG)"
    (
      set +u
      # shellcheck disable=SC1090
      source "$ROS_SETUP"
      exec ros2 launch mavros px4.launch "fcu_url:=$MAVROS_FCU_URL"
    ) >"$MAVROS_LOG" 2>&1 &
    MAVROS_PID=$!
    sleep 2
    if ! kill -0 "$MAVROS_PID" 2>/dev/null; then
      echo "WARN: MAVROS failed to start; PX4 flight still works. Log:" >&2
      tail -30 "$MAVROS_LOG" >&2 || true
      MAVROS_PID=""
    fi
  else
    echo "WARN: ROS 2 setup not found at $ROS_SETUP; skipping MAVROS." >&2
    echo "      Set ROS_SETUP=/path/to/setup.bash, or START_MAVROS=0 to silence." >&2
  fi
fi

if [[ "${START_BRIDGE:-1}" == "1" ]]; then
  ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO:-humble}/setup.bash}"
  BRIDGE_LAUNCH="$SCRIPT_DIR/launch/sensor_bridge.launch.py"
  if [[ -f "$ROS_SETUP" && -f "$BRIDGE_LAUNCH" ]]; then
    echo "ros_gz bridge    : world=$WORLD_NAME model=$ENTITY_NAME (log: $BRIDGE_LOG)"
    (
      set +u
      # shellcheck disable=SC1090
      source "$ROS_SETUP"
      exec ros2 launch "$BRIDGE_LAUNCH" "world:=$WORLD_NAME" "model:=$ENTITY_NAME"
    ) >"$BRIDGE_LOG" 2>&1 &
    BRIDGE_PID=$!
    sleep 2
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
      echo "WARN: ros_gz sensor bridge failed to start; simulation still runs. Log:" >&2
      tail -30 "$BRIDGE_LOG" >&2 || true
      BRIDGE_PID=""
    fi
  else
    echo "WARN: ros_gz bridge skipped (need $ROS_SETUP and $BRIDGE_LAUNCH)." >&2
    echo "      Set START_BRIDGE=0 to silence." >&2
  fi
fi

echo "Gazebo is ready. Starting the PX4 console; Ctrl-C stops this complete launch."
cd "$PX4_ROOTFS"
PX4_GZ_STANDALONE=1 \
PX4_GZ_WORLD="$WORLD_NAME" \
PX4_GZ_MODEL_POSE="$SPAWN_POSE" \
PX4_SYS_AUTOSTART="$AUTOSTART_ID" \
PX4_SIM_MODEL="$SIM_MODEL" \
"$PX4_BIN"
