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
    # The normal city profile is the self-contained 1260 m UAV derivative with
    # rolled-back jo 2.5x centroids / 0.9x footprints and 10--20 m heights.
    # The source 500 m asset remains available explicitly for regression work.
    COORDINATES="$SCRIPT_DIR/maps/city_coordinates_uav.yaml"
    ;;
  city-legacy)
    COORDINATES="$SCRIPT_DIR/maps/city_coordinates.yaml"
    ;;
  mountain)
    COORDINATES="$SCRIPT_DIR/maps/mountain_coordinates.yaml"
    ;;
  -h|--help|help|"")
    cat <<EOF
Usage: $(basename "$0") <city|city-legacy|mountain> [additional gz sim options]

Starts Gazebo Harmonic, waits for the selected world, then starts PX4 SITL.
PX4 itself creates the dynamic x500_city_rgbd_lidar vehicle (default x500 quad
with forward RGB/depth plus downward RGB/depth/lidar sensors).
A repository-owned trailer model is included in each map. The city trailer is
spawn-only; DRIVE_TRAILER=1 is available for the mountain profile.

Environment:
  PX4_DIR=~/PX4-Autopilot  Existing PX4 source/build (firmware is not changed)
  HEADLESS=1              Gazebo server only
  START_XRCE=1            Enable Micro XRCE-DDS Agent/client (default: off)
  START_MAVROS=0         Do not start MAVROS (MAVLink->ROS 2 bridge)
  MAVROS_FCU_URL=...     Override MAVROS fcu_url (default udp://:14540@127.0.0.1:14580)
  START_BRIDGE=0         Do not start the ros_gz camera/lidar bridge
  ROS_SETUP=...          ROS 2 setup.bash to source for MAVROS + bridge (default humble)
  FOLLOW_DRONE=1          Opt in to locking the gz camera to the drone (default: off)
  DRIVE_TRAILER=1         Mountain only: drive the trailer through YAML waypoints
  TRAILER_ROUTE_LOOPS=1   Stop the driver after one complete route (0=repeat)
  TRAILER_ROUTE=slope     Mountain-only terrain-follow safeguard test
  USE_NVIDIA=0            Disable NVIDIA PRIME render variables
  GZ_PARTITION=...        Gazebo transport partition (shared with PX4)
  PHYSICS_ENGINE=...      Override physics engine (default: DART)
  PX4_DAEMON=1            Disable the interactive PX4 shell (for log/CI runs)
EOF
    exit 0
    ;;
  *)
    echo "ERROR: expected 'city', 'city-legacy' or 'mountain', got '$MAP'." >&2
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
trailer = document["trailer"]["spawn_pose_enu"]
print(",".join(str(trailer[key]) for key in keys))
PY
)
[[ ${#MAP_CONFIG[@]} -eq 7 ]] || {
  echo "ERROR: failed to read launch values from $COORDINATES" >&2
  exit 2
}
WORLD_NAME="${MAP_CONFIG[0]}"
WORLD_FILE="${MAP_CONFIG[1]}"
SPAWN_POSE="${MAP_CONFIG[2]}"
AUTOSTART_ID="${MAP_CONFIG[3]}"
SIM_MODEL="${MAP_CONFIG[4]}"
ENTITY_NAME="${MAP_CONFIG[5]}"
TRAILER_SPAWN_POSE="${MAP_CONFIG[6]}"

PX4_DIR="$(realpath -m "${PX4_DIR:-$HOME/PX4-Autopilot}")"
PX4_BUILD="$PX4_DIR/build/px4_sitl_default"
PX4_BIN="$PX4_BUILD/bin/px4"
PX4_ROOTFS="$PX4_BUILD/rootfs"
PX4_ENV="$PX4_ROOTFS/gz_env.sh"

if [[ ! -x "$PX4_BIN" || ! -f "$PX4_ENV" ]]; then
  cat >&2 <<EOF
ERROR: compatible PX4 SITL build was not found under:
       $PX4_DIR

This launcher never changes tracked firmware sources. It only manages the
repository model's unique symlink. For a fresh PC, run
./gazebo/setup_px4_sitl.sh once, then retry this command.
EOF
  exit 3
fi
"$SCRIPT_DIR/link_px4_model.sh" "$PX4_DIR"
CUSTOM_MODEL="${SIM_MODEL#gz_}"
EXPECTED_CUSTOM_MODEL="x500_city_rgbd_lidar"
if [[ "$CUSTOM_MODEL" != "$EXPECTED_CUSTOM_MODEL" ]]; then
  echo "ERROR: coordinate contract selects '$CUSTOM_MODEL'; expected repository model '$EXPECTED_CUSTOM_MODEL'." >&2
  exit 3
fi
CUSTOM_MODEL_SOURCE="$(realpath -m "$REPO_DIR/px4_models/$CUSTOM_MODEL")"
CUSTOM_MODEL_TARGET="$PX4_DIR/Tools/simulation/gz/models/$CUSTOM_MODEL"
if [[ ! -L "$CUSTOM_MODEL_TARGET" || "$(realpath -m "$CUSTOM_MODEL_TARGET")" != "$CUSTOM_MODEL_SOURCE" ]]; then
  echo "ERROR: PX4 custom model must link to this checkout: $CUSTOM_MODEL_TARGET -> $CUSTOM_MODEL_SOURCE" >&2
  exit 3
fi
for model in "$CUSTOM_MODEL" x500 x500_base OakD-Lite LW20; do
  [[ -f "$PX4_DIR/Tools/simulation/gz/models/$model/model.sdf" ]] || {
    echo "ERROR: PX4 Gazebo model asset is missing: $model" >&2
    if [[ "$model" == "$CUSTOM_MODEL" ]]; then
      cat >&2 <<EOF
       Install the checked-in custom model into your PX4 tree first:
         "$SCRIPT_DIR/link_px4_model.sh" "$PX4_DIR"
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
  # Keep Qt Quick and OGRE2 on the same GLX path on PRIME laptops.  Without
  # this, Qt may create a Mesa EGL surface while OGRE renders on NVIDIA.
  export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-xcb_glx}"
fi
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi
if [[ "${HEADLESS:-0}" != "1" && "${FOLLOW_DRONE:-0}" == "1" ]]; then
  export PX4_GZ_FOLLOW=1
  export PX4_GZ_FOLLOW_OFFSET_X="${PX4_GZ_FOLLOW_OFFSET_X:--6.0}"
  export PX4_GZ_FOLLOW_OFFSET_Y="${PX4_GZ_FOLLOW_OFFSET_Y:--6.0}"
  export PX4_GZ_FOLLOW_OFFSET_Z="${PX4_GZ_FOLLOW_OFFSET_Z:-4.0}"
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

# PX4's POSIX SITL rcS starts uxrce_dds_client unconditionally.  In the
# MAVROS-only profile, use PX4's supported `-s` option with a runtime copy of
# rcS from which that single start command is removed.  This prevents the
# startup race (and its misleading "got no ping" errors) without modifying the
# user's PX4 source/build tree.  Fail closed if a future PX4 release changes
# the expected rcS line instead of silently filtering the wrong command.
MAVROS_RCS=""
if [[ "${START_XRCE:-0}" != "1" ]]; then
  PX4_STOCK_RCS="$PX4_BUILD/etc/init.d-posix/rcS"
  MAVROS_RCS="$RUNTIME_DIR/rcS.mavros_only"
  MAVROS_RCS_TMP="$MAVROS_RCS.tmp.$$"
  [[ -f "$PX4_STOCK_RCS" ]] || {
    echo "ERROR: PX4 startup script is missing: $PX4_STOCK_RCS" >&2
    exit 3
  }
  DDS_START_COUNT="$(grep -Ec '^[[:space:]]*uxrce_dds_client[[:space:]]+start([[:space:]]|$)' "$PX4_STOCK_RCS" || true)"
  [[ "$DDS_START_COUNT" == "1" ]] || {
    echo "ERROR: expected exactly one uxrce_dds_client start line in $PX4_STOCK_RCS; found $DDS_START_COUNT." >&2
    exit 3
  }
  awk '!/^[[:space:]]*uxrce_dds_client[[:space:]]+start([[:space:]]|$)/' \
    "$PX4_STOCK_RCS" >"$MAVROS_RCS_TMP"
  if ! /bin/sh -n "$MAVROS_RCS_TMP"; then
    rm -f "$MAVROS_RCS_TMP"
    echo "ERROR: generated MAVROS-only PX4 startup script is invalid." >&2
    exit 3
  fi
  if grep -Eq '^[[:space:]]*uxrce_dds_client[[:space:]]+start([[:space:]]|$)' "$MAVROS_RCS_TMP"; then
    rm -f "$MAVROS_RCS_TMP"
    echo "ERROR: generated MAVROS-only PX4 startup script still starts DDS." >&2
    exit 3
  fi
  mv -f "$MAVROS_RCS_TMP" "$MAVROS_RCS"
fi

stop_process() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    # Some gz-sim launcher wrappers do not terminate on SIGTERM.  Never let
    # Ctrl-C leave a world running or make this cleanup wait indefinitely.
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
}

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
  stop_process "$BRIDGE_PID"
  stop_process "$MAVROS_PID"
  stop_process "$XRCE_PID"
  stop_process "$GAZEBO_PID"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if gz topic -l 2>/dev/null | grep -q '^/world/.*/clock$'; then
  echo "ERROR: another Gazebo world is already active in GZ_PARTITION=$GZ_PARTITION." >&2
  echo "       Close it or choose a different GZ_PARTITION." >&2
  exit 4
fi

PHYSICS_ENGINE="${PHYSICS_ENGINE:-gz-physics-dartsim-plugin}"
GZ_ARGS=(-v4 -r --physics-engine "$PHYSICS_ENGINE")
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
echo "Physics          : $PHYSICS_ENGINE"
if [[ "${HEADLESS:-0}" == "1" ]]; then
  echo "Gazebo camera    : headless"
elif [[ -n "${PX4_GZ_FOLLOW:-}" ]]; then
  echo "Gazebo camera    : follow $ENTITY_NAME"
else
  echo "Gazebo camera    : free"
fi
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

# A user's persisted Gazebo GUI state can override the SDF camera pose. For the
# default free-camera profile, move once to a close spawn view and explicitly
# clear tracking. The vehicle remains visible when it appears, while the user
# can immediately pan / orbit away without fighting a follow controller.
if [[ "${HEADLESS:-0}" != "1" && -z "${PX4_GZ_FOLLOW:-}" ]]; then
  free_camera_ready=0
  for _ in {1..45}; do
    if gz service -i --service /gui/move_to/pose 2>&1 | grep -q "Service providers"; then
      free_camera_ready=1
      break
    fi
    sleep 1
  done
  if [[ "$free_camera_ready" == "1" ]]; then
    IFS=, read -r spawn_x spawn_y spawn_z _ <<<"$SPAWN_POSE"
    read -r camera_x camera_y camera_z < <(
      python3 - "$spawn_x" "$spawn_y" "$spawn_z" <<'PY'
import sys
print(float(sys.argv[1]) - 4.0, float(sys.argv[2]), float(sys.argv[3]) + 3.0)
PY
    )
    # pitch=+0.45 rad points the Gazebo camera down toward the model root.
    camera_reply="$(gz service -s /gui/move_to/pose \
      --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 5000 \
      --req "pose: {position: {x: $camera_x, y: $camera_y, z: $camera_z}, orientation: {x: 0, y: 0.2231063621, z: 0, w: 0.9747941071}}" 2>&1 || true)"
    # CameraTrack::NONE is idempotent and prevents a stale persisted target.
    gz topic -t /gui/track -m gz.msgs.CameraTrack -p 'track_mode: NONE' \
      >/dev/null 2>&1 || true
    if grep -q "data: true" <<<"$camera_reply"; then
      echo "Gazebo camera    : spawn-close free view (tracking disabled)"
    else
      echo "WARN: could not apply the spawn-close free camera pose." >&2
    fi
  else
    echo "WARN: /gui/move_to/pose was not advertised; using the world camera pose." >&2
  fi
fi

# In opt-in follow mode, PX4 requests /gui/follow during its Gazebo startup
# script. On a cold GUI start that service can appear after the world service.
if [[ -n "${PX4_GZ_FOLLOW:-}" ]]; then
  follow_ready=0
  for _ in {1..45}; do
    if ! kill -0 "$GAZEBO_PID" 2>/dev/null; then
      echo "ERROR: Gazebo exited while waiting for the camera follow service." >&2
      tail -80 "$GAZEBO_LOG" >&2 || true
      exit 5
    fi
    if gz service -i --service /gui/follow 2>&1 | grep -q "Service providers"; then
      follow_ready=1
      break
    fi
    sleep 1
  done
  if [[ "$follow_ready" == "1" ]]; then
    echo "Gazebo follow     : service ready"
  else
    echo "WARN: /gui/follow was not advertised; using the spawn-close fallback view." >&2
  fi
fi

if [[ "${START_XRCE:-0}" == "1" ]]; then
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
  if [[ "$MAP" == "city" ]]; then
    echo "ERROR: the city trailer is intentionally spawn-only; leave DRIVE_TRAILER=0." >&2
    exit 6
  fi
  python3 -c 'import gz.transport13, gz.msgs10.pose_v_pb2, yaml' >/dev/null 2>&1 || {
    echo "ERROR: DRIVE_TRAILER=1 needs Gazebo Harmonic Python transport bindings." >&2
    exit 6
  }
  TRAILER_MAP="$MAP"
  if [[ "$TRAILER_MAP" == "city-legacy" ]]; then
    TRAILER_MAP="city"
  fi
  TRAILER_ARGS=("$TRAILER_MAP" --coordinates "$COORDINATES" \
    --loops "${TRAILER_ROUTE_LOOPS:-0}" --route "${TRAILER_ROUTE:-flat}")
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
  if [[ "$MAP" == "city" ]]; then
    echo "Trailer          : spawned at ENU ($TRAILER_SPAWN_POSE), stationary"
  else
    echo "Trailer route    : spawned, stationary (set DRIVE_TRAILER=1 to drive)"
  fi
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
    else
      # PX4's data-link arming check expects a GCS heartbeat. MAVROS defaults
      # to ONBOARD_CONTROLLER, which leaves gcs_connection_lost=true even
      # though MAVLink is connected and causes ordinary arm to be rejected.
      if (
        set +u
        # shellcheck disable=SC1090
        source "$ROS_SETUP"
        for _ in {1..30}; do
          ros2 param set /mavros/sys heartbeat_mav_type GCS >/dev/null 2>&1 && exit 0
          sleep 0.2
        done
        exit 1
      ); then
        echo "MAVROS heartbeat : GCS (PX4 arming data-link check enabled)"
      else
        echo "WARN: could not set MAVROS heartbeat type to GCS." >&2
      fi
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
PX4_ARGS=()
if [[ "${PX4_DAEMON:-0}" == "1" ]]; then
  # `px4 -d` keeps the SITL server in the foreground but omits the pxh shell.
  # Without this, redirecting a non-interactive shell to a file can emit ANSI
  # prompt refreshes indefinitely and consume gigabytes in a few minutes.
  PX4_ARGS+=(-d)
fi
if [[ "${START_XRCE:-0}" != "1" ]]; then
  PX4_ARGS+=(-s "$MAVROS_RCS")
  echo "PX4 transport     : MAVROS/MAVLink only (DDS disabled before rcS startup)"
else
  echo "PX4 transport     : MAVLink + Micro XRCE-DDS"
fi
PX4_GZ_STANDALONE=1 \
PX4_GZ_WORLD="$WORLD_NAME" \
PX4_GZ_MODEL_POSE="$SPAWN_POSE" \
PX4_SYS_AUTOSTART="$AUTOSTART_ID" \
PX4_SIM_MODEL="$SIM_MODEL" \
"$PX4_BIN" "${PX4_ARGS[@]}"
