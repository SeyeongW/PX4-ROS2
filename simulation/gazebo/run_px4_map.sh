#!/usr/bin/env bash
# Launch a checked-in map and dynamically spawn the real PX4 SITL vehicle.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAP="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi
MOVING_LANDING_PROFILE=0

case "$MAP" in
  mpc-landing)
    # Bounded flat baseline for downward-camera ArUco precision landing.
    COORDINATES="$SCRIPT_DIR/maps/mpc_landing_200m.yaml"
    # DART is the qualified engine for the fixed-joint RGB-D X500 payload.
    # Bullet Featherstone can trap the vehicle in the ground contact and then
    # eject it when takeoff thrust rises (see the Stage-10 SITL evidence).
    DEFAULT_PHYSICS_ENGINE="gz-physics-dartsim-plugin"
    ;;
  mpc-landing-moving)
    COORDINATES="$SCRIPT_DIR/maps/mpc_landing_200m_moving.yaml"
    DEFAULT_PHYSICS_ENGINE="gz-physics-dartsim-plugin"
    MOVING_LANDING_PROFILE=1
    ;;
  cju-track)
    COORDINATES="$SCRIPT_DIR/maps/drone_cju_track.yaml"
    DEFAULT_PHYSICS_ENGINE="gz-physics-dartsim-plugin"
    MOVING_LANDING_PROFILE=1
    ;;
  city)
    # The normal city profile is the self-contained 1300 m UAV derivative with
    # uniformly 2.5x-scaled centroids / footprints and 20--50 m heights
    # with an exact 35 m mean.
    COORDINATES="$SCRIPT_DIR/maps/city_coordinates_uav.yaml"
    DEFAULT_PHYSICS_ENGINE="gz-physics-dartsim-plugin"
    ;;
  mountain)
    COORDINATES="$SCRIPT_DIR/maps/mountain_coordinates.yaml"
    DEFAULT_PHYSICS_ENGINE="gz-physics-bullet-featherstone-plugin"
    ;;
  -h|--help|help|"")
    cat <<EOF
Usage: $(basename "$0") <mpc-landing|mpc-landing-moving|cju-track|city|mountain> [additional gz sim options]

Starts Gazebo Harmonic, waits for the selected world, then starts PX4 SITL.
PX4 itself creates the dynamic x500_city_rgbd_lidar vehicle (default x500 quad
with forward RGB/depth plus downward RGB/depth/lidar sensors).
A repository-owned trailer model is included in each map. The stationary
mpc-landing baseline cannot be driven; the default moving profile follows its
map-defined 1,000 m forward / 1,000 m no-turn reverse shuttle with
DRIVE_TRAILER=1.
cju-track selects the real-scale Cheongju University main-stadium world.

Environment:
  PX4_DIR=~/PX4-Autopilot  Existing PX4 source/build (firmware is not changed)
  HEADLESS=1              Gazebo server only
  START_XRCE=1            Enable Micro XRCE-DDS Agent/client (default: off)
  START_MAVROS=0         Do not start MAVROS (MAVLink->ROS 2 bridge)
  MAVROS_FCU_URL=...     Override MAVROS fcu_url (default udp://:14540@127.0.0.1:14580)
  MAVROS_HEARTBEAT_TYPE=GCS  MAVROS identity used for PX4 data-link health
  MAVROS_HEARTBEAT_RATE=10.0 Optional MAVROS heartbeat rate override in Hz
  START_BRIDGE=0         Do not start the sensor bridge (requires START_MAVROS=0)
  ROS_SETUP=...          ROS 2 setup.bash to source for MAVROS + bridge (default humble)
  FOLLOW_DRONE=1          Opt in to locking the gz camera to the drone (default: off)
  GIMBAL=1                Fly the map's gimbal_variant vehicle instead of the
                          baseline one (moving-landing map only): identical
                          sensors plus a slung, tilt-decoupled camera
  DRIVE_TRAILER=1         Drive the city, mountain, 1 km shuttle, or CJU route
  TRAILER_SPEED_M_S=...   Override the map-defined trailer speed for one run
  TRAILER_ROUTE_LOOPS=1   Stop the driver after one complete route (0=repeat)
  TRAILER_ROUTE=slope     Mountain-only terrain-follow safeguard test
  USE_NVIDIA=0            Disable NVIDIA PRIME render variables
  GZ_PARTITION=...        Gazebo transport partition (shared with PX4)
  PHYSICS_ENGINE=...      Override physics engine (mpc-landing/city default: DART)
  PX4_DAEMON=1            Disable the interactive PX4 shell (for log/CI runs)
EOF
    exit 0
    ;;
  *)
    echo "ERROR: unsupported map '$MAP'." >&2
    exit 2
    ;;
esac

COORDINATES="${PX4_MAP_COORDINATES:-$COORDINATES}"

for command in python3 gz timeout; do
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

# The map YAMLs spell their asset paths as `gazebo/worlds/...`, i.e. relative to
# whatever directory holds `gazebo/`.  That used to be the repo root; since the
# regroup it is `simulation/`.  Rebasing here keeps every YAML untouched and
# keeps them honest: they describe simulation assets, so being relative to
# simulation/ is what they actually mean.
mapfile -t MAP_CONFIG < <(
  python3 - "$COORDINATES" "$REPO_DIR/simulation" <<'PY'
import json
import math
import pathlib
import sys
import yaml

document = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pose = document["spawn"]["gazebo_spawn_pose_enu"]
keys = ("x", "y", "z", "roll", "pitch", "yaw")
parameter_ranges = {
    "MIS_TAKEOFF_ALT": (0.0, 100.0),
    "MPC_TKO_SPEED": (0.1, 5.0),
    "MPC_XY_CRUISE": (0.1, 20.0),
    "MPC_XY_VEL_MAX": (0.1, 20.0),
    "MPC_ACC_HOR": (0.1, 20.0),
    "MPC_JERK_AUTO": (0.1, 50.0),
    "MPC_LAND_SPEED": (0.1, 5.0),
    "MPC_LAND_CRWL": (0.1, 5.0),
    "LNDMC_Z_VEL_MAX": (0.01, 5.0),
    "LNDMC_XY_VEL_MAX": (0.01, 20.0),
    "COM_DISARM_LAND": (0.0, 60.0),
    "PLD_BTOUT": (0.0, 50.0),
    "PLD_HACC_RAD": (0.0, 10.0),
    "PLD_VEL_THR": (0.0, 5.0),
    "PLD_FAPPR_ALT": (0.0, 10.0),
    "PLD_SRCH_ALT": (0.0, 100.0),
    "PLD_SRCH_TOUT": (0.0, 100.0),
    "PLD_MAX_SRCH": (0, 100),
}
declared = document["px4_vehicle"].get("sitl_parameter_overrides", {})
overrides = {
    name: declared[name] for name in parameter_ranges if name in declared
}
for name, value in overrides.items():
    low, high = parameter_ranges[name]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not low <= value <= high):
        raise ValueError(f"invalid PX4 SITL parameter {name}={value!r}")
if "PLD_MAX_SRCH" in overrides and not isinstance(overrides["PLD_MAX_SRCH"], int):
    raise ValueError("PLD_MAX_SRCH must be an integer")
if overrides.get("MPC_XY_CRUISE", 0) > overrides.get("MPC_XY_VEL_MAX", math.inf):
    raise ValueError("MPC_XY_CRUISE must not exceed MPC_XY_VEL_MAX")
if overrides.get("MPC_LAND_CRWL", 0) > overrides.get("MPC_LAND_SPEED", math.inf):
    raise ValueError("MPC_LAND_CRWL must not exceed MPC_LAND_SPEED")
if ("LNDMC_Z_VEL_MAX" in overrides
        and "MPC_LAND_CRWL" in overrides
        and overrides["LNDMC_Z_VEL_MAX"] >= overrides["MPC_LAND_CRWL"]):
    raise ValueError("LNDMC_Z_VEL_MAX must be below MPC_LAND_CRWL")
print(document["map"]["gazebo_world_name"])
print(pathlib.Path(sys.argv[2], document["map"]["world_file"]).resolve())
print(",".join(str(pose[key]) for key in keys))
print(document["px4_vehicle"]["airframe_autostart_id"])
print(document["px4_vehicle"]["simulation_model"])
print(document["px4_vehicle"]["runtime_entity_name"])
trailer = document["trailer"]["spawn_pose_enu"]
print(",".join(str(trailer[key]) for key in keys))
gimbal = document["px4_vehicle"].get("gimbal_variant") or {}
print(gimbal.get("simulation_model", ""))
print(gimbal.get("runtime_entity_name", ""))
print(json.dumps(overrides, separators=(",", ":")))
PY
)
[[ ${#MAP_CONFIG[@]} -eq 10 ]] || {
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
GIMBAL_SIM_MODEL="${MAP_CONFIG[7]}"
GIMBAL_ENTITY_NAME="${MAP_CONFIG[8]}"
PX4_PARAM_OVERRIDES="${MAP_CONFIG[9]}"

if [[ "${GIMBAL:-0}" == "1" ]]; then
  [[ -n "$GIMBAL_SIM_MODEL" && -n "$GIMBAL_ENTITY_NAME" ]] || {
    echo "ERROR: GIMBAL=1 but $COORDINATES declares no px4_vehicle.gimbal_variant." >&2
    exit 2
  }
  SIM_MODEL="$GIMBAL_SIM_MODEL"
  ENTITY_NAME="$GIMBAL_ENTITY_NAME"
fi

if [[ "$MAP" == "mpc-landing" && "${DRIVE_TRAILER:-0}" != "0" ]]; then
  echo "ERROR: DRIVE_TRAILER is forbidden in the stationary mpc-landing baseline." >&2
  exit 2
fi

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
# The vehicle is whatever the coordinate contract names, but it must be one
# this checkout owns — that is what makes the symlink check below meaningful.
if [[ ! -f "$REPO_DIR/simulation/px4_models/$CUSTOM_MODEL/model.sdf" ]]; then
  echo "ERROR: coordinate contract selects '$CUSTOM_MODEL', which is not a repository model under $REPO_DIR/simulation/px4_models." >&2
  exit 3
fi
CUSTOM_MODEL_SOURCE="$(realpath -m "$REPO_DIR/simulation/px4_models/$CUSTOM_MODEL")"
CUSTOM_MODEL_TARGET="$PX4_DIR/Tools/simulation/gz/models/$CUSTOM_MODEL"
if [[ ! -L "$CUSTOM_MODEL_TARGET" || "$(realpath -m "$CUSTOM_MODEL_TARGET")" != "$CUSTOM_MODEL_SOURCE" ]]; then
  echo "ERROR: PX4 custom model must link to this checkout: $CUSTOM_MODEL_TARGET -> $CUSTOM_MODEL_SOURCE" >&2
  exit 3
fi
for model in "$CUSTOM_MODEL" x500 x500_base; do
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

export PX4_SIM_MODEL="$SIM_MODEL"
if [[ "$MAP" == "mpc-landing" ]]; then
  for asset in model.sdf model.config; do
    [[ -f "$SCRIPT_DIR/models/moving_platform_aruco/$asset" ]] || {
      echo "ERROR: landing trailer asset is missing: $SCRIPT_DIR/models/moving_platform_aruco/$asset" >&2
      exit 3
    }
  done
  [[ -f "$SCRIPT_DIR/models/aruco_marker_0/materials/textures/aruco_0.png" ]] || {
    echo "ERROR: ArUco ID 0 texture is missing" >&2
    exit 3
  }
fi
if [[ "$MAP" == "city" || "$MOVING_LANDING_PROFILE" == "1" ]]; then
  MOVING_MODELS=(moving_platform_aruco_velocity)
  if [[ "$MAP" == "cju-track" ]]; then
    MOVING_MODELS+=(drone_cju_track_stadium drone_cju_track_running_track)
  elif [[ "$MAP" == "mpc-landing-moving" ]]; then
    MOVING_MODELS+=(mpc_perimeter_patrol_field)
  fi
  for model in "${MOVING_MODELS[@]}"; do
    for asset in model.sdf model.config; do
      [[ -f "$SCRIPT_DIR/models/$model/$asset" ]] || {
        echo "ERROR: moving-landing asset is missing: $SCRIPT_DIR/models/$model/$asset" >&2
        exit 3
      }
    done
  done
  for marker_id in 0 1 2; do
    [[ -f "$SCRIPT_DIR/models/aruco_marker_${marker_id}/materials/textures/aruco_${marker_id}.png" ]] || {
      echo "ERROR: ArUco ID $marker_id texture is missing" >&2
      exit 3
    }
  done
  if [[ "$MAP" == "cju-track" ]]; then
    CJU_MESH_ASSETS=(
      drone_cju_track_running_track/meshes/running_track_surface.obj
      drone_cju_track_running_track/meshes/running_track_surface.mtl
    )
    for asset in "${CJU_MESH_ASSETS[@]}"; do
      [[ -f "$SCRIPT_DIR/models/$asset" ]] || {
        echo "ERROR: drone_cju mesh asset is missing: $SCRIPT_DIR/models/$asset" >&2
        exit 3
      }
    done
  fi
fi

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
WATCHDOG_PID=""
SPAWN_CHECK_PID=""

# PX4 SITL instance 0 owns a host-wide advisory lock.  Starting Gazebo first
# and discovering the conflict only when PX4 launches makes the GUI appear and
# then disappear, which looks like a spawn or physics crash.  Fail before any
# child process is created and print the actual conflicting process instead.
PX4_LOCK_PATH="/tmp/px4_lock-0"
if pgrep -f "$PX4_BIN" >/dev/null 2>&1; then
  echo "ERROR: PX4 SITL instance 0 is already running; refusing to start a second world." >&2
  pgrep -af "$PX4_BIN" >&2 || true
  echo "       Stop the existing PX4/run_px4_map.sh process, then retry." >&2
  exit 4
fi
if command -v ss >/dev/null 2>&1; then
  MAVLINK_PORT_USERS="$(ss -H -lunp 2>/dev/null | grep -E ':(14540|14580)([[:space:]]|$)' || true)"
  if [[ -n "$MAVLINK_PORT_USERS" ]]; then
    echo "ERROR: MAVROS/PX4 UDP port 14540 or 14580 is already in use:" >&2
    echo "$MAVLINK_PORT_USERS" >&2
    echo "       Stop the stale MAVROS/PX4 process, then retry." >&2
    exit 4
  fi
fi
if command -v flock >/dev/null 2>&1; then
  exec {PX4_LOCK_FD}>"$PX4_LOCK_PATH"
  if ! flock -n "$PX4_LOCK_FD"; then
    echo "ERROR: PX4 SITL instance 0 is already running; refusing to start a second world." >&2
    pgrep -af "$PX4_BIN" >&2 || true
    echo "       Stop the existing PX4/run_px4_map.sh process, then retry." >&2
    exec {PX4_LOCK_FD}>&-
    exit 4
  fi
  flock -u "$PX4_LOCK_FD"
  exec {PX4_LOCK_FD}>&-
fi

# PX4's POSIX SITL rcS starts uxrce_dds_client unconditionally.  In the
# MAVROS-only profile, use PX4's supported `-s` option with a runtime copy of
# rcS from which that single start command is removed.  This prevents the
# startup race (and its misleading "got no ping" errors) without modifying the
# user's PX4 source/build tree.  Fail closed if a future PX4 release changes
# the expected rcS line instead of silently filtering the wrong command.
MAVROS_RCS=""
if [[ "${START_XRCE:-0}" != "1" || "$PX4_PARAM_OVERRIDES" != "{}" ]]; then
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
  if [[ "${START_XRCE:-0}" != "1" ]]; then
    awk '!/^[[:space:]]*uxrce_dds_client[[:space:]]+start([[:space:]]|$)/' \
      "$PX4_STOCK_RCS" >"$MAVROS_RCS_TMP"
  else
    awk '{ print }' "$PX4_STOCK_RCS" >"$MAVROS_RCS_TMP"
  fi
  if [[ "$PX4_PARAM_OVERRIDES" != "{}" ]]; then
    printf '\n# Precision-landing SITL safety calibration (runtime only).\n' \
      >>"$MAVROS_RCS_TMP"
    python3 - "$PX4_PARAM_OVERRIDES" >>"$MAVROS_RCS_TMP" <<'PY'
import json
import sys

for name, value in json.loads(sys.argv[1]).items():
    print(f"param set {name} {value}")
PY
  fi
  if ! /bin/sh -n "$MAVROS_RCS_TMP"; then
    rm -f "$MAVROS_RCS_TMP"
    echo "ERROR: generated MAVROS-only PX4 startup script is invalid." >&2
    exit 3
  fi
  if [[ "${START_XRCE:-0}" != "1" ]] && \
      grep -Eq '^[[:space:]]*uxrce_dds_client[[:space:]]+start([[:space:]]|$)' "$MAVROS_RCS_TMP"; then
    rm -f "$MAVROS_RCS_TMP"
    echo "ERROR: generated MAVROS-only PX4 startup script still starts DDS." >&2
    exit 3
  fi
  mv -f "$MAVROS_RCS_TMP" "$MAVROS_RCS"
fi

collect_process_tree() {
  local parent="${1:?missing parent pid}"
  local child
  local children=""
  local children_file="/proc/$parent/task/$parent/children"
  # A short-lived child may disappear between the readability check and the
  # open. Treat that normal /proc race as an empty subtree during cleanup.
  if [[ -r "$children_file" ]]; then
    read -r children <"$children_file" 2>/dev/null || children=""
    for child in $children; do
      collect_process_tree "$child"
    done
  fi
  printf '%s\n' "$parent"
}

stop_process() {
  local pid="${1:-}"
  local process
  local -a process_tree=()
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    # ros2 launch and gz sim both create descendants.  Killing only their
    # launcher PID can orphan camera bridges and TF publishers, leading to
    # duplicate ROS publishers on the next run.
    mapfile -t process_tree < <(collect_process_tree "$pid")
    for process in "${process_tree[@]}"; do
      kill -TERM "$process" 2>/dev/null || true
    done
    for _ in {1..30}; do
      local any_alive=0
      for process in "${process_tree[@]}"; do
        if kill -0 "$process" 2>/dev/null; then
          any_alive=1
          break
        fi
      done
      [[ "$any_alive" == "0" ]] && break
      sleep 0.1
    done
    # Some launcher wrappers do not terminate on SIGTERM.  Never let Ctrl-C
    # leave a world or bridge tree running indefinitely.
    for process in "${process_tree[@]}"; do
      kill -KILL "$process" 2>/dev/null || true
    done
  fi
  wait "$pid" 2>/dev/null || true
}

stop_process_group() {
  local pgid="${1:-}"
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$pgid" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  wait "$pgid" 2>/dev/null || true
}

find_direct_px4_child() {
  local parent="${1:?missing parent pid}"
  local child
  local executable
  [[ -r "/proc/$parent/task/$parent/children" ]] || return 1
  for child in $(<"/proc/$parent/task/$parent/children"); do
    executable="$(readlink -f "/proc/$child/exe" 2>/dev/null || true)"
    if [[ "$executable" == "$PX4_BIN" ]]; then
      printf '%s\n' "$child"
      return 0
    fi
  done
  return 1
}

watch_gazebo_while_px4_runs() {
  local launcher_pid="${1:?missing launcher pid}"
  local gazebo_pid="${2:?missing Gazebo pid}"
  local px4_pid=""
  local gazebo_failed=0

  # This watcher starts just before the foreground PX4 command.  Locate that
  # direct child without changing PX4's access to the interactive terminal.
  for _ in {1..50}; do
    if px4_pid="$(find_direct_px4_child "$launcher_pid")"; then
      break
    fi
    if ! kill -0 "$gazebo_pid" 2>/dev/null; then
      gazebo_failed=1
    fi
    sleep 0.1
  done
  [[ -n "$px4_pid" ]] || return 0

  while kill -0 "$px4_pid" 2>/dev/null; do
    if [[ "$gazebo_failed" == "1" ]] || ! kill -0 "$gazebo_pid" 2>/dev/null; then
      echo "ERROR: Gazebo exited while PX4 was running; stopping the orphan PX4 session." >&2
      kill -INT "$px4_pid" 2>/dev/null || true
      for _ in {1..30}; do
        kill -0 "$px4_pid" 2>/dev/null || break
        sleep 0.1
      done
      kill -TERM "$px4_pid" 2>/dev/null || true
      return 7
    fi
    sleep 0.5
  done
  return 0
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
  stop_process "$SPAWN_CHECK_PID"
  stop_process "$WATCHDOG_PID"
  # The bridge is a dedicated session. Killing the whole process group also
  # catches children which ros2 launch briefly reparents while shutting down.
  stop_process_group "$BRIDGE_PID"
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

PHYSICS_ENGINE="${PHYSICS_ENGINE:-$DEFAULT_PHYSICS_ENGINE}"
GZ_ARGS=(-v4 -r --physics-engine "$PHYSICS_ENGINE")
if [[ "${HEADLESS:-0}" == "1" ]]; then
  GZ_ARGS+=(-s)
fi
GZ_ARGS+=("$@" "$WORLD_FILE")

echo "Map              : $MAP ($WORLD_NAME)"
echo "Coordinate YAML  : $COORDINATES"
echo "PX4 vehicle      : $SIM_MODEL / autostart $AUTOSTART_ID${GIMBAL:+ (GIMBAL=$GIMBAL)}"
echo "Gazebo spawn ENU : $SPAWN_POSE"
echo "Expected entity  : $ENTITY_NAME"
echo "GZ_PARTITION     : $GZ_PARTITION"
echo "Physics          : $PHYSICS_ENGINE"
if [[ "$MAP" == "city" ]]; then
  echo "Landing trailer  : deterministic three-marker velocity model"
  echo "Trailer route    : full-city black-road stop-turn patrol (DRIVE_TRAILER=1)"
elif [[ "$MAP" == "mpc-landing" ]]; then
  echo "Landing trailer  : measured odometry model held at 0 m/s"
  echo "Pair separation  : exactly 10 m on the northeast diagonal"
elif [[ "$MOVING_LANDING_PROFILE" == "1" ]]; then
  echo "Landing trailer  : deterministic velocity model"
  if [[ "$MAP" == "cju-track" ]]; then
    echo "World reference  : real-scale Cheongju University main stadium only"
  else
    echo "World reference  : 1,000 x 100 m forward/reverse shuttle"
  fi
  echo "Motion trigger   : PX4 entity confirmation, independent of mission/ArUco"
fi
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

# A user's persisted Gazebo GUI state can override the SDF camera pose. Apply
# one deterministic free-camera pose and explicitly clear tracking. The long
# shuttle uses a whole-course overview; the other maps retain a close spawn
# view. In either case the user can immediately pan / orbit without fighting a
# follow controller.
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
    if [[ "$MAP" == "mpc-landing-moving" ]]; then
      # Same overview as the world SDF: all 1,000 m of the shuttle remains
      # visible, but CameraTrack::NONE keeps the view fully user-controlled.
      camera_x=500
      camera_y=-683
      camera_z=667
      camera_qx=-0.2550412997
      camera_qy=0.2550413831
      camera_qz=0.6595102517
      camera_qw=0.6595104672
      free_camera_label="1 km overview free view"
    else
      IFS=, read -r spawn_x spawn_y spawn_z _ <<<"$SPAWN_POSE"
      read -r camera_x camera_y camera_z < <(
        python3 - "$spawn_x" "$spawn_y" "$spawn_z" <<'PY'
import sys
print(float(sys.argv[1]) - 4.0, float(sys.argv[2]), float(sys.argv[3]) + 3.0)
PY
      )
      # pitch=+0.45 rad points the Gazebo camera down toward the model root.
      camera_qx=0
      camera_qy=0.2231063621
      camera_qz=0
      camera_qw=0.9747941071
      free_camera_label="spawn-close free view"
    fi
    camera_reply="$(gz service -s /gui/move_to/pose \
      --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 5000 \
      --req "pose: {position: {x: $camera_x, y: $camera_y, z: $camera_z}, orientation: {x: $camera_qx, y: $camera_qy, z: $camera_qz, w: $camera_qw}}" 2>&1 || true)"
    # CameraTrack::NONE is idempotent and prevents a stale persisted target.
    gz topic -t /gui/track -m gz.msgs.CameraTrack -p 'track_mode: NONE' \
      >/dev/null 2>&1 || true
    if grep -q "data: true" <<<"$camera_reply"; then
      echo "Gazebo camera    : $free_camera_label (tracking disabled)"
    else
      echo "WARN: could not apply the map's free-camera pose." >&2
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
    echo "WARN: /gui/follow was not advertised; using the map's free-camera fallback view." >&2
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
  if [[ "$MAP" == "mpc-landing" ]]; then
    echo "ERROR: the $MAP trailer is intentionally stationary; leave DRIVE_TRAILER=0." >&2
    exit 6
  fi
  if [[ "$MAP" == "city" && -z "${TRAILER_START_FILE:-}" \
      && "${START_MAVROS:-1}" != "1" ]]; then
    echo "ERROR: the city trailer takeoff gate needs START_MAVROS=1." >&2
    exit 6
  fi
  python3 -c 'import gz.transport13, gz.msgs10.pose_v_pb2, yaml' >/dev/null 2>&1 || {
    echo "ERROR: DRIVE_TRAILER=1 needs Gazebo Harmonic Python transport bindings." >&2
    exit 6
  }
  TRAILER_ARGS=("$MAP" --coordinates "$COORDINATES" \
    --loops "${TRAILER_ROUTE_LOOPS:-0}" --route "${TRAILER_ROUTE:-flat}")
  if [[ "$MAP" == "city" || "$MOVING_LANDING_PROFILE" == "1" ]]; then
    # The model root is already calibrated to z=0 for the 2.051 m marker
    # surface. Do not apply the mountain driver's terrain-clearance offset.
    TRAILER_ARGS+=(--no-terrain-follow --rate "${TRAILER_COMMAND_RATE_HZ:-50}")
    if [[ -n "${TRAILER_SPEED_M_S:-}" ]]; then
      TRAILER_ARGS+=(--speed "$TRAILER_SPEED_M_S")
    fi
  fi
  if [[ -n "${TRAILER_ROUTE_TIMEOUT:-}" ]]; then
    TRAILER_ARGS+=(--timeout "$TRAILER_ROUTE_TIMEOUT")
  fi
  echo "Trailer route    : waiting for PX4 entity (log: $TRAILER_LOG)"
else
  if [[ "$MAP" == "city" ]]; then
    echo "Trailer          : stationary at patrol start (set DRIVE_TRAILER=1)"
  elif [[ "$MAP" == "mpc-landing" ]]; then
    echo "Trailer          : static at ENU ($TRAILER_SPAWN_POSE)"
  elif [[ "$MOVING_LANDING_PROFILE" == "1" ]]; then
    echo "Trailer          : stationary because DRIVE_TRAILER=0"
  else
    echo "Trailer route    : spawned, stationary (set DRIVE_TRAILER=1 to drive)"
  fi
fi

if [[ "${START_MAVROS:-1}" == "1" ]]; then
  MAVROS_FCU_URL="${MAVROS_FCU_URL:-udp://:14540@127.0.0.1:14580}"
  MAVROS_HEARTBEAT_TYPE="${MAVROS_HEARTBEAT_TYPE:-GCS}"
  MAVROS_HEARTBEAT_RATE="${MAVROS_HEARTBEAT_RATE:-}"
  ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO:-humble}/setup.bash}"
  MAVROS_LAUNCH="$SCRIPT_DIR/launch/mavros_px4_sitl.launch.py"
  MAVROS_CONTRACT="$SCRIPT_DIR/tools/configure_mavros_sitl.py"
  [[ -f "$MAVROS_LAUNCH" ]] || {
    echo "ERROR: repository MAVROS SITL launch is missing: $MAVROS_LAUNCH" >&2
    exit 3
  }
  [[ -f "$MAVROS_CONTRACT" ]] || {
    echo "ERROR: MAVROS SITL contract probe is missing: $MAVROS_CONTRACT" >&2
    exit 3
  }
  if [[ -f "$ROS_SETUP" ]]; then
    echo "MAVROS           : $MAVROS_FCU_URL (log: $MAVROS_LOG)"
    (
      set +u
      # shellcheck disable=SC1090
      source "$ROS_SETUP"
      exec ros2 launch "$MAVROS_LAUNCH" "fcu_url:=$MAVROS_FCU_URL"
    ) >"$MAVROS_LOG" 2>&1 &
    MAVROS_PID=$!
    sleep 2
    if ! kill -0 "$MAVROS_PID" 2>/dev/null; then
      echo "ERROR: requested MAVROS failed to start. Log:" >&2
      tail -30 "$MAVROS_LOG" >&2 || true
      exit 5
    else
      # PX4 advances on Gazebo lockstep time.  Verify the repository wildcard
      # reached the UAS and representative data/time plugins before PX4 can
      # emit a single state sample.  Otherwise MAVROS periodically switches
      # its headers between Unix epoch and PX4 boot time as its filter resets.
      contract_arguments=(
        --timeout 30
        --heartbeat-type "$MAVROS_HEARTBEAT_TYPE"
      )
      if [[ -n "$MAVROS_HEARTBEAT_RATE" ]]; then
        contract_arguments+=(--heartbeat-rate "$MAVROS_HEARTBEAT_RATE")
      fi
      if ! (
        set +u
        # shellcheck disable=SC1090
        source "$ROS_SETUP"
        python3 "$MAVROS_CONTRACT" "${contract_arguments[@]}"
      ); then
        echo "ERROR: MAVROS SITL clock contract failed; refusing to start PX4." >&2
        tail -30 "$MAVROS_LOG" >&2 || true
        exit 5
      fi
    fi
  else
    echo "ERROR: requested MAVROS needs ROS 2 setup at $ROS_SETUP." >&2
    echo "       Set ROS_SETUP=/path/to/setup.bash, or START_MAVROS=0 to skip." >&2
    exit 3
  fi
fi

if [[ "${START_BRIDGE:-1}" == "1" ]]; then
  ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO:-humble}/setup.bash}"
  BRIDGE_LAUNCH="$SCRIPT_DIR/launch/sensor_bridge.launch.py"
  if [[ -f "$ROS_SETUP" && -f "$BRIDGE_LAUNCH" ]]; then
    # On a gimbal run nothing consumes the front/down/depth/lidar streams, so
    # bridge only the gimbal camera — that is the traffic that was lagging the
    # debug image.  Override with BRIDGE_GIMBAL_ONLY=0 to bridge everything.
    # MUST be a bool literal (true/false): the bridge declares gimbal_only as
    # BOOL, and `-p gimbal_only:=1` is parsed as INTEGER and kills the node on
    # startup — which takes /clock and the whole mission down with it.
    case "${BRIDGE_GIMBAL_ONLY:-${GIMBAL:-0}}" in
      1|true|yes|on) BRIDGE_GIMBAL_ONLY=true ;;
      *)             BRIDGE_GIMBAL_ONLY=false ;;
    esac
    echo "ros_gz bridge    : world=$WORLD_NAME model=$ENTITY_NAME gimbal_only=$BRIDGE_GIMBAL_ONLY (log: $BRIDGE_LOG)"
    setsid bash -c '
      set +u
      # shellcheck disable=SC1090
      source "$1"
      exec ros2 launch "$2" "world:=$3" "model:=$4" "gimbal_only:=$5"
    ' bridge "$ROS_SETUP" "$BRIDGE_LAUNCH" "$WORLD_NAME" "$ENTITY_NAME" \
      "$BRIDGE_GIMBAL_ONLY" >"$BRIDGE_LOG" 2>&1 &
    BRIDGE_PID=$!
    sleep 2
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
      echo "WARN: ros_gz sensor bridge failed to start; simulation still runs. Log:" >&2
      tail -30 "$BRIDGE_LOG" >&2 || true
      BRIDGE_PID=""
    fi
  else
    echo "WARN: ros_gz bridge skipped (need $ROS_SETUP and $BRIDGE_LAUNCH)." >&2
    echo "      Set START_BRIDGE=0 to silence (also set START_MAVROS=0)." >&2
  fi
fi

# A sim-time MAVROS must see a live Gazebo clock before the FCU connects.  The
# sensor bridge owns that clock conversion in this checkout.  Fail closed
# instead of allowing zero-time headers or a later epoch/boot-time rebase.
if [[ -n "$MAVROS_PID" ]]; then
  if [[ "${START_BRIDGE:-1}" != "1" || -z "$BRIDGE_PID" ]]; then
    echo "ERROR: START_MAVROS=1 requires the /clock sensor bridge in SITL." >&2
    exit 5
  fi
  if ! (
    set +u
    # shellcheck disable=SC1090
    source "$ROS_SETUP"
    timeout 10 ros2 topic echo --qos-reliability best_effort \
      --once /clock >/dev/null 2>&1
  ); then
    echo "ERROR: Gazebo /clock was not received; refusing to start PX4." >&2
    exit 5
  fi
  echo "Gazebo clock     : live before PX4 startup"
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
if [[ -n "$MAVROS_RCS" ]]; then
  PX4_ARGS+=(-s "$MAVROS_RCS")
fi
if [[ "${START_XRCE:-0}" != "1" ]]; then
  echo "PX4 transport     : MAVROS/MAVLink only (DDS disabled before rcS startup)"
else
  echo "PX4 transport     : MAVLink + Micro XRCE-DDS"
fi
if [[ "$PX4_PARAM_OVERRIDES" != "{}" ]]; then
  echo "PX4 native params : $(python3 - "$PX4_PARAM_OVERRIDES" <<'PY'
import json
import sys

print(" ".join(f"{name}={value}" for name, value in json.loads(sys.argv[1]).items()))
PY
)"
fi
LAUNCHER_PID=$$

# PX4 creates the model only after it connects to the already-running world.
# Confirm the expected entity explicitly so a failed or off-screen spawn is
# reported as such instead of being confused with the independent GCS check.
(
  for _ in {1..45}; do
    if gz model --list 2>/dev/null | grep -Fq -- "$ENTITY_NAME"; then
      echo "PX4 spawn check  : confirmed $ENTITY_NAME"
      exit 0
    fi
    sleep 1
  done
  echo "ERROR: PX4 did not create Gazebo entity '$ENTITY_NAME' within 45 seconds." >&2
  echo "       Gazebo log: $GAZEBO_LOG" >&2
  tail -80 "$GAZEBO_LOG" >&2 || true
  kill -TERM "$LAUNCHER_PID" 2>/dev/null || true
  exit 1
) &
SPAWN_CHECK_PID=$!

if [[ "${DRIVE_TRAILER:-0}" == "1" ]]; then
  # Keep the map-defined initial spawn geometry until the aircraft exists.
  # This avoids giving the trailer a head start during PX4 startup. The CJU
  # word-command launcher additionally supplies a gate held until `takeoff`.
  (
    for _ in {1..45}; do
      if gz model --list 2>/dev/null | grep -Fq -- "$ENTITY_NAME"; then
        if [[ -n "${TRAILER_START_FILE:-}" ]]; then
          echo "Trailer route    : PX4 entity confirmed; waiting for takeoff clearance"
          while [[ ! -e "$TRAILER_START_FILE" ]]; do sleep 0.1; done
        elif [[ "$MAP" == "city" ]]; then
          echo "Trailer route    : PX4 entity confirmed; waiting for vehicle takeoff"
          set +u
          # shellcheck disable=SC1090
          source "$ROS_SETUP"
          set -u
          ros2 topic echo --filter 'm.landed_state in (2, 3)' --once \
            /mavros/extended_state mavros_msgs/msg/ExtendedState >/dev/null
        fi
        echo "Trailer route    : starting"
        exec python3 -u "$SCRIPT_DIR/trailer_waypoint_driver.py" \
          "${TRAILER_ARGS[@]}"
      fi
      sleep 1
    done
    echo "ERROR: trailer route did not see PX4 entity within 45 seconds." >&2
    exit 6
  ) >"$TRAILER_LOG" 2>&1 &
  TRAILER_PID=$!
fi

watch_gazebo_while_px4_runs "$LAUNCHER_PID" "$GAZEBO_PID" &
WATCHDOG_PID=$!
set +e
PX4_GZ_STANDALONE=1 \
PX4_GZ_WORLD="$WORLD_NAME" \
PX4_GZ_MODEL_POSE="$SPAWN_POSE" \
PX4_SYS_AUTOSTART="$AUTOSTART_ID" \
PX4_SIM_MODEL="$SIM_MODEL" \
"$PX4_BIN" "${PX4_ARGS[@]}"
PX4_STATUS=$?
wait "$WATCHDOG_PID"
WATCHDOG_STATUS=$?
WATCHDOG_PID=""
set -e
if [[ "$WATCHDOG_STATUS" == "7" ]]; then
  exit 5
fi
if [[ "$PX4_STATUS" != "0" ]]; then
  echo "ERROR: PX4 SITL exited with status $PX4_STATUS; Gazebo and ROS bridges will now stop." >&2
  echo "       Check that no other PX4 instance owns $PX4_LOCK_PATH." >&2
  exit "$PX4_STATUS"
fi
