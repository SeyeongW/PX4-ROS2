#!/usr/bin/env bash
#
# Launch the PX4 truck-mission world (Gazebo Harmonic only -- PX4 SITL itself
# is started separately, see run_px4_sitl.sh in this same directory).
#
# Why two scripts instead of one (like the ArduPilot run_sim.sh): PX4's own
# startup script (px4-rc.gzsim) launches Gazebo FOR you when PX4_GZ_STANDALONE
# is unset, but only for worlds living inside PX4-Autopilot's own
# Tools/simulation/gz/worlds -- it rebuilds the path as
# "$PX4_GZ_WORLDS/$PX4_GZ_WORLD.sdf" and PX4_GZ_WORLDS gets re-exported to that
# fixed path every boot (gz_env.sh), clobbering any override. So for OUR OWN
# world file (gazebo/worlds/truck_mission_px4.sdf, with the ArUco-decaled
# moving_platform_aruco model) we start Gazebo ourselves and tell PX4 to
# attach in standalone mode instead (PX4_GZ_STANDALONE=1 in run_px4_sitl.sh).
#
# Usage:
#   ./gazebo/run_sim_px4.sh                                    # GUI, constant-velocity platform
#   HEADLESS=1 ./gazebo/run_sim_px4.sh                          # server only
#   HEADLESS=1 WORLD=truck_mission_px4_track ./gazebo/run_sim_px4.sh   # vision track-following platform
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLD="${WORLD:-truck_mission_px4}"
PX4_HOME="${PX4_HOME:-$HOME/PX4-Autopilot}"
PX4_SITL_ROOTFS="$PX4_HOME/build/px4_sitl_default/rootfs"

if [[ ! -f "$PX4_SITL_ROOTFS/gz_env.sh" ]]; then
  echo "ERROR: $PX4_SITL_ROOTFS/gz_env.sh not found -- build PX4 first" >&2
  echo "       (cd $PX4_HOME && make px4_sitl_default)" >&2
  exit 1
fi

# gz-transport on loopback (same reasoning as the ArduPilot run_sim.sh: extra
# host interfaces make gz-transport's default pick unreachable to peers).
export GZ_IP="${GZ_IP:-127.0.0.1}"

# MovingPlatformController (loaded by OUR gz sim process, not PX4's) reads
# PX4_SIM_MODEL itself to build the vehicle model name it should wait for
# ("<model>_<px4_instance>", stripping the "gz_" prefix) before it starts
# moving the platform -- must match what run_px4_sitl.sh spawns.
export PX4_SIM_MODEL="${PX4_SIM_MODEL:-gz_x500_mono_cam_down}"

# Pulls in PX4_GZ_MODELS/WORLDS/PLUGINS/SERVER_CONFIG and the matching
# GZ_SIM_RESOURCE_PATH / GZ_SIM_SYSTEM_PLUGIN_PATH / GZ_SIM_SERVER_CONFIG_PATH
# -- without GZ_SIM_SERVER_CONFIG_PATH specifically, gz sim loads NO physics/
# sensors systems at all (our world file has no <plugin> of its own; PX4
# supplies them this way for every world, see server.config).
# gz_env.sh references GZ_SIM_RESOURCE_PATH etc. without a ${VAR:-} default,
# so pre-declare them (empty) or `set -u` above kills the script here.
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
# shellcheck disable=SC1091
source "$PX4_SITL_ROOTFS/gz_env.sh"

# Add our own world/model directory so `model://moving_platform_aruco` and
# `truck_mission_px4.sdf` resolve (prepended so ours wins on any name clash).
export GZ_SIM_RESOURCE_PATH="$SCRIPT_DIR/models:$SCRIPT_DIR/worlds:${GZ_SIM_RESOURCE_PATH:-}"

if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
  echo "Wayland detected -> QT_QPA_PLATFORM=xcb (set QT_QPA_PLATFORM=wayland to override)"
fi

SERVER_ARGS=()
if [[ "${HEADLESS:-0}" == "1" ]]; then
  SERVER_ARGS=(-s)
  echo "HEADLESS=1 -> server only (no GUI)"
fi

# --- trailer cue relay --------------------------------------------------
# The platform (flat_platform, model://moving_platform_aruco) is driven by
# PX4's own MovingPlatformController plugin, not by us -- moving_marker_node
# in drive_mode=external just relays its ACTUAL gz pose as the ENU cue
# (/marker/position, /marker/velocity) that mission_manager_node/
# precision_landing_node follow. MARKER_MOVE=0 to skip (Gazebo/cue only, no
# relay -- e.g. if you're driving the platform by hand for a test).
# NOT exec'd below (unlike the plain-Gazebo version of this script) so this
# background job and the EXIT trap can clean it up on Ctrl-C.
MARKER_MOVE="${MARKER_MOVE:-1}"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$SCRIPT_DIR/../install/setup.bash"
CHILD_PIDS=()
cleanup() {
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

if [[ "$MARKER_MOVE" == "1" && -f "$ROS_SETUP" && -f "$WS_SETUP" ]]; then
  if [[ "$WORLD" == "truck_mission_px4_track" ]]; then
    # moving_platform_track has NO MovingPlatformController -- it's driven by
    # gz VelocityControl (cmd_vel), commanded here by track_follower_node
    # reading the platform's own forward camera (real vision line-following,
    # not a relay of someone else's motion like moving_marker_node below).
    echo "Starting track_follower_node (vision line-following on truck_mission_px4_track) ..."
    bash -c "
      source '$ROS_SETUP'
      source '$WS_SETUP'
      exec ros2 run precision_landing track_follower_node --ros-args \
        --params-file '$SCRIPT_DIR/config/track_follower_px4.yaml'
    " &
    CHILD_PIDS+=("$!")
  else
    echo "Starting moving_marker_node (drive_mode=external, relaying flat_platform's actual pose) ..."
    bash -c "
      source '$ROS_SETUP'
      source '$WS_SETUP'
      exec ros2 run precision_landing moving_marker_node --ros-args \
        -p model:='flat_platform' \
        -p world:='$WORLD' \
        -p move_model:=false \
        -p release_on_arm:=false \
        -p drive_mode:='external'
    " &
    CHILD_PIDS+=("$!")
  fi
else
  echo "MARKER_MOVE=0 or ROS/workspace not sourced -- skipping cue driver (no /marker/position cue)."
fi

echo "Launching world: $SCRIPT_DIR/worlds/$WORLD.sdf"
echo "Next: HEADLESS=1 ./gazebo/run_px4_sitl.sh   (in another terminal)"
gz sim --verbose=1 -r "${SERVER_ARGS[@]}" "$SCRIPT_DIR/worlds/$WORLD.sdf"
