#!/usr/bin/env bash
#
# Launch the iris-with-downward-camera world in Gazebo (gz-sim Harmonic).
#
# Prerequisites (see README): Gazebo Harmonic installed, ardupilot_gazebo
# plugin built at ~/ardupilot_gazebo/build.
#
# Usage:
#   ./run_sim.sh            # start Gazebo with the world
#
# In separate terminals run ArduPilot SITL and the ROS 2 camera bridge:
#   sim_vehicle.py -v ArduCopter -f JSON --console --map
#   ros2 launch ./launch/camera_bridge.launch.py
#
set -euo pipefail

# Repo root (this script lives in <repo>/gazebo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AP_GAZEBO="${AP_GAZEBO:-$HOME/ardupilot_gazebo}"

if [[ ! -d "$AP_GAZEBO" ]]; then
  echo "ERROR: ardupilot_gazebo not found at $AP_GAZEBO" >&2
  echo "       Set AP_GAZEBO=/path/to/ardupilot_gazebo or clone it there." >&2
  exit 1
fi

if [[ ! -d "$AP_GAZEBO/build" ]]; then
  echo "ERROR: ardupilot_gazebo plugin not built ($AP_GAZEBO/build missing)." >&2
  echo "       Build it first (see README)." >&2
  exit 1
fi

# gz-sim needs to find: our model+world, the stock ardupilot_gazebo models,
# and the compiled ArduPilotPlugin / sensor systems.
export GZ_SIM_RESOURCE_PATH="$SCRIPT_DIR/models:$SCRIPT_DIR/worlds:$AP_GAZEBO/models:$AP_GAZEBO/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$AP_GAZEBO/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"

# --- GUI rendering on Wayland ------------------------------------------------
# gz-sim's Qt/ogre2 GUI frequently shows a BLANK WHITE viewport under a native
# Wayland session — the world loads but the 3D render target never presents.
# Forcing Qt onto XWayland (xcb) fixes it. Override with QT_QPA_PLATFORM=wayland.
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
  echo "Wayland detected -> QT_QPA_PLATFORM=xcb (set QT_QPA_PLATFORM=wayland to override)"
fi
# Still white? Drop the render engine to ogre (v1): RENDER_ENGINE=ogre ./run_sim.sh
RENDER_ARGS=()
if [[ -n "${RENDER_ENGINE:-}" ]]; then
  RENDER_ARGS=(--render-engine "$RENDER_ENGINE")
  echo "Using render engine: $RENDER_ENGINE"
fi
# HEADLESS=1 -> run the SERVER only (no GUI window). The down-camera sensor is
# still rendered and published on down_camera/image, so the whole mission and
# its verification (rqt_image_view, /marker/position) work without any GUI.
SERVER_ARGS=()
if [[ "${HEADLESS:-0}" == "1" ]]; then
  SERVER_ARGS=(-s)
  echo "HEADLESS=1 -> server only (no GUI; camera still renders)"
fi

echo "GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"
echo "GZ_SIM_SYSTEM_PLUGIN_PATH=$GZ_SIM_SYSTEM_PLUGIN_PATH"

# --- moving marker platform --------------------------------------------------
# The marker platform is part of the simulated world, so its mover starts here
# together with Gazebo. moving_marker_node teleports the model along a trajectory
# (default: a circle) via the gz set_pose service AND streams its position,
# measured from the drone spawn (world origin == MAVROS local ENU origin), on the
# ROS topic /marker/position. The precision_landing controller follows that cue.
#
# Tunables (env vars):
#   MARKER_MOVE=0      disable (Gazebo only, platform stays put)
#   MARKER_MODEL       gz model name to drive   (default aruco_platform)
#   MARKER_PATTERN     line | circle | static   (default line — constant-direction
#                        motion the CV Kalman/feed-forward tracks exactly, so the
#                        drone matches its velocity and lands cleanly; a circle's
#                        turning velocity defeats the tracker and tips the drone)
#   MARKER_SPEED       path speed  m/s           (default 3.0 — needs vel_max>3 on
#                        the controller so the drone can match it AND correct)
#   MARKER_AMP         half-stroke (line) / radius (circle)  m  (default 250 → the
#                        line runs 500 m straight (2·amp) before turning around)
#   MARKER_CE MARKER_CN  trajectory centre, ENU m (default 3.0 0.0 — clear of the
#                        drone spawn at 0,0 so the platform never clips it)
#   MARKER_Z           model-origin height m      (default 0.0 = box on ground)
MARKER_MOVE="${MARKER_MOVE:-1}"
MARKER_PGID=""
if [[ "$MARKER_MOVE" == "1" ]]; then
  WS_SETUP="$SCRIPT_DIR/../install/setup.bash"
  ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
  if [[ -f "$ROS_SETUP" && -f "$WS_SETUP" ]]; then
    echo "Starting moving_marker_node (model=${MARKER_MODEL:-aruco_platform} pattern=${MARKER_PATTERN:-line}) ..."
    # setsid → the mover (ros2 run wrapper + the node it spawns) gets its OWN
    # process group, so the EXIT trap can kill the WHOLE tree (kill -- -PGID).
    # Killing just the `ros2 run` wrapper would orphan the node and they would
    # pile up across restarts.
    setsid bash -c "
      source '$ROS_SETUP'
      source '$WS_SETUP'
      exec ros2 run precision_landing moving_marker_node --ros-args \
        -p model:='${MARKER_MODEL:-aruco_platform}' \
        -p pattern:='${MARKER_PATTERN:-line}' \
        -p speed:='${MARKER_SPEED:-3.0}' \
        -p amplitude:='${MARKER_AMP:-250.0}' \
        -p center_e:='${MARKER_CE:-3.0}' \
        -p center_n:='${MARKER_CN:-0.0}' \
        -p z:='${MARKER_Z:-0.0}'
    " &
    MARKER_PGID=$!                 # with setsid, child PID == its new PGID
    trap 'kill -- -"$MARKER_PGID" 2>/dev/null' EXIT INT TERM
  else
    echo "WARN: ROS workspace not built ($WS_SETUP missing) — marker will NOT move." >&2
    echo "      Build it: colcon build --symlink-install --packages-select precision_landing" >&2
    echo "      (or set MARKER_MOVE=0 to silence this and run Gazebo only.)" >&2
  fi
fi

echo "Launching world: iris_down_camera_runway.sdf"
# Not exec'd, so the EXIT trap fires and cleans up the mover on Ctrl-C.
gz sim -v4 -r "${SERVER_ARGS[@]}" "${RENDER_ARGS[@]}" "$SCRIPT_DIR/worlds/iris_down_camera_runway.sdf"
