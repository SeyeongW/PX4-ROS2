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
# The camera bridge (gz->ROS) and the line-processing overlay viewer
# (rqt_image_view /perception/track_debug) start automatically with this script
# (BRIDGE=0 / VIEW=0 to disable). In a separate terminal run ArduPilot SITL:
#   sim_vehicle.py -v ArduCopter -f JSON --console --map
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

# --- gz-transport on loopback ------------------------------------------------
# This host has extra interfaces (Tailscale 100.x, docker0, wifi). gz-transport's
# default interface pick can land on the Tailscale/wifi address, so the GUI,
# server and gz tools advertise on an address peers can't reach — symptoms:
#   [Err] SceneBroadcaster: Timed out waiting for state
#   NodeShared::RecvSrvRequest() error sending response: Host unreachable
# Pin everything to loopback (all sim processes are local). Override with GZ_IP=...
export GZ_IP="${GZ_IP:-127.0.0.1}"
echo "GZ_IP=$GZ_IP (gz-transport pinned to loopback; override with GZ_IP=...)"

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
#   MARKER_START_DELAY stationary warm-up s        (default 0.0 — for the mounted
#                        scenario set e.g. 20 so the drone's EKF settles & arms on
#                        the STILL truck before it drives off and the drone launches)
#   MARKER_RELEASE_ON_ARM release DetachableJoint on arm  (default true — frees the
#                        truck-mounted drone to take off; set false for legacy world)
#   MARKER_DRIVE       velocity | teleport  (default velocity — drive the truck by
#                        real velocity via its VelocityControl plugin so a jointed
#                        drone is carried with momentum and launches cleanly; use
#                        teleport for the legacy flat-marker decal with no plugin)
# FOLLOW_TRACK=1 (default) → drive the platform by CAMERA line-following around
#   the painted 400 m track (track_follower_node reads the platform's forward
#   camera and steers it along the white line). FOLLOW_TRACK=0 → fall back to the
#   legacy preset-trajectory mover (moving_marker_node, a hard-coded line/circle).
#   Both publish /marker/position and release the DetachableJoint on arm, so the
#   ride→launch→precision-land pipeline is identical either way.
# track_follower tuning: ALL gains live in config/track_follower.yaml (edit &
# restart). Quick CLI overrides (apply only when set):
#   FOLLOW_SPEED   cruise speed cap m/s
#   FOLLOW_BW      steering loop bandwidth ωₙ rad/s  (≈ P strength)
#   FOLLOW_ZETA    damping ratio ζ                   (raise to kill oscillation)
#   MARKER_START_DELAY  stationary warm-up s (e.g. 20 to arm on the still truck)
# Steering: lane fit on the IPM (bird's-eye) ground plane → metric cross-track[m]/
# heading[rad]/curvature[1/m], then a 2nd-order law (k_y=ωₙ²/v, k_ψ=2ζωₙ) +
# curvature feed-forward — see node header / docs.
FOLLOW_TRACK="${FOLLOW_TRACK:-1}"
MARKER_MOVE="${MARKER_MOVE:-1}"
WS_SETUP="$SCRIPT_DIR/../install/setup.bash"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"

# Track every background helper's process GROUP (mover, camera bridge, viewer) so
# one EXIT/INT/TERM trap tears the whole lot down. Each helper is setsid'd into
# its own group; kill -- -PGID gets the `ros2` wrapper AND the node it spawns.
CHILD_PGIDS=()
cleanup() { for g in "${CHILD_PGIDS[@]}"; do kill -- -"$g" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

if [[ "$MARKER_MOVE" == "1" ]]; then
  if [[ -f "$ROS_SETUP" && -f "$WS_SETUP" ]]; then
    # setsid → the mover (ros2 run wrapper + the node it spawns) gets its OWN
    # process group, so the EXIT trap can kill the WHOLE tree (kill -- -PGID).
    # Killing just the `ros2 run` wrapper would orphan the node and they would
    # pile up across restarts.
    if [[ "$FOLLOW_TRACK" == "1" ]]; then
      echo "Starting track_follower_node (camera line-following the 400 m track) ..."
      # All tunable gains live in config/track_follower.yaml (edit & restart). The
      # three quick env knobs override the file ONLY when explicitly set, so the
      # YAML stays the single source of truth otherwise.
      PARAMS_FILE="$SCRIPT_DIR/config/track_follower.yaml"
      FOLLOW_OVR=""
      [[ -n "${FOLLOW_SPEED:-}" ]] && FOLLOW_OVR="$FOLLOW_OVR -p forward_speed:=$FOLLOW_SPEED"
      [[ -n "${FOLLOW_BW:-}"    ]] && FOLLOW_OVR="$FOLLOW_OVR -p ctrl_bw:=$FOLLOW_BW"
      [[ -n "${FOLLOW_ZETA:-}"  ]] && FOLLOW_OVR="$FOLLOW_OVR -p zeta:=$FOLLOW_ZETA"
      setsid bash -c "
        source '$ROS_SETUP'
        source '$WS_SETUP'
        exec ros2 run precision_landing track_follower_node --ros-args \
          --params-file '$PARAMS_FILE' \
          -p model:='${MARKER_MODEL:-aruco_platform}' \
          -p start_delay:='${MARKER_START_DELAY:-0.0}' \
          -p release_on_arm:='${MARKER_RELEASE_ON_ARM:-true}' \
          $FOLLOW_OVR
      " &
    else
      echo "Starting moving_marker_node (model=${MARKER_MODEL:-aruco_platform} pattern=${MARKER_PATTERN:-line}) ..."
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
          -p z:='${MARKER_Z:-0.0}' \
          -p start_delay:='${MARKER_START_DELAY:-0.0}' \
          -p release_on_arm:='${MARKER_RELEASE_ON_ARM:-true}' \
          -p drive_mode:='${MARKER_DRIVE:-velocity}'
      " &
    fi
    CHILD_PGIDS+=("$!")            # with setsid, child PID == its new PGID
  else
    echo "WARN: ROS workspace not built ($WS_SETUP missing) — platform will NOT move." >&2
    echo "      Build it: colcon build --symlink-install --packages-select precision_landing" >&2
    echo "      (or set MARKER_MOVE=0 to silence this and run Gazebo only.)" >&2
  fi
fi

# --- camera bridge (front + down cameras: gz -> ROS) -------------------------
# BRIDGE=1 (default) auto-starts the ros_gz_bridge so /front_camera/image and
# /down_camera/image appear in ROS as soon as Gazebo publishes them — no separate
# terminal. track_follower then gets images and its overlay (/perception/track_
# debug) goes live. Set BRIDGE=0 to run the bridge yourself.
BRIDGE="${BRIDGE:-1}"
if [[ "$BRIDGE" == "1" && -f "$ROS_SETUP" && -f "$WS_SETUP" ]]; then
  echo "Starting camera bridge (front+down cameras -> ROS) ..."
  setsid bash -c "
    source '$ROS_SETUP'
    source '$WS_SETUP'
    exec ros2 launch '$SCRIPT_DIR/launch/camera_bridge.launch.py'
  " &
  CHILD_PGIDS+=("$!")
fi

# --- live overlay viewer -----------------------------------------------------
# VIEW=1 (default, unless HEADLESS or no $DISPLAY) opens rqt_image_view on the
# line-processing overlay so you SEE how the lane is detected and how it drives.
# VIEW=0 to skip the window (then just run: rqt_image_view /perception/track_debug).
VIEW="${VIEW:-1}"
if [[ "$VIEW" == "1" && "${HEADLESS:-0}" != "1" && -n "${DISPLAY:-}" \
      && -f "$ROS_SETUP" && -f "$WS_SETUP" ]]; then
  echo "Opening overlay viewer: rqt_image_view /perception/track_debug ..."
  setsid bash -c "
    source '$ROS_SETUP'
    source '$WS_SETUP'
    exec ros2 run rqt_image_view rqt_image_view /perception/track_debug
  " &
  CHILD_PGIDS+=("$!")
fi

echo "Launching world: iris_down_camera_runway.sdf"
# Not exec'd, so the EXIT trap fires and cleans up the helpers on Ctrl-C.
gz sim -v4 -r "${SERVER_ARGS[@]}" "${RENDER_ARGS[@]}" "$SCRIPT_DIR/worlds/iris_down_camera_runway.sdf"
