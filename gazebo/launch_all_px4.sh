#!/usr/bin/env bash
# One-terminal launcher for the PX4 truck-mission stack: Gazebo -> PX4 SITL
# (standalone attach) -> common_bringup (mavros+camera+aruco+precision_landing)
# -> mission_manager, then an interactive console in THIS terminal to trigger
# the return-and-land (instead of a separate `ros2 topic pub .../task_complete`
# terminal) -- same idea as launch_all.sh (ArduPilot), ported to PX4.
#
# IMPORTANT: common_bringup.launch.py already brings up mavros (px4.launch)
# internally -- this script does NOT launch mavros separately. Running a
# second standalone mavros alongside it was a real mistake made once during
# testing: both instances fight over the same FCU link, burn 200%+ CPU each,
# and produce confusing arm-denied/EKF symptoms that look like a sim bug but
# aren't. See [[px4_migration]] memory if you're reading this after the fact.
#
# Usage (inside the sim container):
#   ./gazebo/launch_all_px4.sh
#   MISSION_AREA_E=5 MISSION_AREA_N=0 BATTERY_S=25 ./gazebo/launch_all_px4.sh   # quick-return test
#
# Env overrides (all optional):
#   WORLD=truck_mission_px4 (default) | truck_mission_px4_track   (vision track-following platform)
#   PLATFORM_VEL, PLATFORM_HEADING_DEG   PX4_GZ_PLATFORM_VEL/HEADING_DEG passthrough (moving_platform, ignored for _track)
#   MISSION_AREA_E/N, TRIGGER_DIST, LOITER_S, BATTERY_S, FLIGHT_ALT   mission_manager args
#   PATROL_ROUTE   waypoint-list YAML (default: gazebo/config/patrol_route.yaml -- see
#                  below for why this is a default, not opt-in). PATROL_ROUTE='' for
#                  the legacy single-waypoint mode instead (auto-RETURNs after loiter_s).
#   HEADLESS=1 (default) / HEADLESS=0 for the Gazebo GUI
#   CMD_FIFO   path of the command FIFO (default: $LOG_DIR/cmd.fifo, printed
#              at startup). Only read from when stdin isn't a TTY (e.g.
#              `docker exec -T ...` or any backgrounded/redirected run) --
#              feed commands from another shell with `echo land > $CMD_FIFO`.
#              A single `echo ... > fifo` closes its writer end right after,
#              which would normally EOF the reader -- the non-TTY branch below
#              guards against that the same way [[docker_sim_env]]'s
#              MAVProxy/sim_vehicle.py FIFO trick does (`( while true; do cat
#              fifo; done ) | consumer`), so the console survives any number
#              of separate `echo` writes instead of dying after the first one.
#
# Why patrol_route is on by default: mission_manager_node's "legacy single-
# waypoint" mode (patrol_route unset) auto-RETURNs on its own after loiter_s
# seconds at the mission point -- it was built as a benchmark/regression mode
# for the capstone A*/MPC work, not for "fly around until I type land". A
# fresh run of this script with no PATROL_ROUTE override landed on its own
# after ~15s with no `land` typed -- surprising for the S-PBL scenario
# (mission until the OPERATOR says land), so the script now defaults to
# looping the "+"-shaped patrol_route.yaml (empty-map-safe, no obstacles to
# avoid) instead of the single-point legacy mode.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$(mktemp -d /tmp/px4_mission_logs.XXXXXX)"
echo "logs: $LOG_DIR (gazebo.log / px4.log / bringup.log / mission.log)"
CMD_FIFO="${CMD_FIFO:-$LOG_DIR/cmd.fifo}"
mkfifo "$CMD_FIFO"

WORLD="${WORLD:-truck_mission_px4}"
PATROL_ROUTE="${PATROL_ROUTE-$REPO_DIR/gazebo/config/patrol_route.yaml}"
# The drone's down-camera gz topic embeds the WORLD name (gz-sim's default
# topic-naming fallback) -- common_bringup.launch.py's default only matches
# the truck_mission_px4 world, so override it for any other world.
GZ_CAM_IMG="/world/$WORLD/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image"
GZ_CAM_INFO="/world/$WORLD/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/camera_info"
HEADLESS="${HEADLESS:-1}"
# mission_manager_node's ros2 params are declared as DOUBLE -- an integer-
# looking override (e.g. BATTERY_S=25) makes rclpy raise
# InvalidParameterTypeException at startup, so force a decimal point on
# whatever the caller passed.
as_float() { awk -v v="$1" 'BEGIN{printf "%.3f", v}'; }
MISSION_AREA_E="$(as_float "${MISSION_AREA_E:-5.0}")"
MISSION_AREA_N="$(as_float "${MISSION_AREA_N:-0.0}")"
TRIGGER_DIST="$(as_float "${TRIGGER_DIST:-50.0}")"
LOITER_S="$(as_float "${LOITER_S:-15.0}")"
BATTERY_S="$(as_float "${BATTERY_S:-120.0}")"
FLIGHT_ALT="$(as_float "${FLIGHT_ALT:-5.0}")"

CHILD_PGIDS=()
cleanup() {
  echo ""
  echo "shutting down Gazebo/PX4/mission..."
  for g in "${CHILD_PGIDS[@]}"; do kill -- -"$g" 2>/dev/null; done
  rm -f "$CMD_FIFO"
}
trap cleanup EXIT INT TERM

wait_for() {   # wait_for "description" timeout_s check_command...
  local desc="$1" timeout="$2"; shift 2
  local waited=0
  until "$@" >/dev/null 2>&1; do
    sleep 1; waited=$((waited + 1))
    if [[ $waited -ge $timeout ]]; then
      echo "WARN: timed out waiting for $desc after ${timeout}s -- continuing anyway, check $LOG_DIR" >&2
      return 1
    fi
  done
  echo "$desc ready (${waited}s)"
  return 0
}

# --- 1) Gazebo (+ trailer driver: moving_marker_node external mode, or
#        track_follower_node for WORLD=truck_mission_px4_track -- both are
#        started automatically by run_sim_px4.sh based on $WORLD) ------------
echo "[1/5] starting Gazebo ($WORLD)..."
setsid bash -c "
  HEADLESS='$HEADLESS' WORLD='$WORLD' PX4_GZ_PLATFORM_VEL='${PLATFORM_VEL:-}' PX4_GZ_PLATFORM_HEADING_DEG='${PLATFORM_HEADING_DEG:-}' \
    '$SCRIPT_DIR/run_sim_px4.sh' > '$LOG_DIR/gazebo.log' 2>&1
" &
CHILD_PGIDS+=("$!")
wait_for "Gazebo (world clock topic)" 60 bash -c "gz topic -l 2>/dev/null | grep -q '/world/$WORLD/clock'"

# --- 1b) front-camera bridge (track world only) -- track_follower_node reads
# /front_camera/image over ROS; moving_platform_track's sensor sets an
# explicit short <topic> so this is a plain 1:1 bridge, unlike the drone's
# down-camera (long auto gz path, bridged inside common_bringup below). Not
# needed for the default world (no front camera on moving_platform_aruco).
if [[ "$WORLD" == "truck_mission_px4_track" ]]; then
  echo "[1b/5] starting front-camera bridge (track_follower_node's vision input)..."
  setsid bash -c "
    source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
    exec ros2 run ros_gz_bridge parameter_bridge \
      '/front_camera/image@sensor_msgs/msg/Image[gz.msgs.Image' \
      '/front_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo' \
      --ros-args -r __node:=front_camera_bridge > '$LOG_DIR/front_camera_bridge.log' 2>&1
  " &
  CHILD_PGIDS+=("$!")
fi

# --- 2) PX4 SITL (standalone attach) -----------------------------------------
echo "[2/5] starting PX4 SITL..."
setsid bash -c "'$SCRIPT_DIR/run_px4_sitl.sh' > '$LOG_DIR/px4.log' 2>&1" &
CHILD_PGIDS+=("$!")
# NOT "Ready for takeoff" -- that only appears once mission_manager_node (step 4)
# has auto-fixed NAV_DLL_ACT, which hasn't started yet at this point.
wait_for "PX4 (startup script done)" 45 bash -c "grep -q 'Startup script returned successfully' '$LOG_DIR/px4.log'"

# --- 3) common_bringup (mavros + camera bridge + aruco + precision_landing) -
# gz_camera_image_topic/info default (in common_bringup.launch.py) only
# matches the truck_mission_px4 world name baked into the drone's auto long
# gz topic path -- override for any other world.
echo "[3/5] starting common_bringup (mavros/camera/aruco/precision_landing)..."
setsid bash -c "
  source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
  source '$REPO_DIR/install/setup.bash'
  exec ros2 launch precision_landing common_bringup.launch.py \
    gz_camera_image_topic:='$GZ_CAM_IMG' gz_camera_info_topic:='$GZ_CAM_INFO' \
    > '$LOG_DIR/bringup.log' 2>&1
" &
CHILD_PGIDS+=("$!")
wait_for "MAVROS<->PX4 link" 60 bash -c "grep -q 'connected: true' <(timeout 3 ros2 topic echo /mavros/state --once 2>/dev/null)"

# --- 4) mission_manager -------------------------------------------------------
PATROL_ARG=""
[[ -n "$PATROL_ROUTE" ]] && PATROL_ARG="-p patrol_route:=$PATROL_ROUTE"
echo "[4/5] starting mission (mission_area=($MISSION_AREA_E,$MISSION_AREA_N) battery=${BATTERY_S}s patrol_route=${PATROL_ROUTE:-<legacy single point, auto-RETURNs after loiter_s>})..."
setsid bash -c "
  source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
  source '$REPO_DIR/install/setup.bash'
  exec ros2 run precision_landing mission_manager_node --ros-args \
    -p mission_area_e:=$MISSION_AREA_E -p mission_area_n:=$MISSION_AREA_N \
    -p trigger_dist:=$TRIGGER_DIST -p loiter_s:=$LOITER_S \
    -p battery_capacity_s:=$BATTERY_S -p flight_alt:=$FLIGHT_ALT \
    $PATROL_ARG \
    > '$LOG_DIR/mission.log' 2>&1
" &
CHILD_PGIDS+=("$!")

# --- interactive console --------------------------------------------------
run_console() {
  while true; do
    if ! read -r -p "mission> " cmd; then
      echo ""
      echo "stdin closed -- exiting console. Mission keeps running in the background;" >&2
      echo "use 'ros2 topic pub /mission/task_complete std_msgs/msg/Bool \"{data: true}\"' to land, or rerun with a real TTY." >&2
      break
    fi
    case "$cmd" in
      land|복귀|return)
        ros2 topic pub /mission/task_complete std_msgs/msg/Bool '{data: true}' --once
        echo "task_complete sent -- watch 'log' for RETURN -> LANDING -> PRECLAND -> disarmed."
        ;;
      status)
        echo "mission phase: $(timeout 2 ros2 topic echo /mission/phase --once 2>/dev/null | head -1)"
        echo "battery:       $(timeout 2 ros2 topic echo /mission/battery_s --once 2>/dev/null | head -1)"
        echo "mavros mode:   $(timeout 2 ros2 topic echo /mavros/state --field mode --once 2>/dev/null)"
        echo "aruco:         $(timeout 2 ros2 topic echo /perception/aruco_detected --field data --once 2>/dev/null)"
        ;;
      log)
        echo "-- tailing mission log, Ctrl+C returns to this prompt (does NOT stop the mission) --"
        trap '' INT
        tail -n 20 -f "$LOG_DIR/mission.log"
        trap cleanup EXIT INT TERM
        ;;
      help|"")
        echo "  land / 복귀 / return  -- send task_complete (drone returns and lands)"
        echo "  status                -- mission phase / battery / mavros mode / aruco detection"
        echo "  log                   -- tail mission log (Ctrl+C to come back here)"
        echo "  quit / exit           -- shut down Gazebo/PX4/mission"
        ;;
      quit|exit)
        break
        ;;
      *)
        echo "unknown command '$cmd' -- try: help"
        ;;
    esac
  done
}

echo ""
echo "=== all up. commands: land | status | log | help | quit ==="
if [[ -t 0 ]]; then
  run_console
else
  # No TTY on our own stdin (docker exec -T, backgrounded, output redirected
  # to a file, ...). Reading directly from it would just see an immediate EOF
  # and -- before this was fixed -- spin printing the help text as fast as the
  # CPU allows, which once filled the host's disk with a 15GB+ log in under an
  # hour. Feed the console from $CMD_FIFO instead, with a `while true; do cat
  # fifo; done` wrapper keeping a permanent writer around so one `echo cmd >
  # fifo` closing its writer end doesn't EOF us into the same trap again
  # (same trick as [[docker_sim_env]]'s sim_vehicle.py/MAVProxy FIFO pattern).
  # Feed it through process substitution rather than a plain `... | run_console`
  # pipe: a pipeline only returns once BOTH sides exit, and the wrapper loop
  # above never exits on its own (it blocks in `cat` waiting for the next
  # writer) -- with a plain pipe, `quit` would make run_console return but the
  # script would then hang forever instead of reaching cleanup. Explicitly
  # killing the feeder after run_console returns avoids that.
  echo "no TTY on stdin -- send commands from another shell instead:"
  echo "  echo land   > $CMD_FIFO"
  echo "  echo status > $CMD_FIFO"
  echo "  echo quit   > $CMD_FIFO"
  exec 3< <(while true; do cat "$CMD_FIFO"; done)
  FIFO_FEEDER_PID=$!
  run_console <&3
  exec 3<&-
  kill "$FIFO_FEEDER_PID" 2>/dev/null
fi
