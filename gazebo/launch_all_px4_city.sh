#!/usr/bin/env bash
# One-terminal launcher for the city-map capstone stack: Gazebo (city_map_mission,
# trailer + 274 buildings) -> trailer driver (moving_marker_node, pattern=route) ->
# PX4 SITL (standalone attach, spawns ON the trailer) -> common_bringup
# (mavros+camera+aruco+precision_landing) -> mission_manager (mpc_enable=true,
# A*+SFC+MPC obstacle avoidance against obstacle_map_city.yaml), then an
# interactive console in THIS terminal to trigger the return-and-land -- same
# structure as launch_all_px4.sh (the S-PBL flat-map version), see that
# script's header for the FIFO/mavros-conflict notes that apply here too.
#
# Usage (inside the sim container):
#   ./gazebo/launch_all_px4_city.sh
#   BATTERY_S=600 ./gazebo/launch_all_px4_city.sh   # longer run before auto-RTL
#
# Env overrides (all optional):
#   TRAILER_ROUTE   default gazebo/config/trailer_route_city.yaml (48 waypoints,
#                   A*-planned, obstacle-free by construction -- see
#                   gazebo/tools/generate_trailer_route.py)
#   TRAILER_SPEED   trailer path speed, m/s (default 2.0)
#   PATROL_ROUTE    drone's own MISSION-leg patrol, default
#                   gazebo/config/patrol_route_city.yaml
#   MISSION_AREA_E/N, TRIGGER_DIST, LOITER_S, BATTERY_S, FLIGHT_ALT   mission_manager args
#   HEADLESS=1 (default) / HEADLESS=0 for the Gazebo GUI
#   CMD_FIFO   see launch_all_px4.sh's header -- identical mechanism.
#
# 2026-07-13: trailer ground-contact height verified (settles level at the
# real, pad-free ground height). A 4x4 footprint tipped repeatedly under real
# driving even on flat ground and was reverted to 5x5 (see
# moving_platform_aruco's model.sdf comment) -- 5x5's actual route-following
# drive on THIS map/route is being (re)verified now, and trailer+PX4+
# mission_manager together as ONE pipeline has NOT been run end-to-end yet --
# see task #10 in this session's work.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$(mktemp -d /tmp/px4_city_mission_logs.XXXXXX)"
echo "logs: $LOG_DIR (gazebo.log / trailer.log / px4.log / bringup.log / mission.log)"
CMD_FIFO="${CMD_FIFO:-$LOG_DIR/cmd.fifo}"
mkfifo "$CMD_FIFO"

WORLD="city_map_mission"
TRAILER_ROUTE="${TRAILER_ROUTE:-$REPO_DIR/gazebo/config/trailer_route_city.yaml}"
TRAILER_SPEED="${TRAILER_SPEED:-2.0}"
PATROL_ROUTE="${PATROL_ROUTE-$REPO_DIR/gazebo/config/patrol_route_city.yaml}"
OBSTACLE_MAP="${OBSTACLE_MAP:-$REPO_DIR/gazebo/config/obstacle_map_city.yaml}"
# The drone's down-camera gz topic embeds the WORLD name (gz-sim's default
# topic-naming fallback) -- common_bringup.launch.py's default only matches
# the truck_mission_px4 world, so override it here too (same reasoning as
# launch_all_px4.sh).
GZ_CAM_IMG="/world/$WORLD/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image"
GZ_CAM_INFO="/world/$WORLD/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/camera_info"
HEADLESS="${HEADLESS:-1}"
as_float() { awk -v v="$1" 'BEGIN{printf "%.3f", v}'; }
MISSION_AREA_E="$(as_float "${MISSION_AREA_E:--150.0}")"
MISSION_AREA_N="$(as_float "${MISSION_AREA_N:-100.0}")"
TRIGGER_DIST="$(as_float "${TRIGGER_DIST:-50.0}")"
LOITER_S="$(as_float "${LOITER_S:-15.0}")"
BATTERY_S="$(as_float "${BATTERY_S:-300.0}")"
FLIGHT_ALT="$(as_float "${FLIGHT_ALT:-5.0}")"

CHILD_PGIDS=()
cleanup() {
  echo ""
  echo "shutting down Gazebo/trailer/PX4/mission..."
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

# --- 1) Gazebo (city_map_mission: terrain + 274 buildings + trailer) --------
echo "[1/6] starting Gazebo (city_map_mission)..."
setsid bash -c "
  HEADLESS='$HEADLESS' USE_NVIDIA='${USE_NVIDIA:-1}' \
    '$SCRIPT_DIR/run_world.sh' city-mission > '$LOG_DIR/gazebo.log' 2>&1
" &
CHILD_PGIDS+=("$!")
wait_for "Gazebo (world clock topic)" 90 bash -c "gz topic -l 2>/dev/null | grep -q '/world/$WORLD/clock'"

# --- 2) trailer driver (moving_marker_node, pattern=route, VelocityControl) -
# Unlike truck_mission_px4 (PX4's own MovingPlatformController, driven by
# run_sim_px4.sh itself), city_map_mission's trailer
# (model://moving_platform_aruco_route) is driven externally over gz's
# standard VelocityControl plugin, so this is its own step here instead of
# being bundled into the Gazebo-launch script.
echo "[2/6] starting trailer driver (route=$TRAILER_ROUTE, speed=${TRAILER_SPEED}m/s)..."
setsid bash -c "
  source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
  source '$REPO_DIR/install/setup.bash'
  exec ros2 run precision_landing moving_marker_node --ros-args \
    -p world:='$WORLD' -p model:='flat_platform' \
    -p cmd_vel_topic:='/model/flat_platform/cmd_vel' \
    -p pattern:='route' -p route_file:='$TRAILER_ROUTE' -p speed:=$TRAILER_SPEED \
    -p drive_mode:='velocity' -p move_model:=true -p release_on_arm:=false \
    -p start_delay:=15.0 \
    > '$LOG_DIR/trailer.log' 2>&1
" &
CHILD_PGIDS+=("$!")

# --- 3) PX4 SITL (standalone attach, spawns ON the trailer) -----------------
echo "[3/6] starting PX4 SITL..."
setsid bash -c "MAP='city-mission' '$SCRIPT_DIR/run_px4_sitl.sh' > '$LOG_DIR/px4.log' 2>&1" &
CHILD_PGIDS+=("$!")
wait_for "PX4 (startup script done)" 45 bash -c "grep -q 'Startup script returned successfully' '$LOG_DIR/px4.log'"

# --- 4) common_bringup (mavros + camera bridge + aruco + precision_landing) -
echo "[4/6] starting common_bringup (mavros/camera/aruco/precision_landing)..."
setsid bash -c "
  source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
  source '$REPO_DIR/install/setup.bash'
  exec ros2 launch precision_landing common_bringup.launch.py \
    gz_camera_image_topic:='$GZ_CAM_IMG' gz_camera_info_topic:='$GZ_CAM_INFO' \
    > '$LOG_DIR/bringup.log' 2>&1
" &
CHILD_PGIDS+=("$!")
wait_for "MAVROS<->PX4 link" 60 bash -c "grep -q 'connected: true' <(timeout 3 ros2 topic echo /mavros/state --once 2>/dev/null)"

# --- 5) mission_manager (A*+SFC+MPC obstacle avoidance, city patrol) --------
echo "[5/6] starting mission (mission_area=($MISSION_AREA_E,$MISSION_AREA_N) battery=${BATTERY_S}s patrol_route=$PATROL_ROUTE obstacle_map=$OBSTACLE_MAP)..."
setsid bash -c "
  source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
  source '$REPO_DIR/install/setup.bash'
  exec ros2 run precision_landing mission_manager_node --ros-args \
    -p mission_area_e:=$MISSION_AREA_E -p mission_area_n:=$MISSION_AREA_N \
    -p trigger_dist:=$TRIGGER_DIST -p loiter_s:=$LOITER_S \
    -p battery_capacity_s:=$BATTERY_S -p flight_alt:=$FLIGHT_ALT \
    -p patrol_route:='$PATROL_ROUTE' \
    -p mpc_enable:=true -p obstacle_map:='$OBSTACLE_MAP' \
    > '$LOG_DIR/mission.log' 2>&1
" &
CHILD_PGIDS+=("$!")

echo "[6/6] all services launched -- waiting for the interactive console..."

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
      trailer-log)
        echo "-- tailing trailer log, Ctrl+C returns to this prompt --"
        trap '' INT
        tail -n 20 -f "$LOG_DIR/trailer.log"
        trap cleanup EXIT INT TERM
        ;;
      help|"")
        echo "  land / 복귀 / return  -- send task_complete (drone returns and lands)"
        echo "  status                -- mission phase / battery / mavros mode / aruco detection"
        echo "  log                   -- tail mission log (Ctrl+C to come back here)"
        echo "  trailer-log           -- tail trailer driver log (Ctrl+C to come back here)"
        echo "  quit / exit           -- shut down Gazebo/trailer/PX4/mission"
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
echo "=== all up. commands: land | status | log | trailer-log | help | quit ==="
if [[ -t 0 ]]; then
  run_console
else
  # See launch_all_px4.sh's header for why this FIFO indirection exists.
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
