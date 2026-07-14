#!/usr/bin/env bash
# One-terminal launcher for the city-map capstone stack when the TRAILER is
# driven by hand (gazebo/tools/trailer_keyboard_ctrl.py in a SEPARATE terminal)
# instead of moving_marker_node's pattern=route autopilot. Starts everything
# else in the background (Gazebo, trailer position-relay-only, PX4 SITL,
# mavros/camera/aruco/precision_landing, mission_manager), then drops into the
# same interactive console as launch_all_px4_city.sh. Only 2 terminals needed
# total: this one, and the keyboard-control script.
#
# Usage (inside the sim container):
#   ./gazebo/launch_keyboard_px4_city.sh          # terminal 1
#   python3 gazebo/tools/trailer_keyboard_ctrl.py # terminal 2
#
# Why a separate "position-relay-only" trailer node instead of just skipping
# it: mission_manager_node's MOUNTED->TAKEOFF trigger and its whole RETURN/
# d_truck logic need a live truck position on /marker/position. That comes
# from moving_marker_node relaying Gazebo's actual pose of the model -- it's
# just told (move_model=false, drive_mode=external) not to ALSO drive it,
# since the keyboard script owns cmd_vel exclusively (both writing to the same
# gz topic would fight each other).
#
# Env overrides (all optional):
#   PATROL_ROUTE, MISSION_AREA_E/N, TRIGGER_DIST, LOITER_S, BATTERY_S,
#   FLIGHT_ALT, OBSTACLE_MAP, MPC_ENABLE (default false -- see the two still-
#   open MPC/A* issues in this session's notes before flipping this to true)
#   HEADLESS=1 (default) / HEADLESS=0 for the Gazebo GUI
#   CMD_FIFO   see launch_all_px4.sh's header for the no-TTY FIFO mechanism.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$(mktemp -d /tmp/px4_city_keyboard_logs.XXXXXX)"
echo "logs: $LOG_DIR (gazebo.log / trailer.log / px4.log / bringup.log / mission.log)"
CMD_FIFO="${CMD_FIFO:-$LOG_DIR/cmd.fifo}"
mkfifo "$CMD_FIFO"

WORLD="city_map_mission"
PATROL_ROUTE="${PATROL_ROUTE-$REPO_DIR/gazebo/config/patrol_route_city.yaml}"
OBSTACLE_MAP="${OBSTACLE_MAP:-$REPO_DIR/gazebo/config/obstacle_map_city.yaml}"
MPC_ENABLE="${MPC_ENABLE:-false}"
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
  echo "shutting down Gazebo/trailer-relay/PX4/mission (keyboard terminal is yours to close separately)..."
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
echo "[1/5] starting Gazebo (city_map_mission)..."
setsid bash -c "
  HEADLESS='$HEADLESS' USE_NVIDIA='${USE_NVIDIA:-1}' \
    '$SCRIPT_DIR/run_world.sh' city-mission > '$LOG_DIR/gazebo.log' 2>&1
" &
CHILD_PGIDS+=("$!")
wait_for "Gazebo (world clock topic)" 90 bash -c "gz topic -l 2>/dev/null | grep -q '/world/$WORLD/clock'"

# --- 2) trailer position relay ONLY (no driving -- the keyboard script owns
# cmd_vel in the other terminal; this just reports Gazebo's actual pose of
# the model on /marker/position so mission_manager knows where the truck is).
echo "[2/5] starting trailer position relay (NOT driving -- use the keyboard script for that)..."
setsid bash -c "
  source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
  source '$REPO_DIR/install/setup.bash'
  exec ros2 run precision_landing moving_marker_node --ros-args \
    -p world:='$WORLD' -p model:='flat_platform' \
    -p pattern:='static' -p drive_mode:='external' -p move_model:=false \
    > '$LOG_DIR/trailer.log' 2>&1
" &
CHILD_PGIDS+=("$!")

# --- 3) PX4 SITL (standalone attach, spawns ON the trailer) -----------------
echo "[3/5] starting PX4 SITL..."
setsid bash -c "MAP='city-mission' '$SCRIPT_DIR/run_px4_sitl.sh' > '$LOG_DIR/px4.log' 2>&1" &
CHILD_PGIDS+=("$!")
wait_for "PX4 (startup script done)" 45 bash -c "grep -q 'Startup script returned successfully' '$LOG_DIR/px4.log'"

# --- 4) common_bringup (mavros + camera bridge + aruco + precision_landing) -
echo "[4/5] starting common_bringup (mavros/camera/aruco/precision_landing)..."
setsid bash -c "
  source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
  source '$REPO_DIR/install/setup.bash'
  exec ros2 launch precision_landing common_bringup.launch.py \
    gz_camera_image_topic:='$GZ_CAM_IMG' gz_camera_info_topic:='$GZ_CAM_INFO' \
    > '$LOG_DIR/bringup.log' 2>&1
" &
CHILD_PGIDS+=("$!")
wait_for "MAVROS<->PX4 link" 60 bash -c "grep -q 'connected: true' <(timeout 3 ros2 topic echo /mavros/state --once 2>/dev/null)"

# --- 5) mission_manager (mpc_enable defaults false -- see MPC_ENABLE above) -
echo "[5/5] starting mission (mission_area=($MISSION_AREA_E,$MISSION_AREA_N) mpc_enable=$MPC_ENABLE patrol_route=$PATROL_ROUTE)..."
setsid bash -c "
  source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
  source '$REPO_DIR/install/setup.bash'
  exec ros2 run precision_landing mission_manager_node --ros-args \
    -p mission_area_e:=$MISSION_AREA_E -p mission_area_n:=$MISSION_AREA_N \
    -p trigger_dist:=$TRIGGER_DIST -p loiter_s:=$LOITER_S \
    -p battery_capacity_s:=$BATTERY_S -p flight_alt:=$FLIGHT_ALT \
    -p patrol_route:='$PATROL_ROUTE' \
    -p mpc_enable:=$MPC_ENABLE -p obstacle_map:='$OBSTACLE_MAP' \
    > '$LOG_DIR/mission.log' 2>&1
" &
CHILD_PGIDS+=("$!")

echo ""
echo "=== all up. Now in ANOTHER terminal: python3 gazebo/tools/trailer_keyboard_ctrl.py ==="
echo "=== drive the trailer within ${TRIGGER_DIST}m of ($MISSION_AREA_E,$MISSION_AREA_N) to trigger takeoff ==="

# --- interactive console (same as launch_all_px4_city.sh) -------------------
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
        echo "mavros armed:  $(timeout 2 ros2 topic echo /mavros/state --field armed --once 2>/dev/null)"
        ;;
      log)
        echo "-- tailing mission log, Ctrl+C returns to this prompt (does NOT stop the mission) --"
        trap '' INT
        tail -n 20 -f "$LOG_DIR/mission.log"
        trap cleanup EXIT INT TERM
        ;;
      trailer-log)
        echo "-- tailing trailer relay log, Ctrl+C returns to this prompt --"
        trap '' INT
        tail -n 20 -f "$LOG_DIR/trailer.log"
        trap cleanup EXIT INT TERM
        ;;
      help|"")
        echo "  land / 복귀 / return  -- send task_complete (drone returns and lands)"
        echo "  status                -- mission phase / battery / mavros mode+armed"
        echo "  log                   -- tail mission log (Ctrl+C to come back here)"
        echo "  trailer-log           -- tail trailer relay log (Ctrl+C to come back here)"
        echo "  quit / exit           -- shut down Gazebo/trailer-relay/PX4/mission"
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

if [[ -t 0 ]]; then
  run_console
else
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
