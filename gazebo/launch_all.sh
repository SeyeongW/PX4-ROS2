#!/usr/bin/env bash
# One-terminal launcher: Gazebo -> SITL -> MPC mission, then an interactive
# console in THIS SAME terminal to trigger the return-and-land (instead of
# opening a separate terminal for `ros2 topic pub .../task_complete`).
#
# Still 3 separate background processes underneath (Gazebo / SITL / ROS
# mission stack are independent programs that must all keep running) -- this
# script just starts them in the right order with readiness checks, logs each
# to its own file, and gives you ONE foreground prompt instead of 3 terminals.
#
# Usage (inside the sim container):
#   ./gazebo/launch_all.sh
#   MISSION_AREA_E=20 MISSION_AREA_N=16 PATROL_ROUTE= ./gazebo/launch_all.sh   # legacy single-point test
#
# Env overrides (all optional, defaults match the obstacle_field MPC patrol
# test used throughout this session):
#   WORLD, VIEW                  passed straight to run_sim.sh
#   PATROL_ROUTE                 default gazebo/config/patrol_route.yaml ('' -> legacy single point)
#   OBSTACLE_MAP                 default gazebo/config/obstacle_map.yaml
#   MISSION_AREA_E/N, TRIGGER_DIST, LOITER_S   mission_manager args
#   MPC_ENABLE                   default true
#
# Readiness checks (Gazebo world clock topic, then AP: Frame: line, then
# mavros connected) are a first version written from the documented startup
# behaviour, not from a live test run in this environment -- if a wait times
# out or an auto-detect misses (e.g. FRAME_CLASS wording differs), that
# timeout duration or grep pattern is the first thing to adjust in this file.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$(mktemp -d /tmp/px4_mission_logs.XXXXXX)"
echo "logs: $LOG_DIR (gazebo.log / sitl.log / mission.log)"

WORLD="${WORLD:-obstacle_field}"
VIEW="${VIEW:-0}"
PATROL_ROUTE="${PATROL_ROUTE-$REPO_DIR/gazebo/config/patrol_route.yaml}"
OBSTACLE_MAP="${OBSTACLE_MAP:-$REPO_DIR/gazebo/config/obstacle_map.yaml}"
MISSION_AREA_E="${MISSION_AREA_E:-0.0}"
MISSION_AREA_N="${MISSION_AREA_N:-0.0}"
TRIGGER_DIST="${TRIGGER_DIST:-10.0}"
LOITER_S="${LOITER_S:-10.0}"
MPC_ENABLE="${MPC_ENABLE:-true}"

# --- tear-down: kill every process GROUP this script started, whatever way
# the script exits (normal quit, Ctrl+C, error) -----------------------------
CHILD_PGIDS=()
cleanup() {
  echo ""
  echo "shutting down Gazebo/SITL/mission..."
  for g in "${CHILD_PGIDS[@]}"; do kill -- -"$g" 2>/dev/null; done
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

# --- 1) Gazebo ---------------------------------------------------------------
echo "[1/3] starting Gazebo (WORLD=$WORLD)..."
setsid bash -c "WORLD='$WORLD' VIEW='$VIEW' '$SCRIPT_DIR/run_sim.sh' > '$LOG_DIR/gazebo.log' 2>&1" &
CHILD_PGIDS+=("$!")
wait_for "Gazebo (world clock topic)" 60 bash -c "gz topic -l 2>/dev/null | grep -q '/world/$WORLD/clock'"

# --- 2) SITL ------------------------------------------------------------------
echo "[2/3] starting SITL..."
FIFO=$(mktemp -u /tmp/mav_cmd.XXXXXX.fifo)
mkfifo "$FIFO"
setsid bash -c "( while true; do cat '$FIFO'; done ) | sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --no-rebuild -w > '$LOG_DIR/sitl.log' 2>&1" &
CHILD_PGIDS+=("$!")
wait_for "SITL (frame line in boot log)" 45 bash -c "grep -q 'AP: Frame:' '$LOG_DIR/sitl.log'"
if grep -q 'AP: Frame: UNSUPPORTED' "$LOG_DIR/sitl.log" 2>/dev/null; then
  echo "  frame class unsupported -> auto-fixing (FRAME_CLASS=1, FRAME_TYPE=1)"
  echo 'param set FRAME_CLASS 1' > "$FIFO"
  echo 'param set FRAME_TYPE 1' > "$FIFO"
  sleep 2
fi

# --- 3) MPC mission launch -----------------------------------------------------
echo "[3/3] starting mission (mpc_enable=$MPC_ENABLE, patrol_route=${PATROL_ROUTE:-<legacy single point>})..."
PATROL_ARG=""
[[ -n "$PATROL_ROUTE" ]] && PATROL_ARG="patrol_route:=$PATROL_ROUTE"
setsid bash -c "
  source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
  source '$REPO_DIR/install/setup.bash'
  exec ros2 launch precision_landing truck_mission_mpc.launch.py \
    mpc_enable:=$MPC_ENABLE obstacle_map:=$OBSTACLE_MAP $PATROL_ARG \
    mission_area_e:=$MISSION_AREA_E mission_area_n:=$MISSION_AREA_N \
    trigger_dist:=$TRIGGER_DIST loiter_s:=$LOITER_S \
    > '$LOG_DIR/mission.log' 2>&1
" &
CHILD_PGIDS+=("$!")
wait_for "MAVROS<->SITL link" 60 bash -c "grep -q 'connected: true' <(timeout 3 ros2 topic echo /mavros/state --once 2>/dev/null)"

# --- trajectory recording (for gazebo/plot_3d_trajectory.py --log) --------
RECORD_PID=""
RECORD_OUT=""
start_recording() {
  RECORD_OUT="$LOG_DIR/trajectory_$(date +%H%M%S).csv"
  setsid bash -c "
    source /opt/ros/\${ROS_DISTRO:-humble}/setup.bash
    exec python3 '$SCRIPT_DIR/record_trajectory.py' --out '$RECORD_OUT'
  " > "$LOG_DIR/record.log" 2>&1 &
  RECORD_PID="$!"
  CHILD_PGIDS+=("$RECORD_PID")
  echo "recording -> $RECORD_OUT (type 'record' again to stop and flush it)"
}
stop_recording() {
  [[ -z "$RECORD_PID" ]] && return
  kill -INT -- -"$RECORD_PID" 2>/dev/null   # SIGINT -> graceful flush+close, not a hard kill
  sleep 0.3
  echo "recording stopped -> $RECORD_OUT"
  echo "plot it (on the HOST, not in this container): python3 gazebo/plot_3d_trajectory.py --log $RECORD_OUT --out fig_result"
  RECORD_PID=""
}

# --- interactive console --------------------------------------------------
echo ""
echo "=== all up. commands: land | record | status | log | help | quit ==="
while true; do
  read -r -p "mission> " cmd
  case "$cmd" in
    land|복귀|return)
      ros2 topic pub /mission/task_complete std_msgs/msg/Bool '{data: true}' --once
      echo "task_complete sent -- watch 'log' for RETURN -> LANDING -> ...  \
disarmed onto platform -> shutting down node  before doing anything else."
      ;;
    record|녹화)
      if [[ -z "$RECORD_PID" ]]; then start_recording; else stop_recording; fi
      ;;
    status)
      echo "phase:   $(timeout 2 ros2 topic echo /mission/phase --once 2>/dev/null | head -1)"
      echo "battery: $(timeout 2 ros2 topic echo /mission/battery_s --once 2>/dev/null | head -1)"
      [[ -n "$RECORD_PID" ]] && echo "recording: $RECORD_OUT"
      ;;
    log)
      echo "-- tailing mission log, Ctrl+C returns to this prompt (does NOT stop the mission) --"
      trap '' INT   # swallow Ctrl+C here so it only kills the tail, not the whole script
      tail -n 20 -f "$LOG_DIR/mission.log"
      trap cleanup EXIT INT TERM
      ;;
    help|"")
      echo "  land / 복귀 / return  -- send task_complete (drone returns and lands)"
      echo "  record / 녹화         -- start/stop saving flight (t,e,n,alt) to CSV for the 3D plot"
      echo "  status                -- current mission phase / battery / recording state"
      echo "  log                   -- tail mission log (Ctrl+C to come back here)"
      echo "  quit / exit           -- shut down Gazebo/SITL/mission (stops recording first)"
      ;;
    quit|exit)
      stop_recording
      break
      ;;
    *)
      echo "unknown command '$cmd' -- try: help"
      ;;
  esac
done
