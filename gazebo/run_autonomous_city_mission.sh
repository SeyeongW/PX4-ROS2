#!/usr/bin/env bash
# Run the city map, PX4 SITL and the A*/SFC/B-spline/MPC mission as one unit.
# Native PX4 L1-style AUTO.MISSION is a separate, mutually-exclusive A/B run;
# see docs/px4_l1_bspline_cross_validation.md.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PX4_MSGS_SETUP="${PX4_MSGS_SETUP:-$HOME/px4_ros2_ws/install/setup.bash}"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
PX4_PARAM_BIN="$PX4_DIR/build/px4_sitl_default/bin/px4-param"
MISSION_CONFIG="${MISSION_CONFIG:-$REPO_DIR/autonomy_planner/config/city_uav_1m_mission.yaml}"
MAP_YAML="${MAP_YAML:-$SCRIPT_DIR/maps/city_coordinates_uav.yaml}"
RUNTIME_DIR="${PX4_AUTONOMY_RUNTIME_DIR:-/tmp/px4_city_autonomy_${USER:-user}}"
SIM_LOG="$RUNTIME_DIR/simulation.log"
MISSION_LOG="$RUNTIME_DIR/mission.log"
TRAILER_LOG="$RUNTIME_DIR/trailer_motion.log"
FRONT_ARUCO_LOG="$RUNTIME_DIR/front_aruco.log"
DOWN_ARUCO_LOG="$RUNTIME_DIR/down_aruco.log"

if [[ "${1:-}" == "--plan-only" ]]; then
  export PYTHONPATH="$REPO_DIR/autonomy_planner${PYTHONPATH:+:$PYTHONPATH}"
  exec python3 -m autonomy_planner.planning_screening \
    --repository-root "$REPO_DIR"
fi

for setup in "$ROS_SETUP" "$PX4_MSGS_SETUP"; do
  if [[ ! -f "$setup" ]]; then
    echo "ERROR: ROS setup is missing: $setup" >&2
    echo "       Build matching PX4 v1.17 px4_msgs in ~/px4_ros2_ws first." >&2
    exit 2
  fi
  set +u
  # shellcheck disable=SC1090
  source "$setup"
  set -u
done
export PYTHONPATH="$REPO_DIR/autonomy_planner${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$REPO_DIR/camera_detection:$PYTHONPATH"
python3 -c 'import gz.transport13, numpy, px4_msgs, rclpy, scipy, yaml' || {
  echo "ERROR: autonomy Python/ROS dependencies are incomplete." >&2
  exit 2
}

mkdir -p "$RUNTIME_DIR"
export GZ_PARTITION="${GZ_PARTITION:-px4_city_autonomy_${USER:-user}}"
SIM_PID=""
TRAILER_PID=""
FRONT_ARUCO_PID=""
DOWN_ARUCO_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for process_id in "$DOWN_ARUCO_PID" "$FRONT_ARUCO_PID" "$TRAILER_PID"; do
    if [[ -n "$process_id" ]] && kill -0 "$process_id" 2>/dev/null; then
      kill -TERM "$process_id" 2>/dev/null || true
      wait "$process_id" 2>/dev/null || true
    fi
  done
  if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
    kill -TERM -- "-$SIM_PID" 2>/dev/null || true
    for _ in {1..15}; do
      kill -0 "$SIM_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SIM_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "Mission authority  : ROS 2 OFFBOARD (B-spline + local jerk MPC)"
echo "A/B note           : stop this process before native PX4 AUTO.MISSION L1-style testing"
echo "Goal ENU           : x=200 m, y=-125 m"
echo "Sensor safety      : native Gazebo depth + downward lidar, 10 m clearance"
echo "GZ_PARTITION       : $GZ_PARTITION"
echo "Simulation log     : $SIM_LOG"
echo "Mission log        : $MISSION_LOG"
echo "Trailer log        : $TRAILER_LOG"
echo "ArUco logs         : $FRONT_ARUCO_LOG / $DOWN_ARUCO_LOG"

MISSION_EXTRA_ARGS=()
if [[ -n "${MISSION_EXTRA_ROS_ARGS:-}" ]]; then
  # Intended for repeatable local diagnostics, for example:
  # MISSION_EXTRA_ROS_ARGS='-p takeoff_hold_s:=60.0'
  read -r -a MISSION_EXTRA_ARGS <<<"$MISSION_EXTRA_ROS_ARGS"
fi

setsid env \
  START_MAVROS=0 \
  START_BRIDGE=1 \
  START_XRCE=1 \
  PX4_DAEMON=1 \
  PHYSICS_ENGINE="${PHYSICS_ENGINE:-gz-physics-dartsim-plugin}" \
  GZ_PARTITION="$GZ_PARTITION" \
  PX4_MAP_RUNTIME_DIR="$RUNTIME_DIR/map_launcher" \
  "$SCRIPT_DIR/run_px4_map.sh" city "$@" >"$SIM_LOG" 2>&1 &
SIM_PID=$!

ready=0
for _ in {1..120}; do
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    echo "ERROR: PX4/Gazebo launcher stopped before DDS became ready." >&2
    tail -100 "$SIM_LOG" >&2 || true
    exit 3
  fi
  if ros2 topic list 2>/dev/null | grep -qx '/fmu/out/vehicle_local_position_v1'; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  echo "ERROR: timed out waiting for PX4 /fmu DDS topics." >&2
  tail -100 "$SIM_LOG" >&2 || true
  exit 3
fi

# This is a fully onboard SITL mission, so no QGroundControl heartbeat is
# expected. PX4 v1.17 otherwise rejects arming while NAV_DLL_ACT is non-zero.
if [[ ! -x "$PX4_PARAM_BIN" ]]; then
  echo "ERROR: PX4 parameter tool is missing: $PX4_PARAM_BIN" >&2
  exit 3
fi
"$PX4_PARAM_BIN" set NAV_DLL_ACT 0 >/dev/null

# The map launcher owns Gazebo, PX4, XRCE and the Harmonic-native sensor
# bridge.  These three ROS processes remain subordinate to this wrapper so a
# single Ctrl-C cannot leave a second controller or moving platform behind.
python3 -u -m autonomy_planner.trailer_motion_node \
  --ros-args --params-file "$REPO_DIR/autonomy_planner/config/trailer_motion.yaml" \
  >"$TRAILER_LOG" 2>&1 &
TRAILER_PID=$!

python3 -u -m camera_detection.aruco_pose_node \
  --ros-args -r __node:=front_aruco_pose \
  --params-file "$REPO_DIR/camera_detection/config/aruco_front.yaml" \
  >"$FRONT_ARUCO_LOG" 2>&1 &
FRONT_ARUCO_PID=$!

python3 -u -m camera_detection.aruco_pose_node \
  --ros-args -r __node:=down_aruco_pose \
  --params-file "$REPO_DIR/camera_detection/config/aruco_down.yaml" \
  >"$DOWN_ARUCO_LOG" 2>&1 &
DOWN_ARUCO_PID=$!

sleep 2
for process_spec in \
  "$TRAILER_PID:$TRAILER_LOG:trailer motion" \
  "$FRONT_ARUCO_PID:$FRONT_ARUCO_LOG:front ArUco" \
  "$DOWN_ARUCO_PID:$DOWN_ARUCO_LOG:down ArUco"; do
  IFS=: read -r process_id process_log process_name <<<"$process_spec"
  if ! kill -0 "$process_id" 2>/dev/null; then
    echo "ERROR: $process_name process stopped during startup." >&2
    tail -80 "$process_log" >&2 || true
    exit 4
  fi
done

python3 -u -m autonomy_planner.mission_node \
  --ros-args \
  --params-file "$MISSION_CONFIG" \
  -p "map_yaml:=$MAP_YAML" \
  "${MISSION_EXTRA_ARGS[@]}" 2>&1 | tee "$MISSION_LOG"
