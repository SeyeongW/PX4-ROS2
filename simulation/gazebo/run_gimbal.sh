#!/usr/bin/env bash
# One command to fly the gimbal vehicle. Everything the stack needs is set here
# rather than left to the caller: the XRCE agent (which lives in /usr/local/bin,
# not where run_px4_map.sh looks), GZ_PARTITION, and the gimbal camera's wider
# tan(vfov/2) that mission_manager would otherwise default to the down camera's.
#
#   ./simulation/gazebo/run_gimbal.sh              CJU gimbal + perception
#   ./simulation/gazebo/run_gimbal.sh mission      CJU word-command mission + ArUco landing
#   ./simulation/gazebo/run_gimbal.sh baseline     body-fixed direct landing, to compare
#   LANDING_MAP=mpc-landing-moving ./simulation/gazebo/run_gimbal.sh mission
#                                      legacy 1 km shuttle mission
#   FOLLOW_DRONE=1 ./simulation/gazebo/run_gimbal.sh   optional camera tracking
#   ARUCO_VIEW=0 ./simulation/gazebo/run_gimbal.sh mission   disable the automatic viewer
#   MISSION_VIEW=0 ./simulation/gazebo/run_gimbal.sh mission disable the live map viewer
#   CJU_LOG_ROOT=/data/cju ./simulation/gazebo/run_gimbal.sh mission
#                                      override the persistent artifact root
#
# Ctrl-C stops everything it started.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="${1:-}"
LANDING_MAP="${LANDING_MAP:-cju-track}"
case "$LANDING_MAP" in
  mpc-landing-moving)
    LANDING_WORLD="mpc_landing_200m_moving"
    LANDING_COORDINATES="$SCRIPT_DIR/maps/mpc_landing_200m_moving.yaml"
    ;;
  cju-track)
    LANDING_WORLD="drone_cju"
    LANDING_COORDINATES="$SCRIPT_DIR/maps/drone_cju_track.yaml"
    ;;
  *)
    echo "unknown LANDING_MAP '$LANDING_MAP' (expected mpc-landing-moving or cju-track)" >&2
    exit 2
    ;;
esac

GIMBAL=1
case "$MODE" in
  ""|gimbal) RUN_MISSION=0 ;;
  mission)   RUN_MISSION=1 ;;
  baseline)  RUN_MISSION=1; GIMBAL=0 ;;
  -h|--help|help)
    sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  *) echo "unknown mode '$MODE' (try: gimbal | mission | baseline)" >&2; exit 2 ;;
esac

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PX4_MSGS_SETUP="${PX4_MSGS_SETUP:-${HOME}/px4_ros2_ws/install/setup.bash}"
export GZ_PARTITION="${GZ_PARTITION:-px4_ros2_${USER:-user}}"
if [[ -z "${PX4_MAP_RUNTIME_DIR:-}" ]]; then
  CJU_LOG_ROOT="${CJU_LOG_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/px4-ros2-wang/cju}"
  mkdir -p "$CJU_LOG_ROOT"
  PX4_MAP_RUNTIME_DIR="$(mktemp -d \
    "$CJU_LOG_ROOT/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
else
  mkdir -p "$PX4_MAP_RUNTIME_DIR"
  if find "$PX4_MAP_RUNTIME_DIR" -mindepth 1 -maxdepth 1 -print -quit \
      | grep -q .; then
    echo "ERROR: PX4_MAP_RUNTIME_DIR is not empty: $PX4_MAP_RUNTIME_DIR" >&2
    exit 2
  fi
fi
export PX4_MAP_RUNTIME_DIR
LANDING_COORDINATES_SOURCE="$LANDING_COORDINATES"
LANDING_COORDINATES="$PX4_MAP_RUNTIME_DIR/map.yaml"
cp -- "$LANDING_COORDINATES_SOURCE" "$LANDING_COORDINATES"

# trailer_cue_node publishes Gazebo ENU relative to the drone's local origin.
# Read the run's immutable map snapshot so planning and postflight export use
# exactly the same coordinate contract even if the source YAML later changes.
mapfile -t LANDING_CONFIG < <(
  python3 - "$LANDING_COORDINATES" <<'PY'
import json
import pathlib
import sys

import yaml

document = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pose = document["spawn"]["gazebo_spawn_pose_enu"]
print(json.dumps([float(pose["x"]), float(pose["y"])]))
trailer = document["trailer"]
marker_world_z = (float(trailer["spawn_pose_enu"]["z"])
                  + float(trailer["marker_surface_height_m"]))
local_origin_z = float(document["frames"]["mavros_local"]["origin_enu_m"][2])
print(f"{marker_world_z - local_origin_z:.9f}")
print(float(document.get("mission", {}).get("cruise_altitude_m", 6.0)))
print(trailer["odometry_topic"])
print(float(trailer["cruise_speed_m_s"]))
PY
)
LANDING_SPAWN_ENU="${LANDING_CONFIG[0]}"
LANDING_DECK_Z="${LANDING_CONFIG[1]}"
LANDING_TAKEOFF_ALT="${LANDING_CONFIG[2]}"
LANDING_ODOMETRY_TOPIC="${LANDING_CONFIG[3]}"
LANDING_TRAILER_SPEED="${LANDING_CONFIG[4]}"

XRCE_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_xrce.log"
SIM_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_sim.log"
STACK_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_stack.log"
VIEW_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_aruco_view.log"
MISSION_VIEW_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_mission_view.log"
CUE_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_cue.log"
MISSION_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_mission.log"
ODOMETRY_LOG="$PX4_MAP_RUNTIME_DIR/trailer_odometry.jsonl"
ODOMETRY_ERROR_LOG="$PX4_MAP_RUNTIME_DIR/trailer_odometry.err"
RUN_MANIFEST="$PX4_MAP_RUNTIME_DIR/manifest.tsv"
CSV_EXPORT_LOG="$PX4_MAP_RUNTIME_DIR/flight_csv_export.log"
echo "Run artifacts     : $PX4_MAP_RUNTIME_DIR"

PIDS=()
REQUIRED_PIDS=()
REQUIRED_NAMES=()
WATCHDOG_PID=""
TRAILER_GATE_DIR=""
TRAILER_START_FILE=""
SIM_PID=""
DRIVE_TRAILER_FOR_RUN=1
TRAILER_SPEED_FOR_RUN="${TRAILER_SPEED_M_S:-$LANDING_TRAILER_SPEED}"
if [[ "$LANDING_MAP" == "cju-track" ]]; then
  DRIVE_TRAILER_FOR_RUN=0
fi
RUN_STARTED_UTC="$(date -u +%FT%TZ)"
GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || printf unknown)"
GIT_BRANCH="$(git -C "$REPO_DIR" branch --show-current 2>/dev/null || printf unknown)"
GIT_DIRTY=0
[[ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=normal 2>/dev/null)" ]] \
  && GIT_DIRTY=1
{
  printf 'run_id\t%s\n' "$(basename "$PX4_MAP_RUNTIME_DIR")"
  printf 'started_utc\t%s\n' "$RUN_STARTED_UTC"
  printf 'git_commit\t%s\n' "$GIT_COMMIT"
  printf 'git_branch\t%s\n' "$GIT_BRANCH"
  printf 'git_dirty\t%s\n' "$GIT_DIRTY"
  printf 'mode\t%s\n' "${MODE:-gimbal}"
  printf 'map\t%s\n' "$LANDING_MAP"
  printf 'world\t%s\n' "$LANDING_WORLD"
  printf 'coordinates\t%s\n' 'map.yaml'
  printf 'coordinates_source\t%s\n' "$LANDING_COORDINATES_SOURCE"
  printf 'coordinates_sha256\t%s\n' "$(sha256sum "$LANDING_COORDINATES" | cut -d' ' -f1)"
  printf 'gimbal\t%s\n' "$GIMBAL"
  printf 'takeoff_alt_m\t%s\n' "$LANDING_TAKEOFF_ALT"
  printf 'flight_control_owner\t%s\n' 'px4_native'
  printf 'trailer_speed_m_s\t%s\n' "$TRAILER_SPEED_FOR_RUN"
} >"$RUN_MANIFEST"
git -C "$REPO_DIR" status --short >"$PX4_MAP_RUNTIME_DIR/git_status.txt" || true
cleanup() {
  local status=$?
  local run_result="failed"
  local sim_state=""
  local ulog_paths=()
  local ulog_source=""
  local csv_export="missing"
  local odometry_samples=0
  trap - EXIT INT TERM
  echo
  echo "stopping..."
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  for pid in "${PIDS[@]:-}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  # run_px4_map owns a detached sensor-bridge group and cleans it internally.
  # Let that wrapper finish before using this invocation's group KILL fallback.
  if [[ -n "$SIM_PID" ]]; then
    for _ in {1..100}; do
      sim_state="$(awk '{print $3}' "/proc/$SIM_PID/stat" 2>/dev/null || true)"
      [[ -z "$sim_state" || "$sim_state" == "Z" ]] && break
      sleep 0.1
    done
  fi
  for pid in "${PIDS[@]:-}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  mapfile -t ulog_paths < <(
    sed -n 's|.*Opened full log file: \./||p' "$SIM_LOG" 2>/dev/null
  )
  if (( ${#ulog_paths[@]} == 1 )); then
    ulog_source="${PX4_DIR:-$HOME/PX4-Autopilot}/build/px4_sitl_default/rootfs/${ulog_paths[0]}"
    if [[ -f "$ulog_source" ]]; then
      cp -- "$ulog_source" "$PX4_MAP_RUNTIME_DIR/flight.ulg" ||
        echo "WARNING: could not preserve PX4 ULog: $ulog_source" >&2
    else
      echo "WARNING: PX4 ULog not found: $ulog_source" >&2
    fi
  else
    echo "WARNING: expected one PX4 ULog path, found ${#ulog_paths[@]}" >&2
  fi
  if grep -q 'PRECLAND -> DONE' "$MISSION_LOG" 2>/dev/null; then
    run_result="done"
  elif [[ "$status" == "130" ]]; then
    run_result="interrupted"
  elif [[ "$status" == "0" ]]; then
    run_result="completed"
  fi
  [[ -f "$ODOMETRY_LOG" ]] && odometry_samples="$(wc -l <"$ODOMETRY_LOG")"
  {
    printf 'finished_utc\t%s\n' "$(date -u +%FT%TZ)"
    printf 'exit_status\t%s\n' "$status"
    printf 'result\t%s\n' "$run_result"
    printf 'flight_ulg\t%s\n' "$([[ -s "$PX4_MAP_RUNTIME_DIR/flight.ulg" ]] && printf present || printf missing)"
    printf 'odometry_samples\t%s\n' "$odometry_samples"
  } >>"$RUN_MANIFEST"
  if [[ -s "$PX4_MAP_RUNTIME_DIR/flight.ulg" ]]; then
    if python3 "$SCRIPT_DIR/tools/export_flight_1hz.py" \
        "$PX4_MAP_RUNTIME_DIR" >"$CSV_EXPORT_LOG" 2>&1; then
      csv_export="present"
    else
      echo "WARNING: 1 Hz flight CSV export failed; see $CSV_EXPORT_LOG" >&2
    fi
  fi
  {
    printf 'flight_csv_1hz\t%s\n' "$csv_export"
    printf 'flight_summary_csv\t%s\n' "$csv_export"
    if [[ "$csv_export" == "present" ]]; then
      printf 'flight_csv_schema\t%s\n' 'cju_flight_1hz_v3'
      printf 'flight_csv_rate_hz\t1\n'
      printf 'flight_csv_contract\t%s\n' \
        'one-second interval time-weighted means plus native-rate extrema'
    fi
  } >>"$RUN_MANIFEST"
  if [[ -n "$TRAILER_GATE_DIR" ]]; then
    rm -f "$TRAILER_START_FILE"
    rmdir "$TRAILER_GATE_DIR" 2>/dev/null || true
  fi
  echo "artifacts saved   : $PX4_MAP_RUNTIME_DIR"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ "$LANDING_MAP" == "cju-track" ]]; then
  if [[ "$RUN_MISSION" == "1" ]]; then
    TRAILER_GATE_DIR="$(mktemp -d /tmp/drone_cju_trailer.XXXXXX)"
    TRAILER_START_FILE="$TRAILER_GATE_DIR/start"
    DRIVE_TRAILER_FOR_RUN=1
  fi
fi

# Never kill a process this invocation does not own.  A stale stack is reported
# and must be stopped explicitly by its owner.
if pgrep -f 'px4_sitl_default/bin/px4|native_gz_sensor_bridge|landing_mpc/lib/landing_mpc/|python3 .*simulation/gazebo/(trailer_waypoint_driver.py|tools/aruco_debug_viewer.py)' >/dev/null 2>&1; then
  echo "ERROR: another sim/perception stack is already running." >&2
  pgrep -af 'px4_sitl_default/bin/px4|native_gz_sensor_bridge|landing_mpc/lib/landing_mpc/|python3 .*simulation/gazebo/(trailer_waypoint_driver.py|tools/aruco_debug_viewer.py)' >&2 || true
  exit 4
fi

pgrep -f 'MicroXRCEAgent.*8888' >/dev/null 2>&1 || {
  echo "starting Micro XRCE-DDS agent"
  setsid MicroXRCEAgent udp4 -p 8888 >"$XRCE_LOG" 2>&1 &
  PIDS+=($!)
  sleep 1
}

echo "=== Gazebo + PX4 (GIMBAL=$GIMBAL) — log: $SIM_LOG ==="
GIMBAL="$GIMBAL" FOLLOW_DRONE="${FOLLOW_DRONE:-0}" \
START_XRCE=1 START_MAVROS="${START_MAVROS:-1}" PX4_DAEMON=1 \
DRIVE_TRAILER="$DRIVE_TRAILER_FOR_RUN" TRAILER_SPEED_M_S="$TRAILER_SPEED_FOR_RUN" \
TRAILER_START_FILE="$TRAILER_START_FILE" PX4_MAP_COORDINATES="$LANDING_COORDINATES" \
  setsid "$SCRIPT_DIR/run_px4_map.sh" "$LANDING_MAP" >"$SIM_LOG" 2>&1 &
SIM_PID=$!
PIDS+=("$SIM_PID")
REQUIRED_NAMES+=(simulator)
REQUIRED_PIDS+=("$SIM_PID")

echo -n "waiting for the vehicle to spawn"
vehicle_ready=0
for _ in $(seq 120); do
  if gz topic -l 2>/dev/null | grep -q 'gimbal_camera/image\|down_camera/image'; then
    vehicle_ready=1
    break
  fi
  if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
    echo; echo "ERROR: the simulator exited. Last lines:" >&2
    tail -15 "$SIM_LOG" >&2
    exit 1
  fi
  echo -n "."; sleep 2
done
if [[ "$vehicle_ready" != "1" ]]; then
  echo; echo "ERROR: vehicle sensor topics did not appear. Last lines:" >&2
  tail -15 "$SIM_LOG" >&2
  exit 1
fi
echo " up."

setsid stdbuf -oL gz topic -e --json-output -t "$LANDING_ODOMETRY_TOPIC" \
  >"$ODOMETRY_LOG" 2>"$ODOMETRY_ERROR_LOG" &
ODOMETRY_PID=$!
PIDS+=("$ODOMETRY_PID")
REQUIRED_NAMES+=(trailer_odometry)
REQUIRED_PIDS+=("$ODOMETRY_PID")
for _ in {1..50}; do
  [[ -s "$ODOMETRY_LOG" ]] && break
  kill -0 "$ODOMETRY_PID" 2>/dev/null || break
  sleep 0.1
done
if [[ ! -s "$ODOMETRY_LOG" ]]; then
  echo "ERROR: no trailer odometry on $LANDING_ODOMETRY_TOPIC" >&2
  cat "$ODOMETRY_ERROR_LOG" >&2 || true
  exit 5
fi
echo "Trailer odometry : $ODOMETRY_LOG"

# ROS's setup scripts read unset variables, so -u has to stand down for them.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
[[ -f "$PX4_MSGS_SETUP" ]] && source "$PX4_MSGS_SETUP"
# shellcheck disable=SC1091
source "$REPO_DIR/install/setup.bash"
set -u
python3 -c 'import px4_msgs' >/dev/null 2>&1 || {
  echo "ERROR: px4_msgs not found; set PX4_MSGS_SETUP to its install/setup.bash." >&2
  exit 3
}

if [[ "$GIMBAL" == "1" ]]; then
  echo "=== gimbal + perception — log: $STACK_LOG ==="
  # Same-frame markers describe one rigid deck, so gross disagreement is an
  # invalid measurement.  Keep the safety gate tunable without disabling it.
  PAIR_GATE_ARG=(
    max_pair_disagreement_m:="${MAX_PAIR_DISAGREEMENT_M:-1.0}"
  )
  setsid ros2 launch landing_mpc gimbal_perception.launch.py \
    world:="$LANDING_WORLD" deck_z:="$LANDING_DECK_Z" "${PAIR_GATE_ARG[@]}" \
    >"$STACK_LOG" 2>&1 &
  STACK_PID=$!
  PIDS+=("$STACK_PID")
  REQUIRED_NAMES+=(perception)
  REQUIRED_PIDS+=("$STACK_PID")
else
  echo "=== baseline perception (body-fixed camera) ==="
  setsid ros2 run landing_mpc aruco_detector_node --ros-args -p use_sim_time:=true \
    >"$STACK_LOG" 2>&1 &
  PIDS+=("$!"); REQUIRED_NAMES+=(aruco_detector); REQUIRED_PIDS+=("$!")
  setsid ros2 run landing_mpc marker_tf_node --ros-args -p use_sim_time:=true \
    -p deck_z:="$LANDING_DECK_Z" \
    >>"$STACK_LOG" 2>&1 &
  PIDS+=("$!"); REQUIRED_NAMES+=(marker_tf); REQUIRED_PIDS+=("$!")
  setsid ros2 run landing_mpc marker_kf_node --ros-args -p use_sim_time:=true \
    -p deck_z:="$LANDING_DECK_Z" \
    >>"$STACK_LOG" 2>&1 &
  PIDS+=("$!"); REQUIRED_NAMES+=(marker_kf); REQUIRED_PIDS+=("$!")
fi
sleep 5

if [[ "${HEADLESS:-0}" != "1" && "${ARUCO_VIEW:-1}" == "1" ]]; then
  echo -n "waiting for the first ArUco debug frame"
  if ! timeout "${ARUCO_VIEW_TIMEOUT_S:-30}" \
      ros2 topic echo --once /aruco/debug_image/compressed \
      >/dev/null 2>&1; then
    echo
    echo "ERROR: /aruco/debug_image/compressed produced no frame." >&2
    exit 6
  fi
  echo " received."

  : >"$VIEW_LOG"
  setsid python3 -u "$SCRIPT_DIR/tools/aruco_debug_viewer.py" \
    >"$VIEW_LOG" 2>&1 &
  PIDS+=($!)
  ARUCO_VIEW_PID=$!
  ARUCO_VIEW_READY=0
  echo -n "waiting for the ArUco result window"
  for _ in $(seq 1 100); do
    if grep -q 'displaying .* detector frames' "$VIEW_LOG"; then
      ARUCO_VIEW_READY=1
      break
    fi
    if ! kill -0 "$ARUCO_VIEW_PID" 2>/dev/null; then
      break
    fi
    echo -n "."
    sleep 0.1
  done
  echo
  if [[ "$ARUCO_VIEW_READY" != "1" ]]; then
    echo "ERROR: the ArUco result window did not receive a frame." >&2
    cat "$VIEW_LOG" >&2
    exit 6
  fi
  echo "=== ArUco result window ready ==="
fi

if [[ "$RUN_MISSION" == "1" ]]; then
  echo "=== mission — log: $MISSION_LOG ==="
  setsid ros2 run landing_mpc trailer_cue_node --ros-args \
    -p use_sim_time:=true -p world:="$LANDING_WORLD" \
    -p "spawn_enu:=$LANDING_SPAWN_ENU" -p deck_z:="$LANDING_DECK_Z" \
    >"$CUE_LOG" 2>&1 &
  CUE_PID=$!
  PIDS+=("$CUE_PID")
  REQUIRED_NAMES+=(trailer_cue)
  REQUIRED_PIDS+=("$CUE_PID")
  sleep 2
  MISSION_ARGS=(-p auto_start:=true)
  if [[ "$LANDING_MAP" == "cju-track" ]]; then
    MISSION_ARGS=(
      -p auto_start:=false
      -p takeoff_alt:="$LANDING_TAKEOFF_ALT"
      -p "mission_map_yaml:=$LANDING_COORDINATES"
    )
  fi
  setsid ros2 run landing_mpc mission_manager_node --ros-args -p use_sim_time:=true \
    "${MISSION_ARGS[@]}" \
    >"$MISSION_LOG" 2>&1 &
  MISSION_PID=$!
  PIDS+=("$MISSION_PID")
  REQUIRED_NAMES+=(mission_manager)
  REQUIRED_PIDS+=("$MISSION_PID")

  if [[ "$LANDING_MAP" == "cju-track" && "${HEADLESS:-0}" != "1" \
      && "${MISSION_VIEW:-1}" == "1" ]]; then
    if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
      echo "WARNING: live mission map skipped: no GUI display" >&2
    else
      setsid python3 -u "$SCRIPT_DIR/tools/cju_mission_ui.py" \
        --live --map "$LANDING_COORDINATES" \
        >"$MISSION_VIEW_LOG" 2>&1 &
      MISSION_VIEW_PID=$!
      PIDS+=("$MISSION_VIEW_PID")
      sleep 1
      if ! kill -0 "$MISSION_VIEW_PID" 2>/dev/null; then
        echo "WARNING: live mission map failed; see $MISSION_VIEW_LOG" >&2
      fi
    fi
  fi
fi

required_alive() {
  local index
  for index in "${!REQUIRED_PIDS[@]}"; do
    if ! kill -0 "${REQUIRED_PIDS[$index]}" 2>/dev/null \
        || [[ "$(awk '{print $3}' "/proc/${REQUIRED_PIDS[$index]}/stat" 2>/dev/null)" == "Z" ]]; then
      echo "ERROR: required component exited: ${REQUIRED_NAMES[$index]}" >&2
      return 1
    fi
  done
}
watch_required() {
  while sleep 0.5; do
    required_alive || {
      kill -TERM "$$" 2>/dev/null || true
      return
    }
  done
}
watch_required &
WATCHDOG_PID=$!

cat <<EOF

  ready.  Ctrl-C here stops everything.

  watch the camera :  ros2 run rqt_image_view rqt_image_view /gimbal_camera/image
  ArUco result view : opens automatically (ARUCO_VIEW=0 disables it)
                      JPEG topic, ~20 kB/frame instead of 1.4 MB raw
                      (marker outline, centre-to-centre dx/dy in px and m, and
                       the FILL gauge that shows the near-field blind zone)
  watch the gimbal :  rviz2      (Fixed Frame: px4_local_enu, then Add -> TF)
  live mission map :  opens automatically (MISSION_VIEW=0 disables it)
EOF

if [[ "$RUN_MISSION" == "1" && "$LANDING_MAP" == "cju-track" ]]; then
  retry_command() {
    if ! required_alive; then
      echo "  $1 실패 — 필수 구성요소가 종료됨" >&2
      exit 7
    fi
    echo "  $1 시간초과 — Gazebo ▶/clock 확인 후 같은 명령 재입력" >&2
  }
  wait_for_states() {
    local waiter pid_index status=0
    setsid timeout "$2" bash -c \
      'ros2 topic echo /mission/state std_msgs/msg/String 2>/dev/null | grep -m1 -E "^data: ($1)$"' \
      _ "$1" >/dev/null &
    waiter=$!
    pid_index=${#PIDS[@]}
    PIDS+=("$waiter")
    wait "$waiter" || status=$?
    unset 'PIDS[pid_index]'
    return "$status"
  }
  send_until_state() {
    local command="$1" states="$2" timeout_s="$3" publisher pid_index status=0
    setsid ros2 topic pub --rate 5 --wait-matching-subscriptions 1 --print 100000 \
      /mission/command std_msgs/msg/String "{data: '$command'}" >/dev/null 2>&1 &
    publisher=$!
    pid_index=${#PIDS[@]}
    PIDS+=("$publisher")
    wait_for_states "$states" "$timeout_s" || status=$?
    kill -TERM -- "-$publisher" 2>/dev/null || true
    wait "$publisher" 2>/dev/null || true
    unset 'PIDS[pid_index]'
    return "$status"
  }

  commands=(takeoff land)
  step=0
  echo '  명령 순서: takeoff → 자동 A*/geometry B-spline/PX4 Goto → land'
  while (( step < ${#commands[@]} )); do
    IFS= read -r -p '  명령> ' command || break
    if [[ "$command" != "${commands[$step]}" ]]; then
      echo "  지금 입력할 명령: ${commands[$step]}" >&2
      continue
    fi
    case "$command" in
      takeoff)
        echo '  Phase 0: 상태/센서/PX4/경로계획기 PRECHECK 확인 중...'
        echo "  Phase 1: PX4 native takeoff — 고도 ${LANDING_TAKEOFF_ALT} m..."
        send_until_state takeoff 'TAKEOFF|MISSION_PLAN|MISSION|HOVER' 10 || {
          retry_command 'TAKEOFF 상태 확인'
          continue
        }
        wait_for_states 'MISSION|HOVER' 120 || {
          retry_command 'geometry B-spline/PX4 Goto 상태 확인'
          continue
        }
        touch "$TRAILER_START_FILE"
        echo '  Phase 2: YAML A*→geometry B-spline을 PX4 Goto로 추종 중...'
        wait_for_states HOVER 180 || {
          retry_command 'Phase 2 HOVER 확인'
          continue
        }
        echo "  Phase 2 완료 — (50,50) 호버, 트레일러 ${TRAILER_SPEED_FOR_RUN} m/s"
        ;;
      land)
        send_until_state land 'RETURN_PLAN|RETURN|PRECLAND|DONE' 30 || {
          retry_command 'Phase 3 경로 계획 확인'
          continue
        }
        echo '  Phase 3: A*→geometry B-spline으로 트레일러에 접근한 뒤 PX4 NAV_PRECLAND 수행 중...'
        wait_for_states DONE 180 || {
          retry_command '착륙 완료 확인'
          continue
        }
        echo '  Phase 3 완료 — 이동 트레일러 착륙 및 무장해제'
        ;;
    esac
    printf 'command_accepted\t%s\n' "$command" >>"$RUN_MANIFEST"
    ((step += 1))
  done
fi

# Surface the lines worth reading instead of making anyone tail three files.
# Keep the pipeline in an owned group so a watchdog signal interrupts `wait`
# immediately instead of being deferred behind a foreground `tail -f`.
setsid bash -c '
  tail -f "$1" "$2" 2>/dev/null \
    | grep --line-buffered -E "gimbal:|detections|state|ABORT|ERROR|WARN"
' tail "$STACK_LOG" "$MISSION_LOG" &
TAIL_PID=$!
PIDS+=("$TAIL_PID")
wait "$TAIL_PID" || true
