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
#   DIRECT_LANDING=1 LANDING_MAP=city ... mission
#                                      city takeoff above a held trailer, then land
#   LANDING_MAP=city ./simulation/gazebo/run_gimbal.sh mission
#                                      city GPS-SITL mission (10 m deck takeoff)
#   FOLLOW_DRONE=1 ./simulation/gazebo/run_gimbal.sh   optional camera tracking
#   ARUCO_VIEW=0 ./simulation/gazebo/run_gimbal.sh mission   disable the automatic viewer
#   MISSION_VIEW=0 ./simulation/gazebo/run_gimbal.sh mission disable the live map viewer
#   TRAILER_CUE_SOURCE=gps TRAILER_LINK=sim \
#     ./simulation/gazebo/run_gimbal.sh mission
#                                      MAVLink-in-the-loop GPS simulation
#   CJU_LOG_ROOT=/data/cju ./simulation/gazebo/run_gimbal.sh mission
#                                      override the persistent artifact root
#   CITY_LOG_ROOT=/data/city LANDING_MAP=city ./simulation/gazebo/run_gimbal.sh mission
#                                      override city run artifacts
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
  city)
    LANDING_WORLD="applepark_city_uav"
    LANDING_COORDINATES="$SCRIPT_DIR/maps/city_coordinates_uav.yaml"
    ;;
  *)
    echo "unknown LANDING_MAP '$LANDING_MAP' (expected mpc-landing-moving, cju-track, or city)" >&2
    exit 2
    ;;
esac
LANDING_COORDINATES="${PX4_MAP_COORDINATES:-$LANDING_COORDINATES}"

GIMBAL=1
case "$MODE" in
  ""|gimbal) RUN_MISSION=0 ;;
  mission)   RUN_MISSION=1 ;;
  baseline)  RUN_MISSION=1; GIMBAL=0 ;;
  -h|--help|help)
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  *) echo "unknown mode '$MODE' (try: gimbal | mission | baseline)" >&2; exit 2 ;;
esac
DIRECT_LANDING="${DIRECT_LANDING:-0}"
if [[ "$DIRECT_LANDING" != "0" && "$DIRECT_LANDING" != "1" ]]; then
  echo "DIRECT_LANDING must be 0 or 1" >&2
  exit 2
fi
if [[ "$DIRECT_LANDING" == "1" \
    && ( "$RUN_MISSION" != "1" \
         || ( "$LANDING_MAP" != "mpc-landing-moving" \
              && "$LANDING_MAP" != "city" ) ) ]]; then
  echo "DIRECT_LANDING=1 requires mission mode and a city or mpc-landing-moving map" >&2
  exit 2
fi
AUTO_SEQUENCE="${AUTO_SEQUENCE:-0}"
if [[ "$AUTO_SEQUENCE" != "0" && "$AUTO_SEQUENCE" != "1" ]]; then
  echo "AUTO_SEQUENCE must be 0 or 1" >&2
  exit 2
fi
PATH_ONLY="${PATH_ONLY:-0}"
PATH_ONLY_REPLANS="${PATH_ONLY_REPLANS:-3}"
PATH_ONLY_TIMEOUT_S="${PATH_ONLY_TIMEOUT_S:-600}"
if [[ "$PATH_ONLY" != "0" && "$PATH_ONLY" != "1" ]]; then
  echo "PATH_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "$PATH_ONLY" == "1" \
    && ( "$RUN_MISSION" != "1" || "$LANDING_MAP" != "city" \
         || "$DIRECT_LANDING" == "1" || "$AUTO_SEQUENCE" != "1" ) ]]; then
  echo "PATH_ONLY=1 requires an automatic non-direct city mission" >&2
  exit 2
fi
if ! [[ "$PATH_ONLY_REPLANS" =~ ^[1-9][0-9]*$ \
    && "$PATH_ONLY_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]]; then
  echo "PATH_ONLY_REPLANS and PATH_ONLY_TIMEOUT_S must be positive integers" >&2
  exit 2
fi
if [[ "$AUTO_SEQUENCE" == "1" \
    && ( "$RUN_MISSION" != "1" || "$DIRECT_LANDING" == "1" \
         || ( "$LANDING_MAP" != "cju-track" \
              && "$LANDING_MAP" != "city" ) ) ]]; then
  echo "AUTO_SEQUENCE=1 requires a non-direct city or cju-track mission" >&2
  exit 2
fi

# A map profile owns its route start unless the caller explicitly overrides
# it.  This keeps the visual WP0 mission and the quick WP12 landing YAML as two
# reproducible configurations instead of a pile of shell-only coordinates.
if [[ -z "${TRAILER_START_INDEX+x}" ]]; then
  TRAILER_START_INDEX="$(python3 - "$LANDING_COORDINATES" <<'PY'
import pathlib
import sys

import yaml

document = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
trailer = document.get("trailer", {})
value = trailer.get("start_index", 0)
if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise SystemExit("trailer.start_index must be a nonnegative integer")
waypoints = trailer.get("waypoints_enu_m")
if waypoints is not None and value >= len(waypoints):
    raise SystemExit("trailer.start_index is outside waypoints_enu_m")
print(value)
PY
)"
fi
if ! [[ "$TRAILER_START_INDEX" =~ ^[0-9]+$ ]]; then
  echo "TRAILER_START_INDEX must be a nonnegative integer" >&2
  exit 2
fi
if [[ "$LANDING_MAP" == "city" ]]; then
  python3 - "$LANDING_COORDINATES" "$TRAILER_START_INDEX" "$DIRECT_LANDING" <<'PY'
import math
import pathlib
import sys

import yaml

document = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
trailer = document["trailer"]
start_index = int(sys.argv[2])
waypoints = trailer["waypoints_enu_m"]
if start_index >= len(waypoints):
    raise SystemExit("TRAILER_START_INDEX is outside waypoints_enu_m")
spawn = trailer["spawn_pose_enu"]
start = waypoints[start_index]
if not (math.isclose(float(spawn["x"]), float(start[0]), abs_tol=1.0e-9)
        and math.isclose(float(spawn["y"]), float(start[1]), abs_tol=1.0e-9)):
    raise SystemExit(
        "trailer.spawn_pose_enu must match waypoints_enu_m[TRAILER_START_INDEX]"
    )
if start_index != 0 and sys.argv[3] != "1":
    raise SystemExit(
        "a nonzero city trailer start_index requires DIRECT_LANDING=1"
    )
PY
fi
EXIT_ON_DONE="${EXIT_ON_DONE:-0}"
if [[ "$EXIT_ON_DONE" != "0" && "$EXIT_ON_DONE" != "1" ]]; then
  echo "EXIT_ON_DONE must be 0 or 1" >&2
  exit 2
fi
if [[ "$EXIT_ON_DONE" == "1" && ( "$RUN_MISSION" != "1" \
    || ( "$DIRECT_LANDING" != "1" && "$AUTO_SEQUENCE" != "1" ) ) ]]; then
  echo "EXIT_ON_DONE=1 requires direct landing or an automatic mission sequence" >&2
  exit 2
fi

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PX4_MSGS_SETUP="${PX4_MSGS_SETUP:-${HOME}/px4_ros2_ws/install/setup.bash}"
DEFAULT_TRAILER_CUE_SOURCE=gazebo
DEFAULT_TRAILER_LINK=1
DEFAULT_GPS_MAX_DISTANCE_M=200.0
DEFAULT_LANDING_DONE_TIMEOUT_S=180
DEFAULT_MISSION_HOVER_TIMEOUT_S=180
if [[ "$LANDING_MAP" == "city" ]]; then
  DEFAULT_TRAILER_CUE_SOURCE=gps
  DEFAULT_TRAILER_LINK=sim
  DEFAULT_GPS_MAX_DISTANCE_M=2000.0
  DEFAULT_LANDING_DONE_TIMEOUT_S=600
  DEFAULT_MISSION_HOVER_TIMEOUT_S=300
fi
TRAILER_CUE_SOURCE="${TRAILER_CUE_SOURCE:-$DEFAULT_TRAILER_CUE_SOURCE}"
TRAILER_LINK="${TRAILER_LINK:-$DEFAULT_TRAILER_LINK}"
TRAILER_DEV="${TRAILER_DEV:-/dev/ttyUSB0}"
TRAILER_BAUD="${TRAILER_BAUD:-57600}"
TRAILER_SYSID="${TRAILER_SYSID:-1}"
GPS_INPUT_TIMEOUT_S="${GPS_INPUT_TIMEOUT_S:-1.0}"
GPS_CUE_TIMEOUT_S="${GPS_CUE_TIMEOUT_S:-1.0}"
GPS_MAX_DISTANCE_M="${GPS_MAX_DISTANCE_M:-$DEFAULT_GPS_MAX_DISTANCE_M}"
ALLOW_EXTERNAL_GPS_SITL="${ALLOW_EXTERNAL_GPS_SITL:-0}"
GPS_SIM_POSITION_NOISE_M="${GPS_SIM_POSITION_NOISE_M:-0.03}"
GPS_SIM_VELOCITY_NOISE_M_S="${GPS_SIM_VELOCITY_NOISE_M_S:-0.02}"
GPS_SIM_DELAY_S="${GPS_SIM_DELAY_S:-0.08}"
GPS_SIM_DROPOUT="${GPS_SIM_DROPOUT:-0.0}"
GPS_SIM_SEED="${GPS_SIM_SEED:-1}"
GPS_SIM_POSITION_RATE_HZ="${GPS_SIM_POSITION_RATE_HZ:-5.0}"
GPS_SIM_STATUS_RATE_HZ="${GPS_SIM_STATUS_RATE_HZ:-1.0}"
case "$TRAILER_CUE_SOURCE" in
  gazebo|gps) ;;
  *)
    echo "TRAILER_CUE_SOURCE must be 'gazebo' or 'gps'" >&2
    exit 2
    ;;
esac
if [[ "$TRAILER_CUE_SOURCE" == "gps" && "$TRAILER_LINK" != "sim" \
    && "$ALLOW_EXTERNAL_GPS_SITL" != "1" ]]; then
  echo "GPS+SITL mixes a physical GPS target with the Gazebo ArUco target." >&2
  echo "Use flight/trailer_link/run_gps_cue.sh for hardware, or set" >&2
  echo "ALLOW_EXTERNAL_GPS_SITL=1 only for an intentional wiring test." >&2
  exit 2
fi
if [[ "$TRAILER_CUE_SOURCE" == "gps" && "$TRAILER_LINK" != "sim" \
    && -z "${TRAILER_DECK_Z_M:-}" ]]; then
  echo "GPS cue requires measured TRAILER_DECK_Z_M in PX4 local ENU." >&2
  exit 2
fi
if [[ "$TRAILER_LINK" != "0" && "$TRAILER_LINK" != "1" \
    && "$TRAILER_LINK" != "sim" ]]; then
  echo "TRAILER_LINK must be 0, 1, or sim (Gazebo MAVLink PTY)" >&2
  exit 2
fi
if [[ "$TRAILER_LINK" == "sim" && "$TRAILER_CUE_SOURCE" != "gps" ]]; then
  echo "TRAILER_LINK=sim requires TRAILER_CUE_SOURCE=gps" >&2
  exit 2
fi
if ! [[ "$TRAILER_BAUD" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAILER_BAUD must be a positive integer" >&2
  exit 2
fi
python3 - "$GPS_MAX_DISTANCE_M" "$GPS_INPUT_TIMEOUT_S" \
  "$GPS_CUE_TIMEOUT_S" "$GPS_SIM_POSITION_RATE_HZ" \
  "$GPS_SIM_STATUS_RATE_HZ" <<'PY'
import math
import sys

names = ('GPS_MAX_DISTANCE_M', 'GPS_INPUT_TIMEOUT_S', 'GPS_CUE_TIMEOUT_S',
         'GPS_SIM_POSITION_RATE_HZ', 'GPS_SIM_STATUS_RATE_HZ')
for name, text in zip(names, sys.argv[1:]):
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f'{name} must be finite and positive')
PY
if ! [[ "$TRAILER_SYSID" =~ ^[0-9]+$ ]] \
    || (( TRAILER_SYSID < 1 || TRAILER_SYSID > 254 )); then
  echo "TRAILER_SYSID must be 1..254" >&2
  exit 2
fi
if [[ "$TRAILER_CUE_SOURCE" == "gps" && "$TRAILER_LINK" == "1" ]]; then
  TRAILER_CANON="$(readlink -f "$TRAILER_DEV" 2>/dev/null || true)"
  if [[ -z "$TRAILER_CANON" || ! -c "$TRAILER_CANON" ]]; then
    echo "trailer radio is not a serial device: $TRAILER_DEV" >&2
    exit 2
  fi
fi
export GZ_PARTITION="${GZ_PARTITION:-px4_ros2_${USER:-user}}"
if [[ -z "${PX4_MAP_RUNTIME_DIR:-}" ]]; then
  if [[ "$LANDING_MAP" == "city" ]]; then
    RUN_LOG_ROOT="${CITY_LOG_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/px4-ros2-jo/city}"
  else
    RUN_LOG_ROOT="${CJU_LOG_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/px4-ros2-jo/cju}"
  fi
  mkdir -p "$RUN_LOG_ROOT"
  PX4_MAP_RUNTIME_DIR="$(mktemp -d \
    "$RUN_LOG_ROOT/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
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

# An explicit speed override is part of this run's immutable YAML snapshot,
# not just a shell-only driver setting. This keeps the live YAML view, planner
# metadata and postflight report on the same effective trailer speed.
if [[ -n "${TRAILER_SPEED_M_S+x}" ]]; then
  python3 - "$LANDING_COORDINATES" "$TRAILER_SPEED_M_S" <<'PY'
import math
import pathlib
import sys

import yaml

path = pathlib.Path(sys.argv[1])
speed = float(sys.argv[2])
if not math.isfinite(speed) or speed <= 0.0:
    raise SystemExit("TRAILER_SPEED_M_S must be finite and positive")
document = yaml.safe_load(path.read_text(encoding="utf-8"))
document["trailer"]["cruise_speed_m_s"] = speed
path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
PY
fi

# A city sortie starts on the selected trailer route waypoint.  Keep the
# checked-in expansion map as the independent geometry fixture, but make this
# run snapshot place the PX4 model and its local origin on that deck pose.
if [[ "$LANDING_MAP" == "city" ]]; then
  python3 - "$LANDING_COORDINATES" <<'PY'
import pathlib
import sys

import yaml

path = pathlib.Path(sys.argv[1])
document = yaml.safe_load(path.read_text(encoding="utf-8"))
trailer = document["trailer"]
trailer_pose = trailer["spawn_pose_enu"]
deck_z = (float(trailer_pose["z"])
          + float(trailer["marker_surface_height_m"]))
spawn = document["spawn"]["gazebo_spawn_pose_enu"]
old_origin_z = float(document["frames"]["px4_local"]["origin_enu_m"][2])
base_link_offset_z = old_origin_z - float(spawn["z"])
spawn.update({
    "x": float(trailer_pose["x"]),
    "y": float(trailer_pose["y"]),
    "z": deck_z,
})
document["spawn"]["surface"] = "moving_trailer_marker_surface"
origin = [spawn["x"], spawn["y"], deck_z + base_link_offset_z]
document["frames"]["px4_local"]["origin_enu_m"] = origin
document["frames"]["px4_local"]["origin_reference"] = (
    "PX4 x500 base_link at rest on the trailer "
    f"WP{int(trailer.get('start_index', 0))} deck")
document["frames"]["mavros_local"]["origin_enu_m"] = origin
path.write_text(
    yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
PY
fi

# trailer_cue_node publishes Gazebo ENU relative to the drone's local origin.
# Read the run's immutable map snapshot so planning and postflight export use
# exactly the same coordinate contract even if the source YAML later changes.
mapfile -t LANDING_CONFIG < <(
  python3 - "$LANDING_COORDINATES" "$REPO_DIR/simulation" <<'PY'
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

import yaml

document = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pose = document["spawn"]["gazebo_spawn_pose_enu"]
print(json.dumps([float(pose["x"]), float(pose["y"])]))
trailer = document["trailer"]
marker_world_z = (float(trailer["spawn_pose_enu"]["z"])
                  + float(trailer["marker_surface_height_m"]))
frames = document["frames"]
local_frame = frames.get("mavros_local") or frames.get("px4_local")
if local_frame is None:
    raise KeyError("frames requires mavros_local or px4_local")
local_origin_z = float(local_frame["origin_enu_m"][2])
print(f"{marker_world_z - local_origin_z:.9f}")
print(float(document.get("mission", {}).get("cruise_altitude_m", 6.0)))
print(trailer["odometry_topic"])
print(float(trailer["cruise_speed_m_s"]))
print(float(trailer["command_rate_hz"]))
world_path = pathlib.Path(
    sys.argv[2], document["map"]["world_file"]).resolve()
world = ET.parse(world_path).getroot().find("world")
spherical = world.find("spherical_coordinates")
print(float(spherical.findtext("latitude_deg")))
print(float(spherical.findtext("longitude_deg")))
print(float(spherical.findtext("elevation")))
print(float(spherical.findtext("heading_deg", "0")))
print(float(trailer["marker_surface_height_m"]))
gimbal = document["px4_vehicle"].get("gimbal_variant") or {}
print(gimbal.get("runtime_entity_name", ""))
PY
)
LANDING_SPAWN_ENU="${LANDING_CONFIG[0]}"
LANDING_DECK_Z="${LANDING_CONFIG[1]}"
LANDING_TAKEOFF_ALT="${LANDING_CONFIG[2]}"
LANDING_ODOMETRY_TOPIC="${LANDING_CONFIG[3]}"
LANDING_TRAILER_SPEED="${LANDING_CONFIG[4]}"
LANDING_CUE_RATE="${LANDING_CONFIG[5]}"
GPS_SIM_LAT="${LANDING_CONFIG[6]}"
GPS_SIM_LON="${LANDING_CONFIG[7]}"
GPS_SIM_ELEVATION="${LANDING_CONFIG[8]}"
GPS_SIM_HEADING="${LANDING_CONFIG[9]}"
GPS_SIM_ANTENNA_Z="${LANDING_CONFIG[10]}"
LANDING_GIMBAL_ENTITY="${LANDING_CONFIG[11]}"
GPS_DECK_Z="${TRAILER_DECK_Z_M:-$LANDING_DECK_Z}"
if [[ "$LANDING_MAP" == "city" ]]; then
  if [[ "$GIMBAL" == "1" && -z "$LANDING_GIMBAL_ENTITY" ]]; then
    echo "city YAML has no gimbal variant; using its body-fixed down camera"
    GIMBAL=0
  fi
fi

XRCE_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_xrce.log"
SIM_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_sim.log"
STACK_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_stack.log"
VIEW_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_aruco_view.log"
MISSION_VIEW_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_mission_view.log"
CUE_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_cue.log"
GPS_LOG="$PX4_MAP_RUNTIME_DIR/trailer_gps.log"
GPS_SIM_LOG="$PX4_MAP_RUNTIME_DIR/trailer_mavlink_emulator.log"
GPS_SIM_PTY_LINK="$PX4_MAP_RUNTIME_DIR/trailer_mavlink.pty"
MISSION_LOG="$PX4_MAP_RUNTIME_DIR/gimbal_mission.log"
ODOMETRY_LOG="$PX4_MAP_RUNTIME_DIR/trailer_odometry.jsonl"
ODOMETRY_ERROR_LOG="$PX4_MAP_RUNTIME_DIR/trailer_odometry.err"
RUN_MANIFEST="$PX4_MAP_RUNTIME_DIR/manifest.tsv"
CSV_EXPORT_LOG="$PX4_MAP_RUNTIME_DIR/flight_csv_export.log"
EXPERIMENT_LOGGER_LOG="$PX4_MAP_RUNTIME_DIR/experiment_logger.log"
echo "Run artifacts     : $PX4_MAP_RUNTIME_DIR"

PIDS=()
REQUIRED_PIDS=()
REQUIRED_NAMES=()
WATCHDOG_PID=""
TRAILER_GATE_DIR=""
TRAILER_START_FILE=""
SIM_PID=""
EXPERIMENT_LOGGER_PID=""
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
  printf 'trailer_cue_source\t%s\n' "$TRAILER_CUE_SOURCE"
  printf 'trailer_link_owned\t%s\n' "$TRAILER_LINK"
  printf 'gps_deck_z_m\t%s\n' "$GPS_DECK_Z"
  printf 'gps_sim_position_noise_m\t%s\n' "$GPS_SIM_POSITION_NOISE_M"
  printf 'gps_sim_velocity_noise_m_s\t%s\n' "$GPS_SIM_VELOCITY_NOISE_M_S"
  printf 'gps_sim_delay_s\t%s\n' "$GPS_SIM_DELAY_S"
  printf 'gps_sim_dropout\t%s\n' "$GPS_SIM_DROPOUT"
  printf 'gps_sim_seed\t%s\n' "$GPS_SIM_SEED"
  printf 'gps_sim_position_rate_hz\t%s\n' "$GPS_SIM_POSITION_RATE_HZ"
  printf 'gps_sim_status_rate_hz\t%s\n' "$GPS_SIM_STATUS_RATE_HZ"
  printf 'gps_input_timeout_s\t%s\n' "$GPS_INPUT_TIMEOUT_S"
  printf 'gps_cue_timeout_s\t%s\n' "$GPS_CUE_TIMEOUT_S"
  printf 'gps_max_distance_m\t%s\n' "$GPS_MAX_DISTANCE_M"
  printf 'takeoff_alt_m\t%s\n' "$LANDING_TAKEOFF_ALT"
  printf 'flight_control_owner\t%s\n' 'mission_manager_mpc_then_px4_precland'
  printf 'trailer_speed_m_s\t%s\n' "$TRAILER_SPEED_FOR_RUN"
  printf 'trailer_start_index\t%s\n' "$TRAILER_START_INDEX"
  printf 'simulation_speed_factor\t%s\n' "${PX4_SIM_SPEED_FACTOR:-1.0}"
  printf 'gz_partition\t%s\n' "$GZ_PARTITION"
  printf 'headless\t%s\n' "${HEADLESS:-0}"
  printf 'auto_sequence\t%s\n' "$AUTO_SEQUENCE"
  printf 'exit_on_done\t%s\n' "$EXIT_ON_DONE"
  printf 'path_only\t%s\n' "$PATH_ONLY"
  printf 'path_only_replans\t%s\n' "$PATH_ONLY_REPLANS"
  printf 'run_profile\t%s\n' "${LANDING_RUN_PROFILE:-manual}"
  printf 'experiment_logger_schema\t%s\n' 'jo_experiment_logger_v2'
} >"$RUN_MANIFEST"
git -C "$REPO_DIR" status --short >"$PX4_MAP_RUNTIME_DIR/git_status.txt" || true
cleanup() {
  local status=$?
  local run_result="failed"
  local sim_state=""
  local ulog_paths=()
  local ulog_source=""
  local csv_export="missing"
  local experiment_samples="missing"
  local experiment_summary="missing"
  local odometry_samples=0
  trap - EXIT INT TERM
  echo
  echo "stopping..."
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  # Let the observer consume MissionManager's final EXPERIMENT_METRICS line
  # before terminating it. It remains non-authoritative and is never allowed
  # to delay flight shutdown by more than two seconds.
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$EXPERIMENT_LOGGER_PID" \
        && "$pid" == "$EXPERIMENT_LOGGER_PID" ]] && continue
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  if [[ -n "$EXPERIMENT_LOGGER_PID" ]]; then
    for _ in {1..20}; do
      if grep -q 'EXPERIMENT_METRICS' "$MISSION_LOG" 2>/dev/null; then
        # /rosout delivery and the redirected mission log are independent.
        # Keep the observer alive for one callback window after the final
        # controller metrics become visible on disk.
        sleep 0.5
        break
      fi
      sleep 0.05
    done
    kill -TERM -- "-$EXPERIMENT_LOGGER_PID" 2>/dev/null || true
  fi
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
    printf 'experiment_metrics_csv\t%s\n' "$csv_export"
    if [[ "$csv_export" == "present" ]]; then
      printf 'flight_csv_schema\t%s\n' 'cju_flight_1hz_v3'
      printf 'flight_csv_rate_hz\t1\n'
      printf 'flight_csv_contract\t%s\n' \
        'one-second interval time-weighted means plus native-rate extrema'
    fi
    compgen -G "$PX4_MAP_RUNTIME_DIR/experiment_[0-9]*Z.csv" >/dev/null \
      && experiment_samples="present"
    compgen -G "$PX4_MAP_RUNTIME_DIR/experiment_[0-9]*_summary.csv" >/dev/null \
      && experiment_summary="present"
    printf 'experiment_logger_samples_csv\t%s\n' "$experiment_samples"
    printf 'experiment_logger_summary_csv\t%s\n' "$experiment_summary"
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

if [[ "$DIRECT_LANDING" == "1" \
    || "$LANDING_MAP" == "cju-track" || "$LANDING_MAP" == "city" ]]; then
  if [[ "$RUN_MISSION" == "1" ]]; then
    TRAILER_GATE_DIR="$(mktemp -d "/tmp/${LANDING_MAP}_trailer.XXXXXX")"
    TRAILER_START_FILE="$TRAILER_GATE_DIR/start"
    DRIVE_TRAILER_FOR_RUN=1
  fi
fi

# Never kill a process this invocation does not own.  A stale stack is reported
# and must be stopped explicitly by its owner.
if pgrep -f 'px4_sitl_default/bin/px4|native_gz_sensor_bridge|landing_mpc/lib/landing_mpc/|trailer_link/lib/trailer_link/trailer_target_node|trailer_mavlink_emulator.py|python3 .*simulation/gazebo/(trailer_waypoint_driver.py|tools/aruco_debug_viewer.py)' >/dev/null 2>&1; then
  echo "ERROR: another sim/perception stack is already running." >&2
  pgrep -af 'px4_sitl_default/bin/px4|native_gz_sensor_bridge|landing_mpc/lib/landing_mpc/|trailer_link/lib/trailer_link/trailer_target_node|trailer_mavlink_emulator.py|python3 .*simulation/gazebo/(trailer_waypoint_driver.py|tools/aruco_debug_viewer.py)' >&2 || true
  exit 4
fi
if [[ "$TRAILER_CUE_SOURCE" == "gps" && "$TRAILER_LINK" != "0" ]] \
    && pgrep -f 'trailer_link/lib/trailer_link/trailer_gps_node' >/dev/null 2>&1; then
  echo "ERROR: another trailer GPS radio reader is already running." >&2
  pgrep -af 'trailer_link/lib/trailer_link/trailer_gps_node' >&2 || true
  exit 4
fi

pgrep -f 'MicroXRCEAgent.*8888' >/dev/null 2>&1 || {
  echo "starting Micro XRCE-DDS agent"
  setsid MicroXRCEAgent udp4 -p 8888 >"$XRCE_LOG" 2>&1 &
  PIDS+=($!)
  sleep 1
}

echo "=== Gazebo + PX4 (GIMBAL=$GIMBAL) — log: $SIM_LOG ==="
RESET_TRAILER_FOR_RUN="$DIRECT_LANDING"
[[ "$LANDING_MAP" == "city" ]] && RESET_TRAILER_FOR_RUN=1
GIMBAL="$GIMBAL" FOLLOW_DRONE="${FOLLOW_DRONE:-0}" \
START_XRCE=1 START_MAVROS="${START_MAVROS:-1}" PX4_DAEMON=1 \
DRIVE_TRAILER="$DRIVE_TRAILER_FOR_RUN" TRAILER_SPEED_M_S="$TRAILER_SPEED_FOR_RUN" \
TRAILER_START_INDEX="${TRAILER_START_INDEX:-0}" \
RESET_TRAILER_FROM_YAML="$RESET_TRAILER_FOR_RUN" \
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
topic_publisher_count() {
  { ros2 topic info "$1" 2>/dev/null || true; } \
    | awk '/Publisher count:/ {print $3; found=1} END {if (!found) print 0}'
}
require_single_publisher() {
  local topic="$1" count=0
  for _ in {1..50}; do
    count="$(topic_publisher_count "$topic")"
    [[ "$count" == "1" ]] && return 0
    [[ "$count" -gt 1 ]] && break
    sleep 0.1
  done
  echo "ERROR: $topic needs exactly one publisher; found $count." >&2
  return 1
}

if [[ "$GIMBAL" == "1" ]]; then
  echo "=== gimbal + perception — log: $STACK_LOG ==="
  # Same-frame markers describe one rigid deck, so gross disagreement is an
  # invalid measurement.  Keep the safety gate tunable without disabling it.
  PAIR_GATE_ARG=(
    max_pair_disagreement_m:="${MAX_PAIR_DISAGREEMENT_M:-1.0}"
  )
  CITY_DETECTOR_ARGS=()
  if [[ "$LANDING_MAP" == "city" ]]; then
    # Begin slewing before a 20 px marker is resolvable and be fully pointed
    # by the 35 m GPS acquire boundary.  At 5 m/s closure this leaves four
    # simulated seconds for the normal 90 deg/s gimbal instead of demanding a
    # last-second snap.
    GIMBAL_AIM_START="${GIMBAL_AIM_START_RANGE_M:-60.0}"
    GIMBAL_AIM_FULL="${GIMBAL_AIM_FULL_RANGE_M:-35.0}"
    CITY_DETECTOR_ARGS=(
      min_marker_px:="${CITY_MIN_MARKER_PX:-20.0}"
      debug_dir:="$PX4_MAP_RUNTIME_DIR/aruco_debug"
      gimbal_attitude_source:=camera_imu
      entry_fix_window_s:="${CITY_ENTRY_FIX_WINDOW_S:-2.0}"
      prefer_cue_aim:=true
    )
  else
    GIMBAL_AIM_START="${GIMBAL_AIM_START_RANGE_M:-10.0}"
    GIMBAL_AIM_FULL="${GIMBAL_AIM_FULL_RANGE_M:-9.0}"
  fi
  setsid ros2 launch landing_mpc gimbal_perception.launch.py \
    world:="$LANDING_WORLD" model_name:="$LANDING_GIMBAL_ENTITY" \
    deck_z:="$LANDING_DECK_Z" \
    aim_start_range_m:="$GIMBAL_AIM_START" \
    aim_full_range_m:="$GIMBAL_AIM_FULL" "${PAIR_GATE_ARG[@]}" \
    "${CITY_DETECTOR_ARGS[@]}" \
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
  if [[ "$TRAILER_CUE_SOURCE" == "gazebo" ]]; then
    setsid ros2 run landing_mpc trailer_cue_node --ros-args \
      -p use_sim_time:=true -p world:="$LANDING_WORLD" \
      -p "spawn_enu:=$LANDING_SPAWN_ENU" -p deck_z:="$LANDING_DECK_Z" \
      -p rate_hz:="$LANDING_CUE_RATE" \
      >"$CUE_LOG" 2>&1 &
    CUE_PID=$!
    PIDS+=("$CUE_PID")
    REQUIRED_NAMES+=(trailer_cue)
    REQUIRED_PIDS+=("$CUE_PID")
  else
    TRAILER_PREFIX="$(ros2 pkg prefix trailer_link 2>/dev/null || true)"
    if [[ "$(readlink -f "$TRAILER_PREFIX" 2>/dev/null || true)" \
        != "$(readlink -f "$REPO_DIR/install/trailer_link")" ]]; then
      echo "ERROR: trailer_link is not built in this workspace." >&2
      exit 3
    fi
    if [[ "$TRAILER_LINK" == "sim" ]]; then
      setsid python3 -u "$SCRIPT_DIR/tools/trailer_mavlink_emulator.py" \
        --odometry-topic "$LANDING_ODOMETRY_TOPIC" \
        --pty-link "$GPS_SIM_PTY_LINK" \
        --latitude-deg "$GPS_SIM_LAT" --longitude-deg "$GPS_SIM_LON" \
        --elevation-m "$GPS_SIM_ELEVATION" \
        --heading-deg "$GPS_SIM_HEADING" \
        --antenna-z-m "$GPS_SIM_ANTENNA_Z" --sysid "$TRAILER_SYSID" \
        --position-noise-std-m "$GPS_SIM_POSITION_NOISE_M" \
        --velocity-noise-std-m-s "$GPS_SIM_VELOCITY_NOISE_M_S" \
        --delay-s "$GPS_SIM_DELAY_S" \
        --dropout-probability "$GPS_SIM_DROPOUT" --seed "$GPS_SIM_SEED" \
        --position-rate-hz "$GPS_SIM_POSITION_RATE_HZ" \
        --status-rate-hz "$GPS_SIM_STATUS_RATE_HZ" \
        >"$GPS_SIM_LOG" 2>&1 &
      GPS_SIM_PID=$!
      PIDS+=("$GPS_SIM_PID")
      REQUIRED_NAMES+=(trailer_mavlink_emulator)
      REQUIRED_PIDS+=("$GPS_SIM_PID")
      for _ in {1..50}; do
        [[ -c "$GPS_SIM_PTY_LINK" ]] && break
        kill -0 "$GPS_SIM_PID" 2>/dev/null || break
        sleep 0.1
      done
      if [[ ! -c "$GPS_SIM_PTY_LINK" ]]; then
        echo "ERROR: Gazebo MAVLink emulator did not create its PTY." >&2
        cat "$GPS_SIM_LOG" >&2 || true
        exit 7
      fi
      TRAILER_DEV="$GPS_SIM_PTY_LINK"
    fi
    if [[ "$TRAILER_LINK" != "0" ]]; then
      setsid ros2 run trailer_link trailer_gps_node --ros-args \
        -p use_sim_time:=true -p serial_device:="$TRAILER_DEV" \
        -p baud:="$TRAILER_BAUD" -p target_sysid:="$TRAILER_SYSID" \
        >"$GPS_LOG" 2>&1 &
      GPS_PID=$!
      PIDS+=("$GPS_PID")
      REQUIRED_NAMES+=(trailer_gps)
      REQUIRED_PIDS+=("$GPS_PID")
    fi
    setsid ros2 run trailer_link trailer_target_node --ros-args \
      -p use_sim_time:=true -p deck_z_m:="$GPS_DECK_Z" \
      -p stale_after_s:="$GPS_INPUT_TIMEOUT_S" \
      -p max_distance_m:="$GPS_MAX_DISTANCE_M" \
      >"$CUE_LOG" 2>&1 &
    CUE_PID=$!
    PIDS+=("$CUE_PID")
    REQUIRED_NAMES+=(trailer_gps_cue)
    REQUIRED_PIDS+=("$CUE_PID")
  fi
  sleep 2
  require_single_publisher /marker/cue || exit 7
  require_single_publisher /marker/cue_velocity || exit 7
  if [[ "$TRAILER_CUE_SOURCE" == "gps" ]]; then
    require_single_publisher /trailer/fix || exit 7
    require_single_publisher /trailer/velocity_enu || exit 7
    if ! timeout 15 ros2 topic echo /marker/cue --once >/dev/null 2>&1; then
      echo "ERROR: no coherent GPS cue; inspect $GPS_LOG and $CUE_LOG." >&2
      [[ "$TRAILER_LINK" == "sim" ]] && cat "$GPS_SIM_LOG" >&2 || true
      exit 7
    fi
  fi
  MISSION_ARGS=(-p auto_start:=true)
  printf -v LANDING_TARGET_SPEED_ARG '%.9f' "$TRAILER_SPEED_FOR_RUN"
  if [[ "$DIRECT_LANDING" == "1" ]]; then
    MISSION_ARGS+=(
      -p landing_target_min_speed_m_s:="$LANDING_TARGET_SPEED_ARG"
    )
    if [[ "$LANDING_MAP" != "city" ]]; then
      MISSION_ARGS+=( -p landing_gps_preacquire_range_m:=35.0 )
    fi
  fi
  if [[ ( "$LANDING_MAP" == "cju-track" || "$LANDING_MAP" == "city" ) \
      && "$DIRECT_LANDING" != "1" ]]; then
    MISSION_ARGS=(
      -p auto_start:=false
      -p takeoff_alt:="$LANDING_TAKEOFF_ALT"
      -p "mission_map_yaml:=$LANDING_COORDINATES"
    )
  elif [[ "$LANDING_MAP" == "city" ]]; then
    MISSION_ARGS+=(
      -p takeoff_alt:="$LANDING_TAKEOFF_ALT"
      -p "mission_map_yaml:=$LANDING_COORDINATES"
    )
  fi
  if [[ "$LANDING_MAP" == "city" ]]; then
    MISSION_ARGS+=(
      # A 12 m/s horizontal ceiling leaves 5 m/s closing speed on the
      # 7 m/s trailer; PX4 and the path controller use the same ceiling.
      -p path_mpc_speed_m_s:=12.0
      -p path_mpc_v_max_m_s:=12.0
      # Match the checked-in PX4 city dynamics (MPC_ACC_HOR=3,
      # MPC_JERK_AUTO=4) so the controller can track 12 m/s straights.
      -p path_mpc_a_max_m_s2:=3.0
      -p path_mpc_jerk_m_s3:=4.0
      # LandingMPC limits relative velocity.  7 m/s deck + 5 m/s closure
      # preserves the same 12 m/s absolute horizontal ceiling.
      -p landing_mpc_v_max_m_s:=5.0
      # RETURN uses the checked-in 3 m/s^2 profile toward the moving target.
      # The static outbound leg brakes earlier for its one-metre clearance.
      -p path_speed_profile_a_max_m_s2:=3.0
      -p mission_path_speed_profile_a_max_m_s2:=0.5
      -p path_terminal_goto_enabled:=true
      -p landing_gps_preacquire_range_m:=35.0
      # At 5x the measured vision throughput is about 3.3 Hz and arrives in
      # bursts.  Retain three-fix qualification for two seconds; visual/GPS
      # velocity, freshness, and alignment gates remain unchanged.
      -p entry_fix_window_s:="${CITY_ENTRY_FIX_WINDOW_S:-2.0}"
      # The 5x camera/KF and delayed simulated GPS retain a measured ~0.7 m
      # steady offset.  Learn at most 1 m while ACQUIRE is level, then latch
      # that calibration before descent; the final 0.3 m residual gate stays.
      -p bias_max_m:=1.0
      # Full mission RETURN may acquire the 7 m/s deck from the GPS cue before
      # ArUco is resolvable; TAKEOFF remains mission-first because auto_start
      # is false in this profile.
      -p landing_target_min_speed_m_s:="$LANDING_TARGET_SPEED_ARG"
    )
  fi
  if [[ "$TRAILER_CUE_SOURCE" == "gps" ]]; then
    MISSION_ARGS+=( -p cue_timeout_s:="$GPS_CUE_TIMEOUT_S" )
  fi
  if [[ "$PATH_ONLY" == "1" ]]; then
    # RETURN still exercises moving-target A* -> SFC -> B-spline -> MPC, but
    # this observer profile cannot enter LandingMPC or PX4 PRECLAND.
    MISSION_ARGS+=(
      -p landing_target_min_speed_m_s:=0.0
      -p landing_gps_preacquire_range_m:=0.0
      -r /marker/entry_valid:=/path_only/marker_entry_valid
    )
  fi
  setsid ros2 run landing_mpc mission_manager_node --ros-args -p use_sim_time:=true \
    "${MISSION_ARGS[@]}" \
    >"$MISSION_LOG" 2>&1 &
  MISSION_PID=$!
  PIDS+=("$MISSION_PID")
  REQUIRED_NAMES+=(mission_manager)
  REQUIRED_PIDS+=("$MISSION_PID")

  EXPERIMENT_LOGGER_PREFIX="$(
    ros2 pkg prefix experiment_logger 2>/dev/null || true)"
  if [[ "$(readlink -f "$EXPERIMENT_LOGGER_PREFIX" 2>/dev/null || true)" \
      == "$(readlink -f "$REPO_DIR/install/experiment_logger" \
          2>/dev/null || true)" ]]; then
    setsid ros2 run experiment_logger experiment_logger_node --ros-args \
      -p use_sim_time:=true \
      -p output_dir:="$PX4_MAP_RUNTIME_DIR" \
      -p map_yaml:="$LANDING_COORDINATES" \
      -p run_id:="$(basename "$PX4_MAP_RUNTIME_DIR")" \
      >"$EXPERIMENT_LOGGER_LOG" 2>&1 &
    EXPERIMENT_LOGGER_PID=$!
    PIDS+=("$EXPERIMENT_LOGGER_PID")
    echo "Experiment CSV    : $PX4_MAP_RUNTIME_DIR/experiment_<UTC>.csv"
    echo "Experiment summary: $PX4_MAP_RUNTIME_DIR/experiment_<UTC>_summary.csv"
  else
    echo "WARNING: experiment_logger is not built in this workspace; " \
         "run colcon build --packages-select experiment_logger" >&2
  fi

  if [[ "$DIRECT_LANDING" == "1" ]]; then
    setsid bash -c '
      until grep -q -- "TAKEOFF -> LANDING_ACQUIRE " "$1"; do sleep 0.1; done
      touch "$2"
    ' _ "$MISSION_LOG" "$TRAILER_START_FILE" &
    PIDS+=("$!")
  fi

  if [[ ( "$LANDING_MAP" == "cju-track" || "$LANDING_MAP" == "city" ) \
      && ( "${HEADLESS:-0}" != "1" || -n "${MISSION_VIEW+x}" ) \
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

if [[ "$RUN_MISSION" == "1" && "$DIRECT_LANDING" != "1" \
    && ( "$LANDING_MAP" == "cju-track" || "$LANDING_MAP" == "city" ) ]]; then
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
  wait_for_return_replans() {
    local accepted=0
    local target=$((PATH_ONLY_REPLANS + 1))
    local deadline=$((SECONDS + PATH_ONLY_TIMEOUT_S))
    while (( accepted < target )); do
      required_alive || return 1
      if grep -qE ' -> (ABORT|LANDING_ACQUIRE|LANDING_DESCEND|PRECLAND|DONE)' \
          "$MISSION_LOG" 2>/dev/null; then
        return 2
      fi
      (( SECONDS >= deadline )) && return 124
      accepted="$(grep -c \
        'global A\*/B-spline:.*target drift' "$MISSION_LOG" \
        2>/dev/null || true)"
      sleep 0.2
    done
    # Give the final atomic MarkerArray and planner log one callback cycle to
    # reach the read-only logger before teardown.
    sleep 0.5
    required_alive || return 1
    ! grep -qE ' -> (ABORT|LANDING_ACQUIRE|LANDING_DESCEND|PRECLAND|DONE)' \
      "$MISSION_LOG" 2>/dev/null
  }

  commands=(takeoff mission land)
  step=0
  echo '  명령 순서: takeoff → mission → land'
  while (( step < ${#commands[@]} )); do
    if [[ "$AUTO_SEQUENCE" == "1" ]]; then
      command="${commands[$step]}"
      echo "  자동 명령> $command"
    else
      IFS= read -r -p '  명령> ' command || break
    fi
    if [[ "$command" != "${commands[$step]}" ]]; then
      echo "  지금 입력할 명령: ${commands[$step]}" >&2
      continue
    fi
    case "$command" in
      takeoff)
        echo '  Phase 0: 상태/센서/PX4/경로계획기 PRECHECK 확인 중...'
        echo "  Phase 1: PX4 native takeoff — 고도 ${LANDING_TAKEOFF_ALT} m..."
        send_until_state takeoff 'TAKEOFF|READY' 10 || {
          retry_command 'TAKEOFF 상태 확인'
          continue
        }
        wait_for_states READY 120 || {
          retry_command 'READY 호버 확인'
          continue
        }
        # Complete the deck takeoff before releasing the trailer route.
        touch "$TRAILER_START_FILE"
        echo "  Phase 1 완료 — ${LANDING_TAKEOFF_ALT} m READY, mission 명령 대기"
        ;;
      mission)
        send_until_state mission 'MISSION_PLAN|MISSION|HOVER' 10 || {
          retry_command 'Phase 2 경로 계획 확인'
          continue
        }
        wait_for_states 'MISSION|HOVER' 120 || {
          retry_command 'geometry B-spline/TrackingMPC 상태 확인'
          continue
        }
        echo '  Phase 2: YAML A*→geometry B-spline을 TrackingMPC로 추종 중...'
        wait_for_states HOVER \
          "${MISSION_HOVER_TIMEOUT_S:-$DEFAULT_MISSION_HOVER_TIMEOUT_S}" || {
          retry_command 'Phase 2 HOVER 확인'
          continue
        }
        echo "  Phase 2 완료 — 목표 지점 호버, 트레일러 ${TRAILER_SPEED_FOR_RUN} m/s"
        ;;
      land)
        send_until_state land 'RETURN_PLAN|RETURN|LANDING_ACQUIRE|LANDING_DESCEND|PRECLAND|DONE' 30 || {
          retry_command 'Phase 3 경로 계획 확인'
          continue
        }
        if [[ "$PATH_ONLY" == "1" ]]; then
          echo "  Path-only: ${TRAILER_SPEED_FOR_RUN} m/s 트레일러 RETURN 동적 경로/SFC ${PATH_ONLY_REPLANS}회 재계획 관측 중..."
          wait_for_return_replans || {
            retry_command 'RETURN 동적 재계획/SFC 관측'
            exit 8
          }
          echo '  Path-only 완료 — 착륙 전 경로 생성·추종·SFC 데이터 저장'
          printf 'command_accepted\t%s\n' "$command" >>"$RUN_MANIFEST"
          exit 0
        fi
        echo '  Phase 3: TrackingMPC 접근 → LandingMPC 정렬·하강 → PX4 NAV_PRECLAND 수행 중...'
        wait_for_states DONE \
          "${LANDING_DONE_TIMEOUT_S:-$DEFAULT_LANDING_DONE_TIMEOUT_S}" || {
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
    | grep --line-buffered -E "gimbal:|detections|state| -> |ABORT|ERROR|WARN"
' tail "$STACK_LOG" "$MISSION_LOG" &
TAIL_PID=$!
PIDS+=("$TAIL_PID")
if [[ "$EXIT_ON_DONE" == "1" ]]; then
  echo "waiting for PRECLAND -> DONE (automatic cleanup enabled)"
  setsid timeout "${LANDING_DONE_TIMEOUT_S:-$DEFAULT_LANDING_DONE_TIMEOUT_S}" \
      bash -c '
        until grep -q -- "PRECLAND -> DONE" "$1" 2>/dev/null; do
          sleep 0.1
        done
      ' _ "$MISSION_LOG" &
  DONE_WAIT_PID=$!
  PIDS+=("$DONE_WAIT_PID")
  if wait "$DONE_WAIT_PID"; then
    echo "landing result    : DONE (PX4 landed and auto-disarmed)"
    exit 0
  fi
  echo "ERROR: direct landing did not reach DONE before timeout" >&2
  exit 8
fi
wait "$TAIL_PID" || true
