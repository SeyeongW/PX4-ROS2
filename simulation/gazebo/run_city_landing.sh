#!/usr/bin/env bash
# Run one of the two checked-in city mission-to-moving-deck profiles.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_NAME="${1:-}"
PATH_ONLY_PROFILE=0
case "$PROFILE_NAME" in
  visual|fast)
    PROFILE_PATH="$SCRIPT_DIR/profiles/city_landing_${PROFILE_NAME}.yaml"
    ;;
  path)
    # Reuse the proven 5x/headless/live-YAML profile; only the automatic
    # terminal condition changes from touchdown to accepted RETURN replans.
    PROFILE_PATH="$SCRIPT_DIR/profiles/city_landing_fast.yaml"
    PATH_ONLY_PROFILE=1
    ;;
  -h|--help|help|"")
    cat <<EOF
Usage: $(basename "$0") <visual|fast|path>

  visual  1x automatic full mission with Gazebo + ArUco + live map
  fast    YAML-owned 5x automatic full mission, route-map-only view
  path    YAML-only 5x dynamic RETURN planning/tracking/SFC, no landing
EOF
    exit 0
    ;;
  *)
    echo "unknown profile '$PROFILE_NAME' (expected visual, fast, or path)" >&2
    exit 2
    ;;
esac

mapfile -t PROFILE_CONFIG < <(
  python3 - "$PROFILE_PATH" <<'PY'
import math
import pathlib
import sys

import yaml

path = pathlib.Path(sys.argv[1]).resolve()
document = yaml.safe_load(path.read_text(encoding="utf-8"))
if document.get("schema_version") != 1:
    raise SystemExit("landing profile schema_version must be 1")
profile = document.get("profile")
if not isinstance(profile, dict):
    raise SystemExit("landing profile requires a profile mapping")

def boolean(name):
    value = profile.get(name)
    if not isinstance(value, bool):
        raise SystemExit(f"profile.{name} must be boolean")
    return "1" if value else "0"

def positive(name):
    value = profile.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"profile.{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"profile.{name} must be finite and positive")
    # Keep a decimal point so ROS 2 CLI overrides infer DOUBLE, not INTEGER.
    return repr(value)

coordinates = (path.parent / str(profile.get("coordinates", ""))).resolve()
if not coordinates.is_file():
    raise SystemExit(f"profile coordinates do not exist: {coordinates}")
factor = profile.get("simulation_speed_factor")
if isinstance(factor, bool) or not isinstance(factor, (int, float)):
    raise SystemExit("profile.simulation_speed_factor must be numeric")
factor = float(factor)
if not math.isfinite(factor) or factor <= 0.0:
    raise SystemExit("profile.simulation_speed_factor must be finite and positive")

print(profile.get("name", path.stem))
print(coordinates)
print(boolean("direct_landing"))
print(boolean("auto_sequence"))
print(f"{factor:.9g}")
print(boolean("headless"))
print(boolean("aruco_view"))
print(boolean("mission_view"))
print(boolean("exit_on_done"))
print(positive("gps_position_rate_hz"))
print(positive("gps_status_rate_hz"))
print(positive("gps_input_timeout_s"))
print(positive("gps_cue_timeout_s"))
PY
)
if (( ${#PROFILE_CONFIG[@]} != 13 )); then
  echo "invalid landing profile: $PROFILE_PATH" >&2
  exit 2
fi

echo "City mission profile: ${PROFILE_CONFIG[0]}"
echo "Coordinate YAML   : ${PROFILE_CONFIG[1]}"
echo "Simulation speed  : ${PROFILE_CONFIG[4]}x"
PROFILE_ENV=(
  LANDING_RUN_PROFILE="${PROFILE_CONFIG[0]}"
  PATH_ONLY="$PATH_ONLY_PROFILE"
  EXIT_ON_DONE="${PROFILE_CONFIG[8]}"
)
if [[ "$PATH_ONLY_PROFILE" == "1" ]]; then
  PROFILE_ENV=(
    LANDING_RUN_PROFILE=city_path_tracking_fast
    PATH_ONLY=1
    EXIT_ON_DONE=0
    TRAILER_SPEED_M_S="${TRAILER_SPEED_M_S:-10.0}"
  )
  echo "Trailer speed     : ${TRAILER_SPEED_M_S:-10.0} m/s"
fi
exec env \
  "${PROFILE_ENV[@]}" \
  LANDING_MAP=city \
  PX4_MAP_COORDINATES="${PROFILE_CONFIG[1]}" \
  DIRECT_LANDING="${PROFILE_CONFIG[2]}" \
  AUTO_SEQUENCE="${PROFILE_CONFIG[3]}" \
  PX4_SIM_SPEED_FACTOR="${PROFILE_CONFIG[4]}" \
  HEADLESS="${PROFILE_CONFIG[5]}" \
  ARUCO_VIEW="${PROFILE_CONFIG[6]}" \
  MISSION_VIEW="${PROFILE_CONFIG[7]}" \
  GPS_SIM_POSITION_RATE_HZ="${PROFILE_CONFIG[9]}" \
  GPS_SIM_STATUS_RATE_HZ="${PROFILE_CONFIG[10]}" \
  GPS_INPUT_TIMEOUT_S="${PROFILE_CONFIG[11]}" \
  GPS_CUE_TIMEOUT_S="${PROFILE_CONFIG[12]}" \
  "$SCRIPT_DIR/run_gimbal.sh" mission
