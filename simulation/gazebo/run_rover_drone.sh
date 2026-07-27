#!/usr/bin/env bash
#
# Launch the Phase 1 combined world (skid_rover + iris drone on the 400 m track).
#
# This only starts GAZEBO (+ optional camera bridge). The two ArduPilot SITL
# instances run in their OWN terminals so each has its own MAVProxy console:
#
#   Terminal 1 (drone, instance 0 -> FDM 9002 / MAVLink 14550):
#     sim_vehicle.py -v ArduCopter -f JSON -I0 --console --map
#
#   Terminal 2 (rover, instance 1 -> FDM 9012 / MAVLink 14560):
#     sim_vehicle.py -v Rover --model JSON -I1 -w --console --map \
#       --add-param-file="$PWD/gazebo/config/rover_skid.parm"
#
#   Terminal 3 (this script):
#     ./gazebo/run_rover_drone.sh
#
# Order does not matter: the plugins and SITL retry until they connect.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AP_GAZEBO="${AP_GAZEBO:-$HOME/ardupilot_gazebo}"

if [[ ! -d "$AP_GAZEBO/build" ]]; then
  echo "ERROR: ardupilot_gazebo plugin not built at $AP_GAZEBO/build." >&2
  echo "       Set AP_GAZEBO=/path/to/ardupilot_gazebo (with a built ./build)." >&2
  exit 1
fi

# gz-transport on loopback (this host has extra interfaces; pin to localhost).
export GZ_IP="${GZ_IP:-127.0.0.1}"

# Find our models+worlds, the stock ardupilot_gazebo models, and the plugins.
export GZ_SIM_RESOURCE_PATH="$SCRIPT_DIR/models:$SCRIPT_DIR/worlds:$AP_GAZEBO/models:$AP_GAZEBO/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$AP_GAZEBO/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"

# Qt onto XWayland under a Wayland session (gz GUI shows blank white otherwise).
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

SERVER_ARGS=()
if [[ "${HEADLESS:-0}" == "1" ]]; then
  SERVER_ARGS=(-s)
  echo "HEADLESS=1 -> server only (no GUI)"
fi

cat <<EOF
GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH
GZ_SIM_SYSTEM_PLUGIN_PATH=$GZ_SIM_SYSTEM_PLUGIN_PATH

Start the two SITL instances in separate terminals:
  Drone:  sim_vehicle.py -v ArduCopter -f JSON -I0 --console --map
  Rover:  sim_vehicle.py -v Rover --model JSON -I1 -w --console --map \\
            --add-param-file="$REPO_DIR/simulation/gazebo/config/rover_skid.parm"

Launching world: rover_drone_runway.sdf
EOF

exec gz sim -v4 -r "${SERVER_ARGS[@]}" "$SCRIPT_DIR/worlds/rover_drone_runway.sdf"
