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
# In separate terminals run ArduPilot SITL and the ROS 2 camera bridge:
#   sim_vehicle.py -v ArduCopter -f JSON --console --map
#   ros2 launch ./launch/camera_bridge.launch.py
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

# gz-sim needs to find: our model+world, the stock ardupilot_gazebo models,
# and the compiled ArduPilotPlugin / sensor systems.
export GZ_SIM_RESOURCE_PATH="$SCRIPT_DIR/models:$SCRIPT_DIR/worlds:$AP_GAZEBO/models:$AP_GAZEBO/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$AP_GAZEBO/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"

echo "GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"
echo "GZ_SIM_SYSTEM_PLUGIN_PATH=$GZ_SIM_SYSTEM_PLUGIN_PATH"
echo "Launching world: iris_down_camera_runway.sdf"

exec gz sim -v4 -r "$SCRIPT_DIR/worlds/iris_down_camera_runway.sdf"
