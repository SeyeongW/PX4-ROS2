#!/usr/bin/env bash
# Build the PX4 v1.17 ROS 2 message package used by the autonomy controller.
set -Eeuo pipefail

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PX4_MSGS_WS="$(realpath -m "${PX4_MSGS_WS:-$HOME/px4_ros2_ws}")"
PX4_MSGS_VERSION="${PX4_MSGS_VERSION:-v1.17.0}"
SOURCE_DIR="$PX4_MSGS_WS/src/px4_msgs"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ERROR: ROS 2 setup is missing: $ROS_SETUP" >&2
  exit 2
fi
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

for command in git colcon python3; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is missing: $command" >&2
    exit 2
  }
done

if [[ -e "$SOURCE_DIR" ]]; then
  if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    echo "ERROR: $SOURCE_DIR exists but is not a px4_msgs Git checkout." >&2
    exit 3
  fi
  echo "Existing px4_msgs checkout detected; branch and files are unchanged."
else
  mkdir -p "$PX4_MSGS_WS/src"
  git clone --depth 1 --branch "$PX4_MSGS_VERSION" \
    https://github.com/PX4/px4_msgs.git "$SOURCE_DIR"
fi

if [[ ! -f "$PX4_MSGS_WS/install/px4_msgs/share/px4_msgs/package.bash" ]]; then
  colcon --log-base "$PX4_MSGS_WS/log" build \
    --base-paths "$PX4_MSGS_WS/src" \
    --build-base "$PX4_MSGS_WS/build" \
    --install-base "$PX4_MSGS_WS/install" \
    --packages-select px4_msgs
else
  echo "px4_msgs is already built; no rebuild needed."
fi

set +u
# shellcheck disable=SC1090
source "$PX4_MSGS_WS/install/setup.bash"
set -u
python3 -c 'from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition'
if ! python3 -c 'from pymavlink import mavutil' >/dev/null 2>&1; then
  echo "Installing optional pymavlink transport for the native PX4 L1-style A/B experiment."
  python3 -m pip install --user 'pymavlink>=2.4.40'
fi
python3 -c 'from pymavlink import mavutil; assert mavutil.mavlink.MAV_AUTOPILOT_PX4 == 12'
python3 -c 'import gz.msgs10, gz.transport13, numpy, scipy, yaml'
echo "Autonomy dependencies ready: $PX4_MSGS_WS/install/setup.bash"
