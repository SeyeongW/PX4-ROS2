#!/usr/bin/env bash
# One-time PX4 SITL setup. Existing tracked firmware sources are untouched;
# only the repository-owned, collision-free Gazebo model symlink is managed.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PX4_DIR="$(realpath -m "${PX4_DIR:-$HOME/PX4-Autopilot}")"
PX4_VERSION="${PX4_VERSION:-v1.17.0}"
PX4_INSTALL_PREREQS="${PX4_INSTALL_PREREQS:-1}"

if [[ -e "$PX4_DIR" ]]; then
  if [[ ! -d "$PX4_DIR/.git" ]]; then
    echo "ERROR: $PX4_DIR exists but is not a PX4 Git checkout." >&2
    exit 2
  fi
  echo "Existing PX4 checkout detected; branch/tag and tracked firmware files will not be changed."
else
  echo "Cloning PX4 Autopilot $PX4_VERSION into $PX4_DIR"
  git clone --recursive --branch "$PX4_VERSION" \
    https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
fi

CUSTOM_MODEL_NAME="x500_city_rgbd_lidar"
CUSTOM_MODEL_SOURCE="$REPO_DIR/px4_models/$CUSTOM_MODEL_NAME"
CUSTOM_MODEL_TARGET="$PX4_DIR/Tools/simulation/gz/models/$CUSTOM_MODEL_NAME"
"$SCRIPT_DIR/link_px4_model.sh" "$PX4_DIR"

if [[ ! -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ]]; then
  if [[ "$PX4_INSTALL_PREREQS" == "1" ]]; then
    PREREQ_SCRIPT="$PX4_DIR/Tools/setup/ubuntu.sh"
    if [[ ! -x "$PREREQ_SCRIPT" ]]; then
      echo "ERROR: PX4 prerequisite helper is missing: $PREREQ_SCRIPT" >&2
      exit 3
    fi
    echo "Installing PX4's official Ubuntu prerequisites (sudo may prompt once)"
    "$PREREQ_SCRIPT" --no-nuttx --no-sim-tools
  else
    echo "Skipping PX4 prerequisites because PX4_INSTALL_PREREQS=0."
  fi
  echo "Building PX4 SITL (this does not launch Gazebo)"
  make -C "$PX4_DIR" px4_sitl_default
else
  echo "PX4 SITL is already built; no rebuild needed."
fi

test -x "$PX4_DIR/build/px4_sitl_default/bin/px4"
test -L "$CUSTOM_MODEL_TARGET"
test "$(realpath -m "$CUSTOM_MODEL_TARGET")" = "$(realpath -m "$CUSTOM_MODEL_SOURCE")"
test -f "$CUSTOM_MODEL_TARGET/model.sdf"
test -f "$PX4_DIR/Tools/simulation/gz/models/x500/model.sdf"
test -f "$PX4_DIR/Tools/simulation/gz/models/OakD-Lite/model.sdf"
test -f "$PX4_DIR/Tools/simulation/gz/models/LW20/model.sdf"
echo "PX4 SITL ready: $PX4_DIR/build/px4_sitl_default/bin/px4"
