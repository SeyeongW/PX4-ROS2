#!/usr/bin/env bash
# One-time PX4 SITL setup for a fresh clone. Existing firmware checkouts are untouched.
set -Eeuo pipefail

PX4_DIR="$(realpath -m "${PX4_DIR:-$HOME/PX4-Autopilot}")"
PX4_VERSION="${PX4_VERSION:-v1.17.0}"

if [[ -e "$PX4_DIR" ]]; then
  if [[ ! -d "$PX4_DIR/.git" ]]; then
    echo "ERROR: $PX4_DIR exists but is not a PX4 Git checkout." >&2
    exit 2
  fi
  echo "Existing PX4 checkout detected; branch/tag and firmware files will not be changed."
else
  echo "Cloning PX4 Autopilot $PX4_VERSION into $PX4_DIR"
  git clone --recursive --branch "$PX4_VERSION" \
    https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
fi

if [[ ! -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ]]; then
  echo "Building PX4 SITL (this does not launch Gazebo)"
  make -C "$PX4_DIR" px4_sitl_default
else
  echo "PX4 SITL is already built; no rebuild needed."
fi

test -x "$PX4_DIR/build/px4_sitl_default/bin/px4"
test -f "$PX4_DIR/Tools/simulation/gz/models/x500_mono_cam_down/model.sdf"
echo "PX4 SITL ready: $PX4_DIR/build/px4_sitl_default/bin/px4"
