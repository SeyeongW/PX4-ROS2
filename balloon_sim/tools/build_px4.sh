#!/usr/bin/env bash
# Source-only reproducible dependency recipe. Prerequisites are documented in README.
# Existing checkouts must match the pin and be clean; this script never resets them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
revision=d6f12ad1c4f70ad3230afd7d86e971421e02fef4
classic_revision=6697ab169ceab512dc706acea63df4c882662c60
checkout="${1:-$PWD/external/PX4-Autopilot}"
jobs="${BALLOON_BUILD_JOBS:-2}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'BALLOON_BUILD_JOBS must be a positive integer' >&2; exit 2; }
if [[ ! -e "$checkout" ]]; then
  mkdir -p "$(dirname "$checkout")"
  git clone --branch v1.17.0 --single-branch https://github.com/PX4/PX4-Autopilot.git "$checkout"
fi
[[ -d "$checkout/.git" || -f "$checkout/.git" ]] || { echo "Not a Git checkout: $checkout" >&2; exit 1; }
[[ "$(git -C "$checkout" rev-parse HEAD)" == "$revision" ]] || { echo "PX4 must be at $revision; no automatic reset performed." >&2; exit 1; }
[[ -z "$(git -C "$checkout" status --porcelain --untracked-files=no)" ]] || { echo 'PX4 has working changes; use a separate clean checkout.' >&2; exit 1; }
git -C "$checkout" submodule update --init --recursive
make -C "$checkout" -j"$jobs" px4_sitl_default
classic="$checkout/Tools/simulation/gazebo-classic/sitl_gazebo-classic"
[[ "$(git -C "$classic" rev-parse HEAD)" == "$classic_revision" ]] || { echo 'Unexpected Gazebo Classic submodule revision' >&2; exit 1; }
# Pinned classic CMake uses Qt5 unconditionally; ON supplies its Qt5 setup.
# Only the explicit physics/sensor targets below are built.
cmake -S "$classic" -B "$checkout/build/balloon_gazebo_classic" \
  -DCMAKE_BUILD_TYPE=Release \
  -D_MAVLINK_INCLUDE_DIR="$checkout/build/px4_sitl_default/mavlink" \
  -DBUILD_GSTREAMER_PLUGIN=ON -DBUILD_ROS2_PLUGINS=OFF -DENABLE_UNIT_TESTS=OFF
cmake --build "$checkout/build/balloon_gazebo_classic" --parallel "$jobs" --target \
  gazebo_motor_model gazebo_multirotor_base_plugin gazebo_mavlink_interface \
  gazebo_imu_plugin gazebo_magnetometer_plugin gazebo_barometer_plugin \
  gazebo_gps_plugin gazebo_groundtruth_plugin
printf 'Built pinned upstream PX4 and Gazebo plugins in %s\n' "$checkout"
printf '%s\n' 'This clean upstream build still requires its own SITL flight validation.'
