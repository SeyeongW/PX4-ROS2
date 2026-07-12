#!/usr/bin/env bash
#
# Start PX4 SITL and attach it to an ALREADY-RUNNING Gazebo world (started by
# run_sim_px4.sh, or run_world.sh for the city/mountain maps, in this same
# directory) as the x500_mono_cam_down vehicle, spawned above the platform/
# pad (with a small margin to avoid an initial collision).
#
# Usage:
#   ./gazebo/run_sim_px4.sh          # terminal 1 (Gazebo, truck_mission_px4)
#   ./gazebo/run_px4_sitl.sh         # terminal 2 (this script)
#   ros2 launch mavros px4.launch fcu_url:="udp://:14540@"   # terminal 3
#
#   ./gazebo/run_world.sh mountain       # terminal 1 (Gazebo, map-only)
#   MAP=mountain ./gazebo/run_px4_sitl.sh   # terminal 2 (this script)
#
#   ./gazebo/run_world.sh city           # terminal 1
#   MAP=city ./gazebo/run_px4_sitl.sh       # terminal 2
#
# MAP=city|mountain only sets the PX4_GZ_MODEL_POSE default below (each map's
# drone_spawn_pad/drone_launch_pad top, +~0.2-0.3m clearance) -- it does not
# start Gazebo itself, run_world.sh does that. PX4_GZ_MODEL_POSE set directly
# always wins over MAP.
#
set -euo pipefail

PX4_HOME="${PX4_HOME:-$HOME/PX4-Autopilot}"
PX4_BIN="$PX4_HOME/build/px4_sitl_default/bin/px4"
ROOTFS="$PX4_HOME/build/px4_sitl_default/rootfs"

if [[ ! -x "$PX4_BIN" ]]; then
  echo "ERROR: $PX4_BIN not found/executable -- build PX4 first" >&2
  exit 1
fi

# Default spawn pose per map (pad top + clearance so the vehicle doesn't spawn
# embedded in the pad). truck_mission_px4 (platform top ~2.05m) keeps the
# original 0,0,2.3 default when MAP is unset.
case "${MAP:-}" in
  mountain) DEFAULT_POSE="-80,-80,0.4" ;;   # drone_launch_pad top z=0.16
  city)     DEFAULT_POSE="-120,115,-2.8" ;; # drone_spawn_pad top z=-3.019558902
  "")       DEFAULT_POSE="0,0,2.3" ;;
  *)        echo "ERROR: unknown MAP '$MAP' (expected city or mountain)." >&2; exit 2 ;;
esac

export GZ_IP="${GZ_IP:-127.0.0.1}"
export PX4_GZ_STANDALONE=1                       # attach to the running world instead of launching one
export PX4_SIM_MODEL="${PX4_SIM_MODEL:-gz_x500_mono_cam_down}"
export PX4_GZ_MODEL_POSE="${PX4_GZ_MODEL_POSE:-$DEFAULT_POSE}"
export HEADLESS=1                                # no separate gz GUI window from PX4's side

cd "$ROOTFS"
# -d: daemon mode, no interactive pxh shell. Without it, running in a
# background/non-interactive shell (no controlling tty) makes the pxh prompt
# re-print itself in a tight loop -- observed generating ~400MB/16s of log,
# a real risk given how little disk headroom this host usually has (see
# [[docker_sim_env]]/[[px4_migration]]). If you need the interactive console
# for debugging (`commander check` etc.), use the FIFO trick documented in
# [[px4_migration]] instead of dropping -d here.
exec "$PX4_BIN" -d
