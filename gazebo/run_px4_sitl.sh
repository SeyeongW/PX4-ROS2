#!/usr/bin/env bash
#
# Start PX4 SITL and attach it to an ALREADY-RUNNING Gazebo world (started by
# run_sim_px4.sh in this same directory) as the x500_mono_cam_down vehicle,
# spawned above the moving_platform_aruco platform (platform top ~2.05m, so
# spawn a bit above that to avoid an initial collision).
#
# Usage:
#   ./gazebo/run_sim_px4.sh          # terminal 1 (Gazebo)
#   ./gazebo/run_px4_sitl.sh         # terminal 2 (this script)
#   ros2 launch mavros px4.launch fcu_url:="udp://:14540@"   # terminal 3
#
set -euo pipefail

PX4_HOME="${PX4_HOME:-$HOME/PX4-Autopilot}"
PX4_BIN="$PX4_HOME/build/px4_sitl_default/bin/px4"
ROOTFS="$PX4_HOME/build/px4_sitl_default/rootfs"

if [[ ! -x "$PX4_BIN" ]]; then
  echo "ERROR: $PX4_BIN not found/executable -- build PX4 first" >&2
  exit 1
fi

export GZ_IP="${GZ_IP:-127.0.0.1}"
export PX4_GZ_STANDALONE=1                       # attach to the running world instead of launching one
export PX4_SIM_MODEL="${PX4_SIM_MODEL:-gz_x500_mono_cam_down}"
export PX4_GZ_MODEL_POSE="${PX4_GZ_MODEL_POSE:-0,0,2.3}"   # platform top is ~2.05m
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
