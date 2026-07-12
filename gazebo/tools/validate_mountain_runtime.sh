#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/gazebo/validation/runtime"
LAUNCH_LOG="$RUNTIME_DIR/ugv_drone_harmonic.log"
STATS_LOG="$RUNTIME_DIR/ugv_drone_harmonic_stats.log"
GPU_LOG="$RUNTIME_DIR/ugv_drone_harmonic_gpu.csv"
PMON_LOG="$RUNTIME_DIR/ugv_drone_harmonic_pmon.log"
PROCESS_LOG="$RUNTIME_DIR/ugv_drone_harmonic_processes.log"
ERROR_LOG="$RUNTIME_DIR/ugv_drone_harmonic_error_scan.log"
SCREENSHOT="$RUNTIME_DIR/ugv_drone_harmonic_gazebo.png"

mkdir -p "$RUNTIME_DIR"

SERVER_PID="$(pgrep -f -o 'gz sim server' || true)"
GUI_PID="$(pgrep -f -o 'gz sim gui' || true)"
if [[ -z "$SERVER_PID" || -z "$GUI_PID" ]]; then
  echo "The Harmonic server and GUI must already be running." >&2
  exit 1
fi

WINDOW_ID="$(
  xwininfo -root -tree |
    awk '/"Gazebo Sim":/ && /\("gz-sim-gui" "Gazebo GUI"\)/ && !id {id=$1} END {print id}'
)"
if [[ -z "$WINDOW_ID" ]]; then
  WINDOW_ID="$(
    xwininfo -root -tree |
      awk '/"Gazebo( Sim)?":/ && !id {id=$1} END {print id}'
  )"
fi
if [[ -z "$WINDOW_ID" ]]; then
  echo "The Harmonic Gazebo window was not found." >&2
  exit 2
fi

TARGET_WINDOW="$WINDOW_ID" perl -MX11::Protocol -e '
  my $x = X11::Protocol->new();
  my $window = hex($ENV{TARGET_WINDOW});
  my $atom = $x->atom("_NET_ACTIVE_WINDOW");
  my $event = $x->pack_event(
    name => "ClientMessage", window => $window, type => $atom, format => 32,
    data => pack("L5", 2, 0, 0, 0, 0)
  );
  my $mask = $x->pack_event_mask("SubstructureRedirect", "SubstructureNotify");
  $x->SendEvent($x->{root}, 0, $mask, $event);
  $x->flush();
'

sleep 3
import -window "$WINDOW_ID" "$SCREENSHOT"

STATS_TOPIC="$(
  gz topic -l |
    awk '/^\/world\/(ugv_drone_mountain_map|ugv_drone_mountain)\/stats$/ && !topic {topic=$0} END {print topic}'
)"
if [[ -z "$STATS_TOPIC" ]]; then
  echo "No mountain-world stats topic was found in the active Gazebo partition." >&2
  exit 3
fi

set +e
timeout 14s gz topic -e -t "$STATS_TOPIC" >"$STATS_LOG" 2>&1 &
STATS_PID=$!
set -e

printf 'timestamp,gpu_util_percent,memory_used_mib,power_w\n' >"$GPU_LOG"
for _ in {1..10}; do
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits |
    head -n 1 >>"$GPU_LOG"
  sleep 1
done

set +e
wait "$STATS_PID"
STATS_STATUS=$?
set -e
if [[ "$STATS_STATUS" -ne 0 && "$STATS_STATUS" -ne 124 ]]; then
  echo "gz topic stats exited with status $STATS_STATUS" >&2
fi
if ! grep -q 'real_time_factor:' "$STATS_LOG"; then
  echo "Gazebo stats capture contains no real-time-factor sample: $STATS_TOPIC" >&2
  exit 4
fi

nvidia-smi pmon -c 1 >"$PMON_LOG" 2>&1
ps -eo pid,ppid,pgid,sid,stat,%cpu,%mem,rss,etime,args |
  awk '/[g]z sim|[r]uby.*gz/ {print}' >"$PROCESS_LOG"

if grep -Ein \
  '\[Err\]|unable to find|failed to load|exception|segmentation fault|core dumped|multiple sub-trees|floating body' \
  "$LAUNCH_LOG" >"$ERROR_LOG"; then
  ERROR_COUNT="$(wc -l <"$ERROR_LOG")"
else
  : >"$ERROR_LOG"
  ERROR_COUNT=0
fi

awk -F',' '
  NR > 1 {
    for (i=2; i<=4; i++) gsub(/^ +| +$/, "", $i)
    gpu=$2+0; memory=$3+0; power=$4+0
    n++; gpu_sum+=gpu; memory_sum+=memory; power_sum+=power
    if (n==1 || gpu<gpu_min) gpu_min=gpu
    if (n==1 || gpu>gpu_max) gpu_max=gpu
    if (n==1 || memory<memory_min) memory_min=memory
    if (n==1 || memory>memory_max) memory_max=memory
  }
  END {
    printf("gpu_samples=%d gpu_min=%.3f gpu_mean=%.3f gpu_max=%.3f vram_min=%.3f vram_mean=%.3f vram_max=%.3f power_mean=%.3f\n",
      n, gpu_min, n?gpu_sum/n:0, gpu_max, memory_min, n?memory_sum/n:0,
      memory_max, n?power_sum/n:0)
  }
' "$GPU_LOG"

printf 'server_pid=%s gui_pid=%s window_id=%s\n' "$SERVER_PID" "$GUI_PID" "$WINDOW_ID"
printf 'stats_topic=%s\n' "$STATS_TOPIC"
printf 'load_error_count=%s\n' "$ERROR_COUNT"
printf 'screenshot=%s\n' "$SCREENSHOT"
printf 'stats_log=%s\n' "$STATS_LOG"
printf 'gpu_log=%s\n' "$GPU_LOG"
printf 'pmon_log=%s\n' "$PMON_LOG"
