#!/usr/bin/env bash
# One command to fly the gimbal vehicle. Everything the stack needs is set here
# rather than left to the caller: the XRCE agent (which lives in /usr/local/bin,
# not where run_px4_map.sh looks), GZ_PARTITION, and the gimbal camera's wider
# tan(vfov/2) that mission_manager would otherwise default to the down camera's.
#
#   ./gazebo/run_gimbal.sh              gimbal + perception, moving trailer
#   ./gazebo/run_gimbal.sh mission      ...plus the full landing mission
#   ./gazebo/run_gimbal.sh baseline     body-fixed camera instead, to compare
#
# Ctrl-C stops everything it started.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-}"

GIMBAL=1
case "$MODE" in
  ""|gimbal) RUN_MISSION=0 ;;
  mission)   RUN_MISSION=1 ;;
  baseline)  RUN_MISSION=1; GIMBAL=0 ;;
  -h|--help|help)
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  *) echo "unknown mode '$MODE' (try: gimbal | mission | baseline)" >&2; exit 2 ;;
esac

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
export GZ_PARTITION="${GZ_PARTITION:-px4_ros2_${USER:-user}}"

PIDS=()
# Kill the WHOLE stack, not just our direct children.  Two components detach
# themselves and used to survive Ctrl-C, then break the NEXT run: the sensor
# bridge (run_px4_map.sh starts it with setsid) and the perception nodes
# (ros2 launch/run spawn them in their own process groups).  A leftover bridge
# publishing a stale /clock — or none — is exactly the "won't take off" freeze.
# So sweep by pattern, TERM first, then KILL what refuses.  MicroXRCEAgent is
# deliberately left alone: it is cheap and shared, and killing it slows restart.
STACK_PATTERNS=(
  'run_px4_map.sh precision-landing-moving'
  'px4_sitl_default/bin/px4'
  'gz sim'
  'native_gz_sensor_bridge'
  'static_transform_publisher'
  'gimbal_perception.launch'
  'landing_mpc/lib/landing_mpc/'
  'ros2 run landing_mpc'
)
_sweep() {
  local sig="$1"
  for pat in "${STACK_PATTERNS[@]}"; do pkill "$sig" -f "$pat" 2>/dev/null || true; done
}
cleanup() {
  trap - EXIT INT TERM
  echo
  echo "stopping..."
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  _sweep -TERM
  sleep 2
  _sweep -KILL      # anything that ignored TERM (setsid-detached children)
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# The launcher refuses to start beside a stale instance, and leftover pieces
# from a previous Ctrl-C are the usual reason this script appears to hang or
# never takes off.  Clear the FULL stack, not just PX4, before starting.
if pgrep -f 'px4_sitl_default/bin/px4|native_gz_sensor_bridge|landing_mpc/lib/landing_mpc/' >/dev/null 2>&1; then
  echo "Leftover sim/perception processes from a previous run; clearing them first."
  _sweep -TERM
  sleep 2
  _sweep -KILL
  sleep 2
fi

pgrep -f 'MicroXRCEAgent.*8888' >/dev/null 2>&1 || {
  echo "starting Micro XRCE-DDS agent"
  MicroXRCEAgent udp4 -p 8888 >/tmp/gimbal_xrce.log 2>&1 &
  PIDS+=($!)
  sleep 1
}

echo "=== Gazebo + PX4 (GIMBAL=$GIMBAL) — log: /tmp/gimbal_sim.log ==="
GIMBAL="$GIMBAL" FOLLOW_DRONE=1 START_XRCE=1 START_MAVROS=0 PX4_DAEMON=1 \
DRIVE_TRAILER=1 TRAILER_SPEED_M_S="${TRAILER_SPEED_M_S:-3.0}" \
  "$SCRIPT_DIR/run_px4_map.sh" precision-landing-moving >/tmp/gimbal_sim.log 2>&1 &
PIDS+=($!)

echo -n "waiting for the vehicle to spawn"
for _ in $(seq 120); do
  if gz topic -l 2>/dev/null | grep -q 'gimbal_camera/image\|down_camera/image'; then
    break
  fi
  if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
    echo; echo "ERROR: the simulator exited. Last lines:" >&2
    tail -15 /tmp/gimbal_sim.log >&2
    exit 1
  fi
  echo -n "."; sleep 2
done
echo " up."

# ROS's setup scripts read unset variables, so -u has to stand down for them.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1091
source "$REPO_DIR/install/setup.bash"
set -u

if [[ "$GIMBAL" == "1" ]]; then
  echo "=== gimbal + perception — log: /tmp/gimbal_stack.log ==="
  ros2 launch landing_mpc gimbal_perception.launch.py >/tmp/gimbal_stack.log 2>&1 &
  PIDS+=($!)
  # The gimbal camera is wider than the body-fixed one this defaults to.
  VFOV_ARG=(-p tan_vfov_half:=0.7722)
else
  echo "=== baseline perception (body-fixed camera) ==="
  ros2 run landing_mpc aruco_detector_node --ros-args -p use_sim_time:=true \
    >/tmp/gimbal_stack.log 2>&1 & PIDS+=($!)
  ros2 run landing_mpc marker_tf_node --ros-args -p use_sim_time:=true \
    >>/tmp/gimbal_stack.log 2>&1 & PIDS+=($!)
  ros2 run landing_mpc marker_kf_node --ros-args -p use_sim_time:=true \
    >>/tmp/gimbal_stack.log 2>&1 & PIDS+=($!)
  VFOV_ARG=()
fi
sleep 5

if [[ "$RUN_MISSION" == "1" ]]; then
  echo "=== mission — log: /tmp/gimbal_mission.log ==="
  ros2 run landing_mpc trailer_cue_node --ros-args -p use_sim_time:=true \
    >/tmp/gimbal_cue.log 2>&1 &
  PIDS+=($!)
  sleep 2
  ros2 run landing_mpc mission_manager_node --ros-args -p use_sim_time:=true \
    "${VFOV_ARG[@]}" -p auto_start:=true >/tmp/gimbal_mission.log 2>&1 &
  PIDS+=($!)
fi

cat <<EOF

  ready.  Ctrl-C here stops everything.

  watch the camera :  ros2 run rqt_image_view rqt_image_view /gimbal_camera/image
  watch the DETECTOR: ros2 run rqt_image_view rqt_image_view /aruco/debug_image
                      (pick the .../compressed entry in the dropdown — it is a
                       JPEG topic, ~20 kB/frame instead of 1.4 MB raw)
                      (marker outline, centre-to-centre dx/dy in px and m, and
                       the FILL gauge that shows the near-field blind zone)
  watch the gimbal :  rviz2      (Fixed Frame: map, then Add -> TF)
  score a run      :  python3 scratchpad/gimbal_chase_monitor.py

EOF

# Surface the lines worth reading instead of making anyone tail three files.
tail -f /tmp/gimbal_stack.log /tmp/gimbal_mission.log 2>/dev/null \
  | grep --line-buffered -E 'gimbal:|detections|state|ABORT|ERROR|WARN' || true
