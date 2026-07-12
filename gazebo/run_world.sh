#!/usr/bin/env bash
# Run the checked-in city or mountain map using only this repository's assets.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$MAP" in
  city)
    WORLD="$SCRIPT_DIR/worlds/city_map/city_map.world"
    DESCRIPTION="500 x 500 m city"
    ;;
  mountain)
    WORLD="$SCRIPT_DIR/worlds/mountain_map.world"
    DESCRIPTION="300 x 300 m mountain (map-only)"
    ;;
  mountain-sitl)
    WORLD="$SCRIPT_DIR/worlds/ugv_drone.world"
    DESCRIPTION="300 x 300 m mountain with ArduPilot SITL model"
    ;;
  -h|--help|help|"")
    cat <<EOF
Usage: $(basename "$0") <city|mountain|mountain-sitl> [gz sim options]

  city          500 x 500 m city, fully self-contained
  mountain      300 x 300 m mountain, fully self-contained map preview
  mountain-sitl Mountain plus Iris/ArduPilot integration (external plugin needed)

Environment:
  PAUSED=1       Open without starting physics
  HEADLESS=1     Run server-only
  USE_NVIDIA=0   Disable NVIDIA PRIME variables
  GZ_BIN=gz      Override the Gazebo CLI
  AP_GAZEBO=...  Override ardupilot_gazebo for mountain-sitl
EOF
    exit 0
    ;;
  *)
    echo "ERROR: unknown map '$MAP' (expected city, mountain, or mountain-sitl)." >&2
    exit 2
    ;;
esac

GZ_BIN="${GZ_BIN:-gz}"
PHYSICS_ENGINE="${PHYSICS_ENGINE:-gz-physics-bullet-featherstone-plugin}"
AP_GAZEBO="${AP_GAZEBO:-$HOME/ardupilot_gazebo}"

if ! command -v "$GZ_BIN" >/dev/null 2>&1 || ! "$GZ_BIN" sim --help >/dev/null 2>&1; then
  echo "ERROR: Gazebo Harmonic의 'gz sim'을 찾지 못했습니다." >&2
  echo "       sudo bash $SCRIPT_DIR/install_apt_deps.sh" >&2
  exit 2
fi
if [[ ! -f "$WORLD" ]]; then
  echo "ERROR: world file is missing: $WORLD" >&2
  exit 2
fi

# city and mountain are deliberately closed over these repository paths.
RESOURCE_PATHS=("$SCRIPT_DIR/models" "$SCRIPT_DIR/worlds")
if [[ "$MAP" == "mountain-sitl" ]]; then
  if [[ ! -f "$AP_GAZEBO/models/iris_with_standoffs/model.sdf" ]]; then
    echo "ERROR: $AP_GAZEBO/models/iris_with_standoffs/model.sdf is missing." >&2
    echo "       For repository-only map editing use: $0 mountain" >&2
    exit 3
  fi
  if [[ ! -f "$AP_GAZEBO/build/libArduPilotPlugin.so" ]]; then
    echo "ERROR: $AP_GAZEBO/build/libArduPilotPlugin.so is missing." >&2
    echo "       Build ardupilot_gazebo, or use: $0 mountain" >&2
    exit 3
  fi
  RESOURCE_PATHS+=("$AP_GAZEBO/models" "$AP_GAZEBO/worlds")
  export GZ_SIM_SYSTEM_PLUGIN_PATH="$AP_GAZEBO/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
fi
# city/mountain (map-only) declare libOpticalFlowSystem.so/libGstCameraSystem.so
# for the PX4 x500_mono_cam_down vehicle (see run_px4_sitl.sh MAP=city|mountain)
# -- those .so files, and x500_mono_cam_down's own model:// sub-includes, live
# in PX4-Autopilot's own build/Tools output, not this repo, so point
# GZ_SIM_SYSTEM_PLUGIN_PATH/GZ_SIM_RESOURCE_PATH at them when present. Without
# the resource path piece specifically, the model still "spawns" (entity +
# correct sensor topics all show up) but every sensor topic stays silent
# forever -- found by actually attaching PX4 and comparing gz topic -e output
# against the known-good truck_mission_px4 world (2026-07-12), not from docs.
# Harmless no-op otherwise (plain ArduPilot/no-PX4 preview never needs this).
PX4_HOME="${PX4_HOME:-$HOME/PX4-Autopilot}"
PX4_GZ_PLUGINS="$PX4_HOME/build/px4_sitl_default/src/modules/simulation/gz_plugins"
PX4_GZ_MODELS="$PX4_HOME/Tools/simulation/gz/models"
PX4_GZ_WORLDS="$PX4_HOME/Tools/simulation/gz/worlds"
if [[ "$MAP" != "mountain-sitl" && -d "$PX4_GZ_PLUGINS" ]]; then
  export GZ_SIM_SYSTEM_PLUGIN_PATH="$PX4_GZ_PLUGINS${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
  RESOURCE_PATHS+=("$PX4_GZ_MODELS" "$PX4_GZ_WORLDS")
fi
if [[ -n "${GZ_SIM_RESOURCE_PATH:-}" ]]; then
  RESOURCE_PATHS+=("$GZ_SIM_RESOURCE_PATH")
fi
export GZ_SIM_RESOURCE_PATH="$(IFS=:; echo "${RESOURCE_PATHS[*]}")"
export GZ_IP="${GZ_IP:-127.0.0.1}"

if [[ "${USE_NVIDIA:-1}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __VK_LAYER_NV_optimus=NVIDIA_only
  export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
fi
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

ARGS=(-v4 --physics-engine "$PHYSICS_ENGINE")
if [[ "${PAUSED:-0}" != "1" ]]; then
  ARGS+=(-r)
fi
if [[ "${HEADLESS:-0}" == "1" ]]; then
  ARGS+=(-s)
fi
ARGS+=("$@" "$WORLD")

echo "Map: $DESCRIPTION"
echo "World: $WORLD"
echo "Physics: $PHYSICS_ENGINE"
echo "Repository resources: $SCRIPT_DIR/models:$SCRIPT_DIR/worlds"
if [[ "$MAP" == "mountain-sitl" ]]; then
  echo "ArduPilot plugin: $AP_GAZEBO/build/libArduPilotPlugin.so"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  timeout 4s nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
fi

exec "$GZ_BIN" sim "${ARGS[@]}"
