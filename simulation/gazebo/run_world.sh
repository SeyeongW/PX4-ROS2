#!/usr/bin/env bash
# Run a checked-in map using only this repository's assets.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$MAP" in
  city)
    WORLD="$SCRIPT_DIR/worlds/applepark_city_uav/applepark_uav.world"
    DESCRIPTION="Apple Park UAV city (XY-spaced buildings, stationary trailer, no PX4 drone)"
    DEFAULT_PHYSICS_ENGINE="gz-physics-dartsim-plugin"
    ;;
  mountain)
    WORLD="$SCRIPT_DIR/worlds/ugv_drone_mountain_map.world"
    DESCRIPTION="UGV/drone 300 x 300 m mountain (trailer included, no PX4 drone)"
    DEFAULT_PHYSICS_ENGINE="gz-physics-bullet-featherstone-plugin"
    ;;
  mpc-landing-moving)
    WORLD="$SCRIPT_DIR/worlds/mpc_landing_200m_moving.world"
    DESCRIPTION="1,000 x 100 m forward/reverse shuttle world (stationary preview, no PX4 drone)"
    DEFAULT_PHYSICS_ENGINE="gz-physics-dartsim-plugin"
    ;;
  cju-track)
    WORLD="$SCRIPT_DIR/worlds/drone_cju.world"
    DESCRIPTION="real-scale Cheongju University main stadium only (stationary preview, no PX4 drone)"
    DEFAULT_PHYSICS_ENGINE="gz-physics-dartsim-plugin"
    ;;
  -h|--help|help|"")
    cat <<EOF
Usage: $(basename "$0") <city|mountain|mpc-landing-moving|cju-track> [gz sim options]

  city                Current UAV city + stationary trailer (no PX4 drone)
  mountain            300 x 300 m mountain + stationary trailer (no PX4 drone)
  mpc-landing-moving  1,000 x 100 m field + 1 km forward/reverse shuttle
  cju-track           Real-scale Cheongju University main stadium only

Environment:
  PAUSED=1       Open without starting physics
  HEADLESS=1     Run server-only
  USE_NVIDIA=0   Disable NVIDIA PRIME variables
  GZ_BIN=gz      Override the Gazebo CLI

For a dynamically spawned, controllable PX4 x500 use:
  ./gazebo/run_px4_map.sh city
  ./gazebo/run_px4_map.sh mountain
  ./gazebo/run_px4_map.sh mpc-landing-moving
  ./gazebo/run_px4_map.sh cju-track
EOF
    exit 0
    ;;
  *)
    echo "ERROR: unknown map '$MAP' (expected city, mountain, mpc-landing-moving, or cju-track)." >&2
    exit 2
    ;;
esac

GZ_BIN="${GZ_BIN:-gz}"
PHYSICS_ENGINE="${PHYSICS_ENGINE:-$DEFAULT_PHYSICS_ENGINE}"

if ! command -v "$GZ_BIN" >/dev/null 2>&1 || ! "$GZ_BIN" sim --help >/dev/null 2>&1; then
  echo "ERROR: Gazebo Harmonic의 'gz sim'을 찾지 못했습니다." >&2
  echo "       sudo bash $SCRIPT_DIR/install_apt_deps.sh" >&2
  exit 2
fi
if [[ ! -f "$WORLD" ]]; then
  echo "ERROR: world file is missing: $WORLD" >&2
  exit 2
fi

# Every profile is deliberately closed over these repository paths.
RESOURCE_PATHS=("$SCRIPT_DIR/models" "$SCRIPT_DIR/worlds")
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
  export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-xcb_glx}"
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
if command -v nvidia-smi >/dev/null 2>&1; then
  timeout 4s nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
fi

exec "$GZ_BIN" sim "${ARGS[@]}"
