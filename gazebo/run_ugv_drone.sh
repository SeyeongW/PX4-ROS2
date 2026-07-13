#!/usr/bin/env bash
# Launch the 300 x 300 m mountain world with the RTX NVIDIA GPU selected.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AP_GAZEBO="${AP_GAZEBO:-$HOME/ardupilot_gazebo}"
WORLD="${WORLD:-$SCRIPT_DIR/worlds/ugv_drone.world}"
GZ_BIN="${GZ_BIN:-gz}"
PHYSICS_ENGINE="${PHYSICS_ENGINE:-gz-physics-bullet-featherstone-plugin}"

# A fresh clone can always open the complete mountain terrain without
# downloading a vehicle model.  When the external SITL model/plugin is not
# installed, fall back to the repository-only map entry point.  Set
# REQUIRE_SITL=1 to turn a missing ArduPilot installation into a hard error.
if [[ ! -f "$AP_GAZEBO/models/iris_with_standoffs/model.sdf" || \
      ! -f "$AP_GAZEBO/build/libArduPilotPlugin.so" ]]; then
  if [[ "${REQUIRE_SITL:-0}" == "1" ]]; then
    echo "ERROR: ardupilot_gazebo model/plugin is missing under $AP_GAZEBO." >&2
    exit 3
  fi
  echo "INFO: ardupilot_gazebo is incomplete; opening the self-contained map preview."
  echo "      For SITL, install the plugin and rerun this command."
  exec "$SCRIPT_DIR/run_world.sh" mountain "$@"
fi

if ! command -v "$GZ_BIN" >/dev/null 2>&1; then
  echo "ERROR: Gazebo Harmonic의 'gz' 명령을 찾지 못했습니다." >&2
  echo "       먼저: sudo bash $SCRIPT_DIR/install_apt_deps.sh" >&2
  exit 2
fi

# Gazebo Classic 11 also installs a command named `gz`, but it has no `sim`
# subcommand.  Detect that case before producing a confusing launch failure.
if ! "$GZ_BIN" sim --help >/dev/null 2>&1; then
  echo "ERROR: 현재 '$GZ_BIN'은 Gazebo Classic CLI이며 'gz sim'을 지원하지 않습니다." >&2
  echo "       Harmonic을 병행 설치하려면:" >&2
  echo "         sudo bash $SCRIPT_DIR/install_apt_deps.sh" >&2
  exit 2
fi

if [[ ! -f "$WORLD" ]]; then
  echo "ERROR: world 파일이 없습니다: $WORLD" >&2
  exit 2
fi

# Keep all local assets reproducible and avoid a Fuel download at launch time.
RESOURCE_PATHS=("$SCRIPT_DIR/models" "$SCRIPT_DIR/worlds")
if [[ -d "$AP_GAZEBO/models" ]]; then
  RESOURCE_PATHS+=("$AP_GAZEBO/models")
fi
if [[ -d "$AP_GAZEBO/worlds" ]]; then
  RESOURCE_PATHS+=("$AP_GAZEBO/worlds")
fi
RESOURCE_PATHS+=("${GZ_SIM_RESOURCE_PATH:-}")
export GZ_SIM_RESOURCE_PATH="$(IFS=:; echo "${RESOURCE_PATHS[*]}")"

if [[ -d "$AP_GAZEBO/build" ]]; then
  export GZ_SIM_SYSTEM_PLUGIN_PATH="$AP_GAZEBO/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
else
  echo "WARNING: $AP_GAZEBO/build가 없어 ArduPilotPlugin을 로드할 수 없습니다." >&2
  echo "         지형 편집은 가능하지만 드론 SITL 연동에는 플러그인 빌드가 필요합니다." >&2
fi

# PRIME render offload: select the discrete NVIDIA GPU (RTX 5060 on this host).
# Set USE_NVIDIA=0 only for software/iGPU troubleshooting.
if [[ "${USE_NVIDIA:-1}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __VK_LAYER_NV_optimus=NVIDIA_only
  export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
fi

# Qt / Ogre2 is more reliable through XWayland on this machine's Wayland session.
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

export GZ_IP="${GZ_IP:-127.0.0.1}"

SERVER_ARGS=()
if [[ "${HEADLESS:-0}" == "1" ]]; then
  SERVER_ARGS=(-s)
fi

# DART in Harmonic 8 does not construct this world's OBJ collision shape.
# Bullet Featherstone loads the terrain collision mesh correctly.
RUN_ARGS=(-v4 --physics-engine "$PHYSICS_ENGINE")
if [[ "${PAUSED:-0}" != "1" ]]; then
  RUN_ARGS+=(-r)
fi

echo "World: $WORLD"
echo "Physics: $PHYSICS_ENGINE"
echo "Resources: $GZ_SIM_RESOURCE_PATH"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
fi
echo
echo "ArduPilot SITL (별도 터미널):"
echo "  cd $HOME/ardupilot"
echo "  sim_vehicle.py -v ArduCopter -f JSON -I0 --console --map"
echo
echo "GPU 사용률 확인 (별도 터미널):"
echo "  watch -n 1 'nvidia-smi pmon -s um -c 1; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader'"

exec "$GZ_BIN" sim "${RUN_ARGS[@]}" "${SERVER_ARGS[@]}" "$WORLD"
