#!/usr/bin/env bash
# Link every repository-owned Gazebo model under a PX4-name that cannot
# collide with PX4's built-in models. Existing real paths are never removed.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PX4_DIR="$(realpath -m "${1:-${PX4_DIR:-$HOME/PX4-Autopilot}}")"
MODEL_ROOT="$PX4_DIR/Tools/simulation/gz/models"

die() {
  echo "ERROR: $*" >&2
  exit 3
}

[[ -d "$PX4_DIR/.git" ]] || die "$PX4_DIR is not a PX4 Git checkout."
[[ -d "$MODEL_ROOT" ]] || die "PX4 Gazebo model directory is missing: $MODEL_ROOT"

link_model() {
  local model_name="$1"
  local source target current temp_dir
  source="$(realpath -m "$REPO_DIR/simulation/px4_models/$model_name")"
  target="$MODEL_ROOT/$model_name"

  [[ -d "$source" && -f "$source/model.sdf" && -f "$source/model.config" ]] || \
    die "repository model is incomplete: $source"

  if [[ -L "$target" ]]; then
    current="$(realpath -m "$target")"
    if [[ "$current" == "$source" ]]; then
      echo "PX4 model link ready: $target -> $source"
      return 0
    fi

    # Prepare the replacement next to the destination, then rename it into
    # place. rename(2) makes the visible symlink switch atomic. Only an
    # existing symlink is eligible for replacement; real files/directories
    # fail below.
    temp_dir="$(mktemp -d -p "$MODEL_ROOT" ".${model_name}.link.XXXXXX")"
    cleanup_temp() {
      [[ ! -L "$temp_dir/$model_name" ]] || rm -- "$temp_dir/$model_name"
      rmdir -- "$temp_dir" 2>/dev/null || true
    }
    trap cleanup_temp EXIT INT TERM
    ln -s -- "$source" "$temp_dir/$model_name"
    mv -Tf -- "$temp_dir/$model_name" "$target"
    rmdir -- "$temp_dir"
    trap - EXIT INT TERM
    echo "Replaced stale PX4 model symlink atomically: $target -> $source"
  elif [[ -e "$target" ]]; then
    cat >&2 <<EOF
ERROR: refusing to replace the existing non-symlink PX4 model path:
       $target

The launcher will not delete or rename user/PX4 files. Move that path to a
backup yourself, then rerun this script. The required destination must be a
symlink to:
       $source
EOF
    exit 3
  else
    # Creating a symlink is atomic and fails instead of clobbering a path that
    # may have appeared concurrently.
    ln -s -- "$source" "$target" || \
      die "could not create $target (a path may have appeared concurrently)."
    echo "Installed repository-linked PX4 model: $target -> $source"
  fi

  [[ -L "$target" ]] || die "installed model path is not a symlink: $target"
  [[ "$(realpath -m "$target")" == "$source" ]] || \
    die "installed model link does not resolve to this checkout: $target"
  [[ -f "$target/model.sdf" && -f "$target/model.config" ]] || \
    die "installed model link is incomplete: $target"
}

# Install every model this checkout owns, so a vehicle that merge-includes
# another repository model (x500_gimbal_rgbd_lidar -> x500_city_rgbd_lidar +
# gimbal_down) resolves without a second install step.
shopt -s nullglob
installed=0
for model_dir in "$REPO_DIR"/simulation/px4_models/*/; do
  [[ -f "$model_dir/model.sdf" ]] || continue
  link_model "$(basename "${model_dir%/}")"
  installed=$((installed + 1))
done
[[ "$installed" -gt 0 ]] || die "no models found under $REPO_DIR/simulation/px4_models"
