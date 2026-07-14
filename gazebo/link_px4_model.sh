#!/usr/bin/env bash
# Link the repository-owned vehicle under a PX4-name that cannot collide with
# PX4's built-in x500_depth_lidar model. Existing real paths are never removed.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PX4_DIR="$(realpath -m "${1:-${PX4_DIR:-$HOME/PX4-Autopilot}}")"
MODEL_NAME="x500_city_rgbd_lidar"
SOURCE="$(realpath -m "$REPO_DIR/px4_models/$MODEL_NAME")"
MODEL_ROOT="$PX4_DIR/Tools/simulation/gz/models"
TARGET="$MODEL_ROOT/$MODEL_NAME"

die() {
  echo "ERROR: $*" >&2
  exit 3
}

[[ -d "$PX4_DIR/.git" ]] || die "$PX4_DIR is not a PX4 Git checkout."
[[ -d "$MODEL_ROOT" ]] || die "PX4 Gazebo model directory is missing: $MODEL_ROOT"
[[ -d "$SOURCE" && -f "$SOURCE/model.sdf" && -f "$SOURCE/model.config" ]] || \
  die "repository vehicle model is incomplete: $SOURCE"

if [[ -L "$TARGET" ]]; then
  current="$(realpath -m "$TARGET")"
  if [[ "$current" == "$SOURCE" ]]; then
    echo "PX4 model link ready: $TARGET -> $SOURCE"
    exit 0
  fi

  # Prepare the replacement next to the destination, then rename it into
  # place. rename(2) makes the visible symlink switch atomic. Only an existing
  # symlink is eligible for replacement; real files/directories fail below.
  temp_dir="$(mktemp -d -p "$MODEL_ROOT" ".${MODEL_NAME}.link.XXXXXX")"
  cleanup_temp() {
    [[ ! -L "$temp_dir/$MODEL_NAME" ]] || rm -- "$temp_dir/$MODEL_NAME"
    rmdir -- "$temp_dir" 2>/dev/null || true
  }
  trap cleanup_temp EXIT INT TERM
  ln -s -- "$SOURCE" "$temp_dir/$MODEL_NAME"
  mv -Tf -- "$temp_dir/$MODEL_NAME" "$TARGET"
  rmdir -- "$temp_dir"
  trap - EXIT INT TERM
  echo "Replaced stale PX4 model symlink atomically: $TARGET -> $SOURCE"
elif [[ -e "$TARGET" ]]; then
  cat >&2 <<EOF
ERROR: refusing to replace the existing non-symlink PX4 model path:
       $TARGET

The launcher will not delete or rename user/PX4 files. Move that path to a
backup yourself, then rerun this script. The required destination must be a
symlink to:
       $SOURCE
EOF
  exit 3
else
  # Creating a symlink is atomic and fails instead of clobbering a path that
  # may have appeared concurrently.
  ln -s -- "$SOURCE" "$TARGET" || \
    die "could not create $TARGET (a path may have appeared concurrently)."
  echo "Installed repository-linked PX4 model: $TARGET -> $SOURCE"
fi

[[ -L "$TARGET" ]] || die "installed model path is not a symlink: $TARGET"
[[ "$(realpath -m "$TARGET")" == "$SOURCE" ]] || \
  die "installed model link does not resolve to this checkout: $TARGET"
[[ -f "$TARGET/model.sdf" && -f "$TARGET/model.config" ]] || \
  die "installed model link is incomplete: $TARGET"
