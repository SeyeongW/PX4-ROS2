#!/usr/bin/env bash
# Process-scoped flight-control authority lock for PX4-ROS22 launchers.
#
# Source this library from a launcher, then call acquire_control_authority.
# The library deliberately installs no signal/EXIT traps.  The caller keeps the
# returned file descriptor open, and the kernel releases the flock when the
# launcher (or the process it execs) exits.  This makes it compatible with an
# existing launcher cleanup trap.

if [[ -z "${_PX4_ROS22_INVOKED_COMMAND+x}" ]]; then
  printf -v _PX4_ROS22_INVOKED_COMMAND '%q ' "$0" "$@"
  _PX4_ROS22_INVOKED_COMMAND="${_PX4_ROS22_INVOKED_COMMAND% }"
fi

# Print the lock path used by one vehicle authority domain.
resolve_control_authority_lock_file() {
  if [[ $# -ne 1 || -z "${1:-}" ]]; then
    echo "ERROR: resolve_control_authority_lock_file requires a vehicle authority ID." >&2
    return 64
  fi

  if [[ -n "${PX4_ROS22_CONTROL_LOCK_FILE:-}" ]]; then
    printf '%s\n' "$PX4_ROS22_CONTROL_LOCK_FILE"
  else
    printf '/tmp/px4_ros22_%s.control.lock\n' "$1"
  fi
}

# Acquire and retain a non-blocking exclusive flight-control authority lock.
#
# Usage:
#   acquire_control_authority <stack_name> <vehicle_authority_id>
#
# A conflicting owner is reported immediately with exit status 73.  On
# success, PX4_ROS22_CONTROL_LOCK_FD remains open for the process lifetime.
acquire_control_authority() {
  if [[ $# -ne 2 || -z "${1:-}" || -z "${2:-}" ]]; then
    echo "ERROR: acquire_control_authority requires <stack_name> <vehicle_authority_id>." >&2
    return 64
  fi
  if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to acquire flight-control authority." >&2
    return 69
  fi
  if [[ -n "${PX4_ROS22_CONTROL_LOCK_FD:-}" ]]; then
    echo "ERROR: this process already holds a control-authority lock." >&2
    return 70
  fi

  local stack_name="$1"
  local vehicle_authority_id="$2"
  local lock_file
  local lock_fd
  local lock_user
  local lock_hostname
  local start_time_utc
  local invoked_command

  lock_file="$(resolve_control_authority_lock_file "$vehicle_authority_id")" || return $?

  # Do not truncate here: a losing contender must be able to print the current
  # owner's metadata without changing it.
  if ! exec {lock_fd}<>"$lock_file"; then
    echo "ERROR: cannot open control-authority lock file: $lock_file" >&2
    return 74
  fi
  if ! flock -n -x "$lock_fd"; then
    echo "ERROR: flight-control authority is already held (lock: $lock_file)." >&2
    if [[ -s "$lock_file" ]]; then
      echo "Current control-authority owner:" >&2
      sed 's/^/  /' "$lock_file" >&2
    else
      echo "Current control-authority owner metadata is unavailable." >&2
    fi
    exec {lock_fd}>&-
    return 73
  fi

  lock_user="$(id -un 2>/dev/null || printf '%s' "${USER:-unknown}")"
  lock_hostname="$(hostname 2>/dev/null || printf 'unknown')"
  start_time_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  invoked_command="${PX4_ROS22_INVOKED_COMMAND:-$_PX4_ROS22_INVOKED_COMMAND}"

  # Only the successful owner replaces the human-readable metadata.
  if ! : >"$lock_file" || ! {
    printf 'stack_name=%s\n' "$stack_name"
    printf 'vehicle_authority_id=%s\n' "$vehicle_authority_id"
    printf 'pid=%s\n' "$$"
    printf 'user=%s\n' "$lock_user"
    printf 'hostname=%s\n' "$lock_hostname"
    printf 'start_time_utc=%s\n' "$start_time_utc"
    printf 'invoked_command=%s\n' "$invoked_command"
  } >&"$lock_fd"; then
    echo "ERROR: cannot write control-authority metadata: $lock_file" >&2
    exec {lock_fd}>&-
    return 74
  fi

  PX4_ROS22_CONTROL_LOCK_FD="$lock_fd"
  PX4_ROS22_CONTROL_LOCK_FILE_RESOLVED="$lock_file"
  PX4_ROS22_CONTROL_STACK_NAME="$stack_name"
  PX4_ROS22_VEHICLE_AUTHORITY_ID="$vehicle_authority_id"
  return 0
}
