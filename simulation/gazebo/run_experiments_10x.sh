#!/usr/bin/env bash
# Sequential paper runs; one PX4/Gazebo instance owns the shared ports at once.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
RUNS="${2:-10}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TIMEOUT_S="${BATCH_RUN_TIMEOUT_S:-1800}"
KILL_AFTER_S="${BATCH_KILL_AFTER_S:-120}"
FAIL_FAST="${BATCH_FAIL_FAST:-0}"

case "$MODE" in
  cju|city|city-path|city-yaml) ;;
  -h|--help|help|"")
    cat <<EOF
Usage: $(basename "$0") <cju|city|city-path|city-yaml> [runs=10]

  cju        CJU full mission/landing, headless, 1x
  city       city full mission/landing, headless, 5x
  city-path  Gazebo city dynamic planning/tracking/SFC, 5x, no landing
  city-yaml  offline city-YAML A*/B-spline/MPC rollout, no Gazebo/PX4

All requested runs are attempted and stored under ~/.local/state by default.
Set BATCH_FAIL_FAST=1 to stop on the first failed or incomplete run.
EOF
    exit 0
    ;;
  *)
    echo "unknown mode '$MODE' (expected cju, city, city-path, or city-yaml)" >&2
    exit 2
    ;;
esac
if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ \
    && "$RUN_TIMEOUT_S" =~ ^[1-9][0-9]*$ \
    && "$KILL_AFTER_S" =~ ^[1-9][0-9]*$ \
    && "$DRY_RUN" =~ ^[01]$ && "$FAIL_FAST" =~ ^[01]$ ]]; then
  echo "runs/timeouts must be positive integers; DRY_RUN/BATCH_FAIL_FAST must be 0 or 1" >&2
  exit 2
fi

REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
STATE_ROOT="${XDG_STATE_HOME:-$HOME/.local/state}/px4-ros2-jo/batches"
BATCH_ROOT="${BATCH_ROOT:-$STATE_ROOT/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"

# This is the deterministic, figure-producing YAML rollout shown in the paper
# images.  It deliberately bypasses ROS, PX4, Gazebo and experiment_logger.
if [[ "$MODE" == "city-yaml" ]]; then
  command=(python3
    "$REPO_DIR/flight/path_plan/tools/run_city_yaml_offline_batch.py"
    --runs "$RUNS" --output "$BATCH_ROOT")
  [[ "$DRY_RUN" == "1" ]] && command+=(--dry-run)
  [[ "$FAIL_FAST" == "1" ]] && command+=(--fail-fast)
  exec "${command[@]}"
fi

if [[ "$DRY_RUN" == "0" ]]; then
  command -v timeout >/dev/null || {
    echo "GNU timeout is required" >&2
    exit 3
  }
  if [[ ! -x "$REPO_DIR/install/experiment_logger/lib/experiment_logger/experiment_logger_node" ]]; then
    echo "experiment_logger is not built; run: colcon build --packages-select experiment_logger --symlink-install" >&2
    exit 3
  fi
fi

if [[ "$DRY_RUN" == "0" ]]; then
  if [[ -e "$BATCH_ROOT" ]]; then
    echo "batch output already exists: $BATCH_ROOT" >&2
    exit 2
  fi
  mkdir -p "$BATCH_ROOT"
  printf 'run_index,mode,exit_status,result,sample_csv,summary_csv\n' \
    >"$BATCH_ROOT/batch_runs.csv"
fi

failures=0
for ((run = 1; run <= RUNS; run++)); do
  printf -v run_name 'run_%02d' "$run"
  run_dir="$BATCH_ROOT/$run_name"
  partition_mode="${MODE//-/_}"
  common=(PX4_MAP_RUNTIME_DIR="$run_dir"
    GZ_PARTITION="jo_batch_${partition_mode}_$$_${run_name}")
  clean_env=(env -u TRAILER_SPEED_M_S -u TRAILER_START_INDEX \
    -u PX4_MAP_COORDINATES -u TRAILER_DECK_Z_M -u TRAILER_DEV \
    -u PATH_ONLY_REPLANS -u PATH_ONLY_TIMEOUT_S)
  case "$MODE" in
    cju)
      expected=done
      command=("${clean_env[@]}" "${common[@]}" LANDING_MAP=cju-track
        LANDING_RUN_PROFILE=cju_batch
        DIRECT_LANDING=0 PATH_ONLY=0 AUTO_SEQUENCE=1 EXIT_ON_DONE=1 HEADLESS=1
        TRAILER_CUE_SOURCE=gazebo TRAILER_LINK=1 ALLOW_EXTERNAL_GPS_SITL=0
        ARUCO_VIEW=0 MISSION_VIEW=0
        PX4_SIM_SPEED_FACTOR="${CJU_BATCH_SPEED_FACTOR:-1.0}"
        "$SCRIPT_DIR/run_gimbal.sh" mission)
      ;;
    city)
      expected=done
      command=("${clean_env[@]}" "${common[@]}" LANDING_MAP=city
        LANDING_RUN_PROFILE=city_batch
        DIRECT_LANDING=0 PATH_ONLY=0 AUTO_SEQUENCE=1 EXIT_ON_DONE=1 HEADLESS=1
        TRAILER_CUE_SOURCE=gps TRAILER_LINK=sim ALLOW_EXTERNAL_GPS_SITL=0
        ARUCO_VIEW=0 MISSION_VIEW=0
        PX4_SIM_SPEED_FACTOR="${CITY_BATCH_SPEED_FACTOR:-5.0}"
        GPS_SIM_POSITION_RATE_HZ=10.0 GPS_SIM_STATUS_RATE_HZ=2.0
        GPS_INPUT_TIMEOUT_S=2.0 GPS_CUE_TIMEOUT_S=2.0
        "$SCRIPT_DIR/run_gimbal.sh" mission)
      ;;
    city-path)
      expected=completed
      command=("${clean_env[@]}" "${common[@]}" TRAILER_SPEED_M_S=10.0
        PATH_ONLY_REPLANS="${PATH_ONLY_REPLANS:-3}"
        TRAILER_CUE_SOURCE=gps TRAILER_LINK=sim ALLOW_EXTERNAL_GPS_SITL=0
        "$SCRIPT_DIR/run_city_landing.sh" path)
      ;;
  esac

  echo "[$run/$RUNS] $MODE -> $run_dir"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  '
    printf '%q ' timeout --foreground --signal=TERM \
      --kill-after="${KILL_AFTER_S}s" \
      "$RUN_TIMEOUT_S" "${command[@]}"
    printf '\n'
    continue
  fi

  status=0
  timeout --foreground --signal=TERM --kill-after="${KILL_AFTER_S}s" \
    "$RUN_TIMEOUT_S" "${command[@]}" || status=$?
  result="$(awk -F '\t' '$1 == "result" {value=$2} END {print value}' \
    "$run_dir/manifest.tsv" 2>/dev/null || true)"
  mapfile -t summaries < <(
    compgen -G "$run_dir/experiment_[0-9]*_summary.csv" | sort
  )
  mapfile -t samples < <(
    compgen -G "$run_dir/experiment_[0-9]*Z.csv" | sort
  )
  summary="${summaries[0]:-}"
  sample="${samples[0]:-}"
  export_status="$(awk -F '\t' \
    '$1 == "flight_csv_1hz" {value=$2} END {print value}' \
    "$run_dir/manifest.tsv" 2>/dev/null || true)"
  printf '%d,%s,%d,%s,%s,%s\n' \
    "$run" "$MODE" "$status" "$result" "$sample" "$summary" \
    >>"$BATCH_ROOT/batch_runs.csv"
  if [[ "$status" != "0" || "$result" != "$expected" \
      || "${#samples[@]}" != "1" || ! -s "$sample" \
      || "${#summaries[@]}" != "1" || ! -s "$summary" \
      || "$export_status" != "present" \
      || ! -s "$run_dir/flight_1hz.csv" \
      || ! -s "$run_dir/flight_summary.csv" \
      || ! -s "$run_dir/experiment_metrics.csv" ]]; then
    echo "run $run failed: status=$status result=${result:-missing} " \
         "sample=${sample:-missing} summary=${summary:-missing} " \
         "flight_csv=${export_status:-missing}" >&2
    ((failures += 1))
    if [[ "$FAIL_FAST" == "1" ]]; then
      exit 1
    fi
  fi
done

if [[ "$DRY_RUN" == "0" ]]; then
  echo "${RUNS}x artifacts: $BATCH_ROOT ($failures failed)"
  (( failures == 0 )) || exit 1
fi
