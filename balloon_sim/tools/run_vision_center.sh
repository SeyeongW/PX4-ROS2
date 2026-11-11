#!/usr/bin/env bash
# 호환용 별칭 -> ./tools/run_tailsitter_sim.sh 실행
set -eo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/run_tailsitter_sim.sh" "$@"
