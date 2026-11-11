#!/usr/bin/env bash
# 호환용 별칭 -> ./tools/restart_flight.sh 실행
set -eo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/restart_flight.sh" "$@"
