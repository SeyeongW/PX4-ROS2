#!/usr/bin/env bash
# Relative-OSQP precision-landing wrapper.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_px4_precision_landing.sh" "$@"
