#!/bin/bash
# 온보드 풍선 추적 실행 (detector + interceptor)
# 사용법: ./run.sh [interceptor 추가 인자...]
# 예시:   ./run.sh
#         ./run.sh --takeoff-alt 20 --pn-gain 5
#         ./run.sh --url /dev/ttyTHS1 --baud 921600
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/tmp/balloon_interceptor"
mkdir -p "$LOG_DIR"

DETECTOR_PID=

cleanup() {
    echo ""
    echo "종료 중..."
    if [ -n "$DETECTOR_PID" ] && kill -0 "$DETECTOR_PID" 2>/dev/null; then
        kill "$DETECTOR_PID" 2>/dev/null
        wait "$DETECTOR_PID" 2>/dev/null || true
    fi
    echo "로그: $LOG_DIR/"
}
trap cleanup EXIT INT TERM

echo "=== 풍선 추적 시작 ==="
echo "detector 로그: $LOG_DIR/detector.log"
echo "interceptor 로그: $LOG_DIR/interceptor.log"
echo ""

# 1) detector: 카메라 → 풍선 검출 → UDP
python3 "$DIR/detector.py" \
    --port 15760 \
    2>&1 | tee "$LOG_DIR/detector.log" &
DETECTOR_PID=$!

sleep 1
if ! kill -0 "$DETECTOR_PID" 2>/dev/null; then
    echo "detector 시작 실패"
    exit 1
fi
echo "detector 시작됨 (PID $DETECTOR_PID)"

# 2) interceptor: MAVLink + 비전 → OFFBOARD 비행
#    첫 웨이포인트 수신 시 자동 ARM → TAKEOFF → OFFBOARD
python3 -u "$DIR/interceptor.py" \
    --udp-port 15760 \
    --log "$LOG_DIR/interceptor.log" \
    "$@"
