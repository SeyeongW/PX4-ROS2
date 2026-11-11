#!/bin/bash
# 실기체 쿼드 테일시터(Quad-Tailsitter) 온보드 통합 런처 (드론 탑재 컴퓨터 Jetson/Pi에서 실행)
# - detector.py (/dev/video0 노즈콘 카메라 풍선 검출 -> UDP 127.0.0.1:15760)
# - tailsitter_onboard.py (지상국 SiK COMMAND_INT 수신 + 50Hz SO(3) 자세/추력 + 3D LOS/PN 종말 유도)
# 기본 실행은 오프라인 파라미터 검사만 수행하며, --execute 인자를 주어야 실제 카메라/FCU 제어가 시작됩니다.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 - "$SCRIPT_DIR" "$@" <<'PY'
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import tailsitter_onboard

argv = sys.argv[2:]
args = tailsitter_onboard.parse_args(argv)
if not args.execute:
    raise SystemExit(tailsitter_onboard.main(argv))

children = []
stop_signal = None


def stop(signum, _frame):
    global stop_signal
    stop_signal = signum


for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, stop)


def shutdown():
    for sig, timeout in ((signal.SIGINT, 2.0), (signal.SIGTERM, 2.0), (signal.SIGKILL, 1.0)):
        for child in children:
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        for child in children:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        while True:
            alive = False
            for child in children:
                try:
                    os.killpg(child.pid, 0)
                    alive = True
                except ProcessLookupError:
                    pass
            if not alive:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)


with tempfile.TemporaryDirectory(prefix='tailsitter_onboard_') as temporary:
    runtime = Path(temporary)
    env = dict(os.environ, TMPDIR=str(runtime), PYTHONDONTWRITEBYTECODE='1')
    result = 1
    try:
        with (runtime / 'detector.log').open('w') as detector_log:
            detector = subprocess.Popen(
                [sys.executable, str(root / 'detector.py'), '--execute', '--port', str(args.udp_port)],
                env=env, stdout=detector_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            children.append(detector)
            command = [sys.executable, '-u', str(root / 'tailsitter_onboard.py'), *argv]
            if not args.log:
                command += ['--log', str(runtime / 'tailsitter_onboard.log')]
            flight = subprocess.Popen(command, env=env, start_new_session=True)
            children.append(flight)
            while stop_signal is None and detector.poll() is None and flight.poll() is None:
                time.sleep(0.05)
            if stop_signal is not None:
                result = 128 + stop_signal
            elif detector.poll() is not None:
                print('온보드 카메라 검출 프로세스(detector.py)가 종료되어 테일시터 제어를 중단합니다.', file=sys.stderr)
                print((runtime / 'detector.log').read_text()[-4000:], file=sys.stderr)
            else:
                result = flight.returncode
    finally:
        shutdown()
raise SystemExit(result)
PY
