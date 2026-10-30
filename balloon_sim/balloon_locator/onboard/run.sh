#!/bin/bash
# Legacy PN launcher, separate from the Gazebo UI/3m/Bezier mission.
# Default: offline argument validation. Only --execute enables real I/O/auto ARM.
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

root=Path(sys.argv[1])
sys.path.insert(0,str(root))
import interceptor

argv=sys.argv[2:]
args=interceptor.parse_args(argv)
if not args.execute:
    raise SystemExit(interceptor.main(argv))

children=[]
stop_signal=None
def stop(signum,_frame):
    global stop_signal
    stop_signal=signum
for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(signum,stop)

def shutdown():
    # Signal whole groups, including descendants left by an exited leader.
    for sig,timeout in ((signal.SIGINT,2.),(signal.SIGTERM,2.),(signal.SIGKILL,1.)):
        for child in children:
            try:os.killpg(child.pid,sig)
            except ProcessLookupError:pass
        deadline=time.monotonic()+timeout
        for child in children:
            try:child.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:pass
        while True:
            alive=False
            for child in children:
                try:os.killpg(child.pid,0);alive=True
                except ProcessLookupError:pass
            if not alive:return
            if time.monotonic()>=deadline:break
            time.sleep(.02)

with tempfile.TemporaryDirectory(prefix='balloon_onboard_') as temporary:
    runtime=Path(temporary)
    env=dict(os.environ,TMPDIR=str(runtime),PYTHONDONTWRITEBYTECODE='1')
    result=1
    try:
        with (runtime/'detector.log').open('w') as detector_log:
            detector=subprocess.Popen([sys.executable,str(root/'detector.py'),'--execute','--port',str(args.udp_port)],
                env=env,stdout=detector_log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(detector)
            command=[sys.executable,'-u',str(root/'interceptor.py'),*argv]
            if not args.log:command+=['--log',str(runtime/'interceptor.log')]
            flight=subprocess.Popen(command,env=env,start_new_session=True)
            children.append(flight)
            while stop_signal is None and detector.poll() is None and flight.poll() is None:
                time.sleep(.05)
            if stop_signal is not None:result=128+stop_signal
            elif detector.poll() is not None:
                print('검출 프로세스가 종료되어 legacy 제어를 중단합니다.',file=sys.stderr)
                print((runtime/'detector.log').read_text()[-4000:],file=sys.stderr)
            else:result=flight.returncode
    finally:
        shutdown()
raise SystemExit(result)
PY
