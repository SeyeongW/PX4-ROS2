#!/usr/bin/env bash
# Local-only simulation. Never starts SIYI hardware nodes or onboard flight scripts.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Native PX4 uses shared /tmp sockets even in separate network namespaces.
# Lock the existing project directory; no persistent lock file is necessary.
exec {simulation_lock}< .
flock -n "$simulation_lock" || { echo 'Balloon simulation is already running in this workspace.' >&2; exit 75; }
source /opt/ros/humble/setup.bash
[[ -f install/local_setup.bash ]] || { echo 'Build first: ./tools/build_workspace.sh' >&2; exit 1; }
source install/local_setup.bash
export BALLOON_WORKSPACE_ROOT="$PWD"
for package in jol_sim jol_perception balloon_locator; do
  prefix="$(ros2 pkg prefix "$package")"
  [[ "$prefix" == "$PWD/install/"* ]] || { echo "Wrong overlay for $package: $prefix. Rebuild this workspace." >&2; exit 1; }
done
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}"
export ROS_LOCALHOST_ONLY=1 GAZEBO_IP=127.0.0.1
export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://127.0.0.1:11373}" GAZEBO_MODEL_DATABASE_URI=""
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${JOL_NVIDIA:-1}" == 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
fi
if [[ "${1:-}" == --exec ]]; then
  shift
  [[ $# -gt 0 ]] || { echo '--exec requires a command' >&2; exit 2; }
  exec unshare --user --map-current-user --keep-caps --net bash -c 'ip link set lo up && exec "$@"' balloon-check "$@"
fi
exec unshare --user --map-current-user --keep-caps --net bash -c 'ip link set lo up && exec python3 - "$@"' balloon-sim "$@" <<'PY'
import os,signal,subprocess,sys,tempfile,time
from pathlib import Path
requested=[0]
def request_shutdown(signum,frame):requested[0]=signum
for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(signum,request_shutdown)
def group_running(group):
    # Linux is already required by unshare. Ignore adopted zombies, which hold no files.
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=stat.read_text().rsplit(') ',1)[1].split()
            if fields[0]!='Z' and int(fields[2])==group:return True
        except (OSError,ValueError,IndexError):pass
    return False
with tempfile.TemporaryDirectory(prefix='px4_balloon_') as temporary:
    print('Private simulation runtime (removed on exit): '+temporary,flush=True)
    env=dict(os.environ,BALLOON_RUNTIME_DIR=temporary,TMPDIR=temporary,ROS_LOG_DIR=temporary+'/ros_log')
    command=['ros2','launch','balloon_locator','gazebo.launch.py',*sys.argv[1:]]
    process=subprocess.Popen(command,env=env,start_new_session=True)
    try:
        while process.poll() is None and not requested[0]:time.sleep(.1)
    finally:
        # A failed ROS launch parent may leave descendants alive. Stop the whole
        # group before deleting runtime files, even when the leader already exited.
        for signum,grace in ((signal.SIGINT,15),(signal.SIGTERM,5),(signal.SIGKILL,2)):
            try:os.killpg(process.pid,signum)
            except ProcessLookupError:break
            deadline=time.monotonic()+grace
            while time.monotonic()<deadline:
                process.poll()
                if not group_running(process.pid):break
                time.sleep(.05)
            if not group_running(process.pid):break
        process.wait()
    code=128+requested[0] if requested[0] else process.returncode
raise SystemExit(code)
PY
