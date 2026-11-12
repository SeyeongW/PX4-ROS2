"""Exercise the launcher supervisor with a fake ROS command, never an aircraft."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import pytest


LAUNCHER = Path(__file__).resolve().parents[2] / 'tools/run_gazebo.sh'


@pytest.fixture
def runtime_dir():
    with tempfile.TemporaryDirectory(prefix='balloon_launcher_test_') as folder:
        yield Path(folder)


def running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


@pytest.mark.parametrize('parent_exits', [True, False])
def test_supervisor_stops_descendants_before_removing_runtime(runtime_dir, parent_exits):
    # A ROS launch process can exit/crash while a descendant still owns files.
    fake = runtime_dir / 'ros2'
    fake.write_text(f'#!{sys.executable}\n'
                    'import os,subprocess,sys\n'
                    'from pathlib import Path\n'
                    'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])\n'
                    'Path(os.environ["PID_FILE"]).write_text(str(p.pid))\n'
                    + ('' if parent_exits else 'import time;time.sleep(60)\n'))
    fake.chmod(0o700)
    source = LAUNCHER.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    marker = runtime_dir / 'child.pid'
    process = subprocess.Popen([sys.executable, '-c', source], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=dict(os.environ, PATH=f'{runtime_dir}:{os.environ["PATH"]}',
                                        TMPDIR=str(runtime_dir), PID_FILE=str(marker)))
    child = None
    try:
        if not parent_exits:
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists(), 'fake ROS launch did not start'
            process.terminate()
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == (0 if parent_exits else 143), stderr
        child = int(marker.read_text())
        assert not running(child), 'ROS launch exited but its descendant is still running'
        runtime = Path(stdout.split('Private simulation runtime (removed on exit): ', 1)[1].splitlines()[0])
        assert not runtime.exists()
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        if child is None and marker.exists():
            child = int(marker.read_text())
        if child is not None and running(child):
            os.kill(child, signal.SIGKILL)


def test_project_lock_rejects_second_launcher_without_starting_ros():
    descriptor = os.open(LAUNCHER.parent.parent, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([str(LAUNCHER), '--exec', 'true'], text=True,
                                capture_output=True, timeout=10)
        assert result.returncode == 75
        assert 'already running' in result.stderr
    finally:
        os.close(descriptor)
