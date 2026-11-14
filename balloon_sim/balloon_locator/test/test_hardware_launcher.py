"""Exercise launcher wiring with shell stubs; never source ROS or open devices."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('installation', ['missing', 'wrong_overlay', 'local'])
@pytest.mark.parametrize('check_result', [0, 2])
def test_launcher_uses_local_install_and_stops_on_failed_check(tmp_path, installation, check_result):
    root = tmp_path / 'balloon_sim'
    (root / 'tools').mkdir(parents=True)
    script = root / 'tools/run_hardware_gcs.sh'
    shutil.copyfile(Path(__file__).parents[2] / 'tools/run_hardware_gcs.sh', script)
    if installation != 'missing':
        (root / 'install').mkdir()
        (root / 'install/local_setup.bash').touch()
    calls = tmp_path / 'calls'
    stub = tmp_path / 'shell_stubs'
    stub.write_text('''source() { printf 'source %s\\n' "$1" >> "$CALL_LOG"; }
ros2() {
  printf 'ros2 %s\\n' "$*" >> "$CALL_LOG"
  if [[ "$1 $2" == 'pkg prefix' ]]; then printf '%s\\n' "$FAKE_PREFIX"; return; fi
  if [[ "$1 $3" == 'run hardware_check' ]]; then return "$CHECK_RESULT"; fi
  return 0
}
''')
    result = subprocess.run(['bash', str(script), '--with-bridge'], text=True,
                            capture_output=True, timeout=10,
                            env=dict(os.environ, BASH_ENV=str(stub), CALL_LOG=str(calls), CHECK_RESULT=str(check_result),
                                     FAKE_PREFIX=str(root / 'install/balloon_locator')
                                     if installation == 'local' else '/old/install/balloon_locator'))
    log = calls.read_text() if calls.exists() else ''
    assert 'source ' + str(tmp_path / 'install') not in log
    assert 'source ' + str(root / 'install/setup.bash') not in log
    if installation == 'local' and check_result == 0:
        assert result.returncode == 0, result.stderr
        assert 'run balloon_locator hardware_check' in log
        assert 'run balloon_locator detection_viewer_node' in log
        assert 'simulation_mode:=false' in log
        assert 'run balloon_locator mavlink_bridge_node' in log
        assert 'launch balloon_locator ground_station.launch.py' in log
    elif installation == 'local':
        assert result.returncode == check_result
        assert 'run balloon_locator hardware_check' in log
        assert 'detection_viewer_node' not in log
        assert 'mavlink_bridge_node' not in log
        assert 'ros2 launch' not in log
    else:
        assert result.returncode != 0
        assert 'ros2 run' not in log and 'ros2 launch' not in log
