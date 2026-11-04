"""Exercise the real launch wiring with harmless subprocesses, never Gazebo."""
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip('launch')
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess, OpaqueFunction, Shutdown, TimerAction


WORKSPACE = Path(__file__).resolve().parents[3]


def load_launch(package, filename):
    path = WORKSPACE / 'src' / package / 'launch' / filename
    spec = importlib.util.spec_from_file_location(filename, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fake_launch(monkeypatch, tmp_path, *, fail=None, code=9, sim=False, overrides=()):
    """Keep actual launch actions/conditions/handlers; substitute only child executables."""
    module = load_launch('jol_sim' if sim else 'jol_perception',
                         'sim.launch.py' if sim else 'monitor.launch.py')
    monkeypatch.setattr(module, 'get_package_share_directory',
                        lambda package: str(WORKSPACE / 'src' / package))
    if sim:
        monkeypatch.setattr(module, 'get_package_prefix', lambda package: str(WORKSPACE / 'install' / package))
    processes, generated = [], []
    started = tmp_path / 'children-started'

    def fake_process(role, condition=None):
        child = ('import signal,sys,time; from pathlib import Path; '
                 'signal.signal(signal.SIGINT,lambda *_:sys.exit(0)); '
                 f'Path({str(started)!r}).touch(); '
                 f'time.sleep({0.03 if role == fail else 30}); sys.exit({code})')
        process = ExecuteProcess(cmd=[sys.executable, '-c', child], name=role, condition=condition)
        processes.append(process)
        return process

    def fake_execute(*, cmd, condition=None, **kwargs):
        if sim and cmd[0] == 'gzserver':
            generated.extend(Path(x) for x in cmd if str(x).endswith(('.world', '.yaml')))
        return fake_process(cmd[0] if sim else 'camera', condition)

    monkeypatch.setattr(module, 'ExecuteProcess', fake_execute)
    if not sim:
        monkeypatch.setattr(module, 'Node', lambda *, executable, condition=None, **kwargs:
                            fake_process(executable, condition))
        # Defense in depth: never load any simulator even if a condition regresses.
        def forbid_simulator(context):
            raise AssertionError('A monitor-only test attempted to include simulation')
        monkeypatch.setattr(module, 'IncludeLaunchDescription',
                            lambda *args, condition=None, **kwargs:
                            OpaqueFunction(function=forbid_simulator, condition=condition))
    model = tmp_path / 'model.onnx'
    model.touch()  # Static launch check only; model initialization belongs to the camera health check.
    args = [] if sim else ['simulation:=false', f'perception_python:={sys.executable}', f'model_path:={model}']
    service = LaunchService()
    service.context.launch_configurations.update(argument.split(':=', 1)
                                                 for argument in args + list(overrides))
    service.include_launch_description(module.generate_launch_description())
    service.include_launch_description(LaunchDescription([
        TimerAction(period=0.4, actions=[Shutdown(reason='test complete')])]))
    try:
        result = service.run()
        # All started siblings have been reaped when LaunchService returns.
        assert all(process.return_code is not None for process in processes
                   if (process.process_details or {}).get('pid'))
        return result, started.exists()
    finally:
        for path in generated:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize('role', ['tracker', 'camera', 'pointing'])
@pytest.mark.parametrize('code', [0, 9])
def test_required_monitor_exit_is_failure(monkeypatch, tmp_path, role, code):
    result, started = run_fake_launch(monkeypatch, tmp_path, fail=role, code=code)
    assert started and result == 1


@pytest.mark.parametrize('role', ['rviz2', 'experiment_ui', None])
def test_optional_or_orderly_shutdown_is_success(monkeypatch, tmp_path, role):
    result, started = run_fake_launch(monkeypatch, tmp_path, fail=role)
    assert started and result == 0


@pytest.mark.parametrize('role,code,expected', [
    ('gzserver', 9, 1), ('gzserver', 0, 1), ('gzclient', 9, 0), (None, 0, 0)])
def test_sim_server_required_client_optional(monkeypatch, tmp_path, role, code, expected):
    result, started = run_fake_launch(monkeypatch, tmp_path, sim=True, fail=role, code=code)
    assert started and result == expected


@pytest.mark.parametrize('arguments', [
    ['pointing:=1', 'turret_period:=1'],
    ['turret_lead:=1', 'prediction_horizon:=0.5'],
    ['model_path:=/missing/jol-model.onnx'],
    ['perception_python:=/missing/jol-python'],
    ['camera_mount:=laptop'],
    ['lidar_mount_mode:=invalid'],
    ['observation:=true', 'observation_scene:=/missing/scene.json'],
    ['observation:=true', 'observation_standoff:=nan'],
])
def test_invalid_configuration_starts_no_children(monkeypatch, tmp_path, arguments):
    result, started = run_fake_launch(monkeypatch, tmp_path, overrides=arguments)
    assert result == 1 and not started


def test_disabled_camera_and_pointing_do_not_require_their_configuration(monkeypatch, tmp_path):
    result, started = run_fake_launch(monkeypatch, tmp_path, overrides=[
        'camera:=false', 'pointing:=false', 'turret_lead:=100',
        'model_path:=/missing/jol-model.onnx', 'perception_python:=/missing/jol-python'])
    assert started and result == 0


def test_observation_preview_can_start_and_exit_without_stopping_tracking(monkeypatch, tmp_path):
    # 관찰 미리보기는 비행 제어기가 아니므로 종료해도 기존 추적 노드는 유지한다.
    result, started = run_fake_launch(monkeypatch, tmp_path, fail='observation',
                                     overrides=['observation:=true'])
    assert started and result == 0
