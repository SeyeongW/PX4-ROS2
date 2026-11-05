"""Launch preparation must keep explicit values after PX4 airframe defaults."""
import importlib.util
from pathlib import Path

import pytest


def launch_module():
    path = Path(__file__).resolve().parents[1] / 'launch/monitor.launch.py'
    spec = importlib.util.spec_from_file_location('px4_startup_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_airframe_applies_and_checks_compiled_default_values(tmp_path):
    module = launch_module()
    source = tmp_path / 'original/etc'
    airframe = source / 'init.d-posix/airframes/1045_reference'
    airframe.parent.mkdir(parents=True)
    defaults = 'param set-default FW_THR_MIN 0.05\nparam set-default MPC_XY_VEL_MAX 8\n'
    airframe.write_text(defaults)
    rootfs = tmp_path / 'run'
    rootfs.mkdir()
    copied = module.prepare_px4_parameter_overrides(source, rootfs, airframe.name,
        {'FW_THR_MIN': 0., 'MPC_XY_VEL_MAX': 12.})
    content = (copied / 'init.d-posix/airframes' / airframe.name).read_text()
    assert airframe.read_text() == defaults
    assert content.startswith(defaults)
    assert content.index('param set FW_THR_MIN 0') > content.index('param set-default FW_THR_MIN')
    assert 'param compare FW_THR_MIN 0 ||' in content
    assert 'param compare MPC_XY_VEL_MAX 12 ||' in content
    assert 'exit 1' in content and 'px4_parameter_overrides.tsv' in content


@pytest.mark.parametrize('name,value', [('MPC_XY_VEL_MAX', float('nan')),
                                       ('MPC_XY_VEL_MAX', float('inf')),
                                       ('MPC_XY_VEL_MAX;bad', 2)])
def test_invalid_override_cannot_write_a_startup_file(tmp_path, name, value):
    with pytest.raises(ValueError, match='Invalid PX4 parameter override'):
        launch_module().prepare_px4_parameter_overrides(tmp_path / 'source', tmp_path / 'run',
                                                        '1045_reference', {name: value})
    assert not (tmp_path / 'run/etc').exists()
