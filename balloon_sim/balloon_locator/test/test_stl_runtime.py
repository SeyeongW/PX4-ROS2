"""Compatibility registration must not activate a fixed-wing mission or alter UI controls."""
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from balloon_locator import stl_runtime


@pytest.fixture
def physical_profile(monkeypatch):
    profile = dict(
        aircraft=stl_runtime.AIRCRAFT_TYPE, data_status='assumed_not_measured',
        mass_kg=2.4, home_world=[-2., 0., .2], home_yaw_world=0.,
        sources={'motor': 'manufacturer test table; chosen propeller'},
        flight_parameters={'max_speed': 1.5, 'continuous_takeoff': True},
    )
    loader = ModuleType('balloon_locator.stl_airframe')
    loader.load_profile = lambda path: profile
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    return profile


def test_mc_registration_keeps_physics_provenance_without_fw_policy(physical_profile):
    before = deepcopy(physical_profile)
    profile = stl_runtime.multicopter_profile('local.json')
    # These are the actual NUB mission/adapter selectors: no transitions and no
    # old 25g reference profile. Physical simulation uses the unmodified JSON.
    assert not profile.get('flight_parameters', {})
    assert profile['aircraft'] == stl_runtime.AIRCRAFT_TYPE
    assert profile['mass_kg'] == 2.4
    assert profile['data_status'] == 'assumed_not_measured'
    assert profile['home_world'] == [-2., 0., .2]
    assert profile['sources'] == before['sources']
    profile['home_world'][0] = 99.
    assert physical_profile == before


def test_old_aircraft_profile_cannot_be_silently_registered(physical_profile):
    physical_profile['aircraft'] = 'tailsitter_20cm'
    with pytest.raises(ValueError, match='matching_aircraft_profile'):
        stl_runtime.multicopter_profile('old.json')


def test_profile_validation_failure_is_not_replaced_by_default(monkeypatch):
    def fail(path):
        raise ValueError('invalid_motor_profile')

    loader = ModuleType('balloon_locator.stl_airframe')
    loader.load_profile = fail
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    with pytest.raises(ValueError, match='invalid_motor_profile'):
        stl_runtime.multicopter_profile('bad.json')


def test_other_types_keep_original_checks_and_registration_is_restored(physical_profile):
    calls = []

    def original(kind, path=''):
        calls.append((kind, path))
        if kind == 'iris_proxy':
            return {}
        raise ValueError('invalid_aircraft_type')

    module = SimpleNamespace(aircraft_profile=original)
    with pytest.raises(ValueError, match='invalid_aircraft_type'):
        with stl_runtime.register_multicopter_profile(module):
            assert module.aircraft_profile(stl_runtime.AIRCRAFT_TYPE, 'local.json')['mass_kg'] == 2.4
            assert module.aircraft_profile('iris_proxy') == {}
            module.aircraft_profile('unregistered', 'other.json')
    assert module.aircraft_profile is original
    assert calls == [('iris_proxy', ''), ('unregistered', 'other.json')]


@pytest.mark.parametrize('entrypoint,module_name', [
    ('main_mission', 'static_flight'), ('main_adapter', 'mavros_adapter'),
])
def test_wrappers_preserve_reused_node_runtime_guards(monkeypatch, physical_profile, entrypoint, module_name):
    module = ModuleType('jol_perception.' + module_name)
    original = lambda kind, path='': {}
    module.aircraft_profile = original
    observed = []

    def main(args=None):
        observed.append(args)
        assert not module.aircraft_profile(stl_runtime.AIRCRAFT_TYPE, 'local.json').get('flight_parameters')
        # A guard in the original node must propagate through the wrapper.
        raise ValueError('simulation_time_required')

    module.main = main
    if entrypoint == 'main_mission':
        import balloon_locator
        monkeypatch.setattr(balloon_locator, 'stl_mission', module, raising=False)
    else:
        package = ModuleType('jol_perception')
        setattr(package, module_name, module)
        monkeypatch.setitem(sys.modules, 'jol_perception', package)
    with pytest.raises(ValueError, match='simulation_time_required'):
        getattr(stl_runtime, entrypoint)(args=['--ros-args'])
    assert observed == [['--ros-args']]
    assert module.aircraft_profile is original


@pytest.mark.parametrize('aircraft,fresh,mode,expected', [
    (stl_runtime.AIRCRAFT_TYPE, True, 'live', stl_runtime.AIRCRAFT_LABEL),
    (stl_runtime.AIRCRAFT_TYPE, True, 'locked_static_terminal', stl_runtime.AIRCRAFT_LABEL+' · 락 좌표 유지(가림)'),
    (stl_runtime.AIRCRAFT_TYPE, False, 'live', 'original caption'),
    ('tailsitter20cm_reference', True, 'live', 'original caption'),
])
def test_ui_caption_override_preserves_status_detail_and_controls(aircraft, fresh, mode, expected):
    class Label:
        def text(self):
            return self.value

        def setText(self, value):
            self.value = value

    class Dashboard:
        def __init__(self):
            self.observer_policy = Label()
            self.static_buttons = {'static_preview': SimpleNamespace(setEnabled=lambda value: None)}
            self.node = SimpleNamespace(state_fresh=lambda topic: fresh,
                packets={'static_status': {'aircraft_type': aircraft, 'coordinate_mode': mode}})
            self.controls_refreshed = False

        def refresh_static_flight(self, now):
            self.controls_refreshed = now
            self.observer_policy.setText('original caption\nOFFBOARD · 비무장\n발사 준비: 대기')

    window = stl_runtime.aircraft_dashboard(Dashboard)()
    window.refresh_static_flight(12.)
    assert window.controls_refreshed == 12.
    assert window.observer_policy.text() == expected + '\nOFFBOARD · 비무장\n발사 준비: 대기'


def test_ui_entrypoint_preserves_original_shutdown_and_restores_class(monkeypatch):
    class Dashboard:
        pass

    module = ModuleType('jol_perception.experiment_ui')
    module.Dashboard = Dashboard
    module.ExperimentNode = type('ExperimentNode', (), {})
    original_node = module.ExperimentNode
    observed = []

    def main(args=None):
        observed.append(args)
        assert issubclass(module.Dashboard, Dashboard)
        assert module.Dashboard is not Dashboard
        assert issubclass(module.ExperimentNode, original_node)
        assert module.ExperimentNode is not original_node
        raise RuntimeError('UI_stopped')

    module.main = main
    package = ModuleType('jol_perception')
    package.experiment_ui = module
    monkeypatch.setitem(sys.modules, 'jol_perception', package)
    with pytest.raises(RuntimeError, match='UI_stopped'):
        stl_runtime.main_ui(args=['--ros-args'])
    assert observed == [['--ros-args']]
    assert module.Dashboard is Dashboard
    assert module.ExperimentNode is original_node


@pytest.mark.parametrize('changes,original_allowed,expected', [
    ({'reason': 'ground_reference_preparation_pending'}, True, False),
    ({'controller_ready': False}, True, False),
    ({'plan_ready': True, 'phase': 'preview_ready'}, True, False),
    ({'reason': 'preview_failed: controller_not_ready'}, True, True),
    ({}, False, False),
    ({'aircraft_type': 'tailsitter20cm_reference', 'plan_ready': True}, True, True),
])
def test_preview_guard_blocks_duplicate_reference_but_allows_valid_retry(changes, original_allowed, expected):
    status = dict(aircraft_type=stl_runtime.AIRCRAFT_TYPE, phase='locked',
                  controller_ready=True, plan_ready=False, reason='')
    status.update(changes)

    class Node:
        packets = {'static_status': status}

        def static_action_allowed(self, action):
            return original_allowed

    node = stl_runtime.aircraft_experiment_node(Node)()
    assert node.static_action_allowed('static_preview') is expected
    # Other requests retain their existing state checks unchanged.
    assert node.static_action_allowed('static_launch') is original_allowed


def test_dashboard_disables_preview_during_auto_reference_then_reenables_retry():
    state = dict(aircraft_type=stl_runtime.AIRCRAFT_TYPE, phase='locked',
                 controller_ready=True, plan_ready=False, reason='ground_reference_preparation_pending')

    class Button:
        enabled = None

        def setEnabled(self, value):
            self.enabled = value

    class Dashboard:
        def __init__(self):
            self.node = SimpleNamespace(state_fresh=lambda topic: True, packets={'static_status': state})
            self.static_buttons = {'static_preview': Button()}
            self.observer_policy = SimpleNamespace(text=lambda: 'caption\ndetail', setText=lambda value: None)

        def refresh_static_flight(self, now):
            self.static_buttons['static_preview'].setEnabled(True)

    window = stl_runtime.aircraft_dashboard(Dashboard)()
    window.refresh_static_flight(1.)
    assert window.static_buttons['static_preview'].enabled is False
    state['reason'] = 'preview_failed: controller_not_ready'
    window.refresh_static_flight(2.)
    assert window.static_buttons['static_preview'].enabled is True
    state.update(plan_ready=True, phase='preview_ready', reason='')
    window.refresh_static_flight(3.)
    assert window.static_buttons['static_preview'].enabled is False
