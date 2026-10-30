"""Reuse NUB's simulation-only MC mission and UI for the local STL aircraft.

The compatibility identifier selects the existing general multicopter branch;
it does not select Iris geometry or replace the model's physical parameters.
NUB source files and their simulation-time/readiness checks remain unchanged.
"""
from contextlib import contextmanager
from copy import deepcopy


AIRCRAFT_TYPE = 'stl_pacer_quad'
AIRCRAFT_LABEL = 'PX4 · MAVROS · STL/PACER · 총 2kg(가정)'


def multicopter_profile(path):
    """Validate the actual model profile, retaining provenance but no FW policy.

    NUB uses a nonempty ``flight_parameters`` mapping to activate its VTOL/FW
    mission. MC trajectory limits are supplied separately as ROS parameters by
    our launch file, within NUB's existing general-MC envelope.
    """
    from .stl_airframe import load_profile

    profile = deepcopy(load_profile(path))
    if profile.get('aircraft') != AIRCRAFT_TYPE:
        raise ValueError('STL_runtime_requires_matching_aircraft_profile')
    profile.pop('flight_parameters', None)
    return profile


@contextmanager
def register_multicopter_profile(module):
    """Register one simulation model for the duration of a reused entrypoint."""
    original = module.aircraft_profile

    def aircraft_profile(aircraft_type, path=''):
        if aircraft_type == AIRCRAFT_TYPE:
            return multicopter_profile(path)
        return original(aircraft_type, path)

    module.aircraft_profile = aircraft_profile
    try:
        yield
    finally:
        module.aircraft_profile = original


def main_mission(args=None):
    from . import stl_mission

    with register_multicopter_profile(stl_mission):
        return stl_mission.main(args=args)


def main_adapter(args=None):
    from jol_perception import mavros_adapter

    with register_multicopter_profile(mavros_adapter):
        return mavros_adapter.main(args=args)


def preview_retry_ready(status):
    """Lock already requests a preview; don't restart its pending reference step."""
    return (status.get('phase') == 'locked'
            and status.get('controller_ready') is True
            and status.get('plan_ready') is False
            and status.get('reason') != 'ground_reference_preparation_pending')


def aircraft_experiment_node(base):
    """Apply the same preview guard to buttons, shortcuts and direct UI requests."""
    class AircraftExperimentNode(base):
        def static_action_allowed(self, action):
            allowed = super().static_action_allowed(action)
            status = self.packets.get('static_status', {})
            if action == 'static_preview' and status.get('aircraft_type') == AIRCRAFT_TYPE:
                return allowed and preview_retry_ready(status)
            return allowed

    return AircraftExperimentNode


def aircraft_dashboard(base):
    """Change only the live aircraft caption; preserve all existing UI actions."""
    class AircraftDashboard(base):
        def refresh_static_flight(self, now):
            super().refresh_static_flight(now)
            if not self.node.state_fresh('static_status'):
                return
            status = self.node.packets.get('static_status', {})
            if status.get('aircraft_type') == AIRCRAFT_TYPE:
                if not preview_retry_ready(status):
                    self.static_buttons['static_preview'].setEnabled(False)
                text = self.observer_policy.text()
                _, separator, detail = text.partition('\n')
                label = AIRCRAFT_LABEL
                if status.get('coordinate_mode') == 'locked_static_terminal':
                    label += ' · 락 좌표 유지(가림)'
                self.observer_policy.setText(label + separator + detail)

    return AircraftDashboard


def main_ui(args=None):
    from jol_perception import experiment_ui

    original = experiment_ui.Dashboard
    original_node = experiment_ui.ExperimentNode
    experiment_ui.Dashboard = aircraft_dashboard(original)
    experiment_ui.ExperimentNode = aircraft_experiment_node(original_node)
    try:
        # Reuse Qt construction, ROS shutdown and signal handling unchanged.
        return experiment_ui.main(args=args)
    finally:
        experiment_ui.Dashboard = original
        experiment_ui.ExperimentNode = original_node
