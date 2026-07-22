"""ROS-independent status helpers for the PX4/MAVROS landing baseline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from enum import Enum
from numbers import Integral, Real
from typing import Any


COMMAND_TRANSPORT = 'mavros_position_target'
COMMAND_TOPICS = (
    '/mavros/setpoint_raw/local',
    '/mavros/set_mode',
    '/mavros/cmd/arming',
)
TOUCHDOWN_MODE = 'px4_auto_land'


def sanitize_json_value(value: Any) -> Any:
    """Return a strict-JSON-safe copy, mapping non-finite numbers to null."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return sanitize_json_value(value.value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): sanitize_json_value(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    if hasattr(value, 'tolist'):
        return sanitize_json_value(value.tolist())
    if hasattr(value, 'item'):
        return sanitize_json_value(value.item())
    raise TypeError(f'unsupported status value type: {type(value).__name__}')


def dumps_strict_json(payload: Mapping[str, Any]) -> str:
    """Serialize without emitting non-standard NaN/Infinity tokens."""
    return json.dumps(
        sanitize_json_value(payload),
        allow_nan=False,
        separators=(',', ':'),
        sort_keys=True,
    )


def landing_observation(position_z: float, platform_height: float,
                        land_align_radius: float,
                        descend_cone: float) -> tuple[float, float]:
    """Compute read-only landing geometry from controller state."""
    height = max(float(position_z) - float(platform_height), 0.05)
    funnel = max(float(land_align_radius), float(descend_cone) * height)
    return height, funnel


def build_precision_landing_status(
        *, control_stack: str, baseline_run_id: str,
        vehicle_authority_id: str,
        stage: str, connected: bool, armed: bool, flight_mode: str,
        auto_takeoff: bool, require_enable: bool, position_enu_m: list[float],
        yaw_enu_rad: float, command_velocity_enu_m_s: list[float],
        command_yaw_enu_rad: float, command_age_s: float | None,
        marker_fresh: bool, marker_fresh_ticks: int, cue_enabled: bool,
        cue_fresh: bool, cue_age_s: float | None, cue_velocity_fresh: bool,
        cue_velocity_age_s: float | None, kf_initialized: bool,
        kf_state: list[float], kf_covariance_diagonal: list[float],
        kf_miss_count: int, height_above_marker_m: float,
        funnel_radius_m: float, last_horizontal_error_m: float | None,
        uses_sim_ground_truth_control_input: bool) -> dict[str, Any]:
    """Build the stable, flat status schema used for baseline measurements."""
    return {
        'schema_version': 1,
        'control_stack': control_stack,
        'baseline_run_id': baseline_run_id,
        'vehicle_authority_id': vehicle_authority_id,
        'command_transport': COMMAND_TRANSPORT,
        'command_topics': list(COMMAND_TOPICS),
        'stage': stage,
        'connected': connected,
        'armed': armed,
        'flight_mode': flight_mode,
        'auto_takeoff': auto_takeoff,
        'require_enable': require_enable,
        'position_enu_m': position_enu_m,
        'yaw_enu_rad': yaw_enu_rad,
        'command_velocity_enu_m_s': command_velocity_enu_m_s,
        'command_yaw_enu_rad': command_yaw_enu_rad,
        'command_age_s': command_age_s,
        'marker_fresh': marker_fresh,
        'marker_fresh_ticks': marker_fresh_ticks,
        'cue_enabled': cue_enabled,
        'cue_fresh': cue_fresh,
        'cue_age_s': cue_age_s,
        'cue_velocity_fresh': cue_velocity_fresh,
        'cue_velocity_age_s': cue_velocity_age_s,
        'kf_initialized': kf_initialized,
        'kf_state': kf_state,
        'kf_covariance_diagonal': kf_covariance_diagonal,
        'kf_miss_count': kf_miss_count,
        'height_above_marker_m': height_above_marker_m,
        'funnel_radius_m': funnel_radius_m,
        'last_horizontal_error_m': last_horizontal_error_m,
        'uses_sim_ground_truth_control_input':
            uses_sim_ground_truth_control_input,
        'touchdown_mode': TOUCHDOWN_MODE,
    }
