"""Small source-level contracts for the single production mission path."""

from pathlib import Path
import re


PACKAGE = Path(__file__).resolve().parents[1]
MAIN = PACKAGE / "precision_landing" / "mpc_precision_landing_node.py"
TARGET = PACKAGE / "precision_landing" / "target_runtime.py"
COMMAND = PACKAGE / "precision_landing" / "command_runtime.py"
MISSION = PACKAGE / "precision_landing" / "mission_runtime.py"
RUNTIME = PACKAGE / "precision_landing" / "production_runtime.py"
RUNNER = PACKAGE.parent / "gazebo" / "run_precision_landing_experiment.sh"
CONFIG = PACKAGE / "config" / "mpc_precision_landing.yaml"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_production_main_has_no_gazebo_or_legacy_mpc_import():
    source = _source(MAIN)
    assert "import gz" not in source
    assert ".moving_trailer" not in source
    assert ".mpc_core" not in source


def test_timestamped_vehicle_and_rk4_models_are_wired_to_production():
    source = _source(MAIN) + _source(TARGET) + _source(COMMAND)
    assert "VehicleResponseEstimator(" in source
    assert "vehicle_response_estimator.update(" in source
    assert "vehicle_response_estimator.predict(" in source
    assert "TrailerRk4Predictor(" in source
    assert "trailer_motion_predictor.predict(" in source


def test_precision_handoff_requires_robust_aruco_motion_model():
    source = _source(MISSION)
    sensor_gate = source[source.index("acquisition_sensor_ready = bool(") :]
    sensor_gate = sensor_gate[: sensor_gate.index("acquisition_ready = bool(")]
    assert "self.vision_position_model_qualified" in sensor_gate
    assert "self._vehicle_acceleration_ready_for_relative_mpc()" in sensor_gate


def test_first_marker_frames_do_not_integrate_vehicle_velocity():
    source = _source(COMMAND)
    acquisition = source[
        source.index("def _stage_position_only_acquisition_command(") :
    ]
    acquisition = acquisition[: acquisition.index("def _solver_tick(")]
    assert "acquisition_capture_velocity_hold" not in acquisition
    assert "desired_velocity[:2] += measured_velocity[:2]" not in acquisition
    assert "command_velocity = desired_velocity.copy()" in acquisition
    assert "last_position_servo_velocity_enu" not in acquisition


def test_async_moving_deck_hold_has_no_false_50hz_outer_slew():
    source = _source(COMMAND)
    fallback = source[source.index("def _async_failure_command(") :]
    fallback = fallback[: fallback.index("def _cached_solver_command_valid(")]
    assert 'self._float("control_rate_hz")' not in fallback
    assert "safe_velocity = desired_velocity" in fallback


def test_descent_forms_one_physical_gate_snapshot_without_cache_churn():
    source = _source(MISSION)
    descent = source[source.index("def _tick_precision_descent(") :]
    descent = descent[: descent.index("def _tick_final_approach(")]
    decision = descent.index("require_solver_command=False")
    first_stage = descent.index("self._stage_trailer_relative_command(")
    assert decision < first_stage
    assert "if not self._solver_cache_safe_for_descent(now):" not in descent


def test_heartbeat_is_publish_only_and_solver_timer_owns_results():
    main_source = _source(MAIN)
    command_source = _source(COMMAND)
    heartbeat = main_source[main_source.index("def _heartbeat_tick(") :]
    heartbeat = heartbeat[: heartbeat.index("def _measured_rate_hz(")]
    assert "take_completed(" not in heartbeat
    assert "_apply_solver_result(" not in heartbeat
    assert "_expire_cached_solver_command(" not in heartbeat
    assert "self._publish_setpoint(command)" in heartbeat

    solver = command_source[command_source.index("def _solver_tick(") :]
    solver = solver[: solver.index("def _apply_solver_result(")]
    assert "take_completed(" in solver
    assert "_apply_solver_result(" in solver
    assert "_expire_cached_solver_command(" in solver
    assert "with self._solver_context_lock:" in solver


def test_one_control_tick_commits_only_its_final_solver_snapshot():
    source = _source(MISSION)
    control = source[source.index("def _control_tick(") :]
    control = control[: control.index("def _control_tick_impl(")]
    assert "with self._solver_context_lock:" in control
    assert "self._control_tick_snapshot_candidate = None" in control
    assert "self._pending_solver_snapshot = snapshot" in control
    assert "self._latest_solver_snapshot = snapshot" in control


def test_marker_reacquisition_keeps_qualified_motion_while_descent_gates_it():
    source = _source(TARGET)
    velocity = source[source.index("def _control_target_velocity_enu(") :]
    velocity = velocity[: velocity.index(
        "def _vehicle_acceleration_ready_for_relative_mpc("
    )]
    assert "vision_covariance_rejected" not in velocity
    assert "not self._vision_position_covariance_safe()" not in velocity
    assert '"aruco_dropout_hold"' in velocity

    mission = _source(MISSION)
    descent = mission[mission.index("def _descent_permission(") :]
    descent = descent[: descent.index("def _terminal_occlusion_estimate_safe(")]
    assert "position_covariance_m2=covariance" in descent
    assert "maximum_vision_age_s" not in descent


def test_qualified_terminal_tracking_uses_one_bounded_motion_lifetime():
    source = _source(TARGET)
    helper = source[
        source.index("def _vision_motion_continuation_limit_s(") :
    ]
    helper = helper[: helper.index("def _control_target_velocity_enu(")]
    assert "self.vision_position_model_qualified" in helper
    assert "LandingPhase.PRECISION_ALIGN" in helper
    assert "LandingPhase.FINAL_APPROACH" in helper
    assert "LandingPhase.TOUCHDOWN_CONFIRM" in helper
    assert '"vision_velocity_terminal_hold_s"' in helper

    # Callback reset, cache age, ordinary dropout and terminal timeout share it.
    assert (
        source + _source(MISSION)
    ).count("self._vision_motion_continuation_limit_s()") >= 5


def test_close_range_optical_gap_has_no_phase_boundary_deadband():
    source = _source(MISSION)
    descent = source[source.index("def _tick_precision_descent(") :]
    descent = descent[: descent.index("def _tick_final_approach(")]
    assert "terminal_gap_entry = bool(" in descent
    assert "self._terminal_occlusion_estimate_safe(estimate)" in descent
    assert '"terminal height reached during optical gap"' in descent
    assert "terminal_occlusion_max_clearance_m: 6.0" in _source(CONFIG)


def test_relative_mpc_handoff_requires_existing_one_second_acquisition_dwell():
    source = _source(MISSION)
    marker_track = source[source.index("def _tick_marker_track_down(") :]
    marker_track = marker_track[: marker_track.index("def _tick_precision_align(")]
    assert "self.marker_tracking_acquisition_since_s = mission_now" in marker_track
    assert 'self._float("precision_alignment_dwell_s")' in marker_track
    assert "acquisition dwell ready" in marker_track


def test_production_solver_timer_is_50_hz():
    assert "mpc_solver_rate_hz: 50.0" in _source(CONFIG)


def test_external_rk4_pad_horizon_is_wired_to_osqp():
    source = _source(TARGET) + _source(COMMAND)
    assert "trailer_motion_predictor.predict_horizon(" in source
    assert '"landing_pad_positions_horizon_enu_m"' in source
    assert '"landing_pad_velocities_horizon_enu_m_s"' in source
    assert '"landing_pad_accelerations_horizon_enu_m_s2"' in source
    assert "yaw_variance = float(motion_covariance[2, 2])" in source
    assert "yaw_variance = float(estimate_covariance[6, 6])" not in source


def test_contact_qp_uses_deck_constrained_acceleration():
    source = _source(COMMAND) + _source(MISSION)
    assert "if self._contact_entry_latched or self.phase == (" in source
    assert "control_vehicle_acceleration = np.asarray(" in source
    assert "control_vehicle_velocity[2] = float(pad_velocity[2])" in source
    assert source.count(
        "desired_velocity[:2] += relative_position_correction"
    ) == 1


def test_single_setpoint_publisher_and_publish_site():
    source = _source(MAIN)
    assert len(re.findall(r"create_publisher\(\s*PositionTarget", source)) == 1
    assert source.count("self.setpoint_pub.publish(") == 1


def test_production_stack_is_fixed_relative_osqp_hold():
    source = _source(RUNTIME)
    assert "PRODUCTION_CONTROLLER_TYPE = RELATIVE_LANDING_MPC_CONTROLLER_TYPE" in source
    assert 'PRODUCTION_SOLVER_NAME = "osqp"' in source
    assert "PRODUCTION_FAILURE_FALLBACK = HOLD_FALLBACK" in source


def test_one_runner_varies_only_environment_trailer_speed():
    source = _source(RUNNER)
    assert 'TRAILER_SPEED_VALUE="${TRAILER_SPEED_M_S:-5.0}"' in source
    assert 'if [[ "$ACTION" == "environment" ]]' in source
    assert 'CONTROLLER_RUNNER="$SCRIPT_DIR/run_px4_precision_landing.sh"' in source
    assert "TRAILER_SPEED_M_S" not in _source(MAIN)


def test_px4_composite_touchdown_can_disarm_without_estimator_or_solver():
    source = _source(MISSION)
    composite = source.index('source == "px4_composite_on_ground"')
    estimator = source.index(
        "estimate = self._current_target_estimate()", composite
    )
    request = source.index("self._request_arm(\n                    False, now", composite)
    assert composite < request < estimator


def test_moving_target_latency_is_not_double_compensated():
    production_source = (
        _source(MAIN)
        + _source(TARGET)
        + _source(COMMAND)
        + _source(MISSION)
    )
    assert "landing_motion_lead_time_s" not in production_source
    assert "landing_motion_lead_time_s" not in _source(CONFIG)


def test_future_trailer_odometry_is_deferred_until_state_watermark_advances():
    source = _source(MAIN) + _source(TARGET)
    assert "self._pending_trailer_odometry" in source
    assert "def _retry_pending_trailer_odometry(" in source
    assert 'result.reason == "future timestamp"' in source
