"""Focused contracts for MPC horizon extraction and MAVROS axis masks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from precision_landing.async_landing_control import (
    LandingCommandLimits,
    LandingSolveSnapshot,
    _command_constraint_rejection,
)
from precision_landing.landing_controller import (
    RELATIVE_MPC_ACCELERATION_HORIZON_INDEX,
    RELATIVE_MPC_ACCELERATION_LEAD_MS,
    RELATIVE_MPC_COMMAND_HORIZON_INDEX,
    RELATIVE_MPC_COMMAND_LEAD_MS,
    LandingControlCommand,
    LandingControlInput,
    PFeedforwardLandingController,
    RelativeMpcLandingController,
)


class _SuccessfulSolver:
    def solve(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return SimpleNamespace(
            success=True,
            message="solved",
            solve_time_s=0.001,
            iterations=3,
            deadline_missed=False,
            positions_m=np.asarray(
                ((10.0, 20.0, 30.0), (11.0, 21.0, 31.0))
            ),
            velocities_m_s=np.asarray(
                ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
            ),
            accelerations_m_s2=np.asarray(
                ((0.1, 0.2, 0.3), (1.1, 1.2, 1.3))
            ),
        )


def _control_input(
    *, clearance=1.0, descent_speed=0.0, pad_horizons=None
):
    horizon_context = {}
    if pad_horizons is not None:
        horizon_context = {
            "landing_pad_positions_horizon_enu_m": pad_horizons[0],
            "landing_pad_velocities_horizon_enu_m_s": pad_horizons[1],
            "landing_pad_accelerations_horizon_enu_m_s2": pad_horizons[2],
        }
    return LandingControlInput(
        vehicle_position_enu_m=np.zeros(3),
        vehicle_velocity_enu_m_s=np.zeros(3),
        vehicle_acceleration_enu_m_s2=np.zeros(3),
        vehicle_yaw_enu_rad=0.0,
        reference_positions_enu_m=np.zeros((2, 3)),
        reference_velocities_enu_m_s=np.zeros((2, 3)),
        target_yaw_enu_rad=0.0,
        horizontal_velocity_limit_m_s=20.0,
        vertical_velocity_limit_m_s=20.0,
        landing_pad_position_enu_m=np.zeros(3),
        landing_pad_velocity_enu_m_s=np.asarray((0.0, 0.0, 0.25)),
        landing_pad_acceleration_enu_m_s2=np.zeros(3),
        target_clearance_m=clearance,
        landing_constraints_enabled=True,
        descent_allowed=True,
        relative_descent_speed_m_s=descent_speed,
        **horizon_context,
    )


def _controller():
    return RelativeMpcLandingController(
        _SuccessfulSolver(),
        PFeedforwardLandingController(1.0, 1.0),
    )


def test_relative_mpc_uses_one_coherent_horizon_for_velocity_and_acceleration():
    command = _controller().compute(_control_input())

    assert RELATIVE_MPC_COMMAND_HORIZON_INDEX == 1
    assert RELATIVE_MPC_COMMAND_LEAD_MS == 200
    assert RELATIVE_MPC_ACCELERATION_HORIZON_INDEX == 1
    assert RELATIVE_MPC_ACCELERATION_LEAD_MS == 200
    np.testing.assert_allclose(command.velocity_setpoint_enu_m_s, (4, 5, 6))
    np.testing.assert_allclose(
        command.acceleration_setpoint_enu_m_s2[:2], (1.1, 1.2)
    )
    assert np.isnan(command.acceleration_setpoint_enu_m_s2[2])
    assert command.acceleration_enabled_axes == (True, True, False)
    assert command.extraction_policy == (
        "mpc_horizon_lead_200ms"
    )


def test_zero_clearance_marks_guarded_contact_settle_extraction():
    pad_positions = np.zeros((2, 3))
    pad_velocities = np.asarray(((1.0, 1.0, 0.25), (2.0, 3.0, 0.25)))
    pad_accelerations = np.zeros((2, 3))
    command = _controller().compute(
        _control_input(
            clearance=0.0,
            descent_speed=0.4,
            pad_horizons=(
                pad_positions,
                pad_velocities,
                pad_accelerations,
            ),
        )
    )

    # Current pad velocity is (0, 0); preserve only the future sample's
    # relative correction (4, 5) - (2, 3) at physical contact.
    np.testing.assert_allclose(command.velocity_setpoint_enu_m_s[:2], (2, 2))
    assert command.velocity_setpoint_enu_m_s[2] == pytest.approx(-0.15)
    np.testing.assert_allclose(
        command.acceleration_setpoint_enu_m_s2[:2], (1.1, 1.2)
    )
    assert np.isnan(command.acceleration_setpoint_enu_m_s2[2])
    assert command.extraction_policy == "contact_settle_vertical_override"
    assert "contact_settle_vertical_override" in command.status
    assert command.acceleration_enabled_axes == (True, True, False)


def test_relative_controller_forwards_one_immutable_pad_horizon():
    positions = np.asarray(((0.1, 0.01, 0.0), (0.2, 0.04, 0.0)))
    velocities = np.asarray(((1.0, 0.2, 0.0), (1.0, 0.4, 0.0)))
    accelerations = np.asarray(((0.0, 2.0, 0.0), (0.0, 2.0, 0.0)))
    control_input = _control_input(
        pad_horizons=(positions, velocities, accelerations)
    )
    # The controller boundary owns defensive copies.
    positions.fill(999.0)
    solver = _SuccessfulSolver()
    controller = RelativeMpcLandingController(
        solver,
        PFeedforwardLandingController(1.0, 1.0),
    )

    command = controller.compute(control_input)

    assert command.valid
    np.testing.assert_allclose(
        solver.kwargs["landing_pad_positions_horizon_enu_m"],
        ((0.1, 0.01, 0.0), (0.2, 0.04, 0.0)),
    )
    np.testing.assert_allclose(
        solver.kwargs["landing_pad_velocities_horizon_enu_m_s"],
        velocities,
    )
    np.testing.assert_allclose(
        solver.kwargs["landing_pad_accelerations_horizon_enu_m_s2"],
        accelerations,
    )


def test_landing_control_input_rejects_partial_or_misaligned_pad_horizon():
    with pytest.raises(ValueError, match="must be provided together"):
        LandingControlInput(
            **{
                **_control_input().__dict__,
                "landing_pad_positions_horizon_enu_m": np.zeros((2, 3)),
            }
        )

    with pytest.raises(ValueError, match="same length"):
        _control_input(
            pad_horizons=(
                np.zeros((2, 3)),
                np.zeros((3, 3)),
                np.zeros((2, 3)),
            )
        )


def test_axis_mask_requires_nan_for_ignored_acceleration_axis():
    disabled = np.full(3, np.nan)
    with pytest.raises(ValueError, match="disabled acceleration axes"):
        LandingControlCommand(
            disabled,
            np.zeros(3),
            np.asarray((1.0, 2.0, 0.0)),
            0.0,
            False,
            True,
            True,
            True,
            False,
            "invalid-afz",
            0.0,
            "relative_landing_mpc",
            "relative_landing_mpc",
            mpc_attempted=True,
            mpc_success=True,
            acceleration_enabled_axes=(True, True, False),
        )


def test_async_acceptance_understands_xy_only_acceleration():
    control_input = _control_input()
    command = _controller().compute(control_input)
    snapshot = LandingSolveSnapshot(
        generation=1,
        created_monotonic_s=1.0,
        source_stamp_s=1.0,
        phase="TRACK_AND_LAND",
        time_reset_count=0,
        control_input=control_input,
        target_covariance=None,
        solver_deadline_s=0.1,
        constraint_limits=LandingCommandLimits(20.0, 20.0, 20.0),
    )

    assert _command_constraint_rejection(snapshot, command) is None


def test_async_snapshot_deep_freezes_pad_horizon():
    horizons = tuple(np.zeros((2, 3), dtype=float) for _ in range(3))
    control_input = _control_input(pad_horizons=horizons)
    snapshot = LandingSolveSnapshot(
        generation=1,
        created_monotonic_s=1.0,
        source_stamp_s=1.0,
        phase="TRACK_AND_LAND",
        time_reset_count=0,
        control_input=control_input,
        target_covariance=None,
        solver_deadline_s=0.1,
        constraint_limits=LandingCommandLimits(20.0, 20.0, 20.0),
    )

    frozen = snapshot.control_input.landing_pad_positions_horizon_enu_m
    assert frozen is not None
    assert not frozen.flags.writeable
    horizons[0][0, 0] = 123.0
    assert frozen[0, 0] == 0.0


def test_mavros_encoder_masks_afz_but_keeps_afx_afy():
    source = (
        Path(__file__).resolve().parents[1]
        / "precision_landing"
        / "mpc_precision_landing_node.py"
    ).read_text(encoding="utf-8")
    publisher = source[source.index("    def _publish_setpoint(") :]
    publisher = publisher[: publisher.index("    def _request_mode(")]

    assert "command.acceleration_enabled_axes" in publisher
    assert "PositionTarget.IGNORE_AFX" in publisher
    assert "PositionTarget.IGNORE_AFY" in publisher
    assert "PositionTarget.IGNORE_AFZ" in publisher
    assert "if not command.acceleration_enabled:" not in publisher


def test_vehicle_response_observer_keeps_enabled_xy_acceleration():
    source = (
        Path(__file__).resolve().parents[1]
        / "precision_landing"
        / "mpc_precision_landing_node.py"
    ).read_text(encoding="utf-8")
    publisher = source[source.index("    def _publish_setpoint(") :]
    publisher = publisher[: publisher.index("    def _request_mode(")]

    assert "observer_acceleration = np.zeros(3, dtype=float)" in publisher
    assert "command.acceleration_enabled_axes" in publisher
    assert "observer_acceleration[axis] = acceleration[axis]" in publisher
    assert (
        "self.last_command_acceleration_enu = observer_acceleration"
        in publisher
    )
