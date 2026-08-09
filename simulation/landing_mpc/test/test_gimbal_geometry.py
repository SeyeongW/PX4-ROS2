import numpy as np

from landing_mpc.frame import gimbal_axis_flu, gimbal_joint_angles
from landing_mpc.gimbal_control_node import _nadir_handoff


def test_gimbal_crosses_nadir_without_yaw_flip():
    yaw = 0.0
    pitch = -np.pi / 2.0

    for offset_deg in (5.0, 1.0, 0.0, -1.0, -5.0):
        offset = np.deg2rad(offset_deg)
        direction = np.array([np.sin(offset), 0.0, -np.cos(offset)])
        yaw, roll, pitch = gimbal_joint_angles(
            direction, yaw_hold=yaw, pitch_hold=pitch)

        assert roll == 0.0
        assert abs(yaw) < 1e-12
        assert np.isclose(pitch, -np.pi / 2.0 + offset)
        assert np.allclose(gimbal_axis_flu(yaw, pitch), direction, atol=1e-12)


def test_nadir_handoff_has_no_source_transition_step():
    depression = np.deg2rad(np.linspace(0.0, 90.0, 901))
    targets = np.column_stack((np.cos(depression), np.zeros_like(depression),
                               -np.sin(depression)))
    axes = np.asarray([
        _nadir_handoff(target, np.deg2rad(25.0)) for target in targets
    ])

    assert np.allclose(axes[depression <= np.deg2rad(25.0)], [0.0, 0.0, -1.0])
    assert np.allclose(axes[depression >= np.deg2rad(50.0)],
                       targets[depression >= np.deg2rad(50.0)])
    jumps = np.rad2deg(np.arccos(np.clip(np.sum(axes[:-1] * axes[1:], axis=1),
                                        -1.0, 1.0)))
    assert float(np.max(jumps)) < 1.0
