"""Regression tests for sample-time pairing and retryable sensor admission."""

import numpy as np

from precision_landing.capture_state_history import VehicleStateHistory
from precision_landing.target_fusion import (
    FusionParameters,
    TimestampedTargetEstimator,
)


def test_position_is_interpolated_at_velocity_sample_time():
    history = VehicleStateHistory(duration_s=3.0)
    history.add_position(10.0, (1.0, 2.0, 3.0))
    history.add_position(10.2, (3.0, 6.0, 7.0))

    paired = history.interpolate_position(10.1, maximum_gap_s=0.25)

    np.testing.assert_allclose(paired, (2.0, 4.0, 5.0))


def test_exact_position_sample_does_not_require_attitude_history():
    history = VehicleStateHistory(duration_s=3.0)
    history.add_position(20.0, (4.0, 5.0, 6.0))

    paired = history.interpolate_position(20.0, maximum_gap_s=0.25)

    np.testing.assert_allclose(paired, (4.0, 5.0, 6.0))


def test_future_measurement_timestamp_can_be_retried_after_watermark_advances():
    estimator = TimestampedTargetEstimator(
        FusionParameters(future_tolerance_s=0.02)
    )
    covariance = np.diag((0.01, 0.01, 0.01, 0.01))
    stamp_s = 30.0
    stamp_key_ns = 30_000_000_000

    early = estimator.submit_vision(
        "vision_down",
        stamp_s,
        stamp_key_ns,
        (1.0, 2.0, 3.0),
        0.0,
        covariance,
        received_px4_time_s=29.9,
    )
    retried = estimator.submit_vision(
        "vision_down",
        stamp_s,
        stamp_key_ns,
        (1.0, 2.0, 3.0),
        0.0,
        covariance,
        received_px4_time_s=30.0,
    )
    duplicate = estimator.submit_vision(
        "vision_down",
        stamp_s,
        stamp_key_ns,
        (1.0, 2.0, 3.0),
        0.0,
        covariance,
        received_px4_time_s=30.0,
    )

    assert not early.queued
    assert early.reason == "future timestamp"
    assert retried.queued
    assert not duplicate.queued
    assert duplicate.reason == "duplicate timestamp"
    statistics = estimator.statistics["vision_down"]
    assert statistics.future == 1
    assert statistics.duplicate == 1

