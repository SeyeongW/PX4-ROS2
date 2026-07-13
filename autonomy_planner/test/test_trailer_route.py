import math

import numpy as np
import pytest

from autonomy_planner.trailer_route import JerkLimitedTrailerRoute


def test_default_route_contract_and_velocity_vector():
    route = JerkLimitedTrailerRoute()

    assert route.distance_m == pytest.approx(328.01371922548684)
    assert route.peak_speed_m_s == pytest.approx(3.0)
    assert route.cruise_time_s > 100.0
    cruise = route.sample(
        2.0 * route.jerk_ramp_time_s
        + route.constant_acceleration_time_s
        + 1.0
    )
    assert cruise.phase == "cruise"
    assert cruise.velocity_enu_m_s[0] == pytest.approx(-2.99987456, abs=1e-7)
    assert cruise.velocity_enu_m_s[1] == pytest.approx(-0.02743788, abs=1e-7)
    assert cruise.velocity_enu_m_s[2] == pytest.approx(0.0)


def test_dense_samples_respect_jerk_acceleration_speed_and_endpoint():
    route = JerkLimitedTrailerRoute(
        max_acceleration_m_s2=0.8,
        max_jerk_m_s3=0.4,
    )
    samples = [route.sample(t) for t in np.linspace(0.0, route.duration_s, 20001)]

    assert max(np.linalg.norm(s.velocity_enu_m_s) for s in samples) <= 3.0 + 1e-9
    assert max(np.linalg.norm(s.acceleration_enu_m_s2) for s in samples) <= 0.8 + 1e-9
    assert max(np.linalg.norm(s.jerk_enu_m_s3) for s in samples) <= 0.4 + 1e-9
    assert all(
        a.distance_m <= b.distance_m + 1e-10
        for a, b in zip(samples, samples[1:])
    )

    endpoint = route.sample(route.duration_s)
    assert endpoint.complete
    assert endpoint.phase == "hold"
    assert np.allclose(endpoint.position_enu_m, (-128.0, -128.0, 0.0), atol=1e-10)
    assert np.allclose(endpoint.velocity_enu_m_s, 0.0, atol=1e-12)
    assert np.allclose(endpoint.acceleration_enu_m_s2, 0.0, atol=1e-12)
    assert np.allclose(endpoint.jerk_enu_m_s3, 0.0, atol=1e-12)
    assert np.allclose(route.sample(route.duration_s + 60.0).position_enu_m,
                       endpoint.position_enu_m)


def test_short_route_reduces_peak_speed_without_losing_bounds():
    route = JerkLimitedTrailerRoute(
        start_enu_m=(0.0, 0.0, 1.0),
        goal_enu_m=(1.0, 0.0, 1.0),
        cruise_speed_m_s=3.0,
        max_acceleration_m_s2=1.0,
        max_jerk_m_s3=0.5,
    )

    assert route.peak_speed_m_s < route.cruise_speed_m_s
    assert route.cruise_time_s == pytest.approx(0.0, abs=1e-12)
    assert route.peak_acceleration_m_s2 <= route.max_acceleration_m_s2
    before = route.sample(math.nextafter(route.duration_s, 0.0))
    after = route.sample(route.duration_s)
    assert before.distance_m == pytest.approx(1.0, abs=1e-8)
    assert np.allclose(after.position_enu_m, (1.0, 0.0, 1.0))


def test_future_prediction_is_bounded_at_the_goal():
    route = JerkLimitedTrailerRoute()
    predicted = route.predict_position(route.duration_s - 1.0, 30.0)
    assert np.allclose(predicted, route.goal_enu_m)

    with pytest.raises(ValueError):
        route.predict_position(0.0, -0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_enu_m": (0.0, 0.0)},
        {"goal_enu_m": (0.0, float("nan"), 0.0)},
        {"cruise_speed_m_s": 0.0},
        {"max_acceleration_m_s2": -1.0},
        {"max_jerk_m_s3": float("inf")},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        JerkLimitedTrailerRoute(**kwargs)
