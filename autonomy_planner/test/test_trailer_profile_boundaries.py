import numpy as np
import pytest

from autonomy_planner.trailer_route import (
    JerkLimitedTrailerRoute,
    JerkLimitedWaypointRoute,
)


def test_scurve_is_position_velocity_acceleration_continuous_at_boundaries():
    route = JerkLimitedTrailerRoute()
    boundaries = np.cumsum([phase[1] for phase in route._phases])[:-1]
    epsilon = 1e-7
    for boundary in boundaries:
        before = route.sample(float(boundary - epsilon))
        after = route.sample(float(boundary + epsilon))
        assert np.linalg.norm(after.position_enu_m - before.position_enu_m) < 1e-5
        assert np.linalg.norm(after.velocity_enu_m_s - before.velocity_enu_m_s) < 1e-5
        assert np.linalg.norm(
            after.acceleration_enu_m_s2 - before.acceleration_enu_m_s2
        ) < 1e-5


def test_scurve_vector_norm_limits_and_exact_route_projection():
    route = JerkLimitedTrailerRoute(
        cruise_speed_m_s=3.0,
        max_acceleration_m_s2=0.9,
        max_jerk_m_s3=0.45,
    )
    for elapsed in np.linspace(0.0, route.duration_s, 5001):
        sample = route.sample(float(elapsed))
        assert np.linalg.norm(sample.velocity_enu_m_s) <= 3.0 + 1e-10
        assert np.linalg.norm(sample.acceleration_enu_m_s2) <= 0.9 + 1e-10
        assert np.linalg.norm(sample.jerk_enu_m_s3) <= 0.45 + 1e-10
        cross_track = sample.position_enu_m - (
            route.start_enu_m + route.direction_enu * sample.distance_m
        )
        assert np.linalg.norm(cross_track) < 1e-10
    assert route.sample(route.duration_s).position_enu_m == pytest.approx(
        [-128.0, -128.0, 0.0]
    )


def test_safe_city_waypoint_route_stops_at_each_corner_without_diagonal_cut():
    waypoints = np.asarray([
        [200.0, -125.0, 0.0],
        [200.0, -100.0, 0.0],
        [-128.0, -100.0, 0.0],
        [-128.0, -128.0, 0.0],
    ])
    route = JerkLimitedWaypointRoute(
        waypoints,
        cruise_speed_m_s=3.0,
        max_acceleration_m_s2=1.0,
        max_jerk_m_s3=0.5,
    )
    corner_times = np.cumsum([segment.duration_s for segment in route.segments])
    for index, corner_time in enumerate(corner_times[:-1], start=1):
        sample = route.sample(float(corner_time))
        assert sample.position_enu_m == pytest.approx(waypoints[index])
        assert np.linalg.norm(sample.velocity_enu_m_s) < 1e-12
        assert np.linalg.norm(sample.acceleration_enu_m_s2) < 1e-12

    for elapsed in np.linspace(0.0, route.duration_s, 3001):
        sample = route.sample(float(elapsed))
        x, y, _ = sample.position_enu_m
        on_first = abs(x - 200.0) < 1e-8 and -125.0 <= y <= -100.0
        on_second = abs(y + 100.0) < 1e-8 and -128.0 <= x <= 200.0
        on_third = abs(x + 128.0) < 1e-8 and -128.0 <= y <= -100.0
        assert on_first or on_second or on_third
        assert np.linalg.norm(sample.velocity_enu_m_s) <= 3.0 + 1e-10
        assert np.linalg.norm(sample.acceleration_enu_m_s2) <= 1.0 + 1e-10
        assert np.linalg.norm(sample.jerk_enu_m_s3) <= 0.5 + 1e-10

    final = route.sample(route.duration_s)
    assert final.complete
    assert final.position_enu_m == pytest.approx(waypoints[-1])
    assert final.distance_m == pytest.approx(25.0 + 328.0 + 28.0)


def test_waypoint_route_rejects_duplicate_or_malformed_waypoints():
    with pytest.raises(ValueError, match="shape"):
        JerkLimitedWaypointRoute([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="distinct"):
        JerkLimitedWaypointRoute([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
