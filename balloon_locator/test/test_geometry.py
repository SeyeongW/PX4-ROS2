import numpy as np
import pytest

from balloon_locator import geometry as g


def test_boresight_directions():
    ray = lambda az, el: g.pixel_ray_ned(960, 540, 960, 540, 1000, az, el)
    assert ray(0, 0) == pytest.approx([1, 0, 0], abs=1e-9)         # 북
    assert ray(90, 0) == pytest.approx([0, 1, 0], abs=1e-9)        # 동
    assert ray(0, 30)[2] < 0                                        # 위쪽 = D 음수
    assert ray(0, 30) == pytest.approx([np.cos(np.radians(30)), 0, -np.sin(np.radians(30))], abs=1e-9)


def test_pixel_offsets_map_to_expected_sides():
    right = g.pixel_ray_ned(1160, 540, 960, 540, 1000, 0, 0)
    below = g.pixel_ray_ned(960, 740, 960, 540, 1000, 0, 0)
    assert right[1] > 0 and below[2] > 0                           # 오른쪽 -> 동(+), 아래 -> D(+)


def test_range_and_roundtrip_position():
    f = g.focal_px(1920, 80.0, 2.0)
    truth = np.array([40.0, 30.0, -20.0])
    r = np.linalg.norm(truth)
    az = np.degrees(np.arctan2(truth[1], truth[0]))
    el = np.degrees(np.arcsin(-truth[2] / r))
    diameter_px = 0.3 * f / r
    pos, cov, rng = g.locate_balloon(
        960, 540, diameter_px, width_px=1920, height_px=1080, hfov_deg_at_1x=80.0, zoom=2.0,
        az_deg=az, el_deg=el, balloon_diameter_m=0.3, offset_ned=(1, 2, 3))
    assert rng == pytest.approx(r)
    assert pos == pytest.approx(truth + [1, 2, 3], abs=1e-6)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_covariance_larger_along_ray_than_across_at_small_pixel_sigma():
    ray = np.array([1.0, 0, 0])
    cov = g.measurement_covariance(ray, 100.0, 10.0, 2000.0, pixel_sigma_px=0.5)
    assert cov[0, 0] > cov[1, 1]                                    # 지름 기반 거리 오차가 지배적
