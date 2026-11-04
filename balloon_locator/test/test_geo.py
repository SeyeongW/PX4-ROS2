import importlib.util
import math
import pathlib

import pytest

from balloon_locator import geo

_spec = importlib.util.spec_from_file_location(
    'offboard_follower', pathlib.Path(__file__).resolve().parents[1] / 'onboard' / 'offboard_follower.py')
of = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(of)

CAM = (37.5, 127.0)


def test_declination_rotates_magnetic_north_to_true():
    n, e = geo.rotate_declination(100.0, 0.0, -8.5)                 # 자북 = 진북에서 서쪽 8.5°
    assert math.degrees(math.atan2(e, n)) == pytest.approx(-8.5)
    assert math.hypot(n, e) == pytest.approx(100.0)


def test_ned_to_global_roundtrip_distance_and_altitude():
    lat, lon, alt = geo.ned_to_global(60.0, 80.0, -20.0, *CAM, 1.5)
    assert of.horiz_dist_m(*CAM, lat, lon) == pytest.approx(100.0, rel=1e-4)
    assert lat > CAM[0] and lon > CAM[1]                            # 북동
    assert alt == pytest.approx(21.5)                               # 카메라 높이 1.5 + 풍선 20


def test_target_ok_limits():
    home = CAM
    lat, lon, _ = geo.ned_to_global(100.0, 0.0, 0.0, *CAM, 0.0)
    assert of.target_ok((lat, lon, 20.0), home, 150.0, 5.0, 60.0)
    assert not of.target_ok((lat, lon, 2.0), home, 150.0, 5.0, 60.0)           # 너무 낮음
    assert not of.target_ok((lat, lon, 20.0), home, 50.0, 5.0, 60.0)           # 너무 멂
    assert not of.target_ok((float('nan'), lon, 20.0), home, 150.0, 5.0, 60.0)


def test_jump_filter_requires_consistent_new_position():
    f = of.JumpFilter(max_jump=10.0, confirm=3)
    a = geo.ned_to_global(0.0, 0.0, -20.0, *CAM, 0.0)
    b = geo.ned_to_global(50.0, 0.0, -20.0, *CAM, 0.0)
    assert f(a)                                                     # 첫 표적은 채택
    assert f(geo.ned_to_global(3.0, 0.0, -20.0, *CAM, 0.0))         # 작은 이동은 채택
    assert not f(b) and not f(b)                                    # 점프는 보류
    assert f(b)                                                     # 3회 연속이면 채택
    assert not f(a)                                                 # 다시 점프 -> 보류
