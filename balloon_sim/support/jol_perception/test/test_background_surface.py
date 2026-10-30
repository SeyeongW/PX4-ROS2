import numpy as np

from jol_perception.background_surface import BackgroundSurface


def test_plane_fills_scan_holes_but_keeps_objects_and_unseen_extent():
    random = np.random.default_rng(4)
    angle = random.uniform(-np.pi, np.pi, 1800)
    radius = random.uniform(4., 30., 1800)
    # An offset, slightly sloping ground plane, not a hardcoded z=0 mask.
    x, y = radius*np.cos(angle), radius*np.sin(angle)
    floor = np.c_[x, y, 1.7+.02*x+random.normal(0., .005, len(x))]
    clutter = random.uniform([-10, -10, 3], [10, 10, 8], (200, 3))
    surface = BackgroundSurface.fit(np.r_[floor, clutter])
    assert surface is not None
    points = np.array([[2., 0., 1.74], [2., 0., 1.94], [45., 0., 2.6], [0., 0., 10.]])
    assert surface.contains(points).tolist() == [True, False, False, False]
    before = surface.plane.copy()
    surface.contains(np.tile([0., 0., 5.], (1000, 1)))
    assert np.array_equal(surface.plane, before)  # Live objects never train it.


def test_insufficient_or_nonplanar_background_has_no_invented_floor():
    assert BackgroundSurface.fit(np.zeros((10, 3))) is None
    assert BackgroundSurface.fit(np.zeros((300, 3))) is None
    points = np.random.default_rng(3).uniform(-10., 10., (1000, 3))
    assert BackgroundSurface.fit(points) is None
