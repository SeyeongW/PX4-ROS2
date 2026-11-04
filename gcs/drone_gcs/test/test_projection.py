#!/usr/bin/env python3
"""Pan/zoom arithmetic for the map canvas.

The canvas has exactly one coordinate transform, so these tests are where view
bugs get caught — no Qt and no window needed.
"""

from __future__ import annotations

import math

import pytest

from drone_gcs.map_pack import Bounds
from drone_gcs.projection import MAX_PX_PER_M, MIN_FIT_FRACTION, MapProjection

CITY = Bounds(-650.0, 650.0, -650.0, 650.0)
VIEWPORT = (1000, 800)


@pytest.fixture
def fitted():
    return MapProjection.fit(CITY, VIEWPORT)


# ------------------------------------------------------------------------- fit
def test_fit_shows_the_whole_map_centered(fitted):
    assert fitted.center_enu_m == (0.0, 0.0)
    # The short axis is the binding one: 800 px over 1300 m.
    assert fitted.px_per_m == pytest.approx(800 / 1300)

    visible = fitted.visible_bounds()
    assert visible.y_min <= CITY.y_min and visible.y_max >= CITY.y_max
    assert visible.x_min <= CITY.x_min and visible.x_max >= CITY.x_max


def test_fit_honours_a_margin():
    margin = MapProjection.fit(CITY, VIEWPORT, margin_px=50)
    assert margin.px_per_m == pytest.approx((800 - 100) / 1300)


def test_fit_survives_a_zero_sized_viewport():
    """Widgets report 0 x 0 before their first layout pass."""
    projection = MapProjection.fit(CITY, (0, 0))
    assert projection.viewport_px == (1, 1)
    assert math.isfinite(projection.px_per_m)


# ------------------------------------------------------------------- transform
def test_center_maps_to_viewport_center(fitted):
    assert fitted.enu_to_screen(0.0, 0.0) == pytest.approx((500.0, 400.0))


def test_north_is_up_and_east_is_right(fitted):
    center = fitted.enu_to_screen(0.0, 0.0)
    north = fitted.enu_to_screen(0.0, 100.0)
    east = fitted.enu_to_screen(100.0, 0.0)

    assert north[1] < center[1]  # north is a smaller screen y
    assert east[0] > center[0]


@pytest.mark.parametrize("point", [(0.0, 0.0), (587.0, 580.0), (-649.0, 12.5), (200.0, -128.0)])
def test_screen_roundtrip(fitted, point):
    assert fitted.screen_to_enu(*fitted.enu_to_screen(*point)) == pytest.approx(point)


def test_scale_conversions_are_inverse(fitted):
    assert fitted.px_to_metres(fitted.metres_to_px(37.0)) == pytest.approx(37.0)


def test_screen_rect_of_puts_north_at_the_top(fitted):
    """Raster layers rely on this: image row 0 must land on the map's north edge."""
    left, top, width, height = fitted.screen_rect_of(CITY)

    assert width > 0 and height > 0
    assert (left, top) == pytest.approx(fitted.enu_to_screen(CITY.x_min, CITY.y_max))
    assert (left + width, top + height) == pytest.approx(
        fitted.enu_to_screen(CITY.x_max, CITY.y_min)
    )


# ------------------------------------------------------------------------ panning
def test_pan_moves_content_with_the_cursor(fitted):
    """Dragging right must move the map right, i.e. look further west."""
    panned = fitted.panned_by(100.0, 0.0)
    assert panned.center_enu_m[0] < fitted.center_enu_m[0]

    moved = panned.enu_to_screen(0.0, 0.0)
    assert moved[0] == pytest.approx(500.0 + 100.0)


def test_pan_is_reversible_inside_the_map():
    projection = MapProjection.fit(CITY, VIEWPORT).zoomed(4.0)
    there_and_back = projection.panned_by(60.0, -40.0).panned_by(-60.0, 40.0)
    assert there_and_back.center_enu_m == pytest.approx(projection.center_enu_m)


def test_center_cannot_leave_the_map(fitted):
    """Otherwise a stray drag loses the world off-screen with no way back."""
    for x, y in [(1e6, 1e6), (-1e6, 0.0), (0.0, -1e6)]:
        centered = fitted.centered_on(x, y)
        assert CITY.contains(*centered.center_enu_m)

    far = fitted.panned_by(-1e6, -1e6)
    assert CITY.contains(*far.center_enu_m)


# ------------------------------------------------------------------------ zooming
def test_zoom_at_cursor_pins_the_point_under_it(fitted):
    """The defining property of a good wheel zoom."""
    for sx, sy in [(120.0, 90.0), (500.0, 400.0), (940.0, 700.0)]:
        anchor = fitted.screen_to_enu(sx, sy)
        zoomed = fitted.zoomed_at(sx, sy, 2.0)
        assert zoomed.px_per_m == pytest.approx(2.0 * fitted.px_per_m)
        assert zoomed.enu_to_screen(*anchor) == pytest.approx((sx, sy), abs=1e-6)


def test_zoom_about_the_center_keeps_the_center(fitted):
    zoomed = fitted.zoomed(3.0)
    assert zoomed.center_enu_m == pytest.approx(fitted.center_enu_m)
    assert zoomed.enu_to_screen(0.0, 0.0) == pytest.approx((500.0, 400.0))


def test_zoom_out_is_floored_so_the_map_stays_visible(fitted):
    out = fitted
    for _ in range(40):
        out = out.zoomed(0.5)

    assert out.px_per_m == pytest.approx(out.min_px_per_m())
    on_screen_m = out.viewport_px[1] / out.px_per_m
    assert on_screen_m == pytest.approx(CITY.height_m / MIN_FIT_FRACTION)


def test_zoom_in_is_capped(fitted):
    deep = fitted
    for _ in range(80):
        deep = deep.zoomed(1.5)
    assert deep.px_per_m == pytest.approx(MAX_PX_PER_M)


def test_zoom_at_the_cap_does_not_drift(fitted):
    """A pinned anchor must stay pinned even when the zoom request is clamped."""
    capped = fitted.zoomed(1e9)
    anchor = capped.screen_to_enu(300.0, 200.0)
    again = capped.zoomed_at(300.0, 200.0, 4.0)

    assert again.px_per_m == pytest.approx(MAX_PX_PER_M)
    assert again.enu_to_screen(*anchor) == pytest.approx((300.0, 200.0), abs=1e-6)


# ------------------------------------------------------------------------- resize
def test_resize_keeps_the_center_and_zoom(fitted):
    resized = fitted.resized((1600, 900))

    assert resized.center_enu_m == fitted.center_enu_m
    assert resized.px_per_m == fitted.px_per_m
    assert resized.enu_to_screen(0.0, 0.0) == pytest.approx((800.0, 450.0))


def test_projection_is_immutable(fitted):
    """View changes return new objects, so no repaint sees a half-applied state."""
    before = (fitted.center_enu_m, fitted.px_per_m)
    fitted.panned_by(100.0, 100.0)
    fitted.zoomed(2.0)
    assert (fitted.center_enu_m, fitted.px_per_m) == before
