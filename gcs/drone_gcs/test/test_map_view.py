#!/usr/bin/env python3
"""Map canvas behaviour, driven offscreen.

These run without a display (`QT_QPA_PLATFORM=offscreen`) and without ROS, and
they exercise the real widget: a frame is actually painted, so a crash in any
layer fails the test.  What they cannot check is whether the picture *looks*
right — `tools/preview_map.py` renders a PNG for that.
"""

from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from drone_gcs.map_pack import Bounds, MapPack  # noqa: E402
from drone_gcs.world_state import WorldState  # noqa: E402

pytest.importorskip("PySide6", reason="no Qt binding installed")

from drone_gcs.map_view import MapView  # noqa: E402
from drone_gcs.qt import QtCore, QtGui, QtWidgets  # noqa: E402

from test_map_pack import CITY_PACK, minimal_descriptor, write_pack  # noqa: E402

VIEWPORT = (900, 600)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def pack(tmp_path):
    return MapPack.load(
        write_pack(
            tmp_path / "unit",
            minimal_descriptor(
                spawn_enu_m=[10.0, -20.0],
                entities=[{"name": "trailer", "footprint_m": [5.0, 3.0]}],
                markers=[{"name": "goal", "enu_m": [40.0, 20.0]}],
            ),
            buildings={
                "buildings": [
                    {
                        "id": "b1",
                        "outer": [[0, 0], [20, 0], [20, 10], [0, 10]],
                        "roof_z_m": 30.0,
                        "foundation_z_m": 0.0,
                    }
                ]
            },
        )
    )


@pytest.fixture
def view(qapp, pack):
    widget = MapView(pack)
    widget.resize(*VIEWPORT)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def paint(view) -> QtGui.QPixmap:
    """Render one frame; raises if any layer throws."""
    return view.grab()


# ----------------------------------------------------------------------- basics
def test_fits_the_map_after_the_first_layout(view, pack):
    """A widget is constructed before it has a size, so the fit must not stick."""
    assert view.projection.viewport_px == VIEWPORT
    assert view.projection.px_per_m == pytest.approx(
        min(
            (VIEWPORT[0] - 24) / pack.bounds.width_m,
            (VIEWPORT[1] - 24) / pack.bounds.height_m,
        )
    )
    assert view.projection.center_enu_m == pack.bounds.center_m


def test_paints_every_layer_without_error(view):
    view.state.global_path.set([(0, 0, 15), (30, 30, 15)])
    view.state.trajectory.set([(0, 0, 15), (30, 30, 15)])
    view.state.mpc_preview.set([(0, 0, 15), (5, 5, 15)])
    view.state.corridor = [((-5, -5), (5, 5))]
    view.state.goal_enu_m = (40.0, 20.0, 15.0)
    view.state.waypoints_enu_m = [(10.0, 10.0, 15.0)]
    view.state.drone.update(0.0, 0.0, 15.0, yaw_rad=0.5, speed_m_s=4.0)
    view.state.entity("trailer").update(-30.0, -20.0, 0.0, yaw_rad=1.0)
    view.state.set_depth(12.0)

    pixmap = paint(view)
    assert (pixmap.width(), pixmap.height()) == VIEWPORT


def test_paints_with_no_pack_at_all(qapp):
    """The window opens before a map is chosen; it must not crash."""
    widget = MapView()
    widget.resize(*VIEWPORT)
    assert not widget.grab().isNull()
    widget.close()


def test_paints_an_empty_state(view):
    assert not paint(view).isNull()


# ------------------------------------------------------------------ map swapping
def test_set_pack_swaps_the_world_and_resets_the_view(view, tmp_path):
    other = MapPack.load(
        write_pack(
            tmp_path / "other",
            minimal_descriptor(name="other", bounds_enu_m={"x": [0.0, 500.0], "y": [0.0, 500.0]}),
        )
    )
    view.state.drone.update(0.0, 0.0)
    assert len(view.state.drone.trail) == 1

    view.set_pack(other)

    assert view.pack.name == "other"
    assert view.projection.bounds == Bounds(0.0, 500.0, 0.0, 500.0)
    assert view.projection.center_enu_m == (250.0, 250.0)
    # A trail from the old world would be nonsense in the new one.
    assert len(view.state.drone.trail) == 0
    assert not paint(view).isNull()


def test_a_pack_without_optional_layers_still_paints(qapp, tmp_path):
    bare = MapPack.load(write_pack(tmp_path / "bare", minimal_descriptor(layers=["basemap"])))
    widget = MapView(bare)
    widget.resize(*VIEWPORT)
    assert not widget.grab().isNull()
    widget.close()


# --------------------------------------------------------------------- the view
def test_resize_refits_until_the_operator_takes_over(view, pack):
    before = view.projection.px_per_m
    view.resize(1800, 1200)
    assert view.projection.px_per_m > before  # re-fitted to the bigger window

    view._set_projection(view.projection.zoomed(4.0))
    chosen = view.projection.px_per_m
    view.resize(900, 600)
    assert view.projection.px_per_m == pytest.approx(chosen)  # kept


def test_follow_drone_centres_on_the_vehicle(view):
    view.state.drone.update(30.0, -15.0)
    view.set_follow_drone(True)
    view._on_tick()

    assert view.projection.center_enu_m == pytest.approx((30.0, -15.0))

    view.state.drone.update(35.0, -15.0)
    view._on_tick()
    assert view.projection.center_enu_m == pytest.approx((35.0, -15.0))


def test_follow_is_dropped_when_the_operator_drags(view):
    view.state.drone.update(30.0, -15.0)
    view.set_follow_drone(True)
    drag(view, QtCore.QPointF(400, 300), QtCore.QPointF(460, 340))
    assert not view.follows_drone()


def test_fit_map_restores_the_whole_map(view, pack):
    view._set_projection(view.projection.zoomed(8.0).centered_on(50.0, 50.0))
    view.fit_map()

    assert view.projection.center_enu_m == pack.bounds.center_m
    visible = view.projection.visible_bounds()
    assert visible.x_min <= pack.bounds.x_min and visible.x_max >= pack.bounds.x_max


# ------------------------------------------------------------------- interaction
def click(view, point: QtCore.QPointF, modifiers=QtCore.Qt.NoModifier) -> None:
    for kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease):
        view.mousePressEvent if False else None
        event = QtGui.QMouseEvent(
            kind, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, modifiers
        )
        if kind == QtCore.QEvent.MouseButtonPress:
            view.mousePressEvent(event)
        else:
            view.mouseReleaseEvent(event)


def drag(view, start: QtCore.QPointF, end: QtCore.QPointF) -> None:
    view.mousePressEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
        )
    )
    view.mouseMoveEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, end, QtCore.Qt.NoButton,
            QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
        )
    )
    view.mouseReleaseEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease, end, QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
        )
    )


def test_click_on_open_ground_picks_a_goal(view):
    seen = []
    view.goalPicked.connect(lambda x, y: seen.append((x, y)))

    target = (-60.0, 30.0)  # open ground, away from the one building
    click(view, QtCore.QPointF(*view.projection.enu_to_screen(*target)))

    assert seen and seen[0] == pytest.approx(target, abs=0.5)


def test_shift_click_picks_a_waypoint(view):
    goals, waypoints = [], []
    view.goalPicked.connect(lambda x, y: goals.append((x, y)))
    view.waypointPicked.connect(lambda x, y: waypoints.append((x, y)))

    point = QtCore.QPointF(*view.projection.enu_to_screen(-60.0, 30.0))
    click(view, point, QtCore.Qt.ShiftModifier)

    assert waypoints and not goals


def test_click_on_a_building_reports_it_instead_of_moving_the_goal(view):
    goals, buildings = [], []
    view.goalPicked.connect(lambda x, y: goals.append((x, y)))
    view.buildingPicked.connect(lambda i, z: buildings.append((i, z)))

    click(view, QtCore.QPointF(*view.projection.enu_to_screen(10.0, 5.0)))

    assert buildings == [("b1", 30.0)]
    assert not goals


def test_dragging_pans_and_does_not_pick_a_goal(view):
    goals = []
    view.goalPicked.connect(lambda x, y: goals.append((x, y)))
    before = view.projection.center_enu_m

    drag(view, QtCore.QPointF(400, 300), QtCore.QPointF(460, 340))

    assert view.projection.center_enu_m != before
    assert not goals  # a drag is not a click


def test_click_outside_the_map_is_ignored(view):
    goals = []
    view.goalPicked.connect(lambda x, y: goals.append((x, y)))

    # The fitted view has letterbox margins; a corner pixel is off-map.
    click(view, QtCore.QPointF(2.0, 2.0))

    assert not goals


def test_keys_toggle_follow_and_reset_the_view(view):
    def press(key):
        view.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier))

    press(QtCore.Qt.Key_F)
    assert view.follows_drone()
    press(QtCore.Qt.Key_F)
    assert not view.follows_drone()

    fitted = view.projection.px_per_m
    press(QtCore.Qt.Key_Plus)
    assert view.projection.px_per_m > fitted
    press(QtCore.Qt.Key_R)
    assert view.projection.px_per_m == pytest.approx(fitted)


def test_cursor_readout_tracks_the_pointer(view):
    view.mouseMoveEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(*view.projection.enu_to_screen(12.0, -8.0)),
            QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
        )
    )
    assert view.cursor_enu_m() == pytest.approx((12.0, -8.0), abs=0.5)


# --------------------------------------------------------- the real city map pack
@pytest.mark.skipif(not CITY_PACK.is_dir(), reason="city_uav pack not baked yet")
def test_city_pack_paints_at_several_zooms(qapp):
    pack = MapPack.load(CITY_PACK)
    widget = MapView(pack)
    widget.resize(1000, 700)
    widget.show()
    qapp.processEvents()

    widget.state.drone.update(587.0, 580.0, 25.0, yaw_rad=math.pi, speed_m_s=4.0)
    widget.state.entity("trailer").update(-587.0, -512.0, 0.0)
    widget.state.set_depth(8.0)

    for zoom in (1.0, 4.0, 20.0):
        widget._set_projection(widget.projection.zoomed(zoom).centered_on(587.0, 580.0))
        assert not widget.grab().isNull()
    widget.close()
