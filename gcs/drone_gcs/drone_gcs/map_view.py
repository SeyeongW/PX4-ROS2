#!/usr/bin/env python3
"""The map canvas — the centre of the GCS.

Draws a `MapPack` (static world) plus a `WorldState` (live telemetry and planner
output) in ENU metres, and turns mouse/key input into view changes and operator
intent signals.

Rendering split, and why:

* **Rasters** (basemap, occupancy) are blitted into a screen rectangle computed
  by the projection.  A world transform with a negative y scale would flip them,
  and image row 0 must land on the map's north edge.
* **Vectors** (buildings, paths, corridor) are drawn under the projection's world
  transform with *cosmetic* pens, so 205 polygons cost one transform instead of
  205 Python loops and line widths stay in pixels at any zoom.
* **Icons, labels and the HUD** are drawn in screen space, so they keep their
  size when the operator zooms.

Repaints are coalesced by a timer rather than fired per message: telemetry
arrives at 20-50 Hz from several topics and painting on each one would spend the
whole frame budget on redundant frames.
"""

from __future__ import annotations

import math

from . import theme
from .map_pack import Bounds, MapPack
from .projection import MapProjection
from .qt import QtCore, QtGui, QtWidgets, Signal
from .world_state import WorldState

REPAINT_HZ = 30
ZOOM_PER_WHEEL_STEP = 1.15
FIT_MARGIN_PX = 12

# Screen-space sizes, in pixels, that do not change with zoom.
DRONE_ICON_PX = 13.0
MARKER_ICON_PX = 7.0
LABEL_OFFSET_PX = 10

# A building is only worth outlining once it is a few pixels across.
MIN_BUILDING_PX = 2.0


def _event_pos(event) -> QtCore.QPointF:
    """Pointer position, working on both bindings.

    Qt 6 deprecated `QMouseEvent.pos()` in favour of `position()`; PyQt5 has only
    the former.
    """
    getter = getattr(event, "position", None)
    return QtCore.QPointF(getter()) if getter is not None else QtCore.QPointF(event.pos())


def _color(spec: str, alpha: float = 1.0) -> QtGui.QColor:
    color = QtGui.QColor(spec)
    if alpha < 1.0:
        color.setAlphaF(alpha)
    return color


def _pen(spec: str, width: float, *, style=QtCore.Qt.SolidLine, alpha: float = 1.0):
    pen = QtGui.QPen(_color(spec, alpha))
    pen.setWidthF(width)
    pen.setStyle(style)
    pen.setCosmetic(True)  # width in device pixels, immune to the world transform
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    return pen


class MapView(QtWidgets.QWidget):
    """Interactive ENU map.

    Signals carry operator intent in map coordinates; wiring them to ROS is the
    shell's job, so this widget stays usable with no middleware at all.
    """

    goalPicked = Signal(float, float)          # ENU x, y
    waypointPicked = Signal(float, float)      # ENU x, y (shift-click)
    buildingPicked = Signal(str, float)        # id, roof_z_m
    viewChanged = Signal()

    def __init__(self, pack: MapPack | None = None, state: WorldState | None = None, parent=None):
        super().__init__(parent)
        self.state = state if state is not None else WorldState()
        self._pack: MapPack | None = None
        self._projection = MapProjection.fit(Bounds(-1.0, 1.0, -1.0, 1.0), (1, 1))

        self._basemap: QtGui.QPixmap | None = None
        self._occupancy_tint: QtGui.QPixmap | None = None
        # Building rings as QPolygonF in ENU, built once per pack rather than per frame.
        self._building_polys: list[tuple[str, float, QtGui.QPolygonF, list[QtGui.QPolygonF]]] = []

        self._follow_drone = False
        self._drag_origin: QtCore.QPointF | None = None
        self._dragged = False
        self._cursor_enu: tuple[float, float] | None = None
        # Until the operator pans or zooms, the view keeps re-fitting the map.
        # A widget's first layout pass arrives after construction, so a fit done
        # at set_pack() time would otherwise be frozen at the placeholder size.
        self._user_view = False

        self.setMinimumSize(320, 240)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000 / REPAINT_HZ))
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        if pack is not None:
            self.set_pack(pack)

    # -------------------------------------------------------------------- the map
    def set_pack(self, pack: MapPack) -> None:
        """Swap the world.  Resets the view and drops the old map's rasters."""
        self._pack = pack
        self._basemap = self._load_raster(pack.basemap)
        self._occupancy_tint = (
            None if pack.occupancy is None else self._make_occupancy_tint(pack.occupancy.path)
        )
        self._building_polys = self._build_polygons(pack)
        self._projection = MapProjection.fit(pack.bounds, self._viewport(), FIT_MARGIN_PX)
        self.state.drone.clear_trail()
        for track in self.state.entities.values():
            track.clear_trail()
        self.viewChanged.emit()
        self.update()

    @property
    def pack(self) -> MapPack | None:
        return self._pack

    @property
    def projection(self) -> MapProjection:
        return self._projection

    def _load_raster(self, layer) -> QtGui.QPixmap | None:
        if layer is None:
            return None
        pixmap = QtGui.QPixmap(str(layer.path))
        return None if pixmap.isNull() else pixmap

    def _build_polygons(self, pack: MapPack):
        out = []
        for building in pack.buildings:
            outer = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in building.outer])
            holes = [
                QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in ring])
                for ring in building.holes
            ]
            out.append((building.id, building.roof_z_m, outer, holes))
        return out

    # ------------------------------------------------------------------- the view
    def fit_map(self) -> None:
        if self._pack is not None:
            self._projection = MapProjection.fit(
                self._pack.bounds, self._viewport(), FIT_MARGIN_PX
            )
            self._user_view = False
            self.viewChanged.emit()
            self.update()

    def set_follow_drone(self, follow: bool) -> None:
        self._follow_drone = bool(follow)
        if self._follow_drone:
            self._user_view = True  # a deliberate view choice; resizing keeps it
        self.update()

    def follows_drone(self) -> bool:
        return self._follow_drone

    def cursor_enu_m(self) -> tuple[float, float] | None:
        """Where the pointer is in map coordinates, for the status bar readout."""
        return self._cursor_enu

    def _viewport(self) -> tuple[int, int]:
        return (max(self.width(), 1), max(self.height(), 1))

    def _set_projection(self, projection: MapProjection) -> None:
        self._projection = projection
        self._user_view = True
        self.viewChanged.emit()
        self.update()

    # -------------------------------------------------------------------- ticking
    def _on_tick(self) -> None:
        if self._follow_drone and self.state.drone.is_valid():
            self._projection = self._projection.centered_on(*self.state.drone.xy)
        self.update()

    # --------------------------------------------------------------------- events
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._user_view or self._pack is None:
            # The operator chose this view; a window resize must not throw it away.
            self._projection = self._projection.resized(self._viewport())
        else:
            self._projection = MapProjection.fit(
                self._pack.bounds, self._viewport(), FIT_MARGIN_PX
            )
        self.viewChanged.emit()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        steps = event.angleDelta().y() / 120.0
        if steps == 0.0:
            return
        position = event.position() if hasattr(event, "position") else event.posF()
        self._set_projection(
            self._projection.zoomed_at(
                position.x(), position.y(), ZOOM_PER_WHEEL_STEP**steps
            )
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_origin = _event_pos(event)
            self._dragged = False

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        position = _event_pos(event)
        self._cursor_enu = self._projection.screen_to_enu(position.x(), position.y())
        if self._drag_origin is None:
            return
        dx = position.x() - self._drag_origin.x()
        dy = position.y() - self._drag_origin.y()
        if abs(dx) + abs(dy) > 2.0:
            self._dragged = True
            self._drag_origin = position
            # Dragging the map by hand means the view is no longer the drone's.
            self._follow_drone = False
            self._set_projection(self._projection.panned_by(dx, dy))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() != QtCore.Qt.LeftButton:
            return
        was_drag, self._drag_origin, self._dragged = self._dragged, None, False
        if was_drag:
            return
        position = _event_pos(event)
        x, y = self._projection.screen_to_enu(position.x(), position.y())
        if self._pack is not None and not self._pack.bounds.contains(x, y):
            return
        hit = self._building_at(x, y)
        if hit is not None:
            self.buildingPicked.emit(hit[0], hit[1])
            return
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            self.waypointPicked.emit(x, y)
        else:
            self.goalPicked.emit(x, y)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        key = event.key()
        if key == QtCore.Qt.Key_F:
            self.set_follow_drone(not self._follow_drone)
        elif key == QtCore.Qt.Key_R:
            self._follow_drone = False
            self.fit_map()
        elif key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self._set_projection(self._projection.zoomed(ZOOM_PER_WHEEL_STEP))
        elif key == QtCore.Qt.Key_Minus:
            self._set_projection(self._projection.zoomed(1.0 / ZOOM_PER_WHEEL_STEP))
        else:
            super().keyPressEvent(event)

    def _building_at(self, x: float, y: float):
        """The building under an ENU point, if any.  Holes count as outside."""
        point = QtCore.QPointF(x, y)
        for building_id, roof_z, outer, holes in self._building_polys:
            if not outer.containsPoint(point, QtCore.Qt.OddEvenFill):
                continue
            if any(hole.containsPoint(point, QtCore.Qt.OddEvenFill) for hole in holes):
                continue
            return (building_id, roof_z)
        return None

    # -------------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), _color(theme.BACKGROUND))

        pack, projection = self._pack, self._projection
        if pack is None:
            self._draw_centered_text(painter, "맵 팩이 없습니다")
            painter.end()
            return

        layers = pack.layers or ["basemap", "occupancy", "buildings", "geofence", "markers"]

        if "basemap" in layers:
            self._draw_raster(painter, self._basemap, pack.basemap, 1.0)

        # Buildings first, then the no-fly tint *over* them: the operator needs to
        # see how far the planner's inflated obstacle reaches past the wall, and a
        # tint underneath an opaque footprint would only show as a fringe.
        transform = self._world_transform(projection)
        if "buildings" in layers:
            painter.save()
            painter.setWorldTransform(transform)
            self._draw_buildings(painter, pack)
            painter.restore()
        if "occupancy" in layers and self._occupancy_tint is not None:
            self._draw_occupancy(painter, pack)

        painter.save()
        painter.setWorldTransform(transform)
        if "geofence" in layers:
            self._draw_geofence(painter, pack)
        self._draw_corridor(painter)
        self._draw_paths(painter)
        self._draw_trails(painter, pack)
        painter.restore()

        if "markers" in layers:
            self._draw_markers(painter, pack)
        self._draw_goal_and_waypoints(painter)
        self._draw_entities(painter, pack)
        self._draw_drone(painter)
        self._draw_scale_bar(painter)
        painter.end()

    def _world_transform(self, projection: MapProjection) -> QtGui.QTransform:
        """ENU metres -> device pixels, as a Qt transform (y negated)."""
        cx, cy = projection.center_enu_m
        width, height = projection.viewport_px
        scale = projection.px_per_m
        transform = QtGui.QTransform()
        transform.translate(0.5 * width, 0.5 * height)
        transform.scale(scale, -scale)
        transform.translate(-cx, -cy)
        return transform

    # --- rasters
    def _draw_raster(self, painter, pixmap, layer, opacity: float) -> None:
        if pixmap is None or layer is None:
            return
        left, top, width, height = self._projection.screen_rect_of(layer.bounds)
        painter.save()
        painter.setOpacity(opacity)
        # Smooth only when zoomed out past the source resolution; at high zoom the
        # operator wants to see the actual pixels, not an interpolated guess.
        painter.setRenderHint(
            QtGui.QPainter.SmoothPixmapTransform, width < pixmap.width()
        )
        painter.drawPixmap(QtCore.QRectF(left, top, width, height), pixmap,
                           QtCore.QRectF(pixmap.rect()))
        painter.restore()

    def _draw_occupancy(self, painter, pack: MapPack) -> None:
        """Tint the cells the planner cannot enter at cruise altitude."""
        self._draw_raster(painter, self._occupancy_tint, pack.occupancy, theme.OCCUPANCY_ALPHA)

    @staticmethod
    def _make_occupancy_tint(path) -> QtGui.QPixmap | None:
        """Blocked pixels -> solid colour, free pixels -> fully transparent.

        Done once per pack load with an explicit pixel pass.  Qt's mask-from-colour
        route is shorter but its in/out and colour0/colour1 conventions are easy to
        invert silently, and an inverted occupancy overlay would tint exactly the
        space that *is* flyable.
        """
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            grid = np.asarray(image.convert("L"))
        blocked = grid == 0
        color = _color(theme.OCCUPANCY)
        height, width = grid.shape
        # Format_ARGB32 is BGRA in memory on little-endian hosts.
        buffer = np.zeros((height, width, 4), dtype=np.uint8)
        buffer[blocked] = (color.blue(), color.green(), color.red(), 255)
        qimage = QtGui.QImage(
            buffer.tobytes(), width, height, 4 * width, QtGui.QImage.Format_ARGB32
        )
        # tobytes() gives Qt a temporary; copy before the buffer goes out of scope.
        return QtGui.QPixmap.fromImage(qimage.copy())

    # --- vectors (drawn under the world transform)
    def _draw_buildings(self, painter, pack: MapPack) -> None:
        visible = self._projection.visible_bounds()
        cruise_z = pack.cruise_z_m
        min_size_m = self._projection.px_to_metres(MIN_BUILDING_PX)

        blocked_brush = QtGui.QBrush(_color(theme.BUILDING_FILL, theme.BUILDING_ALPHA))
        clear_brush = QtGui.QBrush(_color(theme.BUILDING_OVERFLYABLE, theme.BUILDING_ALPHA))
        courtyard_brush = QtGui.QBrush(_color(theme.COURTYARD))
        painter.setPen(_pen(theme.BUILDING_EDGE, theme.W_OUTLINE))

        for building, (_id, roof_z, outer, holes) in zip(pack.buildings, self._building_polys):
            x0, y0, x1, y1 = building.aabb_xy
            if x1 < visible.x_min or x0 > visible.x_max:
                continue
            if y1 < visible.y_min or y0 > visible.y_max:
                continue
            if (x1 - x0) < min_size_m and (y1 - y0) < min_size_m:
                continue
            painter.setBrush(
                blocked_brush if pack.blocks_at(building, cruise_z) else clear_brush
            )
            painter.drawPolygon(outer)
            if holes:
                painter.setBrush(courtyard_brush)
                for hole in holes:
                    painter.drawPolygon(hole)

    def _draw_geofence(self, painter, pack: MapPack) -> None:
        bounds = pack.bounds
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(_pen(theme.GEOFENCE, theme.W_PATH, style=QtCore.Qt.DashLine))
        painter.drawRect(
            QtCore.QRectF(bounds.x_min, bounds.y_min, bounds.width_m, bounds.height_m)
        )

    def _draw_corridor(self, painter) -> None:
        if not self.state.corridor:
            return
        painter.setBrush(QtGui.QBrush(_color(theme.SFC, 0.10)))
        painter.setPen(_pen(theme.SFC, theme.W_HAIRLINE, alpha=0.55))
        for (x0, y0), (x1, y1) in self.state.corridor:
            painter.drawRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))

    def _draw_paths(self, painter) -> None:
        painter.setBrush(QtCore.Qt.NoBrush)
        for layer, color, width, style in (
            (self.state.global_path, theme.ASTAR, theme.W_PATH, QtCore.Qt.DashLine),
            (self.state.trajectory, theme.BSPLINE, theme.W_TRAJECTORY, QtCore.Qt.SolidLine),
            (self.state.mpc_preview, theme.MPC, theme.W_TRAJECTORY, QtCore.Qt.SolidLine),
        ):
            if len(layer.points) < 2:
                continue
            painter.setPen(_pen(color, width, style=style))
            painter.drawPolyline(
                QtGui.QPolygonF([QtCore.QPointF(p[0], p[1]) for p in layer.points])
            )

    def _draw_trails(self, painter, pack: MapPack) -> None:
        painter.setBrush(QtCore.Qt.NoBrush)
        if len(self.state.drone.trail) > 1:
            painter.setPen(_pen(theme.TRAIL, theme.W_TRAIL, alpha=0.9))
            painter.drawPolyline(
                QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in self.state.drone.trail])
            )
        for spec in pack.entities:
            track = self.state.entities.get(spec.name)
            if not spec.trail or track is None or len(track.trail) < 2:
                continue
            painter.setPen(_pen(spec.color, theme.W_TRAIL, alpha=0.55))
            painter.drawPolyline(
                QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in track.trail])
            )

    # --- screen-space overlays
    def _draw_markers(self, painter, pack: MapPack) -> None:
        painter.setFont(self._small_font())
        for marker in pack.markers:
            sx, sy = self._projection.enu_to_screen(*marker.enu_m)
            if not self._on_screen(sx, sy):
                continue
            color = _color(marker.color)
            painter.setPen(_pen(marker.color, 1.6))
            painter.setBrush(QtCore.Qt.NoBrush)
            half = MARKER_ICON_PX
            painter.drawEllipse(QtCore.QPointF(sx, sy), half, half)
            painter.drawLine(QtCore.QPointF(sx - half - 3, sy), QtCore.QPointF(sx + half + 3, sy))
            painter.drawLine(QtCore.QPointF(sx, sy - half - 3), QtCore.QPointF(sx, sy + half + 3))
            painter.setPen(color)
            painter.drawText(QtCore.QPointF(sx + LABEL_OFFSET_PX, sy - 6), marker.label)

    def _draw_goal_and_waypoints(self, painter) -> None:
        painter.setFont(self._small_font())
        for index, point in enumerate(self.state.waypoints_enu_m, start=1):
            sx, sy = self._projection.enu_to_screen(point[0], point[1])
            painter.setPen(_pen(theme.WAYPOINT, 1.8))
            painter.setBrush(QtGui.QBrush(_color(theme.WAYPOINT, 0.35)))
            painter.drawEllipse(QtCore.QPointF(sx, sy), 6.0, 6.0)
            painter.setPen(_color(theme.TEXT))
            painter.drawText(QtCore.QPointF(sx + LABEL_OFFSET_PX, sy - 4), str(index))

        goal = self.state.goal_enu_m
        if goal is None:
            return
        sx, sy = self._projection.enu_to_screen(goal[0], goal[1])
        painter.setPen(_pen(theme.GOAL, 2.4))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPointF(sx, sy), 9.0, 9.0)
        painter.drawLine(QtCore.QPointF(sx - 13, sy), QtCore.QPointF(sx + 13, sy))
        painter.drawLine(QtCore.QPointF(sx, sy - 13), QtCore.QPointF(sx, sy + 13))

    def _draw_entities(self, painter, pack: MapPack) -> None:
        """Dynamic objects: an oriented footprint at true scale, plus a relation line."""
        painter.setFont(self._small_font())
        for spec in pack.entities:
            track = self.state.entities.get(spec.name)
            if track is None or not track.is_valid():
                continue
            fresh = track.is_fresh()
            sx, sy = self._projection.enu_to_screen(*track.xy)
            length_px = max(self._projection.metres_to_px(spec.footprint_m[0]), 6.0)
            width_px = max(self._projection.metres_to_px(spec.footprint_m[1]), 4.0)

            painter.save()
            painter.translate(sx, sy)
            # Screen y is down, so a CCW-from-east ENU yaw rotates clockwise here.
            painter.rotate(-math.degrees(track.yaw_rad))
            painter.setPen(_pen(spec.color, 1.8, alpha=1.0 if fresh else 0.4))
            painter.setBrush(QtGui.QBrush(_color(spec.color, 0.35 if fresh else 0.12)))
            painter.drawRect(
                QtCore.QRectF(-0.5 * length_px, -0.5 * width_px, length_px, width_px)
            )
            # Nose mark, so heading is readable even on a square footprint.
            painter.drawLine(
                QtCore.QPointF(0.5 * length_px, 0.0),
                QtCore.QPointF(0.5 * length_px + 8.0, 0.0),
            )
            painter.restore()

            painter.setPen(_color(spec.color) if fresh else _color(spec.color, 0.45))
            label = spec.label if fresh else f"{spec.label} (끊김)"
            painter.drawText(QtCore.QPointF(sx + LABEL_OFFSET_PX, sy + 14), label)
            self._draw_relation(painter, spec, track)

    def _draw_relation(self, painter, spec, track) -> None:
        """Line + distance from the drone to an entity — the pursuit/capture cue."""
        drone = self.state.drone
        if not drone.is_valid() or not drone.is_fresh():
            return
        dx, dy = self._projection.enu_to_screen(*drone.xy)
        ex, ey = self._projection.enu_to_screen(*track.xy)
        painter.setPen(_pen(spec.color, theme.W_HAIRLINE, style=QtCore.Qt.DotLine, alpha=0.7))
        painter.drawLine(QtCore.QPointF(dx, dy), QtCore.QPointF(ex, ey))
        distance = math.dist(drone.xy, track.xy)
        painter.setPen(_color(theme.TEXT_DIM))
        painter.drawText(
            QtCore.QPointF(0.5 * (dx + ex) + 6, 0.5 * (dy + ey) - 6), f"{distance:.1f} m"
        )

    def _draw_drone(self, painter) -> None:
        drone = self.state.drone
        if not drone.is_valid():
            return
        fresh = drone.is_fresh()
        sx, sy = self._projection.enu_to_screen(*drone.xy)
        color = _color(theme.DRONE if fresh else theme.DRONE_STALE)

        if self.state.depth_is_fresh() and math.isfinite(self.state.depth_m):
            self._draw_depth_cone(painter, sx, sy, drone.yaw_rad, self.state.depth_m)

        painter.save()
        painter.translate(sx, sy)
        painter.rotate(-math.degrees(drone.yaw_rad))
        painter.setPen(_pen(theme.BACKGROUND, 1.2))
        painter.setBrush(QtGui.QBrush(color))
        half = DRONE_ICON_PX
        painter.drawPolygon(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(half, 0.0),
                    QtCore.QPointF(-0.6 * half, 0.7 * half),
                    QtCore.QPointF(-0.2 * half, 0.0),
                    QtCore.QPointF(-0.6 * half, -0.7 * half),
                ]
            )
        )
        painter.restore()

        painter.setPen(_pen(theme.DRONE if fresh else theme.DRONE_STALE, 1.0, alpha=0.5))
        painter.setBrush(QtCore.Qt.NoBrush)
        heading_px = 34.0
        painter.drawLine(
            QtCore.QPointF(sx, sy),
            QtCore.QPointF(
                sx + heading_px * math.cos(drone.yaw_rad),
                sy - heading_px * math.sin(drone.yaw_rad),
            ),
        )
        if not fresh:
            painter.setFont(self._small_font())
            painter.setPen(_color(theme.DRONE_STALE))
            painter.drawText(QtCore.QPointF(sx + LABEL_OFFSET_PX, sy - 12), "텔레메트리 끊김")

    def _draw_depth_cone(self, painter, sx, sy, yaw, distance_m) -> None:
        """Forward nearest-obstacle reading, drawn to scale ahead of the nose."""
        radius = self._projection.metres_to_px(min(distance_m, 60.0))
        if radius < 4.0:
            return
        half_angle = 20.0
        painter.setPen(_pen(theme.DEPTH_CONE, theme.W_HAIRLINE, alpha=0.8))
        painter.setBrush(QtGui.QBrush(_color(theme.DEPTH_CONE, 0.12)))
        rect = QtCore.QRectF(sx - radius, sy - radius, 2 * radius, 2 * radius)
        start = (math.degrees(yaw) - half_angle) * 16
        painter.drawPie(rect, int(start), int(2 * half_angle * 16))

    def _draw_scale_bar(self, painter) -> None:
        """A bar of round metres — the only honest way to read distance off a map."""
        target_px = 120.0
        metres = self._nice_length(self._projection.px_to_metres(target_px))
        length_px = self._projection.metres_to_px(metres)
        x0 = 14.0
        y0 = self.height() - 18.0
        painter.setFont(self._small_font())
        painter.setPen(_pen(theme.TEXT, 2.0))
        painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x0 + length_px, y0))
        painter.drawLine(QtCore.QPointF(x0, y0 - 4), QtCore.QPointF(x0, y0 + 4))
        painter.drawLine(
            QtCore.QPointF(x0 + length_px, y0 - 4), QtCore.QPointF(x0 + length_px, y0 + 4)
        )
        painter.setPen(_color(theme.TEXT))
        label = f"{metres:g} m" if metres < 1000 else f"{metres / 1000:g} km"
        painter.drawText(QtCore.QPointF(x0 + length_px + 8, y0 + 4), label)

    @staticmethod
    def _nice_length(raw_m: float) -> float:
        """Round a length down to 1, 2 or 5 times a power of ten."""
        if raw_m <= 0.0:
            return 1.0
        exponent = math.floor(math.log10(raw_m))
        base = 10.0**exponent
        for step in (5.0, 2.0, 1.0):
            if raw_m >= step * base:
                return step * base
        return base

    # --------------------------------------------------------------------- helpers
    def _on_screen(self, sx: float, sy: float, slack: float = 40.0) -> bool:
        return -slack <= sx <= self.width() + slack and -slack <= sy <= self.height() + slack

    def _small_font(self) -> QtGui.QFont:
        font = self.font()
        font.setPointSizeF(max(font.pointSizeF() - 1.0, 7.0))
        return font

    def _draw_centered_text(self, painter, text: str) -> None:
        painter.setPen(_color(theme.TEXT_DIM))
        painter.drawText(self.rect(), QtCore.Qt.AlignCenter, text)
