#!/usr/bin/env python3
"""ENU metres <-> screen pixels.

The map canvas owns exactly one of these.  Every layer draws in ENU metres and
this class is the only place that knows about pixels, so a pan/zoom bug can only
live here — and it is testable without Qt.

Screen axes point right and **down**; ENU y points north, so the y scale is
negated.  Zoom is a single `px_per_m` for both axes: square pixels, no shear.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .map_pack import Bounds

# A view may zoom out until the map is this fraction of the viewport, and in
# until one metre spans this many pixels.  The floor keeps a map from shrinking
# to a dot; the ceiling stops float precision from showing at extreme zoom.
MIN_FIT_FRACTION = 0.25
MAX_PX_PER_M = 20.0


@dataclass(frozen=True)
class MapProjection:
    """An immutable view: which ENU point sits at the viewport centre, and how zoomed in.

    Immutability is deliberate — `pan`, `zoom_at` and friends return a new
    projection, so a view change is one assignment and never a half-applied
    mutation mid-repaint.
    """

    center_enu_m: tuple[float, float]
    px_per_m: float
    viewport_px: tuple[int, int]
    bounds: Bounds

    # ------------------------------------------------------------------ builders
    @staticmethod
    def fit(bounds: Bounds, viewport_px: tuple[int, int], margin_px: int = 0) -> "MapProjection":
        """The view that shows the whole map, centred, with `margin_px` to spare."""
        width, height = _positive_viewport(viewport_px)
        usable_w = max(width - 2 * margin_px, 1)
        usable_h = max(height - 2 * margin_px, 1)
        px_per_m = min(usable_w / bounds.width_m, usable_h / bounds.height_m)
        return MapProjection(
            center_enu_m=bounds.center_m,
            px_per_m=px_per_m,
            viewport_px=(width, height),
            bounds=bounds,
        )

    def resized(self, viewport_px: tuple[int, int]) -> "MapProjection":
        """Same centre and zoom in a new viewport size."""
        return replace(self, viewport_px=_positive_viewport(viewport_px))

    # ------------------------------------------------------------------ transform
    def enu_to_screen(self, x: float, y: float) -> tuple[float, float]:
        cx, cy = self.center_enu_m
        width, height = self.viewport_px
        return (
            0.5 * width + (x - cx) * self.px_per_m,
            0.5 * height - (y - cy) * self.px_per_m,
        )

    def screen_to_enu(self, sx: float, sy: float) -> tuple[float, float]:
        cx, cy = self.center_enu_m
        width, height = self.viewport_px
        return (
            cx + (sx - 0.5 * width) / self.px_per_m,
            cy - (sy - 0.5 * height) / self.px_per_m,
        )

    def metres_to_px(self, metres: float) -> float:
        return metres * self.px_per_m

    def px_to_metres(self, px: float) -> float:
        return px / self.px_per_m

    # -------------------------------------------------------------------- extents
    def visible_bounds(self) -> Bounds:
        """The ENU rectangle currently on screen — used to cull layers."""
        x_min, y_max = self.screen_to_enu(0.0, 0.0)
        x_max, y_min = self.screen_to_enu(*[float(v) for v in self.viewport_px])
        return Bounds(x_min, x_max, y_min, y_max)

    def screen_rect_of(self, bounds: Bounds) -> tuple[float, float, float, float]:
        """`bounds` as a screen rectangle (left, top, width, height).

        Rasters are drawn through this rather than through a flipped world
        transform, so image row 0 reliably lands on the map's north edge.
        """
        left, top = self.enu_to_screen(bounds.x_min, bounds.y_max)
        right, bottom = self.enu_to_screen(bounds.x_max, bounds.y_min)
        return (left, top, right - left, bottom - top)

    # --------------------------------------------------------------- view changes
    def panned_by(self, dx_px: float, dy_px: float) -> "MapProjection":
        """Drag the map by a screen delta (content follows the cursor)."""
        cx, cy = self.center_enu_m
        return self._clamped(
            replace(
                self,
                center_enu_m=(cx - dx_px / self.px_per_m, cy + dy_px / self.px_per_m),
            )
        )

    def centered_on(self, x: float, y: float) -> "MapProjection":
        return self._clamped(replace(self, center_enu_m=(x, y)))

    def zoomed_at(self, sx: float, sy: float, factor: float) -> "MapProjection":
        """Scale by `factor`, keeping the ENU point under (sx, sy) pinned there.

        This is what makes wheel-zoom feel right: the thing under the cursor does
        not slide away.
        """
        anchor = self.screen_to_enu(sx, sy)
        scaled = replace(self, px_per_m=self._clamp_scale(self.px_per_m * factor))
        # Where the anchor would land after scaling, and the shift that undoes it.
        moved = scaled.enu_to_screen(*anchor)
        return scaled.panned_by(sx - moved[0], sy - moved[1])

    def zoomed(self, factor: float) -> "MapProjection":
        """Scale about the viewport centre."""
        width, height = self.viewport_px
        return self.zoomed_at(0.5 * width, 0.5 * height, factor)

    # ------------------------------------------------------------------- clamping
    def min_px_per_m(self) -> float:
        """Zoomed all the way out: the map still fills `MIN_FIT_FRACTION` of the view."""
        width, height = self.viewport_px
        return MIN_FIT_FRACTION * min(width / self.bounds.width_m, height / self.bounds.height_m)

    def _clamp_scale(self, px_per_m: float) -> float:
        return max(self.min_px_per_m(), min(MAX_PX_PER_M, px_per_m))

    def _clamped(self, projection: "MapProjection") -> "MapProjection":
        """Keep the centre inside the map so the view cannot be lost off-world.

        Panning is otherwise free: at high zoom the operator must be able to put
        an edge of the map in the middle of the screen to inspect it.
        """
        cx, cy = projection.center_enu_m
        bounds = projection.bounds
        return replace(
            projection,
            center_enu_m=(
                min(max(cx, bounds.x_min), bounds.x_max),
                min(max(cy, bounds.y_min), bounds.y_max),
            ),
        )


def _positive_viewport(viewport_px) -> tuple[int, int]:
    width, height = int(viewport_px[0]), int(viewport_px[1])
    # A widget reports 0 x 0 before its first layout pass; 1 px keeps the
    # arithmetic finite until a real resize arrives.
    return (max(width, 1), max(height, 1))
