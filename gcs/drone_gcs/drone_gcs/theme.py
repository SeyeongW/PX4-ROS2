#!/usr/bin/env python3
"""Colours and pen widths.

The planner-stage colours are deliberately the same ones
`path_plan/tools/visualize_pipeline.py` and `pursuit_sim.py` use for their
figures, so a live screen and an offline plot of the same route read as the same
picture instead of forcing a mental re-mapping.
"""

from __future__ import annotations

# --- planner stages (shared with the path_plan figures)
ASTAR = "#e8590c"      # global A* polyline
SFC = "#1c7ed6"        # safe-flight-corridor boxes
BSPLINE = "#2f9e44"    # optimised trajectory
MPC = "#ae3ec9"        # MPC preview horizon
OBSTACLE = "#9aa4ad"   # building footprints

# --- map surface
BACKGROUND = "#14171a"
GEOFENCE = "#4b5563"
OCCUPANCY = "#e03131"        # tint over what the planner cannot enter
BUILDING_FILL = "#7e848c"
BUILDING_EDGE = "#5a5f66"
BUILDING_OVERFLYABLE = "#5c6b52"  # only used when the planner may overfly
COURTYARD = "#2b2f34"

# --- vehicle and targets
DRONE = "#f8f9fa"
DRONE_STALE = "#868e96"
TRAIL = "#74c0fc"
DEPTH_CONE = "#fab005"
GOAL = "#f03e3e"
WAYPOINT = "#fcc419"
MARKER_DEFAULT = "#adb5bd"
CAPTURE_RING = "#f59f00"

# --- HUD
TEXT = "#e9ecef"
TEXT_DIM = "#909296"
HUD_PANEL = "#1f2429"

# --- pens (px; cosmetic so zoom does not change them)
W_HAIRLINE = 1.0
W_PATH = 2.0
W_TRAJECTORY = 2.6
W_TRAIL = 1.8
W_OUTLINE = 1.0

# Layer opacity for the raster overlays.
OCCUPANCY_ALPHA = 0.28
BUILDING_ALPHA = 0.85
