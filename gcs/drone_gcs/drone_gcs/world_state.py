#!/usr/bin/env python3
"""What the map draws — plain data, no Qt and no ROS.

The ROS link (or the replay reader) fills this; the canvas only reads it.  Keeping
the two apart means the drawing code can be exercised from a test or a preview
script with no middleware running.

Freshness is part of the state.  A GCS that keeps drawing a drone at its last
known spot after telemetry dies is lying, so every live field carries the
timestamp it arrived and asks `is_fresh()` before it is trusted.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

# Position/attitude go stale fast; mission phase and planner output are allowed to
# be old because they only change when something happens.
POSE_TIMEOUT_S = 1.5
ENTITY_TIMEOUT_S = 3.0
DEPTH_TIMEOUT_S = 1.5

TRAIL_MAX_POINTS = 4000
# Trail points closer together than this add nothing at any usable zoom.
TRAIL_MIN_STEP_M = 0.5


def _now() -> float:
    """Monotonic wall clock for staleness.  Never the ROS clock — a paused sim
    must still be reported as stale rather than looking live forever."""
    return time.monotonic()


@dataclass
class Track:
    """A live position + heading with a decaying trail behind it."""

    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_rad: float = 0.0
    speed_m_s: float = 0.0
    stamp_s: float = -1.0
    timeout_s: float = POSE_TIMEOUT_S
    trail: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=TRAIL_MAX_POINTS)
    )

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x_m, self.y_m)

    def is_valid(self) -> bool:
        """True once any position has ever arrived."""
        return self.stamp_s >= 0.0

    def is_fresh(self, now: float | None = None) -> bool:
        if self.stamp_s < 0.0:
            return False
        return ((_now() if now is None else now) - self.stamp_s) < self.timeout_s

    def update(self, x, y, z=0.0, yaw_rad=0.0, speed_m_s=0.0, now: float | None = None) -> None:
        stamp = _now() if now is None else now
        moved = not self.trail or math.dist((x, y), self.trail[-1]) >= TRAIL_MIN_STEP_M
        self.x_m, self.y_m, self.z_m = float(x), float(y), float(z)
        self.yaw_rad = float(yaw_rad)
        self.speed_m_s = float(speed_m_s)
        self.stamp_s = stamp
        if moved:
            self.trail.append((float(x), float(y)))

    def clear_trail(self) -> None:
        self.trail.clear()


@dataclass
class PathLayer:
    """One polyline the planner published, with the time it arrived."""

    points: list[tuple[float, float, float]] = field(default_factory=list)
    stamp_s: float = -1.0

    def set(self, points, now: float | None = None) -> None:
        self.points = [(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0) for p in points]
        self.stamp_s = _now() if now is None else now

    def clear(self) -> None:
        self.points = []
        self.stamp_s = -1.0

    def __bool__(self) -> bool:
        return bool(self.points)

    def length_m(self) -> float:
        """Path length in 3D — what the operator compares against straight-line distance."""
        return sum(
            math.dist(a, b) for a, b in zip(self.points, self.points[1:])
        )


@dataclass
class VehicleStatus:
    """Flight-stack state, straight off MAVROS."""

    connected: bool = False
    armed: bool = False
    mode: str = ""
    landed_state: str = ""
    battery_pct: float | None = None
    stamp_s: float = -1.0

    def is_fresh(self, now: float | None = None) -> bool:
        if self.stamp_s < 0.0:
            return False
        return ((_now() if now is None else now) - self.stamp_s) < POSE_TIMEOUT_S


@dataclass
class PlannerStatus:
    """Enough to tell "thinking" from "stuck".

    A* -> SFC -> B-spline takes seconds on this map, so a replan that shows no
    progress indicator reads as a frozen GUI.
    """

    replans: int = 0
    planning_since_s: float | None = None
    last_plan_duration_s: float | None = None
    corridor_boxes: int = 0
    mpc_solve_ms: float | None = None
    note: str = ""

    def begin_plan(self, now: float | None = None) -> None:
        self.planning_since_s = _now() if now is None else now

    def end_plan(self, now: float | None = None) -> None:
        if self.planning_since_s is not None:
            self.last_plan_duration_s = (_now() if now is None else now) - self.planning_since_s
        self.planning_since_s = None
        self.replans += 1

    def is_planning(self) -> bool:
        return self.planning_since_s is not None

    def planning_elapsed_s(self, now: float | None = None) -> float:
        if self.planning_since_s is None:
            return 0.0
        return (_now() if now is None else now) - self.planning_since_s


@dataclass
class WorldState:
    """Everything live the canvas and the status panel read."""

    drone: Track = field(default_factory=Track)
    entities: dict[str, Track] = field(default_factory=dict)

    global_path: PathLayer = field(default_factory=PathLayer)
    trajectory: PathLayer = field(default_factory=PathLayer)
    mpc_preview: PathLayer = field(default_factory=PathLayer)
    # Safe-flight-corridor boxes as ((x0, y0), (x1, y1)) in ENU.
    corridor: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)

    goal_enu_m: tuple[float, float, float] | None = None
    waypoints_enu_m: list[tuple[float, float, float]] = field(default_factory=list)

    status: VehicleStatus = field(default_factory=VehicleStatus)
    planner: PlannerStatus = field(default_factory=PlannerStatus)

    # Forward nearest-obstacle distance from the depth camera.
    depth_m: float = math.inf
    depth_stamp_s: float = -1.0
    # Mission FSM phase, e.g. SEARCH / DESCEND / TOUCHDOWN.
    mission_phase: str = ""
    marker_detected: bool = False
    marker_enu_m: tuple[float, float, float] | None = None

    # ------------------------------------------------------------------- entities
    def entity(self, name: str) -> Track:
        """Get or create the track for a named dynamic object."""
        track = self.entities.get(name)
        if track is None:
            track = Track(timeout_s=ENTITY_TIMEOUT_S)
            self.entities[name] = track
        return track

    # -------------------------------------------------------------------- queries
    def depth_is_fresh(self, now: float | None = None) -> bool:
        if self.depth_stamp_s < 0.0:
            return False
        return ((_now() if now is None else now) - self.depth_stamp_s) < DEPTH_TIMEOUT_S

    def set_depth(self, metres: float, now: float | None = None) -> None:
        self.depth_m = float(metres)
        self.depth_stamp_s = _now() if now is None else now

    def distance_to_goal_m(self) -> float | None:
        """Straight-line distance, not path length — the operator's sanity check."""
        if self.goal_enu_m is None or not self.drone.is_valid():
            return None
        return math.dist(self.drone.xy, self.goal_enu_m[:2])

    def eta_s(self) -> float | None:
        """Remaining time along the planned trajectory at the current speed.

        Falls back to straight-line distance when nothing is planned.  Returns
        None while stopped, because "infinity" is not useful on a panel.
        """
        speed = self.drone.speed_m_s
        if speed < 0.2:
            return None
        remaining = self._remaining_path_m() or self.distance_to_goal_m()
        if remaining is None:
            return None
        return remaining / speed

    def distance_to_entity_m(self, name: str) -> float | None:
        """Drone-to-entity xy distance — the pursuit/landing capture check."""
        track = self.entities.get(name)
        if track is None or not track.is_valid() or not self.drone.is_valid():
            return None
        return math.dist(self.drone.xy, track.xy)

    def _remaining_path_m(self) -> float | None:
        """Trajectory length from the nearest point ahead of the drone to its end."""
        points = self.trajectory.points or self.global_path.points
        if len(points) < 2 or not self.drone.is_valid():
            return None
        here = self.drone.xy
        nearest = min(
            range(len(points)), key=lambda i: math.dist(here, points[i][:2])
        )
        return sum(
            math.dist(a, b) for a, b in zip(points[nearest:], points[nearest + 1:])
        )
