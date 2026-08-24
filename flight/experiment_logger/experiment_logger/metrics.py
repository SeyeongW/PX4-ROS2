"""Pure metric accumulation shared by the ROS node and unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

import yaml


SCHEMA_VERSION = "jo_experiment_logger_v2"
REQUIRED_METRICS = (
    "path_length_m",
    "tracking_error_mean_m",
    "tracking_error_max_m",
    "tracking_error_rmse_m",
    "min_clearance_m",
    "astar_plan_time_ms",
    "mpc_solve_time_ms",
    "replan_count",
    "aruco_detection_rate_pct",
    "relative_xy_error_m",
    "landing_xy_error_m",
    "touchdown_relative_speed_m_s",
    "sfc_generation_time_ms",
    "sfc_min_width_m",
    "sfc_avg_width_m",
    "sfc_corridor_count",
    "sfc_violation_count",
)

FLIGHT_STATES = frozenset({
    "TAKEOFF", "MISSION_PLAN", "MISSION", "HOVER", "RETURN_PLAN",
    "RETURN", "LANDING_ACQUIRE", "LANDING_DESCEND", "PRECLAND",
})
TRACKING_STATES = frozenset({"MISSION", "RETURN"})
RELATIVE_STATES = frozenset({
    "RETURN", "LANDING_ACQUIRE", "LANDING_DESCEND", "PRECLAND",
})
LANDING_STATES = frozenset({
    "LANDING_ACQUIRE", "LANDING_DESCEND", "PRECLAND",
})
SFC_TRACKING_STATES = frozenset({"MISSION", "RETURN_PLAN", "RETURN"})

_PLAN_LATENCY = re.compile(
    r"global A\*/B-spline:\s+\d+\s+samples,.*?"
    r"\d+\s+A\* expansions,\s*([0-9.eE+-]+)\s+s(?:,|$)")
_EXPERIMENT_METRICS = re.compile(r"EXPERIMENT_METRICS\s+(.+)$")
_SFC_GENERATION = re.compile(r"\bSFC\s+([0-9.eE+-]+)\s+ms(?:,|$)")


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def _maximum(values: Iterable[float]) -> float:
    values = list(values)
    return max(values) if values else math.nan


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.total_squared += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.nan

    @property
    def rmse(self) -> float:
        return (math.sqrt(self.total_squared / self.count)
                if self.count else math.nan)

    @property
    def max(self) -> float:
        return self.maximum if self.count else math.nan

    @property
    def min(self) -> float:
        return self.minimum if self.count else math.nan


@dataclass(frozen=True)
class Obstacle:
    name: str
    low_x: float
    low_y: float
    high_x: float
    high_y: float


@dataclass(frozen=True)
class ClearanceMap:
    """The same YAML obstacle geometry expressed in its mission map frame."""

    path: Path
    frame_id: str
    cos_heading: float
    sin_heading: float
    spawn_x: float
    spawn_y: float
    origin_x: float
    origin_y: float
    required_clearance_m: float
    planning_clearance_m: float
    obstacles: tuple[Obstacle, ...]
    sha256: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClearanceMap":
        path = Path(path).expanduser().resolve()
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        mission = document["mission"]
        frame = document["frames"][mission["coordinate_frame"]]
        heading = math.radians(float(frame.get("heading_deg_enu", 0.0)))
        origin = frame.get("origin_enu_m", [0.0, 0.0, 0.0])
        spawn = document["spawn"]["gazebo_spawn_pose_enu"]
        clearance = float(mission.get(
            "vehicle_clearance_xy_m",
            mission.get("obstacle_clearance_m", 0.0)))
        planning_clearance = clearance + float(
            mission.get("bspline_clearance_margin_m", 0.0))
        obstacles = []
        if "obstacles" in mission:
            for item in mission["obstacles"]:
                center = item["center_m"]
                size = item["size_m"]
                obstacles.append(Obstacle(
                    str(item["name"]),
                    float(center[0]) - 0.5 * float(size[0]),
                    float(center[1]) - 0.5 * float(size[1]),
                    float(center[0]) + 0.5 * float(size[0]),
                    float(center[1]) + 0.5 * float(size[1])))
        elif mission.get("obstacle_source") == "city_buildings":
            for item in document["obstacles"]["buildings"]:
                bounds = item["aabb_xy_m"]
                obstacles.append(Obstacle(
                    str(item["id"]),
                    float(bounds["min"][0]), float(bounds["min"][1]),
                    float(bounds["max"][0]), float(bounds["max"][1])))
        else:
            raise ValueError("unsupported mission obstacle source")
        if not obstacles or not _finite(
                heading, origin[0], origin[1], spawn["x"], spawn["y"],
                clearance, planning_clearance) or clearance < 0.0 \
                or planning_clearance < clearance:
            raise ValueError("invalid mission map clearance contract")
        return cls(
            path=path,
            frame_id=str(mission["coordinate_frame"]),
            cos_heading=math.cos(heading),
            sin_heading=math.sin(heading),
            spawn_x=float(spawn["x"]), spawn_y=float(spawn["y"]),
            origin_x=float(origin[0]), origin_y=float(origin[1]),
            required_clearance_m=clearance,
            planning_clearance_m=planning_clearance,
            obstacles=tuple(obstacles),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def local_to_map(self, x: float, y: float) -> tuple[float, float]:
        """PX4-local ENU to the YAML mission frame (row-vector contract)."""
        x = float(x) + self.spawn_x - self.origin_x
        y = float(y) + self.spawn_y - self.origin_y
        return (x * self.cos_heading + y * self.sin_heading,
                -x * self.sin_heading + y * self.cos_heading)

    def clearance(self, local_x: float, local_y: float):
        """Physical horizontal distance to the nearest immutable AABB."""
        x, y = self.local_to_map(local_x, local_y)
        best_distance = math.inf
        best_name = ""
        for obstacle in self.obstacles:
            dx = max(obstacle.low_x - x, 0.0, x - obstacle.high_x)
            dy = max(obstacle.low_y - y, 0.0, y - obstacle.high_y)
            distance = math.hypot(dx, dy)
            if distance < best_distance:
                best_distance = distance
                best_name = obstacle.name
        return (best_distance,
                best_distance - self.required_clearance_m,
                best_name)


def parse_experiment_metrics(message: str) -> dict[str, float]:
    match = _EXPERIMENT_METRICS.search(str(message))
    if match is None:
        return {}
    values = {}
    for token in match.group(1).split():
        key, separator, raw = token.partition("=")
        if not separator:
            continue
        try:
            values[key] = float(raw)
        except ValueError:
            continue
    return values


def parse_plan_latency_ms(message: str) -> float | None:
    match = _PLAN_LATENCY.search(str(message))
    if match is None:
        return None
    seconds = float(match.group(1))
    return 1000.0 * seconds if math.isfinite(seconds) and seconds >= 0 else None


def parse_sfc_generation_ms(message: str) -> float | None:
    match = _SFC_GENERATION.search(str(message))
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) and value >= 0.0 else None


@dataclass
class ExperimentAccumulator:
    clearance_map: ClearanceMap | None = None
    state: str = "UNKNOWN"
    path_length_m: float = 0.0
    path_length_xy_m: float = 0.0
    last_path_position: tuple[float, float, float] | None = None
    latest_setpoint: tuple[float, float, float] | None = None
    latest_setpoint_time_s: float | None = None
    latest_cue: tuple[float, float, float] | None = None
    tracking: RunningStats = field(default_factory=RunningStats)
    relative: RunningStats = field(default_factory=RunningStats)
    min_clearance_m: float = math.inf
    min_clearance_residual_m: float = math.inf
    closest_obstacle: str = ""
    active_plan_seq: int = 0
    seen_plan_sequences: set[int] = field(default_factory=set)
    return_plan_commit_count: int = 0
    plan_latencies_ms: list[float] = field(default_factory=list)
    sfc_generation_times_ms: list[float] = field(default_factory=list)
    sfc_widths: RunningStats = field(default_factory=RunningStats)
    sfc_corridor_counts: RunningStats = field(default_factory=RunningStats)
    sfc_latest_corridor_count: int | None = None
    active_sfc_boxes: tuple[tuple[float, ...], ...] = ()
    sfc_evaluated_samples: int = 0
    sfc_violation_samples: int = 0
    sfc_violation_events: int = 0
    sfc_last_violation: bool = False
    aruco_hits: int = 0
    aruco_frames: int = 0
    latest_aruco_detected: int | None = None
    official: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    first_ros_time_s: float | None = None
    last_ros_time_s: float | None = None
    last_tracking_error_m: float = math.nan
    last_clearance_m: float = math.nan
    last_clearance_residual_m: float = math.nan
    last_relative_xy_error_m: float = math.nan

    def set_state(self, state: str) -> None:
        state = str(state).strip() or "UNKNOWN"
        if state not in FLIGHT_STATES:
            self.last_path_position = None
        self.state = state

    def set_setpoint(self, x: float, y: float, z: float, now_s: float) -> None:
        if _finite(x, y, z, now_s):
            self.latest_setpoint = (float(x), float(y), float(z))
            self.latest_setpoint_time_s = float(now_s)

    def set_cue(self, x: float, y: float, z: float) -> None:
        if _finite(x, y, z):
            self.latest_cue = (float(x), float(y), float(z))

    def set_plan_sequence(self, sequence: int) -> None:
        sequence = int(sequence)
        if sequence in self.seen_plan_sequences:
            return
        self.seen_plan_sequences.add(sequence)
        if self.state in {"RETURN_PLAN", "RETURN"}:
            self.return_plan_commit_count += 1
        if sequence > self.active_plan_seq:
            self.active_plan_seq = sequence

    def add_sfc_snapshot(self, sequence: int, boxes) -> bool:
        sequence = int(sequence)
        if sequence in self.seen_plan_sequences:
            return False
        normalized = []
        for low, high in boxes:
            low = tuple(float(value) for value in low)
            high = tuple(float(value) for value in high)
            if (len(low) != 3 or len(high) != 3
                    or not _finite(*low, *high)
                    or any(upper <= lower
                           for lower, upper in zip(low, high))):
                return False
            normalized.append((*low, *high))
        if not normalized:
            return False
        self.set_plan_sequence(sequence)
        widths = [min(box[3] - box[0], box[4] - box[1])
                  for box in normalized]
        for width in widths:
            self.sfc_widths.add(width)
        self.sfc_corridor_counts.add(len(normalized))
        self.sfc_latest_corridor_count = len(normalized)
        self.active_sfc_boxes = tuple(normalized)
        self.sfc_last_violation = False
        return True

    def clear_active_sfc(self) -> None:
        self.active_sfc_boxes = ()
        self.sfc_last_violation = False

    @property
    def replan_count(self) -> int:
        """Accepted RETURN replacements after its first committed route."""
        return max(0, self.return_plan_commit_count - 1)

    def add_aruco(self, detected: bool) -> None:
        self.latest_aruco_detected = int(bool(detected))
        if self.state in LANDING_STATES:
            self.aruco_frames += 1
            self.aruco_hits += int(bool(detected))

    def add_log(self, message: str) -> bool:
        changed = False
        latency = parse_plan_latency_ms(message)
        if latency is not None:
            self.plan_latencies_ms.append(latency)
            changed = True
        sfc_generation = parse_sfc_generation_ms(message)
        if sfc_generation is not None:
            self.sfc_generation_times_ms.append(sfc_generation)
            changed = True
        values = parse_experiment_metrics(message)
        if values:
            self.official.update(values)
            changed = True
        return changed

    def add_position(self, x: float, y: float, z: float, now_s: float,
                     setpoint_max_age_s: float = 0.5) -> dict[str, float]:
        if not _finite(x, y, z, now_s):
            raise ValueError("vehicle position and time must be finite")
        position = (float(x), float(y), float(z))
        now_s = float(now_s)
        if self.first_ros_time_s is None:
            self.first_ros_time_s = now_s
        self.last_ros_time_s = now_s
        self.sample_count += 1

        if self.state in FLIGHT_STATES:
            if self.last_path_position is not None:
                dx = position[0] - self.last_path_position[0]
                dy = position[1] - self.last_path_position[1]
                dz = position[2] - self.last_path_position[2]
                self.path_length_xy_m += math.hypot(dx, dy)
                self.path_length_m += math.sqrt(dx * dx + dy * dy + dz * dz)
            self.last_path_position = position

        tracking_error = math.nan
        if (self.state in TRACKING_STATES
                and self.latest_setpoint is not None
                and self.latest_setpoint_time_s is not None
                and 0.0 <= now_s - self.latest_setpoint_time_s
                <= float(setpoint_max_age_s)):
            tracking_error = math.hypot(
                position[0] - self.latest_setpoint[0],
                position[1] - self.latest_setpoint[1])
            self.tracking.add(tracking_error)

        clearance = clearance_residual = math.nan
        if self.state in FLIGHT_STATES and self.clearance_map is not None:
            clearance, clearance_residual, obstacle = (
                self.clearance_map.clearance(position[0], position[1]))
            if clearance < self.min_clearance_m:
                self.min_clearance_m = clearance
                self.min_clearance_residual_m = clearance_residual
                self.closest_obstacle = obstacle

        relative_error = math.nan
        if self.latest_cue is not None:
            relative_error = math.hypot(
                position[0] - self.latest_cue[0],
                position[1] - self.latest_cue[1])
            if self.state in RELATIVE_STATES:
                self.relative.add(relative_error)

        sfc_violation = math.nan
        if (self.state in SFC_TRACKING_STATES
                and self.active_sfc_boxes and self.clearance_map is not None):
            map_x, map_y = self.clearance_map.local_to_map(
                position[0], position[1])
            tolerance = 1.0e-6
            inside = any(
                box[0] - tolerance <= map_x <= box[3] + tolerance
                and box[1] - tolerance <= map_y <= box[4] + tolerance
                for box in self.active_sfc_boxes)
            violation = not inside
            sfc_violation = float(violation)
            self.sfc_evaluated_samples += 1
            self.sfc_violation_samples += int(violation)
            if violation and not self.sfc_last_violation:
                self.sfc_violation_events += 1
            self.sfc_last_violation = violation
        elif self.state not in SFC_TRACKING_STATES:
            self.sfc_last_violation = False

        self.last_tracking_error_m = tracking_error
        self.last_clearance_m = clearance
        self.last_clearance_residual_m = clearance_residual
        self.last_relative_xy_error_m = relative_error
        return {
            "vehicle_x_enu_m": position[0],
            "vehicle_y_enu_m": position[1],
            "vehicle_z_enu_m": position[2],
            "tracking_error_m": tracking_error,
            "clearance_m": clearance,
            "clearance_residual_m": clearance_residual,
            "relative_xy_error_m": relative_error,
            "sfc_violation": sfc_violation,
        }

    def summary(self) -> dict[str, object]:
        official = self.official
        marker_frames = int(official.get("marker_frames", self.aruco_frames))
        marker_hits = int(official.get("marker_hits", self.aruco_hits))
        aruco_rate = (100.0 * marker_hits / marker_frames
                      if marker_frames else math.nan)
        mpc_count = int(official.get("mpc_count", 0))
        mpc_total_ms = float(official.get("mpc_total_ms", math.nan))
        mpc_mean_ms = (mpc_total_ms / mpc_count
                       if mpc_count and math.isfinite(mpc_total_ms)
                       else math.nan)
        astar_mean_ms = _mean(self.plan_latencies_ms)
        min_clearance = (self.min_clearance_m
                         if math.isfinite(self.min_clearance_m) else math.nan)
        min_residual = (
            self.min_clearance_residual_m
            if math.isfinite(self.min_clearance_residual_m) else math.nan)
        landing_xy = float(official.get("landing_xy_error_m", math.nan))
        touchdown_speed = float(official.get(
            "touchdown_relative_speed_3d_m_s", math.nan))
        sfc_generation_mean = _mean(self.sfc_generation_times_ms)
        sfc_corridor_count = (
            self.sfc_latest_corridor_count
            if self.sfc_latest_corridor_count is not None else math.nan)
        sfc_violation_count = (
            self.sfc_violation_events
            if self.sfc_evaluated_samples else math.nan)
        sfc_violation_rate = (
            100.0 * self.sfc_violation_samples / self.sfc_evaluated_samples
            if self.sfc_evaluated_samples else math.nan)
        required = {
            "path_length_m": self.path_length_m,
            "tracking_error_mean_m": self.tracking.mean,
            "tracking_error_max_m": self.tracking.max,
            "tracking_error_rmse_m": self.tracking.rmse,
            "min_clearance_m": min_clearance,
            # Existing interface exposes accepted A*+B-spline pipeline age,
            # not an isolated A* timer. The scope field below makes this clear.
            "astar_plan_time_ms": astar_mean_ms,
            "mpc_solve_time_ms": mpc_mean_ms,
            "replan_count": self.replan_count,
            "aruco_detection_rate_pct": aruco_rate,
            "relative_xy_error_m": self.relative.mean,
            "landing_xy_error_m": landing_xy,
            "touchdown_relative_speed_m_s": touchdown_speed,
            "sfc_generation_time_ms": sfc_generation_mean,
            "sfc_min_width_m": self.sfc_widths.min,
            "sfc_avg_width_m": self.sfc_widths.mean,
            "sfc_corridor_count": sfc_corridor_count,
            "sfc_violation_count": sfc_violation_count,
        }
        missing = [name for name, value in required.items()
                   if isinstance(value, float) and not math.isfinite(value)]
        return {
            **required,
            "path_length_xy_m": self.path_length_xy_m,
            "active_plan_seq": self.active_plan_seq,
            "accepted_plan_count": len(self.seen_plan_sequences),
            "return_plan_commit_count": self.return_plan_commit_count,
            "tracking_error_samples": self.tracking.count,
            "tracking_error_scope": "xy_setpoint_error_mission_return",
            "min_clearance_residual_m": min_residual,
            "closest_obstacle": self.closest_obstacle,
            "min_clearance_scope": "physical_aabb_xy_distance",
            "astar_plan_count": len(self.plan_latencies_ms),
            "astar_plan_time_max_ms": _maximum(self.plan_latencies_ms),
            "astar_plan_time_scope": "accepted_astar_bspline_pipeline_elapsed",
            "astar_isolated_timer_available": 0,
            "global_plan_pipeline_latency_ms": astar_mean_ms,
            "mpc_solve_count": mpc_count,
            "mpc_solve_time_max_ms": float(
                official.get("mpc_max_ms", math.nan)),
            "mpc_solve_time_scope": "tracking_and_landing_mpc_wall_mean",
            "aruco_hits": marker_hits,
            "aruco_frames": marker_frames,
            "aruco_metric_scope": "precision_landing_until_touchdown",
            "relative_xy_error_max_m": self.relative.max,
            "relative_xy_error_samples": self.relative.count,
            "relative_xy_error_scope": "mean_return_and_landing",
            "sfc_generation_time_count": len(self.sfc_generation_times_ms),
            "sfc_generation_time_max_ms": _maximum(
                self.sfc_generation_times_ms),
            "sfc_generation_time_scope": (
                "accepted_active_path_cover_polyline_wall"),
            "sfc_plan_count": self.sfc_corridor_counts.count,
            "sfc_corridor_count_mean": self.sfc_corridor_counts.mean,
            "sfc_corridor_count_max": self.sfc_corridor_counts.max,
            "sfc_evaluated_samples": self.sfc_evaluated_samples,
            "sfc_violation_sample_count": self.sfc_violation_samples,
            "sfc_violation_rate_pct": sfc_violation_rate,
            "sfc_violation_event_count": self.sfc_violation_events,
            "sfc_violation_scope": (
                "contiguous_vehicle_excursions_outside_active_xy_box_union"),
            "sfc_planning_clearance_m": (
                self.clearance_map.planning_clearance_m
                if self.clearance_map else math.nan),
            "landing_error_3d_m": float(
                official.get("landing_error_3d_m", math.nan)),
            "touchdown_relative_speed_3d_m_s": touchdown_speed,
            "touchdown_relative_vertical_speed_m_s": float(official.get(
                "touchdown_relative_vertical_speed_m_s", math.nan)),
            "precland_attempts": int(official.get("precland_attempts", 0)),
            "precland_recoveries": int(official.get("precland_recoveries", 0)),
            "landing_descend_recoveries": int(official.get(
                "landing_descend_recoveries", 0)),
            # Compatibility names used by the existing ULog exporter.
            "path_tracking_rmse_m": self.tracking.rmse,
            "path_tracking_error_max_m": self.tracking.max,
            "marker_detection_rate_pct": aruco_rate,
            "mpc_solve_mean_ms": mpc_mean_ms,
            "mpc_solve_max_ms": float(official.get("mpc_max_ms", math.nan)),
            "required_metrics_complete": int(not missing),
            "missing_metrics": "|".join(missing),
        }
