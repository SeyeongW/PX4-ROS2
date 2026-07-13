"""Safety contract for PX4 multicopter L1-style AUTO.MISSION experiments.

This module intentionally does not plan a path.  It accepts the output of the
A*/SFC planner only after the planner has emitted a machine-checkable safety
certificate.  Separating planning from mission upload prevents a convenient
waypoint uploader from silently becoming a second, unchecked planner.

The name "L1" here means the multicopter L1-style crossing point implemented by
PX4 ``PositionSmoothing::_getL1Point``.  It is not the legacy fixed-wing L1
controller.  PX4 v1.17 fixed-wing guidance uses NPFG.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = "px4_l1_collision_checked_plan/v2"
PX4_MC_L1_MIN_LOOKAHEAD_M = 5.0
MINIMUM_SURFACE_CLEARANCE_M = 10.0
PX4_MC_TURN_COS_THRESHOLD = 0.98
MAX_SITL_MISSION_ITEMS = 10_000


class PlanValidationError(ValueError):
    """Raised when an experiment plan does not satisfy the upload contract."""


def _vec3(value: object, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise PlanValidationError(f"{field} must be a finite [x, y, z] vector")
    return array


def _positive(value: object, field: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    valid = math.isfinite(number) and (number >= 0.0 if allow_zero else number > 0.0)
    if not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise PlanValidationError(f"{field} must be a finite {qualifier} number")
    return number


@dataclass(frozen=True)
class AxisAlignedCorridor:
    """One obstacle-free axis-aligned box in Gazebo-world ENU metres."""

    minimum_enu_m: tuple[float, float, float]
    maximum_enu_m: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], index: int) -> "AxisAlignedCorridor":
        minimum = _vec3(value.get("min_enu_m"), f"corridor_boxes[{index}].min_enu_m")
        maximum = _vec3(value.get("max_enu_m"), f"corridor_boxes[{index}].max_enu_m")
        if np.any(maximum <= minimum):
            raise PlanValidationError(
                f"corridor_boxes[{index}] must have max strictly greater than min"
            )
        return cls(tuple(float(v) for v in minimum), tuple(float(v) for v in maximum))

    def contains(self, point_enu_m: Sequence[float], inset_m: float = 0.0) -> bool:
        point = _vec3(point_enu_m, "corridor point")
        inset = _positive(inset_m, "corridor inset", allow_zero=True)
        lower = np.asarray(self.minimum_enu_m) + inset
        upper = np.asarray(self.maximum_enu_m) - inset
        return bool(np.all(lower <= point) and np.all(point <= upper))

    def contains_with_horizontal_inset(
        self, point_enu_m: Sequence[float], horizontal_inset_m: float
    ) -> bool:
        """Check an XY turn envelope without inventing a vertical L1 radius.

        PX4's multicopter L1-style crossing point is an along-track look-ahead,
        not a five-metre spherical vehicle envelope.  Vertical safety is already
        certified by the 3-D occupancy/SFC and the configured surface clearance.
        """

        point = _vec3(point_enu_m, "corridor point")
        inset = _positive(
            horizontal_inset_m, "horizontal corridor inset", allow_zero=True
        )
        lower = np.asarray(self.minimum_enu_m, dtype=float)
        upper = np.asarray(self.maximum_enu_m, dtype=float)
        return bool(
            np.all(lower[:2] + inset <= point[:2])
            and np.all(point[:2] <= upper[:2] - inset)
            and lower[2] <= point[2] <= upper[2]
        )


@dataclass(frozen=True)
class L1TripletCertificate:
    """Certificate that one previous/current/next triplet fits a safe SFC box.

    ``center_waypoint_index`` addresses the current waypoint, therefore valid
    values are 1 through ``len(waypoints)-2``.  Requiring all three vertices to
    fit the same convex box also encloses their full triangle.  The box is inset
    by the conservative L1 turn envelope so tracking does not rely on an
    unchecked corner cut.
    """

    center_waypoint_index: int
    corridor_box_index: int
    is_turning_triplet: bool
    turn_envelope_radius_m: float
    validated_min_clearance_m: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], index: int) -> "L1TripletCertificate":
        try:
            center = int(value["center_waypoint_index"])
            box = int(value["corridor_box_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanValidationError(
                f"l1_triplet_certificates[{index}] needs integer waypoint and box indices"
            ) from exc
        turning = value.get("is_turning_triplet")
        if not isinstance(turning, bool):
            raise PlanValidationError(
                f"l1_triplet_certificates[{index}].is_turning_triplet must be boolean"
            )
        return cls(
            center_waypoint_index=center,
            corridor_box_index=box,
            is_turning_triplet=turning,
            turn_envelope_radius_m=_positive(
                value.get("turn_envelope_radius_m"),
                f"l1_triplet_certificates[{index}].turn_envelope_radius_m",
                allow_zero=True,
            ),
            validated_min_clearance_m=_positive(
                value.get("validated_min_clearance_m"),
                f"l1_triplet_certificates[{index}].validated_min_clearance_m",
            ),
        )


def canonical_plan_digest(mapping: Mapping[str, object]) -> str:
    """Return SHA-256 of a plan while excluding its self-referential digest."""

    canonical = dict(mapping)
    canonical.pop("plan_sha256", None)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def with_computed_digest(mapping: Mapping[str, object]) -> dict[str, object]:
    """Copy a plan mapping and add its canonical digest."""

    result = dict(mapping)
    result["plan_sha256"] = canonical_plan_digest(result)
    return result


@dataclass(frozen=True)
class CollisionCheckedL1Plan:
    """Validated immutable plan consumed by the MAVLink mission adapter."""

    map_revision: str
    occupancy_sha256: str
    planner_run_id: str
    world_home_enu_m: tuple[float, float, float]
    waypoints_enu_m: tuple[tuple[float, float, float], ...]
    corridors: tuple[AxisAlignedCorridor, ...]
    certificates: tuple[L1TripletCertificate, ...]
    hard_radius_m: float
    min_clearance_m: float
    collision_check_resolution_m: float
    minimum_surface_clearance_m: float
    plan_sha256: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "CollisionCheckedL1Plan":
        if mapping.get("schema_version") != SCHEMA_VERSION:
            raise PlanValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}; got {mapping.get('schema_version')!r}"
            )
        if mapping.get("coordinate_frame") != "gazebo_world_enu":
            raise PlanValidationError("coordinate_frame must be 'gazebo_world_enu'")

        proof = mapping.get("planner_proof")
        if not isinstance(proof, Mapping):
            raise PlanValidationError("planner_proof mapping is required")
        required_flags = (
            "astar_occupancy_checked",
            "sfc_checked",
            "continuous_segment_checked",
            "hard_radius_applied",
            "minimum_surface_clearance_applied",
        )
        missing_flags = [name for name in required_flags if proof.get(name) is not True]
        if missing_flags:
            raise PlanValidationError(
                "planner_proof must explicitly certify: " + ", ".join(missing_flags)
            )

        identifiers: dict[str, str] = {}
        for name in ("map_revision", "occupancy_sha256", "planner_run_id"):
            value = str(mapping.get(name, "")).strip()
            if not value:
                raise PlanValidationError(f"{name} is required")
            identifiers[name] = value
        occupancy_digest = identifiers["occupancy_sha256"].lower()
        if len(occupancy_digest) != 64 or any(
            c not in "0123456789abcdef" for c in occupancy_digest
        ):
            raise PlanValidationError("occupancy_sha256 must be 64 lowercase hex characters")

        raw_waypoints = mapping.get("waypoints_enu_m")
        if not isinstance(raw_waypoints, Sequence) or isinstance(raw_waypoints, (str, bytes)):
            raise PlanValidationError("waypoints_enu_m must be a sequence")
        waypoints = tuple(
            tuple(float(v) for v in _vec3(point, f"waypoints_enu_m[{index}]"))
            for index, point in enumerate(raw_waypoints)
        )
        if len(waypoints) < 3:
            raise PlanValidationError(
                "at least three waypoints are required to exercise triplet crossing"
            )
        for index in range(1, len(waypoints)):
            separation = np.linalg.norm(
                np.asarray(waypoints[index]) - np.asarray(waypoints[index - 1])
            )
            if separation < 1e-3:
                raise PlanValidationError(f"waypoints {index - 1} and {index} are duplicates")

        raw_corridors = mapping.get("corridor_boxes")
        if not isinstance(raw_corridors, Sequence) or not raw_corridors:
            raise PlanValidationError("at least one corridor_boxes entry is required")
        corridors = tuple(
            AxisAlignedCorridor.from_mapping(value, index)
            for index, value in enumerate(raw_corridors)
            if isinstance(value, Mapping)
        )
        if len(corridors) != len(raw_corridors):
            raise PlanValidationError("every corridor_boxes entry must be a mapping")

        raw_certificates = mapping.get("l1_triplet_certificates")
        if not isinstance(raw_certificates, Sequence):
            raise PlanValidationError("l1_triplet_certificates must be a sequence")
        certificates = tuple(
            L1TripletCertificate.from_mapping(value, index)
            for index, value in enumerate(raw_certificates)
            if isinstance(value, Mapping)
        )
        if len(certificates) != len(raw_certificates):
            raise PlanValidationError("every l1_triplet_certificates entry must be a mapping")

        hard_radius = _positive(mapping.get("hard_radius_m"), "hard_radius_m")
        min_clearance = _positive(mapping.get("min_clearance_m"), "min_clearance_m")
        check_resolution = _positive(
            mapping.get("collision_check_resolution_m"), "collision_check_resolution_m"
        )
        surface_clearance = _positive(
            mapping.get("minimum_surface_clearance_m"),
            "minimum_surface_clearance_m",
        )
        if surface_clearance + 1e-9 < MINIMUM_SURFACE_CLEARANCE_M:
            raise PlanValidationError(
                "minimum_surface_clearance_m must be at least 10 m"
            )
        if min_clearance + 1e-9 < hard_radius:
            raise PlanValidationError("min_clearance_m must be at least hard_radius_m")
        if check_resolution > hard_radius / 2.0:
            raise PlanValidationError(
                "collision_check_resolution_m must be no greater than half hard_radius_m"
            )

        expected_centers = set(range(1, len(waypoints) - 1))
        actual_centers = [certificate.center_waypoint_index for certificate in certificates]
        if set(actual_centers) != expected_centers or len(actual_centers) != len(expected_centers):
            raise PlanValidationError(
                "exactly one L1 certificate is required for every interior waypoint"
            )
        for certificate in certificates:
            center = certificate.center_waypoint_index
            if not 0 <= certificate.corridor_box_index < len(corridors):
                raise PlanValidationError(f"triplet {center} references an invalid corridor box")
            expected_turning = triplet_is_turning(
                waypoints[center - 1], waypoints[center], waypoints[center + 1]
            )
            if certificate.is_turning_triplet != expected_turning:
                raise PlanValidationError(
                    f"triplet {center} turning classification does not match its geometry"
                )
            if (
                certificate.is_turning_triplet
                and certificate.turn_envelope_radius_m + 1e-9
                < PX4_MC_L1_MIN_LOOKAHEAD_M
            ):
                raise PlanValidationError(
                    f"triplet {center} turn envelope must cover PX4's 5 m minimum L1 look-ahead"
                )
            if certificate.validated_min_clearance_m + 1e-9 < hard_radius:
                raise PlanValidationError(f"triplet {center} does not preserve hard clearance")
            box = corridors[certificate.corridor_box_index]
            for point_index in (center - 1, center, center + 1):
                if not box.contains_with_horizontal_inset(
                    waypoints[point_index], certificate.turn_envelope_radius_m
                ):
                    raise PlanValidationError(
                        f"triplet {center} waypoint {point_index} is outside its corridor "
                        "after applying the L1 turn envelope"
                    )

        for point_index, waypoint in enumerate(waypoints):
            if not any(box.contains(waypoint) for box in corridors):
                raise PlanValidationError(f"waypoint {point_index} is outside every SFC box")

        supplied_digest = str(mapping.get("plan_sha256", "")).lower()
        expected_digest = canonical_plan_digest(mapping)
        if supplied_digest != expected_digest:
            raise PlanValidationError(
                f"plan_sha256 mismatch: expected {expected_digest}, "
                f"got {supplied_digest or '<missing>'}"
            )

        return cls(
            map_revision=identifiers["map_revision"],
            occupancy_sha256=occupancy_digest,
            planner_run_id=identifiers["planner_run_id"],
            world_home_enu_m=tuple(
                float(v) for v in _vec3(mapping.get("world_home_enu_m"), "world_home_enu_m")
            ),
            waypoints_enu_m=waypoints,
            corridors=corridors,
            certificates=certificates,
            hard_radius_m=hard_radius,
            min_clearance_m=min_clearance,
            collision_check_resolution_m=check_resolution,
            minimum_surface_clearance_m=surface_clearance,
            plan_sha256=supplied_digest,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CollisionCheckedL1Plan":
        plan_path = Path(path)
        text = plan_path.read_text(encoding="utf-8")
        if plan_path.suffix.lower() == ".json":
            mapping = json.loads(text)
        else:
            mapping = yaml.safe_load(text)
        if not isinstance(mapping, Mapping):
            raise PlanValidationError("plan document root must be a mapping")
        return cls.from_mapping(mapping)

    def patrol_waypoints(self, cycles: int) -> tuple[tuple[float, float, float], ...]:
        """Return a finite out-and-back patrol without skipping checked segments."""

        if isinstance(cycles, bool) or int(cycles) != cycles or cycles < 1:
            raise PlanValidationError("patrol cycles must be a positive integer")
        forward = list(self.waypoints_enu_m)
        reverse_without_goal = list(reversed(forward[:-1]))
        one_cycle = forward + reverse_without_goal
        route: list[tuple[float, float, float]] = []
        for _ in range(int(cycles)):
            if route:
                route.extend(one_cycle[1:])
            else:
                route.extend(one_cycle)
        # The PX4 SITL board is built with CONFIG_NUM_MISSION_ITMES_SUPPORTED=10000.
        if len(route) > MAX_SITL_MISSION_ITEMS:
            raise PlanValidationError(
                f"expanded patrol has {len(route)} points; "
                f"PX4 SITL limit is {MAX_SITL_MISSION_ITEMS}"
            )
        return tuple(route)


def certify_triplets_in_single_corridors(
    waypoints_enu_m: Sequence[Sequence[float]],
    corridor_boxes: Sequence[AxisAlignedCorridor],
    *,
    turn_envelope_radius_m: float = PX4_MC_L1_MIN_LOOKAHEAD_M,
    validated_min_clearance_m: float,
) -> tuple[L1TripletCertificate, ...]:
    """Create strict convex-box certificates or fail without weakening safety.

    This helper is for the A*/SFC export stage.  It never invents a corridor and
    never snaps a point.  If no single inset convex box encloses a triplet, the
    upstream planner must add a safe waypoint/corridor or select B-spline mode.
    """

    points = tuple(
        _vec3(point, f"waypoint[{index}]")
        for index, point in enumerate(waypoints_enu_m)
    )
    if len(points) < 3:
        raise PlanValidationError("at least three waypoints are required")
    envelope = _positive(turn_envelope_radius_m, "turn_envelope_radius_m")
    if envelope + 1e-9 < PX4_MC_L1_MIN_LOOKAHEAD_M:
        raise PlanValidationError("turn envelope cannot be less than PX4's 5 m L1 minimum")
    clearance = _positive(validated_min_clearance_m, "validated_min_clearance_m")
    certificates: list[L1TripletCertificate] = []
    for center in range(1, len(points) - 1):
        turning = triplet_is_turning(
            points[center - 1], points[center], points[center + 1]
        )
        applied_envelope = envelope if turning else 0.0
        selected: int | None = None
        for box_index, box in enumerate(corridor_boxes):
            if all(
                box.contains_with_horizontal_inset(points[index], applied_envelope)
                for index in (center - 1, center, center + 1)
            ):
                selected = box_index
                break
        if selected is None:
            raise PlanValidationError(
                f"triplet centered at waypoint {center} lacks a convex SFC box "
                f"with {applied_envelope:g} m inset"
            )
        certificates.append(
            L1TripletCertificate(
                center,
                selected,
                turning,
                applied_envelope,
                clearance,
            )
        )
    return tuple(certificates)


def triplet_is_turning(
    previous_enu_m: Sequence[float],
    current_enu_m: Sequence[float],
    following_enu_m: Sequence[float],
) -> bool:
    """Conservatively classify a waypoint direction change using PX4's 0.98 gate.

    Firmware ``_isTurning`` compares live velocity to the current target rather
    than just waypoint geometry.  A geometric heading change with cosine below
    0.98 is therefore treated as turning; a collinear triplet does not turn and
    does not receive a fictitious 5 m obstacle inflation.
    """

    previous = _vec3(previous_enu_m, "previous waypoint")
    current = _vec3(current_enu_m, "current waypoint")
    following = _vec3(following_enu_m, "following waypoint")
    incoming = current[:2] - previous[:2]
    outgoing = following[:2] - current[:2]
    incoming_norm = float(np.linalg.norm(incoming))
    outgoing_norm = float(np.linalg.norm(outgoing))
    if incoming_norm < 1e-6 or outgoing_norm < 1e-6:
        return True
    cosine = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
    return cosine < PX4_MC_TURN_COS_THRESHOLD
