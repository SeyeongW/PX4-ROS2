"""Export a common 3-D A*/SFC result as a signed PX4 L1 experiment plan."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence
import uuid

import numpy as np
import yaml

from .planner_3d import (
    HierarchicalAStarPlanner3D,
    HierarchicalAStar3DConfig,
    PrismOccupancy3D,
    VehicleEnvelope3D,
    load_preferred_city_map,
    plan_validated_waypoints_3d,
)
from .px4_l1_plan import (
    AxisAlignedCorridor,
    MINIMUM_SURFACE_CLEARANCE_M,
    PX4_MC_L1_MIN_LOOKAHEAD_M,
    PlanValidationError,
    SCHEMA_VERSION,
    certify_triplets_in_single_corridors,
    with_computed_digest,
)


class L1ExportError(RuntimeError):
    """Raised when A*/SFC planning cannot prove a safe native-PX4 route."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _occupancy_digest(
    map_path: Path,
    envelope: VehicleEnvelope3D,
    *,
    minimum_z_m: float,
    maximum_z_m: float,
    ground_z_m: float,
    minimum_surface_clearance_m: float,
) -> str:
    payload = {
        "map_sha256": _sha256_file(map_path),
        "vehicle_envelope": asdict(envelope),
        "minimum_z_m": minimum_z_m,
        "maximum_z_m": maximum_z_m,
        "ground_z_m": ground_z_m,
        "minimum_surface_clearance_m": minimum_surface_clearance_m,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _map_name(path: Path) -> str:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, Mapping):
        raise L1ExportError("coordinate map root must be a mapping")
    name = str(document.get("map", {}).get("name", path.stem))
    return name.strip() or path.stem


def export_common_astar_sfc_plan(
    *,
    repository_root: str | Path,
    start_enu_m: Sequence[float],
    goal_enu_m: Sequence[float],
    world_home_enu_m: Sequence[float] | None = None,
    map_path: str | Path | None = None,
    planner_run_id: str | None = None,
    minimum_z_m: float = 0.0,
    maximum_z_m: float = 120.0,
    ground_z_m: float = 0.0,
    minimum_surface_clearance_m: float = 10.0,
    collision_check_resolution_m: float = 0.25,
    l1_lookahead_m: float = PX4_MC_L1_MIN_LOOKAHEAD_M,
    sfc_deadline_s: float = 20.0,
) -> dict[str, object]:
    """Run the common planner and return a validated, digest-signed manifest.

    The generated SFC uses the normal decomposed 1 m-class UAV hard envelope.
    The additional 5 m value is kept as an L1 triplet certificate inset, not
    silently substituted for ``hard_horizontal_radius_m`` in A* occupancy.
    This deliberately strict A/B gate can reject the L1 branch while leaving
    the B-spline branch and its ordinary corridor usable.
    """

    if minimum_surface_clearance_m + 1e-9 < MINIMUM_SURFACE_CLEARANCE_M:
        raise L1ExportError(
            "native PX4 L1 experiment requires at least 10 m surface clearance"
        )
    root = Path(repository_root).expanduser().resolve()
    if map_path is None:
        city_map = load_preferred_city_map(root, prefer_uav_map=True)
    else:
        from .planner_core import CityMap

        city_map = CityMap.from_yaml(map_path)
    envelope = VehicleEnvelope3D.from_city_map_defaults(city_map)
    occupancy = PrismOccupancy3D.from_vehicle_envelope(
        city_map,
        envelope,
        minimum_z_m=minimum_z_m,
        maximum_z_m=maximum_z_m,
        ground_z_m=ground_z_m,
        minimum_surface_clearance_m=minimum_surface_clearance_m,
    )
    base_config = HierarchicalAStar3DConfig()
    planner_config = replace(
        base_config,
        coarse=replace(
            base_config.coarse,
            safe_clearance_m=l1_lookahead_m,
            preferred_clearance_m=l1_lookahead_m,
            clearance_weight=4.0,
        ),
        fine=replace(
            base_config.fine,
            safe_clearance_m=l1_lookahead_m,
            preferred_clearance_m=l1_lookahead_m,
            clearance_weight=4.0,
        ),
    )
    planner = HierarchicalAStarPlanner3D(occupancy, planner_config)
    result = plan_validated_waypoints_3d(
        planner,
        start_enu_m,
        goal_enu_m,
        minimum_clearance_m=0.0,
        validation_step_m=collision_check_resolution_m,
        # Keep normal common-corridor padding here.  Boxes may grow to 10 m in
        # free space, and the separate certificate below insists on a complete
        # 5 m inset.  This avoids turning PX4's 5 m along-track look-ahead into
        # a global 5 m hard obstacle inflation.
        sfc_minimum_overlap_m=1.0,
        sfc_initial_margin_m=0.5,
        sfc_maximum_margin_m=max(10.0, l1_lookahead_m + 5.0),
        sfc_expansion_step_m=0.5,
        sfc_maximum_segment_length_m=4.0,
        sfc_maximum_subdivision_depth=16,
        sfc_maximum_expansion_iterations=50_000,
        sfc_deadline_s=sfc_deadline_s,
    )
    if not result.success or result.corridor is None:
        raise L1ExportError(
            "COMMON_ASTAR_SFC_FAILED: " + result.metrics.reason
        )

    # Use the exact dense SFC centerline rather than adding an unchecked
    # simplification between boxes.  This is still the same A*/SFC solution
    # that feeds the B-spline optimizer.
    waypoints = np.asarray(result.corridor.centerline_enu_m, dtype=float)
    if len(waypoints) < 3:
        raise L1ExportError(
            "L1_EXPERIMENT_NEEDS_THREE_WAYPOINTS: select a route with a real triplet"
        )
    corridors = tuple(
        AxisAlignedCorridor(box.minimum_enu_m, box.maximum_enu_m)
        for box in result.corridor.boxes
    )
    try:
        certificates = certify_triplets_in_single_corridors(
            waypoints,
            corridors,
            turn_envelope_radius_m=l1_lookahead_m,
            validated_min_clearance_m=envelope.hard_horizontal_radius_m,
        )
    except PlanValidationError as exc:
        raise L1ExportError(
            "L1_SFC_CERTIFICATE_INFEASIBLE: B-spline remains available; " + str(exc)
        ) from exc

    raw_minimum_clearance = min(
        occupancy.segment_minimum_clearance(
            left,
            right,
            inflated=False,
            sample_step_m=collision_check_resolution_m,
        )
        for left, right in zip(waypoints, waypoints[1:])
    )
    if raw_minimum_clearance + 1e-9 < envelope.hard_horizontal_radius_m:
        raise L1ExportError("raw route clearance is below the hard vehicle envelope")

    source_path = city_map.source_path.resolve()
    map_digest = _sha256_file(source_path)
    map_revision = f"{_map_name(source_path)}:{map_digest[:16]}"
    try:
        map_source = str(source_path.relative_to(root))
    except ValueError:
        map_source = str(source_path)
    anchor = (
        np.asarray(city_map.px4_origin_enu_m, dtype=float)
        if world_home_enu_m is None
        else np.asarray(world_home_enu_m, dtype=float)
    )
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise L1ExportError("world_home_enu_m must contain three finite values")

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_frame": "gazebo_world_enu",
        "map_revision": map_revision,
        "map_source": map_source,
        "map_sha256": map_digest,
        "occupancy_sha256": _occupancy_digest(
            source_path,
            envelope,
            minimum_z_m=minimum_z_m,
            maximum_z_m=maximum_z_m,
            ground_z_m=ground_z_m,
            minimum_surface_clearance_m=minimum_surface_clearance_m,
        ),
        "planner_run_id": planner_run_id or f"astar-sfc-{uuid.uuid4()}",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "world_home_enu_m": [float(value) for value in anchor],
        "hard_radius_m": envelope.hard_horizontal_radius_m,
        "min_clearance_m": max(
            envelope.hard_horizontal_radius_m, raw_minimum_clearance
        ),
        "collision_check_resolution_m": collision_check_resolution_m,
        "vehicle_envelope": asdict(envelope),
        "planner_proof": {
            "astar_occupancy_checked": True,
            "sfc_checked": True,
            "continuous_segment_checked": True,
            "hard_radius_applied": True,
            "minimum_surface_clearance_applied": True,
        },
        "minimum_surface_clearance_m": float(minimum_surface_clearance_m),
        "guidance_contract": {
            "mode": "px4_mc_l1_style_auto_mission",
            "firmware_lookahead_minimum_m": PX4_MC_L1_MIN_LOOKAHEAD_M,
            "certified_corridor_inset_m": l1_lookahead_m,
            "trajectory_setpoint_publisher": False,
        },
        "waypoints_enu_m": waypoints.tolist(),
        "corridor_boxes": [
            {
                "min_enu_m": [float(value) for value in corridor.minimum_enu_m],
                "max_enu_m": [float(value) for value in corridor.maximum_enu_m],
            }
            for corridor in corridors
        ],
        "l1_triplet_certificates": [
            {
                "center_waypoint_index": certificate.center_waypoint_index,
                "corridor_box_index": certificate.corridor_box_index,
                "is_turning_triplet": certificate.is_turning_triplet,
                "turn_envelope_radius_m": certificate.turn_envelope_radius_m,
                "validated_min_clearance_m": certificate.validated_min_clearance_m,
            }
            for certificate in certificates
        ],
        "planner_metrics": asdict(result.metrics),
    }
    signed = with_computed_digest(manifest)
    # Round-trip through the consumer before allowing a file to be emitted.
    from .px4_l1_plan import CollisionCheckedL1Plan

    CollisionCheckedL1Plan.from_mapping(signed)
    return signed


def write_plan(path: str | Path, manifest: Mapping[str, object]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="common 3-D A*/SFC to PX4 multicopter L1 experiment exporter"
    )
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument("--map")
    parser.add_argument("--start", nargs=3, type=float, default=(-120.0, 115.0, 11.5))
    parser.add_argument("--goal", nargs=3, type=float, default=(200.0, -128.0, 11.5))
    parser.add_argument("--world-home", nargs=3, type=float)
    parser.add_argument("--planner-run-id")
    parser.add_argument("--maximum-z-m", type=float, default=120.0)
    parser.add_argument("--surface-clearance-m", type=float, default=10.0)
    parser.add_argument("--collision-check-resolution-m", type=float, default=0.25)
    parser.add_argument("--l1-lookahead-m", type=float, default=5.0)
    parser.add_argument("--sfc-deadline-s", type=float, default=20.0)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = export_common_astar_sfc_plan(
            repository_root=args.repository_root,
            map_path=args.map,
            start_enu_m=args.start,
            goal_enu_m=args.goal,
            world_home_enu_m=args.world_home,
            planner_run_id=args.planner_run_id,
            maximum_z_m=args.maximum_z_m,
            minimum_surface_clearance_m=args.surface_clearance_m,
            collision_check_resolution_m=args.collision_check_resolution_m,
            l1_lookahead_m=args.l1_lookahead_m,
            sfc_deadline_s=args.sfc_deadline_s,
        )
        output = write_plan(args.output, manifest)
        print(
            f"EXPORTED {output} plan={manifest['plan_sha256']} "
            f"waypoints={len(manifest['waypoints_enu_m'])}"
        )
        return 0
    except (L1ExportError, PlanValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
