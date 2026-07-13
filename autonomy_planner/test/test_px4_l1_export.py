from types import SimpleNamespace

import pytest
import yaml

from autonomy_planner.planner_3d import CorridorBox3D, WaypointPlanMetrics3D
from autonomy_planner.px4_l1_export import (
    L1ExportError,
    export_common_astar_sfc_plan,
    write_plan,
)
from autonomy_planner.px4_l1_plan import CollisionCheckedL1Plan


def coordinate_map(tmp_path):
    path = tmp_path / "synthetic_city.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "map": {
                    "name": "synthetic_city",
                    "gazebo_world_name": "synthetic_city",
                    "bounds_enu_m": {"x": [-50.0, 50.0], "y": [-50.0, 50.0]},
                },
                "frames": {"px4_local": {"origin_enu_m": [0.0, 0.0, 0.0]}},
                "planner_defaults": {
                    "vehicle_minimum_xy_size_m": [1.0, 1.0],
                    "hard_inflation_radius_m": 1.45,
                    "preferred_clearance_m": 2.15,
                },
                "obstacles": {"buildings": []},
            }
        ),
        encoding="utf-8",
    )
    return path


def fake_result(boxes):
    centerline = ((0.0, 0.0, 10.0), (10.0, 0.0, 10.0), (20.0, 10.0, 10.0))
    corridor = SimpleNamespace(centerline_enu_m=centerline, boxes=boxes)
    metrics = WaypointPlanMetrics3D(
        success=True,
        reason="GOAL_REACHED",
        elapsed_s=0.1,
        astar_elapsed_s=0.05,
        sfc_elapsed_s=0.05,
        expanded_nodes=10,
        raw_waypoint_count=3,
        output_waypoint_count=3,
        corridor_box_count=len(boxes),
        path_length_m=24.14,
        minimum_clearance_m=8.0,
        cache_hit=False,
    )
    return SimpleNamespace(success=True, corridor=corridor, metrics=metrics)


def test_exports_common_plan_and_roundtrips_consumer(monkeypatch, tmp_path):
    map_path = coordinate_map(tmp_path)
    large_box = CorridorBox3D((-10.0, -10.0, 4.0), (30.0, 20.0, 16.0))
    result = fake_result((large_box, large_box))
    monkeypatch.setattr(
        "autonomy_planner.px4_l1_export.plan_validated_waypoints_3d",
        lambda *args, **kwargs: result,
    )
    manifest = export_common_astar_sfc_plan(
        repository_root=tmp_path,
        map_path=map_path,
        start_enu_m=(0.0, 0.0, 10.0),
        goal_enu_m=(20.0, 10.0, 10.0),
        planner_run_id="synthetic-run",
    )
    destination = write_plan(tmp_path / "l1_plan.yaml", manifest)
    loaded = CollisionCheckedL1Plan.load(destination)
    assert loaded.planner_run_id == "synthetic-run"
    assert len(loaded.waypoints_enu_m) == 3
    assert manifest["guidance_contract"]["trajectory_setpoint_publisher"] is False
    assert manifest["minimum_surface_clearance_m"] == pytest.approx(10.0)
    assert manifest["planner_proof"]["minimum_surface_clearance_applied"] is True


def test_l1_certificate_failure_does_not_weaken_corridor(monkeypatch, tmp_path):
    map_path = coordinate_map(tmp_path)
    narrow_box = CorridorBox3D((-1.0, -1.0, 9.0), (21.0, 11.0, 11.0))
    result = fake_result((narrow_box, narrow_box))
    monkeypatch.setattr(
        "autonomy_planner.px4_l1_export.plan_validated_waypoints_3d",
        lambda *args, **kwargs: result,
    )
    with pytest.raises(L1ExportError, match="B-spline remains available"):
        export_common_astar_sfc_plan(
            repository_root=tmp_path,
            map_path=map_path,
            start_enu_m=(0.0, 0.0, 10.0),
            goal_enu_m=(20.0, 10.0, 10.0),
        )
