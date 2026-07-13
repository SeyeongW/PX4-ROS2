import copy

import pytest

from autonomy_planner.px4_l1_plan import (
    AxisAlignedCorridor,
    CollisionCheckedL1Plan,
    PlanValidationError,
    SCHEMA_VERSION,
    certify_triplets_in_single_corridors,
    with_computed_digest,
)


def valid_plan_mapping():
    mapping = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_frame": "gazebo_world_enu",
        "map_revision": "applepark_uav-test",
        "occupancy_sha256": "a" * 64,
        "planner_run_id": "astar-sfc-test-001",
        "world_home_enu_m": [0.0, 0.0, 0.0],
        "hard_radius_m": 1.35,
        "min_clearance_m": 2.0,
        "collision_check_resolution_m": 0.25,
        "minimum_surface_clearance_m": 10.0,
        "planner_proof": {
            "astar_occupancy_checked": True,
            "sfc_checked": True,
            "continuous_segment_checked": True,
            "hard_radius_applied": True,
            "minimum_surface_clearance_applied": True,
        },
        "waypoints_enu_m": [
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
            [20.0, 10.0, 10.0],
        ],
        "corridor_boxes": [
            {"min_enu_m": [-10.0, -10.0, 0.0], "max_enu_m": [30.0, 20.0, 20.0]}
        ],
        "l1_triplet_certificates": [
            {
                "center_waypoint_index": 1,
                "corridor_box_index": 0,
                "is_turning_triplet": True,
                "turn_envelope_radius_m": 5.0,
                "validated_min_clearance_m": 2.0,
            }
        ],
    }
    return with_computed_digest(mapping)


def test_valid_plan_and_out_and_back_patrol():
    plan = CollisionCheckedL1Plan.from_mapping(valid_plan_mapping())
    assert plan.plan_sha256
    assert plan.patrol_waypoints(2) == (
        (0.0, 0.0, 10.0),
        (10.0, 0.0, 10.0),
        (20.0, 10.0, 10.0),
        (10.0, 0.0, 10.0),
        (0.0, 0.0, 10.0),
        (10.0, 0.0, 10.0),
        (20.0, 10.0, 10.0),
        (10.0, 0.0, 10.0),
        (0.0, 0.0, 10.0),
    )


def test_digest_detects_any_modified_coordinate():
    mapping = valid_plan_mapping()
    mapping["waypoints_enu_m"][1][0] = 10.1
    with pytest.raises(PlanValidationError, match="plan_sha256 mismatch"):
        CollisionCheckedL1Plan.from_mapping(mapping)


def test_all_safety_proof_flags_are_required():
    mapping = valid_plan_mapping()
    mapping["planner_proof"]["continuous_segment_checked"] = False
    mapping = with_computed_digest(mapping)
    with pytest.raises(PlanValidationError, match="continuous_segment_checked"):
        CollisionCheckedL1Plan.from_mapping(mapping)


def test_surface_clearance_cannot_be_weakened_below_ten_metres():
    mapping = valid_plan_mapping()
    mapping["minimum_surface_clearance_m"] = 9.99
    mapping = with_computed_digest(mapping)
    with pytest.raises(PlanValidationError, match="at least 10 m"):
        CollisionCheckedL1Plan.from_mapping(mapping)


def test_five_metre_l1_envelope_is_not_optional():
    mapping = valid_plan_mapping()
    mapping["l1_triplet_certificates"][0]["turn_envelope_radius_m"] = 4.99
    mapping = with_computed_digest(mapping)
    with pytest.raises(PlanValidationError, match="5 m minimum"):
        CollisionCheckedL1Plan.from_mapping(mapping)


def test_certification_fails_closed_when_triplet_spans_boxes():
    waypoints = ((0.0, 0.0, 10.0), (10.0, 0.0, 10.0), (20.0, 0.0, 10.0))
    boxes = (
        AxisAlignedCorridor((-6.0, -6.0, 4.0), (11.0, 6.0, 16.0)),
        AxisAlignedCorridor((9.0, -6.0, 4.0), (26.0, 6.0, 16.0)),
    )
    with pytest.raises(PlanValidationError, match="lacks a convex SFC box"):
        certify_triplets_in_single_corridors(
            waypoints,
            boxes,
            turn_envelope_radius_m=5.0,
            validated_min_clearance_m=2.0,
        )


def test_l1_lookahead_inset_is_horizontal_not_a_fictitious_vertical_radius():
    waypoints = ((0.0, 0.0, 10.5), (10.0, 0.0, 10.5), (20.0, 10.0, 10.5))
    # The 3-D SFC proves the flight layer, but it need not provide five metres
    # below the centreline: L1 look-ahead is horizontal and is not a sphere.
    box = AxisAlignedCorridor((-6.0, -6.0, 10.4), (26.0, 16.0, 12.0))
    certificates = certify_triplets_in_single_corridors(
        waypoints,
        (box,),
        turn_envelope_radius_m=5.0,
        validated_min_clearance_m=2.0,
    )
    assert certificates[0].is_turning_triplet is True
    assert certificates[0].corridor_box_index == 0


def test_duplicate_certificates_are_rejected():
    mapping = valid_plan_mapping()
    mapping["l1_triplet_certificates"].append(
        copy.deepcopy(mapping["l1_triplet_certificates"][0])
    )
    mapping = with_computed_digest(mapping)
    with pytest.raises(PlanValidationError, match="exactly one"):
        CollisionCheckedL1Plan.from_mapping(mapping)
