# JO experiment logger

`experiment_logger` is a read-only sidecar. It only subscribes to existing JO
topics and `/rosout`; it publishes no command and does not change mission,
planner, MPC, perception, or PX4 behavior.

## Metric contract

| CSV field | Existing source | Definition in the summary row |
|---|---|---|
| `path_length_m` | `/mission/vehicle_position` | 3-D distance accumulated from `TAKEOFF` through `PRECLAND` |
| `tracking_error_{mean,max,rmse}_m` | `/mission/vehicle_position` + `/fmu/in/trajectory_setpoint` | Horizontal local-ENU setpoint error in `MISSION` and `RETURN` |
| `min_clearance_m` | vehicle position + immutable mission YAML | Minimum horizontal distance to a physical obstacle AABB during flight |
| `astar_plan_time_ms` | accepted `global A*/B-spline` `/rosout` record | Mean accepted A* + B-spline pipeline elapsed time; isolated A* timing is not exposed by the current controller |
| `mpc_solve_time_ms` | existing `EXPERIMENT_METRICS` `/rosout` record | Mean TrackingMPC + LandingMPC wall-clock solve time |
| `replan_count` | `/mission/active_plan_markers` + `/mission/state` | Accepted `RETURN` route replacements after its first committed route |
| `aruco_detection_rate_pct` | `/aruco/detected`, reconciled with `EXPERIMENT_METRICS` | Detection hits / frames during precision landing, before touchdown |
| `relative_xy_error_m` | vehicle position + `/marker/cue` | Mean horizontal vehicle-to-trailer error in `RETURN` and landing phases |
| `landing_xy_error_m` | existing touchdown `EXPERIMENT_METRICS` | Bias-corrected horizontal error at PX4-confirmed touchdown |
| `touchdown_relative_speed_m_s` | existing touchdown `EXPERIMENT_METRICS` | 3-D vehicle-to-trailer relative speed at PX4-confirmed touchdown |
| `sfc_generation_time_ms` | planner `/rosout` timing field | Mean wall-clock time for the `cover_polyline` call that certifies each accepted active path |
| `sfc_min_width_m` | `/mission/active_plan_markers` | Minimum `min(scale.x, scale.y)` over accepted SFC boxes |
| `sfc_avg_width_m` | `/mission/active_plan_markers` | Mean horizontal width over accepted SFC boxes |
| `sfc_corridor_count` | `/mission/active_plan_markers` | Box count in the latest accepted SFC |
| `sfc_violation_count` | vehicle position + active SFC boxes | Contiguous vehicle excursions outside the active horizontal SFC-box union in route-tracking states |

The implementation's collision/SFC contract is planar; marker-box height is
only a visualization slab. The summary additionally stores the out-of-corridor
sample count and rate, while the required count remains sample-rate-independent
excursion events. `sfc_corridor_count` is the box count in the latest accepted
corridor snapshot.

The summary also preserves the established exporter names such as
`path_tracking_rmse_m`, `marker_detection_rate_pct`,
`touchdown_relative_speed_3d_m_s`, and `mpc_solve_mean_ms`.

## Standalone use

```bash
ros2 run experiment_logger experiment_logger_node --ros-args \
  -p output_dir:=/path/to/run \
  -p map_yaml:=/path/to/coordinates.yaml \
  -p run_id:=real_001
```

The node creates `experiment_<UTC timestamp>.csv` and
`experiment_<UTC timestamp>_summary.csv`. Empty summary cells mean the source
was unavailable or the applicable phase did not complete; `missing_metrics`
lists those fields explicitly.
