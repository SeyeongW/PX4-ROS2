# `drone_cju` world

`worlds/drone_cju.world` is the stadium-only Cheongju University world. Its
runtime coordinate contract is `maps/drone_cju_track.yaml`; the existing
launcher key remains `cju-track`.

```bash
./simulation/gazebo/run_world.sh cju-track       # map preview
./simulation/gazebo/run_px4_map.sh cju-track     # PX4 + stationary trailer
./simulation/gazebo/run_gimbal.sh mission        # full mission
```

For the full mission, enter exactly `takeoff → mission → land`. After
`takeoff` settles at 5 m, the node holds `READY`; `mission` plans and flies the
map leg. The canonical
stadium endpoint remains `(0,0)`. The trailer and drone spawn together on the
track centre at the integer coordinate `(5,0)`, then the trailer repeats the
straight `(5,0) → (5,50) → (5,0)` route at 1.0 m/s without turning its body. It becomes
the ArUco landing target when `land` is entered.

The mission groups are:

- Phase 0 `PRECHECK`: fail closed on PX4 feedback, the live cue, planner, and
  Offboard readiness.
- Phase 1 `TAKEOFF`: PX4 `NAV_TAKEOFF` to 5 m; PX4 owns the climb profile.
- Phase 2: A* supplies obstacle topology, a geometry-only B-spline reinforces
  that spatial route, and PX4 Goto flies it to map `(50,50)` and `HOVER`.
- Phase 3: `land` builds one A* → geometry-only B-spline `RETURN`, then
  atomically retargets only its 2.5 m-safe tail every two seconds from the
  latest live GPS/cue. Full replanning is used only when no safe tail connector
  exists. Inside the 6 m YAML-safe handoff corridor the companion then
  publishes `LandingTargetPose`, requests PX4 `NAV_PRECLAND`, and stops all
  Offboard setpoints. PX4 owns the remaining dynamics, touchdown verdict and
  auto-disarm; ArUco only corrects the target position.

The B-spline has no flight speed or P/V/A schedule. PX4 Goto/PositionSmoothing
owns the route timing through `MPC_XY_CRUISE=3`, `MPC_ACC_HOR`, and
`MPC_JERK_AUTO`; `MPC_XY_VEL_MAX=10` is the hard horizontal cap.
The 6 m lookahead is a spatial path target, not a spline time/speed schedule;
exact segment checks shorten it near obstacles.
`mission_manager_node` remains the only PX4 Offboard setpoint publisher.

`/marker/cue` and `/marker/cue_velocity` use the explicit frame ID
`px4_local_enu`: the PX4 local origin converted to ENU, not the rotated
`stadium_endpoint` YAML coordinates. Its XY origin coincides with the drone
spawn in this map. `gimbal_cue.log` reports position, velocity, and publisher
delay once per second.

Geometry, source references, and calibration are documented in
`CJU_STADIUM_WORLD.md`.

Each run is preserved below
`${XDG_STATE_HOME:-$HOME/.local/state}/px4-ros2-wang/cju/`. Alongside the raw
`flight.ulg`, ROS/Gazebo logs, and 50 Hz `trailer_odometry.jsonl`, cleanup writes:

- `map.yaml`: the immutable coordinate/map snapshot used by that run;
- `flight_1hz.csv`: one row per one-second interval, with time-weighted means
  and native-rate maxima/p95 values so short speed spikes remain visible;
- `flight_summary.csv`: one run-level row with peak speed, acceleration,
  descent rate, data gaps, failsafe/dropout counts, and a PASS/WARN/FAIL flag.

Re-export an existing artifact without rerunning the simulator:

```bash
python3 -m pip install 'pyulog==1.2.3'  # once; install_apt_deps.sh also does this
python3 simulation/gazebo/tools/export_flight_1hz.py /path/to/run-directory
```

The ULog is the authoritative lossless record; the 1 Hz CSV is the
paper/plotting view. Coordinates are PX4 local NED converted to local ENU, with
additional map `(x,y)` columns in the YAML `stadium_endpoint` frame.
