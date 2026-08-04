# `drone_cju` world

`worlds/drone_cju.world` is the stadium-only Cheongju University world. Its
runtime coordinate contract is `maps/drone_cju_track.yaml`; the existing
launcher key remains `cju-track`.

```bash
./simulation/gazebo/run_world.sh cju-track       # map preview
./simulation/gazebo/run_px4_map.sh cju-track     # PX4 + stationary trailer
./simulation/gazebo/run_gimbal.sh mission        # full mission
```

For the full mission, enter exactly `takeoff → mission → land`. The trailer stays
still for takeoff, begins its 3.0 m/s track route when `mission` is accepted, and
becomes the ArUco landing target when `land` is entered. `mission_manager_node` is the
only PX4 Offboard setpoint publisher.

Geometry, source references, and calibration are documented in
`CJU_STADIUM_WORLD.md`.
