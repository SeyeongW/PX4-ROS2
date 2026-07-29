# DRONE_CJU_TRACK working copy

`DRONE_CJU_TRACK.world` is the isolated working copy of the CJU stadium version
that previously occupied `mpc_landing_200m_moving.world`. The default moving
profile has since returned to a compact flat obstacle-patrol map; this CJU
world and its dedicated models remain unchanged as the preserved stadium
variant.

The working copy uses two dedicated model directories:

- `models/drone_cju_track_running_track`: one continuous red eight-lane track
- `models/drone_cju_track_stadium`: the stadium with one smooth, curved blue
  grandstand roof mesh in place of the seven faceted roof panels

The roof OBJ, continuous track-surface OBJ, local materials, and lane
primitives are all generated locally:

```bash
python3 simulation/gazebo/gen_drone_cju_track_models.py
```

## Run

Preview only:

```bash
./simulation/gazebo/run_world.sh cju-track
```

PX4 plus the moving landing platform:

```bash
DRIVE_TRAILER=1 TRAILER_SPEED_M_S=3 \
  ./simulation/gazebo/run_px4_map.sh cju-track
```

Gimbal and perception stack:

```bash
LANDING_MAP=cju-track ./simulation/gazebo/run_gimbal.sh
```

Full landing mission:

```bash
LANDING_MAP=cju-track ./simulation/gazebo/run_gimbal.sh mission
```

The SDF world name is `DRONE_CJU_TRACK`, and the dedicated coordinate contract
is `maps/drone_cju_track.yaml`.
