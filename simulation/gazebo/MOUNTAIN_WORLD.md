# UGV drone mountain world

`worlds/ugv_drone_mountain_map.world` is the Gazebo Harmonic 300 m mountain flight map.
Its terrain, texture and tree collision assets are self-contained under
`gazebo/models`. `run_px4_map.sh` connects that world to the existing PX4 SITL
build and PX4 dynamically spawns the real `x500_mono_cam_down` vehicle.

## Map layout

- Bounds: 300 x 300 m (`x`, `y` = -150..150 m)
- Terrain elevation: 0..40 m
- Retained summits: approximately `(-75, 75, 40)` and `(55, 80, 20.078)`
- Mountain relief: six broad deterministic ridge / foothill envelopes around
  the retained hills; nonzero relief covers 70.4795% of the map
- Actual collision-triangle grade: p99 `66.9281%`, maximum `78.0509%`
- Launch pad: `(-80, -80)`, top surface `z=0.16 m`
- PX4 x500 spawn: `(-80, -80, 0.16)`, yaw `0.785398 rad`; its landing gear settles approximately 1.3 cm onto the pad
- Forest: 288 uniformly 2x-scaled textured trees (216 pines, 72 oaks) with
  matching 2x trunk collisions seated 3 mm below each terrain-disk minimum
- Maze: removed; runtime wall collisions and visuals are both zero
- Obstacle body: one static compound link containing only the 288 tree trunks / 576 tree visuals
- Vehicle: PX4 airframe 4014, `gz_x500_mono_cam_down`, runtime entity `x500_mono_cam_down_0`
- Moving platform: `flat_platform`, selectively ported from the `seo` branch
- Physics: Bullet Featherstone, 2 ms step / 500 Hz

The central exact-flat clearing is the ellipse
`(x/42)^2 + (y/30)^2 <= 1`; the complete 12 x 12 m launch-pad footprint is also
held at `z=0`. Tree collisions are conservatively buried to the minimum sampled
height under their complete disk. This avoids the former short tree shelves,
which created artificial slope spikes, while keeping static trunks from
floating above the terrain.

The flat south-west pad is deliberate: the generated heightmap is exactly zero
at `(-80, -80)`, so the vehicle retains a deterministic contact surface. Keep
an absolute altitude above 50 m when first crossing the main summit, then tune
the route after checking actual clearance.

## Run

From the `PX4-ROS2` repository root, one command starts Gazebo, the optional
Micro XRCE-DDS Agent and PX4 SITL in the correct order:

```bash
./gazebo/run_px4_map.sh mountain
```

On a fresh PC, run `./gazebo/setup_px4_sitl.sh` once first. It runs PX4's
official Ubuntu general-prerequisite helper, which may request sudo once. An
existing `~/PX4-Autopilot` checkout or firmware version is never changed. The launcher
sets the resource paths, standalone PX4 environment, Wayland/XWayland
workaround and NVIDIA PRIME variables for the RTX 5060.

The exact two-process equivalent is:

```bash
export GZ_SIM_RESOURCE_PATH="$PWD/gazebo/models:$PWD/gazebo/worlds:$HOME/PX4-Autopilot/Tools/simulation/gz/models"
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__VK_LAYER_NV_optimus=NVIDIA_only \
QT_QPA_PLATFORM=xcb \
gz sim -v4 -r \
  --physics-engine gz-physics-bullet-featherstone-plugin \
  gazebo/worlds/ugv_drone_mountain_map.world

# second terminal, after Gazebo reports ready
cd ~/PX4-Autopilot/build/px4_sitl_default/rootfs
source ./gz_env.sh
PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=ugv_drone_mountain_map \
PX4_GZ_MODEL_POSE='-80,-80,0.16,0,0,0.785398' \
PX4_SYS_AUTOSTART=4014 PX4_SIM_MODEL=gz_x500_mono_cam_down \
../bin/px4
```

Useful launcher switches:

```bash
HEADLESS=1 ./gazebo/run_px4_map.sh mountain    # server-only performance test
START_XRCE=0 ./gazebo/run_px4_map.sh mountain  # PX4/Gazebo without ROS 2 DDS
USE_NVIDIA=0 ./gazebo/run_px4_map.sh mountain  # GPU troubleshooting only
DRIVE_TRAILER=1 TRAILER_ROUTE_LOOPS=1 ./gazebo/run_px4_map.sh mountain
```

Verify that rendering is on the RTX GPU in another terminal:

```bash
watch -n 1 'nvidia-smi pmon -s um -c 1; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader'
```

`WorldStats` in the Gazebo window shows the real-time factor. A sustained value
near 1.0 means the simulation is keeping up. The 2 ms / 500 Hz setting is the
known-good baseline retained from the previous map; record fresh real-time
factor and RTX 5060 samples after every terrain or obstacle change. For
map-only inspection, `PAUSED=1` avoids spending CPU on physics.

The 72-tree baseline GUI validation on 2026-07-11 passed. Across 137 stats
samples, one diagnostic-load transient was `0.5458`; the other 136 samples
were `0.9987..1.0006` with mean `0.99988`. The GUI displayed `100.00%` RTF.
RTX 5060 utilization was `12..14%` (mean `13.2%`) with 659 MiB VRAM, and
the launch/resource error scan found 0 failures. The screenshot and raw logs
are under `gazebo/validation/runtime/`; the concise result is
`ugv_drone_harmonic_runtime.log`.

The older GUI performance logs remain under `gazebo/validation/runtime/` as
historical baselines. The current terrain / trailer revision has its own
runtime record in `trailer_waypoint_validation.log`: all nine central-flat
waypoints and all nine optional terrain-follow safeguard waypoints passed with
no impulse launch or excessive tilt. The latter is kinematic height following,
not physical wheel / suspension traction.

The visual and collision OBJ files each contain 66,049 vertices and 131,072
triangles. If later additions reduce the real-time factor, decimate only the
collision mesh while retaining the current visual OBJ and generated heightmap.

Do not omit the Bullet Featherstone argument when launching this first map
version. Harmonic 8's default DART backend logs that SDF mesh collision
construction is not implemented and would leave the mountain visual without a
matching collision surface. `run_world.sh mountain` and `run_px4_map.sh
mountain` select Bullet automatically.
The forest deliberately uses one static link. Bullet Featherstone rejects a
static model made from many unjointed links as multiple floating subtrees;
the generated compound link keeps all 288 trunk collisions and 576 visuals
active without fake joints.

For a later high-fidelity dynamics comparison, change `max_step_size` back to
`0.001` and `real_time_update_rate` to `1000`; expect slower-than-real-time
execution on this exact full scene unless SITL timing or forest complexity is
also reduced.

## Rebuild and validate

The deterministic terrain formula, natural texture, tree meshes and layout
snapshots are stored with the models. The builder recreates the heightmap,
reseats every tree, and generates byte-identical visual / collision OBJ meshes
and the compound obstacle SDF. Rebuild and run the static checks with:

```bash
python3 gazebo/tools/build_mountain_300_assets.py
python3 gazebo/tools/validate_mountain_300_assets.py
```

The checked result is written to
`gazebo/validation/ugv_drone_mountain_300_static.log`. The validator asserts
the 300 m bounds, 40 m / 20 m summits, deterministic ridge pixels, unchanged
spawn, collision counts, the one-link Bullet-compatible obstacle structure,
all absolute poses against the preserved source layouts, zero maze runtime
entities, all 288 conservatively buried trunk disks, the full
launch-pad footprint, zero blue-dominant terrain pixels, byte-identical
visual/collision geometry, neutral scene background and the absence of
operational references to `sim_assets`.

The previous 200 m Harmonic world and models are retained under
`gazebo/backups/pre_mountain_300m_20260711/`.

## Sources and licenses

The first Harmonic port copied the following local model assets from
`FSD_Vehicle`:

- the original 200 m `ugv_mou_terrain` visual/collision OBJ assets
- the original `ugv_mou_forest_obstacles` 177-tree primitive model

The removed central maze was derived from the local modified
`engcang/gazebo_maps` Height Maze asset; its old source snapshot is retained
for provenance but is not read by the builder or included at runtime. The active 300 m terrain is generated
locally from analytic compact hills, broad anisotropic ridges, an elliptical
trailer clearing and deterministic grade caps; no third-party DEM or satellite
image is embedded. Pine and oak DAE models are from the Gazebo model collection
included with the source asset. The corresponding license texts are preserved as:

- `gazebo/licenses/height_maze_BSD-3-Clause.txt`
- `gazebo/licenses/tree_models_CC-BY-3.0.txt`

The two generated terrain OBJ files are about 13 MiB each and are committed as
ordinary Git blobs. A fresh clone therefore receives the collision geometry
without a separate Git LFS fetch.

The copied local bird / terrain assets do not contain an explicit upstream
license file. They are suitable for this local workspace, but confirm their
redistribution rights before pushing the assets to a public repository.

## Open-source mountain references

- [Gazebo Terrain Generator](https://github.com/saiaravind19/gazebo_terrain_generator)
  (BSD-3-Clause): Harmonic terrain generation and its Joshimath DEM sample.
- [Gazebo DEM / heightmap guide](https://gazebosim.org/api/sim/10/heightmap_dem.html):
  official GeoTIFF, heightmap collision, texture and blend examples.
- [SDFormat heightmap geometry](https://sdformat.org/spec/1.12/geometry):
  canonical `uri`, world-unit `size`, `pos` and sampling semantics.
- [Gazebo Sim Mount St. Helens example](https://github.com/gazebosim/gz-sim/blob/gz-sim8/examples/worlds/dem_volcano.sdf):
  official SDF 1.9 DEM and mountain-view GUI configuration.
- [PX4 ridge world](https://github.com/PX4/PX4-gazebo-models/blob/main/worlds/ridge.sdf)
  (BSD-3-Clause): lightweight polyline ridge for repeatable terrain-clearance tests.
- [PX4 windy world](https://github.com/PX4/PX4-gazebo-models/blob/main/worlds/windy.sdf):
  a useful second-stage reference after terrain-only flight is stable.
- [engcang/gazebo_maps](https://github.com/engcang/gazebo_maps) (BSD-3-Clause):
  mountain / cliff layouts, but its launch instructions target Gazebo Classic.
- [FastNoiseLite](https://github.com/Auburn/FastNoiseLite) (MIT): reference for
  fixed-seed ridged noise and domain-warp concepts; no library code is copied.
- [terrain-erosion-3-ways](https://github.com/dandrino/terrain-erosion-3-ways)
  (MIT): reference for low-frequency ridge and erosion structure; no source or
  generated asset is copied.

Do not copy satellite imagery or Mapbox-derived data solely because the terrain
generator code is BSD licensed; map data providers have separate terms.
