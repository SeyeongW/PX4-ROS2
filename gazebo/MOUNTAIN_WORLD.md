# UGV drone mountain world

`worlds/ugv_drone.world` is the Gazebo Harmonic version of the local 300 m
mountain flight map. Its terrain, texture, tree and maze collision assets are
self-contained under `gazebo/models`, so those map assets require neither a
Gazebo Fuel download nor the `sim_assets` workspace. The Iris vehicle and
ArduPilot system plugin continue to come from the documented
`~/ardupilot_gazebo` installation.

## Map layout

- Bounds: 300 x 300 m (`x`, `y` = -150..150 m)
- Terrain elevation: 0..40 m
- Retained summits: approximately `(-75, 75, 40)` and `(55, 80, 20.078)`
- Mountain relief: seven warped, low-frequency ridge / foothill envelopes
  around the retained hills; nonzero relief covers 56.68% of the map
- Launch pad: `(-80, -80)`, top surface `z=0.16 m`
- SITL Iris spawn: `(-80, -80, 0.355)`, yaw 45 degrees; its landing gear contacts the `z=0.16 m` pad exactly
- Map-preview drone origin: `(-80, -80, 0.16)`, normalized to its landing-gear contact plane on the same pad
- Forest: 288 uniformly 2x-scaled textured trees (216 pines, 72 oaks) with
  matching 2x trunk collisions and an exact local terrain seat under every trunk disk
- Maze: 72 unique collision walls, each 3.75 m high, near the map centre
  (the imported source snapshot keeps all 73 entries, including one exact duplicate)
- Obstacle body: one static compound link containing all tree and maze geometry
- Vehicle: `iris_with_down_camera`, ArduPilot JSON FDM UDP port 9002
- Physics: Bullet Featherstone, 2 ms step / 500 Hz

The central maze core (`x=-42..42`, `y=-30..30`) and the complete 12 x 12 m
launch-pad footprint are explicitly held at `z=0`. The generator also expands
each tree's flat core by one 1.171875 m grid-cell diagonal beyond the trunk
radius. Consequently no wall or tree collision is positioned from a centre
sample alone: their full contact footprints are checked against the same
8-bit height field that generates both OBJ meshes.

The flat south-west pad is deliberate: the generated heightmap is exactly zero
at `(-80, -80)`, so the vehicle retains a deterministic contact surface. Keep
an absolute altitude above 50 m when first crossing the main summit, then tune
the route after checking actual clearance.

## Run

Gazebo and SITL run in separate terminals. The order does not matter.

Terminal 1, from the `PX4-ROS2` repository root:

```bash
./gazebo/run_ugv_drone.sh
```

The launcher sets `GZ_SIM_RESOURCE_PATH`, the ArduPilot plugin path, the
Wayland/XWayland workaround and the NVIDIA PRIME variables for the RTX 5060.
The equivalent direct command is:

```bash
export GZ_SIM_RESOURCE_PATH="$PWD/gazebo/models:$PWD/gazebo/worlds:$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HOME/ardupilot_gazebo/build"
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__VK_LAYER_NV_optimus=NVIDIA_only \
QT_QPA_PLATFORM=xcb \
gz sim -v4 -r \
  --physics-engine gz-physics-bullet-featherstone-plugin \
  gazebo/worlds/ugv_drone.world
```

Terminal 2:

```bash
cd ~/ardupilot
sim_vehicle.py -v ArduCopter -f JSON -I0 --console --map
```

Useful launcher switches:

```bash
PAUSED=1 ./gazebo/run_ugv_drone.sh       # inspect before physics starts
HEADLESS=1 ./gazebo/run_ugv_drone.sh     # server-only performance test
USE_NVIDIA=0 ./gazebo/run_ugv_drone.sh   # GPU troubleshooting only
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

After adding the ridges and exact contact patches to the 288-tree / 2x forest,
the repository-only preview was run again on the RTX 5060. Across 58
post-initialization samples the real-time factor was `0.986500..1.018512`
with mean `1.000001`; the Gazebo GUI GPU process sampled at `18%` with
376 MiB total VRAM in use. Gazebo Harmonic initialized the world and Bullet
compound body without a model, resource or physics load error, and reported
the preview pose as `(-80, -80, 0.16)`. The concise result is
`mountain_tree288_runtime.log`.

The visual and collision OBJ files each contain 66,049 vertices and 131,072
triangles. If later additions reduce the real-time factor, decimate only the
collision mesh while retaining the current visual OBJ and generated heightmap.

Do not omit the Bullet Featherstone argument when launching this first map
version. Harmonic 8's default DART backend logs that SDF mesh collision
construction is not implemented and would leave the mountain visual without a
matching collision surface. `run_ugv_drone.sh` selects Bullet automatically.
The forest and maze deliberately share one static link. Bullet Featherstone
rejects a static model made from many unjointed links as multiple floating
subtrees, disabling those links; the generated compound link keeps all 360
collisions (288 trunks and 72 walls) and 648 visuals active without fake joints.

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
all absolute poses against the preserved source layouts, source-maze
deduplication, all 72 oriented wall footprints, all 288 trunk disks, the full
launch-pad footprint, zero blue-dominant terrain pixels, byte-identical
visual/collision geometry, neutral scene background and the absence of
operational references to `sim_assets`.

The previous 200 m Harmonic world and models are retained under
`gazebo/backups/pre_mountain_300m_20260711/`.

## Copied sources and licenses

The source files were preserved without edits under `worlds/legacy`:

- `ugv_drone_flat_classic.world` from `ugv_ws3`
- `ugv_world_classic.world` from `FSD_Vehicle`

The first Harmonic port copied the following local models from `FSD_Vehicle`:

- the original 200 m `ugv_mou_terrain` visual/collision OBJ assets
- the original `ugv_mou_forest_obstacles` 177-tree primitive model
- `bird` (DAE and its PNG texture, used by `worlds/ugv.world`)

The central maze layout is derived from the local modified
`engcang/gazebo_maps` Height Maze asset. The active 300 m terrain is generated
locally from analytic compact hills, anisotropic ridges, deterministic value
noise, protected plateaus and edge tapering; no third-party DEM or satellite
image is embedded. Pine and oak DAE models are from the Gazebo model collection
included with the source asset. The corresponding license texts are preserved as:

- `gazebo/licenses/height_maze_BSD-3-Clause.txt`
- `gazebo/licenses/tree_models_CC-BY-3.0.txt`

The executable `worlds/ugv.world`, model manifests and model SDF files were
ported from SDF 1.6 to SDF 1.9. The incompatible Gazebo Classic
`libgazebo_ros_state.so` plugin was removed; use `ros_gz_bridge` for ROS 2.

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
