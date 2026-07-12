# PX4-ready Gazebo maps

The `jo` branch keeps every map world, collision mesh and texture inside this
repository. The main launcher starts Gazebo Harmonic and then PX4 SITL; PX4
dynamically creates the real, motor-driven `x500_mono_cam_down` entity. There
is no static preview-drone substitute in either world.

```bash
git clone -b jo https://github.com/SeyeongW/PX4-ROS2.git
cd PX4-ROS2
./gazebo/setup_px4_sitl.sh       # first PC / one time only
./gazebo/run_px4_map.sh city
# or
./gazebo/run_px4_map.sh mountain
```

The setup helper pins a new checkout to the tested PX4 `v1.17.0`, but never
changes or checks out an existing `~/PX4-Autopilot`. The launch uses airframe
`4014`, standalone Gazebo mode, and the spawn pose from the checked-in YAML.
It also starts the local Micro XRCE-DDS Agent when available, enabling PX4
ROS 2 `/fmu/*` topics; MAVLink / PX4 flight remains available without it.

Map-only inspection is still available with `./gazebo/run_world.sh city` or
`mountain`, but those commands intentionally contain no drone. Use
`run_px4_map.sh` whenever a flyable PX4 vehicle is required.

The equivalent direct commands below force rendering onto the NVIDIA GPU on a
hybrid laptop without using the launcher script:

```bash
# Mountain
GZ_SIM_RESOURCE_PATH="$PWD/gazebo/models:$PWD/gazebo/worlds" __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only QT_QPA_PLATFORM=xcb gz sim -v4 -r --physics-engine gz-physics-bullet-featherstone-plugin "$PWD/gazebo/worlds/ugv_drone_map.world"

# City
GZ_SIM_RESOURCE_PATH="$PWD/gazebo/models:$PWD/gazebo/worlds" __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only QT_QPA_PLATFORM=xcb gz sim -v4 -r --physics-engine gz-physics-bullet-featherstone-plugin "$PWD/gazebo/worlds/applepark_city/applepark.world"
```

## City

- world: `gazebo/worlds/applepark_city/applepark.world`
- size: 500 x 500 m
- 274 buildings; each checked-in pre-update height is further scaled by a
  deterministic pseudo-random factor in `2.001288..3.476403`
- terrain-seated foundations and every XY footprint remain unchanged; only
  roof vectors are raised (`15.907114..109.621338 m` above ground)
- visual and Bullet terrain surfaces share the same `-5.299..0.856 m` Z range
- spawn: `(-120, 115)`
- non-photographic OSM road / land-use texture
- Bullet-compatible 129 x 129 terrain collision OBJ

The road generator used a square 2318.81 m coordinate system, while the old
crop applied a 2124.58 m Y scale. Correcting that transform reduced asphalt
overlap from 109 buildings / 104,410 pixels to 5 / 237 pixels. Exact DAE
footprints plus a one-pixel (0.244 m) clearance remove the remaining road and
casing pixels; final visible road/building overlap is zero. Coordinates are in
`gazebo/validation/city/road_building_overlap_coordinates.csv`.

Road and building geometry originate from OpenStreetMap-derived data. See
`worlds/applepark_city/OSM_ATTRIBUTION.txt` and
`licenses/applepark_terrain_BSD-3-Clause.txt`.

The source terrain generator assigned one centroid elevation to each flat
building base. On sloped ground that produced downhill gaps up to 0.732 m and
uphill terrain intersections up to 2.485 m. The checked DAE extends the bottom
vertices beneath both the visual heightmap and Bullet collision surface by at
least 0.05 m. The latest height pass then preserves those exact foundations,
all roads and all XY coordinates while raising only the 13,872 roof vectors.
Foundation values are recorded in
`gazebo/validation/city/building_foundation_alignment.csv`; the stable
SHA-256-derived factor, old roof and new roof for every component are in
`gazebo/validation/city/building_height_scaling.csv`.

Gazebo's OGRE2 renderer normalizes an image heightmap against the largest
pixel actually present. The 500 m crop has a maximum value of 152 rather than
255, so the visual height size is `26.6 * 152 / 255 = 15.855686274509806 m`.
This keeps the rendered road surface exactly on the separately generated
Bullet collision mesh instead of letting it cut through the buildings.

## Mountain

- world: `gazebo/worlds/ugv_drone_map.world`
- size: 300 x 300 m
- retained 40 m / 20.078 m summits plus seven deterministic branching ridges
- 288 uniformly 2x-scaled trees on exact protected contact patches
- artificial maze removed (zero maze collisions and zero maze visuals)
- spawn: `(-80, -80)`

For the actual PX4-controlled x500 with downward monocular camera, run:

```bash
./gazebo/run_px4_map.sh mountain
```

Exact world ENU coordinates, PX4 spawn-relative NED conversion, all 274 city
building polygons (including the courtyard hole), and all 288 mountain tree
collision cylinders are exported to:

- `gazebo/maps/city_coordinates.yaml`
- `gazebo/maps/mountain_coordinates.yaml`

Regenerate the YAML and the two 2-D planning references with:

```bash
python3 gazebo/tools/generate_path_planning_assets.py
```

## Validation

```bash
python3 gazebo/tools/build_city_collision.py
python3 gazebo/tools/validate_self_contained_maps.py
```

The validation asserts asset hashes, local URI closure, PX4 launch contracts, city
road alignment, deterministic building-height factors and deterministic
mountain geometry. It checks the launch-pad footprint, every tree trunk disk,
the absence of maze entities, both YAML files and both reference images. Generated collision
OBJ files are committed directly so a fresh clone needs no separate LFS
download.
