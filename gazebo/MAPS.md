# Self-contained Gazebo maps

The `jo` branch keeps every runtime world, mesh and texture required for map
preview inside this repository. Gazebo Harmonic is the only simulator needed
for the two preview commands.

```bash
git clone -b jo https://github.com/SeyeongW/PX4-ROS2.git
cd PX4-ROS2
./gazebo/run_world.sh city
# or
./gazebo/run_world.sh mountain
```

Both previews include the repository-local `map_preview_drone` on the launch
pad, so a fresh clone displays the map and a drone without Fuel downloads or
an `ardupilot_gazebo` checkout. This is a static map marker rather than a
flyable vehicle. Its model origin is normalized to the landing-gear contact
plane, which is seated exactly on each pad surface.

The equivalent direct commands below force rendering onto the NVIDIA GPU on a
hybrid laptop without using the launcher script:

```bash
# Mountain
GZ_SIM_RESOURCE_PATH="$PWD/gazebo/models:$PWD/gazebo/worlds" __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only QT_QPA_PLATFORM=xcb gz sim -v4 -r --physics-engine gz-physics-bullet-featherstone-plugin "$PWD/gazebo/worlds/mountain_map.world"

# City
GZ_SIM_RESOURCE_PATH="$PWD/gazebo/models:$PWD/gazebo/worlds" __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only QT_QPA_PLATFORM=xcb gz sim -v4 -r --physics-engine gz-physics-bullet-featherstone-plugin "$PWD/gazebo/worlds/city_map/city_map.world"
```

## City

- world: `gazebo/worlds/city_map/city_map.world`
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
`worlds/city_map/OSM_ATTRIBUTION.txt` and
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

- world: `gazebo/worlds/mountain_map.world`
- size: 300 x 300 m
- retained 40 m / 20.078 m summits plus seven deterministic branching ridges
- 288 uniformly 2x-scaled trees on exact protected contact patches (the
  original 72-wall central maze was removed 2026-07-13; see MOUNTAIN_WORLD.md)
- spawn: `(-80, -80)`

For an actual ArduPilot-controlled Iris, install `ardupilot_gazebo` and run:

```bash
./gazebo/run_world.sh mountain-sitl
```

The original convenience command `./gazebo/run_ugv_drone.sh` uses SITL when
the external model/plugin is installed and otherwise falls back to the local
preview automatically.

## Validation

```bash
python3 gazebo/tools/build_city_collision.py
python3 gazebo/tools/validate_self_contained_maps.py
```

The validation asserts asset hashes, local URI closure, drone presence, city
road alignment, deterministic building-height factors and deterministic
mountain geometry. It samples every maze footprint, launch-pad footprint and
every tree trunk disk against the final quantized terrain. Generated collision
OBJ files are committed directly so a fresh clone needs no separate LFS
download.
