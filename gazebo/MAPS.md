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

# Spawn PX4 and drive the included seo-derived trailer for one route
DRIVE_TRAILER=1 TRAILER_ROUTE_LOOPS=1 ./gazebo/run_px4_map.sh city
```

The setup helper pins a new checkout to the tested PX4 `v1.17.0`, runs PX4's
official Ubuntu general-prerequisite installer (one sudo prompt may appear),
but never changes or checks out an existing `~/PX4-Autopilot`. Set
`PX4_INSTALL_PREREQS=0` only when those prerequisites are already installed.
The launch uses airframe
`4014`, standalone Gazebo mode, and the spawn pose from the checked-in YAML.
It also starts the local Micro XRCE-DDS Agent when available, enabling PX4
ROS 2 `/fmu/*` topics; MAVLink / PX4 flight remains available without it.

Map-only inspection is still available with `./gazebo/run_world.sh city` or
`mountain`; the trailer is part of each world, but those commands intentionally
contain no PX4 drone. Use
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
- every building XY footprint and randomized height remains unchanged; all
  foundations now span `z=-0.05..0` and roofs are
  `15.907114..109.621338 m` above the common datum
- visual heightmap and Bullet collision OBJ are both completely flat at `z=0`
- PX4 spawn: `(-120, 115, 0.16)`; trailer spawn: `(-175, 140, 0)`
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

The old sloped source terrain produced downhill gaps and uphill intersections.
The trailer-safe pass now uses one exact world datum and extends every checked
DAE prism 0.05 m below it. The latest height pass preserves all roads and XY
coordinates while retaining the raised 13,872 roof vectors.
Foundation values are recorded in
`gazebo/validation/city/building_foundation_alignment.csv`; the stable
SHA-256-derived factor, old roof and new roof for every component are in
`gazebo/validation/city/building_height_scaling.csv`.

The visual asset is an all-255 257×257 heightmap with `<size_z>1` and
`<pos_z>-1`; OGRE2 therefore draws it at exactly `z=0`. The committed 129×129
Bullet OBJ has the same `z=0` value at every vertex.

## Mountain

- world: `gazebo/worlds/ugv_drone_map.world`
- size: 300 x 300 m
- retained 40 m / 20.078 m summits plus six broad deterministic ridges
- actual collision-triangle grade is p99 `66.93%` and max `78.05%`
- 288 uniformly 2x-scaled trees seated with a 3 mm burial allowance; the old
  short tree-contact shelves which caused slope spikes are gone
- artificial maze removed (zero maze collisions and zero maze visuals)
- PX4 spawn: `(-80, -80, 0.16)`; trailer spawn: `(0, 0, 0)`
- exact-flat trailer corridor: ellipse `(x/42)^2 + (y/30)^2 <= 1`

For the actual PX4-controlled x500 with downward monocular camera, run:

```bash
./gazebo/run_px4_map.sh mountain
```

## Trailer waypoint control

The model selectively ported from `origin/seo` is not a wheeled trailer; it is
a 5×5 m VelocityControl moving landing platform. Both maps include it as
`flat_platform`, and its operational routes stay on collision-checked planes.
The standalone driver uses Gazebo Transport directly and does not require
MAVROS:

```bash
# Terminal 1
GZ_PARTITION=trailer_test ./gazebo/run_world.sh mountain

# Terminal 2: exact-flat cross route
GZ_PARTITION=trailer_test ./gazebo/trailer_waypoint_driver.py mountain --loops 1

# Optional grade-following safeguard validation outside the central plane
GZ_PARTITION=trailer_test ./gazebo/trailer_waypoint_driver.py mountain --route slope --loops 1
```

The grade-following mode samples a 6×6 m envelope below the body and commands
vertical velocity before terrain can penetrate the platform. This prevents the
contact impulse that previously launched it, but it does not turn the platform
into a physical wheeled rover.

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
python3 gazebo/tools/flatten_city_assets.py --check
python3 gazebo/tools/validate_self_contained_maps.py
```

The validation asserts asset hashes, local URI closure, PX4 launch contracts, city
road alignment, deterministic building-height factors and deterministic
mountain geometry. It checks the launch-pad footprint, every tree trunk disk,
the absence of maze entities, both YAML files and both reference images. Generated collision
OBJ files are committed directly so a fresh clone needs no separate LFS
download.
