# PX4-ready Gazebo maps

The `jo` branch keeps every map world, collision mesh and texture inside this
repository. The main launcher starts Gazebo Harmonic and then PX4 SITL; PX4
dynamically creates the real, motor-driven `x500_city_rgbd_lidar` entity. There
is no static preview-drone substitute in either world.

```bash
git clone -b jo https://github.com/SeyeongW/PX4-ROS2.git
cd PX4-ROS2
./gazebo/setup_px4_sitl.sh       # first PC / one time only
./gazebo/run_px4_map.sh city
# or
./gazebo/run_px4_map.sh mountain

# Mountain-only optional trailer route
DRIVE_TRAILER=1 TRAILER_ROUTE_LOOPS=1 ./gazebo/run_px4_map.sh mountain
```

The setup helper pins a new checkout to the tested PX4 `v1.17.0`, runs PX4's
official Ubuntu general-prerequisite installer (one sudo prompt may appear),
but never changes the branch/tag or tracked source of an existing
`~/PX4-Autopilot`. It only manages the unique
`Tools/simulation/gz/models/x500_city_rgbd_lidar` symlink; a real path at that
location is preserved and reported as an error. Set
`PX4_INSTALL_PREREQS=0` only when those prerequisites are already installed.
The launch uses the stock x500 airframe `4001`, the repository's
`x500_city_rgbd_lidar` model, standalone Gazebo mode, and the spawn pose from the
checked-in YAML.
The default is MAVROS / MAVLink only. The launcher passes a validated runtime
copy of PX4's rcS through `px4 -s` with only the unconditional DDS start line
removed, so the client never starts and cannot emit a transient no-agent
error. The PX4 source/build remains untouched. Set `START_XRCE=1` to use the
stock rcS and Agent explicitly.
In GUI mode the camera starts close to the spawn but remains freely
controllable; it is not locked to the vehicle. Set `FOLLOW_DRONE=1` only when
an explicit PX4 follow view is wanted.

Map-only inspection is still available with `./gazebo/run_world.sh city` or
`mountain`; the city trailer is stationary at `(-587,-512)`, and those commands
intentionally contain no PX4 drone. The original 500 m source city remains
available as `./gazebo/run_world.sh city-legacy`. Use
`run_px4_map.sh` whenever a flyable PX4 vehicle is required.

The former experimental 3D planner / SFC / B-spline / MPC stack has been
retired from this branch. This package now keeps the city geometry contract,
PX4 vehicle and sensor runtime only.

The equivalent direct commands below force rendering onto the NVIDIA GPU on a
hybrid laptop without using the launcher script:

```bash
# Mountain
GZ_SIM_RESOURCE_PATH="$PWD/gazebo/models:$PWD/gazebo/worlds" __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only QT_QPA_PLATFORM=xcb gz sim -v4 -r --physics-engine gz-physics-bullet-featherstone-plugin "$PWD/gazebo/worlds/ugv_drone_map.world"

# Current UAV city
GZ_SIM_RESOURCE_PATH="$PWD/gazebo/models:$PWD/gazebo/worlds" __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only QT_XCB_GL_INTEGRATION=xcb_glx QT_QPA_PLATFORM=xcb gz sim -v4 -r --physics-engine gz-physics-dartsim-plugin "$PWD/gazebo/worlds/applepark_city_uav/applepark_uav.world"
```

## Active UAV city

- world: `gazebo/worlds/applepark_city_uav/applepark_uav.world`
- coordinate contract: `gazebo/maps/city_coordinates_uav.yaml`
- flat ground: 1300 x 1300 m at exactly `z=0` (`-650..650 m` ENU)
- 205 active buildings (69 / 274 removed, the nearest integer to one quarter);
  retained XY footprints and centroids both use a uniform `2.5x` transform of
  the initial `origin/main` city
- building IDs and centroid XY coordinates remain unchanged from the preceding
  active layout; foundations stay at `-0.05 m`, while only roof heights are
  remapped from the source hash order into `30..70 m`
- removals use seed `7577`, 5x5 spatial Hamilton quotas and stable SHA-256
  ranking; all 25 regions contain removals, so no artificial diagonal corridor
  is baked into the map
- every retained building uses the deterministic hash-rank `30..70 m`
  skyline; foundations extend from `-0.05 m` to the flat `z=0` datum
- the visual is one closed triangulated DAE; 205 exact DART polyline-prism
  collisions use the same YAML rings, so the courtyard remains open and all footprints remain disjoint,
  the physical minimum gap is `0.971942 m`, and invisible
  outward/undercoverage error is zero
- expanded building footprints cover zero strict-asphalt pixels; `2.5x` is the
  largest reference-faithful scale before road conflicts begin
- PX4 model-root spawn: `(587, 580, 0)` on the north-east road end
- trailer spawn: `(-587, -512, 0)` on the opposite south-west road end;
  separation is `1603.352737 m`, both sites are asphalt-only and retain at
  least `63 m` center clearance from the restored map boundary
- `(-600,-600)` has 50 m of visual and physical ground to each nearby edge;
  the former dark affine-fill border is replaced by a smooth
  edge-clamped fade without changing any building coordinate
- A* polygon rings and vertical limits are exported deterministically to
  `gazebo/maps/city_uav_building_vertices.csv`; the fixed goal is
  `(200,-128)` and its former blocking structure (`building_265`) is removed
- DART cannot load this city's DAE directly as collision in Gazebo Harmonic
  8.14, so the DAE remains visual-only while exact DART-supported polyline
  prisms provide physical buildings without destabilizing PX4 flight
- the trailer has zero commanded mean speed in the city profile; the retained
  SEO controller adds its stock small force/torque perturbations after PX4 spawns
- the GUI includes `GzSceneManager`, so the checked-in custom GUI renders the
  scene instead of opening a black 3D panel

## Legacy source city

- world: `gazebo/worlds/applepark_city/applepark.world`
- size: 500 x 500 m
- 274 buildings using a deterministic hash-rank skyline bounded to `10..20 m`;
  historical `2.001288..3.476403x` factors remain in the audit CSV only
- every building XY footprint remains unchanged; all foundations span
  `z=-0.05..0` and roofs are `10..20 m` above the common datum
- visual heightmap and Bullet collision OBJ are both completely flat at `z=0`
- PX4 model-root spawn: `(-120, 115, 0)`; settled `base_link` / PX4 local
  origin: `(-120, 115, 0.24)`; trailer spawn: `(-175, 140, 0)`
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
DAE prism 0.05 m below it. The latest pass preserves all roads, XY coordinates
and flat foundations while applying the deterministic `jo` skyline factor to
every roof vector.
Foundation values are recorded in
`gazebo/validation/city/building_foundation_alignment.csv`; the stable
SHA-256-derived factor, old roof and new roof for every component are in
`gazebo/validation/city/building_height_scaling.csv`.

The visual asset is an all-255 256×256 heightmap with `<size_z>0.001` and
`<pos_z>-0.001`; OGRE2 therefore draws its top at exactly `z=0` while its
otherwise-visible heightmap skirt is only 1 mm deep. The terrain visual does
not cast a large backing-volume shadow. The committed 129×129 Bullet OBJ has
the same `z=0` value at every vertex.

## Mountain

- world: `gazebo/worlds/ugv_drone_map.world`
- size: 300 x 300 m
- retained 40 m / 20.078 m summits plus six broad deterministic ridges
- actual collision-triangle grade is p99 `66.93%` and max `78.05%`
- 288 uniformly 2x-scaled trees seated with a 3 mm burial allowance; the old
  short tree-contact shelves which caused slope spikes are gone
- artificial maze removed (zero maze collisions and zero maze visuals)
- PX4 model-root spawn: `(-80, -80, 0.16)`; settled `base_link` / PX4 local
  origin: `(-80, -80, 0.40)`; trailer spawn: `(0, 0, 0)`
- exact-flat trailer corridor: ellipse `(x/42)^2 + (y/30)^2 <= 1`

For the actual PX4-controlled x500 with downward monocular camera, run:

```bash
./gazebo/run_px4_map.sh mountain
```

## Trailer waypoint control

The city now uses `seo`'s default `moving_platform_aruco` PX4 landing-platform
trailer (5 x 5 m deck, 2.05 m top, 1 x 1 m ArUco marker). Its original
`MovingPlatformController` is retained, while the city launcher defaults
`PX4_GZ_PLATFORM_VEL=0`, so its commanded mean speed is zero unless the
operator explicitly requests motion. The stock SEO controller still adds
small force/torque perturbations after the PX4 vehicle appears; map-only runs
remain fixed while the controller waits for that vehicle. It is not a wheeled vehicle. The
mountain retains its optional VelocityControl landing-platform route. The
standalone mountain driver uses Gazebo Transport directly and does not require
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

Exact world ENU coordinates, PX4 spawn-relative NED conversion, all 205 active
city building polygons (including the courtyard hole), the 274-building legacy
source, and all 288 mountain tree collision cylinders are exported to:

- `gazebo/maps/city_coordinates_uav.yaml` (active UAV city, full XYZ AABBs)
- `gazebo/maps/city_uav_building_vertices.csv` (ordered A* outer/hole vertices and vertical limits)
- `gazebo/maps/city_coordinates.yaml`
- `gazebo/maps/mountain_coordinates.yaml`

Regenerate the active YAML/world/mesh and its 2-D reduction reference with:

```bash
python3 gazebo/tools/expand_city_for_uav.py
python3 gazebo/tools/render_city_uav_reference.py
```

## Validation

```bash
python3 gazebo/tools/build_city_collision.py
python3 gazebo/tools/flatten_city_assets.py --check
python3 gazebo/tools/validate_city_uav_expansion.py
python3 gazebo/tools/validate_self_contained_maps.py
```

The validation asserts asset hashes, local URI closure, PX4 launch contracts,
city road alignment, the active deterministic `30..70 m` skyline, retained
historical height-factor audit data, and deterministic mountain geometry. It
checks PX4 spawn clearance, the spatial-random reduction audit, all exact DART
city collision prisms, every tree trunk disk, the absence of maze entities,
the YAML/CSV coordinate files and the planning
references. Generated assets are committed directly so a fresh clone needs no
separate LFS download.
