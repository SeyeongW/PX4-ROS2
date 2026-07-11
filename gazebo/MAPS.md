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
an `ardupilot_gazebo` checkout.

## City

- world: `gazebo/worlds/applepark_city/applepark.world`
- size: 500 x 500 m
- 274 buildings with deterministic 2–5x height scaling
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

## Mountain

- world: `gazebo/worlds/ugv_drone_map.world`
- size: 300 x 300 m
- smooth 40 m and 20 m summits, 72 trees and 72 maze walls
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
road alignment and deterministic mountain geometry. Generated collision OBJ
files are stored with Git LFS.
