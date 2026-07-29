# Cheongju University stadium moving-landing variant

`cju-track` is an original, low-poly reconstruction inspired by
Cheongju University's **Main Complex Stadium (building 40)**. It is not a
survey-grade digital twin, and it is not the similarly named municipal
Cheongju Stadium.

The stadium is preserved in `DRONE_CJU_TRACK.world` with the internal world
name `DRONE_CJU_TRACK`. The physical ground is 280 x 280 m to contain the
north-south stadium and its simplified outer landmarks. The default
`mpc-landing-moving` profile is now the separate 1,000 x 100 m obstacle-free
forward/reverse shuttle benchmark.

## Reference facts

The university's published material describes a stadium completed in 2000
with a 400 m track, football field, approximately 10,000 seats, a covered
royal-box building, broadcast and lighting rooms, storage, changing and shower
facilities, and a section of natural-stone seating. The official facility
photograph and ground-level VR show a light tan dirt field, a mostly dark-gray
track, a reddish straight section, a blue covered stand, open concrete /
natural-stone stands, trees, football goals, and a blue court near one curve.

Sources used for visual reference:

- [Cheongju University facility page](https://www.cju.ac.kr/www/contents.do?key=6858)
- [Cheongju University campus map](https://www.cju.ac.kr/site/campusmap/sub.jsp)
- [Cheongju University 70-year history PDF](https://www.cju.ac.kr/DATA/download/history/cts5453_file7.pdf)
- [Official ground-level stadium VR](https://www.cju.ac.kr/common/vr/ground/sports_complex/index.html)
- [OpenStreetMap location](https://www.openstreetmap.org/?mlat=36.65440&mlon=127.49591#map=18/36.65440/127.49591)

The official images and VR are references only. They are not copied, embedded,
or redistributed by this repository. The model uses newly generated SDF
primitives and plain materials. OpenStreetMap is credited under its
[copyright and ODbL terms](https://www.openstreetmap.org/copyright).

## Simulation approximations

- WGS84 centre: approximately `36.65440, 127.49591`, elevation `74 m`
- Stadium long-axis heading: `96.3 deg` from ENU +x, equivalent to about
  `6.3 deg` west of north
- Track centreline: `84.39 m` straights and `36.8 m` semicircle radius,
  giving `400.001 m` per lap
- Eight lanes are a visual estimate; the source material does not publish a
  lane count
- Buildings, stands, court, trees, and floodlights are deliberately simplified
  landmarks, not surveyed dimensions
- Track, field, and painted lines are visual-only so their thin surfaces do not
  change the exact `z=0` landing-vehicle contact datum
- The existing 5 x 5 m platform keeps yaw at zero because its Gazebo
  `VelocityControl` consumes linear commands in model axes. A directional
  truck/trailer mesh should be added only together with body-frame velocity
  conversion and yaw steering.

The trailer starts at the southern point of the track at
`(37.0710678, 37.0710678)`, preserving the existing exact 10 m initial
separation from the PX4 spawn at `(30, 30)`. Its entity name, ArUco geometry,
deck height, command topic, odometry topic, and dynamic-pose topic are
unchanged.

## Run

Map-only inspection:

```bash
./simulation/gazebo/run_world.sh cju-track
```

PX4 plus a 3 m/s moving platform:

```bash
DRIVE_TRAILER=1 \
TRAILER_SPEED_M_S=3 \
./simulation/gazebo/run_px4_map.sh cju-track
```

Full gimbal landing experiment:

```bash
LANDING_MAP=cju-track ./simulation/gazebo/run_gimbal.sh
```

Regenerate the original procedural assets after editing a generator:

```bash
python3 simulation/gazebo/gen_drone_cju_track_models.py
```
