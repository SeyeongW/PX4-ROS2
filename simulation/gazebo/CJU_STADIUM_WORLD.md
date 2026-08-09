# Cheongju University Stadium World

`worlds/drone_cju.world` is a minimal stadium-only map. Its visible fixed map
contains one low ground pad, the field, red running track, basketball court,
and jokgu court.
Mission barriers, the moving trailer, and the drone remain because they are
runtime experiment assets rather than stadium decoration.

The canonical local frame is `stadium_endpoint`. Its `(0, 0)` is OSM node
`12730808466`, the south-east running-track tangent at the bottom-right of a
north-up photo/map, at WGS84 `36.6540480, 127.4964451`. Local +x points about
`6.307 deg` north of east and +y follows the stadium long axis to the north.
The map therefore extends mainly left (`-x`) and up (`+y`) from the origin.
Mission anchors, facility centres, and the enlarged jokgu dimensions stay on
integer metres; the smooth OSM-scale track retains its measured decimal values.

| Feature | Centre `(x, y)` m | Size / bounds m |
| --- | ---: | ---: |
| running track / route centre | `(-44, 46)` | `x=[-87.4,0]`, `y=[-43.35,135.35]` |
| stadium field | `(-44, 48)` | `68 x 105` |
| basketball court | `(-44, 113)` | `28 x 15` |
| jokgu court | `(-44, -18)` | `10 x 20` |
| support ground | `(-42, 49)` | `128 x 226` |

The 48-segment-per-half-curve outline follows the measured scale of
[OSM track way 1374978221](https://www.openstreetmap.org/way/1374978221) and
[site way 431113163](https://www.openstreetmap.org/way/431113163), not a
survey-grade digital twin. The red surface has eight 1.22 m lanes and nine
continuous white boundaries. The basketball playing surface uses the
[FIBA 28 x 15 m specification](https://www.venueguide.fiba.basketball/vanue-design).
The jokgu surface is intentionally enlarged to `10 x 20 m`, aligned vertically
with the stadium long axis, and kept fully inside the running track.
Stands, royal box, canopy, rails, trees, goals, court equipment, and plaza
geometry remain intentionally omitted.

Run the integrated experiment:

```bash
./simulation/gazebo/run_gimbal.sh mission
```

Enter `takeoff`, `mission`, and `land`. The drone patrols at 5 m around barriers
centred at `(-47,2)`, `(-41,12)`, `(-47,22)`, and `(-41,32)`. A 1 m planner
grid keeps mission waypoints integer-valued; the trailer moves at 3 m/s and
the landing phase uses ArUco/MPC.

Regenerate the runtime models after editing their generator:

```bash
python3 simulation/gazebo/gen_cju_stadium_model.py
python3 simulation/gazebo/gen_drone_cju_track_models.py
```
