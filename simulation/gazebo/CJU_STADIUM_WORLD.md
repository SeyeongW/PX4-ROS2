# Cheongju University Stadium World

`worlds/drone_cju.world` is a minimal stadium-only map. Its visible fixed map
contains one low ground pad, the field, red running track, basketball court,
and jokgu court.
Mission barriers, the moving trailer, and the drone remain because they are
runtime experiment assets rather than stadium decoration.

The canonical local frame is `stadium_endpoint`. Its `(0, 0)` is the integer
south-west staging point beside the running track, at approximately WGS84
`36.653960886920, 127.495466874950`. Local +x points about `6.307 deg` north of
east and +y follows the stadium long axis to the north. This puts the stadium
centre at the simple positive coordinate `(44, 46)`, close to `(50, 50)`.
Mission anchors and facility centres stay on integer metres; the smooth
OSM-scale track retains its measured decimal dimensions.

| Feature | Centre `(x, y)` m | Size / bounds m |
| --- | ---: | ---: |
| running track / route centre | `(44, 46)` | `x=[0.6,88]`, `y=[-43.35,135.35]` |
| stadium field | `(44, 48)` | `68 x 105` |
| basketball court | `(44, 113)` | `28 x 15` |
| jokgu court | `(44, -18)` | `10 x 20` |
| support ground | `(46, 49)` | `128 x 226` |

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

Enter `takeoff`, `mission`, then `land`. Phase 0 `PRECHECK` validates PX4 feedback, the
live cue, planner, and Offboard readiness; Phase 1 requests PX4
`NAV_TAKEOFF` to 5 m and holds `READY`. In Phase 2, `mission` makes YAML A* provide obstacle topology, a validated
geometry-only B-spline reinforces that spatial path, and TrackingMPC tracks it to
`(50,50)` and `HOVER`. The `stadium_endpoint` remains `(0,0)`;
the drone and trailer start together on the track centre at the integer map
coordinate `(5,0)`, with the drone on the 2.051 m deck. The drone flies at
0.5 m/s to 5 m, then uses 1 m-grid A* topology plus the geometry B-spline/
TrackingMPC route to `(50,50)` around twenty
`0.45 x 0.35 x 10 m` barriers at seed-5053-based integer centres sampled from
`(0,0)` to `(50,50)` and filtered inside the infield, with barrier 2 manually
placed above barrier 17: `(33,10)`, `(18,39)`,
`(21,25)`, `(49,41)`, `(31,36)`, `(45,11)`, `(39,21)`, `(42,50)`, `(17,18)`,
`(22,33)`, `(44,33)`, `(24,12)`, `(39,0)`, `(30,23)`, `(49,1)`, `(21,3)`,
`(15,34)`, `(42,41)`, `(35,49)`, and `(28,46)`. The trailer moves
50 m forward along the stadium long axis and 50 m backward at 1 m/s, repeating
that fixed segment. A*/B-spline, return-tail, and landing-entry checks use a
1.5 m Euclidean XY radius around each physical obstacle AABB; runtime checks
retain the hard 1 m radius. In Phase 3 the gimbal remains nadir beyond 10 m,
blends toward the trailer over 10→9 m of horizontal GPS/cue range, and points directly inside
9 m. `land` stays in `RETURN` until three distinct KF-accepted ArUco fixes arrive
within 0.5 s and the live cue segment passes the 1.5 m planning-clearance check.
LandingMPC then acquires at fixed
altitude, descends after alignment, and hands the low-altitude final approach to
PX4 `NAV_PRECLAND`. PX4 owns contact detection and auto-disarm.

The B-spline owns no flight speed or P/V/A schedule. TrackingMPC derives a
braking reference from the accepted spatial path, while PX4 retains the
lower-level position, velocity, attitude, and motor-control loops.

Regenerate the runtime models after editing their generator:

```bash
python3 simulation/gazebo/gen_cju_stadium_model.py
python3 simulation/gazebo/gen_drone_cju_track_models.py
```
