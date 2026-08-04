# Cheongju University Main Complex Stadium

`worlds/drone_cju.world` contains only Cheongju University's Main Complex
Stadium (campus facility 40). It is an OSM-calibrated procedural model, not a
survey-grade digital twin and not a model of the full campus.

Reference geometry:

| Feature | Approximate size |
| --- | ---: |
| stadium site | 226.8 x 128.3 m |
| running-track outside edge | 179.0 x 87.1 m |
| sand football field | 111.6 x 70.8 m |
| north basketball court | 27.0 x 17.2 m |
| south basketball court | 29.6 x 17.9 m |

The centre is approximately `36.654417, 127.495904` at 74 m elevation and the
long-axis heading is 96.3 degrees from ENU east. The model follows the permanent
layout visible in the [official facility photograph](https://cju.ac.kr/site/www/images/contents/cts6858_01.jpg),
[official stadium VR](https://www2.cju.ac.kr/common/vr/ground/sports_complex/index.html),
and these OSM ways: [site 431113163](https://www.openstreetmap.org/way/431113163),
[track 1374978221](https://www.openstreetmap.org/way/1374978221), and
[field 431605916](https://www.openstreetmap.org/way/431605916).

The stadium has a beige sand field, a red eight-lane mission track, two blue
courts, west royal-box canopy, concrete terraces, green
safety rails, football goal nets, basketball equipment, and perimeter trees.
Temporary festival stages and unsupported floodlight towers are omitted. No
source photo or downloaded texture is embedded; geometry and materials are
generated locally.

Run the integrated experiment:

```bash
./simulation/gazebo/run_gimbal.sh mission
```

Enter `takeoff`, `mission`, `land`. The drone takes off and patrols at 5 m;
`mission` starts the trailer at 3.0 m/s and repeatedly replans an A* patrol from
the vehicle's actual position inside the YAML-defined `226.8 x 128.3 m` stadium
geofence and around the four `4.5 x 0.35 x 10.0 m` barriers.
`land` interrupts planning or patrol and starts the moving-trailer ArUco/MPC
landing sequence.

Regenerate the two canonical runtime models after editing their generators:

```bash
python3 simulation/gazebo/gen_drone_cju_track_models.py
```
