# 1 km moving-trailer forward/reverse shuttle

The default `mpc-landing-moving` profile is the repository's primary
moving-target landing environment. Its legacy file and internal world name
`mpc_landing_200m_moving` are retained for ROS and Gazebo topic compatibility.
The current experiment is a straight, obstacle-free 1,000 m shuttle.

## Coordinate and ground contract

- Nominal ENU bounds: `x=[0,1000]`, `y=[0,100] m`
- Visible ground size: `1,000 x 100 m`, centred at `(500,50)`
- Invisible collision support: `1,010 x 110 m`, centred at `(500,50)`
- Ground datum: `z=0`
- Trailer footprint: `5 x 5 m`
- Obstacles: none

The collision support extends 5 m past every nominal map edge. At the
`x=0` and `x=1000` reversal planes, the full platform footprint is therefore
supported with another 2.5 m to the physical collision edge. The yellow
boundary lines continue to mark the exact nominal 1,000 x 100 m experiment.

## Spawn and motion contract

- Trailer spawn and west endpoint: `(0,50,0)`
- East endpoint: `(1000,50,0)`
- Drone spawn: `(15,40,0)`
- Initial horizontal separation: `18.028 m`
- Outbound leg: 1,000 m along ENU +x
- Inbound leg: 1,000 m along ENU -x
- Complete cycle: 2,000 m
- Trailer yaw: fixed at `0`; reversal changes velocity sign without turning
- Default cruise speed: `3 m/s`
- Acceleration limit: `2 m/s²`
- Command rate: `50 Hz`
- Repetition: continuous until stopped

At cruise speed, the nominal leg and cycle times are 333.33 s and 666.67 s,
respectively; acceleration, braking, and endpoint reversal add a small amount
to elapsed runtime. `TRAILER_ROUTE_LOOPS=1` means one complete outbound and
return cycle, not a single 1,000 m leg.

The blue ground line is the exact `y=50` shuttle centreline. Two yellow discs
mark the west and east reversal planes. The platform is a fixed-yaw
VelocityControl model, so negative-x motion on the inbound leg is true
translational reverse rather than a visually hidden 180-degree turn.

## Regenerate

```bash
python3 simulation/gazebo/gen_mpc_perimeter_patrol_model.py
```

## Run

Map-only preview:

```bash
./simulation/gazebo/run_world.sh mpc-landing-moving
```

PX4 and continuously shuttling trailer:

```bash
DRIVE_TRAILER=1 ./simulation/gazebo/run_px4_map.sh mpc-landing-moving
```

One complete 2,000 m cycle validation:

```bash
DRIVE_TRAILER=1 TRAILER_ROUTE_LOOPS=1 \
  ./simulation/gazebo/run_px4_map.sh mpc-landing-moving
```

Gimbal perception with the moving trailer:

```bash
./simulation/gazebo/run_gimbal.sh
```

The Gazebo camera is free by default. Opt in to tracking only when required:

```bash
FOLLOW_DRONE=1 ./simulation/gazebo/run_gimbal.sh
```

`run_gimbal.sh` selects `mpc-landing-moving` by default and starts this shuttle
with `DRIVE_TRAILER=1`. The exact machine-readable experiment contract and
paper-oriented evaluation metrics are in
`maps/mpc_landing_200m_moving.yaml`.
