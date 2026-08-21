# landing_mpc

Relative-coordinate translational **MPC** for precision landing on a **moving
ArUco target** (~3 m/s, curved manoeuvre). PX4 SITL + Offboard.

Built from `MPC_동적착륙_설계문서.docx` / `LANDING_MPC_SPEC.md`. The core idea:
precision landing on a mover is a **relative-frame rendezvous (docking)** — drive
relative position → 0 *and* relative velocity → 0. The MPC is a translational
double integrator (position + velocity state, 3-axis acceleration input); PX4's
attitude loop handles the quaternion dynamics (differential flatness, §2.1).

## Files (spec §6)

| file | role |
|------|------|
| `model.py`     | discrete double integrator in relative coords, condensing maps |
| `predictor.py` | target prediction: const-vel / const-accel / sliding-window poly |
| `mpc.py`       | cost + constraints, two-stage convex QP, receding horizon |
| `reference.py` | MPC plan → 50 Hz interpolated setpoint stream |
| `frame.py`     | ENU↔NED conversion + gimbal joint chain, unit-tested (§4) |

Perception + mission nodes (the live stack, launched by `gazebo/run_gimbal.sh`):

| node | role |
|------|------|
| `aruco_detector_node` | marker ladder → landing point, camera optical frame |
| `marker_tf_node`      | + vehicle state & gimbal joints → local ENU |
| `marker_kf_node`      | const-velocity KF + coast |
| `trailer_cue_node`    | long-range target cue (gz trailer pose) |
| `gimbal_control_node` | aims the gimbal, publishes joint encoders + TF |
| `mission_manager_node`| phase sequencing, MPC descent, the single setpoint authority |

## Design notes (do not "improve" without rebutting the rationale — spec §2)

- **Relative coordinates** (`p = p_drone - p_target`): fuses "chase" and "align"
  into one problem. Absolute PX4 setpoint = target prediction + relative command.
- **Target-accel feed-forward** enters the free response (net accel =
  `a_drone - a_target`): the key to tracking a curved target without lag (§3.1).
- **Safety cone** `p_z ≥ k·‖p_xy‖` gates descent on horizontal alignment (docking
  approach corridor). To keep every solve a QP, it is applied via a **two-stage**
  solve: xy first, then z with a *linear* per-step lower bound from the predicted
  `‖p_xy‖`. No nonlinear program, no local minima.
- **Terminal cost** `w_f(‖p_N‖² + ‖v_N‖²)` drives the horizon end to zero
  relative position *and* velocity — a soft touchdown.

## Verify (no PX4/ROS needed)

```bash
python3 -m landing_mpc.frame    # ENU<->NED + gimbal chain unit test
```

## PX4 SITL — landing on the moving ArUco trailer

One command brings up Gazebo + PX4 + the perception chain + the mission, with
every switch already set:

```bash
./gazebo/run_gimbal.sh mission        # gimbal vehicle, moving trailer
./gazebo/run_gimbal.sh baseline       # body-fixed camera, for comparison
HEADLESS=1 ./gazebo/run_gimbal.sh mission
```

Score a run against Gazebo ground truth (not the mission's own estimate):

```bash
python3 scratchpad/landing_center_check.py
```

Measured: touches down on the 3 m/s deck every run, 0.42-0.78 m from the deck
centre. See `docs/nodes.md` for the per-node derivations.
