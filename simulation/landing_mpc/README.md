# landing_mpc

PX4-native precision landing on a moving ArUco target. The active CJU stack
plans obstacle topology with A*, reinforces only the spatial geometry with a
B-spline, and hands flight dynamics and landing verdicts to PX4.

`mpc.py`, `predictor.py`, and `reference.py` are retained legacy/research
modules; `run_gimbal.sh mission` does not use them for active vehicle control.

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

Perception + mission nodes (the live stack, launched by
`simulation/gazebo/run_gimbal.sh`):

| node | role |
|------|------|
| `aruco_detector_node` | marker ladder → landing point, camera optical frame |
| `marker_tf_node`      | + vehicle state & gimbal joints → local ENU |
| `marker_kf_node`      | const-velocity KF + coast |
| `trailer_cue_node`    | long-range target cue (gz trailer pose) |
| `gimbal_control_node` | aims the gimbal, publishes joint encoders + TF |
| `mission_manager_node`| A* + geometry-only B-spline route, PX4 Goto and native PRECLAND handoff |

## CJU mission phases

- **Phase 0 — `PRECHECK`**: validate PX4 feedback, the live trailer cue, the
  planner, and Offboard readiness before arming.
- **Phase 1 — `TAKEOFF`**: request PX4 `NAV_TAKEOFF` to 5 m; PX4 owns the climb profile.
- **Phase 2 — map route**: A* supplies obstacle topology, a geometry-only
  B-spline reinforces the spatial path, and PX4 Goto flies it to `(50,50)` and
  `HOVER`. The spline publishes no speed or P/V/A schedule. PX4
  Goto/PositionSmoothing owns the temporal profile through
  `MPC_XY_CRUISE=3`, `MPC_ACC_HOR`, and `MPC_JERK_AUTO`, with
  `MPC_XY_VEL_MAX=10` as the hard horizontal cap.
- **Phase 3 — landing**: `land` builds one A* → geometry-only B-spline
  `RETURN`, then replaces only a validated 2.5 m-safe tail every two seconds
  from the latest live GPS/cue. A full replan is the fallback only when no safe
  tail connector exists; the prior route remains active throughout. Within the validated 6 m
  handoff corridor it publishes `LandingTargetPose`, requests PX4
  `NAV_PRECLAND`, and stops all Offboard setpoints. PX4 owns speed,
  acceleration, attitude, descent, contact detection and auto-disarm. ArUco
  only corrects the reported target position.

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
./simulation/gazebo/run_gimbal.sh mission        # CJU: takeoff → mission → land
./simulation/gazebo/run_gimbal.sh baseline       # CJU body-fixed camera comparison
HEADLESS=1 ./simulation/gazebo/run_gimbal.sh mission
LANDING_MAP=mpc-landing-moving ./simulation/gazebo/run_gimbal.sh mission
```
