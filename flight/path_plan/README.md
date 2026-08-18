# path_plan

Global + local path generation for the PX4 city UAV mission (`applepark_city_uav`
map). Pipeline of four ROS 2 nodes:

```
A*  ──global_path──▶  SFC  ──corridor──▶  B-spline  ──trajectory──▶  MPC ──cmd_vel──▶ PX4
(grid search)      (safe boxes)        (smooth curve)          (tracking + depth avoid)
```

- **Cruise band 20–30 m AGL.** The occupancy model inflates every building roof
  upward by `vertical_margin + roof_clearance` (≈10 m), so under a 30 m ceiling
  nothing above ~19.6 m can be overflown. This map's buildings are 15.9–109 m
  (70 % > 30 m), so avoidance is effectively **lateral**; buildings that reach
  into the band are routed around, shorter ones are still treated as obstacles.

## Layout

| File | Role |
|------|------|
| `world_model.py` | AABB obstacle field, free-space queries, city-YAML loader |
| `astar.py` / `astar_node.py` | global grid A\* |
| `sfc.py` / `sfc_node.py` | safe flight corridor (free convex boxes) |
| `uniform_bspline.py` + `bspline_optimizer.py` / `bspline_node.py` | ego-planner corridor B-spline (L-BFGS + rebound) |
| `mpc_ros.py` / `mpc.py` / `mpc_node.py` | unicycle tracking MPC (+ double-integrator alt.) |
| `ros_msgs.py` | (de)serialisation to std/nav/geometry msgs |
| `launch/path_plan.launch.py` | wires the pipeline |
| `config/city_uav.yaml` | per-node parameters |

Per-node algorithms, numerical characteristics, and parameter tuning:
[`docs/nodes.md`](docs/nodes.md).

---

# Running

### 0. Build (once, from the workspace root)
```bash
cd ~/ros2_ws/PX4-ROS2
colcon build --packages-select path_plan
source install/setup.bash
```

### 1. Offline — no ROS/sim (fastest way to test & see figures)
Runs A\* → SFC → B-spline → MPC on the map and writes PNGs to `figures/`:
```bash
cd path_plan
PYTHONPATH=$PWD python3 tools/visualize_pipeline.py \
    --start 587 580 25  --goal -300 -300 25            # add --waypoints below
```

### 2. The Gazebo + PX4 simulation (separate terminal)
```bash
./gazebo/run_px4_map.sh city          # Gazebo Harmonic + PX4 SITL + bridges
```
Spawn is the map's `city_drone_spawn` (currently ENU ≈ `587, 580`).

### 3. The planning pipeline (ROS 2)
```bash
ros2 launch path_plan path_plan.launch.py
```
Wiring: `astar → /path_plan/global_path → bspline → /path_plan/trajectory → mpc`.
Feed the MPC from the sim and consume its command:
- **in**  `/path_plan/odometry` (`nav_msgs/Odometry`, ENU pose + world vel)
- **in**  `/path_plan/depth` (`sensor_msgs/Range`, forward nearest obstacle)
- **out** `/path_plan/cmd_vel` (`geometry_msgs/TwistStamped`, vx,vy,vz + yaw-rate)
  → bridge to a PX4 OFFBOARD `TrajectorySetpoint` (bridge TBD).

Run a single node with a config, e.g.:
```bash
ros2 run path_plan astar_node --ros-args \
    --params-file install/path_plan/share/path_plan/config/city_uav.yaml \
    -p map_yaml:=$PWD/gazebo/maps/city_coordinates_uav.yaml
```

# Route: start / goal / waypoints

Set the route in **`config/city_uav.yaml`** under `/astar_planner` (ENU metres):

```yaml
start_enu_m: [587.0, 580.0, 25.0]     # usually the PX4 spawn
goal_enu_m:  [-300.0, -300.0, 25.0]
waypoints_enu_m: []                   # optional ordered via-points
```

- **z (altitude)**: use the 20–30 m cruise band → `25` is mid-band.
- **bounds**: keep inside the map geofence (this map: ±630 m). Points must be
  collision-free (A\* logs `start/goal cell blocked` otherwise).
- **waypoints**: a flat list `[x,y,z, x,y,z, ...]`. The path is **forced through
  each** (`AStarPlanner3D.plan_through`), so single start→goal always finds a
  near-straight line — to make a **hard/slalom route**, drop via-points *off*
  the straight line and near building gaps, e.g.:
  ```yaml
  waypoints_enu_m: [300.0, 300.0, 25.0,  0.0, 100.0, 25.0,  -150.0, -100.0, 25.0]
  ```
- **at runtime**: publish a `geometry_msgs/PoseStamped` to `astar_planner/goal`
  to replan to a new goal live.
- **offline preview**: same via-points on the visualizer:
  ```bash
  PYTHONPATH=$PWD python3 tools/visualize_pipeline.py \
      --start 587 580 25 --goal -300 -300 25 \
      --waypoints 300 300 25  0 100 25  -150 -100 25
  ```

> Map note: the current city map has 205 buildings, all 10–20 m tall; because the
> planner enforces ~10 m roof clearance, cruise at 25 m still routes laterally
> around most of them.

---

# Math ⇄ Code

Notation: ENU world frame, metres; per-axis where axes decouple.

## 1. Occupancy — `world_model.py`

Vehicle-centre free test (Minkowski-inflated obstacles):

```
free(p) ⇔ p ∈ [bounds_min, bounds_max]  ∧  ∀ box B:  p ∉ B
```

- Point-in-box (all-axis interval test) and its vectorised negation →
  `WorldModel.is_free` (`p[:,None,:] ≥ boxes_min ∧ p[:,None,:] ≤ boxes_max`).
- Roof inflation `z1 + vertical_margin + roof_clearance` → `from_city_yaml`
  (this is what makes the 20–30 m band lateral).
- AABB–AABB overlap (separating axis): `overlap ⇔ ∀axis lo ≤ B.max ∧ hi ≥ B.min`
  → `WorldModel.box_is_free` (used by SFC).

## 2. A\* — `astar.py`

Structured like **BigZaphod/AStar**: a domain-agnostic core plus injected
callbacks (their `ASPathNodeSource`).

```
f(n) = g(n) + h(n),     h(n) = ‖world(n) − goal‖₂        (Euclidean, admissible)
```

- Generic best-first core (`neighbors` + `heuristic` callbacks, `f=g+h`,
  relaxation `if tentative < g_score[nb]`) → `a_star_search`.
- Grid adapter injecting 26-connectivity + cell↔world + free test →
  `AStarPlanner3D` (`_NEIGHBORS`, `_NEIGHBOR_COST`, `neighbors`/`heuristic`
  closures in `plan`).
- **Shaped (pluggable) cost function** → `AStarPlanner3D._edge_cost`:
  ```
  cost(n→m) = step·(1 + w_clear·max(0, clear_pref − clearance(m)) + w_alt·|z_m − z_pref|)
              + w_climb·max(0, z_m − z_n)
  ```
  short **and** clear of walls (clearance term), band-centred altitude, minimal
  climbing. All extra terms ≥ 0 ⇒ the straight-line heuristic stays admissible.
- Nearest-obstacle distance (point→AABB gap, min over boxes) → `WorldModel.clearance`.
- Line-of-sight string-pulling (drop waypoint if straight segment stays free) →
  `AStarPlanner3D.shortcut`.

## 3. SFC — `sfc.py`

One **guaranteed-free** convex box **per collision-free seed point** (one per
B-spline control point), grown by face-by-face inflation:

```
box ← p ± eps            # p is free ⇒ tiny seed is free
while box_is_free(grow(box, face, step)) ∧ extent_face < max_extent:
    box ← grow(box, face, step)
```

- Free-point box → `SafeFlightCorridor._box_for_point` / `boxes_for_points`
  (used per control point by the optimizer).
- Face inflation loop → `_inflate` (over `_FACES`).
- Polyline → dense free seeds → boxes → `SafeFlightCorridor.build` (standalone
  corridor viz).
- Point→box assignment → `assign_boxes`.

Seeding from free points (not a whole segment's AABB) is what keeps every box
obstacle-free — the earlier bug returned huge segment slabs that contained
buildings.

## 4. B-spline — `bspline_optimizer.py` + `uniform_bspline.py`

Port of **QingZhuanya/corridor_Bspline_optimization** (ego-planner lineage).
Interior control points of a **uniform cubic B-spline** are optimised by
**L-BFGS** on an unconstrained weighted sum with analytic gradients:

```
f = λ1·smoothness + λ2·corridor + λ3·feasibility + λ4·fitness
```

Uniform B-spline derivatives are control-point differences with knot span `ts`
(`v=(qᵢ₊₁−qᵢ)/ts`, `a=…/ts²`, `jerk=…/ts³`) → `UniformBspline`; initial control
points via least-squares `parameterize_to_bspline` (zero boundary velocity ⇒ no
endpoint speed spike).

Cost terms (each returns cost + ∂/∂q, vectorised) in `BsplineOptimizer`:
- **smoothness** `Σ‖jerkᵢ‖²` → `_smoothness`.
- **corridor** per control point/axis, C² cubic→quadratic penalty keeping it
  `demarcation` inside its free box (`a=3d,b=−3d²,c=d³`) → `_corridor`.
- **feasibility** squared excess of `|v|,|a|` over limits → `_feasibility`.
- **fitness** anisotropic guide pull `f=(x·v)²/25+‖x×v‖²/1`,
  `x=(qᵢ₋₁+4qᵢ+qᵢ₊₁)/6−refᵢ` → `_fitness`.
- L-BFGS over free control points → `_solve` (`scipy` `L-BFGS-B`).
- **rebound loop** `check_collision_and_rebound`: optimise → sample → if the
  curve still hits an obstacle, refresh the free boxes at the (collision-free)
  guide and raise λ2, re-optimise; ≤ `max_rebound` rounds → `optimize`.

## 5. MPC — `mpc_ros.py` (active) / `mpc.py` (alternative)

**Active controller** = mpc_ros-style (Geonhee-LEE/mpc_ros, Udacity lineage)
**kinematic unicycle** tracking MPC. Inputs are yaw-rate `ω` and accel `a`, so
the drone noses along the path and the forward depth camera looks ahead;
altitude is a decoupled hold.

Reference fit + initial errors (in the **vehicle frame**, cubic `f(x)=Σcₖxᵏ`):

```
cte₀  = f(0) = c₀            (cross-track error)
epsi₀ = −atan(f'(0)) = −atan(c₁)   (heading error)
```

Model + error dynamics (step dt):

```
x⁺=x+v cos ψ dt,  y⁺=y+v sin ψ dt,  ψ⁺=ψ+ω dt,  v⁺=v+a dt
cte⁺  = (f(x) − y) + v sin(epsi) dt
epsi⁺ = (ψ − atan f'(x)) + ω dt
```

Cost (mpc_ros weight structure):

```
J = Σ w_cte cte² + w_epsi epsi² + w_v (v−v_ref)²
  + Σ w_ω ω² + w_a a²  +  Σ w_dω Δω² + w_da Δa²
```

- Reference→vehicle-frame transform + cubic fit + `cte₀,epsi₀` →
  `UnicycleMPC.solve` (`_poly_fit`, `_poly_val`, `_poly_der`).
- Nonlinear model rollout + cost (single shooting; decision vector
  `U=[ω…,a…]`) → `UnicycleMPC._rollout`.
- SLSQP with input box bounds `|ω|≤ω_max, |a|≤a_max` (mpc_ros uses IPOPT/CppAD
  with dynamics as equality constraints; single-shooting + bounds is the
  dependency-light equivalent) → `UnicycleMPC.solve`.
- Receding horizon: apply first `(ω₀,a₀)` → world velocity setpoint
  `v(cos,sin)(ψ+ω₀dt)` + yaw-rate `ω₀`; warm-start shift for next tick.
- Decoupled altitude hold: `vz = clip(z_kp·(z_ref−z), ±vz_max)`.
- Depth avoidance: lateral reference offset ⟂ tangent + speed scaling →
  `mpc.depth_avoidance_offset`, applied in `mpc_node.MPCNode._control_tick`
  (scales `v_ref`).

**Alternative** (`mpc.py`, `TrackingMPC`): per-axis double integrator, condensed
QP `X=Φx₀+ΓU`, `J=½UᵀHU+fᵀU` s.t. `|a|≤a_max, |v|≤v_max` (SLSQP with a velocity
`LinearConstraint`). Holonomic (no heading), kept for comparison.

---

## Known limitations

- B-spline speed briefly overshoots `v_max` (~5.7 vs 5 m/s) on the accel-in /
  accel-out transients; the feasibility penalty is soft and the MPC enforces
  `v_max` hard. Raising `lambda_feas` trades this against smoothness.
- Corridor safety is guaranteed by the **collision-check + rebound loop** (the
  sampled curve is verified free), not by a formal convex-hull proof; the loop
  converged in 1 iteration on the full route.
- Depth avoidance is a reactive lateral reference shift, not a re-plan; a
  persistent blockage should trigger an A\* replan (goal-side hook TBD).

## References

- `corridor_Bspline_optimization` (QingZhuanya, ego-planner lineage):
  https://github.com/QingZhuanya/corridor_Bspline_optimization — **integrated**
  as the B-spline backend (`uniform_bspline.py`, `bspline_optimizer.py`):
  uniform B-spline, smoothness/corridor-rebound/feasibility/fitness costs,
  L-BFGS + `check_collision_and_rebound`, ported to 3D and solved with scipy
  `L-BFGS-B`. We keep our own 3D A\* frontend (not their 2D hybrid A*).
- `mpc_ros` (Geonhee-LEE): ground-vehicle tracking MPC — https://github.com/Geonhee-LEE/mpc_ros
  — **integrated** as the active controller (`mpc_ros.py`): unicycle model,
  cubic reference fit, cte/epsi error dynamics and cost, adapted to a
  cruise-altitude multirotor and solved with scipy single shooting.
- `high_mpc` (UZH-RPG): CasADi quadrotor NMPC — https://github.com/uzh-rpg/high_mpc
  (not integrated; would need CasADi/acados).
- A\* structure follows `BigZaphod/AStar` (generic callback core).

The prior `autonomy_planner` package was removed; its code remains in git history
at commit `313d95d` if any part needs porting.
