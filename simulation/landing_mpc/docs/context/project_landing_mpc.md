---
name: project_landing_mpc
description: "landing_mpc pkg — relative-coord MPC for moving-ArUco precision landing; Steps 0-3 validated, Step 4 PX4 untested"
metadata: 
  node_type: memory
  type: project
  originSessionId: 8420beaf-12bc-4574-81f6-c789a51152bb
  modified: 2026-07-23T08:16:02.996Z
---

New `landing_mpc` ROS2 package (jo branch, created 2026-07-22) built from
`MPC_동적착륙_설계문서.docx` + `LANDING_MPC_SPEC.md`. Relative-coordinate
translational double-integrator MPC for precision landing on a moving ArUco
target (~3 m/s curved). Treats landing as relative-frame docking: drive
`p_rel`→0 and `v_rel`→0.

Files: model.py, predictor.py (const-vel/const-accel/poly), mpc.py, baseline.py,
sim.py, frame.py (ENU↔NED, unit-tested), px4_node.py, eval/plot_run.py +
eval/montecarlo.py (matplotlib, Agg, PNGs in eval/plots/). docs/nodes.md has a
plain-language math↔code walkthrough of every module.

Key impl choice: safety cone `p_z ≥ k·‖p_xy‖` kept as a **QP** via a two-stage
solve (xy axes first → predicted ‖p_xy‖ → z solve with a *linear* per-step lower
bound), NOT one nonlinear SOCP — the coupled SLSQP version was too slow (>2 min).
Target-accel feed-forward folded into the free response (net accel =
a_drone − a_target) is the curved-target lag fix. Consistent with [[project_mpc_pointmass]].

Status (2026-07-22): Steps 0-3 PASS in `python3 -m landing_mpc.sim` (steady xy
err ~1.6 cm, touchdown |v_rel| ~0.10 m/s on straight + 8 m circle, 0 cone
violations, solve ≤9 ms). Step 5 eval done: montecarlo MPC 24/25 vs baseline
22/25, MPC touchdown ~0.11 m/s flat vs baseline degrading on sharp/fast curves.
Step 4 `px4_node.py` (consumes precision_landing `/marker/position` +
`/marker/velocity` ENU → LandingMPC → NED TrajectorySetpoint, position-led;
opt-in auto_engage via VehicleCommand). SITL FLIGHT CONFIRMED (2026-07-22,
headless): real PX4 x500 (gz_x500) took off to 5 m then MPC landed on a moving
local-ENU marker — touchdown xy 0.23 m, |v_rel| 0.65 m/s (firmer than sim ~0.1
due to real attitude/actuator dynamics; terminal-descent tuning is future work).
Orchestrator that did takeoff+land: scratchpad/sitl_land.py.

SITL gotchas discovered: (1) this PX4 build uses MESSAGE VERSIONING — outputs are
suffixed, so subscribe to `/fmu/out/vehicle_local_position_v1` NOT
`vehicle_local_position` (now a `local_pos_topic` param); inputs
(offboard_control_mode, trajectory_setpoint, vehicle_command) are UNVERSIONED.
(2) `/fmu/out/*` are best_effort QoS — `ros2 topic echo` needs
`--qos-reliability best_effort`. (3) MicroXRCEAgent is at /usr/local/bin (not
~/.local/bin where run_px4_map.sh looks); pre-start `MicroXRCEAgent udp4 -p 8888`.
(4) PX4 daemon shell (`px4 -d`/backgrounded) spews ANSI and can write GBs — a
4.7 GB task log filled /tmp; redirect PX4 stdout to /dev/null for headless runs.
Lightest boot: `cd ~/PX4-Autopilot && HEADLESS=1 make px4_sitl gz_x500` (origin
spawn, avoids city world/local-frame offset).

precision_landing_200m WORLD LANDING CONFIRMED (2026-07-22): pulled origin/jo
(merge; it also added a separate precision_landing MPC stack — mpc_core.py etc. —
which the USER SAID TO IGNORE, keep landing_mpc). Kept only the new world files.
Ran landing_mpc over XRCE (no code changes) via `START_XRCE=1 START_MAVROS=0
START_BRIDGE=0 HEADLESS=1 PX4_DAEMON=1 ./gazebo/run_precision_landing_200m.sh` +
`ros2 run landing_mpc landing_mpc_sitl_demo`. Drone spawns world (30,30), ArUco
trailer at world (37.07,37.07) = LOCAL ENU (7.07,7.07), deck marker z=1.811 local.
Result: takeoff 8 m -> landed ON the 2 m deck, 0.27 m from marker, |v_rel| 0.8 m/s.
Added `landing_mpc_sitl_demo` (sitl_demo.py) entry point = takeoff+land mission
with a fixed LOCAL-ENU cue.

PERCEPTION PIPELINE (single-purpose nodes, user's explicit architecture rule —
one node per role, clear names, integrate later):
  aruco_detector_node -> /aruco/pose_cam (camera optical frame; knows no drone)
  marker_tf_node      -> /marker/measured (pure geometric transform to ENU)
  marker_kf_node      -> /marker/position + /marker/velocity + /marker/valid
                         (const-velocity KF; COASTS when detections stop)
**TRIED AND REVERTED: routing APPROACH through a jerk-limited MPC.** Motivation
was sound (a raw setpoint 50-100 m away is a huge position error, so PX4's own
controller saturates and pitches over). It DID smooth the flight — peak tilt
109 deg -> 17 deg — but the vehicle then never caught the 3 m/s trailer at all:
camera-on-target fell 36% -> **2%**, DESCEND was never entered, at BOTH
approach a_max 2.0 and 4.0. Reverted to the plain position-led setpoint, which
is rougher but does close the distance. Dead mpc_app/_ref_app removed.
Lesson: smoothness bought at the cost of never reaching the target is not a win.

**INSTRUMENTED DIAGNOSIS (2026-07-23) — stop guessing, this is the data.**
`scratchpad/vis_diag.py` projects the camera optical axis onto the deck plane
(tilt-aware footprint centre) and compares "geometry says visible" vs "detector
fired". Result over 155 s:
  - geometry OK -> detected **55/59 s (93%)**. THE DETECTOR IS FINE.
  - the camera was **NOT pointed at the target 62% of the flight**.
  - tilt >10 deg for 41 s, peak 109 deg.
So the blocker is camera POINTING, not detection. Detection also freezes
permanently mid-run (count stuck at 615 for 40+ s) whenever the vehicle loses
the target — the 26% "rate" is a stale average, not a live one.
**Lowering the MPC's own a_max 4.0 -> 2.0 did NOT help** (on-target 38%->36%,
tilt>10 deg 41 s -> 41 s, unchanged) — because APPROACH and ABORT do NOT use the
MPC at all: they send raw position setpoints and **PX4's own position controller**
generates the acceleration, hence the tilt. The lever is therefore PX4-side
(`MPC_ACC_HOR_MAX`, `MPC_XY_VEL_MAX`, settable via the map yaml's
`sitl_parameter_overrides`) or routing APPROACH through the jerk-limited MPC too
— or a gimbal. This is exactly the "P gain too high" the user warned about, but
it is PX4's gain, not ours. See [[feedback_gentle_control]].

**THE GIMBAL IS NOW THE RIGHT ANSWER** (2026-07-23, end state). Two mission-manager
bugs were found and fixed by derivation, and each revealed the next layer:
 1. radius-based DESCEND bail-out: threshold 2*r(h) SHRINKS with altitude, so
    descending from a ~4.8 m acquisition always tripped near h=5 m. Identical
    failure at 1.5 and 3 m/s PROVED it was logic, not target speed. Removed —
    alignment is already enforced by the MPC corridor. Hunting 15-16 -> **0**.
 2. remaining blocker is TILT: entering DESCEND with a 3 m/s velocity mismatch
    makes the vehicle tilt to catch up, and the body-fixed down camera swings
    off target — at h=10 m the visible radius is 5.5 m but only **10-15 deg of
    tilt moves the line of sight 1.8-2.7 m**, so the marker leaves frame within
    ~3 s and the mission aborts (7 cycles, no landing) despite a healthy 26%
    overall detection rate.
So the user's earlier gimbal idea is now the correct next step: the cue solved
the RANGE problem, and tilt decoupling is precisely what a gimbal fixes.
Cheaper alternatives to try first: require velocity match (not just proximity)
before the handoff so the vehicle does not need to tilt hard in DESCEND, and/or
raise abort_grace so the KF coasts through the tilt transient.

**APPROACH HIGH, NOT LOW** (2026-07-23): the acquisition cone r(h) GROWS with
altitude while the marker stays resolvable far above (30 m => still 3.4 px per
ArUco cell). Raising approach_alt 8 m -> 16 m took the detection rate from
**0.8% to 37.7%**. (I initially told the user the opposite — lowering altitude
widens the cone — which is wrong; r is proportional to h.)
Also derived: the MPC descent corridor should BE the camera cone, so
`cone_k = 1/tan(vfov/2) ~= 1.59`; the old 0.25 allowed |p_xy| <= 4h, six times
wider than the camera sees, so it descended off-centre and lost the marker.
STILL UNRESOLVED: even with both fixes the mission oscillates
DESCEND<->APPROACH (15 hunts, no landing in 170 s) while chasing the 3 m/s
trailer. Last change (UNVERIFIED): the drift check measured distance to the CUE
while DESCEND tracks the VISION estimate — self-conflicting; now measured
against the tracked target. Needs a re-run.

**THE VISION REGION IS A CONE, NOT A RADIUS** — the single most useful geometric
fact found. A down-looking camera sees the marker only while the horizontal
offset satisfies `r(h) = h*tan(vfov/2) - s_m/2` (h = height above deck).
At the 8 m approach altitude that is 4.3 m, NOT tens of metres — and r(h)=0 at
h = s_m/(2 tan(vfov/2)) = 1.19 m, which is exactly the blind zone, so the FOV
cone and the blind zone are the same geometry at its two ends. SITL handoffs
happened at 4.5/3.4/1.7 m, matching. Derive handoff radii from altitude with
this; never hard-code one.

`mission_manager_node` (single Offboard authority) implements the handoff:
IDLE→TAKEOFF→APPROACH(cue)→DESCEND(vision)→TOUCHDOWN, plus ABORT/recover, with
hysteresis and a COMMIT HEIGHT below which vision loss is expected and the KF
coast is ridden down instead of aborting. `trailer_cue_node` stands in for the
target's telemetry (reads gz trailer pose). FULL PIPELINE VERIFIED on the 3 m/s
circling trailer with REAL ArUco: landed, xy 0.51 m, |v_rel| 1.71 m/s — but that
is much worse than the gz-cheat cue (0.24 m, 0.10 m/s), and it hunted
DESCEND<->APPROACH twice before committing. Terminal-phase estimate quality
inside the narrow cone is the open problem.

**VISION CANNOT DRIVE THE CHASE — architectural finding (2026-07-23).** Detection
rate is 15% on a static target (60-80% when hovering overhead) but collapses to
**0.3% (6/1943 frames) chasing the 3 m/s circling trailer**: during a long fast
chase the airframe is tilted and the target is 50-90 m away, so the down camera
almost never sees the marker. The KF then coasts forever from a stale fix and
diverges (error grew 174 -> 216 m, `valid` correctly False the whole time — a
constant-velocity coast cannot follow a circling target anyway). Conclusion:
vision is a TERMINAL-PHASE sensor only. The stack needs the cue->vision handoff
the original design doc describes (fly to an externally supplied target cue,
switch to vision only when close and overhead) — that is a mission-manager
responsibility, not something more KF tuning can fix.

KF COAST VERIFIED in SITL (static target): last fix at 1.60 m above deck, zero fixes from 1.13 m
down (matches the computed 1.19 m FOV limit), KF kept publishing throughout and
`valid` auto-flipped False after max_coast=3 s. Drone landed through the blind
zone. KF sigma_meas default 0.06 m is taken from the MEASURED static accuracy,
not guessed. KF vs MPC framing that helped the user: KF = estimator (looks
backward, fuses measurements, handles noise/dropouts); MPC = controller (looks
forward, optimizes inputs, handles cost/constraints); they chain, not compete.

PERCEPTION DETAIL (2026-07-23), split into single-purpose nodes per user request:
`aruco_detector_node` (image -> /aruco/pose_cam in the CAMERA OPTICAL frame,
nothing else) and `marker_tf_node` (/aruco/pose_cam + vehicle state ->
/marker/position + /marker/velocity in local ENU). The camera->ENU maths lives
in frame.py (`camera_offset_to_enu`) and is unit-tested with the rest of
`python3 -m landing_mpc.frame`.

MEASURED (static hover over the marker, truth known):
  8 m: 0.068 m | 6 m: 0.024 m | 4 m: 0.059 m | 3.0 m and below: NO DETECTION
Bias ~0 — the detector + TF chain is CORRECT. Detection rate ~15% of frames
overall (near 100% when stationary and in view). Dynamic-flight error is much
worse (~0.66 m, zero bias = scatter not lag) because the airframe tilts.
Marker size CONFIRMED 1.5 m by measuring the texture (400/520 px x 1.95 m plane
= 1.5000 m) — the model's 1.95 m plane includes the quiet zone; do NOT "fix"
marker_size_m to 1.95. The 3.0 m detection cliff matches the FOV computation
exactly (marker overflows below 1.19 m above the deck) -> still needs KF coast
or a nested small marker.
Gotchas: ros_gz camera topics are BEST_EFFORT (a default RELIABLE subscription
receives NOTHING, only a QoS warning). Interpolating vehicle state to the IMAGE
stamp cut 1.17 m -> 0.66 m. TRIED AND REVERTED: keying the pose history by PX4's
own timestamp with a running-min clock offset — measured WORSE (0.66 -> 1.08 m
with a lag-signature bias); receipt-time keying wins.

HARD-WON SITL LESSONS (2026-07-23) — all four broke flight and none showed up in
the offline sim: (1) **use_sim_time is mandatory** — the MPC plan's time axis is
SIM time; on the wall clock with RTF<1 (GUI+camera bridge) the reference is
sampled ahead of physics and the drone overshoots. (2) **Never replay the warm
start when the QP fails** — it keeps accelerating along the stale plan and
diverges; fall back to a velocity-SATURATED PD (a raw PD on a 70 m error hит
21 m/s and crashed into the trailer). (3) **The safety cone must be a SOFT
z-reference, not a hard floor** — at 70 m out it demanded ~18 m of climb the
2 s horizon cannot deliver, so every solve failed (observed qp_fail=128/128).
(4) **Velocity limits need a jerk-aware feasible envelope** — near v_max with
accel still positive, the jerk limit cannot reverse accel fast enough, so a
hard |v|<=v_max is unsolvable (~50% failures). Use `_reachable_velocity()`.
Always log `qp_fail` — it is the fastest way to see the MPC is not actually
running. Jerk limit: `j_max*dt` = accel change per step; measured knee is
**0.2 m/s²** (0.1 costs 4x tracking error + corridor dips, 0.05 fails to land,
0.8 caused the airframe to lurch).

MOVING platform CONFIRMED too (2026-07-22): `run_precision_landing_200m_moving.sh`
drives the trailer in a circle (center world (72.43,72.43), r=50 m, default 9 m/s;
override with TRAILER_SPEED_M_S). Added `landing_mpc_sitl_demo_moving`
(sitl_demo_moving.py) which reads the live trailer pose via gz.transport13
(/world/<world>/dynamic_pose/info), converts world->local ENU (subtract spawn
30,30; deck z=1.811), finite-diffs velocity, and chases+lands. At 3 m/s the drone
intercepted the trailer ~90 m away and landed on the 2 m deck within 0.24 m at
|v_rel| ~1.0 m/s (cone_k lowered to 0.25, v_max 7 for the long chase). 9 m/s is
beyond this simple demo. GUI+camera run: FOLLOW_DRONE=1 START_BRIDGE=1 (cameras
/down_camera/image ~15Hz, /front_camera/image ~13Hz); drone spawns at world
(30,30) in a 200 m world so it looks empty unless FOLLOW_DRONE or Entity Tree
->Move to. Two scripts: precision_landing_200m (static) vs _200m_moving (circle).
Sibling of [[project_path_plan]].
Env: ROS humble, px4_msgs at ~/install, PX4-Autopilot + Micro-XRCE agent present.
