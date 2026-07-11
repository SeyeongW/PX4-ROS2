"""Short-horizon MPC for the MISSION-leg obstacle-avoidance servo -- the
"MPC(QP)" stage of the A*(front-end) -> corridor(back-end) -> MPC pipeline
in research_direction. Solved via projected GRADIENT DESCENT WITH BACKTRACKING
LINE SEARCH on plain numpy, NOT a QP solver library (osqp/cvxpy/casadi): this
repo's Docker image build is disk-space constrained (see docker_sim_env
memory), and the problem here is small enough (horizon ~10, 2 controls per
step) that a hand-rolled solver converges in well under a control tick -- see
gazebo/mpc_demo.py for a convergence/timing/stress check.

Backtracking (not a fixed step size) matters here: the smooth part of the
cost is a quadratic form in v with curvature that scales with corridor_weight
AND with horizon (v_0 influences every later position through the cumulative-
sum dynamics, so its Hessian block grows like ~horizon^2). A fixed step tuned
for the "no active corridor constraint" case DIVERGES the moment a target
sits outside the current box (the exact common case right after a replan) --
found by a closed-loop stress test, not just eyeballing single solves. Every
step is only ACCEPTED if it doesn't increase the cost (Armijo-free monotone
backtracking), so the solver cannot diverge for any weight/horizon/dt
combination; if a step can't reduce the cost within max_backtracks halvings
it stays put for that iteration. If a HARD corridor guarantee is ever needed,
swap this for osqp behind the same solve() signature.

Corridor constraint is a heavily-weighted SOFT quadratic penalty, not a hard
QP inequality -- a deliberate simplification for this first version. Velocity
bounds ARE hard (exact box projection every iteration).

State is point-mass kinematics in the (E,N) plane. ArduPilot GUIDED already
handles the real nonlinear attitude/motor dynamics beneath the velocity
setpoint we send (see the NMPC discussion in research_direction: NMPC would
model that nonlinear layer directly, which we deliberately don't -- so plain
linear/kinematic MPC is enough here).
"""
import numpy as np


def _cost(v, p0, v0, targets, dt, q_track, r_smooth, r_effort, corridor_box,
         corridor_weight):
    pos_seq = p0[None, :] + dt * np.cumsum(v, axis=0)
    c = q_track * np.sum((pos_seq - targets) ** 2)
    if corridor_box is not None:
        over_e = np.clip(pos_seq[:, 0] - corridor_box.e_max, 0, None)
        under_e = np.clip(corridor_box.e_min - pos_seq[:, 0], 0, None)
        over_n = np.clip(pos_seq[:, 1] - corridor_box.n_max, 0, None)
        under_n = np.clip(corridor_box.n_min - pos_seq[:, 1], 0, None)
        c += corridor_weight * np.sum(over_e ** 2 + under_e ** 2 +
                                      over_n ** 2 + under_n ** 2)
    prev = np.vstack([v0[None, :], v[:-1]])
    c += r_smooth * np.sum((v - prev) ** 2)
    c += r_effort * np.sum(v ** 2)
    return c


def _grad(v, p0, v0, targets, dt, q_track, r_smooth, r_effort, corridor_box,
          corridor_weight):
    pos_seq = p0[None, :] + dt * np.cumsum(v, axis=0)
    g_p = 2.0 * q_track * (pos_seq - targets)            # d(tracking)/dp_k
    if corridor_box is not None:
        over_e = np.clip(pos_seq[:, 0] - corridor_box.e_max, 0, None)
        under_e = np.clip(corridor_box.e_min - pos_seq[:, 0], 0, None)
        over_n = np.clip(pos_seq[:, 1] - corridor_box.n_max, 0, None)
        under_n = np.clip(corridor_box.n_min - pos_seq[:, 1], 0, None)
        g_p[:, 0] += 2.0 * corridor_weight * (over_e - under_e)
        g_p[:, 1] += 2.0 * corridor_weight * (over_n - under_n)

    # d(cost)/dv_m = dt * sum_{k=m}^{N-1} g_p[k] -- v_m affects every LATER
    # position via the cumsum, so its gradient is a reverse partial sum of
    # the per-position gradient.
    grad = dt * np.cumsum(g_p[::-1], axis=0)[::-1]

    # Smoothness (penalise jerk relative to v0 and between consecutive
    # steps) + a small effort term to keep the solution well-posed.
    prev = np.vstack([v0[None, :], v[:-1]])
    grad += 2.0 * r_smooth * (v - prev)
    nxt_diff = np.vstack([v[1:] - v[:-1], np.zeros((1, 2))])
    grad -= 2.0 * r_smooth * nxt_diff
    grad += 2.0 * r_effort * v
    return grad


def solve(pos, vel, target, target_vel, corridor_box, vmax, dt, horizon,
         q_track=1.0, r_smooth=0.08, r_effort=0.01, corridor_weight=40.0,
         iters=80, step0=2.0, max_backtracks=20):
    """Returns the (vE, vN) command to apply NOW. Receding horizon: only the
    first step of the solved sequence is used -- the caller re-solves every
    tick with fresh state/target/corridor, discarding the rest of the plan.

    pos, vel      : current (e, n) position and (vE, vN) velocity (the latter
                    has no direct sensor feedback in mission_manager_node, so
                    the caller passes the last COMMANDED velocity as a proxy).
    target        : (e, n) the servo is chasing -- a static patrol waypoint,
                    or a moving cue.
    target_vel    : (vE, vN) feed-forward for a moving target (0,0 for a
                    static waypoint) -- target is predicted forward as
                    target + target_vel * t over the horizon, same idea as
                    _servo_to's feed-forward but projected multiple steps out.
    corridor_box  : planner.Box the position is softly penalised for leaving,
                    or None (no corridor available -- falls back to a purely
                    unconstrained tracking servo, still safe in open space).
    step0         : starting step size each outer iteration tries before
                    backtracking (halving) down to one that doesn't increase
                    the cost -- NOT a fixed learning rate.
    """
    p0 = np.asarray(pos, dtype=float)
    v0 = np.asarray(vel, dtype=float)
    tgt0 = np.asarray(target, dtype=float)
    tgt_vel = np.asarray(target_vel, dtype=float)
    v = np.zeros((horizon, 2))          # decision vars v_0..v_{N-1}, warm-started at 0

    t = (dt * np.arange(1, horizon + 1))[:, None]       # (N,1): t_1..t_N
    targets = tgt0[None, :] + tgt_vel[None, :] * t       # predicted target per step
    args = (p0, v0, targets, dt, q_track, r_smooth, r_effort, corridor_box,
           corridor_weight)

    cost = _cost(v, *args)
    for _ in range(iters):
        grad = _grad(v, *args)
        gnorm2 = float(np.sum(grad ** 2))
        if gnorm2 < 1e-12:
            break
        step = step0
        for _ in range(max_backtracks):
            v_trial = np.clip(v - step * grad, -vmax, vmax)
            cost_trial = _cost(v_trial, *args)
            if cost_trial <= cost:
                v, cost = v_trial, cost_trial
                break
            step *= 0.5
        # else: no step this halving budget reduced the cost -- stay put
        # (already at/near a local optimum of the projected problem).

    return float(v[0, 0]), float(v[0, 1])
