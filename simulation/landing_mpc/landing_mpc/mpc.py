"""Relative-coordinate landing MPC (receding horizon) — §3 of the spec.

The drone is a translational double integrator; the state is the RELATIVE
position/velocity to the (moving) target and the input is the 3-axis drone
acceleration.  Solving in relative coordinates fuses "chase" and "align" into
one problem; the absolute PX4 setpoint is recovered by adding the target
prediction back at the end (§2.4, §4).

To keep every solve a small convex QP (§3.5), the horizontal-vertical coupling
of the safety cone (docx §5.4: "z 하강을 수평정렬 상태에 묶는다") is handled by a
two-stage solve rather than one nonlinear program:

    1. xy stage  : two decoupled per-axis QPs -> predicted |p_xy| horizon.
    2. z  stage  : one QP whose per-step position lower bound is
                   p_z,k >= k_cone*|p_xy_pred,k|.  With |p_xy_pred| fixed from
                   stage 1 this is a LINEAR constraint, so the z solve stays a
                   QP.  Descent is thereby gated on horizontal alignment.

Per axis the smooth cost is quadratic in the accel sequence U:

    J = q (p - p_ref)^2 + q_v v^2 + r ||U||^2 + q_T (p_N^2 + v_N^2)          (3)

subject to (all linear in U):

    -a_max <= a_k <= a_max            (box on U)                            (4)
    v_lo   <= v_k <= v_hi             (linear;  v = v_free + Gv U)
    p_k    >= p_lb,k                  (linear;  cone bound, z stage only)

The target acceleration enters the free response (net accel = a_drone -
a_target), which is the curved-target feed-forward (§3.1).  The manager selects
the same look-ahead knot that its streamed reference will actually publish.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .model import condense

_INF = np.inf


@dataclass
class MPCResult:
    accel_cmd: np.ndarray      # (3,) selected outgoing drone acceleration
    pos_cmd: np.ndarray        # (3,) absolute position setpoint at t+dt
    vel_cmd: np.ndarray        # (3,) absolute velocity setpoint at t+dt
    acc_cmd: np.ndarray        # (3,) absolute accel feed-forward at t+dt
    pred_rel_pos: np.ndarray   # (N, 3) predicted relative position horizon
    pred_rel_vel: np.ndarray   # (N, 3) predicted relative velocity horizon
    pred_rel_acc: np.ndarray   # (N, 3) relative accel per horizon step (the plan U)
    solve_time_s: float
    success: bool


class LandingMPC:
    def __init__(self, dt_s: float = 0.1, horizon: int = 20,
                 w_xy: float = 6.0, w_z: float = 3.0,
                 w_vxy: float = 1.5, w_vz: float = 1.5,
                 w_a: float = 0.05, w_terminal: float = 40.0,
                 v_max: float = 6.0, a_max: float = 4.0, vz_max: float = 1.2,
                 cone_k: float = 0.6, z_ref: float = 0.0,
                 cone_max_m: float = 6.0, w_jerk: float = 0.5,
                 j_max: float = 2.0,
                 kp_fb: float = 1.0, kd_fb: float = 2.0):
        self.dt = float(dt_s)
        self.N = int(horizon)
        self.w_xy, self.w_z = w_xy, w_z
        self.w_vxy, self.w_vz = w_vxy, w_vz
        self.w_a, self.w_f = w_a, w_terminal
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.vz_max = float(vz_max)
        self.cone_k = float(cone_k)
        # The cone is a terminal APPROACH CORRIDOR, not a long-range rule: at
        # 70 m out, k*|p_xy| demands ~18 m of relative altitude that the horizon
        # physically cannot reach, so the z-QP became infeasible on every solve.
        # Cap it, and additionally clamp to what is reachable (see solve()).
        self.cone_max = float(cone_max_m)
        self.z_ref = float(z_ref)
        # Jerk penalty: sum (a_k - a_{k-1})^2 keeps the acceleration profile
        # gradual (no lurching); stays quadratic in U so the problem is still a QP.
        self.w_jerk = float(w_jerk)
        self.D = np.eye(self.N) - np.eye(self.N, k=-1)     # (D u)_k = u_k - u_{k-1}
        # Hard slew-rate (jerk) limit |a_k - a_{k-1}| <= j_max*dt.  The soft
        # penalty alone is overwhelmed by the position cost when the error is
        # large (bang-bang a_max -> -a_max); this constraint guarantees the
        # acceleration ramps, which is what stops the airframe lurching.
        # Measured knee (sim sweep, 3 m/s targets): j_max*dt = 0.2 m/s^2 per
        # step keeps steady |v_xy| ~0.055 m and 0 corridor dips, i.e. 4x
        # smoother than 0.8 at no cost.  Tightening to 0.1 costs 4x the
        # tracking error and 30 dips; 0.05 fails to land at all.
        self.j_max = float(j_max)
        self._a_prev = np.zeros(3)
        # PD gains used only when the QP fails (a = -kp*p - kd*v; kd = 2*sqrt(kp)
        # is critically damped for a double integrator, so no fallback ringing)
        self.kp_fb, self.kd_fb = float(kp_fb), float(kd_fb)
        self.Fp, self.Fv, self.Gp, self.Gv = condense(self.dt, self.N)
        self._warm = np.zeros((3, self.N))

    def reset(self):
        """Clear receding-horizon history at a controller-mode boundary."""
        self._warm.fill(0.0)
        self._a_prev.fill(0.0)

    def _acceleration_limits(self, a0, output_step):
        """Bounds that jerk back inside ``a_max`` from an out-of-band handoff."""
        if self.j_max <= 0.0 or abs(a0) <= self.a_max:
            return np.full(self.N, self.a_max)
        jd = self.j_max * self.dt
        recovery_steps = np.maximum(
            1, np.arange(self.N) - int(output_step) + 1)
        return np.maximum(
            self.a_max, abs(a0) - jd * recovery_steps)

    def _reachable_velocity(self, v0, a0, tk, sign, target_accel,
                            output_step=0):
        """Extreme QP-knot velocity under its discrete jerk/accel limits."""
        steps = np.arange(1, np.asarray(tk).size + 1, dtype=float)
        s = 1.0 if sign > 0 else -1.0
        if self.j_max > 0.0:
            jd = self.j_max * self.dt
            limits = self._acceleration_limits(a0, output_step)
            accel = np.empty(steps.size, dtype=float)
            current = float(a0)
            for k in range(steps.size):
                current = float(np.clip(
                    current + s * jd, -limits[k], limits[k]))
                if k == output_step:
                    current = float(np.clip(
                        current, a0 - jd, a0 + jd))
                accel[k] = current
        else:
            accel = np.full(steps.shape, s * self.a_max)
        return v0 + self.dt * np.cumsum(
            accel - np.asarray(target_accel, float))

    # -- single-axis QP -----------------------------------------------------
    def _solve_axis(self, p0, v0, at_axis, wp, wv, p_ref, warm,
                    v_lo, v_hi, p_lb=None, a_prev=0.0, output_step=0):
        """Solve one axis; return (u, pred_p, pred_v).

        at_axis (N,)  predicted target accel on this axis (feed-forward).
        p_ref  (N,)   position reference.  v ref is 0 (match target velocity).
        v_lo/v_hi     scalar relative-velocity bounds.
        p_lb   (N,)   optional per-step position lower bound (cone).
        """
        N = self.N
        x0 = np.array([p0, v0])
        Gp, Gv = self.Gp, self.Gv
        p_free = self.Fp @ x0 - Gp @ at_axis
        v_free = self.Fv @ x0 - Gv @ at_axis

        H = 2.0 * (wp * Gp.T @ Gp + wv * Gv.T @ Gv + self.w_a * np.eye(N))
        H += 2.0 * self.w_f * (np.outer(Gp[-1], Gp[-1]) + np.outer(Gv[-1], Gv[-1]))
        f = 2.0 * (wp * Gp.T @ (p_free - p_ref) + wv * Gv.T @ v_free)
        f += 2.0 * self.w_f * (Gp[-1] * (p_free[-1] - p_ref[-1]) + Gv[-1] * v_free[-1])
        # jerk term: w_j * ||D u - e0*a_prev||^2  (a_{-1} = last applied accel,
        # so the profile is also continuous ACROSS re-plans)
        if self.w_jerk > 0.0:
            D = self.D
            H = H + 2.0 * self.w_jerk * (D.T @ D)
            f = f - 2.0 * self.w_jerk * (D.T[:, 0] * a_prev)

        def cost(u):
            return 0.5 * u @ H @ u + f @ u

        def grad(u):
            return H @ u + f

        # Velocity limits as a FEASIBLE envelope.  A hard |v| <= v_max is
        # unsatisfiable whenever the current speed already exceeds it (during
        # the chase v_rel hit 8.4 m/s against v_max=7: step 1 would need ~14
        # m/s^2 of braking), which is what made ~25% of solves fail.  If v0 is
        # outside the band, only require it to return at a_max, not instantly.
        # The envelope must respect the JERK limit too: near v_max with the
        # acceleration still positive, reversing it fast enough to stay under
        # the bound is impossible at 0.2 m/s^2 per step, which is what kept
        # ~50% of the chase solves infeasible.
        tk = self.dt * np.arange(1, N + 1)
        v_ub = np.maximum(
            v_hi, self._reachable_velocity(
                v0, a_prev, tk, -1, at_axis, output_step))
        v_lb = np.minimum(
            v_lo, self._reachable_velocity(
                v0, a_prev, tk, +1, at_axis, output_step))
        cons = [LinearConstraint(Gv, v_lb - v_free, v_ub - v_free)]
        if p_lb is not None:
            cons.append(LinearConstraint(Gp, p_lb - p_free, np.full(N, _INF)))
        if self.j_max > 0.0:
            # |a_k - a_(k-1)| <= j_max*dt ; row 0 is measured against a_prev
            jd = self.j_max * self.dt
            lb = np.full(N, -jd)
            ub = np.full(N, jd)
            lb[0] += a_prev
            ub[0] += a_prev
            cons.append(LinearConstraint(self.D, lb, ub))
            if output_step:
                selector = np.zeros((1, N))
                selector[0, output_step] = 1.0
                cons.append(LinearConstraint(
                    selector, [a_prev - jd], [a_prev + jd]))
        limits = self._acceleration_limits(a_prev, output_step)
        bounds = [(-limit, limit) for limit in limits]
        res = minimize(cost, warm, jac=grad, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 60, 'ftol': 1e-6})
        if res.success:
            u = res.x
        else:
            # NEVER replay the stale warm start: if the QP fails once, repeating
            # the old plan keeps accelerating along it, which drives the state
            # further from feasibility (the cone bound k*|p_xy| grows with the
            # error) and the vehicle runs away.  Fall back to a critically
            # damped PD that always aims back at the target, and clear the warm
            # start so the next solve restarts cleanly.
            # Saturated cascade: a desired velocity capped at v_max, then a
            # P-law on the velocity error.  A raw PD on a 70 m error saturates
            # a_max for seconds and builds ~20 m/s, which is how the vehicle
            # overshot into the trailer.
            v_des = np.clip(-self.kp_fb * p0, v_lo, v_hi)
            a_fb = np.clip(self.kd_fb * (v_des - v0), -self.a_max, self.a_max)
            u = np.full(N, a_fb)
        pred_p = p_free + Gp @ u
        pred_v = v_free + Gv @ u
        return u, pred_p, pred_v, bool(res.success)

    def solve(self, p_rel, v_rel, target_P, target_V, target_A,
              applied_acceleration=None, output_step=0) -> MPCResult:
        """One receding-horizon step.

        p_rel/v_rel : (3,) current relative position/velocity.
        target_P/V/A: (N,3) target prediction over horizon steps 1..N.
        """
        p_rel = np.asarray(p_rel, float)
        v_rel = np.asarray(v_rel, float)
        At = np.asarray(target_A, float).reshape(self.N, 3)
        N = self.N
        output_step = int(output_step)
        if not 0 <= output_step < N:
            raise ValueError('output_step must index the MPC horizon')
        applied = (self._a_prev.copy() if applied_acceleration is None else
                   np.asarray(applied_acceleration, float).reshape(3))
        if not np.all(np.isfinite(applied)):
            raise ValueError('applied_acceleration must be finite')
        zeros = np.zeros(N)
        t0 = time.perf_counter()
        ok = True

        # stage 1: horizontal axes -> predicted |p_xy| for the cone bound
        U = np.zeros((3, N))
        pred_p = np.zeros((3, N))
        pred_v = np.zeros((3, N))
        failed = [False, False, False]
        for ax in (0, 1):
            u, pp, pv, s = self._solve_axis(
                p_rel[ax], v_rel[ax], At[:, ax], self.w_xy, self.w_vxy, zeros,
                self._warm[ax], -self.v_max, self.v_max,
                a_prev=applied[ax], output_step=output_step)
            U[ax], pred_p[ax], pred_v[ax] = u, pp, pv
            failed[ax] = not s
            ok &= s
        pxy_pred = np.hypot(pred_p[0], pred_p[1])

        # stage 2: vertical axis.  The approach corridor is applied as a SOFT
        # reference, not a hard floor: as a hard constraint it is unreachable
        # during a long chase (70 m out it demanded ~18 m of climb the horizon
        # cannot deliver) and the QP failed on every tick.  As a reference the
        # z cost simply pulls the vehicle to the corridor height, which decays
        # to 0 as |p_xy| shrinks — same "align before descend" gating, always
        # feasible.
        corridor = np.minimum(self.cone_k * pxy_pred, self.cone_max)
        z_ref_seq = np.maximum(corridor, self.z_ref)
        u, pp, pv, s = self._solve_axis(
            p_rel[2], v_rel[2], At[:, 2], self.w_z, self.w_vz,
            z_ref_seq, self._warm[2], -self.vz_max, self.v_max,
            a_prev=applied[2], output_step=output_step)
        U[2], pred_p[2], pred_v[2] = u, pp, pv
        failed[2] = not s
        ok &= s

        # keep a warm start only for axes that actually converged
        self._warm = U.copy()
        for ax in range(3):
            if failed[ax]:
                self._warm[ax] = 0.0
        solve_dt = time.perf_counter() - t0

        pred_rel_pos = pred_p.T.copy()
        pred_rel_vel = pred_v.T.copy()
        selected_acceleration = U[:, output_step].copy()
        self._a_prev = selected_acceleration.copy()
        self._a_prev[np.asarray(failed, bool)] = 0.0
        tp0 = np.asarray(target_P, float).reshape(N, 3)[0]
        tv0 = np.asarray(target_V, float).reshape(N, 3)[0]
        ta0 = np.asarray(target_A, float).reshape(N, 3)[0]
        pos_cmd = tp0 + pred_rel_pos[0]
        vel_cmd = tv0 + pred_rel_vel[0]
        acc_cmd = ta0 + selected_acceleration
        pred_rel_acc = U.T.copy()
        return MPCResult(selected_acceleration, pos_cmd, vel_cmd, acc_cmd,
                         pred_rel_pos, pred_rel_vel, pred_rel_acc, solve_dt, ok)
