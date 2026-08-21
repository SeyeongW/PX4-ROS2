"""ego-planner-style B-spline optimizer (port of QingZhuanya/corridor_Bspline_opt).

Interior control points of a uniform cubic B-spline are optimised by **L-BFGS**
on an unconstrained weighted sum with analytic gradients:

    f = λ1·smoothness + λ2·corridor + λ3·feasibility + λ4·fitness

- **smoothness**  Σ‖jerk_i‖²,  jerk_i=(q_{i+3}-3q_{i+2}+3q_{i+1}-q_i)/ts³
- **corridor**    per control point / axis, a C² cubic→quadratic penalty that
  keeps it ``demarcation`` inside its free SFC box (rebound term)
- **feasibility** squared excess of |v|,|a| (control-point differences) over limits
- **fitness**     anisotropic pull to the A* guide (weak along, strong across)

Wrapped in ``check_collision_and_rebound``: optimise → sample → if the curve
still hits an obstacle, refresh the offending control points' free boxes at the
(collision-free) guide and raise λ2, then re-optimise; ≤ ``max_rebound`` rounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .sfc import Corridor, SafeFlightCorridor
from .uniform_bspline import UniformBspline
from .world_model import WorldModel


@dataclass
class OptimizeResult:
    spline: UniformBspline
    corridor: Corridor
    guide_points: np.ndarray          # (n,3) reference aligned to control points
    rebound_iters: int
    collision_free: bool
    free_fraction: float
    solver_success: bool = True
    solver_status: int = 0
    solver_message: str = ''
    solution_finite: bool = True

    @property
    def accepted(self) -> bool:
        """Finite solver result whose sampled chordal path is collision-free."""
        return bool(
            self.solver_success
            and self.solution_finite
            and self.collision_free
            and np.isfinite(self.spline.ts)
            and self.spline.ts > 0.0
            and np.all(np.isfinite(self.spline.q)))


class BsplineOptimizer:
    def __init__(self, world: WorldModel,
                 *, cruise_speed_m_s: float | None = 4.0,
                 ctrl_spacing_m: float = 5.0, order: int = 3,
                 max_vel: float = 5.0, max_acc: float = 1.5,
                 lambda_smooth: float = 5.0, lambda_dist: float = 0.5,
                 lambda_feas: float = 1.0, lambda_fit: float = 0.05,
                 demarcation_m: float = 0.3, max_rebound: int = 10,
                 dist_growth: float = 2.0, sfc: SafeFlightCorridor | None = None,
                 strict_validation: bool = False):
        self.world = world
        # None leaves the curve as geometry only. The flight controller then
        # owns its time profile instead of inheriting a simulated cruise speed.
        self.cruise = (None if cruise_speed_m_s is None
                       else float(cruise_speed_m_s))
        self.ctrl_spacing = float(ctrl_spacing_m)
        self.p = int(order)
        self.max_vel = float(max_vel)
        self.max_acc = float(max_acc)
        self.l_smooth = float(lambda_smooth)
        self.l_dist0 = float(lambda_dist)
        self.l_feas = float(lambda_feas)
        self.l_fit = float(lambda_fit)
        self.d = float(demarcation_m)
        self.max_rebound = int(max_rebound)
        self.dist_growth = float(dist_growth)
        self.sfc = sfc or SafeFlightCorridor(world)
        self.strict_validation = bool(strict_validation)

    # ------------------------------------------------------------ cost terms
    def _smoothness(self, q, ts):
        jerk = (q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3]) / ts ** 3
        cost = float(np.sum(jerk * jerk))
        g = 2.0 * jerk / ts ** 3
        grad = np.zeros_like(q)
        grad[0:-3] += -g
        grad[1:-2] += 3 * g
        grad[2:-1] += -3 * g
        grad[3:] += g
        return cost, grad

    def _feasibility(self, q, ts):
        grad = np.zeros_like(q)
        v = np.diff(q, axis=0) / ts
        ex_v = np.clip(np.abs(v) - self.max_vel, 0.0, None) * np.sign(v)
        cost = float(np.sum(ex_v * ex_v))
        gv = 2.0 * ex_v / ts
        grad[:-1] += -gv
        grad[1:] += gv
        a = np.diff(q, n=2, axis=0) / ts ** 2
        ex_a = np.clip(np.abs(a) - self.max_acc, 0.0, None) * np.sign(a)
        cost += float(np.sum(ex_a * ex_a))
        ga = 2.0 * ex_a / ts ** 2
        grad[:-2] += ga
        grad[1:-1] += -2 * ga
        grad[2:] += ga
        return cost, grad

    def _corridor(self, q, lo, hi):
        d = self.d
        a, b, c = 3 * d, -3 * d * d, d ** 3
        grad = np.zeros_like(q)
        # upper wall:  updist = d - (hi - q)
        up = d - (hi - q)
        small = up < d
        act = up >= 0.0
        cost_u = np.where(act, np.where(small, np.clip(up, 0, None) ** 3,
                                        a * up * up + b * up + c), 0.0)
        grad_u = np.where(act, np.where(small, 3 * up * up, 2 * a * up + b), 0.0)
        # lower wall:  lowdist = d - (q - lo),  d(lowdist)/dq = -1
        low = d - (q - lo)
        smalll = low < d
        actl = low >= 0.0
        cost_l = np.where(actl, np.where(smalll, np.clip(low, 0, None) ** 3,
                                         a * low * low + b * low + c), 0.0)
        grad_l = -np.where(actl, np.where(smalll, 3 * low * low, 2 * a * low + b), 0.0)
        grad += grad_u + grad_l
        return float(np.sum(cost_u + cost_l)), grad

    def _fitness(self, q, ref):
        A2, B2 = 25.0, 1.0
        grad = np.zeros_like(q)
        x = (q[:-2] + 4 * q[1:-1] + q[2:]) / 6.0 - ref[1:-1]      # (n-2,3)
        vdir = ref[2:] - ref[:-2]
        norm = np.linalg.norm(vdir, axis=1, keepdims=True)
        vhat = np.divide(vdir, norm, out=np.zeros_like(vdir), where=norm > 1e-9)
        along = np.sum(x * vhat, axis=1, keepdims=True)           # (n-2,1)
        perp2 = np.sum(x * x, axis=1, keepdims=True) - along * along
        cost = float(np.sum(along * along / A2 + perp2 / B2))
        gx = 2 * along * vhat / A2 + (2 * x - 2 * along * vhat) / B2
        grad[:-2] += gx / 6.0
        grad[1:-1] += 4 * gx / 6.0
        grad[2:] += gx / 6.0
        return cost, grad

    # ------------------------------------------------------------- optimisation
    def _solve(self, q0, ts, lo, hi, ref, free_idx, l_dist):
        fixed = q0.copy()

        def assemble(x):
            q = fixed.copy()
            q[free_idx] = x.reshape(-1, 3)
            return q

        def cost_and_grad(x):
            q = assemble(x)
            cs, gs = self._smoothness(q, ts)
            cd, gd = self._corridor(q, lo, hi)
            if self.cruise is None:
                cf, gf = 0.0, np.zeros_like(q)
            else:
                cf, gf = self._feasibility(q, ts)
            ct, gt = self._fitness(q, ref)
            total = (self.l_smooth * cs + l_dist * cd
                     + self.l_feas * cf + self.l_fit * ct)
            grad = (self.l_smooth * gs + l_dist * gd
                    + self.l_feas * gf + self.l_fit * gt)
            return total, grad[free_idx].ravel()

        res = minimize(cost_and_grad, q0[free_idx].ravel(), jac=True,
                       method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-6})
        return assemble(res.x), res

    def _exact_collision_status(self, spline: UniformBspline, count: int):
        """Continuously check each chord of this sampled spline representation."""
        _, positions, _, _ = spline.sample(count)
        if len(positions) < 2 or not np.all(np.isfinite(positions)):
            return False, 0.0
        free = np.fromiter(
            (self.world.segment_is_free_exact(a, b)
             for a, b in zip(positions[:-1], positions[1:])),
            dtype=bool,
            count=len(positions) - 1,
        )
        return bool(np.all(free)), float(np.mean(free))

    def _collision_status(self, spline: UniformBspline, count: int):
        if self.strict_validation:
            return self._exact_collision_status(spline, count)
        _, positions, _, _ = spline.sample(count)
        free = self.world.is_free(positions)
        return bool(np.all(free)), float(np.mean(free))

    def optimize(self, astar_waypoints_m: np.ndarray) -> OptimizeResult:
        wp = np.asarray(astar_waypoints_m, dtype=float).reshape(-1, 3)
        length = float(np.sum(np.linalg.norm(np.diff(wp, axis=0), axis=1)))
        intervals = (int(np.ceil(length / self.ctrl_spacing))
                     if self.strict_validation
                     else int(length / self.ctrl_spacing))
        K = max(self.p + 2, intervals + 1)
        interp_pts = self._resample(wp, K)                    # position samples
        ts = (1.0 if self.cruise is None else
              max(length / (self.cruise * (K + 2 - self.p)), 1e-2))
        q = UniformBspline.parameterize_to_bspline(ts, interp_pts)  # (K+2,3)
        n = len(q)

        # Enforce exact clamping at start and end to guarantee Pos(0) == wp[0]
        q[:self.p] = wp[0]
        q[-self.p:] = wp[-1]

        ref = self._resample(wp, n)                           # guide per ctrl pt
        corridor = self.sfc.boxes_for_points(ref)             # free box per ctrl pt

        free_idx = np.arange(self.p, n - self.p)
        l_dist = self.l_dist0
        iters = 0
        solver_success = False
        solver_status = -1
        solver_message = "optimizer did not run"
        solution_finite = False

        def output(spline, collision_free, free_fraction):
            return OptimizeResult(
                spline, corridor, ref, iters, collision_free, free_fraction,
                solver_success, solver_status, solver_message, solution_finite)

        for iters in range(1, self.max_rebound + 1):
            previous = q
            candidate, solver = self._solve(
                q, ts, corridor.boxes_min, corridor.boxes_max,
                ref, free_idx, l_dist)
            solution_finite = bool(
                np.all(np.isfinite(candidate))
                and np.isfinite(float(solver.fun)))
            solver_success = bool(solver.success)
            solver_status = int(solver.status)
            solver_message = str(solver.message)
            if (self.strict_validation
                    and (not solver_success or not solution_finite)):
                spline = UniformBspline(previous, ts)
                collision_free, frac = self._collision_status(
                    spline, max(200, 4 * n))
                return output(spline, collision_free, frac)
            q = candidate
            spline = UniformBspline(q, ts)
            collision_free, frac = self._collision_status(
                spline, max(200, 4 * n))
            if collision_free:
                return output(spline, True, frac)
            # rebound: refresh boxes for control points near collisions, raise λ2
            self._refresh_boxes(corridor, q, ref)
            l_dist *= self.dist_growth
        spline = UniformBspline(q, ts)
        collision_free, frac = self._collision_status(
            spline, max(200, 4 * n))
        if not self.strict_validation:
            collision_free = False
        return output(spline, collision_free, frac)

    def _refresh_boxes(self, corridor: Corridor, q, ref):
        """Reseed each control point's box at its (free) guide point.

        Control points that drifted into obstacles get a fresh free box at the
        collision-free guide; the raised λ2 then pulls them back."""
        fresh = self.sfc.boxes_for_points(ref)
        corridor.boxes_min[:] = fresh.boxes_min
        corridor.boxes_max[:] = fresh.boxes_max

    @staticmethod
    def _resample(wp: np.ndarray, n: int) -> np.ndarray:
        d = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))))
        if d[-1] <= 0:
            return np.repeat(wp[:1], n, axis=0)
        targets = np.linspace(0.0, d[-1], n)
        return np.column_stack([np.interp(targets, d, wp[:, k]) for k in range(3)])
