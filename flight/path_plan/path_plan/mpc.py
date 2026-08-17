"""Local tracking **MPC** (receding-horizon) for reference following + braking.

Per axis the vehicle is a discrete **double integrator** with acceleration input

    x_{k+1} = A x_k + B a_k,     x = [p, v]^T,
    A = [[1, dt], [0, 1]],   B = [[0.5 dt^2], [dt]].                        (1)

Rolling (1) out over a horizon of ``N`` steps gives the condensed prediction

    X = Phi x_0 + Gamma U                                                   (2)

with U = [a_0, ..., a_{N-1}]^T.  Selecting positions (P = Sp X) and velocities
(V = Sv X) makes both affine in U:  P = p_free + Gp U,  V = v_free + Gv U.

Cost (tracking + control effort + terminal), quadratic in U:

    J = sum_k [ q (p_k - p_ref,k)^2 + q_v (v_k - v_ref,k)^2 + r a_k^2 ]
        + q_T (p_N - p_ref,N)^2
      = 1/2 U^T H U + f^T U + const                                        (3)

    H = 2( q Gp^T Gp + q_v Gv^T Gv + r I + q_T Gp[-1]^T Gp[-1] )
    f = 2( q Gp^T (p_free - p_ref) + q_v Gv^T (v_free - v_ref) + ... )

subject to actuator and safety limits (linear in U):

    -a_max <= a_k <= a_max            (box bounds on U)                     (4)
    -v_max <= v_k <= v_max            (linear:  -v_max <= v_free + Gv U <= v_max)
    |a_k - a_(k-1)| <= j_max dt       (linear acceleration-slew bound)       (5)

The caller selects the look-ahead knot it actually publishes and anchors that
input to the last published acceleration.  Axes are dynamically decoupled, so
three small QPs are solved independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def _condense(dt: float, N: int):
    """Build Phi-derived position/velocity maps: p = p_free + Gp U, v = v_free + Gv U.

    Returns (Fp, Fv, Gp, Gv) where p_free = Fp x0, v_free = Fv x0 (x0 = [p0, v0]).
    """
    Fp = np.zeros((N, 2))
    Fv = np.zeros((N, 2))
    Gp = np.zeros((N, N))
    Gv = np.zeros((N, N))
    for k in range(1, N + 1):
        Fp[k - 1] = (1.0, k * dt)
        Fv[k - 1] = (0.0, 1.0)
        for j in range(k):
            # contribution of a_j to p_k and v_k (exact double-integrator sums)
            lag = k - 1 - j
            Gp[k - 1, j] = dt * dt * (0.5 + lag)
            Gv[k - 1, j] = dt
    return Fp, Fv, Gp, Gv


@dataclass
class MPCResult:
    acceleration_cmd: np.ndarray   # (3,) selected output acceleration
    predicted_pos: np.ndarray      # (N, 3)
    predicted_vel: np.ndarray      # (N, 3)
    predicted_acc: np.ndarray      # (N, 3)
    success: bool


class TrackingMPC:
    def __init__(self, dt_s: float = 0.1, horizon: int = 20,
                 v_max: float = 5.0, a_max: float = 3.0,
                 j_max: float = 2.0,
                 q_pos: float = 4.0, q_vel: float = 0.4, r_acc: float = 0.05,
                 q_terminal: float = 20.0):
        self.dt = float(dt_s)
        self.N = int(horizon)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.j_max = float(j_max)
        self.q, self.qv, self.r, self.qT = q_pos, q_vel, r_acc, q_terminal
        self.Fp, self.Fv, self.Gp, self.Gv = _condense(self.dt, self.N)
        self.D = np.eye(self.N) - np.eye(self.N, k=-1)
        self._warm = np.zeros((3, self.N))
        self._a_prev = np.zeros(3)

    def reset(self):
        """Forget a plan when the accepted B-spline changes discontinuously."""
        self._warm.fill(0.0)
        self._a_prev.fill(0.0)

    def _solve_axis(self, p0, v0, p_ref, v_ref, warm, a_prev,
                    output_step):
        x0 = np.array([p0, v0])
        p_free = self.Fp @ x0                      # eq. (2) free response
        v_free = self.Fv @ x0
        Gp, Gv = self.Gp, self.Gv

        # eq. (3) Hessian and gradient
        H = 2.0 * (self.q * Gp.T @ Gp + self.qv * Gv.T @ Gv + self.r * np.eye(self.N))
        H += 2.0 * self.qT * np.outer(Gp[-1], Gp[-1])
        f = 2.0 * (self.q * Gp.T @ (p_free - p_ref)
                   + self.qv * Gv.T @ (v_free - v_ref))
        f += 2.0 * self.qT * Gp[-1] * (p_free[-1] - p_ref[-1])

        def cost(u):
            return 0.5 * u @ H @ u + f @ u

        def grad(u):
            return H @ u + f

        # eq. (4) velocity limits as a linear constraint on U
        constraints = [LinearConstraint(
            Gv, -self.v_max - v_free, self.v_max - v_free)]
        if self.j_max > 0.0:
            jerk_step = self.j_max * self.dt
            lower = np.full(self.N, -jerk_step)
            upper = np.full(self.N, jerk_step)
            lower[0] += a_prev
            upper[0] += a_prev
            constraints.append(LinearConstraint(self.D, lower, upper))
            if output_step:
                selector = np.zeros((1, self.N))
                selector[0, output_step] = 1.0
                constraints.append(LinearConstraint(
                    selector, [a_prev - jerk_step], [a_prev + jerk_step]))
        bounds = [(-self.a_max, self.a_max)] * self.N
        res = minimize(cost, warm, jac=grad, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 60, "ftol": 1e-4})
        if res.success and np.all(np.isfinite(res.x)):
            return res.x, True
        # Never replay a stale warm start after a failed solve.  A bounded
        # constant deceleration is deterministic and moves every axis toward a
        # hold while the mission manager falls back to its last safe route.
        brake = float(np.clip(-v0 / max(self.N * self.dt, self.dt),
                              -self.a_max, self.a_max))
        return np.full(self.N, brake), False

    def solve(self, position, velocity, reference_positions, reference_velocities,
              applied_acceleration=None, output_step=0) -> MPCResult:
        """Solve all three axes for the current state and reference horizon.

        ``reference_positions`` / ``reference_velocities`` are (N, 3) look-ahead
        samples of the B-spline trajectory ahead of the vehicle.  When a caller
        streams a look-ahead knot, ``output_step`` anchors that exact knot to
        ``applied_acceleration`` so the outgoing acceleration slew stays bounded.
        """
        pos = np.asarray(position, float)
        vel = np.asarray(velocity, float)
        ref_p = np.asarray(reference_positions, float).reshape(self.N, 3)
        ref_v = np.asarray(reference_velocities, float).reshape(self.N, 3)
        output_step = int(output_step)
        if not 0 <= output_step < self.N:
            raise ValueError('output_step must index the MPC horizon')
        applied = (self._a_prev.copy() if applied_acceleration is None else
                   np.asarray(applied_acceleration, float).reshape(3))
        if not np.all(np.isfinite(applied)):
            raise ValueError('applied_acceleration must be finite')

        U = np.zeros((3, self.N))
        ok = True
        for ax in range(3):
            u, solved = self._solve_axis(
                pos[ax], vel[ax], ref_p[:, ax], ref_v[:, ax],
                self._warm[ax], applied[ax], output_step)
            U[ax] = u
            ok &= solved
            self._warm[ax] = u if solved else 0.0
            self._a_prev[ax] = u[output_step] if solved else 0.0

        # forward-simulate the applied horizon for diagnostics / preview
        pred_pos = np.column_stack([self.Fp @ np.array([pos[a], vel[a]]) + self.Gp @ U[a]
                                    for a in range(3)])
        pred_vel = np.column_stack([self.Fv @ np.array([pos[a], vel[a]]) + self.Gv @ U[a]
                                    for a in range(3)])
        return MPCResult(
            U[:, output_step].copy(), pred_pos, pred_vel, U.T.copy(), ok)


def depth_avoidance_offset(nearest_m: float, tangent_xy: np.ndarray,
                           trigger_m: float = 10.0, emergency_m: float = 4.0,
                           lateral_m: float = 7.0) -> tuple[np.ndarray, float]:
    """Turn a forward depth reading into a lateral reference offset + speed scale.

    Returns ``(offset_xy_world, speed_scale)``.  Beyond ``trigger_m`` there is no
    effect.  The offset is perpendicular to the path tangent (left normal), and
    the cruise speed is scaled down linearly to zero at ``emergency_m``.
    """
    if not np.isfinite(nearest_m) or nearest_m >= trigger_m:
        return np.zeros(2), 1.0
    normal = np.array([-tangent_xy[1], tangent_xy[0]], float)
    n = np.linalg.norm(normal)
    normal = normal / n if n > 1e-9 else np.array([0.0, 1.0])
    closeness = np.clip((trigger_m - nearest_m) / (trigger_m - emergency_m), 0.0, 1.0)
    speed_scale = float(np.clip((nearest_m - emergency_m) / (trigger_m - emergency_m),
                                0.0, 1.0))
    return normal * lateral_m * closeness, speed_scale
