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

Only the first input ``a_0`` is applied; the horizon is re-solved every tick
(receding horizon).  Axes are dynamically decoupled, so three small QPs are
solved independently.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize

# SLSQP evaluates trial points outside the box bounds during its line search and
# clips them itself -- harmless and expected, but scipy emits a RuntimeWarning each
# time, which floods the log at the control rate. Silence just that message.
warnings.filterwarnings(
    "ignore", message="Values in x were outside bounds",
    category=RuntimeWarning, module="scipy.optimize._slsqp_py")


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
    acceleration_cmd: np.ndarray   # (3,) first input a_0 to apply now
    predicted_pos: np.ndarray      # (N, 3)
    predicted_vel: np.ndarray      # (N, 3)
    success: bool


class TrackingMPC:
    def __init__(self, dt_s: float = 0.1, horizon: int = 20,
                 v_max: float = 5.0, a_max: float = 3.0,
                 q_pos: float = 4.0, q_vel: float = 0.4, r_acc: float = 0.05,
                 q_terminal: float = 20.0):
        self.dt = float(dt_s)
        self.N = int(horizon)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.q, self.qv, self.r, self.qT = q_pos, q_vel, r_acc, q_terminal
        self.Fp, self.Fv, self.Gp, self.Gv = _condense(self.dt, self.N)
        self._warm = np.zeros((3, self.N))

    def _solve_axis(self, p0, v0, p_ref, v_ref, warm) -> np.ndarray:
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
        vel_con = LinearConstraint(Gv, -self.v_max - v_free, self.v_max - v_free)
        bounds = [(-self.a_max, self.a_max)] * self.N
        warm = np.clip(warm, -self.a_max, self.a_max)   # start inside the box bounds
        res = minimize(cost, warm, jac=grad, method="SLSQP",
                       bounds=bounds, constraints=[vel_con],
                       options={"maxiter": 60, "ftol": 1e-4})
        return res.x if res.success else np.clip(warm, -self.a_max, self.a_max)

    def solve(self, position, velocity, reference_positions, reference_velocities
              ) -> MPCResult:
        """Solve all three axes for the current state and reference horizon.

        ``reference_positions`` / ``reference_velocities`` are (N, 3) look-ahead
        samples of the B-spline trajectory ahead of the vehicle.
        """
        pos = np.asarray(position, float)
        vel = np.asarray(velocity, float)
        ref_p = np.asarray(reference_positions, float).reshape(self.N, 3)
        ref_v = np.asarray(reference_velocities, float).reshape(self.N, 3)

        U = np.zeros((3, self.N))
        ok = True
        for ax in range(3):
            u = self._solve_axis(pos[ax], vel[ax], ref_p[:, ax], ref_v[:, ax],
                                 self._warm[ax])
            U[ax] = u
        self._warm = U

        # forward-simulate the applied horizon for diagnostics / preview
        pred_pos = np.column_stack([self.Fp @ np.array([pos[a], vel[a]]) + self.Gp @ U[a]
                                    for a in range(3)])
        pred_vel = np.column_stack([self.Fv @ np.array([pos[a], vel[a]]) + self.Gv @ U[a]
                                    for a in range(3)])
        return MPCResult(U[:, 0].copy(), pred_pos, pred_vel, ok)


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
