"""Compact descent MPC: centre over the marker, then come down.

Deliberately NOT an import of `simulation/landing_mpc`. That stack solves a
harder problem (rendezvous with a 3 m/s deck, gimbal slew budgets, a two-stage
cone QP) and it lives under simulation/ because half of it reads target truth
out of Gazebo. A vehicle should not carry the simulation stack to fly, and this
descent is a much smaller problem: a marker that is roughly stationary under a
vehicle that is already above it.

So this is the same *idea* — a receding-horizon QP over a double integrator in
coordinates relative to the marker — reduced to what the hardware case needs:

    state    x = [p, v]     p = vehicle - marker, per axis
    input    a              commanded acceleration
    horizon  N steps of dt, condensed so the QP is over `a` alone

    minimise   Σ w_p‖p_k‖² + w_v‖v_k‖² + w_a‖a_k‖² + w_j‖a_k - a_{k-1}‖²
    subject to |a| ≤ a_max,  |a_k - a_{k-1}| ≤ j_max·dt,  |v| ≤ v_max

The vertical axis is not solved by the QP. It is a plain rate command gated on
horizontal alignment:

    v_z = -v_descent   while ‖p_xy‖ ≤ align_radius, else 0

which is the same "align before you descend" rule the big stack enforces with a
cone, expressed in the one line the simple case actually needs. Getting this
wrong is the failure that matters — descending while off-centre walks the
marker out of the camera's view and the vehicle lands on stale information.

No scipy: the constrained problem is solved by projected gradient descent on the
condensed cost, which is convex, small (N≈15) and warm-started from the previous
solve. That keeps the flight package's dependencies to numpy.
"""

from __future__ import annotations

import numpy as np


class DescentMPC:
    """Horizontal-plane MPC over marker-relative coordinates.

    All tuning lives in the node that constructs this, never in a launch file.
    """

    def __init__(self, dt: float, horizon: int, v_max: float, a_max: float,
                 j_max: float, w_p: float, w_v: float, w_a: float,
                 w_j: float, iterations: int = 60):
        self.dt = float(dt)
        self.N = int(horizon)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.j_max = float(j_max)
        self.w_p, self.w_v = float(w_p), float(w_v)
        self.w_a, self.w_j = float(w_a), float(w_j)
        self.iterations = int(iterations)

        # Condensing: p = F_p x0 + G_p a,  v = F_v x0 + G_v a.  Built once —
        # they depend only on dt and N, so rebuilding them per solve would be
        # pure waste at the rate this runs.
        n, dt_ = self.N, self.dt
        k = np.arange(1, n + 1)
        self.F_p = np.column_stack([np.ones(n), k * dt_])
        self.F_v = np.column_stack([np.zeros(n), np.ones(n)])
        self.G_p = np.zeros((n, n))
        self.G_v = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                # a_j acts for half a step on position the step it is applied,
                # then as accumulated velocity for every step after.
                self.G_p[i, j] = dt_ * dt_ * (0.5 + (i - j))
                self.G_v[i, j] = dt_
        # first-difference matrix for the jerk term
        self.D = np.eye(n) - np.eye(n, k=-1)

        self._warm = {0: np.zeros(n), 1: np.zeros(n)}
        self._a_prev = {0: 0.0, 1: 0.0}
        self.fails = 0
        self.solves = 0

    # ------------------------------------------------------------------ solve
    def _solve_axis(self, axis: int, p0: float, v0: float) -> np.ndarray:
        x0 = np.array([p0, v0])
        p_free = self.F_p @ x0
        v_free = self.F_v @ x0
        a_prev = self._a_prev[axis]

        # cost = w_p‖p‖² + w_v‖v‖² + w_a‖a‖² + w_j‖Da - e0·a_prev‖²
        H = (2.0 * (self.w_p * self.G_p.T @ self.G_p
                    + self.w_v * self.G_v.T @ self.G_v
                    + self.w_a * np.eye(self.N)
                    + self.w_j * self.D.T @ self.D))
        f = 2.0 * (self.w_p * self.G_p.T @ p_free
                   + self.w_v * self.G_v.T @ v_free)
        e0 = np.zeros(self.N)
        e0[0] = a_prev
        f -= 2.0 * self.w_j * (self.D.T @ e0)

        # Projected gradient with a fixed step from the Lipschitz constant, so
        # it cannot diverge no matter how the weights are retuned.
        step = 1.0 / max(np.linalg.norm(H, 2), 1e-9)
        a = self._warm[axis].copy()
        for _ in range(self.iterations):
            a = self._project(a - step * (H @ a + f), a_prev)
        self._warm[axis] = a
        return a

    def _project(self, a: np.ndarray, a_prev: float) -> np.ndarray:
        """Clamp onto the acceleration, jerk and velocity limits.

        Order matters: acceleration first (a hard actuator bound), then jerk
        relative to the previous applied command, then velocity, which is a
        consequence of the others rather than a limit on `a` directly — so it is
        enforced by scaling back the accelerations that would overshoot it.
        """
        a = np.clip(a, -self.a_max, self.a_max)
        lim = self.j_max * self.dt
        prev = a_prev
        for i in range(a.size):
            a[i] = np.clip(a[i], prev - lim, prev + lim)
            prev = a[i]
        # velocity envelope
        v = self.G_v @ a
        over = np.abs(v) > self.v_max
        if np.any(over):
            worst = float(np.max(np.abs(v)))
            a *= self.v_max / worst
        return a

    def step(self, p_rel: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
        """One horizontal solve. Returns the acceleration to apply now, [ax, ay].

        On any numerical failure it falls back to a critically damped PD rather
        than replaying the previous plan — a stale plan keeps accelerating along
        a direction that is no longer valid, which is how a controller runs away
        instead of degrading.
        """
        self.solves += 1
        out = np.zeros(2)
        for axis in (0, 1):
            try:
                a = self._solve_axis(axis, float(p_rel[axis]), float(v_rel[axis]))
                if not np.all(np.isfinite(a)):
                    raise FloatingPointError('non-finite solution')
                out[axis] = a[0]
            except Exception:
                self.fails += 1
                self._warm[axis] = np.zeros(self.N)
                kp = 1.0
                kd = 2.0 * np.sqrt(kp)
                v_des = np.clip(-kp * float(p_rel[axis]), -self.v_max, self.v_max)
                out[axis] = np.clip(kd * (v_des - float(v_rel[axis])),
                                    -self.a_max, self.a_max)
            self._a_prev[axis] = float(out[axis])
        return out

    def reset(self) -> None:
        self._warm = {0: np.zeros(self.N), 1: np.zeros(self.N)}
        self._a_prev = {0: 0.0, 1: 0.0}
