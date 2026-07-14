"""Uniform cubic B-spline (ego-planner convention).

Unlike a clamped spline, a **uniform** B-spline has equally spaced knots, which
makes its derivatives simple control-point differences with knot interval ``ts``:

    v_i    = (q_{i+1} − q_i) / ts
    a_i    = (q_{i+2} − 2 q_{i+1} + q_i) / ts²
    jerk_i = (q_{i+3} − 3 q_{i+2} + 3 q_{i+1} − q_i) / ts³

The optimizer (`bspline_optimizer.py`) penalises exactly these differences, so
smoothness/feasibility gradients are closed form.  Curve **evaluation** for
sampling/plots uses ``scipy.interpolate.BSpline`` with a uniform knot vector.

``parameterize_to_bspline`` recovers control points that make the curve pass
through a set of sample points with prescribed boundary velocity/acceleration —
the ego-planner ``parameterizeToBspline`` least-squares fit.  Passing zero
boundary velocity yields a curve that starts and stops at rest (good for
takeoff/landing and no endpoint speed spike).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


class UniformBspline:
    def __init__(self, control_points: np.ndarray, ts: float, degree: int = 3):
        self.q = np.asarray(control_points, dtype=float).reshape(-1, 3)
        self.ts = float(ts)
        self.p = int(degree)
        n = len(self.q)
        if n <= self.p:
            raise ValueError("need more than `degree` control points")
        # uniform knot vector, spacing ts; valid domain is [knots[p], knots[n]]
        self.knots = np.arange(n + self.p + 1, dtype=float) * self.ts

    # ------------------------------------------------------- control-point diffs
    def velocity_points(self) -> np.ndarray:
        return np.diff(self.q, axis=0) / self.ts

    def acceleration_points(self) -> np.ndarray:
        return np.diff(self.q, n=2, axis=0) / (self.ts ** 2)

    # ------------------------------------------------------------------ sampling
    def duration(self) -> float:
        return (len(self.q) - self.p) * self.ts

    def _bspline(self, deriv: int = 0) -> BSpline:
        spline = BSpline(self.knots, self.q, self.p, extrapolate=False)
        return spline.derivative(deriv) if deriv else spline

    def sample(self, num: int = 200):
        """Return (t, position, velocity, acceleration) over the valid domain.

        The evaluation times are clamped strictly inside the half-open domain
        ``[knots[p], knots[n])``.  Without this, floating-point round-off makes
        ``t0 + duration`` land a hair past ``knots[n]``, where scipy's
        ``extrapolate=False`` returns NaN; the old ``nan_to_num`` then silently
        turned the final sample into (0, 0, 0), corrupting the trajectory tail
        (and hence any MPC reference / termination built from it).
        """
        t0 = self.knots[self.p]
        t_end = self.knots[len(self.q)]
        duration = self.duration()
        t = np.clip(t0 + np.linspace(0.0, duration, num), t0, np.nextafter(t_end, t0))
        pos = self._bspline(0)(t)
        vel = self._bspline(1)(t)
        acc = self._bspline(2)(t)
        return np.linspace(0.0, duration, num), pos, vel, acc

    # ------------------------------------------------------------- construction
    @staticmethod
    def parameterize_to_bspline(ts: float, points: np.ndarray,
                                start_end_derivatives=None) -> np.ndarray:
        """Least-squares control points through ``points`` (ego-planner).

        For K sample points the uniform cubic relation at each sample is
        ``(q_i + 4 q_{i+1} + q_{i+2})/6 = point_i``; two velocity and two
        acceleration rows pin the boundaries.  Yields K+2 control points.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        K = len(pts)
        if K < 2:
            raise ValueError("parameterize needs at least two points")
        if start_end_derivatives is None:
            z = np.zeros(3)
            v0 = v1 = a0 = a1 = z
        else:
            v0, v1, a0, a1 = (np.asarray(d, float) for d in start_end_derivatives)

        rows = K + 4
        cols = K + 2
        A = np.zeros((rows, cols))
        b = np.zeros((rows, 3))
        for i in range(K):                       # position constraints
            A[i, i:i + 3] = (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)
            b[i] = pts[i]
        A[K, 0:3] = (-1.0 / (2 * ts), 0.0, 1.0 / (2 * ts));      b[K] = v0
        A[K + 1, K - 1:K + 2] = (-1.0 / (2 * ts), 0.0, 1.0 / (2 * ts)); b[K + 1] = v1
        A[K + 2, 0:3] = (1.0 / ts ** 2, -2.0 / ts ** 2, 1.0 / ts ** 2); b[K + 2] = a0
        A[K + 3, K - 1:K + 2] = (1.0 / ts ** 2, -2.0 / ts ** 2, 1.0 / ts ** 2)
        b[K + 3] = a1
        ctrl, *_ = np.linalg.lstsq(A, b, rcond=None)
        return ctrl
