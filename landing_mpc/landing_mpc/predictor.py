"""Target motion prediction over the MPC horizon (§3.6 of the spec).

Three modes of increasing curvature capacity:

    const_vel   : straight target, p += v*dt each step, a = 0.
    const_accel : gentle curve, hold the current a_target constant.
    poly        : sharp curve, least-squares polynomial fit of the recent
                  target track, differentiated for v and a.

All return (P, V, A): (N, 3) predicted target position / velocity /
acceleration for horizon steps k = 1..N (i.e. t0+dt .. t0+N*dt).
"""

from __future__ import annotations

from collections import deque

import numpy as np


def predict_const_vel(p, v, dt, N):
    p = np.asarray(p, float)
    v = np.asarray(v, float)
    ks = np.arange(1, N + 1)[:, None]
    P = p[None, :] + ks * dt * v[None, :]
    V = np.tile(v, (N, 1))
    A = np.zeros((N, 3))
    return P, V, A


def predict_const_accel(p, v, a, dt, N):
    p = np.asarray(p, float)
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    ks = np.arange(1, N + 1)[:, None] * dt
    P = p[None, :] + ks * v[None, :] + 0.5 * ks * ks * a[None, :]
    V = v[None, :] + ks * a[None, :]
    A = np.tile(a, (N, 1))
    return P, V, A


class PolyPredictor:
    """Sliding-window polynomial fit of the target track (§3.6 mode 3).

    Reuses the poly-fit idea from the legacy unicycle code: keep the last
    ``window`` (t, p) samples, fit degree-``deg`` polynomials per axis, then
    evaluate p / v / a on the horizon.
    """

    def __init__(self, window: int = 12, deg: int = 2):
        self.window = int(window)
        self.deg = int(deg)
        self._buf = deque(maxlen=self.window)   # (t, np.array([x,y,z]))

    def update(self, t: float, p):
        self._buf.append((float(t), np.asarray(p, float)))

    def ready(self) -> bool:
        return len(self._buf) >= self.deg + 1

    def predict(self, t0: float, dt: float, N: int):
        ts = np.array([s[0] for s in self._buf])
        ps = np.array([s[1] for s in self._buf])          # (M, 3)
        t_eval = t0 + dt * np.arange(1, N + 1)
        P = np.zeros((N, 3))
        V = np.zeros((N, 3))
        A = np.zeros((N, 3))
        deg = min(self.deg, len(self._buf) - 1)
        for ax in range(3):
            c = np.polyfit(ts, ps[:, ax], deg)            # highest power first
            P[:, ax] = np.polyval(c, t_eval)
            V[:, ax] = np.polyval(np.polyder(c, 1), t_eval)
            A[:, ax] = np.polyval(np.polyder(c, 2), t_eval)
        return P, V, A
