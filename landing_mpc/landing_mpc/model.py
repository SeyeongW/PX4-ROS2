"""Discrete double-integrator model in RELATIVE (target-frame) coordinates.

Per axis the relative state is x = [p, v]^T with p = p_drone - p_target,
v = v_drone - v_target and dynamics

    p_dot = v,   v_dot = a_drone - a_target.                                 (1)

The drone acceleration ``a_drone`` is the control input; the target
acceleration ``a_target`` is a known time-varying feed-forward (§3.1 of the
spec).  Being linear with no position term in v_dot, (1) admits the exact
constant-acceleration discretisation

    p += v*dt + 0.5*a*dt^2,   v += a*dt                                      (2)

for which RK4 and Euler coincide, so no integrator choice is needed.

Rolling (2) out over ``N`` steps gives the condensed prediction (per axis)

    P = Fp x0 + Gp (U - At),   V = Fv x0 + Gv (U - At)                       (3)

where U = [a0..a_{N-1}] is the drone-accel sequence and At the target-accel
sequence.  Gp/Gv are the same maps for U and for -At because both enter (1)
through the identical channel a_drone - a_target.
"""

from __future__ import annotations

import numpy as np


def condense(dt: float, N: int):
    """Per-axis condensing maps (Fp, Fv, Gp, Gv) for P = Fp x0 + Gp A, etc.

    ``A`` is the net acceleration sequence acting on the axis (drone minus
    target).  x0 = [p0, v0].
    """
    Fp = np.zeros((N, 2))
    Fv = np.zeros((N, 2))
    Gp = np.zeros((N, N))
    Gv = np.zeros((N, N))
    for k in range(1, N + 1):
        Fp[k - 1] = (1.0, k * dt)
        Fv[k - 1] = (0.0, 1.0)
        for j in range(k):
            lag = k - 1 - j                       # steps a_j acts before step k
            Gp[k - 1, j] = dt * dt * (0.5 + lag)
            Gv[k - 1, j] = dt
    return Fp, Fv, Gp, Gv


def step_state(p: np.ndarray, v: np.ndarray, a: np.ndarray, dt: float):
    """Advance one plant tick with the exact constant-accel update (eq. 2).

    Vectorised over axes; ``p``/``v``/``a`` are (3,) arrays.
    """
    p = np.asarray(p, float)
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    p_next = p + v * dt + 0.5 * a * dt * dt
    v_next = v + a * dt
    return p_next, v_next
