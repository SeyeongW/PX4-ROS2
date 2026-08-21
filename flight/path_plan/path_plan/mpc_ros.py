"""mpc_ros-style **kinematic tracking MPC** (Geonhee-LEE/mpc_ros, Udacity lineage).

Ported/adapted from the ground-vehicle formulation to a multirotor flying at a
cruise altitude: the horizontal motion is tracked with a **unicycle /
differential-drive** model whose inputs are yaw-rate ``omega`` and linear
acceleration ``a``; the vertical axis is a decoupled altitude hold.  Nosing the
vehicle along the path also points the forward depth camera where it is going.

Reference as a polynomial (mpc_ros)
-----------------------------------
A window of the reference path ahead is expressed in the **vehicle frame** and
fit by a cubic polynomial ``f(x) = c0 + c1 x + c2 x^2 + c3 x^3``.  In that frame
the vehicle sits at (x, y, psi) = (0, 0, 0), so the initial errors are

    cte_0  = f(0) - 0 = c0                       (cross-track error)
    epsi_0 = 0 - atan(f'(0)) = -atan(c1)         (heading error)

Model + error dynamics (mpc_ros / Udacity MPC), step dt
-------------------------------------------------------
    x_{t+1}    = x_t + v_t cos(psi_t) dt
    y_{t+1}    = y_t + v_t sin(psi_t) dt
    psi_{t+1}  = psi_t + omega_t dt
    v_{t+1}    = v_t + a_t dt
    cte_{t+1}  = (f(x_t) - y_t) + v_t sin(epsi_t) dt
    epsi_{t+1} = (psi_t - atan(f'(x_t))) + omega_t dt

Cost (tracking + effort + input smoothness), the mpc_ros weight structure
-------------------------------------------------------------------------
    J = sum_t [ w_cte cte_t^2 + w_epsi epsi_t^2 + w_v (v_t - v_ref)^2 ]
      + sum_t [ w_omega omega_t^2 + w_a a_t^2 ]
      + sum_t [ w_domega (omega_{t+1}-omega_t)^2 + w_da (a_{t+1}-a_t)^2 ]

Solved by **single shooting**: the decision vector is the input sequence
U = [omega_0..omega_{N-1}, a_0..a_{N-1}]; states are rolled out through the
nonlinear model, and SLSQP minimises J under input box bounds |omega|<=omega_max,
|a|<=a_max (mpc_ros uses IPOPT/CppAD with the dynamics as equality constraints;
single shooting with bounds is the dependency-light equivalent here).  Only the
first input pair is applied (receding horizon).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


def _poly_fit(x: np.ndarray, y: np.ndarray, order: int = 3) -> np.ndarray:
    """Least-squares polynomial coefficients [c0, c1, ..., c_order]."""
    order = min(order, max(1, len(x) - 1))
    
    # 궤적이 직선(Linear)에 가까운 경우, 강제로 1차식(Linear fit)으로 낮춤 (RankWarning 방지)
    if len(x) >= 3:
        dx = x[-1] - x[0]
        dy = y[-1] - y[0]
        dist_sq = dx**2 + dy**2
        if dist_sq > 1e-8:
            # 시작점과 끝점을 잇는 직선과 중간 점들 사이의 최대 직교 거리 오차 계산
            cross = np.abs(dx * (y - y[0]) - dy * (x - x[0]))
            max_dev = np.max(cross) / np.sqrt(dist_sq)
            if max_dev < 5e-2:  # If deviation is < 5cm, treat as straight line to prevent polyfit divergence
                order = min(order, 1)
        else:
            order = min(order, 1)

    return np.polyfit(x, y, order)[::-1]  # numpy returns highest-power first


def _poly_val(c: np.ndarray, x):
    return sum(ck * x ** k for k, ck in enumerate(c))


def _poly_der(c: np.ndarray, x):
    return sum(k * ck * x ** (k - 1) for k, ck in enumerate(c) if k >= 1)


@dataclass
class UnicycleMPCResult:
    velocity_world: np.ndarray   # (3,) vx, vy, vz setpoint in ENU
    yaw_rate: float              # omega_0 [rad/s]
    predicted_xy: np.ndarray     # (N, 2) horizon in world frame (preview)
    cte0: float
    epsi0: float
    success: bool


@dataclass
class Weights:
    cte: float = 2.0
    epsi: float = 4.0
    v: float = 1.0
    omega: float = 0.5
    a: float = 0.05
    domega: float = 2.0
    da: float = 0.1


class UnicycleMPC:
    def __init__(self, dt_s: float = 0.1, horizon: int = 20,
                 v_ref: float = 4.0, v_max: float = 5.0,
                 omega_max: float = 1.2, a_max: float = 3.0,
                 z_kp: float = 0.8, vz_max: float = 2.0,
                 weights: Weights | None = None, max_iter: int = 40):
        self.dt = float(dt_s)
        self.max_iter = int(max_iter)
        self.N = int(horizon)
        self.v_ref = float(v_ref)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.a_max = float(a_max)
        self.z_kp = float(z_kp)
        self.vz_max = float(vz_max)
        self.w = weights or Weights()
        self._warm = np.zeros(2 * self.N)

    # ------------------------------------------------------------- rollout/cost
    def _rollout(self, U: np.ndarray, v0: float, cte0: float, epsi0: float,
                 coeffs: np.ndarray):
        N, dt = self.N, self.dt
        omega, acc = U[:N], U[N:]
        x = y = psi = 0.0
        v, cte, epsi = v0, cte0, epsi0
        xs = np.empty(N)
        ys = np.empty(N)
        J = 0.0
        for t in range(N):
            J += (self.w.cte * cte * cte + self.w.epsi * epsi * epsi
                  + self.w.v * (v - self.v_ref) ** 2
                  + self.w.omega * omega[t] ** 2 + self.w.a * acc[t] ** 2)
            if t + 1 < N:
                J += (self.w.domega * (omega[t + 1] - omega[t]) ** 2
                      + self.w.da * (acc[t + 1] - acc[t]) ** 2)
            f = _poly_val(coeffs, x)
            psides = np.arctan(_poly_der(coeffs, x))
            
            # Runge-Kutta 4 (RK4) integration
            v_mid = v + 0.5 * acc[t] * dt
            psi_mid = psi + 0.5 * omega[t] * dt
            epsi_mid = epsi + 0.5 * omega[t] * dt
            
            v_end = v + acc[t] * dt
            psi_end = psi + omega[t] * dt
            epsi_end = epsi + omega[t] * dt
            
            k1_x = v * np.cos(psi)
            k2_x = v_mid * np.cos(psi_mid)
            k4_x = v_end * np.cos(psi_end)
            
            k1_y = v * np.sin(psi)
            k2_y = v_mid * np.sin(psi_mid)
            k4_y = v_end * np.sin(psi_end)
            
            x_next = x + (dt / 6.0) * (k1_x + 4 * k2_x + k4_x)
            y_next = y + (dt / 6.0) * (k1_y + 4 * k2_y + k4_y)
            psi_next = psi_end
            v_next = v_end
            
            # 최적화기(SLSQP)의 Gradient 폭발을 막기 위해 오차(Error) 상태는 Euler 투영 유지
            cte_next = (f - y) + v * np.sin(epsi) * dt
            epsi_next = (psi - psides) + omega[t] * dt
            
            x, y, psi, v, cte, epsi = (x_next, y_next, psi_next, v_next,
                                       cte_next, epsi_next)
            xs[t], ys[t] = x, y
        return J, xs, ys

    def solve(self, position_m, yaw_rad: float, speed_m_s: float,
              altitude_z_m: float, reference_positions_world) -> UnicycleMPCResult:
        pos = np.asarray(position_m, float)
        ref = np.asarray(reference_positions_world, float).reshape(-1, 3)

        # reference window in the vehicle frame (rotate by -yaw, translate)
        c, s = np.cos(-yaw_rad), np.sin(-yaw_rad)
        R = np.array([[c, -s], [s, c]])
        rel = (ref[:, :2] - pos[:2]) @ R.T
        coeffs = _poly_fit(rel[:, 0], rel[:, 1], order=3)
        cte0 = float(_poly_val(coeffs, 0.0))
        epsi0 = float(-np.arctan(_poly_der(coeffs, 0.0)))

        def cost(U):
            return self._rollout(U, speed_m_s, cte0, epsi0, coeffs)[0]

        bounds = [(-self.omega_max, self.omega_max)] * self.N \
            + [(-self.a_max, self.a_max)] * self.N
        res = minimize(cost, self._warm, method="SLSQP", bounds=bounds,
                       options={"maxiter": self.max_iter, "ftol": 1e-3})
        U = res.x if res.success else np.clip(self._warm, -self.a_max, self.a_max)
        # warm-start shift for the next tick
        self._warm = np.concatenate([U[1:self.N], U[self.N - 1:self.N],
                                     U[self.N + 1:], U[2 * self.N - 1:]])
        omega0, a0 = float(U[0]), float(U[self.N])

        # apply first input: new heading/speed -> world velocity setpoint
        new_yaw = yaw_rad + omega0 * self.dt
        new_v = float(np.clip(speed_m_s + a0 * self.dt, 0.0, self.v_max))
        vx = new_v * np.cos(new_yaw)
        vy = new_v * np.sin(new_yaw)

        # decoupled altitude hold: P on z error -> vz setpoint (mpc_ros is planar)
        z_ref = float(ref[min(len(ref) - 1, self.N // 2), 2])
        vz = float(np.clip(self.z_kp * (z_ref - altitude_z_m), -self.vz_max, self.vz_max))

        # roll predicted horizon back into the world frame for preview
        _, xs, ys = self._rollout(U, speed_m_s, cte0, epsi0, coeffs)
        Rinv = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                         [np.sin(yaw_rad), np.cos(yaw_rad)]])
        pred = pos[:2] + np.column_stack([xs, ys]) @ Rinv.T

        return UnicycleMPCResult(np.array([vx, vy, vz]), omega0, pred,
                                 cte0, epsi0, bool(res.success))
