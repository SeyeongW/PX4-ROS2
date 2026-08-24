"""High-rate reference streaming from the MPC plan (jerk-free setpoints).

The MPC re-solves at a modest rate (e.g. 10 Hz) but PX4 wants a dense, smooth
setpoint stream (e.g. 50 Hz).  Instead of holding the first horizon point (a
staircase reference), we stream the WHOLE predicted trajectory, sampled at the
true elapsed time.

Theory — why a quadratic is the *exact* interpolation, not an approximation:

The MPC plant is a double integrator with the acceleration held CONSTANT across
each horizon step (a zero-order hold on the input, a_k on t in [t_k, t_{k+1})).
Integrating exactly inside a step gives, with s = t - t_k in [0, dt]:

    a(t) = a_k                                  (piecewise constant)          (1)
    v(t) = v_k + a_k * s                         (piecewise linear)           (2)
    p(t) = p_k + v_k * s + 1/2 * a_k * s^2       (piecewise quadratic)        (3)

At s = dt these reproduce the MPC's own knot states (p_{k+1}, v_{k+1}) exactly,
because (3) is the same constant-accel update the condensing used.  So the
continuous optimal plan is a C^1 quadratic spline: position C^1, velocity
continuous & piecewise linear, acceleration piecewise constant.  Sampling it
yields position/velocity/acceleration setpoints that are mutually CONSISTENT,
which is what PX4's position-led controller needs to move smoothly.

The MPC solves in RELATIVE coordinates.  First reconstruct its absolute drone
trajectory q at the preview clock tau_p = tau + L.  PX4 already adds the
streamed velocity as feed-forward, so subtract the solve-time drone velocity's
inertial displacement over L-D instead of feeding it through PX4's position P
term a second time.  D is the bounded age of the vehicle state when the
setpoint reaches PX4; retaining v_d0*D places the command at the controller's
current position instead of one transport interval behind it:

    v_d0        = v_t(0) + v_rel(0)
    p_cmd(tau)  = p_t(tau_p) + p_rel(tau_p) - v_d0 * (L - D)                  (4)
    v_cmd(tau)  = v_t(tau_p) + v_rel(tau_p)
    a_cmd(tau)  = a_t(tau_p) + a_rel(tau_p)

The subtraction is constant for the life of a plan, so position, velocity and
acceleration remain mutually derivative-consistent.  D is clamped to [0,L].
It removes the artificial v_d0*L position carrot at constant velocity while
retaining only the state-age compensation and the small acceleration preview
that made L=0 unable to start a chase.

Symbols:
    dt      MPC step (horizon knot spacing) [s]
    N       horizon length (number of steps); plan spans [0, N*dt]
    t_k     = k*dt, the k-th knot time from the solve instant [s]
    tau     elapsed time since the solve instant [s]
    L       fixed absolute-plan look-ahead [s]
    D       bounded vehicle-state age at PX4 consumption [s]
    tau_p   preview clock, clamp(tau + L, 0, N*dt) [s]
    s       intra-step time, s = tau_p - t_k in [0, dt] [s]
    p_k,v_k relative position/velocity knots (3,) [m], [m/s]
    a_k     relative acceleration held over step k (3,) [m/s^2]
    p_t,v_t,a_t  target predicted position/velocity/accel at tau [m],[m/s],[m/s^2]
"""

from __future__ import annotations

import numpy as np


class HorizonReference:
    """Holds one MPC plan and samples it (O(1)) at any elapsed time.

    Relative plan knots are P_k = [p0; pred_pos] and V_k = [v0; pred_vel]
    (k = 0..N), with per-step relative accel A_k = pred_acc (k = 0..N-1).
    The target is extrapolated analytically from its state at the solve instant.
    """

    def __init__(self, lead_s: float = 0.0, state_age_s: float = 0.0):
        """``lead_s`` (L) is a constant look-ahead for the absolute plan.

        Without it the command at tau = 0 is the drone's CURRENT state ("stay
        put"), and it only pulls forward as tau grows — so the effective
        look-ahead sawtooths between 0 and the solve period on every re-plan,
        exciting an oscillation at the solve rate.  Sampling at tau + L keeps
        the commanded point a fixed L seconds ahead of *now*, continuously
        across re-plans.  Position subtracts the solve-time vehicle velocity
        over ``L-D``, where ``D=state_age_s`` is the bounded age of that state
        when PX4 consumes the command.  This prevents double-counting constant
        motion without anchoring the position setpoint behind the vehicle.  L
        = one MPC step and D = one control tick are the natural runtime choices.
        """
        self._ok = False
        self.lead = float(lead_s)
        self.state_age = float(state_age_s)

    def set_plan(self, p_rel0, v_rel0, pred_pos, pred_vel, pred_acc, dt,
                 p_t0, v_t0, a_t0):
        """Install a fresh plan sampled from a solve at tau = 0.

        p_rel0/v_rel0 : (3,) relative state at the solve instant (knot 0).
        pred_pos/vel  : (N,3) relative knots for k = 1..N (from MPCResult).
        pred_acc      : (N,3) relative accel per step k = 0..N-1 (the plan U).
        p_t0/v_t0/a_t0: (3,) target state at the solve instant (feed-forward).
        """
        self.Pk = np.vstack([np.asarray(p_rel0, float), np.asarray(pred_pos, float)])
        self.Vk = np.vstack([np.asarray(v_rel0, float), np.asarray(pred_vel, float)])
        self.Ak = np.asarray(pred_acc, float)
        self.dt = float(dt)
        self.N = self.Ak.shape[0]
        self.T = self.N * self.dt
        self.p_t0 = np.asarray(p_t0, float)
        self.v_t0 = np.asarray(v_t0, float)
        self.a_t0 = np.asarray(a_t0, float)
        preview_lead = float(np.clip(self.lead, 0.0, self.T))
        compensated_lead = max(
            0.0, preview_lead - float(np.clip(
                self.state_age, 0.0, preview_lead)))
        self._position_lead_compensation = (
            self.v_t0 + self.Vk[0]) * compensated_lead
        self._ok = True

    def ready(self) -> bool:
        return self._ok

    def reset(self):
        self._ok = False

    def sample(self, tau: float):
        """Return absolute (pos, vel, acc) setpoint at elapsed time ``tau``.

        Implements eqs. (1)-(4): pick the active step, evaluate the exact
        quadratic in relative coords, add the analytic target extrapolation.
        """
        tau_preview = float(np.clip(tau + self.lead, 0.0, self.T))
        j = min(int(tau_preview / self.dt), self.N - 1)  # active step index
        s = tau_preview - j * self.dt                # intra-step time in [0, dt]

        a_rel = self.Ak[j]                                   # (1)
        v_rel = self.Vk[j] + a_rel * s                       # (2)
        p_rel = self.Pk[j] + self.Vk[j] * s + 0.5 * a_rel * s * s   # (3)

        p_t = (self.p_t0 + self.v_t0 * tau_preview
               + 0.5 * self.a_t0 * tau_preview * tau_preview)
        v_t = self.v_t0 + self.a_t0 * tau_preview
        a_t = self.a_t0

        return (p_t + p_rel - self._position_lead_compensation,
                v_t + v_rel, a_t + a_rel)                    # (4)


def _selftest():
    # Build a plan from a known relative state + accel sequence; verify the
    # sampler reproduces the double-integrator knots at s = dt (continuity).
    dt, N = 0.1, 5
    rng = np.random.default_rng(0)
    p0 = rng.normal(size=3); v0 = rng.normal(size=3)
    A = rng.normal(size=(N, 3))
    # roll out the exact double integrator to get the knots the MPC would report
    P = np.zeros((N, 3)); V = np.zeros((N, 3))
    p, v = p0.copy(), v0.copy()
    for k in range(N):
        p = p + v * dt + 0.5 * A[k] * dt * dt
        v = v + A[k] * dt
        P[k], V[k] = p, v
    ref = HorizonReference()
    ref.set_plan(p0, v0, P, V, A, dt, np.zeros(3), np.zeros(3), np.zeros(3))
    # at each knot time the sample must equal the rolled-out knot
    for k in range(1, N + 1):
        pk, vk, ak = ref.sample(k * dt)
        assert np.allclose(pk, P[k - 1], atol=1e-9), (k, pk, P[k - 1])
        assert np.allclose(vk, V[k - 1], atol=1e-9), (k, vk, V[k - 1])
    # mid-step continuity: velocity is the derivative of position (finite diff)
    t = 0.23
    p_a, _, _ = ref.sample(t)
    p_b, v_b, _ = ref.sample(t + 1e-6)
    assert np.allclose((p_b - p_a) / 1e-6, v_b, atol=1e-3)
    # target extrapolation: pure const-velocity target adds v_t * tau
    ref.set_plan(np.zeros(3), np.zeros(3), np.zeros((N, 3)), np.zeros((N, 3)),
                 np.zeros((N, 3)), dt, np.array([1.0, 0, 0]),
                 np.array([2.0, 0, 0]), np.zeros(3))
    p_c, v_c, _ = ref.sample(0.5)
    assert np.allclose(p_c, [1.0 + 2.0 * 0.5, 0, 0]) and np.allclose(v_c, [2, 0, 0])
    print('reference sampler self-test: PASS')


if __name__ == '__main__':
    _selftest()
