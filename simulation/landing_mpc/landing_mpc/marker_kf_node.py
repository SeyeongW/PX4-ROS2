"""marker_kf_node — ONE job: turn noisy, intermittent marker fixes into a
continuous state estimate, and COAST when the camera loses the marker.

This is the estimator half of the stack (the MPC is the controller half): the
Kalman filter answers "where is the target now, and how fast is it moving",
fusing past measurements; the MPC then answers "what acceleration do I command".

Why it is needed: marker fixes are intermittent during oblique viewing and the
handover between the two 1.3 m markers and centred 0.30 m marker.  With no new
measurement the filter briefly predicts from the last estimate instead of
dropping the target immediately.

Subscribes
    /marker/measured    geometry_msgs/PointStamped   raw per-detection fix (ENU)
Publishes
    /marker/position    geometry_msgs/PointStamped   filtered/coasted (ENU)
    /marker/velocity    geometry_msgs/Vector3Stamped filtered (ENU)
    /marker/valid       std_msgs/Bool                false once coasting too long

Model — constant velocity in the horizontal plane (z is the known deck height,
so it is not estimated).  With state x = [px, py, vx, vy]^T and step dt:

    F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]        state transition
    H = [[1,0,0,0],[0,1,0,0]]                              we measure position
    Q = sigma_a^2 * [[dt^4/4,0,dt^3/2,0],                  accel-driven process
                     [0,dt^4/4,0,dt^3/2],                   noise (a white accel
                     [dt^3/2,0,dt^2,0],                     of std sigma_a)
                     [0,dt^3/2,0,dt^2]]
    R = sigma_m^2 * I                                      measurement noise

    predict:  x <- F x            P <- F P F^T + Q
    update:   y <- z - H x        S <- H P H^T + R
              K <- P H^T S^-1     x <- x + K y     P <- (I - K H) P

sigma_m defaults to 0.06 m, which is what the detector actually measured in a
static hover (0.024-0.068 m at 4-8 m altitude) — not a guess.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Bool

from .parameter_utils import (
    DEFAULT_DECK_Z_M,
    require_finite,
    require_nonnegative,
    require_positive,
)


class MarkerKfNode(Node):
    def __init__(self):
        super().__init__('marker_kf_node')
        p = self.declare_parameter
        self.rate = require_positive('rate_hz', p('rate_hz', 50.0).value)
        self.sigma_m = require_positive(
            'sigma_meas_m', p('sigma_meas_m', 0.06).value)      # measured
        self.sigma_a = require_positive(
            'sigma_accel_m_s2',
            p('sigma_accel_m_s2', 1.5).value)   # target agility
        self.deck_z = require_finite(
            'deck_z', p('deck_z', DEFAULT_DECK_Z_M).value)
        # Coasting is dead reckoning: trust it for the last metre, not forever.
        self.max_coast = require_nonnegative(
            'max_coast_s', p('max_coast_s', 3.0).value)
        # A fix this far from the prediction is treated as an outlier (gating),
        # in units of the innovation standard deviation.
        self.gate_sigma = require_positive(
            'gate_sigma', p('gate_sigma', 5.0).value)

        self.x = None                     # [px, py, vx, vy]
        self.P = np.eye(4)
        self.R = (self.sigma_m ** 2) * np.eye(2)
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self._t_last = None               # last predict time
        self._t_meas = None               # last accepted measurement time
        self._rejected = 0
        # Ceiling on the lag correction.  A fix older than this is not a
        # pipeline delay, it is a bad or badly stamped message, and
        # extrapolating it by v*lag would fling the estimate rather than
        # sharpen it.  Also guards the case where the stamp is not sim time.
        self.max_lag = require_nonnegative(
            'max_lag_s', p('max_lag_s', 0.5).value)
        self._lag_sum = 0.0
        self._lag_sq = 0.0
        self._lag_n = 0
        # Floor under the measured lag jitter, so the gate is sane before there
        # is enough history to estimate it (and never zero afterwards).  Half a
        # camera frame period is the irreducible part: a fix can be stamped
        # anywhere inside the frame it came from.
        self.lag_jitter_floor = require_nonnegative(
            'lag_jitter_floor_s',
            p('lag_jitter_floor_s', 0.02).value)

        self.pos_pub = self.create_publisher(PointStamped, '/marker/position', 10)
        self.vel_pub = self.create_publisher(Vector3Stamped, '/marker/velocity', 10)
        self.valid_pub = self.create_publisher(Bool, '/marker/valid', 10)
        self.create_subscription(PointStamped, '/marker/measured',
                                 self._on_meas, 10)
        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info(
            f'marker_kf_node: const-velocity KF @ {self.rate:.0f} Hz, '
            f'sigma_meas={self.sigma_m} m, coast<={self.max_coast} s')

    # -------------------------------------------------------------- filter
    def _Q(self, dt):
        s2 = self.sigma_a ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        return s2 * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ])

    def _predict(self, dt):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _lag_jitter(self):
        """Spread of the observed pipeline lag, never below the frame floor."""
        if self._lag_n < 2:
            return self.lag_jitter_floor
        mean = self._lag_sum / self._lag_n
        var = max(0.0, self._lag_sq / self._lag_n - mean * mean)
        return max(float(np.sqrt(var)), self.lag_jitter_floor)

    def _initialise(self, z, t):
        """Start, or restart, the filter from one current marker fix."""
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([
            self.sigma_m ** 2, self.sigma_m ** 2, 4.0, 4.0])
        self._t_last = t
        self._t_meas = t

    def _on_meas(self, msg: PointStamped):
        """Fold in one fix, AT THE INSTANT IT WAS TAKEN.

        The stamp on this message is the camera's capture time: the detector
        copies it off the image (`m.header.stamp = msg.header.stamp`) and
        `marker_tf_node` interpolates the vehicle state to it rather than to
        "now".  The whole chain carries that instant deliberately — and this
        node used to throw it away and use the ARRIVAL time instead, which
        silently asserts that the marker was where it is seen to be at the
        moment the message happened to land.  Against a deck moving at 3 m/s
        that is a pure lag error of v*(pipeline delay), always pointing behind
        the target's motion, and no amount of filtering removes a bias.

        The filter has already been advanced past the capture instant by the
        50 Hz `_tick`, and rewinding a KF is far more machinery than this
        deserves.  For a constant-velocity target the equivalent move is to
        advance the MEASUREMENT to the filter's clock instead — z + v*dt with
        the filter's own velocity estimate — which is unbiased under exactly
        the model the filter already assumes.
        """
        z = np.array([msg.point.x, msg.point.y])
        t = self._now()
        expired = (self._t_meas is not None
                   and t - self._t_meas > self.max_coast)
        if self.x is None or expired:
            if expired:
                self.get_logger().info(
                    'expired marker track recovered from a fresh fix')
            self._initialise(z, t)
            return
        t_cap = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # bring the state up to now, then bring the MEASUREMENT forward to meet
        # it across the pipeline delay (see the docstring)
        dt = max(0.0, t - self._t_last)
        if dt > 0:
            self._predict(dt)
            self._t_last = t
        lag = float(np.clip(self._t_last - t_cap, 0.0, self.max_lag))
        z = z + self.x[2:4] * lag
        self._lag_sum += lag
        self._lag_sq += lag * lag
        self._lag_n += 1
        if self._lag_n % 200 == 0:
            mean = self._lag_sum / self._lag_n
            self.get_logger().info(
                f'pipeline lag mean {1000 * mean:.0f} ms — worth '
                f'{np.linalg.norm(self.x[2:4]) * mean:.2f} m at the target'
                f"'s current speed, which is what this correction removes")
        y = z - self.H @ self.x
        # The extrapolated measurement is only as good as the velocity used to
        # extrapolate it, so it carries that velocity's covariance over the lag:
        # Var(z + v*lag) = R + lag^2 * P_vv.  Leaving it at R alone makes the
        # gate reject its own correction — measured at switch-on, a 134 ms lag
        # against a 3 m/s deck moves the fix 1.07 m while a 0.06 m sensor sigma
        # sizes the gate at ~0.3 m, so every fix read as a 6-sigma outlier, the
        # filter coasted, `/marker/valid` went false and the mission held at the
        # vision gate reporting "no marker fix yet" with the detector at 35%.
        # Inflating S is also what breaks the bootstrap: the gate is widest
        # exactly while the velocity estimate is poor, and tightens by itself as
        # the filter converges.
        # ...and the lag is not a constant either.  Var(v*lag) has TWO terms —
        # lag^2*Var(v), above, and |v|^2*Var(lag) — and dropping the second one
        # leaves the gate too tight by exactly the amount the pipeline jitters:
        # at 3 m/s a 30 ms spread in arrival time is 0.09 m, larger than the
        # 0.06 m sensor sigma the gate was sized from.  Measured after adding
        # only the first term, innovations sat at 5.2-5.6 sigma against a
        # 5-sigma gate, i.e. rejected for being marginally over a bound that
        # was missing a real source of spread.  The jitter is estimated from
        # the lags actually seen rather than assumed.
        S = (self.H @ self.P @ self.H.T + self.R
             + (lag ** 2) * self.P[2:4, 2:4]
             + (self._lag_jitter() ** 2) * float(self.x[2:4] @ self.x[2:4])
             * np.eye(2))
        # chi-square style gate: reject fixes far outside the innovation spread
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > self.gate_sigma ** 2:
            self._rejected += 1
            self.get_logger().warn(
                f'outlier fix rejected (mahalanobis {np.sqrt(d2):.1f} sigma), '
                f'total {self._rejected}', throttle_duration_sec=2.0)
            return
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self._t_meas = t

    # ---------------------------------------------------------------- output
    def _tick(self):
        if self.x is None:
            self.valid_pub.publish(Bool(data=False))
            return
        t = self._now()
        dt = max(0.0, t - self._t_last)
        if dt > 0:
            self._predict(dt)            # no measurement -> pure coast
            self._t_last = t

        coasting = t - (self._t_meas if self._t_meas is not None else t)
        valid = coasting <= self.max_coast
        self.valid_pub.publish(Bool(data=bool(valid)))

        stamp = self.get_clock().now().to_msg()
        pm = PointStamped()
        pm.header.stamp = stamp
        pm.header.frame_id = 'map'
        pm.point.x, pm.point.y, pm.point.z = (float(self.x[0]), float(self.x[1]),
                                              float(self.deck_z))
        self.pos_pub.publish(pm)

        vm = Vector3Stamped()
        vm.header.stamp = stamp
        vm.header.frame_id = 'map'
        vm.vector.x, vm.vector.y, vm.vector.z = (float(self.x[2]),
                                                 float(self.x[3]), 0.0)
        self.vel_pub.publish(vm)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerKfNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
