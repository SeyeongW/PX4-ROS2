"""Is the EKF actually aided, or only pretending? — shared by every mission node.

WHY THIS EXISTS
---------------
`/mavros/local_position/pose` keeps publishing even when PX4's EKF has no
position aiding at all. When the GPS quality checks fail, EKF2 falls back to
CONSTANT POSITION mode: it stops fusing GNSS and holds the estimate still by
feeding itself a zero-velocity measurement. The pose topic looks completely
normal — it is simply no longer attached to the world.

That state is what a pose-only preflight cannot see. It reported PASSED,
prompted the operator to approve the arm, and PX4 then refused with "GPS speed
accuracy" — a rejection the node never learned about, because MAVROS cannot
decode PX4's event stream and logs it as `FCU: EVENT 133277 with args -0-0-...`.

Measured on the pad, disarmed, while that was happening:

    EKF2_REQ_SACC        0.5 m/s        (the vehicle's own parameter)
    GPS vel_acc          0.37-0.54 m/s  (oscillating across it)
    const_pos_mode       true           (so: GNSS was not being fused)
    local_position z     -5.98 m        (sitting on the ground)

The GPS speed accuracy straddled the threshold, so the ~10 s of *continuous*
passes EKF2 requires before it will START fusing GNSS never accumulated. The
z of -5.98 m is the same fact seen from the other side: with nothing correcting
it, the height estimate had walked six metres away from the ground it was
standing on.

The fix for that pad is to stop straddling the threshold: EKF2_REQ_SACC is now
1.0 m/s on the vehicle, and DEFAULT_SPEED_ACC_MAX below mirrors it. Both have to
move together — see the comment on that constant.

ONE FLAG, THREE CAUSES
----------------------
The reading above — const_pos_mode means GNSS is unfused — is only true of one
of the three things PX4 raises that flag for. In v1.18
(ekf_helper.cpp, get_ekf_soln_status):

    const_pos_mode = fake_pos || valid_fake_pos || vehicle_at_rest

`vehicle_at_rest` is the land detector's at_rest, fed in from control.cpp. It
says the airframe is not moving. It says NOTHING about aiding — and a preflight
runs, by definition, on an airframe that is not moving. Refusing on the raw flag
therefore refuses every preflight on this firmware, which is what it did:

    fix_type 4 (DGPS), 17 sats, vel_acc 0.48 m/s, h_acc 2.2 m
    const_pos_mode        true      (vehicle_at_rest — parked on the pad)
    velocity_horiz        true      \
    pos_horiz_abs         true       > GNSS demonstrably being fused
    pred_pos_horiz_abs    true      /

So the flag is read together with the aiding flags rather than alone. A real
fake_pos fallback drops velocity_horiz and pos_horiz_abs — that is the pad state
at the top of this file, and it still blocks. A parked, fully aided EKF does not.

So this module answers one question — can PX4 hold position right now? — from
telemetry the node already receives, and answers it BEFORE the operator is
asked to approve an arm that cannot succeed.

ONLY POSITIVE EVIDENCE BLOCKS
-----------------------------
A missing `/mavros/estimator_status` never blocks. Silence means a different
autopilot or a different plugin list, not a bad estimate, and grounding a
vehicle over a topic that was never wired is a worse failure than the one this
module exists to catch. Everything here refuses only on a state it has actually
observed.

Kept free of rclpy so the rules can be exercised without a vehicle
(`test_estimator.py`).
"""

from __future__ import annotations

#: The vehicle's EKF2_REQ_SACC [m/s], mirrored here so the number in our
#: messages is the number the operator will find in QGC.
#:
#: RAISED from PX4's 0.5 default to 1.0 on purpose. On our pad the GPS speed
#: accuracy oscillates around 0.4-0.55 m/s (see the measurements above), so at
#: 0.5 the ~10 s of CONTINUOUS passes EKF2 needs before it starts fusing GNSS
#: never accumulated and the vehicle sat in constant-position mode. 1.0 clears
#: that spread with margin. It buys the tolerance by accepting a looser velocity
#: estimate at fusion start, which is the trade we want for a low, slow landing
#: over a marker — it is NOT a number to carry into fast waypoint flight.
#:
#: THIS CONSTANT ALONE CHANGES NOTHING ON THE VEHICLE. It is our copy of the
#: threshold; the one PX4 enforces lives in EKF2_REQ_SACC and has to be set
#: there too, or the arm is still refused at 0.5 and our messages quote a limit
#: that is not the one being applied.
DEFAULT_SPEED_ACC_MAX = 1.0

#: Telemetry older than this is treated as absent rather than as fact.
DEFAULT_STALE_AFTER_S = 3.0

#: MAV_GPS_FIX_TYPE, for messages that a human has to act on.
_FIX_NAMES = {
    0: 'no GPS', 1: 'no fix', 2: '2D fix', 3: '3D fix', 4: 'DGPS',
    5: 'RTK float', 6: 'RTK fixed', 7: 'static', 8: 'PPP',
}

#: Below this there is no usable horizontal position from GNSS.
_MIN_FIX_TYPE = 3


class EstimatorHealth:
    """The EKF's aiding state, from `estimator_status` plus the raw GPS.

    Fed plain numbers rather than ROS messages, and asked two questions:
    `blocking_reason()` — something is provably wrong, do not offer the arm —
    and `warning()` — this will probably bite, say so but let it fly.
    """

    def __init__(self, speed_acc_max: float = DEFAULT_SPEED_ACC_MAX,
                 stale_after: float = DEFAULT_STALE_AFTER_S) -> None:
        self.speed_acc_max = float(speed_acc_max)
        self.stale_after = float(stale_after)

        self._t_status: float | None = None
        self.const_pos_mode = False
        self.velocity_horiz = False
        self.pos_horiz_abs = False
        self.gps_glitch = False

        self._t_gps: float | None = None
        self.fix_type = 0
        self.satellites = 0
        self.h_acc = float('nan')      # m
        self.vel_acc = float('nan')    # m/s

    # ------------------------------------------------------------------ input
    def on_status(self, t: float, *, const_pos_mode: bool,
                  velocity_horiz: bool, pos_horiz_abs: bool,
                  gps_glitch: bool = False) -> None:
        self._t_status = float(t)
        self.const_pos_mode = bool(const_pos_mode)
        self.velocity_horiz = bool(velocity_horiz)
        self.pos_horiz_abs = bool(pos_horiz_abs)
        self.gps_glitch = bool(gps_glitch)

    def on_gps(self, t: float, *, fix_type: int, satellites: int,
               h_acc_m: float, vel_acc_m_s: float) -> None:
        self._t_gps = float(t)
        self.fix_type = int(fix_type)
        self.satellites = int(satellites)
        self.h_acc = float(h_acc_m)
        self.vel_acc = float(vel_acc_m_s)

    # ---------------------------------------------------------------- queries
    def status_fresh(self, t: float) -> bool:
        return (self._t_status is not None
                and (t - self._t_status) <= self.stale_after)

    def gps_fresh(self, t: float) -> bool:
        return (self._t_gps is not None
                and (t - self._t_gps) <= self.stale_after)

    def gps_detail(self, t: float) -> str:
        """The numbers an operator needs to decide whether to move the vehicle."""
        if not self.gps_fresh(t):
            return 'no GPS telemetry (/mavros/gpsstatus/gps1/raw)'
        fix = _FIX_NAMES.get(self.fix_type, f'fix {self.fix_type}')
        return (f'GPS vel_acc {self.vel_acc:.2f} m/s '
                f'(EKF2_REQ_SACC {self.speed_acc_max:.2f}), '
                f'h_acc {self.h_acc:.2f} m, {self.satellites} sats, {fix}')

    def blocking_reason(self, t: float) -> str | None:
        """Why the vehicle provably cannot hold position, or None."""
        if self.status_fresh(t):
            # const_pos_mode ALONE IS NOT EVIDENCE. See ONE FLAG, THREE CAUSES
            # above: on PX4 v1.18 it is also raised for a vehicle merely sitting
            # still, so a preflight that refuses on it refuses every preflight.
            # The aiding flags are what separate a real fake_pos fallback from a
            # parked-but-perfectly-aided EKF, so they qualify it here.
            if self.const_pos_mode and not (self.velocity_horiz
                                            and self.pos_horiz_abs):
                return ('EKF is in CONSTANT-POSITION mode — it is NOT fusing '
                        'GNSS, so the local position is dead reckoning and PX4 '
                        'will refuse OFFBOARD with "GPS speed accuracy". '
                        + self.gps_detail(t))
            if not self.velocity_horiz:
                return ('EKF has no horizontal velocity estimate — no position '
                        'aiding source is being fused. ' + self.gps_detail(t))
            if not self.pos_horiz_abs:
                return ('EKF has no absolute horizontal position — GNSS is not '
                        'yet fused. ' + self.gps_detail(t))
        if self.gps_fresh(t) and self.fix_type < _MIN_FIX_TYPE:
            return (f'GPS has no 3D fix '
                    f'({_FIX_NAMES.get(self.fix_type, self.fix_type)}, '
                    f'{self.satellites} sats)')
        return None

    def warning(self, t: float) -> str | None:
        """Flying is possible, but this is the thing that will bite. Or None.

        Marginal speed accuracy is a warning rather than a block on purpose:
        once EKF2 HAS started fusing GNSS it keeps fusing across brief excursions
        past EKF2_REQ_SACC, so a vehicle in this state can be perfectly
        controllable. It is only fatal before fusion starts — and that case is
        already caught above as constant-position mode.
        """
        if self.gps_glitch:
            return 'EKF reports a GPS GLITCH — the position may jump'
        if self.gps_fresh(t) and self.vel_acc > self.speed_acc_max:
            return (f'GPS speed accuracy is marginal: {self.gps_detail(t)}. '
                    f'GNSS fusion is running now, but if it ever drops out it '
                    f'will not restart until vel_acc stays under '
                    f'{self.speed_acc_max:.2f} m/s for ~10 s')
        return None

    def summary(self, t: float) -> str:
        """One line for the PASSED log, so a good run records its own margin."""
        if not self.status_fresh(t):
            return 'no /mavros/estimator_status (not checked)'
        return f'EKF fusing GNSS — {self.gps_detail(t)}'
