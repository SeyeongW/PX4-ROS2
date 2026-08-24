"""The EKF-aiding rules, exercised without a vehicle.

`EstimatorHealth` is ROS-free precisely so this can run anywhere. What is
asserted here is the property the module exists for: **a published pose is not
evidence that the vehicle can hold position**, and the one state that proves it
cannot — constant-position mode — must stop the mission before the operator is
asked to approve an arm PX4 is going to refuse.

The numbers in `test_the_state_that_actually_grounded_the_vehicle` are the ones
measured on the pad, so a regression here fails against the real event rather
than an invented one.
"""

from mpc_landing.estimator import DEFAULT_SPEED_ACC_MAX, EstimatorHealth


def _aided(h, t=0.0, **kw):
    """Feed a healthy, GNSS-aided estimator."""
    flags = dict(const_pos_mode=False, velocity_horiz=True, pos_horiz_abs=True)
    flags.update(kw)
    h.on_status(t, **flags)
    h.on_gps(t, fix_type=4, satellites=12, h_acc_m=0.8, vel_acc_m_s=0.15)


def test_a_healthy_estimator_blocks_nothing():
    h = EstimatorHealth()
    _aided(h)
    assert h.blocking_reason(0.0) is None
    assert h.warning(0.0) is None
    assert 'fusing GNSS' in h.summary(0.0)


def test_the_state_that_actually_grounded_the_vehicle():
    """Pad measurements: GNSS unfused, pose still publishing, PX4 refusing.

    A real fake_pos fallback drops the aiding flags with the mode — that pairing
    is what makes it distinguishable from the parked vehicle below.
    """
    h = EstimatorHealth(speed_acc_max=0.5)          # the vehicle's EKF2_REQ_SACC
    h.on_status(0.0, const_pos_mode=True, velocity_horiz=False,
                pos_horiz_abs=False)
    h.on_gps(0.0, fix_type=4, satellites=12, h_acc_m=1.95, vel_acc_m_s=0.506)

    why = h.blocking_reason(0.0)
    assert why is not None
    # The message has to name the mode AND quote the numbers, because the
    # operator's next move is to walk the vehicle somewhere with more sky.
    assert 'CONSTANT-POSITION' in why
    assert '0.51 m/s' in why and '0.50' in why
    assert '12 sats' in why


def test_silence_never_blocks():
    """No estimator_status is a different autopilot, not a bad estimate."""
    h = EstimatorHealth()
    assert h.blocking_reason(0.0) is None
    assert h.warning(0.0) is None
    assert 'not checked' in h.summary(0.0)


def test_a_parked_but_aided_vehicle_is_not_grounded():
    """PX4 v1.18 raises const_pos_mode for vehicle_at_rest too.

    Measured on the pad with the airframe simply standing still: DGPS fix, 17
    sats, GNSS demonstrably fused. Blocking on the raw flag here would refuse
    every preflight on this firmware, because every preflight is run parked.
    """
    h = EstimatorHealth()
    h.on_status(0.0, const_pos_mode=True, velocity_horiz=True,
                pos_horiz_abs=True)
    h.on_gps(0.0, fix_type=4, satellites=17, h_acc_m=2.215, vel_acc_m_s=0.484)

    assert h.blocking_reason(0.0) is None
    assert h.warning(0.0) is None


def test_stale_telemetry_is_absent_rather_than_true():
    h = EstimatorHealth(stale_after=3.0)
    h.on_status(0.0, const_pos_mode=True, velocity_horiz=False,
                pos_horiz_abs=False)
    assert h.blocking_reason(1.0) is not None
    assert h.blocking_reason(10.0) is None


def test_no_aiding_source_blocks_even_without_const_pos_mode():
    h = EstimatorHealth()
    h.on_status(0.0, const_pos_mode=False, velocity_horiz=False,
                pos_horiz_abs=False)
    assert 'no horizontal velocity' in h.blocking_reason(0.0)


def test_no_3d_fix_blocks():
    h = EstimatorHealth()
    h.on_gps(0.0, fix_type=1, satellites=3, h_acc_m=99.0, vel_acc_m_s=9.0)
    assert 'no 3D fix' in h.blocking_reason(0.0)


def test_marginal_speed_accuracy_warns_but_flies():
    """Once fusion HAS started it survives excursions past EKF2_REQ_SACC, so
    this is a warning. The fatal case is const_pos_mode, tested above."""
    h = EstimatorHealth(speed_acc_max=DEFAULT_SPEED_ACC_MAX)
    _aided(h)
    h.on_gps(0.0, fix_type=4, satellites=11,
             h_acc_m=1.2, vel_acc_m_s=DEFAULT_SPEED_ACC_MAX + 0.12)
    assert h.blocking_reason(0.0) is None
    assert 'marginal' in h.warning(0.0)


def test_the_pad_spread_is_inside_the_raised_default():
    """The 0.37-0.55 m/s the pad actually produced is why the default moved.

    At PX4's 0.5 it straddled the limit and EKF2 never started fusing; the
    raised default has to clear that whole spread without a warning, or the
    change bought nothing.
    """
    h = EstimatorHealth()
    _aided(h)
    for vel_acc in (0.37, 0.45, 0.506, 0.55):
        h.on_gps(0.0, fix_type=4, satellites=12, h_acc_m=1.95,
                 vel_acc_m_s=vel_acc)
        assert h.blocking_reason(0.0) is None
        assert h.warning(0.0) is None, vel_acc


def test_a_gps_glitch_warns():
    h = EstimatorHealth()
    _aided(h, gps_glitch=True)
    assert 'GLITCH' in h.warning(0.0)
