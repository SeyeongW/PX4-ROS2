"""The gating rules, exercised without a vehicle.

`GateState` is ROS-free precisely so this can run anywhere. What is asserted
here is the safety property the whole node exists for: **nothing between
preflight and the descent advances without an explicit approval.**

That property is what `auto_after_arm` narrows, and the tests at the foot of
this file pin down exactly how far: the operator may delegate the two gates that
pause a flight already under way, never the ARM itself, and `abort` continues to
reach every phase either way.
"""

from mpc_landing.mission import GATES, GateState, Phase


def test_sequence_needs_an_approval_at_every_gate():
    g = GateState()
    assert g.phase is Phase.PRECHECK

    g.checks_passed()
    assert g.phase is Phase.READY_TO_ARM and g.waiting

    ok, _ = g.approve()
    assert ok and g.phase is Phase.ARMING

    g.armed_confirmed()
    assert g.phase is Phase.READY_TO_TAKEOFF and g.waiting

    assert g.approve()[0] and g.phase is Phase.TAKEOFF

    g.altitude_reached()
    assert g.phase is Phase.READY_TO_SEARCH and g.waiting

    assert g.approve()[0] and g.phase is Phase.SEARCH

    # ...and only here does it move on its own.
    g.marker_acquired()
    assert g.phase is Phase.DESCEND and not g.waiting

    g.touched_down()
    g.finished()
    assert g.phase is Phase.DONE


def test_a_gate_never_releases_itself():
    """The point of the whole exercise: no non-approve call may cross a gate."""
    for gate, nudges in (
        (Phase.READY_TO_ARM, ('armed_confirmed', 'altitude_reached',
                              'marker_acquired', 'checks_passed')),
        (Phase.READY_TO_TAKEOFF, ('altitude_reached', 'marker_acquired',
                                  'armed_confirmed')),
        (Phase.READY_TO_SEARCH, ('marker_acquired', 'armed_confirmed',
                                 'altitude_reached')),
    ):
        g = GateState(phase=gate)
        for name in nudges:
            getattr(g, name)()
            assert g.phase is gate, f'{name}() escaped {gate.value}'


def test_approve_is_refused_when_no_gate_is_pending():
    """An early approval must be rejected, not banked.

    Silently swallowing it would leave the operator believing they had
    authorised a step they had not.
    """
    for phase in (Phase.PRECHECK, Phase.TAKEOFF, Phase.SEARCH, Phase.DESCEND):
        ok, msg = GateState(phase=phase).approve()
        assert not ok
        assert 'not a gate' in msg


def test_abort_works_from_every_phase_including_gates():
    for phase in Phase:
        if phase in (Phase.DONE, Phase.ABORT):
            continue
        g = GateState(phase=phase)
        g.abort('test')
        assert g.phase is Phase.ABORT, f'could not abort from {phase.value}'
        assert g.abort_reason == 'test'


def test_marker_acquisition_only_counts_from_search():
    """The automatic transition must not fire before the operator opened SEARCH."""
    for phase in (Phase.PRECHECK, Phase.READY_TO_ARM, Phase.TAKEOFF,
                  Phase.READY_TO_SEARCH):
        g = GateState(phase=phase)
        g.marker_acquired()
        assert g.phase is not Phase.DESCEND


def test_every_gate_has_a_prompt():
    for gate in GATES:
        assert GateState(phase=gate).prompt, f'{gate.value} has no prompt'


# --------------------------------------------------------------------------
# PX4 keeps offboard only while setpoints keep arriving. These pin the phases
# where a lapse would drop the vehicle out of offboard mid-mission.
# --------------------------------------------------------------------------

def test_setpoint_stream_covers_every_armed_phase():
    """A gap in any of these is a PX4 offboard dropout, not a cosmetic issue."""
    must_stream = (Phase.ARMING, Phase.READY_TO_TAKEOFF, Phase.TAKEOFF,
                   Phase.READY_TO_SEARCH, Phase.SEARCH, Phase.DESCEND)
    for phase in must_stream:
        assert GateState(phase=phase).needs_setpoint_stream, \
            f'{phase.value} would starve the offboard stream'


def test_no_stream_before_arming_or_after_landing():
    """Streaming setpoints at a disarmed vehicle on the pad is not harmless —
    it is what lets a stray approval put it straight into offboard."""
    for phase in (Phase.PRECHECK, Phase.READY_TO_ARM, Phase.TOUCHDOWN,
                  Phase.DONE, Phase.ABORT):
        assert not GateState(phase=phase).needs_setpoint_stream, \
            f'{phase.value} should not be streaming setpoints'


def test_ready_to_takeoff_streams_even_though_it_is_not_flying():
    """The gate that catches people out: it is a WAIT, but the vehicle is
    already armed and in offboard, so the stream must not pause for it."""
    g = GateState(phase=Phase.READY_TO_TAKEOFF)
    assert g.waiting and not g.flying
    assert g.needs_setpoint_stream


# ------------------------------------------------- one approval per flight
# `auto_after_arm` exists so an armed airframe is not left on the pad with live
# props waiting for a keystroke. What must survive it: the ARM stays a human
# decision, and abort still works from everywhere.
def _armed_and_flying(g):
    g.checks_passed()
    g.approve()
    g.armed_confirmed()
    return g


def test_auto_after_arm_still_stops_at_the_arm():
    """The one gate that may never be delegated."""
    g = GateState(auto_after_arm=True)
    g.checks_passed()
    assert g.phase is Phase.READY_TO_ARM and g.waiting


def test_auto_after_arm_runs_the_rest_without_asking():
    g = GateState(auto_after_arm=True)
    g.checks_passed()
    assert g.approve()[0]
    g.armed_confirmed()
    assert g.phase is Phase.TAKEOFF and not g.waiting
    g.altitude_reached()
    assert g.phase is Phase.SEARCH and not g.waiting


def test_the_delegated_gates_are_still_entered_and_recorded():
    """A skipped step a reader has to notice is missing is a worse log."""
    g = _armed_and_flying(GateState(auto_after_arm=True))
    seen = [frm for frm, _to, _why in g.history]
    assert Phase.READY_TO_TAKEOFF in seen
    assert any(why == 'auto-released after arm' for _f, _t, why in g.history)


def test_without_the_flag_nothing_changes():
    g = _armed_and_flying(GateState())
    assert g.phase is Phase.READY_TO_TAKEOFF and g.waiting


def test_abort_still_reaches_every_phase_when_gates_are_delegated():
    """The control that actually matters once it is airborne."""
    for step in range(3):
        g = GateState(auto_after_arm=True)
        g.checks_passed()
        g.approve()
        if step >= 1:
            g.armed_confirmed()
        if step >= 2:
            g.altitude_reached()
        g.abort('operator aborted')
        assert g.phase is Phase.ABORT


def test_the_arm_gate_cannot_be_made_auto_releasable():
    """Structural, not a convention — see mission.AUTO_RELEASABLE."""
    from mpc_landing.mission import AUTO_RELEASABLE
    assert Phase.READY_TO_ARM not in AUTO_RELEASABLE
    assert set(AUTO_RELEASABLE) < set(GATES)
