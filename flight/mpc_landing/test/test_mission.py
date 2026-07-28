"""The gating rules, exercised without a vehicle.

`GateState` is ROS-free precisely so this can run anywhere. What is asserted
here is the safety property the whole node exists for: **nothing between
preflight and the descent advances without an explicit approval.**
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
