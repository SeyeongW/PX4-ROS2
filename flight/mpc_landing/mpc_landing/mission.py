"""Gated mission sequence: every irreversible step waits for a human `approve`.

The shape of this state machine is lifted from `flight/offboard`'s
`drone_core_controller.cpp` — the same MAVROS handles (set_mode / arm /
takeoff / land), the same "ask, then confirm from telemetry rather than
believing the service reply" discipline. What is added is the part that matters
on a real vehicle: **the sequence does not advance on its own.**

    PRECHECK ──approve──► ARM ──approve──► TAKEOFF ──approve──► SEARCH
                                                                  │
                                                       marker seen│ (no gate)
                                                                  ▼
                                                               DESCEND
                                                                  │
                                                                  ▼
                                                              TOUCHDOWN

Only the last transition is automatic, and deliberately so: once the vehicle is
committed to a descent onto a marker it has actually seen, waiting on a human
adds risk rather than removing it. Everything before that — arming, leaving the
ground, and beginning to look for the marker at altitude — is a decision a
person makes.

`GateState` is kept free of ROS and MAVROS types on purpose, so the sequencing
rules can be exercised without a flight controller (see `test_mission.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    """Mission phases, in the order they occur."""

    PRECHECK = 'PRECHECK'      # running preflight checks
    READY_TO_ARM = 'READY_TO_ARM'   # checks passed, waiting for approval
    ARMING = 'ARMING'          # arm command sent, waiting for FCU confirmation
    READY_TO_TAKEOFF = 'READY_TO_TAKEOFF'   # armed, waiting for approval
    TAKEOFF = 'TAKEOFF'        # climbing to takeoff_alt
    READY_TO_SEARCH = 'READY_TO_SEARCH'     # at altitude, waiting for approval
    SEARCH = 'SEARCH'          # holding altitude, looking for the marker
    DESCEND = 'DESCEND'        # marker acquired, MPC flying the descent
    TOUCHDOWN = 'TOUCHDOWN'    # on the ground, disarming
    DONE = 'DONE'
    ABORT = 'ABORT'


#: Phases that exist only to wait for a human. Nothing else may advance them.
GATES = (Phase.READY_TO_ARM, Phase.READY_TO_TAKEOFF, Phase.READY_TO_SEARCH)

#: What `approve` does at each gate.
_GATE_RELEASES = {
    Phase.READY_TO_ARM: Phase.ARMING,
    Phase.READY_TO_TAKEOFF: Phase.TAKEOFF,
    Phase.READY_TO_SEARCH: Phase.SEARCH,
}

#: Human-readable prompt for each gate, so the operator is told exactly what
#: they are authorising rather than just "approve?".
GATE_PROMPTS = {
    Phase.READY_TO_ARM:
        'preflight checks PASSED — approve to ARM the vehicle',
    Phase.READY_TO_TAKEOFF:
        'vehicle is ARMED and props are live — approve to TAKE OFF',
    Phase.READY_TO_SEARCH:
        'holding at target altitude — approve to begin MARKER SEARCH',
}


@dataclass
class CheckResult:
    """One preflight check: a name, a verdict, and why."""

    name: str
    passed: bool
    detail: str = ''

    def __str__(self) -> str:
        mark = 'PASS' if self.passed else 'FAIL'
        return f'[{mark}] {self.name}' + (f' — {self.detail}' if self.detail else '')


@dataclass
class GateState:
    """The sequencing rules, with no ROS or MAVROS in sight.

    Every transition the mission can make goes through one of the methods here,
    so "what is allowed to happen next" lives in exactly one place and can be
    tested without a vehicle.
    """

    phase: Phase = Phase.PRECHECK
    abort_reason: str = ''
    history: list = field(default_factory=list)

    # ------------------------------------------------------------- transitions
    def _to(self, phase: Phase, why: str = '') -> None:
        if phase is self.phase:
            return
        self.history.append((self.phase, phase, why))
        self.phase = phase

    def checks_passed(self) -> None:
        """Preflight is green — stop and wait for a human."""
        if self.phase is Phase.PRECHECK:
            self._to(Phase.READY_TO_ARM, 'preflight checks passed')

    def approve(self) -> tuple[bool, str]:
        """Release the current gate. Returns (accepted, message).

        Refusing when no gate is pending is not pedantry: an approval that
        arrives early would otherwise be silently swallowed, and the operator
        would believe they had authorised a step they had not.
        """
        nxt = _GATE_RELEASES.get(self.phase)
        if nxt is None:
            return False, (f'nothing to approve — phase is {self.phase.value}, '
                           f'which is not a gate')
        prompt = GATE_PROMPTS[self.phase]
        self._to(nxt, 'operator approved')
        return True, f'approved: {prompt}'

    def armed_confirmed(self) -> None:
        if self.phase is Phase.ARMING:
            self._to(Phase.READY_TO_TAKEOFF, 'FCU reports armed')

    def altitude_reached(self) -> None:
        if self.phase is Phase.TAKEOFF:
            self._to(Phase.READY_TO_SEARCH, 'reached target altitude')

    def marker_acquired(self) -> None:
        """The one automatic transition — see the module docstring."""
        if self.phase is Phase.SEARCH:
            self._to(Phase.DESCEND, 'marker acquired')

    def touched_down(self) -> None:
        if self.phase is Phase.DESCEND:
            self._to(Phase.TOUCHDOWN, 'touchdown detected')

    def finished(self) -> None:
        """Reached a safe, disarmed end — from a normal touchdown OR from an
        abort that has since landed and disarmed, so both terminate cleanly in
        DONE instead of one of them looping on the land command forever."""
        if self.phase in (Phase.TOUCHDOWN, Phase.ABORT):
            self._to(Phase.DONE, 'disarmed')

    def abort(self, reason: str) -> None:
        """Always allowed, from anywhere, including from a gate."""
        if self.phase in (Phase.DONE, Phase.ABORT):
            return
        self.abort_reason = reason
        self._to(Phase.ABORT, reason)

    # ----------------------------------------------------------------- queries
    @property
    def waiting(self) -> bool:
        return self.phase in GATES

    @property
    def prompt(self) -> str:
        return GATE_PROMPTS.get(self.phase, '')

    @property
    def flying(self) -> bool:
        """Is the vehicle airborne under our command?"""
        return self.phase in (Phase.TAKEOFF, Phase.READY_TO_SEARCH,
                              Phase.SEARCH, Phase.DESCEND)

    @property
    def needs_setpoint_stream(self) -> bool:
        """Must a setpoint go out THIS tick to keep offboard control?

        Wider than `flying`, and the difference is PX4-specific and dangerous:
        PX4 will not ENTER offboard until setpoints have been streaming, and it
        drops straight back out if the stream lapses for ~0.5 s. So the stream
        has to start in ARMING — before the mode request — and must not pause at
        READY_TO_TAKEOFF, which is a gate that happens AFTER the vehicle is
        armed and in offboard.

        Getting this wrong on ArduPilot is invisible (GUIDED does not care);
        getting it wrong on PX4 means the mode request is rejected, or the
        vehicle falls out of offboard while waiting for a human.
        """
        return self.phase in (Phase.ARMING, Phase.READY_TO_TAKEOFF,
                              Phase.TAKEOFF, Phase.READY_TO_SEARCH,
                              Phase.SEARCH, Phase.DESCEND)
