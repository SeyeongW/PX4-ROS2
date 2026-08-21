"""The probe's verdict, without a radio.

Its whole value is telling two failures apart that look identical on a telemetry
screen: "the GPS has no fix yet" (wait) and "the position stream is switched
off" (change a parameter). Getting that backwards sends an operator to change
settings on a working vehicle, or to stand in a field waiting for a fix that was
already there.
"""

import collections

from trailer_link.radio_probe import GPS_MSG, POSITION_MSG, verdict


def _counts(**kw):
    return collections.Counter(kw)


def test_silence_is_reported_as_the_link_not_the_gps():
    head, action = verdict(_counts(), None, 0)
    assert 'NOTHING ON THE WIRE' in head
    assert 'baud' in action and 'one reader' in action


def test_a_position_on_the_wire_sends_the_search_downstream():
    """The radio half being fine is itself a finding — it moves the hunt."""
    head, action = verdict(_counts(**{POSITION_MSG: 50, GPS_MSG: 20}), 4, 14)
    assert 'POSITION IS ARRIVING' in head
    assert 'ROS_DOMAIN_ID' in action


def test_no_fix_is_not_a_misconfiguration():
    """The autopilot is RIGHT to withhold a position it does not have."""
    for fix in (0, 1, 2):
        head, action = verdict(_counts(**{GPS_MSG: 20, 'HEARTBEAT': 10}),
                               fix, 4)
        assert 'NO POSITION YET' in head
        assert 'Nothing is misconfigured' in action
        assert 'SR' not in action.replace('SRx', '')   # no parameter advice


def test_a_fix_with_no_position_message_is_the_stream_setting():
    """The case that started this: every reassuring number, no coordinate."""
    head, action = verdict(_counts(**{GPS_MSG: 20, 'HEARTBEAT': 10}), 3, 14)
    assert 'STREAM IS SWITCHED OFF' in head
    assert 'SR1_POSITION' in action


def test_the_two_gps_cases_never_give_the_same_advice():
    """The distinction the tool exists for, asserted as a distinction."""
    no_fix = verdict(_counts(**{GPS_MSG: 20}), 1, 5)
    fixed = verdict(_counts(**{GPS_MSG: 20}), 4, 14)
    assert no_fix[0] != fixed[0] and no_fix[1] != fixed[1]


def test_traffic_without_any_gps_message_blames_the_streams():
    head, action = verdict(_counts(HEARTBEAT=10, SYS_STATUS=10), None, 0)
    assert 'NO GPS MESSAGES' in head
    assert 'SRx_EXT_STAT' in action
