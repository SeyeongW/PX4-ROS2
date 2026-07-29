"""SIYI framing, checked without a gimbal.

A wire protocol is exactly the kind of thing that looks right and is wrong by a
sign or a scale factor, and on hardware that failure looks like "the gimbal
ignored me". So the framing is pinned here against the standard CRC vector and
against the two conventions most likely to be got backwards: deci-degrees and
the flipped yaw sign.
"""

import struct

from siyi_gimbal import protocol as p


def test_crc_matches_the_standard_check_vectors():
    """Pin BOTH initial values, because the wrong one is easy to talk yourself into.

    The CCITT name is usually quoted with init 0xFFFF (check value 0x29B1), but
    the SIYI framing used here requires **init 0**. At init 0 the check value of
    '123456789' is 0x31C3 (the XMODEM variant).

    Asserting the 0xFFFF vector too proves the polynomial and bit order are
    genuine CCITT, so a future edit cannot quietly swap the algorithm and still
    pass on the one value we happen to use.
    """
    assert p.crc16(b'123456789') == 0x31C3            # init 0 — what we send
    assert p.crc16(b'123456789', 0xFFFF) == 0x29B1    # proves it is real CCITT


def test_frame_layout_and_self_consistency():
    pkt = p.encode(p.SET_ANGLE, struct.pack('<hh', 0, -900), seq=7)
    assert pkt[0] == 0x55 and pkt[1] == 0x66
    h1, h2, rack, plen, seq, cmd = struct.unpack('<BBBHHB', pkt[:8])
    assert (rack, plen, seq, cmd) == (1, 4, 7, p.SET_ANGLE)
    # length field is the PAYLOAD, not the frame
    assert len(pkt) == plen + 10
    got, = struct.unpack('<H', pkt[-2:])
    assert got == p.crc16(pkt[:-2])


def test_angles_go_out_as_deci_degrees_with_yaw_flipped():
    yaw_raw, pitch_raw = struct.unpack('<hh', p.set_angle(30.0, -45.0)[8:-2])
    assert pitch_raw == -450, 'pitch must be deci-degrees, unflipped'
    assert yaw_raw == -300, 'yaw must be deci-degrees AND sign-flipped for SIYI'


def test_look_down_is_nadir():
    yaw_raw, pitch_raw = struct.unpack('<hh', p.look_down()[8:-2])
    assert (yaw_raw, pitch_raw) == (0, -900)


def test_commands_are_clamped_to_the_a8_minis_travel():
    # -120 deg of pitch does not exist on this gimbal; it must not wrap or
    # overflow the int16, it must clamp.
    _, pitch_raw = struct.unpack('<hh', p.set_angle(0.0, -120.0)[8:-2])
    assert pitch_raw == -900
    yaw_raw, _ = struct.unpack('<hh', p.set_angle(200.0, 0.0)[8:-2])
    assert yaw_raw == -1350
    ok, why = p.clamped(200.0, -120.0)
    assert not ok and 'yaw' in why and 'pitch' in why


def test_decode_round_trips_and_rejects_corruption():
    pkt = p.look_down(seq=3)
    cmd, payload = p.decode(pkt)
    assert cmd == p.SET_ANGLE and len(payload) == 4

    bad = bytearray(pkt)
    bad[-1] ^= 0xFF
    assert p.decode(bytes(bad)) is None, 'a bad CRC must not decode'
    assert p.decode(b'') is None
    assert p.decode(b'\x00\x01\x02') is None
    assert p.decode(b'\x55\x66' + b'\x00' * 20) is None   # header ok, CRC not


def test_attitude_reply_unflips_yaw():
    # gimbal reports (z=yaw, y=pitch, x=roll) in deci-degrees
    payload = struct.pack('<hhhhhh', 300, -900, 12, 0, 0, 0)
    roll, pitch, yaw = p.parse_attitude(payload)
    assert (round(roll, 1), round(pitch, 1)) == (1.2, -90.0)
    assert round(yaw, 1) == -30.0, 'reported yaw must be un-flipped to match set_angle'


def test_parse_attitude_tolerates_a_short_payload():
    assert p.parse_attitude(b'\x00\x00') is None
