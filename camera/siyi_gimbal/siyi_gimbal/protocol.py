"""SIYI gimbal wire protocol — framing, CRC and the two commands this needs.

Kept free of ROS and sockets so the framing can be checked against known
vectors without a gimbal on the desk; see `test_protocol.py`.

Frame layout (little-endian throughout):

    55 66 | rack | len | seq | cmd | payload… | crc16
     ^      ^      ^     ^     ^                ^
     |      |      |     |     |                CRC-16/CCITT over everything before it
     |      |      |     |     command id
     |      |      |     rolling sequence number
     |      |      payload length, NOT total length
     |      1 for a command that wants no ack
     header

Two conventions here will bite anyone reading the SIYI datasheet alongside this:

* **Angles are deci-degrees**, so -90 deg goes on the wire as -900.
* **Yaw is negated.** SIYI counts yaw the opposite way round from the ENU/MAVLink
  convention, and both call sites in the reference implementation send `-yaw`.
  `set_angle()` takes yaw in the normal convention and flips it here, once, so
  that nothing above this module has to remember.
"""

from __future__ import annotations

import struct

from . import siyi_commands as cmds

HEADER1 = 0x55
HEADER2 = 0x66

# Command ids come from the command TABLE, not from constants repeated here.
# `siyi_commands` is the single place a command's id, payload layout and meaning
# are written down; duplicating even one of them is how the two drift apart.
SET_ANGLE = cmds.SET_ANGLE.cid
CENTER = cmds.CENTER.cid
ACQUIRE_GIMBAL_ATTITUDE = cmds.ACQUIRE_GIMBAL_ATTITUDE.cid
GIMBAL_ROTATION = cmds.GIMBAL_ROTATION.cid
ABSOLUTE_ZOOM = cmds.ABSOLUTE_ZOOM.cid
ACQUIRE_FIRMWARE_VERSION = cmds.ACQUIRE_FIRMWARE_VERSION.cid

#: A8 mini mechanical limits, degrees. Commanding past these is silently
#: clamped by the gimbal, which looks identical to a command that never
#: arrived — so it is clamped here instead, where it can be logged.
PITCH_MIN_DEG, PITCH_MAX_DEG = -90.0, 25.0
YAW_MIN_DEG, YAW_MAX_DEG = -135.0, 135.0

#: Straight down. The whole point of the arming behaviour.
NADIR_PITCH_DEG = -90.0


def crc16(data: bytes, initial: int = 0) -> int:
    """CRC-16/CCITT, poly 0x1021, no reflection, no output xor.

    Note the initial value is 0, not the 0xFFFF the CCITT name usually implies —
    that is what the reference implementation actually uses, and a gimbal
    rejects anything else. `test_protocol.py` pins the standard check vector.
    """
    crc = initial
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 \
                else (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def encode(command_id: int, payload: bytes = b'', seq: int = 0) -> bytes:
    """Wrap a payload in a SIYI frame."""
    head = struct.pack('<BBBHHB', HEADER1, HEADER2, 1, len(payload), seq,
                       command_id)
    body = head + payload
    return body + struct.pack('<H', crc16(body))


def set_angle(yaw_deg: float, pitch_deg: float, seq: int = 0) -> bytes:
    """Absolute-angle command, clamped to the A8 mini's travel.

    `yaw_deg` is in the normal convention (positive = right); the SIYI sign flip
    happens here.
    """
    yaw = max(YAW_MIN_DEG, min(YAW_MAX_DEG, float(yaw_deg)))
    pitch = max(PITCH_MIN_DEG, min(PITCH_MAX_DEG, float(pitch_deg)))
    payload = struct.pack('<hh', int(round(-yaw * 10)), int(round(pitch * 10)))
    return encode(SET_ANGLE, payload, seq)


def look_down(seq: int = 0) -> bytes:
    """Point the camera straight down, level in yaw."""
    return set_angle(0.0, NADIR_PITCH_DEG, seq)


def request_attitude(seq: int = 0) -> bytes:
    return encode(ACQUIRE_GIMBAL_ATTITUDE, b'', seq)


def clamped(yaw_deg: float, pitch_deg: float) -> tuple[bool, str]:
    """Was a requested angle inside the gimbal's travel? For logging."""
    out = []
    if not YAW_MIN_DEG <= yaw_deg <= YAW_MAX_DEG:
        out.append(f'yaw {yaw_deg:.1f} -> [{YAW_MIN_DEG}, {YAW_MAX_DEG}]')
    if not PITCH_MIN_DEG <= pitch_deg <= PITCH_MAX_DEG:
        out.append(f'pitch {pitch_deg:.1f} -> [{PITCH_MIN_DEG}, {PITCH_MAX_DEG}]')
    return (not out), '; '.join(out)


def decode(packet: bytes):
    """Parse one frame. Returns (command_id, payload) or None if it is not valid.

    Returning None rather than raising on a bad CRC is deliberate: this reads a
    UDP socket that also carries replies to commands we never sent, and a
    partial or foreign datagram is an ordinary event, not an error.
    """
    if len(packet) < 10 or packet[0] != HEADER1 or packet[1] != HEADER2:
        return None
    _h1, _h2, _rack, plen, _seq, cmd = struct.unpack('<BBBHHB', packet[:8])
    if len(packet) < plen + 10:
        return None
    frame = packet[:plen + 10]
    got, = struct.unpack('<H', frame[-2:])
    if got != crc16(frame[:-2]):
        return None
    return cmd, frame[8:-2]


def parse_attitude(payload: bytes):
    """ACQUIRE_GIMBAL_ATTITUDE reply -> (roll, pitch, yaw) in degrees.

    The gimbal reports (z, y, x) — yaw, pitch, roll — each in deci-degrees, and
    yaw comes back with the same flipped sign the command used, so it is
    un-flipped here to match `set_angle`'s convention.
    """
    if len(payload) < 12:
        return None
    z, y, x, _sz, _sy, _sx = struct.unpack('<hhhhhh', payload[:12])
    return x * 0.1, y * 0.1, -z * 0.1
