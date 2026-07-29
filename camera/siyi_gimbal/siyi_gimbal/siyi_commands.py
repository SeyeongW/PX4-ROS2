"""SIYI command set — the "header file" for the gimbal protocol.

This module is a REFERENCE, not logic. It declares every command id, its payload
layout, and what it actually does, so that calling code reads as intent
(`siyi.set_angle(...)`) instead of as magic numbers, and so that adding a
command is a one-line table entry rather than an archaeology session.

Commands marked **VERIFIED** are exercised by `test_protocol.py`; the rest are
declared from the SIYI command reference but have not been put on a real gimbal
by this repo, so treat their payloads as documentation rather than as tested.

---------------------------------------------------------------------------
FRAME (little-endian throughout, identical for every command)

    55 66 | rack | len | seq | cmd | payload… | crc16
     ^      ^      ^     ^     ^                ^
     |      |      |     |     |                CRC-16/CCITT, init 0, over all
     |      |      |     |     |                bytes before it
     |      |      |     |     command id, from the table below
     |      |      |     rolling sequence number
     |      |      PAYLOAD length — not the frame length
     |      1 = command, 0 = reply needs no ack
     header

UNITS AND SIGNS — the two things that silently break everything:
  * angles on the wire are DECI-DEGREES  (-90 deg -> -900)
  * SIYI counts YAW the opposite way from ENU/MAVLink, so every yaw is negated
    on the way out and un-negated on the way in. `protocol.set_angle` does this
    once so nothing above it has to remember.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One protocol command: what to send, and what it means."""

    cid: int                 #: command id byte
    name: str
    request_fmt: str         #: struct format of the payload we send ('' = none)
    reply_fmt: str           #: struct format of the payload it sends back
    doc: str
    verified: bool = False   #: exercised by this repo's tests

    def __str__(self) -> str:
        mark = 'VERIFIED' if self.verified else 'declared'
        return f'0x{self.cid:02X} {self.name} [{mark}] — {self.doc}'


# ===========================================================================
# ATTITUDE — pan / tilt / roll. What this repo actually flies.
# ===========================================================================

SET_ANGLE = Command(
    0x0E, 'SET_ANGLE', '<hh', '',
    'ABSOLUTE angle command: (yaw, pitch) in deci-degrees. Yaw is SIGN-FLIPPED '
    'relative to the normal convention. This is the one to use for "look at a '
    'fixed angle" — e.g. nadir is (0, -900). Roll is not commandable; the '
    'gimbal levels it itself.',
    verified=True)

GIMBAL_ROTATION = Command(
    0x07, 'GIMBAL_ROTATION', '<bb', '',
    'RATE command: (yaw_rate, pitch_rate) as signed percentages of the maximum '
    'slew (-100..100), 0 = stop. Use for joystick-style continuous motion, NOT '
    'for holding an angle — it keeps moving until told otherwise, so a dropped '
    'stop packet leaves the gimbal turning.')

CENTER = Command(
    0x08, 'CENTER', '<B', '',
    'Recentre: payload 1 = do it. Returns the gimbal to its neutral forward '
    'pose. Note this shares an id with RESET_ATTITUDE in the vendor docs.',
    verified=True)

ACQUIRE_GIMBAL_ATTITUDE = Command(
    0x0D, 'ACQUIRE_GIMBAL_ATTITUDE', '', '<hhhhhh',
    'Poll the current attitude. Reply is (yaw, pitch, roll, yaw_rate, '
    'pitch_rate, roll_rate) in deci-degrees and deci-deg/s, with yaw again '
    'sign-flipped. The only way to know a command was obeyed.',
    verified=True)

ACQUIRE_GIMBAL_CONFIG_INFO = Command(
    0x0A, 'ACQUIRE_GIMBAL_CONFIG_INFO', '', '<BBBBBBB',
    'Read the current mode configuration (recording state, motion mode, '
    'mounting orientation, …).')

SET_WEAK_CONTROL = Command(
    0x71, 'SET_WEAK_CONTROL', '<B', '<B',
    'Select the attitude-control strength/mode. Changing this alters how the '
    'gimbal responds to SET_ANGLE, so leave it alone unless you know why.')

READ_CONTROL_MODE = Command(
    0x27, 'READ_CONTROL_MODE', '', '<B',
    'Which control mode the gimbal is in (lock / follow / FPV).')

# ===========================================================================
# CAMERA — zoom and focus
# ===========================================================================

ABSOLUTE_ZOOM = Command(
    0x0F, 'ABSOLUTE_ZOOM', '<BB', '<B',
    'Zoom to an absolute factor: (integer_part, fractional_tenths), so 4.5x is '
    '(4, 5). The A8 mini is digital-zoom only up to 6x — zooming in narrows the '
    'FOV, which directly shrinks the ground area a landing marker can be found '
    'in, so the landing path deliberately never calls this.')

MANUAL_ZOOM_AND_AUTO_FOCUS = Command(
    0x05, 'MANUAL_ZOOM_AND_AUTO_FOCUS', '<b', '<H',
    'Continuous zoom: 1 = in, -1 = out, 0 = stop. Autofocus runs afterwards. '
    'Reply is the resulting zoom level x10.')

AUTO_FOCUS = Command(
    0x04, 'AUTO_FOCUS', '<B', '<B',
    'Trigger one autofocus cycle (payload 1).')

MANUAL_FOCUS = Command(
    0x06, 'MANUAL_FOCUS', '<b', '<B',
    'Continuous focus: 1 = far, -1 = near, 0 = stop.')

# ===========================================================================
# MEDIA
# ===========================================================================

PHOTO = Command(
    0x0C, 'PHOTO', '<B', '',
    'Media command, selected by the payload byte: 0 = take photo, '
    '2 = start/stop recording, 3 = motion mode, 4 = mounting orientation. '
    'One id doing five jobs is the vendor\'s design, not a mistake here.')

FUNCTION_FEEDBACK_INFO = Command(
    0x0B, 'FUNCTION_FEEDBACK_INFO', '', '<B',
    'Result of the last PHOTO-family action: 0 = success, non-zero = the '
    'failure code (SD card missing, already recording, …).')

SET_IMAGE_TYPE = Command(
    0x11, 'SET_IMAGE_TYPE', '<B', '',
    'Which video stream the RTSP feed carries (RGB / thermal / split view). '
    'Affects the picture the RTSP bridge receives, NOT the control link.')

# ===========================================================================
# RANGEFINDER AND THERMAL — present on ZT30-class units, not on the A8 mini
# ===========================================================================

READ_RANGEFINDER = Command(
    0x15, 'READ_RANGEFINDER', '', '<H',
    'Laser rangefinder distance in decimetres. The A8 mini has no rangefinder; '
    'it answers with zero.')

SET_THERMAL_PALETTE = Command(
    0x1B, 'SET_THERMAL_PALETTE', '<B', '',
    'Thermal false-colour palette (0 = white hot). Thermal units only.')

READ_TEMP_FULL_SCREEN = Command(
    0x14, 'READ_TEMP_FULL_SCREEN', '<B', '<HHHHHH',
    'Frame-wide min/max temperature and where they are. Thermal units only.')

GET_THERMAL_MODE = Command(0x33, 'GET_THERMAL_MODE', '', '<B',
                           'Read the thermal gain mode. Thermal units only.')
SET_THERMAL_MODE = Command(0x34, 'SET_THERMAL_MODE', '<B', '<B',
                           'Set the thermal gain mode. Thermal units only.')

# ===========================================================================
# TELEMETRY AND SYSTEM
# ===========================================================================

ACQUIRE_FIRMWARE_VERSION = Command(
    0x01, 'ACQUIRE_FIRMWARE_VERSION', '', '<BBBBBBBBBBBB',
    'Camera, gimbal and zoom firmware versions. Cheapest way to prove the link '
    'works at all — if this answers, the IP, port and CRC are all correct.')

HARDWARE_ID = Command(
    0x02, 'HARDWARE_ID', '', '<BBBBBB',
    'Model identifier. Distinguishes an A8 mini from a ZT30, which matters '
    'because half the commands above only exist on one of them.')

REQUEST_CONTINUOUS_DATA = Command(
    0x25, 'REQUEST_CONTINUOUS_DATA', '<BB', '',
    'Ask the gimbal to STREAM a data type at a fixed rate instead of being '
    'polled: (data_type, frequency_code). Cheaper than polling '
    'ACQUIRE_GIMBAL_ATTITUDE, but it keeps sending after you stop listening.')

ATTITUDE_EXTERNAL = Command(
    0x22, 'ATTITUDE_EXTERNAL', '<Iffffff', '',
    'Feed the gimbal the AIRCRAFT attitude (time_ms, then quaternion and rates) '
    'so it can compensate for airframe motion itself. Only useful when the '
    'gimbal is not already getting this from the flight controller.')

VELOCITY_EXTERNAL = Command(
    0x26, 'VELOCITY_EXTERNAL', '<Ifff', '',
    'Feed the gimbal the aircraft velocity. NOTE: this id collides with '
    'READ_ENCODERS in the vendor documentation — do not use both.')

READ_ENCODERS = Command(
    0x26, 'READ_ENCODERS', '', '<hhh',
    'Raw joint encoder angles. Same id as VELOCITY_EXTERNAL; which one the '
    'gimbal means depends on direction and firmware.')

READ_VOLTAGES = Command(0x2A, 'READ_VOLTAGES', '', '<HH',
                        'Internal rail voltages, for diagnosing brownouts.')
READ_THRESHOLDS = Command(0x28, 'READ_THRESHOLDS', '', '<hhh',
                          'Read the configured motion thresholds.')
SET_THRESHOLDS = Command(0x29, 'SET_THRESHOLDS', '<hhh', '',
                         'Set the motion thresholds. Changes how the gimbal '
                         'decides it is being commanded vs drifting.')


#: Every command declared above, by id. Note two ids appear twice by the
#: vendor's own design (0x08 CENTER/RESET_ATTITUDE, 0x26
#: VELOCITY_EXTERNAL/READ_ENCODERS), so this maps to the meaning this repo uses.
ALL = {c.cid: c for c in (
    SET_ANGLE, GIMBAL_ROTATION, CENTER, ACQUIRE_GIMBAL_ATTITUDE,
    ACQUIRE_GIMBAL_CONFIG_INFO, SET_WEAK_CONTROL, READ_CONTROL_MODE,
    ABSOLUTE_ZOOM, MANUAL_ZOOM_AND_AUTO_FOCUS, AUTO_FOCUS, MANUAL_FOCUS,
    PHOTO, FUNCTION_FEEDBACK_INFO, SET_IMAGE_TYPE,
    READ_RANGEFINDER, SET_THERMAL_PALETTE, READ_TEMP_FULL_SCREEN,
    GET_THERMAL_MODE, SET_THERMAL_MODE,
    ACQUIRE_FIRMWARE_VERSION, HARDWARE_ID, REQUEST_CONTINUOUS_DATA,
    ATTITUDE_EXTERNAL, VELOCITY_EXTERNAL, READ_VOLTAGES,
    READ_THRESHOLDS, SET_THRESHOLDS,
)}


def describe(cid: int) -> str:
    """Human-readable name for a command id, for logs and debugging."""
    cmd = ALL.get(cid)
    return str(cmd) if cmd else f'0x{cid:02X} <unknown>'


if __name__ == '__main__':      # `python3 -m siyi_gimbal.siyi_commands`
    for cid in sorted(ALL):
        print(ALL[cid])
