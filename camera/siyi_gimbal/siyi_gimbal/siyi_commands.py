"""SIYI command set — the "header file" for the gimbal protocol.

This module is a REFERENCE, not logic. It declares every command id, its payload
layout, and what it actually does, so that calling code reads as intent
(`siyi.set_angle(...)`) instead of as magic numbers, and so that adding a
command is a one-line table entry rather than an archaeology session.

Transcribed from MAVProxy's SIYI module and cross-checked against ArduPilot's
`AP_Mount_Siyi`. Commands marked **VERIFIED** are exercised by
`test_protocol.py`; the rest are declared from the reference implementations but
have not been put on a real gimbal by this repo, so treat their payloads as
documentation rather than as tested.

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


# ===========================================================================
# PAYLOAD VALUE TABLES
#
# The command ids above say WHICH message; these say what goes INSIDE it.
# Every magic number the protocol uses is named here — sub-commands, mode
# enums, result codes — so that no call site anywhere contains a bare integer.
# Values are cross-checked against ArduPilot's `AP_Mount_Siyi.h` enums and
# MAVProxy's call sites.
# ===========================================================================


class PhotoFunction:
    """Payload byte for PHOTO (0x0C).

    One command id doing six unrelated jobs is the vendor's design. Note that
    LOCK / FOLLOW / FPV are the gimbal's MOTION MODE and have nothing to do with
    photography — sending 3 here does not take a picture, it changes how the
    gimbal responds to aircraft yaw.
    """

    TAKE_PICTURE = 0        #: capture one still to the SD card
    HDR_TOGGLE = 1          #: toggle HDR on/off
    RECORD_VIDEO_TOGGLE = 2  #: start recording, or stop if already recording
    LOCK_MODE = 3           #: yaw holds a WORLD heading, ignoring the airframe
    FOLLOW_MODE = 4         #: yaw follows the airframe heading
    FPV_MODE = 5            #: gimbal rigidly follows the airframe (no isolation)


class FunctionFeedback:
    """Reply byte of FUNCTION_FEEDBACK_INFO (0x0B) — the result of the last PHOTO."""

    SUCCESS = 0
    FAILED_TO_TAKE_PHOTO = 1
    HDR_ON = 2
    HDR_OFF = 3
    FAILED_TO_RECORD_VIDEO = 4   #: usually means no SD card

    NAMES = {0: 'success', 1: 'photo failed', 2: 'HDR on', 3: 'HDR off',
             4: 'record failed (SD card?)'}


class MotionMode:
    """Gimbal motion mode, as REPORTED in ACQUIRE_GIMBAL_CONFIG_INFO (0x0A).

    To SET it, send the matching PhotoFunction value — the numbers differ, which
    is exactly the sort of trap this table exists to remove.
    """

    LOCK = 0
    FOLLOW = 1
    FPV = 2

    NAMES = {0: 'lock', 1: 'follow', 2: 'fpv'}
    #: mode -> the PHOTO sub-command that selects it
    SET_BY = {0: PhotoFunction.LOCK_MODE,
              1: PhotoFunction.FOLLOW_MODE,
              2: PhotoFunction.FPV_MODE}


class MountingDirection:
    """How the gimbal is bolted on, from ACQUIRE_GIMBAL_CONFIG_INFO (0x0A).

    UPSIDE_DOWN inverts the sign of the pitch the gimbal reports and accepts, so
    a nadir command is NOT -90 deg on an inverted mount.
    """

    UNDEFINED = 0
    NORMAL = 1
    UPSIDE_DOWN = 2

    NAMES = {0: 'undefined', 1: 'normal', 2: 'upside down'}


class RecordingStatus:
    OFF = 0
    ON = 1
    NO_CARD = 2
    DATA_LOSS = 3

    NAMES = {0: 'off', 1: 'recording', 2: 'no SD card', 3: 'data loss'}


class HdrStatus:
    OFF = 0
    ON = 1


class VideoOutput:
    HDMI = 0
    CVBS = 1


class HardwareModel:
    """Reply of HARDWARE_ID (0x02). Which unit is on the other end.

    Worth checking before sending anything exotic: half the commands in this
    file exist only on the ZT-series. The A8 mini has no rangefinder and no
    thermal sensor.
    """

    UNKNOWN = 0
    A2 = 1
    A8 = 2
    ZR10 = 3
    ZR30 = 4
    ZT6 = 5
    ZT30 = 6

    NAMES = {0: 'unknown', 1: 'A2', 2: 'A8 mini', 3: 'ZR10', 4: 'ZR30',
             5: 'ZT6', 6: 'ZT30'}
    #: models that actually have a thermal sensor / rangefinder
    THERMAL = (5, 6)
    RANGEFINDER = (3, 4, 6)


class CameraImageType:
    """Payload byte of SET_IMAGE_TYPE (0x11) — which lens feeds the RTSP stream.

    Only meaningful on multi-sensor units. On an A8 mini there is one lens, so
    this does nothing. Changing it changes the picture `rtsp_bridge` receives,
    NOT the control link.
    """

    PIP_ZOOM_MAIN_WIDE_SUB = 0
    PIP_WIDE_MAIN_ZOOM_SUB = 1
    PIP_ZOOM_MAIN_THERMAL_SUB = 2
    ZOOM_MAIN_THERMAL_SUB = 3
    ZOOM_MAIN_WIDE_SUB = 4
    WIDE_MAIN_THERMAL_SUB = 5
    WIDE_MAIN_ZOOM_SUB = 6
    THERMAL_MAIN_ZOOM_SUB = 7
    THERMAL_MAIN_WIDE_SUB = 8


class ThermalPalette:
    """Payload byte of SET_THERMAL_PALETTE (0x1B). Thermal units only."""

    WHITE_HOT = 0
    REVERSED = 1
    RAINBOW = 2
    IRONBOW = 3


class ContinuousData:
    """First payload byte of REQUEST_CONTINUOUS_DATA (0x25) — what to stream.

    The second byte is the rate CODE, not a frequency in Hz. Streaming is
    cheaper than polling, but it keeps arriving after you stop reading, so a
    node that subscribes and dies leaves the gimbal talking to nobody.
    """

    ATTITUDE = 1
    LASER_RANGE = 2
    ENCODERS = 3
    MOTOR_CURRENT = 4

    #: rate code -> Hz, as the vendor defines them
    RATE_HZ = {0: 0, 1: 2, 2: 4, 3: 5, 4: 10, 5: 20, 6: 50, 7: 100}


class Direction:
    """Payload byte for the continuous MANUAL_ZOOM / MANUAL_FOCUS commands.

    These LATCH: the gimbal keeps zooming or focusing until STOP arrives, so a
    dropped stop packet runs the lens to its limit.
    """

    IN = 1        #: zoom in / focus far
    STOP = 0
    OUT = -1      #: zoom out / focus near


#: Payload byte that means "yes, do it" for the commands that take a bare flag
#: (CENTER, AUTO_FOCUS). Not a boolean on the wire — the vendor spells it 1.
TRIGGER = 1

# --- mechanical limits, A8 mini -------------------------------------------
# Commanding past these is silently clamped BY THE GIMBAL, which is
# indistinguishable from a command that never arrived, so `protocol.set_angle`
# clamps here instead where it can be logged.
A8_PITCH_MIN_DEG, A8_PITCH_MAX_DEG = -90.0, 25.0
A8_YAW_MIN_DEG, A8_YAW_MAX_DEG = -135.0, 135.0
A8_ZOOM_MAX = 6.0            #: digital only; there is no optical zoom
A8_MAX_SLEW_DEG_S = 90.0     #: what GIMBAL_ROTATION's +/-100 percent maps to

#: Scale factor between wire units and degrees. Angles are DECI-degrees.
DEG_PER_LSB = 0.1


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
