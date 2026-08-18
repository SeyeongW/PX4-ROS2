"""radio_probe — what is actually coming over the radio, and what to do about it.

    ros2 run trailer_link radio_probe
    ros2 run trailer_link radio_probe --device /dev/ttyUSB1 --seconds 20

THE FAILURE THIS EXISTS FOR
---------------------------
A silent `/trailer/fix` with a perfectly healthy-looking link. The telemetry
screen shows satellites, fix type and HDOP, so the GPS is obviously working and
the radio is obviously connected — and yet no coordinate ever arrives.

Those are two different messages, and on ArduPilot they are in two different
stream groups:

    GPS_RAW_INT           sats, fix type, HDOP        SRx_EXT_STAT
    GLOBAL_POSITION_INT   LATITUDE AND LONGITUDE      SRx_POSITION

`trailer_gps_node` publishes only from the second one, because only the second
one carries a position. So a link with `SRx_POSITION = 0` produces exactly the
symptom above: every reassuring number arrives, and the one number the mission
needs never does. Reading a stats line cannot distinguish that from "the GPS has
no fix yet", which is the other cause and needs the opposite response — waiting,
not a parameter change.

So this listens for a few seconds and says which of the two it is, in words that
name the next action. It is READ-ONLY unless asked otherwise: it opens the port,
counts, and prints.

ONE PORT, ONE READER. Stop trailer_gps_node (or run_px4) before probing — a
serial port cannot be shared, and the loser of that race just sees nothing.

Needs pymavlink:  pip install pymavlink
"""

from __future__ import annotations

import argparse
import collections
import sys
import time

try:
    from pymavlink import mavutil
except ImportError as exc:                       # pragma: no cover
    raise ImportError(
        'radio_probe needs pymavlink — `pip install pymavlink`') from exc


#: MAV_GPS_FIX_TYPE, for a line a human has to act on.
FIX_NAMES = {
    0: 'no GPS', 1: 'no fix', 2: '2D fix', 3: '3D fix', 4: 'DGPS',
    5: 'RTK float', 6: 'RTK fixed', 7: 'static', 8: 'PPP',
}

#: The message the whole pipeline hangs on.
POSITION_MSG = 'GLOBAL_POSITION_INT'
#: The one that makes a dead pipeline look alive.
GPS_MSG = 'GPS_RAW_INT'

#: Below this there is no usable position, and an autopilot is CORRECT not to
#: send GLOBAL_POSITION_INT — nothing is broken, the receiver is not ready.
MIN_FIX_TYPE = 3


def verdict(counts, fix_type: int | None, sats: int) -> tuple[str, str]:
    """(headline, what to do) from the message inventory. No I/O, so testable.

    Ordered by how far down the chain the failure is, because the first thing
    that is wrong is the only thing worth reporting: a dead link makes every
    later question meaningless.
    """
    total = sum(counts.values())
    if total == 0:
        return ('NOTHING ON THE WIRE',
                'No MAVLink at all. Check the radio pair is powered and bound, '
                'the USB device is the right one, and the baud matches (SiK '
                'defaults to 57600). If trailer_gps_node or run_px4 is running, '
                'stop it first — one serial port takes one reader.')

    if POSITION_MSG in counts:
        return ('POSITION IS ARRIVING',
                f'{POSITION_MSG} is on the wire, so the radio half is fine and '
                f'trailer_gps_node should be publishing /trailer/fix. If it is '
                f'not, the problem is downstream: check the node is running, '
                f'and that it and its consumer share ROS_DOMAIN_ID and '
                f'RMW_IMPLEMENTATION.')

    if GPS_MSG not in counts:
        return ('LINK UP, BUT NO GPS MESSAGES',
                f'MAVLink is arriving but neither {GPS_MSG} nor {POSITION_MSG} '
                f'is. The autopilot is sending some streams and not others — '
                f'raise SRx_EXT_STAT and SRx_POSITION on the serial port the '
                f'radio is plugged into (SERIAL1 -> SR1_*, SERIAL2 -> SR2_*).')

    if fix_type is not None and fix_type < MIN_FIX_TYPE:
        name = FIX_NAMES.get(fix_type, f'fix {fix_type}')
        return ('THE TRAILER HAS NO POSITION YET',
                f'GPS reports {name} with {sats} satellites, so the autopilot is '
                f'correctly not sending {POSITION_MSG} — there is no position to '
                f'send. Nothing is misconfigured. Take the trailer somewhere with '
                f'sky and wait for a 3D fix.')

    return ('THE POSITION STREAM IS SWITCHED OFF',
            f'GPS has a {FIX_NAMES.get(fix_type, fix_type)} with {sats} '
            f'satellites, so a position EXISTS — but {POSITION_MSG} is not being '
            f'sent, while {GPS_MSG} is. They are different stream groups. Set '
            f'SRx_POSITION >= 2 on the serial port the radio is on (SERIAL1 -> '
            f'SR1_POSITION, SERIAL2 -> SR2_POSITION), or re-run this with '
            f'--request-position to ask for it over the link.')


def _request_position(master, rate_hz: float) -> None:
    """Ask the autopilot to start sending GLOBAL_POSITION_INT.

    A runtime request, not a parameter write: it does not survive a power cycle,
    which is why it is a diagnosis aid and not the fix. The fix is SRx_POSITION.
    """
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        int(1e6 / max(rate_hz, 0.1)),            # interval, microseconds
        0, 0, 0, 0, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='What is arriving over the trailer radio, and why '
                    '/trailer/fix may be silent.')
    ap.add_argument('--device', default='/dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=57600)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--request-position', action='store_true',
                    help='ask the autopilot to send GLOBAL_POSITION_INT '
                         '(runtime only; the durable fix is SRx_POSITION)')
    args = ap.parse_args(argv)

    print(f'\ntrailer radio probe — {args.seconds:.0f} s on {args.device} @ '
          f'{args.baud}\n')
    try:
        master = mavutil.mavlink_connection(args.device, baud=args.baud)
    except Exception as e:                                   # noqa: BLE001
        print(f'  cannot open {args.device}: {e}\n')
        print('  Is the radio plugged in?  ls /dev/ttyUSB*')
        print('  Is something else holding the port?  '
              'ros2 node list | grep trailer_gps')
        return 2

    counts: collections.Counter = collections.Counter()
    fix_type: int | None = None
    sats = 0
    hdop = float('nan')
    last_pos = None
    sysid = None
    requested = False

    t0 = time.time()
    while time.time() - t0 < args.seconds:
        msg = master.recv_match(blocking=True, timeout=1.0)
        if msg is None:
            continue
        mtype = msg.get_type()
        if mtype == 'BAD_DATA':
            counts['(corrupt frames)'] += 1
            continue
        counts[mtype] += 1
        if sysid is None:
            sysid = msg.get_srcSystem()
            print(f'  link up — MAVLink from system {sysid}\n')
            if args.request_position:
                _request_position(master, 5.0)
                requested = True
                print(f'  asked system {sysid} for {POSITION_MSG} at 5 Hz\n')
        if mtype == GPS_MSG:
            fix_type = int(msg.fix_type)
            sats = int(msg.satellites_visible)
            hdop = msg.eph / 100.0 if msg.eph not in (0, 65535) else float('nan')
        elif mtype == POSITION_MSG:
            last_pos = (msg.lat * 1e-7, msg.lon * 1e-7, msg.alt * 1e-3)

    elapsed = max(time.time() - t0, 1e-6)

    print('  count   rate   message')
    for mtype, n in counts.most_common():
        print(f'  {n:5d}  {n / elapsed:5.1f}   {mtype}')
    if not counts:
        print('  (none)')

    print()
    if fix_type is not None:
        print(f'  gps       {FIX_NAMES.get(fix_type, fix_type)}, {sats} sats, '
              f'hdop {hdop:.1f}')
    if last_pos:
        print(f'  position  {last_pos[0]:.7f}, {last_pos[1]:.7f}  '
              f'alt {last_pos[2]:.1f} m AMSL')
    else:
        print(f'  position  NONE — no {POSITION_MSG} in {elapsed:.0f} s')

    headline, action = verdict(counts, fix_type, sats)
    print(f'\n  {headline}\n')
    for line in _wrap(action, 74):
        print(f'  {line}')
    if requested and last_pos is None:
        print('\n  The stream was requested and still nothing came, so the '
              'autopilot\n  refused or does not support the request — set '
              'SRx_POSITION directly.')
    print()
    return 0 if last_pos else 1


def _wrap(text: str, width: int) -> list:
    """Wrap without importing textwrap for one call."""
    out, line = [], ''
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f'{line} {word}'.strip()
    if line:
        out.append(line)
    return out


if __name__ == '__main__':
    sys.exit(main())
