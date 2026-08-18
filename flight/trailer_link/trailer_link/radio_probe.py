"""Report what is arriving over the field trailer's MAVLink radio.

This is the read-only diagnostic from the deployed ``wang`` branch.  It tells
apart a missing GPS fix from a disabled ``GLOBAL_POSITION_INT`` stream.  The
optional request is temporary and does not write autopilot parameters.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time

try:
    from pymavlink import mavutil
except ImportError as exc:  # pragma: no cover - deployment dependency
    raise ImportError('radio_probe needs pymavlink') from exc


FIX_NAMES = {
    0: 'no GPS', 1: 'no fix', 2: '2D fix', 3: '3D fix', 4: 'DGPS',
    5: 'RTK float', 6: 'RTK fixed', 7: 'static', 8: 'PPP',
}
POSITION_MSG = 'GLOBAL_POSITION_INT'
GPS_MSG = 'GPS_RAW_INT'
MIN_FIX_TYPE = 3


def verdict(counts, fix_type: int | None, sats: int) -> tuple[str, str]:
    """Return an operator verdict and action from a message inventory."""
    if sum(counts.values()) == 0:
        return (
            'NOTHING ON THE WIRE',
            'Check radio power/binding, USB device and baud. Stop any other '
            'reader first because one serial port takes one reader.',
        )
    if POSITION_MSG in counts:
        return (
            'POSITION IS ARRIVING',
            f'{POSITION_MSG} is on the wire. If /trailer/fix is silent, check '
            'the ROS node, ROS_DOMAIN_ID and RMW_IMPLEMENTATION.',
        )
    if GPS_MSG not in counts:
        return (
            'LINK UP, BUT NO GPS MESSAGES',
            'Raise SRx_EXT_STAT and SRx_POSITION on the serial port used by '
            'the radio (SERIAL1 -> SR1_*, SERIAL2 -> SR2_*).',
        )
    if fix_type is not None and fix_type < MIN_FIX_TYPE:
        return (
            'THE TRAILER HAS NO POSITION YET',
            f'GPS reports {FIX_NAMES.get(fix_type, fix_type)} with {sats} '
            'satellites. Nothing is misconfigured; get sky view and wait for '
            'a 3D fix.',
        )
    return (
        'THE POSITION STREAM IS SWITCHED OFF',
        f'GPS has {FIX_NAMES.get(fix_type, fix_type)} with {sats} satellites, '
        f'but {POSITION_MSG} is absent. Set SRx_POSITION >= 5, or use '
        '--request-position for a temporary 5 Hz request.',
    )


def _request_position(master, rate_hz: float) -> None:
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        int(1.0e6 / max(rate_hz, 0.1)),
        0, 0, 0, 0, 0,
    )


def _wrap(text: str, width: int) -> list[str]:
    output, line = [], ''
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            output.append(line)
            line = word
        else:
            line = f'{line} {word}'.strip()
    if line:
        output.append(line)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Inspect the trailer radio position stream.')
    parser.add_argument('--device', default='/dev/ttyUSB0')
    parser.add_argument('--baud', type=int, default=57600)
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument(
        '--request-position', action='store_true',
        help='temporarily request GLOBAL_POSITION_INT at 5 Hz')
    args = parser.parse_args(argv)

    print(f'\ntrailer radio probe — {args.seconds:.0f}s on '
          f'{args.device} @ {args.baud}\n')
    try:
        master = mavutil.mavlink_connection(args.device, baud=args.baud)
    except Exception as exc:  # noqa: BLE001
        print(f'cannot open {args.device}: {exc}')
        return 2

    counts: collections.Counter = collections.Counter()
    fix_type: int | None = None
    sats = 0
    last_position = None
    requested = False
    start = time.time()
    while time.time() - start < args.seconds:
        message = master.recv_match(blocking=True, timeout=1.0)
        if message is None:
            continue
        message_type = message.get_type()
        if message_type == 'BAD_DATA':
            counts['(corrupt frames)'] += 1
            continue
        counts[message_type] += 1
        if sum(counts.values()) == 1:
            print(f'link up — MAVLink system {message.get_srcSystem()}\n')
            if args.request_position:
                _request_position(master, 5.0)
                requested = True
                print(f'asked for {POSITION_MSG} at 5 Hz\n')
        if message_type == GPS_MSG:
            fix_type = int(message.fix_type)
            sats = int(message.satellites_visible)
        elif message_type == POSITION_MSG:
            last_position = (
                message.lat * 1.0e-7,
                message.lon * 1.0e-7,
                message.alt * 1.0e-3,
            )

    elapsed = max(time.time() - start, 1.0e-6)
    print('count   rate   message')
    for message_type, count in counts.most_common():
        print(f'{count:5d}  {count / elapsed:5.1f}   {message_type}')
    if not counts:
        print('(none)')
    if last_position is not None:
        print(f'position {last_position[0]:.7f}, {last_position[1]:.7f}, '
              f'{last_position[2]:.1f}m AMSL')

    headline, action = verdict(counts, fix_type, sats)
    print(f'\n{headline}\n')
    for line in _wrap(action, 74):
        print(line)
    if requested and last_position is None:
        print('\nRequest failed; configure SRx_POSITION persistently.')
    print()
    return 0 if last_position is not None else 1


if __name__ == '__main__':
    sys.exit(main())
