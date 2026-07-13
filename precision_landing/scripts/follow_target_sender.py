#!/usr/bin/env python3
"""FOLLOW_TARGET (#144) sender for PX4 AUTO.FOLLOW_TARGET -- LAPTOP side.

Reads the laptop's own GPS (a USB NMEA receiver) and streams MAVLink
FOLLOW_TARGET to PX4 over the laptop's telemetry link, so PX4 in
AUTO.FOLLOW_TARGET follows the laptop. This is the piece QGroundControl-mobile
would normally do for you; a desktop/laptop QGC can't, hence this script.

Runs on the LAPTOP, NOT the Jetson. Independent of ROS / follow_precland_node
(that runs on the Jetson over USB; this uses a SEPARATE telemetry link). Only
needs pymavlink (+ pyserial for a serial GPS):
    pip install pymavlink pyserial

Examples
  # 1) GROUND CHECK ONLY -- read GPS, print fix status live, send nothing.
  #    Use this to confirm the receiver gets a fix before flying.
  python3 follow_target_sender.py --gps /dev/ttyUSB0 --gps-baud 9600 --check

  # 2) RUN -- read GPS, send FOLLOW_TARGET to PX4 via the telemetry radio.
  python3 follow_target_sender.py --gps /dev/ttyUSB0 --gps-baud 9600 \
        --mavlink /dev/ttyUSB1 --mavlink-baud 57600

  # 3) FIXED test coordinate (no GPS) to a UDP/SITL PX4.
  python3 follow_target_sender.py --fixed 37.5665 126.9780 30 \
        --mavlink udp:127.0.0.1:14550

FOLLOW_TARGET is a broadcast-style message (no target_system field): PX4's
follow-target estimator consumes it from whichever link it arrives on.
"""

import argparse
import math
import sys
import time


# --- NMEA GPS reader -------------------------------------------------------
class GpsState:
    def __init__(self):
        self.lat = None          # decimal degrees
        self.lon = None
        self.alt = 0.0           # m MSL (GGA)
        self.fix_quality = 0     # GGA: 0=no fix, 1=GPS, 2=DGPS, ...
        self.num_sats = 0
        self.hdop = 99.9
        self.rmc_valid = False   # RMC status A(valid)/V(invalid)
        self.speed = 0.0         # m/s (from RMC SOG)
        self.course = 0.0        # deg (from RMC COG)
        self.last_update = 0.0

    def has_fix(self):
        return self.lat is not None and (self.fix_quality > 0 or self.rmc_valid)


def _nmea_to_deg(val, hemi):
    """ddmm.mmmm + N/S/E/W -> signed decimal degrees."""
    if not val:
        return None
    v = float(val)
    deg = int(v / 100)
    minutes = v - deg * 100
    dec = deg + minutes / 60.0
    if hemi in ('S', 'W'):
        dec = -dec
    return dec


def _checksum_ok(line):
    if '*' not in line:
        return False
    body, _, cs = line[1:].partition('*')
    try:
        want = int(cs[:2], 16)
    except ValueError:
        return False
    got = 0
    for ch in body:
        got ^= ord(ch)
    return got == want


def parse_nmea(line, st):
    """Update GpsState from one NMEA sentence. Returns True if it was a
    recognised position sentence."""
    line = line.strip()
    if not line.startswith('$') or not _checksum_ok(line):
        return False
    f = line.split('*')[0].split(',')
    typ = f[0][3:]   # strip talker id ($GP/$GN/$GA...) -> GGA/RMC/...
    try:
        if typ == 'GGA' and len(f) >= 10:
            lat = _nmea_to_deg(f[2], f[3])
            lon = _nmea_to_deg(f[4], f[5])
            if lat is not None and lon is not None:
                st.lat, st.lon = lat, lon
            st.fix_quality = int(f[6]) if f[6] else 0
            st.num_sats = int(f[7]) if f[7] else 0
            st.hdop = float(f[8]) if f[8] else 99.9
            st.alt = float(f[9]) if f[9] else st.alt
            st.last_update = time.time()
            return True
        if typ == 'RMC' and len(f) >= 9:
            st.rmc_valid = (f[2] == 'A')
            lat = _nmea_to_deg(f[3], f[4])
            lon = _nmea_to_deg(f[5], f[6])
            if lat is not None and lon is not None:
                st.lat, st.lon = lat, lon
            if f[7]:
                st.speed = float(f[7]) * 0.514444     # knots -> m/s
            if f[8]:
                st.course = float(f[8])
            st.last_update = time.time()
            return True
    except (ValueError, IndexError):
        return False
    return False


# --- FOLLOW_TARGET send ----------------------------------------------------
def make_master(conn, baud, source_system):
    from pymavlink import mavutil
    kwargs = {'source_system': source_system, 'source_component': 190}
    if any(conn.startswith(p) for p in ('udp', 'tcp')):
        master = mavutil.mavlink_connection(conn, **kwargs)
    else:
        master = mavutil.mavlink_connection(conn, baud=baud, **kwargs)
    return master


def send_follow_target(master, st):
    # est_capabilities bitmask: bit0(1)=position, bit1(2)=velocity.
    caps = 0b1
    vel = [0.0, 0.0, 0.0]
    if st.speed > 0.05:
        caps |= 0b10
        rad = math.radians(st.course)
        vel = [st.speed * math.cos(rad),   # North
               st.speed * math.sin(rad),   # East
               0.0]
    master.mav.follow_target_send(
        int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF,   # timestamp ms
        caps,
        int(st.lat * 1e7), int(st.lon * 1e7), float(st.alt),
        vel, [0.0, 0.0, 0.0],          # vel, accel
        [1.0, 0.0, 0.0, 0.0],          # attitude_q
        [0.0, 0.0, 0.0],               # rates
        [0.0, 0.0, 0.0],               # position_cov
        0)                             # custom_state


def fix_str(st):
    q = {0: 'NO FIX', 1: 'GPS', 2: 'DGPS', 4: 'RTK-fix', 5: 'RTK-float'}.get(
        st.fix_quality, str(st.fix_quality))
    ll = f'{st.lat:.6f},{st.lon:.6f}' if st.lat is not None else '--,--'
    return (f'fix={q} sats={st.num_sats} hdop={st.hdop:.1f} '
            f'alt={st.alt:.0f}m pos=({ll}) rmc={"valid" if st.rmc_valid else "-"}')


def main():
    ap = argparse.ArgumentParser(description='PX4 FOLLOW_TARGET sender (laptop GPS)')
    ap.add_argument('--gps', help='GPS serial device, e.g. /dev/ttyUSB0')
    ap.add_argument('--gps-baud', type=int, default=9600)
    ap.add_argument('--fixed', nargs=3, type=float, metavar=('LAT', 'LON', 'ALT'),
                    help='send a fixed coordinate instead of reading GPS')
    ap.add_argument('--mavlink', help='PX4 link, e.g. /dev/ttyUSB1 or udp:host:port')
    ap.add_argument('--mavlink-baud', type=int, default=57600)
    ap.add_argument('--rate', type=float, default=2.0, help='send rate Hz (default 2)')
    ap.add_argument('--sysid', type=int, default=255, help='our (GCS) system id')
    ap.add_argument('--check', action='store_true',
                    help='GPS ground-check only: print fix status, send nothing')
    args = ap.parse_args()

    st = GpsState()

    # --- fixed-coordinate mode (no GPS) ---
    if args.fixed:
        st.lat, st.lon, st.alt = args.fixed
        st.fix_quality = 1
        st.rmc_valid = True
        ser = None
    elif args.gps:
        try:
            import serial
        except ImportError:
            sys.exit('pyserial 필요: pip install pyserial')
        try:
            ser = serial.Serial(args.gps, args.gps_baud, timeout=1.0)
        except Exception as e:
            sys.exit(f'GPS 포트 열기 실패 {args.gps}: {e}')
    else:
        sys.exit('--gps 또는 --fixed 중 하나는 필요합니다 (--help)')

    # --- ground-check mode: just read + print, no MAVLink ---
    if args.check:
        print(f'[check] {args.gps or "fixed"} 읽는 중... (Ctrl-C 종료)')
        last = 0.0
        try:
            while True:
                if ser is not None:
                    raw = ser.readline().decode('ascii', 'ignore')
                    parse_nmea(raw, st)
                now = time.time()
                if now - last >= 1.0:
                    last = now
                    ok = 'OK ✅' if st.has_fix() and st.num_sats >= 4 else '대기...'
                    print(f'[{ok}] {fix_str(st)}')
                if ser is None:
                    time.sleep(1.0)
        except KeyboardInterrupt:
            print('\n종료')
        return

    # --- run mode: read GPS + stream FOLLOW_TARGET ---
    if not args.mavlink:
        sys.exit('전송에는 --mavlink 이 필요합니다 (또는 --check 로 확인만)')
    master = make_master(args.mavlink, args.mavlink_baud, args.sysid)
    print(f'[send] MAVLink={args.mavlink} rate={args.rate}Hz sysid={args.sysid}')

    period = 1.0 / max(args.rate, 0.1)
    next_send = time.time()
    warned = False
    try:
        while True:
            if ser is not None:
                raw = ser.readline().decode('ascii', 'ignore')
                parse_nmea(raw, st)
            now = time.time()
            if now >= next_send:
                next_send = now + period
                if st.has_fix():
                    send_follow_target(master, st)
                    warned = False
                    print(f'[TX] {fix_str(st)}          ', end='\r')
                elif not warned:
                    warned = True
                    print(f'\n[WAIT] GPS 픽스 없음 -- 송출 대기: {fix_str(st)}')
    except KeyboardInterrupt:
        print('\n종료')


if __name__ == '__main__':
    main()
