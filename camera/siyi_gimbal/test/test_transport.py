"""Serial framing, without a serial port.

UDP hands you whole datagrams, so one read is one frame and there is nothing to
get wrong. A UART is a byte stream: a frame can arrive split across reads, two
can arrive in one, and the line can contain noise or the tail of a frame that
was in flight before the port was opened. Every one of those shows up in flight
as "the gimbal sometimes ignores me", so they are pinned here.

A fake serial object stands in for pyserial, which keeps this runnable on a
desktop with no gimbal and no UART.
"""

import struct

from siyi_gimbal import protocol as p
from siyi_gimbal import transport as tp


class FakeSerial:
    """Just enough of pyserial for SerialTransport: a byte stream you feed."""

    def __init__(self):
        self.rx = bytearray()      # bytes the "gimbal" has sent us
        self.tx = bytearray()      # bytes we have written

    @property
    def in_waiting(self):
        return len(self.rx)

    def read(self, n):
        out = bytes(self.rx[:n])
        del self.rx[:n]
        return out

    def write(self, data):
        self.tx += data

    def close(self):
        pass


def _serial_transport():
    """A SerialTransport wired to a FakeSerial, bypassing __init__'s port open."""
    t = tp.SerialTransport.__new__(tp.SerialTransport)
    t.port, t.baud = '/dev/fake', 115200
    t.ser = FakeSerial()
    t._buf = bytearray()
    return t


def _attitude_frame(pitch_deg=-90.0):
    payload = struct.pack('<hhhhhh', 0, int(pitch_deg * 10), 0, 0, 0, 0)
    return p.encode(p.ACQUIRE_GIMBAL_ATTITUDE, payload)


def test_a_whole_frame_in_one_read():
    t = _serial_transport()
    t.ser.rx += _attitude_frame()
    frames = t.read()
    assert len(frames) == 1
    cmd, payload = p.decode(frames[0])
    assert cmd == p.ACQUIRE_GIMBAL_ATTITUDE
    assert p.parse_attitude(payload)[1] == -90.0


def test_a_frame_split_across_reads_is_reassembled():
    """The failure UDP cannot have: half a frame arrives, then the rest."""
    t = _serial_transport()
    frame = _attitude_frame()
    t.ser.rx += frame[:5]
    assert t.read() == [], 'half a frame must not be returned'
    t.ser.rx += frame[5:]
    frames = t.read()
    assert len(frames) == 1 and frames[0] == frame


def test_two_frames_in_one_read_are_both_returned():
    t = _serial_transport()
    t.ser.rx += _attitude_frame(-90.0) + _attitude_frame(-45.0)
    frames = t.read()
    assert len(frames) == 2
    assert p.parse_attitude(p.decode(frames[0])[1])[1] == -90.0
    assert p.parse_attitude(p.decode(frames[1])[1])[1] == -45.0


def test_leading_noise_is_discarded_and_the_stream_resynchronises():
    """A UART carries whatever was on the wire before we opened it."""
    t = _serial_transport()
    t.ser.rx += b'\x00\xff\x12garbage' + _attitude_frame()
    frames = t.read()
    assert len(frames) == 1
    assert p.decode(frames[0])[0] == p.ACQUIRE_GIMBAL_ATTITUDE


def test_buffer_cannot_grow_without_bound_on_a_bad_line():
    """Noise that happens to look like a header must not exhaust memory."""
    t = _serial_transport()
    # a header claiming a huge payload that never arrives
    t.ser.rx += bytes([p.HEADER1, p.HEADER2, 1]) + struct.pack('<H', 60000) \
        + b'\x00' * 9000
    t.read()
    assert len(t._buf) <= 4096


def test_send_writes_the_exact_frame():
    t = _serial_transport()
    t.send(p.look_down())
    assert bytes(t.ser.tx) == p.look_down()
    assert t.ser.tx[0] == 0x55 and t.ser.tx[1] == 0x66


def test_factory_rejects_an_unknown_transport():
    try:
        tp.make('carrier-pigeon', host='h', port=1, device='/dev/x', baud=1)
    except ValueError as exc:
        assert 'udp' in str(exc) and 'serial' in str(exc)
    else:
        raise AssertionError('an unknown transport must not be accepted')


def test_udp_transport_treats_a_datagram_as_a_frame():
    t = tp.UdpTransport('127.0.0.1', 37999)
    assert 'udp' in t.description and '37999' in t.description
    assert t.read() == []          # nothing sent, nothing waiting
    t.close()
