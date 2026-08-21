"""How the SIYI bytes get to the gimbal — over ethernet, or over a wire.

The gimbal speaks ONE protocol. `protocol.py` builds the frame — 55 66, length,
sequence, command, payload, CRC — and those bytes are byte-for-byte identical no
matter how they are delivered. What differs is only the pipe:

    UDP     the gimbal is a network device with an IP. The frame goes in one
            datagram to 192.168.144.25:37260. Needs the vehicle ethernet up.

    SERIAL  the gimbal is wired to the companion's UART pins. The same frame is
            written to /dev/ttyTHS1 at 115200 baud. No network involved.

So "serial support" is not a second protocol or a second driver — it is the same
packets down a different wire. That is why this file is small and why nothing
above it changes: `siyi_gimbal_node` asks a transport to `send()` and to hand
back whatever arrived, and does not know or care which one it got.

The distinction that DOES matter is framing. UDP delivers whole datagrams, so
one read is one frame. A serial line is a byte stream with no message
boundaries: a frame can arrive split across reads, or two can arrive in one, so
the serial transport keeps a buffer and cuts frames out of it using the length
field. Getting that wrong shows up as intermittent "the gimbal ignores me",
which is the worst kind of bug to chase on a vehicle.
"""

from __future__ import annotations

import socket

from . import protocol as siyi


class Transport:
    """Common interface. `send` one frame, `read` returns whole frames."""

    def send(self, packet: bytes) -> None:
        raise NotImplementedError

    def read(self) -> list[bytes]:
        """Every COMPLETE frame available right now. Never blocks."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def description(self) -> str:
        raise NotImplementedError


class UdpTransport(Transport):
    """The gimbal as a network peer. One datagram in, one datagram out."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def send(self, packet: bytes) -> None:
        self.sock.sendto(packet, (self.host, self.port))

    def read(self) -> list[bytes]:
        out = []
        while True:
            try:
                data, _addr = self.sock.recvfrom(1024)
            except (BlockingIOError, OSError):
                return out
            out.append(data)          # a datagram IS a frame

    def close(self) -> None:
        self.sock.close()

    @property
    def description(self) -> str:
        return f'udp {self.host}:{self.port}'


class SerialTransport(Transport):
    """The gimbal on a UART. Same bytes, but a stream with no boundaries.

    `pyserial` is imported lazily so that a machine with no serial hardware —
    every desktop running the SITL side of this repo — can still import this
    module and run the UDP path.
    """

    def __init__(self, port: str, baud: int, timeout: float = 0.0):
        try:
            import serial
        except ImportError as exc:      # pragma: no cover - environment issue
            raise RuntimeError(
                'pyserial is required for the serial transport '
                '(pip3 install pyserial)') from exc
        self.port, self.baud = port, int(baud)
        try:
            self.ser = serial.Serial(port, int(baud), timeout=timeout)
        except Exception as exc:
            # pyserial wraps the OS error in SerialException, so the useful
            # distinction is in the TEXT, not the exception type. Matching on
            # the type alone silently loses the advice below — which is the
            # whole reason for catching it here.
            text = str(exc).lower()
            if 'no such file' in text or 'errno 2' in text:
                hint = (f'{port} does not exist. On a Jetson the header UART is '
                        f'/dev/ttyTHS1; a USB-TTL adapter is usually '
                        f'/dev/ttyUSB0 (ls /dev/tty* to see what is there). '
                        f'Use transport:=udp if the gimbal is on the vehicle '
                        f'network instead.')
            elif 'permission' in text or 'errno 13' in text:
                hint = (f'no permission to open {port} — add yourself to the '
                        f'dialout group (sudo usermod -aG dialout $USER) and '
                        f'log back in')
            elif 'busy' in text or 'errno 16' in text:
                hint = (f'{port} is already open — another node or a serial '
                        f'monitor still holds it')
            else:
                hint = f'could not open {port} @{baud}: {exc}'
            raise RuntimeError(hint) from exc
        self._buf = bytearray()

    def send(self, packet: bytes) -> None:
        self.ser.write(packet)

    def read(self) -> list[bytes]:
        """Cut whole frames out of the byte stream.

        A SIYI frame is 8 header bytes + a payload whose length the header
        declares + 2 CRC bytes, so the length is knowable from the first 8
        bytes. Anything that is not a valid header is discarded one byte at a
        time until the stream re-synchronises — line noise and half-frames from
        before we opened the port are ordinary, not errors.
        """
        waiting = self.ser.in_waiting
        if waiting:
            self._buf += self.ser.read(waiting)

        frames = []
        while len(self._buf) >= 10:
            if self._buf[0] != siyi.HEADER1 or self._buf[1] != siyi.HEADER2:
                del self._buf[0]           # resynchronise
                continue
            plen = int.from_bytes(self._buf[3:5], 'little')
            total = plen + 10
            if len(self._buf) < total:
                break                      # rest of the frame has not arrived
            frames.append(bytes(self._buf[:total]))
            del self._buf[:total]
        # A buffer that only grows means we are locked onto garbage that looks
        # like a header; cap it so a bad wire cannot exhaust memory.
        if len(self._buf) > 4096:
            del self._buf[:-1024]
        return frames

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:               # pragma: no cover
            pass

    @property
    def description(self) -> str:
        return f'serial {self.port} @{self.baud}'


def make(kind: str, *, host: str, port: int, device: str, baud: int) -> Transport:
    """Build the transport named by `kind` ('udp' or 'serial')."""
    kind = kind.lower()
    if kind == 'udp':
        return UdpTransport(host, port)
    if kind == 'serial':
        return SerialTransport(device, baud)
    raise ValueError(f"transport must be 'udp' or 'serial', got '{kind}'")
