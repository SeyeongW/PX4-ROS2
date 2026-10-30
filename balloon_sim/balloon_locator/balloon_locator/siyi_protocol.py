"""
프레임: STX(0x5566) | CTRL(1) | DATA_LEN(2,LE) | SEQ(2,LE) | CMD_ID(1) | DATA | CRC16(2,LE)
CRC16 = XMODEM (poly 0x1021, init 0). 명령 ID/페이로드 형식은 SIYI SDK 문서 기준이며
펌웨어 버전에 따라 다를 수 있으니 실기에서 한 번 확인할 것 (README 참고).
"""
import struct
from typing import NamedTuple

STX = b'\x55\x66'

CMD_MANUAL_ZOOM = 0x05      # int8: 1 줌인, 0 정지, -1 줌아웃
CMD_CENTER = 0x08           # uint8: 1 = 원점 복귀
CMD_ROTATE = 0x07           # int8 yaw_speed, int8 pitch_speed (-100..100)
CMD_ATTITUDE = 0x0D         # 응답: int16 x6 (yaw,pitch,roll,각속도 3개), 0.1deg 단위
CMD_ABSOLUTE_ZOOM = 0x0F    # uint8 정수부, uint8 소수부(0.1 단위)
CMD_CURRENT_ZOOM = 0x18     # 응답: uint8 정수부, uint8 소수부

HEADER_LEN = 8  # STX2 + CTRL1 + LEN2 + SEQ2 + CMD1


class ProtocolError(ValueError):
    pass


class Frame(NamedTuple):
    cmd_id: int
    seq: int
    data: bytes
    ctrl: int


def crc16_xmodem(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build_frame(cmd_id: int, data: bytes = b'', seq: int = 0, need_ack: bool = True) -> bytes:
    body = STX + struct.pack('<BHHB', 1 if need_ack else 0, len(data), seq & 0xFFFF, cmd_id) + data
    return body + struct.pack('<H', crc16_xmodem(body))


def parse_frame(buf: bytes) -> Frame:
    if len(buf) < HEADER_LEN + 2 or buf[:2] != STX:
        raise ProtocolError('bad header')
    ctrl, data_len, seq, cmd_id = struct.unpack_from('<BHHB', buf, 2)
    total = HEADER_LEN + data_len + 2
    if len(buf) < total:
        raise ProtocolError('truncated frame')
    body = buf[:total - 2]
    (crc,) = struct.unpack_from('<H', buf, total - 2)
    if crc != crc16_xmodem(body):
        raise ProtocolError('crc mismatch')
    return Frame(cmd_id, seq, bytes(buf[HEADER_LEN:HEADER_LEN + data_len]), ctrl)


def _clamp_i8(v: float) -> int:
    return max(-100, min(100, int(round(v))))


def encode_rotate(yaw_speed: float, pitch_speed: float) -> bytes:
    return struct.pack('<bb', _clamp_i8(yaw_speed), _clamp_i8(pitch_speed))


def encode_absolute_zoom(zoom: float) -> bytes:
    zoom = max(1.0, zoom)
    whole = int(zoom)
    tenth = min(9, int(round((zoom - whole) * 10)))
    return struct.pack('<BB', whole, tenth)


def decode_zoom(data: bytes) -> float:
    if len(data) < 2:
        raise ProtocolError('zoom payload too short')
    whole, tenth = struct.unpack_from('<BB', data)
    return whole + tenth / 10.0


def decode_attitude(data: bytes):
    """(yaw, pitch, roll) [deg]. 각속도(뒤 6바이트)는 쓰지 않는다."""
    if len(data) < 6:
        raise ProtocolError('attitude payload too short')
    yaw, pitch, roll = struct.unpack_from('<hhh', data)
    return yaw / 10.0, pitch / 10.0, roll / 10.0