import pytest

from balloon_locator import siyi_protocol as sp


def test_crc_xmodem_check_value():
    assert sp.crc16_xmodem(b'123456789') == 0x31C3


def test_attitude_request_matches_sdk_example():
    # SIYI SDK 문서의 "짐벌 자세 요청" 예시 프레임
    assert sp.build_frame(sp.CMD_ATTITUDE, b'', seq=0, need_ack=True).hex() == '556601000000000de805'


def test_roundtrip():
    raw = sp.build_frame(sp.CMD_ROTATE, sp.encode_rotate(30, -40), seq=7)
    fr = sp.parse_frame(raw)
    assert (fr.cmd_id, fr.seq, fr.data) == (sp.CMD_ROTATE, 7, bytes([30, 256 - 40]))


def test_bad_crc_and_header():
    raw = bytearray(sp.build_frame(sp.CMD_ATTITUDE, b'\x00' * 12))
    raw[-1] ^= 0xFF
    with pytest.raises(sp.ProtocolError):
        sp.parse_frame(bytes(raw))
    with pytest.raises(sp.ProtocolError):
        sp.parse_frame(b'\x00\x00' + bytes(raw[2:]))


def test_rotate_clamped_and_zoom_codec():
    assert sp.encode_rotate(500, -500) == bytes([100, 256 - 100])
    assert sp.decode_zoom(sp.encode_absolute_zoom(4.5)) == pytest.approx(4.5)


def test_decode_attitude_units():
    import struct
    data = struct.pack('<hhhhhh', 900, -300, 15, 0, 0, 0)
    assert sp.decode_attitude(data) == pytest.approx((90.0, -30.0, 1.5))
