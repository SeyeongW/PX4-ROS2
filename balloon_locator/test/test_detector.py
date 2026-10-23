import cv2
import numpy as np

from balloon_locator.balloon_detector import detect_balloons, parse_hsv_ranges

RED = parse_hsv_ranges([0, 120, 70, 10, 255, 255, 170, 120, 70, 180, 255, 255])


def _sky(w=640, h=480):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (200, 140, 60)    # BGR 하늘색
    return img


def test_finds_red_circle_center_and_diameter():
    img = _sky()
    cv2.circle(img, (320, 200), 20, (0, 0, 220), -1)
    dets = detect_balloons(img, RED)
    assert len(dets) == 1
    d = dets[0]
    assert abs(d.cx - 320) < 1.5 and abs(d.cy - 200) < 1.5 and abs(d.diameter_px - 40) < 3


def test_rejects_elongated_shape_and_small_noise():
    img = _sky()
    cv2.rectangle(img, (100, 100), (260, 120), (0, 0, 220), -1)    # 가늘고 긴 빨강
    cv2.circle(img, (400, 300), 2, (0, 0, 220), -1)                # 너무 작음
    assert detect_balloons(img, RED) == []


def test_scale_keeps_original_coordinates():
    img = _sky(1280, 720)
    cv2.circle(img, (900, 400), 30, (0, 0, 220), -1)
    d = detect_balloons(img, RED, scale=0.5)[0]
    assert abs(d.cx - 900) < 3 and abs(d.diameter_px - 60) < 5


def test_hsv_parse_validation():
    import pytest
    with pytest.raises(ValueError):
        parse_hsv_ranges([1, 2, 3])
