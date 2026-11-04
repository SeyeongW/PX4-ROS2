import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

HsvRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


@dataclass
class Detection:
    cx: float
    cy: float
    diameter_px: float
    circularity: float
    score: float


def parse_hsv_ranges(flat: Sequence[int]) -> List[HsvRange]:
    """[h0,s0,v0,h1,s1,v1, ...] -> [((h0,s0,v0),(h1,s1,v1)), ...]. 빨강은 색상이 0/180에서 갈라지므로 범위 2개."""
    if len(flat) % 6 != 0 or not flat:
        raise ValueError('hsv_ranges 길이는 6의 배수(범위당 lo3+hi3)여야 함')
    v = [int(x) for x in flat]
    return [((v[i], v[i + 1], v[i + 2]), (v[i + 3], v[i + 4], v[i + 5])) for i in range(0, len(v), 6)]


def detect_balloons(bgr: np.ndarray, ranges: Sequence[HsvRange], *, min_diameter_px=6.0,
                    max_diameter_px=1000.0, min_circularity=0.6, min_fill=0.55,
                    scale=1.0, morph_px=3) -> List[Detection]:
    """색 마스크 -> 윤곽 -> 원형도/채움비 필터. 좌표와 지름은 원본 해상도 기준으로 돌려준다."""
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else bgr
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if morph_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_px, morph_px))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Detection] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1.0:
            continue
        diameter = math.sqrt(4.0 * area / math.pi) / scale
        if not (min_diameter_px <= diameter <= max_diameter_px):
            continue
        perim = cv2.arcLength(c, True)
        circ = 4.0 * math.pi * area / (perim * perim) if perim > 0 else 0.0
        (_, _), radius = cv2.minEnclosingCircle(c)
        fill = area / (math.pi * radius * radius) if radius > 0 else 0.0
        if circ < min_circularity or fill < min_fill:
            continue
        m = cv2.moments(c)
        if m['m00'] <= 0:
            continue
        out.append(Detection(m['m10'] / m['m00'] / scale, m['m01'] / m['m00'] / scale,
                             diameter, circ, circ * fill))
    out.sort(key=lambda d: d.score * d.diameter_px, reverse=True)
    return out
