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
    bbox: Tuple[float, float, float, float] = (0., 0., 0., 0.)


def parse_hsv_ranges(flat: Sequence[int]) -> List[HsvRange]:
    """[h0,s0,v0,h1,s1,v1, ...] -> [((h0,s0,v0),(h1,s1,v1)), ...]. 빨강은 색상이 0/180에서 갈라지므로 범위 2개."""
    if len(flat) % 6 != 0 or not flat:
        raise ValueError('hsv_ranges 길이는 6의 배수(범위당 lo3+hi3)여야 함')
    v = [int(x) for x in flat]
    for i in range(0, len(v), 6):
        lo, hi = v[i:i+3], v[i+3:i+6]
        if any(not 0 <= a <= b <= maximum for a, b, maximum in zip(lo, hi, (180, 255, 255))):
            raise ValueError('invalid_hsv_range')
    return [((v[i], v[i + 1], v[i + 2]), (v[i + 3], v[i + 4], v[i + 5])) for i in range(0, len(v), 6)]


def detect_balloons(bgr: np.ndarray, ranges: Sequence[HsvRange], *, min_diameter_px=6.0,
                    max_diameter_px=1000.0, min_circularity=0.6, min_fill=0.55,
                    scale=1.0, morph_px=3) -> List[Detection]:
    """색 마스크 -> 윤곽 -> 원형도/채움비 필터. 좌표와 지름은 원본 해상도 기준으로 돌려준다."""
    if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3
            or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 2
            or not np.isfinite([scale, min_diameter_px, max_diameter_px, min_circularity, min_fill]).all()
            or not 0 < scale <= 1 or not 0 < min_diameter_px <= max_diameter_px
            or not 0 <= min_circularity <= 1 or not 0 <= min_fill <= 1
            or morph_px < 0 or morph_px > 31 or morph_px % 2 == 0 and morph_px != 0):
        raise ValueError('invalid_detector_input')
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    original_mask = mask
    if scale != 1.0:
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    if morph_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_px, morph_px))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Detection] = []
    for c in contours:
        # Morphology/downsampling proposes a region only. Measuring diameter
        # there systematically shrinks a small target and overestimates range.
        x, y, w, h = cv2.boundingRect(c)
        x0, y0 = max(0, int(x/scale)-3), max(0, int(y/scale)-3)
        x1, y1 = min(bgr.shape[1], math.ceil((x+w)/scale)+3), min(bgr.shape[0], math.ceil((y+h)/scale)+3)
        originals, _ = cv2.findContours(original_mask[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not originals:
            continue
        c = max(originals, key=cv2.contourArea) + np.array([[[x0, y0]]])
        x, y, w, h = cv2.boundingRect(c)
        if (x <= x0 or y <= y0 or x+w >= x1 or y+h >= y1
                or x <= 0 or y <= 0 or x+w >= bgr.shape[1] or y+h >= bgr.shape[0] or len(c) < 10):
            continue  # A truncated silhouette cannot provide a known diameter.
        area = cv2.contourArea(c)
        if area < 1.0:
            continue
        center, axes, _ = cv2.fitEllipse(c)
        if min(axes) <= 0 or min(axes)/max(axes) < .8:
            continue
        # Contour coordinates lie at pixel centers; include their pixel support.
        diameter = float(np.mean(axes) + 1.)
        if not (min_diameter_px <= diameter <= max_diameter_px):
            continue
        perim = cv2.arcLength(c, True)
        circ = 4.0 * math.pi * area / (perim * perim) if perim > 0 else 0.0
        (_, _), radius = cv2.minEnclosingCircle(c)
        fill = area / (math.pi * radius * radius) if radius > 0 else 0.0
        if circ < min_circularity or fill < min_fill:
            continue
        out.append(Detection(float(center[0]), float(center[1]), diameter,
                             circ, circ * fill, (float(x), float(y), float(w), float(h))))
    out.sort(key=lambda d: d.score * d.diameter_px, reverse=True)
    return out
