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
                or x <= 0 or y <= 0 or x+w >= bgr.shape[1] or y+h >= bgr.shape[0] or len(c) < 6):
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


_YOLO_CACHE = {}
_YOLO_WARNED = set()


def _ensure_ultralytics_on_path():
    """Ensure user/conda/venv site-packages are on sys.path when launched under /usr/bin/python3."""
    import glob
    import os
    import sys
    from pathlib import Path
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass
    home = str(Path.home())
    patterns = [
        f'{home}/.local/lib/python*/site-packages',
        f'{home}/*venv*/lib/python*/site-packages',
        f'{home}/.*venv*/lib/python*/site-packages',
        f'{home}/miniconda3/envs/*/lib/python*/site-packages',
        f'{home}/miniconda3/lib/python*/site-packages',
        f'{home}/anaconda3/envs/*/lib/python*/site-packages',
        f'{home}/anaconda3/lib/python*/site-packages',
    ]
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        patterns.insert(0, f'{conda_prefix}/lib/python*/site-packages')
    for pat in patterns:
        for sp in sorted(glob.glob(pat), reverse=True):
            if sp not in sys.path:
                sys.path.append(sp)


def _get_yolo_model(model_path: str):
    """Lazily load and cache the Ultralytics YOLO model (e.g., ~/droneD.pt)."""
    from pathlib import Path
    if not model_path:
        return None
    p = Path(model_path).expanduser()
    key = str(p)
    if key in _YOLO_CACHE:
        return _YOLO_CACHE[key]
    if not p.is_file():
        if key not in _YOLO_WARNED:
            print(f'[YOLO] Model file not found at {key}, using silhouette detector', flush=True)
            _YOLO_WARNED.add(key)
        return None
    try:
        _ensure_ultralytics_on_path()
        from ultralytics import YOLO
        model = YOLO(key)
        _YOLO_CACHE[key] = model
        print(f'[YOLO] Loaded trained drone detector model: {key}', flush=True)
        return model
    except Exception as exc:
        if key not in _YOLO_WARNED:
            print(f'[YOLO] Could not load {key} ({exc}), using silhouette detector', flush=True)
            _YOLO_WARNED.add(key)
        return None


def _sky_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of Gazebo / outdoor sky pixels (bright sky / light-blue background)."""
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    return (b > 115) & (g > 115) & ((b - r) >= -10) & ((b - g) >= -15)


def _refine_drone_roi(bgr: np.ndarray, sky: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """Extract sub-pixel horizontal span (w_px) and center (cx, cy) of the drone silhouette inside [x0:x1, y0:y1]."""
    h_img, w_img = bgr.shape[:2]
    x0, y0 = max(1, x0), max(1, y0)
    x1, y1 = min(w_img - 1, x1), min(h_img - 1, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    sub_sky = sky[y0:y1, x0:x1]
    border_sky = np.concatenate([
        sub_sky[0, :], sub_sky[-1, :], sub_sky[:, 0], sub_sky[:, -1]
    ])
    if border_sky.size == 0 or np.mean(border_sky) < 0.70:
        return None
    fg = ~sub_sky
    ys, xs = np.nonzero(fg)
    if xs.size < 1:
        return None
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    # Reject silhouettes clipped by the ROI or image boundary
    if xmin <= 0 or ymin <= 0 or xmax >= (x1 - x0 - 1) or ymax >= (y1 - y0 - 1):
        return None
    gx0, gy0 = x0 + xmin, y0 + ymin
    gx1, gy1 = x0 + xmax, y0 + ymax
    if gx0 <= 1 or gy0 <= 1 or gx1 >= w_img - 2 or gy1 >= h_img - 2:
        return None

    # Sub-pixel edge refinement along sky-to-foreground intensity transition
    roi_gray = np.mean(bgr[y0:y1, x0:x1].astype(np.float32), axis=2)
    sky_vals = roi_gray[sub_sky]
    fg_vals = roi_gray[fg]
    sky_ref = float(np.median(sky_vals)) if sky_vals.size > 0 else 190.0
    fg_ref = float(np.percentile(fg_vals, 25)) if fg_vals.size > 0 else 40.0
    contrast = max(sky_ref - fg_ref, 20.0)

    # Left & right tip fractional sub-pixel extension
    left_rows = ys[xs == xmin]
    right_rows = ys[xs == xmax]
    left_ext = 0.0
    right_ext = 0.0
    if xmin > 0 and left_rows.size > 0:
        darkness_l = np.clip((sky_ref - roi_gray[left_rows, xmin - 1]) / contrast, 0.0, 1.0)
        darkness_c = np.clip((sky_ref - roi_gray[left_rows, xmin]) / contrast, 0.1, 1.0)
        left_ext = float(np.max(darkness_l) + 0.5 * (np.max(darkness_c) - 0.5))
    if xmax + 1 < (x1 - x0) and right_rows.size > 0:
        darkness_r = np.clip((sky_ref - roi_gray[right_rows, xmax + 1]) / contrast, 0.0, 1.0)
        darkness_c = np.clip((sky_ref - roi_gray[right_rows, xmax]) / contrast, 0.1, 1.0)
        right_ext = float(np.max(darkness_r) + 0.5 * (np.max(darkness_c) - 0.5))

    x_left_sub = float(xmin) - np.clip(left_ext, -0.45, 0.95)
    x_right_sub = float(xmax) + np.clip(right_ext, -0.45, 0.95)
    w_px = float(max(1.5, x_right_sub - x_left_sub + 1.0))
    h_px = float(max(1.5, ymax - ymin + 1.0))
    if w_px / max(h_px, 1.0) > 16.0 or h_px / max(w_px, 1.0) > 5.0:
        return None
    cx = float(x0 + 0.5 * (x_left_sub + x_right_sub))
    cy = float(y0 + 0.5 * (ymin + ymax))
    return cx, cy, w_px, h_px, float(gx0), float(gy0)


def detect_drones(
    bgr: np.ndarray,
    *,
    model_path: str = '',
    conf_thres: float = 0.05,
    min_diameter_px: float = 2.0,
    max_diameter_px: float = 1000.0,
) -> List[Detection]:
    """Detect target x500 quadrotor using ~/droneD.pt YOLO model + sub-pixel silhouette span refinement."""
    if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3
            or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 2):
        raise ValueError('invalid_detector_input')

    h_img, w_img = bgr.shape[:2]
    sky = _sky_mask(bgr)
    out: List[Detection] = []

    # 1. Primary detector: user's trained YOLO model (~/droneD.pt)
    model = _get_yolo_model(model_path)
    if model is not None:
        try:
            eff_conf = min(float(conf_thres), 0.05)
            imgsz = 1280 if max(h_img, w_img) >= 960 else 640
            results = model.predict(bgr, imgsz=imgsz, conf=eff_conf, verbose=False)
            boxes_xyxy = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for (bx1, by1, bx2, by2), sc in zip(xyxy, confs):
                    boxes_xyxy.append((float(bx1), float(by1), float(bx2), float(by2), float(sc)))

            # If full-frame missed a small distant drone, run magnified center crop
            if not boxes_xyxy:
                cw = min(w_img, 320 if w_img <= 640 else 480)
                ch = min(h_img, 240 if h_img <= 360 else 360)
                cx0, cy0 = (w_img - cw) // 2, (h_img - ch) // 2
                crop = bgr[cy0:cy0 + ch, cx0:cx0 + cw]
                c_res = model.predict(crop, imgsz=640, conf=eff_conf, verbose=False)
                for r in c_res:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for (bx1, by1, bx2, by2), sc in zip(xyxy, confs):
                        boxes_xyxy.append((
                            float(bx1 + cx0), float(by1 + cy0),
                            float(bx2 + cx0), float(by2 + cy0), float(sc)
                        ))

            for bx1, by1, bx2, by2, sc in boxes_xyxy:
                bw, bh = bx2 - bx1, by2 - by1
                pad = max(8, int(0.45 * max(bw, bh)))
                refined = _refine_drone_roi(
                    bgr, sky, int(bx1) - pad, int(by1) - pad, int(bx2) + pad, int(by2) + pad
                )
                if refined is not None:
                    cx, cy, w_px, h_px, gx0, gy0 = refined
                    if min_diameter_px <= w_px <= max_diameter_px:
                        out.append(Detection(
                            cx=cx, cy=cy, diameter_px=w_px,
                            circularity=0.90, score=max(sc, 0.88),
                            bbox=(gx0, gy0, w_px, h_px),
                        ))
                elif (bx1 > 1 and by1 > 1 and bx2 < w_img - 1 and by2 < h_img - 1
                      and min_diameter_px <= bw <= max_diameter_px):
                    out.append(Detection(
                        cx=0.5 * (bx1 + bx2), cy=0.5 * (by1 + by2), diameter_px=float(bw),
                        circularity=0.85, score=float(sc),
                        bbox=(float(bx1), float(by1), float(bw), float(bh)),
                    ))
        except Exception:
            pass

    if out:
        out.sort(key=lambda d: d.score * d.diameter_px, reverse=True)
        return out

    # 2. Long-range / synthetic sky-contrast silhouette assist (for distant x500 in Gazebo sky)
    if np.mean(sky) < 0.12:
        return out
    fg = (~sky).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    parts = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 1 or w > int(0.65 * w_img) or h > int(0.65 * h_img):
            continue
        if x <= 1 or y <= 1 or (x + w) >= w_img - 1 or (y + h) >= h_img - 1:
            continue
        # Check if this component is above the ground horizon (surrounded by sky below it)
        by_check = min(h_img - 1, y + h + 5)
        if np.mean(sky[y + h:by_check, max(0, x - 3):min(w_img, x + w + 3)]) < 0.65:
            continue
        parts.append((x, y, x + w, y + h, int(area), float(centroids[i][0]), float(centroids[i][1])))

    if not parts:
        return out

    # Group nearby quadrotor sub-parts (body, arms, 4 rotor disks) into a unified x500 cluster
    parts.sort(key=lambda item: item[4], reverse=True)
    used = [False] * len(parts)
    for i, seed in enumerate(parts):
        if used[i]:
            continue
        used[i] = True
        cx0, cy0 = seed[5], seed[6]
        span_est = max(seed[2] - seed[0], seed[3] - seed[1], 14)
        gx0, gy0, gx1, gy1 = seed[0], seed[1], seed[2], seed[3]
        for j, other in enumerate(parts):
            if used[j]:
                continue
            if math.hypot(other[5] - cx0, other[6] - cy0) <= max(32.0, 2.8 * span_est):
                used[j] = True
                gx0, gy0 = min(gx0, other[0]), min(gy0, other[1])
                gx1, gy1 = max(gx1, other[2]), max(gy1, other[3])

        pad = max(6, int(0.35 * max(gx1 - gx0, gy1 - gy0, 6)))
        refined = _refine_drone_roi(bgr, sky, gx0 - pad, gy0 - pad, gx1 + pad, gy1 + pad)
        if refined is not None:
            cx, cy, w_px, h_px, rx0, ry0 = refined
            if min_diameter_px <= w_px <= max_diameter_px:
                out.append(Detection(
                    cx=cx, cy=cy, diameter_px=w_px,
                    circularity=0.90, score=0.93,
                    bbox=(rx0, ry0, w_px, h_px),
                ))
        else:
            w_px = float(max(2.0, gx1 - gx0))
            h_px = float(max(2.0, gy1 - gy0))
            if min_diameter_px <= w_px <= max_diameter_px:
                out.append(Detection(
                    cx=0.5 * (gx0 + gx1), cy=0.5 * (gy0 + gy1), diameter_px=w_px,
                    circularity=0.88, score=0.90,
                    bbox=(float(gx0), float(gy0), w_px, h_px),
                ))

    out.sort(key=lambda d: d.score * d.diameter_px, reverse=True)
    return out

