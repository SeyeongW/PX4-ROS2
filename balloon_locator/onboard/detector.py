#!/usr/bin/env python3
"""온보드: 위방향 카메라로 풍선 검출 → (detected, dx_px, dy_px, radius_px) UDP 송신.

색 지정 방법 (--hsv-ranges):
  [lo_H lo_S lo_V hi_H hi_S hi_V] 6개씩, 여러 범위는 붙여서 12개·18개···
  빨강처럼 H=0/180 갈라진 색은 범위 두 쌍(12개) 사용.

  ── OpenCV HSV 값 (H: 0-180, S: 0-255, V: 0-255) ──────────────────────
  빨강   (lo)  0  120  70  (hi) 10 255 255  +  (lo) 170 120 70 (hi) 180 255 255
  주황        10  150  80       25 255 255
  노랑        25  150  80       35 255 255
  연두        35  100  60       65 255 255
  초록        65   80  50       85 255 255
  하늘        85   80  50      100 255 255
  파랑       100   80  50      130 255 255
  남색/보라  130   80  50      160 255 255
  분홍       160   80  50      170 255 255
  흰색         0    0 180      180  30 255   (채도 낮고 밝음)
  ──────────────────────────────────────────────────────────────────────

물리적 풍선 직경(m) 불필요 — 드론 유도에는 픽셀 오프셋만 사용.
최소/최대 반지름(px)은 너무 작은 잡음 / 너무 큰 오검출 제거용.

송신 포맷 (UDP, 4×float32 little-endian 16 바이트):
  [detected(0/1), dx_px, dy_px, radius_px]
  dx_px: 이미지 중심 기준 오른쪽 양수
  dy_px: 이미지 중심 기준 위가 양수 (이미지 y 반전)

수신 측: onboard/offboard_follower.py
"""
import argparse
import socket
import struct
import time

import cv2
import numpy as np

from balloon_detector import detect_balloons, parse_hsv_ranges


def run(args):
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f'카메라 열기 실패: {args.device}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    ranges = parse_hsv_ranges(args.hsv_ranges)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.host, args.port)

    t_log = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        cx0, cy0 = w / 2.0, h / 2.0

        dets = detect_balloons(
            frame, ranges,
            min_diameter_px=args.min_radius * 2,
            max_diameter_px=args.max_radius * 2,
            min_circularity=args.min_circularity,
            min_fill=args.min_fill,
        )

        if dets:
            best = dets[0]                          # score 기준 가장 신뢰도 높은 것
            dx = best.cx - cx0
            dy = -(best.cy - cy0)                   # 이미지 y 아래 양수 → 반전
            r = best.diameter_px / 2.0
            payload = struct.pack('<4f', 1.0, dx, dy, r)

            if args.display:
                cv2.circle(frame, (int(best.cx), int(best.cy)), int(r), (0, 255, 0), 2)
                cv2.putText(frame, f'r={r:.0f}px dx={dx:.0f} dy={dy:.0f}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            payload = struct.pack('<4f', 0.0, 0.0, 0.0, 0.0)

        sock.sendto(payload, dst)

        now = time.monotonic()
        if now - t_log > 2.0:
            if dets:
                print(f'[detector] 검출: r={dets[0].diameter_px/2:.0f}px  '
                      f'dx={dets[0].cx-cx0:.0f}  dy={-(dets[0].cy-cy0):.0f}', flush=True)
            else:
                print('[detector] 미검출', flush=True)
            t_log = now

        if args.display:
            cv2.imshow('balloon detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if args.display:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument('--device', default='/dev/video0')
    ap.add_argument('--width',  type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps',    type=int, default=30)
    ap.add_argument('--host',   default='127.0.0.1')
    ap.add_argument('--port',   type=int, default=15760)

    # 빨강 풍선 기본값 (0-10 + 170-180)
    ap.add_argument(
        '--hsv-ranges', type=int, nargs='+',
        default=[0, 120, 70, 10, 255, 255, 170, 120, 70, 180, 255, 255],
        metavar='INT',
        help='HSV 범위, 6개씩 [lo_H lo_S lo_V hi_H hi_S hi_V ...]. 독스트링 색상표 참고.')

    ap.add_argument('--min-radius',      type=float, default=10.0,
                    help='최소 반지름 [px] — 잡음 제거')
    ap.add_argument('--max-radius',      type=float, default=400.0,
                    help='최대 반지름 [px] — 오검출 제거')
    ap.add_argument('--min-circularity', type=float, default=0.6,
                    help='원형도 하한 [0-1]')
    ap.add_argument('--min-fill',        type=float, default=0.55,
                    help='채움비 하한 [0-1]')
    ap.add_argument('--display', action='store_true',
                    help='디버그 창 (헤드리스 환경에서는 끌 것)')

    run(ap.parse_args())


if __name__ == '__main__':
    main()
