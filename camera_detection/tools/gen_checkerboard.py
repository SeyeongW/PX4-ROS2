#!/usr/bin/env python3
"""카메라 캘리브레이션용 체커보드(SVG) 생성기.

ROS 2 `camera_calibration cameracalibrator` 규약에 맞춥니다:
  --size  = 내부 코너 개수 (cols × rows)  → 사각형 수 = (cols+1) × (rows+1)
  --square= 사각형 한 변 실측(m)

SVG 는 mm 단위가 고정된 벡터라 100%(실제 크기)로 인쇄하면 칸 크기가 정확합니다.
PNG 로 뽑아 "페이지에 맞춤"으로 인쇄하면 크기가 틀어져 캘리브레이션이 부정확해집니다.

사용:
    python3 gen_checkerboard.py                       # 기본 8x6 코너, 25mm, A4 가로
    python3 gen_checkerboard.py --cols 8 --rows 6 --square-mm 25 --out board.svg

인쇄 후:
    1) 프린터 대화상자에서 배율 100% / "실제 크기"(맞춤 끄기)로 인쇄
    2) 자로 한 칸을 재서 실제 mm 확인 → cameracalibrator 의 --square 를 그 값(m)으로
    3) 평평한 판(폼보드/하드보드)에 구김 없이 부착
"""

import argparse


def generate_svg(cols_corners, rows_corners, square_mm, margin_mm):
    # 내부 코너 개수 → 사각형 개수
    n_cols = cols_corners + 1
    n_rows = rows_corners + 1
    board_w = n_cols * square_mm
    board_h = n_rows * square_mm
    total_w = board_w + 2 * margin_mm
    total_h = board_h + 2 * margin_mm + 12  # 하단 라벨 공간

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w}mm" height="{total_h}mm" '
        f'viewBox="0 0 {total_w} {total_h}">')
    # 흰 배경(quiet zone 포함) — 검출 안정화
    parts.append(f'<rect x="0" y="0" width="{total_w}" height="{total_h}" '
                 f'fill="white"/>')
    # 검은 사각형만 그림(체스판 패턴). (row+col) 짝수 = 검정.
    for r in range(n_rows):
        for c in range(n_cols):
            if (r + c) % 2 == 0:
                x = margin_mm + c * square_mm
                y = margin_mm + r * square_mm
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{square_mm}" '
                    f'height="{square_mm}" fill="black"/>')
    # 얇은 외곽 테두리(자를 때 기준)
    parts.append(
        f'<rect x="{margin_mm}" y="{margin_mm}" width="{board_w}" '
        f'height="{board_h}" fill="none" stroke="black" '
        f'stroke-width="0.2"/>')
    # 라벨: 캘리브레이터 명령 인자
    label = (f'checkerboard  --size {cols_corners}x{rows_corners}  '
             f'--square {square_mm/1000:.3f}   '
             f'({n_cols}x{n_rows} squares, {square_mm}mm each) '
             f'-- PRINT AT 100%, verify with ruler')
    parts.append(
        f'<text x="{margin_mm}" y="{total_h - 4}" '
        f'font-family="monospace" font-size="4" fill="black">{label}</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


def main():
    ap = argparse.ArgumentParser(description='체커보드 SVG 생성')
    ap.add_argument('--cols', type=int, default=8,
                    help='가로 내부 코너 개수 (cameracalibrator --size 의 앞 숫자)')
    ap.add_argument('--rows', type=int, default=6,
                    help='세로 내부 코너 개수 (--size 의 뒤 숫자)')
    ap.add_argument('--square-mm', type=float, default=25.0,
                    help='사각형 한 변(mm)')
    ap.add_argument('--margin-mm', type=float, default=10.0,
                    help='흰 여백(mm)')
    ap.add_argument('--out', default='checkerboard.svg', help='출력 SVG 경로')
    args = ap.parse_args()

    svg = generate_svg(args.cols, args.rows, args.square_mm, args.margin_mm)
    with open(args.out, 'w') as f:
        f.write(svg)

    n_cols, n_rows = args.cols + 1, args.rows + 1
    print(f'wrote {args.out}')
    print(f'  코너 {args.cols}x{args.rows}  사각형 {n_cols}x{n_rows}  '
          f'{args.square_mm}mm/칸')
    print(f'  보드 {n_cols*args.square_mm:.0f}x{n_rows*args.square_mm:.0f}mm '
          f'(+여백 {args.margin_mm}mm)')
    print(f'  캘리브레이터: ros2 run camera_calibration cameracalibrator '
          f'--size {args.cols}x{args.rows} --square {args.square_mm/1000:.3f} '
          f'image:=/down_camera/image_raw camera:=/down_camera')


if __name__ == '__main__':
    main()
