#!/usr/bin/env python3
"""
시뮬레이션용 ArUco 마커 텍스처 PNG 생성 스크립트.

실행 (최초 1회, Gazebo 실행 전):
    python3 gazebo/gen_aruco_model.py

결과:
    gazebo/models/aruco_marker_0/materials/textures/aruco_0.png   (레거시 평지 마커용, 단일 ID=0)
    gazebo/models/aruco_platform/materials/textures/aruco_{0,1,2}.png
        aruco_platform 위 3중 동심원 마커(model.sdf의 marker_vis_large/medium/small)용.
        ID/물리적 크기는 model.sdf에서 결정되고, 여기서는 텍스처(픽셀 이미지)만 생성한다.
"""
import os
import sys

try:
    import cv2
    import cv2.aruco as aruco
except ImportError:
    print("ERROR: opencv-python 이 설치되지 않았습니다.")
    print("       pip3 install opencv-python")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
adict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


def _draw_marker(marker_id: int, px: int):
    # OpenCV >= 4.7 renamed drawMarker -> generateImageMarker.
    if hasattr(aruco, "generateImageMarker"):
        return aruco.generateImageMarker(adict, marker_id, px)
    return aruco.drawMarker(adict, marker_id, px)


def gen_marker(marker_id: int, out_path: str, px: int = 400, border: int = 60):
    marker = _draw_marker(marker_id, px)
    # White border simulates a printed marker's quiet zone.
    bordered = cv2.copyMakeBorder(marker, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=255)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, bordered)
    print(f"✓ {out_path}  ({bordered.shape[1]}×{bordered.shape[0]} px, ID={marker_id})")


# Legacy single flat-ground marker (model://aruco_marker_0).
gen_marker(0, os.path.join(SCRIPT_DIR, "models", "aruco_marker_0",
                            "materials", "textures", "aruco_0.png"))

# Multi-size concentric markers on the moving platform (model://aruco_platform).
# IDs must match model.sdf's marker_vis_large/medium/small albedo_map paths.
PLATFORM_TEX_DIR = os.path.join(SCRIPT_DIR, "models", "aruco_platform", "materials", "textures")
for marker_id in (0, 1, 2):
    gen_marker(marker_id, os.path.join(PLATFORM_TEX_DIR, f"aruco_{marker_id}.png"))

print("이제 ./run_sim.sh 로 Gazebo를 실행하면 마커가 월드에 나타납니다.")
