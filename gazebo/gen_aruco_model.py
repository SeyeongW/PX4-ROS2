#!/usr/bin/env python3
"""
시뮬레이션용 ArUco 마커 텍스처 PNG 생성 스크립트.

실행 (최초 1회, Gazebo 실행 전):
    python3 gazebo/gen_aruco_model.py

결과:
    gazebo/models/aruco_marker_0/materials/textures/aruco_0.png
"""
import os
import sys

try:
    import cv2
    import cv2.aruco as aruco
    import numpy as np
except ImportError:
    print("ERROR: opencv-python 이 설치되지 않았습니다.")
    print("       pip3 install opencv-python")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_DIR    = os.path.join(SCRIPT_DIR, "models", "aruco_marker_0", "materials", "textures")
os.makedirs(TEX_DIR, exist_ok=True)

# Generate DICT_4X4_50 marker ID=0 at 400×400 px
adict  = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker = aruco.generateImageMarker(adict, 0, 400)

# Add white border (simulates printed marker with quiet zone)
bordered = cv2.copyMakeBorder(marker, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)

out = os.path.join(TEX_DIR, "aruco_0.png")
cv2.imwrite(out, bordered)

print(f"✓ 마커 텍스처 생성 완료: {out}")
print(f"  사이즈: {bordered.shape[1]}×{bordered.shape[0]} px")
print(f"  이제 ./run_sim.sh 로 Gazebo를 실행하면 마커가 월드에 나타납니다.")
