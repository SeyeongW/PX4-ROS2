#!/usr/bin/env python3
"""400 m 육상 트랙(운동장) Gazebo 모델 생성 스크립트.

표준 400 m 트랙(직선 84.39 m × 2 + 반지름 36.8 m 반원 × 2, 둘레 = 400 m)의
중심선을 기준으로, 좌·우 두 개의 흰색 경계선(차선 양쪽 라인)과 그 사이를 채우는
붉은 노면 띠를 만든다. 플랫폼(트럭)은 전방 카메라로 이 두 라인을 보고 그 **사이**
(차선 가운데)로 주행한다 — 한쪽 라인을 밟고 가는 게 아니라 두 라인 사이를 유지.

실행 (최초 1회, Gazebo 실행 전):
    python3 gazebo/gen_track_model.py

결과:
    gazebo/models/running_track/model.sdf
    gazebo/models/running_track/model.config

좌표 규약:
    중심선의 아래쪽 직선이 월드 y = 0 위에 놓이도록 트랙 중심을 (0, R)에 둔다.
    따라서 (3, 0) 부근에서 동쪽(+x)을 보고 출발하는 플랫폼은 곧장 라인 위에 선다.
    한 바퀴: 아래직선(서→동) → 오른쪽 반원 → 위직선(동→서) → 왼쪽 반원 → 닫힘.
"""
import math
import os

# --- 트랙 치수 (표준 400 m) -------------------------------------------------
L = 84.39          # 직선 길이 [m]
R = 36.8           # 반원 반지름 [m]  (2·L + 2πR ≈ 400.0 m)
CX, CY = 0.0, R    # 트랙 중심 → 아래직선이 y=0 위에 놓인다

LANE_HALF     = 1.0    # 중심선 ~ 각 경계선까지의 거리 [m] (차선 폭 = 2·LANE_HALF)
#   차선 폭 2.0 m. 곡선의 코너 컷은 차선을 넓히는 대신 track_follower 가 IPM(탑뷰)
#   미터 평면에서 적합해 횡오차/곡률을 정확히 잡는 방식으로 해결한다.
SURFACE_WIDTH = 2.2    # 두 라인 사이를 채우는 붉은 노면 띠 폭 [m] (라인이 띠 위에 얹힘)
LINE_WIDTH    = 0.15   # 경계선 폭 [m]
SURFACE_Z     = 0.010  # 노면 높이 (지면 z=0 과의 z-fighting 회피)
LINE_Z        = 0.020  # 라인 높이 (노면보다 위)
SEG_H         = 0.005  # 박스 두께 [m]

ARC_SEGS = 48          # 반원당 분할 수 (라인 곡선 매끄러움)

SURFACE_RGBA = (0.72, 0.25, 0.18, 1.0)   # 테라코타(붉은) 트랙 노면
LINE_RGBA    = (0.95, 0.95, 0.95, 1.0)   # 흰색 경계선(좌/우)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "models", "running_track")


def path_segments(d=0.0):
    """중심선에서 안쪽(트랙 중심 방향)으로 d 만큼 평행이동한 스타디움 경로를
    (cx, cy, yaw, length) 박스 세그먼트 리스트로 반환.

    d=0   → 중심선(노면 띠의 중앙)
    d>0   → 안쪽(트랙 중심 y=R 쪽) 경계선  (반원 반지름 R-d)
    d<0   → 바깥쪽 경계선                    (반원 반지름 R+d = R+|d|)

    직선은 1개의 큰 박스로, 반원은 ARC_SEGS 개의 현(chord) 박스로 근사한다.
    각 박스는 중심점에 놓고 경로 접선(heading)만큼 yaw 회전시킨다.
    """
    r = R - d                                   # 이 경로의 반원 반지름
    segs = []

    # 1) 아래 직선: y = d, heading = 0 (동쪽)
    segs.append((CX, d, 0.0, L))

    # 2) 위 직선: y = 2R - d, heading = π (서쪽)
    segs.append((CX, 2 * R - d, math.pi, L))

    # 3) 오른쪽 반원: 중심 (+L/2, R), θ ∈ [-π/2, +π/2], 반지름 r
    # 4) 왼쪽  반원: 중심 (-L/2, R), θ ∈ [+π/2, +3π/2], 반지름 r
    chord = (math.pi * r) / ARC_SEGS * 1.08     # 이 반지름 기준 현 길이, 살짝 겹침
    for sign, theta0 in ((+1, -math.pi / 2), (-1, +math.pi / 2)):
        cx_c = CX + sign * (L / 2)
        for i in range(ARC_SEGS):
            theta = theta0 + (i + 0.5) * (math.pi / ARC_SEGS)
            x = cx_c + r * math.cos(theta)
            y = CY + r * math.sin(theta)
            yaw = theta + math.pi / 2           # 접선 방향
            segs.append((x, y, yaw, chord))
    return segs


def box_visual(name, cx, cy, cz, yaw, length, width, height, rgba):
    r, g, b, a = rgba
    return f"""      <visual name="{name}">
        <pose>{cx:.4f} {cy:.4f} {cz:.4f} 0 0 {yaw:.6f}</pose>
        <geometry><box><size>{length:.4f} {width:.4f} {height:.4f}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>
      </visual>"""


def build_sdf():
    visuals = []
    # 1) 노면(붉은 띠): 중심선(d=0)을 따라 SURFACE_WIDTH 폭으로 깐다
    for n, (cx, cy, yaw, length) in enumerate(path_segments(0.0)):
        visuals.append(box_visual(f"surf_{n}", cx, cy, SURFACE_Z, yaw,
                                  length, SURFACE_WIDTH, SEG_H, SURFACE_RGBA))
    # 2) 두 경계선(안쪽 d=+LANE_HALF, 바깥쪽 d=-LANE_HALF)을 노면 위에 얹는다.
    #    플랫폼은 이 두 라인 사이로 주행한다. (중앙선은 월드에 그리지 않고,
    #    track_follower 가 두 라인을 적합해 '가상' 중앙선을 만들어 추종한다.)
    for side, d in (("in", +LANE_HALF), ("out", -LANE_HALF)):
        for n, (cx, cy, yaw, length) in enumerate(path_segments(d)):
            visuals.append(box_visual(f"line_{side}_{n}", cx, cy, LINE_Z, yaw,
                                      length, LINE_WIDTH, SEG_H, LINE_RGBA))

    body = "\n".join(visuals)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- 자동 생성: gen_track_model.py. 직접 수정하지 말 것. -->
  <!-- 표준 400 m 트랙: 직선 {L} m × 2 + 반지름 {R} m 반원 × 2.
       붉은 노면 띠 위에 좌·우 두 흰색 경계선. 플랫폼 전방 카메라가 두 라인
       사이(차선 중앙)를 유지하며 주행한다. 중심선 아래 직선이 y=0 위에
       놓이도록 중심을 (0,{R})에 둠. 차선 폭 = 2·{LANE_HALF} m. -->
  <model name="running_track">
    <static>true</static>
    <link name="link">
{body}
    </link>
  </model>
</sdf>
"""


def build_config():
    return """<?xml version="1.0"?>
<model>
  <name>running_track</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Auto-generated 400 m athletic track (red surface + white guide line).</description>
</model>
"""


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    sdf_path = os.path.join(MODEL_DIR, "model.sdf")
    cfg_path = os.path.join(MODEL_DIR, "model.config")
    with open(sdf_path, "w") as f:
        f.write(build_sdf())
    with open(cfg_path, "w") as f:
        f.write(build_config())
    perim = 2 * L + 2 * math.pi * R
    print(f"✓ 트랙 모델 생성 완료: {sdf_path}")
    print(f"  중심선 둘레 ≈ {perim:.2f} m (직선 {L}×2 + 반원 R={R})")
    print(f"  차선 폭 {2 * LANE_HALF} m (경계선 2줄), 노면 폭 {SURFACE_WIDTH} m, "
          f"라인 폭 {LINE_WIDTH} m, 반원 분할 {ARC_SEGS}")
    print(f"  이제 run_sim.sh 로 Gazebo를 실행하면 트랙이 월드에 나타납니다.")


if __name__ == "__main__":
    main()
