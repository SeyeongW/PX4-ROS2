"""Is `_landing_point` biased when the camera is tilted? Synthetic, exact truth.

The SITL runs report the two flanking markers disagreeing about the deck centre
by 0.35 m on average and 1.35 m at worst, which lands straight on the descent
target as a bias.  The detector docstring justifies the homography path against
`t + R@o` on a synthetic sweep, but that sweep evidently did not cover the case
that bites: the homography places the landing point exactly in the IMAGE and
then back-projects it at the MARKER CENTRE's depth.  Those two depths are only
equal when the deck is perpendicular to the optical axis.  The gimbal aims at
the target, not at nadir, so during a chase they are not.

Build the exact geometry, project the corners, and measure.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

FX = FY = 388.5
CX, CY = 400.0, 300.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
MARKERS = {0: (1.3, np.array([-1.1, 0.0])), 2: (1.3, np.array([1.1, 0.0]))}


def corners_world(size, centre):
    """Marker corners on the deck (z=0), TL,TR,BR,BL as ArUco orders them."""
    s = size / 2.0
    return np.array([[centre[0] - s, centre[1] + s, 0.0],
                     [centre[0] + s, centre[1] + s, 0.0],
                     [centre[0] + s, centre[1] - s, 0.0],
                     [centre[0] - s, centre[1] - s, 0.0]])


def cam_from_deck(tilt_deg, height, offset):
    """Camera pose over a level deck: `height` up, `offset` along +x, tilted.

    Returns (R_cw, t_cw) mapping deck coords -> camera optical coords
    (+X right, +Y down, +Z forward).
    """
    # nadir: optical +Z = deck -z, +X = deck +x, +Y = deck -y
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # Tilt about the optical Y axis, i.e. IN the plane the markers are offset
    # along.  Tilting about X instead leaves both markers at the same depth and
    # the whole effect under test vanishes — which is exactly how a sweep can
    # report 0.000 everywhere while the real thing is off by a third of a metre.
    th = math.radians(tilt_deg)
    Rt = np.array([[math.cos(th), 0.0, math.sin(th)],
                   [0.0, 1.0, 0.0],
                   [-math.sin(th), 0.0, math.cos(th)]])
    R_cw = Rt @ R0
    C = np.array([offset[0], offset[1], height])    # camera centre, deck coords
    return R_cw, -R_cw @ C


def project(P, R_cw, t_cw):
    Pc = (R_cw @ P.T).T + t_cw
    uv = (K @ Pc.T).T
    return uv[:, :2] / uv[:, 2:3], Pc


def landing_point_current(objp, corners, t, o):
    """What aruco_detector_node._landing_point does today."""
    Hm = cv2.getPerspectiveTransform(objp[:, :2].astype(np.float32),
                                     corners.astype(np.float32))
    p = Hm @ np.array([o[0], o[1], 1.0])
    u, v = p[0] / p[2], p[1] / p[2]
    z = float(t[2])                                  # <-- marker-centre depth
    return np.array([(u - CX) / FX * z, (v - CY) / FY * z, z])


def landing_point_plane(objp, corners, t, o):
    """Same image point, but back-projected onto the MARKER'S OWN PLANE.

    The plane is recovered from the homography itself — no rotation matrix, so
    the planar-PnP mirror ambiguity that `t + R@o` trips over cannot reach it.
    K^-1 H = [r1 r2 t]/lambda with |r1| = |r2| = 1 fixes lambda, and r3 = r1 x r2
    is the plane normal.  Then intersect the landing point's ray with that plane.
    """
    Hm = cv2.getPerspectiveTransform(objp[:, :2].astype(np.float32),
                                     corners.astype(np.float32))
    M = np.linalg.inv(K) @ Hm
    lam = (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1])) / 2.0
    M = M / lam
    if M[2, 2] < 0:                                  # marker must be in front
        M = -M
    r1, r2, t_h = M[:, 0], M[:, 1], M[:, 2]
    n = np.cross(r1, r2)
    p = Hm @ np.array([o[0], o[1], 1.0])
    d = np.linalg.inv(K) @ np.array([p[0] / p[2], p[1] / p[2], 1.0])
    denom = float(n @ d)
    if abs(denom) < 1e-9:
        return landing_point_current(objp, corners, t, o)
    return d * (float(n @ t_h) / denom)


print(f'{"tilt":>5} {"h":>5} {"off":>5} | {"current: id0":>13} {"id2":>7} '
      f'{"disagree":>9} | {"plane: id0":>11} {"id2":>7} {"disagree":>9}')
print('-' * 82)
for tilt in (0.0, 10.0, 20.0, 30.0):
    for h, off in ((12.0, 3.0), (6.0, 1.5), (4.0, 1.0), (3.0, 0.5)):
        R_cw, t_cw = cam_from_deck(tilt, h, np.array([off, 0.0]))
        truth_cam = (R_cw @ np.array([0.0, 0.0, 0.0])) + t_cw   # deck centre
        cur, pln = {}, {}
        for mid, (size, centre) in MARKERS.items():
            W = corners_world(size, centre)
            uv, Pc = project(W, R_cw, t_cw)
            if uv.min() < 0 or uv[:, 0].max() > 800 or uv[:, 1].max() > 600:
                break                                # out of frame, skip case
            ok, rvec, tv = cv2.solvePnP(
                np.float32([[-size / 2, size / 2, 0], [size / 2, size / 2, 0],
                            [size / 2, -size / 2, 0], [-size / 2, -size / 2, 0]]),
                uv.astype(np.float32), K, np.zeros(5),
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            objp = np.float32([[-size / 2, size / 2, 0], [size / 2, size / 2, 0],
                               [size / 2, -size / 2, 0], [-size / 2, -size / 2, 0]])
            o = -centre                              # marker centre -> deck centre
            cur[mid] = landing_point_current(objp, uv, tv.reshape(3), o)
            pln[mid] = landing_point_plane(objp, uv, tv.reshape(3), o)
        if len(cur) < 2:
            continue
        ec = {m: np.linalg.norm(cur[m] - truth_cam) for m in cur}
        ep = {m: np.linalg.norm(pln[m] - truth_cam) for m in pln}
        dc = np.linalg.norm(cur[0] - cur[2])
        dp = np.linalg.norm(pln[0] - pln[2])
        print(f'{tilt:5.0f} {h:5.1f} {off:5.1f} | {ec[0]:13.3f} {ec[2]:7.3f} '
              f'{dc:9.3f} | {ep[0]:11.3f} {ep[2]:7.3f} {dp:9.3f}')
