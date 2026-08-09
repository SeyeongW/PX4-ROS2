"""ENU (ROS2) <-> NED (PX4) conversion (§4 of the spec).

PX4 setpoints are NED; ROS2 works in ENU.  The map swaps x/y and flips z, and
is its own inverse.  Sign mistakes here are a common bug source, so this is the
one place the conversion lives and it is unit-tested (run ``python3 -m
landing_mpc.frame``).

    NED_x =  ENU_y   (North  = East-North's North)
    NED_y =  ENU_x   (East)
    NED_z = -ENU_z   (Down   = -Up)
"""

from __future__ import annotations

import math

import numpy as np


def enu_to_ned(v):
    """Map an ENU 3-vector (position/velocity/accel) to NED. Self-inverse."""
    v = np.asarray(v, float)
    return np.array([v[1], v[0], -v[2]])


# NED->ENU is the same swap-and-flip.
ned_to_enu = enu_to_ned


# --------------------------------------------------------------------------
# Down-camera chain: marker in the camera optical frame -> world ENU offset.
#
# solvePnP returns the marker in the CAMERA OPTICAL frame (REP-103: +X right,
# +Y down, +Z forward).  The camera is pitched +90 deg so it looks straight
# down, giving, for an optical point (u, v, w):
#
#     p_FLU = (-v, -u, -w)          optical -> body FLU (forward-left-up)   (1)
#
# The drone attitude then rotates body into the world.  PX4 publishes q as
# FRD->NED, so with D = diag(1,-1,-1) (FLU->FRD) and S mapping NED->ENU:
#
#     p_FRD = D p_FLU                                                       (2)
#     p_NED = R(q) p_FRD                                                    (3)
#     p_ENU = S p_NED,   S = [[0,1,0],[1,0,0],[0,0,-1]]                     (4)
#
# Skipping (3) — ignoring attitude — makes the marker swing whenever the
# airframe tilts, which a landing controller must never see.
# --------------------------------------------------------------------------
_S_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_D_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def quat_to_rot(q):
    """Rotation matrix from a PX4 quaternion [w, x, y, z] (FRD body -> NED)."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def optical_to_flu(p_opt):
    """Eq. (1): down-looking camera optical (right, down, forward) -> body FLU."""
    u, v, w = (float(c) for c in p_opt)
    return np.array([-v, -u, -w])


def camera_offset_to_enu(p_opt, q):
    """Eqs. (1)-(4): marker offset in the camera -> offset in world ENU."""
    return _S_NED_ENU @ (quat_to_rot(q) @ (_D_FLU_FRD @ optical_to_flu(p_opt)))


def flu_dir_to_enu(d_flu, q):
    """Eqs. (2)-(4) on a direction: body FLU -> world ENU."""
    d = np.asarray(d_flu, float)
    return _S_NED_ENU @ (quat_to_rot(q) @ (_D_FLU_FRD @ d))


def enu_dir_to_flu(d_enu, q):
    """Inverse of (2)-(4): a world-ENU direction -> body FLU.

    Every matrix in the forward chain is its own inverse (S swaps x/y and flips
    z; D = diag(1,-1,-1)) and R(q) is orthogonal, so the inverse is just the
    same chain read backwards with a transpose in the middle.  This is what
    turns "look at that point over there" into something the gimbal can aim.
    """
    d = np.asarray(d_enu, float)
    return _D_FLU_FRD @ (quat_to_rot(q).T @ (_S_NED_ENU @ d))


# --------------------------------------------------------------------------
# Gimbal chain.
#
# The camera hangs on a yaw/roll/pitch joint chain instead of being bolted to
# the airframe, so the two questions change:
#
#   (a) pointing  — which joint angles aim the lens at a given direction?
#   (b) perception — the camera attitude is no longer the airframe attitude,
#       so `camera_offset_to_enu` above must NOT be used on gimbal images.
#
# (a) Joint chain, all angles as the JointPositionController commands them.
# gimbal_down declares yaw about (0,0,-1), roll about (-1,0,0), pitch about
# (0,1,0), and the whole gimbal is mounted with a 3.14 rad yaw that the camera
# sensor's own 3.14 rad yaw cancels.  Composing those, with roll held at 0 (a
# 2-DOF aim needs no third axis; roll would only spin the image):
#
#     d_FLU = ( cos p cos y,  -cos p sin y,  sin p )                       (5)
#
# which has two equivalent downward-pointing solutions
#
#     p0 = asin(d_z),          y0 = atan2(-d_y, d_x)                      (6)
#     p1 = -pi - p0,           y1 = y0 + pi                              (7)
#
# Nadir (d = (0,0,-1)) gives p = -90 deg, inside the joint's -135 deg limit,
# and leaves y undefined — that is real gimbal lock, not a bug, so the caller
# holds the previous yaw there (see `gimbal_joint_angles`).
#
# (b) The camera-mounted IMU reports the LENS's own world attitude, which is
# exactly the quantity the airframe no longer supplies.  Its sensor frame is
# Gazebo's (+X along the optical axis, +Y left, +Z up), so an optical point
# (u, v, w) is (w, -u, -v) there, and one rotation finishes the job.
# --------------------------------------------------------------------------
_PITCH_MIN, _PITCH_MAX = -2.35619, 0.7854          # gimbal_down joint limits


def gimbal_joint_angles(d_flu, yaw_hold=0.0, lock_tol=0.05,
                        pitch_hold=-np.pi / 2.0):
    """Eqs. (6)-(7): body-FLU direction -> continuous gimbal commands.

    ``yaw_hold`` is returned unchanged when the aim is within ``lock_tol`` of
    straight down, where yaw is geometrically undefined; otherwise a hovering
    gimbal would spin on detector noise for no optical gain.  Of the exact
    joint-space solutions allowed by the pitch limits, the one nearest the
    current ``yaw_hold``/``pitch_hold`` is used to avoid a 180-degree yaw jump
    when the optical axis crosses nadir.
    """
    d = np.asarray(d_flu, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return float(yaw_hold), 0.0, -np.pi / 2.0
    d = d / n
    pitch0 = float(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    yaw0 = float(np.arctan2(-d[1], d[0]))
    exact = ((yaw0, pitch0), (yaw0 + np.pi, -np.pi - pitch0))
    candidates = []
    for yaw, pitch in exact:
        if _PITCH_MIN <= pitch <= _PITCH_MAX:
            yaw = float(yaw_hold) + (yaw - float(yaw_hold) + np.pi) % (
                2.0 * np.pi) - np.pi
            cost = max(abs(yaw - float(yaw_hold)),
                       abs(pitch - float(pitch_hold)))
            candidates.append((cost, yaw, pitch))
    if candidates:
        _, yaw, pitch = min(candidates, key=lambda item: item[0])
    else:
        yaw = float(yaw_hold) + (yaw0 - float(yaw_hold) + np.pi) % (
            2.0 * np.pi) - np.pi
        pitch = float(np.clip(pitch0, _PITCH_MIN, _PITCH_MAX))
    horizontal = float(np.hypot(d[0], d[1]))
    if horizontal < lock_tol:
        yaw = float(yaw_hold)
    return yaw, 0.0, pitch


def gimbal_axis_flu(yaw, pitch):
    """Eq. (5): joint commands -> the optical axis they aim, in body FLU."""
    return np.array([np.cos(pitch) * np.cos(yaw),
                     -np.cos(pitch) * np.sin(yaw),
                     np.sin(pitch)])


def optical_to_gz_camera(p_opt):
    """Optical (right, down, forward) -> Gazebo camera frame (fwd, left, up)."""
    u, v, w = (float(c) for c in p_opt)
    return np.array([w, -u, -v])


def gimbal_camera_offset_to_enu(p_opt, q_cam):
    """Marker offset in the gimbal camera -> ENU, via the camera's own attitude.

    ``q_cam`` is the camera IMU's [w, x, y, z], i.e. camera frame -> world ENU.

    CAUTION: in SITL the camera IMU reports against GAZEBO's world, which is
    not PX4's local frame — measured 27.3 deg apart, because the world's
    magnetic_field already encodes a declination that PX4's EKF applies again.
    Mixing this result with a PX4 local position is therefore wrong.  Use
    `gimbal_joint_offset_to_enu` for anything that touches PX4 coordinates;
    this one is for working in Gazebo's own frame (e.g. checking against truth).
    """
    return quat_to_rot(q_cam) @ optical_to_gz_camera(p_opt)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def gimbal_camera_rotation(yaw, roll, pitch):
    """Joint angles -> rotation taking a CAMERA-frame vector into body FLU.

    Reading the chain outward from the airframe: the gimbal is bolted on with a
    3.14 rad mount yaw, then the three joints turn about (0,0,-1), (-1,0,0) and
    (0,1,0) as gimbal_down declares them, and finally the camera sensor carries
    its own 3.14 rad yaw inside the last link:

        R = Rz(pi - yaw) Rx(-roll) Ry(pitch) Rz(pi)

    Applying it to the camera's +X reproduces `gimbal_axis_flu` exactly, which
    is what the self-test checks.
    """
    return (_rot_z(np.pi - float(yaw)) @ _rot_x(-float(roll))
            @ _rot_y(float(pitch)) @ _rot_z(np.pi))


def rot_to_quat(R):
    """Rotation matrix -> [w, x, y, z], via the largest-diagonal branch."""
    R = np.asarray(R, float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


# --------------------------------------------------------------------------
# TF chain for the gimbal, so RViz (and anything else that speaks tf2) can see
# where the lens is actually looking.
#
# Every gimbal link shares the gimbal model origin, so at zero joint angles the
# link frames coincide and a revolute joint is just "translate to the anchor,
# rotate, translate back".  The anchors are the joints' own <pose> values in
# gimbal_down, and the mount offset is where x500_gimbal_rgbd_lidar hangs the
# gimbal: model (-0.0412, 0, 0.272) with base_link at model z 0.24.
#
# The leaf is `gimbal_camera_optical_frame`, which is the name the camera
# sensor already advertises via <optical_frame_id>, and the last hop uses the
# same (-pi/2, 0, -pi/2) convention the fixed cameras use in
# sensor_bridge.launch.py, so both cameras look identical to a tf2 consumer.
# --------------------------------------------------------------------------
GIMBAL_MOUNT_XYZ = (-0.0412, 0.0, 0.032)        # base_link -> gimbal origin
_ANCHOR_YAW = np.array([-0.02552, 0.0, 0.0])
_ANCHOR_ROLL = np.array([0.0, 0.0, -0.1619])
_ANCHOR_PITCH = np.array([-0.0412, 0.0, -0.162])
_SENSOR_XYZ = np.array([-0.0412, 0.0, -0.162])


def _revolute(anchor, R):
    """Link-to-link transform for a joint whose frames coincide at zero."""
    return R, anchor - R @ anchor


def gimbal_tf_chain(yaw, roll, pitch):
    """Joint angles -> [(parent, child, xyz, quat[w,x,y,z]), ...].

    Composing the whole chain reproduces `gimbal_camera_rotation` exactly; the
    self-test asserts it, so the tf2 tree and the analytic perception maths can
    never silently disagree.
    """
    R_mount = _rot_z(np.pi)
    R_yaw, t_yaw = _revolute(_ANCHOR_YAW, _rot_z(-float(yaw)))
    R_roll, t_roll = _revolute(_ANCHOR_ROLL, _rot_x(-float(roll)))
    R_pitch, t_pitch = _revolute(_ANCHOR_PITCH, _rot_y(float(pitch)))
    # The camera SENSOR sits inside the last link at its own pose, and its pi
    # yaw is what cancels the mount's.  Fold that hop in here so that
    # `gimbal_camera_link` is the sensor frame itself (+X along the optical
    # axis), matching how sensor_bridge names the fixed cameras.
    R_cam = R_pitch @ _rot_z(np.pi)
    t_cam = t_pitch + R_pitch @ _SENSOR_XYZ
    # gz camera link -> REP-103 optical axes, same convention as the fixed cams
    R_optical = _rot_z(-np.pi / 2) @ _rot_x(-np.pi / 2)
    return [
        ('base_link', 'cgo3_mount_link',
         np.array(GIMBAL_MOUNT_XYZ), rot_to_quat(R_mount)),
        ('cgo3_mount_link', 'cgo3_vertical_arm_link', t_yaw, rot_to_quat(R_yaw)),
        ('cgo3_vertical_arm_link', 'cgo3_horizontal_arm_link', t_roll,
         rot_to_quat(R_roll)),
        ('cgo3_horizontal_arm_link', 'gimbal_camera_link', t_cam,
         rot_to_quat(R_cam)),
        ('gimbal_camera_link', 'gimbal_camera_optical_frame',
         np.zeros(3), rot_to_quat(R_optical)),
    ]


def gimbal_joint_offset_to_enu(p_opt, joints, q):
    """Marker offset in the gimbal camera -> local ENU, staying in PX4's frame.

    ``joints`` is (yaw, roll, pitch) as the joint encoders report them and
    ``q`` is PX4's attitude quaternion.  The joint rotation is body-relative,
    so the only world reference used is PX4's own — no Gazebo world frame
    enters, and the declination offset that afflicts the camera IMU cannot
    apply.  This is also the real-hardware chain: encoders plus FC attitude.
    """
    yaw, roll, pitch = joints
    p_flu = gimbal_camera_rotation(yaw, roll, pitch) @ optical_to_gz_camera(p_opt)
    return flu_dir_to_enu(p_flu, q)


def _selftest():
    assert np.allclose(enu_to_ned([1, 2, 3]), [2, 1, -3])
    assert np.allclose(ned_to_enu(enu_to_ned([1, 2, 3])), [1, 2, 3])
    # a purely-East ENU velocity is purely-East (+y) in NED
    assert np.allclose(enu_to_ned([5, 0, 0]), [0, 5, 0])
    # climbing (+z ENU) is negative Down in NED
    assert np.allclose(enu_to_ned([0, 0, 1]), [0, 0, -1])

    # --- down-camera chain ---
    q_level = [1.0, 0.0, 0.0, 0.0]                    # level, heading north
    # marker straight below at 5 m: optical (0,0,5) -> 5 m DOWN, no horizontal
    assert np.allclose(camera_offset_to_enu([0, 0, 5], q_level), [0, 0, -5])
    # 1 m toward the image top = 1 m FORWARD (north) when level
    assert np.allclose(camera_offset_to_enu([0, -1, 5], q_level), [0, 1, -5])
    # 1 m to the image right = 1 m to the drone's RIGHT = east when level
    assert np.allclose(camera_offset_to_enu([1, 0, 5], q_level), [1, 0, -5])
    # yaw +90 deg (nose east): what was forward must now read east
    s = np.sqrt(0.5)
    assert np.allclose(camera_offset_to_enu([0, -1, 5], [s, 0, 0, s]),
                       [1, 0, -5], atol=1e-6)
    # a pure roll/pitch must preserve range (rotations are isometries)
    for q in (q_level, [s, 0, 0, s], [0.9239, 0.0, 0.3827, 0.0]):
        e = camera_offset_to_enu([0.3, -0.4, 5.0], q)
        assert abs(np.linalg.norm(e) - np.linalg.norm([0.3, -0.4, 5.0])) < 1e-6

    # --- ENU direction -> body FLU (inverse of the same chain) ---
    # level: "down" in the world is "down" for the body too
    assert np.allclose(enu_dir_to_flu([0, 0, -1], q_level), [0, 0, -1])
    # level, nose north: east in the world is the body's right, i.e. -y in FLU
    assert np.allclose(enu_dir_to_flu([1, 0, 0], q_level), [0, -1, 0])
    assert np.allclose(enu_dir_to_flu([0, 1, 0], q_level), [1, 0, 0])
    # nose east: world east is now straight ahead
    assert np.allclose(enu_dir_to_flu([1, 0, 0], [s, 0, 0, s]), [1, 0, 0],
                       atol=1e-6)
    # it must undo the forward chain exactly, at any attitude
    for q in (q_level, [s, 0, 0, s], [0.9239, 0.0, 0.3827, 0.0]):
        flu = np.array([0.2, -0.5, -3.0])
        assert np.allclose(enu_dir_to_flu(flu_dir_to_enu(flu, q), q), flu,
                           atol=1e-9)

    # --- gimbal pointing: eq. (6) must invert eq. (5) ---
    # nadir is pitch -90 deg, and yaw is undefined there so it is held
    yaw, roll, pitch = gimbal_joint_angles([0, 0, -1], yaw_hold=0.37)
    assert roll == 0.0 and abs(pitch + np.pi / 2) < 1e-9 and yaw == 0.37
    # level camera, marker ahead: pitch 0, yaw 0
    yaw, _, pitch = gimbal_joint_angles([1, 0, 0])
    assert abs(yaw) < 1e-9 and abs(pitch) < 1e-9
    # a target to the body's LEFT (+y FLU) needs a NEGATIVE yaw command,
    # because the joint axis is (0,0,-1)
    yaw, _, _ = gimbal_joint_angles([1, 1, 0])
    assert abs(yaw + np.pi / 4) < 1e-9
    for d in ([1, 0, 0], [1, 1, -1], [0.3, -0.8, -4.0], [-1, 0.2, -2.0]):
        d = np.asarray(d, float) / np.linalg.norm(d)
        yaw, _, pitch = gimbal_joint_angles(d)
        assert np.allclose(gimbal_axis_flu(yaw, pitch), d, atol=1e-9), d
    # the pitch limit is respected even when asked to look up steeply
    _, _, pitch = gimbal_joint_angles([0, 0, 1])
    assert abs(pitch - 0.7854) < 1e-9

    # --- gimbal perception chain (camera IMU, no airframe attitude) ---
    # Lens pointing straight down, camera "up" (+Z_cam) toward the east:
    # +90 deg about +Y sends X_cam -> world down and Z_cam -> world east,
    # leaving Y_cam (the camera's left) pointing north.
    q_down = [s, 0.0, s, 0.0]
    # the optical axis is X_cam, so a marker 5 m ahead of the lens is 5 m down
    assert np.allclose(gimbal_camera_offset_to_enu([0, 0, 5], q_down),
                       [0, 0, -5], atol=1e-6)
    # image right (+u) is -Y_cam, i.e. SOUTH for this attitude
    assert np.allclose(gimbal_camera_offset_to_enu([1, 0, 5], q_down),
                       [0, -1, -5], atol=1e-6)
    # and it is a rotation, so range is preserved at any camera attitude
    for q in (q_down, q_level, [0.9239, 0.0, 0.3827, 0.0]):
        e = gimbal_camera_offset_to_enu([0.3, -0.4, 5.0], q)
        assert abs(np.linalg.norm(e) - np.linalg.norm([0.3, -0.4, 5.0])) < 1e-6

    # --- gimbal joint chain: the frame-safe perception path ---
    # The full rotation must agree with the pointing formula it was derived
    # alongside: the camera's +X is the optical axis, so R @ x_hat is eq. (5).
    for yaw, roll, pitch in [(0.0, 0.0, -np.pi / 2), (0.4, 0.0, -1.2),
                             (-0.9, 0.2, 0.3), (2.0, -0.1, -0.7)]:
        R = gimbal_camera_rotation(yaw, roll, 0.0 + pitch)
        if abs(roll) < 1e-12:
            assert np.allclose(R @ [1, 0, 0], gimbal_axis_flu(yaw, pitch),
                               atol=1e-9), (yaw, pitch)
        # and it must stay a rotation
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
    # Nadir, level airframe: a marker 5 m along the optical axis is 5 m below,
    # and the ENU result must not depend on the gimbal's yaw at all.
    for yaw in (0.0, 0.7, -2.1):
        e = gimbal_joint_offset_to_enu([0, 0, 5], (yaw, 0.0, -np.pi / 2), q_level)
        assert np.allclose(e, [0, 0, -5], atol=1e-9), yaw
    # A tilted airframe must NOT move the answer when the joints hold nadir in
    # the body — that is the coupling the gimbal exists to break, and here it
    # is the caller's job to command the compensating joint angle:
    # --- tf2 chain must reproduce the analytic rotation, exactly ---
    for yaw, roll, pitch in [(0.0, 0.0, -np.pi / 2), (0.4, 0.0, -1.2),
                             (-0.9, 0.2, 0.3), (2.0, -0.1, -0.7)]:
        R_chain, t_chain = np.eye(3), np.zeros(3)
        for _parent, child, xyz, quat in gimbal_tf_chain(yaw, roll, pitch):
            if child == 'gimbal_camera_optical_frame':
                break                       # stop at the sensor frame
            t_chain = t_chain + R_chain @ np.asarray(xyz, float)
            R_chain = R_chain @ quat_to_rot(quat)
        assert np.allclose(R_chain, gimbal_camera_rotation(yaw, roll, pitch),
                           atol=1e-9), (yaw, roll, pitch)
        # rot_to_quat must round-trip every hop it produced
        for _p, _c, _xyz, quat in gimbal_tf_chain(yaw, roll, pitch):
            assert abs(np.linalg.norm(quat) - 1.0) < 1e-9
    # At zero joints the lens sits on the centreline 0.13 m below base_link —
    # the geometry the vehicle model was designed around.
    R0, t0 = np.eye(3), np.zeros(3)
    for _parent, child, xyz, quat in gimbal_tf_chain(0.0, 0.0, 0.0):
        if child == 'gimbal_camera_optical_frame':
            break
        t0 = t0 + R0 @ np.asarray(xyz, float)
        R0 = R0 @ quat_to_rot(quat)
    assert np.allclose(t0, [0.0, 0.0, -0.13], atol=1e-6), t0
    # the optical leaf must agree with optical_to_gz_camera
    *_, (_p, _c, _xyz, q_opt) = gimbal_tf_chain(0.0, 0.0, 0.0)
    for p_opt in ([0, 0, 1], [1, 0, 0], [0.3, -0.4, 5.0]):
        assert np.allclose(quat_to_rot(q_opt) @ p_opt,
                           optical_to_gz_camera(p_opt), atol=1e-9)

    pitched = [0.9239, 0.0, 0.3827, 0.0]              # 45 deg airframe pitch
    d_flu = enu_dir_to_flu([0, 0, -1], pitched)       # where "down" now is
    yaw_c, roll_c, pitch_c = gimbal_joint_angles(d_flu)
    e = gimbal_joint_offset_to_enu([0, 0, 5], (yaw_c, roll_c, pitch_c), pitched)
    assert np.allclose(e, [0, 0, -5], atol=1e-6), e
    print('frame conversion self-test: PASS '
          '(ENU/NED + down-camera + gimbal chains)')


if __name__ == '__main__':
    _selftest()
