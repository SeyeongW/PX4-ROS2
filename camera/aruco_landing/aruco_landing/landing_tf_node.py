"""landing_tf_node — the TF chain the landing stack measures through.

WHY THIS NODE EXISTS
--------------------
`aruco_pose_node` solves the marker in the CAMERA optical frame, then asks tf2
to express it in the frame the mission flies in.  `mpc_landing_node` subtracts
that pose from the MAVROS local position (`p_rel0 = p_d - tgt`), so the marker
has to arrive in **map**, the MAVROS local ENU frame.

Nothing on this aircraft published any of that.  Every other piece was already
here — MAVROS has the vehicle pose, siyi_gimbal has the gimbal attitude — but
no one joined them, so `lookup_transform` threw on every single frame and the
detector published zero poses while drawing a perfect green square on the debug
view.  That failure is indistinguishable from "the camera cannot see the
marker", which is how it survived as long as it did.

    map ──(this node, from /mavros/local_position/pose)──> base_link
                                                              │
                       (this node, static lever arm) ─────────┤
                                                              ▼
                                                        gimbal_mount
                                                              │
                    (this node, from ~/attitude) ─────────────┤
                                                              ▼
                                              down_camera_optical_frame

WHY map→base_link IS PUBLISHED HERE AND NOT BY MAVROS
-----------------------------------------------------
MAVROS can do it (`local_position/tf/send`), but it is off by default and lives
in a config file passed to a launch this repo deliberately does not own — the
docstring of `fixed_marker_landing.launch.py` explains why MAVROS is started by
hand.  A TF chain that silently depends on a parameter in someone else's launch
file is the same class of bug this node was written to fix.

Publishing it from the same PoseStamped that `mpc_landing_node` flies on has a
second, better property: the marker is transformed with EXACTLY the vehicle
pose the controller differences it against, so an estimator jump cancels out
instead of appearing as marker motion.

If you would rather MAVROS owned it, set `publish_map_tf:=false` — two
publishers of the same transform is a fight tf2 does not referee.

WHICH FRAME THE GIMBAL REPORTS ITS ANGLES IN
--------------------------------------------
This is the one thing that cannot be settled from a desk, and getting it
backwards applies the airframe rotation twice.  It matters: at a 5 m search
height a 10 deg airframe tilt handled wrongly puts the marker ~0.9 m off, which
is larger than the whole 0.35 m touchdown tolerance.

`ros2 run siyi_gimbal gimbal_monitor` decides it in the field — tilt the
airframe 15-20 deg in pitch and read its verdict.  Then set
`gimbal_attitude_reference` to match:

  'stabilized'  DEFAULT.  Roll and pitch are earth-referenced, yaw is
                body-referenced.  This is what a 3-axis stabilized gimbal in
                follow mode physically does: roll and pitch are held against
                GRAVITY, which is an earth measurement, while yaw is a joint
                angle off the mount — which is why commanding `nadir_yaw 0`
                means "aligned with the airframe" and not "aligned with north".
                Note that gimbal_monitor only exercises PITCH, so its
                'EARTH-referenced' verdict is a verdict about roll/pitch and is
                consistent with this setting, not with 'earth'.

  'body'        The report is pure joint angles.  Compose with the airframe.

  'earth'       The report is a full world attitude INCLUDING yaw.  Only pick
                this if the gimbal's yaw datum really is the world and not the
                mount; a gimbal yaw that reads ~0 while the aircraft turns is
                the giveaway that it is NOT this.

ANGLE SIGNS
-----------
SIYI reports aerospace-signed angles (pitch up +, yaw right +, roll right +) —
`siyi_gimbal_node` documents "pitch is negative-down on this gimbal", and it
commands -90 for nadir.  base_link is ROS ENU (X forward, Y LEFT, Z UP), where
those two conventions disagree in sign on pitch and yaw.  `rotation_body_from
_gimbal` is where that conversion lives, and `test_landing_tf.py` pins the
nadir case so a sign flip fails a test instead of a landing.

Interfaces
----------
Subscribes  /mavros/local_position/pose   geometry_msgs/PoseStamped
            /siyi_gimbal_node/attitude    geometry_msgs/Vector3Stamped (deg)
Publishes   /tf
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

# camera_link (X along the lens, Y left, Z up) -> optical (X right, Y down,
# Z forward), the REP-103 rotation.  Columns are the optical axes written in
# camera_link.
_OPTICAL_FROM_LINK = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def _rx(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_body_from_gimbal(roll_deg: float, pitch_deg: float,
                              yaw_deg: float) -> np.ndarray:
    """Gimbal angles -> R(ENU-body <- camera_link).

    The sign flips on pitch and yaw are the whole point.  A rotation of +theta
    about ENU +Y carries +X (forward) toward -Z (down), i.e. it pitches the nose
    DOWN, while the gimbal calls nose-up positive — so pitch enters as -theta.
    The same argument about ENU +Z (up) and yaw-right-positive gives -psi.  Roll
    needs no flip: +phi about ENU +X lifts the left side, which is the
    right-side-down the gimbal means.

    Nadir (0, -90, 0) therefore has to come out as Ry(+90): lens forward along
    base -Z, camera up along base +X.  That is what the test asserts.
    """
    return (_rz(-math.radians(yaw_deg))
            @ _ry(-math.radians(pitch_deg))
            @ _rx(math.radians(roll_deg)))


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (x, y, z, w), via the numerically stable branch."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array([0.25 * scale,
                                   (matrix[0, 1] + matrix[1, 0]) / scale,
                                   (matrix[0, 2] + matrix[2, 0]) / scale,
                                   (matrix[2, 1] - matrix[1, 2]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array([(matrix[0, 1] + matrix[1, 0]) / scale,
                                   0.25 * scale,
                                   (matrix[1, 2] + matrix[2, 1]) / scale,
                                   (matrix[0, 2] - matrix[2, 0]) / scale])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array([(matrix[0, 2] + matrix[2, 0]) / scale,
                                   (matrix[1, 2] + matrix[2, 1]) / scale,
                                   0.25 * scale,
                                   (matrix[1, 0] - matrix[0, 1]) / scale])
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def quaternion_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    """(x, y, z, w) -> rotation matrix, normalising first."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0 or not math.isfinite(norm):
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def enu_yaw_from_rotation(rotation: np.ndarray) -> float:
    """Heading of an ENU body frame, radians CCW from East."""
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def camera_rotation_in_body(reference: str, roll_deg: float, pitch_deg: float,
                            yaw_deg: float,
                            rotation_map_from_body: np.ndarray) -> np.ndarray:
    """R(base_link <- camera_link) under each gimbal attitude convention.

    'body'        the report is already relative to the mount, so it IS the
                  answer and the vehicle attitude never enters.

    'earth'       the report is a world attitude, so it has to be brought back
                  into base_link by the inverse of the vehicle rotation.

    'stabilized'  roll and pitch are world (held against gravity) but yaw is a
                  joint angle.  The camera's world heading is therefore the
                  vehicle's heading plus the gimbal's yaw joint, and the rest is
                  the 'earth' case.
    """
    if reference == 'body':
        return rotation_body_from_gimbal(roll_deg, pitch_deg, yaw_deg)

    if reference == 'earth':
        rotation_map_from_camera = rotation_body_from_gimbal(
            roll_deg, pitch_deg, yaw_deg)
    elif reference == 'stabilized':
        # Gimbal yaw is a joint off the mount; ENU yaw is left-positive, so the
        # right-positive gimbal joint subtracts.
        heading = (enu_yaw_from_rotation(rotation_map_from_body)
                   - math.radians(yaw_deg))
        rotation_map_from_camera = (_rz(heading)
                                    @ _ry(-math.radians(pitch_deg))
                                    @ _rx(math.radians(roll_deg)))
    else:
        raise ValueError(f'unknown gimbal_attitude_reference: {reference}')

    return rotation_map_from_body.T @ rotation_map_from_camera


class LandingTfNode(Node):
    def __init__(self):
        super().__init__('landing_tf_node')
        self._declare()
        g = self.get_parameter

        self.map_frame = str(g('map_frame').value)
        self.base_frame = str(g('base_frame').value)
        self.mount_frame = str(g('gimbal_mount_frame').value)
        self.optical_frame = str(g('camera_optical_frame').value)
        self.reference = str(g('gimbal_attitude_reference').value)
        if self.reference not in ('stabilized', 'body', 'earth'):
            raise ValueError(
                f'gimbal_attitude_reference must be stabilized, body or earth '
                f'(got {self.reference!r})')
        self.publish_map_tf = bool(g('publish_map_tf').value)
        self.mount_xyz = [float(v) for v in g('gimbal_mount_xyz_m').value]
        self.assume_nadir_s = float(g('assume_nadir_after_s').value)
        self.nadir_pitch = float(g('nadir_pitch_deg').value)

        self.broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_mount()

        # The vehicle rotation is needed by two of the three conventions and by
        # map->base_link.  Identity until the first pose arrives, which is
        # harmless because nothing is published before then either.
        self.rotation_map_from_body = np.eye(3)
        self._gimbal: tuple[float, float, float] | None = None
        self._gimbal_t: float | None = None
        self._pose: PoseStamped | None = None
        # The grace period has to run from somewhere, and startup is the only
        # honest origin before the first packet: without it the very first tick
        # would announce a 3 s silence that has not happened yet, and the
        # operator would learn to ignore the one warning that says the gimbal
        # link is dead.
        self._start_t = self.get_clock().now().nanoseconds * 1e-9

        self.create_subscription(PoseStamped, str(g('vehicle_pose_topic').value),
                                 self._on_pose, _sensor_qos())
        self.create_subscription(Vector3Stamped,
                                 str(g('gimbal_attitude_topic').value),
                                 self._on_gimbal, 10)

        # The camera transform is republished on a timer rather than only when a
        # gimbal packet lands.  The gimbal answers at 2 Hz, but the ArUco lookup
        # is at the image's CAPTURE stamp, and tf2 refuses to extrapolate past
        # its newest sample — a sparse chain turns into ExtrapolationException,
        # which rejects exactly the frames a fast camera makes freshest.  A
        # stabilized gimbal's attitude is near-constant between reports, so
        # holding the last one is honest.
        rate_hz = float(g('tf_rate_hz').value)
        self.create_timer(1.0 / rate_hz, self._publish_camera)
        self.create_timer(5.0, self._report)

        self.get_logger().info(
            f'landing_tf_node: {self.map_frame} -> {self.base_frame} -> '
            f'{self.mount_frame} -> {self.optical_frame} at {rate_hz:.0f} Hz, '
            f'gimbal attitude treated as {self.reference.upper()}-referenced '
            f'(confirm with `ros2 run siyi_gimbal gimbal_monitor`)')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these may be set. The launch file passes none."""
        p = self.declare_parameter
        p('map_frame', 'map')
        p('base_frame', 'base_link')
        p('gimbal_mount_frame', 'gimbal_mount')
        # Must match aruco_pose_node's `optical_frame_id`, which is the frame it
        # stamps its solvePnP result in.
        p('camera_optical_frame', 'down_camera_optical_frame')
        # See the module docstring. gimbal_monitor decides this in the field.
        p('gimbal_attitude_reference', 'stabilized')
        # Turn off if MAVROS is configured with local_position/tf/send instead.
        p('publish_map_tf', True)
        # MEASURE THIS. Where the gimbal's rotation centre sits relative to the
        # FCU origin, in base_link ENU metres (X forward, Y left, Z up) — a
        # camera slung 12 cm forward of the FCU reports the marker 12 cm behind
        # where it is. The touchdown gate is 0.35 m, so a lever arm this stack
        # ignores spends a third of the budget before the controller starts.
        # Left at zero it is not wrong, it is unmeasured.
        p('gimbal_mount_xyz_m', [0.0, 0.0, 0.0])
        p('vehicle_pose_topic', '/mavros/local_position/pose')
        p('gimbal_attitude_topic', '/siyi_gimbal_node/attitude')
        # 50 Hz: the gap between the newest TF sample and an image stamped at
        # capture is what tf2 has to interpolate across, and at 50 Hz the worst
        # case is 20 ms.
        p('tf_rate_hz', 50.0)
        # If the gimbal never reports (poll_attitude off, a dead serial link),
        # fall back to the commanded nadir after this long rather than publish
        # nothing. A landing stack that stops because a STATUS packet was lost,
        # while the gimbal is in fact pointing exactly where it was told to,
        # fails for the wrong reason. Set to 0 to require real feedback.
        p('assume_nadir_after_s', 3.0)
        p('nadir_pitch_deg', -90.0)

    # -------------------------------------------------------------- callbacks
    def _on_pose(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        self.rotation_map_from_body = quaternion_to_rotation(q.x, q.y, q.z, q.w)
        self._pose = msg
        if not self.publish_map_tf:
            return
        t = TransformStamped()
        # The pose's OWN stamp, not `now`: tf2 interpolates on these, and
        # restamping would quietly tell it the vehicle was here more recently
        # than the estimator ever said.
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.broadcaster.sendTransform(t)

    def _on_gimbal(self, msg: Vector3Stamped) -> None:
        self._gimbal = (float(msg.vector.x), float(msg.vector.y),
                        float(msg.vector.z))
        self._gimbal_t = self.get_clock().now().nanoseconds * 1e-9

    # ----------------------------------------------------------------- output
    def _publish_mount(self) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.mount_frame
        (t.transform.translation.x, t.transform.translation.y,
         t.transform.translation.z) = self.mount_xyz
        t.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t)

    def _gimbal_angles(self) -> tuple[float, float, float] | None:
        """The last reported angles, or the commanded nadir if it has gone quiet."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._gimbal is not None and self._gimbal_t is not None:
            if now - self._gimbal_t <= max(self.assume_nadir_s, 1.0):
                return self._gimbal
        if self.assume_nadir_s <= 0.0:
            return None
        silent_since = self._gimbal_t if self._gimbal_t is not None \
            else self._start_t
        silent_s = now - silent_since
        if silent_s < self.assume_nadir_s:
            # Still inside the grace period. Publishing nothing for a moment is
            # better than publishing a guess, and the detector's own gates
            # already tolerate a short gap.
            return None
        self.get_logger().warn(
            f'no gimbal attitude for {silent_s:.0f} s — assuming the commanded '
            f'nadir (pitch {self.nadir_pitch:.0f} deg). Ranges stay valid; '
            f'horizontal offsets are only as good as that assumption.',
            throttle_duration_sec=10.0)
        return (0.0, self.nadir_pitch, 0.0)

    def _publish_camera(self) -> None:
        angles = self._gimbal_angles()
        if angles is None:
            return
        # 'body' is self-contained; the other two are expressed through the
        # vehicle rotation, so publishing before a pose has arrived would put
        # the camera in a frame built on an identity that is not true.
        if self.reference != 'body' and self._pose is None:
            self.get_logger().warn(
                f'waiting for {self.get_parameter("vehicle_pose_topic").value} '
                f'— {self.reference} attitude needs the vehicle rotation',
                throttle_duration_sec=10.0)
            return

        roll, pitch, yaw = angles
        rotation = camera_rotation_in_body(self.reference, roll, pitch, yaw,
                                           self.rotation_map_from_body)
        x, y, z, w = rotation_to_quaternion(rotation @ _OPTICAL_FROM_LINK)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.mount_frame
        t.child_frame_id = self.optical_frame
        t.transform.rotation.x = x
        t.transform.rotation.y = y
        t.transform.rotation.z = z
        t.transform.rotation.w = w
        self.broadcaster.sendTransform(t)

    def _report(self) -> None:
        if self._pose is None:
            self.get_logger().warn(
                'no vehicle pose yet — is MAVROS up? Without it there is no '
                f'{self.map_frame} -> {self.base_frame} and the detector will '
                'reject every pose with tf_lookup.', throttle_duration_sec=10.0)


def _sensor_qos() -> QoSProfile:
    """MAVROS publishes BEST_EFFORT; a RELIABLE subscriber would get nothing."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


def main(args=None):
    rclpy.init(args=args)
    node = LandingTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
