"""marker_tf_node — ONE job: camera-frame marker pose -> world (local ENU).

Knows nothing about images or ArUco.  It takes a marker pose expressed in the
camera optical frame, the vehicle state at the SAME instant, and produces the
target contract the landing MPC consumes.

Subscribes
    /aruco/pose_cam                     geometry_msgs/PoseStamped (camera frame)
    /fmu/out/vehicle_attitude           px4_msgs/VehicleAttitude  (FRD->NED q)
    /fmu/out/vehicle_local_position_v1  px4_msgs/VehicleLocalPosition (NED)
    /gimbal/joint_state                 sensor_msgs/JointState (gimbal mode)
Publishes
    /marker/measured                    geometry_msgs/PointStamped   local ENU
        One point per detection — raw, unsmoothed, and absent whenever the
        marker is not visible.  `marker_kf_node` turns this into continuous
        /marker/position + /marker/velocity estimator outputs.

Two things this node exists to get right:

1. **Time alignment.**  The image is already ~1 frame old when its pose arrives.
   Transforming with the *current* vehicle state smears the marker along the
   direction of travel (measured ~1.2 m bias on a diagonal approach; correcting
   it recovered ~0.3 m).  So the vehicle position/attitude are buffered and
   interpolated back to `pose_cam.header.stamp`.

2. **Attitude compensation.**  The camera offset is rotated into the world
   through whatever the CAMERA's attitude actually is — and that is the one
   place the two vehicles differ:

   * ``camera_frame=body`` (default): the lens is bolted to the airframe, so
     the airframe attitude IS the camera attitude and
     `frame.camera_offset_to_enu` applies it.
   * ``camera_frame=gimbal``: the lens hangs on joints, so the airframe
     attitude alone is the WRONG rotation — it would inject the very tilt the
     gimbal just removed.  `frame.gimbal_joint_offset_to_enu` composes the
     airframe attitude with the gimbal's own joint angles instead.

   Getting this wrong is silent: the marker still appears, just displaced by
   h*tan(tilt) — metres at altitude — which is exactly the error class this
   node exists to remove.

   The camera also carries an IMU that reports the lens attitude directly, and
   it is deliberately NOT used: it is referenced to GAZEBO's world, while the
   position it would be added to is in PX4's local frame, and the two are 27.3
   deg apart in SITL.  Joint angles are body-relative, so composing them with
   PX4's attitude keeps the whole chain in one frame.

Filtering and velocity estimation deliberately live in `marker_kf_node`, not
here: this node is a pure geometric transform.

For the gimbal camera with ``use_deck_z=true`` the transform uses the actual
lens origin and intersects the measured camera ray with that known plane.  This
deliberately throws away solvePnP range, which is the least reliable part of a
20--30 pixel marker pose.  The body-camera path stays backward-compatible;
``use_deck_z=false`` remains the raw-range diagnostic path.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import Imu, JointState

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

from .frame import (LOCAL_ENU_FRAME_ID, camera_offset_to_enu,
                    gimbal_camera_offset_to_enu,
                    gimbal_camera_origin_to_enu,
                    gimbal_joint_offset_to_enu)
from .parameter_utils import (
    DEFAULT_DECK_Z_M,
    require_finite,
    require_nonempty,
    require_positive,
)


def _px4_qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


def sample_at(hist, t):
    """Linear-interpolate a (time, vector) history at time ``t``."""
    if not hist:
        return None
    if t <= hist[0][0]:
        return hist[0][1]
    if t >= hist[-1][0]:
        return hist[-1][1]
    for i in range(len(hist) - 1):
        t0, v0 = hist[i]
        t1, v1 = hist[i + 1]
        if t0 <= t <= t1:
            w = 0.0 if (t1 - t0) < 1e-9 else (t - t0) / (t1 - t0)
            return (1.0 - w) * v0 + w * v1
    return hist[-1][1]


def ray_to_horizontal_plane(origin_enu, ray_enu, plane_z_m):
    """Intersect a forward camera ray with a known horizontal world plane.

    ArUco pose range is ill-conditioned when a marker is only a few pixels
    wide.  The pixel still gives a useful ray, and the deck height is known, so
    use only that direction.  ``None`` is fail-closed for a parallel ray or an
    intersection behind the camera.
    """
    origin = np.asarray(origin_enu, float)
    ray = np.asarray(ray_enu, float)
    plane_z = float(plane_z_m)
    if (origin.shape != (3,) or ray.shape != (3,)
            or not np.all(np.isfinite(np.r_[origin, ray, plane_z]))
            or abs(float(ray[2])) < 1.0e-9):
        return None
    scale = (plane_z - float(origin[2])) / float(ray[2])
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    point = origin + scale * ray
    point[2] = plane_z
    return point


class MarkerTfNode(Node):
    def __init__(self):
        super().__init__('marker_tf_node')
        p = self.declare_parameter
        self.hist_s = require_positive(
            'pose_history_s', p('pose_history_s', 1.0).value)
        # The deck plane height is known geometry; camera range is the noisy
        # part, so optionally pin z instead of trusting solvePnP range.
        self.deck_z = require_finite(
            'deck_z', p('deck_z', DEFAULT_DECK_Z_M).value)
        self.use_deck_z = bool(p('use_deck_z', True).value)
        self.local_pos_topic = require_nonempty(
            'local_pos_topic',
            p('local_pos_topic',
              '/fmu/out/vehicle_local_position_v1').value)
        self.camera_frame = str(p('camera_frame', 'body').value)
        if self.camera_frame not in ('body', 'gimbal'):
            raise ValueError(
                f"camera_frame must be 'body' or 'gimbal', got "
                f"'{self.camera_frame}'")
        self.gimbal_attitude_source = str(
            p('gimbal_attitude_source', 'joints_px4').value)
        if self.gimbal_attitude_source not in ('joints_px4', 'camera_imu'):
            raise ValueError(
                'gimbal_attitude_source must be joints_px4 or camera_imu, got '
                f"'{self.gimbal_attitude_source}'")
        if (self.camera_frame != 'gimbal'
                and self.gimbal_attitude_source != 'joints_px4'):
            raise ValueError(
                'gimbal_attitude_source applies only to camera_frame=gimbal')
        joint_topic = str(
            p('joint_state_topic', '/gimbal/joint_state').value)
        if self.camera_frame == 'gimbal':
            joint_topic = require_nonempty('joint_state_topic', joint_topic)

        self._pos_hist = deque()
        self._att_hist = deque()
        self._joint_hist = deque()
        self._camera_att_hist = deque()

        self.meas_pub = self.create_publisher(PointStamped, '/marker/measured', 10)
        self.create_subscription(PoseStamped, '/aruco/pose_cam', self._on_pose, 10)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude',
                                 self._on_att, _px4_qos())
        self.create_subscription(VehicleLocalPosition, self.local_pos_topic,
                                 self._on_pos, _px4_qos())
        if self.camera_frame == 'gimbal':
            self.create_subscription(JointState, joint_topic,
                                     self._on_joints, 10)
            if self.gimbal_attitude_source == 'camera_imu':
                camera_imu_topic = require_nonempty(
                    'camera_imu_topic',
                    p('camera_imu_topic', '/gimbal_camera/imu').value)
                self.create_subscription(
                    Imu, camera_imu_topic, self._on_camera_imu,
                    qos_profile_sensor_data)
        if not self.get_parameter('use_sim_time').value:
            self.get_logger().warn(
                'use_sim_time=false: image stamps are sim time, so the pose '
                'interpolation will be wrong. Launch with -p use_sim_time:=true')
        rotation_source = (
            'the camera IMU'
            if (self.camera_frame == 'gimbal'
                and self.gimbal_attitude_source == 'camera_imu')
            else ('the airframe attitude + gimbal joints'
                  if self.camera_frame == 'gimbal'
                  else 'the airframe attitude'))
        self.get_logger().info(
            f'marker_tf_node: /aruco/pose_cam + vehicle state -> '
            f'/marker/measured (camera_frame={self.camera_frame}, rotating '
            f'through {rotation_source})')

    # ---------------------------------------------------------- state buffers
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _push(self, hist, value):
        # Samples are keyed by RECEIPT time on the ROS clock.  Keying by PX4's
        # own timestamp was tried (PX4 runs a different epoch, so the offset was
        # estimated as a running minimum of receipt-sample) and measured WORSE:
        # dynamic error 0.66 m -> 1.08 m with a large lag-signature bias, because
        # the offset estimate is itself biased.  Receipt time keeps the bias
        # near zero; the residual is scatter, not lag.
        t = self._now()
        hist.append((t, value))
        while len(hist) > 2 and t - hist[0][0] > self.hist_s:
            hist.popleft()

    def _on_att(self, msg):
        self._push(self._att_hist, np.array(msg.q, float))     # [w,x,y,z]

    def _on_pos(self, msg):
        self._push(self._pos_hist, np.array([msg.y, msg.x, -msg.z]))  # -> ENU

    def _on_joints(self, msg):
        # gimbal_control_node publishes these in (yaw, roll, pitch) order.
        if len(msg.position) >= 3:
            self._push(self._joint_hist, np.array(msg.position[:3], float))

    def _on_camera_imu(self, msg):
        q = msg.orientation
        self._push(
            self._camera_att_hist,
            np.array([q.w, q.x, q.y, q.z], float))

    def _warn_no_joints(self):
        """Loud, because the alternative is silently dropping every detection.

        Throttled rather than once-only: the useful case is noticing it is
        STILL missing minutes into a run, not just at startup.
        """
        self.get_logger().warn(
            'camera_frame=gimbal but no /gimbal/joint_state yet — detections '
            'are being discarded. Is gimbal_control_node running against the '
            'gimbal vehicle (GIMBAL=1)?',
            throttle_duration_sec=5.0)

    # ---------------------------------------------------------------- transform
    def _on_pose(self, msg: PoseStamped):
        if not self._pos_hist:
            return
        use_camera_imu = (
            self.camera_frame == 'gimbal'
            and self.gimbal_attitude_source == 'camera_imu')
        if not use_camera_imu and not self._att_hist:
            return
        if (self.camera_frame == 'gimbal' and not use_camera_imu
                and not self._joint_hist):
            self._warn_no_joints()
            return
        if use_camera_imu and not self._camera_att_hist:
            self.get_logger().warn(
                'gimbal_attitude_source=camera_imu but no camera IMU sample '
                'yet; detections are being discarded',
                throttle_duration_sec=5.0)
            return
        t_img = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p_d = sample_at(self._pos_hist, t_img)

        p_opt = np.array([msg.pose.position.x, msg.pose.position.y,
                          msg.pose.position.z])
        if self.camera_frame == 'gimbal':
            if use_camera_imu:
                q_camera = sample_at(self._camera_att_hist, t_img)
                q_camera = q_camera / max(
                    float(np.linalg.norm(q_camera)), 1.0e-9)
                offset = gimbal_camera_offset_to_enu(p_opt, q_camera)
                # The lens lever is 0.13 m against a 23 m ray.  Using the PX4
                # base position here avoids mixing its yaw with Gazebo's
                # world-referenced camera IMU; the resulting city-SITL XY
                # error measured 3 cm, versus 1.1 m for that mixed frame.
                camera_origin = p_d
            else:
                # Interpolated to the image stamp for the same reason the
                # vehicle state is: during a slew the joints move several
                # degrees per frame.
                q = sample_at(self._att_hist, t_img)
                q = q / max(float(np.linalg.norm(q)), 1e-9)
                joints = sample_at(self._joint_hist, t_img)
                offset = gimbal_joint_offset_to_enu(p_opt, joints, q)
                camera_origin = (
                    p_d + gimbal_camera_origin_to_enu(joints, q))
        else:
            q = sample_at(self._att_hist, t_img)
            q = q / max(float(np.linalg.norm(q)), 1e-9)
            offset = camera_offset_to_enu(p_opt, q)
            camera_origin = p_d
        if self.use_deck_z and self.camera_frame == 'gimbal':
            marker = ray_to_horizontal_plane(
                camera_origin, offset, self.deck_z)
            if marker is None:
                self.get_logger().warn(
                    'marker ray does not meet the known deck plane in front '
                    'of the camera; dropping the fix',
                    throttle_duration_sec=2.0)
                return
        else:
            # Preserve the body-camera contract and the raw-range diagnostic.
            marker = p_d + offset
            if self.use_deck_z:
                marker[2] = self.deck_z

        pm = PointStamped()
        pm.header.stamp = msg.header.stamp
        pm.header.frame_id = LOCAL_ENU_FRAME_ID
        pm.point.x, pm.point.y, pm.point.z = (float(marker[0]), float(marker[1]),
                                              float(marker[2]))
        self.meas_pub.publish(pm)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerTfNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
