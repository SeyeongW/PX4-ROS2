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
        marker is not visible.  `marker_kf_node` turns this into the continuous
        /marker/position + /marker/velocity the MPC consumes.

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

KNOWN OFFSET: the camera offset is added to the vehicle position rather than to
the LENS position, and on the gimbal vehicle the lens sits 0.13 m below
base_link.  Measured in flight as a 0.125 m shortfall in the reported marker z
(1.686 m against a true 1.811 m).  It is left in because `use_deck_z` pins z
anyway, and the horizontal component of the lever arm is under 5 cm.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import JointState

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

from .frame import camera_offset_to_enu, gimbal_joint_offset_to_enu


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


class MarkerTfNode(Node):
    def __init__(self):
        super().__init__('marker_tf_node')
        p = self.declare_parameter
        self.hist_s = float(p('pose_history_s', 1.0).value)
        # The deck plane height is known geometry; camera range is the noisy
        # part, so optionally pin z instead of trusting solvePnP range.
        self.deck_z = float(p('deck_z', 1.811).value)
        self.use_deck_z = bool(p('use_deck_z', True).value)
        self.local_pos_topic = str(
            p('local_pos_topic', '/fmu/out/vehicle_local_position_v1').value)
        self.camera_frame = str(p('camera_frame', 'body').value)
        if self.camera_frame not in ('body', 'gimbal'):
            raise ValueError(
                f"camera_frame must be 'body' or 'gimbal', got "
                f"'{self.camera_frame}'")
        joint_topic = str(p('joint_state_topic', '/gimbal/joint_state').value)

        self._pos_hist = deque()
        self._att_hist = deque()
        self._joint_hist = deque()
        self._n = 0

        self.meas_pub = self.create_publisher(PointStamped, '/marker/measured', 10)
        self.create_subscription(PoseStamped, '/aruco/pose_cam', self._on_pose, 10)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude',
                                 self._on_att, _px4_qos())
        self.create_subscription(VehicleLocalPosition, self.local_pos_topic,
                                 self._on_pos, _px4_qos())
        if self.camera_frame == 'gimbal':
            self.create_subscription(JointState, joint_topic,
                                     self._on_joints, 10)
        if not self.get_parameter('use_sim_time').value:
            self.get_logger().warn(
                'use_sim_time=false: image stamps are sim time, so the pose '
                'interpolation will be wrong. Launch with -p use_sim_time:=true')
        rotation_source = ('the airframe attitude + gimbal joints'
                           if self.camera_frame == 'gimbal'
                           else 'the airframe attitude')
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
        if not self._att_hist:
            return
        if self.camera_frame == 'gimbal' and not self._joint_hist:
            self._warn_no_joints()
            return
        t_img = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p_d = sample_at(self._pos_hist, t_img)
        q = sample_at(self._att_hist, t_img)
        q = q / max(float(np.linalg.norm(q)), 1e-9)

        p_opt = np.array([msg.pose.position.x, msg.pose.position.y,
                          msg.pose.position.z])
        if self.camera_frame == 'gimbal':
            # Interpolated to the image stamp for the same reason the vehicle
            # state is: during a slew the joints move several degrees per frame.
            joints = sample_at(self._joint_hist, t_img)
            offset = gimbal_joint_offset_to_enu(p_opt, joints, q)
        else:
            offset = camera_offset_to_enu(p_opt, q)
        marker = p_d + offset
        if self.use_deck_z:
            marker[2] = self.deck_z

        pm = PointStamped()
        pm.header.stamp = msg.header.stamp
        pm.header.frame_id = 'map'
        pm.point.x, pm.point.y, pm.point.z = (float(marker[0]), float(marker[1]),
                                              float(marker[2]))
        self.meas_pub.publish(pm)
        self._n += 1


def main(args=None):
    rclpy.init(args=args)
    node = MarkerTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
