"""gimbal_control_node — ONE job: keep the lens pointed at the target.

This is the node `px4_models/x500_gimbal_rgbd_lidar/model.sdf` promises.  It
exists because of a measured failure, not a preference: with a body-fixed down
camera the vehicle was **not pointed at the target 62% of a 155 s flight**,
while the detector fired 93% of the time whenever geometry said the marker was
visible.  A multirotor accelerates by tilting, so

    tan(theta) = a_h / g   =>   line-of-sight shift = h * a_h / g

and at h = 10 m even a gentle 2 m/s^2 chase walks the aim 2.0 m inside a 5.5 m
visible radius.  Chasing and seeing are in direct conflict on a fixed camera.
A gimbal removes the conflict instead of trading one against the other.

Why PX4's own gimbal driver is not used: `GZGimbal` accepts MAVLink setpoints
in *body* angles, so a body-fixed command tilts with the airframe — the exact
coupling we are trying to break.  Stabilisation has to be closed against the
world, so it is closed here.

Subscribes
    /fmu/out/vehicle_attitude           px4_msgs/VehicleAttitude  (FRD->NED q)
    /fmu/out/vehicle_local_position_v1  px4_msgs/VehicleLocalPosition (NED)
    /marker/cue                         PointStamped  long-range target (ENU)
    /marker/position                    PointStamped  vision/KF target (ENU)
    /marker/valid                       Bool          is the vision fix usable
    (gz) /world/<world>/model/<model>/joint_state       gz.msgs.Model
Publishes
    (gz) /model/<model>/command/gimbal_yaw|roll|pitch   gz.msgs.Double
    /gimbal/joint_state                 sensor_msgs/JointState  (yaw,roll,pitch)
    /gimbal/aim_error_deg               Float32   DESIRED vs measured aim, i.e.
                                        including whatever the slew limiter
                                        swallowed (see `_publish_joints`)
    /tf                                 the gimbal joint chain, ending at
                                        `gimbal_camera_optical_frame`, plus
                                        map->base_link from PX4's own state so
                                        RViz can place the whole thing

This node owns BOTH directions of the Gazebo gimbal interface — commands out,
encoders in — so nothing else has to speak gz-transport to use the gimbal.
`/gimbal/joint_state` is what `marker_tf_node` needs to place a detection in
the world, and it is deliberately the ENCODER angle rather than the command:
the servos run with the integral term disabled, so they hold a small
gravity droop that a command-only chain would never see.

Control law, all in `frame.py` and unit-tested there:

    1. pick a look-at point (vision when valid, else the cue, else nadir)
    2. direction in world ENU  ->  body FLU          `enu_dir_to_flu`
    3. body FLU  ->  joint commands                  `gimbal_joint_angles`

Step 2 is the entire trick: the airframe attitude enters with a TRANSPOSE, so
whatever tilt the vehicle adopts is subtracted back out of the joint command.

The loop is deliberately open at this level — the joint servo closes the inner
loop in Gazebo — and the encoders are used to MEASURE the residual, in the
spirit of the rest of this package: instrument it, do not assume it.

Note on frames: the camera also carries an IMU, and using it directly would be
the obvious way to get the lens attitude.  It is NOT used here, because it
reports against GAZEBO's world while everything else in this stack lives in
PX4's local frame, and the two sit 27.3 deg apart (the world's magnetic_field
already encodes a declination that PX4's EKF then applies again).  Joint angles
are body-relative and so carry no world frame at all.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformBroadcaster

from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model as GzModel
from gz.transport13 import Node as GzNode
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

from .frame import (LOCAL_ENU_FRAME_ID, enu_dir_to_flu, flu_dir_to_enu,
                    gimbal_axis_flu, gimbal_joint_angles, gimbal_tf_chain,
                    rot_to_quat)
from .parameter_utils import (
    DEFAULT_DECK_Z_M,
    require_between,
    require_finite,
    require_nonempty,
    require_nonnegative,
    require_positive,
)

# gimbal_down's joint names, in the (yaw, roll, pitch) order this node uses.
JOINT_NAMES = ('cgo3_vertical_arm_joint', 'cgo3_horizontal_arm_joint',
               'cgo3_camera_joint')


def _px4_qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


def _rate_limit(current, desired, max_step):
    """Move `current` toward `desired` by at most `max_step`, shortest way."""
    delta = desired - current
    # yaw is an angle: never take the long way round
    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    return current + max(-max_step, min(max_step, delta))


def _nadir_handoff(direction, start_depression):
    """Blend continuously from nadir to a shallow target direction."""
    down = np.array([0.0, 0.0, -1.0])
    target = np.asarray(direction, float)
    norm = float(np.linalg.norm(target))
    if norm < 1e-9:
        return down
    target /= norm
    if start_depression <= 0.0:
        return target
    depression = math.atan2(max(0.0, -target[2]),
                            max(float(np.hypot(target[0], target[1])), 1e-6))
    end_depression = min(2.0 * start_depression, math.pi / 2.0)
    if depression <= start_depression:
        return down
    if depression >= end_depression:
        return target
    weight = (depression - start_depression) / (
        end_depression - start_depression)
    blended = (1.0 - weight) * down + weight * target
    return blended / np.linalg.norm(blended)


class GimbalControlNode(Node):
    def __init__(self):
        super().__init__('gimbal_control_node')
        p = self.declare_parameter
        # The gz entity, not the SDF model name: PX4 spawns it with a _0 suffix.
        self.model = require_nonempty(
            'model_name',
            p('model_name', 'x500_gimbal_rgbd_lidar_0').value)
        self.world = require_nonempty(
            'world', p('world', 'mpc_landing_200m_moving').value)
        self.rate_hz = require_positive(
            'rate_hz', p('rate_hz', 50.0).value)
        # Slew limit, per [[feedback_gentle_control]]: a gimbal that snaps
        # smears the image and shakes the airframe through the joint reaction.
        # 90 deg/s crosses a full hemisphere in 2 s, far quicker than a chase.
        self.max_rate = math.radians(require_positive(
            'max_rate_deg_s', p('max_rate_deg_s', 90.0).value))
        self.deck_z = require_finite(
            'deck_z', p('deck_z', DEFAULT_DECK_Z_M).value)
        self.vision_timeout_s = require_positive(
            'vision_timeout_s', p('vision_timeout_s', 1.0).value)
        self.cue_timeout_s = require_positive(
            'cue_timeout_s', p('cue_timeout_s', 2.0).value)
        # Below this height the target fills the frame and pointing off-nadir
        # buys nothing, so stop chasing and hold nadir for a clean touchdown.
        self.nadir_below_m = require_nonnegative(
            'nadir_below_m', p('nadir_below_m', 2.0).value)
        # Do not aim at a target until the look-down angle is steep enough to be
        # worth it.  Chasing a cue 57 m out and 14 m below is a ~14 deg
        # depression: the 1.3 m marker is unresolvable at that range AND lies
        # edge-on (measured 0% detection during the whole approach), so aiming
        # there sees nothing — and because the cue is nearly on the horizon,
        # its AZIMUTH swings as the vehicle chases, which slews the gimbal yaw
        # up to 180 deg (measured slew_sat 43-90%) and rolls the image so hard
        # it looks like the drone flipped (it did not — the flight log holds
        # the airframe at <=5 deg tilt throughout).  So hold nadir until the
        # geometry is steep; the image stays stable and the yaw stops thrashing.
        # 25 deg starts a continuous nadir-to-target handoff; it reaches the
        # target direction at 50 deg.  Applying the same handoff to cue and
        # vision prevents either source from creating a 65–80 deg aim step.
        self.min_cue_depression = math.radians(require_between(
            'min_cue_depression_deg',
            p('min_cue_depression_deg', 25.0).value,
            0.0, 90.0, upper_inclusive=False))
        self.report_s = require_positive(
            'report_period_s', p('report_period_s', 5.0).value)
        self.local_pos_topic = require_nonempty(
            'local_pos_topic',
            p('local_pos_topic',
              '/fmu/out/vehicle_local_position_v1').value)
        # TF is how you SEE the gimbal tracking instead of reading numbers.
        self.publish_tf = bool(p('publish_tf', True).value)
        # Nothing else in landing_mpc publishes the vehicle's own transform, so
        # without this the gimbal chain would float unattached in RViz.  Turn
        # it off if another node owns map->base_link.
        self.publish_base_tf = bool(p('publish_base_tf', True).value)
        self.map_frame = str(p('map_frame', LOCAL_ENU_FRAME_ID).value)
        self.base_frame = str(p('base_frame', 'base_link').value)
        self.tf_rate_hz = float(p('tf_rate_hz', 30.0).value)
        if self.publish_tf:
            self.map_frame = require_nonempty('map_frame', self.map_frame)
            self.base_frame = require_nonempty('base_frame', self.base_frame)
            self.tf_rate_hz = require_positive(
                'tf_rate_hz', self.tf_rate_hz)

        self._q = None                       # airframe attitude [w,x,y,z]
        self._p_d = None                     # drone position, local ENU
        self._cue = self._cue_t = None
        self._vis = self._vis_t = None
        self._valid = False
        self._measured = None                # (yaw, roll, pitch) from encoders
        self._yaw = self._roll = 0.0
        self._pitch = -math.pi / 2.0         # start looking down
        self._desired = None                 # (yaw, pitch) BEFORE the slew limit
        self._err_sum = 0.0
        self._err_n = 0
        self._err_max = 0.0
        self._lag_sum = 0.0
        self._lag_max = 0.0
        self._sat_n = 0
        self._source = 'nadir'

        self.gz = GzNode()
        base = f'/model/{self.model}/command/gimbal_'
        self._pub_yaw = self.gz.advertise(base + 'yaw', Double)
        self._pub_roll = self.gz.advertise(base + 'roll', Double)
        self._pub_pitch = self.gz.advertise(base + 'pitch', Double)
        self.gz.subscribe(GzModel,
                          f'/world/{self.world}/model/{self.model}/joint_state',
                          self._on_joint_state)

        self.joint_pub = self.create_publisher(JointState, '/gimbal/joint_state',
                                               10)
        self.err_pub = self.create_publisher(Float32, '/gimbal/aim_error_deg', 10)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude',
                                 self._on_att, _px4_qos())
        self.create_subscription(VehicleLocalPosition, self.local_pos_topic,
                                 self._on_pos, _px4_qos())
        self.create_subscription(PointStamped, '/marker/cue', self._on_cue, 10)
        self.create_subscription(PointStamped, '/marker/position',
                                 self._on_vis, 10)
        self.create_subscription(Bool, '/marker/valid', self._on_valid, 10)

        self.tf_pub = TransformBroadcaster(self) if self.publish_tf else None

        self.create_timer(1.0 / self.rate_hz, self._tick)
        if self.tf_pub is not None:
            # Slower than the control loop: TF is for consumers and RViz, and
            # 30 Hz is already finer than the 25 Hz camera it describes.
            self.create_timer(1.0 / self.tf_rate_hz, self._broadcast_tf)
        self.create_timer(self.report_s, self._report)
        self.get_logger().info(
            f'gimbal_control_node: aiming {self.model} at the target '
            f'({self.rate_hz:.0f} Hz, slew '
            f'{math.degrees(self.max_rate):.0f} deg/s)')

    # ------------------------------------------------------------- callbacks
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_att(self, m):
        self._q = np.array(m.q, float)

    def _on_pos(self, m):
        self._p_d = np.array([m.y, m.x, -m.z])          # NED -> ENU

    def _on_cue(self, m):
        self._cue = np.array([m.point.x, m.point.y, m.point.z])
        self._cue_t = self._now()

    def _on_vis(self, m):
        self._vis = np.array([m.point.x, m.point.y, m.point.z])
        self._vis_t = self._now()

    def _on_valid(self, m):
        self._valid = bool(m.data)

    def _on_joint_state(self, msg: GzModel):
        """Gazebo joint encoders (gz-transport callback, NOT the ROS thread)."""
        angles = {j.name: j.axis1.position for j in msg.joint}
        if not all(name in angles for name in JOINT_NAMES):
            return
        first = self._measured is None
        self._measured = tuple(float(angles[name]) for name in JOINT_NAMES)
        if first:
            # SEED the command from the encoders instead of asserting nadir.
            # The joints start at 0 but `self._pitch` was initialised to -pi/2,
            # so the very first command was a 90 deg STEP.  The rate limiter
            # does not help: it limits how fast the COMMAND moves, and the
            # joint controller still saw ~1.5 rad of error, asked for
            # p_gain*1.5 = 3 N.m and sat at its 1 N.m cmd_max.  Measured in
            # flight as `servo max=89.56 deg`.  That torque has an equal and
            # opposite reaction on the airframe, and 1 N.m against an x500's
            # ~0.06 kg.m^2 roll inertia is 16 rad/s^2 — enough to matter.
            # Starting from where the joints actually are makes the first
            # command a no-op and lets the slew limiter do its job.
            self._yaw, self._roll, self._pitch = self._measured

    # ------------------------------------------------------------ look-at
    def _look_at(self):
        """The point to aim at, in local ENU, or None to hold nadir.

        Vision wins when it is valid and fresh; otherwise the cue carries the
        aim through the long approach, which is precisely the phase where the
        fixed camera lost the marker.  Aiming at the cue before the marker is
        resolvable is what lets vision acquire at all.
        """
        now = self._now()
        if (self._valid and self._vis is not None
                and now - self._vis_t < self.vision_timeout_s):
            self._source = 'vision'
            return self._vis
        if self._cue is not None and now - self._cue_t < self.cue_timeout_s:
            self._source = 'cue'
            return self._cue
        self._source = 'nadir'
        return None

    def _desired_axis_flu(self):
        """Where the lens should point, as a body-FLU unit vector."""
        target = self._look_at()
        if target is None or self._p_d is None:
            d_enu = np.array([0.0, 0.0, -1.0])
        else:
            d_enu = target - self._p_d
            height = self._p_d[2] - self.deck_z
            hold_nadir = height < self.nadir_below_m or np.linalg.norm(d_enu) < 1e-3
            if hold_nadir:
                d_enu = np.array([0.0, 0.0, -1.0])
                self._source = 'nadir'
            else:
                target_axis = d_enu / np.linalg.norm(d_enu)
                d_enu = _nadir_handoff(d_enu, self.min_cue_depression)
                if np.allclose(d_enu, [0.0, 0.0, -1.0]):
                    self._source = f'nadir ({self._source} too shallow)'
                elif not np.allclose(d_enu, target_axis):
                    self._source += ' handoff'
        return enu_dir_to_flu(d_enu, self._q)

    # ----------------------------------------------------------------- loop
    def _tick(self):
        if self._q is None:
            return
        yaw_d, roll_d, pitch_d = gimbal_joint_angles(self._desired_axis_flu(),
                                                     yaw_hold=self._yaw,
                                                     pitch_hold=self._pitch)
        step = self.max_rate / self.rate_hz
        yaw_next = _rate_limit(self._yaw, yaw_d, step)
        roll_next = _rate_limit(self._roll, roll_d, step)
        pitch_next = _rate_limit(self._pitch, pitch_d, step)
        desired_axis = gimbal_axis_flu(yaw_d, pitch_d)
        limited_axis = gimbal_axis_flu(yaw_next, pitch_next)
        optical_residual = math.acos(float(np.clip(
            np.dot(desired_axis, limited_axis), -1.0, 1.0)))
        if optical_residual > 1e-4:
            self._sat_n += 1
        self._desired = (yaw_d, pitch_d)
        self._yaw, self._roll, self._pitch = yaw_next, roll_next, pitch_next

        for publisher, value in ((self._pub_yaw, self._yaw),
                                 (self._pub_roll, self._roll),
                                 (self._pub_pitch, self._pitch)):
            message = Double()
            message.data = float(value)
            publisher.publish(message)

        self._publish_joints()

    def _publish_joints(self):
        """Republish the encoders for `marker_tf_node`, and score the aim.

        TWO errors are scored, because they fail for different reasons and the
        second one used to be invisible:

          servo error  commanded (post-slew-limit) vs encoder — how well the
                       joint servo holds what it was told.
          aim lag      DESIRED (pre-limit) vs encoder — how far the lens is
                       from the target, including everything the rate limiter
                       swallowed.  Scoring only the first made a slew-saturated
                       gimbal look perfect: the limiter guarantees the servo
                       always gets a reachable command, so the error it is
                       measured against is small precisely when tracking is
                       worst.  `aim lag` is what actually decides whether the
                       marker is in frame, so it is what gets published.

        Both are evaluated in the body frame, so no world frame enters and the
        PX4/Gazebo heading offset cannot contaminate the numbers.
        """
        if self._measured is None:
            return
        yaw_m, roll_m, pitch_m = self._measured
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [yaw_m, roll_m, pitch_m]
        self.joint_pub.publish(message)

        measured = gimbal_axis_flu(yaw_m, pitch_m)

        def _angle_to(axis):
            cos = float(np.dot(axis, measured))
            return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

        error = _angle_to(gimbal_axis_flu(self._yaw, self._pitch))
        self._err_sum += error
        self._err_max = max(self._err_max, error)
        self._err_n += 1

        lag = error if self._desired is None else _angle_to(
            gimbal_axis_flu(self._desired[0], self._desired[1]))
        self._lag_sum += lag
        self._lag_max = max(self._lag_max, lag)
        self.err_pub.publish(Float32(data=float(lag)))

    def _broadcast_tf(self):
        """Publish the gimbal joint chain, and the vehicle it hangs from.

        The joint hops come from the ENCODERS, so what RViz draws is where the
        lens actually is, not where it was asked to be.  The chain geometry
        lives in `frame.gimbal_tf_chain` alongside the analytic rotation the
        perception path uses, and the self-test asserts the two agree — so the
        picture and the maths cannot drift apart.
        """
        stamp = self.get_clock().now().to_msg()
        transforms = []

        if self.publish_base_tf and self._p_d is not None and self._q is not None:
            # PX4 gives FRD->NED; map(ENU)->base_link(FLU) is the same chain
            # `flu_dir_to_enu` applies, so build it from the basis vectors.
            R = np.column_stack([flu_dir_to_enu(axis, self._q)
                                 for axis in np.eye(3)])
            transforms.append(self._transform(
                stamp, self.map_frame, self.base_frame, self._p_d,
                rot_to_quat(R)))

        if self._measured is not None:
            yaw_m, roll_m, pitch_m = self._measured
            for parent, child, xyz, quat in gimbal_tf_chain(yaw_m, roll_m,
                                                            pitch_m):
                parent = self.base_frame if parent == 'base_link' else parent
                transforms.append(self._transform(stamp, parent, child, xyz,
                                                  quat))

        if transforms:
            self.tf_pub.sendTransform(transforms)

    @staticmethod
    def _transform(stamp, parent, child, xyz, quat):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(xyz[0])
        t.transform.translation.y = float(xyz[1])
        t.transform.translation.z = float(xyz[2])
        t.transform.rotation.w = float(quat[0])
        t.transform.rotation.x = float(quat[1])
        t.transform.rotation.y = float(quat[2])
        t.transform.rotation.z = float(quat[3])
        return t

    def _report(self):
        if not self._err_n:
            self.get_logger().warn(
                f'gimbal: source={self._source} '
                f'pitch={math.degrees(self._pitch):.1f} deg '
                f'yaw={math.degrees(self._yaw):.1f} deg — no joint encoders on '
                f'/world/{self.world}/model/{self.model}/joint_state, so '
                f'/gimbal/joint_state is silent and marker_tf_node will drop '
                f'every detection. Check the world and model_name parameters.')
            return
        sat = 100.0 * self._sat_n / self._err_n
        self.get_logger().info(
            f'gimbal: source={self._source} '
            f'pitch={math.degrees(self._pitch):.1f} deg '
            f'yaw={math.degrees(self._yaw):.1f} deg  '
            f'aim_lag mean={self._lag_sum / self._err_n:.2f} deg '
            f'max={self._lag_max:.2f} deg  '
            f'servo mean={self._err_sum / self._err_n:.2f} deg '
            f'max={self._err_max:.2f} deg  slew_sat={sat:.0f}%')
        if sat > 10.0:
            self.get_logger().warn(
                f'gimbal slew saturated {sat:.0f}% of the last '
                f'{self.report_s:.0f} s at '
                f'{math.degrees(self.max_rate):.0f} deg/s — the aim is being '
                f'dragged off target by the vehicle manoeuvre, not by the '
                f'pointing law.  Slow the approach (mission_manager '
                f'los_rate_budget_deg_s) or raise max_rate_deg_s.')
        self._err_sum = self._lag_sum = 0.0
        self._err_n = self._sat_n = 0
        self._err_max = self._lag_max = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = GimbalControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
