#!/usr/bin/env python3
"""Moving ArUco marker — Gazebo mover + ENU coordinate broadcaster.

Drives the ground ArUco marker along a trajectory inside Gazebo and, on the
SAME tick, publishes the marker's position as a local-ENU PointStamped. This
is the "cue" the precision_landing controller flies toward BEFORE the down
camera can see the marker; once vision acquires it, the controller hands off
to its visual servo and lands.

Geometry note: the drone spawns at the Gazebo world origin and MAVROS sets its
local-position (EKF) origin there too, so for this world the local ENU frame
coincides with the Gazebo world XY plane (E = world x, N = world y). The marker
position is therefore published unchanged as the ENU cue.

Moving the model uses gz-transport's world `set_pose` service (teleport — no
physics needed for a static ground decal). If the gz Python bindings are not
importable the node degrades to publish-only (useful for the "static marker,
virtual cue" case, or replaying a real external publisher).

Topic / service contract:
  out (ROS)  <marker_topic>                 geometry_msgs/PointStamped  ENU cue
  call (gz)  /world/<world>/set_pose        gz.msgs.Pose -> gz.msgs.Boolean

Trajectory patterns (param `pattern`):
  static : stay at (center_e, center_n)
  line   : constant-speed straight back-and-forth along East, centred on
           (center_e, center_n) — see _position() for the exact triangle-wave
           profile (not a literal sine; velocity magnitude is constant).
  circle : e = center_e + amplitude·cos(w t),  n = center_n + amplitude·sin(w t)
  cross  : constant-speed "+"-shaped patrol — out-and-back along +East, then
           -East, then +North, then -North (each leg length = amplitude),
           looping. Exercises BOTH the drone's East-West and North-South
           obstacle-avoidance legs (line only exercises East-West), so the
           obstacle map's clear lane must stay clear along both axes.
  where w = speed / amplitude (rad/s) so `speed` is the path speed in m/s.
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, Vector3Stamped
from mavros_msgs.msg import State
from rcl_interfaces.msg import ParameterDescriptor

# gz-transport is optional: without it we still publish the ENU cue, we just
# can't physically move the Gazebo model (publish-only / external-mover mode).
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.pose_pb2 import Pose as GzPose
    from gz.msgs10.boolean_pb2 import Boolean as GzBoolean
    from gz.msgs10.empty_pb2 import Empty as GzEmpty
    from gz.msgs10.twist_pb2 import Twist as GzTwist
    from gz.msgs10.pose_v_pb2 import Pose_V as GzPoseV
    _GZ_OK = True
except Exception as _e:        # pragma: no cover - depends on host gz install
    _GZ_OK = False
    _GZ_ERR = _e


class MovingMarkerNode(Node):
    def __init__(self):
        super().__init__('moving_marker_node')

        # --- Parameters -----------------------------------------------------
        self.world = self.declare_parameter('world', 'iris_down_camera_runway').value
        self.model = self.declare_parameter('model', 'aruco_marker_0').value
        self.marker_topic = self.declare_parameter('marker_topic', '/marker/position').value
        self.vel_topic = self.declare_parameter('vel_topic', '/marker/velocity').value
        self.rate = self.declare_parameter('rate', 50.0).value          # Hz (higher
        #   keeps the teleport step small at high speed: step = speed / rate)
        self.pattern = self.declare_parameter('pattern', 'line').value  # static|line|circle|cross
        self.center_e = self.declare_parameter('center_e', 1.0).value   # m, ENU East
        self.center_n = self.declare_parameter('center_n', 0.0).value   # m, ENU North
        self.amplitude = self.declare_parameter('amplitude', 1.5).value  # m
        self.speed = self.declare_parameter('speed', 0.3).value         # m/s path speed
        # 'cross' + drive_mode=velocity only: closed-loop P-servo gain/arrive-radius
        # toward each "+"-shape waypoint, using the ACTUAL gz pose as feedback (see
        # _cross_velocity_closed_loop). Replaces the old open-loop analytic velocity
        # command, which assumed the physical model tracks a commanded velocity with
        # zero lag -- with real mass/VelocityControl dynamics it doesn't, so an
        # instantaneous open-loop direction flip (in-arm reversal at each amplitude
        # extreme, or the 90-degree axis switch passing through centre) overshoots
        # past the intended turn point. Closed-loop feedback decelerates smoothly
        # approaching each waypoint (proportional term shrinks) and self-corrects any
        # drift every tick, bounding the overshoot regardless of the exact plugin
        # dynamics (observed: still colliding with obstacles at the old open-loop
        # scheme even after widening clear_lane_half_width once already).
        self.cross_kp = self.declare_parameter('cross_kp', 0.6).value          # 1/s
        self.cross_arrive_radius = self.declare_parameter(
            'cross_arrive_radius', 1.0).value                                  # m
        # Hard acceleration cap on the OUTPUT command, on top of the P-servo
        # above. The P-servo smooths deceleration approaching a waypoint (its
        # proportional term naturally shrinks near the target) but NOT the
        # departure: the instant the target switches to the next waypoint,
        # the remaining distance is suddenly large again, so the P-servo's
        # desired velocity jumps toward full speed in the new direction
        # within a single tick. A PID-style feedback law can't fix this on
        # its own -- P/I/D all react to the CURRENT error, which itself has a
        # step discontinuity at a waypoint switch, so their output inherits
        # that jump (D on a step error is actually worse: "derivative kick").
        # Rate-limiting the OUTPUT itself is what guarantees smooth accel in
        # both directions, independent of the servo gain -- clamp the
        # per-tick change in commanded velocity to cross_accel_max * dt.
        self.cross_accel_max = self.declare_parameter(
            'cross_accel_max', 2.0).value                                     # m/s^2
        self._cross_cmd_vel = [0.0, 0.0]    # last rate-limited output (state across ticks)
        # Hold STATIONARY at the start position for this many seconds before the
        # trajectory begins. For the mounted-on-moving-truck scenario the drone
        # rides the truck and must sit still long enough for its EKF/GPS to settle
        # and arm; only then does the truck drive off (and the drone launches WHILE
        # moving). 0 = start moving immediately (legacy landing-demo behaviour).
        # dynamic_typing so a bare integer override (e.g. start_delay:=20) is
        # accepted, not rejected as "INTEGER, expecting DOUBLE"; coerce to float.
        self.start_delay = float(self.declare_parameter(
            'start_delay', 0.0,
            ParameterDescriptor(dynamic_typing=True)).value)  # s
        self.z = self.declare_parameter('z', 0.002).value               # m, ground height
        self.frame_id = self.declare_parameter('frame_id', 'map').value
        # move_model=false → publish the cue only, leave the Gazebo model alone.
        self.move_model = self.declare_parameter('move_model', True).value
        # HOW to move the model:
        #   velocity : stream the trajectory VELOCITY as gz.msgs.Twist to the model's
        #              VelocityControl plugin. Real momentum, so a fix-jointed drone
        #              is carried and lifts off cleanly when released (mounted truck).
        #   teleport : set_pose every tick (a kinematic decal; legacy flat-marker
        #              demo where the model has no VelocityControl plugin).
        #   external : something ELSE drives the model (PX4's own
        #              MovingPlatformController plugin on moving_platform_aruco,
        #              see gazebo/worlds/truck_mission_px4.sdf) -- this node only
        #              relays its actual gz pose as the cue (position + finite-
        #              differenced velocity), commands nothing, and ignores
        #              move_model/release_on_arm (that model has no
        #              DetachableJoint; the drone just rests on its collision
        #              surface via gravity until it lifts off under its own thrust).
        self.drive_mode = self.declare_parameter('drive_mode', 'velocity').value
        self.cmd_vel_topic = self.declare_parameter(
            'cmd_vel_topic', '/model/aruco_platform/cmd_vel').value
        # DetachableJoint release: the drone rides the truck secured by a fixed
        # joint (declared in aruco_platform/model.sdf). The instant the drone arms
        # for takeoff we publish gz.msgs.Empty on this gz topic to sever the joint,
        # freeing it to lift off FROM the moving truck. release_on_arm=false leaves
        # the joint alone (e.g. the legacy ground-launch world has no joint).
        self.detach_topic = self.declare_parameter(
            'detach_topic', '/aruco_platform/detach').value
        self.release_on_arm = self.declare_parameter('release_on_arm', True).value

        # angular rate so |velocity| == speed along the path
        self.w = (self.speed / self.amplitude) if self.amplitude > 1e-6 else 0.0

        self.pub = self.create_publisher(PointStamped, self.marker_topic, 10)
        # Publish the marker's VELOCITY explicitly (ENU vE, vN) so the controller
        # feeds it forward directly instead of finite-differencing the position
        # (no lag/noise → tighter velocity match → the drone stays level and
        # lands cleanly on the moving platform). Derived from the node's own
        # successive positions, so it is exact for any pattern and never hard-codes
        # a speed.
        self.vel_pub = self.create_publisher(Vector3Stamped, self.vel_topic, 10)
        self._prev_traj = None   # (e, n, t) analytic target, for the drive velocity
        self._actual = None      # (x, y) latest ACTUAL truck pose from gz (vel mode)
        self._actual_prev = None  # (e, n, t) previous actual pose, for external-mode finite diff

        # "+"-shape waypoints for the closed-loop 'cross' driver: East tip, centre,
        # North tip, centre, West tip, centre, South tip, centre -- looped. Visiting
        # order differs from the old open-loop _cross_position (which did a full
        # out-and-back on one axis before switching), but covers the same 4 arms and
        # keeps every leg exactly axis-aligned, which is what obstacle_map.yaml's
        # clear_lane assumes.
        A = self.amplitude
        self._cross_wps = [
            (self.center_e + A, self.center_n), (self.center_e, self.center_n),
            (self.center_e, self.center_n + A), (self.center_e, self.center_n),
            (self.center_e - A, self.center_n), (self.center_e, self.center_n),
            (self.center_e, self.center_n - A), (self.center_e, self.center_n),
        ]
        self._cross_idx = 0

        # --- gz mover -------------------------------------------------------
        self.gz = None
        self.detach_pub = None
        self.cmd_vel_pub = None
        self.set_pose_srv = f'/world/{self.world}/set_pose'
        if (self.move_model or self.release_on_arm or self.drive_mode == 'external'):
            if _GZ_OK:
                self.gz = GzNode()
                if self.move_model:
                    if self.drive_mode == 'velocity':
                        self.cmd_vel_pub = self.gz.advertise(self.cmd_vel_topic, GzTwist)
                        # In velocity mode the analytic target drifts from the truck
                        # when sim runs slower than wall-clock (velocity integrates in
                        # sim time, the target advances in wall time), so read the
                        # ACTUAL truck pose from gz and report THAT as the cue.
                        pose_topic = f'/world/{self.world}/dynamic_pose/info'
                        self.gz.subscribe(GzPoseV, pose_topic, self._gz_pose_cb)
                        self.get_logger().info(
                            f'driving model "{self.model}" by VELOCITY via {self.cmd_vel_topic}; '
                            f'cue from actual pose ({pose_topic})')
                    else:
                        self.get_logger().info(
                            f'moving model "{self.model}" via {self.set_pose_srv} (teleport)')
                elif self.drive_mode == 'external':
                    # Not driving anything ourselves -- just relay the actual
                    # pose (some other plugin, e.g. PX4's MovingPlatformController,
                    # moves the model).
                    pose_topic = f'/world/{self.world}/dynamic_pose/info'
                    self.gz.subscribe(GzPoseV, pose_topic, self._gz_pose_cb)
                    self.get_logger().info(
                        f'external drive mode: relaying actual pose of "{self.model}" '
                        f'from {pose_topic} as the cue (not commanding movement)')
                if self.release_on_arm:
                    self.detach_pub = self.gz.advertise(self.detach_topic, GzEmpty)
                    self.get_logger().info(
                        f'will release DetachableJoint via {self.detach_topic} on arm')
            else:
                self.get_logger().warn(
                    f'gz Python bindings unavailable ({_GZ_ERR}); '
                    'publishing cue only (model will NOT move / joint NOT released).')

        # Watch the FCU arm state so we can sever the DetachableJoint the instant
        # the drone arms for takeoff (rising edge false→true), releasing it from
        # the moving truck. Sim-only glue lives here (not in mission_manager) to
        # keep the mission brain hardware-portable.
        self._armed = False
        self._released = False
        if self.release_on_arm:
            self.create_subscription(State, '/mavros/state', self._state_cb, 10)

        self.get_logger().info(
            f'pattern={self.pattern} center=({self.center_e:.2f},{self.center_n:.2f}) '
            f'amp={self.amplitude:.2f} speed={self.speed:.2f} -> {self.marker_topic}')

        self.t0 = self.get_clock().now()
        self.create_timer(1.0 / self.rate, self.tick)

    # -----------------------------------------------------------------------
    def _traj_time(self):
        """Elapsed trajectory time (s), possibly NEGATIVE during the
        stationary start_delay warm-up -- callers that need to know "are we
        still in warm-up" (e.g. _cross_velocity) want the unclamped value;
        _position clamps it to 0 itself."""
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9 - self.start_delay

    def _position(self):
        """Marker (E, N) at the current time for the configured pattern."""
        # Stationary warm-up: stay put until start_delay has elapsed, then run the
        # trajectory from t=0 (so motion begins smoothly from the start position).
        t = max(self._traj_time(), 0.0)
        if self.pattern == 'static':
            return self.center_e, self.center_n
        if self.pattern == 'circle':
            return (self.center_e + self.amplitude * math.cos(self.w * t),
                    self.center_n + self.amplitude * math.sin(self.w * t))
        if self.pattern == 'cross':
            return self._cross_position(t)
        # default: line — CONSTANT-SPEED straight runs along East (a triangle wave,
        # not a sin: the velocity is constant in magnitude so the drone's velocity
        # feed-forward matches it exactly and it lands cleanly). The platform runs
        # straight out to +amplitude, reverses, runs straight to −amplitude and
        # back — one continuous straight leg is 2·amplitude metres long.
        if self.amplitude < 1e-6:
            return self.center_e, self.center_n
        A = self.amplitude
        phase = (self.speed * t) % (4.0 * A)     # distance into the back-and-forth
        if phase < A:
            d = phase                            # centre → +A
        elif phase < 3.0 * A:
            d = 2.0 * A - phase                  # +A → −A  (the long 2A straight leg)
        else:
            d = phase - 4.0 * A                  # −A → centre
        return self.center_e + d, self.center_n

    def _cross_position(self, t):
        """"+"-shaped patrol: four constant-speed out-and-back legs in order
        +East, -East, +North, -South -- each leg is centre->amplitude->centre
        (length 2A), so the full loop period is 8A. Direction changes only at
        arm boundaries (centre), matching the same constant-speed-with-
        direction-reversal style as the 'line' pattern above."""
        if self.amplitude < 1e-6:
            return self.center_e, self.center_n
        A = self.amplitude
        period = 8.0 * A
        phase = (self.speed * t) % period
        arm = int(phase // (2.0 * A))            # 0:+E 1:-E 2:+N 3:-N
        local = phase - arm * (2.0 * A)          # 0..2A within this arm
        d = local if local < A else (2.0 * A - local)   # centre->A->centre
        if arm == 0:
            return self.center_e + d, self.center_n
        elif arm == 1:
            return self.center_e - d, self.center_n
        elif arm == 2:
            return self.center_e, self.center_n + d
        else:
            return self.center_e, self.center_n - d

    def _cross_velocity(self, raw_t):
        """Exact instantaneous velocity for _cross_position(), NOT finite-
        differenced. Finite-differencing two consecutive _position() samples
        breaks specifically for 'cross': at the ~1 control tick that straddles
        an arm boundary (e.g. -East ending -> +North starting), the two
        samples come from DIFFERENT axes, so (e-pe)/dt, (n-pn)/dt spuriously
        blends both into one diagonal velocity pulse for that tick. Gazebo
        integrates that pulse into REAL position before the next tick's
        (correct) velocity zeroes the spurious axis again -- the position kick
        does NOT undo. Over many arm switches this accumulates into a
        steadily growing off-axis offset (observed: the platform drifting off
        the intended "+" shape and eventually clipping an obstacle that was
        placed assuming exact axis alignment). Always exactly axis-aligned by
        construction, so this bug class cannot occur."""
        if raw_t < 0.0 or self.amplitude < 1e-6:
            return 0.0, 0.0
        A = self.amplitude
        period = 8.0 * A
        phase = (self.speed * raw_t) % period
        arm = int(phase // (2.0 * A))
        local = phase - arm * (2.0 * A)
        sign = 1.0 if local < A else -1.0   # centre->+A leg vs +A->centre leg
        if arm == 0:
            return sign * self.speed, 0.0
        elif arm == 1:
            return -sign * self.speed, 0.0
        elif arm == 2:
            return 0.0, sign * self.speed
        else:
            return 0.0, -sign * self.speed

    def _cross_velocity_closed_loop(self):
        """Closed-loop replacement for _cross_velocity (drive_mode='velocity'
        only): P-servo toward self._cross_wps[self._cross_idx] using the actual
        gz pose as feedback, advancing to the next waypoint on arrival. Speed is
        capped at self.speed and decays proportionally near the target, so the
        commanded velocity reaches ~0 right at each turn instead of flipping
        sign/axis instantaneously -- see the cross_kp/cross_arrive_radius
        declare_parameter comment for why. The RETURNED value is additionally
        hard-rate-limited to cross_accel_max (see that parameter's comment):
        the P-servo above smooths deceleration approaching a waypoint but not
        departure, since the moment the target switches the desired velocity
        jumps toward full speed in the new direction within one tick."""
        if self._traj_time() < 0.0:
            v_desired_e, v_desired_n = 0.0, 0.0   # stationary start_delay warm-up
        else:
            cur_e, cur_n = self._actual if self._actual is not None else \
                (self.center_e, self.center_n)
            target = self._cross_wps[self._cross_idx]
            err_e, err_n = target[0] - cur_e, target[1] - cur_n
            dist = math.hypot(err_e, err_n)
            if dist < self.cross_arrive_radius:
                self._cross_idx = (self._cross_idx + 1) % len(self._cross_wps)
                target = self._cross_wps[self._cross_idx]
                err_e, err_n = target[0] - cur_e, target[1] - cur_n
                dist = math.hypot(err_e, err_n)
            if dist < 1e-6:
                v_desired_e, v_desired_n = 0.0, 0.0
            else:
                speed_cmd = min(self.speed, self.cross_kp * dist)
                v_desired_e, v_desired_n = speed_cmd * err_e / dist, speed_cmd * err_n / dist

        dt = 1.0 / self.rate
        max_step = self.cross_accel_max * dt
        cmd_e, cmd_n = self._cross_cmd_vel
        d_e, d_n = v_desired_e - cmd_e, v_desired_n - cmd_n
        # Clamp the STEP VECTOR's magnitude, not each axis independently --
        # clamping components separately lets the combined magnitude reach
        # sqrt(2)*max_step whenever both axes need to change at once (exactly
        # what happens right at the 90-degree centre switch: the old axis's
        # error is falling to 0 while the new axis's error is rising from 0,
        # both at once) -- found by a closed-loop simulation checking the
        # ACTUAL |accel| every tick, not by inspection.
        d_mag = math.hypot(d_e, d_n)
        if d_mag > max_step and d_mag > 1e-9:
            scale = max_step / d_mag
            d_e *= scale
            d_n *= scale
        self._cross_cmd_vel = [cmd_e + d_e, cmd_n + d_n]
        return tuple(self._cross_cmd_vel)

    def tick(self):
        now = self.get_clock().now()
        stamp = now.to_msg()

        if self.drive_mode == 'external':
            self._tick_external(now, stamp)
            return

        e, n = self._position()       # analytic trajectory TARGET

        if self.pattern == 'cross' and self.drive_mode == 'velocity':
            # Closed-loop P-servo toward the current "+"-shape waypoint using the
            # ACTUAL gz pose (see _cross_velocity_closed_loop) -- decelerates
            # approaching each turn instead of flipping direction open-loop, so
            # real momentum/plugin dynamics can't overshoot past it.
            vx, vy = self._cross_velocity_closed_loop()
        elif self.pattern == 'cross':
            # Teleport mode has no physics/momentum, so the open-loop analytic
            # velocity (see _cross_velocity docstring) is exact and simpler.
            vx, vy = self._cross_velocity(self._traj_time())
        else:
            # Drive velocity = analytic trajectory derivative via finite
            # difference. Fine for line/circle/static: they never switch
            # axes, so two consecutive samples never blend different
            # directions the way 'cross' can.
            vx = vy = 0.0
            if self._prev_traj is not None:
                pe, pn, pt = self._prev_traj
                dt = (now - pt).nanoseconds * 1e-9
                if dt > 1e-4:
                    vx = (e - pe) / dt
                    vy = (n - pn) / dt
        self._prev_traj = (e, n, now)

        # Move the model: real velocity (carries the jointed drone with momentum so
        # it lifts off cleanly on release) or legacy teleport.
        if self.gz is not None:
            if self.drive_mode == 'velocity':
                self._set_model_velocity(vx, vy)
            else:
                self._set_model_pose(e, n)

        # Cue = where the truck ACTUALLY is. Velocity mode: report the actual gz
        # pose (the analytic target drifts under real-time-factor < 1, since the
        # velocity integrates in sim time but the target advances in wall time).
        # Teleport mode: actual == analytic, so the analytic target is exact.
        if self.drive_mode == 'velocity' and self._actual is not None:
            cue_e, cue_n = self._actual
        else:
            cue_e, cue_n = e, n

        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.point.x = cue_e    # ENU East  (== Gazebo world x)
        msg.point.y = cue_n    # ENU North (== Gazebo world y)
        msg.point.z = 0.0
        self.pub.publish(msg)

        # Cue velocity = the (smooth) commanded velocity, which the truck tracks.
        vel = Vector3Stamped()
        vel.header.stamp = stamp
        vel.header.frame_id = self.frame_id
        vel.vector.x = vx
        vel.vector.y = vy
        self.vel_pub.publish(vel)

    def _tick_external(self, now, stamp):
        """drive_mode='external': relay the actual gz pose of a model driven by
        something else (PX4's MovingPlatformController) as the cue -- publish
        nothing until the first pose arrives, then finite-difference successive
        actual positions for the velocity cue (no analytic trajectory/commanded
        velocity exists in this mode, unlike 'velocity'/'teleport')."""
        if self._actual is None:
            return
        e, n = self._actual
        vx = vy = 0.0
        if self._actual_prev is not None:
            pe, pn, pt = self._actual_prev
            dt = (now - pt).nanoseconds * 1e-9
            if dt > 1e-4:
                vx = (e - pe) / dt
                vy = (n - pn) / dt
        self._actual_prev = (e, n, now)

        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.point.x = e
        msg.point.y = n
        msg.point.z = 0.0
        self.pub.publish(msg)

        vel = Vector3Stamped()
        vel.header.stamp = stamp
        vel.header.frame_id = self.frame_id
        vel.vector.x = vx
        vel.vector.y = vy
        self.vel_pub.publish(vel)

    def _gz_pose_cb(self, msg):
        # Latest ACTUAL pose of the truck model (gz dynamic_pose/info, world frame).
        for p in msg.pose:
            if p.name == self.model:
                self._actual = (p.position.x, p.position.y)
                return

    def _set_model_velocity(self, vx, vy):
        if self.cmd_vel_pub is None:
            return
        # Truck has no yaw, so world ENU velocity maps straight to the Twist.
        twist = GzTwist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.linear.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def _set_model_pose(self, e, n):
        req = GzPose()
        req.name = self.model
        req.position.x = float(e)
        req.position.y = float(n)
        req.position.z = float(self.z)
        req.orientation.w = 1.0
        # Blocking request with a short timeout; set_pose returns quickly.
        ok, _rep = self.gz.request(self.set_pose_srv, req, GzPose, GzBoolean, 100)
        if not ok:
            # Throttle: don't spam if the world/service isn't up yet.
            self.get_logger().warn(
                f'set_pose to {self.set_pose_srv} failed (is Gazebo running?)',
                throttle_duration_sec=5.0)

    def _state_cb(self, msg):
        # Release the DetachableJoint once, on the disarmed→armed rising edge.
        if msg.armed and not self._armed and not self._released:
            self._release_joint()
        self._armed = msg.armed

    def _release_joint(self):
        if self.detach_pub is None:
            return
        # gz DetachableJoint severs on ANY message to its <topic>; send Empty.
        self.detach_pub.publish(GzEmpty())
        self._released = True
        self.get_logger().info(
            f'drone armed -> released DetachableJoint ({self.detach_topic})')


def main(args=None):
    rclpy.init(args=args)
    node = MovingMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
