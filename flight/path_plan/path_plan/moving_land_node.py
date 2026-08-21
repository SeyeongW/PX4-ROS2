#!/usr/bin/env python3
"""Terminal landing onto the MOVING trailer — self-contained landing logic.

This is the single file the pursuit hands off to for the terminal land. It owns ALL
of its own tuning (declared with defaults below), so the launch file adds it with NO
parameter block — at landing time you "just load this file". While it holds authority
it is the ONLY node streaming setpoints (the pursuit bridge stops when it latches the
handoff gate), so there is never a setpoint fight.

Where the moving target comes from is deliberately behind a small PLUGGABLE estimator
so the landing behaviour can later switch from the broadcast cue to on-board ArUco
vision by editing ONLY this file:

    TargetEstimator.estimate(now) -> (E, N, vE, vN, valid)   # map-ENU pos + velocity

  * CueVelocityEstimator  (used now): reads the trailer's broadcast /trailer/position
    and /trailer/velocity. Velocity-based estimate — the drone matches the trailer's
    velocity while converging, so it can settle over a target that keeps moving.
  * ArucoEstimator        (later): documented stub that will fuse the down-camera
    ArUco offset with the drone pose. Swap it in at ``_make_estimator`` — nothing
    else in the node, and no launch/param change, has to move.

Control: a first-order velocity servo that matches the target — v = v_target_ff + kp*e
— plus a descent funnel (wide up high, narrowing to the deck) computed from the height
ABOVE the deck. When low and centred over the deck it FORCE-disarms so the drone settles
onto the moving platform (AUTO.LAND would descend vertically and miss a moving target).

Frames: the drone position comes from /path_plan/odometry, which the pursuit bridge
already republishes in map ENU — the same frame the trailer cue lives in, so no offset
maths is needed here. Velocity setpoints go out on /mavros/setpoint_velocity/cmd_vel
(the same topic/mechanism the bridge used), and world-ENU velocity is translation-
invariant, so PX4 stays in OFFBOARD across the handoff with no frame juggling.

Topic contract:
  in  /pursuit/land_enable      std_msgs/Bool (latched)   landing-authority gate
  in  /path_plan/odometry       nav_msgs/Odometry         drone state (map ENU)
  in  /mavros/state             mavros_msgs/State
  in  /trailer/position         geometry_msgs/PointStamped   (CueVelocityEstimator)
  in  /trailer/velocity         geometry_msgs/Vector3Stamped (CueVelocityEstimator)
  out /mavros/setpoint_velocity/cmd_vel  geometry_msgs/TwistStamped  OFFBOARD setpoint
  out /moving_land/debug        std_msgs/String            status
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)

from geometry_msgs.msg import PointStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, String
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandLong

from .ros_msgs import positions_to_path


DT = 0.05  # 20 Hz control period


# ---------------------------------------------------------------------------
# Pluggable target estimators. Each returns the moving target's map-ENU position
# and velocity (feed-forward) plus a validity flag. Swap the one built in
# _make_estimator to change how the landing tracks the trailer — nothing else moves.
# ---------------------------------------------------------------------------
class TargetEstimator:
    def estimate(self, now):
        """Return (E, N, vE, vN, valid) in map ENU. valid=False -> coast/hold."""
        raise NotImplementedError


class CueVelocityEstimator(TargetEstimator):
    """Velocity-based estimate from the trailer's broadcast cue (used now).

    Tracks the last /trailer/position and /trailer/velocity. The velocity is fed
    forward so the drone matches the trailer's motion while the proportional term
    closes the remaining offset — the key to settling over a target that keeps moving.
    """

    def __init__(self, node: Node, pos_topic: str, vel_topic: str, timeout: float):
        self._node = node
        self._timeout = timeout
        self._pos = None
        self._pos_stamp = None
        self._vel = [0.0, 0.0]
        self._vel_stamp = None
        node.create_subscription(PointStamped, pos_topic, self._pos_cb, 10)
        node.create_subscription(Vector3Stamped, vel_topic, self._vel_cb, 10)

    def _pos_cb(self, msg: PointStamped):
        self._pos = [msg.point.x, msg.point.y]
        self._pos_stamp = self._node.get_clock().now()

    def _vel_cb(self, msg: Vector3Stamped):
        self._vel = [msg.vector.x, msg.vector.y]
        self._vel_stamp = self._node.get_clock().now()

    def _fresh(self, stamp, now):
        if stamp is None:
            return False
        return (now - stamp).nanoseconds * 1e-9 < self._timeout

    def estimate(self, now):
        if self._pos is None or not self._fresh(self._pos_stamp, now):
            return (0.0, 0.0, 0.0, 0.0, False)
        vE, vN = self._vel if self._fresh(self._vel_stamp, now) else (0.0, 0.0)
        return (self._pos[0], self._pos[1], vE, vN, True)


class ArucoEstimator(TargetEstimator):  # pragma: no cover - future vision path
    """Vision estimate from the down-camera ArUco marker (future swap-in).

    Placeholder for the later on-board-vision landing: it would subscribe to
    /perception/aruco_offset + /perception/aruco_detected and the drone pose, project
    the normalised image offset to a world (E, N) point at the height above the deck,
    and run a constant-velocity Kalman filter to produce (E, N, vE, vN) — see
    precision_landing_node._measure_marker_world / the KF there for the exact maths.
    Wire it up in _make_estimator when the camera path is enabled; no other change is
    needed in this node, and the launch/params stay the same.
    """

    def __init__(self, node: Node):
        self._node = node
        node.get_logger().warn(
            "ArucoEstimator is a stub; falling back to no valid target. "
            "Implement the vision fusion here to enable camera-based landing.")

    def estimate(self, now):
        return (0.0, 0.0, 0.0, 0.0, False)


class Stage(Enum):
    DORMANT = 0     # not enabled: stream nothing (bridge still owns setpoints)
    LANDING = 1     # enabled: track + descend onto the moving deck
    DONE = 2        # force-disarmed onto the platform


class MovingLandNode(Node):
    def __init__(self):
        super().__init__("moving_land_node")
        p = self.declare_parameter

        # --- ALL tuning lives here (the launch passes nothing) ----------------
        # Target selection: which estimator to use. 'cue' = velocity-based broadcast
        # cue (now); 'aruco' = on-board vision (future). Change the default to switch.
        self.estimator_mode = str(p("estimator", "cue").value)
        # Cue topics (CueVelocityEstimator) + freshness.
        self.pos_topic = str(p("position_topic", "/trailer/position").value)
        self.vel_topic = str(p("velocity_topic", "/trailer/velocity").value)
        self.cue_timeout = float(p("cue_timeout", 1.0).value)
        # Handoff + drone-state topics.
        self.enable_topic = str(p("enable_topic", "/pursuit/land_enable").value)
        self.odom_topic = str(p("odom_topic", "/path_plan/odometry").value)
        # Horizontal velocity servo (v = v_target_ff + vel_gain * error, clamped).
        # The final phase uses the gentler land_vel_gain so it doesn't snap violently
        # at the moving deck (the earlier 0.9 gain spiked accel to ~35 m/s^2).
        self.vel_gain = float(p("vel_gain", 0.9).value)          # 1/s (approach)
        self.land_vel_gain = float(p("land_vel_gain", 0.45).value)  # 1/s (final, gentle)
        self.vel_max = float(p("vel_max", 10.0).value)           # m/s (> trailer speed)
        # Slew-rate (acceleration) limit on the published setpoint so neither the final
        # servo nor the glide->final handoff commands a violent jerk. Bounds |dv|/tick.
        self.accel_limit = float(p("accel_limit", 6.0).value)    # m/s^2
        # Deck geometry. The marker/deck sits platform_height above the ground; all
        # vertical logic uses the height ABOVE the deck.
        self.platform_height = float(p("platform_height", 1.25).value)  # m (trailer deck)
        self.land_align_radius = float(p("land_align_radius", 0.35).value)  # m
        self.descend_rate = float(p("descend_rate", 0.8).value)   # m/s, gentle final descent
        self.descend_min_scale = float(p("descend_min_scale", 0.3).value)
        self.land_clearance = float(p("land_clearance", 0.25).value)  # m above deck -> disarm
        # Below this height above the deck, finish straight down on the cue centre.
        self.final_descent_h = float(p("final_descent_h", 1.0).value)
        # --- Predicted-intercept descent via the REAL B-spline pipeline -----------
        # The drone is faster than the trailer, so it aims at where the trailer WILL be
        # (constant-velocity prediction: pos + vel*T_go) and descends there on a smooth
        # 3D curve. Rather than a hand-rolled glide, this feeds a drone->intercept-deck
        # global_path straight into the existing ego-planner bspline_node (its ground
        # floor is now 0 so the curve can reach the deck) and lets mpc_node track it;
        # we forward that MPC command as our setpoint (single authority). Regenerated at
        # descent_period, not every tick (L-BFGS is too heavy for 20 Hz). No KF is needed
        # while the trailer velocity is broadcast; a KF slots behind the estimator only
        # once the target comes from noisy on-board vision.
        self.approach_speed = float(p("approach_speed", 9.0).value)   # m/s used to solve T_go
        self.min_time_to_go = float(p("min_time_to_go", 1.0).value)   # s, T_go floor
        self.descent_period = float(p("descent_replan_period", 1.0).value)  # s, spline refresh
        self.global_path_topic = str(p("global_path_topic", "/path_plan/global_path").value)
        self.mpc_cmd_topic = str(p("mpc_cmd_topic", "/path_plan/cmd_vel").value)

        # --- State ------------------------------------------------------------
        self.stage = Stage.DORMANT
        self.land_enabled = False
        self.mav_state = State()
        self.pos = [0.0, 0.0, 0.0]     # drone map-ENU position
        self.cmd_vel = [0.0, 0.0, 0.0]
        self.drone_vel = [0.0, 0.0, 0.0]   # latest odom world velocity
        self._prev_pub = [0.0, 0.0, 0.0]   # last published setpoint (for slew limiting)
        self.dbg_tick = 0
        self._mpc_cmd = None           # latest mpc_node cmd (tracking the descent spline)
        self._last_descent = None      # time of the last descent global_path publish

        # --- Publishers -------------------------------------------------------
        self.sp_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10)
        self.debug_pub = self.create_publisher(String, "/moving_land/debug", 10)
        latched = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # Descent global_path -> bspline_node (bypasses A*: the terminal descent is a
        # short open-air diagonal above the trailer, no grid search needed).
        self.path_pub = self.create_publisher(Path, self.global_path_topic, latched)

        # --- Subscribers ------------------------------------------------------
        self.create_subscription(State, "/mavros/state", self._state_cb, 10)
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(Bool, self.enable_topic, self._enable_cb, latched)
        # mpc_node's output (it tracks the descent spline we publish); we forward it.
        self.create_subscription(
            TwistStamped, self.mpc_cmd_topic, self._mpc_cmd_cb, 10)

        # --- Target estimator (the pluggable swap point) ----------------------
        self.estimator = self._make_estimator()

        # --- Service client ---------------------------------------------------
        self.command_cli = self.create_client(CommandLong, "/mavros/cmd/command")

        self.create_timer(DT, self.tick)
        self.get_logger().info(
            f"moving_land ready (DORMANT); estimator='{self.estimator_mode}', "
            f"deck={self.platform_height:.2f} m, waiting for {self.enable_topic}")

    # -----------------------------------------------------------------------
    def _make_estimator(self) -> TargetEstimator:
        # Single swap point: change 'cue' -> 'aruco' (or the default above) to move
        # the whole landing onto on-board vision, with no other edits here or in the
        # launch file.
        if self.estimator_mode == "aruco":
            return ArucoEstimator(self)
        return CueVelocityEstimator(
            self, self.pos_topic, self.vel_topic, self.cue_timeout)

    # -------------------------------------------------------------- callbacks
    def _state_cb(self, msg: State):
        self.mav_state = msg

    def _odom_cb(self, msg: Odometry):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        self.pos[2] = msg.pose.pose.position.z
        v = msg.twist.twist.linear
        self.drone_vel = [v.x, v.y, v.z]     # for seeding the slew limiter at handoff

    def _enable_cb(self, msg: Bool):
        self.land_enabled = msg.data

    def _mpc_cmd_cb(self, msg: TwistStamped):
        self._mpc_cmd = msg.twist

    # -------------------------------------------------------------- main loop
    def tick(self):
        if not self.mav_state.connected:
            return

        # Authority gate: stay DORMANT and stream NOTHING until the pursuit hands off,
        # and hand authority straight back if the gate is revoked.
        if not self.land_enabled:
            if self.stage == Stage.LANDING:
                self._log("land_enable revoked -> DORMANT")
            self.stage = Stage.DORMANT
            return
        if self.stage == Stage.DORMANT:
            self._log("land_enable -> LANDING (taking setpoint authority)")
            self.stage = Stage.LANDING
            # Seed the slew limiter with the drone's current velocity so the first
            # setpoint continues the cruise motion instead of braking to zero.
            self._prev_pub = list(self.drone_vel)

        if self.stage == Stage.DONE:
            if self.mav_state.armed:
                self._force_disarm()        # keep re-sending until it takes
            return

        # ----- LANDING: track the moving target and descend onto the deck -----
        self.cmd_vel = [0.0, 0.0, 0.0]
        now = self.get_clock().now()
        tE, tN, vE, vN, valid = self.estimator.estimate(now)

        if not valid:
            # No target this tick: hold position + altitude (coast) rather than drift.
            self._publish_velocity()
            self.dbg_tick += 1
            if self.dbg_tick % 20 == 0:
                self._log("target invalid -> holding")
            return

        err = math.hypot(tE - self.pos[0], tN - self.pos[1])   # true offset NOW
        h = self._height_above_deck()

        if h < self.final_descent_h:
            # Final phase: aim at the trailer's ACTUAL position (no lead) so the drone
            # centres on the moving deck and settles, matching its velocity. Use the
            # gentle land gain so it eases in instead of snapping.
            self._servo_to(tE, tN, vE, vN, gain=self.land_vel_gain)
            t_go = 0.0
            if err < self.land_align_radius:
                if h < self.land_clearance:
                    self._log(f"centred ({err:.2f} m) over deck at h={h:.2f} m "
                              f"-> DONE (disarm)")
                    self.stage = Stage.DONE
                    return
                self.cmd_vel[2] = -self.descend_rate * self.descend_min_scale
            else:
                self.cmd_vel[2] = 0.0        # hold height, converge onto the deck
        else:
            # Glide phase: descend on a REAL B-spline. Publish a drone->intercept-deck
            # global_path (refreshed at descent_period) into bspline_node, and FORWARD
            # mpc_node's tracking command as our setpoint. The predicted intercept
            # (pos + vel*t_go) leads the moving deck; the spline's z drops to it and
            # MPC's altitude hold follows it down.
            t_go, aimE, aimN = self._intercept(tE, tN, vE, vN)
            self._maybe_publish_descent(now, aimE, aimN)
            if self._mpc_cmd is not None:
                self.cmd_vel = [self._mpc_cmd.linear.x, self._mpc_cmd.linear.y,
                                self._mpc_cmd.linear.z]
            else:
                # MPC output not up yet: fall back to a direct servo toward the aim.
                self._servo_to(aimE, aimN, vE, vN)

        self.dbg_tick += 1
        if self.dbg_tick % 20 == 0:
            src = "mpc" if (h >= self.final_descent_h and self._mpc_cmd) else "servo"
            self._log(f"LANDING err={err:.2f} m h={h:.2f} m vz={self.cmd_vel[2]:+.2f} "
                      f"t_go={t_go:.1f}s src={src}")
        self._publish_velocity()

    def _maybe_publish_descent(self, now, aimE, aimN):
        """At descent_period, publish a drone->intercept-deck global_path so the
        existing bspline_node fits a smooth 3D descent spline for mpc_node to track."""
        if (self._last_descent is not None
                and (now - self._last_descent).nanoseconds * 1e-9 < self.descent_period):
            return
        self._last_descent = now
        pts = np.array([[self.pos[0], self.pos[1], self.pos[2]],
                        [aimE, aimN, self.platform_height]], float)
        self.path_pub.publish(positions_to_path(pts, now.to_msg()))

    # -------------------------------------------------------------- helpers
    def _servo_to(self, tE, tN, vffE=0.0, vffN=0.0, gain=None):
        """First-order velocity servo toward (tE, tN) with target-velocity feed-
        forward: v = v_ff + gain * error, clamped. Returns the horizontal error."""
        g = self.vel_gain if gain is None else gain
        eE = tE - self.pos[0]
        eN = tN - self.pos[1]
        self.cmd_vel[0] = self._clamp(vffE + g * eE, -self.vel_max, self.vel_max)
        self.cmd_vel[1] = self._clamp(vffN + g * eN, -self.vel_max, self.vel_max)
        return math.hypot(eE, eN)

    def _intercept(self, tE, tN, vE, vN):
        """Constant-velocity intercept: smallest T_go>0 such that flying straight at
        ``approach_speed`` from the drone reaches the trailer's predicted position
        ``(tE,tN) + (vE,vN)*T``. Solves |R + V T| = v T  ->  quadratic in T. Returns
        (T_go, aimE, aimN); falls back to a range/speed estimate if no real intercept
        exists (e.g. trailer momentarily faster than the drone)."""
        rx = tE - self.pos[0]
        ry = tN - self.pos[1]
        v = self.approach_speed
        a = vE * vE + vN * vN - v * v
        b = 2.0 * (rx * vE + ry * vN)
        c = rx * rx + ry * ry
        t = None
        if abs(a) < 1e-6:
            if abs(b) > 1e-6:
                t = -c / b
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sq = math.sqrt(disc)
                for cand in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
                    if cand > 1e-3 and (t is None or cand < t):
                        t = cand
        if t is None or t <= 0.0:
            t = math.hypot(rx, ry) / max(v, 0.1)     # can't formally intercept -> coast in
        t = max(t, self.min_time_to_go)
        return t, tE + vE * t, tN + vN * t

    def _height_above_deck(self):
        return max(self.pos[2] - self.platform_height, 0.05)

    def _publish_velocity(self):
        # Slew-rate limit: bound |dv| per tick to accel_limit*DT so the setpoint (final
        # servo AND the glide->final handoff) can't jerk. This is what tames the ~35
        # m/s^2 spikes seen at touchdown while leaving smooth commands untouched.
        dv_max = self.accel_limit * DT
        for i in range(3):
            self.cmd_vel[i] = self._clamp(self.cmd_vel[i],
                                          self._prev_pub[i] - dv_max,
                                          self._prev_pub[i] + dv_max)
        self._prev_pub = list(self.cmd_vel)
        sp = TwistStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"
        sp.twist.linear.x = self.cmd_vel[0]   # East
        sp.twist.linear.y = self.cmd_vel[1]   # North
        sp.twist.linear.z = self.cmd_vel[2]   # Up
        self.sp_pub.publish(sp)

    def _force_disarm(self):
        # MAV_CMD_COMPONENT_ARM_DISARM (400): param1=0 disarm, param2=21196 force
        # magic (bypasses the "still flying" check) so the drone settles on the deck.
        req = CommandLong.Request()
        req.command = 400
        req.param1 = 0.0
        req.param2 = 21196.0
        self.command_cli.call_async(req)

    def _log(self, text):
        self.get_logger().info(text)
        self.debug_pub.publish(String(data=text))

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))


def main(args=None):
    rclpy.init(args=args)
    node = MovingLandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
