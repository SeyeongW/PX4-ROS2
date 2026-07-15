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

import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)

from geometry_msgs.msg import PointStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandLong


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
        self.vel_gain = float(p("vel_gain", 0.9).value)          # 1/s
        self.vel_max = float(p("vel_max", 8.0).value)            # m/s
        # Deck geometry + descent funnel. The marker/deck sits platform_height above
        # the ground; all vertical logic uses the height ABOVE the deck.
        self.platform_height = float(p("platform_height", 1.25).value)  # m (trailer deck)
        self.land_align_radius = float(p("land_align_radius", 0.35).value)  # m, funnel floor
        self.descend_cone = float(p("descend_cone", 0.35).value)  # m error / m height
        self.descend_rate = float(p("descend_rate", 0.8).value)   # m/s downward
        self.descend_min_scale = float(p("descend_min_scale", 0.3).value)
        self.land_clearance = float(p("land_clearance", 0.25).value)  # m above deck -> disarm
        # Below this height above the deck, finish straight down on the cue centre.
        self.final_descent_h = float(p("final_descent_h", 1.0).value)

        # --- State ------------------------------------------------------------
        self.stage = Stage.DORMANT
        self.land_enabled = False
        self.mav_state = State()
        self.pos = [0.0, 0.0, 0.0]     # drone map-ENU position
        self.cmd_vel = [0.0, 0.0, 0.0]
        self.dbg_tick = 0

        # --- Publishers -------------------------------------------------------
        self.sp_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10)
        self.debug_pub = self.create_publisher(String, "/moving_land/debug", 10)

        # --- Subscribers ------------------------------------------------------
        self.create_subscription(State, "/mavros/state", self._state_cb, 10)
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, qos_profile_sensor_data)
        latched = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, self.enable_topic, self._enable_cb, latched)

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

    def _enable_cb(self, msg: Bool):
        self.land_enabled = msg.data

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

        # Horizontal: match the target velocity and close the offset.
        err = self._servo_to(tE, tN, vE, vN)

        # Vertical: descent funnel keyed on the height above the deck.
        h = self._height_above_deck()
        funnel = max(self.land_align_radius, self.descend_cone * h)
        if h < self.final_descent_h:
            # Near the deck: finish straight down once centred; converge first if not.
            if err < self.land_align_radius:
                if h < self.land_clearance:
                    self._log(f"centred ({err:.2f} m) over deck at h={h:.2f} m "
                              f"-> DONE (disarm)")
                    self.stage = Stage.DONE
                    return
                self.cmd_vel[2] = -self.descend_rate * self.descend_min_scale
            else:
                self.cmd_vel[2] = 0.0
        elif err <= funnel:
            # Inside the funnel: creep down, faster the more centred.
            scale = max(self.descend_min_scale, min(1.0, 1.0 - err / funnel))
            self.cmd_vel[2] = -self.descend_rate * scale
        else:
            self.cmd_vel[2] = 0.0            # too far off for this height: converge

        self.dbg_tick += 1
        if self.dbg_tick % 20 == 0:
            self._log(f"LANDING err={err:.2f} m h={h:.2f} m "
                      f"vz={self.cmd_vel[2]:+.2f} funnel={funnel:.2f}")
        self._publish_velocity()

    # -------------------------------------------------------------- helpers
    def _servo_to(self, tE, tN, vffE=0.0, vffN=0.0):
        """First-order velocity servo toward (tE, tN) with target-velocity feed-
        forward: v = v_ff + vel_gain * error, clamped. Returns the horizontal error."""
        eE = tE - self.pos[0]
        eN = tN - self.pos[1]
        self.cmd_vel[0] = self._clamp(vffE + self.vel_gain * eE, -self.vel_max, self.vel_max)
        self.cmd_vel[1] = self._clamp(vffN + self.vel_gain * eN, -self.vel_max, self.vel_max)
        return math.hypot(eE, eN)

    def _height_above_deck(self):
        return max(self.pos[2] - self.platform_height, 0.05)

    def _publish_velocity(self):
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
