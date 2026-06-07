#!/usr/bin/env python3
"""Track-line follower — drives the platform around the painted 400 m track.

The platform (aruco_platform) carries a FORWARD-looking camera pitched down at
the ground. The track is a lane bounded by TWO white lines. Rather than a single
centroid, this node fits the lane numerically: it samples the lane CENTRE at
several look-ahead rows (centre = midpoint between the two detected line edges in
that row band), then runs a LEAST-SQUARES POLYNOMIAL FIT — np.polyfit, degree 2
(a curve) when enough points span the ROI, degree 1 (a straight line) otherwise.
The fitted curve x = P(row) is the estimated lane path ahead. Steering combines
two readings off that fit: the CROSS-TRACK error P(near row) (how far off-centre
the lane is right at the vehicle) and a LOOK-AHEAD error P(far row) (where the
lane heads, which anticipates curves); the degree-2 coefficient (path curvature)
also trims forward speed in tight bends. So the platform tracks the fitted
line/curve down the MIDDLE of the lane. It
drives the truck via its gz VelocityControl plugin (the same cmd_vel topic the
old preset-trajectory mover used), so the rest of the truck mission is unchanged:

  * the drone rides ON TOP, secured by the DetachableJoint, and
  * on arm we sever that joint so the drone launches FROM the moving truck, and
  * we publish the truck's ACTUAL (E,N) as /marker/position so the precision-
    landing controller can later fly back and land on it.

So this is a drop-in replacement for moving_marker_node in the track world: the
truck's path now comes from CAMERA line-following instead of a hard-coded sine.

Steering geometry: gz VelocityControl takes a body/world Twist. To avoid any
world-vs-body frame ambiguity for the LINEAR part, we read the truck's actual
yaw from gz dynamic_pose/info and send the linear velocity already rotated into
the WORLD frame (v·[cos ψ, sin ψ]); the angular part is a pure yaw rate about
+Z (identical in world and body for a flat-driving vehicle). Image x grows to
the right, +yaw is CCW (left), so a line offset to the right (error>0) commands
a negative (right) yaw rate: omega = -Kp · error.

Topic / service contract:
  in  (ROS)  <image_topic>                  sensor_msgs/Image    front camera
  in  (ROS)  /mavros/state                  mavros_msgs/State    arm edge
  out (ROS)  <marker_topic>                 geometry_msgs/PointStamped  ENU cue
  out (ROS)  <debug_topic>/compressed       sensor_msgs/CompressedImage (opt.)
  pub (gz)   <cmd_vel_topic>                gz.msgs.Twist        drive command
  pub (gz)   <detach_topic>                 gz.msgs.Empty        joint release
  sub (gz)   /world/<world>/dynamic_pose/info  gz.msgs.Pose_V    actual pose
"""

import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CompressedImage
from mavros_msgs.msg import State

# gz-transport is optional: without it we still run the vision/steering loop and
# publish the cue, we just can't drive the Gazebo model or release the joint.
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.empty_pb2 import Empty as GzEmpty
    from gz.msgs10.twist_pb2 import Twist as GzTwist
    from gz.msgs10.pose_v_pb2 import Pose_V as GzPoseV
    _GZ_OK = True
except Exception as _e:        # pragma: no cover - depends on host gz install
    _GZ_OK = False
    _GZ_ERR = _e


def _imgmsg_to_bgr(msg: Image):
    # Manual sensor_msgs/Image -> BGR ndarray. Avoids cv_bridge, whose compiled
    # boost module breaks under NumPy 2.x (_ARRAY_API not found). Mirrors the
    # camera_detection aruco_detector conversion.
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    h, w = msg.height, msg.width
    enc = msg.encoding
    if enc in ('rgb8', 'bgr8'):
        img = buf.reshape(h, w, 3)
        if enc == 'rgb8':
            img = img[:, :, ::-1]              # RGB -> BGR
        return np.ascontiguousarray(img)
    if enc in ('mono8', '8UC1'):
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(buf.reshape(h, w, 3))


def _yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class TrackFollowerNode(Node):
    def __init__(self):
        super().__init__('track_follower_node')

        # --- Parameters -----------------------------------------------------
        self.world = self.declare_parameter('world', 'iris_down_camera_runway').value
        self.model = self.declare_parameter('model', 'aruco_platform').value
        self.image_topic = self.declare_parameter('image_topic', '/front_camera/image').value
        self.marker_topic = self.declare_parameter('marker_topic', '/marker/position').value
        self.cmd_vel_topic = self.declare_parameter(
            'cmd_vel_topic', '/model/aruco_platform/cmd_vel').value
        self.detach_topic = self.declare_parameter(
            'detach_topic', '/aruco_platform/detach').value
        self.frame_id = self.declare_parameter('frame_id', 'map').value

        # forward_speed is the CRUISE CAP, not a fixed command — the actual speed is
        # computed each tick from the path (see _drive): a curvature limit, an
        # error-based ease-off, and acceleration limits make it ramp up and down
        # gradually (like the drone easing in as it nears the pad), never a hard set.
        self.forward_speed = self.declare_parameter('forward_speed', 3.0).value   # m/s, cap

        # --- Dynamic longitudinal speed profile -----------------------------
        # v_target = min( cap, sqrt(lat_accel_max/|κ|), cap/(1+err_gain·err) ), then
        # the commanded speed chases v_target under accel_max/decel_max limits so it
        # changes smoothly with travel instead of being explicitly pinned.
        self.lat_accel_max = self.declare_parameter('lat_accel_max', 2.5).value   # m/s^2
        self.accel_max = self.declare_parameter('accel_max', 1.5).value           # m/s^2
        self.decel_max = self.declare_parameter('decel_max', 2.5).value           # m/s^2
        # min_speed kept reasonably high so the car always has forward motion to
        # steer with (lateral authority ∝ speed); crawling kills controllability.
        self.min_speed = self.declare_parameter('min_speed', 1.0).value           # m/s

        # --- Steering control, designed as a 2nd-order lateral loop ----------
        # The lane-keeping loop is modelled as ÿ + 2ζωₙẏ + ωₙ²y = 0 (y = lateral
        # error [m]). The kinematic model ẏ = v·ψ, ψ̇ = ω with the feedback law
        #   ω = v·κ_ff + k_ψ·ψ + k_y·e_y
        # gives ωₙ² = v·k_y and 2ζωₙ = k_ψ, so for a target bandwidth ωₙ and a
        # chosen damping ζ the gains are FIXED by the math:
        #   k_y = ωₙ²/v   (cross-track gain, [ (rad/s)/m ])
        #   k_ψ = 2ζωₙ    (heading gain,      [ 1/s ])
        # ζ = 0.707 is the maximally-flat / -3dB-bandwidth-equals-ωₙ sweet spot
        # (no overshoot, no weaving) — exactly the case the Bode exercise singles
        # out (BW = ωₙ when ζ = 1/√2). v·κ is curvature feed-forward so a constant-
        # radius bend needs no steady-state error.
        self.ctrl_bw = self.declare_parameter('ctrl_bw', 1.2).value        # ωₙ [rad/s]
        # ζ ≥ 1 (critically/over-damped). 0.707 is the no-lag optimum, but the real
        # loop has camera/processing lag (worse under the GUI, which drops camera
        # frames) that eats phase margin and leaves a P-dominant loop OSCILLATING.
        # ζ>1 raises the heading (D) gain k_ψ=2ζωₙ so it converges without ringing,
        # while P=k_y=ωₙ²/v stays put. Raise FOLLOW_ZETA more if it still hunts.
        self.zeta = self.declare_parameter('zeta', 1.2).value              # damping ratio
        self.kappa_ff = self.declare_parameter('kappa_ff', 1.0).value      # curvature FF scale
        # EMA low-pass on the lane estimate (e_y/ψ/κ): 1.0 = OFF. Default off — the
        # lag it adds hurts convergence (it was making the oscillation worse, not
        # better); damping is handled by ζ instead. Set <1 only for raw-noise jitter.
        self.ctrl_lpf_alpha = self.declare_parameter('ctrl_lpf_alpha', 1.0).value
        self.yaw_rate_max = self.declare_parameter('yaw_rate_max', 0.8).value     # rad/s
        # How many consecutive lane-lost frames to coast (hold last estimate) before
        # declaring the lane truly lost and easing to a stop.
        self.lost_hold_frames = int(self.declare_parameter('lost_hold_frames', 8).value)
        # Hold still this long at the start so the drone's EKF settles and it can
        # arm on the STATIONARY truck before it drives off (mounted-launch).
        self.start_delay = float(self.declare_parameter('start_delay', 0.0).value)  # s
        self.release_on_arm = self.declare_parameter('release_on_arm', True).value
        self.publish_debug = self.declare_parameter('publish_debug', True).value

        # --- Camera geometry for inverse-perspective mapping (IPM) -----------
        # Known from the model SDF (aruco_platform front_camera): the camera sits
        # cam_x forward of the platform origin, cam_h above the ground, pitched
        # cam_pitch down; cam_fov is the horizontal FOV. With these we map each
        # image pixel onto the ground plane analytically (no calibration), so the
        # lane fit is done in METRES where the two lines are truly parallel — this
        # removes the perspective bias that made a plain image-centroid cut the
        # inside of curves.
        self.cam_fov = self.declare_parameter('cam_fov', 1.60).value       # rad, horizontal
        self.cam_h = self.declare_parameter('cam_height', 0.85).value      # m above ground
        self.cam_pitch = self.declare_parameter('cam_pitch', 0.61).value   # rad, down
        self.cam_x = self.declare_parameter('cam_x', 0.60).value           # m fwd of origin
        # Known half lane width [m]: lets a band that sees only ONE line still place
        # the lane centre (centre = Y_line - half_lane toward the lane), so curves
        # where the far line leaves frame don't drop to "lost".
        self.half_lane = self.declare_parameter('half_lane', 1.0).value    # m

        # Lane detection. Grayscale threshold isolates the white lines; the ROI is
        # split into n_bands strips, each giving one lane-centre pixel at (min+max)/2
        # of its white columns (midpoint of the two line edges). Those pixels are
        # IPM-projected to ground (X fwd, Y left) and a curve Y=P(X) is least-squares
        # fit there (degree 2 = curve, 1 = straight line).
        self.white_thresh = int(self.declare_parameter('white_thresh', 180).value)
        self.roi_top = self.declare_parameter('roi_top', 0.45).value
        self.roi_bot = self.declare_parameter('roi_bot', 0.95).value
        self.n_bands = int(self.declare_parameter('n_bands', 12).value)
        self.band_min_px = int(self.declare_parameter('band_min_px', 6).value)
        self.fit_degree = int(self.declare_parameter('fit_degree', 2).value)  # 2=curve,1=line
        # White columns in a band are split into clusters wherever a gap >
        # line_gap_px appears (clusters need >= min_cluster_px). Two clusters = both
        # lines (centre = midpoint); one cluster = a single line (centre via the
        # known half-lane offset). See _band_centre.
        self.line_gap_px = int(self.declare_parameter('line_gap_px', 20).value)
        self.min_cluster_px = int(self.declare_parameter('min_cluster_px', 2).value)

        # --- State ----------------------------------------------------------
        self._yaw = 0.0          # latest actual truck yaw (world)
        self._pos = None         # latest actual truck (x, y)
        # last good metric lane estimate (held briefly when the lane is lost)
        self._last_ey = 0.0      # lateral offset of lane centre [m], +left
        self._last_psi = 0.0     # lane heading rel. forward [rad], +left
        self._last_kappa = 0.0   # lane curvature [1/m], +left
        self._lost_count = 0
        self._v = 0.0            # current commanded speed [m/s] (ramped, not pinned)
        self._t_prev = None      # last tick time, for the speed ramp dt
        self._v_target = 0.0     # latest speed target (for the HUD)
        self._last_omega = 0.0   # last commanded yaw rate / speed (for the HUD)
        self._last_v = 0.0
        self._fit_deg = 0        # degree of the most recent lane fit (1=line,2=curve)
        self._armed = False
        self._released = False
        self.t0 = self.get_clock().now()

        # --- ROS I/O --------------------------------------------------------
        # RELIABLE (matches the ros_gz_bridge publisher) + KEEP_LAST depth 1 = always
        # the freshest frame, no backlog lag. BEST_EFFORT drops these 640x480 (≈0.9 MB)
        # images irregularly, which starved the control loop and made it weave even on
        # straights (worse under the GUI's heavier load) — so use RELIABLE.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, self.image_topic, self._image_cb, qos)
        self.cue_pub = self.create_publisher(PointStamped, self.marker_topic, 10)
        # Debug overlay published BOTH as raw Image (so `rqt_image_view
        # /perception/track_debug` shows it instantly, no transport fiddling) and
        # compressed (cheap for remote viewers). image_transport naming convention:
        # base topic raw + "<base>/compressed".
        if self.publish_debug:
            self.dbg_raw_pub = self.create_publisher(Image, '/perception/track_debug', 10)
            self.dbg_pub = self.create_publisher(
                CompressedImage, '/perception/track_debug/compressed', 10)
        else:
            self.dbg_raw_pub = self.dbg_pub = None
        if self.release_on_arm:
            self.create_subscription(State, '/mavros/state', self._state_cb, 10)

        # --- gz transport ---------------------------------------------------
        self.gz = None
        self.cmd_vel_pub = None
        self.detach_pub = None
        if _GZ_OK:
            self.gz = GzNode()
            self.cmd_vel_pub = self.gz.advertise(self.cmd_vel_topic, GzTwist)
            self.gz.subscribe(GzPoseV, f'/world/{self.world}/dynamic_pose/info',
                              self._gz_pose_cb)
            if self.release_on_arm:
                self.detach_pub = self.gz.advertise(self.detach_topic, GzEmpty)
            self.get_logger().info(
                f'driving "{self.model}" via {self.cmd_vel_topic}; '
                f'pose from /world/{self.world}/dynamic_pose/info')
        else:
            self.get_logger().warn(
                f'gz Python bindings unavailable ({_GZ_ERR}); vision runs but the '
                'truck will NOT move / joint NOT released.')

        ky = self.ctrl_bw ** 2 / max(0.1, self.forward_speed)
        kpsi = 2.0 * self.zeta * self.ctrl_bw
        self.get_logger().info(
            f'track_follower: image={self.image_topic} v={self.forward_speed:.1f} m/s '
            f'IPM h={self.cam_h:.2f} pitch={self.cam_pitch:.2f} fov={self.cam_fov:.2f} | '
            f'2nd-order ctrl wn={self.ctrl_bw:.2f} zeta={self.zeta:.3f} '
            f'-> k_y={ky:.3f} k_psi={kpsi:.3f}')

    # -----------------------------------------------------------------------
    def _lane_clusters(self, mask, r0):
        """Per-band white-pixel CLUSTERS: returns (row_full, [col_mean, ...] sorted).

        A new cluster starts wherever the column gap exceeds line_gap_px; clusters
        with < min_cluster_px columns are dropped. One cluster = a single line in
        that band's view, two = both lines. The caller decides the lane centre
        (true midpoint with two clusters, known-width offset with one).
        """
        H, W = mask.shape
        n = max(2, self.n_bands)
        bh = max(1, H // n)
        out = []
        for k in range(n):
            y0 = k * bh
            y1 = (k + 1) * bh if k < n - 1 else H
            cols = np.where(mask[y0:y1].any(axis=0))[0]
            if cols.size < self.band_min_px:
                continue
            groups = np.split(cols, np.where(np.diff(cols) > self.line_gap_px)[0] + 1)
            means = [float(g.mean()) for g in groups if g.size >= self.min_cluster_px]
            if means:
                out.append((r0 + (y0 + y1) // 2, means))
        return out

    def _band_centre(self, row, means, w, h):
        """Lane-centre ground point for one band's clusters, or None. Both clusters
        are at the SAME image row → the SAME forward distance X, so their midpoint
        is the geometrically correct lane centre at that X (this stays accurate on
        curves, unlike averaging two whole-line fits). A single line uses the known
        half-lane offset. Returns (X, Y_centre, centre_px, left_px, right_px)."""
        if len(means) >= 2:
            cl, cr = means[0], means[-1]
            gl = self._ground_xy(cl, row, w, h)
            gr = self._ground_xy(cr, row, w, h)
            if gl and gr and gl[0] > 0.0 and gr[0] > 0.0:
                return (0.5 * (gl[0] + gr[0]), 0.5 * (gl[1] + gr[1]),
                        0.5 * (cl + cr), cl, cr)
            return None
        # single line: offset inward by the known half lane width (Yl>0 = left line
        # -> centre = Yl - half_lane; Yl<0 = right line -> + half_lane)
        c = means[0]
        g = self._ground_xy(c, row, w, h)
        if g is None or g[0] <= 0.0:
            return None
        X, Yl = g
        return (X, Yl - math.copysign(self.half_lane, Yl), c, c, c)

    def _ground_xy(self, u, v, w, h):
        """Inverse-perspective map: image pixel (u,v) -> ground point (X,Y) in the
        platform body frame [m], X forward, Y left. Returns None if the pixel's
        ray does not strike the ground ahead.

        Pinhole intrinsics from the horizontal FOV (square pixels), extrinsics from
        the known camera mount (height cam_h, pitch cam_pitch down, cam_x forward).
        Optical frame (z fwd, x right, y down) relative to the body after the pitch:
            z_opt = ( cosθ, 0, -sinθ),  x_opt = (0,-1,0),  y_opt = (-sinθ,0,-cosθ)
        Ray dir r = xn·x_opt + yn·y_opt + z_opt, then intersect plane z=0.
        """
        f = (w / 2.0) / math.tan(self.cam_fov / 2.0)
        xn = (u - w / 2.0) / f
        yn = (v - h / 2.0) / f
        st, ct = math.sin(self.cam_pitch), math.cos(self.cam_pitch)
        rx = ct - yn * st
        ry = -xn
        rz = -(yn * ct + st)
        if rz > -1e-6:                      # ray points at/above the horizon
            return None
        t = -self.cam_h / rz                # >0
        return (self.cam_x + t * rx, t * ry)

    def _image_cb(self, msg: Image):
        bgr = _imgmsg_to_bgr(msg)
        h, w = bgr.shape[:2]
        r0 = int(self.roi_top * h)
        r1 = int(self.roi_bot * h)
        gray = cv2.cvtColor(bgr[r0:r1, :, :], cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)

        bands = self._lane_clusters(mask, r0)

        # Build the VIRTUAL centre line: one centre POINT per band (both lines at the
        # same row → same X → true midpoint; single line → known-width offset), then
        # least-squares fit Y=P(X) through them. That fitted polynomial IS the
        # synthetic centre line we follow.
        coeffs = None          # image-space fit, for the overlay
        overlay = []           # (row, centre_px, left_px, right_px) for the HUD
        gpts_two = []          # centres from bands seeing BOTH lines (unbiased)
        gpts_one = []          # centres from single-line bands (known-width)
        for (row, means) in bands:
            bc = self._band_centre(row, means, w, h)
            if bc is None:
                continue
            X, Yc, cpx, lpx, rpx = bc
            (gpts_two if lpx != rpx else gpts_one).append((X, Yc))
            overlay.append((row, cpx, lpx, rpx))
        # Use ALL centre points together: this is well-conditioned and stable.
        # (Using only the two-line bands sounds more accurate but they are few and
        # bunched in X → the degree-2 fit becomes ill-conditioned and flings the
        # truck off; the small steady straight offset from single-line bands is the
        # safe trade.)
        gpts = gpts_two + gpts_one

        e_y = psi = kappa = 0.0
        X0 = 0.0
        ok = False
        if len(gpts) >= 2:
            Xs = np.array([p[0] for p in gpts]); Ys = np.array([p[1] for p in gpts])
            degm = self.fit_degree if (len(gpts) >= 3 and self.fit_degree >= 2) else 1
            self._fit_deg = degm
            cm = np.polyfit(Xs, Ys, degm)        # virtual centre line Y=P(X) [m]
            dcm = np.polyder(cm)
            X0 = float(Xs.min())
            e_y = float(np.polyval(cm, X0))      # lateral offset of centre [m], +left
            slope = float(np.polyval(dcm, X0))
            psi = math.atan(slope)               # heading of centre [rad], +left
            if degm >= 2:
                kappa = (2.0 * cm[0]) / (1.0 + slope * slope) ** 1.5
            # EMA low-pass on the control inputs to kill frame-to-frame jitter /
            # hunting (a fresh fit each frame can twitch; smoothing steadies the
            # target). alpha=1 → no smoothing, smaller → smoother but more lag.
            a = self.ctrl_lpf_alpha
            e_y = a * e_y + (1 - a) * self._last_ey
            psi = a * psi + (1 - a) * self._last_psi
            kappa = a * kappa + (1 - a) * self._last_kappa
            orows = np.array([o[0] for o in overlay], dtype=float)
            ocols = np.array([o[1] for o in overlay], dtype=float)
            coeffs = np.polyfit(orows, ocols, 2 if len(overlay) >= 3 else 1)
            self._last_ey, self._last_psi, self._last_kappa = e_y, psi, kappa
            self._lost_count = 0
            ok = True

        if not ok:
            # Lane lost: hold the last metric estimate briefly (bridges glare / a
            # band gap), then zero it and crawl so a long miss can't fling us off.
            self._lost_count += 1
            self._fit_deg = 0
            if self._lost_count < self.lost_hold_frames:
                e_y, psi, kappa = self._last_ey, self._last_psi, self._last_kappa

        self._drive(e_y, psi, kappa, ok)
        self._publish_cue(msg.header.stamp)
        if self.dbg_pub is not None:
            self._publish_debug(bgr, mask, r0, r1, overlay, coeffs,
                                e_y, psi, kappa, X0, ok)

    def _drive(self, e_y, psi, kappa, detected):
        if self.cmd_vel_pub is None:
            return
        # Warm-up: stay put until start_delay elapses (drone arms on still truck).
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9
        if t < self.start_delay:
            self.cmd_vel_pub.publish(GzTwist())   # zero
            self._v = self._last_omega = self._last_v = 0.0
            self._t_prev = now
            return
        dt = 0.033 if self._t_prev is None else \
            min(0.1, max(1e-3, (now - self._t_prev).nanoseconds * 1e-9))
        self._t_prev = now

        # --- Dynamic speed: NOT a fixed command, but ONLY curvature-limited + ramp.
        #   v ≤ √(a_lat/|κ|) keeps a lateral-accel budget on bends; an accel/decel
        #   ramp makes it ease up/down with travel. We deliberately do NOT slow on
        #   tracking error: a car is nonholonomic, so going slower REDUCES lateral
        #   authority (you need forward motion to move sideways) and makes the
        #   steering loop oscillate — the opposite of the hovering drone. ---
        if detected or self._lost_count < self.lost_hold_frames:
            v_curve = math.sqrt(self.lat_accel_max / max(abs(kappa), 1e-4))
            v_target = max(self.min_speed, min(self.forward_speed, v_curve))
        else:
            v_target = 0.0                      # lane lost: ease to a stop
        self._v_target = v_target
        dv = v_target - self._v
        amax = self.accel_max if dv > 0.0 else self.decel_max
        self._v += max(-amax * dt, min(amax * dt, dv))
        self._v = max(0.0, self._v)
        v = self._v

        # 2nd-order lateral control with curvature feed-forward. Gains are FIXED at
        # the cruise speed (k_y=ωₙ²/v_cruise, k_ψ=2ζωₙ) — NOT recomputed from the
        # current speed, which would blow k_y up as v drops and drive the loop
        # unstable. At lower speeds the effective ωₙ=√(v·k_y) simply drops, i.e. it
        # corrects more gently, which is safe. Signs are +left: lane left (e_y>0) or
        # heading/curving left commands +ω (CCW/left), driving the error to zero.
        vc = max(self.min_speed, self.forward_speed)
        k_y = self.ctrl_bw ** 2 / vc
        k_psi = 2.0 * self.zeta * self.ctrl_bw
        omega = self.kappa_ff * v * kappa + k_psi * psi + k_y * e_y
        omega = max(-self.yaw_rate_max, min(self.yaw_rate_max, omega))
        self._last_omega, self._last_v = omega, v

        # gz VelocityControl applies cmd_vel in the model's OWN (body) frame
        # (components LinearVelocityCmd / AngularVelocityCmd are body-frame; the
        # world-frame variants are the separate World*Cmd). So this is a plain
        # unicycle command: drive FORWARD along body +x at v and yaw at omega —
        # the truck turns and its heading carries the forward velocity around the
        # curve. (Sending world-frame linear here made it crab sideways the moment
        # it yawed, so it could never hold the lane.)
        twist = GzTwist()
        twist.linear.x = float(v)
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.z = float(omega)
        self.cmd_vel_pub.publish(twist)

    def _publish_cue(self, stamp):
        if self._pos is None:
            return
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.point.x = self._pos[0]   # ENU East  (== Gazebo world x)
        msg.point.y = self._pos[1]   # ENU North (== Gazebo world y)
        msg.point.z = 0.0
        self.cue_pub.publish(msg)

    def _publish_debug(self, bgr, mask, r0, r1, samples, coeffs,
                       e_y, psi, kappa, X0, detected):
        """Render the 'how it sees the lane and how it drives' overlay onto the
        camera frame: detected line pixels, the two line edges, the per-band lane
        centres, the least-squares fit curve, a steering arrow, and a text HUD
        with the METRIC estimate (lateral [m] / heading [deg] / curvature) and the
        2nd-order control state (ωₙ, ζ, yaw rate, speed)."""
        vis = bgr.copy()
        h, w = vis.shape[:2]
        cxc = w // 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1) detected white-line pixels tinted green inside the ROI
        roi = vis[r0:r1, :, :]
        roi[mask > 0] = (0, 255, 0)
        # ROI box + image-centre reference (where we WANT the lane centre)
        cv2.rectangle(vis, (0, r0), (w - 1, r1), (255, 128, 0), 2)
        cv2.line(vis, (cxc, r0), (cxc, r1), (180, 180, 180), 1)

        # 2) per-band detection: left edge / right edge (orange) and lane centre
        #    (yellow), with a thin bar showing the detected lane width.
        for (row, centre, left, right) in samples:
            row = int(row)
            cv2.line(vis, (int(left), row), (int(right), row), (0, 165, 255), 1)
            cv2.circle(vis, (int(left), row), 3, (0, 140, 255), -1)
            cv2.circle(vis, (int(right), row), 3, (0, 140, 255), -1)
            cv2.circle(vis, (int(centre), row), 3, (0, 255, 255), -1)

        # 3) the fitted lane path (magenta, image-space fit just for drawing)
        if coeffs is not None:
            pts = [(int(np.polyval(coeffs, row)), row) for row in range(r0, r1, 3)]
            for i in range(1, len(pts)):
                cv2.line(vis, pts[i - 1], pts[i], (255, 0, 255), 2)

        # 4) steering arrow from bottom centre: tilts toward the commanded turn,
        #    length grows with yaw-rate magnitude. +omega = left (CCW).
        omega = self._last_omega
        frac = max(-1.0, min(1.0, omega / self.yaw_rate_max)) if self.yaw_rate_max else 0.0
        ang = frac * (math.pi / 4.0)            # up to 45 deg off vertical
        base = (cxc, h - 8)
        tip = (int(cxc - 70 * math.sin(ang)), int(h - 8 - 70 * math.cos(ang)))
        cv2.arrowedLine(vis, base, tip, (0, 255, 255), 4, tipLength=0.3)

        # 5) binary-mask inset (top-right) so the raw thresholding is visible
        tw = 150
        th = max(1, int((r1 - r0) * tw / w))
        thumb = cv2.cvtColor(cv2.resize(mask, (tw, th)), cv2.COLOR_GRAY2BGR)
        vis[6:6 + th, w - tw - 6:w - 6] = thumb
        cv2.rectangle(vis, (w - tw - 6, 6), (w - 6, 6 + th), (255, 128, 0), 1)
        cv2.putText(vis, 'mask', (w - tw - 4, 6 + th - 4), font, 0.4, (0, 255, 255), 1)

        # 6) text HUD on a dark panel (top-left): METRIC estimate + control state
        omega = self._last_omega
        deg_s = math.degrees(omega)
        turn = 'LEFT' if omega > 0.02 else 'RIGHT' if omega < -0.02 else 'STRAIGHT'
        state = (f'TRACKING  fit=deg{self._fit_deg}  n={len(samples)}'
                 if detected else f'LANE LOST ({self._lost_count})')
        lines = [
            state,
            f'lateral e_y = {e_y:+.2f} m   (X0={X0:.1f} m)',
            f'heading psi = {math.degrees(psi):+.1f} deg',
            f'curv  kappa = {kappa:+.3f} 1/m',
            f'wn={self.ctrl_bw:.2f} zeta={self.zeta:.3f}',
            f'yaw = {omega:+.2f} rad/s ({deg_s:+.0f} deg/s) {turn}',
            f'speed = {self._last_v:.1f} -> {self._v_target:.1f} m/s',
        ]
        panel = vis.copy()
        cv2.rectangle(panel, (4, 4), (330, 22 + 22 * len(lines)), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.45, vis, 0.55, 0, vis)
        col0 = (0, 255, 0) if detected else (0, 0, 255)
        for i, ln in enumerate(lines):
            cv2.putText(vis, ln, (10, 24 + 22 * i), font, 0.55,
                        col0 if i == 0 else (255, 255, 255), 1, cv2.LINE_AA)

        stamp = self.get_clock().now().to_msg()
        raw = Image()
        raw.header.stamp = stamp
        raw.header.frame_id = 'front_camera'
        raw.height, raw.width = h, w
        raw.encoding = 'bgr8'
        raw.is_bigendian = 0
        raw.step = w * 3
        raw.data = np.ascontiguousarray(vis).tobytes()
        self.dbg_raw_pub.publish(raw)

        ok, enc = cv2.imencode('.jpg', vis)
        if ok:
            out = CompressedImage()
            out.header.stamp = stamp
            out.format = 'jpeg'
            out.data = enc.tobytes()
            self.dbg_pub.publish(out)

    def _gz_pose_cb(self, msg):
        for p in msg.pose:
            if p.name == self.model:
                self._pos = (p.position.x, p.position.y)
                q = p.orientation
                self._yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
                return

    def _state_cb(self, msg):
        if msg.armed and not self._armed and not self._released:
            self._release_joint()
        self._armed = msg.armed

    def _release_joint(self):
        if self.detach_pub is None:
            return
        self.detach_pub.publish(GzEmpty())
        self._released = True
        self.get_logger().info(
            f'drone armed -> released DetachableJoint ({self.detach_topic})')


def main(args=None):
    rclpy.init(args=args)
    node = TrackFollowerNode()
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
