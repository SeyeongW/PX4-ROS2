"""aruco_detector_node — ONE job: find the ArUco marker in the down camera.

Does NOT know about drones, frames or landing.  It converts pixels to a marker
pose in the CAMERA OPTICAL frame and publishes it; `marker_tf_node` turns that
into a world position.

Subscribes
    /down_camera/image          sensor_msgs/Image        (BEST_EFFORT)
    /down_camera/camera_info    sensor_msgs/CameraInfo   (BEST_EFFORT)
Publishes
    /aruco/pose_cam             geometry_msgs/PoseStamped
        position = marker centre in the camera optical frame (REP-103:
        +X right, +Y down, +Z forward), metres.
        header.stamp = the IMAGE capture stamp — downstream must transform with
        the vehicle state at that instant, not "now".
    /aruco/detected             std_msgs/Bool            (every frame)
    /aruco/center_error         geometry_msgs/Vector3Stamped  (on detection)
        x = dx, y = dy, z = hypot(dx, dy): the horizontal offset from the
        optical axis to the LANDING POINT, in METRES, in the camera optical
        frame.  This is the same centring error the debug image draws, just as
        numbers a controller or a plot can read; pixel dx/dy stay on the
        overlay.  Published only when a marker is solved, so its arrival rate
        IS the detection rate.
    /aruco/debug_image/compressed  sensor_msgs/CompressedImage  annotated view,
        JPEG (quality `debug_image_quality`, default 50) — a raw frame is
        ~1.4 MB and stalls rqt_image_view; compressed it is ~20 kB.

The debug image draws what the detector is actually reasoning about: the frame
centre, the marker centre, the line between them, and the dx/dy deviation in
both pixels and metres.  It also carries a FILL gauge — how much of the frame
the marker spans — because the failure that matters near the deck is not a bad
detection, it is the marker growing until it no longer fits:

    r(h) = h*tan(vfov/2) - s_m/2  ->  0   at   h = s_m / (2 tan(vfov/2))

so a 1.5 m marker in this camera is unresolvable below ~0.97 m no matter how
good the tracking is.  The gauge turns that from a surprise into a readout.

Marker geometry: the model's textured plane is 1.95 m but its quiet zone leaves
a 1.50 m black code — measured from the texture (400/520 px x 1.95 m = 1.500 m),
so marker_size_m = 1.5 is the value solvePnP needs.  Range scales linearly with
this number, so do not change it without re-measuring.
"""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool


def img_to_bgr(msg: Image) -> np.ndarray:
    """Image -> BGR ndarray without cv_bridge (it breaks under numpy 2).

    The result is always C-CONTIGUOUS.  This is load-bearing: the rgb8 path
    reverses the channel axis with `[:, :, ::-1]`, which is a NEGATIVE-STRIDE
    view, and cv2 DRAWING functions (line/putText/…) reject such an array with
    "Layout of the output array img is incompatible with cv::Mat".  detectMarkers
    tolerates it, so the detector ran fine until the first frame it tried to
    ANNOTATE — then the node crashed, which read downstream as 0% detection.
    """
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    if msg.encoding in ('rgb8', 'bgr8'):
        img = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)
    if msg.encoding == 'mono8':
        g = rows[:, :msg.width].reshape(msg.height, msg.width)
        return np.ascontiguousarray(np.dstack([g, g, g]))
    raise ValueError(f'unsupported encoding {msg.encoding}')


def bgr_to_compressed(bgr: np.ndarray, header, quality: int) -> CompressedImage:
    """BGR ndarray -> JPEG CompressedImage (no cv_bridge; cv2 does the encode)."""
    ok, buf = cv2.imencode('.jpg', bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    m = CompressedImage()
    m.header = header
    m.format = 'jpeg'
    if ok:
        m.data = buf.tobytes()
    return m


def rvec_to_quat(rvec):
    """OpenCV rotation vector -> [w, x, y, z]."""
    R, _ = cv2.Rodrigues(rvec)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    return [w,
            (R[2, 1] - R[1, 2]) / (4 * w),
            (R[0, 2] - R[2, 0]) / (4 * w),
            (R[1, 0] - R[0, 1]) / (4 * w)]


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        p = self.declare_parameter
        # A LADDER of markers, not one.  Each entry is size and offset from the
        # marker centre to the LANDING POINT, in the marker's own plane.  The
        # detector always reports the landing point, so nothing downstream has
        # to know which marker was used — the handover is invisible above this
        # node.  Defaults match the deck in
        # `gazebo/models/moving_platform_aruco_velocity/model.sdf`.
        # Sizes are chosen so the ladder OVERLAPS rather than merely meets.
        # The big marker is usable only above (offset + s/2)/tan = 2.65 m —
        # being off-centre costs it the bottom of its own range — and the small
        # one only below s*fy/min_px = 3.88 m, so 0.30 m at a 1.3 m offset
        # leaves a 1.2 m band where both work.  0.35 m widens that band but
        # puts h_blind at 0.23 m against a 0.25 m touchdown, i.e. 2 cm of
        # margin; 0.25 m keeps the margin but narrows the band to 0.69 m.
        self.ids = [int(v) for v in p('marker_ids', [0, 2, 1]).value]
        self.sizes = [float(v) for v in p('marker_sizes_m',
                                          [1.3, 1.3, 0.30]).value]
        # flat [x0, y0, x1, y1, ...] because ROS2 has no array-of-arrays param.
        # Offset points from the MARKER CENTRE to the LANDING POINT, in the
        # marker's own plane — so a big marker sitting at deck -1.1 m has offset
        # +1.1 m, its mirror has -1.1 m, and the centred small one has zero.
        offs = [float(v) for v in p('marker_offsets_m', [1.1, 0.0,
                                                         -1.1, 0.0,
                                                         0.0, 0.0]).value]
        if not (len(self.ids) == len(self.sizes) == len(offs) // 2):
            raise ValueError(
                f'marker_ids({len(self.ids)}), marker_sizes_m({len(self.sizes)}) '
                f'and marker_offsets_m({len(offs)}/2) must describe the same '
                f'number of markers')
        self.offsets = {i: np.array([offs[2 * k], offs[2 * k + 1], 0.0])
                        for k, i in enumerate(self.ids)}
        self.msize = dict(zip(self.ids, self.sizes))
        # The largest marker is the one the FILL gauge and h_blind refer to.
        self.target_id = self.ids[0]
        self.size = self.sizes[0]
        dict_name = str(p('aruco_dictionary', 'DICT_4X4_50').value)
        self.report_s = float(p('report_period_s', 5.0).value)

        adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        params = cv2.aruco.DetectorParameters()
        # sub-pixel corner refinement: the corner accuracy sets the pose
        # accuracy, and at altitude the marker is only tens of pixels wide.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(adict, params)

        self.objp = {i: np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0],
                                  [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]],
                                 np.float32)
                     for i, s in self.msize.items()}
        # A marker can be DETECTED long before its pose is worth anything.
        # The small marker is 9.7 px across at 10 m and ArUco will still find
        # it, but the pose off 9 px is nonsense — measured 1.55 m of landing
        # point error, which would poison the estimator and fire the
        # cross-check alarm for no reason.  Detection is not the gate; apparent
        # size is.  30 px over a 6-cell code is 5 px/cell, about the floor at
        # which corner refinement means anything.
        self.min_px = float(p('min_marker_px', 30.0).value)
        # SEEN vs USED, separately.  Only counting the marker that won the
        # selection makes the ladder's central question unanswerable: after a
        # run that reported `id1:1`, there was no way to tell whether the centre
        # marker had been invisible or merely out-voted by a big one, and those
        # call for opposite fixes.  Count both.
        self._seen_by_id = {i: 0 for i in self.ids}
        self._hits_by_id = {i: 0 for i in self.ids}
        self._agree = []        # landing-point disagreement when both are seen
        self.K = None
        self.dist = np.zeros(5)
        self._hits = 0
        self._frames = 0
        # --- instrumentation: detection gaps are the thing we cannot explain,
        # so record when they start/end and dump a frame from inside one.
        self._last_det = None          # time of the last successful detection
        self._gap_logged = False
        self.debug_dir = str(p('debug_dir', '').value)
        self.dump_after = float(p('dump_after_s', 4.0).value)
        self._dumped = 0

        # Which camera to look through.  The body-fixed down camera is the
        # default; point these at /gimbal_camera/* to run the same detector on
        # the tilt-decoupled gimbal payload.  Nothing else in this node changes
        # — it only ever speaks the camera optical frame, and `marker_tf_node`
        # is where the two cameras stop being interchangeable.
        image_topic = str(p('image_topic', '/down_camera/image').value)
        info_topic = str(p('camera_info_topic',
                           '/down_camera/camera_info').value)
        self.frame_id = str(p('optical_frame_id',
                              'down_camera_optical_frame').value)

        self.publish_debug = bool(p('publish_debug_image', True).value)
        # JPEG-compress the debug image.  A raw 800x600 bgr8 frame is 1.4 MB;
        # at 25 Hz that is 35 MB/s of ROS traffic, which is what stalls
        # rqt_image_view.  quality 50 JPEG is ~15-25 kB — a ~60-90x cut — and
        # the overlay is diagnostic, not measured off, so the artefacts do not
        # matter.  Published as CompressedImage on <topic>/compressed, which is
        # the name rqt_image_view already looks for.
        self.debug_quality = int(p('debug_image_quality', 50).value)
        self._last_poly = None         # last seen outline, for the lost frames
        self._last_fill = 0.0
        self._img_h = self._img_w = 0

        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose_cam', 10)
        self.det_pub = self.create_publisher(Bool, '/aruco/detected', 10)
        self.err_pub = self.create_publisher(
            Vector3Stamped,
            str(p('center_error_topic', '/aruco/center_error').value), 10)
        dbg_topic = str(p('debug_image_topic', '/aruco/debug_image').value)
        self.dbg_pub = (self.create_publisher(
            CompressedImage, dbg_topic + '/compressed', 1)
            if self.publish_debug else None)
        self.create_subscription(Image, image_topic, self._on_img,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info_topic,
                                 self._on_info, qos_profile_sensor_data)
        self.create_timer(self.report_s, self._report)
        self.get_logger().info(
            f'aruco_detector_node: {dict_name} id={self.target_id} '
            f'size={self.size} m on {image_topic} '
            f'-> /aruco/pose_cam (camera optical frame)')

    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, float).reshape(3, 3)
            d = np.array(msg.d, float)
            self.dist = d if d.size else np.zeros(5)
            self.get_logger().info(f'intrinsics locked: fx={self.K[0, 0]:.1f} '
                                   f'cx={self.K[0, 2]:.1f} cy={self.K[1, 2]:.1f}')

    def _report(self):
        if not self._frames:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        gap = (now - self._last_det) if self._last_det else float('nan')
        per_id = '  '.join(f'id{i}:{self._hits_by_id[i]}/{self._seen_by_id[i]}'
                           for i in self.ids)
        self.get_logger().info(
            f'detections {self._hits}/{self._frames} frames '
            f'({100.0 * self._hits / self._frames:.1f}%)  [used/seen {per_id}]  '
            f'since_last_detection={gap:.1f}s')
        # The cross-check is only meaningful while both markers are in view, so
        # report it when it happens rather than on a timer of its own.
        if self._agree:
            a = np.array(self._agree)
            worst = float(a.max())
            self.get_logger().info(
                f'landing-point cross-check over {a.size} dual-sighting '
                f'frames: mean {a.mean():.3f} m, max {worst:.3f} m')
            if worst > 0.5:
                self.get_logger().error(
                    f'the two markers disagree about the landing point by up '
                    f'to {worst:.2f} m. They are on ONE rigid deck, so this is '
                    f'a configuration error, not noise — check the sign and '
                    f'axes of marker_offsets_m against the SDF poses (a flipped '
                    f'sign shows up as ~2x the offset).')
            self._agree.clear()

    def _maybe_dump(self, img, now):
        """Write one frame from inside a detection gap, so we can SEE it."""
        if not self.debug_dir or self._dumped >= 6:
            return
        if self._last_det is None or (now - self._last_det) < self.dump_after:
            return
        if self._gap_logged:
            return
        try:
            import os
            os.makedirs(self.debug_dir, exist_ok=True)
            path = os.path.join(self.debug_dir, f'gap_{self._dumped:02d}.png')
            cv2.imwrite(path, img)
            self._dumped += 1
            self._gap_logged = True
            self.get_logger().warn(f'detection gap {now - self._last_det:.1f}s '
                                   f'-> dumped frame {path}')
        except Exception as e:                       # pragma: no cover
            self.get_logger().warn(f'frame dump failed: {e}')

    def _landing_point(self, marker_id, corners, t):
        """Marker pose -> landing point, carried by the CORNERS not by R.

        The obvious way to move from the marker centre to a landing point
        `o` metres away in the deck plane is `t + R @ o`.  Do not: it rides the
        PnP rotation over a lever arm, and a planar square's pose is ambiguous
        exactly when the view is fronto-parallel — which for a down camera over
        a flat deck is the NORMAL case, not an edge case.  Measured on
        synthetic frames of this deck, `R @ o` over the 1.4 m arm was fine at
        most heights and then produced a **49 cm** error at 12 m with the deck
        dead level, because solvePnP returned the mirrored IPPE solution.

        Instead map the landing point through the homography defined by the
        marker's own four corners.  That is exact in the image and uses no
        decomposition, so the ambiguity cannot reach it.

        Getting the right PIXEL is only half of it, though, and the other half
        used to be wrong: the pixel was back-projected at the MARKER CENTRE's
        depth.  Those two depths are equal only when the deck is perpendicular
        to the optical axis, and the gimbal aims at the target rather than at
        nadir, so in flight they are not.  The error is a pure lever arm,
        |o|*sin(tilt) — and because `o` points the opposite way on the mirrored
        marker, the pair disagrees by TWICE it, which is the signature the SITL
        runs kept reporting (mean 0.35 m, max 1.35 m of disagreement).
        Synthetic, exact geometry, tilt taken IN the plane the markers are
        offset along (`scratchpad/landing_point_accuracy.py`):

            tilt      10 deg   20 deg   30 deg
            centre-depth   0.19 m   0.38 m   0.57 m
            this           0.00 m   0.00 m   0.00 m

        So intersect the landing point's ray with the marker's OWN PLANE
        instead of guessing its depth.  The plane comes out of the same
        homography — K^-1 H = [r1 r2 t]/lambda with |r1| = |r2| = 1 fixes the
        scale, and n = r1 x r2 is the normal — so this still never touches a
        PnP rotation and the planar ambiguity still cannot reach it.

        A zero offset needs none of this — the marker centre IS the landing
        point — so that path returns `t` untouched.
        """
        o = self.offsets[marker_id]
        if not np.any(o):
            return t
        obj2 = self.objp[marker_id][:, :2].astype(np.float32)
        Hm = cv2.getPerspectiveTransform(obj2, corners)
        Kinv = np.linalg.inv(self.K)
        M = Kinv @ Hm
        lam = (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1])) / 2.0
        if lam < 1e-9:
            return t
        M = M / lam
        if M[2, 2] < 0.0:                 # the marker must be IN FRONT of us
            M = -M
        n = np.cross(M[:, 0], M[:, 1])    # marker-plane normal, camera frame
        p = Hm @ np.array([o[0], o[1], 1.0])
        if abs(p[2]) < 1e-9:
            return t
        d = Kinv @ np.array([p[0] / p[2], p[1] / p[2], 1.0])
        den = float(n @ d)
        if abs(den) < 1e-9:               # ray parallel to the deck: unusable
            return t
        return d * (float(n @ M[:, 2]) / den)

    def _blind_height(self, marker_id=None):
        """Height at which a marker stops fitting in frame, from intrinsics.

        The vertical half-angle is atan((H/2)/fy), so

            h_blind = s_m / (2 tan(vfov/2)) = s_m * fy / H

        Reported rather than assumed: it is a property of THIS camera and THIS
        marker.  Because it scales with size, the ladder of markers is exactly
        a ladder of h_blind — which is the whole reason the small one exists.
        """
        if self.K is None or not self._img_h:
            return float('nan')
        s = self.msize.get(marker_id, self.size)
        return s * float(self.K[1, 1]) / float(self._img_h)

    def _draw(self, img, header, poly, tvec, gap, used_id):
        """Annotate a frame with the centring error the controller cares about.

        Everything drawn is a quantity something downstream acts on, so the
        picture and the maths cannot drift apart: the crosshair is the
        principal point (the optical axis, NOT the image centre — they differ
        and using the wrong one biases every deviation), the line is the
        centring error, and dx/dy are that error in the camera optical frame,
        which is exactly what `marker_tf_node` rotates into the world.
        """
        h, w = img.shape[:2]
        cx, cy = (float(self.K[0, 2]), float(self.K[1, 2])) if self.K is not None \
            else (w / 2.0, h / 2.0)
        ci = (int(round(cx)), int(round(cy)))
        green, red, amber, white = ((0, 255, 0), (0, 0, 255),
                                    (0, 190, 255), (255, 255, 255))

        # Frame centre: crosshair + ring, stroked dark-then-light so it stays
        # readable over the marker's own white quiet zone — which is exactly
        # where it sits once the vehicle is centred, i.e. when you most need it.
        for col, th in (((0, 0, 0), 3), (white, 1)):
            cv2.line(img, (ci[0] - 22, ci[1]), (ci[0] + 22, ci[1]), col, th)
            cv2.line(img, (ci[0], ci[1] - 22), (ci[0], ci[1] + 22), col, th)
            cv2.circle(img, ci, 5, col, th if th == 1 else 2)

        lines = []
        if poly is not None:
            mc = poly.mean(axis=0)
            mi = (int(round(mc[0])), int(round(mc[1])))
            ok = tvec is not None
            col = green if ok else amber
            cv2.polylines(img, [poly.astype(np.int32)], True, col, 2)
            cv2.circle(img, mi, 5, col, -1)
            # the centring error itself
            cv2.arrowedLine(img, ci, mi, col, 2, tipLength=0.12)
            # dx / dy legs, so the split between them is readable at a glance
            cv2.line(img, ci, (mi[0], ci[1]), col, 1, cv2.LINE_AA)
            cv2.line(img, (mi[0], ci[1]), mi, col, 1, cv2.LINE_AA)
            dx_px, dy_px = mc[0] - cx, mc[1] - cy
            cv2.putText(img, f'dx {dx_px:+.0f}px', ((ci[0] + mi[0]) // 2, ci[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            cv2.putText(img, f'dy {dy_px:+.0f}px', (mi[0] + 6, (ci[1] + mi[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            lines.append((f'dx {dx_px:+7.1f} px', col))
            lines.append((f'dy {dy_px:+7.1f} px', col))
            if ok:
                lines.append((f'dx {tvec[0]:+7.3f} m', col))
                lines.append((f'dy {tvec[1]:+7.3f} m', col))
                lines.append((f'range {tvec[2]:6.2f} m', col))
                lines.append((f'|err| {float(np.hypot(tvec[0], tvec[1])):6.3f} m',
                              col))
        else:
            lines.append((f'NO DETECTION  {gap:.1f}s', red))

        # FILL: how close the marker is to outgrowing the frame.  This is the
        # near-field failure, and it is invisible in a raw image until it has
        # already happened.
        fill = self._last_fill
        bar_col = green if fill < 0.75 else amber if fill < 1.0 else red
        y0 = h - 30
        cv2.rectangle(img, (10, y0), (210, y0 + 14), (60, 60, 60), -1)
        cv2.rectangle(img, (10, y0), (10 + int(200 * min(fill, 1.0)), y0 + 14),
                      bar_col, -1)
        cv2.rectangle(img, (10, y0), (210, y0 + 14), white, 1)
        hb = self._blind_height(used_id)
        cv2.putText(img, f'FILL {100 * fill:3.0f}%   this marker fits only '
                         f'above {hb:.2f} m', (10, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_col, 1, cv2.LINE_AA)
        if fill >= 1.0:
            # ABOVE the bar, not below it: below is off the bottom edge.
            cv2.putText(img, 'BLIND ZONE - marker larger than frame',
                        (10, y0 - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1,
                        cv2.LINE_AA)

        for i, (text, col) in enumerate(lines):
            cv2.putText(img, text, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, col, 1, cv2.LINE_AA)
        rate = 100.0 * self._hits / max(self._frames, 1)
        tag = ('id=--' if used_id is None
               else f'id={used_id} {self.msize[used_id]:.2f}m')
        cv2.putText(img, f'{tag}  det {rate:.0f}%',
                    (w - 250, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1,
                    cv2.LINE_AA)
        self.dbg_pub.publish(bgr_to_compressed(img, header, self.debug_quality))

    def _on_img(self, msg):
        self._frames += 1
        det = Bool()
        if self.K is None:
            self.det_pub.publish(det)
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        img = img_to_bgr(msg)
        self._img_h, self._img_w = img.shape[0], img.shape[1]
        gap = (now - self._last_det) if self._last_det else 0.0
        corners, ids, _ = self.detector.detectMarkers(img)
        seen = {} if ids is None else {
            int(v): corners[k].reshape(4, 2).astype(np.float32)
            for k, v in enumerate(ids.flatten()) if int(v) in self.msize}
        # drop anything too small to carry a pose (see `min_px`)
        seen = {i: c for i, c in seen.items()
                if float(max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))) >= self.min_px}
        if not seen:
            self._maybe_dump(img, now)
            self.det_pub.publish(det)
            if self.dbg_pub is not None:
                # Keep drawing the last outline while it is lost: seeing WHERE
                # it went is the whole point, and a blank frame says nothing.
                self._draw(img.copy(), msg.header, self._last_poly, None, gap,
                           None)
            return

        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        # Solve every marker we can see, each reduced to the SAME landing point.
        solved = {}
        for i, c in seen.items():
            ok, rvec, tv = cv2.solvePnP(self.objp[i], c, self.K, self.dist,
                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            solved[i] = (self._landing_point(i, c, tv.reshape(3)), rvec, c)
            self._seen_by_id[i] += 1
        if not solved:
            self.det_pub.publish(det)
            if self.dbg_pub is not None:
                self._draw(img.copy(), msg.header, next(iter(seen.values())),
                           None, gap, None)
            return

        # Cross-check: two markers on one rigid deck MUST agree about where the
        # landing point is.  A sign error in `marker_offsets_m` shows up here as
        # a ~2*|offset| disagreement, which is the kind of mistake that is
        # otherwise silent and lands the vehicle a metre off.
        if len(solved) > 1:
            pts = [v[0] for v in solved.values()]
            self._agree.append(max(float(np.linalg.norm(a - b))
                                   for i, a in enumerate(pts)
                                   for b in pts[i + 1:]))

        # Prefer the marker that is comfortably in frame — which, as the
        # vehicle descends, hands over from the big one to the small one all by
        # itself.  "Best" = smallest fill, i.e. most margin before it outgrows
        # the view; ties broken toward the larger marker (better pose accuracy).
        def fill_of(c):
            return float(max(
                np.max(np.abs(c[:, 0] - cx)) / max(cx, self._img_w - cx),
                np.max(np.abs(c[:, 1] - cy)) / max(cy, self._img_h - cy)))

        fills = {i: fill_of(v[2]) for i, v in solved.items()}
        usable = [i for i, f in fills.items() if f < 1.0] or list(solved)
        # Bigger is more accurate; between equals (the two flanking markers are
        # the same size) take the one with more margin left in frame.  This
        # choice still decides which marker is REPORTED and drawn.
        best = max(usable, key=lambda i: (self.msize[i], -fills[i]))
        # ...but the landing point itself is the MEAN over the winner's SIZE
        # CLASS, not the winner alone.  The deck is deliberately symmetric — the
        # two big markers sit at -1.1 m and +1.1 m about one landing point — so
        # whatever residual each carries (perspective, corner noise, the
        # homography's range term) points the OPPOSITE way on its mirror, and
        # averaging the pair cancels it instead of inheriting whichever half
        # won.  That symmetry is the entire reason the SDF places them in a
        # pair, and picking one threw it away: over the dual-sighting frames of
        # the last run the two disagreed by 0.23 m on average and 0.65 m at
        # worst, all of it landing on the descent target as a bias.
        # Restricted to equal sizes on purpose — a marker is averaged with its
        # mirror, never with the small centre one, whose pose at 30-50 px is far
        # noisier and would be dragged in at equal weight.
        pair = [i for i in usable if self.msize[i] == self.msize[best]]
        t = np.mean([solved[i][0] for i in pair], axis=0)
        rvec, c = solved[best][1], solved[best][2]
        self._last_fill = fills[best]
        self._hits_by_id[best] += 1
        self._last_poly = c
        self._last_id = best
        if self._last_det is not None and (now - self._last_det) > 1.0:
            self.get_logger().info(
                f'detection RESUMED after {now - self._last_det:.1f}s gap')
        self._last_det = now
        self._gap_logged = False
        self._hits += 1
        det.data = True
        self.det_pub.publish(det)

        # `t` is the LANDING POINT, not the marker centre — the offset was
        # already applied above, so downstream never learns which marker won.
        q = rvec_to_quat(rvec)
        m = PoseStamped()
        m.header.stamp = msg.header.stamp        # IMAGE capture time
        m.header.frame_id = self.frame_id
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (
            float(t[0]), float(t[1]), float(t[2]))
        m.pose.orientation.w, m.pose.orientation.x = float(q[0]), float(q[1])
        m.pose.orientation.y, m.pose.orientation.z = float(q[2]), float(q[3])
        self.pose_pub.publish(m)

        # Centring error as plain numbers: dx, dy, and the distance between the
        # landing point and the optical axis, all in metres.
        e = Vector3Stamped()
        e.header.stamp = msg.header.stamp
        e.header.frame_id = self.frame_id
        e.vector.x, e.vector.y = float(t[0]), float(t[1])
        e.vector.z = float(np.hypot(t[0], t[1]))
        self.err_pub.publish(e)

        if self.dbg_pub is not None:
            self._draw(img, msg.header, c, t, 0.0, best)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
