"""Geometry, stale/uncalibrated rejection, uniqueness and image stride regressions."""
from types import SimpleNamespace
import unittest
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from jol_perception.fusion_core import (DroneNetwork, calibration_error, image_bgr, project_tracks,
                                        fuse, reacquisition_regions, transform_matrix)


def rig_transforms(yaw, pitch):
    """Independent fixture for the Gazebo pan/tilt pivots and physical lever arms."""
    def pose(xyz, rpy=(0., 0., 0.)):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        result[:3, 3] = xyz
        return result
    world_from_mount = pose([0, 0, .7], [0, 0, yaw]) @ pose([0, 0, .3], [0, pitch, 0])
    world_from_camera = (world_from_mount @ pose([.14, -.145, 1.0125])
                         @ pose([0, 0, 0], [-np.pi/2, 0, -np.pi/2]))
    world_from_lidar = world_from_mount @ pose([0, 0, 1.047])
    return world_from_camera, world_from_lidar


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.info = dict(k=[500., 0., 320., 0., 500., 240., 0., 0., 1.], d=[0.]*5,
                         width=640, height=480, frame_id='camera', distortion_model='plumb_bob')
        self.track = dict(id=7, position=[0., 0., 10.], velocity=[1., 0., 0.], size=[2., 2., 2.],
                          nearest_distance=9., observed=True)
        self.packet = dict(stamp=1., frame_id='lidar', tracks=[self.track])

    def test_intrinsics_and_frame_must_be_valid(self):
        self.assertEqual(calibration_error(self.info, 640, 480, 'camera'), '')
        self.assertEqual(calibration_error(self.info, 640, 480, 'wrong'), 'camera_info_mismatch')
        self.info['k'][0] = 0.
        self.assertEqual(calibration_error(self.info, 640, 480, 'camera'), 'invalid_camera_intrinsics')

    def test_projection_fusion_and_one_to_one(self):
        matrix = transform_matrix([0., 0., 0.], [0., 0., 0., 1.])
        projections, reason = project_tracks(self.packet, 1.1, matrix, self.info)
        self.assertEqual(reason, 'ready')
        self.assertTrue(np.allclose(projections[0]['position'], [.1, 0, 10.]))
        detections = [dict(bbox=projections[0]['bbox'], class_name='quadcopter', confidence=.9)] * 2
        result = fuse(projections, detections, matrix, self.info)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 7)
        self.assertFalse(result[0]['subtype_verified'])
        self.assertEqual(result[0]['subtype_status'], 'unvalidated')
        self.assertTrue(np.allclose(result[0]['position'], [.1, 0, 10.]))
        self.assertTrue(np.isfinite(result[0]['camera_bearing_lidar_depth_position']).all())
        self.assertFalse(fuse(projections * 2, detections, matrix, self.info))

    def test_stale_behind_outside_and_coasting_do_not_fuse(self):
        self.assertEqual(project_tracks(self.packet, 1.2, np.eye(4), self.info), ([], 'stale_lidar'))
        for position in ([0, 0, -10], [100, 0, 10]):
            self.track['position'] = position
            self.assertFalse(project_tracks(self.packet, 1., np.eye(4), self.info)[0])
        self.track.update(position=[0., 0., 10.], observed=False)
        self.assertFalse(project_tracks(self.packet, 1., np.eye(4), self.info)[0])

    def test_raw_distortion_and_transform(self):
        self.track['position'] = [1., 0., 10.]
        plain = project_tracks(self.packet, 1., np.eye(4), self.info)[0][0]['bbox']
        self.info['d'][0] = .2
        distorted = project_tracks(self.packet, 1., np.eye(4), self.info)[0][0]['bbox']
        self.assertNotEqual(plain, distorted)
        with self.assertRaises(ValueError):
            transform_matrix([0, 0, 0], [0, 0, 0, 0])

    def test_coasting_search_uses_covariance_without_becoming_a_measurement(self):
        self.track.update(observed=False, confirmed=True, last_seen=.5, size=[.2, .2, .2],
                          state_covariance=(np.eye(6)*.001).tolist(),
                          prediction={'acceleration_std': 2.})
        small = reacquisition_regions(self.packet, 1., np.eye(4), self.info)
        self.assertEqual(len(small), 1)
        self.assertEqual(small[0]['region'][2:], [128, 128])
        self.assertEqual(small[0]['proposal_source'], 'kalman_reacquisition_roi')
        self.assertEqual(small[0]['observation_age_s'], .5)
        self.track['state_covariance'] = np.eye(6).tolist()
        large = reacquisition_regions(self.packet, 1., np.eye(4), self.info)
        self.assertGreater(large[0]['region'][2], small[0]['region'][2])
        self.assertEqual(project_tracks(self.packet, 1., np.eye(4), self.info)[0], [])
        search, _ = project_tracks(self.packet, 1., np.eye(4), self.info, search_max_age=1.)
        detected = dict(bbox=search[0]['bbox'], class_name='quadcopter', confidence=.9)
        self.assertEqual(fuse(search, [detected], np.eye(4), self.info), [])
        self.assertFalse(self.track['observed'])
        for stamp in (.9, 1.2):  # no future posterior and no stale packet
            self.assertEqual(reacquisition_regions(self.packet, stamp, np.eye(4), self.info), [])
        for change in ({'last_seen': -.1}, {'last_seen': 2.}, {'confirmed': False},
                       {'state_covariance': -np.eye(6)}, {'state_covariance': None}):
            packet = dict(self.packet, tracks=[dict(self.track, **change)])
            self.assertEqual(reacquisition_regions(packet, 1., np.eye(4), self.info), [])

    def test_search_roi_needs_real_neural_evidence_and_preserves_full_frame(self):
        output = np.zeros((1, 6, 8400), np.float32)
        calls = []
        detector = DroneNetwork.__new__(DroneNetwork)
        detector.names, detector.input_name, detector.confidence = {0: 'quadcopter', 1: 'fixed-wing'}, 'image', .35
        detector.session = SimpleNamespace(run=lambda *args: calls.append(1) or [output])
        region = dict(region=[64, 64, 128, 128], proposal_source='kalman_reacquisition_roi')
        image = np.zeros((256, 256, 3), np.uint8)
        self.assertEqual(detector.infer(image, [region]), [])
        self.assertEqual(len(calls), 2)  # full frame still runs before the search
        neural = output.copy()
        neural[0, :, 0] = [320, 320, 100, 100, .9, .1]
        outputs = iter([output, neural])
        detector.session = SimpleNamespace(run=lambda *args: [next(outputs)])
        result = detector.infer(image, [region])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['proposal_source'], 'kalman_reacquisition_roi')

    def test_completion_rechecks_original_exposure_against_new_tracks(self):
        try:
            from jol_perception.camera_fusion import CameraFusion, PipelineAudit
            from sensor_msgs.msg import Image
        except ImportError:
            self.skipTest('ROS environment not sourced')
        published = []
        publisher = SimpleNamespace(publish=lambda message: published.append(message))
        msg = Image()
        msg.header.frame_id = 'camera'
        msg.header.stamp.sec, msg.header.stamp.nanosec = 1, 100000000
        msg.width, msg.height = 640, 480
        fresh_packet = dict(stamp=1.1, frame_id='lidar', tracks=[])
        node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1200000000)),
            geometry=lambda original: ([], np.eye(4), fresh_packet, 'ready') if original is msg else None,
            info=self.info, max_age=.8, network=SimpleNamespace(model='fixture'), processed=0, epoch=0,
            pipeline=PipelineAudit(lambda: dict(monotonic_ns=0, ros_s=1.2)),
            get_parameter=lambda name: SimpleNamespace(value=True if name == 'calibration_valid' else 'fixture'),
            detection_pub=publisher, fused_pub=publisher, image_pub=publisher, publish_markers=lambda *args: None)
        context = dict(message=msg, packet=dict(stamp=.5), projections=[], reason='stale_lidar', regions=[])
        pixels = np.arange(480*640*3, dtype=np.uint8).reshape(480, 640, 3)
        CameraFusion.publish_result(node, context, pixels, [], 50.)
        # The published packed image must preserve every final annotated pixel.
        self.assertEqual(bytes(published[2].data), pixels.tobytes())
        self.assertEqual(published[2].step, 1920)
        self.assertEqual(published[2].encoding, 'bgr8')
        stages = ['publish', 'annotation_start', 'annotation_ready', 'image_publish_complete', 'publish_complete']
        ticks = [node.last_timing[name]['monotonic_ns'] for name in stages]
        self.assertEqual(ticks, sorted(ticks))
        result = json.loads(published[1].data)
        self.assertEqual(result['fusion_reason'], 'ready')
        self.assertEqual(result['stamp'], 1.1)
        self.assertEqual(result['time_difference_s'], 0.)
        node.get_parameter = lambda name: SimpleNamespace(value=False if name == 'calibration_valid' else 'fixture')
        published.clear()
        CameraFusion.publish_result(node, context, np.zeros((480, 640, 3), np.uint8), [], 50.)
        self.assertEqual(json.loads(published[1].data)['fusion_reason'], 'uncalibrated')
        self.assertFalse(json.loads(published[1].data)['tracks'])

    def test_rotating_pan_tilt_preserves_world_motion_and_scan_range(self):
        _, scan_lidar = rig_transforms(0., 0.)
        nearest = np.array([49.5, 4., 9.9])
        measured_range = float(np.linalg.norm(nearest - scan_lidar[:3, 3]))
        track = dict(id=17, position=[50., 4., 10.], velocity=[0., 10., 0.], size=[1., 1., .5],
                     nearest_distance=measured_range, nearest_point=nearest.tolist(), observed=True)
        packet = dict(stamp=10., frame_id='world', sensor_frame='jol/lidar',
                      sensor_origin=scan_lidar[:3, 3].tolist(), stabilized=True, tracks=[track])
        expected_world = np.array([50., 5., 10.])  # dt=.1 s, velocity remains in world.
        image_pixels = []
        for yaw, pitch in [(0., 0.), (.15, 0.), (0., -.12), (.15, -.12)]:
            world_from_camera, _ = rig_transforms(yaw, pitch)
            camera_from_world = np.linalg.inv(world_from_camera)
            projections, reason = project_tracks(packet, 10.1, camera_from_world, self.info)
            self.assertEqual(reason, 'ready')
            self.assertEqual(len(projections), 1)
            camera_point = camera_from_world[:3, :3] @ expected_world + camera_from_world[:3, 3]
            pixel = (np.asarray(self.info['k']).reshape(3, 3) @ camera_point)[:2] / camera_point[2]
            image_pixels.append(pixel)
            width, height = projections[0]['bbox'][2:]
            detection = dict(bbox=[pixel[0]-width/2, pixel[1]-height/2, width, height],
                             class_name='quadcopter', confidence=.9)
            result = fuse(projections, [detection], camera_from_world, self.info)[0]
            self.assertTrue(np.allclose(result['position'], expected_world))
            self.assertEqual(result['position_frame_id'], 'world')
            self.assertEqual(result['velocity'], [0., 10., 0.])
            self.assertTrue(np.allclose(result['camera_bearing_lidar_depth_position'], expected_world))
            self.assertEqual(result['nearest_distance'], measured_range)
            self.assertEqual(result['nearest_distance_stamp'], 10.)
            self.assertEqual(result['nearest_distance_sensor_frame'], 'jol/lidar')
            self.assertEqual(result['nearest_point_frame_id'], 'world')
            self.assertEqual(result['nearest_distance_sensor_origin'], packet['sensor_origin'])
            self.assertEqual(result['nearest_point'], nearest.tolist())
        self.assertGreater(image_pixels[1][0], image_pixels[0][0] + 50)  # pan changes u.
        self.assertGreater(image_pixels[2][1], image_pixels[0][1] + 50)  # tilt changes v.
        # The camera offset must rotate around the pivots, not remain at a fixed world XYZ.
        rotated_camera, _ = rig_transforms(.15, -.12)
        self.assertFalse(np.allclose(rotated_camera[:3, 3], [.14, -.145, 2.0125]))

    def test_dynamic_tf_uses_exposure_time_and_never_latest_fallback(self):
        try:
            from jol_perception.camera_fusion import CameraFusion
            from geometry_msgs.msg import TransformStamped
            from rclpy.duration import Duration
            from sensor_msgs.msg import Image
            from tf2_ros import Buffer
        except ImportError:
            self.skipTest('ROS environment not sourced')
        buffer = Buffer(cache_time=Duration(seconds=10.))

        def insert(parent, child, xyz, rpy, stamp, static=False):
            msg = TransformStamped()
            msg.header.frame_id, msg.child_frame_id = parent, child
            msg.header.stamp.sec = int(stamp)
            msg.header.stamp.nanosec = round((stamp-int(stamp))*1e9)
            msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, xyz)
            q = Rotation.from_euler('xyz', rpy).as_quat()
            msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = q
            if static:
                buffer.set_transform_static(msg, 'rotation_fixture')
            else:
                buffer.set_transform(msg, 'rotation_fixture')

        insert('world', 'jol/base', [0, 0, 0], [0, 0, 0], 0., True)
        insert('jol/turret_pitch', 'jol/camera_link', [.14, -.145, 1.0125], [0, 0, 0], 0., True)
        insert('jol/camera_link', 'camera', [0, 0, 0], [-np.pi/2, 0, -np.pi/2], 0., True)
        for stamp, yaw, pitch in [(1., 0., 0.), (2., .2, -.1)]:
            insert('jol/base', 'jol/turret_yaw', [0, 0, .7], [0, 0, yaw], stamp)
            insert('jol/turret_yaw', 'jol/turret_pitch', [0, 0, .3], [0, pitch, 0], stamp)
        warnings = []
        packet = dict(stamp=1.5, frame_id='world', tracks=[dict(self.track, position=[50., 4., 10.])])
        node = SimpleNamespace(tf=buffer, history=[packet], info=self.info, tolerance=.15,
            get_parameter=lambda _: SimpleNamespace(value=True),
            get_logger=lambda: SimpleNamespace(warning=lambda message, **kwargs: warnings.append(message)))
        image = Image()
        image.header.frame_id, image.width, image.height = 'camera', 640, 480
        image.header.stamp.sec, image.header.stamp.nanosec = 1, 500000000
        projections, matrix, _, reason = CameraFusion.geometry(node, image)
        self.assertEqual(reason, 'ready')
        self.assertEqual(len(projections), 1)
        expected, _ = rig_transforms(.1, -.05)
        self.assertTrue(np.allclose(matrix, np.linalg.inv(expected), atol=1e-12))
        latest, _ = rig_transforms(.2, -.1)
        self.assertFalse(np.allclose(matrix, np.linalg.inv(latest)))
        # There is a latest transform at t=2, but no transform at exposure t=2.1.
        packet['stamp'] = 2.1
        image.header.stamp.sec, image.header.stamp.nanosec = 2, 100000000
        self.assertEqual(CameraFusion.geometry(node, image)[3], 'tf_or_geometry_unavailable')
        self.assertEqual(len(warnings), 1)
        packet['stamp'] = 0.
        image.header.stamp.sec, image.header.stamp.nanosec = 0, 0
        self.assertEqual(CameraFusion.geometry(node, image)[3], 'invalid_image_stamp')

    def test_neural_nms_preserves_contradictory_subtype_evidence(self):
        # Stub only the network tensors; this tests decoding, not neural accuracy.
        full, crop = np.zeros((1, 6, 8400), np.float32), np.zeros((1, 6, 8400), np.float32)
        full[0, :, 0] = [320, 320, 80, 80, .1, .8]
        crop[0, :, 0] = [320, 320, 160, 160, .6, .1]
        outputs = iter([full, crop])
        detector = DroneNetwork.__new__(DroneNetwork)
        detector.names, detector.input_name, detector.confidence = {0: 'quadcopter', 1: 'fixed-wing'}, 'image', .35
        detector.session = SimpleNamespace(run=lambda *args: [next(outputs)])
        result = detector.infer(np.zeros((256, 256, 3), np.uint8), [(64, 64, 128, 128)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['class_name'], 'fixed-wing')
        self.assertTrue(result[0]['subtype_ambiguous'])
        self.assertFalse(result[0]['subtype_verified'])
        self.assertEqual(result[0]['subtype_status'], 'ambiguous')
        self.assertEqual({c['class_name'] for c in result[0]['subtype_candidates']}, {'fixed-wing', 'quadcopter'})

    def test_same_anchor_keeps_both_class_scores_without_changing_winner(self):
        raw = np.zeros((1, 6, 8400), np.float32)
        raw[0, :, 0] = [320, 320, 80, 80, .7, .8]
        detector = DroneNetwork.__new__(DroneNetwork)
        detector.names, detector.input_name, detector.confidence = {0: 'quadcopter', 1: 'fixed-wing'}, 'image', .35
        detector.session = SimpleNamespace(run=lambda *args: [raw])
        result = detector.infer(np.zeros((640, 640, 3), np.uint8))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['class_name'], 'fixed-wing')
        self.assertAlmostEqual(result[0]['confidence'], .8)
        self.assertFalse(result[0]['subtype_verified'])
        self.assertEqual(result[0]['subtype_status'], 'ambiguous')
        self.assertTrue(result[0]['subtype_ambiguous'])
        self.assertEqual({c['class_name'] for c in result[0]['subtype_candidates']}, {'fixed-wing', 'quadcopter'})
        raw[0, 4, 0] = .1
        single = detector.infer(np.zeros((640, 640, 3), np.uint8))[0]
        self.assertEqual(single['class_name'], 'fixed-wing')
        self.assertAlmostEqual(single['confidence'], .8)
        self.assertFalse(single['subtype_ambiguous'])
        self.assertFalse(single['subtype_verified'])
        self.assertEqual(single['subtype_status'], 'unvalidated')

    def test_crop_cannot_claim_pixels_outside_its_input(self):
        empty, cropped = np.zeros((1, 6, 8400), np.float32), np.zeros((1, 6, 8400), np.float32)
        cropped[0, :, 0] = [0, 320, 100, 80, .8, .1]
        outputs = iter([empty, cropped])
        detector = DroneNetwork.__new__(DroneNetwork)
        detector.names, detector.input_name, detector.confidence = {0: 'quadcopter', 1: 'fixed-wing'}, 'image', .35
        detector.session = SimpleNamespace(run=lambda *args: [next(outputs)])
        result = detector.infer(np.zeros((480, 640, 3), np.uint8), [(100, 100, 128, 128)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bbox'][0], 100.)
        self.assertTrue(result[0]['box_clipped'])

    def test_rgb_padded_rows(self):
        image = image_bgr(SimpleNamespace(width=2, height=2, step=8, encoding='rgb8',
                          data=bytes([30, 20, 10, 60, 50, 40, 0, 0]*2)))
        self.assertEqual(image.tolist(), [[[10, 20, 30], [40, 50, 60]]]*2)


if __name__ == '__main__':
    unittest.main()
