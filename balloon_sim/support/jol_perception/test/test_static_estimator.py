"""Sensor-only geometry, adverse-input and independent gimbal regressions."""
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from jol_perception.static_estimator_core import (associate_surface, detect_orange_marker,
    gimbal_center_command, detection_center, lidar_acquisition_candidate, lidar_acquisition_command,
    validate_detection, display_clouds, sensor_quality_config,
    calibration_revision, pose_motion, add_sensor_uncertainty)


class StaticEstimatorTests(unittest.TestCase):
    def test_display_filter_keeps_raw_data_and_removes_only_visual_clutter(self):
        raw=np.array([[10.,0.,0.],[10.,0.,.05],[40.,0.,8.],[40.01,0.,8.],
                      [-10.,0.,2.],[70.,0.,3.],[np.nan,0.,2.]])
        before=raw.copy()
        context,target=display_clouds(raw,[0.,0.,.855],[40.,0.,8.])
        np.testing.assert_equal(raw,before)
        self.assertEqual(len(context),2)
        self.assertTrue(np.all(context[:,2]>=.3))
        self.assertEqual(len(target),1)
        np.testing.assert_allclose(target[0],[40.,0.,8.])
        self.assertEqual(display_clouds(raw,[0.,0.,.855])[1].shape,(0,3))

    def setUp(self):
        self.info = dict(k=[500., 0., 320., 0., 500., 240., 0., 0., 1.], d=[0.]*5,
                         width=640, height=480, frame_id='optical', distortion_model='plumb_bob')
        self.detection = dict(bbox=[307.5, 227.5, 25., 25.], confidence=.95)
        self.packet = dict(stamp=2., frame_id='optical', width=640, height=480,
                           detections=[self.detection])
        grid = np.array([[x, y, 40.] for x in (-.4, 0., .4) for y in (-.4, 0., .4)])
        self.points = np.vstack([grid, [4., 3., 40.], [0., 0., -2.]])

    def test_bearing_uses_actual_associated_lidar_range(self):
        result = associate_surface(self.points, [0., 0., 0.], np.eye(4), self.detection, self.info)
        np.testing.assert_allclose(result['position'], [0., 0., 40.], atol=1e-8)
        self.assertEqual(result['points_count'], 9)
        self.assertEqual(result['nearest_distance_m'], 40.)
        self.assertGreater(np.linalg.eigvalsh(result['covariance']).min(), 0.)
        diagnostic = result['reprojection_consistency']
        self.assertEqual(diagnostic['nearest_center_distance_px'], 0.)
        self.assertIn('not_independent_calibration', diagnostic['semantics'])

    def test_clock_and_extrinsic_covariance_propagates_range_and_gimbal_motion(self):
        base = associate_surface(self.points, [0., 0., 0.], np.eye(4), self.detection, self.info)
        config = sensor_quality_config()
        stationary = add_sensor_uncertainty(dict(base), np.eye(4), np.zeros(3), config,
                                            (np.zeros(3), np.zeros(3)), (np.zeros(3), np.zeros(3)))
        moving = add_sensor_uncertainty(dict(base), np.eye(4), np.zeros(3), config,
                                       (np.zeros(3), [0., .8, 0.]), (np.zeros(3), np.zeros(3)))
        extra = np.asarray(moving['covariance'])-stationary['covariance']
        self.assertAlmostEqual(extra[0, 0], (.001*.8*40)**2)
        self.assertGreater(stationary['covariance'][0][0], base['covariance'][0][0])
        with self.assertRaisesRegex(ValueError, 'sensor_position_uncertainty_exceeded'):
            add_sensor_uncertainty(dict(base), np.eye(4), np.zeros(3),
                sensor_quality_config(camera_time_std_s=.02),
                (np.zeros(3), [0., .8, 0.]), (np.zeros(3), np.zeros(3)))
        with self.assertRaisesRegex(ValueError, 'sensor_position_uncertainty_exceeded'):
            add_sensor_uncertainty(dict(base), np.eye(4), np.zeros(3),
                sensor_quality_config(extrinsic_rotation_std_deg=1.),
                (np.zeros(3), np.zeros(3)), (np.zeros(3), np.zeros(3)))

    def test_calibration_revision_changes_on_intrinsic_clock_and_provenance(self):
        config = sensor_quality_config()
        revision = calibration_revision(self.info, 'gazebo_sdf', config)
        self.assertEqual(revision, calibration_revision(dict(reversed(list(self.info.items()))), 'gazebo_sdf', config))
        for info, source, quality in ((dict(self.info, k=[501.]+self.info['k'][1:]), 'gazebo_sdf', config),
                                      (self.info, 'hardware_import', config),
                                      (self.info, 'gazebo_sdf', sensor_quality_config(camera_time_offset_s=-.01))):
            self.assertNotEqual(revision, calibration_revision(info, source, quality))
        self.assertNotEqual(revision, calibration_revision(self.info, 'gazebo_sdf', config,
            {'lidar': dict(parent='base', pose=[.1, 0., 0., 0., 0., 0., 1.])}))
        for values in (dict(camera_time_std_s=-.1), dict(camera_time_offset_s=1.),
                       dict(extrinsic_rotation_std_deg=float('nan')), dict(max_position_std_m=2.)):
            with self.assertRaisesRegex(ValueError, 'invalid_sensor_quality_configuration'):
                sensor_quality_config(**values)

    def test_pose_motion_uses_world_angular_velocity_and_translation(self):
        previous, current = np.eye(4), np.eye(4)
        current[:3, :3] = Rotation.from_rotvec([0., .016, 0.]).as_matrix()
        current[:3, 3] = [.02, 0., 0.]
        velocity, angular = pose_motion(previous, current, .02)
        np.testing.assert_allclose(velocity, [1., 0., 0.])
        np.testing.assert_allclose(angular, [0., .8, 0.])
        with self.assertRaises(ValueError):
            pose_motion(previous, current, 0.)

    def test_rotating_camera_pose_does_not_rotate_world_estimate(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('xyz', [.2, -.1, .8]).as_matrix()
        pose[:3, 3] = [3., 4., 2.]
        world = self.points@pose[:3, :3].T+pose[:3, 3]
        result = associate_surface(world, pose[:3, 3], pose, self.detection, self.info)
        np.testing.assert_allclose(result['position'], pose[:3, :3]@[0., 0., 40.]+pose[:3, 3], atol=1e-8)

    def test_tilted_surface_uses_supported_plane_intersection_not_sample_median(self):
        points = np.array([[x, y, 40.+.5*x] for x in (-.45, -.3, -.15, .1)
                           for y in (-.4, 0., .4)])
        original = associate_surface(points, [0., 0., 0.], np.eye(4), self.detection,
                                     self.info, use_surface_plane=False)
        result = associate_surface(points, [0., 0., 0.], np.eye(4), self.detection, self.info)
        self.assertGreater(abs(original['camera_depth_m']-40.), .1)
        np.testing.assert_allclose(result['position'], [0., 0., 40.], atol=1e-8)
        self.assertEqual(result['surface_model'], 'dominant_plane_ray')
        self.assertTrue(result['surface_fit']['accepted'])
        self.assertEqual(result['surface_fit']['inlier_count'], 12)
        self.assertGreaterEqual(result['covariance'][2][2], .15**2+.02**2-1e-12)
        # Every repeat produces the same fit; no experiment-dependent RNG drift.
        self.assertEqual(result, associate_surface(points, [0., 0., 0.], np.eye(4),
                                                   self.detection, self.info))
        steep = np.array([[x, y, 40.+4*x] for x in (-.05, -.025, 0., .025) for y in (-.4, 0., .4)])
        amplified = associate_surface(steep, [0., 0., 0.], np.eye(4), self.detection, self.info)
        baseline = associate_surface(steep, [0., 0., 0.], np.eye(4), self.detection,
                                     self.info, use_surface_plane=False)
        self.assertTrue(amplified['surface_fit']['accepted'])
        self.assertGreater(amplified['covariance'][2][2], baseline['covariance'][2][2])

    def test_unsupported_sparse_degenerate_and_nonplanar_clouds_keep_guarded_median(self):
        cases = [
            (self.points[:3], 'insufficient_plane_support'),
            (np.tile(self.points[:3], (4, 1)), 'insufficient_plane_support'),
            (np.array([[x, 0., 40.] for x in np.linspace(-.4, .4, 9)]), 'degenerate_inlier_geometry'),
            (np.array([[x, y, 40.+z] for x in (-.4, 0., .4) for y in (-.4, 0., .4)
                       for z in (-.2, 0., .2)]), 'insufficient_plane_support'),
            (np.array([[x, y, 40.+2*x] for x in (.2, .3, .4) for y in (-.4, 0., .4)]),
             'camera_ray_outside_plane_support'),
        ]
        for points, reason in cases:
            with self.subTest(reason=reason):
                result = associate_surface(points, [0., 0., 0.], np.eye(4), self.detection, self.info)
                baseline = associate_surface(points, [0., 0., 0.], np.eye(4), self.detection,
                                             self.info, use_surface_plane=False)
                self.assertFalse(result['surface_fit']['accepted'])
                self.assertEqual(result['surface_fit']['reason'], reason)
                self.assertEqual(result['surface_model'], 'median_depth')
                self.assertEqual(result['position'], baseline['position'])
                self.assertEqual(result['covariance'], baseline['covariance'])

    def test_clutter_depth_ambiguity_and_sparse_returns_reject(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous_lidar_depth'):
            associate_surface(np.vstack([self.points[:9], self.points[:9]*2]),
                              [0., 0., 0.], np.eye(4), self.detection, self.info)
        with self.assertRaisesRegex(ValueError, 'insufficient_associated_lidar_points'):
            associate_surface(self.points[:2], [0., 0., 0.], np.eye(4), self.detection, self.info)

    def test_camera_missing_stale_calibration_and_invalid_bbox_reject(self):
        self.assertIs(validate_detection(self.packet, self.info, 2.1), self.detection)
        for packet in (dict(self.packet, stamp=0.), dict(self.packet, stamp=1.),
                       dict(self.packet, stamp=2.101),
                       dict(self.packet, frame_id='wrong'), dict(self.packet, detections=[]),
                       dict(self.packet, detections=[dict(self.detection, bbox=[0., 0., float('nan'), 10.])]),
                       dict(self.packet, detections=[dict(self.detection, box_clipped=True)])):
            with self.assertRaises(ValueError):
                validate_detection(packet, self.info, 2.1)
        with self.assertRaisesRegex(ValueError, 'invalid_camera_intrinsics'):
            validate_detection(self.packet, dict(self.info, k=[0.]*9), 2.1)

    def test_marker_is_camera_pixel_evidence_and_no_marker_is_empty(self):
        image = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(detect_orange_marker(image), [])
        cv2.rectangle(image, (311, 230), (328, 248), (0, 110, 240), -1)
        detections = detect_orange_marker(image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['source'], 'simulation_color_marker')
        self.assertEqual(detections[0]['bbox'], [311., 230., 18., 19.])
        self.assertEqual(detections[0]['center_px'], [319.5, 239.])

    def test_opencv_pixel_center_corrects_half_pixel_bias_without_changing_yolo(self):
        image = np.zeros((480, 640, 3), np.uint8)
        cv2.rectangle(image, (311, 230), (328, 248), (0, 110, 240), -1)
        detection = detect_orange_marker(image)[0]
        np.testing.assert_equal(detection_center(detection), [319.5, 239.])
        np.testing.assert_equal(detection_center(dict(bbox=detection['bbox'])), [320., 239.5])
        info = dict(self.info, k=[500., 0., 319.5, 0., 500., 239., 0., 0., 1.])
        result = associate_surface(self.points, [0., 0., 0.], np.eye(4), detection, info)
        np.testing.assert_allclose(result['position'], [0., 0., 40.], atol=1e-8)
        np.testing.assert_allclose(gimbal_center_command(detection['bbox'], info, .2, -.15,
            center=detection_center(detection)), [.2, -.15], atol=1e-9)
        for center in ([319.5], [float('nan'), 239.], [0., 0.]):
            with self.assertRaisesRegex(ValueError, 'invalid_detection_center'):
                validate_detection(dict(self.packet, detections=[dict(detection, center_px=center)]), self.info, 2.1)

    def test_lidar_acquisition_uses_whole_airborne_cluster_and_real_frame(self):
        target = np.array([[x, -39.4, z] for x in (-.5, 0., .5) for z in (7.5, 8., 8.5)])
        grounded = np.array([[18.+x, -11., z] for x in np.arange(-1.5, 1.6, .5)
                             for z in np.arange(.5, 3.1, .5)])
        column = np.array([[-17.+x, 19., z] for x in (-.5, 0., .5) for z in np.arange(.5, 5.1, .5)])
        points = np.vstack([target, grounded, column, [0., 0., 0.], [float('nan'), 0., 8.]])
        origin = np.array([.1, -.145, .855])
        candidate = lidar_acquisition_candidate(points, origin)
        np.testing.assert_allclose(candidate, [0., -39.4, 8.])
        self.assertIsNone(lidar_acquisition_candidate(np.vstack([grounded, column]), origin))
        self.assertIsNone(lidar_acquisition_candidate(np.array([[0., -40., 8.]]), origin))
        yaw, pitch = lidar_acquisition_command(candidate, np.eye(4), [.25, -.145, .760])
        self.assertLess(yaw, -1.5)  # camera's right is world -Y, not positive yaw.
        self.assertLess(pitch, 0.)
        pose = np.eye(4); pose[:3, :3] = Rotation.from_euler('z', np.pi/2).as_matrix()
        rotated = lidar_acquisition_command([40., 0., 8.], pose, [0., 0., .76])
        self.assertAlmostEqual(rotated[0], -np.pi/2)
        self.assertIsNone(lidar_acquisition_command([1., 0., 8.], np.eye(4), [0., 0., .76]))

    def test_visual_alignment_sign_limits_and_center(self):
        command = gimbal_center_command(self.detection['bbox'], self.info, .2, -.15)
        np.testing.assert_allclose(command, [.2, -.15], atol=1e-9)
        right_down = gimbal_center_command([350., 270., 20., 20.], self.info, .2, -.15)
        self.assertLess(right_down[0], .2)
        self.assertGreater(right_down[1], -.15)
        self.assertLessEqual(abs(right_down[0]-.2), .04+1e-9)


if __name__ == '__main__':
    unittest.main()
