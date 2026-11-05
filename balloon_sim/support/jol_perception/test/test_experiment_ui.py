"""실제 Qt 위젯과 격리 ROS 서비스로 표시·요청·실패 처리를 검사한다."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import rclpy
from rclpy.context import Context
from rclpy.parameter import Parameter
from rclpy.task import Future
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from jol_perception.experiment_ui import Dashboard, ExperimentNode, SessionRecorder


class ExperimentUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.context = Context()
        rclpy.init(context=self.context, domain_id=179)
        self.node = ExperimentNode(context=self.context, parameter_overrides=[
            Parameter('record_dir', value=self.directory.name), Parameter('service_timeout', value=.15),
            Parameter('raw_bag_uri', value='/known/recordings/test_trial')])
        self.window = Dashboard(self.node)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.node.destroy_node()
        rclpy.shutdown(context=self.context)
        self.directory.cleanup()

    def wait_for(self, condition, seconds=3):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline and not condition():
            QtTest.QTest.qWait(20)
        self.assertTrue(condition())

    def click(self, name):
        button = self.window.findChild(QtWidgets.QPushButton, name)
        self.assertIsNotNone(button)
        QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)

    def test_flight_path_display_is_separate_from_preview_and_sends_no_command(self):
        from jol_perception.observation_path import build_corner_plan
        now=self.node.get_clock().now().nanoseconds*1e-9
        plan=build_corner_plan([-5.,-6.,.1],[5.,-6.,8.])
        plan.update(stamp=now,execution_policy='gazebo_classic_bounded_observation')
        self.node.receive_packet('flight_plan',String(data=json.dumps(plan)))
        self.node.receive_packet('flight',String(data=json.dumps(dict(stamp=now,phase='transfer',target_id=1,
            reason='',position=[-5.,-6.,7.],velocity=[0.,0.,0.],tracking_error_m=.03))))
        self.window.refresh_observation(now)
        self.assertTrue(self.window.plot.flight_path_display)
        self.assertTrue(self.window.plot.observation_path)
        self.assertIn('Gazebo',self.window.observer_policy.text())
        self.assertFalse(self.window.preview_button.isEnabled())
        self.assertIsNone(self.node.pending)
        self.node.receive_packet('flight',String(data=json.dumps(dict(stamp=now,phase='aborted',reason='test'))))
        self.window.refresh_observation(now)
        self.assertFalse(self.window.plot.observation_path)

    def test_actual_qt_buttons_wait_for_ros_response_and_record_events(self):
        calls = []
        def callback(request, response):
            calls.append('stop')
            response.success, response.message = True, '순찰 명령 수락'
            return response
        service = self.node.create_service(Trigger, '/jol/pointing/stop_tracking', callback)
        self.wait_for(lambda: self.node.control_clients['stop_tracking'].service_is_ready())
        self.click('recordButton')
        directory = self.node.recorder.directory
        self.assertIn('UI 기록 중', self.window.record_badge.text())
        self.assertEqual(self.window.record_button.text(), 'UI 기록 종료')
        self.assertIn('원시 rosbag', self.window.record_badge.toolTip())
        self.click('stopTrackingButton')
        self.assertIsNotNone(self.node.pending)
        self.assertFalse(self.window.stop_button.isEnabled())
        self.assertFalse(self.window.resume_button.isEnabled())
        self.wait_for(lambda: self.node.pending is None)
        self.window.refresh()
        self.assertEqual(calls, ['stop'])
        self.assertIn('성공', self.window.command_label.text())
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertNotIn('360° 순찰', self.window.mode_label.text())  # A service ack is not measured mode.
        self.click('recordButton')
        self.assertEqual(self.window.record_button.text(), 'UI 기록 시작')
        rows = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        responses = [row for row in rows if row['phase'] == 'response']
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]['action'], 'stop_tracking')
        self.assertTrue(responses[0]['success'])
        self.assertEqual(responses[0]['method'], 'qt_button')
        self.assertIn('monotonic_ns', responses[0])
        self.assertTrue((directory/'selected_target.csv').is_file())
        self.node.destroy_service(service)

    def test_timeout_is_unconfirmed_and_does_not_leave_buttons_disabled(self):
        class NoResponse:
            def service_is_ready(self):
                return True
            def call_async(self, request):
                return Future()
            def remove_pending_request(self, future):
                self.removed = future
        self.node.control_clients['stop_tracking'] = NoResponse()
        self.click('stopTrackingButton')
        self.assertIsNotNone(self.node.pending)
        self.wait_for(lambda: self.node.pending is None)
        self.window.refresh()
        self.assertTrue(self.window.resume_button.isEnabled())
        self.assertIn('미확인', self.window.command_label.text())
        self.assertEqual(self.node.events[-1]['phase'], 'timeout')
        self.assertIsNone(self.node.events[-1]['success'])

    def test_received_image_joint_elevation_and_stale_data_are_explicit(self):
        self.fixed_clock(10.)
        self.node.receive_packet('pointing', String(data=json.dumps(dict(stamp=10., mode='patrol',
            state='patrolling', auto_acquisition_enabled=False, target_id=None))))
        joint = JointState(name=['yaw_joint', 'pitch_joint'], position=[.3, -.3490658503988659])
        joint.header.stamp.sec = 10
        self.node.receive_joints(joint)
        image = Image(width=2, height=1, encoding='bgr8', step=6, data=bytes([255, 0, 0, 0, 255, 0]))
        self.node.receive_image(image)
        self.window.refresh()
        self.assertIn('360° 순찰', self.window.mode_label.text())
        self.assertIn('고각 20.0°', self.window.status_detail.text())
        pixel = self.window.pixmap.toImage().pixelColor(0, 0)
        self.assertEqual((pixel.red(), pixel.green(), pixel.blue()), (0, 0, 255))
        now = self.node.clocks()['sim_stamp']
        track = dict(id=3, position=[2., 3., 4.], velocity=[1., 0., 0.], observed=True,
                     nearest_distance=5.)
        self.node.receive_packet('tracks', String(data=json.dumps(dict(stamp=now-.8, tracks=[track]))))
        self.node.receive_packet('pointing', String(data=json.dumps(dict(stamp=now, mode='hold',
            state='stale_or_future_tracks', target_id=3))))
        row = self.node.snapshot()
        self.assertTrue(row['tracks_communication_fresh'])
        self.assertGreaterEqual(row['track_source_age_s'], .79)
        self.assertEqual(row['x_m'], 2.)  # Keep genuine measurements, show their age instead of hiding them.
        self.window.refresh()
        self.assertIn('전 갱신', self.window.status_detail.text())
        self.assertIn('LiDAR 통신 수신 중', self.window.status_detail.text())
        self.node.receive_packet('tracks', String(data=json.dumps(dict(stamp=now+5., tracks=[track]))))
        self.window.refresh()
        self.assertIn('미래 시각', self.window.status_detail.text())
        self.node.received['pointing'] -= int(3e9)
        self.window.refresh()
        self.assertIn('미확인', self.window.mode_label.text())
        self.node.receive_packet('tracks', String(data='{"stamp":10,"tracks":[{"id":1,"position":[0]}]}'))
        self.assertEqual(self.node.packets['tracks']['tracks'][0]['id'], 3)

    def fixed_clock(self, stamp):
        self.node.clocks = lambda: dict(sim_stamp=stamp, monotonic_ns=time.monotonic_ns(),
                                       wall_time='2026-10-05T00:00:00.000+00:00')

    def packet(self, key, **values):
        self.node.receive_packet(key, String(data=json.dumps(values)))

    def test_coasting_keeps_last_observation_age_and_distance_matches_table(self):
        # Real mounted02 regression: a 0.095-second state is already 2.595 seconds beyond its last observation.
        self.fixed_clock(42.1)
        track = dict(id=4, position=[2., 3., 4.], velocity=[1., 0., 0.], observed=False,
                     last_seen=39.505, nearest_distance=50.)
        self.packet('tracks', stamp=42.005, tracks=[track])
        self.packet('pointing', stamp=42.09, mode='tracking', target_id=4, source_stamp=42.005,
                    track_position=track['position'], track_velocity=track['velocity'],
                    observed=False, measurement_age_s=42.09-39.505)
        row = self.node.snapshot()
        self.assertAlmostEqual(row['position_source_age_s'], .095)
        self.assertAlmostEqual(row['last_observation_age_s'], 2.595)
        self.assertFalse(row['position_observed'])
        self.assertIsNone(row['nearest_surface_m'])
        self.window.refresh()
        self.assertIn('최종 실측 ROS 39.51', self.window.target_label.text())
        self.assertIn('예측', self.window.target_label.text())
        self.assertRegex(self.window.table.item(0, 1).text(), r'2\.(59|60)s 전 실측')
        self.assertIn('상태 갱신', self.window.table.item(0, 1).text())
        self.assertEqual(self.window.table.item(0, 6).text(), '—')
        self.assertIn('예측', self.window.target_card.detail.text())
        self.assertIn('2.59', self.window.observation_card.value.text())
        self.assertEqual(self.window.distance_card.value.text(), '— m')
        track.update(observed=True, last_seen=42.005)
        self.packet('tracks', stamp=42.005, tracks=[track])
        self.window.refresh()
        self.assertEqual(self.node.snapshot()['nearest_surface_m'], 50.)
        self.assertEqual(self.window.table.item(0, 6).text(), '50.00')
        self.assertEqual(self.window.distance_card.value.text(), '50.00 m')
        self.node.received['tracks'] -= int(3e9)
        self.window.refresh()
        self.assertIsNone(self.node.snapshot()['nearest_surface_m'])
        self.assertIsNone(self.node.snapshot()['surface_stamp'])
        self.assertEqual(self.window.table.item(0, 6).text(), '—')
        self.assertIn('표면 거리 —', self.window.target_label.text())
        self.assertEqual(self.window.distance_card.value.text(), '— m')

    def test_arm_feedback_requires_measured_alignment_not_command_success(self):
        self.fixed_clock(10.)
        self.packet('pointing', stamp=10., mode='tracking', target_id=1,
                    command_success=True, arm=dict(measured_angle_error_deg=63.7, aligned=False),
                    prediction=dict(position=[40., 20., 10.]))
        self.window.refresh()
        self.assertIn('63.7° · 회전 중', self.window.arm_feedback.text())
        self.assertNotIn('정렬 완료', self.window.arm_feedback.text())
        self.assertIn('40.0, 20.0, 10.0', self.window.prediction_label.text())
        self.node.packets['pointing']['arm'].update(measured_angle_error_deg=1.2, aligned=True)
        self.window.refresh()
        self.assertIn('정렬 완료', self.window.arm_feedback.text())
        self.node.received['pointing'] -= int(3e9)
        self.window.refresh()
        self.assertIn('실측 대기', self.window.arm_feedback.text())
        self.assertNotIn('정렬 완료', self.window.arm_feedback.text())

    def test_summary_keeps_stale_image_warning_and_controls_visible_in_every_tab(self):
        self.fixed_clock(10.)
        self.assertEqual(self.window.target_card.value.text(), '—')
        self.assertIn('—', self.window.tabs.tabText(0))
        image = Image(width=2, height=1, encoding='rgb8', step=6, data=bytes(6))
        image.header.stamp.sec = 9
        self.node.receive_image(image)
        self.window.refresh()
        self.assertIn('영상 수신 중', self.window.camera_summary.text())
        self.assertIn('1.00s 전 노출', self.window.camera_summary.text())
        self.node.received['image'] -= int(3e9)
        self.window.resize(1050, 740)
        for index in range(self.window.tabs.count()):
            self.window.tabs.setCurrentIndex(index)
            self.window.refresh()
            self.app.processEvents()
            self.assertIn('과거 영상', self.window.camera_summary.text())
            for button in (self.window.stop_button, self.window.resume_button, self.window.record_button):
                self.assertTrue(button.isVisible())
                position = button.mapTo(self.window.centralWidget(), QtCore.QPoint(0, 0))
                self.assertTrue(self.window.centralWidget().rect().contains(QtCore.QRect(position, button.size())))
        self.assertFalse(self.window.target_input.accessibleName() == '')

    def test_camera_exposure_detection_geometry_and_replayed_fusion_are_distinct(self):
        self.fixed_clock(16.6)
        track = dict(id=1, position=[28., -41., 10.], velocity=[10., 0., 0.], observed=True,
                     last_seen=16.505, nearest_distance=50.)
        self.packet('tracks', stamp=16.505, tracks=[track])
        self.packet('pointing', stamp=16.59, mode='tracking', target_id=1,
                    turret_state=dict(control_state_source='lidar'), arm_state=dict(control_state_source='camera_lidar'))
        self.packet('camera', stamp=16.5, ready=True, fusion_reason='ready', inference_ms=200.)
        self.packet('detections', stamp=16.335, detections=[])
        self.packet('fused', stamp=16.335, tracks=[], fusion_reason='ready')
        image = Image(width=2, height=1, encoding='rgb8', step=6, data=bytes(6))
        image.header.stamp.sec, image.header.stamp.nanosec = 16, 335000000
        self.node.receive_image(image)
        self.window.refresh()
        row = self.node.snapshot()
        self.assertAlmostEqual(row['image_exposure_age_s'], .265)
        self.assertLess(row['image_rx_age_s'], .1)
        self.assertEqual(row['detection_count'], 0)
        self.assertEqual(row['fusion_accepted_count'], 0)
        self.assertIn('NN 검출 0개', self.window.camera_status.text())
        self.assertIn('기하 조건 ready', self.window.camera_status.text())
        self.assertIn('포탑 lidar / 팔 camera_lidar', self.window.target_label.text())
        fused = dict(track, position=[27., -42., 10.], fusion_update=dict(accepted=True))
        self.packet('fused', stamp=16.335, tracks=[fused], fusion_reason='ready')
        self.window.refresh()
        self.assertAlmostEqual(self.node.snapshot()['lidar_minus_fused_stamp_s'], .17)
        self.assertIn('Δt 0.170s', self.window.target_label.text())
        self.assertIn('직접 비교 불가', self.window.target_label.text())
        # A new LiDAR replay updates state, without pretending to expose another image.
        fused.update(current_state_stamp=16.505, source_exposure_stamp=16.335,
                     current_state=dict(position=[29., -40., 10.]), temporal_fusion=dict(valid=True))
        self.packet('fused', stamp=16.335, current_state_stamp=16.55, tracks=[fused], fusion_reason='ready',
                    publication_kind='lidar_replay')
        row = self.node.snapshot()
        self.assertAlmostEqual(row['fusion_stamp'], 16.505)  # Per-track stamp takes precedence.
        self.assertAlmostEqual(row['fusion_exposure_stamp'], 16.335)
        self.assertAlmostEqual(row['lidar_minus_fused_stamp_s'], 0.)
        self.assertEqual(row['detection_count'], 0)
        self.window.refresh()
        self.assertIn('현재 융합 XYZ (29.00, -40.00, 10.00)', self.window.target_label.text())
        fused['temporal_fusion']['valid'] = False
        self.packet('fused', stamp=16.335, tracks=[fused], fusion_reason='ready')
        self.window.refresh()
        self.assertIn('노출 시각 융합 XYZ (27.00, -42.00, 10.00)', self.window.target_label.text())
        self.assertAlmostEqual(self.node.snapshot()['fusion_stamp'], 16.335)

    def test_camera_failure_withdraws_displayed_fusion_and_requires_new_evidence(self):
        self.fixed_clock(10.)
        track = dict(id=1, position=[1., 2., 3.], velocity=[1., 0., 0.], observed=True,
                     last_seen=10., class_name='fixed-wing', confidence=.99,
                     fusion_update=dict(accepted=True), subtype_verified=False)
        self.packet('tracks', stamp=10., tracks=[track])
        self.packet('pointing', stamp=10., mode='tracking', target_id=1)
        self.packet('camera', stamp=10., ready=True, inference_ready=True, fusion_ready=True)
        self.packet('fused', stamp=10., tracks=[track], fusion_reason='ready')
        self.window.refresh()
        self.assertEqual(self.window.table.item(0, 7).text(), 'fixed-wing (기종 미검증)')
        self.assertIn('융합 XYZ', self.window.target_label.text())
        self.packet('camera', stamp=10., ready=False, inference_ready=False, fusion_ready=False,
                    health_reason='inference_failed')
        self.window.refresh()
        self.assertNotIn('융합 XYZ', self.window.target_label.text())
        self.assertEqual(self.window.table.item(0, 7).text(), '—')
        self.assertFalse(self.node.snapshot()['yolo_ready'])
        self.assertIn('inference_failed', self.window.camera_status.text())
        # Late packets cannot outlive a failure just because its heartbeat ages out.
        self.packet('fused', stamp=10., tracks=[track], fusion_reason='ready')
        self.assertFalse(self.node.fusion_packet())
        self.node.received['camera'] -= int(3e9)
        self.packet('fused', stamp=10., tracks=[track], fusion_reason='ready')
        self.window.refresh()
        self.assertNotIn('융합 XYZ', self.window.target_label.text())
        self.assertFalse(self.node.fusion_packet())
        # Recovering the model alone does not restore an old camera correction.
        self.packet('camera', stamp=10., ready=True, inference_ready=True, fusion_ready=True)
        self.assertFalse(self.node.fusion_packet())
        self.packet('fused', stamp=10., tracks=[track], fusion_reason='ready')
        self.assertEqual(self.node.fusion_packet()['tracks'][0]['class_name'], 'fixed-wing')
        self.assertEqual(self.node.fusion_packet()['tracks'][0]['confidence'], .99)
        self.node.received['camera'] -= int(3e9)
        self.assertFalse(self.node.fusion_packet())

    def test_parser_errors_are_visible_and_recorded_without_replacing_good_data(self):
        self.node.start_recording(self.directory.name)
        directory = self.node.recorder.directory
        self.packet('tracks', stamp=1., tracks=[])
        self.node.receive_packet('tracks', String(data='{"stamp": 2, "tracks": [{}]}'))
        self.node.receive_packet('detections', String(data='{"stamp": NaN, "detections": []}'))
        self.window.refresh()
        self.assertEqual(self.node.snapshot()['parse_error_count'], 2)
        self.assertEqual(self.node.packets['tracks']['stamp'], 1.)
        self.assertIn('입력 파싱 오류 2건', self.window.command_label.text())
        self.assertIn('detections:', self.window.command_label.text())
        self.node.stop_recording()
        events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['action'] == 'input_parse_error' for row in events), 2)

    def test_nearest_policy_is_visible_and_malformed_selection_is_rejected(self):
        self.fixed_clock(10.1)
        self.packet('pointing', stamp=10., mode='tracking', target_id=2, requested_target_id=0,
                    selection=dict(policy='nearest_observed_3d', source_stamp=10.,
                                   distance_m=42.3, reason='nearest_handover'))
        self.window.refresh()
        self.assertIn('최근접 자동', self.window.target_card.detail.text())
        self.assertIn('시스템 기준 중심 거리 42.30 m', self.window.target_label.text())
        self.assertEqual(self.window.distance_card.value.text(), '— m')  # Center is not surface distance.
        self.packet('pointing', stamp=10.1, selection=[])
        self.window.refresh()
        self.assertEqual(self.node.packets['pointing']['stamp'], 10.)
        self.assertEqual(self.node.parse_errors['pointing'], 1)

    def test_alignment_and_camera_accounting_are_visible_without_claiming_detection_success(self):
        self.fixed_clock(10.)
        self.packet('pointing', stamp=10., mode='tracking', target_id=1, selection_age_s=.5,
                    arm=dict(measured_angle_error_deg=70., aligned=False, joint_slew_lower_bound_s=.8),
                    turret=dict(measured_angle_error_deg=1., aligned=True, joint_slew_lower_bound_s=.1))
        pipeline = dict(schema_version=1, counters=dict(image_received=100, inference_submitted=72,
            result_published=70, input_rejected=20, queue_dropped=8, inference_failed=1, result_discarded=1))
        self.packet('camera', stamp=10., ready=True, pipeline=pipeline)
        self.window.refresh()
        self.assertIn('선회 중', self.window.mode_label.text())
        self.assertIn('시간 하한 0.80s', self.window.arm_feedback.toolTip())
        self.assertIn('노드 수신 100 / 추론 제출 72 / 결과 발행 70', self.window.camera_status.text())
        self.assertIn('입력 거부 20 / 대기 제외 8', self.window.camera_status.text())
        self.packet('pointing', stamp=10., mode='tracking', target_id=1,
                    arm=dict(measured_angle_error_deg=.5, aligned=True),
                    turret=dict(measured_angle_error_deg=1., aligned=True))
        self.window.refresh()
        self.assertIn('정렬 유지', self.window.mode_label.text())
        self.packet('camera', stamp=11., ready=True, pipeline=dict(counters={'image_received': 'bad'}))
        self.assertEqual(self.node.packets['camera']['stamp'], 10.)
        self.assertEqual(self.node.parse_errors['camera'], 1)

    def test_malformed_control_mappings_and_string_vectors_keep_last_good_state(self):
        self.fixed_clock(10.)
        self.packet('pointing', stamp=10., mode='patrol', target_id=None)
        for field in ('arm', 'turret', 'prediction', 'turret_prediction', 'turret_state', 'arm_state', 'selection'):
            with self.subTest(field=field):
                self.packet('pointing', stamp=10.1, mode='tracking', **{field: []})
                self.window.refresh()
                self.assertEqual(self.node.packets['pointing']['mode'], 'patrol')
        self.packet('tracks', stamp=10., tracks=[])
        self.packet('tracks', stamp=10., tracks=[dict(id=1, position=['1', 2., 3.], velocity=[0., 0., 0.])])
        self.packet('pointing', stamp=10.1, mode='tracking', target_id=[])
        self.packet('fused', stamp=10., tracks=[])
        self.packet('fused', stamp=10.1, tracks=[dict(id=1, position=[1., 2., 3.], velocity=[0., 0., 0.],
                                                   class_name=[])])
        self.window.refresh()
        self.assertEqual(self.node.packets['tracks']['tracks'], [])
        self.assertEqual(self.node.packets['fused']['tracks'], [])
        self.assertEqual(self.node.parse_errors['pointing'], 8)
        self.assertEqual(self.node.parse_errors['tracks'], 1)
        self.assertEqual(self.node.parse_errors['fused'], 1)

    def test_clock_reset_clears_previous_epoch_selection_fusion_and_image_and_records_event(self):
        self.fixed_clock(100.)
        track = dict(id=1, position=[40., 0., 10.], velocity=[0., 10., 0.], observed=True,
                     last_seen=100., nearest_distance=40.)
        self.packet('tracks', stamp=100., tracks=[track])
        self.packet('pointing', stamp=100., mode='tracking', target_id=1)
        self.packet('fused', stamp=100., tracks=[dict(track, fusion_update=dict(accepted=True))])
        self.node.receive_image(Image(width=2, height=1, encoding='rgb8', step=6, data=bytes(6)))
        self.window.refresh()
        self.assertIsNotNone(self.window.pixmap)
        self.assertEqual(self.node.snapshot()['target_id'], 1)
        self.assertTrue(self.node.start_recording(self.directory.name))
        directory = self.node.recorder.directory
        self.fixed_clock(2.)
        row = self.node.snapshot()
        self.window.refresh()
        self.assertIsNone(row['target_id'])
        self.assertIsNone(row['nearest_surface_m'])
        self.assertEqual(row['track_count'], 0)
        self.assertEqual(self.node.history, {})
        self.assertEqual(self.node.packets, {})
        self.assertIsNone(self.node.background_ready)
        self.assertIsNone(self.window.pixmap)
        self.assertIn('미확인', self.window.mode_label.text())
        self.assertEqual(self.window.target_card.value.text(), '—')
        self.packet('pointing', stamp=2., mode='patrol', target_id=None)
        self.window.refresh()
        self.assertIn('360° 순찰', self.window.mode_label.text())
        self.node.stop_recording()
        events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        reset = [event for event in events if event['action'] == 'clock_reset']
        self.assertEqual(len(reset), 1)
        self.assertEqual((reset[0]['previous_stamp'], reset[0]['current_stamp']), (100., 2.))

    def test_manifest_records_only_known_context_and_csv_contains_age_contract(self):
        self.fixed_clock(10.)
        self.node.start_recording(self.directory.name)
        directory = self.node.recorder.directory
        manifest = json.loads((directory/'manifest.json').read_text())
        self.assertEqual(manifest['format_version'], 3)
        self.assertEqual(manifest['context']['raw_bag_uri'], '/known/recordings/test_trial')
        self.assertFalse(manifest['context']['raw_bag_uri_verified'])
        self.assertIsNone(manifest['context']['reported_model'])
        self.packet('camera', stamp=10., model='known-model-source', calibrated=True,
                    requested_provider='CPUExecutionProvider', active_providers=['CPUExecutionProvider'])
        self.packet('fused', stamp=10., tracks=[], calibration_valid=True, calibration_source='simulation_intrinsics')
        self.window.last_csv_write = -float('inf')
        self.window.refresh()
        self.node.stop_recording()
        manifest = json.loads((directory/'manifest.json').read_text())
        self.assertEqual(manifest['context']['reported_model'], 'known-model-source')
        self.assertEqual(manifest['context']['reported_calibration_source'], 'simulation_intrinsics')
        self.assertEqual(len(manifest['context']['ui_source']['sha256']), 64)
        self.assertIn('detections', manifest['telemetry_topics'])
        with (directory/'selected_target.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertIn('last_observation_age_s', rows[0])
        self.assertIn('image_exposure_age_s', rows[0])

    def test_recorder_creates_distinct_sessions_and_exposes_disk_error(self):
        recorder = SessionRecorder()
        self.assertTrue(recorder.start(self.directory.name, True))
        first = recorder.directory
        recorder.write('events.jsonl', {'action': 'sentinel'})
        recorder.close()
        self.assertTrue(recorder.start(self.directory.name, True))
        self.assertNotEqual(first, recorder.directory)
        recorder.close()
        self.assertEqual(json.loads((first/'events.jsonl').read_text()), {'action': 'sentinel'})
        blocked = Path(self.directory.name)/'existing_file'
        blocked.write_text('preserve')
        self.assertFalse(recorder.start(blocked, True))
        self.assertFalse(recorder.active)
        self.assertTrue(recorder.error)
        self.assertEqual(blocked.read_text(), 'preserve')

    def test_main_sigint_closes_qt_and_active_recorder_without_callback_traceback(self):
        # 실제 별도 프로세스에서 Qt 타이머가 기록 중일 때 Ctrl-C와 같은 신호를 보낸다.
        root = Path(self.directory.name)/'signal_exit'
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen', ROS_DOMAIN_ID='190', ROS_LOCALHOST_ONLY='1')
        command = [sys.executable, '-m', 'jol_perception.experiment_ui', '--ros-args',
                   '-p', 'auto_record:=true', '-p', f'record_dir:={root}']
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        try:
            deadline = time.monotonic()+8
            csv_files = []
            while time.monotonic() < deadline and process.poll() is None:
                csv_files = list(root.glob('ui_*/selected_target.csv'))
                if csv_files and len(csv_files[0].read_text().splitlines()) >= 3:
                    break
                time.sleep(.02)
            self.assertTrue(csv_files and len(csv_files[0].read_text().splitlines()) >= 3)
            process.send_signal(signal.SIGINT)
            output, _ = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 0, output)
            for failure in ('Traceback', 'KeyboardInterrupt', 'RCLError'):
                self.assertNotIn(failure, output)
            events = [json.loads(line) for line in (csv_files[0].parent/'events.jsonl').read_text().splitlines()]
            stop = [row for row in events if row['action'] == 'recording' and row['phase'] == 'stop']
            self.assertEqual(len(stop), 1)
            self.assertEqual(stop[0]['method'], 'window_close')
            self.assertIs(stop[0]['success'], True)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)

    def capture(self, name):
        # 명시적으로 요청한 검사에서만 실제 위젯 이미지를 저장한다.
        directory = os.environ.get('JOL_UI_CAPTURE_DIR')
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            self.app.processEvents()
            self.assertTrue(self.window.grab().save(str(path/(name+'.png'))))

    def observer_status(self, **values):
        packet = dict(stamp=10., state='waiting', reason='', ready=False,
                      execution_allowed=False, dynamic_collision_checked=False)
        packet.update(values)
        self.packet('observer', **packet)

    def observer_plan(self, **values):
        packet = dict(stamp=10., valid=True, frame_id='world', planned_at=10.,
                      preview_expires_at=10.5, home=[0., 0., 0.], goal=[6.85, 0., 3.],
                      target_id=1, duration_s=16.42, execution_allowed=False, dynamic_collision_checked=False)
        packet.update(values)
        self.packet('observer_plan', **packet)

    def test_joint_display_requires_fresh_source_stamp_and_receipt_across_reset(self):
        self.fixed_clock(10.)
        joint = JointState(name=['yaw_joint', 'pitch_joint'], position=[.4, -.2])
        for stamp, valid in ((9.5, True), (1., False), (0., False), (10.1, False), (10.04, True)):
            with self.subTest(stamp=stamp):
                joint.header.stamp.sec = int(stamp)
                joint.header.stamp.nanosec = round((stamp-int(stamp))*1e9)
                self.node.receive_joints(joint)
                self.window.refresh()
                row = self.node.snapshot()
                self.assertTrue(self.node.fresh('joints'))
                self.assertEqual(row['yaw_deg'] is not None, valid)
                self.assertEqual(row['elevation_deg'] is not None, valid)
                self.assertEqual(self.window.plot.yaw is not None, valid)
                if not valid:
                    self.assertIn('실제 yaw —° / 고각 —°', self.window.status_detail.text())
        self.node.received['joints'] -= int(3e9)
        self.assertIsNone(self.node.snapshot()['yaw_deg'])
        self.fixed_clock(1.)
        self.window.refresh()
        self.assertIsNone(self.node.joint_stamp)
        self.assertIsNone(self.window.plot.yaw)
        self.node.receive_joints(joint)  # 초기화 이전 시각의 늦은 메시지도 미래값이므로 숨긴다.
        self.assertIsNone(self.node.snapshot()['yaw_deg'])
        joint.header.stamp.sec, joint.header.stamp.nanosec = 1, 0
        self.node.receive_joints(joint)
        self.assertIsNotNone(self.node.snapshot()['yaw_deg'])
        # 0초 자체는 유효한 시뮬레이션 원점이며, 현재 시각과의 차이로 판정한다.
        self.fixed_clock(0.)
        joint.header.stamp.sec = 0
        self.node.receive_joints(joint)
        self.assertIsNotNone(self.node.snapshot()['yaw_deg'])

    def test_valid_preview_requires_explicit_false_execution_contract(self):
        self.fixed_clock(10.)
        self.observer_status(state='preview_ready', ready=True)
        for flag in (True, 0, None, 'false', 'missing'):
            with self.subTest(flag=flag):
                self.observer_plan(samples=[dict(position=[0., 0., 0.])])
                plan = dict(self.node.packets['observer_plan'], execution_allowed=flag)
                if flag == 'missing':
                    plan.pop('execution_allowed')
                self.packet('observer_plan', **plan)
                self.window.refresh()
                self.assertNotIn('observer_plan', self.node.packets)
                self.assertEqual(self.window.plot.observation_path, [])
                self.assertIn('실행 불허', self.window.command_label.text())
        self.assertEqual(self.node.parse_errors['observer_plan'], 5)
        self.observer_plan(execution_allowed=False)
        self.assertIs(self.node.packets['observer_plan']['execution_allowed'], False)

    def test_observer_preview_expiry_and_invalid_input_never_enable_flight(self):
        self.fixed_clock(10.)
        self.window.resize(960, 700)
        self.window.refresh()
        self.assertIn('수신 대기', self.window.observer_summary.text())
        self.assertFalse(self.window.preview_button.isEnabled())
        self.capture('empty_960x700')
        self.observer_status(reason_code='input_missing', input_freshness=dict(
            status=dict(ros_age_s=0., wall_age_s=0.), tracks=dict(ros_age_s=0., wall_age_s=0.),
            own=dict(ros_age_s=None, wall_age_s=None)))
        self.window.refresh()
        self.assertIn('자기 기체 위치', self.window.observer_summary.text())
        self.assertIn('입력 없음: 자기 기체 위치·속도', self.window.observer_summary.text())
        self.assertIn('동적 충돌: 미검증', self.window.observer_detail.text())
        self.capture('missing_own_pose_960x700')
        track = dict(id=1, position=[18., 0., 3.], velocity=[0., 0., 0.], observed=True, last_seen=10.)
        self.packet('tracks', stamp=10., tracks=[track])
        self.packet('pointing', stamp=10., mode='tracking', target_id=1, camera_mount='arm',
                    prediction=dict(horizon_s=.5, position=[18., 0., 3.]))
        self.observer_plan()
        self.observer_status(state='preview_ready', ready=True)
        self.window.refresh()
        self.assertIn('미리보기 · 실행 불가', self.window.observer_summary.text())
        self.assertIn('6.85', self.window.observer_detail.text())
        self.capture('preview_960x700')
        self.fixed_clock(10.6)
        self.window.refresh()
        self.assertIn('만료', self.window.observer_summary.text())
        self.assertNotIn('관찰점 XYZ', self.window.observer_detail.text())
        self.assertIn('최근 계획 기록 · 현재 유효하지 않음', self.window.observer_detail.text())
        self.observer_status(stamp=10.6, state='invalidated', reason='missing pose', reason_code='own_pose_invalid')
        self.packet('observer_plan', stamp=10.6, valid=False, reason='missing pose')
        self.window.refresh()
        self.assertIn('자기 기체 위치', self.window.observer_summary.text())
        self.assertIn('과거 관찰점 6.85', self.window.observer_detail.text())
        self.capture('expired_with_history_960x700')
        self.observer_status(stamp=10.6, ready=True, execution_allowed=True)
        self.window.refresh()
        self.assertNotIn('observer', self.node.packets)
        self.assertIn('계약 오류', self.window.command_label.text())
        self.assertEqual(self.window.command_label.property('tone'), 'error')

    def test_preview_keyboard_calls_only_prepare_and_service_ack_is_not_a_plan(self):
        self.fixed_clock(10.)
        calls = []
        def callback(request, response):
            calls.append('prepare')
            response.success, response.message = True, 'preview accepted'
            return response
        service = self.node.create_service(Trigger, '/jol/observer/prepare', callback)
        self.wait_for(lambda: self.node.control_clients['prepare_observation'].service_is_ready())
        self.window.refresh()
        self.window.preview_button.setFocus()
        QtTest.QTest.keyClick(self.window.preview_button, QtCore.Qt.Key_Space)
        self.assertIsNotNone(self.node.pending)
        self.assertFalse(self.window.preview_button.isEnabled())
        self.wait_for(lambda: self.node.pending is None)
        self.window.refresh()
        self.assertEqual(calls, ['prepare'])
        self.assertIn('계획 상태 확인', self.window.command_label.text())
        self.assertIn('수신 대기', self.window.observer_summary.text())
        self.assertEqual(set(self.node.control_clients), {'stop_tracking', 'resume_auto', 'prepare_observation'})
        self.assertFalse(any('비행 시작' in b.text() for b in self.window.findChildren(QtWidgets.QPushButton)))
        self.window.preview_button.setFocus()
        QtTest.QTest.keyClick(self.window.preview_button, QtCore.Qt.Key_Tab)
        self.assertIsNot(self.app.focusWidget(), self.window.preview_button)
        self.node.destroy_service(service)

    def test_fresh_receipt_cannot_make_an_old_state_current(self):
        self.fixed_clock(10.)
        track = dict(id=1, position=[1., 2., 3.], velocity=[0., 0., 0.], observed=True, last_seen=5.)
        self.packet('tracks', stamp=5., tracks=[track])
        self.packet('pointing', stamp=5., mode='tracking', target_id=1)
        self.packet('camera', stamp=5., ready=True)
        self.observer_plan()
        self.observer_status(stamp=5., state='preview_ready', ready=True)
        self.window.refresh()
        self.assertTrue(self.node.fresh('tracks'))
        self.assertFalse(self.node.state_fresh('tracks'))
        self.assertEqual(self.window.target_card.value.text(), '—')
        self.assertIn('원본 시각 확인', self.window.table.item(0, 1).text())
        self.assertNotEqual(self.window.health_labels['tracks'].property('tone'), 'good')
        self.assertTrue(self.window.plot.stale)
        self.assertIsNone(self.node.snapshot()['yolo_ready'])
        self.assertIn('원본 시각 확인', self.window.observer_summary.text())
        self.observer_status(state='preview_ready', ready=True)
        self.node.received['observer'] -= int(3e9)
        self.window.refresh()
        self.assertNotIn('미리보기 · 실행 불가', self.window.observer_summary.text())
        self.fixed_clock(1.)
        self.window.refresh()
        self.assertEqual(self.node.packets, {})
        self.assertIsNone(self.node.last_observer_plan)

    def test_compact_layout_keeps_controls_and_errors_readable(self):
        self.fixed_clock(10.)
        self.window.resize(960, 700)
        self.observer_status(reason_code='input_missing', input_freshness=dict(status={}, tracks={}, own=None))
        self.node.request_command('stop_tracking')
        self.window.refresh()
        self.assertEqual(self.window.command_label.property('tone'), 'error')
        self.assertIn('실패', self.window.command_label.text())
        for index in range(self.window.tabs.count()):
            self.window.tabs.setCurrentIndex(index)
            self.app.processEvents()
            self.assertLessEqual(self.window.width(), 960)
            self.assertLessEqual(self.window.height(), 700)
            for widget in (self.window.preview_button, self.window.observer_summary,
                           self.window.stop_button, self.window.record_button, self.window.command_label):
                position = widget.mapTo(self.window.centralWidget(), QtCore.QPoint())
                self.assertTrue(self.window.centralWidget().rect().contains(QtCore.QRect(position, widget.size())))
        self.window.tabs.setCurrentIndex(next(i for i in range(self.window.tabs.count()) if self.window.tabs.tabText(i) == '관찰 경로'))
        self.capture('compact_observer_tab_960x700')

    def test_empty_fusion_waiting_packet_does_not_require_a_world_frame(self):
        # 실제 Gazebo 시작·리셋 중 카메라가 원본 트랙을 아직 못 받으면 이 패킷을 발행한다.
        self.fixed_clock(10.)
        self.packet('fused', stamp=10., frame_id='', tracks=[], fusion_reason='waiting_for_tracks')
        self.window.refresh()
        self.assertFalse(self.node.parse_errors)
        self.assertIn('waiting_for_tracks', self.window.camera_status.text())
        track = dict(id=1, position=[1., 2., 3.], velocity=[0., 0., 0.])
        self.packet('fused', stamp=10.1, frame_id='camera_optical', tracks=[track])
        self.assertEqual(self.node.parse_errors['fused'], 1)
        self.assertEqual(self.node.packets['fused']['tracks'], [])

    def test_world_plot_shows_only_valid_finite_observation_path(self):
        from jol_perception.observation_path import build_plan
        self.fixed_clock(10.)
        plan = build_plan([-30., -25., 0.], [25., 25., 3.])
        self.observer_status(state='preview_ready', ready=True)
        self.observer_plan(**plan)
        self.window.refresh()
        self.assertEqual(self.window.plot.observation_goal, plan['goal'])
        self.assertEqual(len(self.window.plot.observation_path), len(plan['samples']))
        self.assertIn('수직 상승·고도·충돌 안전', self.window.plot.toolTip())
        self.window.resize(960, 700)
        self.capture('observation_path_final_960x700')
        self.fixed_clock(10.6)
        self.window.refresh()
        self.assertEqual(self.window.plot.observation_path, [])
        self.assertIsNone(self.window.plot.observation_goal)
        self.assertIn('현재 유효하지 않음', self.window.observer_detail.text())
        for mutation in (dict(frame_id='camera_optical'), dict(samples=[dict(position=[1., 2., '3'])]),
                         dict(samples=[dict(position=[1., 2., float('inf')])])):
            self.observer_plan(**dict(plan, **mutation))
            self.window.refresh()
            self.assertEqual(self.window.plot.observation_path, [])
        self.assertEqual(self.node.parse_errors['observer_plan'], 3)
        self.observer_plan(**plan, stamp=10.6, planned_at=10.6, preview_expires_at=11.1)
        self.observer_status(stamp=10.6, state='preview_ready', ready=True)
        self.window.refresh()
        self.assertTrue(self.window.plot.observation_path)
        self.node.received['observer'] -= int(3e9)
        self.window.refresh()
        self.assertEqual(self.window.plot.observation_path, [])
        self.observer_status(stamp=10.6, state='preview_ready', ready=True)
        self.window.refresh()
        self.assertTrue(self.window.plot.observation_path)
        self.fixed_clock(1.)
        self.window.refresh()
        self.assertEqual(self.window.plot.observation_path, [])


class StaticFlightUITests(unittest.TestCase):
    """새 정적 실험 모드의 실제 Qt 조작과 ROS 서비스 계약을 검사한다."""
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.context = Context()
        rclpy.init(context=self.context, domain_id=189)
        self.node = ExperimentNode(context=self.context, parameter_overrides=[
            Parameter('static_flight_mode', value=True), Parameter('record_dir', value=self.directory.name),
            Parameter('service_timeout', value=.15)])
        self.window = Dashboard(self.node)
        self.window.show()
        self.node.clocks = lambda: dict(wall_time='test', monotonic_ns=time.monotonic_ns(), sim_stamp=10.)
        self.services = []

    def tearDown(self):
        self.window.close()
        for service in self.services:
            self.node.destroy_service(service)
        self.node.destroy_node()
        rclpy.shutdown(context=self.context)
        self.directory.cleanup()

    wait_for = ExperimentUITests.wait_for
    click = ExperimentUITests.click
    packet = ExperimentUITests.packet
    capture = ExperimentUITests.capture

    def status(self, **values):
        packet = dict(stamp=10., phase='coordinate_ready', reason='', coordinate_valid=True,
                      target_world=[39., 2., 7.], target_ned=[8., 39., -7.], locked=False,
                      plan_ready=False, controller_ready=True, can_launch=False,
                      camera_centered=True, camera_alignment_error_deg=0., gimbal_tracking=True,
                      takeoff_height=3., takeoff_height_m=0., position_world=[-5.,-6.,.1],
                      velocity_world=[0.,0.,0.], own_stamp=9.99, odometry_age_s=.01)
        packet.update(values)
        self.packet('static_status', **packet)
        self.window.refresh()

    def connect_services(self, calls):
        for action in ('lock', 'unlock', 'preview', 'launch', 'abort', 'reset'):
            def callback(request, response, action=action):
                calls.append(action)
                response.success, response.message = True, action+' accepted'
                return response
            self.services.append(self.node.create_service(Trigger, '/jol/static_flight/'+action, callback))
        self.wait_for(lambda: all(self.node.control_clients['static_'+action].service_is_ready()
                                 for action in ('lock', 'unlock', 'preview', 'launch', 'abort', 'reset')))

    def test_detection_start_button_requests_only_sensor_search(self):
        calls=[]
        def start(request,response):
            calls.append('sensor_search')
            response.success,response.message=True,'lidar_search'
            return response
        self.services.append(self.node.create_service(Trigger,'/jol/static_target/start_acquisition',start))
        self.wait_for(lambda:self.node.control_clients['static_start_acquisition'].service_is_ready())
        self.status(phase='detecting',coordinate_valid=False,camera_centered=False,target_world=None,target_ned=None)
        self.assertFalse(self.window.resume_button.isEnabled())
        self.packet('static_sensor',stamp=10.,calibration_valid=True,acquisition=dict(stage='waiting_start',started=False))
        self.window.refresh()
        self.assertTrue(self.window.resume_button.isEnabled())
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.click('resumeAutoButton')
        self.wait_for(lambda:self.node.pending is None)
        self.assertEqual(calls,['sensor_search'])
        self.assertFalse(any(event['action']=='static_launch' for event in self.node.events))
        self.packet('static_sensor',stamp=10.,calibration_valid=True,acquisition=dict(stage='slewing',started=True))
        self.status(phase='detecting',coordinate_valid=False,camera_centered=False,target_world=None,target_ned=None)
        self.assertFalse(self.window.resume_button.isEnabled())

    def test_qt_lock_preview_launch_duplicate_abort_and_reset_flow(self):
        calls = []
        self.connect_services(calls)
        self.status()
        self.assertTrue(self.window.static_buttons['static_lock'].isEnabled())
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.click('staticLockButton')
        self.wait_for(lambda: self.node.pending is None)
        # 서비스 수락 자체로 락이나 비행 준비를 만들어 내지 않는다.
        self.assertFalse(self.node.packets['static_status']['locked'])
        self.status(phase='locked', locked=True)
        self.click('observationPreviewButton')
        self.wait_for(lambda: self.node.pending is None)
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.packet('static_plan', stamp=10., valid=True, frame_id='world', home=[-5., -6., .1],
                    goal=[36., 2., 7.], samples=[dict(position=[-5., -6., .1]), dict(position=[36., 2., 7.])], duration_s=30.)
        self.status(phase='preview_ready', locked=True, plan_ready=True, can_launch=True)
        self.assertTrue(self.window.static_buttons['static_launch'].isEnabled())
        self.assertEqual(len(self.window.plot.observation_path), 2)
        self.click('staticLaunchButton')
        self.node.request_command('static_launch')  # 동시에 직접 요청해도 중복 전송되지 않는다.
        self.wait_for(lambda: self.node.pending is None)
        self.window.refresh()
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.node.request_command('static_launch')  # 수락과 상태 전이 사이에도 중복 발사를 막는다.
        self.assertEqual(calls.count('launch'), 1)
        self.status(phase='takeoff', locked=True, plan_ready=True, position_world=[-5., -6., 1.1], takeoff_height_m=1.)
        self.assertTrue(self.window.static_buttons['static_abort'].isEnabled())
        self.assertFalse(self.window.static_buttons['static_reset'].isEnabled())
        self.status(phase='spline_tracking', locked=True, plan_ready=True,
                    position_world=[0., -5., 3.1], takeoff_height_m=3., tracking_error_m=.12)
        self.assertEqual(len(self.window.plot.actual_path), 2)
        self.assertIn('0.120', self.window.speed_card.value.text())
        self.click('staticAbortButton')
        self.wait_for(lambda: self.node.pending is None)
        self.status(phase='aborted', locked=True)
        self.click('staticUnlockButton')
        self.wait_for(lambda: self.node.pending is None)
        self.status()
        self.click('staticResetButton')
        self.wait_for(lambda: self.node.pending is None)
        self.assertEqual(calls, ['lock', 'preview', 'launch', 'abort', 'unlock', 'reset'])
        self.assertFalse(self.node.static_actual_path)

    def test_camera_reset_waits_for_accepted_mission_reset(self):
        calls=[]
        accepted=[False]
        def reset_mission(request,response):
            calls.append('mission')
            response.success,response.message=accepted[0],'ground_reset'
            return response
        def reset_camera(request,response):
            calls.append('camera')
            response.success,response.message=True,'acquisition_reset_pending'
            return response
        self.services.extend([
            self.node.create_service(Trigger,'/jol/static_flight/reset',reset_mission),
            self.node.create_service(Trigger,'/jol/static_target/reset_acquisition',reset_camera)])
        self.wait_for(lambda: all(self.node.control_clients[action].service_is_ready()
                                 for action in ('static_reset','static_reset_acquisition')))
        self.status(phase='aborted')
        self.click('staticResetButton')
        self.wait_for(lambda:self.node.pending is None)
        self.assertEqual(calls,['mission'])
        accepted[0]=True
        self.status(phase='aborted')
        self.click('staticResetButton')
        self.wait_for(lambda:self.node.pending is None)
        self.assertEqual(calls,['mission','mission','camera'])
        self.packet('static_sensor',stamp=10.,calibration_valid=True,acquisition=dict(stage='resetting'))
        self.status(phase='detecting',coordinate_valid=False,camera_centered=False,target_world=None,target_ned=None)
        self.assertFalse(self.window.resume_button.isEnabled())
        self.assertIn('초기 자세 복귀',self.window.observer_summary.text())

    def test_missing_stale_invalid_coordinates_never_enable_launch(self):
        calls = []
        self.connect_services(calls)
        self.window.refresh()
        self.assertFalse(any(button.isEnabled() for button in self.window.static_buttons.values()))
        for mutation in (dict(coordinate_valid=False, target_world=None, target_ned=None),
                         dict(controller_ready=False), dict(camera_centered=False), dict(camera_centered=None),
                         dict(locked=False), dict(plan_ready=False), dict(can_launch=False),
                         dict(stamp=5.)):
            with self.subTest(mutation=mutation):
                values = dict(phase='preview_ready', locked=True, plan_ready=True, can_launch=True)
                values.update(mutation)
                self.status(**values)
                self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
                self.node.request_command('static_launch')
                self.assertIsNone(self.node.pending)
        self.status(phase='preview_ready', locked=True, plan_ready=True, can_launch=True)
        self.node.received['static_status'] -= int(3e9)
        self.window.refresh()
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.status(phase='preview_ready', locked=True, plan_ready=True, can_launch='true')
        self.assertNotIn('static_status', self.node.packets)
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.assertEqual(calls, [])

    def test_reset_requires_fresh_stationary_ground_pose_and_confirmed_px4_disarm(self):
        calls = []
        self.connect_services(calls)
        feedback = dict(phase='aborted', locked=True, flight_backend='mavros_px4',
                        autopilot_connected=True, armed=False, landed_state=1)
        for mutation in (dict(position_world=None), dict(velocity_world=None),
                         dict(own_stamp=None), dict(own_stamp=9.7), dict(own_stamp=10.01),
                         dict(position_world=[-5.,-6.,.26]), dict(velocity_world=[.151,0.,0.]),
                         dict(velocity_world=[.11,.11,0.]), dict(armed=True), dict(armed=None),
                         dict(landed_state=2), dict(landed_state=None), dict(landed_state=True),
                         dict(autopilot_connected=False)):
            with self.subTest(mutation=mutation):
                self.status(**dict(feedback, **mutation))
                self.assertFalse(self.window.static_buttons['static_reset'].isEnabled())
                self.assertFalse(self.node.static_action_allowed('static_reset'))
                self.node.request_command('static_reset')
                self.assertIsNone(self.node.pending)
        self.assertEqual(calls, [])
        # Older status schemas may provide a measured age instead of own_stamp.
        self.status(**feedback)
        packet = dict(self.node.packets['static_status']); packet.pop('own_stamp')
        packet['odometry_age_s'] = .3
        self.packet('static_status', **packet); self.window.refresh()
        self.assertFalse(self.node.static_action_allowed('static_reset'))
        packet['odometry_age_s'] = .01
        self.packet('static_status', **packet); self.window.refresh()
        self.assertTrue(self.node.static_action_allowed('static_reset'))
        packet['stamp'] = 9.7  # Communication delay also ages an old-schema measurement.
        self.packet('static_status', **packet); self.window.refresh()
        self.assertFalse(self.node.static_action_allowed('static_reset'))
        self.status(**feedback)
        self.assertTrue(self.window.static_buttons['static_reset'].isEnabled())
        self.click('staticResetButton')
        self.wait_for(lambda: self.node.pending is None)
        self.assertEqual(calls, ['reset'])
        # Native recordings require the same pose/velocity evidence, without FCU fields.
        self.status(phase='aborted', flight_backend='native')
        self.assertTrue(self.node.static_action_allowed('static_reset'))

    def test_lock_requires_backend_sensor_uncertainty_clearance_readiness(self):
        calls = []
        self.connect_services(calls)
        reason = 'standoff_too_small_for_vehicle_and_sensor_uncertainty'
        self.status(can_lock=False, lock_block_reason=reason)
        self.assertFalse(self.window.static_buttons['static_lock'].isEnabled())
        self.assertFalse(self.node.static_action_allowed('static_lock'))
        self.assertIn('센서 좌표·안전 여유 확인 대기', self.window.observer_summary.text())
        self.assertIn(reason, self.node.static_launch_block_reason())
        self.node.request_command('static_lock')
        self.assertIsNone(self.node.pending)
        self.assertEqual(calls, [])
        self.status(can_lock=True, lock_block_reason='')
        self.assertTrue(self.window.static_buttons['static_lock'].isEnabled())
        self.assertTrue(self.node.static_action_allowed('static_lock'))
        self.click('staticLockButton')
        self.wait_for(lambda: self.node.pending is None)
        self.assertEqual(calls, ['lock'])
        # Old single-flight recordings do not contain these optional fields.
        self.status()
        self.assertTrue(self.window.static_buttons['static_lock'].isEnabled())
        for invalid in ('true', 1, None):
            with self.subTest(invalid=invalid):
                self.status(can_lock=invalid)
                self.assertNotIn('static_status', self.node.packets)
                self.assertFalse(self.window.static_buttons['static_lock'].isEnabled())
        self.status(can_lock=False, lock_block_reason=1)
        self.assertNotIn('static_status', self.node.packets)

    def test_locked_preview_failure_remains_visible_after_sensor_acquisition_ready(self):
        reason = 'preview_failed: path_outside_controller_envelope'
        self.packet('static_sensor', stamp=10., calibration_valid=True,
                    acquisition=dict(stage='ready', started=True))
        self.status(phase='locked', locked=True, plan_ready=False, can_launch=False,
                    reason=reason, launch_block_reason=reason)
        self.assertIn(reason, self.window.observer_summary.text())
        self.assertIn('경로 준비 필요', self.window.observer_summary.text())
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.assertFalse(self.node.static_action_allowed('static_launch'))

    def test_interactive_3d_shows_world_height_actual_flight_and_never_commands_flight(self):
        from matplotlib.backend_bases import MouseEvent
        plan = [[-5., -6., .1], [-5., -6., 3.1], [20., 6., 8.]]
        self.packet('static_plan', stamp=10., valid=True, frame_id='world', home=plan[0],
                    goal=plan[-1], samples=[dict(position=point) for point in plan], duration_s=60.)
        self.status(phase='preview_ready', locked=True, plan_ready=True, can_launch=True,
                    target_world=[21., 6., 8.], goal_world=plan[-1])
        view = self.window.flight_3d
        view.redraw(force=True)
        self.app.processEvents()
        self.assertEqual(self.window.world_views.currentIndex(), 0)
        self.assertEqual(view.axes.name, '3d')
        self.assertEqual([list(axis) for axis in view.plan_line.get_data_3d()],
                         [[-5., -5., 20.], [-6., -6., 6.], [.1, 3.1, 8.]])
        self.assertEqual([list(axis) for axis in view.takeoff_line.get_data_3d()],
                         [[-5., -5.], [-6., -6.], [.1, 3.1]])
        self.assertEqual([list(axis) for axis in view.target_marker.get_data_3d()], [[21.], [6.], [8.]])
        self.assertEqual(view.axes.get_zlabel(), 'World Z [m]')
        self.assertNotEqual(view.plan_line.get_color(), view.actual_line.get_color())
        actual = [[-5., -6., 1.2], [-2., -5., 3.3]]
        self.status(phase='takeoff', locked=True, plan_ready=True, position_world=actual[0])
        self.status(phase='spline_tracking', locked=True, plan_ready=True, position_world=actual[1])
        view.redraw(force=True)
        self.assertEqual([list(axis) for axis in view.actual_line.get_data_3d()],
                         [[-5., -2.], [-6., -5.], [1.2, 3.3]])
        # Exercise the displayed Qt canvas mouse rotation and wheel event.
        view.canvas.draw()
        before = (view.axes.elev, view.axes.azim)
        bounds = view.axes.bbox
        center = QtCore.QPoint(int(bounds.x0+bounds.width/2), view.canvas.height()-int(bounds.y0+bounds.height/2))
        QtTest.QTest.mousePress(view.canvas, QtCore.Qt.LeftButton, pos=center)
        self.app.sendEvent(view.canvas, QtGui.QMouseEvent(QtCore.QEvent.MouseMove,
            QtCore.QPointF(center+QtCore.QPoint(40, 20)), QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        QtTest.QTest.mouseRelease(view.canvas, QtCore.Qt.LeftButton, pos=center+QtCore.QPoint(40, 20))
        self.assertNotEqual((view.axes.elev, view.axes.azim), before)
        width = view.axes.get_xlim3d()[1]-view.axes.get_xlim3d()[0]
        event = MouseEvent('scroll_event', view.canvas, bounds.x0+bounds.width/2, bounds.y0+bounds.height/2, step=1)
        view.canvas.callbacks.process('scroll_event', event)
        self.assertLess(view.axes.get_xlim3d()[1]-view.axes.get_xlim3d()[0], width)
        rotated = (view.axes.elev, view.axes.azim)
        limits = view.axes.get_xlim3d()
        self.status(phase='spline_tracking', locked=True, plan_ready=True, position_world=[0., -4., 4.])
        view.redraw(force=True)
        self.assertEqual((view.axes.elev, view.axes.azim), rotated)
        self.assertEqual(view.axes.get_xlim3d(), limits)
        self.click('reset3DViewButton')
        self.assertEqual((view.axes.elev, view.axes.azim), (25., -60.))
        self.click('worldXYButton')
        self.assertEqual(self.window.world_views.currentIndex(), 1)
        self.click('world3DButton')
        self.assertEqual(self.window.world_views.currentIndex(), 0)
        self.assertIsNone(self.node.pending)
        self.assertFalse(any(event.get('phase') == 'request' for event in self.node.events))
        self.capture('static_3d_interactive_synthetic_test')
        self.status(plan_ready=False)
        view.redraw(force=True)
        self.assertEqual([list(axis) for axis in view.plan_line.get_data_3d()], [[], [], []])

    def test_3d_skips_unchanged_geometry_and_keeps_motion_view_resize_and_forced_redraw(self):
        from unittest.mock import patch
        view = self.window.flight_3d
        plan = dict(home=[0., 0., .1], goal=[40., 0., 8.],
                    samples=[dict(position=[0., 0., .1]), dict(position=[40., 0., 8.])])
        actual = [[0., 0., .1]]
        view.set_path(plan, actual, [40., 0., 8.])
        view.redraw(force=True)
        with patch.object(view.canvas, 'draw_idle') as draw:
            view._last_draw = -float('inf')
            view.set_path(plan, actual, [40., 0., 8.])
            draw.assert_not_called()
            view.set_path(plan, actual+[[0., 0., 1.]], [40., 0., 8.])
            self.assertEqual(draw.call_count, 1)
            view._last_draw = -float('inf')
            view.axes.view_init(elev=35., azim=-30.)
            view.redraw()
            self.assertEqual(draw.call_count, 2)
            view._last_draw = -float('inf')
            view.canvas.resize(view.canvas.width()+10, view.canvas.height()+10)
            draw.reset_mock()  # Qt/Matplotlib may schedule its own resize draw.
            view.redraw()
            draw.assert_called_once()
            view.redraw(force=True)
            self.assertEqual(draw.call_count, 2)

    def test_px4_arming_is_active_and_backend_feedback_source_is_explicit(self):
        calls = []
        self.connect_services(calls)
        self.status(phase='arming', locked=True, plan_ready=True, flight_backend='mavros_px4',
                    feedback_controller='MAVROS_OFFBOARD_PX4', own_pose_source='mavros_px4_ekf',
                    position_world=[-5., -6., .1])
        self.assertIn('PX4 OFFBOARD', self.window.mode_label.text())
        self.assertIn('PX4 · MAVROS', self.window.observer_policy.text())
        self.assertIn('MAVROS_OFFBOARD_PX4', self.window.observer_detail.text())
        self.assertIn('mavros_px4_ekf', self.window.observer_detail.text())
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.assertFalse(self.window.static_buttons['static_reset'].isEnabled())
        self.assertTrue(self.window.static_buttons['static_abort'].isEnabled())

        self.node.request_command('static_launch')
        self.assertEqual(calls, [])
        self.status(phase='autopilot_landing', locked=True, plan_ready=True, flight_backend='mavros_px4',
                    own_pose_source='mavros_px4_ekf')
        self.assertIn('PX4 자동 착륙', self.window.mode_label.text())
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.assertFalse(self.window.static_buttons['static_reset'].isEnabled())
        self.assertFalse(self.window.static_buttons['static_abort'].isEnabled())
        self.status(flight_backend='native', own_pose_source='ideal_gazebo_self_odometry')
        self.assertIn('native', self.window.observer_policy.text())
        self.assertIn('ideal_gazebo_self_odometry', self.window.flight_3d.summary.toolTip())

    def test_tailsitter_screen_discloses_reference_model_and_observed_transition(self):
        self.status(phase='spline_tracking',locked=True,plan_ready=True,flight_backend='mavros_px4',
                    aircraft_type='tailsitter20cm_reference',vtol_state=1,armed=True,
                    autopilot_connected=True,autopilot_mode='OFFBOARD')
        self.assertIn('20cm 테일시터(가정)',self.window.observer_policy.text())
        self.assertIn('전진 전환 중',self.window.observer_policy.text())
        self.assertNotIn('전진 비행',self.window.observer_policy.text())
        self.status(phase='spline_tracking',locked=True,plan_ready=True,flight_backend='mavros_px4',
                    aircraft_type='tailsitter20cm_reference',vtol_state=4,armed=True,
                    autopilot_connected=True,autopilot_mode='OFFBOARD')
        self.assertIn('전진 비행',self.window.observer_policy.text())

    def test_new_accepted_launch_clears_previous_3d_trace_but_rejection_preserves_it(self):
        accepted = [False]
        def reply(request, response):
            response.success = accepted[0]
            response.message = 'accepted' if response.success else 'start rejected'
            return response
        self.services.append(self.node.create_service(Trigger, '/jol/static_flight/launch', reply))
        self.wait_for(lambda: self.node.control_clients['static_launch'].service_is_ready())
        previous = [[-5., -6., 1.], [2., 3., 4.]]
        self.status(phase='preview_ready', locked=True, plan_ready=True, can_launch=True)
        self.node.static_actual_path.extend(previous)
        self.click('staticLaunchButton')
        self.wait_for(lambda: self.node.pending is None)
        self.assertEqual(list(self.node.static_actual_path), previous)
        accepted[0] = True
        self.status(phase='preview_ready', locked=True, plan_ready=True, can_launch=True)
        self.click('staticLaunchButton')
        self.wait_for(lambda: self.node.pending is None)
        self.assertFalse(self.node.static_actual_path)
        self.status(phase='arming', locked=True, plan_ready=True, position_world=[-5., -6., .1])
        self.assertEqual(list(self.node.static_actual_path), [[-5., -6., .1]])

    def test_px4_actual_mode_and_arming_remain_visible_during_flight_and_arrival(self):
        calls = []
        self.connect_services(calls)
        feedback = dict(locked=True, plan_ready=True, flight_backend='mavros_px4',
                        autopilot_connected=True, autopilot_mode='OFFBOARD', armed=True)
        for phase, label in [('arming','시동 중'), ('takeoff','비행 중'),
                             ('spline_tracking','비행 중'), ('arrived','도달 유지')]:
            with self.subTest(phase=phase):
                self.status(phase=phase, **feedback)
                self.assertEqual(self.window.observation_card.value.text(), label)
                self.assertEqual(self.window.observation_card.detail.text(), 'PX4 · OFFBOARD · 무장')
                self.assertIn('PX4 · OFFBOARD · 무장', self.window.observer_policy.text())
                self.assertNotIn('발사 준비: 대기', self.window.observer_policy.text())
                self.assertNotIn('발사 대기', self.window.observer_summary.text())
                self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
                self.assertTrue(self.window.static_buttons['static_abort'].isEnabled())
        self.status(phase='autopilot_landing', **dict(feedback, autopilot_mode='AUTO.LAND'))
        self.assertEqual(self.window.observation_card.value.text(), '착륙 중')
        self.assertIn('PX4 · AUTO.LAND · 무장', self.window.observation_card.detail.text())
        self.status(phase='aborted', **dict(feedback, armed=False, autopilot_mode='POSCTL'))
        self.assertIn('PX4 · POSCTL · 비무장', self.window.observer_policy.text())
        # Missing or stale FCU feedback must not inherit the prior OFFBOARD state.
        self.status(phase='arrived', locked=True, plan_ready=True, flight_backend='mavros_px4')
        self.assertIn('모드 미수신', self.window.observation_card.detail.text())
        self.assertIn('무장 미확인', self.window.observation_card.detail.text())
        self.assertNotIn('OFFBOARD', self.window.observer_policy.text())
        self.node.received['static_status'] -= int(3e9)
        self.window.refresh()
        self.assertEqual(self.window.observation_card.value.text(), '대기')
        self.assertNotIn('OFFBOARD', self.window.observer_policy.text())

    def test_world_ned_gimbal_recording_and_compact_layout(self):
        calls = []
        self.connect_services(calls)
        self.node.start_recording(self.directory.name)
        directory = self.node.recorder.directory
        joint = JointState(name=['gimbal_yaw_joint', 'gimbal_pitch_joint'], position=[.5, -.2])
        joint.header.stamp.sec = 10
        self.node.receive_joints(joint)
        self.window.last_csv_write = -float('inf')
        self.status()
        self.assertIn('(39.000, 2.000, 7.000)', self.window.target_label.text())
        self.assertIn('(8.000, 39.000, -7.000)', self.window.target_label.text())
        self.assertIn('짐벌 yaw 28.6', self.window.turret_feedback.text())
        self.assertFalse(self.window.stop_button.isVisible())
        self.assertTrue(self.window.resume_button.isVisible())
        self.assertEqual(self.window.resume_button.text(),'탐지 시작')
        self.assertFalse(self.window.arm_feedback.isVisible())
        self.window.resize(960, 700)
        self.app.processEvents()
        for button in self.window.static_buttons.values():
            position = button.mapTo(self.window.centralWidget(), QtCore.QPoint())
            self.assertTrue(self.window.centralWidget().rect().contains(QtCore.QRect(position, button.size())))
        self.capture('static_flight_960x700')
        self.node.stop_recording()
        with (directory/'selected_target.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[-1]['static_phase'], 'coordinate_ready')
        self.assertEqual(rows[-1]['ned_x_m'], '8.0')
        telemetry = [json.loads(line) for line in (directory/'telemetry.jsonl').read_text().splitlines()]
        self.assertEqual(telemetry[-1]['topic'], '/jol/static_flight/status')

    def test_lock_with_backend_auto_preview_enables_manual_launch_only(self):
        calls = []
        self.connect_services(calls)
        self.status()
        self.click('staticLockButton')
        self.wait_for(lambda: self.node.pending is None)
        self.packet('static_plan', stamp=10., valid=True, frame_id='world', home=[-5., -6., .1],
                    goal=[38., 2., 7.], samples=[dict(position=[-5., -6., .1]), dict(position=[38., 2., 7.])])
        self.status(phase='preview_ready', locked=True, plan_ready=True, can_launch=True)
        self.assertTrue(self.window.static_buttons['static_launch'].isEnabled())
        self.assertEqual(calls, ['lock'])  # 락과 자동 계획은 비행 명령을 보내지 않는다.
        self.assertIn('발사 버튼으로', self.window.observer_summary.text())
        self.status(phase='locked', locked=True, reason='path_collision')
        self.assertFalse(self.window.static_buttons['static_launch'].isEnabled())
        self.assertIn('경로 준비 필요 · path_collision', self.window.observer_summary.text())
        self.assertTrue(self.window.static_buttons['static_preview'].isEnabled())

    def test_locked_box_uses_matching_exposure_and_removes_stale_overlay(self):
        image = Image(width=640, height=480, encoding='rgb8', step=640*3, data=bytes(640*480*3))
        image.header.stamp.sec = 10
        self.node.receive_image(image)
        self.status(phase='locked', locked=True, target_id=1, lock_bbox=[314., 233., 14., 15.], lock_bbox_stamp=10.)
        self.window.render_image()
        displayed = self.window.camera_label.pixmap().toImage()
        # 축소 후 그린 실제 LOCKED 박스의 색을 검사한다. 검은 원본에는 이 색이 없다.
        def yellow_count(rendered):
            return sum(rendered.pixelColor(x, y).red() > 240 and rendered.pixelColor(x, y).green() > 170
                       and rendered.pixelColor(x, y).blue() < 80 for y in range(rendered.height())
                       for x in range(rendered.width()))
        self.assertGreater(yellow_count(displayed), 10)
        self.assertIs(self.node.display_image(), image)
        # 짐벌이 움직인 다음 영상에 이전 노출 박스를 그대로 투영하지 않는다.
        newer = Image(width=640, height=480, encoding='rgb8', step=640*3, data=bytes(640*480*3))
        newer.header.stamp.sec, newer.header.stamp.nanosec = 10, 100_000_000
        self.node.receive_image(newer)
        self.assertIs(self.node.display_image(), image)  # 보관한 실제 노출 영상에만 그린다.
        self.assertEqual(self.node.snapshot()['image_stamp'], 10.)
        self.status(phase='locked', locked=True, target_id=1, lock_bbox=[314., 233., 14., 15.], lock_bbox_stamp=8.)
        self.window.render_image()
        self.assertEqual(yellow_count(self.window.camera_label.pixmap().toImage()), 0)
        self.assertIs(self.node.display_image(), newer)
        # backend-selected evidence가 없으면 다른 검출을 대신 그리지 않는다.
        self.status(phase='locked', locked=True, target_id=1, lock_bbox=None, lock_bbox_stamp=None)
        self.window.render_image()
        self.assertEqual(yellow_count(self.window.camera_label.pixmap().toImage()), 0)
        self.status(phase='locked', locked=True, target_id=1, lock_bbox=[314., 233., 14., 15.], lock_bbox_stamp=10.05)
        self.window.render_image()  # 이 노출의 실제 영상이 수신되지 않았다.
        self.assertEqual(yellow_count(self.window.camera_label.pixmap().toImage()), 0)
        self.status(locked=False, target_id=1, lock_bbox=[314., 233., 14., 15.], lock_bbox_stamp=10.)
        self.window.render_image()
        self.assertEqual(yellow_count(self.window.camera_label.pixmap().toImage()), 0)

    def test_gimbal_alignment_uses_fresh_camera_residual_and_actual_joint_feedback(self):
        calls = []
        self.connect_services(calls)
        joint = JointState(name=['gimbal_yaw_joint', 'gimbal_pitch_joint'], position=[.35, -.18])
        joint.header.stamp.sec = 10
        self.node.receive_joints(joint)
        self.status(camera_centered=False, camera_alignment_error_deg=19.8, gimbal_tracking=True)
        self.assertIn('짐벌 yaw 20.1°', self.window.turret_feedback.text())
        self.assertIn('중심 잔차 19.80° · 정렬 중', self.window.turret_feedback.text())
        self.assertEqual(self.window.turret_feedback.property('tone'), 'warning')
        self.assertFalse(self.window.static_buttons['static_lock'].isEnabled())
        self.assertFalse(self.node.static_action_allowed('static_lock'))
        self.status(camera_centered=True, camera_alignment_error_deg=.42, gimbal_tracking=True)
        self.assertIn('중심 잔차 0.42° · 정렬 완료', self.window.turret_feedback.text())
        self.assertEqual(self.window.turret_feedback.property('tone'), 'good')
        self.assertTrue(self.window.static_buttons['static_lock'].isEnabled())
        row = self.node.snapshot()
        self.assertEqual(row['camera_alignment_error_deg'], .42)
        self.assertIs(row['camera_centered'], True)
        self.assertIs(row['gimbal_tracking'], True)
        self.node.received['static_status'] -= int(3e9)
        self.window.refresh()
        self.assertIn('중심 잔차 —° · 정렬 미확인', self.window.turret_feedback.text())
        self.assertNotIn('정렬 완료', self.window.turret_feedback.text())
        self.assertIsNone(self.node.snapshot()['camera_alignment_error_deg'])
        self.status(camera_centered=None, camera_alignment_error_deg=None, gimbal_tracking=False)
        self.assertIn('비전 추적 대기', self.window.turret_feedback.text())
        self.status(camera_centered='true', camera_alignment_error_deg=.42, gimbal_tracking=True)
        self.assertNotIn('static_status', self.node.packets)


if __name__ == '__main__':
    unittest.main()
