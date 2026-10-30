"""월드 예측·공식 팔 형상·방향 IK·제어 조건 검사."""
from collections import deque
import copy
import math
from types import SimpleNamespace
from types import MethodType
from threading import RLock
import unittest

import numpy as np
from jol_perception.pointing_core import (ARM_LOWER, ARM_UPPER, TargetSelector, arm_fk,
    choose_turret_state, feedback_step_interval, forecast, joint_slew_lower_bound, limit_step, patrol_step, pose,
    ray_error, solve_arm, solve_turret, turret_fk, turret_geometry, wrap)
from jol_perception.tracking_core import predict_cv, LEGACY_CV_NOISE_MODEL


class PointingTests(unittest.TestCase):
    def test_shutdown_race_is_accepted_only_after_ros_context_stops(self):
        from unittest.mock import patch, MagicMock
        from jol_perception import pointing_node as module
        for active in (True, False):
            executor=MagicMock();executor.spin.side_effect=RuntimeError('DDS take during shutdown')
            node=MagicMock()
            with patch.object(module.rclpy,'init'), patch.object(module.rclpy,'ok',return_value=active), \
                 patch.object(module.rclpy,'shutdown'), patch.object(module,'Pointing',return_value=node), \
                 patch.object(module,'SingleThreadedExecutor',return_value=executor):
                if active:
                    with self.assertRaises(RuntimeError):module.main()
                else:module.main()
            executor.shutdown.assert_called_once()
            node.destroy_node.assert_called_once()

    def track(self, identity=1, stamp=10., observed=True):
        return dict(id=identity, position=[50., 0., 10.], velocity=[0., 10., 0.],
                    state_covariance=np.eye(6).tolist(), position_covariance=np.eye(3).tolist(),
                    prediction=dict(acceleration_std=.7), confirmed=True, hits=3,
                    observed=observed, last_seen=stamp)

    def test_fk_includes_official_offsets_and_rotating_lever_arms(self):
        origin, tip, axis = arm_fk([0., 0., 0., 0.])
        self.assertTrue(np.allclose(origin, [.286, 0., .2045]))
        self.assertTrue(np.allclose(tip-origin, [.3, 0., 0.]))
        self.assertTrue(np.allclose(axis, [1., 0., 0.]))
        origin, _, _ = arm_fk([math.pi/2, 0., 0., 0.])
        self.assertTrue(np.allclose(origin, [.012, .274, .2045]))
        origin, axis = turret_fk([0., 0.])
        self.assertTrue(np.allclose(origin, [.14, -.145, 2.0125]))
        self.assertTrue(np.allclose(axis, [1., 0., 0.]))
        origin, axis = turret_fk([math.pi/2, 0.])
        self.assertTrue(np.allclose(origin, [.145, .14, 2.0125]))
        self.assertTrue(np.allclose(axis, [0., 1., 0.]))

    def test_directional_ik_targets_a_ray_not_an_unreachable_50m_arm_pose(self):
        for angle in np.linspace(-math.pi, math.pi, 25):
            target = [50*math.cos(angle), 50*math.sin(angle), 10.]
            arm, turret = solve_arm(target), solve_turret(target)
            self.assertTrue(arm['success'])
            self.assertTrue(turret['success'])
            self.assertLess(arm['angle_error_deg'], 1e-5)
            self.assertLess(turret['angle_error_deg'], 1e-5)
            self.assertGreater(arm['ray_depth_m'], 40.)
            self.assertTrue((np.asarray(arm['joints']) >= ARM_LOWER).all())
            self.assertTrue((np.asarray(arm['joints']) <= ARM_UPPER).all())
        # 베이스가 움직여도 관절 기하는 유지된다.
        base = pose((2., -3., 1.), yaw=.4)
        target = (base @ np.array([50., 2., 8., 1.]))[:3]
        self.assertLess(solve_arm(target, base)['angle_error_deg'], 1e-5)

    def test_arm_camera_turret_axis_is_independent_of_lidar_counterpitch(self):
        self.assertEqual(turret_geometry('arm')['frame'], 'jol/turret_pitch')
        self.assertEqual(turret_geometry()['frame'], 'jol/camera_optical')
        origin, axis = turret_fk([0., 0.], camera_mount='arm')
        self.assertTrue(np.allclose(origin, [0., 0., 1.]))
        self.assertTrue(np.allclose(axis, [1., 0., 0.]))
        target = [40., 30., 10.]
        for mode in ('payload', 'level'):
            origin, axis = turret_fk([.4, -.3], camera_mount='arm', lidar_mount_mode=mode)
            self.assertTrue(np.allclose(origin, [0., 0., 1.]))
            self.assertTrue(np.allclose(axis, pose(yaw=.4, pitch=-.3)[:3, 0]))
            solution = solve_turret(target, camera_mount='arm', lidar_mount_mode=mode)
            self.assertTrue(solution['success'])
            origin, axis = turret_fk(solution['joints'], camera_mount='arm', lidar_mount_mode=mode)
            self.assertLess(ray_error(origin, axis, target)['angle_error_deg'], 1e-5)
            base = pose((2., -3., 1.), yaw=.6, pitch=.15)
            world_target = (base @ np.r_[target, 1.])[:3]
            self.assertLess(solve_turret(world_target, base, 'arm', mode)['angle_error_deg'], 1e-5)
        with self.assertRaises(ValueError):
            turret_geometry('unknown')

    def test_arm_camera_is_never_used_as_turret_joint_feedback(self):
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import JointState
        from jol_perception.pointing_node import Pointing
        message = JointState()
        message.header.stamp.sec = 10
        message.name, message.position = ['yaw_joint', 'pitch_joint'], [0., 0.]
        calls = []

        def lookup(target, source, stamp):
            calls.append(source)
            self.assertIn(source, ('jol/turret_pitch', 'jol/base'))
            transform = TransformStamped()
            transform.transform.rotation.w = 1.
            transform.transform.translation.z = 1. if source == 'jol/turret_pitch' else 0.
            return transform

        fixture = SimpleNamespace(states={'turret': [message]}, joint_age=.3,
                                  camera_mount='arm', tf=SimpleNamespace(lookup_transform=lookup))
        result, _, _ = Pointing.measured(fixture, 'turret', 10.1)
        self.assertEqual(result['actual_frame'], 'jol/turret_pitch')
        self.assertEqual(result['actual_axis_index'], 0)
        self.assertEqual(result['actual_direction'], [1., 0., 0.])
        self.assertEqual(result['actual_origin'], [0., 0., 1.])
        self.assertEqual(calls, ['jol/turret_pitch', 'jol/base'])

    def test_moving_yaw_deck_arm_base_preserves_world_pointing(self):
        # 장착 변환은 FK·IK에 명시적으로 전달한다.
        target = np.array([40., 30., 10.])
        joint1 = []
        for yaw in (-2.8, -1., 0., 1., 2.8):
            base = pose((0., 0., .7), yaw=yaw) @ pose((-.8, .9, 1.))
            solution = solve_arm(target, base)
            origin, tip, direction = arm_fk(solution['joints'], base)
            self.assertTrue(solution['success'])
            self.assertLess(ray_error(origin, direction, target)['angle_error_deg'], 1e-5)
            self.assertAlmostEqual(np.linalg.norm(tip-origin), .3)
            joint1.append(solution['joints'][0])
        self.assertGreater(np.ptp(joint1), 3.)  # 상대 관절각으로 베이스의 월드 이동을 보상한다.

    def test_joint_limits_singular_target_and_positive_ray(self):
        self.assertFalse(solve_turret([1., 0., 50.])['success'])
        with self.assertRaises(ValueError):
            solve_turret([0., 0., 50.])
        with self.assertRaises(ValueError):
            solve_arm([0., 0., 0.])
        backwards = ray_error([0, 0, 0], [1, 0, 0], [-10, 0, 0])
        self.assertEqual(backwards['angle_error_deg'], 180.)
        self.assertLess(backwards['ray_depth_m'], 0.)

    def test_physical_arm_unwinds_while_continuous_turret_takes_short_arc(self):
        arm_current = np.array([3.1, -.6, .6, 0.])
        arm_desired = np.array([-3.1, -.6, .6, 0.])
        arm = limit_step(arm_current, arm_desired, 1.5, .05, ARM_LOWER, ARM_UPPER)
        self.assertAlmostEqual(arm[0], 3.025)
        self.assertLess(abs(arm[0]-arm_current[0]), .07500001)
        turret = limit_step([3.1, 0.], [-3.1, -.2], [1.2, .8], .05,
                            [-math.pi, -math.pi/6], [math.pi, math.pi/6], True)
        self.assertAlmostEqual(float(wrap(turret[0]-3.1)), .06)
        self.assertAlmostEqual(turret[1], -.04)

    def test_joint_slew_bound_wraps_only_the_continuous_turret(self):
        self.assertAlmostEqual(joint_slew_lower_bound(
            [3.1, 0.], [-3.1, -.2], [1.2, .8], True), .25)
        self.assertAlmostEqual(joint_slew_lower_bound(
            [3.1, -.6, .6, 0.], [-3.1, -.6, .6, 0.], 1.5), 6.2/1.5)
        self.assertEqual(joint_slew_lower_bound([0., 0.], [0., 0.], 1.), 0.)
        for current, goal, rate in [([0.], [1.], 0.), ([float('nan')], [1.], 1.),
                                    ([0.], [float('inf')], 1.), ([0.], [1., 2.], 1.)]:
            with self.assertRaises(ValueError):
                joint_slew_lower_bound(current, goal, rate)

    def test_feedback_budget_never_grows_for_old_future_or_paused_time(self):
        self.assertAlmostEqual(feedback_step_interval(.05, 10., 9.99), .06)
        for age in (0., .05, .1, .25, .3):
            self.assertAlmostEqual(feedback_step_interval(.05, 10., 10.-age), .05)
        self.assertEqual(feedback_step_interval(0., 10., 9.99), 0.)
        for args in [(-.05, 10., 10.), (.05, 10., 10.01),
                     (.05, float('nan'), 10.), (.05, 10., float('inf'))]:
            with self.assertRaises(ValueError):
                feedback_step_interval(*args)
        budget = feedback_step_interval(.05, 10., 9.99)
        reversed_goal = limit_step([.4, 0.], [-.4, 0.], [1.2, .8], budget,
                                  [-math.pi, -math.pi/6], [math.pi, math.pi/6], True)
        self.assertAlmostEqual(reversed_goal[0], .328)
        self.assertAlmostEqual(limit_step([.01], [0.], 1.2, budget, [-math.pi], [math.pi])[0], 0.)

    def test_patrol_revolves_once_in_eight_seconds_and_optical_axis_is_twenty_degrees_up(self):
        current = np.array([3.1, 0.])
        turn = 0.
        for _ in range(80):
            following = patrol_step(current, .1)
            increment = float(wrap(following[0]-current[0]))
            self.assertGreater(increment, 0.)
            self.assertLessEqual(abs(following[1]-current[1]), .080000001)
            turn += increment
            current = following
        self.assertAlmostEqual(turn, 2*math.pi)
        _, optical_axis = turret_fk(current)
        self.assertAlmostEqual(math.degrees(math.asin(optical_axis[2])), 20.)
        self.assertTrue(np.allclose(patrol_step(current, 0.), current))
        for options in ({'elevation_deg': 31}, {'period': 1}, {'period': float('nan')}, {'dt': -.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                patrol_step(current, **{'dt': .1, **options})

    def patrol_node(self, now=10.):
        """Gazebo 실행 없이 시각이 있는 관절 표본으로 콜백을 검사한다."""
        try:
            import json
            from jol_perception.pointing_node import Pointing, NAMES
            from builtin_interfaces.msg import Time
            from sensor_msgs.msg import JointState
        except ImportError:
            self.skipTest('ROS environment not sourced')
        clock = SimpleNamespace(value=now)
        statuses, commands = [], []
        node = SimpleNamespace(horizon=.5, turret_lead=0., max_age=.5, max_coast=1., max_fusion_age=.35,
            camera_mount='turret', lidar_mount_mode='payload',
            fusion_enabled=True, joint_age=.3, arm_rate=1.5, yaw_rate=1.2, pitch_rate=.8, rod_length=.3,
            patrol_elevation=20., patrol_period=8., selector=TargetSelector(), packet=None, fused_packet=None,
            background_ready=True, active=False, last_tick=None, mode='background', patrol_arm_hold=None,
            selection_changed_stamp=None,
            auto_acquisition_enabled=True, track_min_stamp=-math.inf, fusion_min_stamp=-math.inf,
            event_seq=0, last_event=dict(seq=0, action='startup', stamp=None, previous_target_id=None),
            states={key: deque(maxlen=100) for key in NAMES}, last_commands={},
            tf=SimpleNamespace(guard_lock=RLock(), dynamic_times={}, guard_rejections=0, future_rejections=0))

        def ros_time():
            sec, nanosec = divmod(round(clock.value*1e9), 1000000000)
            return Time(sec=sec, nanosec=nanosec)

        def command(key, values, stamp):
            commands.append((key, np.asarray(values).copy()))
            node.last_commands[key] = np.asarray(values).tolist()

        def measured(key, stamp):
            joints = node.latest_joints(key, stamp)
            if joints is None:
                raise ValueError('No measured joints')
            if key == 'turret':
                origin, axis = turret_fk(joints, camera_mount=node.camera_mount, lidar_mount_mode=node.lidar_mount_mode)
                base, extra = np.eye(4), {}
            else:
                turret = node.latest_joints('turret', stamp)
                base = pose((0., 0., .7), yaw=turret[0]) @ pose((-.8, .9, 1.))
                origin, tip, axis = arm_fk(joints, base)
                extra = {'rod_tip': tip.tolist()}
            return dict(actual_origin=origin.tolist(), actual_direction=axis.tolist(),
                        actual_joints=joints.tolist(), **extra), base, joints

        node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            to_msg=ros_time, nanoseconds=round(clock.value*1e9)))
        node.get_logger = lambda: SimpleNamespace(warning=lambda *args, **kwargs: None)
        node.command, node.measured, node.publish_markers = command, measured, lambda *args: None
        node.selection_origin = lambda source_stamp: [0., 0., 0.]
        node.status_pub = SimpleNamespace(publish=lambda msg: statuses.append(json.loads(msg.data)))
        node.markers_pub = SimpleNamespace(publish=lambda msg: None)
        for name in ('base_status', 'acquisition_event', 'stop_tracking', 'resume_auto', 'latest_joints', 'command_reference',
                     'hold_patrol_arm', 'patrol', 'hold_diagnostics', 'hold', 'tick', 'receive_tracks', 'receive_fusion'):
            setattr(node, name, MethodType(getattr(Pointing, name), node))
        for key, q in [('turret', [3.1, 0.]), ('arm', [.1, -.6, .6, .2])]:
            msg = JointState()
            msg.header.stamp = ros_time()
            msg.name, msg.position = NAMES[key], q
            node.states[key].append(msg)
        return node, clock, statuses, commands

    def test_turret_only_tracking_and_markers_work_without_arm_feedback(self):
        from jol_perception.pointing_node import Pointing
        node, clock, statuses, commands = self.patrol_node()
        node.arm_enabled = False
        node.states['arm'].clear()
        node.publish_markers = MethodType(Pointing.publish_markers, node)
        node.packet = dict(stamp=10., frame_id='world', tracks=[self.track()], stabilized=True)
        node.tick()
        status = statuses[-1]
        self.assertEqual(status['command_mode'], 'follow')
        self.assertFalse(status['arm_enabled'])
        self.assertNotIn('arm', status)
        self.assertNotIn('arm_state', status)
        self.assertIn('prediction_state', status)
        self.assertEqual({key for key, *_ in commands}, {'turret'})

    def test_arm_camera_tracking_uses_lidar_ik_while_arm_tracking_continues(self):
        node, clock, statuses, commands = self.patrol_node()
        node.camera_mount = 'arm'
        track = self.track()
        node.packet = dict(stamp=10., frame_id='world', tracks=[track], stabilized=True)
        node.tick()
        status = statuses[-1]
        self.assertEqual(status['camera_mount'], 'arm')
        self.assertTrue(status['arm']['ik_success'])
        self.assertTrue(status['turret']['ik_success'])
        expected = solve_turret(status['turret_prediction']['position'], camera_mount='arm')
        self.assertTrue(np.allclose(status['turret']['desired_joints'], expected['joints']))
        self.assertTrue(np.allclose(status['turret']['ideal_origin'], expected['origin']))

    def test_delayed_exact_tf_does_not_rewind_the_joint_slew_reference(self):
        import json
        from std_msgs.msg import String
        node, clock, statuses, commands = self.patrol_node()
        node.last_tick = 9.95
        old = {'turret': np.array([0., 0.]), 'arm': np.array([0., -.6, .6, 0.])}
        latest = {'turret': np.array([.06, 0.]), 'arm': np.array([.075, -.6, .6, 0.])}
        for key in old:
            node.states[key][-1].position = latest[key].tolist()

        def delayed_geometry(key, now):
            base = np.eye(4) if key == 'turret' else pose((-.8, .9, 1.7))
            if key == 'turret':
                origin, axis = turret_fk(old[key], base)
                extra = {}
            else:
                origin, tip, axis = arm_fk(old[key], base)
                extra = dict(rod_tip=tip.tolist())
            return dict(actual_origin=origin.tolist(), actual_direction=axis.tolist(),
                        actual_joints=old[key].tolist(), actual_tf_stamp=9.95,
                        joint_state_stamp=9.95, **extra), base, old[key]

        node.measured = delayed_geometry
        track = dict(self.track(), position=[0., 50., 10.], velocity=[0., 0., 0.])
        node.receive_tracks(String(data=json.dumps(dict(stamp=10., frame_id='world', tracks=[track]))))
        node.tick()
        status = statuses[-1]
        self.assertTrue(status['command_success'])
        for key, rate in [('turret', 1.2), ('arm', 1.5)]:
            detail = status[key]
            self.assertAlmostEqual(detail['command_joints'][0]-latest[key][0], rate*.05)
            self.assertEqual(detail['command_reference_joint_stamp'], 10.)
            self.assertEqual(detail['actual_tf_stamp'], 9.95)
            self.assertEqual(detail['actual_joints'], old[key].tolist())
            self.assertEqual(detail['command_reference_joints'], latest[key].tolist())
            self.assertFalse(detail['aligned'])
            self.assertGreater(detail['joint_slew_lower_bound_s'], .5)
        self.assertEqual(status['prediction']['horizon_s'], .5)
        self.assertEqual(status['turret_prediction']['horizon_s'], 0.)
        self.assertEqual(status['selection_age_s'], 0.)
        # 같은 ID의 새 패킷이 선택 경과시간을 초기화하면 안 된다.
        clock.value = 10.05
        node.tick()
        self.assertAlmostEqual(statuses[-1]['selection_age_s'], .05)
        self.assertEqual(statuses[-1]['selection_changed_stamp'], 10.)

    def test_stalled_tf_hold_reports_bounded_local_snapshot_without_changing_commands(self):
        from jol_perception.pointing_node import Pointing
        from tf2_ros import TransformException

        node, _, statuses, commands = self.patrol_node()
        node.active = True
        node.packet = dict(stamp=10., frame_id='world', tracks=[self.track()])
        node.tf.dynamic_times = {'jol/turret_pitch': [9000000000, 9690000000]}
        node.tf.dynamic_times.update({f'other_{i}': [9500000000] for i in range(40)})
        node.tf.guard_rejections, node.tf.future_rejections = 7, 2

        def missing_tf(*args):
            node.tf.guard_rejections += 1
            raise TransformException('TF guard missing timestamp brackets')

        node.tf.lookup_transform = missing_tf
        node.measured = MethodType(Pointing.measured, node)
        before = copy.deepcopy(node.tf.dynamic_times)
        node.tick()
        status = statuses[-1]
        self.assertEqual(status['state'], 'hold_invalid_state_or_tf')
        self.assertEqual(status['error'], 'turret_joint_state_or_exact_tf_unavailable')
        self.assertEqual(status['command_mode'], 'hold')
        self.assertFalse(status['command_success'])
        snapshot = status['hold_diagnostics']
        self.assertEqual(snapshot['latest_joint_stamp'], {'turret': 10., 'arm': 10.})
        guard = snapshot['tf_guard']
        self.assertEqual((guard['guard_rejections'], guard['future_rejections']), (8, 2))
        self.assertEqual(guard['dynamic_children_total'], 41)
        self.assertEqual(len(guard['dynamic_children']), 32)
        self.assertTrue(guard['dynamic_children_truncated'])
        self.assertAlmostEqual(guard['dynamic_children']['jol/turret_pitch']['latest_stamp'], 9.69)
        self.assertEqual(guard['dynamic_children']['jol/turret_pitch']['sample_count'], 2)
        self.assertEqual(node.tf.dynamic_times, before)
        self.assertEqual(len(commands), 2)
        for key, values in commands:
            self.assertTrue(np.array_equal(values, node.states[key][-1].position))

    def test_node_compensates_only_fresh_feedback_and_stop_still_holds_measured_arm(self):
        from std_srvs.srv import Trigger
        for age, expected_interval in [(.01, .06), (.25, .05)]:
            for direction in (-1., 1.):
                with self.subTest(age=age, direction=direction):
                    node, clock, statuses, commands = self.patrol_node()
                    node.last_tick = 9.95
                    for key, positions in [('turret', [0., 0.]), ('arm', [0., -.6, .6, 0.])]:
                        message = node.states[key][-1]
                        message.position = positions
                        message.header.stamp.sec, message.header.stamp.nanosec = divmod(round((10.-age)*1e9), 1000000000)
                    node.packet = dict(stamp=10., frame_id='world', tracks=[dict(
                        self.track(), position=[0., direction*50., 10.], velocity=[0., 0., 0.])])
                    node.tick()
                    status = statuses[-1]
                    self.assertTrue(status['command_success'])
                    for key, rate in [('turret', 1.2), ('arm', 1.5)]:
                        detail = status[key]
                        self.assertAlmostEqual(detail['command_joints'][0], direction*rate*expected_interval)
                        self.assertAlmostEqual(detail['command_feedback_age_s'], age)
                        self.assertAlmostEqual(detail['command_age_compensation_s'], expected_interval-.05)
                        self.assertAlmostEqual(detail['command_step_interval_s'], expected_interval)
                    count = len(statuses)
                    node.tick()
                    self.assertEqual(len(statuses), count)  # Paused clock emits no extra motion.
                    self.assertTrue(node.stop_tracking(Trigger.Request(), Trigger.Response()).success)
                    self.assertTrue(np.array_equal(commands[-1][1], node.states['arm'][-1].position))
                    self.assertFalse(node.auto_acquisition_enabled)
                    self.assertIsNone(statuses[-1]['target_id'])

    def test_startup_background_patrol_and_no_detection_patrol_hold_measured_arm(self):
        node, clock, statuses, commands = self.patrol_node()
        node.background_ready = False
        node.tick()
        self.assertEqual(statuses[-1]['mode'], 'background')
        self.assertEqual(statuses[-1]['command_mode'], 'patrol')
        self.assertEqual(statuses[-1]['patrol_elevation_deg'], 20.)
        initial_arm = node.patrol_arm_hold.copy()
        self.assertTrue(np.allclose(initial_arm, [.1, -.6, .6, .2]))
        node.background_ready = True
        clock.value += .1
        node.states['arm'][-1].position = [.2, -.5, .5, .1]
        node.tick()
        self.assertEqual(statuses[-1]['mode'], 'patrol')
        self.assertEqual(statuses[-1]['state'], 'waiting_for_tracks')
        self.assertTrue(np.allclose(node.patrol_arm_hold, initial_arm))
        self.assertTrue(np.allclose([q for key, q in commands if key == 'arm'][-1], initial_arm))

    def test_lidar_acquisition_follow_coast_expiry_and_stop_resume_latch(self):
        import json
        node, clock, statuses, commands = self.patrol_node()
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        packet = dict(stamp=10., frame_id='world', tracks=[self.track()])
        node.receive_tracks(String(data=json.dumps(packet)))
        node.tick()
        self.assertEqual(statuses[-1]['mode'], 'tracking')
        self.assertEqual(statuses[-1]['target_id'], 1)
        self.assertEqual(statuses[-1]['command_mode'], 'follow')
        clock.value = 10.1
        packet['stamp'], packet['tracks'][0]['observed'] = 10.1, False
        node.receive_tracks(String(data=json.dumps(packet)))
        node.tick()
        self.assertEqual(statuses[-1]['state'], 'coasting')
        self.assertEqual(statuses[-1]['mode'], 'tracking')

        response = node.stop_tracking(Trigger.Request(), Trigger.Response())
        self.assertTrue(response.success)
        self.assertFalse(node.auto_acquisition_enabled)
        self.assertIsNone(node.selector.selected_id)
        self.assertIsNone(node.fused_packet)
        self.assertEqual(statuses[-1]['event_action'], 'stop_tracking')
        self.assertEqual(statuses[-1]['last_event']['previous_target_id'], 1)
        # LiDAR 수신이 계속돼도 자동 획득을 끈 동안 재획득하지 않는다.
        clock.value = 10.2
        packet['stamp'], packet['tracks'][0]['observed'], packet['tracks'][0]['last_seen'] = 10.2, True, 10.2
        node.receive_tracks(String(data=json.dumps(packet)))
        node.receive_fusion(String(data=json.dumps(self.fusion_packet(stamp=10.2))))
        self.assertIsNotNone(node.packet)
        self.assertIsNone(node.fused_packet)
        node.tick()
        self.assertEqual(statuses[-1]['mode'], 'patrol')
        self.assertIsNone(statuses[-1]['target_id'])

        self.assertTrue(node.resume_auto(Trigger.Request(), Trigger.Response()).success)
        self.assertTrue(node.auto_acquisition_enabled)
        self.assertEqual(statuses[-1]['event_seq'], 2)
        self.assertIsNone(node.packet)
        node.receive_tracks(String(data=json.dumps(packet)))  # Pre-resume exposure is discarded.
        self.assertIsNone(node.packet)
        clock.value = 10.3
        packet['stamp'], packet['tracks'][0]['last_seen'] = 10.3, 10.3
        for queue in node.states.values():
            queue[-1].header.stamp.nanosec = 300000000
        node.receive_tracks(String(data=json.dumps(packet)))
        node.tick()
        self.assertEqual(statuses[-1]['mode'], 'tracking')
        # 잠깐 오래된 패킷은 유지하되 대상이 만료되면 순찰로 돌아간다.
        clock.value = 10.9
        node.tick()
        self.assertEqual(statuses[-1]['mode'], 'hold')
        clock.value = 11.4
        node.tick()
        self.assertEqual(statuses[-1]['mode'], 'patrol')
        self.assertIsNone(node.selector.selected_id)

    def test_stop_without_joint_state_tf_or_background_and_clock_reset_never_reenables_acquisition(self):
        node, clock, statuses, commands = self.patrol_node()
        from std_srvs.srv import Trigger
        node.states['arm'].clear()
        node.states['turret'].clear()
        node.background_ready = False
        self.assertTrue(node.stop_tracking(Trigger.Request(), Trigger.Response()).success)
        self.assertFalse(commands)
        node.tick()
        self.assertFalse(statuses[-1]['command_success'])
        self.assertEqual(statuses[-1]['mode'], 'background')
        count = len(statuses)
        node.tick()  # Paused/same ROS time cannot advance yaw or republish goals.
        self.assertEqual(len(statuses), count)
        clock.value = 2.
        node.tick()
        self.assertFalse(node.auto_acquisition_enabled)
        self.assertEqual(statuses[-1]['event_action'], 'clock_reset')
        self.assertEqual(statuses[-1]['event_seq'], 2)
        self.assertFalse(node.last_commands)

    def test_pre_reset_future_packets_cannot_poison_track_or_joint_caches(self):
        import json
        from jol_perception.pointing_node import Pointing
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        node, clock, _, _ = self.patrol_node()
        node.tick()
        clock.value = 2.
        node.tick()
        node.receive_tracks(String(data=json.dumps(dict(stamp=10., tracks=[]))))
        node.receive_tracks(String(data=json.dumps(dict(stamp=2., tracks=[]))))
        self.assertEqual(node.packet['stamp'], 2.)
        for stamp in (10., 2.):
            msg = JointState(name=['yaw_joint', 'pitch_joint'], position=[.2, -.1])
            msg.header.stamp.sec = int(stamp)
            Pointing.receive_joints(node, 'turret', msg)
        self.assertEqual(len(node.states['turret']), 1)
        self.assertEqual(node.states['turret'][-1].header.stamp.sec, 2)
        # 작은 /clock 전달 지연을 시간 초기화로 오해하지 않는다.
        node.receive_tracks(String(data=json.dumps(dict(stamp=2.01, frame_id='world', tracks=[self.track(stamp=2.01)]))))
        self.assertEqual(node.packet['stamp'], 2.01)
        self.assertEqual(node.selector.choose(node.packet, 2.)[1], 'stale_or_future_tracks')
        self.assertEqual(node.selector.choose(node.packet, 2.02)[1], 'tracking')

    def test_forecast_uses_command_age_plus_horizon_and_tracker_covariance_noise(self):
        track = self.track()
        original = copy.deepcopy(track)
        result = forecast(track, 10., 10.2, .5)
        self.assertTrue(np.allclose(result['position'], [50., 7., 10.]))
        self.assertAlmostEqual(result['stamp'], 10.7)
        self.assertAlmostEqual(result['propagation_s'], .7)
        expected = predict_cv(np.r_[track['position'], track['velocity']], np.eye(6), .7, .7,
                              LEGACY_CV_NOISE_MODEL)[1]
        self.assertTrue(np.allclose(result['position_covariance'], expected[:3, :3]))
        self.assertEqual(track, original)
        with self.assertRaises(ValueError):
            forecast(track, 12., 10., .5)

    def test_nearest_handover_approach_departure_and_stationary_target(self):
        selector = TargetSelector()
        near, far = self.track(5), self.track(1)
        near['position'], far['position'] = [30., 0., 5.], [45., 0., 5.]
        packet = dict(stamp=10., frame_id='world', tracks=[far, near])
        self.assertEqual(selector.choose(packet, 10.1)[0]['id'], 5)
        self.assertEqual(selector.detail['reason'], 'nearest_acquired')
        # 같은 LiDAR 상태 시각에서 더 가까워진 대상은 다음 평가 때 바로 선택한다.
        near['position'], far['position'] = [40., 0., 5.], [39.99, 0., 5.]
        self.assertEqual(selector.choose(packet, 10.2)[0]['id'], 1)
        self.assertEqual(selector.detail['reason'], 'nearest_handover')
        self.assertEqual(selector.detail['previous_target_id'], 5)
        far['velocity'] = [0., 0., 0.]
        self.assertEqual(selector.choose(packet, 10.3)[0]['id'], 1)
        self.assertEqual(selector.detail['source_stamp'], 10.)
        self.assertAlmostEqual(selector.detail['distance_m'], math.hypot(39.99, 5.))
        self.assertEqual([item['id'] for item in selector.detail['candidate_distances_m']], [1, 5])

    def test_nearest_switches_between_scans_using_current_time_without_mutating_measurements(self):
        selector = TargetSelector()
        first, second = self.track(1), self.track(2)
        first.update(position=[20., 0., 5.], velocity=[10., 0., 0.])
        second.update(position=[21., 0., 5.], velocity=[-10., 0., 0.])
        packet = dict(stamp=10., frame_id='world', tracks=[first, second])
        original = copy.deepcopy(packet)
        self.assertEqual(selector.choose(packet, 10.)[0]['id'], 1)
        self.assertEqual(selector.choose(packet, 10.1)[0]['id'], 2)
        self.assertEqual(selector.detail['position_source'], 'kalman_current_time')
        self.assertEqual(selector.detail['distance_state_stamp'], 10.1)
        candidate = selector.detail['candidate_distances_m'][1]
        np.testing.assert_allclose(candidate['position'], [20., 0., 5.], atol=1e-12)
        self.assertEqual(candidate['source_position'], [21., 0., 5.])
        self.assertEqual(packet, original)
        self.assertIsNone(selector.choose(packet, 10.6)[0])

    def test_one_selected_id_drives_current_turret_and_future_rod_through_handover(self):
        node, clock, statuses, _ = self.patrol_node(now=10.1)
        first, second = self.track(1), self.track(2)
        first.update(position=[30., 0., 10.], velocity=[10., 4., 0.])
        second.update(position=[34., 0., 10.], velocity=[-10., 4., 0.])
        node.packet = dict(stamp=10., frame_id='world', tracks=[first, second])
        for now, expected in ((10.1, first), (10.3, second)):
            clock.value = now
            node.tick()
            status = statuses[-1]
            self.assertEqual(status['target_id'], expected['id'])
            for part, field, horizon in (('turret', 'turret_prediction', 0.), ('arm', 'prediction', .5)):
                goal = status[field]
                self.assertEqual(goal['target_id'], expected['id'])
                self.assertEqual(goal['frame_id'], 'world')
                self.assertAlmostEqual(goal['stamp'], now+horizon)
                expected_position = np.asarray(expected['position']) + (now+horizon-10.)*np.asarray(expected['velocity'])
                np.testing.assert_allclose(goal['position'], expected_position)
                np.testing.assert_allclose(status[part]['target_position'], expected_position)
                self.assertTrue(status[part]['ik_success'])

    def test_nearest_is_three_dimensional_and_uses_translated_system_origin(self):
        selector = TargetSelector()
        high, low = self.track(1), self.track(2)
        high['position'], low['position'] = [3., 0., 20.], [10., 0., 1.]
        high['velocity'] = low['velocity'] = [0., 0., 0.]
        packet = dict(stamp=10., frame_id='world', tracks=[high, low])
        self.assertEqual(selector.choose(packet, 10.1)[0]['id'], 2)
        self.assertEqual(selector.choose(packet, 10.2, origin=[3., 0., 19.])[0]['id'], 1)
        self.assertEqual(selector.detail['origin'], [3., 0., 19.])
        self.assertAlmostEqual(selector.detail['distance_m'], 1.)
        with self.assertRaises(ValueError):
            selector.choose(packet, 10.2, origin=[float('nan'), 0., 0.])

    def test_exact_distance_tie_retains_current_then_lowest_id_independent_of_order(self):
        selector = TargetSelector()
        tracks = [self.track(5), self.track(2), self.track(8)]
        packet = dict(stamp=10., frame_id='world', tracks=tracks)
        self.assertEqual(selector.choose(packet, 10.1)[0]['id'], 2)
        selector.selected_id = 8
        for ordered in (tracks, list(reversed(tracks)), [tracks[1], tracks[2], tracks[0]]):
            packet['tracks'] = ordered
            self.assertEqual(selector.choose(packet, 10.2)[0]['id'], 8)
        tracks[0]['position'][0] -= .000001
        self.assertEqual(selector.choose(packet, 10.3)[0]['id'], 5)

    def test_nearest_recent_prediction_survives_sparse_return_gap_and_manual_override_holds(self):
        selector = TargetSelector()
        near, far = self.track(1), self.track(2)
        near['position'], far['position'] = [20., 0., 5.], [40., 0., 5.]
        packet = dict(stamp=10., frame_id='world', tracks=[near, far])
        self.assertEqual(selector.choose(packet, 10.1)[0]['id'], 1)
        near['observed'] = False
        self.assertEqual(selector.choose(packet, 10.2)[0]['id'], 1)
        self.assertEqual(selector.detail['policy'], 'nearest_recent_3d')
        self.assertFalse(selector.detail['candidate_distances_m'][0]['observed'])
        far['observed'] = False
        self.assertEqual(selector.choose(packet, 10.3)[1], 'coasting')
        self.assertEqual(selector.detail['reason'], 'nearest_retained')
        packet['tracks'], near['observed'] = [near], True
        self.assertEqual(selector.choose(packet, 10.4)[0]['id'], 1)
        # 수동 선택 ID는 더 가까운 대상이 있어도 유지한다.
        far['observed'], packet['tracks'] = True, [near, far]
        selector.request(2)
        self.assertEqual(selector.choose(packet, 10.4)[0]['id'], 2)
        far['observed'] = False
        self.assertEqual(selector.choose(packet, 10.4)[1], 'coasting')
        self.assertEqual(selector.detail['policy'], 'manual_id')
        selector.request(2)
        self.assertEqual(selector.choose(packet, 10.4)[1], 'requested_target_unobserved')
        packet['stamp'] = 11.1
        self.assertEqual(selector.choose(packet, 11.1)[1], 'requested_target_expired')
        selector.request(4)
        self.assertEqual(selector.choose(packet, 11.1)[1], 'requested_target_unavailable')

    def test_nearest_sparse_scan_does_not_make_actuators_chase_each_return(self):
        selector = TargetSelector()
        near, far = self.track(1), self.track(2)
        near['position'], far['position'] = [20., 0., 5.], [-40., 0., 5.]
        packet = dict(stamp=10., frame_id='world', tracks=[near, far])
        for index in range(20):
            stamp = 10.+index*.1
            packet['stamp'] = stamp
            for identity, track in enumerate((near, far)):
                track['observed'] = index % 2 == identity
                if track['observed']:
                    track['last_seen'] = stamp
            self.assertEqual(selector.choose(packet, stamp+.02)[0]['id'], 1)
        # 현재 예측이 더 가까우면 정답 좌표 없이 바로 선택을 바꾼다.
        far['position'] = [-19., 0., 5.]
        self.assertEqual(selector.choose(packet, stamp+.03)[0]['id'], 2)
        # 실측 없는 예측을 무한정 후보로 유지하지 않는다.
        packet['stamp'] = stamp+1.1
        near['observed'], far['observed'] = True, False
        near['last_seen'] = packet['stamp']
        self.assertEqual(selector.choose(packet, packet['stamp'])[0]['id'], 1)
        self.assertIn(dict(id=2, reason='expired_measurement'), selector.detail['rejected_candidates'])
        selector.request(0)
        near['observed'] = False
        self.assertIsNone(selector.choose(packet, packet['stamp'])[0])

    def test_nearest_rejects_unconfirmed_malformed_stale_future_and_duplicate_tracks(self):
        selector = TargetSelector()
        valid = self.track(20)
        tracks = [valid]
        changes = [{'confirmed': False}, {'hits': 2}, {'position': [float('nan'), 0., 0.]},
                   {'velocity': [1., 2.]}, {'last_seen': 9.}, {'last_seen': 11.},
                   {'state_covariance': (-np.eye(6)).tolist()}, {'observed': 'true'},
                   {'hits': float('nan')}, {'position_covariance': []}]
        for identity, change in enumerate(changes, 1):
            track = self.track(identity)
            track.update(position=[1., 0., 0.])
            track.update(change)
            tracks.append(track)
        tracks.extend([self.track(30), self.track(30), None, {'id': 'broken'}])
        packet = dict(stamp=10., frame_id='world', tracks=tracks)
        self.assertEqual(selector.choose(packet, 10.2)[0]['id'], 20)
        self.assertEqual([item['id'] for item in selector.detail['candidate_distances_m']], [20])
        self.assertIn('duplicate_id', [item['reason'] for item in selector.detail['rejected_candidates']])
        self.assertEqual(selector.choose(packet, 10.6)[1], 'stale_or_future_tracks')
        self.assertEqual(selector.choose(packet, 9.9)[1], 'stale_or_future_tracks')
        packet['frame_id'] = 'jol/lidar'
        self.assertEqual(selector.choose(packet, 10.1)[1], 'requires_world_tracks')
        selector.request(0)
        packet = dict(stamp=11., frame_id='world', tracks=[self.track(1, observed=False)])
        self.assertIsNone(selector.choose(packet, 11.)[0])

    def test_node_records_nearest_origin_and_switch_clears_old_camera_posterior(self):
        node, clock, statuses, commands = self.patrol_node()
        calls = []
        node.selection_origin = lambda stamp: calls.append(stamp) or [20., 0., 0.]
        near, far = self.track(1), self.track(2)
        near['position'], far['position'] = [40., 0., 10.], [-30., 0., 10.]
        node.packet = dict(stamp=10., frame_id='world', tracks=[far, near])
        node.tick()
        self.assertEqual(statuses[-1]['target_id'], 1)
        self.assertEqual(statuses[-1]['selection']['origin'], [20., 0., 0.])
        self.assertEqual(statuses[-1]['selection']['origin_tf_stamp'], 10.)
        self.assertEqual(calls, [10.])
        node.fused_packet = self.fusion_packet(near)
        far['position'] = [30., 0., 10.]
        clock.value = 10.1
        node.tick()
        self.assertEqual(statuses[-1]['target_id'], 2)
        self.assertEqual(statuses[-1]['selection']['reason'], 'nearest_handover')
        self.assertIsNone(node.fused_packet)
        self.assertEqual(node.fusion_min_stamp, 10.1)
        self.assertEqual(statuses[-1]['turret_state']['control_state_source'], 'lidar')

    def test_selection_origin_uses_packet_time_and_refuses_unavailable_tf(self):
        from jol_perception.pointing_node import Pointing
        from geometry_msgs.msg import TransformStamped
        calls = []

        def lookup(target, source, stamp):
            calls.append((target, source, stamp.nanoseconds))
            if stamp.nanoseconds != 10200000000:
                raise ValueError('Exact-time TF unavailable')
            transform = TransformStamped()
            transform.transform.translation.x = 12.
            transform.transform.translation.z = 3.
            return transform

        fixture = SimpleNamespace(tf=SimpleNamespace(lookup_transform=lookup))
        self.assertEqual(Pointing.selection_origin(fixture, 10.2), [12., 0., 3.])
        self.assertEqual(calls, [('world', 'jol/base', 10200000000)])
        with self.assertRaisesRegex(ValueError, 'selection_origin_exact_tf_unavailable'):
            Pointing.selection_origin(fixture, 10.3)
        self.assertNotIn(0, [item[2] for item in calls])

        node, _, statuses, _ = self.patrol_node()
        node.selection_origin = lambda stamp: Pointing.selection_origin(fixture, stamp)
        node.packet = dict(stamp=10., frame_id='world', tracks=[self.track()])
        node.tick()
        self.assertIsNone(node.selector.selected_id)
        self.assertEqual(statuses[-1]['mode'], 'hold')
        self.assertEqual(statuses[-1]['selection']['reason'], 'selection_origin_unavailable_or_invalid_packet')

    def fusion_packet(self, track=None, stamp=10.):
        track = copy.deepcopy(track or self.track())
        covariance = np.eye(6)*.1
        track.update(stamp=stamp, lidar_stamp=stamp, position=[50., .1, 10.],
                     source='camera_lidar', position_source='lidar_camera_bearing_ekf',
                     fusion_update={'accepted': True}, state_covariance=covariance.tolist(),
                     position_covariance=covariance[:3, :3].tolist(),
                     velocity_covariance=covariance[3:, 3:].tolist())
        return dict(stamp=stamp, frame_id='world', stabilized=True, calibration_valid=True,
                    fusion_reason='ready', tracks=[track])

    def test_camera_posterior_refines_only_selected_state_without_accumulation(self):
        lidar, packet = self.track(), self.fusion_packet()
        before = copy.deepcopy((lidar, packet))
        state, stamp, status = choose_turret_state(lidar, 10., packet, 10.1)
        self.assertEqual(status['control_state_source'], 'camera_lidar')
        self.assertEqual(status['camera_stamp'], stamp)
        self.assertEqual(status['fusion_lidar_stamp'], 10.)
        self.assertLess(status['fusion_position_variance_m2'], status['lidar_position_variance_m2'])
        self.assertTrue(np.allclose(forecast(state, stamp, 10.1, .1)['position'], [50., 2.1, 10.]))
        self.assertTrue(np.allclose(forecast(lidar, 10., 10.1, .5)['position'], [50., 6., 10.]))
        # 같은 보정 결과를 다시 읽어도 칼만 갱신을 반복하지 않는다.
        self.assertEqual(choose_turret_state(lidar, 10., packet, 10.1), (state, stamp, status))
        self.assertEqual((lidar, packet), before)
        state, stamp, detail = choose_turret_state(lidar, 10., packet, 10.1, enabled=False)
        self.assertIs(state, lidar)
        self.assertEqual(stamp, 10.)
        self.assertEqual(detail['fusion_selection_reason'], 'coordinate_fusion_disabled')

    def test_camera_state_gates_preserve_lidar_acquisition_and_identity(self):
        lidar = self.track()
        cases = [({'calibration_valid': False}, 'fusion_geometry_not_ready'),
                 ({'stabilized': False}, 'fusion_geometry_not_ready'),
                 ({'frame_id': 'camera'}, 'fusion_geometry_not_ready'),
                 ({'fusion_reason': 'stale_lidar'}, 'fusion_geometry_not_ready'),
                 ({'stamp': 9.}, 'stale_or_future_fusion'),
                 ({'stamp': 11.}, 'stale_or_future_fusion'),
                 ({'tracks': []}, 'selected_lidar_id_not_unique_in_fusion')]
        for changes, reason in cases:
            packet = self.fusion_packet()
            packet.update(changes)
            with self.subTest(changes=changes):
                state, stamp, detail = choose_turret_state(lidar, 10., packet, 10.1)
                self.assertIs(state, lidar)
                self.assertEqual(stamp, 10.)
                self.assertEqual(detail['fusion_selection_reason'], reason)
        for changes, reason in [({'id': 2}, 'selected_lidar_id_not_unique_in_fusion'),
                                ({'fusion_update': {'accepted': False}}, 'camera_update_not_accepted'),
                                ({'lidar_stamp': 10.05}, 'invalid_fusion_source_stamp'),
                                ({'stamp': 9.9}, 'invalid_fusion_source_stamp'),
                                ({'prediction': {'process_noise_model': 'continuous_white_acceleration',
                                                 'acceleration_std': .7, 'acceleration_reference_dt_s': .1,
                                                 'acceleration_spectral_density_m2_s3': .049}}, 'fusion_motion_model_mismatch'),
                                ({'velocity': [float('nan'), 0, 0]}, 'invalid_fusion_state_or_covariance'),
                                ({'state_covariance': (-np.eye(6)).tolist()}, 'invalid_fusion_state_or_covariance'),
                                ({'position_covariance': np.eye(3).tolist()}, 'inconsistent_fusion_covariance')]:
            packet = self.fusion_packet()
            packet['tracks'][0].update(changes)
            with self.subTest(changes=changes):
                state, _, detail = choose_turret_state(lidar, 10., packet, 10.1)
                self.assertIs(state, lidar)
                self.assertEqual(detail['fusion_selection_reason'], reason)

    def test_delayed_camera_posterior_must_beat_new_lidar_uncertainty(self):
        lidar = self.track(stamp=10.25)
        lidar['state_covariance'] = (np.eye(6)*.01).tolist()
        lidar['position_covariance'] = (np.eye(3)*.01).tolist()
        old_camera = self.fusion_packet(stamp=10.)
        state, stamp, detail = choose_turret_state(lidar, 10.25, old_camera, 10.3)
        self.assertIs(state, lidar)
        self.assertEqual(stamp, 10.25)
        self.assertEqual(detail['fusion_selection_reason'], 'lidar_has_lower_predicted_uncertainty')

    def test_malformed_fusion_metadata_falls_back_without_stopping_control(self):
        lidar = self.track()
        for malformed in ([], None, 'valid'):
            packet = self.fusion_packet()
            packet['tracks'][0]['temporal_fusion'] = malformed
            selected, stamp, detail = choose_turret_state(lidar, 10., packet, 10.1)
            self.assertIs(selected, lidar)
            self.assertEqual(stamp, 10.)
            self.assertEqual(detail['fusion_selection_reason'], 'invalid_delayed_fusion')
        packet = self.fusion_packet()
        packet['tracks'][0]['id'] = float('inf')
        self.assertEqual(choose_turret_state(lidar, 10., packet, 10.1)[2]['control_state_source'], 'lidar')

    def test_inconsistent_continuous_noise_never_reaches_pointing(self):
        lidar = self.track()
        sigma = lidar['prediction']['acceleration_std']
        lidar['process_noise_model'] = 'continuous_white_acceleration'
        lidar['prediction'].update(process_noise_model='continuous_white_acceleration',
                                   acceleration_reference_dt_s=.1,
                                   acceleration_spectral_density_m2_s3=sigma**2*.1)
        packet = self.fusion_packet(lidar)
        self.assertEqual(choose_turret_state(lidar, 10., packet, 10.1)[2]['control_state_source'], 'camera_lidar')
        packet['tracks'][0]['prediction']['acceleration_spectral_density_m2_s3'] *= 10
        selected, _, detail = choose_turret_state(lidar, 10., packet, 10.1)
        self.assertIs(selected, lidar)
        self.assertEqual(detail['fusion_selection_reason'], 'invalid_fusion_motion_metadata')
        lidar['prediction']['acceleration_reference_dt_s'] = 1.
        with self.assertRaises(ValueError):
            forecast(lidar, 10., 10.1, .5)
        selector = TargetSelector()
        selected, _ = selector.choose(dict(stamp=10., frame_id='world', tracks=[lidar]), 10.1)
        self.assertIsNone(selected)
        self.assertEqual(selector.detail['rejected_candidates'][0]['reason'], 'invalid_track')

    def test_replayed_fusion_separates_evidence_state_and_measurement_freshness(self):
        lidar, packet = self.track(stamp=10.6), self.fusion_packet(stamp=10.)
        lidar['birth_stamp'] = 9.
        posterior = packet['tracks'][0]
        posterior.update(birth_stamp=9., current_state_stamp=10.6, last_measurement_stamp=10.6,
                         temporal_fusion={'valid': True}, current_state={key: copy.deepcopy(posterior[key])
                             for key in ('position', 'velocity', 'state_covariance', 'position_covariance',
                                         'velocity_covariance', 'prediction')})
        packet.update(current_state_stamp=10.6, lidar_stamp=10., last_measurement_stamp=10.6)
        state, stamp, detail = choose_turret_state(lidar, 10.6, packet, 10.65, selection_horizon=.5)
        self.assertEqual(detail['control_state_source'], 'camera_lidar')
        self.assertEqual(stamp, 10.6)
        self.assertAlmostEqual(detail['fusion_age_s'], .65)
        self.assertAlmostEqual(detail['fusion_last_measurement_age_s'], .05)
        self.assertAlmostEqual(detail['uncertainty_comparison_stamp'], 11.15)
        # 새 LiDAR로 오래된 영상을 계속 유효하게 만들지 않는다.
        self.assertEqual(choose_turret_state(lidar, 10.6, packet, 10.81)[2]['fusion_selection_reason'],
                         'stale_or_future_fusion')
        # 예측 시각이 진행돼도 마지막 실측의 경과시간은 유지된다.
        posterior['last_measurement_stamp'] = 10.
        self.assertEqual(choose_turret_state(lidar, 10.6, packet, 10.65)[2]['fusion_selection_reason'],
                         'invalid_fusion_source_stamp')
        posterior['last_measurement_stamp'] = 10.5
        self.assertEqual(choose_turret_state(lidar, 10.6, packet, 10.65)[2]['fusion_selection_reason'],
                         'fusion_missing_latest_lidar_measurement')
        posterior['last_measurement_stamp'] = 10.6
        self.assertEqual(choose_turret_state(lidar, 10.6, packet, 10.65)[2]['control_state_source'], 'camera_lidar')

    def test_stale_rejection_retains_candidate_timestamps(self):
        packet = self.fusion_packet(stamp=9.)
        packet.update(current_state_stamp=9.2, lidar_stamp=8.9)
        detail = choose_turret_state(self.track(), 10., packet, 10.1)[2]
        self.assertEqual(detail['fusion_selection_reason'], 'stale_or_future_fusion')
        self.assertEqual(detail['camera_stamp'], 9.)
        self.assertEqual(detail['fusion_state_stamp'], 9.2)
        self.assertEqual(detail['fusion_lidar_stamp'], 8.9)
        self.assertAlmostEqual(detail['fusion_age_s'], 1.1)

    def test_node_uses_fusion_independently_for_camera_and_rod_horizons(self):
        node, clock, statuses, commands = self.patrol_node()
        track = self.track(stamp=10.)
        track['state_covariance'] = np.diag([.02]*3+[.01]*3).tolist()
        track['position_covariance'] = (np.eye(3)*.02).tolist()
        node.packet = dict(stamp=10., frame_id='world', tracks=[track], stabilized=True)
        node.selector.selected_id, node.selector.last_seen = track['id'], 10.
        node.selection_changed_stamp = 10.
        node.fused_packet = self.fusion_packet(track)
        posterior = node.fused_packet['tracks'][0]
        posterior['state_covariance'] = np.diag([.001]*3+[.1]*3).tolist()
        posterior['position_covariance'] = (np.eye(3)*.001).tolist()
        posterior['velocity_covariance'] = (np.eye(3)*.1).tolist()
        node.tick()
        status = statuses[-1]
        self.assertEqual(status['turret_state']['control_state_source'], 'camera_lidar')
        self.assertEqual(status['arm_state']['control_state_source'], 'lidar')
        self.assertAlmostEqual(status['turret_state']['uncertainty_comparison_stamp'], 10.)
        self.assertAlmostEqual(status['arm_state']['uncertainty_comparison_stamp'], 10.5)
        # 속도 불확실성도 줄면 팔 조준에도 융합 상태를 사용한다.
        posterior['state_covariance'] = (np.eye(6)*.001).tolist()
        posterior['velocity_covariance'] = (np.eye(3)*.001).tolist()
        clock.value = 10.1
        node.tick()
        status = statuses[-1]
        self.assertEqual(status['arm_state']['control_state_source'], 'camera_lidar')
        self.assertTrue(np.allclose(status['prediction']['position'],
                                    forecast(posterior, 10., 10.1, .5)['position']))

    def test_camera_uncertainty_is_compared_at_actual_aim_time(self):
        lidar, packet = self.track(), self.fusion_packet()
        lidar['state_covariance'] = np.diag([.02]*3+[.01]*3).tolist()
        lidar['position_covariance'] = (np.eye(3)*.02).tolist()
        posterior = packet['tracks'][0]
        posterior['state_covariance'] = np.diag([.01]*3+[2.]*3).tolist()
        posterior['position_covariance'] = (np.eye(3)*.01).tolist()
        posterior['velocity_covariance'] = (np.eye(3)*2.).tolist()
        # 촬영 시각에는 위치가 정확해도 속도 불확실성은 더 클 수 있다.
        self.assertEqual(choose_turret_state(lidar, 10., packet, 10.)[2]['control_state_source'], 'camera_lidar')
        state, stamp, detail = choose_turret_state(lidar, 10., packet, 10., selection_horizon=.1)
        self.assertIs(state, lidar)
        self.assertEqual(stamp, 10.)
        self.assertAlmostEqual(detail['uncertainty_comparison_stamp'], 10.1)
        self.assertEqual(detail['fusion_selection_reason'], 'lidar_has_lower_predicted_uncertainty')
        self.assertGreater(detail['fusion_position_variance_m2'], detail['lidar_position_variance_m2'])

    def test_fusion_receiver_drops_out_of_order_duplicate_and_future_exposures(self):
        try:
            import json
            from jol_perception.pointing_node import Pointing
            from std_msgs.msg import String
        except ImportError:
            self.skipTest('ROS environment not sourced')
        warnings = []
        node = SimpleNamespace(fused_packet=None, auto_acquisition_enabled=True, fusion_min_stamp=-math.inf,
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=11000000000)),
            get_logger=lambda: SimpleNamespace(warning=lambda message, **kwargs: warnings.append(message)))
        packet = self.fusion_packet(stamp=10.)
        Pointing.receive_fusion(node, String(data=json.dumps(packet)))
        original = node.fused_packet
        for stamp in (9., 10., 12.):
            Pointing.receive_fusion(node, String(data=json.dumps(self.fusion_packet(stamp=stamp))))
            self.assertIs(node.fused_packet, original)
        revised = dict(packet, current_state_stamp=10.5)
        Pointing.receive_fusion(node, String(data=json.dumps(revised)))
        self.assertEqual(node.fused_packet['current_state_stamp'], 10.5)
        # 같은 영상의 상태 진행은 더 새로운 LiDAR 반영이 있을 때만 허용한다.
        accepted = node.fused_packet
        Pointing.receive_fusion(node, String(data=json.dumps(revised)))
        self.assertIs(node.fused_packet, accepted)
        informed = dict(revised, fusion_revision=1)
        Pointing.receive_fusion(node, String(data=json.dumps(informed)))
        self.assertEqual(node.fused_packet['fusion_revision'], 1)
        accepted = node.fused_packet
        for packet in (informed, revised, dict(revised, current_state_stamp=10.6)):
            Pointing.receive_fusion(node, String(data=json.dumps(packet)))
            self.assertIs(node.fused_packet, accepted)
        Pointing.receive_fusion(node, String(data='[]'))
        self.assertEqual(len(warnings), 1)
        Pointing.receive_background(node, SimpleNamespace(data=False))
        self.assertIsNone(node.fused_packet)

    def test_node_preserves_background_survey_and_holds_after_control_ownership(self):
        try:
            from jol_perception.pointing_node import Pointing
            from builtin_interfaces.msg import Time
        except ImportError:
            self.skipTest('ROS environment not sourced')
        output, commands = [], []
        publisher = SimpleNamespace(publish=lambda message: output.append(message))
        node = SimpleNamespace(active=False, states={'turret': [], 'arm': []}, last_commands={},
                               markers_pub=publisher, status_pub=publisher, joint_age=.3)
        node.command = lambda key, values, stamp: commands.append((key, values))
        status = dict(stamp=10., state='waiting_for_background', command_stamp=None)
        Pointing.hold(node, status, Time(sec=10))
        self.assertFalse(commands)
        self.assertEqual(status['command_mode'], 'preserve_background_survey')
        node.active, node.last_commands = True, {'turret': [1., 0.], 'arm': [0., -.6, .6, 0.]}
        Pointing.hold(node, status, Time(sec=10))
        self.assertEqual(len(commands), 2)
        self.assertFalse(status['command_success'])
        self.assertEqual(status['command_mode'], 'hold')

    def test_node_constructs_with_ros_and_measures_only_exact_stamped_tf(self):
        try:
            import rclpy
            from jol_perception.pointing_node import Pointing
            from sensor_msgs.msg import JointState
            from geometry_msgs.msg import TransformStamped
        except ImportError:
            self.skipTest('ROS environment not sourced')
        rclpy.init()
        node = Pointing()
        try:
            self.assertEqual(node.states['turret'].maxlen, 100)
            node.tick()  # 토픽 수신 전 생성자와 대기 타이머를 검사한다.
        finally:
            node.destroy_node()
            rclpy.shutdown()
        messages, calls = deque(), []
        for seconds in (10, 11):
            message = JointState()
            message.header.stamp.sec = seconds
            message.name, message.position = ['yaw_joint', 'pitch_joint'], [float(seconds), 0.]
            messages.append(message)

        def lookup(target, source, stamp):
            calls.append((target, source, stamp.nanoseconds))
            if stamp.nanoseconds == 11000000000:
                raise ValueError('Newest state TF has not arrived')
            transform = TransformStamped()
            transform.transform.rotation.w = 1.
            transform.transform.translation.x = 3.
            return transform

        fixture = SimpleNamespace(states={'turret': messages}, joint_age=1.2,
                                  tf=SimpleNamespace(lookup_transform=lookup))
        result, _, _ = Pointing.measured(fixture, 'turret', 11.1)
        self.assertEqual(result['joint_state_stamp'], 10.)
        self.assertEqual(result['actual_tf_stamp'], 10.)
        self.assertEqual(result['actual_joints'], [10., 0.])
        self.assertEqual(result['actual_origin'], [3., 0., 0.])
        self.assertNotIn(0, [item[2] for item in calls])  # 최신 TF로 대체하지 않는다.
        fixture.joint_age = .3
        with self.assertRaises(ValueError):
            Pointing.measured(fixture, 'turret', 11.1)

        # 200 Hz 상태의 TF가 0.2초 늦으면 30개 버퍼는 부족하므로 100개를 보존한다.
        dense = deque(maxlen=100)
        for index in range(100):
            nanoseconds = 10405000000+index*5000000
            message = JointState()
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(nanoseconds, 1000000000)
            message.name, message.position = ['yaw_joint', 'pitch_joint'], [nanoseconds*1e-9, 0.]
            dense.append(message)

        def delayed_lookup(target, source, stamp):
            self.assertGreater(stamp.nanoseconds, 0)
            if stamp.nanoseconds > 10700000000:
                raise ValueError('TF lags the joint-state topic by 0.2 s')
            transform = TransformStamped()
            transform.transform.rotation.w = 1.
            return transform

        fixture.states['turret'], fixture.tf.lookup_transform = dense, delayed_lookup
        result, _, _ = Pointing.measured(fixture, 'turret', 10.9)
        self.assertAlmostEqual(result['actual_tf_stamp'], 10.7)
        self.assertEqual(result['actual_tf_stamp'], result['joint_state_stamp'])
        self.assertAlmostEqual(result['actual_joints'][0], 10.7)
        fixture.states['turret'] = deque(list(dense)[-30:])
        with self.assertRaises(ValueError):
            Pointing.measured(fixture, 'turret', 10.9)

    def test_arm_uses_moving_base_tf_at_its_joint_stamp_and_never_latest_base(self):
        try:
            from jol_perception.pointing_node import Pointing
            from sensor_msgs.msg import JointState
            from geometry_msgs.msg import TransformStamped
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest('ROS environment not sourced')
        messages, calls = deque(), []
        joints = [.2, -.6, .6, -.2]
        for stamp in (10, 11):
            message = JointState()
            message.header.stamp.sec = stamp
            message.name, message.position = ['joint1', 'joint2', 'joint3', 'joint4'], joints
            messages.append(message)
        newest_base_missing = [True]

        def lookup(target, frame, stamp):
            time = stamp.nanoseconds*1e-9
            calls.append((frame, time))
            if frame == 'jol/arm_base' and time == 11. and newest_base_missing[0]:
                raise ValueError('Moving arm-base TF has not arrived')
            base = pose((0., 0., .7), yaw=time-10.) @ pose((-.8, .9, 1.))
            if frame == 'jol/arm_base':
                matrix = base
            else:
                origin, tip, _ = arm_fk(joints, base)
                matrix = base @ pose(yaw=joints[0], pitch=sum(joints[1:]))
                matrix[:3, 3] = tip if frame == 'jol/rod_tip' else origin
            message = TransformStamped()
            p, q = message.transform.translation, message.transform.rotation
            p.x, p.y, p.z = matrix[:3, 3]
            q.x, q.y, q.z, q.w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            return message

        fixture = SimpleNamespace(states={'arm': messages}, joint_age=1.2,
                                  tf=SimpleNamespace(lookup_transform=lookup))
        old, old_base, _ = Pointing.measured(fixture, 'arm', 11.1)
        self.assertEqual(old['base_tf_stamp'], 10.)
        self.assertEqual(old['joint_state_stamp'], old['base_tf_stamp'])
        self.assertTrue(np.allclose(old_base, pose((0., 0., .7)) @ pose((-.8, .9, 1.))))
        newest_base_missing[0] = False
        new, new_base, _ = Pointing.measured(fixture, 'arm', 11.1)
        self.assertEqual(new['base_tf_stamp'], 11.)
        self.assertTrue(np.allclose(new_base, pose((0., 0., .7), yaw=1.) @ pose((-.8, .9, 1.))))
        self.assertFalse(np.allclose(old['actual_direction'], new['actual_direction']))
        self.assertNotIn(0., [stamp for _, stamp in calls])


if __name__ == '__main__':
    unittest.main()
