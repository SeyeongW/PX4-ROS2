"""Exercise the real estimator's corrected acquisition queue without a ROS graph."""
from collections import Counter, deque
import time
from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import rclpy

from jol_perception.static_estimator import StaticEstimator, corrected_stamp_seconds
from jol_perception.static_estimator_core import sensor_quality_config, validate_detection


def cloud(stamp, offsets):
    fields = [PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
              for i, name in enumerate(('x', 'y', 'z', 'time'))]
    return point_cloud2.create_cloud(Header(stamp=Time(seconds=stamp).to_msg(), frame_id='lidar'),
                                    fields, [(2., 0., 1., offset) for offset in offsets])


def estimator_queue(message, **quality):
    calls, published = [], []
    publisher = SimpleNamespace(publish=lambda msg: published.append(msg))
    fake = SimpleNamespace(sensor_quality=sensor_quality_config(**quality),
        pending_clouds=deque([(message, time.monotonic())]), max_age=.8, clouds=deque(),
        counters=Counter(), fixed_frame='world', last_result=None,
        registered_pub=publisher, context_pub=publisher, target_points_pub=publisher,
        get_logger=lambda: SimpleNamespace(warning=lambda *a, **kw: None))
    fake.lookup_pose = lambda frame, stamp: (calls.append(stamp) or [0., 0., 0., 0., 0., 0., 1.])
    fake.lookup_matrix = lambda target, source, stamp: (calls.append(stamp) or np.eye(4))
    return fake, calls, published


def test_signed_clock_offsets_apply_once_before_point_acquisition_tf():
    fake, calls, published = estimator_queue(cloud(1.9, [-.1, 0.]), lidar_time_offset_s=.05)
    StaticEstimator.drain_clouds(fake, 2.)
    assert len(fake.clouds) == 1 and not fake.pending_clouds
    assert abs(fake.clouds[0]['stamp']-1.95) < 1e-9
    assert abs(min(calls)-1.85) < 1e-7 and max(calls) <= 2.
    assert published[0].header.stamp == Time(seconds=1.95).to_msg()
    packet = dict(stamp=2., detections=[])
    corrected = StaticEstimator.correct_detection_time(
        SimpleNamespace(sensor_quality=sensor_quality_config(camera_time_offset_s=-.01)), packet)
    assert corrected['stamp'] == 1990000000*1e-9 and corrected['source_stamp'] == 2.
    assert packet['stamp'] == 2.  # UI detections retain original image stamps.


def test_corrected_camera_and_cloud_share_integer_nanosecond_clock_boundary():
    # Float subtraction previously produced 2.3000000000000003 > 2.3 and
    # rejected an acquisition at the exact clock boundary as future.
    current = 2300000000*1e-9
    camera = StaticEstimator.correct_detection_time(
        SimpleNamespace(sensor_quality=sensor_quality_config(camera_time_offset_s=-.01)),
        dict(stamp=2310000000*1e-9))
    lidar = corrected_stamp_seconds(Time(nanoseconds=2280000000).to_msg(), .02)
    assert camera['stamp'] == lidar == current
    info = dict(k=[500., 0., 320., 0., 500., 240., 0., 0., 1.], d=[0.]*5,
                width=640, height=480, frame_id='optical', distortion_model='plumb_bob')
    packet = dict(camera, width=640, height=480, frame_id='optical',
                  detections=[dict(bbox=[312., 232., 16., 16.], confidence=1.)])
    validate_detection(packet, info, current)
    packet['stamp'] += 1e-9
    import pytest
    with pytest.raises(ValueError, match='stale_or_invalid_camera_stamp'):
        validate_detection(packet, info, current)
    for invalid in (float('inf'), float('nan'), 1e308, -1.):
        with pytest.raises(ValueError, match='stale_or_invalid_camera_stamp'):
            StaticEstimator.correct_detection_time(SimpleNamespace(sensor_quality=sensor_quality_config()),
                                                   dict(stamp=invalid))
    message = cloud(2.28, [-.1, 0.])
    message.header.stamp = Time(nanoseconds=2280000000).to_msg()
    fake, calls, _ = estimator_queue(message, lidar_time_offset_s=.02)
    StaticEstimator.drain_clouds(fake, current)
    assert len(fake.clouds) == 1 and not fake.pending_clouds and max(calls) <= current
    message.header.stamp = Time(nanoseconds=2280000001).to_msg()
    fake, calls, _ = estimator_queue(message, lidar_time_offset_s=.02)
    StaticEstimator.drain_clouds(fake, current)
    assert len(fake.pending_clouds) == 1 and not calls


def test_future_cloud_waits_for_clock_and_future_scan_points_are_rejected():
    fake, calls, _ = estimator_queue(cloud(2.1, [-.1, 0.]))
    StaticEstimator.drain_clouds(fake, 2.)
    assert len(fake.pending_clouds) == 1 and not calls and not fake.clouds
    StaticEstimator.drain_clouds(fake, 2.2)
    assert len(fake.clouds) == 1 and max(calls) <= 2.2
    bad, calls, _ = estimator_queue(cloud(1.9, [0., .01]))
    StaticEstimator.drain_clouds(bad, 2.)
    assert not bad.clouds and not calls and bad.counters['invalid_cloud'] == 1


def test_fixed_extrinsic_signature_excludes_dynamic_gimbal_rotation():
    transform = TransformStamped()
    transform.transform.rotation.w = 1.
    transform.transform.translation.x = .084
    calls = []
    fake = SimpleNamespace(info=dict(frame_id='optical'), lidar_frame='lidar', fixed_frame='world',
        tf=SimpleNamespace(parents=dict(optical='gimbal', gimbal='base', lidar='base', base='world'),
            static_frames={'optical', 'lidar', 'base'},
            lookup_transform=lambda parent, child, stamp: (calls.append(child) or transform)))
    signature = StaticEstimator.static_calibration_transforms(fake, 2.)
    assert set(signature) == {'optical', 'lidar', 'base'} and 'gimbal' not in calls
    assert signature['optical']['pose'][0] == .084


def test_actual_node_waits_for_image_clock_and_preserves_only_fresh_coordinates():
    import json
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true', '-p', 'calibration_valid:=true',
                    '-p', 'calibration_source:=offline_queue_fixture',
                    '-p', 'acquisition_requires_start:=false'])
    node = StaticEstimator()
    published = []
    node.estimate_pub = SimpleNamespace(publish=lambda msg: published.append(json.loads(msg.data)))
    node.marker_pub = SimpleNamespace(publish=lambda msg: None)
    try:
        node.get_clock().set_ros_time_override(Time(nanoseconds=2000000000))
        node.info = dict(k=[500., 0., 320., 0., 500., 240., 0., 0., 1.], d=[0.]*5,
                         width=640, height=480, frame_id='optical', distortion_model='plumb_bob')
        for child in ('optical', 'lidar'):
            transform = TransformStamped()
            transform.header.frame_id, transform.child_frame_id = 'world', child
            transform.transform.rotation.w = 1.
            node.tf.set_transform_static(transform, 'offline_queue_fixture')
        node.lidar_frame = 'lidar'
        node.tick()  # Initialize calibration before the frozen observation.
        old = dict(valid=True, stamp=1.99, position=[0., 0., 40.])
        node.last_result, node.last_reason = old, 'ready'
        packet = dict(stamp=2.004, source_stamp=2.004, frame_id='optical', width=640, height=480,
                      detections=[dict(bbox=[312., 232., 16., 16.], confidence=1.)])
        node.pending_detection = (packet, time.monotonic())
        node.clouds.append(dict(stamp=1.95, source_stamp=1.95, origin=[0., 0., 0.], deskewed=True,
            points=np.array([[x, y, 40.] for x in (-.4, 0., .4) for y in (-.4, 0., .4)]),
            motion=(np.zeros(3), np.zeros(3))))
        node.tick()
        assert node.pending_detection[0] is packet and node.last_result is old
        assert not published and node.counters['camera_clock_wait_ticks'] == 1
        node.get_clock().set_ros_time_override(Time(nanoseconds=2004000000))
        node.tick()
        assert node.pending_detection is None and node.last_result['valid']
        assert node.last_processed == 2.004 and published[-1]['valid']
        # A stalled clock must not keep an old coordinate or pending image
        # usable forever; simulation age and wall age retain their own guards.
        node.get_clock().set_ros_time_override(Time(nanoseconds=3000000000))
        node.pending_detection = (dict(packet, stamp=3.004), time.monotonic())
        node.tick()
        assert node.last_result is None and node.pending_detection is not None
        assert published[-1]['reason'] == 'camera_estimate_stale'
        node.pending_detection = (dict(packet, stamp=3.004), time.monotonic()-node.max_age-.01)
        node.tick()
        assert node.pending_detection is None and published[-1]['reason'] == 'camera_result_wall_stale'
        node.pending_detection = (dict(packet, stamp=3.2), time.monotonic())
        node.tick()
        assert node.pending_detection is None and published[-1]['reason'] == 'stale_or_invalid_camera_stamp'
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_lidar_search_only_commands_camera_and_requires_fresh_evidence():
    import json
    rclpy.init()
    node = StaticEstimator()
    commands, coordinates = [], []
    try:
        from rclpy.parameter import Parameter
        node.set_parameters([Parameter('calibration_valid', value=True)])
        node.acquisition_started = True
        node.command_pub = SimpleNamespace(publish=commands.append)
        node.estimate_pub = SimpleNamespace(publish=lambda msg: coordinates.append(json.loads(msg.data)))
        node.info = dict(frame_id='optical')
        node.gimbal = dict(yaw=0., pitch=0., stamp=2.)
        node.clouds.append(dict(stamp=1.95, origin=[.1, -.145, .855],
            points=np.array([[x, -39.4, z] for x in (-.5, 0., .5) for z in (7.5, 8., 8.5)])))
        node.lookup_matrix = lambda target, source, stamp: np.eye(4)
        node.search_gimbal(2.)
        assert len(commands) == 1 and not coordinates and node.last_result is None
        assert commands[0].position[0] < -1.5
        assert node.search_proposal['control_coordinate'] is False
        node.search_gimbal(2.01)
        assert len(commands) == 1  # bounded to 5Hz, independent of the 50Hz estimator tick.
        node.last_visual_stamp = 2.1
        node.gimbal['stamp'] = 2.3
        node.search_gimbal(2.3)
        assert len(commands) == 1  # fresh RGB always owns centering.
        node.last_visual_stamp = None
        node.search_gimbal(3.)
        assert len(commands) == 1  # stale gimbal / cloud cannot point a search.
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_acquisition_waits_for_service_even_with_visible_rgb_then_records_measured_slew():
    import json
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    rclpy.init()
    node = StaticEstimator()
    commands, estimates, statuses = [], [], []
    try:
        node.command_pub = SimpleNamespace(publish=commands.append)
        node.estimate_pub = SimpleNamespace(publish=lambda msg: estimates.append(json.loads(msg.data)))
        node.status_pub = SimpleNamespace(publish=lambda msg: statuses.append(json.loads(msg.data)))
        node.now = lambda: 2.
        node.info = dict(k=[500.,0.,320.,0.,500.,240.,0.,0.,1.], d=[0.]*5,
            width=640, height=480, frame_id='optical', distortion_model='plumb_bob')
        node.gimbal = dict(yaw=0., pitch=0., stamp=2.)
        packet=dict(stamp=2., frame_id='optical', width=640,height=480,
                    detections=[dict(bbox=[312.,232.,16.,16.], confidence=1.)])
        node.pending_detection = packet, time.monotonic()
        node.tick()
        assert not node.acquisition_started and node.pending_detection is None
        assert not estimates and not commands and node.last_result is None
        node.publish_status()
        assert statuses[-1]['acquisition']['stage'] == 'waiting_start'
        denied=node.start_acquisition(Trigger.Request(), Trigger.Response())
        assert not denied.success and denied.message == 'calibration_not_confirmed'
        node.set_parameters([Parameter('calibration_valid',value=True),
                             Parameter('calibration_source',value='unit_measured_fixture')])
        started=node.start_acquisition(Trigger.Request(), Trigger.Response())
        assert started.success and node.acquisition_stage == 'lidar_search'
        repeated=node.start_acquisition(Trigger.Request(), Trigger.Response())
        assert repeated.success and repeated.message == 'acquisition_already_started'
        node.clouds.append(dict(stamp=1.95,origin=[.1,-.145,.855],
            points=np.array([[x,-39.4,z]for x in(-.5,0.,.5)for z in(7.5,8.,8.5)])))
        node.lookup_matrix=lambda *args: np.eye(4)
        node.search_gimbal(2.)
        assert commands and node.acquisition_stage == 'yaw_slew'
        assert commands[0].position[1] == 0. and node.acquisition_motion is None
        state=JointState();state.header.stamp=Time(seconds=2.).to_msg()
        state.name=['gimbal_yaw_joint','gimbal_pitch_joint'];state.position=[-.1,0.]
        node.receive_gimbal(state)
        assert node.acquisition_stage == 'yaw_slew'
        assert node.acquisition_motion['measured_position'] == [-.1,0.]
        assert not estimates and node.last_result is None  # Search position is never a flight coordinate.
        # Seeing RGB while yaw is still moving cannot bypass the ordered phase.
        node.pending_detection=packet,time.monotonic()
        node.tick()
        assert node.pending_detection is None and not node.coarse_acquisition_complete
        assert len(commands) == 1 and not estimates
        target_yaw,target_pitch=node.coarse_gimbal_target
        state.position=[target_yaw,0.];node.receive_gimbal(state)
        node.advance_coarse_gimbal(2.)
        assert node.acquisition_stage == 'pitch_slew' and len(commands) == 2
        assert list(commands[1].position) == node.coarse_gimbal_target and target_pitch < 0.
        assert not node.coarse_acquisition_complete
        state.position=[target_yaw,target_pitch];node.receive_gimbal(state)
        node.advance_coarse_gimbal(2.)
        assert node.coarse_acquisition_complete and node.acquisition_stage == 'visual_centering'
        assert not estimates
        assert [e['stage']for e in node.acquisition_events] == [
            'waiting_start','lidar_search','yaw_slew','pitch_slew','visual_centering']
        reset=node.reset_acquisition(Trigger.Request(),Trigger.Response())
        assert reset.success and node.acquisition_stage == 'resetting' and not node.acquisition_started
        assert not estimates[-1]['valid'] and node.coarse_gimbal_target is None
        node.tick()
        assert list(commands[-1].position) == [0.,0.]
        denied=node.start_acquisition(Trigger.Request(),Trigger.Response())
        assert not denied.success and denied.message == 'returning_to_initial'
        state.position=[0.,0.];node.receive_gimbal(state);node.tick()
        assert node.acquisition_stage == 'waiting_start' and not node.restore_initial_gimbal
        assert node.start_acquisition(Trigger.Request(),Trigger.Response()).success
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_clock_reset_requires_new_acquisition_and_restores_only_with_fresh_joint_feedback():
    from sensor_msgs.msg import JointState
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true'])
    node = StaticEstimator()
    commands = []
    try:
        node.command_pub = SimpleNamespace(publish=commands.append)
        node.get_clock().set_ros_time_override(Time(seconds=20.))
        node.now()
        node.acquisition_started = True
        node.acquisition_initial_position = [0.,-.15]
        node.gimbal = dict(yaw=-1.5,pitch=-.18,stamp=20.)
        node.get_clock().set_ros_time_override(Time(seconds=1.))
        node.now()
        assert not node.acquisition_started and node.acquisition_stage == 'resetting'
        assert node.gimbal is None and node.restore_initial_gimbal
        node.tick()
        assert not commands
        state=JointState();state.header.stamp=Time(seconds=1.).to_msg()
        state.name=['gimbal_yaw_joint','gimbal_pitch_joint'];state.position=[-1.5,-.18]
        node.receive_gimbal(state)
        node.tick()
        assert len(commands) == 1 and list(commands[0].position) == [0.,-.15]
        assert node.restore_initial_gimbal and not node.acquisition_started
        node.tick()
        assert len(commands) == 1
        state.position=[0.,-.15];node.receive_gimbal(state);node.tick()
        assert not node.restore_initial_gimbal and node.acquisition_stage == 'waiting_start'
    finally:
        node.destroy_node()
        rclpy.shutdown()
