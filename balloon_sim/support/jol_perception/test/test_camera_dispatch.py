"""Real ROS executor wakeup with frozen/reset simulated time; no Gazebo or NN."""
import json
import threading
import time

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String

from jol_perception import camera_fusion


def assert_pipeline_balance(node):
    status = node.pipeline_status()
    c = status['counters']
    assert c['image_received'] == c['input_accepted']+c['input_rejected']
    assert c['input_accepted'] == c['inference_submitted']+c['queue_dropped']+int(status['pending'])
    assert c['inference_submitted'] == (c['inference_completed']+c['inference_failed']
                                      +c['result_discarded']+int(status['inflight']))
    assert c['inference_completed'] == c['result_published']+c['publication_failed']
    assert c['detections_published'] >= c['result_published']


@pytest.mark.parametrize('reliable', [False, True])
def test_image_subscription_reliability_is_explicit_and_keeps_bounded_depth(monkeypatch, reliable):
    from rclpy.qos import ReliabilityPolicy
    class Network:
        model, providers = 'qos_fixture', ['CPUExecutionProvider']
        def __init__(self, *args):
            pass
    monkeypatch.setattr(camera_fusion, 'DroneNetwork', Network)
    rclpy.init(args=['--ros-args', '-p', 'camera_reliable:='+str(reliable).lower()])
    node = camera_fusion.CameraFusion()
    try:
        qos = node.image_subscription.qos_profile
        assert qos.depth == 5
        assert qos.reliability == (ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT)
        assert node.pipeline_status()['input_qos']['reliability'] == ('reliable' if reliable else 'best_effort')
    finally:
        node.worker.shutdown(wait=True, cancel_futures=True)
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('simulation,override,expected', [
    ('true', None, 'true'), ('false', None, 'false'),
    ('true', 'false', 'false'), ('false', 'true', 'true')])
def test_launch_selects_reliable_only_for_simulation_unless_explicitly_overridden(monkeypatch, simulation, override, expected):
    """Resolve real launch arguments only; do not execute any process action."""
    import importlib.util
    from pathlib import Path
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location('camera_qos_launch_fixture', root/'src/jol_perception/launch/monitor.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'get_package_share_directory', lambda package: str(root/'src'/package))
    context = LaunchContext()
    context.launch_configurations['simulation'] = simulation
    if override is not None:
        context.launch_configurations['camera_reliable'] = override
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    assert context.launch_configurations['camera_reliable'] == expected


@pytest.fixture
def health_camera(monkeypatch):
    """Exercise executor/publication state with a cheap, deterministic NN fixture."""
    import numpy as np
    class Network:
        model, providers = 'health_fixture', ['CPUExecutionProvider']
        def __init__(self, *args):
            self.fail = False
        def infer(self, image, regions):
            if self.fail:
                raise RuntimeError('fixture inference failure')
            return []
    monkeypatch.setattr(camera_fusion, 'DroneNetwork', Network)
    rclpy.init()
    node = camera_fusion.CameraFusion()
    node.set_parameters([Parameter('use_sim_time', value=True), Parameter('calibration_valid', value=True)])
    node.get_clock().set_ros_time_override(Time(seconds=10.))
    info = CameraInfo()
    info.header.frame_id, info.width, info.height = 'camera', 64, 48
    info.k, info.distortion_model = [50., 0., 32., 0., 50., 24., 0., 0., 1.], 'plumb_bob'
    node.receive_info(info)
    monkeypatch.setattr(node, 'geometry', lambda msg: ([], np.eye(4), dict(stamp=10., frame_id='world'), 'ready'))
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    outputs = {'status': [], 'fused': []}
    for name in outputs:
        monkeypatch.setattr(getattr(node, name+'_pub'), 'publish',
                            lambda msg, key=name: outputs[key].append(json.loads(msg.data)))
    def image(stamp):
        msg = Image()
        msg.header.frame_id = 'camera'
        msg.header.stamp = Time(nanoseconds=round(stamp*1e9)).to_msg()
        msg.width, msg.height, msg.step, msg.encoding = 64, 48, 192, 'bgr8'
        msg.data = bytes(64*48*3)
        node.receive_image(msg)
    def until(predicate):
        deadline = time.monotonic()+1.2
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.005)
        assert predicate()
    try:
        image(10.)
        until(lambda: node.processed == 1)
        yield node, outputs, info, image, until
    finally:
        node.worker.shutdown(wait=True, cancel_futures=True)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def test_status_heartbeat_and_expiry_continue_when_clock_and_camera_stop(health_camera):
    node, outputs, _, _, until = health_camera
    node.publish_status()
    assert outputs['status'][-1]['ready'] is True
    node.last_image_wall_ns = time.monotonic_ns()-2_000_000_000
    count = len(outputs['status'])
    until(lambda: len(outputs['status']) > count)  # Actual steady-clock timer, no /clock progress.
    status = outputs['status'][-1]
    assert status['ready'] is False and status['fusion_ready'] is False
    assert status['health_reason'] == 'camera_stream_stale'
    assert status['stamp'] == 10. and status['result_age_s'] == 0.
    assert node.fusion_seed is None
    assert outputs['fused'][-1]['publication_kind'] == 'invalidation'


def test_pipeline_distinguishes_received_rejected_and_published(health_camera):
    node, _, _, image, _ = health_camera
    for stamp in (0., 10., 11.):
        image(stamp)
    node.receive_image(Image())  # Missing frame takes precedence over invalid time.
    counters = node.pipeline_status()['counters']
    assert counters['image_received'] == 5 and counters['input_accepted'] == 1
    assert counters['input_rejected'] == 4 and counters['result_published'] == 1
    for reason in ('nonpositive_stamp', 'nonincreasing_stamp', 'future_stamp', 'missing_frame'):
        assert counters['input_rejected_'+reason] == 1
    assert_pipeline_balance(node)


def test_pipeline_tracks_clock_wait_worker_replacement_and_epoch_discard(health_camera, monkeypatch):
    node, _, _, image, until = health_camera
    release = threading.Event()
    def blocked(image, regions):
        assert release.wait(1.)
        return []
    monkeypatch.setattr(node.network, 'infer', blocked)
    try:
        node.get_clock().set_ros_time_override(Time(seconds=10.1))
        image(10.15)
        node.tick()
        assert_pipeline_balance(node)
        image(10.16)
        assert node.pipeline.counters['queue_dropped_clock_wait'] == 1
        node.get_clock().set_ros_time_override(Time(seconds=10.2))
        node.tick()
        assert node.future is not None
        image(10.17)
        image(10.18)
        assert node.pipeline.counters['queue_dropped_worker_inflight'] == 1
        assert_pipeline_balance(node)
        node.get_clock().set_ros_time_override(Time(seconds=2.))
        node.check_clock()
        assert node.pipeline.counters['queue_dropped_clock_reset'] == 1
        release.set()
        until(lambda: node.future is None)
        assert node.pipeline.counters['result_discarded_clock_reset'] == 1
        image(2.)
        until(lambda: node.processed == 2)
        assert_pipeline_balance(node)
        assert len(node.pipeline.events) <= 64
    finally:
        release.set()


def test_inference_failure_revokes_previous_fusion_and_next_frame_recovers(health_camera):
    node, outputs, _, image, until = health_camera
    prior = outputs['fused'][-1]
    node.network.fail = True
    node.get_clock().set_ros_time_override(Time(seconds=10.1))
    image(10.1)
    until(lambda: node.failures == 1)
    assert node.fusion_seed is None
    invalid = outputs['fused'][-1]
    assert invalid['tracks'] == [] and invalid['publication_kind'] == 'invalidation'
    assert invalid['stamp'] == prior['stamp'] and invalid['fusion_revision'] > prior['fusion_revision']
    node.publish_status()
    assert outputs['status'][-1]['ready'] is False
    assert node.pipeline.counters['inference_failed'] == 1
    assert_pipeline_balance(node)
    node.network.fail = False
    node.get_clock().set_ros_time_override(Time(seconds=10.2))
    image(10.2)
    until(lambda: node.processed == 2)
    node.publish_status()
    assert outputs['status'][-1]['ready'] is True and node.fusion_seed is not None
    assert_pipeline_balance(node)


def test_pipeline_separates_successful_inference_from_publication_failure(health_camera, monkeypatch):
    node, _, _, image, until = health_camera
    def fail(*args):
        raise ValueError('fixture publication failure')
    monkeypatch.setattr(node, 'publish_result', fail)
    node.get_clock().set_ros_time_override(Time(seconds=10.1))
    image(10.1)
    until(lambda: node.failures == 1)
    c = node.pipeline.counters
    assert c['inference_completed'] == 2 and c['inference_failed'] == 0
    assert c['publication_failed'] == 1 and c['result_published'] == 1
    assert_pipeline_balance(node)


def test_changed_intrinsics_revoke_seed_and_pending_result_not_reinterpreted(health_camera):
    from concurrent.futures import Future
    import numpy as np
    node, outputs, info, image, _ = health_camera
    node.receive_info(info)  # Repeated CameraInfo must not invalidate evidence.
    assert node.fusion_seed is not None
    node.get_clock().set_ros_time_override(Time(seconds=10.1))
    image(10.1)
    context = dict(message=node.latest, regions=[], epoch=node.epoch, timing=node.latest_timing,
                   calibration_revision=getattr(node, 'calibration_revision', 0))
    node.latest = node.latest_timing = None
    future = Future()
    future.set_result((np.zeros((48, 64, 3), np.uint8), [], 1.))
    node.future = future, context
    info.k[0] = 60.
    node.receive_info(info)
    assert node.fusion_seed is None
    node.tick()
    assert outputs['fused'][-1]['fusion_reason'] == 'calibration_changed'
    assert node.fusion_seed is None and node.processed == 2


def test_calibration_parameter_change_without_new_image_revokes_seed(health_camera):
    node, outputs, _, _, _ = health_camera
    node.set_parameters([Parameter('calibration_valid', value=False)])
    node.tick()
    assert node.fusion_seed is None
    assert outputs['fused'][-1]['publication_kind'] == 'invalidation'
    node.publish_status()
    assert outputs['status'][-1]['ready'] is True  # Actual NN readiness is separate from calibration.
    assert outputs['status'][-1]['fusion_ready'] is False


def test_queued_image_keeps_receipt_calibration_revision(health_camera):
    node, outputs, info, image, until = health_camera
    node.get_clock().set_ros_time_override(Time(seconds=10.1))
    image(10.1)  # Queued before the worker starts, not just before it completes.
    info.k[0] = 60.
    node.receive_info(info)
    until(lambda: node.processed == 2)
    result = outputs['fused'][-1]
    assert result['fusion_reason'] == 'calibration_changed'
    assert result['calibration_revision'] > result['image_calibration_revision']


def test_invalidation_replaces_controller_cache_and_returns_to_lidar(health_camera):
    from types import SimpleNamespace
    import numpy as np
    from jol_perception.fusion_core import fuse, project_tracks
    from jol_perception.pointing_node import Pointing
    from jol_perception.pointing_core import choose_turret_state
    node, outputs, _, _, _ = health_camera
    track = dict(id=1, position=[0., 0., 10.], velocity=[0., 0., 0.], size=[.6, .6, .4],
                 state_covariance=(np.eye(6)*.4).tolist(), last_seen=10., observed=True,
                 prediction=dict(horizon_s=.5, acceleration_std=2.))
    projections = project_tracks(dict(stamp=10., frame_id='world', tracks=[track]),
                                 10., np.eye(4), node.info)[0]
    posterior = fuse(projections, [dict(bbox=projections[0]['bbox'], class_name='fixed-wing', confidence=.9)],
                     np.eye(4), node.info)
    seed = dict(node.fusion_seed, tracks=posterior, stabilized=True, fusion_revision=3)
    node.fusion_seed = seed
    controller = SimpleNamespace(auto_acquisition_enabled=True, fusion_min_stamp=float('-inf'),
                                 fused_packet=None, get_clock=node.get_clock, get_logger=node.get_logger)
    Pointing.receive_fusion(controller, String(data=json.dumps(seed)))
    assert choose_turret_state(track, 10., controller.fused_packet, 10.)[2]['control_state_source'] == 'camera_lidar'
    node.invalidate_fusion('camera_stream_stale')
    invalid = outputs['fused'][-1]
    Pointing.receive_fusion(controller, String(data=json.dumps(invalid)))
    assert controller.fused_packet['fusion_revision'] == 4
    selected, _, detail = choose_turret_state(track, 10., controller.fused_packet, 10.)
    assert selected is track and detail['control_state_source'] == 'lidar'
    assert detail['fusion_selection_reason'] == 'fusion_geometry_not_ready'
    Pointing.receive_fusion(controller, String(data=json.dumps(seed)))
    assert controller.fused_packet == invalid  # A queued older branch cannot restore it.


def test_malformed_tracks_are_rejected_before_entering_replay_history():
    from collections import deque
    from types import SimpleNamespace
    warnings = []
    node = SimpleNamespace(check_clock=lambda: 10., tolerance=.15, history=deque(maxlen=60),
                           fusion_seed=None,
                           get_logger=lambda: SimpleNamespace(warning=lambda text, **kwargs: warnings.append(text)))
    track = dict(id=1, observed=True, position=[0., 0., 10.], velocity=[0., 0., 0.], size=[1., 1., 1.])
    malformed = [None, dict(track, prediction=[]), dict(track, observed='false'),
                 dict(track, id=[]), dict(track, position=[float('nan'), 0., 10.])]
    for invalid in malformed:
        packet = dict(stamp=10., frame_id='world', tracks=[invalid])
        camera_fusion.CameraFusion.receive_tracks(node, String(data=json.dumps(packet)))
        assert not node.history
    camera_fusion.CameraFusion.receive_tracks(node, String(data=json.dumps(dict(stamp=10., frame_id='world', tracks=[track]))))
    assert len(node.history) == 1 and len(warnings) == len(malformed)


def test_image_waits_for_delayed_exact_tf_and_times_out_without_relaxing_gap(monkeypatch):
    class Network:
        model, providers = 'empty_actual_inference_fixture', ['CPUExecutionProvider']
        def __init__(self, *args):
            self.regions = []
        def infer(self, image, regions):
            self.regions.append(regions)
            return []
    monkeypatch.setattr(camera_fusion, 'DroneNetwork', Network)
    rclpy.init()
    node = camera_fusion.CameraFusion()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    fused = []
    publish = node.fused_pub.publish
    monkeypatch.setattr(node.fused_pub, 'publish',
                        lambda message: fused.append(json.loads(message.data)) or publish(message))
    def until(predicate):
        deadline = time.monotonic()+1.
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.005)
        assert predicate()
    def transform(stamp):
        msg = TransformStamped()
        msg.header.frame_id, msg.child_frame_id = 'world', 'camera'
        msg.header.stamp = Time(nanoseconds=round(stamp*1e9)).to_msg()
        msg.transform.rotation.w = 1.
        node.tf.set_transform(msg, 'late_tf_fixture')
    def image(stamp):
        msg = Image()
        msg.header.frame_id = 'camera'
        msg.header.stamp = Time(nanoseconds=round(stamp*1e9)).to_msg()
        msg.width, msg.height, msg.step, msg.encoding = 640, 480, 1920, 'bgr8'
        msg.data = bytes(640*480*3)
        return msg
    def tracks(stamp):
        node.receive_tracks(String(data=json.dumps(dict(stamp=stamp, frame_id='world', tracks=[
            dict(id=1, observed=True, confirmed=True, position=[0., 0., 10.],
                 velocity=[0., 0., 0.], size=[.2, .2, .2])]))))
    try:
        node.set_parameters([Parameter('use_sim_time', value=True), Parameter('calibration_valid', value=True)])
        node.get_clock().set_ros_time_override(Time(seconds=10.01))
        node.info = dict(k=[500., 0., 320., 0., 500., 240., 0., 0., 1.], d=[0.]*5,
                         width=640, height=480, frame_id='camera', distortion_model='plumb_bob')
        tracks(10.)
        transform(10.)
        node.receive_image(image(10.01))
        node.tick()
        assert node.latest is not None and node.future is None and node.processed == 0
        assert node.geometry_wait_frames == 1
        transform(10.02)  # actual later bracket arrives through the same TF buffer
        until(lambda: node.processed == 1)
        assert fused[0]['fusion_reason'] == 'ready'
        assert node.network.regions[0]  # ROI could be constructed before inference
        assert node.tf.last_support['camera']['gap_s'] == .02
        assert node.geometry_wait_timeouts == 0

        node.get_clock().set_ros_time_override(Time(seconds=10.11))
        tracks(10.1)
        transform(10.2)  # .18 second gap is still invalid; never accept it after waiting
        node.receive_image(image(10.1))
        until(lambda: node.processed == 2)
        latest = [packet for packet in fused if packet['publication_kind'] == 'exposure_result'][-1]
        assert latest['fusion_reason'] == 'tf_or_geometry_unavailable' and latest['tracks'] == []
        assert node.geometry_wait_timeouts == 1
        assert 'interpolation gap' in node.geometry_error
    finally:
        node.worker.shutdown(wait=True, cancel_futures=True)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def test_continuous_tf_outage_does_not_starve_replaced_images_and_recovers(health_camera, monkeypatch):
    """15Hz replacements over 1s must not renew the 100ms TF wait indefinitely."""
    from types import SimpleNamespace
    import numpy as np
    node, outputs, _, image, until = health_camera
    node.geometry = camera_fusion.CameraFusion.geometry.__get__(node)
    wall = [time.monotonic_ns()]
    monkeypatch.setattr(camera_fusion, 'time', SimpleNamespace(
        monotonic_ns=lambda: wall[0], perf_counter=time.perf_counter))
    detections, regions = [], []
    monkeypatch.setattr(node.detection_pub, 'publish', lambda msg: detections.append(json.loads(msg.data)))
    def infer(pixels, rois):
        regions.append(rois)
        return [dict(bbox=[29., 21., 6., 6.], confidence=.9, class_name='quadcopter')]
    monkeypatch.setattr(node.network, 'infer', infer)
    def tracks(stamp):
        node.receive_tracks(String(data=json.dumps(dict(stamp=stamp, frame_id='world', tracks=[
            dict(id=1, observed=True, confirmed=True, position=[0., 0., 10.],
                 velocity=[0., 0., 0.], size=[.6, .6, .6],
                 state_covariance=(np.eye(6)*.4).tolist(),
                 last_seen=stamp, prediction=dict(horizon_s=.5, acceleration_std=2.))]))))
    def transform(stamp):
        msg = TransformStamped()
        msg.header.frame_id, msg.child_frame_id = 'world', 'camera'
        msg.header.stamp = Time(nanoseconds=round(stamp*1e9)).to_msg()
        msg.transform.rotation.w = 1.
        node.tf.set_transform(msg, 'continuous_outage_fixture')
    transform(10.)
    initial_processed = node.processed
    initial_wall = wall[0]
    for frame in range(1, 17):
        stamp = 10.+frame/15.
        wall[0] = initial_wall+round(frame/15.*1e9)
        node.get_clock().set_ros_time_override(Time(nanoseconds=round(stamp*1e9)))
        tracks(stamp)
        image(stamp)
        node.tick()
        if node.future is not None:
            until(lambda: node.future is None)
        assert_pipeline_balance(node)
    assert node.processed-initial_processed >= 12
    assert detections and all(packet['detections'] for packet in detections)
    assert all(not roi for roi in regions)
    unavailable = [p for p in outputs['fused'] if p['publication_kind'] == 'exposure_result'
                   and p['stamp'] > 10.]
    assert len(unavailable) == len(detections)
    assert all(p['fusion_reason'] == 'tf_or_geometry_unavailable' and not p['tracks'] for p in unavailable)
    assert node.fusion_seed is None
    node.publish_status()
    assert outputs['status'][-1]['inference_ready'] and not outputs['status'][-1]['fusion_ready']

    stamp = 10.+17/15.
    wall[0] = initial_wall+round(17/15.*1e9)
    node.get_clock().set_ros_time_override(Time(nanoseconds=round(stamp*1e9)))
    tracks(stamp)
    transform(stamp)
    image(stamp)
    processed = node.processed
    until(lambda: node.processed == processed+1)
    assert regions[-1]  # Valid exact TF restores LiDAR ROI before inference.
    recovered = [p for p in outputs['fused'] if p['publication_kind'] == 'exposure_result'][-1]
    assert recovered['fusion_reason'] == 'ready' and recovered['tracks']
    assert node.fusion_seed is not None
    assert_pipeline_balance(node)

    # A separate outage after recovery gets its own short wait, preserving ROI
    # opportunity rather than retaining the exhausted old failure budget.
    stamp = 10.+18/15.
    wall[0] = initial_wall+round(18/15.*1e9)
    node.get_clock().set_ros_time_override(Time(nanoseconds=round(stamp*1e9)))
    tracks(stamp)
    image(stamp)
    node.tick()
    assert node.latest is not None and node.future is None
    assert node.tf_wait_budget_start['monotonic_ns'] == wall[0]
    assert_pipeline_balance(node)


@pytest.mark.parametrize('reset', ['calibration', 'clock'])
def test_tf_wait_budget_restarts_after_calibration_or_clock_reset(health_camera, monkeypatch, reset):
    from types import SimpleNamespace
    node, _, info, image, until = health_camera
    wall = [time.monotonic_ns()]
    monkeypatch.setattr(camera_fusion, 'time', SimpleNamespace(
        monotonic_ns=lambda: wall[0], perf_counter=time.perf_counter))
    monkeypatch.setattr(node, 'geometry', lambda msg: (
        [], None, dict(stamp=camera_fusion.stamp_seconds(msg.header.stamp), frame_id='world'),
        'tf_or_geometry_unavailable'))
    node.get_clock().set_ros_time_override(Time(seconds=10.1))
    image(10.1)
    node.tick()
    wall[0] += 200_000_000
    until(lambda: node.processed == 2)  # Exhausted budget permits full-frame NN.
    if reset == 'calibration':
        info.k[0] += 1.
        node.receive_info(info)
        stamp = 10.2
    else:
        stamp = 2.
    node.get_clock().set_ros_time_override(Time(seconds=stamp))
    image(stamp)
    node.tick()
    assert node.latest is not None and node.future is None
    assert node.tf_wait_budget_start['monotonic_ns'] == wall[0]
    assert_pipeline_balance(node)


def test_calibration_change_while_same_image_waits_gives_new_budget_but_not_new_revision(health_camera, monkeypatch):
    from types import SimpleNamespace
    node, outputs, info, image, until = health_camera
    wall = [time.monotonic_ns()]
    monkeypatch.setattr(camera_fusion, 'time', SimpleNamespace(
        monotonic_ns=lambda: wall[0], perf_counter=time.perf_counter))
    monkeypatch.setattr(node, 'geometry', lambda msg: (
        [], None, dict(stamp=10.1, frame_id='world'), 'tf_or_geometry_unavailable'))
    node.get_clock().set_ros_time_override(Time(seconds=10.1))
    image(10.1)
    node.tick()
    wall[0] += 60_000_000
    info.k[0] += 1.
    node.receive_info(info)
    node.tick()
    wall[0] += 50_000_000
    node.tick()
    assert node.latest is not None and node.future is None  # New budget is only 50ms old.
    wall[0] += 51_000_000
    until(lambda: node.processed == 2)
    result = [p for p in outputs['fused'] if p['publication_kind'] == 'exposure_result'][-1]
    assert result['fusion_reason'] == 'calibration_changed' and not result['tracks']
    assert result['image_calibration_revision'] < result['calibration_revision']
    assert_pipeline_balance(node)


@pytest.mark.parametrize('executor_type', [SingleThreadedExecutor, MultiThreadedExecutor])
def test_worker_completion_publishes_during_clock_pause_and_discards_reset_epoch(monkeypatch, executor_type):
    release = threading.Event()
    release.set()

    class Network:
        model, providers = 'deterministic_fixture', ['CPUExecutionProvider']

        def __init__(self, *args):
            pass

        def infer(self, image, regions):
            if not release.wait(2.):
                raise RuntimeError('Test worker release timeout')
            return []

    monkeypatch.setattr(camera_fusion, 'DroneNetwork', Network)
    rclpy.init()
    node = camera_fusion.CameraFusion()
    executor = executor_type() if executor_type is SingleThreadedExecutor else executor_type(num_threads=2)
    executor.add_node(node)
    published = []
    original = node.detection_pub.publish

    def capture(message):
        published.append(json.loads(message.data))
        original(message)

    monkeypatch.setattr(node.detection_pub, 'publish', capture)

    def until(predicate):
        deadline = time.monotonic()+2.
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.01)
        assert predicate()

    def image(stamp):
        message = Image()
        message.header.stamp = Time(nanoseconds=round(stamp*1e9)).to_msg()
        message.header.frame_id = 'fixture_camera'
        message.width, message.height, message.step, message.encoding = 64, 48, 192, 'bgr8'
        message.data = bytes(64*48*3)
        return message

    try:
        node.set_parameters([Parameter('use_sim_time', value=True)])
        node.get_clock().set_ros_time_override(Time(seconds=10.))
        node.receive_image(image(10.))
        # /clock remains at 10.0 forever: neither completion nor next submit may
        # depend on a ROS-time timer becoming ready at 10.1.
        until(lambda: node.processed == 1)
        assert node.get_clock().now().nanoseconds == 10_000_000_000
        assert published[0]['stamp'] == 10.
        timing = published[0]['timing']
        stages = ['image_receive', 'worker_submit', 'worker_start', 'preprocess_end',
                  'worker_end', 'completion_callback', 'fusion_end', 'publish']
        assert [timing[key]['monotonic_ns'] for key in stages] == sorted(timing[key]['monotonic_ns'] for key in stages)
        assert all(timing[key]['ros_s'] == 10. for key in stages)
        node.receive_image(image(10.))  # paused duplicate cannot create more work
        node.tick()
        assert node.future is None and node.latest is None

        release.clear()
        node.get_clock().set_ros_time_override(Time(seconds=11.))
        node.receive_image(image(11.))
        until(lambda: node.future is not None)
        old_epoch = node.epoch
        node.get_clock().set_ros_time_override(Time(seconds=2.))
        node.receive_image(image(2.))
        assert node.epoch == old_epoch+1
        release.set()
        until(lambda: node.processed == 2)
        assert [packet['stamp'] for packet in published] == [10., 2.]
        # A real sensor may be just ahead of a sparsely published /clock. Keep
        # that exposure until its time arrives instead of silently dropping it.
        node.receive_image(image(2.05))
        node.tick()
        assert node.latest is not None and node.future is None
        node.get_clock().set_ros_time_override(Time(seconds=2.1))
        until(lambda: node.processed == 3)
        assert published[-1]['stamp'] == 2.05
    finally:
        release.set()
        node.worker.shutdown(wait=True, cancel_futures=True)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
