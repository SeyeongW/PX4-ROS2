"""Offscreen passive views: concurrent feeds, raw fallback and honest freshness."""
import json
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
import rclpy
from PyQt5 import QtWidgets
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String

from balloon_locator.camera_display import camera_dashboard, camera_node
from balloon_locator.stl_runtime import aircraft_dashboard, aircraft_experiment_node
from jol_perception.experiment_ui import Dashboard, ExperimentNode


@pytest.fixture
def screen():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true', '-p', 'static_flight_mode:=true'])
    node = camera_node(aircraft_experiment_node(ExperimentNode))()
    node.get_clock().set_ros_time_override(Time(seconds=10))
    window = camera_dashboard(aircraft_dashboard(Dashboard))(node, start_timer=False)
    window.show()
    app.processEvents()
    try:
        yield node, window
    finally:
        window.close()
        node.destroy_node()
        rclpy.shutdown()


def frame(value=100, stamp=10):
    image = Image()
    image.header.stamp = Time(seconds=stamp).to_msg()
    image.width, image.height = 40, 30
    image.encoding, image.step = 'rgb8', 120
    image.data = bytes([value] * 3600)
    return image


def test_both_cameras_and_existing_controls_are_visible_without_lock(screen):
    node, window = screen
    node.receive_image(frame())
    raw = frame(200)
    node.receive_onboard_image('raw', raw)
    window.refresh()
    assert window.camera_label.pixmap() is not None
    assert window.onboard_panel.image.pixmap() is not None
    assert window.camera_label.isVisible() and window.onboard_panel.isVisible()
    assert window.flight_3d.isVisible()
    assert {'static_lock', 'static_preview', 'static_launch', 'static_abort'} <= window.static_buttons.keys()
    assert '원본 영상' in window.onboard_panel.summary.text()
    assert '검출 상태 미확인' in window.onboard_panel.summary.text()
    # Repainting retains the original message reference; callbacks have no copies/queues.
    window.refresh()
    assert window.onboard_panel.message is raw
    assert len(node.onboard_frames) == 1


@pytest.mark.parametrize('detected,label', [(False, '풍선 미검출'), (True, '풍선 후보 검출')])
def test_fresh_annotated_is_preferred_and_stale_annotation_falls_back_to_raw(screen, detected, label):
    node, window = screen
    raw, annotated = frame(100), frame(200)
    node.receive_onboard_image('raw', raw)
    node.receive_onboard_image('annotated', annotated)
    node.receive_onboard_observation(String(data=json.dumps(dict(
        observation_stamp=10., frame_fresh=True, detected=detected))))
    window.refresh()
    assert window.onboard_panel.message is annotated
    assert label in window.onboard_panel.summary.text()
    node.onboard_frames['annotated'] = (annotated, time.monotonic_ns() - 3_000_000_000)
    window.refresh()
    assert window.onboard_panel.message is raw
    assert '검출 상태 미확인' in window.onboard_panel.summary.text()


def test_live_raw_video_wins_when_annotation_processing_lags(screen):
    node, window = screen
    raw, annotated = frame(100), frame(200, stamp=9.5)
    node.receive_onboard_image('raw', raw)
    node.receive_onboard_image('annotated', annotated)
    window.refresh()
    assert window.onboard_panel.message is raw


@pytest.mark.parametrize('bad_status', [
    {'observation_stamp': 9., 'frame_fresh': True, 'detected': True},
    {'observation_stamp': 10., 'frame_fresh': False, 'detected': True},
    {'observation_stamp': True, 'frame_fresh': True, 'detected': True},
    [],
])
def test_unmatched_or_invalid_diagnostic_never_claims_a_current_detection(screen, bad_status):
    node, window = screen
    node.receive_onboard_image('annotated', frame())
    node.receive_onboard_observation(String(data=json.dumps(bad_status)))
    window.refresh()
    assert '검출 상태 미확인' in window.onboard_panel.summary.text()


def test_exposure_or_receipt_staleness_hides_both_frozen_images(screen):
    node, window = screen
    node.receive_image(frame())
    node.receive_onboard_image('raw', frame())
    window.refresh()
    # Paused ROS time must not keep a dead stream apparently live.
    node.received['image'] -= 3_000_000_000
    message, received = node.onboard_frames['raw']
    node.onboard_frames['raw'] = (message, received - 3_000_000_000)
    window.refresh()
    assert window.camera_label.pixmap() is None
    assert window.onboard_panel.image.pixmap() is None
    assert '과거' in window.onboard_panel.image.text()
    # Newly received old exposures remain stale too.
    node.receive_onboard_image('raw', frame(stamp=7))
    window.refresh()
    assert window.onboard_panel.image.pixmap() is None


def test_clock_reset_and_invalid_image_clear_previous_view(screen):
    node, window = screen
    node.receive_onboard_image('raw', frame())
    window.refresh()
    malformed = frame()
    malformed.step = 1
    node.receive_onboard_image('raw', malformed)
    window.refresh()
    assert window.onboard_panel.image.pixmap() is None
    assert '형식 오류' in window.onboard_panel.image.text()
    node.get_clock().set_ros_time_override(Time(seconds=1))
    node.check_clock()
    assert node.onboard_frames == {}
    window.refresh()
    assert '수신 대기' in window.onboard_panel.image.text()
