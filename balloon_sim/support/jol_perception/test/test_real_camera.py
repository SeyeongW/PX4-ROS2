"""장치 오선택과 ROI 오류를 하드웨어 없이 검사한다."""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from launch import LaunchContext

from jol_perception import real_camera
from jol_perception.real_camera import RealCamera, require_external_camera, target_message


def test_camera_requires_explicit_stable_external_device(tmp_path, monkeypatch):
    dev, sys = tmp_path / 'dev/v4l', tmp_path / 'sys/class/video4linux'
    usb = tmp_path / 'sys/devices/usb/1-2'
    entry = usb / 'video4linux/video4'
    entry.mkdir(parents=True)
    sys.mkdir(parents=True)
    (dev / 'by-id').mkdir(parents=True)
    (sys / 'video4').symlink_to(entry)
    (entry / 'name').write_text('External camera')
    (usb / 'idVendor').write_text('1234')
    (usb / 'idProduct').write_text('5678')
    target = tmp_path / 'dev/video4'
    target.touch()
    camera = dev / 'by-id/external-video-index0'
    camera.symlink_to(target)
    monkeypatch.setattr(real_camera, 'V4L_DEV', dev)
    monkeypatch.setattr(real_camera, 'V4L_SYS', sys)
    monkeypatch.setattr(Path, 'is_char_device', lambda path: path.exists())
    for invalid in ('', '/dev/video0', str(dev / 'by-id/missing')):
        with pytest.raises(ValueError, match='현재 영상 장치'):
            require_external_camera(invalid)
    assert require_external_camera(str(camera)) == str(camera)
    (usb / 'idVendor').write_text('30c9')
    (usb / 'idProduct').write_text('00d9')
    with pytest.raises(ValueError, match='노트북 내장 HP'):
        require_external_camera(str(camera))
    (usb / 'idVendor').write_text('1234')
    (entry / 'name').write_text('HP True Vision FHD Camera: HP I')
    with pytest.raises(ValueError, match='노트북 내장 HP'):
        require_external_camera(str(camera))


@pytest.mark.skip(reason='Legacy real-hardware launcher is not part of this camera-only simulation bundle')
def test_launch_rejects_missing_camera_before_creating_any_process(monkeypatch):
    launch_file = Path(__file__).parents[1] / 'launch/real_hardware.launch.py'
    generate = runpy.run_path(str(launch_file))['generate_launch_description']
    created = []
    monkeypatch.setitem(generate.__globals__, 'ExecuteProcess', lambda **kw: created.append(kw))
    context = LaunchContext()
    context.launch_configurations['device'] = ''
    with pytest.raises(ValueError, match='고정 장치 경로'):
        generate().entities[-1].execute(context)
    assert not created


def test_target_geometry_and_invalid_regions_never_produce_valid_commands():
    center = target_message(1., 640, 480, [280, 200, 80, 80], selection_id=3)
    assert center['valid'] and center['error_normalized'] == [0., 0.]
    assert center['selection_id'] == 3 and center['source'] == 'manual_roi_csrt'
    corner = target_message(1., 640, 480, [0, 0, 64, 48])
    assert corner['valid'] and corner['error_normalized'] == [-.9, -.9]
    for bbox in (None, [], [1, 2, 3], [-1, 0, 10, 10], [0, 0, 0, 10],
                 [0, 0, 2, 1], [630, 0, 20, 10], [0, 470, 10, 20],
                 [float('nan'), 0, 10, 10], [0, 0, float('inf'), 10]):
        target = target_message(1., 640, 480, bbox)
        assert target['valid'] is False
        assert target['bbox'] is None and target['error_normalized'] is None
        json.dumps(target, allow_nan=False)
    assert target_message(1., 0, 480, [0, 0, 10, 10])['valid'] is False


def test_gtk_unsupported_visibility_keeps_camera_running_until_quit(monkeypatch):
    cleared = []
    node = SimpleNamespace(status={}, overlay=None, done=False, clear=lambda: cleared.append(True))
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: -1.)
    monkeypatch.setattr(cv2, 'waitKey', lambda *args: -1)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    RealCamera.show(node, frame, {'valid': False})
    assert not node.done and not cleared
    monkeypatch.setattr(cv2, 'waitKey', lambda *args: ord('q'))
    RealCamera.show(node, frame, {'valid': False})
    assert node.done and cleared == [True]
