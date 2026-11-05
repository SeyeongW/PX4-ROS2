"""Camera I/O checks use synthetic images and fake capture, never RTSP hardware."""
import threading
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2
import rclpy
import yaml
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from balloon_locator.ros_utils import bgr_to_image, image_to_bgr
from balloon_locator.siyi_camera_node import SiyiCameraNode
from balloon_locator.camera_calibration import CameraCalibration


@pytest.mark.parametrize('encoding', ['bgr8', 'rgb8'])
def test_padded_image_rows_preserve_pixels(encoding):
    pixels=np.arange(18,dtype=np.uint8).reshape(2,3,3)
    rows=np.full((2,12),255,np.uint8);rows[:,:9]=pixels.reshape(2,9)
    message=Image(height=2,width=3,encoding=encoding,step=12,data=rows.tobytes())
    expected=pixels[:,:,::-1] if encoding=='rgb8' else pixels
    np.testing.assert_array_equal(image_to_bgr(message),expected)


@pytest.mark.parametrize('step,data_size', [(8,18),(10,18),(9,19)])
def test_malformed_image_layout_is_rejected(step,data_size):
    message=Image(height=2,width=3,encoding='bgr8',step=step,data=bytes(data_size))
    with pytest.raises(ValueError,match='image_layout'):
        image_to_bgr(message)


@pytest.mark.parametrize('frame', [np.zeros((2,3),np.uint8),np.zeros((2,3,3),np.float32)])
def test_encoder_rejects_non_bgr8(frame):
    with pytest.raises(ValueError,match='bgr8'):
        bgr_to_image(frame,Time(sec=1),'camera')


def test_stop_joins_capture_before_node_destruction():
    joined=[]
    fake=SimpleNamespace(_run=True,_stop_event=threading.Event(),
        _publish_lock=threading.Lock(),
        _thread=SimpleNamespace(join=lambda timeout:joined.append(timeout),is_alive=lambda:False),
        open_timeout=3.,read_timeout=1.)
    SiyiCameraNode.stop(fake)
    assert fake._stop_event.is_set() and joined


@pytest.fixture
def calibration_file():
    document=dict(image_width=640,image_height=480,distortion_model='plumb_bob',
        camera_matrix=dict(rows=3,cols=3,data=[400.,0.,320.,0.,400.,240.,0.,0.,1.]),
        distortion_coefficients=dict(rows=1,cols=5,data=[.1,0.,0.,0.,0.]),
        rectification_matrix=dict(rows=3,cols=3,data=np.eye(3).ravel().tolist()),
        projection_matrix=dict(rows=3,cols=4,data=[400.,0.,320.,0.,0.,400.,240.,0.,0.,0.,1.,0.]))
    with tempfile.TemporaryDirectory(prefix='balloon_camera_test_') as folder:
        path=Path(folder)/'calibration.yaml'
        path.write_text(yaml.safe_dump(document))
        yield path,document


def test_measured_radial_distortion_is_rectified_with_matching_metadata(calibration_file):
    path,_=calibration_file
    calibration=CameraCalibration(path)
    # Synthetic pinhole point (520,340) distorted analytically by k1=0.1.
    x,y=.5,.25;radial=1.+.1*(x*x+y*y)
    frame=np.zeros((480,640,3),np.uint8)
    cv2.circle(frame,(round(320.+400.*x*radial),round(240.+400.*y*radial)),3,(255,255,255),-1)
    rectified=calibration.rectify(frame)
    rows,cols=np.nonzero(rectified[:,:,0]>127)
    assert abs(cols.mean()-520.)<1. and abs(rows.mean()-340.)<1.
    info=calibration.message(Time(sec=7),'camera_optical')
    assert info.header.stamp.sec==7 and info.header.frame_id=='camera_optical'
    assert list(info.d)==[0.]*5 and info.k[0]==400. and info.width==640


@pytest.mark.parametrize('fault',['nonfinite','stereo','rotation','model','wrong_shape'])
def test_invalid_or_unsupported_calibration_is_rejected(calibration_file,fault):
    path,document=calibration_file
    if fault=='nonfinite':document['camera_matrix']['data'][0]=float('nan')
    elif fault=='stereo':document['projection_matrix']['data'][3]=1.
    elif fault=='rotation':document['rectification_matrix']['data'][0]=0.
    elif fault=='model':document['distortion_model']='equidistant'
    else:document['camera_matrix']['cols']=4
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):CameraCalibration(path)


def test_calibration_does_not_silently_resize_images(calibration_file):
    calibration=CameraCalibration(calibration_file[0])
    with pytest.raises(ValueError,match='resolution'):
        calibration.rectify(np.zeros((240,320,3),np.uint8))


@pytest.fixture
def camera(monkeypatch):
    captures=[]
    class Capture:
        def __init__(self,*args):
            self.args=args;self.released=False;self.reads=0
            captures.append(self)
        def isOpened(self):return True
        def read(self):
            self.reads+=1
            if self.reads<=3:
                return True,np.full((2,3,3),self.reads,np.uint8)
            time.sleep(.01)
            return False,None
        def release(self):self.released=True
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    rclpy.init(args=['--ros-args','-p','open_timeout_sec:=0.1','-p','read_timeout_sec:=0.1'])
    node=SiyiCameraNode();images=[];infos=[]
    node.pub=SimpleNamespace(publish=images.append)
    node.info_pub=SimpleNamespace(publish=infos.append)
    deadline=time.monotonic()+1.
    while (not captures or captures[0].reads<4) and time.monotonic()<deadline:time.sleep(.001)
    try:yield node,images,infos,captures
    finally:node.destroy_node();rclpy.shutdown()


def test_reader_uses_timeouts_and_only_latest_frame_then_joins(camera):
    node,images,_,captures=camera
    deadline=time.monotonic()+1.
    while captures[0].reads<4 and time.monotonic()<deadline:time.sleep(.001)
    assert captures[0].args[1:]==(cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,100,cv2.CAP_PROP_READ_TIMEOUT_MSEC,100])
    node._publish_latest()
    assert len(images)==1 and set(images[0].data)=={3}
    node.stop()
    node._publish_latest()
    assert not node._thread.is_alive() and captures[0].released and len(images)==1


@pytest.mark.parametrize('condition',['negative_stamp','stale_frame','future_frame'])
def test_invalid_capture_time_cannot_be_relabelled_as_current(camera,condition):
    node,images,_,_=camera
    now=node.get_clock().now().nanoseconds
    received=now;wall=time.monotonic()
    if condition=='negative_stamp':received=1;node.latency=1.
    elif condition=='stale_frame':wall-=1.
    else:received+=int(10e9)
    with node._frame_lock:node._latest=(np.zeros((2,3,3),np.uint8),received,wall,node._zoom_generation)
    node._publish_latest()
    assert not images


def test_calibrated_camera_gates_zoom_and_publishes_same_exposure_header(camera,calibration_file):
    node,images,infos,_=camera
    node.calibration=CameraCalibration(calibration_file[0])
    frame=np.zeros((480,640,3),np.uint8)
    def publish():
        with node._frame_lock:
            node._latest=(frame,node.get_clock().now().nanoseconds,time.monotonic(),node._zoom_generation)
        node._publish_latest()
    publish();assert not images  # No zoom evidence.
    node._on_zoom(Float32(data=2.));publish();assert not images
    node._on_zoom(Float32(data=1.));node.zoom_received-=3.;publish();assert not images
    node._on_zoom(Float32(data=1.));publish()
    assert len(images)==len(infos)==1
    assert images[0].header==infos[0].header and images[0].width==infos[0].width


def test_stop_waits_for_inflight_publish_before_destruction(camera):
    node,_,_,_=camera
    publishing=threading.Event();allow_publish=threading.Event();stopped=threading.Event()
    def publish(_):
        publishing.set()
        assert allow_publish.wait(1.)
    node.pub=SimpleNamespace(publish=publish)
    worker=threading.Thread(target=node._publish_latest)
    worker.start();assert publishing.wait(1.)
    stopper=threading.Thread(target=lambda:(node.stop(),stopped.set()))
    stopper.start()
    try:
        assert not stopped.wait(.03)
    finally:
        allow_publish.set();worker.join(1.);stopper.join(1.)
    assert stopped.is_set() and not worker.is_alive() and not stopper.is_alive()


@pytest.mark.parametrize('transition',['different_zoom','reacquired_zoom'])
def test_zoom_transition_cannot_relabel_a_pending_old_frame(camera,calibration_file,transition):
    node,images,infos,_=camera
    node.calibration=CameraCalibration(calibration_file[0])
    node._on_zoom(Float32(data=2. if transition=='different_zoom' else 1.))
    frame=np.zeros((480,640,3),np.uint8)
    with node._frame_lock:
        node._latest=(frame,node.get_clock().now().nanoseconds,time.monotonic(),node._zoom_generation)
    if transition=='reacquired_zoom':node.zoom_received-=3.
    node._on_zoom(Float32(data=1.))
    node._publish_latest()
    assert not images and not infos


def test_zoom_change_during_rectification_invalidates_inflight_frame(camera):
    node,images,infos,_=camera
    node._on_zoom(Float32(data=1.))
    def rectify(frame):
        node._on_zoom(Float32(data=2.))
        node._on_zoom(Float32(data=1.))
        return frame
    node.calibration=SimpleNamespace(rectify=rectify)
    with node._frame_lock:
        node._latest=(np.zeros((2,3,3),np.uint8),node.get_clock().now().nanoseconds,
                      time.monotonic(),node._zoom_generation)
    node._publish_latest()
    assert not images and not infos


def test_capture_spanning_zoom_change_is_discarded(camera,monkeypatch):
    node,_,_,_=camera
    node.stop();node._stop_event.clear()
    node.calibration=object();node._on_zoom(Float32(data=1.))
    class ChangingCapture:
        reads=0
        def isOpened(self):return True
        def read(self):
            self.reads+=1
            if self.reads==1:
                node._on_zoom(Float32(data=2.));node._on_zoom(Float32(data=1.))
                return True,np.zeros((2,3,3),np.uint8)
            node._stop_event.set()
            return False,None
        def release(self):pass
    monkeypatch.setattr(cv2,'VideoCapture',lambda *args:ChangingCapture())
    node._loop()
    assert node._latest is None
