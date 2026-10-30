"""Real locator methods: measured exposure time, TF, calibration and expiry."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped,Vector3Stamped
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2D,Detection2DArray

from balloon_locator.balloon_locator_node import BalloonLocatorNode


@pytest.fixture
def locator():
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true','-p','pose_source:=tf',
        '-p','calibration_valid:=true','-p','diameter_rel_sigma:=0.0',
        '-p','focal_rel_sigma:=0.0','-p','pixel_sigma_px:=0.5'])
    node=BalloonLocatorNode();points=[];statuses=[]
    node.pub_pt=SimpleNamespace(publish=points.append)
    node.pub_est=SimpleNamespace(publish=lambda msg:None)
    node.pub_status=SimpleNamespace(publish=lambda msg:statuses.append(json.loads(msg.data)))
    info=CameraInfo();info.header.frame_id='optical';info.width=1280;info.height=960
    info.k=[6000.,0.,639.5,0.,6000.,479.5,0.,0.,1.];info.d=[0.]*5;info.distortion_model='plumb_bob'
    node._on_info(info)
    tf=TransformStamped();tf.header.frame_id='world';tf.child_frame_id='optical'
    tf.transform.rotation.w=1.;tf.transform.translation.x=1.;tf.transform.translation.z=2.
    node.tf.set_transform_static(tf,'fixture')
    try:yield node,points,statuses
    finally:node.destroy_node();rclpy.shutdown()


def feed(node,stamp):
    node.get_clock().set_ros_time_override(Time(nanoseconds=round(stamp*1e9)))
    msg=Detection2DArray();msg.header.frame_id='optical';msg.header.stamp=Time(nanoseconds=round(stamp*1e9)).to_msg()
    det=Detection2D();det.bbox.center.position.x=639.5;det.bbox.center.position.y=479.5
    det.bbox.size_x=det.bbox.size_y=45.;msg.detections=[det]
    node._on_det(msg);node._process_pending();node._publish()
    return msg


def test_exposure_tf_to_station_ned_and_expiration_keep_source_time(locator):
    node,points,statuses=locator
    for stamp in (2.,2.1,2.2):feed(node,stamp)
    assert statuses[-1]['valid']
    assert statuses[-1]['measurement_stamp']==pytest.approx(2.2)
    np.testing.assert_allclose(statuses[-1]['ned'],[0.,1.,-42.],atol=1e-6)
    assert points[-1].header.stamp==Time(nanoseconds=2200000000).to_msg()
    assert statuses[-1]['camera_centered'] and statuses[-1]['ground_truth_used'] is False
    count=len(points)
    node.get_clock().set_ros_time_override(Time(nanoseconds=3200000000));node._publish()
    assert len(points)==count and not statuses[-1]['valid']
    assert statuses[-1]['reason']=='measurement_stale'
    assert statuses[-1]['measurement_stamp']==pytest.approx(2.2)


def test_wall_stale_clock_pause_and_calibration_loss_are_invalid(locator):
    node,points,statuses=locator
    for stamp in (2.,2.1,2.2):feed(node,stamp)
    node.measurement['received']=time.monotonic()-2.
    node._publish();assert not statuses[-1]['valid']
    node.set_parameters([Parameter('calibration_valid',value=False)])
    node._publish();assert not statuses[-1]['valid'] and statuses[-1]['reason']=='calibration_not_confirmed'


def test_old_and_future_detections_cannot_refresh_measurement(locator):
    node,points,statuses=locator
    msg=feed(node,2.)
    node._on_det(msg)
    assert node.skipped['stale']==1
    msg.header.stamp=Time(nanoseconds=3000000000).to_msg()
    node._on_det(msg)
    assert node.skipped['stale']==2
    assert node.measurement['stamp']==2.


def test_calibration_reconfirmation_requires_new_measurements(locator):
    node,points,statuses=locator
    for stamp in (2.,2.1,2.2):feed(node,stamp)
    assert statuses[-1]['valid']
    node.set_parameters([Parameter('calibration_valid',value=False)])
    node._publish()
    node.set_parameters([Parameter('calibration_valid',value=True)])
    feed(node,2.3)
    assert node.est.n==1
    assert not statuses[-1]['valid'] and statuses[-1]['reason']=='insufficient_samples'


def test_exposure_attitude_includes_roll_and_wraps_shortest_arc(locator):
    node,_,_=locator
    for stamp,yaw,roll in ((2.,170.,170.),(2.2,-170.,-170.)):
        attitude=Vector3Stamped();attitude.header.stamp=Time(nanoseconds=round(stamp*1e9)).to_msg()
        attitude.vector.x=yaw;attitude.vector.y=10.;attitude.vector.z=roll
        node._on_att(attitude)
    assert node._attitude_at(2.1)==pytest.approx((180.,10.,180.))
    attitude.header.stamp=Time(nanoseconds=2300000000).to_msg();attitude.vector.z=float('nan')
    node._on_att(attitude)
    assert len(node.att)==2


def test_hardware_locator_applies_roll_calibration_and_siyi_rotation_order(locator):
    node,_,statuses=locator
    node.pose_source='gimbal'
    node.set_parameters([Parameter('roll_sign',value=-1.),Parameter('roll_zero_deg',value=10.),
        Parameter('base_heading_deg',value=90.),Parameter('yaw_sign',value=-1.)])
    for stamp in (1.9,2.1):
        attitude=Vector3Stamped();attitude.header.stamp=Time(nanoseconds=round(stamp*1e9)).to_msg()
        attitude.vector.x=30.;attitude.vector.y=30.;attitude.vector.z=30.
        node._on_att(attitude)
    feed(node,2.)
    yaw,pitch,roll=np.radians([60.,30.,-20.])
    expected=40.*np.array([np.cos(yaw)*np.cos(pitch)-np.sin(yaw)*np.sin(roll)*np.sin(pitch),
        np.sin(yaw)*np.cos(pitch)+np.cos(yaw)*np.sin(roll)*np.sin(pitch),
        -np.cos(roll)*np.sin(pitch)])
    np.testing.assert_allclose(statuses[-1]['ned'],expected,atol=1e-9)
    assert statuses[-1]['station_offset_ned']==[0.,0.,0.]


def test_hardware_requires_calibrated_intrinsics_instead_of_nominal_fov(locator):
    node, points, statuses = locator
    node.pose_source = 'gimbal'
    node.info = None
    feed(node, 2.)
    assert not points and not statuses[-1]['valid']
    assert statuses[-1]['reason'] == 'waiting_for_camera_info'


def test_hardware_rejects_intrinsics_from_another_camera_frame(locator):
    node, points, statuses = locator
    node.pose_source = 'gimbal'
    node.info['frame_id'] = 'another_camera'
    feed(node, 2.)
    assert not points and not statuses[-1]['valid']
    assert statuses[-1]['reason'] == 'camera_frame_mismatch'


def test_mavlink_bridge_uses_source_age_not_repeated_receipt_time():
    from balloon_locator.mavlink_bridge_node import MavlinkBridgeNode
    sent=[];now=time.monotonic()
    fake=SimpleNamespace(point=(40.,0.,-8.),point_stamp=9.9,t_point=now,t_status=now,
        max_age=1.,min_samples=30,max_sigma=2.,expect_frame='local_ned',cam=(37.,127.,1.5),decl=0.,target=(1,191),
        get_clock=lambda:SimpleNamespace(now=lambda:Time(nanoseconds=10000000000)),
        get_logger=lambda:SimpleNamespace(info=lambda *a,**k:None),
        mav=SimpleNamespace(mav=SimpleNamespace(command_int_send=lambda *args:sent.append(args))))
    fake.status=dict(valid=True,frame_id='local_ned',pose_source='gimbal',measurement_stamp=9.9,
        measurement_wall_age=.1,n=30,sigma_m=.3,calibration_valid=True,station_offset_ned=[0.,0.,0.])
    MavlinkBridgeNode._send(fake)
    assert len(sent)==1
    # Both ROS receipts look fresh, but the camera measurement is now old.
    fake.point_stamp=1.;fake.status['measurement_stamp']=1.
    MavlinkBridgeNode._send(fake)
    assert len(sent)==1
    fake.point_stamp=9.9;fake.status['measurement_stamp']=9.9
    fake.status['measurement_wall_age']=2.
    MavlinkBridgeNode._send(fake)
    assert len(sent)==1
    fake.status['measurement_wall_age']=.1
    fake.status['station_offset_ned']=[10.,0.,0.]
    MavlinkBridgeNode._send(fake)
    assert len(sent)==1  # The GPS bridge adds the camera origin exactly once.
    fake.status['station_offset_ned']=[0.,0.,0.];fake.status['calibration_valid']=False
    MavlinkBridgeNode._send(fake)
    assert len(sent)==1
