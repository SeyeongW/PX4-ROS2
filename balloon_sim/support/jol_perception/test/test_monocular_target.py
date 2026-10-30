"""Camera-only scale and uncertainty; no simulator pose can enter estimation."""
import importlib.util
import math
import numpy as np
import pytest


def estimate(*args, **kwargs):
    assert importlib.util.find_spec('jol_perception.monocular_target') is not None
    from jol_perception.monocular_target import estimate_sphere
    return estimate_sphere(*args, **kwargs)


def camera(focal=6000.):
    return dict(k=[focal,0.,639.5,0.,focal,479.5,0.,0.,1.],d=[0.]*5,
        width=1280,height=960,frame_id='optical',distortion_model='plumb_bob')


def detection(distance=40., focal=6000.):
    radius=.15
    diameter=2*focal*radius/math.sqrt(distance**2-radius**2)
    return dict(bbox=[639.5-diameter/2,479.5-diameter/2,diameter,diameter],
        center_px=[639.5,479.5],diameter_px=diameter,confidence=1.)


def test_known_sphere_center_and_world_transform():
    pose=np.eye(4);pose[:3,3]=[1.,2.,3.]
    r=estimate(detection(),camera(),pose,diameter_m=.3,diameter_std_m=0.)
    np.testing.assert_allclose(r['position'],[1.,2.,43.],atol=1e-8)
    assert r['source']=='camera_monocular_size'
    assert r['position_semantics']=='known_sphere_center'
    assert np.linalg.eigvalsh(r['covariance']).max()<=.25


def test_existing_wide_camera_cannot_authorize_40m_30cm_target():
    with pytest.raises(ValueError,match='target_too_small|uncertainty'):
        estimate(detection(focal=462.),camera(462.),np.eye(4),diameter_m=.3,diameter_std_m=0.)


def test_approximate_balloon_size_uncertainty_is_not_hidden():
    with pytest.raises(ValueError,match='uncertainty'):
        estimate(detection(),camera(),np.eye(4),diameter_m=.3,diameter_std_m=.015)


@pytest.mark.parametrize('change',['clipped','not_round','invalid_size','invalid_pose'])
def test_invalid_monocular_measurement_rejected(change):
    d=detection();pose=np.eye(4);diameter=.3
    if change=='clipped':d['box_clipped']=True
    elif change=='not_round':d['bbox'][3]*=.4
    elif change=='invalid_size':diameter=float('nan')
    else:pose[0,0]=float('nan')
    with pytest.raises(ValueError):estimate(d,camera(),pose,diameter_m=diameter,diameter_std_m=0.)


def test_actual_camera_node_has_no_lidar_or_truth_io_and_obeys_start_and_age():
    import json,time
    from types import SimpleNamespace
    import rclpy
    from rclpy.time import Time
    from geometry_msgs.msg import TransformStamped
    from std_srvs.srv import Trigger
    from jol_perception.camera_only_estimator import CameraOnlyEstimator
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true','-p','calibration_valid:=true',
        '-p','calibration_source:=camera_test','-p','balloon_diameter_std_m:=0.0'])
    node=CameraOnlyEstimator();messages=[]
    node.estimate_pub=SimpleNamespace(publish=lambda msg:messages.append(json.loads(msg.data)))
    node.marker_pub=SimpleNamespace(publish=lambda msg:None)
    try:
        assert not any('lidar' in s.topic_name or 'truth' in s.topic_name for s in node.subscriptions)
        assert not any('points' in p.topic_name for p in node.publishers)
        node.get_clock().set_ros_time_override(Time(nanoseconds=2000000000))
        node.info=camera()
        transform=TransformStamped();transform.header.frame_id='world';transform.child_frame_id='optical'
        transform.transform.rotation.w=1.
        node.tf.set_transform_static(transform,'test')
        packet=dict(stamp=2.,frame_id='optical',width=1280,height=960,detections=[detection()])
        node.pending_detection=(packet,time.monotonic());node.tick()
        assert node.last_result is None and node.pending_detection is None
        response=node.start_acquisition(Trigger.Request(),Trigger.Response())
        assert response.success and node.acquisition_stage=='camera_search'
        node.pending_detection=(packet,time.monotonic());node.tick()
        assert messages[-1]['valid'] and messages[-1]['source']=='camera_monocular_size'
        assert messages[-1]['evidence']['ground_truth_used'] is False
        assert messages[-1]['evidence']['lidar_used'] is False
        node.get_clock().set_ros_time_override(Time(nanoseconds=3000000000));node.tick()
        assert node.last_result is None and messages[-1]['reason']=='camera_estimate_stale'
        node.pending_detection=(dict(packet,stamp=3.01),time.monotonic()-2.);node.tick()
        assert messages[-1]['reason']=='camera_result_wall_stale'
        node.last_result=dict(stamp=3.,valid=True)
        from rclpy.parameter import Parameter
        node.set_parameters([Parameter('calibration_valid',value=False)])
        node.tick()
        assert node.last_result is None and messages[-1]['reason']=='calibration_not_confirmed'
        node.reset_acquisition(Trigger.Request(),Trigger.Response())
        assert not node.acquisition_started and node.last_result is None
    finally:
        node.destroy_node();rclpy.shutdown()
