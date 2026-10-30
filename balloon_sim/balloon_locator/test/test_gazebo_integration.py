"""Coordinate/frame and authored-scene contracts; no flight or hardware side effects."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from balloon_locator.gazebo_bridge import mission_estimate
from balloon_locator.gazebo_scene import make_world


def sample():
    return dict(valid=True,frame_id='local_ned',measurement_stamp=4.,pose_source='tf',station_origin_world=[0.,0.,0.],
        north_yaw_world_rad=math.pi/2,ned=[-40.,0.,-10.],covariance=np.diag([.04,.01,.09]).tolist(),
        bbox=[620.,460.,40.,40.],image_size=[1280,960],camera_centered=True,alignment_error_deg=.01,
        calibration_revision='calibrated-sim',calibration_valid=True,measurement_wall_age=.1,
        camera_origin_world=[0.,0.,.8])


def test_station_ned_is_transformed_to_world_before_px4_alignment():
    result=mission_estimate(sample(),4.1,3.)
    assert result['position']==[0.,-40.,10.]
    np.testing.assert_allclose(result['covariance'],np.diag([.01,.04,.09]))
    assert result['stamp']==4. and result['evidence']['estimator']=='balloon_locator'
    assert not result['evidence']['ground_truth_used']


@pytest.mark.parametrize('change',['stale','before_start','invalid','wrong_frame','frame_label','uncertain'])
def test_invalid_coordinate_cannot_reach_the_mission(change):
    status=sample();now=4.1;start=3.
    if change=='stale':now=5.
    if change=='before_start':start=4.05
    if change=='invalid':status['valid']=False
    if change=='wrong_frame':status['north_yaw_world_rad']=0.
    if change=='frame_label':status['frame_id']='flight_ned'
    if change=='uncertain':status['covariance']=(np.eye(3)*.3).tolist()
    with pytest.raises(ValueError):mission_estimate(status,now,start)


def test_world_has_one_balloon_at40m10m_no_lidar_and_native_motors():
    from balloon_locator.runtime_paths import runtime_paths
    from balloon_locator.stl_airframe import build_model
    workspace=Path(__file__).resolve().parents[2]
    paths=runtime_paths(workspace);sim=workspace/'support/jol_sim'
    if not paths['px4_root'].is_dir() or not paths['px4_plugin_dir'].is_dir():
        pytest.skip('external PX4/Gazebo dependency build unavailable')
    profile=workspace/'balloon_locator/config/stl_pacer_quad.json'
    model,_=build_model(sim,paths['px4_root'],paths['px4_plugin_dir'],paths['px4_plugin_dir'],
        profile,workspace/'balloon_locator/meshes/stl_airframe')
    world=make_world(sim,paths['px4_root'],paths['px4_plugin_dir'],paths['px4_plugin_dir'],profile,aircraft_model=model)
    root=ET.fromstring(world);target=root.find("world/model[@name='static_target']")
    np.testing.assert_allclose(np.fromstring(target.findtext('pose'),sep=' ')[:3],[0.,-40.,10.],atol=1e-10)
    assert float(target.findtext('link/visual/geometry/sphere/radius'))==.15
    assert not root.findall(".//plugin[@name='jol_mid360']")
    assert len(root.findall("world/model[@name='stl_pacer_quad']"))==1
    assert len([p for p in root.findall('.//plugin') if p.get('filename','').endswith('libgazebo_motor_model.so')])==4
    assert target.findtext('link/collision/surface/contact/collide_without_contact')=='true'


def test_delayed_start_response_cannot_undo_a_user_reset():
    from types import SimpleNamespace
    from balloon_locator.gazebo_bridge import GazeboBridge
    reset_state=SimpleNamespace(reset_generation=2,started=False,stage='resetting')
    GazeboBridge.started_after_reset(reset_state,SimpleNamespace(),1)
    assert not reset_state.started and reset_state.stage=='resetting'


def test_future_and_reordered_images_cannot_drive_gimbal_search_state():
    from types import SimpleNamespace
    from vision_msgs.msg import Detection2DArray
    from rclpy.time import Time
    from balloon_locator.gazebo_bridge import GazeboBridge
    packets=[]
    state=SimpleNamespace(now=lambda:10.,last_detection_stamp=None,last_visual=None,images=[],
        detection_pub=SimpleNamespace(publish=packets.append))
    message=Detection2DArray();message.header.stamp=Time(seconds=20.).to_msg()
    GazeboBridge.receive_detection(state,message)
    assert not packets and state.last_detection_stamp is None
    message.header.stamp=Time(seconds=9.9).to_msg()
    GazeboBridge.receive_detection(state,message)
    assert len(packets)==1
    message.header.stamp=Time(seconds=9.8).to_msg()
    GazeboBridge.receive_detection(state,message)
    assert len(packets)==1 and state.last_detection_stamp==pytest.approx(9.9)


@pytest.mark.parametrize('change',['uncalibrated','bad_bbox','bad_alignment','bad_origin','wall_stale'])
def test_unusable_camera_evidence_does_not_reach_flight_or_crash_ui(change):
    status=sample()
    if change=='uncalibrated':status['calibration_valid']=False
    if change=='bad_bbox':status['bbox'][2]=float('nan')
    if change=='bad_alignment':status['alignment_error_deg']=float('nan')
    if change=='bad_origin':status['camera_origin_world']=[0.,0.]
    if change=='wall_stale':status['measurement_wall_age']=1.
    with pytest.raises(ValueError):mission_estimate(status,4.1,3.)


def test_invalid_detection_is_not_visual_evidence_or_json_nan():
    import json
    from types import SimpleNamespace
    from vision_msgs.msg import Detection2D,Detection2DArray
    from rclpy.time import Time
    from balloon_locator.gazebo_bridge import GazeboBridge
    packets=[]
    state=SimpleNamespace(now=lambda:10.,last_detection_stamp=None,last_visual=None,images=[],
        detection_pub=SimpleNamespace(publish=packets.append))
    message=Detection2DArray();message.header.stamp=Time(seconds=9.9).to_msg()
    det=Detection2D();det.bbox.size_x=float('nan');det.bbox.size_y=40.
    message.detections=[det]
    GazeboBridge.receive_detection(state,message)
    assert state.last_visual is None
    assert json.loads(packets[-1].data)['detections']==[]


def test_truncated_image_is_discarded_before_overlay_reshape():
    from types import SimpleNamespace
    from sensor_msgs.msg import Image
    from balloon_locator.gazebo_bridge import GazeboBridge
    published=[];state=SimpleNamespace(images=[],annotated_pub=SimpleNamespace(publish=published.append))
    image=Image(width=2,height=2,encoding='rgb8',step=6,data=[0]*9)
    GazeboBridge.receive_image(state,image)
    assert not state.images and not published


def test_status_queue_delay_counts_toward_measurement_wall_age():
    with pytest.raises(ValueError,match='wall_stale'):
        mission_estimate(sample(),4.1,3.,status_age=.75)
