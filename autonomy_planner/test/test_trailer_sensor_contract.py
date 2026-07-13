from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_x500_has_distinct_front_down_rgb_depth_and_down_lidar():
    model = ET.parse(ROOT / "px4_models/x500_city_rgbd_lidar/model.sdf").getroot()
    sensors = {
        sensor.attrib["name"]: sensor
        for sensor in model.findall("./model/link/sensor")
    }
    assert sensors["front_rgb_camera"].attrib["type"] == "camera"
    assert sensors["front_depth_camera"].attrib["type"] == "depth_camera"
    assert sensors["down_rgb_camera"].attrib["type"] == "camera"
    assert sensors["down_depth_camera"].attrib["type"] == "depth_camera"
    assert sensors["lidar"].attrib["type"] == "gpu_lidar"

    topics = {name: sensor.findtext("topic") for name, sensor in sensors.items()}
    assert topics["front_rgb_camera"] == "front_camera/image"
    assert topics["front_depth_camera"] == "front_depth/image"
    assert topics["down_rgb_camera"] == "down_camera/image"
    assert topics["down_depth_camera"] == "down_depth/image"
    assert topics["lidar"] is None  # PX4 needs its world-qualified default path.

    frames = {name: sensor.findtext("gz_frame_id") for name, sensor in sensors.items()}
    assert len(set(frames.values())) == len(frames)
    assert sensors["front_rgb_camera"].findtext("camera/optical_frame_id") \
        == "front_camera_optical_frame"
    assert sensors["front_depth_camera"].findtext("camera/optical_frame_id") \
        == "front_depth_optical_frame"
    assert sensors["down_rgb_camera"].findtext("camera/optical_frame_id") \
        == "down_camera_optical_frame"
    assert sensors["down_depth_camera"].findtext("camera/optical_frame_id") \
        == "down_depth_optical_frame"
    front_depth_size = sensors["front_depth_camera"].find("camera/image")
    down_depth_size = sensors["down_depth_camera"].find("camera/image")
    assert front_depth_size is not None and down_depth_size is not None
    assert (front_depth_size.findtext("width"), front_depth_size.findtext("height")) \
        == ("320", "240")
    assert (down_depth_size.findtext("width"), down_depth_size.findtext("height")) \
        == ("320", "240")
    assert float(sensors["front_depth_camera"].findtext("update_rate")) == 12.0
    assert float(sensors["down_depth_camera"].findtext("update_rate")) == 15.0


def test_bridge_contract_exposes_all_required_ros_topics_and_tf_frames():
    source = (ROOT / "gazebo/launch/sensor_bridge.launch.py").read_text()
    topics = (
        "/front_camera/image", "/front_camera/camera_info",
        "/front_depth/image", "/front_depth/points",
        "/down_camera/image", "/down_camera/camera_info",
        "/down_depth/image", "/down_depth/points",
        "/down_lidar", "/down_lidar/points",
    )
    for topic in topics:
        assert topic in source
    frames = (
        "front_camera_optical_frame", "front_depth_optical_frame",
        "down_camera_optical_frame", "down_depth_optical_frame",
        "lidar_sensor_link",
    )
    for frame in frames:
        assert frame in source
    assert 'LaunchConfiguration("world")' in source
    assert 'LaunchConfiguration("model")' in source


def test_trailer_motion_config_matches_physical_marker_and_exact_route():
    data = yaml.safe_load(
        (ROOT / "autonomy_planner/config/trailer_motion.yaml").read_text()
    )["trailer_motion"]["ros__parameters"]
    assert data["start_enu_m"] == [200.0, -125.0, 0.0]
    assert data["goal_enu_m"] == [-128.0, -128.0, 0.0]
    route = data["route_waypoints_enu_m"]
    assert route == [
        200.0, -125.0, 0.0,
        200.0, -100.0, 0.0,
        -128.0, -100.0, 0.0,
        -128.0, -128.0, 0.0,
    ]
    assert data["cruise_speed_m_s"] == 3.0
    assert data["max_acceleration_m_s2"] > 0.0
    assert data["max_jerk_m_s3"] > 0.0
    assert data["odom_frame"] == "trailer/odom"
    assert data["base_frame"] == "trailer/base_link"
    assert data["deck_frame"] == "trailer/landing_deck"
    assert data["marker_frame"] == "trailer/aruco_marker"
    assert data["deck_height_m"] + data["marker_offset_deck_m"][2] == 1.251
