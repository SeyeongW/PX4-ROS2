from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import pytest


REPOSITORY = Path(__file__).resolve().parents[2]


def test_trailer_model_has_deck_marker_collision_and_velocity_topic():
    path = REPOSITORY / "gazebo/models/trailer_aruco/model.sdf"
    root = ET.parse(path).getroot()

    assert root.attrib["version"] == "1.9"
    assert root.find("./model/link[@name='base_link']") is not None
    assert root.find(".//collision[@name='body_collision']") is not None
    assert root.find(".//collision[@name='landing_deck_collision']") is not None
    marker = root.find(".//visual[@name='aruco_marker_visual']/geometry/plane/size")
    assert marker is not None and marker.text.strip() == "1.3 1.3"
    topic = root.find("./model/plugin/topic")
    assert topic is not None and topic.text.strip() == "/model/trailer/cmd_vel"


def test_trailer_model_ground_clearance_and_deck_top_are_consistent():
    path = REPOSITORY / "gazebo/models/trailer_aruco/model.sdf"
    root = ET.parse(path).getroot()
    body = root.find(".//collision[@name='body_collision']")
    deck = root.find(".//collision[@name='landing_deck_collision']")
    assert body is not None and deck is not None

    body_z = float(body.findtext("pose").split()[2])
    body_height = float(body.findtext("geometry/box/size").split()[2])
    deck_z = float(deck.findtext("pose").split()[2])
    deck_height = float(deck.findtext("geometry/box/size").split()[2])
    assert body_z - 0.5 * body_height == pytest.approx(0.05)
    assert deck_z + 0.5 * deck_height == pytest.approx(1.25)


def test_trailer_texture_is_detectable_dictionary_4x4_50_marker_zero():
    texture = cv2.imread(
        str(REPOSITORY / "gazebo/models/aruco_marker_0/materials/textures/aruco_0.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    assert texture is not None
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        detector = cv2.aruco.ArucoDetector(dictionary)
        _, ids, _ = detector.detectMarkers(texture)
    except AttributeError:
        _, ids, _ = cv2.aruco.detectMarkers(texture, dictionary)
    assert ids is not None and ids.flatten().tolist() == [0]


def test_trailer_texture_quiet_zone_keeps_detectable_marker_one_metre_wide():
    texture = cv2.imread(
        str(REPOSITORY / "gazebo/models/aruco_marker_0/materials/textures/aruco_0.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(texture)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(texture, dictionary)
    assert ids is not None
    detected_side_px = float(
        corners[0][0, 1, 0] - corners[0][0, 0, 0] + 1.0
    )
    rendered_marker_side_m = 1.3 * detected_side_px / float(texture.shape[1])
    assert rendered_marker_side_m == pytest.approx(1.0, abs=0.003)
