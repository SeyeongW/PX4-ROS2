from types import SimpleNamespace

import numpy as np
import pytest

from landing_mpc.frame import (gimbal_axis_flu, gimbal_camera_origin_flu,
                               gimbal_camera_origin_to_enu,
                               gimbal_joint_angles)
from landing_mpc.gimbal_control_node import (GimbalControlNode, _range_handoff,
                                             _rate_limit)
from landing_mpc.marker_tf_node import ray_to_horizontal_plane


def test_known_deck_plane_discards_noisy_aruco_range_but_keeps_its_ray():
    origin = np.array([10.0, -3.0, 24.87])
    short_range = np.array([4.0, 2.0, -20.0])
    wrong_long_range = 3.7 * short_range

    first = ray_to_horizontal_plane(origin, short_range, 1.811)
    second = ray_to_horizontal_plane(origin, wrong_long_range, 1.811)

    assert np.allclose(first, second)
    assert first[2] == pytest.approx(1.811)
    assert ray_to_horizontal_plane(origin, [1.0, 0.0, 0.0], 1.811) is None
    assert ray_to_horizontal_plane(origin, [0.0, 0.0, 1.0], 1.811) is None


def test_gimbal_plane_projection_starts_at_the_lens_not_base_link():
    joints = np.zeros(3)
    q_level = np.array([1.0, 0.0, 0.0, 0.0])

    assert np.allclose(
        gimbal_camera_origin_flu(joints), [0.0, 0.0, -0.13], atol=1e-6)
    assert np.allclose(
        gimbal_camera_origin_to_enu(joints, q_level),
        [0.0, 0.0, -0.13], atol=1e-6)

    moving_joints = np.array([0.6, -0.15, -1.1])
    body_origin = gimbal_camera_origin_flu(moving_joints)
    world_origin = gimbal_camera_origin_to_enu(moving_joints, q_level)
    assert np.isfinite(body_origin).all()
    assert np.linalg.norm(world_origin) == pytest.approx(
        np.linalg.norm(body_origin))


def test_gimbal_crosses_nadir_without_yaw_flip():
    yaw = 0.0
    pitch = -np.pi / 2.0

    for offset_deg in (5.0, 1.0, 0.0, -1.0, -5.0):
        offset = np.deg2rad(offset_deg)
        direction = np.array([np.sin(offset), 0.0, -np.cos(offset)])
        yaw, roll, pitch = gimbal_joint_angles(
            direction, yaw_hold=yaw, pitch_hold=pitch)

        assert roll == 0.0
        assert abs(yaw) < 1e-12
        assert np.isclose(pitch, -np.pi / 2.0 + offset)
        assert np.allclose(gimbal_axis_flu(yaw, pitch), direction, atol=1e-12)


def test_range_handoff_uses_horizontal_gps_distance_and_is_continuous():
    target = np.array([0.8, 0.0, -0.6])
    down = np.array([0.0, 0.0, -1.0])

    # Altitude is deliberately irrelevant: this is the horizontal GPS range.
    assert np.allclose(
        _range_handoff(target, [10.0, 0.0, -100.0], 10.0, 9.0), down)
    assert np.allclose(
        _range_handoff(target, [9.0, 0.0, 100.0], 10.0, 9.0), target)
    middle = _range_handoff(target, [9.5, 0.0, -100.0], 10.0, 9.0)
    assert np.isclose(np.linalg.norm(middle), 1.0)
    assert not np.allclose(middle, down)
    assert not np.allclose(middle, target)

    distances = np.linspace(10.5, 8.5, 2001)
    axes = np.asarray([
        _range_handoff(target, [distance, 0.0, 100.0], 10.0, 9.0)
        for distance in distances
    ])
    adjacent_dot = np.sum(axes[:-1] * axes[1:], axis=1)
    jumps = np.rad2deg(np.arccos(np.clip(adjacent_dot, -1.0, 1.0)))
    assert float(np.max(jumps)) < 1.0


def test_city_gimbal_keeps_fresh_cue_authoritative_over_transient_vision():
    node = SimpleNamespace(
        _cue=np.array([1.0, 2.0, 0.0]), _cue_t=9.9,
        _vis=np.array([80.0, -40.0, 0.0]), _vis_t=9.9,
        _valid=True, prefer_cue_aim=True,
        vision_timeout_s=1.0, cue_timeout_s=2.0,
        _source='nadir', _now=lambda: 10.0,
    )

    assert np.allclose(GimbalControlNode._look_at(node), node._cue)
    assert node._source == 'cue'

    node.prefer_cue_aim = False
    assert np.allclose(GimbalControlNode._look_at(node), node._vis)
    assert node._source == 'vision'

    node.prefer_cue_aim = True
    node._cue_t = 7.0
    assert np.allclose(GimbalControlNode._look_at(node), node._vis)
    assert node._source == 'vision'


def test_gimbal_leaves_joints_uncommanded_until_land_then_tracks():
    node = SimpleNamespace(
        _mission_state='PRECHECK', _landing_started=False, _source='nadir',
        _q=np.array([1.0, 0.0, 0.0, 0.0]), _yaw=0.0,
        _pitch=-np.pi / 2.0,
        _desired_axis_flu=lambda: (_ for _ in ()).throw(
            AssertionError('world tracker ran before land')))

    GimbalControlNode._on_mission_state(node, SimpleNamespace(data='HOVER'))
    assert not node._landing_started
    assert GimbalControlNode._desired_joint_angles(node) is None
    assert node._source == 'uncommanded (waiting for land)'

    node._desired_axis_flu = lambda: np.array([1.0, 0.0, 0.0])
    GimbalControlNode._on_mission_state(
        node, SimpleNamespace(data='RETURN_PLAN'))
    assert node._landing_started
    yaw, roll, pitch = GimbalControlNode._desired_joint_angles(node)
    assert np.allclose([yaw, roll, pitch], [0.0, 0.0, 0.0])

    # DONE is the confirmed landed+disarmed terminal state: stop commands and
    # leave the final measured pose alone.
    GimbalControlNode._on_mission_state(node, SimpleNamespace(data='DONE'))
    assert node._landing_started
    assert GimbalControlNode._desired_joint_angles(node) is None
    assert node._source == 'uncommanded (landed)'


@pytest.mark.parametrize(
    ('mission_state', 'landing_started'),
    [('HOVER', False), ('DONE', True)])
def test_gimbal_idle_tick_mirrors_encoders_without_sending_a_command(
        mission_state, landing_started):
    published_joints = []
    node = SimpleNamespace(
        _measured=(0.4, -0.1, 0.2),
        _mission_state=mission_state,
        _landing_started=landing_started,
        _source='nadir',
        _desired=(1.0, 1.0),
        _yaw=0.0,
        _roll=0.0,
        _pitch=0.0,
        _desired_joint_angles=lambda: GimbalControlNode._desired_joint_angles(
            node),
        _publish_joints=lambda: published_joints.append(True),
    )

    GimbalControlNode._tick(node)

    assert (node._yaw, node._roll, node._pitch) == node._measured
    assert node._desired is None
    assert published_joints == [True]


def test_gimbal_uses_literal_joint_lock_outside_ten_metres_after_land():
    node = SimpleNamespace(
        _landing_started=True, _source='cue',
        _q=np.array([np.cos(np.pi / 12.0), np.sin(np.pi / 12.0), 0.0, 0.0]),
        _p_d=np.array([0.0, 0.0, 5.0]),
        _cue=np.array([10.0, 0.0, 0.0]), _cue_t=1.0,
        _vis=None, _vis_t=None, _valid=False,
        vision_timeout_s=1.0, cue_timeout_s=2.0,
        aim_start_m=10.0, aim_full_m=9.0,
        _yaw=4.0 * np.pi, _pitch=-np.pi / 2.0,
        _now=lambda: 1.1,
    )
    node._look_at = lambda: GimbalControlNode._look_at(node)
    node._desired_axis_flu = lambda: GimbalControlNode._desired_axis_flu(node)

    assert np.allclose(
        GimbalControlNode._desired_joint_angles(node),
        [4.0 * np.pi, 0.0, -np.pi / 2.0])
    assert node._source.startswith('joint lock')


def test_land_entry_joint_lock_is_slower_than_target_tracking():
    pitch_commands = []

    class Publisher:
        def __init__(self, values=None):
            self.values = values

        def publish(self, message):
            if self.values is not None:
                self.values.append(message.data)

    node = SimpleNamespace(
        _measured=(0.0, 0.0, 0.0),
        _yaw=0.0, _roll=0.0, _pitch=0.0,
        _desired=None, _sat_n=0,
        max_rate=np.deg2rad(90.0),
        land_entry_rate=np.deg2rad(30.0),
        rate_hz=50.0,
        _pub_yaw=Publisher(), _pub_roll=Publisher(),
        _pub_pitch=Publisher(pitch_commands),
        _publish_joints=lambda: None,
    )

    def joint_lock():
        node._source = 'joint lock (cue beyond 10 m)'
        return 0.0, 0.0, -np.pi / 2.0

    node._desired_joint_angles = joint_lock
    GimbalControlNode._tick(node)
    assert pitch_commands[-1] == pytest.approx(-np.deg2rad(0.6))

    node._pitch = 0.0

    def target_tracking():
        node._source = 'cue handoff'
        return 0.0, 0.0, -np.pi / 2.0

    node._desired_joint_angles = target_tracking
    GimbalControlNode._tick(node)
    assert pitch_commands[-1] == pytest.approx(-np.deg2rad(1.8))
