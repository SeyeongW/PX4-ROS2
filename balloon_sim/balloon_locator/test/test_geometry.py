import numpy as np
import pytest

from balloon_locator import geometry as g


def test_boresight_directions():
    ray = lambda az, el: g.pixel_ray_ned(960, 540, 960, 540, 1000, az, el)
    assert ray(0, 0) == pytest.approx([1, 0, 0], abs=1e-9)         # 북
    assert ray(90, 0) == pytest.approx([0, 1, 0], abs=1e-9)        # 동
    assert ray(0, 30)[2] < 0                                        # 위쪽 = D 음수
    assert ray(0, 30) == pytest.approx([np.cos(np.radians(30)), 0, -np.sin(np.radians(30))], abs=1e-9)


def test_pixel_offsets_map_to_expected_sides():
    right = g.pixel_ray_ned(1160, 540, 960, 540, 1000, 0, 0)
    below = g.pixel_ray_ned(960, 740, 960, 540, 1000, 0, 0)
    assert right[1] > 0 and below[2] > 0                           # 오른쪽 -> 동(+), 아래 -> D(+)


def test_range_and_roundtrip_position():
    f = g.focal_px(1920, 80.0, 2.0)
    truth = np.array([40.0, 30.0, -20.0])
    r = np.linalg.norm(truth)
    az = np.degrees(np.arctan2(truth[1], truth[0]))
    el = np.degrees(np.arcsin(-truth[2] / r))
    diameter_px = 0.3 * f / r
    pos, cov, rng = g.locate_balloon(
        960, 540, diameter_px, width_px=1920, height_px=1080, hfov_deg_at_1x=80.0, zoom=2.0,
        az_deg=az, el_deg=el, balloon_diameter_m=0.3, offset_ned=(1, 2, 3))
    assert rng == pytest.approx(r)
    assert pos == pytest.approx(truth + [1, 2, 3], abs=1e-6)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_covariance_larger_along_ray_than_across_at_small_pixel_sigma():
    ray = np.array([1.0, 0, 0])
    cov = g.measurement_covariance(ray, 100.0, 10.0, 2000.0, pixel_sigma_px=0.5)
    assert cov[0, 0] > cov[1, 1]                                    # 지름 기반 거리 오차가 지배적


def test_calibrated_intrinsics_and_asymmetric_center_ray():
    info=dict(k=[6000.,0.,639.5,0.,6100.,479.5,0.,0.,1.],d=[0.]*5,
              width=1280,height=960,frame_id='optical',distortion_model='plumb_bob')
    intrinsics=g.calibrated_intrinsics(info)
    np.testing.assert_allclose(g.optical_ray(639.5,479.5,intrinsics),[0.,0.,1.])
    assert g.optical_ray(640.5,480.5,intrinsics)[0]>0
    info['d'][0]=.01
    with pytest.raises(ValueError,match='rectified'):
        g.calibrated_intrinsics(info)


@pytest.mark.parametrize('values',[(0.,.3,1000.),(10.,-.3,1000.),(10.,.3,float('nan'))])
def test_invalid_range_geometry_is_rejected(values):
    with pytest.raises(ValueError):g.range_from_diameter(*values)


def test_siyi_312_rotation_combines_yaw_roll_pitch_in_documented_order():
    # An independent analytic boresight for Rz(yaw) Rx(roll) Ry(pitch).
    yaw,pitch,roll=np.radians([60.,30.,20.])
    expected=np.array([np.cos(yaw)*np.cos(pitch)-np.sin(yaw)*np.sin(roll)*np.sin(pitch),
        np.sin(yaw)*np.cos(pitch)+np.cos(yaw)*np.sin(roll)*np.sin(pitch),
        -np.cos(roll)*np.sin(pitch)])
    rotation=g.ned_from_cam(60.,30.,20.,'zxy')
    np.testing.assert_allclose(rotation@np.array([1.,0.,0.]),expected,atol=1e-12)
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
    assert np.linalg.det(rotation)==pytest.approx(1.)
    assert not np.allclose(rotation,g.ned_from_cam(60.,30.,20.,'zyx'))
    # A positive 90deg roll maps right-of-image to down in NED.
    ray=g.pixel_ray_ned(1100,500,1000,500,1000,0.,0.,90.,'zxy')
    assert ray[2]>0 and abs(ray[1])<1e-12
