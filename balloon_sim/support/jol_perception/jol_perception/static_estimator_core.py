"""Static camera-box / LiDAR association; no simulator truth or ROS dependency.

The orange detector is a controlled simulation marker experiment, not a claim
of learned object recognition. A camera supplies direction; range comes only
from surface returns of the associated LiDAR cluster.
"""
import math
import hashlib
import json

import cv2
import numpy as np

from .fusion_core import calibration_error
from .tracking_core import dbscan, voxel_filter


SENSOR_QUALITY_DEFAULTS = dict(camera_time_offset_s=0., lidar_time_offset_s=0.,
    camera_time_std_s=.001, lidar_time_std_s=.001, extrinsic_translation_std_m=.02,
    extrinsic_rotation_std_deg=.1, max_position_std_m=.5)


def sensor_quality_config(**values):
    """Explicit assumptions; these do not claim a measured hardware calibration."""
    config = dict(SENSOR_QUALITY_DEFAULTS, **values)
    if set(config) != set(SENSOR_QUALITY_DEFAULTS) or not np.isfinite(list(config.values())).all():
        raise ValueError('invalid_sensor_quality_configuration')
    if (any(abs(config[key]) > .5 for key in ('camera_time_offset_s', 'lidar_time_offset_s'))
            or any(not 0 <= config[key] <= .1 for key in ('camera_time_std_s', 'lidar_time_std_s'))
            or not 0 <= config['extrinsic_translation_std_m'] <= 1.
            or not 0 <= config['extrinsic_rotation_std_deg'] <= 5.
            or not 0 < config['max_position_std_m'] <= .5):
        raise ValueError('invalid_sensor_quality_configuration')
    return config


def calibration_revision(info, source, config, static_transforms=None):
    """Freeze intrinsics, provenance and clock/uncertainty assumptions at lock."""
    payload = json.dumps(dict(info=info, source=source, quality=config, static_transforms=static_transforms),
                         sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def pose_motion(previous, current, seconds):
    """Measured TF finite difference; angular velocity is expressed in world."""
    from scipy.spatial.transform import Rotation
    previous, current = np.asarray(previous, float), np.asarray(current, float)
    if (previous.shape != (4, 4) or current.shape != (4, 4)
            or not np.isfinite(np.r_[previous.ravel(), current.ravel(), seconds]).all() or seconds <= 0):
        raise ValueError('invalid_pose_motion_samples')
    return ((current[:3, 3]-previous[:3, 3])/seconds,
            Rotation.from_matrix(current[:3, :3]@previous[:3, :3].T).as_rotvec()/seconds)


def add_sensor_uncertainty(result, camera_pose, lidar_origin, config, camera_motion, lidar_motion):
    """First-order extrinsic and clock covariance, without hiding systematic bias.

    For p=R*r+t, J_theta=-[R*r]x and J_time=v+omega x (p-t).
    The projected cloud/bbox diagnostics are association consistency only: an
    independent calibration target or held-out edges are still needed to measure
    reprojection accuracy. The estimate is never compared with simulator truth.
    """
    config = sensor_quality_config(**config)
    pose = np.asarray(camera_pose, float)
    p = np.asarray(result['position'], float)
    lever = p-pose[:3, 3]
    x, y, z = lever
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    extrinsic = np.eye(3)*config['extrinsic_translation_std_m']**2
    extrinsic += skew@skew.T*math.radians(config['extrinsic_rotation_std_deg'])**2
    timing = np.zeros((3, 3))
    for motion, origin, key in ((camera_motion, pose[:3, 3], 'camera_time_std_s'),
                                (lidar_motion, np.asarray(lidar_origin), 'lidar_time_std_s')):
        velocity, angular = map(lambda value: np.asarray(value, float), motion)
        if velocity.shape != (3,) or angular.shape != (3,) or not np.isfinite(np.r_[velocity, angular]).all():
            raise ValueError('invalid_sensor_motion')
        jacobian = velocity+np.cross(angular, p-origin)
        timing += np.outer(jacobian, jacobian)*config[key]**2
    covariance = np.asarray(result['covariance'], float)+extrinsic+timing
    maximum_std = math.sqrt(max(0., float(np.linalg.eigvalsh(covariance).max())))
    if maximum_std > config['max_position_std_m']:
        raise ValueError('sensor_position_uncertainty_exceeded')
    result.update(covariance=covariance.tolist(), sensor_quality=dict(config,
        extrinsic_covariance_world=extrinsic.tolist(), timing_covariance_world=timing.tolist(),
        maximum_position_std_m=maximum_std, assumptions='configured_not_measured_hardware_accuracy'))
    return result


def display_clouds(points, origin, target=None):
    """Display-only thinning/ground hiding; never used by association or control."""
    points=np.asarray(points,dtype=float)
    points=points[np.isfinite(points).all(axis=1)]
    points=points[np.linalg.norm(points-np.asarray(origin),axis=1)<=65.]
    context=voxel_filter(points[points[:,2]>=.3],.15,sensor_origin=np.asarray(origin))
    neighborhood=(points[np.max(np.abs(points-np.asarray(target)),axis=1)<=1.5]
                  if target is not None else np.empty((0,3)))
    return context,voxel_filter(neighborhood,.05,sensor_origin=np.asarray(origin))


def detect_orange_marker(image, minimum_pixels=8):
    """Find one deliberately orange experiment marker from actual RGB pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (3, 100, 70), (28, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        if area >= minimum_pixels and w >= 3 and h >= 3:
            proposals.append(dict(bbox=[float(x), float(y), float(w), float(h)],
                                  center_px=[x+(w-1)/2, y+(h-1)/2],
                                  center_semantics='opencv_inclusive_pixel_extent',
                                  confidence=min(1., area/(w*h)), class_name='orange_marker',
                                  source='simulation_color_marker', box_clipped=(
                                      x == 0 or y == 0 or x+w == image.shape[1] or y+h == image.shape[0])))
    return sorted(proposals, key=lambda item: item['bbox'][2]*item['bbox'][3], reverse=True)[:1]


def validate_detection(packet, info, now, max_age=.8):
    """Validate geometry and freshness before either fusion or gimbal actuation."""
    stamp = float(packet['stamp'])
    if not math.isfinite(stamp) or stamp <= 0 or not -.5e-9 <= now-stamp <= max_age:
        raise ValueError('stale_or_invalid_camera_stamp')
    reason = calibration_error(info, packet['width'], packet['height'], packet['frame_id'])
    if reason:
        raise ValueError(reason)
    if not isinstance(packet['detections'], list) or len(packet['detections']) != 1:
        raise ValueError('no_detection' if not packet['detections'] else 'ambiguous_detection')
    detection = packet['detections'][0]
    box = np.asarray(detection['bbox'], dtype=float)
    confidence = float(detection['confidence'])
    if (box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) < 2
            or min(box[:2]) < 0 or box[0]+box[2] > packet['width']
            or box[1]+box[3] > packet['height'] or not math.isfinite(confidence)
            or not 0 < confidence <= 1 or detection.get('box_clipped', False)):
        raise ValueError('invalid_or_clipped_bbox')
    detection_center(detection)
    return detection


def detection_center(detection):
    """OpenCV pixel extents and continuous detector boxes have different centers."""
    box = np.asarray(detection['bbox'], float)
    center = np.asarray(detection.get('center_px', box[:2]+box[2:]/2), float)
    if (center.shape != (2,) or not np.isfinite(center).all()
            or np.any(center < box[:2]) or np.any(center > box[:2]+box[2:])):
        raise ValueError('invalid_detection_center')
    return center


def optical_bearing(box, info, center=None):
    center = np.asarray(box[:2], float)+np.asarray(box[2:], float)/2 if center is None else np.asarray(center, float)
    k = np.asarray(info['k'], float).reshape(3, 3)
    d = np.asarray(info['d'], float)
    xy = cv2.undistortPoints(center.reshape(1, 1, 2), k, d if len(d) else None)[0, 0]
    return np.r_[xy, 1.]


def surface_plane_depth(optical, ray, median_depth, k):
    """Fit measured returns only; never extrapolate an unsupported target face.

    ponytail: fixed 200-hypothesis fit for the small single-target ROI; use an
    existing point-cloud solver if large ROIs make this a measured bottleneck.
    Failure keeps the previously guarded median estimate, with a logged reason.
    """
    evidence = dict(accepted=False, reason='insufficient_plane_support',
                    inlier_count=0, points_count=len(optical), support_fraction=0.,
                    residual_rms_m=None, ray_normal_abs_dot=None,
                    second_singular_value_m=None, ray_support_distance_px=None,
                    median_depth_m=float(median_depth), depth_delta_m=None)
    if len(optical) < 8 or len(np.unique(np.round(optical, 6), axis=0)) < 8:
        return None, evidence
    best = np.array([], dtype=int)
    rng = np.random.default_rng(40)
    for _ in range(200):
        ids = rng.choice(len(optical), 3, replace=False)
        normal = np.cross(optical[ids[1]]-optical[ids[0]], optical[ids[2]]-optical[ids[0]])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-5:
            continue
        inliers = np.flatnonzero(np.abs((optical-optical[ids[0]])@(normal/norm)) <= .035)
        if len(inliers) > len(best):
            best = inliers
    evidence.update(inlier_count=len(best), support_fraction=len(best)/len(optical))
    if not len(best):
        evidence['reason'] = 'degenerate_inlier_geometry'
    if len(best) < 8 or len(best) <= len(optical)/2:
        return None, evidence
    chosen = optical[best]
    mean = chosen.mean(axis=0)
    _, singular, vh = np.linalg.svd(chosen-mean, full_matrices=False)
    normal = vh[-1]
    incidence = abs(float(normal@(ray/np.linalg.norm(ray))))
    residual = float(np.sqrt(np.mean(((chosen-mean)@normal)**2)))
    evidence.update(residual_rms_m=residual, ray_normal_abs_dot=incidence,
                    second_singular_value_m=float(singular[1]))
    if singular[1] < .02:
        evidence['reason'] = 'degenerate_inlier_geometry'
    elif incidence < .1:
        evidence['reason'] = 'grazing_camera_ray'
    elif residual > .035:
        evidence['reason'] = 'plane_residual_exceeded'
    else:
        # Camera-normalized convex support prevents a side face from supplying
        # the center ray's depth outside the finite surface actually observed.
        hull = cv2.convexHull((chosen[:, :2]/chosen[:, 2:3]).astype(np.float32))
        support_px = float(cv2.pointPolygonTest(hull, tuple(map(float, ray[:2])), True)*min(k[0, 0], k[1, 1]))
        evidence['ray_support_distance_px'] = support_px
        depth = float(normal@mean)/float(normal@ray)
        if support_px < -1.:
            evidence['reason'] = 'camera_ray_outside_plane_support'
        elif not np.isfinite(depth) or depth <= .2 or not optical[:, 2].min()-.15 <= depth <= optical[:, 2].max()+.15:
            evidence['reason'] = 'invalid_plane_intersection'
        else:
            evidence.update(accepted=True, reason='plane_accepted', depth_delta_m=depth-median_depth)
            return depth, evidence
    return None, evidence


def associate_surface(points_world, lidar_origin_world, world_from_camera, detection, info,
                      min_points=3, depth_gap=.8, use_surface_plane=True):
    """Project actual surface returns and combine their depth with camera bearing.

    A depth-separated clutter cluster cannot silently supply object range. The
    dominant cluster must contain at least three returns and account for most
    candidates. Supported planar returns supply the camera-ray intersection;
    otherwise the guarded median depth remains available. The output describes
    an observed surface point, not a hidden target center or simulator pose.
    """
    points = np.asarray(points_world, dtype=float)
    origin = np.asarray(lidar_origin_world, dtype=float)
    pose = np.asarray(world_from_camera, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or origin.shape != (3,)
            or pose.shape != (4, 4) or not np.isfinite(np.r_[origin, pose.ravel()]).all()):
        raise ValueError('invalid_point_cloud_or_transform')
    points = points[np.isfinite(points).all(axis=1)]
    camera = (points-pose[:3, 3])@pose[:3, :3]
    forward = camera[:, 2] > .2
    points, camera = points[forward], camera[forward]
    if not len(camera):
        raise ValueError('no_lidar_in_camera_view')
    k = np.asarray(info['k'], float).reshape(3, 3)
    d = np.asarray(info['d'], float)
    pixels = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k,
                               d if len(d) else None)[0].reshape(-1, 2)
    x, y, w, h = detection['bbox']
    # Allow a one-pixel image/LiDAR discretization residual, not a wide ROI.
    inside = ((pixels[:, 0] >= x-1) & (pixels[:, 0] <= x+w+1)
              & (pixels[:, 1] >= y-1) & (pixels[:, 1] <= y+h+1))
    points, camera, pixels = points[inside], camera[inside], pixels[inside]
    if len(camera) < min_points:
        raise ValueError('insufficient_associated_lidar_points')
    order = np.argsort(camera[:, 2])
    groups = np.split(order, np.flatnonzero(np.diff(camera[order, 2]) > depth_gap)+1)
    groups = sorted(groups, key=len, reverse=True)
    chosen = groups[0]
    if len(chosen) < min_points or (len(groups) > 1 and len(chosen) < 1.5*len(groups[1])):
        raise ValueError('ambiguous_lidar_depth')
    surface, optical, projected = points[chosen], camera[chosen], pixels[chosen]
    depth = float(np.median(optical[:, 2]))
    mad = float(np.median(np.abs(optical[:, 2]-depth)))
    keep = np.abs(optical[:, 2]-depth) <= max(.15, 3*1.4826*mad)
    surface, optical, projected = surface[keep], optical[keep], projected[keep]
    if len(surface) < min_points or np.ptp(optical[:, 2]) > 3.:
        raise ValueError('incoherent_lidar_surface')
    depth = float(np.median(optical[:, 2]))
    center = detection_center(detection)
    ray = optical_bearing(detection['bbox'], info, center)
    plane_depth, surface_fit = surface_plane_depth(optical, ray, depth, k) if use_surface_plane else (
        None, dict(accepted=False, reason='plane_disabled', median_depth_m=depth))
    if plane_depth is not None:
        depth = plane_depth
    position = pose[:3, :3]@(ray*depth)+pose[:3, 3]
    # These are conservative uncertainty assumptions, not calibrated accuracy.
    # Retain a surface-depth floor instead of declaring scatter/sqrt(N) truth.
    depth_std = max(.15, 1.4826*mad)
    px_std = max(1., .03*max(w, h))
    if surface_fit['accepted']:
        incidence = surface_fit['ray_normal_abs_dot']
        # A grazing plane amplifies both fit residual and bearing uncertainty.
        # This adds a conservative range term; the existing .15m floor stays.
        depth_std = max(depth_std, surface_fit['residual_rms_m']/incidence,
                        px_std*depth/min(k[0, 0], k[1, 1])*math.sqrt(max(0., 1/incidence**2-1)))
    jacobian = np.array([[depth/k[0, 0], 0., ray[0]],
                         [0., depth/k[1, 1], ray[1]], [0., 0., 1.]])
    covariance_camera = jacobian@np.diag([px_std**2, px_std**2, depth_std**2])@jacobian.T
    covariance = pose[:3, :3]@covariance_camera@pose[:3, :3].T+np.eye(3)*.02**2
    residual = projected-center
    reprojection = dict(semantics='lidar_bbox_consistency_not_independent_calibration_error',
        median_offset_px=np.median(residual, axis=0).tolist(),
        nearest_center_distance_px=float(np.linalg.norm(residual, axis=1).min()),
        projected_extent_px=np.ptp(projected, axis=0).tolist(),
        bbox_size_px=[float(w), float(h)])
    return dict(position=position.tolist(), covariance=covariance.tolist(),
                nearest_distance_m=float(np.linalg.norm(surface-origin, axis=1).min()),
                bbox=list(detection['bbox']), center_px=center.tolist(), target_size_m=np.maximum(
                    np.ptp(surface, axis=0), .1).tolist(),
                points_count=int(len(surface)), candidates_count=int(len(points)),
                camera_depth_m=depth, surface_depth_mad_m=mad,
                surface_model='dominant_plane_ray' if plane_depth is not None else 'median_depth',
                surface_fit=surface_fit,
                reprojection_consistency=reprojection,
                camera_bearing=ray.tolist(), position_semantics='camera_ray_lidar_surface_depth',
                uncertainty_semantics='conservative_model_not_calibrated_accuracy')


def gimbal_center_command(box, info, yaw, pitch, gain=.65, max_step=.04, center=None):
    """Positive optical x is right; gimbal +yaw is left and +pitch is down."""
    if not np.isfinite([yaw, pitch, gain, max_step]).all() or not 0 < gain <= 1 or max_step <= 0:
        raise ValueError('invalid_gimbal_state_or_gain')
    ray = optical_bearing(box, info, center)
    dyaw = -math.atan2(ray[0], 1.)/max(.2, math.cos(pitch))
    dpitch = math.atan2(ray[1], math.sqrt(1+ray[0]**2))
    return [math.remainder(yaw+float(np.clip(gain*dyaw, -max_step, max_step)), 2*math.pi),
            float(np.clip(pitch+np.clip(gain*dpitch, -max_step, max_step), -.65, .65))]


def lidar_acquisition_candidate(points_world, lidar_origin_world):
    """Measured compact airborne cluster; camera-search proposal, never a lock.

    Whole clusters retain their bottom/extent so tall grounded clutter cannot
    become a floating proposal merely because an altitude crop removed its base.
    This single-target experiment assumes a compact, isolated airborne marker.
    """
    points, origin = np.asarray(points_world, float), np.asarray(lidar_origin_world, float)
    if points.ndim != 2 or points.shape[1] != 3 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('invalid_acquisition_cloud')
    points = points[np.isfinite(points).all(axis=1)]
    distance = np.linalg.norm(points-origin, axis=1)
    points = voxel_filter(points[(distance >= 3.) & (distance <= 65.) & (points[:, 2] > origin[2]-.5)], .15)
    # Existing DBSCAN is bounded here; dense/complex scenes need the background
    # subtraction pipeline rather than a broader unbounded search.
    if not 3 <= len(points) <= 4000:
        return None
    labels = dbscan(points, .8, 3)
    candidates = []
    for label in np.unique(labels[labels >= 0]):
        cluster = points[labels == label]
        extent = np.ptp(cluster, axis=0)
        if (len(cluster) >= 3 and cluster[:, 2].min() > origin[2]+1.
                and extent.max() <= 2.5 and np.count_nonzero(extent >= .15) >= 2):
            center = np.median(cluster, axis=0)
            candidates.append((np.linalg.norm(center-origin), center))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def lidar_acquisition_command(point_world, world_from_base, pivot_world):
    """Point only the camera using measured TF; camera confirmation is mandatory."""
    point, pose, pivot = np.asarray(point_world, float), np.asarray(world_from_base, float), np.asarray(pivot_world, float)
    if point.shape != (3,) or pose.shape != (4, 4) or pivot.shape != (3,) or not np.isfinite(np.r_[point, pose.ravel(), pivot]).all():
        raise ValueError('invalid_acquisition_transform')
    relative = pose[:3, :3].T@(point-pivot)
    horizontal = float(np.linalg.norm(relative[:2]))
    if horizontal < .2:
        return None
    pitch = -math.atan2(float(relative[2]), horizontal)
    if abs(pitch) > math.pi/6:
        return None
    return [math.atan2(float(relative[1]), float(relative[0])), pitch]
