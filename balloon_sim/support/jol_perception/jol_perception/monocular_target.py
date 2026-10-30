"""Known-size spherical target geometry; no LiDAR or simulator coordinates."""
import math
import cv2
import numpy as np
from .fusion_core import calibration_error
from .static_estimator_core import detection_center, detect_orange_marker


def detect_balloon(image, minimum_pixels=8):
    """Controlled orange simulation balloon, not a trained balloon detector."""
    detections=detect_orange_marker(image,minimum_pixels)
    if not detections:return []
    detection=detections[0]
    mask=cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),(3,100,70),(28,255,255))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if contours:
        contour=max(contours,key=cv2.contourArea)
        if len(contour)>=10:
            center,axes,_=cv2.fitEllipse(contour)
            detection.update(center_px=list(center),diameter_px=float(np.mean(axes)+1.),
                diameter_method='ellipse_contour_pixel_support')
    detection.update(class_name='orange_balloon',source='controlled_balloon_color_marker')
    return detections


def estimate_sphere(detection, info, world_camera, *, diameter_m=.3, diameter_std_m=.01,
                    pixel_std=.5, minimum_diameter_px=12., maximum_std_m=.5,
                    translation_std_m=.02, rotation_std_deg=.1):
    """A known metric diameter resolves scale; approximate size increases covariance.

    Assumes a round silhouette near the optical axis. The real balloon's size,
    shape and camera calibration must be measured before hardware use.
    """
    parameters=np.array([diameter_m,diameter_std_m,pixel_std,minimum_diameter_px,maximum_std_m,
        translation_std_m,rotation_std_deg])
    pose=np.asarray(world_camera,float);box=np.asarray(detection['bbox'],float)
    if (not np.isfinite(parameters).all()or diameter_m<=0 or diameter_std_m<0 or pixel_std<=0
            or minimum_diameter_px<3 or not 0<maximum_std_m<=.5
            or translation_std_m<0 or rotation_std_deg<0
            or pose.shape!=(4,4)or not np.isfinite(pose).all()
            or not np.allclose(pose[3],[0,0,0,1])
            or not np.allclose(pose[:3,:3]@pose[:3,:3].T,np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3,:3]),1.)
            or box.shape!=(4,)or not np.isfinite(box).all()or min(box[2:])<=0):
        raise ValueError('invalid_monocular_geometry')
    if calibration_error(info,info['width'],info['height'],info['frame_id']):
        raise ValueError('calibrated_camera_required')
    if (detection.get('box_clipped')or min(box[:2])<0
            or box[0]+box[2]>info['width']or box[1]+box[3]>info['height']
            or not .65<=box[2]/box[3]<=1.5):
        raise ValueError('clipped_or_nonround_balloon')
    center=detection_center(detection);apparent=float(detection.get('diameter_px',np.mean(box[2:])))
    if not math.isfinite(apparent)or apparent<minimum_diameter_px:
        raise ValueError('camera_target_too_small_for_range')
    k=np.asarray(info['k'],float).reshape(3,3);focal=(k[0,0]+k[1,1])/2
    def optical(measurement):
        u,v,pixels=measurement
        ray=np.array([(u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1],1.])
        alpha=math.atan(pixels/(2*focal))
        return ray/np.linalg.norm(ray)*(diameter_m/(2*math.sin(alpha)))
    measured=np.r_[center,apparent];point=optical(measured);step=1e-3
    jacobian=np.column_stack([(optical(measured+np.eye(3)[i]*step)-
        optical(measured-np.eye(3)[i]*step))/(2*step)for i in range(3)])
    covariance=jacobian@jacobian.T*pixel_std**2
    covariance+=np.outer(point,point)*(diameter_std_m/diameter_m)**2
    rotation=pose[:3,:3];lever=rotation@point
    covariance=rotation@covariance@rotation.T+np.eye(3)*translation_std_m**2
    x,y,z=lever;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    covariance+=skew@skew.T*math.radians(rotation_std_deg)**2
    uncertainty=math.sqrt(max(0.,float(np.linalg.eigvalsh(covariance).max())))
    if uncertainty>maximum_std_m:raise ValueError('camera_range_uncertainty_exceeded')
    return dict(position=(lever+pose[:3,3]).tolist(),covariance=covariance.tolist(),
        source='camera_monocular_size',position_semantics='known_sphere_center',
        optical_range_m=float(np.linalg.norm(point)),camera_bearing=(point/point[2]).tolist(),
        maximum_position_std_m=uncertainty,diameter_m=diameter_m,diameter_std_m=diameter_std_m,
        observed_diameter_px=apparent,assumptions='known_sphere_diameter_and_calibrated_pinhole')
