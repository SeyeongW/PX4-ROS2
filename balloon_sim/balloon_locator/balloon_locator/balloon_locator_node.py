#!/usr/bin/env python3
"""Static balloon observations with exposure-time pose and honest source freshness.

Hardware uses the configured gimbal bearing. TF mode uses calibrated, rectified
CameraInfo and world<-optical at exposure time; its station NED is [world_y,
world_x,-world_z], origin world(0,0,0), north world+Y. No truth input exists.
"""
import bisect
import hashlib
import json
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Vector3Stamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

from .geometry import (calibrated_intrinsics, focal_px, gimbal_to_azel,
                       locate_balloon, measurement_covariance, optical_ray,
                       range_from_diameter)
from .static_estimator import StaticTargetEstimator


class BalloonLocatorNode(Node):
    def __init__(self):
        super().__init__('balloon_locator_node')
        defaults = dict(image_width=1920.0,image_height=1080.0,hfov_deg_at_1x=80.0,
            balloon_diameter_m=.30,base_heading_deg=0.0,yaw_sign=1.0,yaw_zero_deg=0.0,
            pitch_sign=1.0,pitch_zero_deg=0.0,roll_sign=1.0,roll_zero_deg=0.0,
            gimbal_rotation_order='zxy',offset_ned=[0.,0.,0.],pixel_sigma_px=1.0,
            diameter_rel_sigma=.05,focal_rel_sigma=.01,diameter_bias_sigma_px=.25,
            angle_sigma_deg=.2,extrinsic_translation_std_m=.02,
            camera_time_std_s=.001,max_range_m=300.0,min_diameter_px=8.0,
            max_attitude_gap_sec=.3,forget=1.0,position_topic='/balloon/position_ned',
            frame_id='local_ned',pose_source='gimbal',fixed_frame='world',
            camera_info_topic='/camera/camera_info',max_measurement_age_sec=.8,
            min_samples=3,max_sigma_m=.5,center_tolerance_deg=1.0,calibration_valid=False,
            require_camera_info=True)
        for name,value in defaults.items():self.declare_parameter(name,value)
        self.pose_source=self.value('pose_source')
        if self.pose_source not in ('gimbal','tf'):raise ValueError('pose_source must be gimbal or tf')
        positive=('image_width','image_height','balloon_diameter_m','pixel_sigma_px',
                  'max_range_m','min_diameter_px','max_measurement_age_sec','max_sigma_m')
        nonnegative=('diameter_rel_sigma','focal_rel_sigma','diameter_bias_sigma_px',
                     'angle_sigma_deg','extrinsic_translation_std_m','camera_time_std_s')
        if (any(not math.isfinite(self.value(n)) or self.value(n)<=0 for n in positive)
                or any(not math.isfinite(self.value(n)) or self.value(n)<0 for n in nonnegative)
                or self.value('min_samples')<1
                or self.value('gimbal_rotation_order') not in ('zxy','zyx')
                or self.value('roll_sign') not in (-1.,1.)
                or not math.isfinite(self.value('roll_zero_deg'))):raise ValueError('invalid_locator_configuration')
        self.est=StaticTargetEstimator(forget=self.value('forget'))
        self.att=deque(maxlen=200);self.zoom=1.;self.info=None
        self.pending=None;self.measurement=None;self.last_seen_stamp=None;self.last_clock=None
        self.last_published_stamp=None;self.revision=None;self.reason='waiting_for_detection'
        self.skipped=dict(no_att=0,range=0,small=0,stale=0,invalid=0,tf=0)
        self.pub_pt=self.create_publisher(PointStamped,self.value('position_topic'),10)
        self.pub_est=self.create_publisher(PoseWithCovarianceStamped,'/balloon/estimate',10)
        self.pub_status=self.create_publisher(String,'/balloon/status',10)
        self.create_subscription(Detection2DArray,'/balloon/detections',self._on_det,10)
        self.create_subscription(Vector3Stamped,'/gimbal/attitude',self._on_att,50)
        self.create_subscription(Float32,'/gimbal/zoom',self._on_zoom,10)
        self.create_subscription(CameraInfo,self.value('camera_info_topic'),self._on_info,qos_profile_sensor_data)
        self.create_service(Trigger,'/balloon/reset',self._on_reset)
        self.tf=Buffer(cache_time=Duration(seconds=10.)) if self.pose_source=='tf' else None
        self.listener=TransformListener(self.tf,self) if self.tf else None
        self.create_timer(.02,self._process_pending)
        self.create_timer(.1,self._publish)

    def value(self,name):return self.get_parameter(name).value

    def _now(self):
        current=self.get_clock().now().nanoseconds*1e-9
        if self.last_clock is not None and current<self.last_clock-.001:
            self.est.reset();self.pending=self.measurement=self.last_seen_stamp=None
            self.last_published_stamp=None;self.att.clear();self.reason='clock_reset'
        self.last_clock=current
        return current

    def _on_reset(self,_req,resp):
        self.est.reset();self.pending=self.measurement=self.last_seen_stamp=None
        self.last_published_stamp=None;self.reason='reset'
        resp.success=True;resp.message='reset';return resp

    def _on_info(self,msg):
        self.info=dict(k=list(msg.k),d=list(msg.d),width=msg.width,height=msg.height,
            frame_id=msg.header.frame_id,distortion_model=msg.distortion_model,
            binning_x=msg.binning_x,binning_y=msg.binning_y,
            roi_offset=[msg.roi.x_offset,msg.roi.y_offset])

    def _on_zoom(self,msg):
        if math.isfinite(msg.data) and msg.data>=1.:self.zoom=float(msg.data)

    def _on_att(self,msg):
        stamp=Time.from_msg(msg.header.stamp).nanoseconds*1e-9
        if (np.isfinite([stamp,msg.vector.x,msg.vector.y,msg.vector.z]).all() and stamp>0
                and (not self.att or stamp>self.att[-1][0])):
            self.att.append((stamp,msg.vector.x,msg.vector.y,msg.vector.z))

    def _attitude_at(self,t):
        if len(self.att)<2:return None
        i=bisect.bisect_left([a[0] for a in self.att],t)
        if i==0:
            return self.att[0][1:] if abs(self.att[0][0]-t)<1e-9 else None
        if i>=len(self.att):return None
        (t0,y0,p0,r0),(t1,y1,p1,r1)=self.att[i-1],self.att[i]
        if t1-t0>self.value('max_attitude_gap_sec'):return None
        a=(t-t0)/(t1-t0);dy=(y1-y0+180.)%360.-180.
        dr=(r1-r0+180.)%360.-180.
        return y0+a*dy,p0+a*(p1-p0),r0+a*dr

    def _on_det(self,msg):
        current=self._now();stamp=Time.from_msg(msg.header.stamp).nanoseconds*1e-9
        if (not 0<stamp<=current+.1 or current-stamp>self.value('max_measurement_age_sec')
                or self.last_seen_stamp is not None and stamp<=self.last_seen_stamp):
            self.skipped['stale']+=1;return
        self.last_seen_stamp=stamp
        if not msg.detections:
            self.pending=None;self.reason='no_detection';return
        self.pending=(msg,time.monotonic())

    @staticmethod
    def _matrix(transform):
        p,q=transform.transform.translation,transform.transform.rotation
        x,y,z,w=q.x,q.y,q.z,q.w
        if not np.isfinite([p.x,p.y,p.z,x,y,z,w]).all() or abs(x*x+y*y+z*z+w*w-1.)>1e-3:
            raise ValueError('invalid_camera_transform')
        rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        return rotation,np.array([p.x,p.y,p.z])

    def _tf_pose(self,frame,stamp):
        if not frame or frame==self.value('fixed_frame'):raise ValueError('optical_frame_required')
        return self._matrix(self.tf.lookup_transform(self.value('fixed_frame'),frame,
            Time(nanoseconds=round(stamp*1e9))))

    def _calibration_ready(self):
        if self.value('calibration_valid'):return True
        # Reconfirming calibration requires new evidence, never old accumulated
        # samples that happened to precede the calibration loss.
        self.est.reset();self.pending=self.measurement=None;self.last_published_stamp=None
        self.reason='calibration_not_confirmed'
        return False

    def _process_pending(self):
        current=self._now()
        if not self._calibration_ready():return
        if self.pending is None:return
        msg,received=self.pending;stamp=Time.from_msg(msg.header.stamp).nanoseconds*1e-9
        age=self.value('max_measurement_age_sec')
        if current-stamp>age or time.monotonic()-received>age:
            self.pending=None;self.reason='stale_detection';self.skipped['stale']+=1;return
        if stamp>current:return
        if self.measurement and (stamp-self.measurement['stamp']>age
                or received-self.measurement['received']>age):
            self.est.reset();self.measurement=None;self.last_published_stamp=None
        try:
            det=msg.detections[0];u=det.bbox.center.position.x;v=det.bbox.center.position.y
            diameter=(det.bbox.size_x+det.bbox.size_y)/2
            if (not np.isfinite([u,v,diameter,det.bbox.size_x,det.bbox.size_y]).all()
                    or min(det.bbox.size_x,det.bbox.size_y)<=0
                    or not .65<=det.bbox.size_x/det.bbox.size_y<=1.5):raise ValueError('invalid_detection')
            if diameter<self.value('min_diameter_px'):
                self.skipped['small']+=1;raise ValueError('target_too_small')
            intrinsics=calibrated_intrinsics(self.info) if self.info else None
            if (self.pose_source=='tf' or self.value('require_camera_info')) and intrinsics is None:
                self.reason='waiting_for_camera_info';return
            if self.info and (not self.info['frame_id'] or self.info['frame_id']!=msg.header.frame_id):
                raise ValueError('camera_frame_mismatch')
            width,height=(self.info['width'],self.info['height']) if self.info else (int(self.value('image_width')),int(self.value('image_height')))
            if min(u-diameter/2,v-diameter/2)<0 or u+diameter/2>width or v+diameter/2>height:
                raise ValueError('clipped_detection')
            calibration=dict(info=self.info,pose_source=self.pose_source,zoom=self.zoom if intrinsics is None else None,
                configuration={name:self.value(name) for name in ('balloon_diameter_m','diameter_rel_sigma',
                'focal_rel_sigma','diameter_bias_sigma_px','angle_sigma_deg','extrinsic_translation_std_m','hfov_deg_at_1x','base_heading_deg',
                'yaw_sign','yaw_zero_deg','pitch_sign','pitch_zero_deg','roll_sign','roll_zero_deg',
                'gimbal_rotation_order','offset_ned')})
            revision=hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest()
            if self.revision and revision!=self.revision:
                self.est.reset();self.measurement=None;self.last_published_stamp=None
            self.revision=revision
            focal=(intrinsics[0]+intrinsics[1])/2 if intrinsics else focal_px(width,self.value('hfov_deg_at_1x'),self.zoom)
            rng=range_from_diameter(diameter,self.value('balloon_diameter_m'),focal)
            if rng>self.value('max_range_m'):
                self.skipped['range']+=1;raise ValueError('target_out_of_range')
            timing=np.zeros((3,3));origin_world=None
            if self.pose_source=='tf':
                ray_optical=optical_ray(u,v,intrinsics)
                rotation,origin_world=self._tf_pose(msg.header.frame_id,stamp)
                ray_world=rotation@ray_optical;world=origin_world+rng*ray_world
                enu_ned=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
                ray=enu_ned@ray_world;pos=enu_ned@world
                old_rotation,old_origin=self._tf_pose(msg.header.frame_id,stamp-.02)
                # Differentiate the same measured optical lever, including both
                # camera-origin motion and image-pose timing during a slew.
                motion=(origin_world-old_origin+(rotation-old_rotation)@(rng*ray_optical))/.02
                motion=enu_ned@motion
                timing=np.outer(motion,motion)*self.value('camera_time_std_s')**2
                alignment=math.degrees(math.acos(np.clip(ray_optical[2],-1.,1.)))
            else:
                ya=self._attitude_at(stamp)
                if ya is None:self.reason='waiting_for_exposure_attitude';self.skipped['no_att']+=1;return
                az,el=gimbal_to_azel(ya[0],ya[1],self.value('base_heading_deg'),self.value('yaw_sign'),
                    self.value('yaw_zero_deg'),self.value('pitch_sign'),self.value('pitch_zero_deg'))
                # SIYI uses 312 (yaw, roll, pitch), also implemented by
                # ArduPilot/libraries/AP_Mount/AP_Mount_Siyi.cpp. Signs and
                # mounting zeros still require this physical unit's calibration.
                roll=self.value('roll_sign')*(ya[2]-self.value('roll_zero_deg'))
                pos,_,rng=locate_balloon(u,v,diameter,width_px=width,height_px=height,
                    hfov_deg_at_1x=self.value('hfov_deg_at_1x'),zoom=self.zoom,az_deg=az,el_deg=el,
                    balloon_diameter_m=self.value('balloon_diameter_m'),offset_ned=self.value('offset_ned'),intrinsics=intrinsics,
                    roll_deg=roll,rotation_order=self.value('gimbal_rotation_order'))
                ray=(pos-np.asarray(self.value('offset_ned')))/rng
                cx,cy=(intrinsics[2],intrinsics[3]) if intrinsics else (width/2,height/2)
                alignment=math.degrees(math.atan2(math.hypot(u-cx,v-cy),focal))
            random=measurement_covariance(ray,rng,diameter,focal,pixel_sigma_px=self.value('pixel_sigma_px'),
                diameter_rel_sigma=0.,angle_sigma_deg=0.)+timing+np.eye(3)*1e-10
            common=measurement_covariance(ray,rng,diameter,focal,pixel_sigma_px=self.value('diameter_bias_sigma_px'),
                diameter_rel_sigma=math.hypot(self.value('diameter_rel_sigma'),self.value('focal_rel_sigma')),
                angle_sigma_deg=self.value('angle_sigma_deg'))
            common+=np.eye(3)*self.value('extrinsic_translation_std_m')**2
            if self.est.update(pos,random,systematic_covariance=common):
                self.measurement=dict(stamp=stamp,received=received,bbox=[u-diameter/2,v-diameter/2,diameter,diameter],
                    image_size=[int(width),int(height)],camera_centered=alignment<=self.value('center_tolerance_deg'),
                    alignment_error_deg=alignment,camera_origin_world=origin_world.tolist() if origin_world is not None else None,
                    observed_diameter_px=diameter,range_m=rng)
                self.reason='ready'
            else:self.reason='measurement_rejected'
            self.pending=None
        except TransformException:
            self.reason='waiting_for_exposure_tf';self.skipped['tf']+=1
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as error:
            self.reason=str(error);self.pending=None;self.skipped['invalid']+=1

    def _publish(self):
        current=self._now();calibrated=self._calibration_ready();measurement=self.measurement
        source_stamp=measurement['stamp'] if measurement else None
        wall_age=time.monotonic()-measurement['received'] if measurement else None
        valid=bool(measurement and calibrated and self.est.n>=self.value('min_samples')
            and 0<=current-source_stamp<=self.value('max_measurement_age_sec')
            and wall_age<=self.value('max_measurement_age_sec') and self.est.sigma<=self.value('max_sigma_m'))
        reason=('ready' if valid else 'calibration_not_confirmed' if not self.value('calibration_valid') else
            'measurement_stale' if measurement and (current-source_stamp>self.value('max_measurement_age_sec')
                or wall_age>self.value('max_measurement_age_sec')) else
            'uncertainty_exceeded' if self.est.n and self.est.sigma>self.value('max_sigma_m') else
            'insufficient_samples' if measurement and self.est.n<self.value('min_samples') else self.reason)
        mean,cov=self.est.mean,self.est.cov
        if valid and source_stamp!=self.last_published_stamp:
            pt=PointStamped();pt.header.stamp=Time(nanoseconds=round(source_stamp*1e9)).to_msg()
            pt.header.frame_id=self.value('frame_id');pt.point.x,pt.point.y,pt.point.z=map(float,mean)
            self.pub_pt.publish(pt)
            pc=PoseWithCovarianceStamped();pc.header=pt.header;pc.pose.pose.position=pt.point;pc.pose.pose.orientation.w=1.
            full=np.zeros((6,6));full[:3,:3]=cov;pc.pose.covariance=full.flatten().tolist();self.pub_est.publish(pc)
            self.last_published_stamp=source_stamp
        status=dict(stamp=current,valid=valid,reason=reason,measurement_stamp=source_stamp,
            measurement_wall_age=wall_age,n=self.est.n,sigma_m=self.est.sigma if self.est.n else None,
            rejects=self.est.rejects,skipped=self.skipped,zoom=self.zoom,ned=mean.tolist() if mean is not None else None,
            covariance=cov.tolist() if cov is not None else None,frame_id=self.value('frame_id'),
            calibration_revision=self.revision,calibration_valid=bool(self.value('calibration_valid')),
            pose_source=self.pose_source,ground_truth_used=False,
            station_offset_ned=self.value('offset_ned') if self.pose_source=='gimbal' else None,
            station_origin_world=[0.,0.,0.] if self.pose_source=='tf' else None,
            north_yaw_world_rad=math.pi/2 if self.pose_source=='tf' else None)
        if measurement:status.update({k:v for k,v in measurement.items() if k not in ('stamp','received')})
        self.pub_status.publish(String(data=json.dumps(status,allow_nan=False)))


def main(args=None):
    rclpy.init(args=args);node=BalloonLocatorNode()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
