"""Camera-only acquisition/lock source for the isolated ros2_nub simulation."""
import json,math,time
import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import TransformException
from .fusion_core import image_bgr
from .static_estimator import StaticEstimator,stamp_seconds
from .static_estimator_core import validate_detection,calibration_revision,pose_motion
from .monocular_target import detect_balloon,estimate_sphere


class CameraOnlyEstimator(StaticEstimator):
    def __init__(self):
        super().__init__(camera_only=True)
        if not self.get_parameter('use_sim_time').value:raise ValueError('camera_only_simulation_required')
        if self.detector_mode!='orange_marker':raise ValueError('trained_balloon_model_not_configured')
        readonly=ParameterDescriptor(read_only=True)
        self.diameter=float(self.declare_parameter('balloon_diameter_m',.3,readonly).value)
        self.diameter_std=float(self.declare_parameter('balloon_diameter_std_m',.01,readonly).value)
        self.search_elevation=float(self.declare_parameter('search_elevation_deg',12.,readonly).value)
        if not 0<self.diameter<=2 or not 0<=self.diameter_std<=.2 or not 0<=self.search_elevation<=30:
            raise ValueError('invalid_balloon_assumptions')
        self.get_logger().info('NUB camera-only; known-size sphere; no LiDAR subscription or truth input')

    def start_acquisition(self,request,response):
        response=super().start_acquisition(request,response)
        if response.success:
            self.set_acquisition_stage('camera_search',self.now())
            self.last_reason='searching_camera_balloon'
        return response

    def receive_image(self,msg):
        current=self.now();stamp=stamp_seconds(msg.header.stamp)
        if not 0<stamp<=current+.1 or self.last_image_stamp is not None and stamp<=self.last_image_stamp:return
        self.last_image_stamp=stamp
        try:
            image=image_bgr(msg);detections=detect_balloon(image,self.marker_minimum_pixels)
            packet=dict(stamp=stamp,frame_id=msg.header.frame_id,width=msg.width,height=msg.height,
                detections=detections,model='controlled_orange_balloon_not_neural',source='camera_only')
            self.last_image=(msg,image);self.detection_pub.publish(String(data=json.dumps(packet)))
            self.pending_detection=(self.correct_detection_time(packet),time.monotonic())
            self.counters['images_processed']+=1;self.annotate(packet)
        except(ValueError,TypeError,cv2.error)as error:self.invalidate(str(error),current,stamp)

    def search_gimbal(self,current):
        if (not self.acquisition_started or self.gimbal is None or self.info is None
                or not 0<=current-self.gimbal['stamp']<=.3
                or self.last_search is not None and current-self.last_search<.2
                or self.last_visual_stamp is not None and current-self.last_visual_stamp<=self.max_age):return
        self.last_search=current;self.set_acquisition_stage('camera_search',current)
        command=JointState();command.header.stamp=self.get_clock().now().to_msg()
        command.name=['gimbal_yaw_joint','gimbal_pitch_joint']
        command.position=[self.gimbal['yaw']-.2,-math.radians(self.search_elevation)]
        self.command_pub.publish(command)

    def invalidate(self,reason,current,source_stamp=None):
        self.last_result=None;self.last_reason=reason
        self.estimate_pub.publish(String(data=json.dumps(dict(valid=False,reason=reason,
            stamp=source_stamp or current,publication_stamp=current,frame_id=self.fixed_frame,
            target_id=1,position=None,covariance=None,source='camera_monocular_size'))))

    def tick(self):
        current=self.now()
        if not self.acquisition_started:
            self.pending_detection=self.last_result=None;self.drive_initial_gimbal(current)
            self.last_reason='waiting_for_acquisition_start';return
        if self.last_result is not None and current-self.last_result['stamp']>self.max_age:
            self.invalidate('camera_estimate_stale',current)
        if (not self.get_parameter('calibration_valid').value
                or self.get_parameter('calibration_source').value in ('','unspecified')):
            self.pending_detection=None
            if self.last_result is not None or self.last_reason!='calibration_not_confirmed':
                self.invalidate('calibration_not_confirmed',current)
            return
        if self.pending_detection is None:
            self.search_gimbal(current);return
        packet,received=self.pending_detection
        if self.info is None:return
        try:
            source=self.get_parameter('calibration_source').value
            if not self.get_parameter('calibration_valid').value or source in('','unspecified'):
                raise ValueError('calibration_not_confirmed')
            if time.monotonic()-received>self.max_age:raise ValueError('camera_result_wall_stale')
            if .5e-9<packet['stamp']-current<=.1+.5e-9:return
            detection=validate_detection(packet,self.info,current,self.max_age)
            self.last_visual_stamp=packet['stamp'];self.center_gimbal(packet,detection,current)
            pose=self.lookup_matrix(self.fixed_frame,packet['frame_id'],packet['stamp'])
            mounts=self.static_calibration_transforms(packet['stamp'])
            revision=calibration_revision(self.info,source+f'/sphere/{self.diameter}/{self.diameter_std}',self.sensor_quality,mounts)
            if self.calibration_revision and revision!=self.calibration_revision:
                self.invalidate('calibration_changed',current)
                self.calibration_revision=revision
                self.pending_detection=None
                return
            self.calibration_revision=revision
            result=estimate_sphere(detection,self.info,pose,diameter_m=self.diameter,
                diameter_std_m=self.diameter_std,
                translation_std_m=self.sensor_quality['extrinsic_translation_std_m'],
                rotation_std_deg=self.sensor_quality['extrinsic_rotation_std_deg'],
                maximum_std_m=min(.5,self.sensor_quality['max_position_std_m']))
            motion=pose_motion(self.lookup_matrix(self.fixed_frame,packet['frame_id'],packet['stamp']-.02),pose,.02)
            jacobian=np.asarray(motion[0])+np.cross(motion[1],np.asarray(result['position'])-pose[:3,3])
            covariance=np.asarray(result['covariance'])+np.outer(jacobian,jacobian)*self.sensor_quality['camera_time_std_s']**2
            if np.linalg.eigvalsh(covariance).max()>min(.5,self.sensor_quality['max_position_std_m'])**2:
                raise ValueError('camera_range_uncertainty_exceeded')
            result['covariance']=covariance.tolist();bearing=np.asarray(result['camera_bearing'])
            result['maximum_position_std_m']=float(np.sqrt(np.linalg.eigvalsh(covariance).max()))
            alignment=math.degrees(math.atan2(float(np.linalg.norm(bearing[:2])),float(bearing[2])))
            tracking=bool(self.gimbal and 0<=current-self.gimbal['stamp']<=.3)
            evidence=dict(gimbal_centered=tracking and alignment<=self.center_tolerance_deg,
                gimbal_tracking=tracking,camera_alignment_error_deg=alignment,image_size=[packet['width'],packet['height']],
                camera_origin_world=pose[:3,3].tolist(),camera_tf_stamp=packet['stamp'],exposure_stamp=packet['stamp'],
                image_source_stamp=packet.get('source_stamp',packet['stamp']),calibration_fixed_transforms=mounts,
                target_is_static=True,ground_truth_used=False,lidar_used=False,
                known_diameter_assumption_m=self.diameter,diameter_std_m=self.diameter_std,
                configured_not_measured_hardware_accuracy=True)
            self.last_result=dict(result,valid=True,reason='ready',stamp=packet['stamp'],publication_stamp=current,
                frame_id=self.fixed_frame,target_id=1,bbox=detection['bbox'],evidence=evidence,
                calibration_source=source,calibration_revision=revision)
            self.last_reason='ready';self.last_processed=packet['stamp']
            self.set_acquisition_stage('ready'if evidence['gimbal_centered']else'visual_centering',current)
            self.estimate_pub.publish(String(data=json.dumps(self.last_result,allow_nan=False)))
            self.publish_marker(self.last_result);self.counters['coordinate_updates']+=1
        except(ValueError,KeyError,TypeError,cv2.error)as error:
            self.invalidate(str(error),current,packet.get('stamp'))
            if str(error)=='no_detection':self.search_gimbal(current)
        except TransformException:self.last_reason='exposure_tf_unavailable';return
        self.pending_detection=None


def main(args=None):
    cv2.setNumThreads(1);rclpy.init(args=args);node=CameraOnlyEstimator()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
