"""Simulation adapter: balloon_locator station NED -> existing UI/flight interface.

This node never derives a target range, reads Gazebo truth or commands flight.
PX4-local NED conversion remains in the existing flight mission, using its EKF origin.
"""
from array import array
from collections import deque
import json
import math
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Vector3
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker,MarkerArray
from rclpy.duration import Duration

NED_TO_WORLD=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])


def mission_estimate(status,now,started_at,status_age=0.):
    """Reject stale/unready coordinates before adapting the documented station basis."""
    if status.get('valid') is not True:raise ValueError(status.get('reason','balloon_coordinate_invalid'))
    stamp=float(status['measurement_stamp'])
    if not started_at<=stamp<=now or now-stamp>.8:raise ValueError('balloon_measurement_stale')
    wall_age=float(status['measurement_wall_age'])
    if not 0<=status_age<=.8 or not 0<=wall_age+status_age<=.8:
        raise ValueError('balloon_measurement_wall_stale')
    if (status.get('calibration_valid') is not True or not isinstance(status.get('calibration_revision'),str)
            or not status['calibration_revision']):raise ValueError('balloon_calibration_unconfirmed')
    if status.get('frame_id')!='local_ned':raise ValueError('station_frame_label_mismatch')
    if status.get('pose_source')!='tf':raise ValueError('simulation_requires_measured_camera_tf')
    if status.get('station_origin_world')!=[0.,0.,0.] or not math.isclose(status.get('north_yaw_world_rad',0.),math.pi/2):
        raise ValueError('station_frame_unconfirmed')
    point=np.asarray(status['ned'],float);cov=np.asarray(status['covariance'],float)
    if point.shape!=(3,) or cov.shape!=(3,3) or not np.isfinite(point).all() or not np.isfinite(cov).all():
        raise ValueError('invalid_balloon_coordinate')
    if not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min()<0 or np.linalg.eigvalsh(cov).max()>.25:
        raise ValueError('balloon_coordinate_uncertainty')
    # These fields also reach UI drawing and allow_nan=False JSON publication.
    for name,size in (('bbox',4),('image_size',2),('camera_origin_world',3)):
        value=np.asarray(status[name],float)
        if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError('invalid_camera_'+name)
    x,y,width,height=status['bbox'];image_width,image_height=status['image_size']
    alignment=float(status['alignment_error_deg'])
    if (min(width,height,image_width,image_height)<=0 or min(x,y)<0
            or x+width>image_width or y+height>image_height
            or type(status['camera_centered']) is not bool or not math.isfinite(alignment) or alignment<0):
        raise ValueError('invalid_camera_alignment_or_bbox')
    diameter=status.get('observed_diameter_px')
    if diameter is not None and (not math.isfinite(diameter) or diameter<=0):raise ValueError('invalid_camera_diameter')
    return dict(valid=True,reason='ready',stamp=stamp,publication_stamp=now,frame_id='world',target_id=1,
        position=(NED_TO_WORLD@point).tolist(),covariance=(NED_TO_WORLD@cov@NED_TO_WORLD.T).tolist(),
        source='camera_monocular_size',position_semantics='known_sphere_center',bbox=status['bbox'],
        calibration_revision=status['calibration_revision'],calibration_source='gazebo_balloon_locator_camera_info_tf',
        maximum_position_std_m=float(np.sqrt(np.linalg.eigvalsh(cov).max())),
        observed_diameter_px=status.get('observed_diameter_px'),
        evidence=dict(estimator='balloon_locator',image_size=status['image_size'],
            gimbal_centered=status['camera_centered'],gimbal_tracking=True,
            camera_alignment_error_deg=status['alignment_error_deg'],camera_tf_stamp=stamp,
            image_source_stamp=stamp,exposure_stamp=stamp,target_is_static=True,
            lidar_used=False,ground_truth_used=False,camera_origin_world=status.get('camera_origin_world')))


class GazeboBridge(Node):
    def __init__(self):
        super().__init__('balloon_gazebo_bridge')
        if not self.get_parameter('use_sim_time').value:raise ValueError('gazebo_bridge_requires_simulation_time')
        self.started=False;self.start_stamp=None;self.restoring=False;self.initial=None
        self.joints=None;self.joint_stamp=None;self.status={};self.status_wall=0.
        self.last_visual=None;self.rate=None;self.rate_wall=0.;self.last_tick=None
        self.reset_generation=0;self.pending_reset=None;self.last_clock=None;self.stage='waiting_start';self.reason='waiting_for_acquisition_start'
        self.images=deque(maxlen=12);self.last_detection_stamp=None
        self.estimate_pub=self.create_publisher(String,'/jol/static_target/estimate',10)
        self.status_pub=self.create_publisher(String,'/jol/static_target/status',10)
        self.marker_pub=self.create_publisher(MarkerArray,'/jol/static_target/markers',10)
        self.command_pub=self.create_publisher(JointState,'/jol/gimbal/command',10)
        self.annotated_pub=self.create_publisher(Image,'/jol/camera/annotated',qos_profile_sensor_data)
        self.detection_pub=self.create_publisher(String,'/jol/camera/detections',10)
        self.reset_client=self.create_client(Trigger,'/balloon/reset')
        self.create_subscription(String,'/balloon/status',self.receive_status,10)
        self.create_subscription(JointState,'/jol/gimbal/joint_states',self.receive_joints,10)
        self.create_subscription(Vector3,'/gimbal/cmd_rate',self.receive_rate,10)
        self.create_subscription(Image,'/jol/camera/image_raw',self.receive_image,qos_profile_sensor_data)
        self.create_subscription(Detection2DArray,'/balloon/detections',self.receive_detection,10)
        self.create_service(Trigger,'/jol/static_target/start_acquisition',self.start)
        self.create_service(Trigger,'/jol/static_target/reset_acquisition',self.reset)
        self.create_timer(.05,self.tick)

    def now(self):return self.get_clock().now().nanoseconds*1e-9
    def receive_status(self,msg):
        try:
            packet=json.loads(msg.data)
            if not isinstance(packet,dict):raise ValueError('status_must_be_object')
            self.status=packet;self.status_wall=time.monotonic()
        except(ValueError,TypeError):self.status={}
    def receive_joints(self,msg):
        values=dict(zip(msg.name,msg.position))
        if all(k in values and math.isfinite(values[k]) for k in ('gimbal_yaw_joint','gimbal_pitch_joint')):
            self.joints=[values['gimbal_yaw_joint'],values['gimbal_pitch_joint']]
            self.joint_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            if self.initial is None:self.initial=self.joints.copy()
    def receive_rate(self,msg):
        if np.isfinite([msg.x,msg.y]).all():self.rate=[msg.x,msg.y];self.rate_wall=time.monotonic()
    def receive_image(self,msg):
        if (msg.encoding not in ('rgb8','bgr8') or min(msg.width,msg.height)<=0
                or msg.step<msg.width*3 or len(msg.data)!=msg.height*msg.step):return
        self.images.append(msg);self.annotated_pub.publish(msg)
    def receive_detection(self,msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        now=self.now()
        if (not 0<stamp<=now+.1 or now-stamp>.8
                or self.last_detection_stamp is not None and stamp<=self.last_detection_stamp):return
        self.last_detection_stamp=stamp
        rows=[]
        for det in msg.detections:
            b=det.bbox;score=det.results[0].hypothesis.score if det.results else 1.
            if (not np.isfinite([b.center.position.x,b.center.position.y,b.size_x,b.size_y,score]).all()
                    or min(b.size_x,b.size_y)<=0 or not 0<=score<=1.):continue
            rows.append(dict(bbox=[b.center.position.x-b.size_x/2,b.center.position.y-b.size_y/2,b.size_x,b.size_y],
                confidence=score,class_name='balloon'))
        self.detection_pub.publish(String(data=json.dumps(dict(stamp=stamp,frame_id=msg.header.frame_id,detections=rows),allow_nan=False)))
        if rows:self.last_visual=stamp
        for original in reversed(self.images):
            image_stamp=original.header.stamp.sec+original.header.stamp.nanosec*1e-9
            if abs(stamp-image_stamp)>1e-6 or original.header.frame_id!=msg.header.frame_id:continue
            if original.encoding not in ('rgb8','bgr8'):return
            pixels=np.frombuffer(original.data,np.uint8).reshape(original.height,original.step)[:,:original.width*3].reshape(original.height,original.width,3).copy()
            if original.encoding=='rgb8':pixels=cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)
            for row in rows:
                x,y,w,h=map(round,row['bbox']);cv2.rectangle(pixels,(x,y),(x+w,y+h),(60,235,80),2)
            image=Image(header=original.header,height=original.height,width=original.width,encoding='bgr8',step=original.width*3)
            image.data=array('B',pixels.tobytes());self.annotated_pub.publish(image);break
    def start(self,request,response):
        if self.started:response.success=True;response.message='acquisition_already_started';return response
        if self.restoring or self.pending_reset is not None or not self.reset_client.service_is_ready():
            response.success=False;response.message='balloon_reset_or_gimbal_not_ready';return response
        self.reset_generation+=1;generation=self.reset_generation
        self.status={};self.stage='starting';self.pending_reset=self.reset_client.call_async(Trigger.Request())
        self.pending_reset.add_done_callback(lambda future:self.started_after_reset(future,generation))
        response.success=True;response.message='balloon_reset_then_camera_search';return response
    def started_after_reset(self,future,generation):
        if generation!=self.reset_generation:return
        self.pending_reset=None
        if future.exception() or not future.result().success:self.stage='waiting_start';return
        self.started=True;self.start_stamp=self.now();self.stage='camera_search';self.last_visual=None
    def reset(self,request,response):
        self.reset_generation+=1;self.pending_reset=None
        self.started=False;self.start_stamp=None;self.restoring=True;self.status={};self.stage='resetting'
        if self.reset_client.service_is_ready():self.reset_client.call_async(Trigger.Request())
        response.success=True;response.message='camera_acquisition_reset';return response
    def command(self,position):
        message=JointState();message.header.stamp=self.get_clock().now().to_msg()
        message.name=['gimbal_yaw_joint','gimbal_pitch_joint'];message.position=list(position)
        self.command_pub.publish(message)
    def tick(self):
        now=self.now();wall=time.monotonic();dt=min(.05,max(0.,now-(self.last_tick or now)));self.last_tick=now
        if self.last_clock is not None and now<self.last_clock:
            self.reset_generation+=1;self.pending_reset=None;self.started=False;self.stage='waiting_start';self.status={}
            self.last_detection_stamp=self.last_visual=self.start_stamp=self.rate=None;self.images.clear()
        self.last_clock=now
        joint_fresh=bool(self.joints is not None and self.joint_stamp is not None and 0<=now-self.joint_stamp<=.3)
        if self.restoring and joint_fresh:
            if max(abs(math.remainder(self.initial[0]-self.joints[0],2*math.pi)),abs(self.initial[1]-self.joints[1]))<math.radians(1):
                self.restoring=False;self.stage='waiting_start'
            else:self.command(self.initial)
        if self.started and joint_fresh:
            if self.last_visual is None or now-self.last_visual>.5:
                self.command([self.joints[0]-dt,-math.radians(13.)]);self.stage='camera_search'
            elif self.rate is not None and wall-self.rate_wall<=.3:
                # SIYI positive yaw/right and pitch/up are opposite NUB joint signs.
                self.command([self.joints[0]-np.clip(self.rate[0]*.008,-1.2,1.2)*dt,
                              self.joints[1]-np.clip(self.rate[1]*.008,-.8,.8)*dt])
        try:
            if not self.started:raise ValueError('waiting_for_acquisition_start')
            if wall-self.status_wall>.8:raise ValueError('balloon_status_wall_stale')
            if not joint_fresh:raise ValueError('gimbal_feedback_stale')
            packet=mission_estimate(self.status,now,self.start_stamp,wall-self.status_wall)
            self.reason='ready';self.stage='ready' if packet['evidence']['gimbal_centered'] else 'visual_centering'
        except(ValueError,KeyError,TypeError)as error:
            self.reason=str(error);packet=dict(valid=False,reason=self.reason,stamp=now,publication_stamp=now,frame_id='world',
                target_id=1,source='camera_monocular_size',position=None,covariance=None)
        if packet['valid']:
            marker=Marker();marker.header.frame_id='world';marker.header.stamp=self.get_clock().now().to_msg()
            marker.ns='balloon_locator';marker.id=1;marker.type=Marker.SPHERE;marker.action=Marker.ADD
            marker.pose.position.x,marker.pose.position.y,marker.pose.position.z=packet['position'];marker.pose.orientation.w=1.
            marker.scale.x=marker.scale.y=marker.scale.z=.3
            marker.color.r,marker.color.g,marker.color.b,marker.color.a=.1,.9,1.,1.
            marker.lifetime=Duration(seconds=.8).to_msg();self.marker_pub.publish(MarkerArray(markers=[marker]))
        self.estimate_pub.publish(String(data=json.dumps(packet,allow_nan=False)))
        self.status_pub.publish(String(data=json.dumps(dict(stamp=now,ready=packet['valid'],reason=self.reason,
            calibration_valid=True,calibration_source='gazebo_camera_info_tf_exact_sphere',ground_truth_used=False,
            estimator='balloon_locator',acquisition=dict(started=self.started,stage=self.stage,requires_start=True,
                returning_to_initial=self.restoring)),allow_nan=False)))


def main(args=None):
    cv2.setNumThreads(1);rclpy.init(args=args);node=GazeboBridge()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
