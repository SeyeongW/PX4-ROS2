"""영상 오차로 미보정 관절 후보를 계산하며 실물 모터에는 명령하지 않는다."""
import json
import math
from pathlib import Path
import time

import numpy as np

from .pointing_core import ARM_LOWER, ARM_UPPER, limit_step


JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']


class ProposalTracker:
    """영상 촬영 시각과 수신 시각을 각각 검사하는 후보 상태기계."""

    def __init__(self, expiry=.5, gain_deg=5., slew_deg_s=2., tau=.25, deadband=.05):
        values = [expiry, gain_deg, slew_deg_s, tau, deadband]
        if (not np.isfinite(values).all() or not 0 < expiry <= 2.
                or not 0 < gain_deg <= 5. or not 0 < slew_deg_s <= 2.
                or tau <= 0 or not 0 <= deadband < 1.):
            raise ValueError('Invalid proposal freshness, gain, slew, filter or deadband')
        self.expiry, self.gain, self.slew = expiry, math.radians(gain_deg), math.radians(slew_deg_s)
        self.tau, self.deadband = tau, deadband
        self.joints = self.target = self.neutral = None
        self.last_now = self.last_wall = None
        self.reason = 'waiting_for_joint_feedback'
        self.reset()

    def reset(self):
        self.filtered = np.zeros(2)
        self.proposed = None

    def fresh(self, packet, now, wall):
        return bool(packet is not None and 0 <= now-packet['stamp'] <= self.expiry
                    and 0 <= wall-packet['received_wall'] <= self.expiry)

    def validate_stamp(self, stamp, now):
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not np.isfinite([stamp, now]).all() or stamp <= 0
                or not 0 <= now-stamp <= self.expiry):
            raise ValueError('invalid_or_stale_stamp')

    def receive_joints(self, stamp, names, positions, now, wall):
        try:
            self.validate_stamp(stamp, now)
            if (len(names) != len(positions) or len(names) != len(set(names))
                    or not all(name in names for name in JOINT_NAMES)):
                raise ValueError('invalid_joint_names')
            q = np.asarray([positions[names.index(name)] for name in JOINT_NAMES], dtype=float)
            if (q.shape != (4,) or not np.isfinite(q).all()
                    or (q < ARM_LOWER).any() or (q > ARM_UPPER).any()):
                raise ValueError('invalid_or_out_of_bounds_joints')
            if self.joints is not None and stamp <= self.joints['stamp']:
                raise ValueError('non_increasing_joint_stamp')
            if not self.fresh(self.joints, now, wall):
                self.reset()
            self.joints = dict(stamp=float(stamp), received_wall=wall, positions=q)
            if self.neutral is None:
                self.neutral = q.copy()
            return True
        except (ValueError, TypeError, KeyError):
            self.joints = None
            self.reason = 'invalid_joint_feedback'
            self.reset()
            return False

    def receive_target(self, packet, now, wall):
        try:
            if not isinstance(packet, dict) or type(packet.get('valid')) is not bool:
                raise ValueError('invalid_target_packet')
            self.validate_stamp(packet['stamp'], now)
            if packet.get('source') != 'manual_roi_csrt':
                raise ValueError('unsupported_target_source')
            if not packet['valid']:
                self.target = None
                self.reason = 'no_target'
                self.reset()
                return True
            width, height = packet['width'], packet['height']
            if (type(width) is not int or type(height) is not int
                    or not 0 < width <= 8192 or not 0 < height <= 8192):
                raise ValueError('invalid_image_size')
            box = np.asarray(packet['bbox'], dtype=float)
            error = np.asarray(packet['error_normalized'], dtype=float)
            if (box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0
                    or min(box[:2]) < 0 or box[0]+box[2] > width or box[1]+box[3] > height
                    or error.shape != (2,) or not np.isfinite(error).all() or (abs(error) > 1.).any()):
                raise ValueError('invalid_box_or_error')
            center_error = (box[:2]+box[2:]/2.-np.array([width, height])/2.) / (np.array([width, height])/2.)
            if not np.allclose(error, center_error, atol=.001, rtol=0):
                raise ValueError('error_does_not_match_box')
            selection = packet.get('selection_id')
            if selection is not None and (type(selection) is not int or selection < 0):
                raise ValueError('invalid_selection_id')
            if self.target is not None and packet['stamp'] <= self.target['stamp']:
                raise ValueError('non_increasing_target_stamp')
            if (not self.fresh(self.target, now, wall)
                    or self.target['selection_id'] != selection):
                self.reset()
            self.target = dict(stamp=float(packet['stamp']), received_wall=wall,
                               error=error, bbox=box.tolist(), selection_id=selection)
            return True
        except (ValueError, TypeError, KeyError):
            self.target = None
            self.reason = 'invalid_target'
            self.reset()
            return False

    def sample(self, now, wall):
        if not np.isfinite([now, wall]).all():
            raise ValueError('Invalid sample time')
        dt = 0. if self.last_wall is None else wall-self.last_wall
        if (self.last_now is not None and now < self.last_now) or dt < 0:
            self.joints = self.target = None
            self.reason = 'clock_reset'
            self.reset()
        elif dt > self.expiry:
            self.reset()
        self.last_now, self.last_wall = now, wall
        joint_fresh = self.fresh(self.joints, now, wall)
        target_fresh = self.fresh(self.target, now, wall)
        status = dict(stamp=now, state=self.reason, mapping='arm_camera_axes_uncalibrated',
                      camera_mount='arm_user_reported',
                      hardware_writes=False, calibrated=False, depth_available=False,
                      world_bearing_available=False, gravity_stabilization_available=False,
                      joint_feedback_fresh=joint_fresh, target_fresh=target_fresh,
                      joint_stamp=self.joints['stamp'] if self.joints else None,
                      target_stamp=self.target['stamp'] if self.target else None,
                      measured_positions=self.joints['positions'].tolist() if joint_fresh else None,
                      neutral_positions=self.neutral.tolist() if self.neutral is not None else None,
                      proposed_positions=[], error_normalized=None, filtered_error=None,
                      gain_deg=math.degrees(self.gain), slew_deg_s=math.degrees(self.slew),
                      smoothing_tau_s=self.tau, deadband_normalized=self.deadband, expiry_s=self.expiry)
        if not joint_fresh or not target_fresh:
            if not joint_fresh:
                reason = 'stale_joint_feedback' if self.joints else self.reason
                if reason not in ('invalid_joint_feedback', 'clock_reset', 'stale_joint_feedback'):
                    reason = 'waiting_for_joint_feedback'
            else:
                reason = 'stale_target' if self.target else self.reason
                if reason not in ('invalid_target', 'no_target', 'clock_reset', 'stale_target'):
                    reason = 'waiting_for_target'
            status['state'] = reason
            self.reset()
            return status
        # 기준 자세를 고정해 누적 드리프트를 막고 재개 시 현재 실측부터 시작한다.
        if self.proposed is None:
            self.proposed = self.joints['positions'].copy()
            dt = 0.
        dt = min(max(dt, 0.), .25)
        error = self.target['error'].copy()
        error[abs(error) <= self.deadband] = 0.
        self.filtered += -math.expm1(-dt/self.tau)*(error-self.filtered)
        desired = self.neutral.copy()
        desired[[0, 3]] += self.gain*self.filtered
        desired = np.clip(desired, ARM_LOWER, ARM_UPPER)
        self.proposed = limit_step(self.proposed, desired, self.slew, dt, ARM_LOWER, ARM_UPPER)
        status.update(state='uncalibrated_proposal', proposed_positions=self.proposed.tolist(),
                      desired_positions=desired.tolist(), error_normalized=self.target['error'].tolist(),
                      filtered_error=self.filtered.tolist(), selection_id=self.target['selection_id'])
        return status


def main(args=None):
    # 순수 후보 계산은 ROS 환경 없이도 검사할 수 있게 한다.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    class RealTracking(Node):
        def __init__(self):
            super().__init__('jol_real_tracking')
            self.core = ProposalTracker(**{name: float(self.declare_parameter(name, default).value)
                for name, default in dict(expiry=.5, gain_deg=5., slew_deg_s=2., tau=.25, deadband=.05).items()})
            self.detection_count = None
            self.detection_stamp = self.detection_wall = None
            self.status_pub = self.create_publisher(String, '/jol/real_tracking/status', 10)
            self.proposal_pub = self.create_publisher(JointState, '/jol/real_arm/proposed_command', 10)
            self.create_subscription(String, '/jol/real_camera/target', self.target, 10)
            self.create_subscription(String, '/jol/camera/detections', self.detections, 10)
            self.create_subscription(JointState, '/jol/arm/joint_states', self.joints, qos_profile_sensor_data)
            log_path = self.declare_parameter('log_path', '').value
            self.log_file = None
            if log_path:
                path = Path(log_path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                self.log_file = path.open('a', encoding='utf-8', buffering=1)
            self.create_timer(.05, self.tick)
            self.get_logger().info('Inspection-only proposals: arm camera axes are uncalibrated; no hardware commands')

        def now(self):
            return self.get_clock().now().nanoseconds*1e-9

        def target(self, msg):
            try:
                packet = json.loads(msg.data)
            except (ValueError, TypeError):
                packet = None
            self.core.receive_target(packet, self.now(), time.monotonic())

        def joints(self, msg):
            stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            self.core.receive_joints(stamp, list(msg.name), list(msg.position), self.now(), time.monotonic())

        def detections(self, msg):
            try:
                packet = json.loads(msg.data)
                self.core.validate_stamp(packet['stamp'], self.now())
                if not isinstance(packet['detections'], list) or len(packet['detections']) > 256:
                    raise ValueError('Invalid detection list')
                self.detection_count, self.detection_stamp = len(packet['detections']), packet['stamp']
                self.detection_wall = time.monotonic()
            except (ValueError, TypeError, KeyError):
                self.detection_count = self.detection_stamp = self.detection_wall = None

        def tick(self):
            now, wall = self.now(), time.monotonic()
            status = self.core.sample(now, wall)
            detection_fresh = (self.detection_stamp is not None
                and 0 <= now-self.detection_stamp <= self.core.expiry
                and 0 <= wall-self.detection_wall <= self.core.expiry)
            status.update(detection_count=self.detection_count if detection_fresh else None,
                          detection_count_fresh=detection_fresh, detections_auto_select_target=False)
            if status['proposed_positions']:
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name, msg.position = JOINT_NAMES, status['proposed_positions']
                self.proposal_pub.publish(msg)
            encoded = json.dumps(status, allow_nan=False)
            self.status_pub.publish(String(data=encoded))
            if self.log_file:
                try:
                    self.log_file.write(encoded+'\n')
                except OSError as error:
                    self.get_logger().error('Proposal log write failed: '+str(error))
                    self.log_file.close()
                    self.log_file = None

    rclpy.init(args=args)
    node = None
    try:
        node = RealTracking()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node.log_file:
                node.log_file.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
