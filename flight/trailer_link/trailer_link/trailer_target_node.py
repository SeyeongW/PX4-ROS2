"""Convert the trailer GPS link into the landing stack's local-ENU cue.

The coordinate conversion is the field-used ``wang`` relationship::

    target_local = vehicle_local + ENU(vehicle_fix -> trailer_fix)

Only the output adapter is new: position and velocity are emitted together in
``px4_local_enu`` with one source stamp, matching MissionManager's existing
Gazebo cue ABI.  Old fixes are never re-stamped or replayed.
"""

from __future__ import annotations

import math
from collections import deque

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from rclpy.time import Time
from sensor_msgs.msg import NavSatFix, NavSatStatus

from .geodesy import (DEFAULT_MAX_DISTANCE_M, DEFAULT_MAX_INPUT_SKEW_S,
                      DEFAULT_STALE_AFTER_S, RelativeTarget)


LOCAL_ENU_FRAME = 'px4_local_enu'


def _sensor_qos() -> QoSProfile:
    """MAVROS and the radio adapter publish sensor data BEST_EFFORT."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


def _cue_qos() -> QoSProfile:
    """MissionManager's cue subscriptions are RELIABLE."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


def _stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class TrailerTargetNode(Node):
    def __init__(self) -> None:
        super().__init__('trailer_target_node')
        p = self.declare_parameter
        self.trailer_fix_topic = str(
            p('trailer_fix_topic', '/trailer/fix').value)
        self.trailer_velocity_topic = str(
            p('trailer_velocity_topic', '/trailer/velocity_enu').value)
        self.vehicle_fix_topic = str(
            p('vehicle_fix_topic', '/mavros/global_position/global').value)
        self.vehicle_pose_topic = str(
            p('vehicle_pose_topic', '/mavros/local_position/pose').value)
        self.cue_topic = str(p('cue_topic', '/marker/cue').value)
        self.cue_velocity_topic = str(
            p('cue_velocity_topic', '/marker/cue_velocity').value)
        self.vehicle_pose_frame = str(
            p('vehicle_pose_frame_id', 'map').value)
        self.trailer_velocity_frame = str(
            p('trailer_velocity_frame_id', 'earth_enu').value)
        self.output_frame = str(p('output_frame_id', LOCAL_ENU_FRAME).value)
        self.deck_z = float(p('deck_z_m', 0.0).value)
        self.rate_hz = float(p('rate_hz', 20.0).value)
        self.stale_after = float(
            p('stale_after_s', DEFAULT_STALE_AFTER_S).value)
        self.max_input_skew = float(
            p('max_input_skew_s', DEFAULT_MAX_INPUT_SKEW_S).value)
        self.max_distance = float(
            p('max_distance_m', DEFAULT_MAX_DISTANCE_M).value)
        self.max_speed = float(p('max_speed_m_s', 20.0).value)
        self.min_source_rate = float(p('min_source_rate_hz', 4.0).value)
        self.rate_window = float(p('source_rate_window_s', 2.0).value)
        self.stats_period = float(p('stats_period_s', 2.0).value)
        numeric = (self.deck_z, self.rate_hz, self.stale_after,
                   self.max_input_skew, self.max_distance, self.max_speed,
                   self.rate_window, self.stats_period)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('GPS cue parameters must be finite')
        if not all(value > 0.0 for value in numeric[1:]):
            raise ValueError('GPS cue rates and limits must be positive')
        if not math.isfinite(self.min_source_rate) \
                or self.min_source_rate < 0.0:
            raise ValueError(
                'min_source_rate_hz must be finite and nonnegative')
        if self.output_frame != LOCAL_ENU_FRAME:
            raise ValueError(f'output_frame_id must be {LOCAL_ENU_FRAME!r}')

        self.target = RelativeTarget(
            stale_after=self.stale_after,
            max_input_skew=self.max_input_skew,
            max_distance=self.max_distance,
        )
        self._velocity: tuple[float, float] | None = None
        self._velocity_stamp: float | None = None
        self._fix_epoch_ns: int | None = None
        self._velocity_epoch_ns: int | None = None
        self._last_published_epoch_ns: int | None = None
        self._fix_source_times = deque()
        self._vehicle_streams = {
            'fix': self._new_vehicle_stream(),
            'pose': self._new_vehicle_stream(),
        }
        self._vehicle_anchor_epochs: tuple[int, int] | None = None
        self._last_block: str | None = 'starting up'
        self._published = 0

        sensor_qos = _sensor_qos()
        self.create_subscription(NavSatFix, self.trailer_fix_topic,
                                 self._on_trailer_fix, sensor_qos)
        self.create_subscription(Vector3Stamped, self.trailer_velocity_topic,
                                 self._on_trailer_velocity, sensor_qos)
        self.create_subscription(NavSatFix, self.vehicle_fix_topic,
                                 self._on_vehicle_fix, sensor_qos)
        self.create_subscription(PoseStamped, self.vehicle_pose_topic,
                                 self._on_vehicle_pose, sensor_qos)
        cue_qos = _cue_qos()
        self.cue_pub = self.create_publisher(
            PointStamped, self.cue_topic, cue_qos)
        self.cue_velocity_pub = self.create_publisher(
            Vector3Stamped, self.cue_velocity_topic, cue_qos)
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.create_timer(self.stats_period, self._log_stats)
        self.get_logger().info(
            f'GPS cue: {self.trailer_fix_topic} -> '
            f'{self.cue_topic} + {self.cue_velocity_topic} '
            f'({LOCAL_ENU_FRAME}, deck z {self.deck_z:.3f}m)')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_trailer_fix(self, message: NavSatFix) -> None:
        self._fix_epoch_ns = _stamp_ns(message)
        source_time = self._fix_epoch_ns * 1.0e-9
        if self._fix_source_times and source_time < self._fix_source_times[-1]:
            self._fix_source_times.clear()
        if (not self._fix_source_times
                or source_time > self._fix_source_times[-1]):
            self._fix_source_times.append(source_time)
            while (len(self._fix_source_times) > 1
                   and source_time - self._fix_source_times[0]
                   > self.rate_window):
                self._fix_source_times.popleft()
        self.target.on_trailer_fix(
            source_time,
            lat=message.latitude,
            lon=message.longitude,
            alt=message.altitude,
            has_fix=message.status.status >= NavSatStatus.STATUS_FIX,
        )

    def _on_trailer_velocity(self, message: Vector3Stamped) -> None:
        epoch_ns = _stamp_ns(message)
        source_time = epoch_ns * 1.0e-9
        values = (float(message.vector.x), float(message.vector.y),
                  float(message.vector.z))
        if (message.header.frame_id != self.trailer_velocity_frame
                or not all(math.isfinite(value) for value in values)
                or math.hypot(values[0], values[1]) > self.max_speed):
            self._velocity = None
            self._velocity_stamp = None
            self._velocity_epoch_ns = None
            return
        self._velocity = values[:2]
        self._velocity_stamp = source_time
        self._velocity_epoch_ns = epoch_ns

    def _on_vehicle_fix(self, message: NavSatFix) -> None:
        values = (float(message.latitude), float(message.longitude),
                  float(message.altitude))
        if (message.status.status < NavSatStatus.STATUS_FIX
                or not all(math.isfinite(value) for value in values)
                or not -90.0 <= values[0] <= 90.0
                or not -180.0 <= values[1] <= 180.0):
            self._reject_vehicle_stream('fix')
            return
        self._record_vehicle_sample('fix', _stamp_ns(message), values)

    def _on_vehicle_pose(self, message: PoseStamped) -> None:
        if message.header.frame_id != self.vehicle_pose_frame:
            self._reject_vehicle_stream('pose')
            return
        position = message.pose.position
        values = (float(position.x), float(position.y), float(position.z))
        if not all(math.isfinite(value) for value in values):
            self._reject_vehicle_stream('pose')
            return
        self._record_vehicle_sample('pose', _stamp_ns(message), values)

    @staticmethod
    def _new_vehicle_stream() -> dict:
        return {
            'sample': None,
            'last_epoch_ns': None,
            'advance_count': 0,
            'receipt': None,
            'progress_receipt': None,
        }

    def _invalidate_vehicle_anchor(self) -> None:
        self._vehicle_anchor_epochs = None
        self.target._t_vehicle_fix = None
        self.target._t_local = None

    def _reject_vehicle_stream(self, name: str) -> None:
        self._vehicle_streams[name] = self._new_vehicle_stream()
        self._invalidate_vehicle_anchor()

    def _record_vehicle_sample(self, name: str, epoch_ns: int,
                               sample: tuple[float, float, float]) -> None:
        now = self._now()
        state = self._vehicle_streams[name]
        source_time = epoch_ns * 1.0e-9
        if (epoch_ns <= 0 or not math.isfinite(now)
                or source_time > now + self.max_input_skew
                or now - source_time > 2.0 * self.stale_after):
            self._reject_vehicle_stream(name)
            return

        last_epoch = state['last_epoch_ns']
        if last_epoch is not None and epoch_ns < last_epoch:
            self._reject_vehicle_stream(name)
            state = self._vehicle_streams[name]

        if state['last_epoch_ns'] is None or epoch_ns > state['last_epoch_ns']:
            last_progress = state['progress_receipt']
            if (last_progress is None
                    or now - last_progress > self.stale_after):
                state['advance_count'] = 1
            else:
                state['advance_count'] = min(
                    2, int(state['advance_count']) + 1)
            state['last_epoch_ns'] = epoch_ns
            state['progress_receipt'] = now
        state['receipt'] = now
        state['sample'] = sample
        self._refresh_vehicle_anchor()

    def _refresh_vehicle_anchor(self) -> None:
        fix = self._vehicle_streams['fix']
        pose = self._vehicle_streams['pose']
        if (fix['advance_count'] < 2 or pose['advance_count'] < 2
                or fix['sample'] is None or pose['sample'] is None):
            return
        epochs = (int(fix['last_epoch_ns']), int(pose['last_epoch_ns']))
        if abs(epochs[0] - epochs[1]) * 1.0e-9 > self.max_input_skew:
            return
        if epochs == self._vehicle_anchor_epochs:
            return

        now = self._now()
        lat, lon, alt = fix['sample']
        x, y, z = pose['sample']
        self.target.on_vehicle_fix(
            now, lat=lat, lon=lon, alt=alt, has_fix=True)
        self.target.on_vehicle_local(now, x=x, y=y, z=z)
        self._vehicle_anchor_epochs = epochs

    def _vehicle_stream_block(self, now: float) -> str | None:
        for name, label in (('fix', 'global position'),
                            ('pose', 'local ENU position')):
            state = self._vehicle_streams[name]
            if state['advance_count'] < 2:
                return f'vehicle {label} source is not qualified'
            receipt = state['receipt']
            progress = state['progress_receipt']
            if (receipt is None or progress is None
                    or not 0.0 <= now - receipt <= self.stale_after
                    or not 0.0 <= now - progress <= self.stale_after):
                return f'vehicle {label} source is stale'
        fix_epoch = self._vehicle_streams['fix']['last_epoch_ns']
        pose_epoch = self._vehicle_streams['pose']['last_epoch_ns']
        if (fix_epoch is None or pose_epoch is None
                or abs(fix_epoch - pose_epoch) * 1.0e-9
                > self.max_input_skew):
            return (f'vehicle anchor times differ by more than '
                    f'{self.max_input_skew:.3f}s')
        return None

    def _blocking_reason(self, now: float) -> str | None:
        reason = self._vehicle_stream_block(now)
        if reason is not None:
            return reason
        reason = self.target.blocking_reason(now)
        if reason is not None:
            return reason
        if self._velocity is None or self._velocity_stamp is None:
            return 'trailer ENU velocity is missing or invalid'
        velocity_age = now - self._velocity_stamp
        if not 0.0 <= velocity_age <= self.stale_after:
            return 'trailer ENU velocity is stale'
        if (self._fix_epoch_ns is None or self._fix_epoch_ns <= 0
                or self._velocity_epoch_ns != self._fix_epoch_ns):
            return 'trailer position and velocity epochs do not match'
        source_rate = self._source_rate_hz()
        if (self.min_source_rate > 0.0
                and (source_rate is None
                     or source_rate + 1.0e-9 < self.min_source_rate)):
            shown = ('unknown' if source_rate is None
                     else f'{source_rate:.2f}Hz')
            return (f'trailer position rate {shown} is below '
                    f'{self.min_source_rate:.2f}Hz')
        if self._fix_epoch_ns == self._last_published_epoch_ns:
            return 'waiting for a new trailer GPS epoch'
        return None

    def _source_rate_hz(self) -> float | None:
        if len(self._fix_source_times) < 3:
            return None
        span = self._fix_source_times[-1] - self._fix_source_times[0]
        if span <= 0.0:
            return None
        return (len(self._fix_source_times) - 1) / span

    def _tick(self) -> None:
        now = self._now()
        reason = self._blocking_reason(now)
        if reason is not None:
            self._note_block(reason)
            return
        solved = self.target.solve(now)
        source_stamp = self.target.source_stamp()
        if solved is None or source_stamp is None or self._velocity is None:
            self._note_block('GPS cue inputs changed before publish')
            return

        source_stamp = self._velocity_stamp
        stamp = Time(nanoseconds=round(source_stamp * 1.0e9)).to_msg()
        cue = PointStamped()
        cue.header.stamp = stamp
        cue.header.frame_id = self.output_frame
        cue.point.x, cue.point.y = solved[:2]
        cue.point.z = self.deck_z
        velocity = Vector3Stamped()
        velocity.header.stamp = stamp
        velocity.header.frame_id = self.output_frame
        velocity.vector.x, velocity.vector.y = self._velocity
        velocity.vector.z = 0.0
        self.cue_pub.publish(cue)
        self.cue_velocity_pub.publish(velocity)
        self._last_published_epoch_ns = self._fix_epoch_ns
        self._published += 1
        self._note_block(None)

    def _note_block(self, reason: str | None) -> None:
        if reason == 'waiting for a new trailer GPS epoch':
            return
        if reason == self._last_block:
            return
        self._last_block = reason
        if reason is None:
            self.get_logger().info('GPS trailer cue VALID')
        else:
            self.get_logger().warn(f'no GPS trailer cue: {reason}')

    def _log_stats(self) -> None:
        count, self._published = self._published, 0
        self.get_logger().info(
            f'GPS cue output {count / self.stats_period:.1f}Hz')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrailerTargetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
