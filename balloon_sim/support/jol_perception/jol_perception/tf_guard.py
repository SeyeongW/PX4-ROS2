"""TF의 실제 수신 표본 간격을 검사해 긴 누락 구간의 잘못된 자세 보간을 거부한다.

측정 시각의 TF를 요청해도 tf2가 먼 표본 사이를 보간할 수 있으므로 별도 검사가 필요하다."""
from bisect import bisect_left
import math
from threading import RLock

from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from tf2_ros import Buffer, TransformException, TransformListener


class BracketedBuffer(Buffer):
    def __init__(self, max_interpolation_gap=.03, cache_time=None, clock=None):
        if not math.isfinite(max_interpolation_gap) or max_interpolation_gap <= 0:
            raise ValueError('TF interpolation gap must be finite and positive')
        self.max_gap_ns = round(max_interpolation_gap*1e9)
        self.history_ns = cache_time.nanoseconds if cache_time is not None else 10000000000
        self.parents, self.dynamic_times, self.static_frames = {}, {}, set()
        self.guard_lock = RLock()
        self.last_support, self.guard_rejections = {}, 0
        self.source_clock = clock
        self.future_rejections = 0
        super().__init__(cache_time=cache_time)

    @staticmethod
    def _valid(transform):
        p, q = transform.transform.translation, transform.transform.rotation
        finite = all(math.isfinite(value) for value in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        return (finite and abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.) <= .001
                and bool(transform.child_frame_id) and bool(transform.header.frame_id)
                and transform.child_frame_id != transform.header.frame_id)

    def set_transform(self, transform, authority):
        if not self._valid(transform):
            return  # A rejected transform must never become apparent exact support.
        stamp = transform.header.stamp.sec*1000000000+transform.header.stamp.nanosec
        # After a simulation reset, a queued old-epoch message can arrive after
        # clear(). It must not push tf2's newest timestamp far into the future
        # and make all new-epoch measurements look too old. Allow ordinary
        # clock/TF transport reordering; lookup still requires exact brackets.
        if self.source_clock is not None and stamp > self.source_clock.now().nanoseconds+500000000:
            self.future_rejections += 1
            return
        child, parent = transform.child_frame_id, transform.header.frame_id
        with self.guard_lock:
            if self.parents.get(child, parent) != parent:
                self.dynamic_times.pop(child, None)
            self.parents[child] = parent
            self.static_frames.discard(child)
            times = self.dynamic_times.setdefault(child, [])
            index = bisect_left(times, stamp)
            if index == len(times) or times[index] != stamp:
                times.insert(index, stamp)
            # Match the TF cache horizon, retaining one earlier bracket edge.
            cutoff = bisect_left(times, times[-1]-self.history_ns)
            if cutoff > 1:
                del times[:cutoff-1]
            super().set_transform(transform, authority)

    def set_transform_static(self, transform, authority):
        if not self._valid(transform):
            return
        with self.guard_lock:
            self.parents[transform.child_frame_id] = transform.header.frame_id
            self.static_frames.add(transform.child_frame_id)
            self.dynamic_times.pop(transform.child_frame_id, None)
            super().set_transform_static(transform, authority)

    def clear(self):
        with self.guard_lock:
            self.dynamic_times.clear()
            self.parents = {child: parent for child, parent in self.parents.items()
                            if child in self.static_frames}
            self.last_support = {}
            super().clear()

    def _lineage(self, frame):
        path = [frame]
        while frame in self.parents:
            frame = self.parents[frame]
            if frame in path:
                raise TransformException('Cyclic TF parent graph')
            path.append(frame)
        return path

    def _support(self, target, source, stamp):
        if stamp <= 0:
            raise TransformException('TF guard requires a positive acquisition timestamp; latest TF is forbidden')
        if target == source:
            return {}
        source_path, target_path = self._lineage(source), self._lineage(target)
        common = next((frame for frame in source_path if frame in target_path), None)
        if common is None:
            raise TransformException('TF guard has no connected recorded sample path')
        edges = source_path[:source_path.index(common)]+target_path[:target_path.index(common)]
        support = {}
        for child in edges:
            if child in self.static_frames:
                continue
            times = self.dynamic_times.get(child, [])
            index = bisect_left(times, stamp)
            if index < len(times) and times[index] == stamp:
                lower = upper = stamp
            elif index == 0 or index == len(times):
                raise TransformException(f'TF guard missing timestamp brackets for {child} at {stamp}')
            else:
                lower, upper = times[index-1], times[index]
                if upper-lower > self.max_gap_ns:
                    raise TransformException(f'TF interpolation gap for {child}: {(upper-lower)*1e-9:.6f}s '
                                             f'exceeds {self.max_gap_ns*1e-9:.6f}s')
            support[child] = dict(lower_ns=lower, upper_ns=upper, gap_s=(upper-lower)*1e-9)
        return support

    def lookup_transform(self, target_frame, source_frame, time, timeout=Duration()):
        with self.guard_lock:
            self.last_support = {}
            try:
                support = self._support(target_frame, source_frame, time.nanoseconds)
            except TransformException:
                self.guard_rejections += 1
                raise
            result = super().lookup_transform(target_frame, source_frame, time, timeout)
            self.last_support = support
            return result


def guarded_tf(node, cache_time=None):
    """Share the actual listener's samples; do not add a second racing subscriber."""
    gap = float(node.declare_parameter('tf_max_interpolation_gap', .03).value)
    depth = int(node.declare_parameter('tf_subscription_depth', 2000).value)
    if not 1 <= depth <= 100000:
        raise ValueError('TF subscription depth must be between 1 and 100000')
    buffer = BracketedBuffer(gap, cache_time=cache_time, clock=node.get_clock())
    buffer.clock_jump_handle = node.get_clock().create_jump_callback(
        JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True),
        post_callback=lambda jump: buffer.clear())
    listener = TransformListener(buffer, node, qos=QoSProfile(depth=depth))
    return buffer, listener
