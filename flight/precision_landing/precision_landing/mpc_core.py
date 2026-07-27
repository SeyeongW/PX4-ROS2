"""Deterministic mission, estimation and jerk-control primitives.

The module intentionally has no ROS imports.  ROS callbacks only translate
messages at the boundary; all planning and safety decisions can be unit tested.
Internal positions, velocities and accelerations use Gazebo map ENU.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Callable, Sequence

import numpy as np
from scipy import optimize


_EPS = 1.0e-9


class MissionState(Enum):
    IDLE = "idle"
    PREFLIGHT = "preflight"
    OFFBOARD_PRESTREAM = "offboard_prestream"
    ARMING = "arming"
    VERTICAL_TAKEOFF = "vertical_takeoff"
    GLOBAL_PLAN = "global_plan"
    GLOBAL_TRACK = "global_track"
    LOCAL_AVOIDANCE = "local_avoidance"
    WAYPOINT_HOLD = "waypoint_hold"
    RETURN_REQUESTED = "return_requested"
    TRAILER_INTERCEPT_PLAN = "trailer_intercept_plan"
    TRAILER_INTERCEPT_TRACK = "trailer_intercept_track"
    MARKER_SEARCH_FRONT = "marker_search_front"
    MARKER_ACQUIRE_FRONT = "marker_acquire_front"
    CAMERA_HANDOFF = "camera_handoff"
    MARKER_TRACK_DOWN = "marker_track_down"
    VELOCITY_MATCH = "velocity_match"
    PRECISION_ALIGN = "precision_align"
    PRECISION_DESCENT = "precision_descent"
    FINAL_APPROACH = "final_approach"
    TOUCHDOWN_CONFIRM = "touchdown_confirm"
    DISARM = "disarm"
    HOLD_REPLAN = "hold_replan"
    ABORT_CLIMB = "abort_climb"
    FAILSAFE_HOLD = "failsafe_hold"
    EMERGENCY_LAND = "emergency_land"


def enu_to_ned_vector(value: Sequence[float]) -> np.ndarray:
    """Convert a free vector from ENU / FLU axes to NED axes."""

    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("ENU vector must be a finite three-vector")
    return np.asarray((vector[1], vector[0], -vector[2]), dtype=float)


def ned_to_enu_vector(value: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`enu_to_ned_vector`."""

    return enu_to_ned_vector(value)


def front_optical_to_flu(value: Sequence[float]) -> np.ndarray:
    """Optical (+right,+down,+forward) to vehicle FLU for a front camera."""

    optical = np.asarray(value, dtype=float)
    if optical.shape != (3,) or not np.all(np.isfinite(optical)):
        raise ValueError("optical vector must be a finite three-vector")
    return np.asarray((optical[2], -optical[0], -optical[1]), dtype=float)


def down_optical_to_flu(value: Sequence[float]) -> np.ndarray:
    """Optical to FLU for a nadir camera whose image top faces forward.

    Camera +z points down, image +x points vehicle-right, and image +y points
    vehicle-back.  The transform is parameterized at the ROS node boundary if
    a physical installation uses another yaw.
    """

    optical = np.asarray(value, dtype=float)
    if optical.shape != (3,) or not np.all(np.isfinite(optical)):
        raise ValueError("optical vector must be a finite three-vector")
    return np.asarray((-optical[1], -optical[0], -optical[2]), dtype=float)


def marker_position_to_flu(
    value: Sequence[float], frame_id: str, camera: str
) -> np.ndarray:
    """Normalize a marker position to aircraft base-link FLU exactly once.

    Production ArUco nodes publish ``base_link`` after tf2 transforms.  Raw
    optical input is accepted for replay tools, while an unknown frame is
    rejected instead of silently applying the wrong axis convention.
    """

    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("marker position must be a finite three-vector")
    frame = str(frame_id).strip().lstrip("/")
    if frame == "base_link" or frame.endswith("/base_link"):
        return vector.copy()
    if not frame.endswith("optical_frame"):
        raise ValueError(f"unsupported marker frame: {frame!r}")
    if camera == "front":
        return front_optical_to_flu(vector)
    if camera == "down":
        return down_optical_to_flu(vector)
    raise ValueError("camera must be front or down")


def body_velocity_to_enu(value: Sequence[float], yaw_enu_rad: float) -> np.ndarray:
    """Rotate a child/body-frame ROS odometry velocity into world ENU."""

    body = np.asarray(value, dtype=float)
    yaw = float(yaw_enu_rad)
    if body.shape != (3,) or not np.all(np.isfinite(body)) or not math.isfinite(yaw):
        raise ValueError("body velocity and yaw must be finite")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            cosine * body[0] - sine * body[1],
            sine * body[0] + cosine * body[1],
            body[2],
        ),
        dtype=float,
    )


def flu_to_enu_with_px4_quaternion(
    value_flu: Sequence[float], quaternion_wxyz: Sequence[float]
) -> np.ndarray:
    """Rotate a base-link FLU vector with PX4's FRD->NED quaternion."""

    flu = np.asarray(value_flu, dtype=float)
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if flu.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("FLU vector and PX4 quaternion have invalid dimensions")
    norm = float(np.linalg.norm(quaternion))
    if not np.all(np.isfinite(flu)) or not math.isfinite(norm) or norm < _EPS:
        raise ValueError("FLU vector and PX4 quaternion must be finite")
    w, x, y, z = quaternion / norm
    rotation_frd_to_ned = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )
    frd = np.asarray((flu[0], -flu[1], -flu[2]), dtype=float)
    ned = rotation_frd_to_ned @ frd
    return np.asarray((ned[1], ned[0], -ned[2]), dtype=float)


@dataclass(frozen=True)
class TrailerKinematicState:
    stamp_s: float
    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray
    acceleration_enu_m_s2: np.ndarray
    yaw_enu_rad: float = math.pi

    def __post_init__(self) -> None:
        for name in ("position_enu_m", "velocity_enu_m_s", "acceleration_enu_m_s2"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite three-vector")
            object.__setattr__(self, name, value)

    def predict(self, future_s: float) -> "TrailerKinematicState":
        dt = max(0.0, float(future_s))
        position = (
            self.position_enu_m
            + self.velocity_enu_m_s * dt
            + 0.5 * self.acceleration_enu_m_s2 * dt * dt
        )
        velocity = self.velocity_enu_m_s + self.acceleration_enu_m_s2 * dt
        return TrailerKinematicState(
            self.stamp_s + dt, position, velocity, self.acceleration_enu_m_s2,
            self.yaw_enu_rad,
        )


@dataclass(frozen=True)
class InterceptSolution:
    position_enu_m: np.ndarray
    trailer_velocity_enu_m_s: np.ndarray
    eta_s: float
    iterations: int
    path_length_m: float


def iterative_trailer_intercept(
    trailer: TrailerKinematicState,
    path_length_to: Callable[[np.ndarray], float],
    expected_cruise_speed_m_s: float,
    maximum_prediction_s: float = 120.0,
    iterations: int = 4,
    endpoint_enu_m: Sequence[float] | None = None,
) -> InterceptSolution:
    """Repeatedly couple path length / ETA with a moving target prediction."""

    if expected_cruise_speed_m_s <= 0.0 or iterations < 1:
        raise ValueError("cruise speed and iteration count must be positive")
    endpoint = None if endpoint_enu_m is None else np.asarray(endpoint_enu_m, dtype=float)
    eta = 0.0
    length = float(path_length_to(trailer.position_enu_m.copy()))
    if not math.isfinite(length) or length < 0.0:
        raise ValueError("path length callback returned an invalid length")
    eta = min(maximum_prediction_s, length / expected_cruise_speed_m_s)
    prediction = trailer.predict(eta)
    for _ in range(iterations):
        target = prediction.position_enu_m.copy()
        if endpoint is not None:
            route = endpoint - trailer.position_enu_m
            route_norm2 = float(np.dot(route, route))
            if route_norm2 > _EPS:
                ratio = float(np.clip(np.dot(target - trailer.position_enu_m, route) / route_norm2, 0.0, 1.0))
                target = trailer.position_enu_m + ratio * route
        length = float(path_length_to(target))
        if not math.isfinite(length) or length < 0.0:
            raise ValueError("path length callback returned an invalid length")
        eta = min(maximum_prediction_s, length / expected_cruise_speed_m_s)
        prediction = trailer.predict(eta)
    target = prediction.position_enu_m.copy()
    if endpoint is not None:
        route = endpoint - trailer.position_enu_m
        route_norm2 = float(np.dot(route, route))
        if route_norm2 > _EPS:
            ratio = float(np.clip(np.dot(target - trailer.position_enu_m, route) / route_norm2, 0.0, 1.0))
            target = trailer.position_enu_m + ratio * route
    return InterceptSolution(target, prediction.velocity_enu_m_s.copy(), eta, iterations, length)


def _polyline_progress(
    position_xy_m: Sequence[float], waypoints_xy_m: np.ndarray
) -> tuple[float, np.ndarray]:
    """Project a point onto an ordered route, preferring later tie points."""

    position = np.asarray(position_xy_m, dtype=float)
    route = np.asarray(waypoints_xy_m, dtype=float)
    if position.shape != (2,) or route.ndim != 2 or route.shape[1] != 2:
        raise ValueError("polyline projection requires XY point and waypoints")
    segments = np.diff(route, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if len(route) < 2 or np.any(lengths <= _EPS):
        raise ValueError("route needs at least two distinct waypoints")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    candidates: list[tuple[float, float]] = []
    for index, (start, delta, length) in enumerate(
        zip(route[:-1], segments, lengths)
    ):
        fraction = float(
            np.clip(np.dot(position - start, delta) / (length * length), 0.0, 1.0)
        )
        projection = start + fraction * delta
        candidates.append(
            (float(np.linalg.norm(position - projection)), cumulative[index] + fraction * length)
        )
    minimum_distance = min(value[0] for value in candidates)
    progress = max(
        value[1] for value in candidates if value[0] <= minimum_distance + 1.0e-6
    )
    return float(progress), cumulative


def _sample_polyline_route(
    waypoints_xy_m: np.ndarray,
    cumulative_m: np.ndarray,
    progress_m: float,
    speed_m_s: float,
    z_m: float,
) -> TrailerKinematicState:
    route = np.asarray(waypoints_xy_m, dtype=float)
    progress = float(np.clip(progress_m, 0.0, cumulative_m[-1]))
    segment = min(
        len(route) - 2,
        int(np.searchsorted(cumulative_m, progress, side="right") - 1),
    )
    segment = max(0, segment)
    delta = route[segment + 1] - route[segment]
    length = float(np.linalg.norm(delta))
    fraction = (progress - cumulative_m[segment]) / length
    position_xy = route[segment] + fraction * delta
    direction = delta / length
    at_endpoint = progress >= cumulative_m[-1] - 1.0e-6
    velocity_xy = np.zeros(2) if at_endpoint else direction * speed_m_s
    return TrailerKinematicState(
        0.0,
        np.asarray((position_xy[0], position_xy[1], z_m), dtype=float),
        np.asarray((velocity_xy[0], velocity_xy[1], 0.0), dtype=float),
        np.zeros(3, dtype=float),
        math.atan2(direction[1], direction[0]),
    )


def iterative_polyline_intercept(
    trailer: TrailerKinematicState,
    route_waypoints_xy_m: Sequence[Sequence[float]],
    path_length_to: Callable[[np.ndarray], float],
    expected_drone_speed_m_s: float,
    *,
    trailer_cruise_speed_m_s: float = 3.0,
    trailer_acceleration_m_s2: float = 1.0,
    planner_latency_s: float = 0.0,
    iterations: int = 4,
    maximum_prediction_s: float = 180.0,
) -> InterceptSolution:
    """Iterate an intercept on the trailer's known collision-free polyline.

    Projecting a constant-acceleration prediction onto a diagonal start/end
    chord cuts route corners and can place the target inside buildings.  This
    predictor advances scalar arclength along the exact jerk-limited route;
    acceleration is conservatively bounded until cruise speed is reached.
    """

    route = np.asarray(route_waypoints_xy_m, dtype=float)
    if (
        expected_drone_speed_m_s <= 0.0
        or trailer_cruise_speed_m_s <= 0.0
        or trailer_acceleration_m_s2 <= 0.0
        or planner_latency_s < 0.0
        or iterations < 1
        or maximum_prediction_s <= 0.0
    ):
        raise ValueError("intercept timing, speeds and iterations are invalid")
    progress, cumulative = _polyline_progress(trailer.position_enu_m[:2], route)
    initial_speed = min(
        trailer_cruise_speed_m_s,
        max(0.0, float(np.linalg.norm(trailer.velocity_enu_m_s[:2]))),
    )

    def travel_distance(duration_s: float) -> tuple[float, float]:
        duration = max(0.0, float(duration_s))
        ramp_time = max(
            0.0,
            (trailer_cruise_speed_m_s - initial_speed)
            / trailer_acceleration_m_s2,
        )
        accelerating = min(duration, ramp_time)
        distance = (
            initial_speed * accelerating
            + 0.5 * trailer_acceleration_m_s2 * accelerating**2
            + trailer_cruise_speed_m_s * max(0.0, duration - ramp_time)
        )
        speed = min(
            trailer_cruise_speed_m_s,
            initial_speed + trailer_acceleration_m_s2 * duration,
        )
        return distance, speed

    eta = float(planner_latency_s)
    length = 0.0
    prediction = _sample_polyline_route(
        route, cumulative, progress, initial_speed, trailer.position_enu_m[2]
    )
    for _ in range(iterations):
        distance, predicted_speed = travel_distance(eta)
        prediction = _sample_polyline_route(
            route,
            cumulative,
            progress + distance,
            predicted_speed,
            trailer.position_enu_m[2],
        )
        length = float(path_length_to(prediction.position_enu_m.copy()))
        if not math.isfinite(length) or length < 0.0:
            raise ValueError("path length callback returned an invalid length")
        eta = min(
            maximum_prediction_s,
            planner_latency_s + length / expected_drone_speed_m_s,
        )
    distance, predicted_speed = travel_distance(eta)
    prediction = _sample_polyline_route(
        route,
        cumulative,
        progress + distance,
        predicted_speed,
        trailer.position_enu_m[2],
    )
    return InterceptSolution(
        prediction.position_enu_m.copy(),
        prediction.velocity_enu_m_s.copy(),
        eta,
        iterations,
        length,
    )


@dataclass(frozen=True)
class JerkMPCResult:
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    accelerations_m_s2: np.ndarray
    jerks_m_s3: np.ndarray
    success: bool
    deadline_missed: bool
    solve_time_s: float
    iterations: int
    message: str
    cost: float


class JerkInputMPC:
    """Finite-horizon constrained MPC with jerk as the optimized input.

    Position bounds represent a local convex free-space corridor.  Optional
    obstacle spheres are enforced as hard nonlinear inequalities.  A failed or
    late solution is never executed; callers must hold/brake and request a new
    global plan.
    """

    def __init__(
        self,
        dt_s: float = 0.15,
        horizon_steps: int = 8,
        velocity_limit_m_s: float = 5.0,
        acceleration_limit_m_s2: float = 3.0,
        jerk_limit_m_s3: float = 5.0,
        deadline_s: float = 0.075,
        maximum_iterations: int = 60,
    ) -> None:
        if min(dt_s, velocity_limit_m_s, acceleration_limit_m_s2, jerk_limit_m_s3, deadline_s) <= 0.0:
            raise ValueError("MPC timing and limits must be positive")
        if horizon_steps < 2:
            raise ValueError("MPC horizon must contain at least two steps")
        if maximum_iterations < 1:
            raise ValueError("MPC maximum iterations must be positive")
        self.dt_s = float(dt_s)
        self.horizon_steps = int(horizon_steps)
        self.velocity_limit_m_s = float(velocity_limit_m_s)
        self.acceleration_limit_m_s2 = float(acceleration_limit_m_s2)
        self.jerk_limit_m_s3 = float(jerk_limit_m_s3)
        self.deadline_s = float(deadline_s)
        self.maximum_iterations = int(maximum_iterations)
        self._warm = np.zeros((horizon_steps, 3), dtype=float)
        self._previous_jerk = np.zeros(3, dtype=float)
        # Condensed exact triple-integrator dynamics.  Supplying analytic
        # derivatives avoids SLSQP finite-differencing all 3*N jerk inputs,
        # which was the dominant source of >100 ms solve times.
        n = self.horizon_steps
        dt = self.dt_s
        self._position_map = np.zeros((n, n), dtype=float)
        self._velocity_map = np.zeros((n, n), dtype=float)
        self._acceleration_map = np.zeros((n, n), dtype=float)
        for row in range(n):
            for column in range(row + 1):
                lag = row - column
                self._position_map[row, column] = dt**3 * (
                    1.0 / 6.0 + 0.5 * lag + 0.5 * lag * lag
                )
                self._velocity_map[row, column] = dt**2 * (lag + 0.5)
                self._acceleration_map[row, column] = dt
        self._jerk_difference = np.eye(n, dtype=float)
        self._jerk_difference[1:, :-1] -= np.eye(n - 1, dtype=float)
        identity_xyz = np.eye(3, dtype=float)
        self._position_flat_map = np.kron(self._position_map, identity_xyz)

    def rollout(
        self,
        position_m: Sequence[float],
        velocity_m_s: Sequence[float],
        acceleration_m_s2: Sequence[float],
        jerks_m_s3: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.asarray(position_m, dtype=float).copy()
        v = np.asarray(velocity_m_s, dtype=float).copy()
        a = np.asarray(acceleration_m_s2, dtype=float).copy()
        jerks = np.asarray(jerks_m_s3, dtype=float)
        if p.shape != (3,) or v.shape != (3,) or a.shape != (3,):
            raise ValueError("initial MPC state must contain three-vectors")
        if jerks.shape != (self.horizon_steps, 3):
            raise ValueError("jerk sequence has the wrong shape")
        positions = np.empty_like(jerks)
        velocities = np.empty_like(jerks)
        accelerations = np.empty_like(jerks)
        dt = self.dt_s
        for index, jerk in enumerate(jerks):
            p = p + v * dt + 0.5 * a * dt**2 + jerk * dt**3 / 6.0
            v = v + a * dt + 0.5 * jerk * dt**2
            a = a + jerk * dt
            positions[index], velocities[index], accelerations[index] = p, v, a
        return positions, velocities, accelerations

    def _references(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape == (3,):
            return np.repeat(array[None, :], self.horizon_steps, axis=0)
        if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
            raise ValueError("reference must have shape (3,) or (N,3)")
        if len(array) < self.horizon_steps:
            array = np.vstack((array, np.repeat(array[-1][None, :], self.horizon_steps - len(array), axis=0)))
        return array[: self.horizon_steps].copy()

    def solve(
        self,
        position_m: Sequence[float],
        velocity_m_s: Sequence[float],
        acceleration_m_s2: Sequence[float],
        reference_positions_m: Sequence[float] | np.ndarray,
        reference_velocities_m_s: Sequence[float] | np.ndarray,
        corridor_bounds_m: np.ndarray | None = None,
        obstacle_spheres_m: Sequence[tuple[Sequence[float], float]] = (),
    ) -> JerkMPCResult:
        reference_p = self._references(reference_positions_m)
        reference_v = self._references(reference_velocities_m_s)
        if corridor_bounds_m is not None:
            corridor = np.asarray(corridor_bounds_m, dtype=float)
            if corridor.shape == (6,):
                corridor = np.repeat(corridor[None, :], self.horizon_steps, axis=0)
            if corridor.shape != (self.horizon_steps, 6):
                raise ValueError("corridor bounds must be [xmin,xmax,ymin,ymax,zmin,zmax]")
        else:
            corridor = None
        spheres = [(np.asarray(center, dtype=float), float(radius)) for center, radius in obstacle_spheres_m]
        if any(center.shape != (3,) or radius <= 0.0 for center, radius in spheres):
            raise ValueError("obstacle spheres require a 3-D center and positive radius")

        p0 = np.asarray(position_m, dtype=float)
        v0 = np.asarray(velocity_m_s, dtype=float)
        a0 = np.asarray(acceleration_m_s2, dtype=float)
        if p0.shape != (3,) or v0.shape != (3,) or a0.shape != (3,):
            raise ValueError("initial MPC state must contain three-vectors")
        if not np.all(np.isfinite(np.concatenate((p0, v0, a0)))):
            raise ValueError("initial MPC state must be finite")
        steps = np.arange(1, self.horizon_steps + 1, dtype=float)[:, None]
        free_p = p0 + steps * self.dt_s * v0 + 0.5 * (steps * self.dt_s) ** 2 * a0
        free_v = v0 + steps * self.dt_s * a0
        free_a = np.repeat(a0[None, :], self.horizon_steps, axis=0)

        def predicted(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            jerk = flat.reshape(self.horizon_steps, 3)
            return (
                free_p + self._position_map @ jerk,
                free_v + self._velocity_map @ jerk,
                free_a + self._acceleration_map @ jerk,
            )

        def objective(flat: np.ndarray) -> float:
            p, v, a = predicted(flat)
            j = flat.reshape(self.horizon_steps, 3)
            delta_j = np.diff(np.vstack((self._previous_jerk, j)), axis=0)
            terminal = p[-1] - reference_p[-1]
            return float(
                12.0 * np.sum((p - reference_p) ** 2)
                + 3.0 * np.sum((v - reference_v) ** 2)
                + 0.3 * np.sum(a**2)
                + 0.08 * np.sum(j**2)
                + 0.12 * np.sum(delta_j**2)
                + 20.0 * np.dot(terminal, terminal)
            )

        def objective_jacobian(flat: np.ndarray) -> np.ndarray:
            p, v, a = predicted(flat)
            jerk = flat.reshape(self.horizon_steps, 3)
            delta_jerk = self._jerk_difference @ jerk
            delta_jerk[0] -= self._previous_jerk
            gradient = (
                24.0 * self._position_map.T @ (p - reference_p)
                + 6.0 * self._velocity_map.T @ (v - reference_v)
                + 0.6 * self._acceleration_map.T @ a
                + 0.16 * jerk
                + 0.24 * self._jerk_difference.T @ delta_jerk
                + 40.0
                * self._position_map[-1, :, None]
                * (p[-1] - reference_p[-1])[None, :]
            )
            return gradient.reshape(-1)

        constraints: list[dict[str, object]] = []

        def dynamic_limits(flat: np.ndarray) -> np.ndarray:
            _p, v, a = predicted(flat)
            return np.concatenate(
                (
                    self.velocity_limit_m_s**2 - np.sum(v * v, axis=1),
                    self.acceleration_limit_m_s2**2 - np.sum(a * a, axis=1),
                )
            )

        def dynamic_limits_jacobian(flat: np.ndarray) -> np.ndarray:
            _p, velocity, acceleration = predicted(flat)
            count = self.horizon_steps
            jacobian = np.zeros((2 * count, 3 * count), dtype=float)
            for row in range(count):
                for column in range(row + 1):
                    xyz = slice(3 * column, 3 * column + 3)
                    jacobian[row, xyz] = (
                        -2.0 * self._velocity_map[row, column] * velocity[row]
                    )
                    jacobian[count + row, xyz] = (
                        -2.0
                        * self._acceleration_map[row, column]
                        * acceleration[row]
                    )
            return jacobian

        constraints.append(
            {
                "type": "ineq",
                "fun": dynamic_limits,
                "jac": dynamic_limits_jacobian,
            }
        )
        if corridor is not None:
            lower = corridor[:, (0, 2, 4)].reshape(-1) - free_p.reshape(-1)
            upper = corridor[:, (1, 3, 5)].reshape(-1) - free_p.reshape(-1)
            constraints.append(
                optimize.LinearConstraint(self._position_flat_map, lower, upper)
            )
        for center, radius in spheres:
            def obstacle_constraint(
                flat: np.ndarray,
                obstacle_center: np.ndarray = center,
                obstacle_radius: float = radius,
            ) -> np.ndarray:
                positions, _v, _a = predicted(flat)
                delta = positions - obstacle_center
                return np.sum(delta * delta, axis=1) - obstacle_radius**2

            def obstacle_jacobian(
                flat: np.ndarray,
                obstacle_center: np.ndarray = center,
            ) -> np.ndarray:
                positions, _v, _a = predicted(flat)
                delta = positions - obstacle_center
                count = self.horizon_steps
                jacobian = np.zeros((count, 3 * count), dtype=float)
                for row in range(count):
                    for column in range(row + 1):
                        xyz = slice(3 * column, 3 * column + 3)
                        jacobian[row, xyz] = (
                            2.0 * self._position_map[row, column] * delta[row]
                        )
                return jacobian

            constraints.append(
                {
                    "type": "ineq",
                    "fun": obstacle_constraint,
                    "jac": obstacle_jacobian,
                }
            )

        started = time.monotonic()
        result = None
        message = "solver not run"
        timed_out = False

        class _SolveDeadline(RuntimeError):
            pass

        def enforce_deadline(_flat: np.ndarray) -> None:
            if time.monotonic() - started > self.deadline_s:
                raise _SolveDeadline("MPC wall-clock deadline exceeded")

        try:
            result = optimize.minimize(
                objective,
                self._warm.reshape(-1),
                jac=objective_jacobian,
                method="SLSQP",
                bounds=[(-self.jerk_limit_m_s3, self.jerk_limit_m_s3)] * (self.horizon_steps * 3),
                constraints=constraints,
                callback=enforce_deadline,
                options={
                    "maxiter": self.maximum_iterations,
                    "ftol": 1.0e-5,
                    "disp": False,
                },
            )
            message = str(result.message)
        except _SolveDeadline as exc:
            timed_out = True
            message = str(exc)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            message = f"solver exception: {exc}"
        elapsed = time.monotonic() - started
        deadline_missed = timed_out or elapsed > self.deadline_s
        valid = result is not None and bool(result.success) and np.all(np.isfinite(result.x)) and not deadline_missed
        if valid:
            jerks = result.x.reshape(self.horizon_steps, 3)
            p, v, a = self.rollout(position_m, velocity_m_s, acceleration_m_s2, jerks)
            # Recheck all constraints numerically; SLSQP success alone is not a safety proof.
            valid = bool(np.all(dynamic_limits(result.x) >= -1.0e-5))
            if corridor is not None:
                valid = valid and bool(
                    np.all(p >= corridor[:, (0, 2, 4)] - 1.0e-5)
                    and np.all(p <= corridor[:, (1, 3, 5)] + 1.0e-5)
                )
            for center, radius in spheres:
                valid = valid and bool(np.all(np.linalg.norm(p - center, axis=1) >= radius - 1.0e-5))
        if valid and result is not None:
            self._warm[:-1] = jerks[1:]
            self._warm[-1] = jerks[-1]
            self._previous_jerk = jerks[0].copy()
            return JerkMPCResult(p, v, a, jerks, True, deadline_missed, elapsed, int(result.nit), message, float(result.fun))

        # Explicit safe fallback: no stale optimizer command is reused.  Jerk
        # moves acceleration toward braking, and the caller treats success=False
        # as HOLD_REPLAN instead of claiming this fallback is MPC.
        v0 = np.asarray(velocity_m_s, dtype=float)
        a0 = np.asarray(acceleration_m_s2, dtype=float)
        brake_a = np.clip(-1.5 * v0, -self.acceleration_limit_m_s2, self.acceleration_limit_m_s2)
        brake_j = np.clip((brake_a - a0) / self.dt_s, -self.jerk_limit_m_s3, self.jerk_limit_m_s3)
        jerks = np.repeat(brake_j[None, :], self.horizon_steps, axis=0)
        p, v, a = self.rollout(position_m, velocity_m_s, acceleration_m_s2, jerks)
        return JerkMPCResult(
            p, v, a, jerks, False, deadline_missed, elapsed,
            0 if result is None else int(getattr(result, "nit", 0)),
            f"HOLD_REPLAN braking fallback: {message}", float("nan"),
        )


class RelativeTargetEstimator:
    """Timestamp-driven constant-velocity KF for relative xyz/yaw state."""

    def __init__(self, process_noise: float = 0.25, innovation_gate: float = 16.0) -> None:
        if process_noise <= 0.0 or innovation_gate <= 0.0:
            raise ValueError("estimator tunables must be positive")
        self.x = np.zeros(8, dtype=float)  # px,py,pz,vx,vy,vz,yaw,yaw_rate
        self.P = np.eye(8, dtype=float) * 25.0
        self.process_noise = float(process_noise)
        self.innovation_gate = float(innovation_gate)
        self.stamp_s: float | None = None
        self.accepted = 0
        self.rejected = 0

    def predict(self, stamp_s: float) -> None:
        stamp = float(stamp_s)
        if self.stamp_s is None:
            self.stamp_s = stamp
            return
        dt = stamp - self.stamp_s
        if not 0.0 <= dt <= 2.0:
            self.stamp_s = stamp
            self.P += np.eye(8) * 10.0
            return
        F = np.eye(8)
        F[0, 3] = F[1, 4] = F[2, 5] = F[6, 7] = dt
        q = self.process_noise
        Q = np.eye(8) * q * max(dt, 1.0e-3)
        self.x = F @ self.x
        self.x[6] = math.atan2(math.sin(self.x[6]), math.cos(self.x[6]))
        self.P = F @ self.P @ F.T + Q
        self.stamp_s = stamp

    def update(
        self,
        stamp_s: float,
        relative_position_m: Sequence[float],
        relative_yaw_rad: float,
        covariance: np.ndarray,
    ) -> bool:
        self.predict(stamp_s)
        z = np.asarray((*relative_position_m, float(relative_yaw_rad)), dtype=float)
        if z.shape != (4,) or not np.all(np.isfinite(z)):
            self.rejected += 1
            return False
        R = np.asarray(covariance, dtype=float)
        if R.shape == (4,):
            R = np.diag(R)
        if R.shape != (4, 4) or np.any(np.diag(R) <= 0.0):
            raise ValueError("marker covariance must be positive 4x4 or diagonal")
        H = np.zeros((4, 8), dtype=float)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 6] = 1.0
        innovation = z - H @ self.x
        innovation[3] = math.atan2(math.sin(innovation[3]), math.cos(innovation[3]))
        S = H @ self.P @ H.T + R
        mahalanobis = float(innovation @ np.linalg.solve(S, innovation))
        if not math.isfinite(mahalanobis) or mahalanobis > self.innovation_gate:
            self.rejected += 1
            return False
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x += K @ innovation
        self.x[6] = math.atan2(math.sin(self.x[6]), math.cos(self.x[6]))
        identity = np.eye(8)
        self.P = (identity - K @ H) @ self.P @ (identity - K @ H).T + K @ R @ K.T
        self.accepted += 1
        return True


class TouchdownGate:
    """Continuous multi-signal touchdown confirmation; never disarms in air."""

    def __init__(
        self,
        maximum_range_m: float = 0.22,
        maximum_vertical_speed_m_s: float = 0.18,
        dwell_s: float = 1.0,
    ) -> None:
        if min(maximum_range_m, maximum_vertical_speed_m_s, dwell_s) <= 0.0:
            raise ValueError("touchdown thresholds must be positive")
        self.maximum_range_m = float(maximum_range_m)
        self.maximum_vertical_speed_m_s = float(maximum_vertical_speed_m_s)
        self.dwell_s = float(dwell_s)
        self._since_s: float | None = None

    def reset(self) -> None:
        """Clear accumulated dwell after any sensor/control interruption."""

        self._since_s = None

    def update(
        self,
        now_s: float,
        range_to_deck_m: float | None,
        relative_vertical_speed_m_s: float,
        px4_ground_contact: bool,
        px4_landed: bool,
        at_rest: bool,
    ) -> bool:
        range_ok = range_to_deck_m is not None and math.isfinite(range_to_deck_m) and range_to_deck_m <= self.maximum_range_m
        kinematic_ok = abs(relative_vertical_speed_m_s) <= self.maximum_vertical_speed_m_s
        combined = range_ok and kinematic_ok and (px4_landed or (px4_ground_contact and at_rest))
        if not combined:
            self._since_s = None
            return False
        if self._since_s is None:
            self._since_s = float(now_s)
            return False
        return float(now_s) - self._since_s >= self.dwell_s


class MissionCommandGate:
    """Pure command gate proving that normal ``land`` means trailer return."""

    RETURNABLE = {
        MissionState.GLOBAL_PLAN,
        MissionState.GLOBAL_TRACK,
        MissionState.LOCAL_AVOIDANCE,
        MissionState.WAYPOINT_HOLD,
    }

    def __init__(self, initial: MissionState = MissionState.IDLE) -> None:
        self.state = initial
        self.last_reason = "initialized"

    def request_trailer_land(self) -> bool:
        if self.state not in self.RETURNABLE:
            return False
        self.state = MissionState.RETURN_REQUESTED
        self.last_reason = "operator land command: return to moving trailer"
        return True

    def request_emergency_land(self) -> None:
        self.state = MissionState.EMERGENCY_LAND
        self.last_reason = "operator emergency_land command"
