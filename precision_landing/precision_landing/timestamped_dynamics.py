"""Timestamped vehicle-response and trailer-motion prediction.

This module deliberately has no ROS, MAVROS, PX4, or Gazebo imports.  The ROS
boundary is responsible for converting message fields to world-ENU vectors and
PX4 sample-time seconds before calling these helpers.

The two models solve separate problems:

* :class:`VehicleResponseEstimator` is a variable-step linear Kalman filter.
  Per world axis its state is ``[position, velocity, actuator_acceleration,
  disturbance_acceleration]``.  It uses message sample timestamps, models
  first-order actuator response, and predicts the measured state forward to
  the solver snapshot time without mutating its posterior.
* :class:`TrailerRk4Predictor` propagates a planar kinematic-bicycle state with
  RK4.  Steering can be inferred from observed speed and yaw rate, so a caller
  never has to invent steering when the trailer is stationary.

Neither class is a controller and neither publishes commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .target_fusion import validate_covariance, wrap_angle


_EPS = 1.0e-9
_MAX_CONSECUTIVE_INNOVATION_REJECTIONS = 5


def _finite_vector(
    value: Sequence[float], size: int, description: str
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{description} must be a finite {size}-vector")
    return vector.copy()


def _bounded_norm(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum_norm or norm <= _EPS:
        return vector.copy()
    return vector * (maximum_norm / norm)


@dataclass(frozen=True)
class VehicleResponseParameters:
    """Read-only tuning and timing limits for UAV response estimation."""

    actuator_time_constant_s: float = 0.18
    maximum_acceleration_m_s2: float = 15.0
    maximum_disturbance_m_s2: float = 5.0
    duplicate_tolerance_s: float = 1.0e-6
    timestamp_reset_threshold_s: float = 0.50
    maximum_sample_gap_s: float = 0.50
    maximum_prediction_horizon_s: float = 0.25
    position_measurement_variance_m2: float = 4.0e-4
    velocity_measurement_variance_m2_s2: float = 2.5e-3
    kinematic_acceleration_spectral_density_m2_s3: float = 0.10
    actuator_process_variance_m2_s4: float = 0.25
    disturbance_random_walk_spectral_density_m2_s5: float = 0.05
    initial_actuator_acceleration_variance_m2_s4: float = 4.0
    initial_disturbance_variance_m2_s4: float = 2.0
    innovation_nis_gate: float = 36.0
    covariance_eigenvalue_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        values = {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("vehicle response parameters must be finite")
        positive = {
            "actuator_time_constant_s",
            "maximum_acceleration_m_s2",
            "maximum_disturbance_m_s2",
            "duplicate_tolerance_s",
            "timestamp_reset_threshold_s",
            "maximum_sample_gap_s",
            "maximum_prediction_horizon_s",
            "position_measurement_variance_m2",
            "velocity_measurement_variance_m2_s2",
            "initial_actuator_acceleration_variance_m2_s4",
            "initial_disturbance_variance_m2_s4",
            "innovation_nis_gate",
        }
        if any(values[name] <= 0.0 for name in positive):
            raise ValueError(
                "vehicle response timing and bounds must be positive"
            )
        if self.maximum_sample_gap_s <= self.duplicate_tolerance_s:
            raise ValueError(
                "maximum sample gap must exceed duplicate tolerance"
            )
        nonnegative = {
            "kinematic_acceleration_spectral_density_m2_s3",
            "actuator_process_variance_m2_s4",
            "disturbance_random_walk_spectral_density_m2_s5",
            "covariance_eigenvalue_floor",
        }
        if any(values[name] < 0.0 for name in nonnegative):
            raise ValueError("vehicle process noise cannot be negative")


@dataclass(frozen=True)
class VehicleResponseState:
    """Measured vehicle state at one PX4 sample timestamp."""

    sample_time_s: float
    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray
    acceleration_enu_m_s2: np.ndarray
    disturbance_enu_m_s2: np.ndarray
    commanded_acceleration_enu_m_s2: np.ndarray


@dataclass(frozen=True)
class VehicleUpdateResult:
    """Result of ingesting one timestamped position/velocity sample."""

    accepted: bool
    reset_detected: bool
    reason: str
    filter_alpha: float | None
    state: VehicleResponseState | None


@dataclass(frozen=True)
class VehiclePrediction:
    """Non-mutating vehicle-state prediction at the solver snapshot time."""

    valid: bool
    reason: str
    age_s: float | None
    state: VehicleResponseState | None


def _copy_vehicle_state(state: VehicleResponseState) -> VehicleResponseState:
    return VehicleResponseState(
        state.sample_time_s,
        state.position_enu_m.copy(),
        state.velocity_enu_m_s.copy(),
        state.acceleration_enu_m_s2.copy(),
        state.disturbance_enu_m_s2.copy(),
        state.commanded_acceleration_enu_m_s2.copy(),
    )


class VehicleResponseEstimator:
    """Timestamp-driven position/velocity Kalman response estimator.

    Each ENU axis has an independent four-state posterior ``[p, v, a_lag, d]``
    and a 4x4 covariance.  ``a_lag`` follows the commanded acceleration through
    a first-order actuator model, while ``d`` is a random-walk acceleration
    disturbance.  The reported physical acceleration is ``a_lag + d``.

    Position and velocity measurements deliberately have small covariance so
    the posterior stays close to MAVROS rather than becoming an open-loop
    trajectory generator.  A per-axis two-degree-of-freedom NIS gate rejects
    inconsistent position/velocity pairs before a Joseph-form update.  No IMU
    is mixed here because this ROS boundary does not provide a gravity- and
    bias-corrected acceleration in the same ENU frame.
    """

    def __init__(
        self, parameters: VehicleResponseParameters | None = None
    ) -> None:
        self.parameters = parameters or VehicleResponseParameters()
        self._state: VehicleResponseState | None = None
        self._filter_state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._last_nis: np.ndarray | None = None
        self._consecutive_innovation_rejections = 0

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> VehicleResponseState | None:
        if self._state is None:
            return None
        return _copy_vehicle_state(self._state)

    @property
    def covariance(self) -> np.ndarray | None:
        """Return a copy of posterior covariance in ``[axis, state, state]``."""
        if self._covariance is None:
            return None
        return self._covariance.copy()

    @property
    def last_nis(self) -> np.ndarray | None:
        """Return the last per-axis ``[position, velocity]`` NIS values."""
        if self._last_nis is None:
            return None
        return self._last_nis.copy()

    @property
    def consecutive_innovation_rejections(self) -> int:
        return int(self._consecutive_innovation_rejections)

    def reset(self) -> None:
        self._state = None
        self._filter_state = None
        self._covariance = None
        self._last_nis = None
        self._consecutive_innovation_rejections = 0

    def _command(
        self, commanded_acceleration_enu_m_s2: Sequence[float] | None
    ) -> np.ndarray:
        if commanded_acceleration_enu_m_s2 is None:
            if self._state is None:
                return np.zeros(3, dtype=float)
            return self._state.commanded_acceleration_enu_m_s2.copy()
        command = _finite_vector(
            commanded_acceleration_enu_m_s2,
            3,
            "commanded acceleration",
        )
        return _bounded_norm(
            command, self.parameters.maximum_acceleration_m_s2
        )

    def _transition(
        self, dt_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact zero-order-hold transition for one ENU axis."""
        tau = self.parameters.actuator_time_constant_s
        decay = math.exp(-dt_s / tau)
        acceleration_to_velocity = tau * (1.0 - decay)
        acceleration_to_position = (
            tau * dt_s - tau * tau * (1.0 - decay)
        )
        transition = np.asarray(
            (
                (
                    1.0,
                    dt_s,
                    acceleration_to_position,
                    0.5 * dt_s * dt_s,
                ),
                (0.0, 1.0, acceleration_to_velocity, dt_s),
                (0.0, 0.0, decay, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        command_gain = np.asarray(
            (
                0.5 * dt_s * dt_s - acceleration_to_position,
                dt_s - acceleration_to_velocity,
                1.0 - decay,
                0.0,
            ),
            dtype=float,
        )
        return transition, command_gain

    def _process_covariance(
        self, dt_s: float, command_gain: np.ndarray
    ) -> np.ndarray:
        """Build a positive-semidefinite variable-step process covariance."""
        parameters = self.parameters
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        dt4 = dt3 * dt_s
        dt5 = dt4 * dt_s

        # Unmodelled translational acceleration as continuous white noise.
        process = np.zeros((4, 4), dtype=float)
        q_kinematic = (
            parameters.kinematic_acceleration_spectral_density_m2_s3
        )
        process[:2, :2] += q_kinematic * np.asarray(
            ((dt3 / 3.0, dt2 / 2.0), (dt2 / 2.0, dt_s)),
            dtype=float,
        )

        # Bounded actuator-model uncertainty follows the same exact ZOH gain.
        q_actuator = parameters.actuator_process_variance_m2_s4
        process += q_actuator * np.outer(command_gain, command_gain)

        # Exact integrated covariance for d_dot = white noise.  The disturbance
        # drives acceleration, velocity, and position continuously.
        q_disturbance = (
            parameters.disturbance_random_walk_spectral_density_m2_s5
        )
        disturbance_covariance = np.asarray(
            (
                (dt5 / 20.0, dt4 / 8.0, 0.0, dt3 / 6.0),
                (dt4 / 8.0, dt3 / 3.0, 0.0, dt2 / 2.0),
                (0.0, 0.0, 0.0, 0.0),
                (dt3 / 6.0, dt2 / 2.0, 0.0, dt_s),
            ),
            dtype=float,
        )
        process += q_disturbance * disturbance_covariance
        return self._stabilize_covariance(process, apply_floor=False)

    def _stabilize_covariance(
        self, covariance: np.ndarray, *, apply_floor: bool = True
    ) -> np.ndarray:
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = (
            self.parameters.covariance_eigenvalue_floor
            if apply_floor
            else 0.0
        )
        stabilized = (
            eigenvectors
            @ np.diag(np.maximum(eigenvalues, floor))
            @ eigenvectors.T
        )
        return 0.5 * (stabilized + stabilized.T)

    def _enforce_state_bounds(self, state: np.ndarray) -> np.ndarray:
        bounded = np.asarray(state, dtype=float).copy()
        bounded[:, 3] = _bounded_norm(
            bounded[:, 3], self.parameters.maximum_disturbance_m_s2
        )
        total_acceleration = bounded[:, 2] + bounded[:, 3]
        total_acceleration = _bounded_norm(
            total_acceleration,
            self.parameters.maximum_acceleration_m_s2,
        )
        bounded[:, 2] = total_acceleration - bounded[:, 3]
        return bounded

    def _external_state(
        self,
        sample_time_s: float,
        filter_state: np.ndarray,
        command: np.ndarray,
    ) -> VehicleResponseState:
        acceleration = filter_state[:, 2] + filter_state[:, 3]
        return VehicleResponseState(
            float(sample_time_s),
            filter_state[:, 0].copy(),
            filter_state[:, 1].copy(),
            acceleration.copy(),
            filter_state[:, 3].copy(),
            command.copy(),
        )

    def _store_posterior(
        self,
        sample_time_s: float,
        filter_state: np.ndarray,
        covariance: np.ndarray,
        command: np.ndarray,
    ) -> None:
        bounded_state = self._enforce_state_bounds(filter_state)
        self._filter_state = bounded_state
        self._covariance = np.asarray(
            [self._stabilize_covariance(axis) for axis in covariance],
            dtype=float,
        )
        self._state = self._external_state(
            sample_time_s, bounded_state, command
        )

    def update(
        self,
        sample_time_s: float,
        position_enu_m: Sequence[float],
        velocity_enu_m_s: Sequence[float],
        commanded_acceleration_enu_m_s2: Sequence[float] | None = None,
    ) -> VehicleUpdateResult:
        """Ingest one position/velocity sample using its message timestamp."""
        try:
            sample_time = float(sample_time_s)
            position = _finite_vector(position_enu_m, 3, "vehicle position")
            velocity = _finite_vector(
                velocity_enu_m_s, 3, "vehicle velocity"
            )
            command = self._command(commanded_acceleration_enu_m_s2)
        except (TypeError, ValueError):
            return VehicleUpdateResult(
                False, False, "invalid_sample", None, self.state
            )
        if not math.isfinite(sample_time) or sample_time <= 0.0:
            return VehicleUpdateResult(
                False, False, "invalid_timestamp", None, self.state
            )

        if self._state is None:
            filter_state = np.zeros((3, 4), dtype=float)
            filter_state[:, 0] = position
            filter_state[:, 1] = velocity
            initial_diagonal = np.asarray(
                (
                    self.parameters.position_measurement_variance_m2,
                    self.parameters.velocity_measurement_variance_m2_s2,
                    self.parameters.initial_actuator_acceleration_variance_m2_s4,
                    self.parameters.initial_disturbance_variance_m2_s4,
                ),
                dtype=float,
            )
            covariance = np.repeat(
                np.diag(initial_diagonal)[None, :, :], 3, axis=0
            )
            self._store_posterior(
                sample_time, filter_state, covariance, command
            )
            self._consecutive_innovation_rejections = 0
            return VehicleUpdateResult(
                True, False, "initialized", None, self.state
            )

        previous = self._state
        dt = sample_time - previous.sample_time_s
        tolerance = self.parameters.duplicate_tolerance_s
        if abs(dt) <= tolerance:
            return VehicleUpdateResult(
                False, False, "duplicate_timestamp", None, self.state
            )
        if dt < 0.0:
            if -dt >= self.parameters.timestamp_reset_threshold_s:
                self.reset()
                return VehicleUpdateResult(
                    False, True, "timestamp_reset", None, None
                )
            return VehicleUpdateResult(
                False, False, "out_of_order_timestamp", None, self.state
            )
        if dt > self.parameters.maximum_sample_gap_s:
            self.reset()
            return VehicleUpdateResult(
                False, True, "large_sample_gap", None, None
            )

        if self._filter_state is None or self._covariance is None:
            self.reset()
            return VehicleUpdateResult(
                False, True, "invalid_internal_state", None, None
            )

        transition, command_gain = self._transition(dt)
        process_covariance = self._process_covariance(dt, command_gain)
        old_command = previous.commanded_acceleration_enu_m_s2
        predicted_state = np.empty_like(self._filter_state)
        predicted_covariance = np.empty_like(self._covariance)
        for axis in range(3):
            predicted_state[axis] = (
                transition @ self._filter_state[axis]
                + command_gain * old_command[axis]
            )
            predicted_covariance[axis] = self._stabilize_covariance(
                transition
                @ self._covariance[axis]
                @ transition.T
                + process_covariance
            )

        measurement_matrix = np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
            dtype=float,
        )
        measurement_covariance = np.diag(
            (
                self.parameters.position_measurement_variance_m2,
                self.parameters.velocity_measurement_variance_m2_s2,
            )
        )
        innovations: list[np.ndarray] = []
        innovation_covariances: list[np.ndarray] = []
        nis = np.empty(3, dtype=float)
        for axis in range(3):
            measurement = np.asarray(
                (position[axis], velocity[axis]), dtype=float
            )
            innovation = measurement - measurement_matrix @ predicted_state[axis]
            innovation_covariance = (
                measurement_matrix
                @ predicted_covariance[axis]
                @ measurement_matrix.T
                + measurement_covariance
            )
            innovations.append(innovation)
            innovation_covariances.append(innovation_covariance)
            try:
                nis[axis] = float(
                    innovation.T
                    @ np.linalg.solve(innovation_covariance, innovation)
                )
            except np.linalg.LinAlgError:
                self.reset()
                return VehicleUpdateResult(
                    False, True, "singular_innovation_covariance", None, None
                )
        self._last_nis = nis.copy()

        if (
            not np.all(np.isfinite(nis))
            or float(np.max(nis)) > self.parameters.innovation_nis_gate
        ):
            self._consecutive_innovation_rejections += 1
            if (
                self._consecutive_innovation_rejections
                >= _MAX_CONSECUTIVE_INNOVATION_REJECTIONS
            ):
                # Never let one rejected posterior lock the observer into an
                # indefinitely diverging predict/reject cycle. Re-anchor its
                # measured p/v after five consecutive finite timestamped
                # samples; acceleration then restarts conservatively at zero.
                self.reset()
                initialized = self.update(
                    sample_time,
                    position,
                    velocity,
                    commanded_acceleration_enu_m_s2,
                )
                return VehicleUpdateResult(
                    initialized.accepted,
                    False,
                    "innovation_reinitialized",
                    initialized.filter_alpha,
                    initialized.state,
                )
            # Rejected measurements do not corrupt the posterior, but filter
            # time still advances using the valid process model.  This avoids
            # turning one outlier into a later artificial large-gap reset.
            self._store_posterior(
                sample_time,
                predicted_state,
                predicted_covariance,
                command,
            )
            return VehicleUpdateResult(
                False,
                False,
                "innovation_gate_rejected",
                None,
                self.state,
            )

        identity = np.eye(4, dtype=float)
        posterior_state = np.empty_like(predicted_state)
        posterior_covariance = np.empty_like(predicted_covariance)
        velocity_gains = np.empty(3, dtype=float)
        for axis in range(3):
            covariance = predicted_covariance[axis]
            innovation_covariance = innovation_covariances[axis]
            try:
                gain = np.linalg.solve(
                    innovation_covariance,
                    measurement_matrix @ covariance,
                ).T
            except np.linalg.LinAlgError:
                self.reset()
                return VehicleUpdateResult(
                    False, True, "singular_innovation_covariance", None, None
                )
            posterior_state[axis] = (
                predicted_state[axis] + gain @ innovations[axis]
            )
            residual_transform = identity - gain @ measurement_matrix
            # Joseph form remains symmetric/PSD in finite precision.
            posterior_covariance[axis] = self._stabilize_covariance(
                residual_transform @ covariance @ residual_transform.T
                + gain @ measurement_covariance @ gain.T
            )
            velocity_gains[axis] = gain[1, 1]

        self._store_posterior(
            sample_time,
            posterior_state,
            posterior_covariance,
            command,
        )
        self._consecutive_innovation_rejections = 0
        filter_alpha = float(np.clip(np.mean(velocity_gains), 0.0, 1.0))
        return VehicleUpdateResult(
            True, False, "accepted", filter_alpha, self.state
        )

    def predict(
        self,
        solve_time_s: float,
        commanded_acceleration_enu_m_s2: Sequence[float] | None = None,
    ) -> VehiclePrediction:
        """Predict a copy of the last state to ``solve_time_s``.

        A first-order actuator model is integrated analytically while the
        bounded disturbance is held constant.  The estimator's canonical
        timestamp is never advanced by this query.
        """
        if self._state is None:
            return VehiclePrediction(False, "uninitialized", None, None)
        try:
            solve_time = float(solve_time_s)
        except (TypeError, ValueError):
            return VehiclePrediction(
                False, "invalid_prediction_time", None, None
            )
        if not math.isfinite(solve_time) or solve_time <= 0.0:
            return VehiclePrediction(
                False, "invalid_prediction_time", None, None
            )
        state = self._state
        age = solve_time - state.sample_time_s
        tolerance = self.parameters.duplicate_tolerance_s
        if age < -tolerance:
            return VehiclePrediction(
                False, "prediction_before_state", age, None
            )
        if age > self.parameters.maximum_prediction_horizon_s:
            return VehiclePrediction(
                False, "prediction_horizon_exceeded", age, None
            )
        if age <= tolerance:
            return VehiclePrediction(
                True, "current", max(age, 0.0), self.state
            )

        try:
            command = self._command(commanded_acceleration_enu_m_s2)
        except (TypeError, ValueError):
            return VehiclePrediction(False, "invalid_command", age, None)
        if self._filter_state is None:
            return VehiclePrediction(False, "invalid_internal_state", age, None)
        transition, command_gain = self._transition(age)
        predicted_filter_state = np.asarray(
            [
                transition @ self._filter_state[axis]
                + command_gain * command[axis]
                for axis in range(3)
            ],
            dtype=float,
        )
        predicted_filter_state = self._enforce_state_bounds(
            predicted_filter_state
        )
        prediction = self._external_state(
            solve_time, predicted_filter_state, command
        )
        return VehiclePrediction(True, "predicted", age, prediction)


@dataclass(frozen=True)
class TrailerPredictionParameters:
    """Kinematic-bicycle limits and prediction uncertainty."""

    wheelbase_m: float = 2.80
    maximum_steering_rad: float = math.radians(35.0)
    minimum_steering_speed_m_s: float = 0.20
    maximum_acceleration_m_s2: float = 6.0
    maximum_measurement_age_s: float = 1.00
    future_tolerance_s: float = 0.02
    maximum_integration_step_s: float = 0.02
    acceleration_process_variance: float = 0.50
    yaw_acceleration_process_variance: float = math.radians(15.0) ** 2
    position_variance_growth_per_s: float = 1.0e-4
    yaw_variance_growth_per_s: float = math.radians(0.5) ** 2
    speed_variance_growth_per_s: float = 1.0e-3

    def __post_init__(self) -> None:
        values = {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("trailer prediction parameters must be finite")
        nonnegative = {
            "future_tolerance_s",
            "position_variance_growth_per_s",
            "yaw_variance_growth_per_s",
            "speed_variance_growth_per_s",
        }
        if any(values[name] < 0.0 for name in nonnegative):
            raise ValueError(
                "trailer tolerances and covariance growth cannot be negative"
            )
        positive = set(values) - nonnegative
        if any(values[name] <= 0.0 for name in positive):
            raise ValueError("trailer model limits and noise must be positive")
        if self.maximum_steering_rad >= 0.5 * math.pi:
            raise ValueError("maximum steering must be less than pi/2")


@dataclass(frozen=True)
class TrailerBicycleState:
    """Planar bicycle state; covariance order is ``[x, y, yaw, speed]``."""

    sample_time_s: float
    position_enu_m: np.ndarray
    yaw_enu_rad: float
    speed_m_s: float
    acceleration_m_s2: float
    steering_rad: float
    covariance: np.ndarray


@dataclass(frozen=True)
class TrailerPrediction:
    """Trailer prediction and derived world-frame motion."""

    valid: bool
    reason: str
    age_s: float | None
    state: TrailerBicycleState | None
    velocity_enu_m_s: np.ndarray | None
    acceleration_enu_m_s2: np.ndarray | None
    yaw_rate_rad_s: float | None


@dataclass(frozen=True)
class TrailerPredictionHorizon:
    """Future bicycle-model samples aligned with one MPC horizon.

    Sample zero is one ``dt_s`` after ``start_time_s``.  This matches the
    condensed MPC convention, whose first state is at ``t + dt`` rather than
    at the captured initial state.  Covariance order is ``[x, y, yaw, speed]``.
    """

    valid: bool
    reason: str
    start_time_s: float | None
    dt_s: float | None
    positions_enu_m: np.ndarray | None
    velocities_enu_m_s: np.ndarray | None
    accelerations_enu_m_s2: np.ndarray | None
    yaws_enu_rad: np.ndarray | None
    yaw_rates_rad_s: np.ndarray | None
    covariances: np.ndarray | None


def _copy_trailer_state(state: TrailerBicycleState) -> TrailerBicycleState:
    return TrailerBicycleState(
        state.sample_time_s,
        state.position_enu_m.copy(),
        state.yaw_enu_rad,
        state.speed_m_s,
        state.acceleration_m_s2,
        state.steering_rad,
        state.covariance.copy(),
    )


class TrailerRk4Predictor:
    """Timestamped RK4 predictor for a no-slip kinematic bicycle."""

    def __init__(
        self, parameters: TrailerPredictionParameters | None = None
    ) -> None:
        self.parameters = parameters or TrailerPredictionParameters()

    def infer_steering(
        self, speed_m_s: float, yaw_rate_rad_s: float
    ) -> float:
        """Infer bounded steering without dividing by a near-zero speed."""
        speed = float(speed_m_s)
        yaw_rate = float(yaw_rate_rad_s)
        if not math.isfinite(speed) or not math.isfinite(yaw_rate):
            raise ValueError("speed and yaw rate must be finite")
        if abs(speed) < self.parameters.minimum_steering_speed_m_s:
            return 0.0
        steering = math.atan(
            self.parameters.wheelbase_m * yaw_rate / speed
        )
        return float(
            np.clip(
                steering,
                -self.parameters.maximum_steering_rad,
                self.parameters.maximum_steering_rad,
            )
        )

    def state_from_observation(
        self,
        sample_time_s: float,
        position_enu_m: Sequence[float],
        velocity_enu_m_s: Sequence[float],
        yaw_enu_rad: float,
        yaw_rate_rad_s: float,
        covariance: Sequence[Sequence[float]] | Sequence[float],
        acceleration_m_s2: float = 0.0,
    ) -> TrailerBicycleState:
        """Build a bicycle state from timestamped ENU odometry.

        Longitudinal speed is the observed XY velocity projected on the trailer
        heading.  This preserves reverse motion and rejects lateral simulator
        noise instead of treating its norm as forward speed.
        """
        sample_time = float(sample_time_s)
        position = _finite_vector(position_enu_m, 3, "trailer position")
        velocity = _finite_vector(velocity_enu_m_s, 3, "trailer velocity")
        yaw = wrap_angle(yaw_enu_rad)
        yaw_rate = float(yaw_rate_rad_s)
        acceleration = float(acceleration_m_s2)
        if (
            not math.isfinite(sample_time)
            or sample_time <= 0.0
            or not math.isfinite(yaw_rate)
            or not math.isfinite(acceleration)
        ):
            raise ValueError("trailer observation timestamp/motion is invalid")
        acceleration = float(
            np.clip(
                acceleration,
                -self.parameters.maximum_acceleration_m_s2,
                self.parameters.maximum_acceleration_m_s2,
            )
        )
        heading = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=float)
        speed = float(np.dot(velocity[:2], heading))
        steering = self.infer_steering(speed, yaw_rate)
        checked_covariance = validate_covariance(
            covariance, 4, "trailer bicycle state"
        )
        return TrailerBicycleState(
            sample_time,
            position,
            yaw,
            speed,
            acceleration,
            steering,
            checked_covariance,
        )

    def _derivative(
        self,
        state_vector: np.ndarray,
        acceleration_m_s2: float,
        steering_rad: float,
    ) -> np.ndarray:
        yaw = float(state_vector[2])
        speed = float(state_vector[3])
        yaw_rate = (
            speed
            / self.parameters.wheelbase_m
            * math.tan(steering_rad)
        )
        return np.asarray(
            (
                speed * math.cos(yaw),
                speed * math.sin(yaw),
                yaw_rate,
                acceleration_m_s2,
            ),
            dtype=float,
        )

    def _rk4_step(
        self,
        state_vector: np.ndarray,
        dt_s: float,
        acceleration_m_s2: float,
        steering_rad: float,
    ) -> np.ndarray:
        k1 = self._derivative(
            state_vector, acceleration_m_s2, steering_rad
        )
        k2 = self._derivative(
            state_vector + 0.5 * dt_s * k1,
            acceleration_m_s2,
            steering_rad,
        )
        k3 = self._derivative(
            state_vector + 0.5 * dt_s * k2,
            acceleration_m_s2,
            steering_rad,
        )
        k4 = self._derivative(
            state_vector + dt_s * k3,
            acceleration_m_s2,
            steering_rad,
        )
        result = state_vector + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        result[2] = wrap_angle(result[2])
        return result

    def _transition_jacobian(
        self,
        state_vector: np.ndarray,
        dt_s: float,
        acceleration_m_s2: float,
        steering_rad: float,
    ) -> np.ndarray:
        jacobian = np.zeros((4, 4), dtype=float)
        perturbations = (1.0e-5, 1.0e-5, 1.0e-6, 1.0e-5)
        for column, epsilon in enumerate(perturbations):
            plus = state_vector.copy()
            minus = state_vector.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            output_plus = self._rk4_step(
                plus, dt_s, acceleration_m_s2, steering_rad
            )
            output_minus = self._rk4_step(
                minus, dt_s, acceleration_m_s2, steering_rad
            )
            difference = output_plus - output_minus
            difference[2] = wrap_angle(output_plus[2] - output_minus[2])
            jacobian[:, column] = difference / (2.0 * epsilon)
        return jacobian

    def _process_covariance(
        self, state_vector: np.ndarray, dt_s: float
    ) -> np.ndarray:
        yaw = float(state_vector[2])
        acceleration_gain = np.asarray(
            (
                0.5 * dt_s * dt_s * math.cos(yaw),
                0.5 * dt_s * dt_s * math.sin(yaw),
                0.0,
                dt_s,
            ),
            dtype=float,
        )
        yaw_acceleration_gain = np.asarray(
            (0.0, 0.0, 0.5 * dt_s * dt_s, 0.0), dtype=float
        )
        process = (
            self.parameters.acceleration_process_variance
            * np.outer(acceleration_gain, acceleration_gain)
            + self.parameters.yaw_acceleration_process_variance
            * np.outer(yaw_acceleration_gain, yaw_acceleration_gain)
        )
        process += np.diag(
            (
                self.parameters.position_variance_growth_per_s * dt_s,
                self.parameters.position_variance_growth_per_s * dt_s,
                self.parameters.yaw_variance_growth_per_s * dt_s,
                self.parameters.speed_variance_growth_per_s * dt_s,
            )
        )
        return process

    def _derived_motion(
        self, state: TrailerBicycleState
    ) -> tuple[np.ndarray, np.ndarray, float]:
        yaw = state.yaw_enu_rad
        speed = state.speed_m_s
        yaw_rate = (
            speed
            / self.parameters.wheelbase_m
            * math.tan(state.steering_rad)
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        velocity = np.asarray((speed * cosine, speed * sine, 0.0), dtype=float)
        acceleration = np.asarray(
            (
                state.acceleration_m_s2 * cosine - speed * sine * yaw_rate,
                state.acceleration_m_s2 * sine + speed * cosine * yaw_rate,
                0.0,
            ),
            dtype=float,
        )
        return velocity, acceleration, yaw_rate

    def _propagate_vector_and_covariance(
        self,
        state_vector: np.ndarray,
        covariance: np.ndarray,
        duration_s: float,
        acceleration_m_s2: float,
        steering_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate one state interval using bounded RK4 substeps."""
        vector = np.asarray(state_vector, dtype=float).copy()
        propagated_covariance = np.asarray(covariance, dtype=float).copy()
        remaining = float(duration_s)
        while remaining > _EPS:
            step = min(
                remaining, self.parameters.maximum_integration_step_s
            )
            transition = self._transition_jacobian(
                vector, step, acceleration_m_s2, steering_rad
            )
            process = self._process_covariance(vector, step)
            vector = self._rk4_step(
                vector, step, acceleration_m_s2, steering_rad
            )
            propagated_covariance = (
                transition
                @ propagated_covariance
                @ transition.T
                + process
            )
            propagated_covariance = 0.5 * (
                propagated_covariance + propagated_covariance.T
            )
            eigenvalues, eigenvectors = np.linalg.eigh(
                propagated_covariance
            )
            propagated_covariance = (
                eigenvectors
                @ np.diag(np.maximum(eigenvalues, 0.0))
                @ eigenvectors.T
            )
            propagated_covariance = 0.5 * (
                propagated_covariance + propagated_covariance.T
            )
            remaining -= step
        return vector, propagated_covariance

    def _propagate_vector(
        self,
        state_vector: np.ndarray,
        duration_s: float,
        acceleration_m_s2: float,
        steering_rad: float,
    ) -> np.ndarray:
        """Propagate only kinematics when covariance is not requested."""
        vector = np.asarray(state_vector, dtype=float).copy()
        remaining = float(duration_s)
        while remaining > _EPS:
            step = min(
                remaining, self.parameters.maximum_integration_step_s
            )
            vector = self._rk4_step(
                vector, step, acceleration_m_s2, steering_rad
            )
            remaining -= step
        return vector

    def predict(
        self, state: TrailerBicycleState, solve_time_s: float
    ) -> TrailerPrediction:
        """Predict a state to solve time, rejecting future or stale input."""
        try:
            solve_time = float(solve_time_s)
        except (TypeError, ValueError):
            return TrailerPrediction(
                False, "invalid_prediction_time", None, None, None, None, None
            )
        if not math.isfinite(solve_time) or solve_time <= 0.0:
            return TrailerPrediction(
                False, "invalid_prediction_time", None, None, None, None, None
            )
        try:
            age = solve_time - float(state.sample_time_s)
        except (AttributeError, TypeError, ValueError):
            return TrailerPrediction(
                False, "invalid_state_time", None, None, None, None, None
            )
        if not math.isfinite(age):
            return TrailerPrediction(
                False, "invalid_state_time", None, None, None, None, None
            )
        if age < -self.parameters.future_tolerance_s:
            return TrailerPrediction(
                False, "measurement_from_future", age, None, None, None, None
            )
        if age > self.parameters.maximum_measurement_age_s:
            return TrailerPrediction(
                False, "stale_measurement", age, None, None, None, None
            )

        try:
            position = _finite_vector(
                state.position_enu_m, 3, "trailer position"
            )
            covariance = validate_covariance(
                state.covariance, 4, "trailer bicycle state"
            )
            yaw = wrap_angle(state.yaw_enu_rad)
            speed = float(state.speed_m_s)
            acceleration = float(state.acceleration_m_s2)
            steering = float(state.steering_rad)
            if not all(
                math.isfinite(value)
                for value in (speed, acceleration, steering)
            ):
                raise ValueError("trailer motion must be finite")
        except (TypeError, ValueError):
            return TrailerPrediction(
                False, "invalid_state", age, None, None, None, None
            )

        effective_age = max(age, 0.0)
        acceleration = float(
            np.clip(
                acceleration,
                -self.parameters.maximum_acceleration_m_s2,
                self.parameters.maximum_acceleration_m_s2,
            )
        )
        steering = float(
            np.clip(
                steering,
                -self.parameters.maximum_steering_rad,
                self.parameters.maximum_steering_rad,
            )
        )
        vector = np.asarray(
            (position[0], position[1], yaw, speed), dtype=float
        )
        vector, covariance = self._propagate_vector_and_covariance(
            vector,
            covariance,
            effective_age,
            acceleration,
            steering,
        )

        predicted_state = TrailerBicycleState(
            solve_time,
            np.asarray((vector[0], vector[1], position[2]), dtype=float),
            wrap_angle(vector[2]),
            float(vector[3]),
            acceleration,
            steering,
            covariance,
        )
        velocity, acceleration_enu, yaw_rate = self._derived_motion(
            predicted_state
        )
        reason = "current" if effective_age <= _EPS else "predicted"
        return TrailerPrediction(
            True,
            reason,
            effective_age,
            _copy_trailer_state(predicted_state),
            velocity,
            acceleration_enu,
            yaw_rate,
        )

    def predict_horizon(
        self,
        state: TrailerBicycleState,
        start_time_s: float,
        dt_s: float,
        horizon_steps: int,
        *,
        propagate_covariance: bool = True,
    ) -> TrailerPredictionHorizon:
        """Return an RK4 pad trajectory for an existing MPC discretization.

        The observation is first brought to ``start_time_s`` through
        :meth:`predict`, retaining its stale/future fail-closed checks.  The
        accepted current state is then propagated over the future horizon;
        the measurement-age limit is deliberately not re-applied to those
        future samples.  A consumer which needs only MPC ``p/v/a`` can disable
        covariance propagation, avoiding numerical Jacobians while retaining
        exactly the same RK4 kinematics.
        """
        try:
            start_time = float(start_time_s)
            step_size = float(dt_s)
            step_count = int(horizon_steps)
        except (TypeError, ValueError, OverflowError):
            return TrailerPredictionHorizon(
                False,
                "invalid_horizon",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if (
            not math.isfinite(start_time)
            or start_time <= 0.0
            or not math.isfinite(step_size)
            or step_size <= 0.0
            or step_count < 1
            or isinstance(horizon_steps, bool)
        ):
            return TrailerPredictionHorizon(
                False,
                "invalid_horizon",
                start_time if math.isfinite(start_time) else None,
                step_size if math.isfinite(step_size) else None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        # Reject non-integral values such as 2.5 instead of silently
        # shortening the solver horizon through ``int`` conversion.
        try:
            if float(horizon_steps) != float(step_count):
                raise ValueError
        except (TypeError, ValueError):
            return TrailerPredictionHorizon(
                False,
                "invalid_horizon",
                start_time,
                step_size,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        current = self.predict(state, start_time)
        if not current.valid or current.state is None:
            return TrailerPredictionHorizon(
                False,
                current.reason,
                start_time,
                step_size,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        current_state = current.state
        acceleration = float(current_state.acceleration_m_s2)
        steering = float(current_state.steering_rad)
        altitude = float(current_state.position_enu_m[2])
        vector = np.asarray(
            (
                current_state.position_enu_m[0],
                current_state.position_enu_m[1],
                current_state.yaw_enu_rad,
                current_state.speed_m_s,
            ),
            dtype=float,
        )
        covariance = current_state.covariance.copy()
        positions = np.empty((step_count, 3), dtype=float)
        velocities = np.empty((step_count, 3), dtype=float)
        accelerations = np.empty((step_count, 3), dtype=float)
        yaws = np.empty(step_count, dtype=float)
        yaw_rates = np.empty(step_count, dtype=float)
        covariances = (
            np.empty((step_count, 4, 4), dtype=float)
            if bool(propagate_covariance)
            else None
        )

        for index in range(step_count):
            if covariances is not None:
                vector, covariance = (
                    self._propagate_vector_and_covariance(
                        vector,
                        covariance,
                        step_size,
                        acceleration,
                        steering,
                    )
                )
            else:
                vector = self._propagate_vector(
                    vector,
                    step_size,
                    acceleration,
                    steering,
                )
            sample_state = TrailerBicycleState(
                start_time + (index + 1) * step_size,
                np.asarray((vector[0], vector[1], altitude), dtype=float),
                wrap_angle(vector[2]),
                float(vector[3]),
                acceleration,
                steering,
                covariance,
            )
            velocity, acceleration_enu, yaw_rate = self._derived_motion(
                sample_state
            )
            positions[index] = sample_state.position_enu_m
            velocities[index] = velocity
            accelerations[index] = acceleration_enu
            yaws[index] = sample_state.yaw_enu_rad
            yaw_rates[index] = yaw_rate
            if covariances is not None:
                covariances[index] = covariance

        return TrailerPredictionHorizon(
            True,
            "predicted_horizon",
            start_time,
            step_size,
            positions,
            velocities,
            accelerations,
            yaws,
            yaw_rates,
            covariances,
        )
