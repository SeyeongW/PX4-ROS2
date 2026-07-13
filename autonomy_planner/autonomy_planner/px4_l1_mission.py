"""Direct MAVLink adapter for the PX4 multicopter L1-style experiment.

The adapter uploads an already collision-checked A*/SFC route as a PX4
``AUTO.MISSION``.  It deliberately contains no ROS publisher and never emits
PX4 ``TrajectorySetpoint`` messages.  Consequently PX4 Navigator owns the
previous/current/next triplet and ``FlightTaskAuto`` exercises the firmware's
``PositionSmoothing::_getL1Point`` implementation.

This is an experiment adapter, not an alternative local obstacle avoider.
Dynamic-obstacle and trailer-landing tests remain in the B-spline/MPC pipeline.
Only one pipeline may hold guidance authority at a time.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, Iterable, Sequence

import numpy as np

from .px4_l1_plan import (
    CollisionCheckedL1Plan,
    MAX_SITL_MISSION_ITEMS,
    PX4_MC_L1_MIN_LOOKAHEAD_M,
    PlanValidationError,
    triplet_is_turning,
)


# MAVLink common.xml values are repeated here so importing and unit testing the
# safety/conversion helpers does not require pymavlink.  The CLI checks them
# against pymavlink at runtime before it talks to a vehicle.
MAV_AUTOPILOT_PX4 = 12
MAV_TYPE_QUADROTOR = 2
MAV_MODE_FLAG_SAFETY_ARMED = 128
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_MISSION_START = 300
MAV_CMD_GET_HOME_POSITION = 410
MAV_MISSION_TYPE_MISSION = 0
MAV_MISSION_ACCEPTED = 0
MAV_RESULT_ACCEPTED = 0
MAV_RESULT_IN_PROGRESS = 5

UPLOAD_AUTHORITY_TOKEN = "REPLACE_PX4_MISSION"
EXECUTE_AUTHORITY_TOKEN = "AUTO_MISSION_WILL_ARM"
DEFAULT_AUTHORITY_LOCK = "/tmp/px4_ros2_guidance_authority.lock"


class L1ExperimentError(RuntimeError):
    """Raised for transport, authority, or vehicle precondition failures."""


@dataclass(frozen=True)
class HomeGlobal:
    """PX4 home in WGS-84 degrees/metres."""

    latitude_deg: float
    longitude_deg: float
    altitude_msl_m: float

    def validate(self) -> None:
        if not (-90.0 <= self.latitude_deg <= 90.0):
            raise L1ExperimentError("HOME_POSITION latitude is invalid")
        if not (-180.0 <= self.longitude_deg <= 180.0):
            raise L1ExperimentError("HOME_POSITION longitude is invalid")
        if not math.isfinite(self.altitude_msl_m):
            raise L1ExperimentError("HOME_POSITION altitude is invalid")

    @classmethod
    def from_mavlink(cls, message: object) -> "HomeGlobal":
        try:
            home = cls(
                latitude_deg=float(getattr(message, "latitude")) * 1e-7,
                longitude_deg=float(getattr(message, "longitude")) * 1e-7,
                altitude_msl_m=float(getattr(message, "altitude")) * 1e-3,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise L1ExperimentError("malformed HOME_POSITION message") from exc
        home.validate()
        return home


@dataclass(frozen=True)
class MissionItemInt:
    """MAVLink MISSION_ITEM_INT values used by the uploader."""

    seq: int
    frame: int
    command: int
    current: int
    autocontinue: int
    param1: float
    param2: float
    param3: float
    param4: float
    latitude_e7: int
    longitude_e7: int
    relative_altitude_m: float

    def equivalent_to_message(
        self, message: object, *, altitude_tolerance_m: float = 0.02
    ) -> bool:
        """Compare safety-relevant fields of a downloaded mission item."""

        try:
            return (
                int(getattr(message, "seq")) == self.seq
                and int(getattr(message, "frame")) == self.frame
                and int(getattr(message, "command")) == self.command
                and int(getattr(message, "autocontinue")) == self.autocontinue
                and int(getattr(message, "x")) == self.latitude_e7
                and int(getattr(message, "y")) == self.longitude_e7
                and abs(float(getattr(message, "z")) - self.relative_altitude_m)
                <= altitude_tolerance_m
                and abs(float(getattr(message, "param2")) - self.param2) <= 1e-3
            )
        except (AttributeError, TypeError, ValueError):
            return False


@dataclass(frozen=True)
class VehiclePreflight:
    """Vehicle state captured before replacing or starting a mission."""

    target_system: int
    target_component: int
    mode: str
    home: HomeGlobal
    local_ned_m: tuple[float, float, float]


def _geodetic_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    semimajor_m = 6_378_137.0
    eccentricity_sq = 6.69437999014e-3
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_lat = math.sin(latitude)
    cos_lat = math.cos(latitude)
    prime_vertical = semimajor_m / math.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
    return np.array(
        [
            (prime_vertical + altitude_m) * cos_lat * math.cos(longitude),
            (prime_vertical + altitude_m) * cos_lat * math.sin(longitude),
            (prime_vertical * (1.0 - eccentricity_sq) + altitude_m) * sin_lat,
        ],
        dtype=float,
    )


def _ecef_to_geodetic(ecef_m: Sequence[float]) -> tuple[float, float, float]:
    semimajor_m = 6_378_137.0
    eccentricity_sq = 6.69437999014e-3
    x, y, z = (float(v) for v in ecef_m)
    longitude = math.atan2(y, x)
    horizontal = math.hypot(x, y)
    latitude = math.atan2(z, horizontal * (1.0 - eccentricity_sq))
    altitude = 0.0
    for _ in range(12):
        sin_lat = math.sin(latitude)
        prime_vertical = semimajor_m / math.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
        cos_lat = math.cos(latitude)
        if abs(cos_lat) > 1e-10:
            altitude = horizontal / cos_lat - prime_vertical
        else:
            altitude = z / math.copysign(max(abs(sin_lat), 1e-10), sin_lat) - prime_vertical * (
                1.0 - eccentricity_sq
            )
        updated = math.atan2(
            z,
            horizontal * (1.0 - eccentricity_sq * prime_vertical / (prime_vertical + altitude)),
        )
        if abs(updated - latitude) < 1e-13:
            latitude = updated
            break
        latitude = updated
    return math.degrees(latitude), math.degrees(longitude), altitude


def enu_horizontal_to_global(
    home: HomeGlobal,
    east_m: float,
    north_m: float,
) -> tuple[float, float]:
    """Map a local tangent-plane horizontal offset to WGS-84 latitude/longitude.

    PX4 v1.17's mission parser accepts global mission frames but rejects local
    NED mission frames.  The Gazebo route therefore needs this explicit boundary
    conversion.  Mission relative altitude remains the exact Gazebo ENU Z
    difference; it is not taken from the tangent ECEF point's ellipsoid height.
    """

    if not all(math.isfinite(v) for v in (east_m, north_m)):
        raise L1ExperimentError("ENU offset must be finite")
    home.validate()
    latitude = math.radians(home.latitude_deg)
    longitude = math.radians(home.longitude_deg)
    # Columns of this matrix are the E, N, U basis vectors expressed in ECEF.
    enu_to_ecef = np.array(
        [
            [
                -math.sin(longitude),
                -math.sin(latitude) * math.cos(longitude),
                math.cos(latitude) * math.cos(longitude),
            ],
            [
                math.cos(longitude),
                -math.sin(latitude) * math.sin(longitude),
                math.cos(latitude) * math.sin(longitude),
            ],
            [0.0, math.cos(latitude), math.sin(latitude)],
        ],
        dtype=float,
    )
    target_ecef = _geodetic_to_ecef(
        home.latitude_deg, home.longitude_deg, home.altitude_msl_m
    ) + enu_to_ecef @ np.array([east_m, north_m, 0.0], dtype=float)
    target_latitude, target_longitude, _ = _ecef_to_geodetic(target_ecef)
    return target_latitude, target_longitude


def global_to_enu_horizontal(
    home: HomeGlobal,
    latitude_deg: float,
    longitude_deg: float,
) -> tuple[float, float]:
    """Inverse diagnostic conversion used for upload verification and tests."""

    home.validate()
    latitude = math.radians(home.latitude_deg)
    longitude = math.radians(home.longitude_deg)
    delta = _geodetic_to_ecef(
        latitude_deg, longitude_deg, home.altitude_msl_m
    ) - _geodetic_to_ecef(home.latitude_deg, home.longitude_deg, home.altitude_msl_m)
    ecef_to_enu = np.array(
        [
            [-math.sin(longitude), math.cos(longitude), 0.0],
            [
                -math.sin(latitude) * math.cos(longitude),
                -math.sin(latitude) * math.sin(longitude),
                math.cos(latitude),
            ],
            [
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            ],
        ],
        dtype=float,
    )
    enu = ecef_to_enu @ delta
    return float(enu[0]), float(enu[1])


def _assert_route_corridor_certified(
    plan: CollisionCheckedL1Plan,
    route: Sequence[Sequence[float]],
    envelope_m: float,
) -> None:
    """Recheck all patrol turns, including reversals not present in the base plan."""

    for center in range(1, len(route) - 1):
        inset = (
            envelope_m
            if triplet_is_turning(route[center - 1], route[center], route[center + 1])
            else 0.0
        )
        if not any(
            all(
                box.contains_with_horizontal_inset(route[index], inset)
                for index in (center - 1, center, center + 1)
            )
            for box in plan.corridors
        ):
            raise L1ExperimentError(
                f"expanded patrol triplet {center - 1}/{center}/{center + 1} lacks a "
                f"single convex SFC box with {inset:g} m inset"
            )


def build_mission_items(
    plan: CollisionCheckedL1Plan,
    home: HomeGlobal,
    *,
    patrol_cycles: int = 1,
    acceptance_radius_m: float = 2.0,
    minimum_takeoff_altitude_m: float = 5.0,
) -> tuple[MissionItemInt, ...]:
    """Convert a checked Gazebo-world ENU patrol into global-relative mission items."""

    home.validate()
    acceptance = float(acceptance_radius_m)
    if not math.isfinite(acceptance) or acceptance <= 0.0:
        raise L1ExperimentError("acceptance_radius_m must be positive")
    minimum_takeoff = float(minimum_takeoff_altitude_m)
    if not math.isfinite(minimum_takeoff) or minimum_takeoff <= 0.0:
        raise L1ExperimentError("minimum_takeoff_altitude_m must be positive")
    route = plan.patrol_waypoints(patrol_cycles)
    envelope = max(PX4_MC_L1_MIN_LOOKAHEAD_M, acceptance)
    _assert_route_corridor_certified(plan, route, envelope)
    if len(route) > MAX_SITL_MISSION_ITEMS:
        raise L1ExperimentError("route exceeds the PX4 SITL mission item limit")

    anchor = np.asarray(plan.world_home_enu_m, dtype=float)
    first_delta = np.asarray(route[0], dtype=float) - anchor
    if float(np.linalg.norm(first_delta[:2])) > 2.0:
        raise L1ExperimentError(
            "first route point is more than 2 m from world_home_enu_m; "
            "refusing an unchecked takeoff transit"
        )
    if first_delta[2] < minimum_takeoff:
        raise L1ExperimentError(
            f"first route altitude is {first_delta[2]:.2f} m; "
            f"at least {minimum_takeoff:.2f} m is required"
        )

    items: list[MissionItemInt] = []
    for seq, point in enumerate(route):
        offset = np.asarray(point, dtype=float) - anchor
        relative_altitude = float(offset[2])
        if relative_altitude < minimum_takeoff:
            raise L1ExperimentError(
                f"waypoint {seq} altitude {relative_altitude:.2f} m "
                "is below experiment minimum"
            )
        latitude, longitude = enu_horizontal_to_global(home, float(offset[0]), float(offset[1]))
        items.append(
            MissionItemInt(
                seq=seq,
                frame=MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                command=MAV_CMD_NAV_TAKEOFF if seq == 0 else MAV_CMD_NAV_WAYPOINT,
                current=1 if seq == 0 else 0,
                autocontinue=1,
                param1=0.0,
                param2=acceptance,
                param3=0.0,
                param4=float("nan"),
                latitude_e7=int(round(latitude * 1e7)),
                longitude_e7=int(round(longitude * 1e7)),
                relative_altitude_m=relative_altitude,
            )
        )
    return tuple(items)


def parse_ros_graph_exclusivity(topic_info: str, node_list: str) -> None:
    """Fail if another controller can write trajectory setpoints or MAVROS is active."""

    match = re.search(r"Publisher count:\s*(\d+)", topic_info)
    if match is None:
        raise L1ExperimentError(
            "could not prove trajectory-setpoint publisher count from ROS graph"
        )
    publisher_count = int(match.group(1))
    if publisher_count != 0:
        raise L1ExperimentError(
            f"trajectory-setpoint topic has {publisher_count} publisher(s); "
            "stop the B-spline/offboard controller"
        )
    forbidden_fragments = ("mavros", "city_autonomy", "mission_node")
    conflicting = [
        line.strip()
        for line in node_list.splitlines()
        if any(fragment in line.lower() for fragment in forbidden_fragments)
    ]
    if conflicting:
        raise L1ExperimentError(
            "conflicting ROS control nodes are active: " + ", ".join(conflicting)
        )


def assert_ros_guidance_exclusive(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Query the live ROS graph and fail closed if it cannot prove exclusivity."""

    environment = os.environ.copy()
    topic = runner(
        ["ros2", "topic", "info", "/fmu/in/trajectory_setpoint", "--verbose"],
        capture_output=True,
        text=True,
        timeout=5.0,
        env=environment,
        check=False,
    )
    nodes = runner(
        ["ros2", "node", "list"],
        capture_output=True,
        text=True,
        timeout=5.0,
        env=environment,
        check=False,
    )
    if topic.returncode != 0 or nodes.returncode != 0:
        raise L1ExperimentError(
            "ROS graph query failed; refusing AUTO.MISSION because offboard "
            "exclusivity is unproven"
        )
    parse_ros_graph_exclusivity(topic.stdout, nodes.stdout)


class GuidanceAuthorityLease(AbstractContextManager["GuidanceAuthorityLease"]):
    """Process lock plus live ROS graph guard for mutually-exclusive guidance."""

    def __init__(
        self, path: str = DEFAULT_AUTHORITY_LOCK, *, check_ros_graph: bool = True
    ) -> None:
        self.path = Path(path)
        self.check_ros_graph = check_ros_graph
        self._file: object | None = None

    def __enter__(self) -> "GuidanceAuthorityLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.close()
            self._file = None
            raise L1ExperimentError("another guidance experiment owns the authority lock") from exc
        self._file.seek(0)
        self._file.truncate(0)
        self._file.write(f"pid={os.getpid()} mode=PX4_AUTO_MISSION_L1_STYLE\n")
        self._file.flush()
        if self.check_ros_graph:
            try:
                assert_ros_guidance_exclusive()
            except Exception:
                self.__exit__(*sys.exc_info())
                raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None
        return None


def _message_type(message: object) -> str:
    method = getattr(message, "get_type", None)
    if callable(method):
        return str(method())
    return str(getattr(message, "_type", type(message).__name__))


def _receive_until(
    connection: object,
    message_types: Iterable[str],
    *,
    timeout_s: float,
    predicate: Callable[[object], bool] | None = None,
) -> object:
    accepted = set(message_types)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        message = connection.recv_match(
            type=list(accepted), blocking=True, timeout=min(0.5, remaining)
        )
        if message is not None and _message_type(message) in accepted:
            if predicate is None or predicate(message):
                return message
    raise L1ExperimentError(f"timed out waiting for {sorted(accepted)}")


def _send_optional_mission_type(method: Callable[..., object], args: Sequence[object]) -> None:
    try:
        method(*args, MAV_MISSION_TYPE_MISSION)
    except TypeError:
        method(*args)


class MavlinkMissionTransport:
    """Minimal, testable MAVLink mission protocol implementation."""

    def __init__(self, connection: object, *, timeout_s: float = 12.0) -> None:
        self.connection = connection
        self.timeout_s = float(timeout_s)

    @property
    def target_system(self) -> int:
        return int(getattr(self.connection, "target_system", 0))

    @property
    def target_component(self) -> int:
        return int(getattr(self.connection, "target_component", 0))

    def wait_preflight(
        self,
        *,
        mode_decoder: Callable[[object], str],
        max_local_origin_error_m: float = 2.0,
    ) -> VehiclePreflight:
        heartbeat = _receive_until(self.connection, ("HEARTBEAT",), timeout_s=self.timeout_s)
        if int(getattr(heartbeat, "autopilot", -1)) != MAV_AUTOPILOT_PX4:
            raise L1ExperimentError("heartbeat is not from a PX4 autopilot")
        if int(getattr(heartbeat, "type", -1)) != MAV_TYPE_QUADROTOR:
            raise L1ExperimentError("this experiment is restricted to the x500 quadrotor")
        if int(getattr(heartbeat, "base_mode", 0)) & MAV_MODE_FLAG_SAFETY_ARMED:
            raise L1ExperimentError("vehicle must be disarmed before mission replacement")
        mode = str(mode_decoder(heartbeat)).upper()
        if "OFFBOARD" in mode:
            raise L1ExperimentError("PX4 is in OFFBOARD; stop the B-spline controller first")
        if self.target_system <= 0 or self.target_component <= 0:
            raise L1ExperimentError(
                "MAVLink target system/component was not learned from heartbeat"
            )

        self.connection.mav.command_long_send(
            self.target_system,
            self.target_component,
            MAV_CMD_GET_HOME_POSITION,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        home_message = _receive_until(
            self.connection, ("HOME_POSITION",), timeout_s=self.timeout_s
        )
        home = HomeGlobal.from_mavlink(home_message)
        local = _receive_until(
            self.connection, ("LOCAL_POSITION_NED",), timeout_s=self.timeout_s
        )
        local_ned = tuple(float(getattr(local, field)) for field in ("x", "y", "z"))
        if not all(math.isfinite(value) for value in local_ned):
            raise L1ExperimentError("LOCAL_POSITION_NED is not finite")
        if math.hypot(local_ned[0], local_ned[1]) > max_local_origin_error_m:
            raise L1ExperimentError(
                "vehicle is not at the declared world-home anchor; update world_home_enu_m"
            )
        return VehiclePreflight(
            target_system=self.target_system,
            target_component=self.target_component,
            mode=mode,
            home=home,
            local_ned_m=local_ned,
        )

    def _send_item_int(self, item: MissionItemInt) -> None:
        args = (
            self.target_system,
            self.target_component,
            item.seq,
            item.frame,
            item.command,
            item.current,
            item.autocontinue,
            item.param1,
            item.param2,
            item.param3,
            item.param4,
            item.latitude_e7,
            item.longitude_e7,
            item.relative_altitude_m,
        )
        _send_optional_mission_type(self.connection.mav.mission_item_int_send, args)

    def _send_item_float(self, item: MissionItemInt) -> None:
        args = (
            self.target_system,
            self.target_component,
            item.seq,
            MAV_FRAME_GLOBAL_RELATIVE_ALT,
            item.command,
            item.current,
            item.autocontinue,
            item.param1,
            item.param2,
            item.param3,
            item.param4,
            item.latitude_e7 * 1e-7,
            item.longitude_e7 * 1e-7,
            item.relative_altitude_m,
        )
        _send_optional_mission_type(self.connection.mav.mission_item_send, args)

    def upload(self, items: Sequence[MissionItemInt]) -> None:
        """Atomically replace the active mission and require an accepted ACK."""

        if not items or any(item.seq != index for index, item in enumerate(items)):
            raise L1ExperimentError("mission item sequence must be contiguous from zero")
        _send_optional_mission_type(
            self.connection.mav.mission_count_send,
            (self.target_system, self.target_component, len(items)),
        )
        sent: set[int] = set()
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            message = _receive_until(
                self.connection,
                ("MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"),
                timeout_s=max(0.01, deadline - time.monotonic()),
            )
            kind = _message_type(message)
            if kind == "MISSION_ACK":
                result = int(getattr(message, "type", -1))
                if result != MAV_MISSION_ACCEPTED:
                    raise L1ExperimentError(
                        f"PX4 rejected mission upload with MAV_MISSION {result}"
                    )
                if len(sent) != len(items):
                    raise L1ExperimentError(
                        "PX4 acknowledged mission before requesting every item"
                    )
                return
            seq = int(getattr(message, "seq", -1))
            if not 0 <= seq < len(items):
                raise L1ExperimentError(f"PX4 requested invalid mission sequence {seq}")
            if kind == "MISSION_REQUEST_INT":
                self._send_item_int(items[seq])
            else:
                self._send_item_float(items[seq])
            sent.add(seq)
        raise L1ExperimentError("mission upload did not receive MISSION_ACK")

    def download_and_verify(self, expected: Sequence[MissionItemInt]) -> None:
        """Read the uploaded mission back and compare every route coordinate."""

        _send_optional_mission_type(
            self.connection.mav.mission_request_list_send,
            (self.target_system, self.target_component),
        )
        count_message = _receive_until(
            self.connection, ("MISSION_COUNT",), timeout_s=self.timeout_s
        )
        count = int(getattr(count_message, "count", -1))
        if count != len(expected):
            raise L1ExperimentError(f"mission readback count {count} != expected {len(expected)}")
        for seq, item in enumerate(expected):
            _send_optional_mission_type(
                self.connection.mav.mission_request_int_send,
                (self.target_system, self.target_component, seq),
            )
            message = _receive_until(
                self.connection,
                ("MISSION_ITEM_INT",),
                timeout_s=self.timeout_s,
                predicate=lambda candidate, seq=seq: int(getattr(candidate, "seq", -1)) == seq,
            )
            if not item.equivalent_to_message(message):
                raise L1ExperimentError(f"mission readback mismatch at sequence {seq}")

    def start_auto_mission(self, last_item: int) -> None:
        """Start AUTO.MISSION; PX4 accepts this command by switching mode and arming."""

        self.connection.mav.command_long_send(
            self.target_system,
            self.target_component,
            MAV_CMD_MISSION_START,
            0,
            0.0,
            float(last_item),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        acknowledgement = _receive_until(
            self.connection,
            ("COMMAND_ACK",),
            timeout_s=self.timeout_s,
            predicate=lambda message: (
                int(getattr(message, "command", -1)) == MAV_CMD_MISSION_START
            ),
        )
        result = int(getattr(acknowledgement, "result", -1))
        if result not in (MAV_RESULT_ACCEPTED, MAV_RESULT_IN_PROGRESS):
            raise L1ExperimentError(f"PX4 rejected MISSION_START with MAV_RESULT {result}")

    def verify_auto_mission_authority(
        self,
        *,
        mode_decoder: Callable[[object], str],
        item_count: int,
    ) -> tuple[str, int]:
        """Prove that PX4 is armed in AUTO.MISSION and reports a valid item."""

        heartbeat = _receive_until(
            self.connection, ("HEARTBEAT",), timeout_s=self.timeout_s
        )
        mode = str(mode_decoder(heartbeat)).upper()
        armed = bool(
            int(getattr(heartbeat, "base_mode", 0)) & MAV_MODE_FLAG_SAFETY_ARMED
        )
        if not armed or "AUTO" not in mode or "MISSION" not in mode:
            raise L1ExperimentError(
                f"MISSION_START ACKed but authority proof failed: mode={mode} armed={armed}"
            )
        current = _receive_until(
            self.connection, ("MISSION_CURRENT",), timeout_s=self.timeout_s
        )
        sequence = int(getattr(current, "seq", -1))
        if not 0 <= sequence < item_count:
            raise L1ExperimentError(
                f"PX4 reported invalid active mission sequence {sequence}"
            )
        return mode, sequence

    def wait_until_last_waypoint(self, last_item: int, timeout_s: float) -> None:
        """Keep the authority lease until PX4 reports the final item reached."""

        _receive_until(
            self.connection,
            ("MISSION_ITEM_REACHED",),
            timeout_s=timeout_s,
            predicate=lambda message: int(getattr(message, "seq", -1)) == last_item,
        )


def _assert_pymavlink_constants(mavlink: object) -> None:
    expected = {
        "MAV_AUTOPILOT_PX4": MAV_AUTOPILOT_PX4,
        "MAV_TYPE_QUADROTOR": MAV_TYPE_QUADROTOR,
        "MAV_FRAME_GLOBAL_RELATIVE_ALT_INT": MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        "MAV_CMD_NAV_WAYPOINT": MAV_CMD_NAV_WAYPOINT,
        "MAV_CMD_NAV_TAKEOFF": MAV_CMD_NAV_TAKEOFF,
        "MAV_CMD_MISSION_START": MAV_CMD_MISSION_START,
    }
    for name, value in expected.items():
        if int(getattr(mavlink, name)) != value:
            raise L1ExperimentError(f"pymavlink {name} differs from the audited protocol value")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PX4 v1.17 multicopter L1-style AUTO.MISSION cross-validation adapter"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a signed A*/SFC plan only")
    validate.add_argument("--plan", required=True)

    for name, help_text in (
        ("upload", "upload and verify without arming"),
        ("execute", "upload, verify, arm in AUTO.MISSION, and monitor"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--plan", required=True)
        command.add_argument("--connection", default="udpin:0.0.0.0:14550")
        command.add_argument("--patrol-cycles", type=int, default=1)
        command.add_argument("--acceptance-radius-m", type=float, default=2.0)
        command.add_argument("--expect-map-revision", required=True)
        command.add_argument("--expect-plan-sha256", required=True)
        command.add_argument("--authorize-plan-replacement", required=True)
        command.add_argument("--protocol-timeout-s", type=float, default=12.0)
        if name == "execute":
            command.add_argument("--execute-auto-mission", required=True)
            command.add_argument("--mission-timeout-s", type=float, default=900.0)
            command.add_argument("--authority-lock", default=DEFAULT_AUTHORITY_LOCK)
    return parser


def _assert_operator_contract(args: argparse.Namespace, plan: CollisionCheckedL1Plan) -> None:
    if args.expect_map_revision != plan.map_revision:
        raise L1ExperimentError("--expect-map-revision does not match the signed plan")
    if args.expect_plan_sha256.lower() != plan.plan_sha256:
        raise L1ExperimentError("--expect-plan-sha256 does not match the signed plan")
    if args.authorize_plan_replacement != UPLOAD_AUTHORITY_TOKEN:
        raise L1ExperimentError(
            f"type --authorize-plan-replacement {UPLOAD_AUTHORITY_TOKEN} "
            "to replace the PX4 mission"
        )
    if args.command == "execute" and args.execute_auto_mission != EXECUTE_AUTHORITY_TOKEN:
        raise L1ExperimentError(
            f"type --execute-auto-mission {EXECUTE_AUTHORITY_TOKEN}; "
            "MISSION_START arms the vehicle"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan = CollisionCheckedL1Plan.load(args.plan)
        if args.command == "validate":
            print(
                f"VALID plan={plan.plan_sha256} map={plan.map_revision} "
                f"waypoints={len(plan.waypoints_enu_m)} hard_radius={plan.hard_radius_m:.3f}m"
            )
            return 0
        _assert_operator_contract(args, plan)

        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise L1ExperimentError(
                "pymavlink is required for upload: python3 -m pip install --user pymavlink"
            ) from exc
        _assert_pymavlink_constants(mavutil.mavlink)
        connection = mavutil.mavlink_connection(
            args.connection,
            source_system=245,
            source_component=190,
            autoreconnect=True,
        )
        transport = MavlinkMissionTransport(connection, timeout_s=args.protocol_timeout_s)
        preflight = transport.wait_preflight(mode_decoder=mavutil.mode_string_v10)
        items = build_mission_items(
            plan,
            preflight.home,
            patrol_cycles=args.patrol_cycles,
            acceptance_radius_m=args.acceptance_radius_m,
        )
        if args.command == "upload":
            transport.upload(items)
            transport.download_and_verify(items)
            print(f"UPLOADED_AND_VERIFIED items={len(items)} plan={plan.plan_sha256}")
            return 0

        with GuidanceAuthorityLease(args.authority_lock, check_ros_graph=True):
            # Re-read vehicle state after acquiring authority so no stale mode can
            # slip through between an earlier graph check and MISSION_START.
            preflight = transport.wait_preflight(mode_decoder=mavutil.mode_string_v10)
            items = build_mission_items(
                plan,
                preflight.home,
                patrol_cycles=args.patrol_cycles,
                acceptance_radius_m=args.acceptance_radius_m,
            )
            transport.upload(items)
            transport.download_and_verify(items)
            transport.start_auto_mission(len(items) - 1)
            mode, sequence = transport.verify_auto_mission_authority(
                mode_decoder=mavutil.mode_string_v10,
                item_count=len(items),
            )
            print(
                f"AUTO_MISSION_STARTED items={len(items)} plan={plan.plan_sha256}; "
                f"mode={mode} active_seq={sequence}; PX4 now owns guidance"
            )
            transport.wait_until_last_waypoint(len(items) - 1, args.mission_timeout_s)
            print("AUTO_MISSION_FINAL_WAYPOINT_REACHED")
        return 0
    except (L1ExperimentError, PlanValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
