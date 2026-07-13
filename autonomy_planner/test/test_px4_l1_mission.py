from types import SimpleNamespace

import pytest

from autonomy_planner.px4_l1_mission import (
    HomeGlobal,
    L1ExperimentError,
    MAV_AUTOPILOT_PX4,
    MAV_CMD_MISSION_START,
    MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT,
    MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
    MAV_MISSION_ACCEPTED,
    MAV_TYPE_QUADROTOR,
    MavlinkMissionTransport,
    build_mission_items,
    enu_horizontal_to_global,
    global_to_enu_horizontal,
    parse_ros_graph_exclusivity,
)
from autonomy_planner.px4_l1_plan import CollisionCheckedL1Plan
from autonomy_planner.px4_l1_plan import with_computed_digest
from test_px4_l1_plan import valid_plan_mapping


class Message(SimpleNamespace):
    def __init__(self, message_type, **kwargs):
        super().__init__(**kwargs)
        self._message_type = message_type

    def get_type(self):
        return self._message_type


class FakeMav:
    def __init__(self, connection):
        self.connection = connection
        self.sent_items = []
        self.items = ()
        self.commands = []

    def command_long_send(self, *args):
        self.commands.append(args)
        if args[2] == MAV_CMD_MISSION_START:
            self.connection.messages.append(
                Message("COMMAND_ACK", command=MAV_CMD_MISSION_START, result=0)
            )

    def mission_count_send(self, *args):
        count = int(args[2])
        for seq in range(count):
            self.connection.messages.append(Message("MISSION_REQUEST_INT", seq=seq))
        self.connection.messages.append(Message("MISSION_ACK", type=MAV_MISSION_ACCEPTED))

    def mission_item_int_send(self, *args):
        self.sent_items.append(args)

    def mission_item_send(self, *args):
        raise AssertionError("PX4 INT request should not use float mission items")

    def mission_request_list_send(self, *args):
        self.connection.messages.append(Message("MISSION_COUNT", count=len(self.items)))

    def mission_request_int_send(self, *args):
        seq = int(args[2])
        item = self.items[seq]
        self.connection.messages.append(
            Message(
                "MISSION_ITEM_INT",
                seq=item.seq,
                frame=item.frame,
                command=item.command,
                current=item.current,
                autocontinue=item.autocontinue,
                param2=item.param2,
                x=item.latitude_e7,
                y=item.longitude_e7,
                z=item.relative_altitude_m,
            )
        )


class FakeConnection:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.target_system = 1
        self.target_component = 1
        self.mav = FakeMav(self)

    def recv_match(self, type, blocking, timeout):
        accepted = set(type)
        for index, message in enumerate(self.messages):
            if message.get_type() in accepted:
                return self.messages.pop(index)
        return None


def test_enu_global_roundtrip_is_sub_centimetre_over_city_scale():
    home = HomeGlobal(47.397742, 8.545594, 488.0)
    latitude, longitude = enu_horizontal_to_global(home, 625.0, -510.0)
    east, north = global_to_enu_horizontal(home, latitude, longitude)
    assert east == pytest.approx(625.0, abs=0.02)
    assert north == pytest.approx(-510.0, abs=0.02)


def test_builds_takeoff_then_checked_global_waypoints():
    plan = CollisionCheckedL1Plan.from_mapping(valid_plan_mapping())
    home = HomeGlobal(47.397742, 8.545594, 488.0)
    items = build_mission_items(plan, home, patrol_cycles=1)
    assert len(items) == 5
    assert items[0].command == MAV_CMD_NAV_TAKEOFF
    assert all(item.command == MAV_CMD_NAV_WAYPOINT for item in items[1:])
    assert all(item.frame == MAV_FRAME_GLOBAL_RELATIVE_ALT_INT for item in items)
    assert all(item.relative_altitude_m == 10.0 for item in items)
    assert items[0].latitude_e7 == round(home.latitude_deg * 1e7)
    assert items[0].longitude_e7 == round(home.longitude_deg * 1e7)


def test_mission_recheck_keeps_l1_lookahead_horizontal():
    mapping = valid_plan_mapping()
    mapping["corridor_boxes"][0]["min_enu_m"][2] = 9.9
    mapping["corridor_boxes"][0]["max_enu_m"][2] = 10.1
    plan = CollisionCheckedL1Plan.from_mapping(with_computed_digest(mapping))
    items = build_mission_items(plan, HomeGlobal(47.397742, 8.545594, 488.0))
    assert len(items) == 5


def test_ros_graph_guard_rejects_offboard_publisher_and_mavros():
    parse_ros_graph_exclusivity("Publisher count: 0\n", "/aruco_front\n")
    with pytest.raises(L1ExperimentError, match="1 publisher"):
        parse_ros_graph_exclusivity("Publisher count: 1\n", "/aruco_front\n")
    with pytest.raises(L1ExperimentError, match="conflicting ROS"):
        parse_ros_graph_exclusivity("Publisher count: 0\n", "/mavros\n")


def test_mavlink_upload_readback_and_start_protocol():
    plan = CollisionCheckedL1Plan.from_mapping(valid_plan_mapping())
    items = build_mission_items(plan, HomeGlobal(47.0, 8.0, 500.0))
    connection = FakeConnection()
    connection.mav.items = items
    transport = MavlinkMissionTransport(connection, timeout_s=0.2)
    transport.upload(items)
    assert len(connection.mav.sent_items) == len(items)
    transport.download_and_verify(items)
    transport.start_auto_mission(len(items) - 1)
    assert any(command[2] == MAV_CMD_MISSION_START for command in connection.mav.commands)
    connection.messages.extend(
        (
            Message(
                "HEARTBEAT",
                autopilot=MAV_AUTOPILOT_PX4,
                type=MAV_TYPE_QUADROTOR,
                base_mode=128,
            ),
            Message("MISSION_CURRENT", seq=0),
        )
    )
    mode, sequence = transport.verify_auto_mission_authority(
        mode_decoder=lambda _: "AUTO.MISSION", item_count=len(items)
    )
    assert mode == "AUTO.MISSION"
    assert sequence == 0


def test_preflight_rejects_armed_or_non_px4_vehicle():
    home = Message("HOME_POSITION", latitude=470000000, longitude=80000000, altitude=500000)
    local = Message("LOCAL_POSITION_NED", x=0.0, y=0.0, z=0.0)
    wrong = Message(
        "HEARTBEAT",
        autopilot=MAV_AUTOPILOT_PX4 + 1,
        type=MAV_TYPE_QUADROTOR,
        base_mode=0,
    )
    transport = MavlinkMissionTransport(FakeConnection((wrong, home, local)), timeout_s=0.05)
    with pytest.raises(L1ExperimentError, match="not from a PX4"):
        transport.wait_preflight(mode_decoder=lambda _: "MANUAL")


def test_adapter_has_no_ros_setpoint_publisher_api():
    import autonomy_planner.px4_l1_mission as adapter

    assert not hasattr(adapter, "rclpy")
    assert not hasattr(adapter, "TrajectorySetpoint")
