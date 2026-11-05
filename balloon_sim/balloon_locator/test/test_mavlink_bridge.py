"""Hardware configuration checks without opening UDP or serial connections."""
import pytest
import rclpy

from balloon_locator.mavlink_bridge_node import MavlinkBridgeNode, mavutil


@pytest.mark.parametrize('bad_setting',[
    'cam_lat_deg:=91.0', 'cam_lon_deg:=181.0', 'rate_hz:=0.0',
    'max_age_sec:=0.0', 'target_system:=0',
])
def test_invalid_settings_fail_before_opening_hardware(monkeypatch,bad_setting):
    def forbidden_connection(*args,**kwargs):
        pytest.fail('configuration error reached a hardware connection')
    monkeypatch.setattr(mavutil,'mavlink_connection',forbidden_connection)
    rclpy.init(args=['--ros-args','-p','cam_lat_deg:=37.0','-p','cam_lon_deg:=127.0',
                    '-p','cam_height_above_home_m:=1.5','-p',bad_setting])
    try:
        with pytest.raises(RuntimeError,match='invalid_camera_reference'):
            MavlinkBridgeNode()
    finally:
        rclpy.shutdown()
