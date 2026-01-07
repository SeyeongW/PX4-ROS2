#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/srv/vehicle_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <chrono>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class OffboardControl : public rclcpp::Node {
public:
	OffboardControl() : Node("offboard_control") {
		offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
		trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
		vehicle_command_client_ = this->create_client<px4_msgs::srv::VehicleCommand>("/fmu/vehicle_command");
		land_detected_subscription_ = this->create_subscription<px4_msgs::msg::VehicleLandDetected>(
			"/fmu/out/vehicle_land_detected",
			10,
			[this](const px4_msgs::msg::VehicleLandDetected &msg) {
				landed_ = msg.landed;
			});

		while (!vehicle_command_client_->wait_for_service(1s)) {
			if (!rclcpp::ok()) {
				RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for vehicle_command service.");
				return;
			}
			RCLCPP_INFO(this->get_logger(), "vehicle_command service not available, waiting...");
		}

		timer_ = this->create_wall_timer(100ms, [this]() {
			publish_offboard_control_mode();
			publish_trajectory_setpoint();
			advance_state_machine();
		});
	}

private:
	enum class Phase {
		init,
		offboard_requested,
		arm_requested,
		takeoff,
		hover,
		landing,
		disarm_requested,
		done
	};

	rclcpp::TimerBase::SharedPtr timer_;
	rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
	rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
	rclcpp::Client<px4_msgs::srv::VehicleCommand>::SharedPtr vehicle_command_client_;
	rclcpp::Subscription<px4_msgs::msg::VehicleLandDetected>::SharedPtr land_detected_subscription_;
	Phase phase_ = Phase::init;
	uint64_t ticks_in_phase_ = 0;
	bool landed_ = false;

	static constexpr uint64_t kOffboardWarmupTicks = 10;
	static constexpr uint64_t kArmDelayTicks = 10;
	static constexpr uint64_t kTakeoffTicks = 50;
	static constexpr uint64_t kHoverTicks = 50;  // 5s at 100ms tick

	void publish_offboard_control_mode() {
		OffboardControlMode msg{};
		msg.position = true;
		msg.velocity = false;
		msg.acceleration = false;
		msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
		offboard_control_mode_publisher_->publish(msg);
	}

	void publish_trajectory_setpoint() {
		TrajectorySetpoint msg{};
		msg.position = {0.0f, 0.0f, -get_target_altitude_m()};
		msg.yaw = -3.14f;
		msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
		trajectory_setpoint_publisher_->publish(msg);
	}

	float get_target_altitude_m() const {
		switch (phase_) {
		case Phase::landing:
		case Phase::disarm_requested:
		case Phase::done:
			return 0.2f;
		default:
			return 10.0f;
		}
	}

	void request_vehicle_command(uint16_t command, float param1 = 0.0f, float param2 = 0.0f) {
		auto request = std::make_shared<px4_msgs::srv::VehicleCommand::Request>();
		VehicleCommand msg{};
		msg.param1 = param1;
		msg.param2 = param2;
		msg.command = command;
		msg.target_system = 1;
		msg.target_component = 1;
		msg.source_system = 255;
		msg.source_component = 1;
		msg.from_external = true;
		msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
		request->request = msg;

		vehicle_command_client_->async_send_request(
			request,
			[this](rclcpp::Client<px4_msgs::srv::VehicleCommand>::SharedFuture future) {
				if (future.wait_for(1s) == std::future_status::ready) {
					auto reply = future.get()->reply;
					if (reply.result != reply.VEHICLE_CMD_RESULT_ACCEPTED) {
						RCLCPP_WARN(this->get_logger(), "Vehicle command rejected: %u", reply.result);
					}
				} else {
					RCLCPP_WARN(this->get_logger(), "Vehicle command service timeout");
				}
			});
	}

	void advance_state_machine() {
		switch (phase_) {
		case Phase::init:
			if (++ticks_in_phase_ >= kOffboardWarmupTicks) {
				RCLCPP_INFO(this->get_logger(), "Switching to offboard mode");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
				phase_ = Phase::offboard_requested;
				ticks_in_phase_ = 0;
			}
			break;
		case Phase::offboard_requested:
			if (++ticks_in_phase_ >= kArmDelayTicks) {
				RCLCPP_INFO(this->get_logger(), "Arming");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
				phase_ = Phase::arm_requested;
				ticks_in_phase_ = 0;
			}
			break;
		case Phase::arm_requested:
			if (++ticks_in_phase_ >= kArmDelayTicks) {
				RCLCPP_INFO(this->get_logger(), "Taking off to 10m");
				phase_ = Phase::takeoff;
				ticks_in_phase_ = 0;
			}
			break;
		case Phase::takeoff:
			if (++ticks_in_phase_ >= kTakeoffTicks) {
				RCLCPP_INFO(this->get_logger(), "Hovering at 10m");
				phase_ = Phase::hover;
				ticks_in_phase_ = 0;
			}
			break;
		case Phase::hover:
			if (++ticks_in_phase_ >= kHoverTicks) {
				RCLCPP_INFO(this->get_logger(), "Landing");
				landed_ = false;
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
				phase_ = Phase::landing;
				ticks_in_phase_ = 0;
			}
			break;
		case Phase::landing:
			if (landed_) {
				RCLCPP_INFO(this->get_logger(), "Disarming after landing");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f);
				phase_ = Phase::disarm_requested;
				ticks_in_phase_ = 0;
			}
			break;
		case Phase::disarm_requested:
			if (++ticks_in_phase_ >= kArmDelayTicks) {
				RCLCPP_INFO(this->get_logger(), "Mission complete");
				phase_ = Phase::done;
				ticks_in_phase_ = 0;
				rclcpp::shutdown();
			}
			break;
		case Phase::done:
		default:
			break;
		}
	}
};

int main(int argc, char *argv[]) {
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<OffboardControl>());
	rclcpp::shutdown();
	return 0;
}
