#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/srv/vehicle_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <chrono>
#include <cmath>
#include <vector>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class OffboardSimWaypoints : public rclcpp::Node {
public:
	OffboardSimWaypoints() : Node("offboard_sim_waypoints") {
		auto qos_profile = rclcpp::SensorDataQoS();

		offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
		trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
		vehicle_command_client_ = this->create_client<px4_msgs::srv::VehicleCommand>("/fmu/vehicle_command");

		odometry_subscription_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
			"/fmu/out/vehicle_odometry",
			qos_profile,
			[this](const px4_msgs::msg::VehicleOdometry &msg) {
				if (std::isfinite(msg.position[0]) && std::isfinite(msg.position[1]) && std::isfinite(msg.position[2])) {
					current_position_[0] = msg.position[0];
					current_position_[1] = msg.position[1];
					current_position_[2] = msg.position[2];
					position_valid_ = true;
				}
			});

		land_detected_subscription_ = this->create_subscription<px4_msgs::msg::VehicleLandDetected>(
			"/fmu/out/vehicle_land_detected",
			qos_profile,
			[this](const px4_msgs::msg::VehicleLandDetected &msg) {
				landed_ = msg.landed || msg.ground_contact;
			});

		while (!vehicle_command_client_->wait_for_service(1s)) {
			if (!rclcpp::ok()) {
				RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for vehicle_command service.");
				return;
			}
			RCLCPP_INFO(this->get_logger(), "vehicle_command service not available, waiting...");
		}

		timer_ = this->create_wall_timer(100ms, [this]() {
			if (phase_ != Phase::landing && phase_ != Phase::done) {
				publish_offboard_control_mode();
				publish_trajectory_setpoint();
			}
			advance_state_machine();
		});
	}

private:
	enum class Phase {
		init,
		offboard_requested,
		arm_requested,
		mission,
		landing,
		done
	};

	rclcpp::TimerBase::SharedPtr timer_;
	rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
	rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
	rclcpp::Client<px4_msgs::srv::VehicleCommand>::SharedPtr vehicle_command_client_;
	rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odometry_subscription_;
	rclcpp::Subscription<px4_msgs::msg::VehicleLandDetected>::SharedPtr land_detected_subscription_;

	Phase phase_ = Phase::init;
	uint64_t warmup_ticks_ = 0;
	bool command_in_flight_ = false;
	bool command_result_ready_ = false;
	bool command_accepted_ = false;
	bool landed_ = false;
	bool position_valid_ = false;
	std::array<float, 3> current_position_{0.0f, 0.0f, 0.0f};
	std::vector<std::array<float, 3>> waypoints_{
		{0.0f, 0.0f, -10.0f},
		{5.0f, 0.0f, -10.0f},
		{5.0f, 5.0f, -10.0f},
		{0.0f, 5.0f, -10.0f},
		{0.0f, 0.0f, -10.0f}
	};
	size_t current_waypoint_index_ = 0;

	static constexpr float kWaypointToleranceM = 0.5f;
	static constexpr uint64_t kOffboardWarmupTicks = 10;

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
		const auto &target = waypoints_[current_waypoint_index_];
		msg.position = {target[0], target[1], target[2]};
		msg.yaw = -3.14f;
		msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
		trajectory_setpoint_publisher_->publish(msg);
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

		command_in_flight_ = true;
		command_result_ready_ = false;
		vehicle_command_client_->async_send_request(
			request,
			[this](rclcpp::Client<px4_msgs::srv::VehicleCommand>::SharedFuture future) {
				if (future.wait_for(1s) == std::future_status::ready) {
					auto reply = future.get()->reply;
					command_accepted_ = (reply.result == reply.VEHICLE_CMD_RESULT_ACCEPTED);
					command_result_ready_ = true;
					command_in_flight_ = false;
				} else {
					command_in_flight_ = false;
				}
			});
	}

	bool command_accepted() {
		if (!command_result_ready_) {
			return false;
		}
		command_result_ready_ = false;
		return command_accepted_;
	}

	bool reached_waypoint() const {
		if (!position_valid_) {
			return false;
		}
		const auto &target = waypoints_[current_waypoint_index_];
		const float dx = current_position_[0] - target[0];
		const float dy = current_position_[1] - target[1];
		const float dz = current_position_[2] - target[2];
		const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
		return dist <= kWaypointToleranceM;
	}

	void advance_state_machine() {
		switch (phase_) {
		case Phase::init:
			if (warmup_ticks_ < kOffboardWarmupTicks) {
				++warmup_ticks_;
				break;
			}
			if (!command_in_flight_) {
				RCLCPP_INFO(this->get_logger(), "Switching to offboard mode");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
				phase_ = Phase::offboard_requested;
			}
			break;
		case Phase::offboard_requested:
			if (command_accepted() && !command_in_flight_) {
				RCLCPP_INFO(this->get_logger(), "Arming");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
				phase_ = Phase::arm_requested;
			}
			break;
		case Phase::arm_requested:
			if (command_accepted()) {
				RCLCPP_INFO(this->get_logger(), "Starting waypoint mission");
				current_waypoint_index_ = 0;
				phase_ = Phase::mission;
			}
			break;
		case Phase::mission:
			if (reached_waypoint()) {
				if (current_waypoint_index_ + 1 < waypoints_.size()) {
					++current_waypoint_index_;
					RCLCPP_INFO(this->get_logger(), "Next waypoint %zu", current_waypoint_index_);
				} else if (!command_in_flight_) {
					RCLCPP_INFO(this->get_logger(), "All waypoints reached, landing");
					request_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
					phase_ = Phase::landing;
				}
			}
			break;
		case Phase::landing:
			if (landed_ && !command_in_flight_) {
				RCLCPP_INFO(this->get_logger(), "Disarming after landing");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f);
				phase_ = Phase::done;
			}
			break;
		case Phase::done:
			rclcpp::shutdown();
			break;
		default:
			break;
		}
	}
};

int main(int argc, char *argv[]) {
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<OffboardSimWaypoints>());
	rclcpp::shutdown();
	return 0;
}
