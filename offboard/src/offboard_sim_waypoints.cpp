/**
 * @brief Hybrid Offboard Control
 * - Arming/Takeoff Logic: Based on "OffboardSimWaypoints" (Command ACK based)
 * - Mission Logic: Based on "Smooth Interpolation" (No overshoot)
 */

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
#include <array>
#include <vector>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class OffboardHybridMission : public rclcpp::Node {
public:
	OffboardHybridMission() : Node("offboard_hybrid_mission") {
		auto qos_profile = rclcpp::SensorDataQoS();

		offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
		trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
		vehicle_command_client_ = this->create_client<px4_msgs::srv::VehicleCommand>("/fmu/vehicle_command");

		odometry_subscription_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
			"/fmu/out/vehicle_odometry",
			qos_profile,
			[this](const px4_msgs::msg::VehicleOdometry &msg) {
				if (std::isfinite(msg.position[0])) {
					current_position_ = msg.position;
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
			RCLCPP_INFO(this->get_logger(), "Waiting for vehicle_command service...");
		}

		timer_ = this->create_wall_timer(50ms, [this]() {
			if (phase_ != Phase::landing && phase_ != Phase::done) {
				publish_offboard_control_mode();
				publish_trajectory_setpoint();
			}
			advance_state_machine();
			log_state();
		});
	}

private:
	enum class Phase {
		init,
		offboard_requested,
		arm_requested,
		takeoff,
		switch_to_mission,
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
	bool position_valid_ = false;
	bool landed_ = false;
	std::array<float, 3> current_position_{0.0f, 0.0f, 0.0f};
	std::array<float, 3> setpoint_position_{0.0f, 0.0f, 0.0f};
	std::vector<std::array<float, 3>> waypoints_{
		{0.0f, 0.0f, -15.0f},
		{50.0f, 0.0f, -15.0f},
		{50.0f, 50.0f, -15.0f},
		{0.0f, 50.0f, -15.0f},
		{0.0f, 0.0f, -15.0f}
	};
	size_t wp_index_ = 0;

	const float takeoff_alt_ = 15.0f;
	const float flight_speed_ = 3.0f;
	const float dt_ = 0.05f;
	uint64_t log_counter_ = 0;

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
		if (phase_ < Phase::switch_to_mission && position_valid_) {
			setpoint_position_ = current_position_;
			if (setpoint_position_[2] > -0.5f) {
				setpoint_position_[2] = -1.0f;
			}
		}

		msg.position = {setpoint_position_[0], setpoint_position_[1], setpoint_position_[2]};
		msg.yaw = -3.14f;
		msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
		trajectory_setpoint_publisher_->publish(msg);
	}

	void request_vehicle_command(uint16_t command, float param1 = 0.0f, float param2 = 0.0f, float param7 = 0.0f) {
		auto request = std::make_shared<px4_msgs::srv::VehicleCommand::Request>();
		VehicleCommand msg{};
		msg.param1 = param1;
		msg.param2 = param2;
		msg.param7 = param7;
		msg.command = command;
		msg.target_system = 1;
		msg.target_component = 1;
		msg.source_system = 1;
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
				}
				command_in_flight_ = false;
			});
	}

	bool command_accepted() {
		if (!command_result_ready_) {
			return false;
		}
		command_result_ready_ = false;
		return command_accepted_;
	}

	void log_state() {
		if (++log_counter_ >= 20) {
			RCLCPP_INFO(this->get_logger(),
				"Phase: %d | WP: %zu | Curr: [%.1f, %.1f, %.1f]",
				static_cast<int>(phase_), wp_index_,
				current_position_[0], current_position_[1], current_position_[2]);
			log_counter_ = 0;
		}
	}

	void advance_state_machine() {
		if (!position_valid_) {
			return;
		}

		switch (phase_) {
		case Phase::init:
			if (warmup_ticks_ < 10) {
				warmup_ticks_++;
				break;
			}
			if (!command_in_flight_) {
				RCLCPP_INFO(this->get_logger(), "Requesting Offboard Mode...");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f, 0.0f);
				phase_ = Phase::offboard_requested;
			}
			break;

		case Phase::offboard_requested:
			if (command_accepted() && !command_in_flight_) {
				RCLCPP_INFO(this->get_logger(), "Offboard Accepted. Requesting ARM...");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f, 0.0f, 0.0f);
				phase_ = Phase::arm_requested;
			}
			break;

		case Phase::arm_requested:
			if (command_accepted()) {
				RCLCPP_INFO(this->get_logger(), "Arming Accepted. Requesting Takeoff...");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_TAKEOFF, NAN, NAN, takeoff_alt_);
				phase_ = Phase::takeoff;
			}
			break;

		case Phase::takeoff:
			if (current_position_[2] <= -(takeoff_alt_ * 0.9f)) {
				RCLCPP_INFO(this->get_logger(), "Takeoff Complete. Switching to Mission.");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f, 0.0f);
				phase_ = Phase::switch_to_mission;
			}
			break;

		case Phase::switch_to_mission:
			if (command_accepted() || !command_in_flight_) {
				setpoint_position_ = current_position_;
				phase_ = Phase::mission;
			}
			break;

		case Phase::mission: {
			const auto &target = waypoints_[wp_index_];
			float dx = target[0] - setpoint_position_[0];
			float dy = target[1] - setpoint_position_[1];
			float dz = target[2] - setpoint_position_[2];
			float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

			if (dist > 0.2f) {
				float step = flight_speed_ * dt_;
				if (step > dist) {
					step = dist;
				}
				float scale = step / dist;
				setpoint_position_[0] += dx * scale;
				setpoint_position_[1] += dy * scale;
				setpoint_position_[2] += dz * scale;
			} else {
				if (wp_index_ + 1 < waypoints_.size()) {
					wp_index_++;
					RCLCPP_INFO(this->get_logger(), "Reached WP %zu. Next...", wp_index_);
				} else if (!command_in_flight_) {
					RCLCPP_INFO(this->get_logger(), "Mission Done. Landing.");
					request_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND, 0.0f, 0.0f, 0.0f);
					phase_ = Phase::landing;
				}
			}
			break;
		}

		case Phase::landing:
			if (landed_ && !command_in_flight_) {
				RCLCPP_INFO(this->get_logger(), "Landed. Disarming.");
				request_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f, 0.0f, 0.0f);
				phase_ = Phase::done;
			}
			break;

		case Phase::done:
			rclcpp::shutdown();
			break;
		}
	}
};

int main(int argc, char *argv[]) {
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<OffboardHybridMission>());
	rclcpp::shutdown();
	return 0;
}
