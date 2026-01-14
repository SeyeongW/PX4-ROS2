#include <chrono>
#include <cmath>
#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class OffboardRealSimplifiedLog : public rclcpp::Node {
public:
    OffboardRealSimplifiedLog() : Node("offboard_real_log_node") {
        auto qos_profile = rclcpp::SensorDataQoS();

        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        local_pos_subscription_ = this->create_subscription<VehicleLocalPosition>(
            "/fmu/out/vehicle_local_position", qos_profile,
            [this](const VehicleLocalPosition &msg) {
                current_pos_ = {msg.x, msg.y, msg.z};
                current_yaw_meas_ = msg.heading;
                if (!yaw_initialized_) {
                    current_yaw_sp_ = current_yaw_meas_;
                    target_yaw_ = current_yaw_sp_;
                    yaw_initialized_ = true;
                }
            });

        status_subscription_ = this->create_subscription<VehicleStatus>(
            "/fmu/out/vehicle_status", qos_profile,
            [this](const VehicleStatus &msg) { arming_state_ = msg.arming_state; });

        timer_ = this->create_wall_timer(100ms, [this]() {
            if (phase_ == Phase::landing) {
                if (arming_state_ == VehicleStatus::ARMING_STATE_DISARMED) {
                    RCLCPP_INFO(this->get_logger(), ">>> MISSION COMPLETE / DISARMED");
                    rclcpp::shutdown();
                    return;
                }
            }
            if (phase_ != Phase::landing) {
                publish_offboard_control_mode();
                publish_trajectory_setpoint();
            }
            manage_mission_flow();
        });
    }

private:
    enum class Phase { warmup, takeoff, hold, move, landing };

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleLocalPosition>::SharedPtr local_pos_subscription_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_subscription_;

    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;
    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> setpoint_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> target_vel_{0.0f, 0.0f, 0.0f};

    bool yaw_initialized_ = false;
    float current_yaw_meas_ = 0.0f;
    float current_yaw_sp_ = 0.0f;
    float target_yaw_ = 0.0f;
    uint8_t arming_state_ = 0;
    bool arm_sent_ = false;
    bool offboard_sent_ = false;

    const float flight_alt_ = 50.0f;
    float z_sp_ = 0.0f;
    float takeoff_elapsed_ = 0.0f;
    const float takeoff_ramp_time_ = 5.0f;
    const float cruise_speed_ = 5.0f;
    const float wp_switch_radius_ = 10.0f;
    const float yaw_speed_ = 1.0f;

    bool wp_initialized_ = false;
    std::vector<std::array<float,2>> wp_abs_;
    size_t seg_index_ = 0;

    void generate_lawnmower_path(float start_x, float start_y, float initial_yaw) {
        wp_abs_.clear();
        float length = 100.0f;
        float step = 25.0f; 
        float cos_y = std::cos(initial_yaw);
        float sin_y = std::sin(initial_yaw);

        for (float y = 0; y <= 100.1f; y += step) {
            std::vector<float> row_x;
            if (static_cast<int>(std::round(y / step)) % 2 == 0) row_x = {0.0f, length};
            else row_x = {length, 0.0f};
            for (float x : row_x) {
                float rot_x = x * cos_y - y * sin_y;
                float rot_y = x * sin_y + y * cos_y;
                wp_abs_.push_back({start_x + rot_x, start_y + rot_y});
            }
        }
    }

    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup:
                if (yaw_initialized_ && !offboard_sent_) {
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
                    offboard_sent_ = true;
                }
                if (offboard_sent_ && !arm_sent_) {
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
                    arm_sent_ = true;
                    ticks_ = 0;
                }
                if (arm_sent_ && ++ticks_ >= 50) {
                    RCLCPP_INFO(this->get_logger(), ">>> TAKEOFF");
                    phase_ = Phase::takeoff;
                }
                break;

            case Phase::takeoff:
                takeoff_elapsed_ += 0.1f;
                {
                    float s = std::min(1.0f, takeoff_elapsed_ / takeoff_ramp_time_);
                    z_sp_ = (1.0f - s) * current_pos_[2] + s * (-flight_alt_);
                }
                setpoint_pos_[0] = current_pos_[0];
                setpoint_pos_[1] = current_pos_[1];
                target_yaw_ = current_yaw_sp_;

                if (std::abs(current_pos_[2]) >= (flight_alt_ - 1.0f)) {
                    if (!wp_initialized_) {
                        generate_lawnmower_path(current_pos_[0], current_pos_[1], current_yaw_meas_);
                        wp_initialized_ = true;
                        seg_index_ = 0;
                    }
                    phase_ = Phase::hold;
                    ticks_ = 0;
                }
                break;

            case Phase::hold:
                if (++ticks_ >= 20) { 
                    RCLCPP_INFO(this->get_logger(), ">>> MISSION START");
                    phase_ = Phase::move;
                }
                break;

            case Phase::move: {
                if (seg_index_ >= wp_abs_.size()) {
                    RCLCPP_INFO(this->get_logger(), ">>> RTL");
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
                    phase_ = Phase::landing;
                    break;
                }
                const auto &B = wp_abs_[seg_index_];
                float dx = B[0] - current_pos_[0];
                float dy = B[1] - current_pos_[1];
                float dist = std::hypot(dx, dy);

                if (dist > 0.1f) {
                    target_vel_[0] = (dx / dist) * cruise_speed_;
                    target_vel_[1] = (dy / dist) * cruise_speed_;
                }
                if (dist < wp_switch_radius_) seg_index_++;

                setpoint_pos_[0] = B[0];
                setpoint_pos_[1] = B[1];
                z_sp_ = -flight_alt_;
                target_yaw_ = std::atan2(dy, dx);
                
                // Update Yaw
                float diff = target_yaw_ - current_yaw_sp_;
                while (diff > M_PI) diff -= 2.0f * M_PI;
                while (diff < -M_PI) diff += 2.0f * M_PI;
                float step = yaw_speed_ * 0.1f;
                if (std::abs(diff) < step) current_yaw_sp_ = target_yaw_;
                else current_yaw_sp_ += (diff > 0.0f) ? step : -step;
                break;
            }
            case Phase::landing: break;
        }
    }

    void publish_offboard_control_mode() {
        OffboardControlMode msg{};
        msg.position = true;
        msg.velocity = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        offboard_control_mode_publisher_->publish(msg);
    }

    void publish_trajectory_setpoint() {
        TrajectorySetpoint msg{};
        msg.position = {setpoint_pos_[0], setpoint_pos_[1], z_sp_};
        msg.velocity = (phase_ == Phase::move) ? std::array<float, 3>{target_vel_[0], target_vel_[1], 0.0f} : std::array<float, 3>{0.0f, 0.0f, 0.0f};
        msg.yaw = current_yaw_sp_;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        trajectory_setpoint_publisher_->publish(msg);
    }

    void publish_vehicle_command(uint16_t command, float param1 = 0.0f, float param2 = 0.0f) {
        VehicleCommand msg{};
        msg.command = command; msg.param1 = param1; msg.param2 = param2;
        msg.target_system = 1; msg.target_component = 1; msg.source_system = 1; msg.source_component = 1;
        msg.from_external = true; msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        vehicle_command_publisher_->publish(msg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardRealSimplifiedLog>());
    rclcpp::shutdown();
    return 0;
}