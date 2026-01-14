#include <chrono>
#include <cmath>
#include <array>
#include <cstdint>
#include <algorithm>
#include <limits>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <std_msgs/msg/int32.hpp>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class OffboardHoverTracker : public rclcpp::Node {
public:
    OffboardHoverTracker() : Node("offboard_hover_tracker") {
        auto qos_profile = rclcpp::SensorDataQoS();

        flight_alt_ = this->declare_parameter("flight_alt", 25.0f);
        takeoff_ramp_time_ = this->declare_parameter("takeoff_ramp_time", 3.0f);
        target_timeout_s_ = this->declare_parameter("target_timeout_s", 1.0f);
        max_step_m_ = this->declare_parameter("max_step_m", 2.0f);
        gain_forward_m_ = this->declare_parameter("gain_forward_m", 3.0f);
        gain_right_m_ = this->declare_parameter("gain_right_m", 3.0f);

        offboard_control_mode_publisher_ =
            this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ =
            this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ =
            this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        odometry_subscription_ = this->create_subscription<VehicleOdometry>(
            "/fmu/out/vehicle_odometry", qos_profile,
            [this](const VehicleOdometry &msg) {
                current_pos_ = {msg.position[0], msg.position[1], msg.position[2]};

                const float qw = msg.q[0];
                const float qx = msg.q[1];
                const float qy = msg.q[2];
                const float qz = msg.q[3];

                const float siny_cosp = 2.0f * (qw * qz + qx * qy);
                const float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
                current_yaw_meas_ = std::atan2(siny_cosp, cosy_cosp);

                if (!yaw_initialized_) {
                    current_yaw_sp_ = current_yaw_meas_;
                    yaw_initialized_ = true;
                }

                if (phase_ == Phase::warmup || phase_ == Phase::takeoff) {
                    setpoint_pos_[0] = current_pos_[0];
                    setpoint_pos_[1] = current_pos_[1];
                }
            });

        status_subscription_ = this->create_subscription<VehicleStatus>(
            "/fmu/out/vehicle_status_v1", qos_profile,
            [this](const VehicleStatus &msg) { arming_state_ = msg.arming_state; });

        target_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/perception/target_center", qos_profile,
            [this](const geometry_msgs::msg::Point &msg) {
                if (std::isfinite(msg.x) && std::isfinite(msg.y)) {
                    target_offset_x_ = static_cast<float>(msg.x);
                    target_offset_y_ = static_cast<float>(msg.y);
                    target_area_ = static_cast<float>(msg.z);
                    target_valid_ = true;
                    last_target_time_ = this->get_clock()->now();
                } else {
                    target_valid_ = false;
                }
            });

        target_id_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
            "/perception/set_target_id", 10,
            [this](const std_msgs::msg::Int32 &msg) {
                if (msg.data == -1) {
                    RCLCPP_WARN(this->get_logger(), "RTL requested via target ID -1.");
                    start_rtl_sequence();
                    return;
                }

                if (msg.data <= 0) {
                    locked_id_ = 0;
                    RCLCPP_INFO(this->get_logger(), "Target lock released.");
                } else {
                    locked_id_ = msg.data;
                    RCLCPP_INFO(this->get_logger(), "Target locked: ID %d", locked_id_);
                }
            });

        timer_ = this->create_wall_timer(100ms, [this]() {
            if (phase_ == Phase::landing) {
                if (arming_state_ == VehicleStatus::ARMING_STATE_DISARMED) {
                    RCLCPP_INFO(this->get_logger(),
                                ">>> Disarm confirmed. Mission complete. Shutting down.");
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
    enum class Phase { warmup, takeoff, hover, landing };

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_subscription_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_subscription_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr target_id_subscription_;

    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;

    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> setpoint_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 2> hover_xy_{0.0f, 0.0f};

    bool  yaw_initialized_ = false;
    float current_yaw_meas_ = 0.0f;
    float current_yaw_sp_   = 0.0f;

    uint8_t arming_state_ = 0;
    bool arm_sent_ = false;
    bool offboard_sent_ = false;

    const float dt_ = 0.1f;
    const uint64_t warmup_armed_ticks_ = 50;

    float flight_alt_ = 25.0f;
    bool  z_ramp_init_ = false;
    float z_takeoff_start_ = 0.0f;
    float z_sp_ = 0.0f;
    float takeoff_elapsed_ = 0.0f;
    float takeoff_ramp_time_ = 3.0f;

    const float yaw_speed_ = 0.5f;

    float target_offset_x_ = 0.0f;
    float target_offset_y_ = 0.0f;
    float target_area_ = 0.0f;
    bool target_valid_ = false;
    rclcpp::Time last_target_time_{0, 0, RCL_ROS_TIME};

    int32_t locked_id_ = 0;

    float target_timeout_s_ = 1.0f;
    float max_step_m_ = 2.0f;
    float gain_forward_m_ = 3.0f;
    float gain_right_m_ = 3.0f;

    static float wrap_angle(float angle) {
        while (angle > M_PI) angle -= 2.0f * M_PI;
        while (angle < -M_PI) angle += 2.0f * M_PI;
        return angle;
    }

    static float clampf(float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    }

    void update_yaw_smoothing() {
        const float diff = wrap_angle(current_yaw_meas_ - current_yaw_sp_);
        const float step = yaw_speed_ * dt_;

        if (std::abs(diff) < step) current_yaw_sp_ = current_yaw_meas_;
        else current_yaw_sp_ += (diff > 0.0f) ? step : -step;

        current_yaw_sp_ = wrap_angle(current_yaw_sp_);
    }

    void update_takeoff_z_ramp() {
        if (!z_ramp_init_) {
            z_takeoff_start_ = current_pos_[2];
            z_sp_ = z_takeoff_start_;
            takeoff_elapsed_ = 0.0f;
            z_ramp_init_ = true;
        }

        takeoff_elapsed_ += dt_;
        const float s = std::min(1.0f, takeoff_elapsed_ / takeoff_ramp_time_);
        z_sp_ = (1.0f - s) * z_takeoff_start_ + s * (-flight_alt_);
    }

    void publish_offboard_control_mode() {
        OffboardControlMode msg{};
        msg.position = true;
        msg.velocity = false;
        msg.acceleration = false;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        offboard_control_mode_publisher_->publish(msg);
    }

    void publish_trajectory_setpoint() {
        TrajectorySetpoint msg{};

        msg.position[0] = setpoint_pos_[0];
        msg.position[1] = setpoint_pos_[1];

        if (phase_ == Phase::warmup) {
            msg.position[2] = current_pos_[2];
        } else {
            msg.position[2] = z_sp_;
        }

        msg.yaw = current_yaw_sp_;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        trajectory_setpoint_publisher_->publish(msg);
    }

    void publish_vehicle_command(uint16_t command, float param1 = 0.0f, float param2 = 0.0f) {
        VehicleCommand msg{};
        msg.param1 = param1;
        msg.param2 = param2;
        msg.command = command;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        vehicle_command_publisher_->publish(msg);
    }

    void start_rtl_sequence() {
        if (phase_ == Phase::landing) {
            return;
        }
        publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
        phase_ = Phase::landing;
    }

    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup: {
                if (!yaw_initialized_) {
                    if (++ticks_ % 10 == 0) {
                        RCLCPP_INFO(this->get_logger(), "Waiting for odometry to capture yaw...");
                    }
                    break;
                }

                if (!offboard_sent_) {
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
                    offboard_sent_ = true;
                    RCLCPP_INFO(this->get_logger(), "Offboard mode request sent.");
                }

                if (!arm_sent_) {
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
                    arm_sent_ = true;
                    ticks_ = 0;
                    break;
                }

                if (++ticks_ >= warmup_armed_ticks_) {
                    phase_ = Phase::takeoff;
                    ticks_ = 0;
                    z_ramp_init_ = false;
                }
            } break;

            case Phase::takeoff: {
                update_takeoff_z_ramp();
                update_yaw_smoothing();

                if (++ticks_ % 10 == 0) {
                    const float alt = std::abs(current_pos_[2]);
                    RCLCPP_INFO(this->get_logger(), "Climbing alt=%.1f z_sp=%.1f", alt, z_sp_);
                }

                const float alt = std::abs(current_pos_[2]);
                if (alt >= (flight_alt_ - 0.5f) && (takeoff_elapsed_ >= takeoff_ramp_time_ * 0.8f)) {
                    hover_xy_ = {current_pos_[0], current_pos_[1]};
                    setpoint_pos_[0] = hover_xy_[0];
                    setpoint_pos_[1] = hover_xy_[1];
                    z_sp_ = -flight_alt_;
                    RCLCPP_INFO(this->get_logger(), "Reached altitude. Hovering at %.1f, %.1f", 
                                hover_xy_[0], hover_xy_[1]);
                    phase_ = Phase::hover;
                    ticks_ = 0;
                }
            } break;

            case Phase::hover: {
                update_yaw_smoothing();

                const auto now = this->get_clock()->now();
                const bool target_fresh = target_valid_ &&
                    (now - last_target_time_).seconds() <= target_timeout_s_;

                if (locked_id_ > 0 && target_fresh) {
                    const float delta_forward = clampf(-target_offset_y_ * gain_forward_m_,
                                                       -max_step_m_, max_step_m_);
                    const float delta_right = clampf(target_offset_x_ * gain_right_m_,
                                                     -max_step_m_, max_step_m_);

                    setpoint_pos_[0] = current_pos_[0] + delta_forward;
                    setpoint_pos_[1] = current_pos_[1] + delta_right;
                } else {
                    setpoint_pos_[0] = hover_xy_[0];
                    setpoint_pos_[1] = hover_xy_[1];
                }

                if (++ticks_ % 10 == 0) {
                    RCLCPP_INFO(this->get_logger(),
                                "Hover lock=%d target=(%.2f,%.2f) area=%.3f fresh=%d",
                                locked_id_, target_offset_x_, target_offset_y_,
                                target_area_, target_fresh);
                }
            } break;

            case Phase::landing:
                break;
        }
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardHoverTracker>());
    rclcpp::shutdown();
    return 0;
}
