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
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <std_msgs/msg/int32.hpp>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class OffboardHoverTrackerReal : public rclcpp::Node {
public:
    OffboardHoverTrackerReal() : Node("offboard_hover_tracker_real") {
        auto qos_profile = rclcpp::SensorDataQoS();

        // 실기체 비행용 파라미터 선언
        flight_alt_ = this->declare_parameter("flight_alt", 5.0f); // 실기체는 낮게 시작 권장
        max_step_m_ = this->declare_parameter("max_step_m", 1.0f); 
        gain_forward_m_ = this->declare_parameter("gain_forward_m", 4.0f);
        gain_right_m_ = this->declare_parameter("gain_right_m", 4.0f);
        ema_alpha_ = this->declare_parameter("ema_alpha", 0.15f); // 반응성 상향

        // Publishers
        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        // Subscriptions (실기체 토픽명 적용)
        local_pos_subscription_ = this->create_subscription<VehicleLocalPosition>(
            "/fmu/out/vehicle_local_position", qos_profile,
            [this](const VehicleLocalPosition &msg) {
                current_pos_ = {msg.x, msg.y, msg.z};
                current_yaw_meas_ = msg.heading; // LocalPosition은 이미 heading(yaw)을 제공함

                if (!yaw_initialized_) {
                    current_yaw_sp_ = current_yaw_meas_;
                    yaw_initialized_ = true;
                }
            });

        status_subscription_ = this->create_subscription<VehicleStatus>(
            "/fmu/out/vehicle_status", qos_profile,
            [this](const VehicleStatus &msg) {
                arming_state_ = msg.arming_state;
                nav_state_ = msg.nav_state;
            });

        target_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/perception/target_center", qos_profile,
            [this](const geometry_msgs::msg::Point &msg) {
                if (std::isfinite(msg.x) && std::isfinite(msg.y)) {
                    target_offset_x_ = static_cast<float>(msg.x);
                    target_offset_y_ = static_cast<float>(msg.y);
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
                    RCLCPP_INFO(this->get_logger(), "RTL Requested.");
                    publish_cmd(VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
                    phase_ = Phase::landing;
                    return;
                }
                locked_id_ = (msg.data <= 0) ? 0 : msg.data;
            });

        timer_ = this->create_wall_timer(100ms, [this]() {
            if (phase_ != Phase::landing) {
                publish_offboard_control_mode();
                publish_trajectory_setpoint();
            }
            manage_mission_flow();
        });
    }

private:
    enum class Phase { warmup, takeoff, hover, landing };
    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;

    // PX4 통신용 변수
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleLocalPosition>::SharedPtr local_pos_subscription_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_subscription_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr target_id_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 상태 변수
    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f}, setpoint_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 2> hover_xy_{0.0f, 0.0f};
    bool yaw_initialized_ = false;
    float current_yaw_meas_ = 0.0f, current_yaw_sp_ = 0.0f;
    float z_sp_ = 0.0f, takeoff_elapsed_ = 0.0f;
    uint8_t arming_state_, nav_state_;

    // 타겟 변수
    float target_offset_x_ = 0.0f, target_offset_y_ = 0.0f;
    bool target_valid_ = false;
    rclcpp::Time last_target_time_{0, 0, RCL_ROS_TIME};
    int32_t locked_id_ = 0;

    // 파라미터
    float flight_alt_, max_step_m_, gain_forward_m_, gain_right_m_, ema_alpha_;

    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup:
                // 실기체에서는 수동으로 Offboard 전환 후 동작하게끔 안전하게 설계 가능
                if (yaw_initialized_ && ++ticks_ >= 20) {
                    RCLCPP_INFO_ONCE(this->get_logger(), "Arming and Setting Offboard...");
                    publish_cmd(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
                    publish_cmd(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
                    phase_ = Phase::takeoff;
                }
                break;

            case Phase::takeoff:
                takeoff_elapsed_ += 0.1f;
                z_sp_ = std::max(-flight_alt_, -(takeoff_elapsed_ * 0.8f)); // 실기체는 천천히 이륙(0.8m/s)
                setpoint_pos_[0] = current_pos_[0]; 
                setpoint_pos_[1] = current_pos_[1];
                if (std::abs(current_pos_[2] + flight_alt_) < 0.3f) {
                    hover_xy_ = {current_pos_[0], current_pos_[1]};
                    phase_ = Phase::hover;
                    RCLCPP_INFO(this->get_logger(), "Hovering Initialized.");
                }
                break;

            case Phase::hover:
                update_hover_tracking();
                break;

            case Phase::landing:
                break;
        }
    }

    void update_hover_tracking() {
        const auto now = this->get_clock()->now();
        const bool target_fresh = target_valid_ && (now - last_target_time_).seconds() <= 0.8;

        if (locked_id_ > 0 && target_fresh) {
            // Body Frame 오차 계산
            float body_x = -target_offset_y_ * gain_forward_m_;
            float body_y = target_offset_x_ * gain_right_m_;

            // 회전 변환 (Body -> World NED)
            float cos_y = std::cos(current_yaw_meas_);
            float sin_y = std::sin(current_yaw_meas_);
            float world_step_x = std::clamp(body_x * cos_y - body_y * sin_y, -max_step_m_, max_step_m_);
            float world_step_y = std::clamp(body_x * sin_y + body_y * cos_y, -max_step_m_, max_step_m_);

            // EMA 필터 적용
            setpoint_pos_[0] = (1.0f - ema_alpha_) * setpoint_pos_[0] + ema_alpha_ * (current_pos_[0] + world_step_x);
            setpoint_pos_[1] = (1.0f - ema_alpha_) * setpoint_pos_[1] + ema_alpha_ * (current_pos_[1] + world_step_y);
            hover_xy_ = {setpoint_pos_[0], setpoint_pos_[1]};
        } else {
            // 타겟 유실 시 제자리 정지 유지
            setpoint_pos_[0] = (1.0f - 0.2f) * setpoint_pos_[0] + 0.2f * hover_xy_[0];
            setpoint_pos_[1] = (1.0f - 0.2f) * setpoint_pos_[1] + 0.2f * hover_xy_[1];
        }
    }

    void publish_offboard_control_mode() {
        OffboardControlMode msg{};
        msg.position = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        offboard_control_mode_publisher_->publish(msg);
    }

    void publish_trajectory_setpoint() {
        TrajectorySetpoint msg{};
        msg.position = {setpoint_pos_[0], setpoint_pos_[1], z_sp_};
        msg.yaw = current_yaw_sp_;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        trajectory_setpoint_publisher_->publish(msg);
    }

    void publish_cmd(uint16_t cmd, float p1 = 0, float p2 = 0) {
        VehicleCommand msg{};
        msg.command = cmd;
        msg.param1 = p1;
        msg.param2 = p2;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        vehicle_command_publisher_->publish(msg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardHoverTrackerReal>());
    rclcpp::shutdown();
    return 0;
}