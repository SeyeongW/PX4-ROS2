#include <chrono>
#include <cmath>
#include <array>
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
        auto qos = rclcpp::SensorDataQoS();

        // --- 실기체 성능 최적화 파라미터 ---
        flight_alt_ = this->declare_parameter("flight_alt", 3.0f);
        max_step_m_ = this->declare_parameter("max_step_m", 2.5f);
        gain_lateral_ = this->declare_parameter("gain_lateral", 6.5f); // 좌우 반응성 (상향)
        gain_dist_ = this->declare_parameter("gain_dist", 8.5f);       // 전후 반응성 (상향)
        ema_alpha_ = this->declare_parameter("ema_alpha", 0.22f);      // 필터 반응성 (상향)
        target_h_ref_ = 0.45f; // 목표 거리 (사람이 화면 높이의 45%일 때 정지)

        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        local_pos_sub_ = this->create_subscription<VehicleLocalPosition>("/fmu/out/vehicle_local_position", qos,
            [this](const VehicleLocalPosition &msg) {
                current_pos_ = {msg.x, msg.y, msg.z};
                current_yaw_ = msg.heading;
                yaw_init_ = true;
            });

        status_sub_ = this->create_subscription<VehicleStatus>("/fmu/out/vehicle_status", qos,
            [this](const VehicleStatus &msg) { arm_state_ = msg.arming_state; });

        target_sub_ = this->create_subscription<geometry_msgs::msg::Point>("/perception/target_center", qos,
            [this](const geometry_msgs::msg::Point &msg) {
                if (std::isfinite(msg.x) && std::isfinite(msg.y) && std::isfinite(msg.z)) {
                    t_off_x_ = msg.x; t_off_y_ = msg.y; t_off_z_ = msg.z;
                    t_valid_ = true; last_t_time_ = this->get_clock()->now();
                } else { t_valid_ = false; }
            });

        target_id_sub_ = this->create_subscription<std_msgs::msg::Int32>("/perception/set_target_id", 10,
            [this](const std_msgs::msg::Int32 &msg) { locked_id_ = msg.data; });

        timer_ = this->create_wall_timer(100ms, [this]() {
            publish_offboard_control_mode();
            publish_trajectory_setpoint();
            manage_mission();
        });
    }

private:
    void manage_mission() {
        if (!yaw_init_) return;
        if (phase_ == "warmup" && ++ticks_ >= 20) {
            if (arm_state_ != VehicleStatus::ARMING_STATE_ARMED) {
                publish_cmd(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
                publish_cmd(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
            } else { phase_ = "takeoff"; }
        } else if (phase_ == "takeoff") {
            z_sp_ = std::max(-flight_alt_, z_sp_ - 0.08f);
            set_pos_[0] = current_pos_[0]; set_pos_[1] = current_pos_[1];
            if (std::abs(current_pos_[2] + flight_alt_) < 0.3f) { phase_ = "hover"; hover_xy_ = {set_pos_[0], set_pos_[1]}; }
        } else if (phase_ == "hover") { update_tracking(); }
    }

    void update_tracking() {
        bool fresh = t_valid_ && (this->get_clock()->now() - last_t_time_).seconds() <= 0.8;
        if (locked_id_ > 0 && fresh) {
            float b_x_lateral = t_off_x_ * gain_lateral_;
            float b_x_forward = (target_h_ref_ - t_off_z_) * gain_dist_;

            float cos_y = std::cos(current_yaw_), sin_y = std::sin(current_yaw_);
            float w_step_x = std::clamp(b_x_forward * cos_y - b_x_lateral * sin_y, -max_step_m_, max_step_m_);
            float w_step_y = std::clamp(b_x_forward * sin_y + b_x_lateral * cos_y, -max_step_m_, max_step_m_);

            set_pos_[0] = (1.0f - ema_alpha_) * set_pos_[0] + ema_alpha_ * (current_pos_[0] + w_step_x);
            set_pos_[1] = (1.0f - ema_alpha_) * set_pos_[1] + ema_alpha_ * (current_pos_[1] + w_step_y);
            hover_xy_ = {set_pos_[0], set_pos_[1]};
        } else {
            set_pos_[0] = 0.8f * set_pos_[0] + 0.2f * hover_xy_[0];
            set_pos_[1] = 0.8f * set_pos_[1] + 0.2f * hover_xy_[1];
        }
    }

    // 통신 함수들 (기존과 동일)
    void publish_offboard_control_mode() { OffboardControlMode m{}; m.position = true; m.timestamp = this->get_clock()->now().nanoseconds()/1000; offboard_control_mode_publisher_->publish(m); }
    void publish_trajectory_setpoint() { TrajectorySetpoint m{}; m.position = {set_pos_[0], set_pos_[1], z_sp_}; m.yaw = current_yaw_; m.timestamp = this->get_clock()->now().nanoseconds()/1000; trajectory_setpoint_publisher_->publish(m); }
    void publish_cmd(uint16_t c, float p1=0, float p2=0) { VehicleCommand m{}; m.command = c; m.param1 = p1; m.param2 = p2; m.target_system = 1; m.target_component = 1; m.source_system = 1; m.source_component = 1; m.from_external = true; m.timestamp = this->get_clock()->now().nanoseconds()/1000; vehicle_command_publisher_->publish(m); }

    std::string phase_ = "warmup"; int ticks_ = 0; bool yaw_init_ = false; float current_yaw_, z_sp_ = 0;
    std::array<float,3> current_pos_, set_pos_; std::array<float,2> hover_xy_;
    float t_off_x_, t_off_y_, t_off_z_; bool t_valid_; rclcpp::Time last_t_time_; int32_t locked_id_ = 0; uint8_t arm_state_;
    float flight_alt_, max_step_m_, gain_lateral_, gain_dist_, ema_alpha_, target_h_ref_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleLocalPosition>::SharedPtr local_pos_sub_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr target_id_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int a, char **v) { rclcpp::init(a,v); rclcpp::spin(std::make_shared<OffboardHoverTrackerReal>()); rclcpp::shutdown(); return 0; }