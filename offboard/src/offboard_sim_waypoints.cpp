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

        flight_alt_ = this->declare_parameter("flight_alt", 10.0f);
        max_step_m_ = this->declare_parameter("max_step_m", 1.2f); 
        gain_forward_m_ = this->declare_parameter("gain_forward_m", 3.5f);
        gain_right_m_ = this->declare_parameter("gain_right_m", 3.5f);
        ema_alpha_ = this->declare_parameter("ema_alpha", 0.07f); 

        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        odometry_subscription_ = this->create_subscription<VehicleOdometry>(
            "/fmu/out/vehicle_odometry", qos_profile,
            [this](const VehicleOdometry &msg) {
                current_pos_ = {msg.position[0], msg.position[1], msg.position[2]};
                float siny_cosp = 2.0f * (msg.q[0] * msg.q[3] + msg.q[1] * msg.q[2]);
                float cosy_cosp = 1.0f - 2.0f * (msg.q[2] * msg.q[2] + msg.q[3] * msg.q[3]);
                current_yaw_meas_ = std::atan2(siny_cosp, cosy_cosp);
                if (!yaw_initialized_) { current_yaw_sp_ = current_yaw_meas_; yaw_initialized_ = true; }
            });

        target_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/perception/target_center", qos_profile,
            [this](const geometry_msgs::msg::Point &msg) {
                if (std::isfinite(msg.x) && std::isfinite(msg.y)) {
                    target_offset_x_ = static_cast<float>(msg.x);
                    target_offset_y_ = static_cast<float>(msg.y);
                    target_valid_ = true;
                    last_target_time_ = this->get_clock()->now();
                } else { target_valid_ = false; }
            });

        target_id_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
            "/perception/set_target_id", 10,
            [this](const std_msgs::msg::Int32 &msg) {
                // [추가] -1 입력 시 RTL 모드 전환
                if (msg.data == -1) {
                    RCLCPP_INFO(this->get_logger(), "RTL command received. Returning to launch...");
                    publish_cmd(VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
                    phase_ = Phase::landing;
                    return;
                }
                locked_id_ = (msg.data <= 0) ? 0 : msg.data;
            });

        timer_ = this->create_wall_timer(100ms, [this]() {
            // Landing(RTL) 중에는 오프보드 제어 메시지를 보내지 않음 (명령 충돌 방지)
            if (phase_ != Phase::landing) {
                publish_offboard_control_mode();
                publish_trajectory_setpoint();
            }
            manage_mission_flow();
        });
    }

private:
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_subscription_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr target_id_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;

    enum class Phase { warmup, takeoff, hover, landing };
    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;
    
    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f}, setpoint_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 2> hover_xy_{0.0f, 0.0f};
    
    bool yaw_initialized_ = false;
    float current_yaw_meas_ = 0.0f, current_yaw_sp_ = 0.0f;
    float z_sp_ = 0.0f, takeoff_elapsed_ = 0.0f, flight_alt_ = 10.0f;
    float target_offset_x_ = 0.0f, target_offset_y_ = 0.0f;
    bool target_valid_ = false;
    
    rclcpp::Time last_target_time_{0, 0, RCL_ROS_TIME};
    int32_t locked_id_ = 0;
    float max_step_m_, gain_forward_m_, gain_right_m_, ema_alpha_;

    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup:
                if (yaw_initialized_ && ++ticks_ >= 50) {
                    publish_cmd(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
                    publish_cmd(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
                    phase_ = Phase::takeoff; takeoff_elapsed_ = 0.0f;
                }
                break;
            case Phase::takeoff:
                takeoff_elapsed_ += 0.1f;
                z_sp_ = std::max(-flight_alt_, -(takeoff_elapsed_ * 1.5f));
                setpoint_pos_[0] = current_pos_[0]; setpoint_pos_[1] = current_pos_[1];
                if (std::abs(current_pos_[2]) >= (flight_alt_ - 0.5f)) {
                    hover_xy_ = {current_pos_[0], current_pos_[1]};
                    phase_ = Phase::hover;
                }
                break;
            case Phase::hover:
                update_hover_tracking();
                break;
            case Phase::landing:
                // RTL 상태에서는 별도 작업 없음 (PX4 내부 로직에 위임)
                break;
        }
    }

    void update_hover_tracking() {
    const auto now = this->get_clock()->now();
    const bool target_fresh = target_valid_ && (now - last_target_time_).seconds() <= 1.0;

    if (locked_id_ > 0 && target_fresh) {
        // 1. 카메라 좌표계에서의 이동량 계산 (Body Frame 개념)
        // 화면 아래(y>0)가 전진, 화면 오른쪽(x>0)이 우측 이동
        float body_x = -target_offset_y_ * gain_forward_m_;
        float body_y = target_offset_x_ * gain_right_m_;

        // 2. 현재 드론이 바라보는 Yaw 각도를 반영하여 월드 좌표계(NED)로 변환
        // 회전 행렬 적용: 
        // world_x = body_x * cos(yaw) - body_y * sin(yaw)
        // world_y = body_x * sin(yaw) + body_y * cos(yaw)
        float cos_y = std::cos(current_yaw_meas_);
        float sin_y = std::sin(current_yaw_meas_);

        float world_step_x = body_x * cos_y - body_y * sin_y;
        float world_step_y = body_x * sin_y + body_y * cos_y;

        // 3. 변환된 좌표에 최대 보폭 제한(Clamp) 적용
        world_step_x = std::clamp(world_step_x, -max_step_m_, max_step_m_);
        world_step_y = std::clamp(world_step_y, -max_step_m_, max_step_m_);

        // 4. EMA 필터로 부드럽게 업데이트
        setpoint_pos_[0] = (1.0f - ema_alpha_) * setpoint_pos_[0] + ema_alpha_ * (current_pos_[0] + world_step_x);
        setpoint_pos_[1] = (1.0f - ema_alpha_) * setpoint_pos_[1] + ema_alpha_ * (current_pos_[1] + world_step_y);
        
        hover_xy_ = {setpoint_pos_[0], setpoint_pos_[1]};
        }else {
        setpoint_pos_[0] = (1.0f - 0.1f) * setpoint_pos_[0] + 0.1f * hover_xy_[0];
        setpoint_pos_[1] = (1.0f - 0.1f) * setpoint_pos_[1] + 0.1f * hover_xy_[1];
        }
    }

    void publish_offboard_control_mode() {
        OffboardControlMode msg{}; msg.position = true;
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
        VehicleCommand msg{}; msg.command = cmd; msg.param1 = p1; msg.param2 = p2;
        msg.target_system = 1; msg.target_component = 1; msg.source_system = 1; msg.source_component = 1;
        msg.from_external = true; msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        vehicle_command_publisher_->publish(msg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardHoverTracker>());
    rclcpp::shutdown();
    return 0;
}