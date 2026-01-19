#include <chrono>
#include <cmath>
#include <vector>
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

class OffboardTrackerLawnmowerStyle : public rclcpp::Node {
public:
    OffboardTrackerLawnmowerStyle() : Node("offboard_tracker_lawnmower_style") {
        auto qos = rclcpp::SensorDataQoS();

        RCLCPP_INFO(this->get_logger(), "Tracker Started");

        // --- 파라미터 설정 (반응 속도 상향 수치) ---
        flight_alt_ = this->declare_parameter("flight_alt", 5.0f);
        takeoff_ramp_time_ = this->declare_parameter("takeoff_ramp_time", 5.0f);
        
        // 반응성 향상을 위해 Gain 및 Step 상향
        gain_lateral_ = this->declare_parameter("gain_lateral", 7.5f);  // 6.0 -> 7.5
        gain_dist_ = this->declare_parameter("gain_dist", 9.5f);        // 8.0 -> 9.5
        max_step_m_ = this->declare_parameter("max_step_m", 2.5f);      // 2.0 -> 2.5
        yaw_speed_ = this->declare_parameter("yaw_speed", 0.15f);       // Yaw 회전 속도 (Rad/step)

        offboard_control_mode_pub_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_pub_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_pub_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        local_pos_sub_ = this->create_subscription<VehicleLocalPosition>("/fmu/out/vehicle_local_position", qos,
            [this](const VehicleLocalPosition &msg) {
                current_pos_ = {msg.x, msg.y, msg.z};
                current_yaw_meas_ = msg.heading;
                if (!yaw_init_) {
                    current_yaw_sp_ = current_yaw_meas_;
                    yaw_init_ = true;
                }
                if (!local_pos_ready_) {
                    local_pos_ready_ = true;
                    setpoint_pos_ = current_pos_;
                    hover_xy_ = {current_pos_[0], current_pos_[1]};
                    z_sp_ = current_pos_[2];
                }
            });

        status_sub_ = this->create_subscription<VehicleStatus>("/fmu/out/vehicle_status", qos,
            [this](const VehicleStatus &msg) { arming_state_ = msg.arming_state; });

        target_sub_ = this->create_subscription<geometry_msgs::msg::Point>("/perception/target_center", qos,
            [this](const geometry_msgs::msg::Point &msg) {
                if (std::isfinite(msg.x)) {
                    t_off_x_ = static_cast<float>(msg.x); 
                    t_off_z_ = static_cast<float>(msg.z);
                    t_valid_ = true; 
                    last_t_time_ = this->get_clock()->now();
                } else { t_valid_ = false; }
            });

        target_id_sub_ = this->create_subscription<std_msgs::msg::Int32>("/perception/set_target_id", 10,
            [this](const std_msgs::msg::Int32 &msg) {
                if (msg.data <= 0) {
                    RCLCPP_INFO(this->get_logger(), ">>> COMMAND 0: RTL");
                    publish_cmd(VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
                    phase_ = Phase::landing;
                    return;
                }
                locked_id_ = msg.data;
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
    enum class Phase { warmup, takeoff, hold, track, landing };
    Phase phase_ = Phase::warmup;

    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> setpoint_pos_{0,0,0};
    float current_yaw_meas_ = 0, current_yaw_sp_ = 0, z_sp_ = 0;
    bool yaw_init_ = false, t_valid_ = false;
    uint64_t ticks_ = 0;
    uint64_t offboard_setpoint_counter_ = 0;
    uint64_t command_retry_ticks_ = 0;
    bool local_pos_ready_ = false;
    uint8_t arming_state_ = 0;
    float t_off_x_, t_off_z_;
    int32_t locked_id_ = 0;
    rclcpp::Time last_t_time_{0, 0, RCL_ROS_TIME};

    float flight_alt_, takeoff_ramp_time_, gain_lateral_, gain_dist_, max_step_m_, yaw_speed_;
    float takeoff_elapsed_ = 0;
    std::array<float, 2> hover_xy_;
    static constexpr uint64_t kOffboardSetpointRequired = 20;
    static constexpr uint64_t kCommandRetryIntervalTicks = 10;

    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup:
                if (!local_pos_ready_) {
                    break;
                }
                ++ticks_;
                ++offboard_setpoint_counter_;
                if (offboard_setpoint_counter_ >= kOffboardSetpointRequired &&
                    (command_retry_ticks_ == 0 || (ticks_ % kCommandRetryIntervalTicks) == 0)) {
                    publish_cmd(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
                    publish_cmd(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
                    command_retry_ticks_ = ticks_;
                }
                if (arming_state_ == VehicleStatus::ARMING_STATE_ARMED) {
                    phase_ = Phase::takeoff;
                    takeoff_elapsed_ = 0.0f;
                }
                break;

            case Phase::takeoff:
                if (!local_pos_ready_) {
                    break;
                }
                takeoff_elapsed_ += 0.1f;
                {
                    float s = std::min(1.0f, takeoff_elapsed_ / takeoff_ramp_time_);
                    z_sp_ = (1.0f - s) * current_pos_[2] + s * (-flight_alt_);
                }
                setpoint_pos_[0] = current_pos_[0];
                setpoint_pos_[1] = current_pos_[1];
                if (std::abs(current_pos_[2] + flight_alt_) < 0.5f) {
                    hover_xy_ = {current_pos_[0], current_pos_[1]};
                    phase_ = Phase::hold;
                    ticks_ = 0;
                }
                break;

            case Phase::hold:
                if (++ticks_ >= 30) { 
                    RCLCPP_INFO(this->get_logger(), ">>> TRACKING MODE READY");
                    phase_ = Phase::track; 
                }
                break;

            case Phase::track:
                execute_tracking();
                break;

            case Phase::landing:
                if (arming_state_ == VehicleStatus::ARMING_STATE_DISARMED) rclcpp::shutdown();
                break;
        }
    }

    void execute_tracking() {
        bool fresh = t_valid_ && (this->get_clock()->now() - last_t_time_).seconds() <= 0.8;
        if (locked_id_ > 0 && fresh) {
            // 1. 위치 추적 (Body Frame 오차 -> World Frame 변환)
            float b_forward = (0.45f - t_off_z_) * gain_dist_;
            float b_lateral = t_off_x_ * gain_lateral_;

            float c = std::cos(current_yaw_meas_), s = std::sin(current_yaw_meas_);
            float step_x = std::clamp(b_forward * c - b_lateral * s, -max_step_m_, max_step_m_);
            float step_y = std::clamp(b_forward * s + b_lateral * c, -max_step_m_, max_step_m_);

            setpoint_pos_[0] = current_pos_[0] + step_x;
            setpoint_pos_[1] = current_pos_[1] + step_y;
            hover_xy_ = {setpoint_pos_[0], setpoint_pos_[1]};

            // 2. Yaw 추적 로직 추가 (사람을 정면으로 바라보도록 회전)
            // t_off_x는 화면 중앙으로부터의 오차(-1.0 ~ 1.0)
            float target_yaw_err = -t_off_x_ * 0.8f; // 오차에 비례한 회전 각도 계산
            float target_yaw = current_yaw_meas_ + target_yaw_err;

            // Yaw 보간 (부드러운 회전)
            float diff = target_yaw - current_yaw_sp_;
            while (diff > M_PI) diff -= 2.0f * M_PI;
            while (diff < -M_PI) diff += 2.0f * M_PI;

            if (std::abs(diff) < yaw_speed_) current_yaw_sp_ = target_yaw;
            else current_yaw_sp_ += (diff > 0.0f) ? yaw_speed_ : -yaw_speed_;

        } else {
            setpoint_pos_[0] = hover_xy_[0];
            setpoint_pos_[1] = hover_xy_[1];
            // 타겟 유실 시 Yaw는 현재 유지
        }
        z_sp_ = -flight_alt_;
    }

    void publish_offboard_control_mode() {
        OffboardControlMode msg{}; msg.position = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        offboard_control_mode_pub_->publish(msg);
    }

    void publish_trajectory_setpoint() {
        TrajectorySetpoint msg{};
        msg.position = {setpoint_pos_[0], setpoint_pos_[1], z_sp_};
        msg.yaw = current_yaw_sp_;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        trajectory_setpoint_pub_->publish(msg);
    }

    void publish_cmd(uint16_t cmd, float p1 = 0, float p2 = 0) {
        VehicleCommand msg{}; msg.command = cmd; msg.param1 = p1; msg.param2 = p2;
        msg.target_system = 1; msg.target_component = 1; msg.source_system = 1; msg.source_component = 1;
        msg.from_external = true; msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        vehicle_command_pub_->publish(msg);
    }

    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_pub_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_pub_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_pub_;
    rclcpp::Subscription<VehicleLocalPosition>::SharedPtr local_pos_sub_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr target_id_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardTrackerLawnmowerStyle>());
    rclcpp::shutdown();
    return 0;
}
