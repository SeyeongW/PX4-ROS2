#include <chrono>
#include <cmath>
#include <array>
#include <cstdint>
#include <algorithm>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#include <mavros_msgs/srv/set_mode.hpp>
#include <std_msgs/msg/int32.hpp>

using namespace std::chrono_literals;

class ArduPilotHoverTracker : public rclcpp::Node {
public:
    ArduPilotHoverTracker() : Node("ardupilot_hover_tracker") {
        
        flight_alt_ = this->declare_parameter("flight_alt", 10.0f);
        max_step_m_ = this->declare_parameter("max_step_m", 1.2f); 
        gain_forward_m_ = this->declare_parameter("gain_forward_m", 3.5f);
        gain_right_m_ = this->declare_parameter("gain_right_m", 3.5f);
        ema_alpha_ = this->declare_parameter("ema_alpha", 0.07f); 

        // Publishers
        target_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/mavros/setpoint_position/local", 10);

        // Subscriptions
        state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
            "/mavros/state", 10,
            [this](const mavros_msgs::msg::State::SharedPtr msg) { current_state_ = *msg; });

        local_pos_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/mavros/local_position/pose", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                current_pos_ = {static_cast<float>(msg->pose.position.x), 
                                static_cast<float>(msg->pose.position.y), 
                                static_cast<float>(msg->pose.position.z)};
                
                float qz = msg->pose.orientation.z;
                float qw = msg->pose.orientation.w;
                current_yaw_meas_ = 2.0f * std::atan2(qz, qw);

                if (!yaw_initialized_) { current_yaw_sp_ = current_yaw_meas_; yaw_initialized_ = true; }
                if (!local_pos_ready_) { local_pos_ready_ = true; setpoint_pos_ = current_pos_; }
            });

        target_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/perception/target_center", 10,
            [this](const geometry_msgs::msg::Point::SharedPtr msg) {
                if (std::isfinite(msg->x) && std::isfinite(msg->y)) {
                    target_offset_x_ = static_cast<float>(msg->x);
                    target_offset_y_ = static_cast<float>(msg->y);
                    target_valid_ = true;
                    last_target_time_ = this->get_clock()->now();
                } else { target_valid_ = false; }
            });

        target_id_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/perception/set_target_id", 10,
            [this](const std_msgs::msg::Int32::SharedPtr msg) {
                if (msg->data == -1) {
                    RCLCPP_INFO(this->get_logger(), "RTL command received. Returning to launch...");
                    set_mode("RTL");
                    phase_ = Phase::landing;
                    return;
                }
                locked_id_ = (msg->data <= 0) ? 0 : msg->data;
            });

        // Service Clients
        arming_client_ = this->create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
        set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");

        timer_ = this->create_wall_timer(100ms, [this]() {
            if (phase_ == Phase::landing) {
                if (!current_state_.armed) {
                    RCLCPP_INFO(this->get_logger(), ">>> MISSION COMPLETE / DISARMED");
                    rclcpp::shutdown();
                    return;
                }
            }
            publish_target_pose();
            manage_mission_flow();
        });
    }

private:
    enum class Phase { warmup, takeoff, hover, landing };
    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;
    
    mavros_msgs::msg::State current_state_;
    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f}, setpoint_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 2> hover_xy_{0.0f, 0.0f};
    
    bool yaw_initialized_ = false, local_pos_ready_ = false;
    float current_yaw_meas_ = 0.0f, current_yaw_sp_ = 0.0f;
    float takeoff_elapsed_ = 0.0f, flight_alt_ = 10.0f;
    float target_offset_x_ = 0.0f, target_offset_y_ = 0.0f;
    bool target_valid_ = false;
    
    rclcpp::Time last_target_time_{0, 0, RCL_ROS_TIME};
    int32_t locked_id_ = 0;
    float max_step_m_, gain_forward_m_, gain_right_m_, ema_alpha_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_pub_;
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr local_pos_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr target_id_sub_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
    rclcpp::TimerBase::SharedPtr timer_;

    void set_mode(std::string mode) {
        auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        request->custom_mode = mode;
        set_mode_client_->async_send_request(request);
    }

    void arm(bool value) {
        auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
        request->value = value;
        arming_client_->async_send_request(request);
    }

    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup:
                if (local_pos_ready_ && current_state_.mode != "GUIDED" && ticks_ % 20 == 0) {
                    set_mode("GUIDED");
                }
                if (current_state_.mode == "GUIDED" && !current_state_.armed && ticks_ % 20 == 0) {
                    arm(true);
                }
                if (current_state_.armed && current_state_.mode == "GUIDED") {
                    if (++ticks_ >= 50) {
                        RCLCPP_INFO(this->get_logger(), ">>> TAKEOFF");
                        phase_ = Phase::takeoff;
                        takeoff_elapsed_ = 0.0f;
                        ticks_ = 0;
                    }
                } else {
                    ticks_++;
                }
                break;
            case Phase::takeoff:
                takeoff_elapsed_ += 0.1f;
                setpoint_pos_[2] = std::min(flight_alt_, takeoff_elapsed_ * 1.5f);
                setpoint_pos_[0] = current_pos_[0]; setpoint_pos_[1] = current_pos_[1];
                if (std::abs(current_pos_[2] - flight_alt_) < 0.5f) {
                    hover_xy_ = {current_pos_[0], current_pos_[1]};
                    phase_ = Phase::hover;
                }
                break;
            case Phase::hover:
                update_hover_tracking();
                break;
            case Phase::landing: break;
        }
    }

    void update_hover_tracking() {
        const auto now = this->get_clock()->now();
        const bool target_fresh = target_valid_ && (now - last_target_time_).seconds() <= 1.0;

        if (locked_id_ > 0 && target_fresh) {
            // ENU: 화면 아래(y>0)가 전진(+N), 화면 오른쪽(x>0)이 우측 이동(+E)
            float body_n = -target_offset_y_ * gain_forward_m_;
            float body_e = target_offset_x_ * gain_right_m_;

            float cos_y = std::cos(current_yaw_meas_);
            float sin_y = std::sin(current_yaw_meas_);

            float world_step_e = body_e * cos_y + body_n * sin_y;
            float world_step_n = -body_e * sin_y + body_n * cos_y;

            world_step_e = std::clamp(world_step_e, -max_step_m_, max_step_m_);
            world_step_n = std::clamp(world_step_n, -max_step_m_, max_step_m_);

            setpoint_pos_[0] = (1.0f - ema_alpha_) * setpoint_pos_[0] + ema_alpha_ * (current_pos_[0] + world_step_e);
            setpoint_pos_[1] = (1.0f - ema_alpha_) * setpoint_pos_[1] + ema_alpha_ * (current_pos_[1] + world_step_n);
            
            hover_xy_ = {setpoint_pos_[0], setpoint_pos_[1]};
        } else {
            setpoint_pos_[0] = (1.0f - 0.1f) * setpoint_pos_[0] + 0.1f * hover_xy_[0];
            setpoint_pos_[1] = (1.0f - 0.1f) * setpoint_pos_[1] + 0.1f * hover_xy_[1];
        }
        setpoint_pos_[2] = flight_alt_;
    }

    void publish_target_pose() {
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "map";
        msg.pose.position.x = setpoint_pos_[0];
        msg.pose.position.y = setpoint_pos_[1];
        msg.pose.position.z = setpoint_pos_[2];
        msg.pose.orientation.z = std::sin(current_yaw_sp_ / 2.0f);
        msg.pose.orientation.w = std::cos(current_yaw_sp_ / 2.0f);
        target_pose_pub_->publish(msg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArduPilotHoverTracker>());
    rclcpp::shutdown();
    return 0;
}