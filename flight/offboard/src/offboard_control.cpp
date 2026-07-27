#include <chrono>
#include <cmath>
#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#include <mavros_msgs/srv/set_mode.hpp>

using namespace std::chrono_literals;

class ArduPilotOffboardControl : public rclcpp::Node {
public:
    ArduPilotOffboardControl() : Node("ardupilot_offboard_node") {
        
        // Publishers
        target_pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/mavros/setpoint_position/local", 10);
        
        // Subscriptions
        state_subscription_ = this->create_subscription<mavros_msgs::msg::State>(
            "/mavros/state", 10,
            [this](const mavros_msgs::msg::State::SharedPtr msg) { 
                current_state_ = *msg; 
            });

        local_pos_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/mavros/local_position/pose", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                current_pos_ = {static_cast<float>(msg->pose.position.x), 
                                static_cast<float>(msg->pose.position.y), 
                                static_cast<float>(msg->pose.position.z)};
                
                // Simplified yaw extraction from quaternion
                float qz = msg->pose.orientation.z;
                float qw = msg->pose.orientation.w;
                current_yaw_meas_ = 2.0f * std::atan2(qz, qw);

                if (!yaw_initialized_) {
                    current_yaw_sp_ = current_yaw_meas_;
                    target_yaw_ = current_yaw_sp_;
                    yaw_initialized_ = true;
                }
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

            // In ArduPilot/MAVROS, we should stream setpoints even before entering GUIDED
            publish_target_pose();
            manage_mission_flow();
        });
    }

private:
    enum class Phase { warmup, takeoff, hold, move, landing };

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_publisher_;
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr local_pos_subscription_;
    
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;

    mavros_msgs::msg::State current_state_;
    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;
    
    // ENU Coordinates: [East, North, Up]
    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> setpoint_pos_{0.0f, 0.0f, 0.0f};

    bool yaw_initialized_ = false;
    float current_yaw_meas_ = 0.0f;
    float current_yaw_sp_ = 0.0f;
    float target_yaw_ = 0.0f;

    const float flight_alt_ = 50.0f; // 50m Up
    float takeoff_elapsed_ = 0.0f;
    const float takeoff_ramp_time_ = 5.0f;
    const float cruise_speed_ = 5.0f;
    const float wp_switch_radius_ = 10.0f;
    const float yaw_speed_ = 1.0f;

    bool wp_initialized_ = false;
    std::vector<std::array<float,2>> wp_abs_; // {East, North}
    size_t seg_index_ = 0;

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

    void generate_lawnmower_path(float start_east, float start_north, float initial_yaw) {
        wp_abs_.clear();
        float length = 100.0f;
        float step = 25.0f; 
        
        // initial_yaw is ENU (0 = East, CCW)
        float cos_y = std::cos(initial_yaw);
        float sin_y = std::sin(initial_yaw);

        for (float n = 0; n <= 100.1f; n += step) {
            std::vector<float> row_e;
            if (static_cast<int>(std::round(n / step)) % 2 == 0) row_e = {0.0f, length};
            else row_e = {length, 0.0f};
            
            for (float e : row_e) {
                // Rotation in ENU
                float rot_e = e * cos_y - n * sin_y;
                float rot_n = e * sin_y + n * cos_y;
                wp_abs_.push_back({start_east + rot_e, start_north + rot_n});
            }
        }
    }

    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup:
                if (yaw_initialized_ && current_state_.mode != "GUIDED" && ticks_ % 20 == 0) {
                    RCLCPP_INFO(this->get_logger(), "Setting mode to GUIDED...");
                    set_mode("GUIDED");
                }
                if (current_state_.mode == "GUIDED" && !current_state_.armed && ticks_ % 20 == 0) {
                    RCLCPP_INFO(this->get_logger(), "Arming...");
                    arm(true);
                }
                if (current_state_.armed && current_state_.mode == "GUIDED") {
                    ticks_++;
                    if (ticks_ >= 20) {
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
                {
                    float s = std::min(1.0f, takeoff_elapsed_ / takeoff_ramp_time_);
                    setpoint_pos_[2] = (1.0f - s) * current_pos_[2] + s * flight_alt_;
                }
                setpoint_pos_[0] = current_pos_[0];
                setpoint_pos_[1] = current_pos_[1];
                target_yaw_ = current_yaw_sp_;

                if (std::abs(current_pos_[2] - flight_alt_) <= 1.0f) {
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
                    set_mode("RTL");
                    phase_ = Phase::landing;
                    break;
                }
                const auto &B = wp_abs_[seg_index_];
                float de = B[0] - current_pos_[0];
                float dn = B[1] - current_pos_[1];
                float dist = std::hypot(de, dn);

                if (dist < wp_switch_radius_) seg_index_++;

                setpoint_pos_[0] = B[0];
                setpoint_pos_[1] = B[1];
                setpoint_pos_[2] = flight_alt_;
                target_yaw_ = std::atan2(dn, de);
                
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

    void publish_target_pose() {
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "map";
        msg.pose.position.x = setpoint_pos_[0];
        msg.pose.position.y = setpoint_pos_[1];
        msg.pose.position.z = setpoint_pos_[2];
        
        // Simple yaw to quaternion (z-axis only)
        msg.pose.orientation.z = std::sin(current_yaw_sp_ / 2.0f);
        msg.pose.orientation.w = std::cos(current_yaw_sp_ / 2.0f);
        
        target_pose_publisher_->publish(msg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArduPilotOffboardControl>());
    rclcpp::shutdown();
    return 0;
}