#include <chrono>
#include <cmath>
#include <vector>
#include <array>
#include <sstream>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#include <mavros_msgs/srv/set_mode.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

using namespace std::chrono_literals;

class ArduPilotTracker : public rclcpp::Node {
public:
    ArduPilotTracker() : Node("ardupilot_tracker_node") {
        
        RCLCPP_INFO(this->get_logger(), "Tracker Started");

        // Parameters
        flight_alt_ = this->declare_parameter("flight_alt", 5.0f);
        takeoff_ramp_time_ = this->declare_parameter("takeoff_ramp_time", 5.0f);
        gain_lateral_ = this->declare_parameter("gain_lateral", 7.5f);
        gain_dist_ = this->declare_parameter("gain_dist", 9.5f);
        max_step_m_ = this->declare_parameter("max_step_m", 2.5f);
        yaw_speed_ = this->declare_parameter("yaw_speed", 0.15f);

        // Publishers
        target_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/mavros/setpoint_position/local", 10);
        debug_pub_ = this->create_publisher<std_msgs::msg::String>("/offboard_tracking/debug", 10);

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

                if (!yaw_init_) {
                    current_yaw_sp_ = current_yaw_meas_;
                    yaw_init_ = true;
                }
                if (!local_pos_ready_) {
                    local_pos_ready_ = true;
                    setpoint_pos_ = current_pos_;
                }
            });

        target_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/perception/target_center", 10,
            [this](const geometry_msgs::msg::Point::SharedPtr msg) {
                if (std::isfinite(msg->x)) {
                    t_off_x_ = static_cast<float>(msg->x); 
                    t_off_z_ = static_cast<float>(msg->z);
                    t_valid_ = true; 
                    last_t_time_ = this->get_clock()->now();
                } else { t_valid_ = false; }
            });

        target_id_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/perception/set_target_id", 10,
            [this](const std_msgs::msg::Int32::SharedPtr msg) {
                if (msg->data <= 0) {
                    RCLCPP_INFO(this->get_logger(), ">>> COMMAND 0: RTL");
                    set_mode("RTL");
                    phase_ = Phase::landing;
                    return;
                }
                locked_id_ = msg->data;
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

        debug_timer_ = this->create_wall_timer(1s, [this]() { publish_debug(); });
    }

private:
    enum class Phase { warmup, takeoff, hold, track, landing };
    Phase phase_ = Phase::warmup;

    mavros_msgs::msg::State current_state_;
    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> setpoint_pos_{0.0f, 0.0f, 0.0f};
    float current_yaw_meas_ = 0.0f, current_yaw_sp_ = 0.0f;
    bool yaw_init_ = false, t_valid_ = false, local_pos_ready_ = false;
    uint64_t ticks_ = 0;
    bool tracking_active_ = false;
    float t_off_x_ = 0.0f, t_off_z_ = 0.0f;
    int32_t locked_id_ = 0;
    rclcpp::Time last_t_time_{0, 0, RCL_ROS_TIME};

    float flight_alt_, takeoff_ramp_time_, gain_lateral_, gain_dist_, max_step_m_, yaw_speed_;
    float takeoff_elapsed_ = 0.0f;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr local_pos_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr target_id_sub_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr debug_timer_;

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
                    if (++ticks_ >= 20) {
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
                if (std::abs(current_pos_[2] - flight_alt_) < 0.5f) {
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

            case Phase::landing: break;
        }
    }

    void execute_tracking() {
        bool fresh = t_valid_ && (this->get_clock()->now() - last_t_time_).seconds() <= 0.8;
        if (locked_id_ > 0 && fresh) {
            if (!tracking_active_) {
                RCLCPP_INFO(this->get_logger(), ">>> TARGET TRACKING ACTIVE");
                tracking_active_ = true;
            }
            // ENU Conversion: Body to World
            // t_off_z is distance in front, t_off_x is lateral error
            float b_forward = (0.45f - t_off_z_) * gain_dist_;
            float b_lateral = -t_off_x_ * gain_lateral_; // Negative for ENU lateral

            float c = std::cos(current_yaw_meas_), s = std::sin(current_yaw_meas_);
            float step_e = std::clamp(b_forward * s + b_lateral * c, -max_step_m_, max_step_m_);
            float step_n = std::clamp(b_forward * c - b_lateral * s, -max_step_m_, max_step_m_);

            setpoint_pos_[0] = current_pos_[0] + step_e;
            setpoint_pos_[1] = current_pos_[1] + step_n;

            // Yaw Tracking
            float target_yaw_err = -t_off_x_ * 0.8f;
            float target_yaw = current_yaw_meas_ + target_yaw_err;

            float diff = target_yaw - current_yaw_sp_;
            while (diff > M_PI) diff -= 2.0f * M_PI;
            while (diff < -M_PI) diff += 2.0f * M_PI;

            if (std::abs(diff) < yaw_speed_) current_yaw_sp_ = target_yaw;
            else current_yaw_sp_ += (diff > 0.0f) ? yaw_speed_ : -yaw_speed_;

        } else {
            if (tracking_active_) {
                RCLCPP_WARN(this->get_logger(), ">>> TARGET LOST: holding position");
                tracking_active_ = false;
            }
            locked_id_ = 0;
            t_valid_ = false;
            setpoint_pos_[0] = current_pos_[0];
            setpoint_pos_[1] = current_pos_[1];
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

    void publish_debug() {
        std_msgs::msg::String msg{};
        const auto now = this->get_clock()->now();
        const double target_age = t_valid_ ? (now - last_t_time_).seconds() : -1.0;
        const bool fresh = t_valid_ && target_age <= 0.8;

        std::ostringstream oss;
        oss << "phase=" << static_cast<int>(phase_)
            << " armed=" << static_cast<int>(current_state_.armed)
            << " mode=" << current_state_.mode
            << " locked_id=" << locked_id_
            << " tracking=" << (tracking_active_ ? "1" : "0")
            << " target_valid=" << (t_valid_ ? "1" : "0")
            << " target_fresh=" << (fresh ? "1" : "0")
            << " target_age=" << target_age
            << " t_off_x=" << t_off_x_
            << " t_off_z=" << t_off_z_
            << " pos=(" << current_pos_[0] << "," << current_pos_[1] << "," << current_pos_[2] << ")"
            << " sp=(" << setpoint_pos_[0] << "," << setpoint_pos_[1] << "," << setpoint_pos_[2] << ")"
            << " yaw_sp=" << current_yaw_sp_;

        msg.data = oss.str();
        debug_pub_->publish(msg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArduPilotTracker>());
    rclcpp::shutdown();
    return 0;
}
