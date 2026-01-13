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

class OffboardFinalSquare : public rclcpp::Node {
public:
    OffboardFinalSquare() : Node("offboard_final_square") {
        RCLCPP_INFO(this->get_logger(), "\n\n================================================");
        RCLCPP_INFO(this->get_logger(), "   >>> [Version 1.1] Offboard Node Started <<<   ");
        RCLCPP_INFO(this->get_logger(), "================================================\n");

        auto qos_profile = rclcpp::SensorDataQoS();

        // Publishers
        offboard_control_mode_publisher_ =
            this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ =
            this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ =
            this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        // Subscribers (VehicleLocalPosition 사용)
        local_position_subscription_ = this->create_subscription<VehicleLocalPosition>(
            "/fmu/out/vehicle_local_position", qos_profile,
            [this](const VehicleLocalPosition &msg) {
                current_pos_ = {msg.x, msg.y, msg.z};
                current_yaw_meas_ = msg.heading;

                if (!yaw_initialized_) {
                    current_yaw_sp_ = current_yaw_meas_;
                    target_yaw_ = current_yaw_sp_;
                    yaw_initialized_ = true;
                }

                if (phase_ == Phase::warmup || phase_ == Phase::takeoff) {
                    setpoint_pos_[0] = current_pos_[0];
                    setpoint_pos_[1] = current_pos_[1];
                }
            });

        status_subscription_ = this->create_subscription<VehicleStatus>(
            "/fmu/out/vehicle_status_v1", qos_profile,
            [this](const VehicleStatus &msg) {
                arming_state_ = msg.arming_state;
            });

        // Timer: 100ms (10Hz)
        timer_ = this->create_wall_timer(100ms, [this]() {
            if (phase_ == Phase::landing) {
                if (arming_state_ == VehicleStatus::ARMING_STATE_DISARMED) {
                    RCLCPP_INFO(this->get_logger(), ">>> Disarm confirmed. Mission complete. Shutting down.");
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
    enum class Phase { warmup, takeoff, yaw_align, move, landing };

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleLocalPosition>::SharedPtr local_position_subscription_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_subscription_;

    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;
    uint64_t dbg_tick_ = 0;

    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> setpoint_pos_{0.0f, 0.0f, 0.0f};

    bool  yaw_initialized_ = false;
    float current_yaw_meas_ = 0.0f;
    float current_yaw_sp_   = 0.0f;
    float target_yaw_       = 0.0f;

    uint8_t arming_state_ = 0;
    bool arm_sent_ = false;
    bool offboard_sent_ = false;

    const float dt_ = 0.1f;
    const uint64_t warmup_armed_ticks_ = 50; 

    const float flight_alt_ = 50.0f;

    bool  z_ramp_init_ = false;
    float z_takeoff_start_ = 0.0f;
    float z_sp_ = 0.0f;
    float takeoff_elapsed_ = 0.0f;
    const float takeoff_ramp_time_ = 4.0f; 

    const float yaw_speed_ = 0.5f;

    const float yaw_align_tol_rad_ = 0.10f;     
    const uint64_t yaw_align_timeout_ticks_ = 80; 
    uint64_t yaw_align_ticks_ = 0;

    const float wp_switch_radius_ = 10.0f;

    bool wp_initialized_ = false;
    std::array<float,2> start_xy_{0.0f, 0.0f};
    std::vector<std::array<float,2>> wp_abs_;
    std::vector<std::array<float,2>> wp_offsets_ = {
        {0.0f,   0.0f},
        {50.0f, 0.0f},
        {50.0f, 50.0f},
        {0.0f,   50.0f},
        {0.0f,   0.0f}
    };
    size_t seg_index_ = 0;

    const bool use_rtl_for_landing_ = true;
    const bool use_home_xy_then_land_ = false;

    static float wrap_angle(float angle) {
        while (angle > M_PI) angle -= 2.0f * M_PI;
        while (angle < -M_PI) angle += 2.0f * M_PI;
        return angle;
    }

    void update_yaw_smoothing() {
        const float diff = wrap_angle(target_yaw_ - current_yaw_sp_);
        const float step = yaw_speed_ * dt_;

        if (std::abs(diff) < step) current_yaw_sp_ = target_yaw_;
        else current_yaw_sp_ += (diff > 0.0f) ? step : -step;

        current_yaw_sp_ = wrap_angle(current_yaw_sp_);
    }

    // [Reverted] 강제 점프 삭제 -> 부드러운 램프 로직으로 복귀
    void update_takeoff_z_ramp() {
        if (!z_ramp_init_) {
            z_takeoff_start_ = current_pos_[2];
            z_sp_ = z_takeoff_start_; // 현재 위치에서 시작 (점프 없음)
            
            takeoff_elapsed_ = 0.0f;
            z_ramp_init_ = true;
            RCLCPP_INFO(this->get_logger(), "Takeoff Ramp Init: Start Z=%.2f", z_takeoff_start_);
        }

        takeoff_elapsed_ += dt_;
        const float s = std::min(1.0f, takeoff_elapsed_ / takeoff_ramp_time_);
        
        // 부드럽게 상승 (현재 위치 -> -50m)
        float final_alt = -flight_alt_;
        z_sp_ = (1.0f - s) * z_takeoff_start_ + s * final_alt;
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

    void start_landing_sequence() {
        if (use_rtl_for_landing_) {
            RCLCPP_INFO(this->get_logger(), "Landing: RTL (Return-to-Launch).");
            publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
            phase_ = Phase::landing;
            return;
        }

        if (use_home_xy_then_land_) {
            RCLCPP_INFO(this->get_logger(), "Landing: go to home XY then LAND.");
            publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
            phase_ = Phase::landing;
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Landing: NAV_LAND in place.");
        publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
        phase_ = Phase::landing;
    }

    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup: {
                if (!yaw_initialized_) {
                    if (++ticks_ % 10 == 0) {
                        RCLCPP_INFO(this->get_logger(), "Waiting for local position to capture yaw...");
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

                if (++ticks_ % 10 == 0) {
                    const float alt = std::abs(current_pos_[2]);
                    RCLCPP_INFO(this->get_logger(), "Climbing alt=%.1f z_sp=%.1f", alt, z_sp_);
                }

                const float alt = std::abs(current_pos_[2]);
                if (alt >= (flight_alt_ - 1.0f) && (takeoff_elapsed_ >= takeoff_ramp_time_ * 0.8f)) {
                    RCLCPP_INFO(this->get_logger(), "Reached altitude. Initializing waypoints...");

                    if (!wp_initialized_) {
                        start_xy_ = {current_pos_[0], current_pos_[1]};
                        wp_abs_.clear();
                        for (const auto &w : wp_offsets_) {
                            wp_abs_.push_back({start_xy_[0] + w[0], start_xy_[1] + w[1]});
                        }
                        wp_initialized_ = true;
                        seg_index_ = 0;

                        setpoint_pos_[0] = current_pos_[0];
                        setpoint_pos_[1] = current_pos_[1];
                    }

                    yaw_align_ticks_ = 0;
                    phase_ = Phase::yaw_align;
                    ticks_ = 0;
                    dbg_tick_ = 0;
                }
            } break;

            case Phase::yaw_align: {
                if (!wp_initialized_ || wp_abs_.size() < 2) {
                    RCLCPP_ERROR(this->get_logger(), "Waypoints not initialized. Landing for safety.");
                    start_landing_sequence();
                    break;
                }

                setpoint_pos_[0] = current_pos_[0];
                setpoint_pos_[1] = current_pos_[1];
                setpoint_pos_[2] = z_sp_;

                const auto &B = wp_abs_[1];
                const float dx = B[0] - current_pos_[0];
                const float dy = B[1] - current_pos_[1];

                target_yaw_ = std::atan2(dy, dx);
                update_yaw_smoothing();

                const float yaw_err = std::abs(wrap_angle(target_yaw_ - current_yaw_sp_));
                if (++yaw_align_ticks_ % 10 == 0) {
                    RCLCPP_INFO(this->get_logger(), "Yaw-align: err=%.2f rad (%.1f deg)", yaw_err, yaw_err * 57.2958f);
                }

                if (yaw_err < yaw_align_tol_rad_) {
                    RCLCPP_INFO(this->get_logger(), "Yaw aligned. Start moving to WP1.");
                    phase_ = Phase::move;
                    dbg_tick_ = 0;
                } else if (yaw_align_ticks_ >= yaw_align_timeout_ticks_) {
                    RCLCPP_WARN(this->get_logger(), "Yaw-align timeout. Start moving anyway.");
                    phase_ = Phase::move;
                    dbg_tick_ = 0;
                }
            } break;

            case Phase::move: {
                if (!wp_initialized_ || wp_abs_.size() < 2) {
                    RCLCPP_ERROR(this->get_logger(), "Waypoints not initialized. Landing for safety.");
                    start_landing_sequence();
                    break;
                }

                if (seg_index_ + 1 >= wp_abs_.size()) {
                    RCLCPP_INFO(this->get_logger(), "All segments done.");
                    start_landing_sequence();
                    break;
                }

                const std::array<float, 2> A = wp_abs_[seg_index_];
                const std::array<float, 2> B = wp_abs_[seg_index_ + 1];
                const std::array<float, 2> P = {current_pos_[0], current_pos_[1]};

                const float dxB = B[0] - P[0];
                const float dyB = B[1] - P[1];
                const float dist_to_B = std::hypot(dxB, dyB);

                const std::array<float, 2> AB = {B[0] - A[0], B[1] - A[1]};
                const std::array<float, 2> PB = {P[0] - B[0], P[1] - B[1]};
                const float passed_gate = PB[0] * AB[0] + PB[1] * AB[1];

                if (dist_to_B < wp_switch_radius_ || passed_gate > 0.0f) {
                    RCLCPP_INFO(this->get_logger(),
                                "Switch seg: seg=%zu -> %zu (dist=%.1f passed_gate=%.2f)",
                                seg_index_, seg_index_ + 1, dist_to_B, passed_gate);
                    seg_index_++;

                    if (seg_index_ + 1 >= wp_abs_.size()) {
                        RCLCPP_INFO(this->get_logger(), "All segments done.");
                        start_landing_sequence();
                    }
                    break;
                }

                setpoint_pos_[0] = B[0];
                setpoint_pos_[1] = B[1];
                setpoint_pos_[2] = z_sp_;

                target_yaw_ = std::atan2(dyB, dxB);
                update_yaw_smoothing();

                if (++dbg_tick_ % 10 == 0) {
                    RCLCPP_INFO(this->get_logger(),
                                "seg=%zu P(%.1f,%.1f) B(%.1f,%.1f) dist=%.1f passed=%.2f yaw=%.2f->%.2f",
                                seg_index_,
                                P[0], P[1],
                                B[0], B[1],
                                dist_to_B, passed_gate,
                                current_yaw_sp_, target_yaw_);
                }
            } break;

            case Phase::landing:
                break;
        }
    }

};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardFinalSquare>());
    rclcpp::shutdown();
    return 0;
}
