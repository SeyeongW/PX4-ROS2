#include <chrono>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/point.hpp>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class OffboardFinalSquare : public rclcpp::Node {
public:
    OffboardFinalSquare() : Node("offboard_final_square") {
        auto qos_profile = rclcpp::SensorDataQoS();

        // Publishers
        offboard_control_mode_publisher_ =
            this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ =
            this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ =
            this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        // Subscribers
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
            [this](const VehicleStatus &msg) { arming_state_ = msg.arming_state; });
        
        // [핵심 1] 명령(ID, 0, -1) 처리 로직
        target_id_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
            "/perception/set_target_id", 10,
            [this](const std_msgs::msg::Int32 &msg) {
                int cmd = msg.data;

                // 1. RTL (-1)
                if (cmd == -1) {
                    RCLCPP_WARN(this->get_logger(), ">>> COMMAND: LAND (RTL) <<<");
                    start_landing_sequence();
                    target_id_ = 0;
                    is_tracking_active_ = false;
                    return;
                }

                // 2. 다시 비행 (0) -> 호버링 풀고 사각형 비행 복귀
                if (cmd == 0) {
                    RCLCPP_INFO(this->get_logger(), ">>> COMMAND: RESUME FLIGHT (Square) <<<");
                    target_id_ = 0;
                    is_tracking_active_ = false;
                    // 현재 호버링 중이면 Move 모드로 강제 전환
                    if (phase_ == Phase::hover || phase_ == Phase::chase) {
                        phase_ = Phase::move; 
                        // 가까운 웨이포인트부터 다시 시작하도록 로직 추가 가능하나
                        // 여기서는 그냥 하던 인덱스부터 계속 진행
                    }
                    return;
                }

                // 3. 특정 ID 트래킹 (1 이상)
                if (cmd > 0) {
                    RCLCPP_INFO(this->get_logger(), ">>> COMMAND: TRACK ID %d <<<", cmd);
                    target_id_ = cmd;
                    // 아직 추적 활성화는 아님 (YOLO가 위치를 줘야 함)
                }
            });

        // 타겟 위치 정보 수신
        target_info_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/perception/target_pos_info", 10,
            [this](const geometry_msgs::msg::Point &msg) {
                // YOLO가 타겟을 보고 있으면 z > 0 (박스 크기)
                if (target_id_ > 0 && msg.z > 0.0001) {
                    is_tracking_active_ = true;
                    last_target_msg_ = msg;
                } 
                // [추가] ID 지정 안 했는데 뭔가 사람이 감지됨 (호버링용)
                // msg.x가 0이 아니면(뭔가 감지됨) & 현재 트래킹 모드가 아니면 -> 호버링 트리거
                else if (target_id_ == 0 && msg.z > 0.0001) {
                    // ID가 0일 때 사람이 발견되면 -> 'detected_for_hover' 플래그 
                    person_detected_any_ = true;
                } else {
                    is_tracking_active_ = false;
                    person_detected_any_ = false;
                }
            });

        timer_ = this->create_wall_timer(100ms, [this]() {
            if (rtl_requested_.exchange(false)) start_landing_sequence();

            if (phase_ == Phase::landing) {
                if (arming_state_ == VehicleStatus::ARMING_STATE_DISARMED) {
                    rclcpp::shutdown();
                    return;
                }
            } else {
                publish_offboard_control_mode();
                publish_trajectory_setpoint();
                manage_mission_flow();
            }
        });
        start_command_listener();
    }

private:
    enum class Phase { warmup, takeoff, yaw_align, move, hover, chase, landing };

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_subscription_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_subscription_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr target_id_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_info_subscription_;

    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;
    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> setpoint_pos_{0.0f, 0.0f, 0.0f};

    bool  yaw_initialized_ = false;
    float current_yaw_meas_ = 0.0f;
    float current_yaw_sp_   = 0.0f;
    float target_yaw_       = 0.0f;
    uint8_t arming_state_ = 0;
    bool arm_sent_ = false;
    bool offboard_sent_ = false;

    int32_t target_id_ = 0;
    bool is_tracking_active_ = false;
    bool person_detected_any_ = false; // ID 상관없이 사람 발견됨
    
    geometry_msgs::msg::Point last_target_msg_;
    
    // 추적 파라미터
    const float k_yaw = 0.8f;
    const float k_fwd = 3.0f;
    const float desired_area = 0.08f;
    const float max_vel = 2.0f;

    std::atomic<bool> rtl_requested_{false};

    const float dt_ = 0.1f;
    const float flight_alt_ = 5.0f;

    bool  z_ramp_init_ = false;
    float z_takeoff_start_ = 0.0f;
    float z_sp_ = 0.0f;
    float takeoff_elapsed_ = 0.0f;
    const float takeoff_ramp_time_ = 4.0f;
    const float yaw_speed_ = 0.5f;

    bool wp_initialized_ = false;
    std::array<float,2> start_xy_{0.0f, 0.0f};
    std::vector<std::array<float,2>> wp_abs_;
    std::vector<std::array<float,2>> wp_offsets_ = {
        {0.0f, 0.0f}, {50.0f, 0.0f}, {50.0f, 50.0f}, {0.0f, 50.0f}, {0.0f, 0.0f}
    };
    size_t seg_index_ = 0;
    const float wp_switch_radius_ = 2.0f;

    static float wrap_angle(float angle) {
        while (angle > M_PI) angle -= 2.0f * M_PI;
        while (angle < -M_PI) angle += 2.0f * M_PI;
        return angle;
    }
    
    static float clampf(float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    }

    void start_command_listener() {
        std::thread([this]() {
            std::string line;
            while (std::getline(std::cin, line)) { if (line == "rtl") rtl_requested_ = true; }
        }).detach();
    }

    void update_yaw_smoothing() {
        const float diff = wrap_angle(target_yaw_ - current_yaw_sp_);
        const float step = yaw_speed_ * dt_;
        if (std::abs(diff) < step) current_yaw_sp_ = target_yaw_;
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
        msg.position[2] = (phase_ == Phase::warmup) ? current_pos_[2] : z_sp_;
        msg.yaw = current_yaw_sp_;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        trajectory_setpoint_publisher_->publish(msg);
    }

    void publish_vehicle_command(uint16_t command, float param1 = 0.0f, float param2 = 0.0f) {
        VehicleCommand msg{};
        msg.command = command;
        msg.param1 = param1;
        msg.param2 = param2;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000ULL;
        vehicle_command_publisher_->publish(msg);
    }

    void start_landing_sequence() {
        RCLCPP_INFO(this->get_logger(), "Landing: NAV_RETURN_TO_LAUNCH.");
        publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_RETURN_TO_LAUNCH);
        phase_ = Phase::landing;
    }

    void manage_mission_flow() {
        // [상태 전환 로직]
        // 비행 중이고 착륙/이륙 단계가 아닐 때
        if (phase_ != Phase::landing && phase_ != Phase::warmup && phase_ != Phase::takeoff) {
            
            // 1. 추적 명령(ID)이 있고 + 실제 타겟도 보임 -> CHASE
            if (target_id_ > 0 && is_tracking_active_) {
                if (phase_ != Phase::chase) {
                    RCLCPP_WARN(this->get_logger(), ">>> TRACKING START! (ID: %d) <<<", target_id_);
                    phase_ = Phase::chase;
                }
            }
            // 2. ID는 0인데(명령 없음) + 사람이 보임 -> HOVER (일단 멈춤)
            else if (target_id_ == 0 && person_detected_any_) {
                if (phase_ != Phase::hover) {
                    RCLCPP_WARN(this->get_logger(), ">>> PERSON DETECTED! HOVERING <<<");
                    phase_ = Phase::hover;
                    setpoint_pos_ = current_pos_; // 현재 위치 고정
                    target_yaw_ = current_yaw_meas_;
                }
            }
            // 3. 아무것도 안 보이고 ID 명령도 없음 -> MOVE (사각형 비행)
            // (단, Hover 중이었으면 0번 커맨드를 받아야 Move로 풀리도록 설정 가능.
            //  여기서는 '사람 사라지면 다시 비행'이 아니라 '0번 눌러야 비행'을 원하시면 아래 else if 제거)
            /*
            else if (phase_ == Phase::hover && !person_detected_any_) {
                // 사람이 사라져도 가만히 있을 것인가, 다시 움직일 것인가?
                // 사용자 요청: "0을 넣으면 다시 비행" -> 즉, 자동으로 풀리면 안 됨.
                // 따라서 여기서는 아무것도 안 함 (Hover 유지).
            }
            */
        }

        switch (phase_) {
            case Phase::warmup:
                if (!yaw_initialized_) return;
                if (!offboard_sent_) {
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
                    offboard_sent_ = true;
                }
                if (!arm_sent_) {
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
                    arm_sent_ = true;
                }
                if (++ticks_ >= 50) { phase_ = Phase::takeoff; ticks_ = 0; }
                break;

            case Phase::takeoff:
                update_takeoff_z_ramp();
                if (std::abs(current_pos_[2]) >= (flight_alt_ - 1.0f)) {
                    start_xy_ = {current_pos_[0], current_pos_[1]};
                    wp_abs_.clear();
                    for (const auto &w : wp_offsets_) {
                        wp_abs_.push_back({start_xy_[0] + w[0], start_xy_[1] + w[1]});
                    }
                    wp_initialized_ = true;
                    setpoint_pos_ = current_pos_;
                    phase_ = Phase::yaw_align; 
                }
                break;
            
            case Phase::yaw_align:
                phase_ = Phase::move; 
                break;

            case Phase::move: {
                // 평소 사각형 비행
                if (!wp_initialized_) break;
                if (seg_index_ + 1 >= wp_abs_.size()) { 
                    // 한 바퀴 다 돌면 다시 0번부터 (계속 뺑뺑이)
                    seg_index_ = 0; 
                }
                
                const std::array<float, 2> B = wp_abs_[seg_index_ + 1];
                const float dx = B[0] - current_pos_[0];
                const float dy = B[1] - current_pos_[1];
                const float dist = std::hypot(dx, dy);

                if (dist < wp_switch_radius_) seg_index_++;

                setpoint_pos_[0] = B[0];
                setpoint_pos_[1] = B[1];
                target_yaw_ = std::atan2(dy, dx);
                update_yaw_smoothing();
                break;
            }

            case Phase::hover: {
                // 사람 발견 시 제자리 정지
                // (0번 커맨드를 받아야 Phase::move로 풀림)
                if (++ticks_ % 20 == 0) {
                    RCLCPP_INFO(this->get_logger(), "Hovering... Waiting for command (0=Resume, ID=Track)");
                }
                break;
            }

            case Phase::chase: {
                // 트래킹 모드 (ID 지정됨)
                float yaw_error = last_target_msg_.x; 
                target_yaw_ = current_yaw_meas_ + (yaw_error * k_yaw);
                update_yaw_smoothing();

                float current_area = last_target_msg_.z;
                float dist_error = desired_area - current_area; 
                float body_vel_x = clampf(dist_error * k_fwd, -max_vel, max_vel);

                float cos_yaw = std::cos(current_yaw_meas_);
                float sin_yaw = std::sin(current_yaw_meas_);

                setpoint_pos_[0] = current_pos_[0] + (body_vel_x * cos_yaw) * dt_;
                setpoint_pos_[1] = current_pos_[1] + (body_vel_x * sin_yaw) * dt_;
                
                if (++ticks_ % 10 == 0) {
                    RCLCPP_INFO(this->get_logger(), "Tracking ID %d... Vel=%.2f", target_id_, body_vel_x);
                }
                break;
            }

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