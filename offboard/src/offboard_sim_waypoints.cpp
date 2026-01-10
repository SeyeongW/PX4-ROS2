// [헤더 순서: 표준 -> ROS 2 -> PX4]
#include <chrono>
#include <cmath>
#include <vector>
#include <array>
#include <cstdint>
#include <algorithm> // for min, abs

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

// 각도 정규화 유틸리티 (-PI ~ PI)
float wrap_angle(float angle) {
    while (angle > M_PI) angle -= 2.0f * M_PI;
    while (angle < -M_PI) angle += 2.0f * M_PI;
    return angle;
}

class OffboardFinalSquare : public rclcpp::Node {
public:
    OffboardFinalSquare() : Node("offboard_final_square") {
        auto qos_profile = rclcpp::SensorDataQoS();

        // Publishers
        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        // Subscribers
        odometry_subscription_ = this->create_subscription<VehicleOdometry>(
            "/fmu/out/vehicle_odometry", qos_profile,
            [this](const VehicleOdometry &msg) {
                // NED 좌표계: Z는 위쪽이 음수
                current_pos_ = {msg.position[0], msg.position[1], msg.position[2]};
                
                // 이륙 전/초기화 단계일 때 미끼 위치 초기화
                if (phase_ == Phase::takeoff || phase_ == Phase::warmup) {
                    setpoint_pos_ = {msg.position[0], msg.position[1], -flight_alt_};
                    // 초기 Yaw는 북쪽(-3.14)으로 고정하거나 현재 Yaw를 따라가게 설정
                }
            });
            
        status_subscription_ = this->create_subscription<VehicleStatus>(
            "/fmu/out/vehicle_status_v1", qos_profile, // v1 토픽 사용 권장
            [this](const VehicleStatus &msg) {
                arming_state_ = msg.arming_state;
            });

        // Timer: 100ms (10Hz)
        timer_ = this->create_wall_timer(100ms, [this]() {
            // [핵심] 종료 시퀀스: 착륙 단계이고 시동이 꺼졌으면(1) 종료
            if (phase_ == Phase::landing) {
                if (arming_state_ == 1) { // 1 = DISARMED
                    RCLCPP_INFO(this->get_logger(), ">>> Disarm Confirmed. Mission Complete. Shutting Down.");
                    rclcpp::shutdown();
                    return;
                }
            }

            // 착륙 모드가 아닐 때만 Offboard 신호 전송
            if (phase_ != Phase::landing) {
                publish_offboard_control_mode();
                publish_trajectory_setpoint();
            }
            
            manage_mission_flow();
        });
    }

private:
    enum class Phase {
        warmup,      // 1. 예열
        takeoff,     // 2. 이륙
        turn,        // 3. 제자리 회전 (다음 WP 방향)
        wait,        // 4. 회전 후 대기 (1초)
        move,        // 5. 이동
        landing      // 6. 착륙
    };

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_subscription_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_subscription_;

    Phase phase_ = Phase::warmup;
    uint64_t ticks_ = 0;
    
    // 상태 데이터
    std::array<float, 3> current_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> setpoint_pos_{0.0f, 0.0f, 0.0f}; // 이동 목표점(미끼)
    
    float current_yaw_sp_ = -3.14f; // 현재 Yaw Setpoint (초기값 북쪽)
    float target_yaw_ = -3.14f;     // 목표 Yaw

    uint8_t arming_state_ = 0;
    
    // 설정값
    const float flight_alt_ = 50.0f;
    const float flight_speed_ = 5.0f;      // 5m/s 이동 속도
    const float yaw_speed_ = 0.5f;         // 0.5 rad/s (약 30도/초) - 천천히 회전
    const float dt_ = 0.1f;                // 100ms 주기
    const float acceptance_radius_ = 1.0f; // WP 도달 인정 반경

    // 사각형 웨이포인트 (북 -> 동 -> 남 -> 원점)
    std::vector<std::array<float, 2>> waypoints_ = {
        {100.0f, 0.0f},   // WP 1: 북
        {100.0f, 100.0f}, // WP 2: 동
        {0.0f, 100.0f},   // WP 3: 남
        {0.0f, 0.0f}      // WP 4: 원점
    };
    size_t wp_index_ = 0;

    void publish_offboard_control_mode() {
        OffboardControlMode msg{};
        msg.position = true;
        msg.velocity = false;
        msg.acceleration = false;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        offboard_control_mode_publisher_->publish(msg);
    }

    void publish_trajectory_setpoint() {
        TrajectorySetpoint msg{};
        
        // 이륙/예열 중엔 (0,0)에서 고도만 유지
        if (phase_ == Phase::takeoff || phase_ == Phase::warmup) {
            msg.position = {0.0f, 0.0f, -flight_alt_};
            msg.yaw = -3.14f; // 초기 북쪽 고정
            current_yaw_sp_ = -3.14f;
        } 
        else {
            // 미션 중 (회전, 대기, 이동)
            msg.position = {setpoint_pos_[0], setpoint_pos_[1], -flight_alt_};
            msg.yaw = current_yaw_sp_; // 부드럽게 변하는 Yaw 전송
        }

        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
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
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        vehicle_command_publisher_->publish(msg);
    }

    // 부드러운 Yaw 회전 로직
    void update_yaw_smoothing() {
        float diff = wrap_angle(target_yaw_ - current_yaw_sp_);
        float step = yaw_speed_ * dt_;

        if (std::abs(diff) < step) {
            current_yaw_sp_ = target_yaw_;
        } else {
            // 짧은 방향으로 회전
            if (diff > 0) current_yaw_sp_ += step;
            else current_yaw_sp_ -= step;
        }
        current_yaw_sp_ = wrap_angle(current_yaw_sp_);
    }

    void manage_mission_flow() {
        switch (phase_) {
            case Phase::warmup:
                if (++ticks_ >= 10) {
                    RCLCPP_INFO(this->get_logger(), "Warmup Done. Arming & Takeoff...");
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
                    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
                    phase_ = Phase::takeoff;
                    ticks_ = 0;
                }
                break;

            case Phase::takeoff:
                if (++ticks_ % 10 == 0) {
                    RCLCPP_INFO(this->get_logger(), "Climbing... Alt: %.1fm", -current_pos_[2]);
                }
                // 목표 고도 도달 확인
                if (-current_pos_[2] >= (flight_alt_ - 1.0f)) {
                    RCLCPP_INFO(this->get_logger(), "Reached 50m. Planning path to WP %zu...", wp_index_ + 1);
                    
                    // 현재 위치를 시작점으로 설정
                    setpoint_pos_ = {current_pos_[0], current_pos_[1], -flight_alt_};
                    
                    // 첫 번째 목표를 향해 회전부터 시작
                    phase_ = Phase::turn;
                }
                break;

            case Phase::turn:
                {
                    // 목표 WP 방향 계산
                    std::array<float, 2> target = waypoints_[wp_index_];
                    float dx = target[0] - setpoint_pos_[0];
                    float dy = target[1] - setpoint_pos_[1];
                    target_yaw_ = std::atan2(dy, dx); // 목표 각도 설정

                    // Yaw 스무딩 업데이트
                    update_yaw_smoothing();

                    // 목표 각도와 거의 비슷해지면 대기 단계로 (오차 0.05 rad 이내)
                    if (std::abs(wrap_angle(target_yaw_ - current_yaw_sp_)) < 0.05f) {
                        RCLCPP_INFO(this->get_logger(), "Aligned. Waiting 1s...");
                        phase_ = Phase::wait;
                        ticks_ = 0;
                    }
                }
                break;

            case Phase::wait:
                // 제자리에서 Yaw 유지하며 1초 대기
                if (++ticks_ >= 10) { // 10 * 100ms = 1s
                    RCLCPP_INFO(this->get_logger(), "Moving to WP %zu...", wp_index_ + 1);
                    phase_ = Phase::move;
                }
                break;

            case Phase::move:
                {
                    std::array<float, 2> target = waypoints_[wp_index_];
                    float dx = target[0] - setpoint_pos_[0];
                    float dy = target[1] - setpoint_pos_[1];
                    float dist = std::sqrt(dx*dx + dy*dy);

                    // 미끼 이동 (Interpolation)
                    if (dist > 0.1f) {
                        float step = flight_speed_ * dt_;
                        if (step > dist) step = dist;
                        float scale = step / dist;
                        
                        setpoint_pos_[0] += dx * scale;
                        setpoint_pos_[1] += dy * scale;
                    }

                    // 실제 드론 도착 확인
                    float drone_dx = target[0] - current_pos_[0];
                    float drone_dy = target[1] - current_pos_[1];
                    float drone_dist = std::sqrt(drone_dx*drone_dx + drone_dy*drone_dy);

                    if (drone_dist < acceptance_radius_) {
                        RCLCPP_INFO(this->get_logger(), "Reached WP %zu!", wp_index_ + 1);
                        wp_index_++;

                        if (wp_index_ >= waypoints_.size()) {
                            RCLCPP_INFO(this->get_logger(), "All WPs Done. Landing...");
                            publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
                            phase_ = Phase::landing;
                        } else {
                            // 다음 WP를 위해 다시 회전 단계로
                            phase_ = Phase::turn;
                        }
                    }
                }
                break;

            case Phase::landing:
                // 상단 timer 람다 함수에서 arming_state 확인 후 종료함
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