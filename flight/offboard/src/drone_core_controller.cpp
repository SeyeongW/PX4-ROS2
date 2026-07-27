// ---------------------------------------------------------------------------
// Drone core controller (ArduPilot / MAVROS, GUIDED mode)
//
// C++ port of the reference Python DroneCoreController. Instead of streaming
// position setpoints (as offboard_control / precision_landing do), this node
// drives takeoff/land through the CommandTOL services, which is the simplest
// way to fly an ArduCopter in GUIDED:
//
//   connect → set GUIDED → arm → takeoff(alt) → … mission … → land
//
// The sequence runs synchronously in main(): each service call blocks on its
// future via spin_until_future_complete, mirroring the Python original.
// ---------------------------------------------------------------------------
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#include <mavros_msgs/srv/command_tol.hpp>
#include <mavros_msgs/srv/set_mode.hpp>

using namespace std::chrono_literals;

class DroneCoreController : public rclcpp::Node {
public:
    DroneCoreController() : Node("drone_core_controller") {
        // State subscriber
        state_sub_ = create_subscription<mavros_msgs::msg::State>(
            "/mavros/state", 10,
            [this](mavros_msgs::msg::State::SharedPtr msg) { current_state_ = *msg; });

        // Service clients
        mode_cli_    = create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");
        arm_cli_     = create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
        takeoff_cli_ = create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/takeoff");
        land_cli_    = create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/land");
    }

    const mavros_msgs::msg::State& state() const { return current_state_; }

    // Block until the FCU reports a connection.
    void wait_for_connection() {
        RCLCPP_INFO(get_logger(), "Waiting for FCU connection...");
        while (rclcpp::ok() && !current_state_.connected) {
            rclcpp::spin_some(get_node_base_interface());
            std::this_thread::sleep_for(100ms);
        }
        RCLCPP_INFO(get_logger(), "FCU connected.");
    }

    bool set_mode(const std::string& mode) {
        auto req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        req->custom_mode = mode;
        auto res = call(mode_cli_, req, "set_mode");
        return res && res->mode_sent;
    }

    bool set_arm(bool value) {
        auto req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
        req->value = value;
        auto res = call(arm_cli_, req, "arming");
        return res && res->success;
    }

    bool takeoff(float alt) {
        auto req = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
        req->altitude = alt;
        auto res = call(takeoff_cli_, req, "takeoff");
        return res && res->success;
    }

    bool land() {
        auto req = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
        auto res = call(land_cli_, req, "land");
        return res && res->success;
    }

    // Spin until the FCU reports `armed == want`.
    void wait_for_armed(bool want) {
        while (rclcpp::ok() && current_state_.armed != want) {
            rclcpp::spin_some(get_node_base_interface());
            std::this_thread::sleep_for(100ms);
        }
    }

private:
    // Generic blocking service call: wait for the server, send, spin on the future.
    template <typename Client, typename Request>
    auto call(const Client& cli, const Request& req, const char* name)
        -> decltype(cli->async_send_request(req).get()) {
        if (!cli->wait_for_service(5s)) {
            RCLCPP_ERROR(get_logger(), "Service '%s' unavailable", name);
            return nullptr;
        }
        auto future = cli->async_send_request(req);
        if (rclcpp::spin_until_future_complete(get_node_base_interface(), future) !=
            rclcpp::FutureReturnCode::SUCCESS) {
            RCLCPP_ERROR(get_logger(), "Service '%s' call failed", name);
            return nullptr;
        }
        return future.get();
    }

    mavros_msgs::msg::State current_state_;
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr     state_sub_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr         mode_cli_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr     arm_cli_;
    rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr      takeoff_cli_;
    rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr      land_cli_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto drone = std::make_shared<DroneCoreController>();

    // Wait for FCU connection
    drone->wait_for_connection();

    // Change mode to GUIDED (required for ArduPilot autonomous commands)
    if (!drone->set_mode("GUIDED")) {
        RCLCPP_ERROR(drone->get_logger(), "Failed to set GUIDED mode");
        rclcpp::shutdown();
        return 1;
    }

    // Arm vehicle
    if (!drone->set_arm(true)) {
        RCLCPP_ERROR(drone->get_logger(), "Arming failed");
        rclcpp::shutdown();
        return 1;
    }

    // Wait until armed confirmation
    drone->wait_for_armed(true);

    // Takeoff to 2.5 m
    if (drone->takeoff(2.5f)) {
        RCLCPP_INFO(drone->get_logger(), "Takeoff successful");
    }

    // Mission loop placeholder — drone->land() can be called when complete.
    rclcpp::spin(drone);

    rclcpp::shutdown();
    return 0;
}
