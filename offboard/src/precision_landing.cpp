#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#include <mavros_msgs/srv/set_mode.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

using namespace std::chrono_literals;

// ---------------------------------------------------------------------------
// State machine
//   TAKEOFF  : set GUIDED, arm, ramp altitude up to flight_alt (skipped if
//              auto_takeoff=false — then start in IDLE and wait for a human)
//   IDLE     : wait for armed + GUIDED + marker visible
//   ALIGN    : hover at flight_alt, slide laterally until marker is centred
//   DESCEND  : keep marker centred while descending; pause if marker lost
//   DONE     : switch FCU to LAND mode
// ---------------------------------------------------------------------------
enum class Stage { TAKEOFF, IDLE, ALIGN, DESCEND, DONE };

class PrecisionLandingNode : public rclcpp::Node {
public:
    PrecisionLandingNode() : Node("precision_landing_node") {
        flight_alt_   = declare_parameter("flight_alt",   5.0f);  // m, hover altitude
        align_thresh_ = declare_parameter("align_thresh", 0.08f); // normalised, ~5% of FOV
        land_alt_     = declare_parameter("land_alt",     0.7f);  // m, trigger LAND mode
        descend_rate_ = declare_parameter("descend_rate", 0.3f);  // m/s downward
        gain_lateral_ = declare_parameter("gain_lateral", 2.5f);  // m of correction per unit offset
        max_step_     = declare_parameter("max_step",     1.5f);  // m, max lateral move per cycle
        // auto_takeoff: node sets GUIDED, arms and climbs to flight_alt itself.
        // Set false to keep the old behaviour (a human/other node flies it up).
        auto_takeoff_   = declare_parameter("auto_takeoff",   true);
        takeoff_ramp_s_ = declare_parameter("takeoff_ramp_s", 5.0f); // s to reach flight_alt

        stage_ = auto_takeoff_ ? Stage::TAKEOFF : Stage::IDLE;

        setpoint_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            "/mavros/setpoint_position/local", 10);
        debug_pub_ = create_publisher<std_msgs::msg::String>(
            "/precision_landing/debug", 10);

        state_sub_ = create_subscription<mavros_msgs::msg::State>(
            "/mavros/state", 10,
            [this](mavros_msgs::msg::State::SharedPtr m) { mav_state_ = *m; });

        // mavros publishes local_position/pose with BEST_EFFORT (sensor) QoS;
        // subscribe with SensorDataQoS or no messages are delivered.
        pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/mavros/local_position/pose", rclcpp::SensorDataQoS(),
            [this](geometry_msgs::msg::PoseStamped::SharedPtr m) {
                pos_[0] = static_cast<float>(m->pose.position.x);
                pos_[1] = static_cast<float>(m->pose.position.y);
                pos_[2] = static_cast<float>(m->pose.position.z);
            });

        offset_sub_ = create_subscription<geometry_msgs::msg::Point>(
            "/perception/aruco_offset", 10,
            [this](geometry_msgs::msg::Point::SharedPtr m) { offset_ = *m; });

        detected_sub_ = create_subscription<std_msgs::msg::Bool>(
            "/perception/aruco_detected", 10,
            [this](std_msgs::msg::Bool::SharedPtr m) {
                if (m->data) fresh_ = 10; // 10 × 50 ms = 500 ms freshness window
            });

        set_mode_cli_ = create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");
        arming_cli_   = create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");

        sp_[0] = sp_[1] = 0.0f;
        sp_[2] = flight_alt_;

        timer_ = create_wall_timer(50ms, [this]() { tick(); });
    }

private:
    // -----------------------------------------------------------------------
    void tick() {
        if (fresh_ > 0) --fresh_;
        const bool marker_ok = (fresh_ > 0);

        if (!mav_state_.connected) return;

        switch (stage_) {
            // ----------------------------------------------------------------
            // Climb to flight_alt the same way offboard_control does: request
            // GUIDED + arm (re-sent every ~2 s until they take), then stream a
            // rising altitude setpoint. ArduCopter GUIDED takes off by following
            // the climbing setpoint, so we never leave the position stream idle.
            case Stage::TAKEOFF: {
                if (++req_ticks_ % 40 == 1) {           // ~ every 2 s (40 × 50 ms)
                    if (mav_state_.mode != "GUIDED") set_mode("GUIDED");
                    else if (!mav_state_.armed)      arm(true);
                }

                sp_[0] = pos_[0];
                sp_[1] = pos_[1];

                if (mav_state_.armed && mav_state_.mode == "GUIDED") {
                    takeoff_elapsed_ += 0.05f;          // 50 ms dt
                    const float s = std::min(1.0f, takeoff_elapsed_ / takeoff_ramp_s_);
                    sp_[2] = (1.0f - s) * 0.0f + s * flight_alt_;
                    if (std::abs(pos_[2] - flight_alt_) <= 1.0f) {
                        log("→ IDLE (reached flight_alt)"); stage_ = Stage::IDLE;
                    }
                } else {
                    sp_[2] = flight_alt_;               // pre-positioned target
                }
                break;
            }

            // ----------------------------------------------------------------
            case Stage::IDLE:
                sp_ = {pos_[0], pos_[1], flight_alt_};
                if (mav_state_.armed && mav_state_.mode == "GUIDED" && marker_ok) {
                    log("→ ALIGN"); stage_ = Stage::ALIGN;
                }
                break;

            // ----------------------------------------------------------------
            case Stage::ALIGN:
                if (!mav_state_.armed) { log("disarmed → IDLE"); stage_ = Stage::IDLE; break; }
                sp_[2] = flight_alt_;
                if (marker_ok) {
                    correct_lateral();
                    const float err = std::hypot(offset_.x, offset_.y);
                    if (err < align_thresh_) {
                        log("→ DESCEND"); stage_ = Stage::DESCEND;
                    }
                }
                break;

            // ----------------------------------------------------------------
            case Stage::DESCEND:
                if (!mav_state_.armed) { log("disarmed → IDLE"); stage_ = Stage::IDLE; break; }
                if (marker_ok) {
                    correct_lateral();
                    // Descend only while marker is visible — freeze altitude if lost
                    sp_[2] -= descend_rate_ * 0.05f;            // 50 ms dt
                    sp_[2]  = std::max(sp_[2], 0.0f);
                }
                if (pos_[2] < land_alt_) {
                    log("→ DONE (LAND)"); stage_ = Stage::DONE;
                }
                break;

            // ----------------------------------------------------------------
            case Stage::DONE:
                if (mav_state_.mode != "LAND") send_land();
                return; // stop sending position setpoints
        }

        geometry_msgs::msg::PoseStamped sp;
        sp.header.stamp    = now();
        sp.header.frame_id = "map";
        sp.pose.position.x = sp_[0];
        sp.pose.position.y = sp_[1];
        sp.pose.position.z = sp_[2];
        sp.pose.orientation.w = 1.0;
        setpoint_pub_->publish(sp);
    }

    // -----------------------------------------------------------------------
    // Move the lateral setpoint toward the marker.
    //
    // Coordinate convention (downward camera, drone facing +X local):
    //   image right (+offset.x) → world +Y   →  sp_[1] += dx
    //   image down  (+offset.y) → world –X   →  sp_[0] -= dy
    //
    // If your camera is mounted differently, flip the signs via gain_lateral_
    // or swap x/y here.
    // -----------------------------------------------------------------------
    void correct_lateral() {
        const float dx = std::clamp(
            static_cast<float>(offset_.x) * gain_lateral_, -max_step_, max_step_);
        const float dy = std::clamp(
            static_cast<float>(offset_.y) * gain_lateral_, -max_step_, max_step_);
        sp_[1] = pos_[1] + dx;   // image +x → local +y (right)
        sp_[0] = pos_[0] - dy;   // image +y → local −x (backward → drone moves back to centre)
    }

    // -----------------------------------------------------------------------
    void send_land() { set_mode("LAND"); }

    void set_mode(const std::string& mode) {
        auto req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        req->custom_mode = mode;
        set_mode_cli_->async_send_request(req);
    }

    void arm(bool value) {
        auto req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
        req->value = value;
        arming_cli_->async_send_request(req);
    }

    void log(const std::string& s) {
        RCLCPP_INFO(get_logger(), "%s", s.c_str());
        std_msgs::msg::String m; m.data = s;
        debug_pub_->publish(m);
    }

    // Parameters
    float flight_alt_, align_thresh_, land_alt_, descend_rate_, gain_lateral_, max_step_;
    bool  auto_takeoff_{true};
    float takeoff_ramp_s_{5.0f};

    // State
    Stage  stage_;
    mavros_msgs::msg::State mav_state_;
    std::array<float, 3>    pos_{0, 0, 0};
    std::array<float, 3>    sp_{0, 0, 5};
    geometry_msgs::msg::Point offset_;
    int   fresh_{0};
    int   req_ticks_{0};
    float takeoff_elapsed_{0.0f};

    // ROS interfaces
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr setpoint_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr           debug_pub_;
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr      state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr    offset_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr          detected_sub_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr          set_mode_cli_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr      arming_cli_;
    rclcpp::TimerBase::SharedPtr                                  timer_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PrecisionLandingNode>());
    rclcpp::shutdown();
    return 0;
}
