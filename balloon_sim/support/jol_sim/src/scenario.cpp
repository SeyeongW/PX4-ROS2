// Copyright 2026 xogus. SPDX-License-Identifier: Apache-2.0
// 기존 이동 표적 또는 새 정적 표적의 평가 전용 정답을 별도 발행한다.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <gazebo/sensors/sensors.hh>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include "trajectory.hpp"

namespace jol_sim {
class Scenario final : public gazebo::WorldPlugin {
 public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf) override {
    world_ = std::move(world);
    node_ = gazebo_ros::Node::Get(sdf);
    scenario_ = sdf->Get<std::string>("scenario", "multi").first;
    trajectory_ = sdf->Get<std::string>("trajectory", "nonlinear").first;
    radius_ = sdf->Get<double>("radius", 50.0).first;
    speed_ = sdf->Get<double>("speed", 10.0).first;
    altitude_ = sdf->Get<double>("altitude", 10.0).first;
    empty_seconds_ = sdf->Get<double>("empty_seconds", 8.0).first;
    fixed_start_seconds_ = sdf->Get<double>("scenario_start_seconds", -1.0).first;
    wait_background_ = sdf->Get<bool>("wait_for_background", true).first;
    if (!std::isfinite(radius_) || !std::isfinite(speed_) || !std::isfinite(altitude_) ||
        !std::isfinite(empty_seconds_) || radius_ <= 1.0 || speed_ < 0 || altitude_ <= 0 || empty_seconds_ < 0 ||
        !std::isfinite(fixed_start_seconds_) || (fixed_start_seconds_ != -1 && fixed_start_seconds_ < 0) ||
        (trajectory_ != "nonlinear" && trajectory_ != "circle")) {
      gzerr << "Invalid JOL orbit settings; scenario disabled\n";
      return;
    }
    path_ = FlightPath(radius_, speed_, altitude_);
    truth_ = node_->create_publisher<std_msgs::msg::String>("/jol/ground_truth", 10);
    background_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
      "/jol/background_ready", rclcpp::QoS(1).reliable().transient_local(),
      [this](const std_msgs::msg::Bool::SharedPtr msg) { background_ready_ = msg->data; });
    update_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&Scenario::Update, this));
    if (scenario_ == "static_flight") {
      gzmsg << "JOL scenario=static_flight authored static target; ground truth is evaluation only\n";
    } else {
      gzmsg << "JOL scenario=" << scenario_ << " trajectory=" << trajectory_ << " radius=" << radius_
            << " speed=" << speed_ << " altitude=" << altitude_ << "\n";
    }
  }

 private:
  static bool WallBlocks(const ignition::math::Vector3d &origin,
                         const ignition::math::Vector3d &target) {
    const ignition::math::Vector3d low(24.5, -2.0, 0), high(25.5, 2.0, 18.0);
    const auto direction = target - origin;
    double enter = 0.0, leave = 1.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (std::abs(direction[axis]) < 1e-12) {
        if (origin[axis] < low[axis] || origin[axis] > high[axis]) return false;
      } else {
        const double a = (low[axis] - origin[axis]) / direction[axis];
        const double b = (high[axis] - origin[axis]) / direction[axis];
        enter = std::max(enter, std::min(a, b));
        leave = std::min(leave, std::max(a, b));
        if (enter > leave) return false;
      }
    }
    return enter < 1.0 && leave > 0.0;
  }

  void Update() {
    const double now = world_->SimTime().Double();
    // 평가 기준 원점도 실제 충돌 광선 센서의 링크 자세를 사용한다.
    auto turret = world_->ModelByName("fixed_turret");
    auto head = turret ? turret->GetLink("lidar_head") : gazebo::physics::LinkPtr();
    if (!head && turret && scenario_ == "static_flight") head = turret->GetLink("gimbal_pitch");
    if (!head) return;
    const auto lidar_pose = head->WorldPose();
    const auto lidar_origin = lidar_pose.Pos();
    if (scenario_ == "static_flight") {
      // 정적 표적은 SDF에서 고정되어 있다. 위치 갱신이나 비행 제어 입력을 만들지 않는다.
      if (!models_[0]) models_[0] = world_->ModelByName("static_target");
      if (!models_[0]) return;
      if (now < last_publish_) last_publish_ = -1.;
      if (now - last_publish_ < .049) return;
      const auto p = models_[0]->WorldPose().Pos();
      std::ostringstream data;
      data << std::fixed << std::setprecision(6)
           << "{\"stamp\":" << now << ",\"frame_id\":\"world\",\"scenario\":\"static_flight\","
           << "\"phase\":\"static\",\"evaluation_only\":true,\"sensor_origin\":["
           << lidar_origin.X() << ',' << lidar_origin.Y() << ',' << lidar_origin.Z()
           << "],\"target_horizontal_range_m\":" << std::hypot(p.X(), p.Y())
           << ",\"target_azimuth_deg\":" << std::atan2(p.Y(), p.X()) * 180.0 / 3.14159265358979323846
           << ",\"objects\":[{\"id\":1,\"position\":[" << p.X() << ',' << p.Y() << ',' << p.Z()
           << "],\"size\":[0.3,0.3,0.3],\"velocity\":[0,0,0],\"active\":true,"
           << "\"stopped\":true,\"occluded\":false,\"trajectory_id\":\"static_object_1\"}]}";
      std_msgs::msg::String message; message.data = data.str(); truth_->publish(message);
      last_publish_ = now;
      return;
    }
    if (now < last_update_) {
      last_publish_ = -1.0; release_time_ = -1.0; background_ready_at_release_ = false;
    }
    last_update_ = now;
    // 반복 실험은 배경 준비가 늦어져도 지정한 시뮬레이션 시각을 유지한다.
    if (release_time_ < 0 && (fixed_start_seconds_ >= 0 ? now >= fixed_start_seconds_ :
        now >= empty_seconds_ && (!wait_background_ || background_ready_))) {
      release_time_ = fixed_start_seconds_ >= 0 ? fixed_start_seconds_ : now;
      background_ready_at_release_ = background_ready_.load();
    }
    const double elapsed = release_time_ < 0 ? -1.0 : now - release_time_;
    const bool multi = scenario_ == "multi" || scenario_ == "all";
    const bool wall = scenario_ == "occlusion" || scenario_ == "all";
    const bool stop = scenario_ == "stop" || scenario_ == "all";
    const bool reappear = scenario_ == "reappearance" || scenario_ == "all";
    std::string phase = elapsed < 0 ? "background" :
      (multi && (scenario_ == "multi" || elapsed >= 3) ? "multi" : "single");
    if (stop && elapsed >= 8 && elapsed < 13) phase = "stop";
    if (reappear && elapsed >= 18 && elapsed < 22) phase = "absent";
    if (reappear && elapsed >= 22 && elapsed < 25) phase = "reappearance";
    std::ostringstream data;
    data << std::fixed << std::setprecision(5)
         << "{\"stamp\":" << now << ",\"frame_id\":\"world\",\"scenario\":\""
         << scenario_ << "\",\"phase\":\"" << phase
         << "\",\"elapsed\":" << elapsed
         << ",\"trajectory\":\"" << trajectory_ << "\",\"trajectory_version\":1"
         << ",\"trajectory_time_basis\":\"" << (fixed_start_seconds_ >= 0 ?
            "motion_elapsed_since_fixed_sim_start" : "motion_elapsed_since_background_release") << '"'
         << ",\"scenario_start_seconds\":" << fixed_start_seconds_
         << ",\"release_time\":" << release_time_
         << ",\"background_ready_at_release\":" << (background_ready_at_release_ ? "true" : "false")
         << ",\"nominal_speed_mps\":" << speed_
         << ",\"lap_length_m\":" << (trajectory_ == "nonlinear" ? path_.Length() : 2 * 3.14159265358979323846 * radius_)
         << ",\"lidar_origin\":[" << lidar_origin.X() << ',' << lidar_origin.Y() << ',' << lidar_origin.Z()
         << "],\"lidar_pose_source\":\"collision_scanner_head_identity\""
         << ",\"lidar_sensor_resolved\":true,\"lidar_sensor_model\":\"jol_mid360\",\"objects\":[";
    for (std::size_t i = 0; i < models_.size(); ++i) {
      if (!models_[i]) models_[i] = world_->ModelByName("drone_" + std::to_string(i + 1));
      if (!models_[i]) continue;
      bool active = scenario_ != "empty" && elapsed >= 0 && (i == 0 || multi);
      if (scenario_ == "all" && i > 0 && elapsed < 3) active = false;
      if (reappear && i == 0 && elapsed >= 18.0 && elapsed < 22.0) active = false;
      const bool stopped = scenario_ == "stationary" ||
        (stop && i == 0 && elapsed >= 8.0 && elapsed < 13.0);
      const double travel = scenario_ == "stationary" ? 0.0 : std::max(0.0, elapsed) -
        (stop && i == 0 ? std::clamp(elapsed - 8.0, 0.0, 5.0) : 0.0);
      auto motion = path_.At(i, travel, trajectory_ == "nonlinear");
      const auto& p = motion.position;
      if (!active || stopped) motion.velocity = {0.0, 0.0, 0.0};
      // 비활성 드론은 지하로 이동시켜 실제 센서 시야에서 제외한다.
      models_[i]->SetWorldPose(ignition::math::Pose3d(p[0], p[1], active ? p[2] : -100.0 - i,
        0, 0, motion.heading));
      models_[i]->SetLinearVel(ignition::math::Vector3d::Zero);
      models_[i]->SetAngularVel(ignition::math::Vector3d::Zero);
      // 중심만 가려진 경우와 구분하여 충돌 경계의 8개 꼭짓점 가림을 검사한다.
      unsigned blocked_corners = 0;
      const auto bounds = models_[i]->BoundingBox();
      if (active && wall) {
        for (int corner = 0; corner < 8; ++corner) {
          ignition::math::Vector3d target;
          for (std::size_t axis = 0; axis < 3; ++axis)
            target[axis] = corner & (1 << axis) ? bounds.Max()[axis] : bounds.Min()[axis];
          if (WallBlocks(lidar_origin, target)) ++blocked_corners;
        }
      }
      const bool occluded = blocked_corners == 8;
      if (i) data << ',';
      data << "{\"id\":" << i + 1 << ",\"position\":[" << p[0] << ',' << p[1] << ',' << p[2]
           << "],\"velocity\":[" << motion.velocity[0] << ',' << motion.velocity[1] << ',' << motion.velocity[2]
           << "],\"trajectory_id\":\"" << trajectory_ << "_" << i + 1
           << "\",\"motion_elapsed\":" << travel << ",\"path_parameter\":" << motion.parameter
           << ",\"active\":" << (active ? "true" : "false")
           << ",\"occluded\":" << (occluded ? "true" : "false")
           << ",\"partial_occlusion\":" << (blocked_corners > 0 && blocked_corners < 8 ? "true" : "false")
           << ",\"occlusion_corner_fraction\":" << blocked_corners / 8.0
           << ",\"stopped\":" << (stopped ? "true" : "false") << '}';
    }
    data << "]}";
    if (now - last_publish_ >= 0.049) {
      std_msgs::msg::String message;
      message.data = data.str();
      truth_->publish(message);
      last_publish_ = now;
    }
  }
  gazebo::physics::WorldPtr world_;
  std::array<gazebo::physics::ModelPtr, 3> models_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr truth_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr background_sub_;
  gazebo::event::ConnectionPtr update_;
  std::string scenario_, trajectory_;
  FlightPath path_;
  double radius_{50}, speed_{10}, altitude_{10}, empty_seconds_{8}, last_publish_{-1};
  double release_time_{-1}, last_update_{-1};
  double fixed_start_seconds_{-1};
  bool background_ready_at_release_{false};
  bool wait_background_{true};
  std::atomic<bool> background_ready_{false};
};
GZ_REGISTER_WORLD_PLUGIN(Scenario)
}  // namespace jol_sim
