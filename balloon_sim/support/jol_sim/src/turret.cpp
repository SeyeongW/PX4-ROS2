// Copyright 2026 xogus. SPDX-License-Identifier: Apache-2.0
// 기존 포탑 또는 별도 고정 지지대의 카메라 짐벌을 움직이고 측정 TF를 발행한다.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

namespace jol_sim {
class Turret final : public gazebo::ModelPlugin {
 public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = std::move(model);
    node_ = gazebo_ros::Node::Get(sdf);
    const auto camera_mount = sdf->Get<std::string>("camera_mount", "turret").first;
    gimbal_ = camera_mount == "gimbal";
    if (camera_mount != "arm" && camera_mount != "turret" && !gimbal_)
      throw std::runtime_error("camera_mount must be arm, turret or gimbal");
    yaw_name_ = gimbal_ ? "gimbal_yaw_joint" : "yaw_joint";
    pitch_name_ = gimbal_ ? "gimbal_pitch_joint" : "pitch_joint";
    yaw_frame_ = gimbal_ ? "jol/gimbal_yaw" : "jol/turret_yaw";
    pitch_frame_ = gimbal_ ? "jol/gimbal_pitch" : "jol/turret_pitch";
    yaw_ = model_->GetJoint(yaw_name_);
    pitch_ = model_->GetJoint(pitch_name_);
    base_ = model_->GetLink("base");
    yaw_link_ = model_->GetLink(gimbal_ ? "gimbal_yaw" : "yaw");
    pitch_link_ = model_->GetLink(gimbal_ ? "gimbal_pitch" : "pitch");
    lidar_link_ = model_->GetLink("lidar_head");
    lidar_joint_ = model_->GetJoint("lidar_mount_joint");
    const auto lidar_mode = sdf->Get<std::string>("lidar_mount_mode", "payload").first;
    if (lidar_mode != "payload" && lidar_mode != "level")
      throw std::runtime_error("lidar_mount_mode must be payload or level");
    level_lidar_ = lidar_mode == "level";
    if (!yaw_ || !pitch_ || !base_ || !yaw_link_ || !pitch_link_ || (!gimbal_ && !lidar_link_) ||
        (level_lidar_ && !lidar_joint_))
      throw std::runtime_error("Turret is missing its base/yaw/pitch/LiDAR links or joints");
    automatic_ = sdf->Get<bool>("motion", true).first;
    survey_ = sdf->Get<bool>("survey", false).first;
    // 고정 표적 모드에는 지지대의 순찰/서보 구동을 적용하지 않는다.
    if (gimbal_) { automatic_ = false; survey_ = false; }
    commanded_pitch_ = sdf->Get<double>("initial_pitch", 0.).first;
    if (!std::isfinite(commanded_pitch_) || std::abs(commanded_pitch_) > M_PI / 6)
      throw std::runtime_error("Invalid initial gimbal pitch");
    survey_elevation_ = sdf->Get<double>("survey_elevation", M_PI / 9).first;
    yaw_amplitude_ = sdf->Get<double>("yaw_amplitude", M_PI / 4).first;
    pitch_amplitude_ = sdf->Get<double>("pitch_amplitude", M_PI / 18).first;
    period_ = sdf->Get<double>("period", 8.0).first;
    if (!std::isfinite(period_) || period_ < 1 || !std::isfinite(yaw_amplitude_) ||
        !std::isfinite(pitch_amplitude_) || yaw_amplitude_ < 0 || yaw_amplitude_ > M_PI ||
        pitch_amplitude_ < 0 || pitch_amplitude_ > M_PI / 6 ||
        !std::isfinite(survey_elevation_) || std::abs(survey_elevation_) > M_PI / 6 ||
        (survey_ && 2 * M_PI / period_ > 1.2))
      throw std::runtime_error("Invalid turret sweep limits or period");
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
    static_tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(node_);
    states_ = node_->create_publisher<sensor_msgs::msg::JointState>("/jol/turret/joint_states", 10);
    if (gimbal_)
      gimbal_states_ = node_->create_publisher<sensor_msgs::msg::JointState>("/jol/gimbal/joint_states", 10);
    const auto command_callback = [this](sensor_msgs::msg::JointState::SharedPtr msg) {
        if (msg->name.size() != msg->position.size() || msg->name.empty()) {
          RCLCPP_WARN(node_->get_logger(), "Turret command requires named finite joint positions");
          return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        double yaw = commanded_yaw_, pitch = commanded_pitch_;
        for (std::size_t i = 0; i < msg->name.size(); ++i) {
          if (!std::isfinite(msg->position[i]) ||
              (msg->name[i] != yaw_name_ && msg->name[i] != pitch_name_ &&
               msg->name[i] != "yaw_joint" && msg->name[i] != "pitch_joint")) {
            RCLCPP_WARN(node_->get_logger(), "Rejected invalid turret joint name or position");
            return;
          }
          if (msg->name[i] == yaw_name_ || msg->name[i] == "yaw_joint") yaw = std::remainder(msg->position[i], 2 * M_PI);
          else pitch = std::clamp(msg->position[i], -M_PI / 6, M_PI / 6);
        }
        commanded_yaw_ = yaw;
        commanded_pitch_ = pitch;
        automatic_ = false;
      };
    command_ = node_->create_subscription<sensor_msgs::msg::JointState>("/jol/turret/command", 10, command_callback);
    if (gimbal_)
      gimbal_command_ = node_->create_subscription<sensor_msgs::msg::JointState>("/jol/gimbal/command", 10, command_callback);
    std::vector<geometry_msgs::msg::TransformStamped> fixed_transforms{
      Transform("world", "jol/base", base_->WorldPose())};
    if (camera_mount == "turret" || gimbal_) {
      const auto pose = sdf->Get<ignition::math::Pose3d>("camera_pose", {0.14, -0.145, 1.0125, 0, 0, 0}).first;
      fixed_transforms.push_back(Transform(pitch_frame_, "jol/camera_link", pose));
      fixed_transforms.push_back(Transform("jol/camera_link", "jol/camera_optical", {0, 0, 0, -M_PI / 2, 0, -M_PI / 2}));
    }
    if (lidar_link_ && !level_lidar_)
      fixed_transforms.push_back(Transform(gimbal_ ? "jol/base" : pitch_frame_, "jol/lidar",
        (gimbal_ ? base_->WorldPose() : pitch_link_->WorldPose()).Inverse() * lidar_link_->WorldPose()));
    if (gimbal_) {
      // 읽기 전용 TF 별칭. 해당 포탑 관절은 물리 모델에 존재하지 않는다.
      fixed_transforms.push_back(Transform(yaw_frame_, "jol/turret_yaw", ignition::math::Pose3d::Zero));
      fixed_transforms.push_back(Transform(pitch_frame_, "jol/turret_pitch", ignition::math::Pose3d::Zero));
    }
    static_tf_->sendTransform(fixed_transforms);
    move_ = gazebo::event::Events::ConnectWorldUpdateBegin([this](const gazebo::common::UpdateInfo &) { Move(); });
    publish_ = gazebo::event::Events::ConnectWorldUpdateEnd([this]() { Publish(); });
    Publish();
    gzmsg << (gimbal_ ? "JOL fixed pedestal: camera-only gimbal; fixed LiDAR; " : "JOL turret: kinematic yaw/pitch; ")
          << "LiDAR mount=" << lidar_mode
          << "; measured TF and joints at simulation timestamps\n";
  }

 private:
  geometry_msgs::msg::TransformStamped Transform(
      const std::string &parent, const std::string &child, const ignition::math::Pose3d &pose) const {
    geometry_msgs::msg::TransformStamped out;
    const auto time = model_->GetWorld()->SimTime();
    out.header.stamp.sec = time.sec;
    out.header.stamp.nanosec = time.nsec;
    out.header.frame_id = parent;
    out.child_frame_id = child;
    out.transform.translation.x = pose.Pos().X();
    out.transform.translation.y = pose.Pos().Y();
    out.transform.translation.z = pose.Pos().Z();
    out.transform.rotation.x = pose.Rot().X();
    out.transform.rotation.y = pose.Rot().Y();
    out.transform.rotation.z = pose.Rot().Z();
    out.transform.rotation.w = pose.Rot().W();
    return out;
  }

  void Move() {
    std::lock_guard<std::mutex> lock(mutex_);
    const double now = model_->GetWorld()->SimTime().Double();
    const double dt = std::clamp(now - last_move_time_, 0.0, 0.02);
    last_move_time_ = now;
    if (automatic_) {
      const double phase = 2 * M_PI * now / period_;
      commanded_yaw_ = survey_ ? std::remainder(phase, 2 * M_PI) : yaw_amplitude_ * std::sin(phase);
      // 센서 전방은 링크 +X이며 음의 pitch가 위쪽을 향한다.
      commanded_pitch_ = survey_ ? -survey_elevation_ : pitch_amplitude_ * std::sin(phase);
    }
    // ponytail: 운동학 검증용이다. 지연·백래시 검증에는 모터 동역학이 필요하다.
    const double yaw = yaw_->Position(0), pitch = pitch_->Position(0);
    const double yaw_error = std::remainder(commanded_yaw_ - yaw, 2 * M_PI);
    const bool legacy_sweep = automatic_ && !survey_;
    yaw_->SetPosition(0, yaw + (legacy_sweep ? yaw_error : std::clamp(yaw_error, -1.2 * dt, 1.2 * dt)), true);
    pitch_->SetPosition(0, legacy_sweep ? commanded_pitch_ :
      pitch + std::clamp(commanded_pitch_ - pitch, -0.8 * dt, 0.8 * dt), true);
    // level 장착은 센서 몸체를 반대로 회전시켜 실제 수평을 유지한다.
    if (level_lidar_) lidar_joint_->SetPosition(0, -pitch_->Position(0), true);
  }

  void Publish() {
    const double now = model_->GetWorld()->SimTime().Double();
    // 물리는 1 kHz, 측정 TF·관절 상태는 적체를 줄이기 위해 100 Hz로 발행한다.
    if (now >= last_time_ && now - last_time_ < 0.01 - 1e-9) return;
    std::vector<geometry_msgs::msg::TransformStamped> transforms{
      Transform("jol/base", yaw_frame_, base_->WorldPose().Inverse() * yaw_link_->WorldPose()),
      Transform(yaw_frame_, pitch_frame_, yaw_link_->WorldPose().Inverse() * pitch_link_->WorldPose())};
    if (level_lidar_)
      transforms.push_back(Transform("jol/turret_pitch", "jol/lidar",
        pitch_link_->WorldPose().Inverse() * lidar_link_->WorldPose()));
    tf_->sendTransform(transforms);
    sensor_msgs::msg::JointState state;
    const auto time = model_->GetWorld()->SimTime();
    state.header.stamp.sec = time.sec;
    state.header.stamp.nanosec = time.nsec;
    state.header.frame_id = "jol/base";
    state.name = {yaw_name_, pitch_name_};
    state.position = {yaw_->Position(0), pitch_->Position(0)};
    if (now > last_time_ && last_time_ >= 0)
      state.velocity = {std::remainder(state.position[0] - last_yaw_, 2 * M_PI) / (now - last_time_),
                        (state.position[1] - last_pitch_) / (now - last_time_)};
    if (gimbal_states_) gimbal_states_->publish(state);
    // 기존 UI/기록 도구의 토픽 계약만 보존한다. 새 모드에서는 카메라 짐벌 각도이다.
    state.name = {"yaw_joint", "pitch_joint"};
    states_->publish(state);
    last_time_ = now;
    last_yaw_ = state.position[0];
    last_pitch_ = state.position[1];
  }

  gazebo::physics::ModelPtr model_;
  gazebo::physics::JointPtr yaw_, pitch_, lidar_joint_;
  gazebo::physics::LinkPtr base_, yaw_link_, pitch_link_, lidar_link_;
  gazebo_ros::Node::SharedPtr node_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr states_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr gimbal_states_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr command_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr gimbal_command_;
  gazebo::event::ConnectionPtr move_, publish_;
  std::mutex mutex_;
  bool automatic_{true};
  bool survey_{false};
  bool level_lidar_{false};
  bool gimbal_{false};
  std::string yaw_name_, pitch_name_, yaw_frame_, pitch_frame_;
  double yaw_amplitude_{M_PI / 4}, pitch_amplitude_{M_PI / 18}, period_{8};
  double survey_elevation_{M_PI / 9};
  double commanded_yaw_{0}, commanded_pitch_{0};
  double last_time_{-1}, last_yaw_{0}, last_pitch_{0};
  double last_move_time_{0};
};
GZ_REGISTER_MODEL_PLUGIN(Turret)
}  // namespace jol_sim
