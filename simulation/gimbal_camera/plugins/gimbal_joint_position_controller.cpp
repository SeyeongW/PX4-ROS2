// Gazebo Classic model plugin: PID joint position control from a ROS 2 topic.
//
// Why this file exists.  The gz-sim model drives each axis with
// `gz-sim-joint-position-controller-system`, and gazebo_ros_pkgs ships no
// equivalent: `libgazebo_ros_joint_pose_trajectory.so` SETS joint positions
// kinematically with no loop at all, so the joint stops holding an angle
// against disturbance — which is the whole reason the gz README says the
// camera "holds them rock-steady, no tremor".  Routing through
// gazebo_ros_control/ros2_control would work but moves the tuning into a
// controller yaml and drags a controller_manager in for three joints.
//
// So this reproduces gz-sim's JointPositionController control law exactly,
// which is what lets the gains carry over from models/gimbal/model.sdf
// UNCHANGED:
//
//     error = position - target
//     force = -(p*error + i*integral(error) + d*d(error)/dt)   clamped
//
// Built on plain rclcpp rather than gazebo_ros::Node so it needs only
// libgazebo-dev + rclcpp, keeping this folder as self-contained as the gz
// version (which needs no build at all).

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <thread>

#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

namespace gimbal_camera
{

class GimbalJointPositionController : public gazebo::ModelPlugin
{
public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;

    joint_name_ = sdf->Get<std::string>("joint_name", "").first;
    if (joint_name_.empty()) {
      gzerr << "[gimbal_joint_position_controller] <joint_name> is required\n";
      return;
    }
    joint_ = model->GetJoint(joint_name_);
    if (!joint_) {
      gzerr << "[gimbal_joint_position_controller] no joint '" << joint_name_
            << "' in model '" << model->GetName() << "'\n";
      return;
    }

    // Same element names and defaults as the gz-sim system, so an SDF block
    // ports across by changing only <filename> and <sub_topic> -> <topic>.
    p_gain_ = sdf->Get<double>("p_gain", 1.0).first;
    i_gain_ = sdf->Get<double>("i_gain", 0.0).first;
    d_gain_ = sdf->Get<double>("d_gain", 0.0).first;
    i_max_ = sdf->Get<double>("i_max", 0.0).first;
    i_min_ = sdf->Get<double>("i_min", 0.0).first;
    cmd_max_ = sdf->Get<double>("cmd_max", 1000.0).first;
    cmd_min_ = sdf->Get<double>("cmd_min", -1000.0).first;
    target_ = sdf->Get<double>("initial_position", 0.0).first;

    // gz addressed these as <sub_topic> under the model namespace and a bridge
    // remapped them to /gimbal/*_cmd.  Classic has no bridge, so the plugin
    // subscribes to the FINAL topic directly and the contract the keyboard
    // script and README describe is unchanged.
    const std::string topic =
      sdf->Get<std::string>("topic", "/gimbal/" + joint_name_ + "_cmd").first;

    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
    node_ = std::make_shared<rclcpp::Node>(
      "gimbal_joint_controller_" + SanitiseName(joint_name_));
    sub_ = node_->create_subscription<std_msgs::msg::Float64>(
      topic, rclcpp::QoS(10),
      [this](const std_msgs::msg::Float64::SharedPtr msg) {
        target_ = msg->data;
      });
    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);
    spin_thread_ = std::thread([this]() {executor_->spin();});

    update_conn_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&GimbalJointPositionController::OnUpdate, this));

    last_update_ = model_->GetWorld()->SimTime();
    RCLCPP_INFO(
      node_->get_logger(),
      "joint '%s' <- %s  (p=%.3f i=%.5f d=%.4f, cmd +/-%.2f N.m)",
      joint_name_.c_str(), topic.c_str(), p_gain_, i_gain_, d_gain_, cmd_max_);
  }

  ~GimbalJointPositionController() override
  {
    if (executor_) {
      executor_->cancel();
    }
    if (spin_thread_.joinable()) {
      spin_thread_.join();
    }
  }

private:
  static std::string SanitiseName(const std::string & in)
  {
    std::string out = in;
    for (auto & c : out) {
      if (!std::isalnum(static_cast<unsigned char>(c))) {c = '_';}
    }
    return out;
  }

  void OnUpdate()
  {
    const gazebo::common::Time now = model_->GetWorld()->SimTime();
    const double dt = (now - last_update_).Double();
    // A paused or reset world hands us dt <= 0; stepping the integrator on that
    // injects a spike the servo then has to unwind.
    if (dt <= 0.0) {
      last_update_ = now;
      return;
    }
    last_update_ = now;

    const double error = joint_->Position(0) - target_;
    integral_ = std::clamp(integral_ + error * dt, i_min_, i_max_);
    const double derivative = (error - last_error_) / dt;
    last_error_ = error;

    const double force = std::clamp(
      -(p_gain_ * error + i_gain_ * integral_ + d_gain_ * derivative),
      cmd_min_, cmd_max_);
    joint_->SetForce(0, force);
  }

  gazebo::physics::ModelPtr model_;
  gazebo::physics::JointPtr joint_;
  gazebo::event::ConnectionPtr update_conn_;
  gazebo::common::Time last_update_;

  std::string joint_name_;
  double p_gain_{1.0}, i_gain_{0.0}, d_gain_{0.0};
  double i_max_{0.0}, i_min_{0.0}, cmd_max_{1000.0}, cmd_min_{-1000.0};
  double target_{0.0}, integral_{0.0}, last_error_{0.0};

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  std::thread spin_thread_;
};

GZ_REGISTER_MODEL_PLUGIN(GimbalJointPositionController)

}  // namespace gimbal_camera
