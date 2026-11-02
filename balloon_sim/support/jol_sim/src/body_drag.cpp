// Copyright 2026 xogus. SPDX-License-Identifier: Apache-2.0
// Assumed quadratic body drag for the STL multicopter; not a calibrated aerodynamic model.
// F_B = -0.5 * rho * |v_B| * (CdA_B ⊙ v_B), applied at the link centre of mass, still air.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <cmath>
#include <functional>
#include <string>

namespace jol_sim {
class BodyDrag final : public gazebo::ModelPlugin {
 public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override {
    const auto link_name = sdf->Get<std::string>("link_name", "base_link").first;
    rho_ = sdf->Get<double>("air_density", 1.225).first;
    cda_ = sdf->Get<ignition::math::Vector3d>("cda_body_m2", ignition::math::Vector3d::Zero).first;
    link_ = model->GetLink(link_name);
    if (!link_ || !std::isfinite(rho_) || rho_ <= 0.0 || !std::isfinite(cda_.X()) || !std::isfinite(cda_.Y()) ||
        !std::isfinite(cda_.Z()) || cda_.X() < 0.0 || cda_.Y() < 0.0 || cda_.Z() < 0.0) {
      gzerr << "Invalid JOL body drag settings (link=" << link_name << "); drag disabled\n";
      return;
    }
    update_ = gazebo::event::Events::ConnectWorldUpdateBegin(std::bind(&BodyDrag::Update, this));
    gzmsg << "JOL body drag on " << link_name << ": rho=" << rho_ << " CdA_body=" << cda_ << " m^2\n";
  }

 private:
  void Update() {
    const auto v = link_->RelativeLinearVel();  // link-origin velocity in the link frame
    const double speed = v.Length();
    if (!std::isfinite(speed) || speed < 1e-6) return;
    const ignition::math::Vector3d force(-0.5 * rho_ * speed * cda_.X() * v.X(),
                                         -0.5 * rho_ * speed * cda_.Y() * v.Y(),
                                         -0.5 * rho_ * speed * cda_.Z() * v.Z());
    link_->AddRelativeForce(force);
  }

  gazebo::physics::LinkPtr link_;
  gazebo::event::ConnectionPtr update_;
  double rho_{1.225};
  ignition::math::Vector3d cda_;
};
GZ_REGISTER_MODEL_PLUGIN(BodyDrag)
}  // namespace jol_sim
