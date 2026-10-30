// Copyright 2026 xogus. SPDX-License-Identifier: Apache-2.0
// 시뮬레이터와 회귀검사가 공유하는 결정적 궤적·속도 계산이다.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace jol_sim {
struct FlightSample {
  std::array<double, 3> position, velocity;
  double heading, parameter;
};

class FlightPath {
 public:
  FlightPath(double radius = 50.0, double speed = 10.0, double altitude = 10.0)
      : radius_(radius), speed_(speed), altitude_(altitude) {
    if (!std::isfinite(radius) || radius <= 1 || !std::isfinite(speed) || speed < 0 ||
        !std::isfinite(altitude) || altitude <= 0)
      throw std::invalid_argument("Flight radius > 1, speed >= 0 and altitude > 0 must be finite");
    for (std::size_t i = 1; i < distances_.size(); ++i)
      distances_[i] = distances_[i - 1] + Arc((i - 1) * step_, i * step_);
  }

  double Length() const { return distances_.back(); }

  FlightSample At(std::size_t index, double motion_seconds, bool nonlinear = true) const {
    if (index >= 3 || !std::isfinite(motion_seconds) || motion_seconds < 0)
      throw std::invalid_argument("Flight index must be 0..2 and motion time finite/nonnegative");
    if (!nonlinear) {
      const double angle = -1.2 + index * .32 + motion_seconds * speed_ / radius_;
      return {{radius_ * std::cos(angle), radius_ * std::sin(angle), altitude_ + index * 1.2},
              {-speed_ * std::sin(angle), speed_ * std::cos(angle), 0.0}, angle + pi_ / 2, angle};
    }
    // 세 드론은 같은 주기에 1/3바퀴 위상차를 둔다. 위치는 누적 적분 없이 시간에서 계산한다.
    const double distance = std::fmod(motion_seconds * speed_ + index * Length() / 3, Length());
    const auto upper = std::upper_bound(distances_.begin(), distances_.end(), distance);
    const auto segment = std::min<std::size_t>(upper - distances_.begin() - 1, samples_ - 1);
    const double low = segment * step_, high = low + step_;
    double u = low + step_ * (distance - distances_[segment]) /
                           (distances_[segment + 1] - distances_[segment]);
    // 호 길이를 역산해 위치의 미분과 보고된 일정 속도가 일치하게 한다.
    for (int iteration = 0; iteration < 2; ++iteration)
      u = std::clamp(u - (distances_[segment] + Arc(low, u) - distance) / TangentNorm(u), low, high);
    auto sample = Curve(u);
    const double scale = speed_ / Norm(sample.velocity);
    const double rotation = index * .12, c = std::cos(rotation), s = std::sin(rotation);
    for (auto* vector : {&sample.position, &sample.velocity}) {
      const double x = (*vector)[0], y = (*vector)[1];
      (*vector)[0] = c * x - s * y;
      (*vector)[1] = s * x + c * y;
    }
    sample.position[2] += index * .6;
    sample.heading = std::atan2(sample.velocity[1], sample.velocity[0]);
    for (auto& value : sample.velocity) value *= scale;
    return sample;
  }

 private:
  static constexpr double pi_ = 3.14159265358979323846;
  static constexpr std::size_t samples_ = 4096;
  static constexpr double step_ = 2 * pi_ / samples_;
  static double Norm(const std::array<double, 3>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }
  FlightSample Curve(double u) const {
    const double r = radius_ * (1 + .16 * std::sin(2 * u + .4) + .08 * std::sin(3 * u - 1.1));
    const double dr = radius_ * (.32 * std::cos(2 * u + .4) + .24 * std::cos(3 * u - 1.1));
    const double angle = -1.2 + u + .13 * std::sin(2 * u + .25) + .065 * std::sin(3 * u - .6);
    const double da = 1 + .26 * std::cos(2 * u + .25) + .195 * std::cos(3 * u - .6);
    const double z = altitude_ * (1 + .28 * std::sin(2 * u - .5) + .10 * std::sin(5 * u + .4));
    const double dz = altitude_ * (.56 * std::cos(2 * u - .5) + .50 * std::cos(5 * u + .4));
    return {{r * std::cos(angle), r * std::sin(angle), z},
            {dr * std::cos(angle) - r * da * std::sin(angle),
             dr * std::sin(angle) + r * da * std::cos(angle), dz}, 0.0, u};
  }
  double TangentNorm(double u) const { return Norm(Curve(u).velocity); }
  double Arc(double low, double high) const {
    return (high - low) * (TangentNorm(low) + 4 * TangentNorm((low + high) / 2) + TangentNorm(high)) / 6;
  }
  double radius_, speed_, altitude_;
  std::array<double, samples_ + 1> distances_{};
};
}  // namespace jol_sim
