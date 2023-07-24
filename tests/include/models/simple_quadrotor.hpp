// FSC's quadrotor simulator, minified for test stability
// Copyright © 2023 FSC Lab
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef INCLUDE_MODELS_SIMPLE_QUADROTOR_HPP_  // NOLINT
#define INCLUDE_MODELS_SIMPLE_QUADROTOR_HPP_

#include "Eigen/Dense"

namespace fsc {

template <typename S>
class SimpleQuadrotor {
 public:
  using Scalar = S;

  enum { kStateSize = 10, kInputSize = 4, kDerivativeSize = 10 };

  using StateType = Eigen::Matrix<Scalar, kStateSize, 1>;
  using DerivativeType = Eigen::Matrix<Scalar, kDerivativeSize, 1>;
  using InputType = Eigen::Matrix<Scalar, kInputSize, 1>;

  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
  using Quaternion = Eigen::Quaternion<Scalar>;

  enum Block {
    kPositionSize = 3,
    kAttitudeSize = 4,
    kVelocitySize = 3,

    kPositionOffset = 0,
    kAttitudeOffset = kPositionOffset + kPositionSize,
    kVelocityOffset = kAttitudeOffset + kAttitudeSize,

    kForceSize = 1,
    kAngVelSize = 3,

    kForceOffset = 0,
    kAngVeloffset = kForceOffset + kForceSize
  };

  explicit SimpleQuadrotor(Scalar mass, Scalar grav_accel,
                           Scalar quaternion_normalization_gain = Scalar(1))
      : mass_(mass),
        grav_accel_(grav_accel),
        k_nrm_(quaternion_normalization_gain) {}

  bool modelDerivatives([[maybe_unused]] Scalar time, const StateType& state,
                        const InputType& input, DerivativeType& derivative) {
    if (mass_ < Scalar(0)) {
      return false;
    }

    if (k_nrm_ < Scalar(0)) {
      return false;
    }

    const Quaternion attitude(getAttitude(state));
    const Vector3 velocity = getVelocity(state);

    const Vector3 force = getForce(input) * Vector3::UnitZ();
    const Vector3 ang_vel = getAngVel(input);

    const Scalar k = k_nrm_ * (Scalar(1) - attitude.squaredNorm());
    const Vector4 quat_correction = k * attitude.coeffs();
    Quaternion ang_vel_q;
    ang_vel_q.coeffs() << ang_vel, Scalar(0);

    const Vector4 attitude_derivative =
        (attitude * ang_vel_q).coeffs() / Scalar(2) + quat_correction;
    const Vector3 velocity_derivative =
        attitude * force / mass_ + Vector3::UnitZ() * grav_accel_;
    derivative << velocity, attitude_derivative, velocity_derivative;

    return true;
  }

  static Vector3 getPosition(const StateType& state) {
    return state.template head<kPositionSize>();
  }

  static Vector4 getAttitude(const StateType& state) {
    return state.template segment<kAttitudeSize>(kAttitudeOffset);
  }

  static Vector3 getVelocity(const StateType& state) {
    return state.template segment<kVelocitySize>(kVelocityOffset);
  }

  static Scalar getForce(const InputType& input) { return input.coeff(0); }

  static Vector3 getAngVel(const InputType& input) {
    return input.template segment<kAngVelSize>(kAngVeloffset);
  }

  Scalar& mass() { return mass_; }

  Scalar& grav_accel() { return grav_accel_; }

  Scalar& quaternion_normalization_gain() { return k_nrm_; }

 private:
  Scalar mass_;
  Scalar grav_accel_;
  Scalar k_nrm_;
};
}  // namespace fsc

#endif  // INCLUDE_MODELS_SIMPLE_QUADROTOR_HPP_ NOLINT
