// FSC's dynamic system simulator, minified for test stability
// Copyright © 2023 yourname
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

#ifndef INCLUDE_SIMULATION_DYNAMIC_SYSTEM_SIMULATOR_HPP_  // NOLINT
#define INCLUDE_SIMULATION_DYNAMIC_SYSTEM_SIMULATOR_HPP_

#include "Eigen/Dense"

namespace fsc {

template <typename S>
class DynamicSystemSimulator {
 public:
  using System = S;
  using Scalar = typename System::Scalar;
  using StateType = typename System::StateType;
  using InputType = typename System::InputType;
  using DerivativeType = typename System::DerivativeType;

  enum {
    kStateSize = System::kStateSize,
    kInputSize = System::kInputSize,
    kDerivativeSize = System::kDerivativeSize,
    kOrder = 4
  };

  /**
   * @brief Construct a new Dynamic System Simulator object
   *
   * @param sys A class implementing the process model to be simulated
   * @param base_dt The basic time step for simulation
   * @param init_time The initial time in the simulator
   * @param init_state The initial state in the simulator
   * @param init_input The initial input for the simulator
   */
  template <typename SDerived, typename UDerived>
  DynamicSystemSimulator(const System &sys, Scalar base_dt, Scalar init_time,
                         const Eigen::MatrixBase<SDerived> &init_state,
                         const Eigen::MatrixBase<UDerived> &init_input)
      : sys_(sys),
        dt_(base_dt),
        time_(init_time),
        state_(init_state),
        input_(init_input) {}

  bool simulationUpdate() {
    if (dt_ <= Scalar(0) || !std::isfinite(dt_)) {
      puts("Invalid time");
      return false;
    }

    if (!state_.allFinite() || !input_.allFinite()) {
      puts("Invalid state and input");
      return false;
    }

    // We store a matrix (stack of columns) of interim values of the ODE
    // [k_1 ... k_N]
    Eigen::Matrix<Scalar, kDerivativeSize, kOrder> k_array;

    const StateType x_op = state_;

    DerivativeType k;
    const Scalar half_dt = dt_ / Scalar(2);
    // Evaluate the i-th interim value of the ODE (k_{i})
    if (!sys_.modelDerivatives(time_, x_op, input_, k)) {
      return false;
    }
    k_array.col(0) = k;

    if (!sys_.modelDerivatives(time_ + half_dt, x_op + half_dt * k_array.col(0),
                               input_, k)) {
      return false;
    }
    k_array.col(1) = k;

    if (!sys_.modelDerivatives(time_ + half_dt, x_op + half_dt * k_array.col(1),
                               input_, k)) {
      return false;
    }
    k_array.col(2) = k;

    if (!sys_.modelDerivatives(time_ + dt_, x_op + dt_ * k_array.col(2), input_,
                               k)) {
      return false;
    }
    k_array.col(3) = k;

    const Eigen::Matrix<Scalar, kOrder, 1> rk4_coeffs(
        Scalar(1.0 / 6.0), Scalar(2.0 / 6.0), Scalar(2.0 / 6.0),
        Scalar(1.0 / 6.0));

    state_ = x_op + dt_ * k_array * rk4_coeffs;

    time_ += dt_;

    return true;
  }

  template <typename UDerived>
  bool simulationUpdate(const Eigen::MatrixBase<UDerived> &input, Scalar dt);

  template <typename UDerived>
  bool simulationUpdate(const Eigen::MatrixBase<UDerived> &input);

  /**
   * @brief Gets the dynamics system object
   *
   * @return const System& A const reference to the underlying dynamics system
   * object
   */
  const System &sys() const { return sys_; }

  /**
   * @brief Gets the integration time interval currently used inside the
   * simulator
   *
   * @return Scalar The integration time interval in seconds
   */
  Scalar dt() const { return dt_; }

  /**
   * @brief Gets the current time inside the simulator
   *
   * @return Scalar The current time in seconds
   */
  Scalar time() const { return time_; }

  /**
   * @brief Gets the state inside the simulator
   *
   * @return const StateType& A constant reference to the simulator state
   */
  const StateType &state() const { return state_; }

  /**
   * @brief Gets the (mutable) input inside the simulator
   *
   * @return InputType& A mutable reference to the input for the simulator
   */
  InputType &input() { return input_; }

  /**
   * @brief Gets the (constant) input inside the simulator
   *
   * @return const InputType& A constant reference to the input for the
   * simulator
   */
  const InputType &input() const { return input_; }

 private:
  System sys_;
  Scalar dt_;

  Scalar time_{0};
  StateType state_{StateType::Zero()};
  InputType input_{InputType::Zero()};
};
}  // namespace fsc

#endif  // INCLUDE_SIMULATION_DYNAMIC_SYSTEM_SIMULATOR_HPP_ NOLINT
