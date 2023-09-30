// The primary OOP-style interface to the acados MPC
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

#ifndef FSC_MPC_MPC_INTERFACE_HPP_
#define FSC_MPC_MPC_INTERFACE_HPP_

#include <memory>
#include <stdexcept>

#include "Eigen/Dense"
#include "fsc_mpc/internal.hpp"
#include "fsc_mpc/solver_wrapper.hpp"

namespace fsc::control {

class AcadosWrapperException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class MPCInterface {
 public:
  enum {
    kStateSize = details::ToUnderlying(Dimensions::kStateSize),
    kInputSize = details::ToUnderlying(Dimensions::kInputSize),
    kRefSize = details::ToUnderlying(Dimensions::kRefSize),
    kEndRefSize = details::ToUnderlying(Dimensions::kEndRefSize),
    kSamples = details::ToUnderlying(Dimensions::kSamples),
    kCostSize = details::ToUnderlying(Dimensions::kCostSize),
    kBoundsSize = details::ToUnderlying(Dimensions::kBoundsSize),
    kParamSize = details::ToUnderlying(Dimensions::kParamSize)
  };

  // Working variables of the solver

  // Type of the full W matrix. Typically blkdiag(Q, R)
  using CostType = Eigen::Matrix<double, kRefSize, kRefSize>;

  // Type of the full W matrix at the terminal step (W_e). Typically Q
  using EndCostType = Eigen::Matrix<double, kEndRefSize, kEndRefSize>;

  // Type of the Q matrix
  using StateCostType = Eigen::Matrix<double, kCostSize, kCostSize>;

  // Type of state cost weights, i.e. diag(Q)
  using StateCostWeightType = Eigen::Matrix<double, kCostSize, 1>;

  // Type of the R matrix
  using InputCostType = Eigen::Matrix<double, kInputSize, kInputSize>;

  // Type of input cost weights, i.e. diag(R)
  using InputCostWeightType = Eigen::Matrix<double, kInputSize, 1>;

  // Type of input bounds
  using BoundsType = Eigen::Matrix<double, kBoundsSize, 1>;

  // Type of the reference vector. Typically [x_ref; u_ref]
  using RefType = Eigen::Matrix<double, kRefSize, 1>;

  // Type of the reference vector at the terminal step. Typically x_ref
  using EndRefType = Eigen::Matrix<double, kEndRefSize, 1>;

  // Type of the state vector
  using StateType = Eigen::Matrix<double, kStateSize, 1>;

  // Type of the input vector
  using InputType = Eigen::Matrix<double, kInputSize, 1>;

  // Type of the parameter vector
  using ParamType = Eigen::Matrix<double, kParamSize, 1>;

  // Types of working variables stacked along the last axis
  using StateTrajectoryType = Eigen::Matrix<double, kStateSize, Eigen::Dynamic>;
  using InputTrajectoryType = Eigen::Matrix<double, kInputSize, Eigen::Dynamic>;

  // Reference wrapper for declaring (constant) input parameters of Eigen
  // objects
  // https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class
  template <typename T>
  using InRef = const Eigen::Ref<const T> &;

  static const BoundsType kNoBounds;

  MPCInterface();

  explicit MPCInterface(InRef<Eigen::VectorXd> time_steps);

  // Move ctors must be explicitly defaulted since we have a custom dtor
  MPCInterface(MPCInterface &&other) noexcept;

  MPCInterface &operator=(MPCInterface &&other) noexcept;

  ~MPCInterface();

  /**
   * @brief Resets the acados solver
   *
   * @param reset_qp_solver_mem Toggles resetting the memory in the QP solver
   */
  void resetSolver(bool reset_qp_solver_mem = false);

  /**
   * @brief Set the constraints on the solver state at the initial shooting
   * node. This is used to constrain the solver state to the actual system state
   * exactly
   *
   * @param initial_state A state vector containing the latest, actual system
   * state
   */
  void setInitialState(InRef<StateType> initial_state);

  /**
   * @brief Set the reference of the solver at some given shooting node.
   *
   * @param i Index of the shooting node
   * @param ref The reference vector, typically containing a stack of the state
   * reference and input reference
   */
  void setReference(int i, InRef<RefType> ref);

  /**
   * @brief Set the reference of the solver at the terminal shooting node.
   *
   * @param terminal_ref The reference vector at the terminal shooting node,
   * typically containing just the state reference
   */
  void setTerminalReference(InRef<EndRefType> terminal_ref);

  /**
   * @brief Set the state and input cost matrices
   *
   * When specifying cost weights only, do not pass `(...).asDiagonal()` to this
   * function, but use the `setCostWeights`
   *
   * @param Q The state cost matrix
   * @param R The input cost matrix
   */
  void setCosts(InRef<StateCostType> Q, InRef<InputCostType> R);

  /**
   * @brief Set the state and input cost weights
   *
   * @param q_weights A vector of state cost weights
   * @param r_weights A vector of input cost weights
   */
  void setCostWeights(InRef<StateCostWeightType> q_weights,
                      InRef<InputCostWeightType> r_weights);

  /**
   * @brief Set bounds on system inputs
   *
   * Throws an exception if any element in the lower bounds vector are not
   * smaller than corresponding element in upper bounds vector
   *
   * @param lbu The lower bounds on system inputs
   * @param ubu The upper bounds on system inputs
   */
  void setBounds(InRef<BoundsType> lbu, InRef<BoundsType> ubu);

  /**
   * @brief Get the state predicted by the solver at some given shooting node
   *
   * @param i Index of the shooting node
   * @return StateType The state vector
   */
  [[nodiscard]] StateType getState(int i) const;

  /**
   * @brief Get the input predicted by the solver at some given shooting node
   *
   * @param i Index of the shooting node
   * @return InputType The input vector
   */
  [[nodiscard]] InputType getInput(int i) const;

  /**
   * @brief Get the state predicted by the solver over all shooting nodes
   *
   * @return InputType The state vector for each shooting node stacked over the
   * last axis
   */
  [[nodiscard]] StateTrajectoryType getState() const;

  /**
   * @brief Get the input predicted by the solver over all shooting nodes
   *
   * @return InputType The input vector for each shooting node stacked over the
   * last axis
   */
  [[nodiscard]] InputTrajectoryType getInput() const;

  /**
   * @brief Set the state and input reference over all shooting nodes to a
   * single state and input setpoint
   *
   * @param state The state setpoint
   * @param input The input setpoint
   */
  void setReferenceState(InRef<StateType> state, InRef<InputType> input);

  /**
   * @brief Set the state and input reference over all shooting nodes to a
   * trajectory of distinct states and inputs
   *
   * @details State and input references are specified as matrices consisting of
   * column-wise stacks of state and input vectors respectively. The width
   * (number of rows) of the state reference matrix must be equal or one more
   * than that of the input reference matrix. If the width (number of rows) of
   * the state reference or input reference matrix are fewer than N+1 or N,
   * where N is the number of shooting nodes, they are padded column-wise by the
   * last column to N+1 / N.
   *
   * @param state_ref The matrix containing the stack of state references
   * @param input_ref The matrix containing the stack of input references
   */
  void setReferenceTrajectory(InRef<StateTrajectoryType> state_ref,
                              InRef<InputTrajectoryType> input_ref);

  /**
   * @brief Set the online parameters of the solver at some given shooting node
   *
   * @param i Index of the shooting node
   * @param params The parameter vector
   */
  void setParameters(int i, InRef<ParamType> params);

  /**
   * @brief Set the online parameters of the solver that is constant over all
   * shooting nodes
   *
   * @param params The parameter vector
   */
  void setConstantParameters(InRef<ParamType> params);

  /**
   * @brief Sets the print level of the solver
   *
   * @param value The print level
   */
  void setPrintLevel(int value);

  /**
   * @brief Convenience method taking the current system state (to be passed to
   * setInitialState) and solves the optimal control problem, then return the
   * optimized input on the first step of the predicted input trajectory
   *
   * @param state A state vector containing the latest, actual system state
   * @return InputType The first optimized input
   */
  InputType optimize(InRef<StateType> state);

  /**
   * @brief Gets the time step at a given shooting node
   *
   * @param i Index of the shooting node
   * @return double Time step at given shooting node
   */
  [[nodiscard]] double step_length(int i) const;

  /**
   * @brief Gets all the time steps over the entire prediction horizon
   *
   * @return Eigen::VectorXd A vector with size equal to the number of shooting
   * nodes, containing the time steps
   */
  [[nodiscard]] Eigen::VectorXd step_length() const;

  /**
   * @brief Gets the number of shooting nodes
   *
   * @return int Number of the shooting nodes
   */
  [[nodiscard]] int num_mpc_nodes() const;

 private:
  using Capsule = details::Handle<SolverCapsule, FreeCapsule>;

  [[nodiscard]] inline auto *capsule() const { return capsule_.get(); }
  [[nodiscard]] inline auto *config() const { return GetConfig(capsule()); }
  [[nodiscard]] inline auto *dims() const { return GetDims(capsule()); }
  [[nodiscard]] inline auto *in() const { return GetInput(capsule()); }
  [[nodiscard]] inline auto *out() const { return GetOutput(capsule()); }
  [[nodiscard]] inline auto *solver() const { return GetSolver(capsule()); }
  [[nodiscard]] inline auto *opts() const { return GetOpts(capsule()); }

  void init();

  void setState(int i, InRef<StateType> state);

  void setTerminalState(InRef<StateType> state);

  void setInput(int i, InRef<InputType> input);

  Capsule capsule_;
};

}  // namespace fsc::control
#endif  // FSC_MPC_MPC_INTERFACE_HPP_
