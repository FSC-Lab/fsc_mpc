// Copyright (c) 2023 hs293go
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef QUADROTOR_MPCPP_QUADROTOR_MPC_HPP_
#define QUADROTOR_MPCPP_QUADROTOR_MPC_HPP_

#include <memory>
#include <stdexcept>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "quadrotor_mpcpp/internal.hpp"
#include "quadrotor_mpcpp/solver_wrapper.hpp"

namespace control {

namespace details {
template <typename T, int (*D)(T *)>
struct DeleterWrapper {
  inline void operator()(T *obj) const { static_cast<void>(D(obj)); }
};

template <typename T, int (*D)(T *)>
using Handle = std::unique_ptr<T, DeleterWrapper<T, D>>;
}  // namespace details

class AcadosWrapperException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class AcadosMPC {
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

  // Type of the R matrix
  using InputCostType = Eigen::Matrix<double, kInputSize, kInputSize>;

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

  static constexpr double kGravAccel{9.81};

  static const StateCostType kDefaultStateCost;
  static const InputCostType kDefaultInputCost;
  static const RefType kDefaultRef;
  static const StateType kDefaultState;
  static const InputType kDefaultInput;

  AcadosMPC();

  explicit AcadosMPC(InRef<Eigen::VectorXd> time_steps);

  ~AcadosMPC();

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
   * @param Q The state cost matrix
   * @param R The input cost matrix
   */
  void setCosts(InRef<StateCostType> Q, InRef<InputType> R);

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
   * @brief Set the state and input reference over all shooting nodes
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
   * @brief Convenience method taking the current system state (to be passed to
   * setInitialState) and solves the optimal control problem, then return the
   * optimized input on the first step of the predicted input trajectory
   *
   * @param state A state vector containing the latest, actual system state
   * @return InputType The first optimized input
   */
  InputType optimize(InRef<StateType> state);

 private:
  using Capsule =
      details::Handle<acadospp::SolverCapsule, acadospp::FreeCapsule>;

  inline acadospp::SolverCapsule *capsule() { return capsule_.get(); }

  void init();

  void setState(int i, InRef<StateType> state);

  void setTerminalState(InRef<StateType> state);

  void setInput(int i, InRef<InputType> input);

  Capsule capsule_;

  StateType initial_state_{kDefaultState};

  ocp_nlp_config *config_;
  ocp_nlp_dims *dims_;
  ocp_nlp_in *in_;
  ocp_nlp_out *out_;
  ocp_nlp_solver *solver_;
};

}  // namespace control
#endif  // QUADROTOR_MPCPP_QUADROTOR_MPC_HPP_
