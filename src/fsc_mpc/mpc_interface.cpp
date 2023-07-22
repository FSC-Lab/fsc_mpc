// Copyright (c) 2023 hs293go
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "fsc_mpc/mpc_interface.hpp"

#include <limits>

#include "fsc_mpc/internal.hpp"

extern "C" {
#include "acados_c/ocp_nlp_interface.h"
}

#define ACADOS_CHECK(expr)                                                \
  do {                                                                    \
    if (expr) {                                                           \
      throw AcadosWrapperException(#expr " returned nonzero error code"); \
    }                                                                     \
  } while (0)

namespace fsc::control {

constexpr double MPCInterface::kGravAccel;

const MPCInterface::BoundsType MPCInterface::kNoBounds =
    MPCInterface::BoundsType::Constant(1e50);

MPCInterface::MPCInterface() : capsule_(CreateCapsule()) {
  ACADOS_CHECK(CreateSolver(capsule()));
  init();
}

MPCInterface::MPCInterface(InRef<Eigen::VectorXd> time_steps)
    : capsule_(CreateCapsule()) {
  using details::MutData;
  ACADOS_CHECK(CreateSolverWithDiscretization(
      capsule(), static_cast<int>(time_steps.size()), MutData(time_steps)));
  init();
}

MPCInterface::MPCInterface(MPCInterface&& other) noexcept {
  *this = std::move(other);
}

MPCInterface& MPCInterface::operator=(MPCInterface&& other) noexcept {
  using std::swap;
  if (this != &other) {
    swap(other.capsule_, capsule_);
    swap(other.config_, config_);
    swap(other.dims_, dims_);
    swap(other.in_, in_);
    swap(other.out_, out_);
    swap(other.solver_, solver_);
  }
  return *this;
}

MPCInterface::~MPCInterface() { FreeSolver(capsule()); }

void MPCInterface::resetSolver(bool reset_qp_solver_mem) {
  ResetSolver(capsule(), reset_qp_solver_mem);
}

void MPCInterface::setInitialState(InRef<StateType> initial_state) {
  using details::MutData;
  ocp_nlp_constraints_model_set(config_, dims_, in_, 0, "lbx",
                                MutData(initial_state));
  ocp_nlp_constraints_model_set(config_, dims_, in_, 0, "ubx",
                                MutData(initial_state));
}

void MPCInterface::setReference(int i, InRef<RefType> ref) {
  using details::MutData;
  ocp_nlp_cost_model_set(config_, dims_, in_, i, "y_ref", MutData(ref));
}

void MPCInterface::setTerminalReference(InRef<EndRefType> terminal_ref) {
  using details::MutData;
  ocp_nlp_cost_model_set(config_, dims_, in_, num_mpc_nodes(), "y_ref",
                         MutData(terminal_ref));
}

void MPCInterface::setCosts(InRef<StateCostType> Q, InRef<InputCostType> R) {
  using details::MutData;
  CostType costs = CostType::Zero();
  costs.topLeftCorner<kCostSize, kCostSize>() = Q;
  costs.bottomRightCorner<kInputSize, kInputSize>() = R;
  for (int i = 0; i < num_mpc_nodes(); ++i) {
    ocp_nlp_cost_model_set(config_, dims_, in_, i, "W", costs.data());
  }
  ocp_nlp_cost_model_set(config_, dims_, in_, num_mpc_nodes(), "W", MutData(Q));
}

void MPCInterface::setCostWeights(InRef<StateCostWeightType> q_weights,
                                  InRef<InputCostWeightType> r_weights) {
  setCosts(StateCostType(q_weights.asDiagonal()),
           InputCostType(r_weights.asDiagonal()));
}

void MPCInterface::setBounds(InRef<BoundsType> lbu, InRef<BoundsType> ubu) {
  using details::MutData;
  if ((lbu.array() > ubu.array()).any()) {
    throw AcadosWrapperException(
        "Some elements in lower bound are greater than corresponding elements "
        "in upper bound");
  }

  for (int i = 0; i < num_mpc_nodes(); ++i) {
    ocp_nlp_constraints_model_set(config_, dims_, in_, i, "lbu", MutData(lbu));
    ocp_nlp_constraints_model_set(config_, dims_, in_, i, "ubu", MutData(ubu));
  }
}

MPCInterface::StateType MPCInterface::getState(int i) const {
  StateType res;
  ocp_nlp_out_get(config_, dims_, out_, i, "x", res.data());
  return res;
}

MPCInterface::InputType MPCInterface::getInput(int i) const {
  InputType res;
  ocp_nlp_out_get(config_, dims_, out_, i, "u", res.data());
  return res;
}

MPCInterface::StateTrajectoryType MPCInterface::getState() const {
  StateTrajectoryType predicted_states(static_cast<int>(kStateSize),
                                       num_mpc_nodes());
  for (int i = 0; i < num_mpc_nodes(); ++i) {
    predicted_states.col(i) = getState(i);
  }
  return predicted_states;
}

MPCInterface::InputTrajectoryType MPCInterface::getInput() const {
  InputTrajectoryType inputs(static_cast<int>(kInputSize), num_mpc_nodes());
  for (int i = 0; i < num_mpc_nodes(); ++i) {
    inputs.col(i) = getInput(i);
  }
  return inputs;
}

void MPCInterface::setReferenceState(InRef<StateType> state,
                                     InRef<InputType> input) {
  const RefType ref = (RefType() << state, input).finished();
  for (int i = 0; i < num_mpc_nodes(); ++i) {
    setReference(i, ref);
  }
  setTerminalReference(state);
}

void MPCInterface::setReferenceTrajectory(
    InRef<StateTrajectoryType> state_ref,
    InRef<InputTrajectoryType> input_ref) {
  const int n_x_samples = static_cast<int>(state_ref.cols());
  const int n_u_samples = static_cast<int>(input_ref.cols());
  if (n_x_samples != n_u_samples && n_x_samples != n_u_samples + 1) {
    throw AcadosWrapperException(
        "Number of state and input references do not match");
  }

  RefType ref;
  for (int i = 0; i < num_mpc_nodes(); ++i) {
    const int i_state = std::min(n_x_samples - 1, i);
    const int i_input = std::min(n_u_samples - 1, i);
    ref << state_ref.col(i_state), input_ref.col(i_input);
    setReference(i, ref);
  }
  setTerminalReference(
      state_ref.col(std::min<int>(n_x_samples - 1, num_mpc_nodes())));
}

void MPCInterface::setParameters(int i, InRef<ParamType> params) {
  SetParameters(capsule(), i, details::MutData(params));
}

void MPCInterface::setConstantParameters(InRef<ParamType> params) {
  for (int i = 0; i < num_mpc_nodes(); ++i) {
    setParameters(i, params);
  }
}

void MPCInterface::setPrintLevel(int value) {
  ocp_nlp_solver_opts_set(config_, opts_, "print_level", &value);
}

MPCInterface::InputType MPCInterface::optimize(InRef<StateType> state) {
  setInitialState(state);
  ACADOS_CHECK(Solve(capsule()));
  return getInput(0);
}

void MPCInterface::init() {
  using StateIdxs = Eigen::Matrix<int, kStateSize, 1>;
  using InputIdxs = Eigen::Matrix<int, kInputSize, 1>;

  config_ = GetConfig(capsule());
  dims_ = GetDims(capsule());
  in_ = GetInput(capsule());
  out_ = GetOutput(capsule());
  solver_ = GetSolver(capsule());
  opts_ = GetOpts(capsule());

  // Initialize the state constraint
  StateIdxs constrained_state_idx = StateIdxs::LinSpaced(0, kStateSize - 1);
  ocp_nlp_constraints_model_set(config_, dims_, in_, 0, "idxbx",
                                constrained_state_idx.data());

  InputIdxs constrained_input_idx = InputIdxs::LinSpaced(0, kInputSize - 1);
  for (int i = 0; i < num_mpc_nodes(); ++i) {
    ocp_nlp_constraints_model_set(config_, dims_, in_, i, "idxbu",
                                  constrained_input_idx.data());
    setBounds(-kNoBounds, kNoBounds);
  }

  // Initialize the output struct
  for (int i = 0; i < num_mpc_nodes(); ++i) {
    setReference(i, RefType::Zero());
    setState(i, StateType::Zero());
  }
  setTerminalState(StateType::Zero());
}

void MPCInterface::setState(int i, InRef<StateType> state) {
  using details::MutData;
  ocp_nlp_out_set(config_, dims_, out_, i, "x", MutData(state));
}

void MPCInterface::setTerminalState(InRef<StateType> terminal_state) {
  setState(num_mpc_nodes(), terminal_state);
}

void MPCInterface::setInput(int i, InRef<InputType> input) {
  using details::MutData;
  ocp_nlp_out_set(config_, dims_, out_, i, "u", MutData(input));
}

double MPCInterface::step_length(int i) const { return in_->Ts[i]; }

Eigen::VectorXd MPCInterface::step_length() const {
  return Eigen::VectorXd::Map(in_->Ts, num_mpc_nodes());
}

int MPCInterface::num_mpc_nodes() const { return dims_->N; }

}  // namespace fsc::control
