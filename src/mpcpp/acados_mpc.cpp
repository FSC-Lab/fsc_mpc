// Copyright (c) 2023 hs293go
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "mpcpp/acados_mpc.hpp"

#include <limits>

#include "mpcpp/internal.hpp"

extern "C" {
#include "acados_c/ocp_nlp_interface.h"
}

#define ACADOS_CHECK(expr)                                                \
  do {                                                                    \
    if (expr) {                                                           \
      throw AcadosWrapperException(#expr " returned nonzero error code"); \
    }                                                                     \
  } while (0)

namespace control {

constexpr double AcadosMPC::kGravAccel;

const AcadosMPC::BoundsType AcadosMPC::kNoBounds =
    AcadosMPC::BoundsType::Constant(1e50);

AcadosMPC::AcadosMPC() : capsule_(acadospp::CreateCapsule()) {
  ACADOS_CHECK(acadospp::CreateSolver(capsule()));
  init();
}

AcadosMPC::AcadosMPC(InRef<Eigen::VectorXd> time_steps)
    : capsule_(acadospp::CreateCapsule()) {
  using details::MutData;
  ACADOS_CHECK(acadospp::CreateSolverWithDiscretization(
      capsule(), static_cast<int>(time_steps.size()), MutData(time_steps)));
  init();
}

AcadosMPC::AcadosMPC(AcadosMPC&& other) noexcept { *this = std::move(other); }

AcadosMPC& AcadosMPC::operator=(AcadosMPC&& other) noexcept {
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

AcadosMPC::~AcadosMPC() { acadospp::FreeSolver(capsule()); }

void AcadosMPC::resetSolver(bool reset_qp_solver_mem) {
  acadospp::ResetSolver(capsule(), reset_qp_solver_mem);
}

void AcadosMPC::setInitialState(InRef<StateType> initial_state) {
  using details::MutData;
  ocp_nlp_constraints_model_set(config_, dims_, in_, 0, "lbx",
                                MutData(initial_state));
  ocp_nlp_constraints_model_set(config_, dims_, in_, 0, "ubx",
                                MutData(initial_state));
}

void AcadosMPC::setReference(int i, InRef<RefType> ref) {
  using details::MutData;
  ocp_nlp_cost_model_set(config_, dims_, in_, i, "y_ref", MutData(ref));
}

void AcadosMPC::setTerminalReference(InRef<EndRefType> terminal_ref) {
  using details::MutData;
  ocp_nlp_cost_model_set(config_, dims_, in_, num_mpc_nodes(), "y_ref",
                         MutData(terminal_ref));
}

void AcadosMPC::setCosts(InRef<StateCostType> Q, InRef<InputCostType> R) {
  using details::MutData;
  Eigen::Matrix<double, kRefSize, kRefSize> costs;
  costs.topLeftCorner<kCostSize, kCostSize>() = Q;
  costs.bottomRightCorner<kInputSize, kInputSize>() = R;
  for (int i = 0; i < kSamples; ++i) {
    ocp_nlp_cost_model_set(config_, dims_, in_, i, "W", MutData(costs));
  }
  ocp_nlp_cost_model_set(config_, dims_, in_, kSamples, "W", MutData(Q));
}

void AcadosMPC::setCostWeights(InRef<StateCostWeightType> q_weights,
                               InRef<InputCostWeightType> r_weights) {
  setCosts(StateCostType(q_weights.asDiagonal()),
           InputCostType(r_weights.asDiagonal()));
}

void AcadosMPC::setBounds(InRef<BoundsType> lbu, InRef<BoundsType> ubu) {
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

AcadosMPC::StateType AcadosMPC::getState(int i) const {
  StateType res;
  ocp_nlp_out_get(config_, dims_, out_, i, "x", res.data());
  return res;
}

AcadosMPC::InputType AcadosMPC::getInput(int i) const {
  InputType res;
  ocp_nlp_out_get(config_, dims_, out_, i, "u", res.data());
  return res;
}

void AcadosMPC::setReferenceState(InRef<StateType> state,
                                  InRef<InputType> input) {
  const RefType ref = (RefType() << state, input).finished();
  for (int i = 0; i < num_mpc_nodes(); ++i) {
    setReference(i, ref);
  }
  setTerminalReference(state);
}

void AcadosMPC::setReferenceTrajectory(InRef<StateTrajectoryType> state_ref,
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
  setTerminalReference(state_ref.col(std::min<int>(n_x_samples - 1, kSamples)));
}

void AcadosMPC::setParameters(int i, InRef<ParamType> params) {
  acadospp::SetParameters(capsule(), i, details::MutData(params));
}

void AcadosMPC::setConstantParameters(InRef<ParamType> params) {
  for (int i = 0; i < kSamples; ++i) {
    setParameters(i, params);
  }
}

void AcadosMPC::setPrintLevel(int value) {
  ocp_nlp_solver_opts_set(config_, opts_, "print_level", &value);
}

AcadosMPC::InputType AcadosMPC::optimize(InRef<StateType> state) {
  setInitialState(state);
  ACADOS_CHECK(acadospp::Solve(capsule()));
  return getInput(0);
}

void AcadosMPC::init() {
  using StateIdxs = Eigen::Matrix<int, kStateSize, 1>;
  using InputIdxs = Eigen::Matrix<int, kInputSize, 1>;

  config_ = acadospp::GetConfig(capsule());
  dims_ = acadospp::GetDims(capsule());
  in_ = acadospp::GetInput(capsule());
  out_ = acadospp::GetOutput(capsule());
  solver_ = acadospp::GetSolver(capsule());
  opts_ = acadospp::GetOpts(capsule());

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

void AcadosMPC::setState(int i, InRef<StateType> state) {
  using details::MutData;
  ocp_nlp_out_set(config_, dims_, out_, i, "x", MutData(state));
}

void AcadosMPC::setTerminalState(InRef<StateType> terminal_state) {
  setState(kSamples, terminal_state);
}

void AcadosMPC::setInput(int i, InRef<InputType> input) {
  using details::MutData;
  ocp_nlp_out_set(config_, dims_, out_, i, "u", MutData(input));
}

double AcadosMPC::step_length(int i) const { return in_->Ts[i]; }

Eigen::VectorXd AcadosMPC::step_length() const {
  return Eigen::VectorXd::Map(in_->Ts, num_mpc_nodes());
}

int AcadosMPC::num_mpc_nodes() const { return dims_->N; }

}  // namespace control
