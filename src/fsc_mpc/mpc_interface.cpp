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

#include "fsc_mpc/mpc_interface.hpp"

#include "acados_c/ocp_nlp_interface.h"

#define CAT_IMPL(A, B) A##B
#define CAT(A, B) CAT_IMPL(A, B)

#define STRINGIFY_IMPL(A) #A
#define STRINGIFY(A) STRINGIFY_IMPL(A)

#ifndef MODEL_NAME_UPPER
#error MISSING DEFINITION FOR MODEL_NAME_UPPER
#endif

#ifndef MODEL_NAME_LOWER
#error MISSING DEFINITION FOR MODEL_NAME_LOWER
#endif

#define SOLVER_LIB STRINGIFY(CAT(acados_solver_, MODEL_NAME_LOWER).h)

#include SOLVER_LIB

#define ACADOS_CHECK(expr)                                                \
  do {                                                                    \
    if (expr) {                                                           \
      throw AcadosWrapperException(#expr " returned nonzero error code"); \
    }                                                                     \
  } while (0)

#define ACADOS_PARAM(PARM) CAT(CAT(MODEL_NAME_UPPER, _), PARM)

#define ACADOS_OBJ(func) CAT(CAT(MODEL_NAME_LOWER, _), func)

namespace fsc::control {

namespace details {
template <typename T>
constexpr auto MutData(const T& obj) -> std::add_pointer_t<
    std::remove_const_t<std::remove_pointer_t<decltype(obj.data())>>> {
  using ConstElement = std::remove_pointer_t<decltype(obj.data())>;
  using MutPtr = std::add_pointer_t<std::remove_const_t<ConstElement>>;
  return const_cast<MutPtr>(obj.data());
}

enum {
  kStateSize = ACADOS_PARAM(NX),
  kInputSize = ACADOS_PARAM(NU),
  kRefSize = ACADOS_PARAM(NY),
  kEndRefSize = ACADOS_PARAM(NYN),
  kSamples = ACADOS_PARAM(N),
  kCostSize = ACADOS_PARAM(NY) - ACADOS_PARAM(NU),
  kBoundsSize = ACADOS_PARAM(NBU),
  kParamSize = ACADOS_PARAM(NP)
};

using SolverCapsule = ACADOS_OBJ(solver_capsule);

SolverCapsule* CreateCapsule() { return ACADOS_OBJ(acados_create_capsule)(); }

int CreateSolver(SolverCapsule* capsule) {
  return ACADOS_OBJ(acados_create)(capsule);
}

struct FreeCapsule {
  void operator()(SolverCapsule* capsule) {
    std::ignore = ACADOS_OBJ(acados_free_capsule)(capsule);
  }
};

int FreeSolver(SolverCapsule* capsule) {
  return ACADOS_OBJ(acados_free)(capsule);
}

int ResetSolver(SolverCapsule* capsule, bool reset_qp_solver_mem) {
  return ACADOS_OBJ(acados_reset)(capsule,
                                  static_cast<int>(reset_qp_solver_mem));
}

int CreateSolverWithDiscretization(SolverCapsule* capsule, int n_time_steps,
                                   double* new_time_steps) {
  return ACADOS_OBJ(acados_create_with_discretization)(capsule, n_time_steps,
                                                       new_time_steps);
}
ocp_nlp_solver* GetSolver(SolverCapsule* capsule) {
  return ACADOS_OBJ(acados_get_nlp_solver)(capsule);
}

ocp_nlp_config* GetConfig(SolverCapsule* capsule) {
  return ACADOS_OBJ(acados_get_nlp_config)(capsule);
}

ocp_nlp_dims* GetDims(SolverCapsule* capsule) {
  return ACADOS_OBJ(acados_get_nlp_dims)(capsule);
}

ocp_nlp_in* GetInput(SolverCapsule* capsule) {
  return ACADOS_OBJ(acados_get_nlp_in)(capsule);
}

ocp_nlp_out* GetOutput(SolverCapsule* capsule) {
  return ACADOS_OBJ(acados_get_nlp_out)(capsule);
}

ocp_nlp_opts* GetOpts(SolverCapsule* capsule) {
  return static_cast<ocp_nlp_opts*>(ACADOS_OBJ(acados_get_nlp_opts)(capsule));
}

int Solve(SolverCapsule* capsule) { return ACADOS_OBJ(acados_solve)(capsule); }
void SetParameters(SolverCapsule* capsule, int stage, double* value) {
  ACADOS_OBJ(acados_update_params)
  (capsule, stage, value, static_cast<int>(details::kParamSize));
}
// Working variables of the solver

// Type of the full W matrix. Typically blkdiag(Q, R)
using CostType = Eigen::Matrix<double, details::kRefSize, details::kRefSize>;

// Type of the full W matrix at the terminal step (W_e). Typically Q
using EndCostType =
    Eigen::Matrix<double, details::kEndRefSize, details::kEndRefSize>;

// Type of the Q matrix
using StateCostType =
    Eigen::Matrix<double, details::kCostSize, details::kCostSize>;

// Type of state cost weights, i.e. diag(Q)
using StateCostWeightType = Eigen::Matrix<double, details::kCostSize, 1>;

// Type of the R matrix
using InputCostType =
    Eigen::Matrix<double, details::kInputSize, details::kInputSize>;

// Type of input cost weights, i.e. diag(R)
using InputCostWeightType = Eigen::Matrix<double, details::kInputSize, 1>;

// Type of input bounds
using BoundsType = Eigen::Matrix<double, details::kBoundsSize, 1>;

// Type of the reference vector. Typically [x_ref; u_ref]
using RefType = Eigen::Matrix<double, details::kRefSize, 1>;

// Type of the reference vector at the terminal step. Typically x_ref
using EndRefType = Eigen::Matrix<double, details::kEndRefSize, 1>;

// Type of the state vector
using StateType = Eigen::Matrix<double, details::kStateSize, 1>;

// Type of the input vector
using InputType = Eigen::Matrix<double, details::kInputSize, 1>;

// Type of the parameter vector
using ParamType = Eigen::Matrix<double, details::kParamSize, 1>;

// Types of working variables stacked along the last axis
using StateTrajectoryType =
    Eigen::Matrix<double, details::kStateSize, Eigen::Dynamic>;
using InputTrajectoryType =
    Eigen::Matrix<double, details::kInputSize, Eigen::Dynamic>;

const BoundsType kNoBounds = BoundsType::Constant(1e50);
}  // namespace details

struct MPCInterface::Impl {
  Impl() : capsule_(details::CreateCapsule()) {
    ACADOS_CHECK(details::CreateSolver(capsule_.get()));
    init();
  }

  explicit Impl(const VectorCRef& time_steps)
      : capsule_(details::CreateCapsule()) {
    using details::MutData;
    ACADOS_CHECK(details::CreateSolverWithDiscretization(
        capsule_.get(), static_cast<int>(time_steps.size()),
        MutData(time_steps)));
    init();
  }

  ~Impl() { details::FreeSolver(capsule_.get()); }

  void resetSolver(bool reset_qp_solver_mem) const {
    details::ResetSolver(capsule_.get(), reset_qp_solver_mem);
  }

  void setInitialState(const VectorCRef& initial_state) const {
    using details::MutData;
    ocp_nlp_constraints_model_set(cfg_, dims_, in_, 0, "lbx",
                                  MutData(initial_state));
    ocp_nlp_constraints_model_set(cfg_, dims_, in_, 0, "ubx",
                                  MutData(initial_state));
  }

  void setReference(int i, const VectorCRef& ref) const {
    using details::MutData;
    ocp_nlp_cost_model_set(cfg_, dims_, in_, i, "y_ref", MutData(ref));
  }

  void setTerminalReference(const VectorCRef& terminal_ref) const {
    using details::MutData;
    ocp_nlp_cost_model_set(cfg_, dims_, in_, num_mpc_nodes(), "y_ref",
                           MutData(terminal_ref));
  }

  void setCosts(const MatrixCRef& Q, const MatrixCRef& R) const {
    using details::MutData;
    details::CostType costs = details::CostType::Zero();
    costs.topLeftCorner<details::kCostSize, details::kCostSize>() = Q;
    costs.bottomRightCorner<details::kInputSize, details::kInputSize>() = R;
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      ocp_nlp_cost_model_set(cfg_, dims_, in_, i, "W", costs.data());
    }
    ocp_nlp_cost_model_set(cfg_, dims_, in_, num_mpc_nodes(), "W", MutData(Q));
  }

  void setCostWeights(const VectorCRef& q_weights,
                      const VectorCRef& r_weights) const {
    setCosts(details::StateCostType(q_weights.asDiagonal()),
             details::InputCostType(r_weights.asDiagonal()));
  }

  [[nodiscard]] bool setBounds(const VectorCRef& lbu,
                               const VectorCRef& ubu) const {
    using details::MutData;
    if ((lbu.array() > ubu.array()).any()) {
      throw AcadosWrapperException(
          "Some elements in lower bound are greater than corresponding "
          "elements "
          "in upper bound");
      return false;
    }

    for (int i = 0; i < num_mpc_nodes(); ++i) {
      ocp_nlp_constraints_model_set(cfg_, dims_, in_, i, "lbu", MutData(lbu));
      ocp_nlp_constraints_model_set(cfg_, dims_, in_, i, "ubu", MutData(ubu));
    }
    return true;
  }

  [[nodiscard]] Eigen::VectorXd getState(int i) const {
    details::StateType res;
    ocp_nlp_out_get(cfg_, dims_, out_, i, "x", res.data());
    return res;
  }

  [[nodiscard]] Eigen::VectorXd getInput(int i) const {
    details::InputType res;
    ocp_nlp_out_get(cfg_, dims_, out_, i, "u", res.data());
    return res;
  }

  [[nodiscard]] Eigen::MatrixXd getState() const {
    details::StateTrajectoryType predicted_states(
        static_cast<int>(details::kStateSize), num_mpc_nodes());
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      predicted_states.col(i) = getState(i);
    }
    return predicted_states;
  }

  [[nodiscard]] Eigen::MatrixXd getInput() const {
    details::InputTrajectoryType inputs(static_cast<int>(details::kInputSize),
                                        num_mpc_nodes());
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      inputs.col(i) = getInput(i);
    }
    return inputs;
  }

  void setReferenceState(const VectorCRef& state,
                         const VectorCRef& input) const {
    const details::RefType ref =
        (details::RefType() << state, input).finished();
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      setReference(i, ref);
    }
    setTerminalReference(state);
  }

  void setReferenceTrajectory(const MatrixCRef& state_ref,
                              const MatrixCRef& input_ref) const {
    const int n_x_samples = static_cast<int>(state_ref.cols());
    const int n_u_samples = static_cast<int>(input_ref.cols());
    if (n_x_samples != n_u_samples && n_x_samples != n_u_samples + 1) {
      throw AcadosWrapperException(
          "Number of state and input references do not match");
    }

    details::RefType ref;
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      const int i_state = std::min(n_x_samples - 1, i);
      const int i_input = std::min(n_u_samples - 1, i);
      ref << state_ref.col(i_state), input_ref.col(i_input);
      setReference(i, ref);
    }
    setTerminalReference(
        state_ref.col(std::min<int>(n_x_samples - 1, num_mpc_nodes())));
  }

  void setParameters(int i, const VectorCRef& params) const {
    details::SetParameters(capsule_.get(), i, details::MutData(params));
  }

  void setConstantParameters(const VectorCRef& params) const {
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      setParameters(i, params);
    }
  }

  void setPrintLevel(int value) const {
    ocp_nlp_solver_opts_set(cfg_, opts_, "print_level", &value);
  }

  [[nodiscard]] Eigen::VectorXd optimize(const VectorCRef& state) const {
    setInitialState(state);
    ACADOS_CHECK(details::Solve(capsule_.get()));
    return getInput(0);
  }

  void init() {
    using StateIdxs = Eigen::Matrix<int, details::kStateSize, 1>;
    using InputIdxs = Eigen::Matrix<int, details::kInputSize, 1>;

    cfg_ = details::GetConfig(capsule_.get());
    dims_ = details::GetDims(capsule_.get());
    in_ = details::GetInput(capsule_.get());
    out_ = details::GetOutput(capsule_.get());
    solver_ = details::GetSolver(capsule_.get());
    opts_ = details::GetOpts(capsule_.get());

    // Initialize the state constraint
    StateIdxs constrained_state_idx =
        StateIdxs::LinSpaced(0, details::kStateSize - 1);
    ocp_nlp_constraints_model_set(cfg_, dims_, in_, 0, "idxbx",
                                  constrained_state_idx.data());

    InputIdxs constrained_input_idx =
        InputIdxs::LinSpaced(0, details::kInputSize - 1);
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      ocp_nlp_constraints_model_set(cfg_, dims_, in_, i, "idxbu",
                                    constrained_input_idx.data());

      std::ignore = setBounds(-details::kNoBounds, details::kNoBounds);
    }

    // Initialize the output struct
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      setReference(i, details::RefType::Zero());
      setState(i, details::StateType::Zero());
    }
    setTerminalState(details::StateType::Zero());
  }

  void setState(int i, const VectorCRef& state) const {
    using details::MutData;
    ocp_nlp_out_set(cfg_, dims_, out_, i, "x", MutData(state));
  }

  void setTerminalState(const VectorCRef& terminal_state) const {
    setState(num_mpc_nodes(), terminal_state);
  }

  [[nodiscard]] double step_length(int i) const { return in_->Ts[i]; }

  [[nodiscard]] Eigen::VectorXd step_length() const {
    return Eigen::VectorXd::Map(in_->Ts, num_mpc_nodes());
  }

  [[nodiscard]] int num_mpc_nodes() const { return dims_->N; }

  using Capsule = std::unique_ptr<details::SolverCapsule, details::FreeCapsule>;

  Capsule capsule_;

  ocp_nlp_config* cfg_{nullptr};
  ocp_nlp_dims* dims_{nullptr};
  ocp_nlp_in* in_{nullptr};
  ocp_nlp_out* out_{nullptr};
  ocp_nlp_solver* solver_{nullptr};
  ocp_nlp_opts* opts_{nullptr};
};

MPCInterface::MPCInterface() : pimpl_(std::make_unique<Impl>()) {}

MPCInterface::MPCInterface(const VectorCRef& time_steps)
    : pimpl_(std::make_unique<Impl>(time_steps)) {}

MPCInterface::~MPCInterface() = default;

void MPCInterface::resetSolver(bool reset_qp_solver_mem) {
  pimpl_->resetSolver(reset_qp_solver_mem);
}

void MPCInterface::setInitialState(const VectorCRef& initial_state) {
  pimpl_->setInitialState(initial_state);
}

void MPCInterface::setReference(int i, const VectorCRef& ref) {
  pimpl_->setReference(i, ref);
}

void MPCInterface::setTerminalReference(const VectorCRef& terminal_ref) {
  pimpl_->setTerminalReference(terminal_ref);
}

void MPCInterface::setCosts(const MatrixCRef& Q, const MatrixCRef& R) {
  pimpl_->setCosts(Q, R);
}

void MPCInterface::setCostWeights(const VectorCRef& q_weights,
                                  const VectorCRef& r_weights) {
  pimpl_->setCostWeights(q_weights, r_weights);
}

bool MPCInterface::setBounds(const VectorCRef& lbu, const VectorCRef& ubu) {
  return pimpl_->setBounds(lbu, ubu);
}

Eigen::VectorXd MPCInterface::getState(int i) const {
  return pimpl_->getState(i);
}

Eigen::VectorXd MPCInterface::getInput(int i) const {
  return pimpl_->getInput(i);
}

Eigen::MatrixXd MPCInterface::getState() const { return pimpl_->getState(); }

Eigen::MatrixXd MPCInterface::getInput() const { return pimpl_->getInput(); }

void MPCInterface::setReferenceState(const VectorCRef& state,
                                     const VectorCRef& input) {
  pimpl_->setReferenceState(state, input);
}

void MPCInterface::setReferenceTrajectory(const MatrixCRef& state_ref,
                                          const MatrixCRef& input_ref) {
  pimpl_->setReferenceTrajectory(state_ref, input_ref);
}

void MPCInterface::setParameters(int i, const VectorCRef& params) {
  pimpl_->setParameters(i, params);
}

void MPCInterface::setConstantParameters(const VectorCRef& params) {
  pimpl_->setConstantParameters(params);
}

void MPCInterface::setPrintLevel(int value) { pimpl_->setPrintLevel(value); }

Eigen::VectorXd MPCInterface::optimize(const VectorCRef& state) {
  return pimpl_->optimize(state);
}

double MPCInterface::step_length(int i) const { return pimpl_->step_length(i); }

Eigen::VectorXd MPCInterface::step_length() const {
  return pimpl_->step_length();
}

int MPCInterface::num_mpc_nodes() const { return pimpl_->num_mpc_nodes(); }

}  // namespace fsc::control
