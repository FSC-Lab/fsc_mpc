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

#include "fsc_mpc/internal.hpp"
#include "fsc_mpc/solver_wrapper.hpp"

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

const BoundsType kNoBounds = BoundsType::Constant(1e50);

struct MPCInterface::Impl {
  Impl() : capsule_(CreateCapsule()) {
    ACADOS_CHECK(CreateSolver(capsule_.get()));
    init();
  }

  explicit Impl(const VectorCRef& time_steps) : capsule_(CreateCapsule()) {
    using details::MutData;
    ACADOS_CHECK(CreateSolverWithDiscretization(
        capsule_.get(), static_cast<int>(time_steps.size()),
        MutData(time_steps)));
    init();
  }

  ~Impl() { FreeSolver(capsule_.get()); }

  void resetSolver(bool reset_qp_solver_mem) const {
    ResetSolver(capsule_.get(), reset_qp_solver_mem);
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
    CostType costs = CostType::Zero();
    costs.topLeftCorner<kCostSize, kCostSize>() = Q;
    costs.bottomRightCorner<kInputSize, kInputSize>() = R;
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      ocp_nlp_cost_model_set(cfg_, dims_, in_, i, "W", costs.data());
    }
    ocp_nlp_cost_model_set(cfg_, dims_, in_, num_mpc_nodes(), "W", MutData(Q));
  }

  void setCostWeights(const VectorCRef& q_weights,
                      const VectorCRef& r_weights) const {
    setCosts(StateCostType(q_weights.asDiagonal()),
             InputCostType(r_weights.asDiagonal()));
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
    StateType res;
    ocp_nlp_out_get(cfg_, dims_, out_, i, "x", res.data());
    return res;
  }

  [[nodiscard]] Eigen::VectorXd getInput(int i) const {
    InputType res;
    ocp_nlp_out_get(cfg_, dims_, out_, i, "u", res.data());
    return res;
  }

  [[nodiscard]] Eigen::MatrixXd getState() const {
    StateTrajectoryType predicted_states(static_cast<int>(kStateSize),
                                         num_mpc_nodes());
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      predicted_states.col(i) = getState(i);
    }
    return predicted_states;
  }

  [[nodiscard]] Eigen::MatrixXd getInput() const {
    InputTrajectoryType inputs(static_cast<int>(kInputSize), num_mpc_nodes());
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      inputs.col(i) = getInput(i);
    }
    return inputs;
  }

  void setReferenceState(const VectorCRef& state,
                         const VectorCRef& input) const {
    const RefType ref = (RefType() << state, input).finished();
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

  void setParameters(int i, const VectorCRef& params) const {
    SetParameters(capsule_.get(), i, details::MutData(params));
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
    ACADOS_CHECK(Solve(capsule_.get()));
    return getInput(0);
  }

  void init() {
    using StateIdxs = Eigen::Matrix<int, kStateSize, 1>;
    using InputIdxs = Eigen::Matrix<int, kInputSize, 1>;

    cfg_ = GetConfig(capsule_.get());
    dims_ = GetDims(capsule_.get());
    in_ = GetInput(capsule_.get());
    out_ = GetOutput(capsule_.get());
    solver_ = GetSolver(capsule_.get());
    opts_ = GetOpts(capsule_.get());

    // Initialize the state constraint
    StateIdxs constrained_state_idx = StateIdxs::LinSpaced(0, kStateSize - 1);
    ocp_nlp_constraints_model_set(cfg_, dims_, in_, 0, "idxbx",
                                  constrained_state_idx.data());

    InputIdxs constrained_input_idx = InputIdxs::LinSpaced(0, kInputSize - 1);
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      ocp_nlp_constraints_model_set(cfg_, dims_, in_, i, "idxbu",
                                    constrained_input_idx.data());

      std::ignore = setBounds(-kNoBounds, kNoBounds);
    }

    // Initialize the output struct
    for (int i = 0; i < num_mpc_nodes(); ++i) {
      setReference(i, RefType::Zero());
      setState(i, StateType::Zero());
    }
    setTerminalState(StateType::Zero());
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

  using Capsule = details::Handle<SolverCapsule, FreeCapsule>;

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
