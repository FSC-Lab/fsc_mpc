// Thin, Macro-driven wrappers adapting the auto-generated acados library
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

#include "fsc_mpc/solver_wrapper.hpp"

namespace fsc::control {
SolverCapsule* CreateCapsule() { return ACADOS_OBJ(acados_create_capsule)(); }

int CreateSolver(SolverCapsule* capsule) {
  return ACADOS_OBJ(acados_create)(capsule);
}

int FreeCapsule(SolverCapsule* capsule) {
  return ACADOS_OBJ(acados_free_capsule)(capsule);
}

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
  (capsule, stage, value, static_cast<int>(Dimensions::kParamSize));
}
}  // namespace fsc::control
