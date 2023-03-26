// Copyright (c) 2023 hs293go
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef QUADROTOR_MPCPP_SOLVER_WRAPPER_HPP_
#define QUADROTOR_MPCPP_SOLVER_WRAPPER_HPP_

#include <acados/ocp_nlp/ocp_nlp_common.h>
#include <acados_c/ocp_nlp_interface.h>

#include <utility>

#include "acados_solver_quadrotor.h"

#ifndef MODEL_NAME_UPPER
#define MODEL_NAME_UPPER QUADROTOR
#endif

#ifndef MODEL_NAME_LOWER
#define MODEL_NAME_LOWER quadrotor
#endif

#define CAT_IMPL(A, B) A##B
#define CAT(A, B) CAT_IMPL(A, B)

#define ACADOS_PARAM(PARM) CAT(CAT(MODEL_NAME_UPPER, _), PARM)

#define ACADOS_OBJ(func) CAT(CAT(MODEL_NAME_LOWER, _), func)

enum class Dimensions {
  kStateSize = ACADOS_PARAM(NX),
  kInputSize = ACADOS_PARAM(NU),
  kRefSize = ACADOS_PARAM(NY),
  kEndRefSize = ACADOS_PARAM(NYN),
  kSamples = ACADOS_PARAM(N),
  kCostSize = ACADOS_PARAM(NY) - ACADOS_PARAM(NU),
  kParamSize = ACADOS_PARAM(NP)
};

namespace acadospp {
using SolverCapsule = ACADOS_OBJ(solver_capsule);

SolverCapsule* CreateCapsule();

int FreeCapsule(SolverCapsule* capsule);

int CreateSolver(SolverCapsule* capsule);

int FreeSolver(SolverCapsule* capsule);

int CreateSolverWithDiscretization(SolverCapsule* capsule, int n_time_steps,
                                   double* new_time_steps);

ocp_nlp_solver* GetSolver(SolverCapsule* capsule);

ocp_nlp_config* GetConfig(SolverCapsule* capsule);

ocp_nlp_dims* GetDims(SolverCapsule* capsule);

ocp_nlp_in* GetInput(SolverCapsule* capsule);

ocp_nlp_out* GetOutput(SolverCapsule* capsule);

int Solve(SolverCapsule* capsule);

void SetParameters(SolverCapsule* capsule, int stage, double* value);
};  // namespace acadospp

#endif  // QUADROTOR_MPCPP_SOLVER_WRAPPER_HPP_
