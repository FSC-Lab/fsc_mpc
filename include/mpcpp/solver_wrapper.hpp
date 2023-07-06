// Copyright (c) 2023 hs293go
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef MPCPP_SOLVER_WRAPPER_HPP_
#define MPCPP_SOLVER_WRAPPER_HPP_

#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados_c/ocp_nlp_interface.h"
#include "mpcpp/internal.hpp"

#ifndef MODEL_NAME_UPPER
#error MISSING DEFINITION FOR MODEL_NAME_UPPER
#endif

#ifndef MODEL_NAME_LOWER
#error MISSING DEFINITION FOR MODEL_NAME_LOWER
#endif

#define SOLVER_LIB STRINGIFY(CAT(acados_solver_, MODEL_NAME_LOWER).h)

#include SOLVER_LIB

#define ACADOS_PARAM(PARM) CAT(CAT(MODEL_NAME_UPPER, _), PARM)

#define ACADOS_OBJ(func) CAT(CAT(MODEL_NAME_LOWER, _), func)

enum class Dimensions {
  kStateSize = ACADOS_PARAM(NX),
  kInputSize = ACADOS_PARAM(NU),
  kRefSize = ACADOS_PARAM(NY),
  kEndRefSize = ACADOS_PARAM(NYN),
  kSamples = ACADOS_PARAM(N),
  kCostSize = ACADOS_PARAM(NY) - ACADOS_PARAM(NU),
  kBoundsSize = ACADOS_PARAM(NBU),
  kParamSize = ACADOS_PARAM(NP)
};

namespace acadospp {
using SolverCapsule = ACADOS_OBJ(solver_capsule);

SolverCapsule* CreateCapsule();

int FreeCapsule(SolverCapsule* capsule);

int CreateSolver(SolverCapsule* capsule);

int FreeSolver(SolverCapsule* capsule);

int ResetSolver(SolverCapsule* capsule, bool reset_qp_solver_mem);

int CreateSolverWithDiscretization(SolverCapsule* capsule, int n_time_steps,
                                   double* new_time_steps);

ocp_nlp_solver* GetSolver(SolverCapsule* capsule);

ocp_nlp_config* GetConfig(SolverCapsule* capsule);

ocp_nlp_dims* GetDims(SolverCapsule* capsule);

ocp_nlp_in* GetInput(SolverCapsule* capsule);

ocp_nlp_out* GetOutput(SolverCapsule* capsule);

ocp_nlp_opts* GetOpts(SolverCapsule* capsule);

int Solve(SolverCapsule* capsule);

void SetParameters(SolverCapsule* capsule, int stage, double* value);
}  // namespace acadospp

#endif  // MPCPP_SOLVER_WRAPPER_HPP_
