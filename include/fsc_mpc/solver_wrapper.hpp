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

#ifndef FSC_MPC_SOLVER_WRAPPER_HPP_
#define FSC_MPC_SOLVER_WRAPPER_HPP_

#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados_c/ocp_nlp_interface.h"
#include "fsc_mpc/internal.hpp"

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

namespace fsc::control {

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
}  // namespace fsc::control

#endif  // FSC_MPC_SOLVER_WRAPPER_HPP_
