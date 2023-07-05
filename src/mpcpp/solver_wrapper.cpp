#include "mpcpp/solver_wrapper.hpp"

namespace acadospp {
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

int Solve(SolverCapsule* capsule) { return ACADOS_OBJ(acados_solve)(capsule); }

void SetParameters(SolverCapsule* capsule, int stage, double* value) {
  ACADOS_OBJ(acados_update_params)
  (capsule, stage, value, static_cast<int>(Dimensions::kParamSize));
}
}  // namespace acadospp
