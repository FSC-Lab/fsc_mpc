include(CMakeFindDependencyMacro)
find_dependency(Eigen3)
find_dependency(acados)

include("${CMAKE_CURRENT_LIST_DIR}/quadrotor_mpcTargets.cmake")
