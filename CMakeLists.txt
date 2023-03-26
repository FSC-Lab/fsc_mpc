project(quadrotor_mpcpp)
cmake_minimum_required(VERSION 3.14.0)

set(CODEGEN_DIR ${CMAKE_SOURCE_DIR}/lib/c_generated_code)
option(BUILD_EXAMPLES ON)

set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

find_package(acados REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(
  quadrotor
  ${CODEGEN_DIR}/quadrotor_model/quadrotor_expl_ode_fun.c
  ${CODEGEN_DIR}/quadrotor_model/quadrotor_expl_vde_adj.c
  ${CODEGEN_DIR}/quadrotor_model/quadrotor_expl_vde_forw.c)

add_library(solver_wrapper ${CODEGEN_DIR}/acados_solver_quadrotor.c
                           src/quadrotor_mpcpp/solver_wrapper.cpp)
target_include_directories(solver_wrapper PUBLIC ${CODEGEN_DIR})
target_link_libraries(solver_wrapper PUBLIC acados blasfeo hpipm quadrotor)

add_library(quadrotor_mpc src/quadrotor_mpcpp/quadrotor_mpc.cpp)
target_include_directories(quadrotor_mpc PUBLIC include)
target_link_libraries(quadrotor_mpc PUBLIC Eigen3::Eigen solver_wrapper)

include(CTest)
enable_testing()
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
