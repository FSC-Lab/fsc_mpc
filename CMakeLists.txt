project(quadrotor_mpcpp)
cmake_minimum_required(VERSION 3.14.0)

find_package(acados REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter)

set(CODEGEN_DIR ${CMAKE_SOURCE_DIR}/lib/${PROJECT_NAME})
set(CODEGEN_FILES
    ${CODEGEN_DIR}/acados_solver_${PROJECT_NAME}.c
    ${CODEGEN_DIR}/${PROJECT_NAME}_model/${PROJECT_NAME}_expl_ode_fun.c
    ${CODEGEN_DIR}/${PROJECT_NAME}_model/${PROJECT_NAME}_expl_vde_adj.c
    ${CODEGEN_DIR}/${PROJECT_NAME}_model/${PROJECT_NAME}_expl_vde_forw.c)

set(MODEL_NAME_LOWER ${PROJECT_NAME})
string(TOUPPER ${MODEL_NAME_LOWER} MODEL_NAME_UPPER)

# Step 1: Generate the ACADOS solver using python
add_custom_command(
  OUTPUT ${CODEGEN_FILES}
  COMMAND PYTHONPATH=${CMAKE_SOURCE_DIR}/src ${Python_EXECUTABLE}
          build_quadrotor_model.py ${CODEGEN_DIR} --name ${PROJECT_NAME}
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/src/${PROJECT_NAME}"
  COMMENT "Generating Acados solver library"
  VERBATIM)

add_custom_target(
  acados_python_codegen
  DEPENDS ${CODEGEN_FILES}
  COMMENT "Generating Acados solver library")

# Step 2: Compile the generated source AND the cpp API wrapper
add_library(solver_wrapper ${CODEGEN_FILES}
                           src/quadrotor_mpcpp/solver_wrapper.cpp)
add_dependencies(solver_wrapper acados_python_codegen)
target_compile_definitions(
  solver_wrapper PUBLIC "MODEL_NAME_LOWER=${MODEL_NAME_LOWER}"
                        "MODEL_NAME_UPPER=${MODEL_NAME_UPPER}")
target_include_directories(solver_wrapper PUBLIC ${CODEGEN_DIR} include)
target_link_libraries(solver_wrapper PUBLIC acados blasfeo hpipm)

# Step 3: Compile the OOPified MPC solver
add_library(quadrotor_mpc_quadrotor_mpc src/quadrotor_mpcpp/quadrotor_mpc.cpp)
target_include_directories(quadrotor_mpc_quadrotor_mpc PUBLIC include)
target_link_libraries(quadrotor_mpc_quadrotor_mpc PUBLIC Eigen3::Eigen
                                                         solver_wrapper)

add_library(quadrotor_mpc::quadrotor_mpc ALIAS quadrotor_mpc_quadrotor_mpc)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
