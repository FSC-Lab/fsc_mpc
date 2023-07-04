cmake_minimum_required(VERSION 3.14.0)
project(quadrotor_mpcpp VERSION 0.0.1)

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
add_library(quadrotor_mpc_solver ${CODEGEN_FILES}
                                 src/quadrotor_mpcpp/solver_wrapper.cpp)
add_dependencies(quadrotor_mpc_solver acados_python_codegen)
target_compile_definitions(
  quadrotor_mpc_solver PUBLIC "MODEL_NAME_LOWER=${MODEL_NAME_LOWER}"
                              "MODEL_NAME_UPPER=${MODEL_NAME_UPPER}")
target_include_directories(
  quadrotor_mpc_solver
  PUBLIC "$<BUILD_INTERFACE:${CODEGEN_DIR};${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(quadrotor_mpc_solver PUBLIC acados blasfeo hpipm)
add_library(quadrotor_mpc::solver ALIAS quadrotor_mpc_solver)
set_property(TARGET quadrotor_mpc_solver PROPERTY EXPORT_NAME solver)

# Step 3: Compile the OOPified MPC solver
add_library(quadrotor_mpc_quadrotor_mpc src/quadrotor_mpcpp/quadrotor_mpc.cpp)
target_include_directories(
  quadrotor_mpc_quadrotor_mpc
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(quadrotor_mpc_quadrotor_mpc PUBLIC quadrotor_mpc::solver
                                                         Eigen3::Eigen)
set_property(TARGET quadrotor_mpc_quadrotor_mpc PROPERTY EXPORT_NAME
                                                         quadrotor_mpc)
add_library(quadrotor_mpc::quadrotor_mpc ALIAS quadrotor_mpc_quadrotor_mpc)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

include(cmake/install-rules.cmake)
