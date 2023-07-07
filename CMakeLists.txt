cmake_minimum_required(VERSION 3.14.0)
project(mpcpp VERSION 0.0.1)

set(CMAKE_DISABLE_FIND_PACKAGE_FortranLibs ON)
find_package(OpenMP REQUIRED)
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
add_library(mpcpp_solver ${CODEGEN_FILES} src/mpcpp/solver_wrapper.cpp)
add_dependencies(mpcpp_solver acados_python_codegen)
target_compile_definitions(
  mpcpp_solver PUBLIC "MODEL_NAME_LOWER=${MODEL_NAME_LOWER}"
                      "MODEL_NAME_UPPER=${MODEL_NAME_UPPER}")
target_include_directories(
  mpcpp_solver
  PUBLIC "$<BUILD_INTERFACE:${CODEGEN_DIR};${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(mpcpp_solver PUBLIC acados blasfeo hpipm)
add_library(mpcpp::solver ALIAS mpcpp_solver)
set_property(TARGET mpcpp_solver PROPERTY EXPORT_NAME solver)

# Step 3: Compile the OOPified MPC solver
add_library(mpcpp_acados_mpc src/mpcpp/acados_mpc.cpp)
target_include_directories(
  mpcpp_acados_mpc PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(mpcpp_acados_mpc PUBLIC mpcpp::solver Eigen3::Eigen)
set_property(TARGET mpcpp_acados_mpc PROPERTY EXPORT_NAME acados_mpc)
add_library(mpcpp::acados_mpc ALIAS mpcpp_acados_mpc)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

include(cmake/install-rules.cmake)
