cmake_minimum_required(VERSION 3.14.0)
project(fsc_mpc VERSION 0.0.1)

set(CMAKE_DISABLE_FIND_PACKAGE_FortranLibs ON)
find_package(OpenMP REQUIRED)
find_package(acados REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter)

set(CODEGEN_DIR
    ${CMAKE_SOURCE_DIR}/lib/${PROJECT_NAME}
    CACHE PATH "Path to generated files")
set(CODEGEN_FILES
    ${CODEGEN_DIR}/acados_solver_${PROJECT_NAME}.c
    ${CODEGEN_DIR}/${PROJECT_NAME}_model/${PROJECT_NAME}_expl_ode_fun.c
    ${CODEGEN_DIR}/${PROJECT_NAME}_model/${PROJECT_NAME}_expl_vde_adj.c
    ${CODEGEN_DIR}/${PROJECT_NAME}_model/${PROJECT_NAME}_expl_vde_forw.c)

set(MODEL_NAME_LOWER
    ${PROJECT_NAME}
    CACHE INTERNAL "")
string(TOUPPER ${MODEL_NAME_LOWER} MODEL_NAME_UPPER)

set(MPC_HORIZON
    2.0
    CACHE STRING "Prediction horizon of the MPC")

set(MPC_NUM_NODES
    20
    CACHE STRING "Number of shooting nodes of the MPC")

# Step 1: Generate the ACADOS solver using python
add_custom_command(
  OUTPUT ${CODEGEN_FILES}
  COMMAND
    PYTHONPATH=${CMAKE_SOURCE_DIR}/src ${Python_EXECUTABLE}
    build_quadrotor_model.py ${CODEGEN_DIR} --name ${PROJECT_NAME} --horizon
    ${MPC_HORIZON} --n_nodes ${MPC_NUM_NODES}
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/src/${PROJECT_NAME}"
  COMMENT "Generating Acados solver library"
  VERBATIM)

add_custom_target(
  acados_python_codegen
  DEPENDS ${CODEGEN_FILES}
  COMMENT "Generating Acados solver library")

# Step 2: Compile the generated source AND the cpp API wrapper
add_library(fsc_mpc_solver ${CODEGEN_FILES} src/fsc_mpc/solver_wrapper.cpp)
add_dependencies(fsc_mpc_solver acados_python_codegen)
target_compile_features(fsc_mpc_solver PUBLIC cxx_std_17)
target_compile_definitions(
  fsc_mpc_solver PUBLIC "MODEL_NAME_LOWER=${MODEL_NAME_LOWER}"
                        "MODEL_NAME_UPPER=${MODEL_NAME_UPPER}")
target_include_directories(
  fsc_mpc_solver
  PUBLIC "$<BUILD_INTERFACE:${CODEGEN_DIR};${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(fsc_mpc_solver PUBLIC acados blasfeo hpipm)
add_library(fsc_mpc::solver ALIAS fsc_mpc_solver)
set_property(TARGET fsc_mpc_solver PROPERTY EXPORT_NAME solver)
set_property(TARGET fsc_mpc_solver PROPERTY POSITION_INDEPENDENT_CODE ON)

# Step 3: Compile the OOPified MPC solver
add_library(fsc_mpc_mpc_interface src/fsc_mpc/mpc_interface.cpp)
target_include_directories(
  fsc_mpc_mpc_interface
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_compile_features(fsc_mpc_mpc_interface PUBLIC cxx_std_17)
target_link_libraries(fsc_mpc_mpc_interface PUBLIC fsc_mpc::solver
                                                   Eigen3::Eigen)
set_property(TARGET fsc_mpc_mpc_interface PROPERTY EXPORT_NAME mpc_interface)
add_library(fsc_mpc::mpc_interface ALIAS fsc_mpc_mpc_interface)
set_property(TARGET fsc_mpc_mpc_interface PROPERTY POSITION_INDEPENDENT_CODE ON)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

include(cmake/install-rules.cmake)
