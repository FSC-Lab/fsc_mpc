find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(Python COMPONENTS Interpreter)

include(GoogleTest)

# Step 1: Generate test data using python
set(TEST_DATA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/test_data.json)

add_custom_command(
  OUTPUT ${TEST_DATA_FILE}
  COMMAND
    PYTHONPATH=${CMAKE_SOURCE_DIR}/src:${CMAKE_SOURCE_DIR}/examples
    ${Python_EXECUTABLE} generate_test_data.py ${TEST_DATA_FILE}
    ${MPC_CODEGEN_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating test data"
  VERBATIM)

add_custom_target(
  generate_test_data
  DEPENDS ${TEST_DATA_FILE}
  COMMENT "Generating test data")
add_dependencies(generate_test_data acados_python_codegen)

# Step 2: Build the test
add_executable(test_fsc_mpc test_fsc_mpc.cpp)
add_dependencies(test_fsc_mpc generate_test_data)
target_include_directories(test_fsc_mpc PRIVATE include)
target_compile_definitions(test_fsc_mpc
                           PRIVATE TEST_DATA_FILE="${TEST_DATA_FILE}")
target_compile_features(test_fsc_mpc PRIVATE cxx_std_17)
target_link_libraries(test_fsc_mpc PRIVATE fsc_mpc::mpc_interface GTest::gtest
                                           GTest::gtest_main GTest::gmock)

gtest_discover_tests(test_fsc_mpc)

# Step 3: Build the benchmark
add_executable(benchmark_fsc_mpc benchmark_fsc_mpc.cpp)
target_link_libraries(
  benchmark_fsc_mpc PRIVATE benchmark::benchmark benchmark::benchmark_main
                            fsc_mpc::mpc_interface)
