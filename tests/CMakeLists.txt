find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(fscore REQUIRED)

include(GoogleTest)

# Step 1: Generate test data using python

set(TEST_DATA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/test_data.json)

add_custom_command(
  OUTPUT ${TEST_DATA_FILE}
  COMMAND PYTHONPATH=${CMAKE_SOURCE_DIR}/src ${Python_EXECUTABLE}
          generate_test_data.py ${TEST_DATA_FILE} ${CODEGEN_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating test data"
  VERBATIM)

add_custom_target(
  generate_test_data
  DEPENDS ${TEST_DATA_FILE}
  COMMENT "Generating test data")

add_executable(test_quadrotor_mpc test_quadrotor_mpc.cpp)
add_dependencies(test_quadrotor_mpc generate_test_data)
target_compile_definitions(test_quadrotor_mpc
                           PRIVATE TEST_DATA_FILE="${TEST_DATA_FILE}")
target_compile_features(test_quadrotor_mpc PRIVATE cxx_std_17)
target_link_libraries(
  test_quadrotor_mpc PRIVATE quadrotor_mpc::quadrotor_mpc GTest::gtest
                             GTest::gtest_main GTest::gmock fscore::fscore)

gtest_discover_tests(test_quadrotor_mpc)
