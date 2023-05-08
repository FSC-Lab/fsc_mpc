find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(RapidJSON REQUIRED)
include(GoogleTest)

add_executable(test_quadrotor_mpc test_quadrotor_mpc.cpp)
target_compile_definitions(
  test_quadrotor_mpc
  PRIVATE TEST_DATA_FILE="${CMAKE_CURRENT_SOURCE_DIR}/test_data.json")
target_link_libraries(test_quadrotor_mpc PRIVATE quadrotor_mpc::quadrotor_mpc
                                                 GTest::GTest)

gtest_discover_tests(test_quadrotor_mpc)
