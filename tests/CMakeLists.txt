find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(fscore REQUIRED)

include(GoogleTest)

add_executable(test_quadrotor_mpc test_quadrotor_mpc.cpp)
target_compile_definitions(
  test_quadrotor_mpc
  PRIVATE TEST_DATA_FILE="${CMAKE_CURRENT_SOURCE_DIR}/test_data.json")
target_compile_features(test_quadrotor_mpc PRIVATE cxx_std_17)
target_link_libraries(test_quadrotor_mpc PRIVATE quadrotor_mpc::quadrotor_mpc
                                                 GTest::GTest fscore::fscore)

gtest_discover_tests(test_quadrotor_mpc)
