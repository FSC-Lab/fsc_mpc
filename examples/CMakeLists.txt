find_package(fscore REQUIRED)

add_executable(stabilization_example stabilization_example.cpp)
target_link_libraries(stabilization_example PRIVATE fscore::fscore
                                                    quadrotor_mpc)
