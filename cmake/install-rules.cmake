if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR
      "include/quadrotor_mpc-${PROJECT_VERSION}"
      CACHE PATH "")
endif()

# Project is configured with no languages, so tell GNUInstallDirs the lib dir
set(CMAKE_INSTALL_LIBDIR
    lib
    CACHE PATH "")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package quadrotor_mpc)

install(
  DIRECTORY include/ ${CODEGEN_DIR}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT quadrotor_mpc_Development
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  TARGETS quadrotor_mpc_quadrotor_mpc quadrotor_mpc_solver
  EXPORT quadrotor_mpcTargets
  INCLUDES
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

write_basic_package_version_file(
  "${package}ConfigVersion.cmake" COMPATIBILITY SameMajorVersion
                                                ARCH_INDEPENDENT)

# Allow package maintainers to freely override the path for the configs
set(quadrotor_mpc_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_DATADIR}/${package}"
    CACHE PATH "CMake package config location relative to the install prefix")
mark_as_advanced(quadrotor_mpc_INSTALL_CMAKEDIR)

install(
  FILES cmake/install-config.cmake
  DESTINATION "${quadrotor_mpc_INSTALL_CMAKEDIR}"
  RENAME "${package}Config.cmake"
  COMPONENT quadrotor_mpc_Development)

install(
  FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
  DESTINATION "${quadrotor_mpc_INSTALL_CMAKEDIR}"
  COMPONENT quadrotor_mpc_Development)

install(
  EXPORT quadrotor_mpcTargets
  NAMESPACE quadrotor_mpc::
  DESTINATION "${quadrotor_mpc_INSTALL_CMAKEDIR}"
  COMPONENT quadrotor_mpc_Development)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
