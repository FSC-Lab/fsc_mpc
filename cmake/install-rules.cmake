if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR
      "include/fsc_mpc-${PROJECT_VERSION}"
      CACHE PATH "")
endif()

# Project is configured with no languages, so tell GNUInstallDirs the lib dir
set(CMAKE_INSTALL_LIBDIR
    lib
    CACHE PATH "")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package fsc_mpc)

install(
  DIRECTORY include/ ${CODEGEN_DIR}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT fsc_mpc_Development
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  TARGETS fsc_mpc_mpc_interface fsc_mpc_solver
  EXPORT fsc_mpcTargets
  INCLUDES
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

write_basic_package_version_file(
  "${package}ConfigVersion.cmake" COMPATIBILITY SameMajorVersion
                                                ARCH_INDEPENDENT)

# Allow package maintainers to freely override the path for the configs
set(fsc_mpc_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_DATADIR}/${package}"
    CACHE PATH "CMake package config location relative to the install prefix")
mark_as_advanced(fsc_mpc_INSTALL_CMAKEDIR)

install(
  FILES cmake/install-config.cmake
  DESTINATION "${fsc_mpc_INSTALL_CMAKEDIR}"
  RENAME "${package}Config.cmake"
  COMPONENT fsc_mpc_Development)

install(
  FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
  DESTINATION "${fsc_mpc_INSTALL_CMAKEDIR}"
  COMPONENT fsc_mpc_Development)

install(
  EXPORT fsc_mpcTargets
  NAMESPACE fsc_mpc::
  DESTINATION "${fsc_mpc_INSTALL_CMAKEDIR}"
  COMPONENT fsc_mpc_Development)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
