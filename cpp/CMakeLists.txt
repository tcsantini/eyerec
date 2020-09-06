cmake_minimum_required(VERSION 3.10)
project(eyerec VERSION 0.0.0 LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Set the project's namespace
set(NAMESPACE "${PROJECT_NAME}::")

################################################################################
# Dependencies
################################################################################

find_package(OpenCV REQUIRED)

################################################################################
# Source Files
################################################################################

add_library(eyerec
        src/common/ocv_utils.cpp
        src/pupil/detection/PupilDetectionMethod.cpp
        src/pupil/detection/PuRe/PuRe.cpp
        src/pupil/tracking/PupilTrackingMethod.cpp
        src/pupil/tracking/PuReST/PuReST.cpp
        )

# Alias so it can be used within the build tree
add_library(${NAMESPACE}eyerec ALIAS eyerec)

# Public headers
target_include_directories(eyerec
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

# Private headers
target_include_directories(eyerec
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        )

# Linked libraries
target_link_libraries(eyerec ${OpenCV_LIBS})

################################################################################
# Installation
################################################################################

# Include and set some useful functions and variables first
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
set(PACKAGE_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake")
set(PACKAGE_CONFIG_VERSION "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}-targets")

# Install the public headers
install(
        DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )

# Export and write targets
install(
        TARGETS eyerec
        EXPORT "${TARGETS_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
install(
        EXPORT "${TARGETS_EXPORT_NAME}"
        NAMESPACE "${NAMESPACE}"
        DESTINATION "${CONFIG_INSTALL_DIR}"
)

# Create and install config files
configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in"
        "${PACKAGE_CONFIG}"
        INSTALL_DESTINATION "${CONFIG_INSTALL_DIR}"
)
write_basic_package_version_file(
        ${PACKAGE_CONFIG_VERSION}
        VERSION ${VERSION}
        COMPATIBILITY AnyNewerVersion
)
install(
        FILES "${PACKAGE_CONFIG}" "${PACKAGE_CONFIG_VERSION}"
        DESTINATION "${CONFIG_INSTALL_DIR}"
)
