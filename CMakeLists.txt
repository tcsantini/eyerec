cmake_minimum_required(VERSION 3.10)
project(er)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

include_directories(cpp/common)
include_directories(cpp/pupil_detection)
include_directories(cpp/pupil_tracking)

add_library(er
        cpp/common/ocv_utils.cpp
        cpp/pupil_detection/PupilDetectionMethod.cpp
        cpp/pupil_detection/PuRe.cpp
        cpp/pupil_tracking/PupilTrackingMethod.cpp
        cpp/pupil_tracking/PuReST.cpp
        )
target_link_libraries(er ${OpenCV_LIBS})

add_executable(track-pupil cpp/examples/track-pupil.cpp)
target_link_libraries(track-pupil er ${OpenCV_LIBS})

