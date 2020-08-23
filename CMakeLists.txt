cmake_minimum_required(VERSION 3.10)
project(er)

set(CMAKE_CXX_STANDARD 17)

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

add_executable(main cpp/examples/main.cpp)
target_link_libraries(main er ${OpenCV_LIBS})

