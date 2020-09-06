cmake_minimum_required(VERSION 3.10)
project(examples)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif ()

find_package(OpenCV REQUIRED)

# Hack to allow building the example with add_subdirectory
if (eyerec_FOUND)
else ()
    find_package(eyerec REQUIRED)
endif (eyerec_FOUND)

add_executable(track-pupil track-pupil.cpp)

target_link_libraries(track-pupil eyerec::eyerec ${OpenCV_LIBS})

# TODO: install the example properly
