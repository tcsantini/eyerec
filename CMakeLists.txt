cmake_minimum_required(VERSION 3.10)

project(eyerec)

add_subdirectory(cpp)
# Let's signal downstream that we already have the eyerec target
set(eyerec_FOUND 1)

add_subdirectory(cpp/examples)
