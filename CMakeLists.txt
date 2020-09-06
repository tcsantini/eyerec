cmake_minimum_required(VERSION 3.10)

add_subdirectory(cpp)
# Let's signal upstream that we already have the eyerec target
set(eyerec_FOUND 1)

add_subdirectory(cpp/examples)
