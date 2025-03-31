cmake_minimum_required(VERSION 3.8)
project(foundationpose_cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(easy_deploy_tool)
add_subdirectory(detection_6d_foundationpose)

add_subdirectory(simple_tests)
