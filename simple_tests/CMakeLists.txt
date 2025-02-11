cmake_minimum_required(VERSION 3.8)
project(simple_tests)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)

set(source_file 
  src/test_foundationpose.cpp
)

enable_testing()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} ${source_file})   

target_link_libraries(${PROJECT_NAME} PUBLIC 
  GTest::gtest_main
  glog::glog
  ${OpenCV_LIBS}
  deploy_core
  trt_core
  detection_6d_foundationpose
)



target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)