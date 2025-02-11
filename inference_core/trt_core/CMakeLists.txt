cmake_minimum_required(VERSION 3.8)
project(trt_core)


add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)


find_package(CUDA REQUIRED)
find_package(glog REQUIRED)

set(source_file src/trt_core.cpp
                src/trt_core_factory.cpp)


include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED ${source_file}) 


target_link_libraries(${PROJECT_NAME} PUBLIC
  ${CUDA_LIBRARIES}
  nvinfer 
  nvonnxparser
  deploy_core
)


target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)