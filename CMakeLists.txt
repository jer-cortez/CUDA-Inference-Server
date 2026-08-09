cmake_minimum_required(VERSION 3.24)
project(cuda_db LANGUAGES CXX)

# CUDA language support is enabled once kernel work begins (a later
# milestone); left out for now so this builds on machines without nvcc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CUDA_DB_BUILD_TESTS "Build C++ unit tests" ON)
option(CUDA_DB_BUILD_PYTHON "Build the pybind11 extension module" OFF)

# enable_testing() must run at the top level, otherwise ctest invoked against
# the root build directory finds no tests registered by cpp/tests.
if(CUDA_DB_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(cpp)
