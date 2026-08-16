cmake_minimum_required(VERSION 3.24)
project(cuda_db LANGUAGES CXX)

# CUDA is a CMake *language*, enabled conditionally in cmake/CudaTargets.cmake
# rather than in the project() call above, so this file -- and every target
# not opted in via cuda_db_enable_cuda() -- still configures on a machine
# with no nvcc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(CUDA_DB_BUILD_TESTS "Build C++ unit tests" ON)
option(CUDA_DB_BUILD_PYTHON "Build the pybind11 extension module" OFF)
option(CUDA_DB_ENABLE_CUDA "Build CUDA memory/kernel/stream targets" OFF)

# enable_testing() must run at the top level, otherwise ctest invoked against
# the root build directory finds no tests registered by cpp/tests.
if(CUDA_DB_BUILD_TESTS)
    enable_testing()
endif()

# Included here, before any target exists, so CMAKE_CUDA_ARCHITECTURES is set
# by the time add_library/add_executable calls in cpp/ initialize their
# per-target CUDA_ARCHITECTURES property from it. cpp/CMakeLists.txt includes
# this module too, so that directory still configures standalone; the include
# is idempotent.
if(CUDA_DB_ENABLE_CUDA)
    include(CudaTargets)
endif()

add_subdirectory(cpp)
