# Milestone 1: pure C++ scheduler/queue, no CUDA/pybind11/ONNX Runtime yet.
# Those get added to this file as later milestones land.
add_library(cuda_db_core STATIC
    src/scheduler/request_queue.cpp
    src/scheduler/scheduler.cpp
)

target_include_directories(cuda_db_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(cuda_db_core PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(cuda_db_core PUBLIC Threads::Threads)

option(CUDA_DB_BUILD_TESTS "Build C++ unit tests" ON)

if(CUDA_DB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
