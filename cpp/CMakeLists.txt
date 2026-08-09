# Milestone 1: pure C++ scheduler/queue plus a stub engine, no CUDA/ONNX
# Runtime yet. Those get added to this file as later milestones land.
# cuda_db_core deliberately has no pybind11 dependency so it stays
# independently buildable and testable without Python.
add_library(cuda_db_core STATIC
    src/scheduler/request_queue.cpp
    src/scheduler/scheduler.cpp
    src/engine/execution_engine.cpp
)

target_include_directories(cuda_db_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(cuda_db_core PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(cuda_db_core PUBLIC Threads::Threads)

if(CUDA_DB_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# The Python extension is opt-in so the plain CMake + ctest loop keeps working
# on a machine with no Python development headers. scikit-build-core turns it
# on (see pyproject.toml).
if(CUDA_DB_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6  # first release series with Python 3.13 support
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(cuda_db_native src/bindings/module.cpp)
    target_link_libraries(cuda_db_native PRIVATE cuda_db_core)

    # Lands next to python/cuda_db/_native/__init__.py in the installed wheel.
    install(TARGETS cuda_db_native LIBRARY DESTINATION cuda_db/_native)
endif()
