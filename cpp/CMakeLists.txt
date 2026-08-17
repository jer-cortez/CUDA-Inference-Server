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

# Required because this static library gets linked into cuda_db_native, a
# shared module. On Linux/GCC a static library is built non-PIC by default, and
# the link then fails with "relocation R_X86_64_PC32 ... can not be used when
# making a shared object". macOS does not catch this -- Darwin compiles
# position-independent by default -- so it only appears on the GPU box.
# Set unconditionally rather than only under CUDA_DB_BUILD_PYTHON: the cost on
# x86-64 is negligible and it keeps the plain-CMake and pip builds consistent.
set_target_properties(cuda_db_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
target_link_libraries(cuda_db_core PUBLIC Threads::Threads)

# Memory primitives, the stream pool, and the CUDA engine are plain .cpp
# compiled against the CUDA Runtime headers; only files containing __global__
# code (normalize.cu) go through nvcc, which keeps incremental build times
# down. CMake routes .cu sources to nvcc automatically once CudaTargets.cmake
# has called enable_language(CUDA), so no separate kernels library is needed.
# Guarded so a machine with no CUDAToolkit (e.g. the Mac dev loop) never sees
# these.
if(CUDA_DB_ENABLE_CUDA)
    include(CudaTargets)

    target_sources(cuda_db_core PRIVATE
        src/memory/device_buffer.cpp
        src/memory/pinned_buffer.cpp
        src/memory/memory_pool.cpp
        src/kernels/cuda_stream_pool.cpp
        src/kernels/normalize.cu
        src/engine/cuda_execution_engine.cpp
    )
    cuda_db_enable_cuda(cuda_db_core)
endif()

# The ONNX Runtime engine is a plain .cpp -- ORT is consumed as a prebuilt
# shared library, so nothing here goes through nvcc. Independent of
# CUDA_DB_ENABLE_CUDA: ORT's CUDA execution provider lives inside libonnxruntime
# and is selected at session-creation time, so this target does not need our
# own CUDA plumbing to run on the GPU.
if(CUDA_DB_ENABLE_ONNX)
    find_package(ONNXRuntime REQUIRED)

    target_sources(cuda_db_core PRIVATE
        src/engine/onnx_execution_engine.cpp
    )
    target_link_libraries(cuda_db_core PUBLIC onnxruntime::onnxruntime)
    target_compile_definitions(cuda_db_core PUBLIC CUDA_DB_ENABLE_ONNX=1)

    # Diagnostic: times the engine with no scheduler/HTTP/Python in the path, to
    # separate "batching does not pay on this model" from "something above the
    # engine is the bottleneck".
    add_executable(bench_engine tools/bench_engine.cpp)
    target_link_libraries(bench_engine PRIVATE cuda_db_core)
    get_filename_component(CUDA_DB_ORT_LIBDIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    set_target_properties(bench_engine PROPERTIES
        BUILD_RPATH "${CUDA_DB_ORT_LIBDIR}")
endif()

if(CUDA_DB_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# The Python extension is opt-in so the plain CMake + ctest loop keeps working
# on a machine with no Python development headers. scikit-build-core turns it
# on (see pyproject.toml).
if(CUDA_DB_BUILD_PYTHON)
    include(Pybind11Deps)

    pybind11_add_module(cuda_db_native src/bindings/module.cpp)
    target_link_libraries(cuda_db_native PRIVATE cuda_db_core)

    if(CUDA_DB_ENABLE_ONNX)
        # libonnxruntime.so lives in an extracted tarball outside the loader's
        # default search path, so without an RPATH the extension builds fine and
        # then fails at import with "libonnxruntime.so.1: cannot open shared
        # object file". Baking the directory in avoids making every caller
        # remember to set LD_LIBRARY_PATH.
        get_filename_component(CUDA_DB_ORT_LIBDIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
        set_target_properties(cuda_db_native PROPERTIES
            BUILD_RPATH "${CUDA_DB_ORT_LIBDIR}"
            INSTALL_RPATH "${CUDA_DB_ORT_LIBDIR}")
    endif()

    # Lands next to python/cuda_db/_native/__init__.py in the installed wheel.
    install(TARGETS cuda_db_native LIBRARY DESTINATION cuda_db/_native)
endif()
