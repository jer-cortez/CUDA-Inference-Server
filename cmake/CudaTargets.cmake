# Enables CUDA and exposes cuda_db_enable_cuda(<target>) to opt a target into
# it. Only included when CUDA_DB_ENABLE_CUDA is ON (see root CMakeLists.txt),
# so a machine with no nvcc never evaluates this file.

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 75=Turing (T4), 80/86=Ampere (A100/A10/30-series), 89=Ada (L4/40-series) --
# covers the GPU families available on the common cloud providers listed in
# docs/DEV_ENVIRONMENT.md without paying for architectures we don't hit.
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING
    "CUDA architectures to build device code for")

function(cuda_db_enable_cuda target)
    target_link_libraries(${target} PUBLIC CUDA::cudart)
    target_compile_definitions(${target} PUBLIC CUDA_DB_ENABLE_CUDA=1)
    # Needed on any translation unit that includes a CUDA Runtime header
    # (e.g. memory/device_buffer.hpp) without going through nvcc.
    target_include_directories(${target} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    # A target's CUDA_ARCHITECTURES property is initialized from
    # CMAKE_CUDA_ARCHITECTURES at the moment the target is *created*. Targets
    # declared before this module was included therefore capture an empty
    # value and fail at generate time ("CUDA_ARCHITECTURES is empty for
    # target ..."), so set it explicitly here rather than relying on when the
    # add_library/add_executable call happened to run.
    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endfunction()
