# Locates a prebuilt ONNX Runtime GPU release and exposes it as the imported
# target onnxruntime::onnxruntime.
#
# ONNX Runtime is not built from source here -- that takes hours and needs its
# own toolchain. The expectation is the extracted upstream tarball
# (onnxruntime-linux-x64-gpu-<version>), pointed at by ONNXRUNTIME_ROOT as a
# CMake cache variable or an environment variable. docs/DEV_ENVIRONMENT.md
# already sets that on the GPU box.
#
# Only included when CUDA_DB_ENABLE_ONNX is ON, so a machine with no ONNX
# Runtime never evaluates this file.

if(TARGET onnxruntime::onnxruntime)
    return()
endif()

# Cache variable wins; fall back to the environment so an exported
# ONNXRUNTIME_ROOT from the shell just works without re-specifying -D.
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
        "Root of an extracted ONNX Runtime GPU release")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY
    FAIL_MESSAGE
        "ONNX Runtime not found. Download the prebuilt GPU release and point \
ONNXRUNTIME_ROOT at the extracted directory, e.g. \
-DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.19.2 . Naming the \
variable explicitly here because the alternative is a link error much later \
that says nothing about what is actually missing."
)

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
