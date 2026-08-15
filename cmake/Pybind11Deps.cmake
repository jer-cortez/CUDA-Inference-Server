# Fetches pybind11. Split out of cpp/CMakeLists.txt so CUDA_DB_BUILD_PYTHON's
# guarded block stays a plain include() rather than a FetchContent block
# growing in place alongside the CUDA/ONNX Runtime equivalents.

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6  # first release series with Python 3.13 support
)
FetchContent_MakeAvailable(pybind11)
