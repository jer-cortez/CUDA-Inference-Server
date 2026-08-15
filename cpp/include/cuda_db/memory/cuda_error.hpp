#pragma once

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace cuda_db {

// Raised by every CUDA Runtime call this project wraps. Kept distinct from
// InferenceError (engine/execution_engine.hpp) because a device failure --
// OOM, an invalid device id, a driver problem -- is a different failure mode
// from a malformed request or a model-level error, even though both
// ultimately surface to Python as catchable exceptions.
struct CudaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Throws CudaError with cudaGetErrorString(status) prefixed by `what` if
// status is not cudaSuccess. Every DeviceBuffer/PinnedBuffer/MemoryPool call
// into the CUDA Runtime is wrapped in this rather than checked ad hoc, so a
// failure always carries both what we were trying to do and what the driver
// said went wrong.
inline void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw CudaError(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace cuda_db
