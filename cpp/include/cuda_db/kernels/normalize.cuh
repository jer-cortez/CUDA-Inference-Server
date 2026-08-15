#pragma once

#include <cuda_runtime.h>

namespace cuda_db {

// Per-channel mean/stddev normalization over an NCHW tensor, in place.
//
// Only the host-side launcher is declared here -- no __global__ -- so plain
// .cpp translation units (engine/cuda_execution_engine.cpp) can call it
// without being routed through nvcc. The kernel itself lives in normalize.cu.
//
// `data` is `n * c * hw` floats laid out NCHW. `d_mean` and `d_stddev` are
// *device* pointers holding `c` floats each; they are read once per element
// and are expected to be uploaded once at engine construction rather than per
// call. stddev values must be non-zero -- that is validated host-side where
// the values are still readable (see CudaExecutionEngine's constructor),
// because checking here would cost a device-to-host copy on every batch.
//
// Asynchronous: the launch is queued on `stream` and this returns immediately.
// The caller must synchronize before reading `data` back.
//
// Throws CudaError if the dimensions are invalid or the launch fails.
// `n == 0` is a no-op, not an error -- an empty batch is a legal drain result.
void launch_normalize(float* data, int n, int c, int hw, const float* d_mean,
                      const float* d_stddev, cudaStream_t stream);

}  // namespace cuda_db
