#include "cuda_db/kernels/normalize.cuh"

#include "cuda_db/memory/cuda_error.hpp"

namespace cuda_db {
namespace {

constexpr int kBlockSize = 256;

// Grid is capped rather than sized to the input: the grid-stride loop below
// lets any block count cover any element count, so a batch of 1 and a batch
// of 64 launch the same shape and we never ask the driver for an absurd grid.
constexpr long long kMaxBlocks = 1024;

/**
 * @brief In-place per-channel normalization over an NCHW tensor.
 * @details Grid-stride loop over all `total` elements. The channel of a flat
 * NCHW index is `(idx / hw) % c`: dividing by the per-channel plane size `hw`
 * gives the plane number, and taking that mod `c` discards the batch
 * component, leaving the channel. Reciprocal is not precomputed -- `stddev`
 * lives in device memory and is re-read per element, which the L1/read-only
 * cache absorbs since every thread in a warp hits the same few channels.
 */
__global__ void normalize_kernel(float* data, long long total, int c, int hw, const float* mean,
                                 const float* stddev) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (; idx < total; idx += stride) {
        const int channel = static_cast<int>((idx / hw) % c);
        data[idx] = (data[idx] - mean[channel]) / stddev[channel];
    }
}

}  // namespace

/**
 * @brief Queues per-channel normalization of `data` on `stream`.
 * @details Validates shape and pointers host-side, then launches
 * normalize_kernel with a capped grid. Checks cudaGetLastError() immediately
 * after the launch: a bad launch configuration fails asynchronously and would
 * otherwise surface much later as an unrelated error, so it is converted to a
 * CudaError here while the cause is still obvious.
 *
 * @param data Device pointer to `n * c * hw` floats, NCHW, modified in place.
 * @param n Batch size. Zero is a no-op.
 * @param c Channel count; must match the length of `d_mean`/`d_stddev`.
 * @param hw Elements per channel plane (height * width).
 * @param d_mean Device pointer to `c` per-channel means.
 * @param d_stddev Device pointer to `c` per-channel standard deviations, all
 * non-zero (validated by the caller, where the host-side values are visible).
 * @param stream Stream to queue the launch on.
 * @throws CudaError If dimensions are negative, `c`/`hw` are zero for a
 * non-empty batch, a required pointer is null, or the launch fails.
 */
void launch_normalize(float* data, int n, int c, int hw, const float* d_mean,
                      const float* d_stddev, cudaStream_t stream) {
    if (n < 0 || c < 0 || hw < 0) {
        throw CudaError("launch_normalize: dimensions must be non-negative");
    }
    if (n == 0) {
        return;  // empty batch: nothing to normalize, not an error
    }
    if (c == 0 || hw == 0) {
        throw CudaError("launch_normalize: c and hw must be non-zero for a non-empty batch");
    }
    if (data == nullptr || d_mean == nullptr || d_stddev == nullptr) {
        throw CudaError("launch_normalize: null device pointer");
    }

    const long long total = static_cast<long long>(n) * c * hw;
    const long long needed = (total + kBlockSize - 1) / kBlockSize;
    const long long blocks = needed < kMaxBlocks ? needed : kMaxBlocks;

    normalize_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
        data, total, c, hw, d_mean, d_stddev);

    cuda_check(cudaGetLastError(), "normalize_kernel launch");
}

}  // namespace cuda_db
