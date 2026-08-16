#include "cuda_db/engine/cuda_execution_engine.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

#include <cuda_runtime.h>

#include "cuda_db/kernels/normalize.cuh"
#include "cuda_db/memory/cuda_error.hpp"

namespace cuda_db {

/**
 * @brief Builds the engine with default ImageNet normalization constants.
 * @details Delegates to the Options constructor. Defined out of line because
 * the enclosing class is complete here, which is what lets `Options{}` pick
 * up its default member initializers.
 */
CudaExecutionEngine::CudaExecutionEngine() : CudaExecutionEngine(Options{}) {}

/**
 * @brief Builds the engine and uploads the normalization constants.
 * @details Validates mean/stddev host-side while the values are still
 * readable -- a zero stddev would otherwise divide to inf silently on device,
 * and catching it per batch would cost a device-to-host copy on the hot path.
 *
 * @param options Normalization constants, device id, and stream count.
 * @throws CudaError If mean/stddev are empty, differ in length, contain a
 * zero stddev, or the constants cannot be uploaded.
 */
CudaExecutionEngine::CudaExecutionEngine(Options options)
    : options_{std::move(options)},
      pool_{options_.device_id},
      streams_{options_.stream_count, options_.device_id},
      mean_{options_.mean.size() * sizeof(float)},
      stddev_{options_.stddev.size() * sizeof(float)} {
    if (options_.mean.empty()) {
        throw CudaError("CudaExecutionEngine: mean must have at least one channel");
    }
    if (options_.mean.size() != options_.stddev.size()) {
        throw CudaError("CudaExecutionEngine: mean and stddev must have the same length");
    }
    if (std::find(options_.stddev.begin(), options_.stddev.end(), 0.0F) !=
        options_.stddev.end()) {
        throw CudaError("CudaExecutionEngine: stddev entries must be non-zero");
    }

    const std::size_t bytes = options_.mean.size() * sizeof(float);
    cuda_check(cudaMemcpy(mean_.data(), options_.mean.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy (mean)");
    cuda_check(cudaMemcpy(stddev_.data(), options_.stddev.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy (stddev)");
}

/**
 * @brief Normalizes a batch on the GPU and returns the result to host memory.
 * @details The full device round trip, all queued on one stream so the stages
 * stay ordered without intermediate synchronization:
 *   1. copy the batch into pooled pinned staging (pageable memory would make
 *      the "async" copies below silently synchronous),
 *   2. async H2D into a pooled device buffer,
 *   3. launch the normalize kernel in place,
 *   4. async D2H back into the same staging buffer,
 *   5. synchronize once, at the end.
 *
 * The single synchronize before returning is also what makes releasing the
 * pooled handles safe: memory_pool.hpp requires no in-flight work touches a
 * block when its handle is destroyed, and the handles die at the end of this
 * scope.
 *
 * Output shaping mirrors StubExecutionEngine so this is a drop-in substitute:
 * `min(input_elems, output_elems)` floats per row are copied out and the
 * remainder of each output row stays zero.
 *
 * @param batched_input `batch_size * input_elems` floats, row-major NCHW rows.
 * @param batch_size Number of request rows in the batch.
 * @param input_elems Floats per input row; must divide evenly by the channel
 * count, since each row is one NCHW image.
 * @param output_elems Floats per output row.
 * @return std::vector<float> `batch_size * output_elems` floats, row-major.
 * @throws InferenceError If the input length or shape is inconsistent.
 * @throws CudaError If any device operation fails.
 */
std::vector<float> CudaExecutionEngine::run_inference(const std::vector<float>& batched_input,
                                                       std::size_t batch_size,
                                                       std::size_t input_elems,
                                                       std::size_t output_elems) {
    if (batched_input.size() != batch_size * input_elems) {
        throw InferenceError("batched input size does not match batch_size * input_elems");
    }
    if (batch_size == 0) {
        return {};
    }

    const std::size_t channel_count = channels();
    if (input_elems == 0 || input_elems % channel_count != 0) {
        throw InferenceError("input_elems must be a non-zero multiple of the channel count");
    }
    const std::size_t hw = input_elems / channel_count;

    const std::size_t bytes = batched_input.size() * sizeof(float);
    PooledBuffer staging = pool_.acquire_pinned(bytes);
    PooledBuffer device = pool_.acquire_device(bytes);

    std::memcpy(staging.data(), batched_input.data(), bytes);

    cudaStream_t stream = streams_.next();
    cuda_check(
        cudaMemcpyAsync(device.data(), staging.data(), bytes, cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync (H2D)");

    launch_normalize(device.as<float>(), static_cast<int>(batch_size),
                     static_cast<int>(channel_count), static_cast<int>(hw), mean_.as<float>(),
                     stddev_.as<float>(), stream);

    cuda_check(
        cudaMemcpyAsync(staging.data(), device.data(), bytes, cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync (D2H)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    std::vector<float> batched_output(batch_size * output_elems, 0.0F);
    const std::size_t copy_elems = std::min(input_elems, output_elems);
    const float* normalized = staging.as<float>();

    for (std::size_t i = 0; i < batch_size; ++i) {
        std::copy_n(normalized + i * input_elems, copy_elems,
                    batched_output.begin() + static_cast<std::ptrdiff_t>(i * output_elems));
    }

    return batched_output;
}

}  // namespace cuda_db
