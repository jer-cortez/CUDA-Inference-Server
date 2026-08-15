#include "cuda_db/kernels/cuda_stream_pool.hpp"

#include <stdexcept>

#include "cuda_db/memory/cuda_error.hpp"

namespace cuda_db {

CudaStreamPool::CudaStreamPool(std::size_t count, int device_id) : device_id_{device_id} {
    if (count == 0) {
        throw std::invalid_argument("CudaStreamPool: count must be at least 1");
    }
    cuda_check(cudaSetDevice(device_id_), "cudaSetDevice");

    streams_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        cudaStream_t stream = nullptr;
        // Non-blocking: these streams never implicitly synchronize with the
        // default stream, so work on one doesn't stall behind unrelated work
        // queued elsewhere in the process.
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
        streams_.push_back(stream);
    }
}

CudaStreamPool::~CudaStreamPool() {
    for (cudaStream_t stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

cudaStream_t CudaStreamPool::next() noexcept {
    const std::size_t index = cursor_.fetch_add(1, std::memory_order_relaxed) % streams_.size();
    return streams_[index];
}

cudaStream_t CudaStreamPool::at(std::size_t index) const {
    if (index >= streams_.size()) {
        throw std::out_of_range("CudaStreamPool::at: index out of range");
    }
    return streams_[index];
}

void CudaStreamPool::synchronize_all() {
    for (cudaStream_t stream : streams_) {
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

}  // namespace cuda_db
