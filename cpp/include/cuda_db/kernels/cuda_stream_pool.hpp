#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

namespace cuda_db {

// Owns a fixed set of CUDA streams and round-robins callers across them.
// Exists so concurrent batches (or a batch's own upload/compute/download
// stages, once IO binding lands) can overlap on the GPU instead of
// serializing on the default stream. Owns streams only -- never memory; see
// MemoryPool for that.
class CudaStreamPool {
public:
    explicit CudaStreamPool(std::size_t count = 2, int device_id = 0);
    ~CudaStreamPool();

    CudaStreamPool(const CudaStreamPool&) = delete;
    CudaStreamPool& operator=(const CudaStreamPool&) = delete;

    // Round-robin, lock-free: safe to call from multiple threads, though
    // which particular stream a given caller lands on under contention is
    // unspecified beyond "eventually cycles through all of them".
    cudaStream_t next() noexcept;

    cudaStream_t at(std::size_t index) const;
    std::size_t size() const noexcept { return streams_.size(); }

    void synchronize_all();

private:
    std::vector<cudaStream_t> streams_;
    int device_id_;
    std::atomic<std::size_t> cursor_{0};
};

}  // namespace cuda_db
