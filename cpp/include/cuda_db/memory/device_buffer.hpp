#pragma once

#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

#include "cuda_db/memory/cuda_error.hpp"

namespace cuda_db {

// Move-only RAII owner of one cudaMalloc'd allocation. Exists so device
// memory is freed via a destructor rather than manually at every throw/return
// path -- a batch that fails mid-inference must not leak the input tensor it
// already uploaded.
//
// Not pooled: DeviceBuffer is the raw allocation primitive. MemoryPool
// (memory_pool.hpp) is what callers on the hot path should actually use, so a
// cudaMalloc doesn't happen on every batch.
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;

    explicit DeviceBuffer(std::size_t bytes) : bytes_{bytes} {
        if (bytes_ > 0) {
            cuda_check(cudaMalloc(&ptr_, bytes_), "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);  // destructors don't throw; a free failure here is unrecoverable anyway
        }
    }

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_{std::exchange(other.ptr_, nullptr)}, bytes_{std::exchange(other.bytes_, 0)} {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = std::exchange(other.ptr_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* data() noexcept { return ptr_; }
    const void* data() const noexcept { return ptr_; }
    std::size_t size_bytes() const noexcept { return bytes_; }
    bool empty() const noexcept { return ptr_ == nullptr; }

    template <class T>
    T* as() noexcept {
        return static_cast<T*>(ptr_);
    }
    template <class T>
    const T* as() const noexcept {
        return static_cast<const T*>(ptr_);
    }

    // Async H2D/D2H copies, queued on `stream`. The caller is responsible for
    // synchronizing (or otherwise ordering) before touching `src`/`dst` again
    // or letting this buffer's lifetime end -- see MemoryPool's PooledBuffer
    // for the RAII pattern that enforces that ordering.
    void copy_from_host_async(const void* src, std::size_t bytes, cudaStream_t stream) {
        if (bytes > bytes_) {
            throw CudaError("copy_from_host_async: bytes exceeds buffer capacity");
        }
        cuda_check(cudaMemcpyAsync(ptr_, src, bytes, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync (H2D)");
    }

    void copy_to_host_async(void* dst, std::size_t bytes, cudaStream_t stream) const {
        if (bytes > bytes_) {
            throw CudaError("copy_to_host_async: bytes exceeds buffer capacity");
        }
        cuda_check(cudaMemcpyAsync(dst, ptr_, bytes, cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync (D2H)");
    }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace cuda_db
