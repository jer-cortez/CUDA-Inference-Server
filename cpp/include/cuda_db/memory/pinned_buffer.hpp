#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>

#include "cuda_db/memory/cuda_error.hpp"

namespace cuda_db {

// Move-only RAII owner of one cudaHostAlloc'd (page-locked) allocation. Used
// as the host-side staging buffer for async H2D/D2H transfers: a pageable
// std::vector can't back an async cudaMemcpyAsync (the driver may need to
// pin it itself, silently making the "async" copy synchronous), so anything
// on the hot batching path stages through pinned memory instead.
class PinnedBuffer {
public:
    PinnedBuffer() noexcept = default;

    explicit PinnedBuffer(std::size_t bytes) : bytes_{bytes} {
        if (bytes_ > 0) {
            // Portable: visible to every device/context in the process, since
            // the scheduler's worker thread is not necessarily the thread
            // that set the active CUDA device.
            cuda_check(cudaHostAlloc(&ptr_, bytes_, cudaHostAllocPortable), "cudaHostAlloc");
        }
    }

    ~PinnedBuffer() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
        }
    }

    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_{std::exchange(other.ptr_, nullptr)}, bytes_{std::exchange(other.bytes_, 0)} {}

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFreeHost(ptr_);
            }
            ptr_ = std::exchange(other.ptr_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void* data() noexcept { return ptr_; }
    const void* data() const noexcept { return ptr_; }
    std::size_t size_bytes() const noexcept { return bytes_; }
    bool empty() const noexcept { return ptr_ == nullptr; }

    float* host_float() noexcept { return static_cast<float*>(ptr_); }
    const float* host_float() const noexcept { return static_cast<const float*>(ptr_); }
    std::uint8_t* host_u8() noexcept { return static_cast<std::uint8_t*>(ptr_); }
    const std::uint8_t* host_u8() const noexcept { return static_cast<const std::uint8_t*>(ptr_); }

    template <class T>
    T* as() noexcept {
        return static_cast<T*>(ptr_);
    }
    template <class T>
    const T* as() const noexcept {
        return static_cast<const T*>(ptr_);
    }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace cuda_db
