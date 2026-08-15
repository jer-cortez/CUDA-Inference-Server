#include "cuda_db/memory/memory_pool.hpp"

#include <exception>

#include <cuda_runtime.h>

#include "cuda_db/memory/cuda_error.hpp"

namespace cuda_db {

// ---------------------------------------------------------------------------
// PooledBuffer
// ---------------------------------------------------------------------------

PooledBuffer::~PooledBuffer() { release(); }

PooledBuffer& PooledBuffer::operator=(PooledBuffer&& other) noexcept {
    if (this != &other) {
        release();
        pool_ = std::exchange(other.pool_, nullptr);
        ptr_ = std::exchange(other.ptr_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
        requested_ = std::exchange(other.requested_, 0);
        pinned_ = std::exchange(other.pinned_, false);
    }
    return *this;
}

void PooledBuffer::release() noexcept {
    if (pool_ != nullptr && ptr_ != nullptr) {
        pool_->release(ptr_, capacity_, pinned_);
    }
    pool_ = nullptr;
    ptr_ = nullptr;
    capacity_ = 0;
    requested_ = 0;
}

// ---------------------------------------------------------------------------
// MemoryPool
// ---------------------------------------------------------------------------

MemoryPool::MemoryPool(int device_id) : device_id_{device_id} {}

MemoryPool::~MemoryPool() {
    // Every PooledBuffer must have released back to us before we're
    // destroyed -- a live handle at this point means a dangling pointer is
    // about to be created the moment that handle is next touched. Caught
    // here rather than left as a silent use-after-free.
    if (stats_.live_handles != 0) {
        std::terminate();
    }
    trim();
}

std::size_t MemoryPool::bucket_for(std::size_t bytes) {
    constexpr std::size_t kMinBucket = 4096;  // 4 KiB
    if (bytes <= kMinBucket) return kMinBucket;

    std::size_t bucket = kMinBucket;
    while (bucket < bytes) {
        bucket <<= 1;
    }
    return bucket;
}

void* MemoryPool::allocate_block(std::size_t capacity, bool pinned) {
    void* ptr = nullptr;
    if (pinned) {
        cuda_check(cudaHostAlloc(&ptr, capacity, cudaHostAllocPortable), "cudaHostAlloc");
    } else {
        cuda_check(cudaSetDevice(device_id_), "cudaSetDevice");
        cuda_check(cudaMalloc(&ptr, capacity), "cudaMalloc");
    }
    return ptr;
}

void MemoryPool::free_block(void* ptr, bool pinned) noexcept {
    if (pinned) {
        cudaFreeHost(ptr);
    } else {
        cudaFree(ptr);
    }
}

PooledBuffer MemoryPool::acquire_device(std::size_t bytes) {
    const std::size_t capacity = bucket_for(bytes);
    std::lock_guard<std::mutex> lock(mutex_);

    ++stats_.acquisitions;
    auto& free_list = device_free_[capacity];
    void* ptr = nullptr;
    if (!free_list.empty()) {
        ptr = free_list.back();
        free_list.pop_back();
        ++stats_.cache_hits;
        stats_.bytes_cached -= capacity;
    } else {
        ptr = allocate_block(capacity, /*pinned=*/false);
        ++stats_.cudamalloc_calls;
    }

    stats_.bytes_in_use += capacity;
    ++stats_.live_handles;
    return PooledBuffer(this, ptr, capacity, bytes, /*pinned=*/false);
}

PooledBuffer MemoryPool::acquire_pinned(std::size_t bytes) {
    const std::size_t capacity = bucket_for(bytes);
    std::lock_guard<std::mutex> lock(mutex_);

    ++stats_.acquisitions;
    auto& free_list = pinned_free_[capacity];
    void* ptr = nullptr;
    if (!free_list.empty()) {
        ptr = free_list.back();
        free_list.pop_back();
        ++stats_.cache_hits;
        stats_.bytes_cached -= capacity;
    } else {
        ptr = allocate_block(capacity, /*pinned=*/true);
        // Deliberately not counted in cudamalloc_calls -- that counter is
        // specifically about device allocations, the ones that matter for
        // the "did the batch stall on an allocator" question.
    }

    stats_.bytes_in_use += capacity;
    ++stats_.live_handles;
    return PooledBuffer(this, ptr, capacity, bytes, /*pinned=*/true);
}

void MemoryPool::release(void* ptr, std::size_t capacity, bool pinned) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = pinned ? pinned_free_[capacity] : device_free_[capacity];
    free_list.push_back(ptr);

    stats_.bytes_in_use -= capacity;
    stats_.bytes_cached += capacity;
    --stats_.live_handles;
}

void MemoryPool::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [capacity, blocks] : device_free_) {
        for (void* ptr : blocks) {
            free_block(ptr, /*pinned=*/false);
            stats_.bytes_cached -= capacity;
        }
        blocks.clear();
    }
    for (auto& [capacity, blocks] : pinned_free_) {
        for (void* ptr : blocks) {
            free_block(ptr, /*pinned=*/true);
            stats_.bytes_cached -= capacity;
        }
        blocks.clear();
    }
}

MemoryPoolStats MemoryPool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

}  // namespace cuda_db
