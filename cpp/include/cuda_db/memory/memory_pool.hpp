#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cuda_db {

class MemoryPool;

// RAII handle for one block acquired from a MemoryPool. Returns the block to
// the pool's free list on destruction rather than freeing it -- that reuse is
// the entire point of pooling: a batch's input/output/staging buffers are the
// same few sizes every time, so after warmup no cudaMalloc/cudaHostAlloc
// should occur on the hot path at all.
//
// Ownership: a PooledBuffer must not outlive the MemoryPool it came from. It
// must also not be destroyed while a stream still has in-flight work touching
// its memory -- the caller (see engine/batch_workspace.hpp, added when the
// device-resident batching path lands) is responsible for synchronizing the
// stream before letting handles go out of scope.
class PooledBuffer {
public:
    PooledBuffer() noexcept = default;
    ~PooledBuffer();

    PooledBuffer(PooledBuffer&& other) noexcept
        : pool_{std::exchange(other.pool_, nullptr)},
          ptr_{std::exchange(other.ptr_, nullptr)},
          capacity_{std::exchange(other.capacity_, 0)},
          requested_{std::exchange(other.requested_, 0)},
          pinned_{std::exchange(other.pinned_, false)} {}

    PooledBuffer& operator=(PooledBuffer&& other) noexcept;

    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;

    void* data() const noexcept { return ptr_; }
    // The size the caller asked for.
    std::size_t size_bytes() const noexcept { return requested_; }
    // The bucket's actual size, always >= size_bytes(); do not write past
    // size_bytes() just because capacity allows it -- the extra space is
    // slack from bucket rounding, not usable payload.
    std::size_t capacity_bytes() const noexcept { return capacity_; }
    bool empty() const noexcept { return ptr_ == nullptr; }

    template <class T>
    T* as() const noexcept {
        return static_cast<T*>(ptr_);
    }

private:
    friend class MemoryPool;

    PooledBuffer(MemoryPool* pool, void* ptr, std::size_t capacity, std::size_t requested,
                 bool pinned) noexcept
        : pool_{pool}, ptr_{ptr}, capacity_{capacity}, requested_{requested}, pinned_{pinned} {}

    void release() noexcept;

    MemoryPool* pool_ = nullptr;
    void* ptr_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t requested_ = 0;
    bool pinned_ = false;
};

// Observability counters for the pool. `cudamalloc_calls` is the number one
// worth watching in a benchmark: in steady state it should stop growing,
// which is the proof pooling is actually working rather than allocating on
// every batch.
struct MemoryPoolStats {
    std::uint64_t acquisitions = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cudamalloc_calls = 0;
    std::uint64_t bytes_in_use = 0;
    std::uint64_t bytes_cached = 0;
    std::uint64_t live_handles = 0;
};

// Size-bucketed free-list allocator for device and pinned-host memory.
// Buckets are rounded up to a power of two (minimum 4 KiB) so a modest spread
// of request sizes -- e.g. every batch size from 1 to max_batch_size -- still
// collapses onto a small, reusable set of buckets instead of fragmenting.
//
// Thread-safe: guarded by a single mutex. The scheduler's worker thread is
// currently the only caller, but the lock is there from day one so a future
// multi-worker scheduler isn't blocked on revisiting this class.
class MemoryPool {
public:
    explicit MemoryPool(int device_id = 0);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    PooledBuffer acquire_device(std::size_t bytes);
    PooledBuffer acquire_pinned(std::size_t bytes);

    // Frees every currently-cached (i.e. not checked out) block. Does not
    // affect live handles.
    void trim();

    MemoryPoolStats stats() const;

    // Rounds `bytes` up to the bucket it would be served from. Exposed
    // publicly so tests (and callers sizing a workspace up front) can predict
    // capacity_bytes() without acquiring first.
    static std::size_t bucket_for(std::size_t bytes);

private:
    friend class PooledBuffer;

    void* allocate_block(std::size_t capacity, bool pinned);
    void free_block(void* ptr, bool pinned) noexcept;
    // Called by ~PooledBuffer / release(); returns the block to its bucket's
    // free list rather than freeing it.
    void release(void* ptr, std::size_t capacity, bool pinned) noexcept;

    mutable std::mutex mutex_;
    std::unordered_map<std::size_t, std::vector<void*>> device_free_;
    std::unordered_map<std::size_t, std::vector<void*>> pinned_free_;
    int device_id_;
    MemoryPoolStats stats_;
};

}  // namespace cuda_db
