#include "cuda_db/memory/memory_pool.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "cuda_db/memory/device_buffer.hpp"
#include "cuda_db/memory/pinned_buffer.hpp"

using namespace cuda_db;

// Bucket sizes are powers of two with a 4 KiB floor: anything at or below
// 4096 collapses to exactly one bucket, and everything else rounds up to the
// next power of two rather than being served an undersized block.
TEST(MemoryPoolTest, BucketForRoundsUpToPowerOfTwoWithFloor) {
    EXPECT_EQ(MemoryPool::bucket_for(1), 4096u);
    EXPECT_EQ(MemoryPool::bucket_for(4096), 4096u);
    EXPECT_EQ(MemoryPool::bucket_for(4097), 8192u);
    EXPECT_EQ(MemoryPool::bucket_for(65536), 65536u);
    EXPECT_EQ(MemoryPool::bucket_for(65537), 131072u);
}

// The whole point of pooling: releasing a block and immediately re-acquiring
// the same size must hand back the identical pointer rather than allocating
// again, and the stats must show it as a cache hit, not a fresh cudaMalloc.
TEST(MemoryPoolTest, ReleaseThenReacquireReturnsSamePointer) {
    MemoryPool pool;

    void* first_ptr = nullptr;
    {
        PooledBuffer buf = pool.acquire_device(1024);
        first_ptr = buf.data();
    }  // released back to the pool here

    PooledBuffer second = pool.acquire_device(1024);
    EXPECT_EQ(second.data(), first_ptr);

    MemoryPoolStats stats = pool.stats();
    EXPECT_EQ(stats.acquisitions, 2u);
    EXPECT_EQ(stats.cache_hits, 1u);
    EXPECT_EQ(stats.cudamalloc_calls, 1u);
}

// Distinct buckets must never hand out the same block concurrently -- a
// 1 KiB and a 100 KiB request should get independent allocations even if
// both are live at once.
TEST(MemoryPoolTest, DistinctBucketsDoNotAlias) {
    MemoryPool pool;
    PooledBuffer small = pool.acquire_device(1024);
    PooledBuffer large = pool.acquire_device(100 * 1024);

    EXPECT_NE(small.data(), nullptr);
    EXPECT_NE(large.data(), nullptr);
    EXPECT_NE(small.data(), large.data());
    EXPECT_LT(small.capacity_bytes(), large.capacity_bytes());
}

// Moving a PooledBuffer must leave the source empty so its (now-vacated)
// destructor doesn't also release the block the destination now owns --
// that double release would corrupt the free list.
TEST(MemoryPoolTest, MoveLeavesSourceEmptyAndDoesNotDoubleRelease) {
    MemoryPool pool;
    PooledBuffer original = pool.acquire_device(2048);
    void* ptr = original.data();

    PooledBuffer moved = std::move(original);
    EXPECT_TRUE(original.empty());
    EXPECT_EQ(moved.data(), ptr);

    EXPECT_EQ(pool.stats().live_handles, 1u);

    PooledBuffer move_assigned;
    move_assigned = std::move(moved);
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(move_assigned.data(), ptr);
    EXPECT_EQ(pool.stats().live_handles, 1u);
}

// DeviceBuffer round-trips a known pattern through a PinnedBuffer staging
// area: this is the actual H2D-then-D2H path every batch takes, exercised
// end to end rather than trusting the CUDA Runtime calls in isolation.
TEST(MemoryPoolTest, DeviceBufferRoundTripsThroughPinnedStaging) {
    constexpr std::size_t kCount = 256;
    PinnedBuffer host_in(kCount * sizeof(float));
    for (std::size_t i = 0; i < kCount; ++i) {
        host_in.host_float()[i] = static_cast<float>(i) * 1.5f;
    }

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    DeviceBuffer device(kCount * sizeof(float));
    device.copy_from_host_async(host_in.data(), host_in.size_bytes(), stream);

    PinnedBuffer host_out(kCount * sizeof(float));
    device.copy_to_host_async(host_out.data(), host_out.size_bytes(), stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(host_out.host_float()[i], static_cast<float>(i) * 1.5f);
    }
}

// ~MemoryPool asserts every handle was released first (see memory_pool.cpp);
// this proves the converse holds in the normal case -- once every handle has
// gone out of scope, live_handles genuinely reaches zero and destruction is
// safe. A death test isn't used for the violation itself since std::terminate
// is disruptive to run routinely; this test instead pins down the invariant
// std::terminate is guarding.
TEST(MemoryPoolTest, LiveHandlesReachesZeroOnceAllBuffersReleased) {
    auto pool = std::make_unique<MemoryPool>();
    {
        PooledBuffer a = pool->acquire_device(1024);
        PooledBuffer b = pool->acquire_pinned(2048);
        EXPECT_EQ(pool->stats().live_handles, 2u);
    }
    EXPECT_EQ(pool->stats().live_handles, 0u);
    pool.reset();  // must not terminate
}

// Concurrent acquire/release from multiple threads must not leak blocks or
// corrupt the free lists -- the mutex is what's under test here.
TEST(MemoryPoolTest, ConcurrentAcquireReleaseLeavesNoBytesInUse) {
    MemoryPool pool;
    constexpr int kThreads = 4;
    constexpr int kIterationsPerThread = 200;

    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&pool, t] {
            for (int i = 0; i < kIterationsPerThread; ++i) {
                const std::size_t size = 1024 * (1 + ((t + i) % 8));
                PooledBuffer buf = pool.acquire_device(size);
                EXPECT_NE(buf.data(), nullptr);
                // buf releases back to the pool at end of scope.
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    MemoryPoolStats stats = pool.stats();
    EXPECT_EQ(stats.live_handles, 0u);
    EXPECT_EQ(stats.bytes_in_use, 0u);
    EXPECT_EQ(stats.acquisitions, static_cast<std::uint64_t>(kThreads * kIterationsPerThread));
}
