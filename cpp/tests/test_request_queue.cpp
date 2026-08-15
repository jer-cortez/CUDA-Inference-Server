#include "cuda_db/scheduler/request_queue.hpp"

#include <chrono>
#include <set>
#include <thread>

#include <gtest/gtest.h>

using namespace cuda_db;
using namespace std::chrono_literals;

namespace {

InferenceRequest make_request(std::uint64_t id) {
    InferenceRequest req;
    req.request_id = id;
    req.input_data = {static_cast<float>(id)};
    req.arrival_time = std::chrono::steady_clock::now();
    return req;
}

}  // namespace

// Pushing exactly max_batch_size items should trigger an immediate drain,
// well before the timeout elapses.
TEST(RequestQueueTest, DrainsAsSoonAsBatchSizeReached) {
    RequestQueue queue;
    queue.push(make_request(1));
    queue.push(make_request(2));

    const auto start = std::chrono::steady_clock::now();
    std::vector<InferenceRequest> batch = queue.wait_and_drain(2, 500ms);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_EQ(batch.size(), 2u);
    EXPECT_EQ(batch[0].request_id, 1u);
    EXPECT_EQ(batch[1].request_id, 2u);
    EXPECT_LT(elapsed, 100ms);
}

// Fewer than max_batch_size items should still be returned once the
// timeout elapses, rather than blocking forever waiting for more.
TEST(RequestQueueTest, TimesOutAndReturnsPartialBatch) {
    RequestQueue queue;
    queue.push(make_request(1));

    const auto start = std::chrono::steady_clock::now();
    std::vector<InferenceRequest> batch = queue.wait_and_drain(8, 20ms);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_EQ(batch.size(), 1u);
    EXPECT_GE(elapsed, 20ms);
}

// The batching window is timed from the first request's arrival, not from
// the call to wait_and_drain(). A request that lands late must still get the
// full window to find batch-mates -- under a single timed wait starting at
// call entry it would get whatever was left over, or nothing at all.
TEST(RequestQueueTest, BatchWindowStartsAtFirstArrival) {
    RequestQueue queue;

    std::thread producer([&queue] {
        std::this_thread::sleep_for(30ms);
        queue.push(make_request(1));
    });

    const auto start = std::chrono::steady_clock::now();
    std::vector<InferenceRequest> batch = queue.wait_and_drain(8, 20ms);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    producer.join();

    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].request_id, 1u);
    // 30ms idle + a 20ms window that opens only once the push lands.
    EXPECT_GE(elapsed, 50ms);
}

// A zero timeout disables the batching window entirely: the request is served
// as soon as it arrives. The idle wait before it must still block, though --
// see IdleSchedulerDoesNotSpin in test_scheduler.cpp for that half.
TEST(RequestQueueTest, ZeroTimeoutReturnsFirstArrivalImmediately) {
    RequestQueue queue;

    std::thread producer([&queue] {
        std::this_thread::sleep_for(20ms);
        queue.push(make_request(1));
    });

    const auto start = std::chrono::steady_clock::now();
    std::vector<InferenceRequest> batch = queue.wait_and_drain(8, 0ms);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    producer.join();

    ASSERT_EQ(batch.size(), 1u);
    EXPECT_GE(elapsed, 20ms);
    EXPECT_LT(elapsed, 200ms);
}

// wait_and_drain() on an empty queue should block until another thread
// pushes, rather than returning immediately or busy-looping.
TEST(RequestQueueTest, WaitsWhenEmptyThenReceivesPush) {
    RequestQueue queue;

    std::thread producer([&queue] {
        std::this_thread::sleep_for(10ms);
        queue.push(make_request(42));
    });

    std::vector<InferenceRequest> batch = queue.wait_and_drain(1, 500ms);
    producer.join();

    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].request_id, 42u);
}

// shutdown() should unblock a waiter promptly instead of making it wait
// out the full timeout.
TEST(RequestQueueTest, ShutdownUnblocksWaiter) {
    RequestQueue queue;

    std::thread consumer([&queue] {
        const auto start = std::chrono::steady_clock::now();
        std::vector<InferenceRequest> batch = queue.wait_and_drain(8, 5000ms);
        const auto elapsed = std::chrono::steady_clock::now() - start;

        EXPECT_TRUE(batch.empty());
        EXPECT_LT(elapsed, 500ms);
    });

    std::this_thread::sleep_for(10ms);
    queue.shutdown();
    consumer.join();
}

// size() should reflect items currently queued, and should not consume
// them (a subsequent wait_and_drain() should still see the same items).
TEST(RequestQueueTest, SizeReflectsQueuedItemsWithoutDraining) {
    RequestQueue queue;
    EXPECT_EQ(queue.size(), 0u);

    queue.push(make_request(1));
    queue.push(make_request(2));
    EXPECT_EQ(queue.size(), 2u);

    std::vector<InferenceRequest> batch = queue.wait_and_drain(2, 500ms);
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_EQ(queue.size(), 0u);
}

// Multiple threads pushing concurrently should not lose or duplicate any
// items, and push() itself should never crash/corrupt state under
// contention. This is the test most likely to catch a locking bug that the
// single-producer tests above can't exercise.
TEST(RequestQueueTest, ConcurrentPushesAreAllDelivered) {
    RequestQueue queue;

    constexpr int kProducers = 4;
    constexpr int kPushesPerProducer = 100;
    constexpr int kTotalPushes = kProducers * kPushesPerProducer;

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&queue, p] {
            for (int i = 0; i < kPushesPerProducer; ++i) {
                // Encode producer id in the high bits so duplicates/loss
                // across producers are easy to detect below.
                const std::uint64_t id = (static_cast<std::uint64_t>(p) << 32) | i;
                queue.push(make_request(id));
            }
        });
    }
    for (std::thread& producer : producers) {
        producer.join();
    }

    ASSERT_EQ(queue.size(), static_cast<std::size_t>(kTotalPushes));

    std::vector<InferenceRequest> drained;
    while (drained.size() < static_cast<std::size_t>(kTotalPushes)) {
        std::vector<InferenceRequest> batch = queue.wait_and_drain(kTotalPushes, 500ms);
        ASSERT_FALSE(batch.empty()) << "drain stalled before all items were retrieved";
        for (InferenceRequest& req : batch) {
            drained.push_back(std::move(req));
        }
    }

    std::set<std::uint64_t> seen_ids;
    for (const InferenceRequest& req : drained) {
        const bool inserted = seen_ids.insert(req.request_id).second;
        EXPECT_TRUE(inserted) << "duplicate request_id: " << req.request_id;
    }
    EXPECT_EQ(seen_ids.size(), static_cast<std::size_t>(kTotalPushes));
}
