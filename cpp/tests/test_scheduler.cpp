#include "cuda_db/scheduler/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/resource.h>
#include <sys/time.h>
#endif

#include <gtest/gtest.h>

using namespace cuda_db;
using namespace std::chrono_literals;

namespace {

// These tests drive the real StubExecutionEngine (echoes each input row into
// the corresponding output row, zero-padded), so Scheduler's
// batching/threading/promise-resolution logic is exercised against the same
// engine the served path uses -- without needing a GPU or ONNX Runtime.

// Always throws, to exercise Scheduler's exception-propagation path.
class ThrowingExecutionEngine : public IExecutionEngine {
public:
    std::vector<float> run_inference(const std::vector<float>&, std::size_t, std::size_t,
                                      std::size_t) override {
        throw std::runtime_error("engine failure");
    }
};

InferenceRequest make_request(std::uint64_t id, float value) {
    InferenceRequest req;
    req.request_id = id;
    req.input_data = {value};
    req.arrival_time = std::chrono::steady_clock::now();
    return req;
}

}  // namespace

// Submits a full batch (max_batch_size requests) and checks each future
// resolves with the correct request_id and the corresponding echoed value
// -- this is what catches an off-by-one in the batch-row <-> request
// mapping inside run_loop().
TEST(SchedulerTest, ResolvesEachFutureWithMatchingRequestId) {
    auto engine = std::make_shared<StubExecutionEngine>();
    SchedulerConfig config;
    config.max_batch_size = 4;
    config.max_wait = 50ms;
    config.input_elems = 1;
    config.output_elems = 1;

    Scheduler scheduler(engine, config);
    scheduler.start();

    std::vector<std::future<InferenceResult>> futures;
    for (std::uint64_t id = 0; id < 4; ++id) {
        futures.push_back(scheduler.submit(make_request(id, static_cast<float>(id) * 10.0f)));
    }

    for (std::uint64_t id = 0; id < 4; ++id) {
        InferenceResult result = futures[id].get();
        EXPECT_EQ(result.request_id, id);
        ASSERT_EQ(result.logits.size(), 1u);
        EXPECT_FLOAT_EQ(result.logits[0], static_cast<float>(id) * 10.0f);
    }

    scheduler.stop();
}

// A single request, with max_batch_size set high enough that it will never
// be reached -- the request should still resolve once max_wait elapses,
// proving the timeout trigger works independently of the size trigger.
TEST(SchedulerTest, BatchesWithinTimeoutEvenBelowMaxBatchSize) {
    auto engine = std::make_shared<StubExecutionEngine>();
    SchedulerConfig config;
    config.max_batch_size = 100;  // never reached in this test
    config.max_wait = 20ms;
    config.input_elems = 1;
    config.output_elems = 1;

    Scheduler scheduler(engine, config);
    scheduler.start();

    auto future = scheduler.submit(make_request(7, 3.5f));
    InferenceResult result = future.get();  // should not hang past ~max_wait

    EXPECT_EQ(result.request_id, 7u);
    ASSERT_EQ(result.logits.size(), 1u);
    EXPECT_FLOAT_EQ(result.logits[0], 3.5f);

    scheduler.stop();
}

// If the engine throws, every request in the affected batch should have
// its future resolve with that exception rather than hang forever.
TEST(SchedulerTest, EngineExceptionPropagatesToAllFuturesInBatch) {
    auto engine = std::make_shared<ThrowingExecutionEngine>();
    SchedulerConfig config;
    config.max_batch_size = 2;
    config.max_wait = 50ms;
    config.input_elems = 1;
    config.output_elems = 1;

    Scheduler scheduler(engine, config);
    scheduler.start();

    auto future_a = scheduler.submit(make_request(1, 1.0f));
    auto future_b = scheduler.submit(make_request(2, 2.0f));

    EXPECT_THROW(future_a.get(), std::runtime_error);
    EXPECT_THROW(future_b.get(), std::runtime_error);

    scheduler.stop();
}

// stop() should still let requests queued just before it's called drain
// and complete, rather than dropping them (the shutdown-drain guarantee
// discussed for run_loop()).
TEST(SchedulerTest, PendingRequestsCompleteAcrossStop) {
    auto engine = std::make_shared<StubExecutionEngine>();
    SchedulerConfig config;
    config.max_batch_size = 100;  // relies on the timeout, not size, to drain
    config.max_wait = 20ms;
    config.input_elems = 1;
    config.output_elems = 1;

    Scheduler scheduler(engine, config);
    scheduler.start();

    auto future = scheduler.submit(make_request(9, 42.0f));
    scheduler.stop();  // should block until the worker drains and resolves it

    ASSERT_EQ(future.wait_for(0ms), std::future_status::ready);
    InferenceResult result = future.get();
    EXPECT_EQ(result.request_id, 9u);
    EXPECT_FLOAT_EQ(result.logits[0], 42.0f);
}

// The batching counters are what /healthz and the benchmark harness read to
// show requests actually coalesced. Submitting a full batch's worth of
// requests up front should produce far fewer batches than requests.
TEST(SchedulerTest, StatsCountBatchesAndRequests) {
    auto engine = std::make_shared<StubExecutionEngine>();
    SchedulerConfig config;
    config.max_batch_size = 8;
    config.max_wait = 50ms;
    config.input_elems = 1;
    config.output_elems = 1;

    Scheduler scheduler(engine, config);

    SchedulerStats before = scheduler.stats();
    EXPECT_EQ(before.total_batches, 0u);
    EXPECT_EQ(before.total_requests, 0u);
    EXPECT_EQ(before.max_batch_size_seen, 0u);

    scheduler.start();

    std::vector<std::future<InferenceResult>> futures;
    for (std::uint64_t id = 0; id < 8; ++id) {
        futures.push_back(scheduler.submit(make_request(id, 1.0f)));
    }
    for (auto& future : futures) {
        future.get();
    }
    scheduler.stop();

    SchedulerStats after = scheduler.stats();
    EXPECT_EQ(after.total_requests, 8u);
    EXPECT_GE(after.total_batches, 1u);
    EXPECT_LE(after.total_batches, 8u);
    EXPECT_GE(after.max_batch_size_seen, 1u);
    EXPECT_LE(after.max_batch_size_seen, 8u);
}

// arrival_time is stamped by the caller when the request enters the system,
// so the queue-wait counter must reflect time spent waiting for batch-mates,
// not just time spent inside run_loop(). Backdating the stamp gives an exact
// lower bound to assert against without depending on scheduler timing.
TEST(SchedulerTest, QueueWaitLatencyIsRecorded) {
    auto engine = std::make_shared<StubExecutionEngine>();
    SchedulerConfig config;
    config.max_batch_size = 1;
    config.max_wait = 0ms;
    config.input_elems = 1;
    config.output_elems = 1;

    Scheduler scheduler(engine, config);
    scheduler.start();

    InferenceRequest req = make_request(1, 5.0f);
    req.arrival_time = std::chrono::steady_clock::now() - 50ms;

    scheduler.submit(std::move(req)).get();
    scheduler.stop();

    SchedulerStats after = scheduler.stats();
    EXPECT_GE(after.max_queue_wait_us, 50'000u);
    EXPECT_GE(after.total_queue_wait_us, 50'000u);
    EXPECT_GE(after.avg_queue_wait_ms(), 50.0);
}

// Counters must read zero -- not divide by zero or report garbage -- before
// any request has been served, since /healthz is polled on a cold server.
TEST(SchedulerTest, LatencyStatsAreZeroBeforeFirstRequest) {
    auto engine = std::make_shared<StubExecutionEngine>();
    Scheduler scheduler(engine, SchedulerConfig{});

    SchedulerStats stats = scheduler.stats();
    EXPECT_EQ(stats.total_queue_wait_us, 0u);
    EXPECT_DOUBLE_EQ(stats.avg_queue_wait_ms(), 0.0);
    EXPECT_DOUBLE_EQ(stats.max_queue_wait_ms(), 0.0);
    EXPECT_DOUBLE_EQ(stats.avg_exec_ms(), 0.0);
}

#ifndef _WIN32
// An idle scheduler must park on the condition variable, not spin. This is a
// real regression guard: with a single timed wait, max_wait == 0ms made
// cv_.wait_for() return immediately on every iteration and run_loop() burned
// a whole core doing nothing. Measured in consumed CPU time rather than wall
// time, because that is the actual symptom.
TEST(SchedulerTest, IdleSchedulerDoesNotSpin) {
    auto engine = std::make_shared<StubExecutionEngine>();
    SchedulerConfig config;
    config.max_batch_size = 8;
    config.max_wait = 0ms;  // the configuration that used to spin
    config.input_elems = 1;
    config.output_elems = 1;

    Scheduler scheduler(engine, config);
    scheduler.start();

    const auto cpu_used = [] {
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        const auto to_us = [](const timeval& tv) {
            return static_cast<std::uint64_t>(tv.tv_sec) * 1'000'000u +
                   static_cast<std::uint64_t>(tv.tv_usec);
        };
        return to_us(usage.ru_utime) + to_us(usage.ru_stime);
    };

    const std::uint64_t before = cpu_used();
    std::this_thread::sleep_for(200ms);
    const std::uint64_t consumed = cpu_used() - before;

    // A spinning worker consumes ~200ms of CPU over this window; a parked one
    // consumes essentially none. The 50ms bound leaves room for whatever else
    // the test binary is doing without being anywhere near the spin case.
    EXPECT_LT(consumed, 50'000u);

    SchedulerStats idle = scheduler.stats();
    EXPECT_EQ(idle.total_batches, 0u);
    EXPECT_EQ(idle.total_requests, 0u);

    // Still responsive after idling -- a parked worker that never wakes would
    // pass the CPU check above while being completely broken.
    InferenceResult result = scheduler.submit(make_request(1, 7.0f)).get();
    EXPECT_EQ(result.request_id, 1u);
    EXPECT_FLOAT_EQ(result.logits[0], 7.0f);

    scheduler.stop();
}
#endif  // _WIN32

// A request whose input length disagrees with config_.input_elems must be
// failed and dropped rather than concatenated -- otherwise it would shift
// every following row's slice boundary and silently corrupt the results of
// well-formed requests sharing its batch.
TEST(SchedulerTest, MisSizedRequestFailsWithoutCorruptingBatchMates) {
    auto engine = std::make_shared<StubExecutionEngine>();
    SchedulerConfig config;
    config.max_batch_size = 3;
    config.max_wait = 50ms;
    config.input_elems = 1;
    config.output_elems = 1;

    Scheduler scheduler(engine, config);
    scheduler.start();

    InferenceRequest bad = make_request(2, 0.0f);
    bad.input_data = {1.0f, 2.0f};  // two floats where one is expected

    auto future_a = scheduler.submit(make_request(1, 11.0f));
    auto future_bad = scheduler.submit(std::move(bad));
    auto future_c = scheduler.submit(make_request(3, 33.0f));

    EXPECT_THROW(future_bad.get(), InferenceError);

    InferenceResult result_a = future_a.get();
    EXPECT_EQ(result_a.request_id, 1u);
    EXPECT_FLOAT_EQ(result_a.logits[0], 11.0f);

    InferenceResult result_c = future_c.get();
    EXPECT_EQ(result_c.request_id, 3u);
    EXPECT_FLOAT_EQ(result_c.logits[0], 33.0f);

    scheduler.stop();
}
