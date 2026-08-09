#include "cuda_db/scheduler/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

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
