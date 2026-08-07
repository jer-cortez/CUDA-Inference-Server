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

// Minimal throwaway IExecutionEngine implementation used only by these
// tests. Does no real inference -- just echoes the first `output_elems`
// floats of each input row (zero-padded if input_elems < output_elems) --
// so Scheduler's batching/threading/promise-resolution logic can be
// exercised without a GPU or ONNX Runtime.
class FakeExecutionEngine : public IExecutionEngine {
public:
    std::vector<float> run_inference(const std::vector<float>& batched_input,
                                      std::size_t batch_size,
                                      std::size_t input_elems,
                                      std::size_t output_elems) override {
        std::vector<float> output(batch_size * output_elems, 0.0f);
        for (std::size_t i = 0; i < batch_size; ++i) {
            const std::size_t copy_count = std::min(input_elems, output_elems);
            std::copy_n(batched_input.begin() + i * input_elems, copy_count,
                        output.begin() + i * output_elems);
        }
        return output;
    }
};

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
    auto engine = std::make_shared<FakeExecutionEngine>();
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
    auto engine = std::make_shared<FakeExecutionEngine>();
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
    auto engine = std::make_shared<FakeExecutionEngine>();
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
