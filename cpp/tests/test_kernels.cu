#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "cuda_db/engine/cuda_execution_engine.hpp"
#include "cuda_db/kernels/normalize.cuh"
#include "cuda_db/memory/cuda_error.hpp"
#include "cuda_db/memory/device_buffer.hpp"
#include "cuda_db/scheduler/scheduler.hpp"

using namespace cuda_db;
using namespace std::chrono_literals;

namespace {

// The CUDA build can be configured on a machine that has the toolkit but no
// usable device (or a driver mismatch). Skipping beats failing: a red suite
// there would say "your kernel is broken" when the real answer is "there is
// no GPU here".
bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

#define SKIP_WITHOUT_CUDA_DEVICE()                            \
    do {                                                      \
        if (!cuda_device_available()) {                       \
            GTEST_SKIP() << "no CUDA device available";       \
        }                                                     \
    } while (false)

DeviceBuffer upload(const std::vector<float>& host) {
    DeviceBuffer buffer(host.size() * sizeof(float));
    cuda_check(cudaMemcpy(buffer.data(), host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy (test upload)");
    return buffer;
}

std::vector<float> download(const DeviceBuffer& buffer, std::size_t elems) {
    std::vector<float> host(elems, 0.0F);
    cuda_check(cudaMemcpy(host.data(), buffer.data(), elems * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy (test download)");
    return host;
}

InferenceRequest make_request(std::uint64_t id, std::vector<float> input) {
    InferenceRequest req;
    req.request_id = id;
    req.input_data = std::move(input);
    req.arrival_time = std::chrono::steady_clock::now();
    return req;
}

}  // namespace

// One batch, three channels, four elements per plane, with distinct per-channel
// constants -- so a kernel that used the wrong index arithmetic (e.g. idx % c
// instead of (idx / hw) % c) would produce visibly wrong values rather than
// coincidentally correct ones.
TEST(NormalizeKernelTest, AppliesPerChannelMeanAndStddev) {
    SKIP_WITHOUT_CUDA_DEVICE();

    constexpr int kN = 1;
    constexpr int kC = 3;
    constexpr int kHW = 4;

    const std::vector<float> input{
        1.0F, 2.0F, 3.0F, 4.0F,        // channel 0
        10.0F, 20.0F, 30.0F, 40.0F,    // channel 1
        100.0F, 200.0F, 300.0F, 400.0F  // channel 2
    };
    const std::vector<float> mean{1.0F, 10.0F, 100.0F};
    const std::vector<float> stddev{2.0F, 5.0F, 100.0F};

    DeviceBuffer data = upload(input);
    DeviceBuffer d_mean = upload(mean);
    DeviceBuffer d_stddev = upload(stddev);

    launch_normalize(data.as<float>(), kN, kC, kHW, d_mean.as<float>(), d_stddev.as<float>(),
                     nullptr);
    cuda_check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");

    const std::vector<float> result = download(data, input.size());
    ASSERT_EQ(result.size(), input.size());

    for (std::size_t i = 0; i < input.size(); ++i) {
        const std::size_t channel = (i / kHW) % kC;
        const float expected = (input[i] - mean[channel]) / stddev[channel];
        EXPECT_FLOAT_EQ(result[i], expected) << "element " << i;
    }
}

// Every element must be visited even when the element count far exceeds the
// capped grid -- this is what the grid-stride loop exists for, and a plain
// one-thread-per-element kernel would silently leave the tail untouched.
TEST(NormalizeKernelTest, GridStrideCoversMoreElementsThanThreads) {
    SKIP_WITHOUT_CUDA_DEVICE();

    constexpr int kN = 64;
    constexpr int kC = 3;
    constexpr int kHW = 8192;  // 1.5M elements, well past 1024 blocks * 256 threads

    const std::vector<float> input(static_cast<std::size_t>(kN) * kC * kHW, 7.0F);
    const std::vector<float> mean{7.0F, 7.0F, 7.0F};
    const std::vector<float> stddev{2.0F, 2.0F, 2.0F};

    DeviceBuffer data = upload(input);
    DeviceBuffer d_mean = upload(mean);
    DeviceBuffer d_stddev = upload(stddev);

    launch_normalize(data.as<float>(), kN, kC, kHW, d_mean.as<float>(), d_stddev.as<float>(),
                     nullptr);
    cuda_check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");

    // (7 - 7) / 2 == 0 everywhere; any element the kernel missed is still 7.
    const std::vector<float> result = download(data, input.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        ASSERT_FLOAT_EQ(result[i], 0.0F) << "element " << i << " was not visited";
    }
}

// An empty batch is a legal drain result, so it must be a no-op rather than an
// error or an empty-grid launch.
TEST(NormalizeKernelTest, ZeroBatchIsNoOp) {
    SKIP_WITHOUT_CUDA_DEVICE();

    const std::vector<float> mean{1.0F};
    const std::vector<float> stddev{1.0F};
    DeviceBuffer d_mean = upload(mean);
    DeviceBuffer d_stddev = upload(stddev);

    EXPECT_NO_THROW(launch_normalize(nullptr, 0, 1, 1, d_mean.as<float>(), d_stddev.as<float>(),
                                      nullptr));
}

TEST(NormalizeKernelTest, RejectsNullDataForNonEmptyBatch) {
    SKIP_WITHOUT_CUDA_DEVICE();

    const std::vector<float> mean{1.0F};
    const std::vector<float> stddev{1.0F};
    DeviceBuffer d_mean = upload(mean);
    DeviceBuffer d_stddev = upload(stddev);

    EXPECT_THROW(
        launch_normalize(nullptr, 1, 1, 1, d_mean.as<float>(), d_stddev.as<float>(), nullptr),
        CudaError);
}

// Caught at construction, where the host-side values are still readable --
// on device this would silently produce inf instead of an error.
TEST(CudaExecutionEngineTest, RejectsZeroStddev) {
    SKIP_WITHOUT_CUDA_DEVICE();

    CudaExecutionEngine::Options options;
    options.mean = {0.0F, 0.0F, 0.0F};
    options.stddev = {1.0F, 0.0F, 1.0F};

    EXPECT_THROW(CudaExecutionEngine{options}, CudaError);
}

TEST(CudaExecutionEngineTest, RejectsInputElemsNotDivisibleByChannels) {
    SKIP_WITHOUT_CUDA_DEVICE();

    CudaExecutionEngine engine;  // 3 channels
    const std::vector<float> input(5, 1.0F);

    EXPECT_THROW(engine.run_inference(input, 1, 5, 5), InferenceError);
}

// The GPU path must produce the same values a host-side computation would --
// this is the end-to-end check that staging, H2D, kernel, and D2H all line up.
TEST(CudaExecutionEngineTest, NormalizesBatchThroughDeviceRoundTrip) {
    SKIP_WITHOUT_CUDA_DEVICE();

    CudaExecutionEngine::Options options;
    options.mean = {1.0F, 2.0F, 3.0F};
    options.stddev = {2.0F, 4.0F, 8.0F};
    CudaExecutionEngine engine{options};

    constexpr std::size_t kBatch = 2;
    constexpr std::size_t kHW = 2;
    constexpr std::size_t kInputElems = 3 * kHW;

    std::vector<float> input(kBatch * kInputElems);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i);
    }

    const std::vector<float> output = engine.run_inference(input, kBatch, kInputElems, kInputElems);
    ASSERT_EQ(output.size(), kBatch * kInputElems);

    for (std::size_t i = 0; i < input.size(); ++i) {
        const std::size_t channel = ((i % kInputElems) / kHW) % 3;
        const float expected = (input[i] - options.mean[channel]) / options.stddev[channel];
        EXPECT_FLOAT_EQ(output[i], expected) << "element " << i;
    }
}

// Truncate/zero-pad semantics must match StubExecutionEngine so the two are
// interchangeable behind IExecutionEngine.
TEST(CudaExecutionEngineTest, ZeroPadsWhenOutputElemsExceedsInputElems) {
    SKIP_WITHOUT_CUDA_DEVICE();

    CudaExecutionEngine::Options options;
    options.mean = {0.0F, 0.0F, 0.0F};
    options.stddev = {1.0F, 1.0F, 1.0F};
    CudaExecutionEngine engine{options};

    const std::vector<float> input{1.0F, 2.0F, 3.0F};  // 1 batch, 3 channels, hw=1
    const std::vector<float> output = engine.run_inference(input, 1, 3, 5);

    ASSERT_EQ(output.size(), 5u);
    EXPECT_FLOAT_EQ(output[0], 1.0F);
    EXPECT_FLOAT_EQ(output[1], 2.0F);
    EXPECT_FLOAT_EQ(output[2], 3.0F);
    EXPECT_FLOAT_EQ(output[3], 0.0F);
    EXPECT_FLOAT_EQ(output[4], 0.0F);
}

// The payoff of pooling: after the first batch warms the free lists, repeated
// same-shape batches must stop calling cudaMalloc entirely.
TEST(CudaExecutionEngineTest, PoolStopsAllocatingAfterWarmup) {
    SKIP_WITHOUT_CUDA_DEVICE();

    CudaExecutionEngine engine;
    const std::vector<float> input(3 * 16, 1.0F);

    engine.run_inference(input, 1, 3 * 16, 3 * 16);
    const std::uint64_t allocs_after_warmup = engine.pool_stats().cudamalloc_calls;
    EXPECT_GT(allocs_after_warmup, 0u);

    for (int i = 0; i < 10; ++i) {
        engine.run_inference(input, 1, 3 * 16, 3 * 16);
    }

    const MemoryPoolStats stats = engine.pool_stats();
    EXPECT_EQ(stats.cudamalloc_calls, allocs_after_warmup) << "pool allocated on the hot path";
    EXPECT_GT(stats.cache_hits, 0u);
}

// The whole point of milestone 2: a request submitted through the real
// Scheduler comes back correct, with the batch-row <-> request mapping intact
// across the GPU round trip.
TEST(CudaExecutionEngineTest, ResolvesFuturesThroughScheduler) {
    SKIP_WITHOUT_CUDA_DEVICE();

    CudaExecutionEngine::Options options;
    options.mean = {0.0F, 0.0F, 0.0F};
    options.stddev = {1.0F, 1.0F, 1.0F};  // identity, so outputs equal inputs
    auto engine = std::make_shared<CudaExecutionEngine>(options);

    SchedulerConfig config;
    config.max_batch_size = 4;
    config.max_wait = 50ms;
    config.input_elems = 3;
    config.output_elems = 3;

    Scheduler scheduler(engine, config);
    scheduler.start();

    std::vector<std::future<InferenceResult>> futures;
    for (std::uint64_t id = 0; id < 4; ++id) {
        const float base = static_cast<float>(id) * 10.0F;
        futures.push_back(scheduler.submit(make_request(id, {base, base + 1.0F, base + 2.0F})));
    }

    for (std::uint64_t id = 0; id < 4; ++id) {
        const InferenceResult result = futures[id].get();
        EXPECT_EQ(result.request_id, id);
        ASSERT_EQ(result.logits.size(), 3u);

        const float base = static_cast<float>(id) * 10.0F;
        EXPECT_FLOAT_EQ(result.logits[0], base);
        EXPECT_FLOAT_EQ(result.logits[1], base + 1.0F);
        EXPECT_FLOAT_EQ(result.logits[2], base + 2.0F);
    }

    scheduler.stop();
}
