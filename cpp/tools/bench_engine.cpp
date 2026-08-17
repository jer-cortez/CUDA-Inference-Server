// Times OnnxExecutionEngine::run_inference directly, with no scheduler, no
// thread pool, no HTTP and no Python in the path.
//
// Exists to answer one question the end-to-end benchmark cannot: is batch-N
// inference actually sublinear on this model and GPU? If per-request cost falls
// as the batch grows here, batching is sound and any regression measured by
// benchmarks/load_test.py lives above this layer (scheduler, executor, or the
// server). If per-request cost is flat or rising here, batching itself is not
// paying and no amount of scheduler tuning will fix it.
//
// The binary lands in the binary dir of the CMakeLists that declares it
// (cpp/), not alongside this source file:
//
//   ./build-cuda/cpp/bench_engine models/resnet50.onnx

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cuda_db/engine/onnx_execution_engine.hpp"

namespace {

constexpr std::size_t kInputElems = 3 * 224 * 224;
constexpr std::size_t kOutputElems = 1000;
constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 30;

double median_ms(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

std::vector<std::size_t> parse_buckets(const std::string& spec) {
    std::vector<std::size_t> buckets;
    std::size_t start = 0;
    while (start <= spec.size()) {
        const std::size_t comma = spec.find(',', start);
        const std::string token = spec.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            buckets.push_back(static_cast<std::size_t>(std::stoul(token)));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return buckets;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "models/resnet50.onnx";
    // Second argument is the bucket set, so the padding policy can be explored
    // without a rebuild:
    //   "1,2,4,8"  four shapes
    //   "8"        one shape -- every batch padded to 8
    const std::string bucket_spec = argc > 2 ? argv[2] : "1,2,4,8";
    // Third argument toggles the device-resident path, so its effect can be
    // measured rather than assumed:
    //   bench_engine model.onnx "8" bound
    //   bench_engine model.onnx "8" host
    const std::string path = argc > 3 ? argv[3] : "bound";

    cuda_db::OnnxExecutionEngine::Options options;
    options.model_path = model_path;
    options.batch_buckets = parse_buckets(bucket_spec);
    options.use_io_binding = (path != "host");
    std::cout << "buckets: " << bucket_spec << "   path: " << path << "\n";

    std::cout << "loading " << model_path << " ...\n";
    cuda_db::OnnxExecutionEngine engine{std::move(options)};
    // Reported rather than inferred from the flag: the bound path falls back
    // silently if its buffers cannot be allocated, and a benchmark that
    // mislabels which path it measured is worse than no benchmark.
    std::cout << "io_binding active: " << (engine.io_binding_active() ? "yes" : "no") << "\n";

    std::cout << "\n" << std::left << std::setw(8) << "batch" << std::right
              << std::setw(14) << "batch ms" << std::setw(16) << "per-request ms"
              << std::setw(14) << "req/s" << "\n";
    std::cout << std::string(52, '-') << "\n";

    double baseline_per_request = 0.0;

    for (std::size_t batch : {1u, 2u, 4u, 8u, 16u}) {
        std::vector<float> input(batch * kInputElems, 0.5F);

        // Warm this specific shape: ORT allocates workspace per input shape, so
        // the first call at a new batch size is not representative.
        for (int i = 0; i < kWarmupIters; ++i) {
            engine.run_inference(input, batch, kInputElems, kOutputElems);
        }

        std::vector<double> samples;
        samples.reserve(kTimedIters);
        for (int i = 0; i < kTimedIters; ++i) {
            const auto start = std::chrono::steady_clock::now();
            engine.run_inference(input, batch, kInputElems, kOutputElems);
            const auto elapsed = std::chrono::steady_clock::now() - start;
            samples.push_back(
                std::chrono::duration<double, std::milli>(elapsed).count());
        }

        const double batch_ms = median_ms(samples);
        const double per_request_ms = batch_ms / static_cast<double>(batch);
        if (batch == 1) {
            baseline_per_request = per_request_ms;
        }

        std::cout << std::left << std::setw(8) << batch << std::right << std::fixed
                  << std::setprecision(2) << std::setw(14) << batch_ms
                  << std::setw(16) << per_request_ms << std::setw(14)
                  << (1000.0 / per_request_ms) << "\n";
    }

    std::cout << "\nIf per-request ms falls as batch grows, batching pays at the engine\n"
                 "level and a regression in load_test.py is above this layer.\n"
                 "If it stays flat, batching is not helping on this model/GPU and the\n"
                 "scheduler cannot recover that.\n";
    if (baseline_per_request > 0.0) {
        std::cout << "batch-1 baseline: " << std::fixed << std::setprecision(2)
                  << baseline_per_request << " ms/request ("
                  << (1000.0 / baseline_per_request) << " req/s ceiling)\n";
    }

    // The measurements above hold the shape constant, which is exactly what a
    // real dynamic batcher does NOT do: it emits whatever batch size happened
    // to accumulate. This phase reproduces that churn, because an ORT
    // dynamic-shape session re-plans per distinct input shape and that cost
    // would be invisible to a fixed-shape benchmark.
    std::cout << "\n--- varying batch size (what the scheduler actually produces) ---\n";

    // Pre-warm every shape so this measures steady-state shape switching, not
    // first-touch allocation.
    for (std::size_t batch : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        std::vector<float> input(batch * kInputElems, 0.5F);
        for (int i = 0; i < 3; ++i) {
            engine.run_inference(input, batch, kInputElems, kOutputElems);
        }
    }

    std::mt19937 rng{1234};
    std::uniform_int_distribution<std::size_t> pick{1, 8};

    std::vector<double> churn_samples;
    double churn_requests = 0.0;
    for (int i = 0; i < 60; ++i) {
        const std::size_t batch = pick(rng);
        std::vector<float> input(batch * kInputElems, 0.5F);

        const auto start = std::chrono::steady_clock::now();
        engine.run_inference(input, batch, kInputElems, kOutputElems);
        const auto elapsed = std::chrono::steady_clock::now() - start;

        churn_samples.push_back(
            std::chrono::duration<double, std::milli>(elapsed).count());
        churn_requests += static_cast<double>(batch);
    }

    double churn_total = 0.0;
    for (double sample : churn_samples) {
        churn_total += sample;
    }

    std::cout << "median batch ms   : " << std::fixed << std::setprecision(2)
              << median_ms(churn_samples) << "\n";
    std::cout << "per-request ms    : " << (churn_total / churn_requests) << "\n";
    std::cout << "req/s             : " << (1000.0 * churn_requests / churn_total) << "\n";
    std::cout << "\nCompare against the fixed-shape rows above. If these are far worse,\n"
                 "the cost is shape switching, and the fix is to pad batches to a small\n"
                 "set of fixed sizes so the session only ever sees a few shapes.\n";

    return 0;
}
