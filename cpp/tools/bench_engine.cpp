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
//   ./build-cuda/cpp/tools/bench_engine models/resnet50.onnx

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <utility>
#include <iomanip>
#include <iostream>
#include <string>
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

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "models/resnet50.onnx";

    cuda_db::OnnxExecutionEngine::Options options;
    options.model_path = model_path;

    std::cout << "loading " << model_path << " ...\n";
    cuda_db::OnnxExecutionEngine engine{std::move(options)};

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

    return 0;
}
