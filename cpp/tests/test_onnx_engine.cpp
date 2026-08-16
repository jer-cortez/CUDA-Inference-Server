#include "cuda_db/engine/onnx_execution_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using namespace cuda_db;

namespace {

constexpr std::size_t kInputElems = 3 * 224 * 224;
constexpr std::size_t kOutputElems = 1000;

// The model is a multi-hundred-megabyte artifact produced by
// models/export_resnet.py and excluded from git, so it is not present on a
// fresh clone or in CI. Skipping beats failing: a red suite there would report
// "the engine is broken" when the real answer is "nobody exported a model".
// CUDA_DB_TEST_MODEL overrides the default path.
std::string model_path() {
    if (const char* env = std::getenv("CUDA_DB_TEST_MODEL")) {
        return env;
    }
    return "models/resnet50.onnx";
}

bool model_available() {
    std::ifstream file(model_path(), std::ios::binary);
    return file.good();
}

#define SKIP_WITHOUT_MODEL()                                                    \
    do {                                                                        \
        if (!model_available()) {                                               \
            GTEST_SKIP() << "no model at " << model_path()                      \
                         << "; run models/export_resnet.py or set "             \
                            "CUDA_DB_TEST_MODEL";                               \
        }                                                                       \
    } while (false)

// Returns a unique_ptr rather than a value: OnnxExecutionEngine is
// non-copyable and non-movable (it owns an Ort::Session), so returning it by
// value would lean on C++17 guaranteed copy elision. This sidesteps that
// entirely.
std::unique_ptr<OnnxExecutionEngine> make_engine() {
    OnnxExecutionEngine::Options options;
    options.model_path = model_path();
    return std::make_unique<OnnxExecutionEngine>(std::move(options));
}

// Deterministic pseudo-random input. A fixed formula rather than a real image
// because these tests check shape/consistency invariants, not classification
// accuracy -- that is the integration test's job.
std::vector<float> make_input(std::size_t batch_size, float seed) {
    std::vector<float> input(batch_size * kInputElems);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.001F + seed);
    }
    return input;
}

}  // namespace

TEST(OnnxEngineTest, RejectsEmptyModelPath) {
    OnnxExecutionEngine::Options options;
    options.model_path = "";
    EXPECT_THROW(OnnxExecutionEngine{std::move(options)}, InferenceError);
}

TEST(OnnxEngineTest, ReportsCudaProviderAvailable) {
    SKIP_WITHOUT_MODEL();

    const std::unique_ptr<OnnxExecutionEngine> engine = make_engine();

    // Construction with require_cuda (the default) would already have thrown,
    // so this mostly documents what the session was built with -- and prints
    // the list when it is not what you expected.
    EXPECT_FALSE(engine->providers().empty());
    for (const std::string& provider : engine->providers()) {
        std::cout << "  provider: " << provider << "\n";
    }
}

TEST(OnnxEngineTest, SingleRequestProducesImageNetLogits) {
    SKIP_WITHOUT_MODEL();

    const std::unique_ptr<OnnxExecutionEngine> engine = make_engine();
    const std::vector<float> input = make_input(1, 0.0F);

    const std::vector<float> output =
        engine->run_inference(input, 1, kInputElems, kOutputElems);

    ASSERT_EQ(output.size(), kOutputElems);
    // A model that silently returned zeros would still have the right shape,
    // so assert the output actually varies.
    EXPECT_NE(output.front(), output.back());
}

// The milestone's central risk: if the export baked in a static batch
// dimension, exactly one of these sizes binds and the rest throw. Dynamic
// batching would be dead on arrival.
TEST(OnnxEngineTest, AcceptsVaryingBatchSizesOnOneSession) {
    SKIP_WITHOUT_MODEL();

    const std::unique_ptr<OnnxExecutionEngine> engine = make_engine();

    for (std::size_t batch_size : {std::size_t{1}, std::size_t{3}, std::size_t{8}}) {
        const std::vector<float> input = make_input(batch_size, 0.5F);
        const std::vector<float> output =
            engine->run_inference(input, batch_size, kInputElems, kOutputElems);

        EXPECT_EQ(output.size(), batch_size * kOutputElems)
            << "batch size " << batch_size << " did not bind";
    }
}

// Batching must not change any individual result. If row i of a batch differs
// from that same input run alone, the batch is being reshaped or strided
// wrongly -- which no shape assertion would catch, since the output size would
// still be correct.
TEST(OnnxEngineTest, BatchedRowMatchesSameInputRunAlone) {
    SKIP_WITHOUT_MODEL();

    const std::unique_ptr<OnnxExecutionEngine> engine = make_engine();

    constexpr std::size_t kBatch = 3;
    std::vector<float> batched(kBatch * kInputElems);
    std::vector<std::vector<float>> rows;
    for (std::size_t i = 0; i < kBatch; ++i) {
        rows.push_back(make_input(1, static_cast<float>(i)));
        std::copy(rows[i].begin(), rows[i].end(),
                  batched.begin() + static_cast<std::ptrdiff_t>(i * kInputElems));
    }

    const std::vector<float> batched_output =
        engine->run_inference(batched, kBatch, kInputElems, kOutputElems);
    ASSERT_EQ(batched_output.size(), kBatch * kOutputElems);

    for (std::size_t i = 0; i < kBatch; ++i) {
        const std::vector<float> solo =
            engine->run_inference(rows[i], 1, kInputElems, kOutputElems);
        ASSERT_EQ(solo.size(), kOutputElems);

        const float* batched_row = batched_output.data() + i * kOutputElems;

        // The invariant that actually matters: batching must not change the
        // prediction. Checked before the elementwise comparison because this
        // is the one that would still fail if batching were genuinely broken,
        // no matter how the numeric tolerance were tuned.
        const auto batched_top = std::max_element(batched_row, batched_row + kOutputElems);
        const auto solo_top = std::max_element(solo.begin(), solo.end());
        ASSERT_EQ(std::distance(batched_row, batched_top),
                  std::distance(solo.begin(), solo_top))
            << "row " << i << ": batching changed the predicted class";

        // Stated as a single worst-case bound rather than 1000 per-element
        // assertions, so a failure reports the largest drift instead of
        // whichever class happened to trip first.
        //
        // The bound is absolute, not relative: measured drift is ~1.2e-3
        // regardless of logit magnitude (it looks the same on a logit of 1.7
        // as on one of 0.003), because it comes from float error accumulating
        // through 50 layers when cuDNN picks a different convolution
        // algorithm for a batched run than a single-image one. That is a
        // constant noise floor, not proportional scaling.
        //
        // 1e-2 leaves ~8x headroom over the observed floor while staying far
        // below the logit range (roughly -10..15), so a genuine batching bug
        // -- which produces unrelated values, not 0.1% agreement -- still
        // fails loudly. The argmax check above is the real guard.
        float max_diff = 0.0F;
        std::size_t worst_class = 0;
        for (std::size_t j = 0; j < kOutputElems; ++j) {
            const float diff = std::fabs(batched_row[j] - solo[j]);
            if (diff > max_diff) {
                max_diff = diff;
                worst_class = j;
            }
        }

        EXPECT_LT(max_diff, 1e-2F)
            << "row " << i << ": largest drift " << max_diff << " at class " << worst_class
            << " (batched " << batched_row[worst_class] << " vs solo " << solo[worst_class]
            << ")";
    }
}

TEST(OnnxEngineTest, RejectsMismatchedInputLength) {
    SKIP_WITHOUT_MODEL();

    const std::unique_ptr<OnnxExecutionEngine> engine = make_engine();
    const std::vector<float> too_short(kInputElems - 1, 0.0F);

    EXPECT_THROW(engine->run_inference(too_short, 1, kInputElems, kOutputElems),
                 InferenceError);
}

TEST(OnnxEngineTest, EmptyBatchReturnsEmpty) {
    SKIP_WITHOUT_MODEL();

    const std::unique_ptr<OnnxExecutionEngine> engine = make_engine();
    EXPECT_TRUE(engine->run_inference({}, 0, kInputElems, kOutputElems).empty());
}
