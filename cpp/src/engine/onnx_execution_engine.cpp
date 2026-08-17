#include "cuda_db/engine/onnx_execution_engine.hpp"

#include <algorithm>
#include <array>
#include <utility>

#include <onnxruntime_cxx_api.h>

namespace cuda_db {

// Holds every ONNX Runtime type, so onnxruntime_cxx_api.h stays out of the
// public header and callers built without ORT can still include it.
struct OnnxExecutionEngine::Impl {
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    // No AllocatedStringPtr members here: that type is a unique_ptr with a
    // stateful deleter (it carries the OrtAllocator*), so it has no default
    // constructor and cannot be a default-initialized member. The model's
    // input/output names are copied into std::string in the constructor
    // instead, and the allocated originals are released right after.
    Impl() : env{ORT_LOGGING_LEVEL_WARNING, "cuda_db"} {}
};

namespace {

// ORT reports failures by throwing Ort::Exception. Callers of this project
// only know about InferenceError, so translate at the boundary rather than
// leaking a third-party exception type through IExecutionEngine.
template <class Fn>
auto translate_ort_errors(const char* what, Fn&& fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const Ort::Exception& e) {
        throw InferenceError(std::string(what) + ": " + e.what());
    }
}

}  // namespace

/**
 * @brief Loads the model and creates a CUDA-backed inference session.
 * @details Appends the CUDA execution provider before creating the session.
 * When `options.require_cuda` is set and this ORT build has no
 * CUDAExecutionProvider compiled in, construction fails rather than silently
 * running on CPU -- a fallback would still produce correct predictions, so
 * nothing downstream would notice until the throughput benchmark came out
 * meaningless. This catches the "CPU-only ORT tarball was downloaded" mistake;
 * it does not prove every operator was placed on the GPU at run time.
 *
 * Input/output names are queried from the model rather than hardcoded, so a
 * re-export that renames them does not silently break binding.
 *
 * @param options Model path, device id, and the CUDA-required flag.
 * @throws InferenceError If the model path is empty, the model fails to load,
 * or CUDA was required but not resolved.
 */
OnnxExecutionEngine::OnnxExecutionEngine(Options options)
    : options_{std::move(options)}, impl_{std::make_unique<Impl>()} {
    if (options_.model_path.empty()) {
        throw InferenceError("OnnxExecutionEngine: model_path is empty");
    }

    translate_ort_errors("failed to initialize ONNX Runtime session", [&] {
        // Value-initialized: this is a C struct, and default-initialization
        // would leave arena/cudnn/stream fields holding garbage that gets
        // handed straight to the provider.
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = options_.device_id;

        // Critical for a dynamic batcher, and NOT the value {} gives us:
        // OrtCudnnConvAlgoSearchExhaustive is enum 0, so zero-initialization
        // silently selects an exhaustive cuDNN algorithm benchmark that re-runs
        // for every new input shape. A batch-size-1 server sees one shape and
        // pays it once; this server produces a different shape per batch size,
        // so the search fires repeatedly during serving and costs far more than
        // batching saves -- measured at 0.1-0.7x the throughput of batch-size-1
        // before this line existed, worst where batch sizes varied most.
        // Heuristic picks an algorithm from cuDNN's cost model instead, which is
        // effectively free per shape.
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        impl_->session_options.AppendExecutionProvider_CUDA(cuda_options);
        impl_->session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        impl_->session = Ort::Session{impl_->env, options_.model_path.c_str(),
                                      impl_->session_options};
        impl_->memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return 0;
    });

    for (const std::string& provider : Ort::GetAvailableProviders()) {
        providers_.push_back(provider);
    }

    if (options_.require_cuda &&
        std::find(providers_.begin(), providers_.end(), "CUDAExecutionProvider") ==
            providers_.end()) {
        throw InferenceError(
            "ONNX Runtime resolved no CUDAExecutionProvider; the build is CPU-only. "
            "Set require_cuda=false to run anyway, but note every latency number "
            "would then be a CPU number.");
    }

    if (!std::is_sorted(options_.batch_buckets.begin(), options_.batch_buckets.end())) {
        throw InferenceError("OnnxExecutionEngine: batch_buckets must be ascending");
    }

    translate_ort_errors("failed to query model input/output names", [&] {
        Ort::AllocatorWithDefaultOptions allocator;
        // Locals, not members: the std::string assignments below copy the
        // characters, so the ORT-allocated buffers can be freed on scope exit.
        const Ort::AllocatedStringPtr input = impl_->session.GetInputNameAllocated(0, allocator);
        const Ort::AllocatedStringPtr output = impl_->session.GetOutputNameAllocated(0, allocator);
        input_name_ = input.get();
        output_name_ = output.get();
        return 0;
    });

    if (options_.warm_buckets) {
        warm_buckets();
    }
}

/**
 * @brief Runs one throwaway inference per bucket so each shape is planned.
 * @details ORT plans and allocates workspace the first time it sees an input
 * shape. Without this, the first real request at each bucket pays that cost and
 * it shows up as a latency spike in p99 rather than as startup time.
 *
 * Shapes are assumed to be 3x224x224 per row, matching the rest of this class.
 * Failures are deliberately swallowed: warmup is an optimization, and a model
 * that cannot run a given bucket will report that clearly on the first real
 * request rather than failing construction for a shape nobody may ever use.
 */
void OnnxExecutionEngine::warm_buckets() {
    constexpr std::size_t kInputElems = 3 * 224 * 224;
    constexpr std::size_t kOutputElems = 1000;

    for (const std::size_t bucket : options_.batch_buckets) {
        if (bucket == 0) {
            continue;
        }
        try {
            const std::vector<float> input(bucket * kInputElems, 0.0F);
            run_inference(input, bucket, kInputElems, kOutputElems);
        } catch (const std::exception&) {
            // See above: not fatal.
        }
    }
}

// Defined here, not defaulted in the header, because Impl is incomplete there.
OnnxExecutionEngine::~OnnxExecutionEngine() = default;

/**
 * @brief Smallest configured bucket >= batch_size.
 * @details Returns `batch_size` unchanged when it exceeds every bucket, which
 * runs at its own shape and pays the re-plan cost rather than failing.
 */
std::size_t OnnxExecutionEngine::bucket_for(std::size_t batch_size) const {
    for (const std::size_t bucket : options_.batch_buckets) {
        if (batch_size <= bucket) {
            return bucket;
        }
    }
    return batch_size;
}

/**
 * @brief Runs one batch through the model, padded to a bucket size.
 * @details The batch is padded up to the next configured bucket and the extra
 * output rows are dropped, so the session only ever sees a handful of input
 * shapes.
 *
 * That padding is what makes dynamic batching viable here at all. An ORT
 * dynamic-shape session re-plans per distinct shape; feeding it the raw batch
 * size (which varies on nearly every batch) measured 15.14 ms/request against
 * 1.44 ms/request at a fixed shape on an RTX A4000 -- a 10x penalty that left
 * the batcher slower than serving requests one at a time. Padded rows waste a
 * little compute; re-planning wastes an order of magnitude more.
 *
 * The unpadded path still wraps `batched_input` without copying. The padded
 * path necessarily copies once into a zero-filled buffer.
 *
 * @param batched_input `batch_size * input_elems` already-normalized floats,
 * NCHW. See the header's input contract -- this engine does not normalize.
 * @param batch_size Number of request rows in the batch.
 * @param input_elems Floats per input row (3 * 224 * 224 for ResNet-50).
 * @param output_elems Floats per output row (1000 ImageNet logits).
 * @return std::vector<float> `batch_size * output_elems` logits, row-major, in
 * the same row order as the input. Padded rows are not returned.
 * @throws InferenceError If the input length is inconsistent, the model
 * returns an unexpected element count, or the session fails.
 */
std::vector<float> OnnxExecutionEngine::run_inference(const std::vector<float>& batched_input,
                                                       std::size_t batch_size,
                                                       std::size_t input_elems,
                                                       std::size_t output_elems) {
    if (batched_input.size() != batch_size * input_elems) {
        throw InferenceError("batched input size does not match batch_size * input_elems");
    }
    if (batch_size == 0) {
        return {};
    }

    const std::size_t padded_batch = bucket_for(batch_size);

    // Only materialized when padding is actually needed, so an exact-bucket
    // batch keeps the zero-copy path.
    std::vector<float> padded_input;
    const float* input_data = batched_input.data();
    if (padded_batch != batch_size) {
        padded_input.assign(padded_batch * input_elems, 0.0F);
        std::copy(batched_input.begin(), batched_input.end(), padded_input.begin());
        input_data = padded_input.data();
    }

    return translate_ort_errors("ONNX Runtime inference failed", [&] {
        const std::array<std::int64_t, 4> input_shape{
            static_cast<std::int64_t>(padded_batch), 3, 224, 224};

        // const_cast is required by the ORT C++ API, which takes a mutable
        // pointer even for inputs it only reads. The buffer is not modified.
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            impl_->memory_info, const_cast<float*>(input_data), padded_batch * input_elems,
            input_shape.data(), input_shape.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        std::vector<Ort::Value> outputs =
            impl_->session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                               output_names, 1);

        if (outputs.empty() || !outputs.front().IsTensor()) {
            throw InferenceError("ONNX Runtime returned no output tensor");
        }

        const std::size_t produced =
            static_cast<std::size_t>(
                outputs.front().GetTensorTypeAndShapeInfo().GetElementCount());
        const std::size_t expected = padded_batch * output_elems;
        if (produced != expected) {
            throw InferenceError("model produced " + std::to_string(produced) +
                                 " elements, expected " + std::to_string(expected));
        }

        // Truncate to the real batch: rows beyond batch_size are padding and
        // belong to no request.
        const float* data = outputs.front().GetTensorData<float>();
        return std::vector<float>(data, data + batch_size * output_elems);
    });
}

}  // namespace cuda_db
