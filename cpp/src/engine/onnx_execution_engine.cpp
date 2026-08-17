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
}

// Defined here, not defaulted in the header, because Impl is incomplete there.
OnnxExecutionEngine::~OnnxExecutionEngine() = default;

/**
 * @brief Runs one batch through the model.
 * @details Builds the input tensor shape from `batch_size` on every call
 * rather than caching it -- that is precisely what lets a single session serve
 * the variable batch sizes the scheduler produces, and it depends on the model
 * having been exported with a dynamic batch axis (see models/export_resnet.py).
 *
 * The input buffer is wrapped, not copied: Ort::Value::CreateTensor over
 * non-const data borrows `batched_input` for the duration of the call, and the
 * CUDA EP performs the host-to-device transfer internally.
 *
 * @param batched_input `batch_size * input_elems` already-normalized floats,
 * NCHW. See the header's input contract -- this engine does not normalize.
 * @param batch_size Number of request rows in the batch.
 * @param input_elems Floats per input row (3 * 224 * 224 for ResNet-50).
 * @param output_elems Floats per output row (1000 ImageNet logits).
 * @return std::vector<float> `batch_size * output_elems` logits, row-major,
 * in the same row order as the input.
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

    return translate_ort_errors("ONNX Runtime inference failed", [&] {
        const std::array<std::int64_t, 4> input_shape{
            static_cast<std::int64_t>(batch_size), 3, 224, 224};

        // const_cast is required by the ORT C++ API, which takes a mutable
        // pointer even for inputs it only reads. The buffer is not modified.
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            impl_->memory_info, const_cast<float*>(batched_input.data()),
            batched_input.size(), input_shape.data(), input_shape.size());

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
        const std::size_t expected = batch_size * output_elems;
        if (produced != expected) {
            throw InferenceError("model produced " + std::to_string(produced) +
                                 " elements, expected " + std::to_string(expected));
        }

        const float* data = outputs.front().GetTensorData<float>();
        return std::vector<float>(data, data + produced);
    });
}

}  // namespace cuda_db
