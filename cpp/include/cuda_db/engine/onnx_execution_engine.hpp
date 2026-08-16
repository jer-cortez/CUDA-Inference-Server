#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cuda_db/engine/execution_engine.hpp"

namespace cuda_db {

// Real inference backend: an ONNX Runtime session running on the CUDA
// execution provider. Third implementation of IExecutionEngine, alongside
// StubExecutionEngine (no GPU needed) and CudaExecutionEngine (custom kernels,
// no model).
//
// INPUT CONTRACT -- read before wiring anything upstream. This engine expects
// input that is ALREADY normalized (ImageNet per-channel mean/stddev), because
// that is what ResNet-50's weights were trained against. Normalization
// currently happens on the CPU in python/cuda_db/preprocessing/image_utils.py.
// The normalize CUDA kernel (kernels/normalize.cuh) deliberately does NOT run
// in this path -- applying it here as well would double-normalize and produce
// confident nonsense. Folding that kernel in belongs with the IoBinding work,
// where input stays device-resident end to end.
//
// Data path: host buffers are handed straight to Ort::Session::Run and the
// CUDA EP performs the H2D copy internally. That leaves one redundant copy per
// batch versus binding pre-allocated DeviceBuffers via Ort::IoBinding, which is
// a deliberate later optimization -- getting correct predictions and getting
// zero-copy predictions are separate problems, and debugging them together
// makes it impossible to tell a preprocessing bug from a binding bug.
class OnnxExecutionEngine : public IExecutionEngine {
public:
    struct Options {
        std::string model_path;
        int device_id = 0;
        // ORT falls back to the CPU provider when the CUDA EP cannot load,
        // which yields correct answers at a fraction of the speed -- and a
        // meaningless throughput benchmark. When true, construction fails
        // instead of silently degrading.
        bool require_cuda = true;
    };

    explicit OnnxExecutionEngine(Options options);
    ~OnnxExecutionEngine() override;

    OnnxExecutionEngine(const OnnxExecutionEngine&) = delete;
    OnnxExecutionEngine& operator=(const OnnxExecutionEngine&) = delete;

    // `batched_input` must hold batch_size * input_elems already-normalized
    // floats, laid out NCHW. The input shape is rebuilt from `batch_size` on
    // every call, which is what lets one session serve the variable batch
    // sizes the scheduler produces -- and why the model must be exported with
    // a dynamic batch axis.
    std::vector<float> run_inference(const std::vector<float>& batched_input,
                                      std::size_t batch_size, std::size_t input_elems,
                                      std::size_t output_elems) override;

    // Execution providers compiled into this ONNX Runtime build. Note the
    // limitation: this reflects the build, not per-node placement at run time,
    // so it proves the CUDA EP *could* be used, not that every operator
    // actually ran on the GPU. Confirming the latter means watching nvidia-smi
    // during a run or enabling ORT profiling. Still worth logging at startup --
    // no CUDAExecutionProvider here guarantees CPU-only.
    const std::vector<std::string>& providers() const noexcept { return providers_; }

    const std::string& input_name() const noexcept { return input_name_; }
    const std::string& output_name() const noexcept { return output_name_; }

private:
    // ORT types are confined to the .cpp so this header stays includable from
    // translation units built without ONNX Runtime on the include path.
    struct Impl;

    Options options_;
    std::unique_ptr<Impl> impl_;
    std::vector<std::string> providers_;
    std::string input_name_;
    std::string output_name_;
};

}  // namespace cuda_db
