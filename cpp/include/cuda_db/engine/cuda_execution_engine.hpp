#pragma once

#include <cstddef>
#include <vector>

#include "cuda_db/engine/execution_engine.hpp"
#include "cuda_db/kernels/cuda_stream_pool.hpp"
#include "cuda_db/memory/device_buffer.hpp"
#include "cuda_db/memory/memory_pool.hpp"

namespace cuda_db {

// GPU-resident backend that exercises the full device round trip: pinned host
// staging -> async H2D -> normalize kernel -> async D2H. It does not run a
// model; that arrives with ONNX Runtime in the next milestone. Its purpose is
// to prove the memory/stream plumbing composes correctly under the real
// Scheduler before model inference is layered on.
//
// Deliberately additive rather than a replacement for StubExecutionEngine:
// the stub is what keeps the macOS dev loop and CPU-only CI working, so both
// implementations of IExecutionEngine coexist and the caller picks.
//
// Thread-safety: safe to call run_inference() from multiple threads. MemoryPool
// is internally locked and CudaStreamPool hands out streams round-robin, so
// concurrent batches land on different streams. The Scheduler currently drives
// it from a single worker thread regardless.
class CudaExecutionEngine : public IExecutionEngine {
public:
    struct Options {
        // ImageNet per-channel constants -- the values a torchvision
        // ResNet/MobileNet export expects its input to already be normalized
        // with, so this preprocessing matches the model landing in milestone 3.
        std::vector<float> mean{0.485F, 0.456F, 0.406F};
        std::vector<float> stddev{0.229F, 0.224F, 0.225F};
        int device_id = 0;
        // Two streams is enough to overlap one batch's download with the next
        // batch's upload; more only helps once multiple workers exist.
        std::size_t stream_count = 2;
    };

    // Two overloads rather than a defaulted `Options options = {}` parameter:
    // GCC rejects that, because a nested aggregate's default member
    // initializers are not usable in a default argument at that point in the
    // enclosing class definition. The no-arg form delegates, out of line, to
    // the other one.
    CudaExecutionEngine();
    explicit CudaExecutionEngine(Options options);

    std::vector<float> run_inference(const std::vector<float>& batched_input,
                                      std::size_t batch_size, std::size_t input_elems,
                                      std::size_t output_elems) override;

    // Exposed so tests and the benchmark harness can assert the pool actually
    // stops calling cudaMalloc once warmed up -- the observable proof that
    // pooling works.
    MemoryPoolStats pool_stats() const { return pool_.stats(); }

    std::size_t channels() const noexcept { return options_.mean.size(); }

private:
    Options options_;
    MemoryPool pool_;
    CudaStreamPool streams_;
    // Uploaded once at construction rather than per batch: they are a handful
    // of floats and never change.
    DeviceBuffer mean_;
    DeviceBuffer stddev_;
};

}  // namespace cuda_db
