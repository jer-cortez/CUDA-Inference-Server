#pragma once

#include <cstddef>
#include <vector>

namespace cuda_db {

// Abstract inference backend the Scheduler drives. The real implementation
// (added in a later milestone) wraps an ONNX Runtime session with the CUDA
// execution provider; for now, tests and local development use a stub
// implementation so the scheduler/queue logic can be built and verified
// without a GPU.
struct IExecutionEngine {
    virtual ~IExecutionEngine() = default;

    // `batched_input` is `batch_size` requests' inputs concatenated
    // contiguously, `input_elems` floats each. Returns `batch_size` output
    // rows concatenated contiguously, `output_elems` floats each, in the
    // same order as the input rows.
    virtual std::vector<float> run_inference(const std::vector<float>& batched_input,
                                              std::size_t batch_size,
                                              std::size_t input_elems,
                                              std::size_t output_elems) = 0;
};

}  // namespace cuda_db