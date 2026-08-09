#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cuda_db {

// Raised when a request cannot be served: a malformed input, or a failure
// inside the backend. The binding layer registers this as a Python exception
// so callers get a catchable error instead of a hard abort.
struct InferenceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

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

// Placeholder backend used until ONNX Runtime lands. Echoes each input row
// into the corresponding output row, truncating or zero-padding to reach
// `output_elems`. Deterministic and shape-correct, so the full
// Python -> pybind11 -> Scheduler round trip can be exercised on a machine
// with no GPU while still producing per-request-distinguishable output.
struct StubExecutionEngine : IExecutionEngine {
    std::vector<float> run_inference(const std::vector<float>& batched_input,
                                      std::size_t batch_size,
                                      std::size_t input_elems,
                                      std::size_t output_elems) override;
};

}  // namespace cuda_db
