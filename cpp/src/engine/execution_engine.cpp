#include "cuda_db/engine/execution_engine.hpp"

#include <algorithm>

namespace cuda_db {

/**
 * @brief Stand-in for the real backend: copies each input row to its output row.
 * @details Copies `min(input_elems, output_elems)` floats from input row `i`
 * to output row `i`; the remainder of the output row stays zero. The output
 * is allocated zero-filled up front, so no explicit padding step is needed.
 *
 * Validates the input length rather than trusting the caller, since a short
 * buffer would otherwise read past the end while slicing rows.
 *
 * @param batched_input `batch_size * input_elems` floats, row-major.
 * @param batch_size Number of request rows in the batch.
 * @param input_elems Floats per input row.
 * @param output_elems Floats per output row.
 * @return std::vector<float> `batch_size * output_elems` floats, row-major.
 * @throws InferenceError If `batched_input` is not exactly
 * `batch_size * input_elems` floats long.
 */
std::vector<float> StubExecutionEngine::run_inference(const std::vector<float>& batched_input,
                                                       std::size_t batch_size,
                                                       std::size_t input_elems,
                                                       std::size_t output_elems) {
    if (batched_input.size() != batch_size * input_elems) {
        throw InferenceError("batched input size does not match batch_size * input_elems");
    }

    std::vector<float> batched_output(batch_size * output_elems, 0.0F);
    const std::size_t copy_elems = std::min(input_elems, output_elems);

    for (std::size_t i = 0; i < batch_size; ++i) {
        const auto row_begin = batched_input.begin() + static_cast<std::ptrdiff_t>(i * input_elems);
        std::copy(row_begin, row_begin + static_cast<std::ptrdiff_t>(copy_elems),
                  batched_output.begin() + static_cast<std::ptrdiff_t>(i * output_elems));
    }

    return batched_output;
}

}  // namespace cuda_db
