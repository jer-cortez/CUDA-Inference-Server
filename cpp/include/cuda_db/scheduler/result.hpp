#pragma once

#include <cstdint>
#include <vector>

namespace cuda_db {

struct InferenceResult {
    uint64_t request_id;
    std::vector<float> logits;
};

}  // namespace cuda_db
