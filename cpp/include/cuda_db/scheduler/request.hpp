#pragma once

#include <chrono>
#include <cstdint>
#include <future>
#include <vector>

#include "cuda_db/scheduler/result.hpp"

namespace cuda_db {

struct InferenceRequest {
    uint64_t request_id;
    std::vector<float> input_data;
    std::chrono::steady_clock::time_point arrival_time;
    std::promise<InferenceResult> result_promise;
};

}  // namespace cuda_db
