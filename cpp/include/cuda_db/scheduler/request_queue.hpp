#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>


#include "cuda_db/scheduler/request.hpp"

namespace cuda_db {

struct RequestQueue {
    void push(InferenceRequest&& request);
    std::vector<InferenceRequest> wait_and_drain(std::size_t max_batch_size, std::chrono::milliseconds timeout);
    void shutdown();
    
    std::size_t size() const;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<InferenceRequest> queue_;
    bool stop_requested_ = false;
};

}  // namespace cuda_db