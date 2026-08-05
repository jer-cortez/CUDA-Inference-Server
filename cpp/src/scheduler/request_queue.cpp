#include "cuda_db/scheduler/request.hpp"

namespace cuda_db { 

void RequestQueue::push(InferenceRequest&& request) { 
    { 
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(request));
    }

    cv_.notify_one();
}


}
