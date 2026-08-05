#include "cuda_db/scheduler/request_queue.hpp"

#include <algorithm>

namespace cuda_db {

/**
 * @brief Enqueues a request and wakes the consumer.
 * @details Producer-side entry point. Moves `request` onto the back of the
 * internal deque under lock, then notifies one waiter. Safe to call from
 * multiple producer threads concurrently.
 *
 * @param request The request to enqueue. Moved from; the caller's object is
 * left in a valid but unspecified state after this call.
 */
void RequestQueue::push(InferenceRequest&& request) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(request));
    }
    cv_.notify_one();
}

/**
 * @brief Blocks until a batch is ready, then drains and returns it.
 * @details Consumer-side entry point; intended to be called by a single
 * consumer thread in a loop. Blocks until EITHER `max_batch_size` requests
 * are queued OR `timeout` elapses, whichever comes first -- this dual
 * condition is the dynamic-batching trigger. Also returns early if
 * shutdown() has been called.
 *
 * @param max_batch_size Maximum number of requests to drain in one call, and
 * the queue-size threshold that triggers an early return.
 * @param timeout Maximum time to wait for `max_batch_size` requests to
 * accumulate before returning whatever is available.
 * @return std::vector<InferenceRequest> The drained batch. May contain fewer
 * than `max_batch_size` items (timeout case) or be empty (woken by
 * shutdown() with nothing queued).
 *
 * @note This call does not throw on timeout; a partial or empty batch is the
 * normal, expected outcome, not an error.
 */
std::vector<InferenceRequest> RequestQueue::wait_and_drain(std::size_t max_batch_size, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(lock, timeout, [this, max_batch_size] {
        return queue_.size() >= max_batch_size || stop_requested_;
    });

    std::vector<InferenceRequest> batch;
    const std::size_t drain_count = std::min(max_batch_size, queue_.size());
    batch.reserve(drain_count);
    for (std::size_t i = 0; i < drain_count; ++i) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
    }
    return batch;
}

}
