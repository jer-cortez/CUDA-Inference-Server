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
 * The wait is split into two phases, which matters for both correctness and
 * CPU cost:
 *
 * Phase 1 waits, untimed, for the queue to become non-empty. An idle server
 * therefore parks on the condition variable instead of burning a core -- a
 * single timed wait would spin here whenever `timeout` is 0ms, since
 * cv_.wait_for(lock, 0ms, pred) returns immediately on a false predicate.
 *
 * Phase 2 then opens the batching window. Because it starts only once the
 * first request has arrived, `timeout` measures how long that request waits
 * for company, not how long this call has been running. Under a single timed
 * wait a request arriving late in the window would get an arbitrarily
 * truncated -- possibly zero -- share of it.
 *
 * @param max_batch_size Maximum number of requests to drain in one call, and
 * the queue-size threshold that triggers an early return.
 * @param timeout How long, measured from the first request's arrival, to wait
 * for `max_batch_size` requests to accumulate before returning whatever is
 * available. A timeout of 0ms disables phase 2 entirely, so each request is
 * served as soon as it arrives.
 * @return std::vector<InferenceRequest> The drained batch. May contain fewer
 * than `max_batch_size` items (timeout case) or be empty (woken by
 * shutdown() with nothing queued).
 *
 * @note This call does not throw on timeout; a partial or empty batch is the
 * normal, expected outcome, not an error.
 */
std::vector<InferenceRequest> RequestQueue::wait_and_drain(std::size_t max_batch_size, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Phase 1: park until there is something to batch, or we are shutting down.
    cv_.wait(lock, [this] {
        return !queue_.empty() || stop_requested_;
    });

    // Woken by shutdown() with nothing queued: nothing to do. Returning here
    // is what lets the worker loop distinguish "idle" from "drained and done".
    if (queue_.empty()) {
        return {};
    }

    // Phase 2: the batching window, timed from the first arrival above.
    if (timeout > std::chrono::milliseconds::zero()) {
        cv_.wait_for(lock, timeout, [this, max_batch_size] {
            return queue_.size() >= max_batch_size || stop_requested_;
        });
    }

    std::vector<InferenceRequest> batch;
    const std::size_t drain_count = std::min(max_batch_size, queue_.size());
    batch.reserve(drain_count);
    for (std::size_t i = 0; i < drain_count; ++i) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
    }
    return batch;
}

/**
 * @brief Signals shutdown and wakes any blocked waiter.
 * @details Sets the internal stop flag under lock, then notifies all
 * waiters. Does not clear or drain the queue -- any items still enqueued
 * remain there so a subsequent wait_and_drain() call can still retrieve
 * them; this only unblocks threads currently waiting so they can check the
 * stop condition and drain whatever remains.
 */
void RequestQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
    }
    cv_.notify_all();
}

/**
 * @brief Returns the current number of queued requests.
 * @details Snapshot only -- by the time the caller observes the returned
 * value, concurrent push()/wait_and_drain() calls on other threads may have
 * already changed it. Intended for tests and observability, not for
 * synchronization decisions.
 *
 * @return std::size_t Number of requests in the queue at the moment the
 * lock was held.
 */
std::size_t RequestQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

}
