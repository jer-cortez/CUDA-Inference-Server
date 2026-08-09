#include "cuda_db/scheduler/scheduler.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>

namespace cuda_db {

/**
 * @brief Constructs a Scheduler bound to a given engine and configuration.
 * @details Only stores `engine` and `config`; does not start the worker
 * thread. Call start() separately to begin processing.
 *
 * @param engine The inference backend the worker loop will call for each
 * batch. Shared ownership since it must outlive the worker thread.
 * @param config Batch-size/timeout/tensor-shape settings used by run_loop().
 */
Scheduler::Scheduler(std::shared_ptr<IExecutionEngine> engine, SchedulerConfig config)
    : engine_{std::move(engine)}, config_{config}
{
}

/**
 * @brief Destroys the Scheduler, ensuring the worker thread is stopped.
 * @details Delegates to stop(), so callers are not required to call stop()
 * themselves before the object goes out of scope.
 */
Scheduler::~Scheduler() {
    stop();
}

/**
 * @brief Starts the background worker thread that runs run_loop().
 * @details Safe to call multiple times; a second call while already running
 * is a no-op. Uses an atomic exchange so concurrent start() calls cannot
 * both pass the "not yet running" check and spawn two worker threads.
 */
void Scheduler::start() {
    if (running_.exchange(true)) return ;
    worker_ = std::thread(&Scheduler::run_loop, this);
}

/**
 * @brief Stops the worker thread and waits for it to finish.
 * @details Clears the running flag, calls queue_.shutdown() to unblock the
 * worker if it is currently waiting in wait_and_drain(), then joins the
 * worker thread. Any requests still queued at the time of the call are
 * still processed by the worker before it exits -- see run_loop().
 */
void Scheduler::stop() {
    running_= false;
    queue_.shutdown();

    if (worker_.joinable()) worker_.join();
}

/**
 * @brief Enqueues a request and returns a future for its result.
 * @details Producer-facing entry point. Retrieves the future from the
 * request's promise before moving the request into the queue, since the
 * request is unusable after being moved from.
 *
 * @param request The request to submit. Moved from.
 * @return std::future<InferenceResult> Resolves once the worker thread has
 * processed the batch containing this request, or holds an exception if
 * the engine call for that batch failed.
 */
std::future<InferenceResult> Scheduler::submit(InferenceRequest&& request) {
    auto future = request.result_promise.get_future();
    queue_.push(std::move(request));
    return future;
}

/**
 * @brief Returns a snapshot of the batching counters.
 * @details Each counter is read independently, so the three values may not
 * correspond to the exact same instant if the worker is mid-batch. That is
 * acceptable: these are observability counters, not a transactional view.
 *
 * @return SchedulerStats Batches processed, requests processed, and the
 * largest batch seen so far.
 */
SchedulerStats Scheduler::stats() const {
    SchedulerStats snapshot;
    snapshot.total_batches = total_batches_.load(std::memory_order_relaxed);
    snapshot.total_requests = total_requests_.load(std::memory_order_relaxed);
    snapshot.max_batch_size_seen = max_batch_size_seen_.load(std::memory_order_relaxed);
    return snapshot;
}

/**
 * @brief Worker thread body: repeatedly batches and processes requests.
 * @details Runs until stop() has been called AND the queue is fully
 * drained -- an empty batch alone is not sufficient to exit, since it may
 * simply mean the timeout elapsed with nothing queued while still running.
 * This ordering guarantees requests queued just before shutdown are still
 * processed rather than silently dropped.
 *
 * Each drained request is first checked against `config_.input_elems`. A
 * mis-sized input is failed immediately and dropped from the batch, because
 * concatenating it would shift every following row's slice boundary and
 * silently corrupt other callers' results.
 *
 * For each surviving batch: concatenates the batch's inputs into one
 * contiguous buffer (request `i`'s input occupies input-row `i`), calls the
 * engine, then slices output-row `i` back out for request `i` and resolves
 * its promise. If the engine call throws, every request in the batch has
 * the exception set on its promise instead, so no caller is left waiting
 * on a future that will never resolve.
 */
void Scheduler::run_loop() {
    while (true) {
        std::vector<InferenceRequest> drained = queue_.wait_and_drain(config_.max_batch_size, config_.max_wait);
        if (drained.empty()) {
            if (!running_) {
                break;
            }
            continue;
        }

        std::vector<InferenceRequest> batch;
        batch.reserve(drained.size());
        for (InferenceRequest& req : drained) {
            if (req.input_data.size() != config_.input_elems) {
                req.result_promise.set_exception(std::make_exception_ptr(
                    InferenceError("input size does not match configured input_elems")));
                continue;
            }
            batch.push_back(std::move(req));
        }
        if (batch.empty()) {
            continue;
        }

        total_batches_.fetch_add(1, std::memory_order_relaxed);
        total_requests_.fetch_add(batch.size(), std::memory_order_relaxed);
        std::uint64_t previous_max = max_batch_size_seen_.load(std::memory_order_relaxed);
        while (batch.size() > previous_max &&
               !max_batch_size_seen_.compare_exchange_weak(previous_max, batch.size(),
                                                            std::memory_order_relaxed)) {
        }

        std::vector<float> batched_input;
        batched_input.reserve(batch.size() * config_.input_elems);
        for (const InferenceRequest& req : batch) {
            batched_input.insert(batched_input.end(), req.input_data.begin(),
                                  req.input_data.end());
        }

        try {
            std::vector<float> batched_output = engine_->run_inference(
                batched_input, batch.size(), config_.input_elems, config_.output_elems);

            if (batched_output.size() != batch.size() * config_.output_elems) {
                throw InferenceError("engine returned an unexpected output size");
            }

            for (std::size_t i = 0; i < batch.size(); ++i) {
                InferenceResult result;
                result.request_id = batch[i].request_id;
                result.logits.assign(
                    batched_output.begin() + i * config_.output_elems,
                    batched_output.begin() + (i + 1) * config_.output_elems);
                batch[i].result_promise.set_value(std::move(result));
            }
        } catch (...) {
            std::exception_ptr eptr = std::current_exception();
            for (InferenceRequest& req : batch) {
                req.result_promise.set_exception(eptr);
            }
        }      
    }
}

}