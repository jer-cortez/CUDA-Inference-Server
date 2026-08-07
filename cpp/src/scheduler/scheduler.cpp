#include "cuda_db/scheduler/scheduler.hpp"

#include <algorithm>

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
 * @brief Worker thread body: repeatedly batches and processes requests.
 * @details Runs until stop() has been called AND the queue is fully
 * drained -- an empty batch alone is not sufficient to exit, since it may
 * simply mean the timeout elapsed with nothing queued while still running.
 * This ordering guarantees requests queued just before shutdown are still
 * processed rather than silently dropped.
 *
 * For each non-empty batch: concatenates the batch's inputs into one
 * contiguous buffer (request `i`'s input occupies input-row `i`), calls the
 * engine, then slices output-row `i` back out for request `i` and resolves
 * its promise. If the engine call throws, every request in the batch has
 * the exception set on its promise instead, so no caller is left waiting
 * on a future that will never resolve.
 */
void Scheduler::run_loop() {
    while (true) { 
        std::vector<InferenceRequest> batch = queue_.wait_and_drain(config_.max_batch_size, config_.max_wait);
        if (batch.empty()) { 
            if (!running_) { 
                break;
            }
            continue;
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