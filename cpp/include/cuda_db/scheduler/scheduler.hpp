#pragma once

#include <atomic>
#include <thread>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <future>

#include "cuda_db/scheduler/request_queue.hpp"
#include "cuda_db/engine/execution_engine.hpp"

namespace cuda_db {
struct SchedulerConfig {
    std::size_t max_batch_size = 1;
    std::chrono::milliseconds max_wait{0};
    std::size_t input_elems = 0;
    std::size_t output_elems = 0;
};

// Snapshot of what the worker thread has processed so far. Used by the
// /healthz endpoint and the benchmark harness to show that requests actually
// coalesced into batches instead of being served one at a time.
struct SchedulerStats {
    std::uint64_t total_batches = 0;
    std::uint64_t total_requests = 0;
    std::uint64_t max_batch_size_seen = 0;
};


struct Scheduler {
    Scheduler(std::shared_ptr<IExecutionEngine> engine, SchedulerConfig config);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void start();
    void stop();
    std::future<InferenceResult> submit(InferenceRequest&& request);
    SchedulerStats stats() const;

    std::shared_ptr<IExecutionEngine> engine_;
    SchedulerConfig config_;
    RequestQueue queue_;
    std::thread worker_;
    std::atomic<bool> running_{false};

    std::atomic<std::uint64_t> total_batches_{0};
    std::atomic<std::uint64_t> total_requests_{0};
    std::atomic<std::uint64_t> max_batch_size_seen_{0};

private:
    void run_loop();
};

}
