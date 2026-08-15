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
    // Defaults to the same 5ms the Python layer uses, so a default-constructed
    // config batches rather than serving every request the instant it lands.
    std::chrono::milliseconds max_wait{5};
    std::size_t input_elems = 0;
    std::size_t output_elems = 0;
};

// Snapshot of what the worker thread has processed so far. Used by the
// /healthz endpoint and the benchmark harness to show that requests actually
// coalesced into batches instead of being served one at a time.
//
// The latency counters split the two halves of a request's server-side time:
// queue wait is how long it sat waiting for batch-mates (i.e. what a larger
// max_wait buys or costs), and exec time is what the engine spent on the
// batch. Together they explain a throughput/latency trade-off that total
// wall-clock time alone does not.
struct SchedulerStats {
    std::uint64_t total_batches = 0;
    std::uint64_t total_requests = 0;
    std::uint64_t max_batch_size_seen = 0;

    // Summed over every request: dequeue time minus arrival_time.
    std::uint64_t total_queue_wait_us = 0;
    std::uint64_t max_queue_wait_us = 0;
    // Summed over every batch: wall time inside the engine call.
    std::uint64_t total_exec_us = 0;

    // Zero rather than NaN before the first request, so callers (and the JSON
    // encoder behind /healthz) never have to special-case a cold server.
    double avg_queue_wait_ms() const {
        if (total_requests == 0) return 0.0;
        return static_cast<double>(total_queue_wait_us) / total_requests / 1000.0;
    }
    double max_queue_wait_ms() const {
        return static_cast<double>(max_queue_wait_us) / 1000.0;
    }
    double avg_exec_ms() const {
        if (total_batches == 0) return 0.0;
        return static_cast<double>(total_exec_us) / total_batches / 1000.0;
    }
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
    std::atomic<std::uint64_t> total_queue_wait_us_{0};
    std::atomic<std::uint64_t> max_queue_wait_us_{0};
    std::atomic<std::uint64_t> total_exec_us_{0};

private:
    void run_loop();
};

}
