#pragma once

#include <atomic>
#include <thread>
#include <chrono>
#include <cstddef>
#include <memory>
#include <future>

#include "cuda_db/scheduler/request_queue.hpp"
#include "cuda_db/engine/execution_engine.hpp"

namespace cuda_db { 
struct SchedulerConfig { 
    std::size_t max_batch_size; 
    std::chrono::milliseconds max_wait;
    std::size_t input_elems; 
    std::size_t output_elems; 
};


struct Scheduler { 
    Scheduler(std::shared_ptr<IExecutionEngine> engine, SchedulerConfig config);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void start(); 
    void stop();
    std::future<InferenceResult> submit(InferenceRequest&& request);

    std::shared_ptr<IExecutionEngine> engine_; 
    SchedulerConfig config_;
    RequestQueue queue_; 
    std::thread worker_;
    std::atomic<bool> running_{false};

private: 
    void run_loop();
};

}