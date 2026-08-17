// pybind11 bindings for the C++ scheduler. Deliberately the only file in the
// project that includes pybind11 -- cuda_db_core stays Python-free so it can
// be built and unit-tested on its own.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "cuda_db/engine/execution_engine.hpp"
#include "cuda_db/scheduler/scheduler.hpp"

#ifdef CUDA_DB_ENABLE_ONNX
#include "cuda_db/engine/onnx_execution_engine.hpp"
#endif

namespace py = pybind11;

namespace {

// Python-facing settings object. Mirrors SchedulerConfig but takes the wait
// as a plain integer of milliseconds, since std::chrono types don't cross the
// binding boundary usefully.
struct RuntimeConfig {
    std::size_t max_batch_size = 8;
    std::int64_t max_wait_ms = 5;
    std::size_t input_elems = 3 * 224 * 224;
    std::size_t output_elems = 1000;
    // Empty means "use the stub engine", which is what keeps the server
    // runnable on a machine with no GPU or no exported model.
    std::string model_path;
};

// What Python actually holds: an engine, the scheduler driving it, and the
// request-id counter. Kept here rather than in cuda_db_core because it exists
// purely to give the binding layer something to own.
class InferenceRuntime {
public:
    explicit InferenceRuntime(const RuntimeConfig& config)
        : engine_{make_engine(config)},
          scheduler_{engine_, to_scheduler_config(config)},
          input_elems_{config.input_elems},
          engine_name_{config.model_path.empty() ? "stub" : "onnx"} {
        scheduler_.start();
    }

    ~InferenceRuntime() { scheduler_.stop(); }

    InferenceRuntime(const InferenceRuntime&) = delete;
    InferenceRuntime& operator=(const InferenceRuntime&) = delete;

    // Blocking by design: the caller is expected to run this off the asyncio
    // event loop via run_in_executor. The GIL is released around the wait, so
    // other Python threads keep running while the scheduler batches and
    // executes -- which is what lets concurrent requests coalesce at all.
    //
    // Returns (request_id, output). The id is returned rather than stored on
    // the runtime because several executor threads call this concurrently.
    py::tuple predict(
        py::array_t<float, py::array::c_style | py::array::forcecast> input) {
        const auto elems = static_cast<std::size_t>(input.size());
        if (elems != input_elems_) {
            throw py::value_error("expected " + std::to_string(input_elems_) +
                                  " input elements, got " + std::to_string(elems));
        }

        cuda_db::InferenceRequest request;
        request.request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
        const float* data = input.data();
        request.input_data.assign(data, data + elems);
        request.arrival_time = std::chrono::steady_clock::now();

        const std::uint64_t request_id = request.request_id;
        cuda_db::InferenceResult result;
        {
            py::gil_scoped_release release;
            std::future<cuda_db::InferenceResult> future = scheduler_.submit(std::move(request));
            result = future.get();
        }

        // Re-acquired the GIL by here, so it is safe to allocate the output.
        py::array_t<float> output(static_cast<py::ssize_t>(result.logits.size()));
        std::copy(result.logits.begin(), result.logits.end(), output.mutable_data());
        return py::make_tuple(request_id, std::move(output));
    }

    // Idempotent: Scheduler::stop() is safe to call more than once, so an
    // explicit shutdown() followed by the destructor is fine.
    void shutdown() { scheduler_.stop(); }

    py::dict stats() const {
        const cuda_db::SchedulerStats snapshot = scheduler_.stats();
        py::dict out;
        // Surfaced so /healthz can answer "is the real model serving?". A stub
        // silently standing in for ONNX returns plausibly-shaped output, so
        // there is otherwise no way to tell from the response alone.
        out["engine"] = engine_name_;
        out["total_batches"] = snapshot.total_batches;
        out["total_requests"] = snapshot.total_requests;
        out["max_batch_size_seen"] = snapshot.max_batch_size_seen;
        out["queue_depth"] = static_cast<std::uint64_t>(scheduler_.queue_.size());
        // Milliseconds, matching PredictionResponse.latency_ms, so the
        // client-observed and server-side numbers are directly comparable.
        out["avg_queue_wait_ms"] = snapshot.avg_queue_wait_ms();
        out["max_queue_wait_ms"] = snapshot.max_queue_wait_ms();
        out["avg_exec_ms"] = snapshot.avg_exec_ms();
        return out;
    }

private:
    // Chooses the backend from the config. An empty model_path selects the
    // stub; a non-empty one requires an ONNX-enabled build, and says so
    // explicitly rather than silently serving echoed inputs -- a server that
    // looks healthy while returning fake predictions is the worse failure.
    static std::shared_ptr<cuda_db::IExecutionEngine> make_engine(const RuntimeConfig& config) {
        if (config.model_path.empty()) {
            return std::make_shared<cuda_db::StubExecutionEngine>();
        }
#ifdef CUDA_DB_ENABLE_ONNX
        cuda_db::OnnxExecutionEngine::Options options;
        options.model_path = config.model_path;
        // One shape, equal to the largest batch the scheduler can produce, so
        // every batch pads up to it and ORT never re-plans. Derived from the
        // scheduler config rather than left at a default: a bucket smaller than
        // max_batch_size would let oversized batches through unpadded, and a
        // larger one would waste compute on every call. See the measurements in
        // onnx_execution_engine.hpp for why a single shape is required.
        options.batch_buckets = {config.max_batch_size};
        return std::make_shared<cuda_db::OnnxExecutionEngine>(std::move(options));
#else
        throw std::runtime_error(
            "model_path was set but this build has no ONNX Runtime support; "
            "reconfigure with -DCUDA_DB_ENABLE_ONNX=ON");
#endif
    }

    static cuda_db::SchedulerConfig to_scheduler_config(const RuntimeConfig& config) {
        cuda_db::SchedulerConfig out;
        out.max_batch_size = config.max_batch_size;
        out.max_wait = std::chrono::milliseconds(config.max_wait_ms);
        out.input_elems = config.input_elems;
        out.output_elems = config.output_elems;
        return out;
    }

    // Held as the interface, not a concrete type, so the backend is a
    // construction-time choice.
    std::shared_ptr<cuda_db::IExecutionEngine> engine_;
    cuda_db::Scheduler scheduler_;
    std::size_t input_elems_;
    std::string engine_name_;
    std::atomic<std::uint64_t> next_request_id_{0};
};

}  // namespace

PYBIND11_MODULE(cuda_db_native, m) {
    m.doc() = "Native scheduler and execution engine for cuda_db.";

    // Engine/scheduler failures arrive in Python as a catchable exception
    // rather than an abort.
    py::register_exception<cuda_db::InferenceError>(m, "InferenceError");

    py::class_<RuntimeConfig>(m, "RuntimeConfig")
        .def(py::init([](std::size_t max_batch_size, std::int64_t max_wait_ms,
                         std::size_t input_elems, std::size_t output_elems,
                         std::string model_path) {
                 RuntimeConfig config;
                 config.max_batch_size = max_batch_size;
                 config.max_wait_ms = max_wait_ms;
                 config.input_elems = input_elems;
                 config.output_elems = output_elems;
                 config.model_path = std::move(model_path);
                 return config;
             }),
             py::arg("max_batch_size") = 8, py::arg("max_wait_ms") = 5,
             py::arg("input_elems") = 3 * 224 * 224, py::arg("output_elems") = 1000,
             py::arg("model_path") = "")
        .def_readwrite("max_batch_size", &RuntimeConfig::max_batch_size)
        .def_readwrite("max_wait_ms", &RuntimeConfig::max_wait_ms)
        .def_readwrite("input_elems", &RuntimeConfig::input_elems)
        .def_readwrite("output_elems", &RuntimeConfig::output_elems)
        .def_readwrite("model_path", &RuntimeConfig::model_path);

    py::class_<InferenceRuntime>(m, "InferenceRuntime")
        .def(py::init<const RuntimeConfig&>(), py::arg("config"))
        .def("predict", &InferenceRuntime::predict, py::arg("input"),
             "Run one request through the batching scheduler and return "
             "(request_id, output). Blocks; call it from a thread-pool "
             "executor, not the event loop.")
        .def("shutdown", &InferenceRuntime::shutdown)
        .def("stats", &InferenceRuntime::stats)
        .def("__enter__", [](InferenceRuntime& self) -> InferenceRuntime& { return self; })
        .def("__exit__", [](InferenceRuntime& self, const py::object&, const py::object&,
                            const py::object&) { self.shutdown(); });
}
