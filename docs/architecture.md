# Architecture

How a request travels through the system, and why each layer is built the way it
is. Measurements behind the performance claims are in [benchmarks.md](benchmarks.md).

## Request lifecycle

1. Pure **ASGI admission middleware** reserves one of a fixed number of
   request slots before reading the body. It enforces the streamed byte limit
   and one deadline across upload, parsing, executor queueing, and inference.
2. **FastAPI** parses `POST /predict` (JSON) or `POST /predict/raw` (binary
   float32) and validates tensor length and finite values.
3. The handler submits the array to a dedicated **thread-pool executor** and
   awaits its wrapped future. This is the load-bearing step: the native call blocks, and
   running it on the event loop would serialize every request and prevent any
   batch from ever forming.
4. The **pybind11 binding** copies the tensor into an `InferenceRequest`,
   releases the GIL, submits to the scheduler, and blocks on a `std::future`.
   Releasing the GIL is what lets other executor threads keep submitting while
   this one waits — without it the "concurrency" would be nominal.
5. The **request queue** accepts the request and wakes the worker.
6. The **scheduler worker** drains a batch, concatenates the inputs, calls the
   engine, then resolves each request's promise with its own output row.
7. The future completes, the binding reacquires the GIL, and the response
   returns up the same path.

The admission permit belongs jointly to the HTTP task and its native future.
If the client disconnects or the deadline expires after submission, the HTTP
task ends but its slot stays occupied until native inference actually returns.
Async cancellation cannot stop a C++/CUDA call, so releasing sooner would let
replacement requests accumulate behind work that is still using the GPU.

## Threading model

Three kinds of thread, deliberately:

- **The event loop** (one) never blocks. It only parses requests and dispatches.
- **Executor threads** (default 8) block inside `predict()` with the GIL
  released. Admission provides the end-to-end bound; requests above the worker
  count can wait in the bounded executor queue.
- **The scheduler worker** (one) owns all batching and inference. Single by
  design: one GPU, one ONNX Runtime session, and a second worker would contend
  for both while making batch composition nondeterministic.

Everything below the binding is Python-free, which is what allows the C++ core
to be built and unit-tested without an interpreter.

## Components

### RequestQueue — `cpp/src/scheduler/request_queue.cpp`

A `std::deque` behind a mutex and condition variable. `wait_and_drain` is the
whole batching primitive, in two phases:

1. Park until the queue is non-empty (or shutdown).
2. Then wait up to `max_wait` for the queue to reach `max_batch_size`.

The two phases exist so the batching window is timed **from the first arrival**,
not from when the worker happened to call in. A single `wait_for` would start the
clock at an arbitrary moment and produce a window that varies with worker timing.

An idle server parks in phase 1 indefinitely rather than spinning on a timeout —
there is a test asserting the worker does not wake when nothing is queued.

*Why a plain mutex rather than a lock-free queue:* batching windows are
milliseconds and there is one consumer. A lock-free MPMC queue would add real
complexity to save microseconds that the 5 ms window makes irrelevant.

### Scheduler — `cpp/src/scheduler/scheduler.cpp`

Owns the queue, the worker thread, and a `shared_ptr<IExecutionEngine>`.

The worker loop only exits when the queue is empty **and** stop has been
requested. Checking the stop flag first would drop requests that arrived just
before shutdown, leaving their futures unfulfilled — callers would see
`broken_promise` rather than an answer.

Engine exceptions are caught and set on *every* promise in the batch. A batch
that throws must not leave any caller blocked forever on a future.

Row `i` of the batch maps to row `i` of the output. That invariant is what a
batching bug would silently violate, so it is tested directly rather than
inferred from output shape.

### Execution engines — `cpp/src/engine/`

Three implementations behind one interface, chosen at construction:

| engine | purpose |
|---|---|
| `StubExecutionEngine` | echoes input; no GPU. Keeps the laptop dev loop and CPU-only CI meaningful |
| `CudaExecutionEngine` | pinned staging → H2D → `normalize` kernel → D2H. Exercises the memory/stream plumbing without a model |
| `OnnxExecutionEngine` | the real path: ONNX Runtime session on the CUDA execution provider |

`/healthz` reports which one is live, because a stub returns plausibly-shaped
output and is otherwise indistinguishable from the real model in a response.

**Single-shape padding.** The ONNX engine pads every batch up to
`max_batch_size` and discards the extra output rows. ONNX Runtime retains a
compiled plan for only the most recent input shape, so a scheduler emitting a
different size per batch triggers a re-plan almost every call — measured at a
10x per-request penalty, enough to make batching slower than not batching. Two
distinct shapes cost as much as four, so only collapsing to one works. The cost
is wasted compute on padded rows; the benefit is an order of magnitude more.

**Device-resident I/O.** The engine preallocates pinned host and device buffers
sized for the largest bucket and binds the device tensors through
`Ort::IoBinding`, so ORT allocates and copies nothing per call and the padded
tail of the input buffer is zeroed once rather than per request. The original
host-pointer path is retained behind `use_io_binding` as both a fallback and the
reference the bound path is tested against — the two must agree bit-for-bit.

This measured ~6% faster at the engine and produced no end-to-end throughput
change, because by then the engine was not the constraint. See
[benchmarks.md](benchmarks.md#device-resident-io-a-real-engine-gain-that-bought-no-throughput).

**Normalization boundary.** ResNet-50 expects ImageNet-normalized input. That
happens once, on the CPU, in `python/cuda_db/preprocessing/image_utils.py`. The
`normalize` CUDA kernel deliberately does **not** run in the ONNX path —
applying both would double-normalize and produce confident nonsense that still
looks shape-correct. The kernel belongs to the device-resident path, where input
never returns to the host.

### Memory — `cpp/src/memory/`

- `DeviceBuffer` / `PinnedBuffer` — move-only RAII around `cudaMalloc` and
  `cudaHostAlloc`. Freeing via destructors means a batch that throws mid-flight
  cannot leak the tensor it already uploaded.
- `MemoryPool` — size-bucketed free lists for both device and pinned memory,
  so no allocation happens on the hot path after warmup. `cudamalloc_calls` in
  its stats is the observable proof: it stops growing in steady state.
- `CudaStreamPool` — a small set of non-blocking streams, round-robin, so
  upload/compute/download can overlap rather than serializing on the default
  stream.

Pinned host memory matters specifically for async copies: a pageable buffer
forces the driver to stage it internally, quietly making an "async" `memcpy`
synchronous.

### Binding — `cpp/src/bindings/module.cpp`

The only file in the project that includes pybind11, which is what keeps
`cuda_db_core` independently buildable.

`predict()` is deliberately blocking rather than returning an awaitable. C++
futures do not integrate with the asyncio event loop, and faking an async API
would hide where the blocking actually happens. Python submits it to the
dedicated executor and adapts its future into an asyncio future explicitly.

## Configuration

| env var | default | effect |
|---|---|---|
| `CUDA_DB_MODEL_PATH` | `""` | empty selects the stub engine |
| `CUDA_DB_MAX_BATCH_SIZE` | 8 | batch cap, and the padded shape the engine uses |
| `CUDA_DB_MAX_WAIT_MS` | 5 | batching window, timed from first arrival |
| `CUDA_DB_EXECUTOR_WORKERS` | 8 | native calls allowed to reach/block in the scheduler concurrently; excess admitted work waits in the bounded executor queue |
| `CUDA_DB_MAX_INFLIGHT_REQUESTS` | 16 | immediate admission cap, acquired before body upload |
| `CUDA_DB_MAX_REQUEST_BYTES` | 4194304 | streamed request-body cap, with or without `Content-Length` |
| `CUDA_DB_REQUEST_TIMEOUT_MS` | 30000 | deadline across upload, validation, queueing, and inference |
| `CUDA_DB_REQUIRE_GPU` | false | require a non-empty model path and CUDA-backed ONNX engine |

`max_wait_ms` is the main tuning knob in principle, but measurement showed
widening it past the default does not help here: it fills batches from 7.0 to
7.9 of 8 while adding wait time, and net throughput falls.

The load-test admission limit must be at least the concurrency under test unless
the purpose is to measure overload rejection. Settings are validated at startup;
sizes, worker counts, capacities, and deadlines must be positive.
JSON parsing is synchronous Python work, so it cannot be interrupted in the
middle of a parse; the handler checks the deadline immediately afterward and
before native submission. Once submitted, native GPU work also cannot be
preempted safely, so a timed-out request retains its admission slot until that
work returns.

## Startup and shutdown

Startup runs one zero-input prediction through the executor, scheduler, and
engine and verifies the output shape and finiteness before `/readyz` can return
200. `CUDA_DB_REQUIRE_GPU=true` also requires the ONNX engine; its CUDA provider
configuration plus the probe proves the serving path runs, but does not prove
that every model operator was placed on the GPU. Confirm placement with GPU
utilization or ONNX Runtime profiling during the AWS benchmark.

Shutdown marks the process unready and closes admission first. It waits for all
HTTP requests and detached native futures before stopping the scheduler, so an
executor task cannot submit after the scheduler has stopped. A native driver or
kernel hang cannot be killed safely inside the process; production should give
the process a finite termination grace period and let the supervisor replace it.
This lifecycle assumes one application process per GPU. Multiple Uvicorn workers
would each create a separate model, scheduler, admission cap, and CUDA context.

## Known limits

- **The frontend caps throughput near 470 req/s.** Above concurrency 16 the
  bottleneck is FastAPI/uvicorn, not the scheduler — server-reported time inside
  `predict()` stays flat while client-observed latency climbs.
- **The scheduler still concatenates the batch on the host** (~4.8 MB per batch)
  before the engine sees it. The engine's own padding copy is gone — it stages
  into a reused pinned buffer — but removing this one needs `IExecutionEngine` to
  accept per-request buffers rather than one pre-joined vector, so the packing
  could happen on-device instead.
- **Low concurrency is worse than no batching**, structurally. Padding to a fixed
  shape means a batch of 1 does `max_batch_size` rows of work.
- **Admission is process-local.** The documented limits and readiness lifecycle
  assume the recommended single Uvicorn worker. A future multi-process deployment
  needs an external/global admission layer.
