# Architecture

How a request travels through the system, and why each layer is built the way it
is. Measurements behind the performance claims are in [benchmarks.md](benchmarks.md).

## Request lifecycle

1. **FastAPI** receives `POST /predict` (JSON) or `POST /predict/raw` (binary
   float32) and validates the tensor length.
2. The handler hands the array to a **thread-pool executor** via
   `run_in_executor`. This is the load-bearing step: the native call blocks, and
   running it on the event loop would serialize every request and prevent any
   batch from ever forming.
3. The **pybind11 binding** copies the tensor into an `InferenceRequest`,
   releases the GIL, submits to the scheduler, and blocks on a `std::future`.
   Releasing the GIL is what lets other executor threads keep submitting while
   this one waits — without it the "concurrency" would be nominal.
4. The **request queue** accepts the request and wakes the worker.
5. The **scheduler worker** drains a batch, concatenates the inputs, calls the
   engine, then resolves each request's promise with its own output row.
6. The future completes, the binding reacquires the GIL, and the response
   returns up the same path.

## Threading model

Three kinds of thread, deliberately:

- **The event loop** (one) never blocks. It only parses requests and dispatches.
- **Executor threads** (default 8) block inside `predict()` with the GIL
  released. Their count bounds how many requests can be in flight at the
  scheduler at once, so it must be at least the target concurrency.
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
would hide where the blocking actually happens. Offloading is Python's job, in
one visible `run_in_executor` call.

## Configuration

| env var | default | effect |
|---|---|---|
| `CUDA_DB_MODEL_PATH` | `""` | empty selects the stub engine |
| `CUDA_DB_MAX_BATCH_SIZE` | 8 | batch cap, and the padded shape the engine uses |
| `CUDA_DB_MAX_WAIT_MS` | 5 | batching window, timed from first arrival |
| `CUDA_DB_EXECUTOR_WORKERS` | 8 | must be ≥ target concurrency or requests queue before reaching the scheduler |

`max_wait_ms` is the main tuning knob in principle, but measurement showed
widening it past the default does not help here: it fills batches from 7.0 to
7.9 of 8 while adding wait time, and net throughput falls.

## Known limits

- **The frontend caps throughput near 470 req/s.** Above concurrency 16 the
  bottleneck is FastAPI/uvicorn, not the scheduler — server-reported time inside
  `predict()` stays flat while client-observed latency climbs.
- **~6 ms per batch cycle** is spent copying: once concatenating the batch, once
  padding it. A persistent padded buffer written directly by the scheduler would
  remove one.
- **Low concurrency is worse than no batching**, structurally. Padding to a fixed
  shape means a batch of 1 does `max_batch_size` rows of work.
