# Benchmarks

Measures what dynamic batching actually buys: throughput and latency against an
otherwise identical server configured to serve one request at a time.

## Running

On the GPU box, with a model exported (`python models/export_resnet.py`):

```bash
pip install -e ".[dev,bench]"

CUDA_DB_MODEL_PATH=models/resnet50.onnx python benchmarks/load_test.py \
    --mode both --concurrency 1,2,4,8,16,32 --num-requests 300 --warmup 20

python benchmarks/plot_results.py
```

Results land in `benchmarks/results/` as JSON, CSV, and two PNGs. The harness
spawns its own uvicorn per mode, since batching config is read at startup;
`--url` targets an already-running server instead.

## Methodology

Each of these is a way the number could have been wrong.

**Both modes run the same code path.** "Serial" is the ordinary server with
`CUDA_DB_MAX_BATCH_SIZE=1` and `CUDA_DB_MAX_WAIT_MS=0` — not a separate endpoint
or a bypass. The comparison isolates batching rather than comparing two
implementations.

**Requests use a binary endpoint.** `POST /predict/raw` takes the tensor as raw
little-endian float32. The JSON endpoint cannot be used for measurement:

| | JSON | binary |
|---|---|---|
| payload for one 3×224×224 input | 3.11 MB | 0.60 MB |
| encode + decode cost | ~90 ms | ~0 (`np.frombuffer` is a view) |

ResNet-50 on an A4000 runs in single-digit to low-teens milliseconds, so JSON
would have dominated the measurement. Worse, that ~90 ms is paid *per request*
regardless of batching, so it would have compressed the ratio between the two
modes and **understated** the batching win.

**Warmup is discarded.** CUDA context creation, cuDNN algorithm selection, and
ORT graph optimization all land on the first requests — hundreds of milliseconds
that would otherwise dominate p99. Default 20 requests, thrown away before the
clock starts. Server counters are also re-read after warmup, so reported batch
statistics describe the measured phase only.

**Percentiles are nearest-rank, not interpolated.** A reported p99 is an
observation that actually happened rather than a synthesized value between two
samples.

**The run aborts rather than reporting a misleading number** when:
- `/healthz` does not report `engine == "onnx"` — otherwise the stub engine is
  being benchmarked, and it does no real work
- dynamic mode never coalesced (`max_batch_size_seen <= 1`) — otherwise the two
  "modes" are the same configuration compared against itself

`--allow-stub` bypasses both, for testing the harness itself. Numbers produced
that way describe the harness, not inference.

## Reading the results

**Observed mean batch size** (`total_requests / total_batches`) is reported
alongside throughput, because it is what explains the throughput. A 3× speedup
with a mean batch of 3.1 is a coherent story; a 3× speedup with a mean batch of
1.2 means something else is going on and the number should not be trusted.

Latency is expected to get *worse* at low concurrency under dynamic batching:
with nothing to batch against, a request simply waits out `max_wait_ms` before
being served alone. That is the tradeoff the system makes, not a defect. The win
appears once arrival rate is high enough that the batch fills before the timer
expires.

The stub engine inverts the result entirely — it does no work, so the batching
window is pure added latency with nothing to amortize. This is a useful sanity
check that the harness measures what it claims to.

## Results

Hardware: NVIDIA RTX A4000 (16 GB, 9 vCPU), ResNet-50 via ONNX Runtime CUDA EP,
`max_batch_size=8`, `max_wait_ms=5`, 300 measured requests per point after 50
discarded warmup requests.

### Headline

At the concurrency where the frontend is not yet the limit:

| metric @ concurrency 16 | serial (batch 1) | dynamic batching | change |
|---|---|---|---|
| throughput | 415 req/s | **468 req/s** | **1.13x** |
| p50 latency | 35.4 ms | **28.7 ms** | 19% lower |
| p99 latency | **50.2 ms** | 58.3 ms | 16% higher |
| mean batch size | 1.0 | 6.8 | — |

This is the batching tradeoff in its expected form: more throughput and a lower
median, paid for in the tail. Median improves because higher throughput drains
the queue faster; p99 worsens because a request can arrive just after a batch
closes and wait out the full window, and because every batch is padded to a
fixed size (see below) so a nearly-empty batch still costs a full one.

### Where batching does not win, and why

| concurrency | serial | dynamic | speedup |
|---|---|---|---|
| 1 | 155 | 49 | 0.32x |
| 4 | 425 | 145 | 0.34x |
| 8 | 420 | 246 | 0.59x |
| **16** | **415** | **468** | **1.13x** |
| 32 | 402 | 263 | 0.65x |
| 64 | 375 | 142 | 0.38x |

**Below concurrency 16** the loss is structural, not a bug. The ONNX engine pads
every batch to a single fixed shape (see below), so a batch of 1 does 8 rows of
work — 12.1 ms instead of 2.3 ms. Throughput only overtakes one-at-a-time
serving once batches average above ~4.7 of 8.

**Above concurrency 16** neither mode is measuring the batcher. Server-reported
time inside `predict()` stays flat at ~22 ms while the client-observed latency
grows to 232 ms — the gap is the FastAPI/uvicorn frontend saturating at roughly
470 req/s, and serial degrades there too. Those rows describe the HTTP layer.

### The finding that mattered: ORT re-plans on every shape change

The first working end-to-end measurement had dynamic batching running about **5x
slower** than no batching at all. Timing the engine in isolation
(`cpp/tools/bench_engine.cpp`, no scheduler, no HTTP, no Python) separated the
two possible causes:

| fixed batch size | per-request | req/s |
|---|---|---|
| 1 | 2.31 ms | 434 |
| 4 | 1.44 ms | 693 |
| 8 | 1.36 ms | 737 |

So batching *does* pay at the engine — up to 1.7x. But feeding the engine the
batch sizes a scheduler actually produces (random 1-8, every shape pre-warmed)
collapsed it:

| shapes in use | median call | throughput |
|---|---|---|
| 4 (buckets 1,2,4,8) | 76.1 ms | 75 req/s |
| 2 (buckets 4,8) | 77.3 ms | 91 req/s |
| **1 (bucket 8)** | **12.2 ms** | **372 req/s** |

Two shapes cost the same as four. An ORT dynamic-shape session retains a plan
for only the most recent shape, so *any* switching pays a full re-plan —
bucketing to a small set does nothing, and only collapsing to a single shape
works. The fix is in `OnnxExecutionEngine`: pad every batch up to
`max_batch_size`, discard the padded output rows. After it, server-side `exec`
time is a flat 10.9 ms at every concurrency level, matching the standalone
batch-8 measurement exactly.

Two earlier hypotheses were wrong and are recorded here because the measurements
that killed them are the useful part: exhaustive cuDNN algorithm search (changing
it moved nothing) and GIL contention in the response path (measured at 0.079 ms
per response, ~0.6 ms for a batch of eight — three orders of magnitude too small).

### Remaining headroom

The engine offers 1.7x; 1.13x is delivered. The difference is accounted for:

- **~6 ms per batch cycle** in the scheduler/engine path. The batch is
  concatenated into one buffer (~4.8 MB memcpy), then copied again to pad to the
  bucket size. Reusing a persistent padded buffer would remove one copy.
- **~10 ms per request** of HTTP/uvicorn/Pydantic overhead, which is also what
  caps both modes near 470 req/s.

Widening the batching window does not help: at `max_wait_ms=30` batches fill to
7.9 of 8, but throughput *drops* to 432 req/s, because the extra wait is added to
a batch that was already effectively full.
