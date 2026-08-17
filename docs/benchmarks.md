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

<!-- Fill in after the first real run on the GPU box. Include: GPU model, ORT
version, model, and the throughput/p99 figures at the concurrency level where
the modes diverge most. Paste the two PNGs from benchmarks/results/. -->

_Not yet recorded._
