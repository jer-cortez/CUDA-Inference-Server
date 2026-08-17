"""Dynamic batching vs. serial (batch-size-1) throughput and latency.

Produces the project's headline number. Run on the GPU box:

    CUDA_DB_MODEL_PATH=models/resnet50.onnx python benchmarks/load_test.py \\
        --mode both --concurrency 1,2,4,8,16,32 --num-requests 300 --warmup 20

Fairness rules baked in, because each one is a way the result could be wrong:

* Both modes run the SAME server code path. "Serial" is the ordinary server
  configured with max_batch_size=1 / max_wait_ms=0, never a separate route, so
  the comparison isolates batching rather than two implementations.
* Warmup requests are discarded. CUDA context creation, cuDNN algorithm
  selection and ORT graph optimization all land on the first few requests and
  would otherwise dominate p99.
* Sanity gates abort the run if the stub engine is serving, or if nothing
  actually coalesced in dynamic mode -- either would yield a confident,
  meaningless number.
* Requests use the binary /predict/raw endpoint. JSON costs ~90 ms per ResNet
  request in encode+decode, which is paid per request regardless of batching
  and would understate the batching win.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import RequestResult, make_payload, send_request  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_INPUT_ELEMS = 3 * 224 * 224
DEFAULT_OUTPUT_ELEMS = 1000

# Server-side batching configuration per mode. Everything else is held equal.
MODE_CONFIG = {
    "dynamic": {"CUDA_DB_MAX_BATCH_SIZE": "8", "CUDA_DB_MAX_WAIT_MS": "5"},
    "serial": {"CUDA_DB_MAX_BATCH_SIZE": "1", "CUDA_DB_MAX_WAIT_MS": "0"},
}


@dataclass
class RunResult:
    mode: str
    concurrency: int
    num_requests: int
    failures: int
    wall_clock_s: float
    throughput_rps: float
    latency_ms: dict[str, float]
    server_stats: dict = field(default_factory=dict)

    @property
    def observed_mean_batch(self) -> float:
        """Requests per batch, as the server actually saw it.

        This is the number that *explains* the throughput result, so it is
        reported alongside rather than left implicit.
        """
        batches = self.server_stats.get("total_batches") or 0
        requests = self.server_stats.get("total_requests") or 0
        return requests / batches if batches else 0.0


def percentiles(values_ms: list[float]) -> dict[str, float]:
    ordered = sorted(values_ms)

    def pct(p: float) -> float:
        if not ordered:
            return 0.0
        # Nearest-rank, clamped. Avoids interpolation so a reported p99 is an
        # observation that actually happened rather than a synthesized value.
        index = min(len(ordered) - 1, max(0, int(round(p / 100.0 * len(ordered))) - 1))
        return ordered[index]

    return {
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "mean": statistics.fmean(ordered) if ordered else 0.0,
        "min": ordered[0] if ordered else 0.0,
        "max": ordered[-1] if ordered else 0.0,
    }


async def wait_until_ready(base_url: str, timeout_s: float = 180.0) -> dict:
    """Poll /healthz until the server answers.

    The timeout is generous because loading ResNet-50 and initializing the CUDA
    execution provider can take tens of seconds on a cold process.
    """
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/healthz")
                if response.status_code == 200:
                    return response.json()
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError(f"server at {base_url} did not become ready within {timeout_s}s")


async def fetch_stats(base_url: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        return (await client.get(f"{base_url}/healthz")).json()


def check_engine(stats: dict, require_onnx: bool) -> None:
    engine = stats.get("engine", "unknown")
    if require_onnx and engine != "onnx":
        raise SystemExit(
            f"refusing to benchmark: server reports engine={engine!r}, not 'onnx'. "
            "Set CUDA_DB_MODEL_PATH to an exported model, or pass --allow-stub "
            "if you are deliberately measuring the harness itself."
        )


async def run_phase(
    base_url: str,
    payload: bytes,
    count: int,
    concurrency: int,
) -> list[RequestResult]:
    """Fire `count` requests holding `concurrency` of them in flight."""
    semaphore = asyncio.Semaphore(concurrency)
    # Client-side connection pool must not be the bottleneck: if it were
    # smaller than the target concurrency, requests would queue in the client
    # and never reach the scheduler together.
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)

    async with httpx.AsyncClient(timeout=300.0, limits=limits) as client:
        url = f"{base_url}/predict/raw"

        async def one() -> RequestResult:
            async with semaphore:
                return await send_request(client, url, payload)

        return await asyncio.gather(*(one() for _ in range(count)))


async def measure(
    base_url: str,
    mode: str,
    concurrency: int,
    num_requests: int,
    warmup: int,
    input_elems: int,
    require_onnx: bool,
) -> RunResult:
    payload = make_payload(input_elems)

    stats_before = await fetch_stats(base_url)
    check_engine(stats_before, require_onnx)

    if warmup:
        await run_phase(base_url, payload, warmup, concurrency)

    # Read counters after warmup so the reported batching stats describe the
    # measured phase only.
    baseline = await fetch_stats(base_url)

    started = time.perf_counter()
    results = await run_phase(base_url, payload, num_requests, concurrency)
    wall_clock_s = time.perf_counter() - started

    stats_after = await fetch_stats(base_url)
    measured_stats = {
        key: stats_after.get(key, 0) - baseline.get(key, 0)
        for key in ("total_batches", "total_requests")
    }
    # total_batches/total_requests above are differenced, so they describe this
    # run. The rest are the scheduler's own cumulative counters and cannot be
    # differenced meaningfully -- a high-water mark and two running averages --
    # so they are named to say so. observed_mean_batch, derived from the
    # differenced pair, is the per-run figure to quote.
    measured_stats["max_batch_size_seen"] = stats_after.get("max_batch_size_seen", 0)
    measured_stats["cumulative_avg_queue_wait_ms"] = stats_after.get("avg_queue_wait_ms", 0.0)
    measured_stats["cumulative_avg_exec_ms"] = stats_after.get("avg_exec_ms", 0.0)
    measured_stats["engine"] = stats_after.get("engine", "unknown")

    ok = [r for r in results if r.ok]
    failures = len(results) - len(ok)

    # Only meaningful above concurrency 1: with a single client there is never
    # more than one request in flight, so a batch size of 1 is the correct
    # outcome, not a failure. Enforced above that, where failing to coalesce
    # would mean the two modes are the same configuration compared against
    # itself.
    if (
        mode == "dynamic"
        and require_onnx
        and concurrency > 1
        and measured_stats["max_batch_size_seen"] <= 1
    ):
        raise SystemExit(
            f"refusing to report: dynamic mode never coalesced at concurrency={concurrency} "
            "(max_batch_size_seen <= 1), so this would be two identical configurations "
            "compared against each other. Raise CUDA_DB_MAX_WAIT_MS, or check that "
            "--executor-workers is at least the concurrency level."
        )

    return RunResult(
        mode=mode,
        concurrency=concurrency,
        num_requests=num_requests,
        failures=failures,
        wall_clock_s=wall_clock_s,
        throughput_rps=len(ok) / wall_clock_s if wall_clock_s > 0 else 0.0,
        latency_ms=percentiles([r.latency_s * 1000.0 for r in ok]),
        server_stats=measured_stats,
    )


class ServerProcess:
    """Runs uvicorn with a mode's batching config, since those are read at startup."""

    def __init__(self, mode: str, port: int, model_path: str, input_elems: int,
                 output_elems: int, executor_workers: int):
        self.mode = mode
        self.port = port
        env = os.environ.copy()
        env.update(MODE_CONFIG[mode])
        env["CUDA_DB_INPUT_ELEMS"] = str(input_elems)
        env["CUDA_DB_OUTPUT_ELEMS"] = str(output_elems)
        env["CUDA_DB_EXECUTOR_WORKERS"] = str(executor_workers)
        if model_path:
            env["CUDA_DB_MODEL_PATH"] = model_path
        self._env = env
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> str:
        self._process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "cuda_db.server.app:app",
             "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "warning"],
            cwd=REPO_ROOT,
            env=self._env,
        )
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *_exc) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)


def write_results(results: list[RunResult], output_stem: str, metadata: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / f"{output_stem}.json"
    payload = {
        "metadata": metadata,
        "runs": [
            {**asdict(run), "observed_mean_batch": round(run.observed_mean_batch, 2)}
            for run in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = RESULTS_DIR / f"{output_stem}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["mode", "concurrency", "throughput_rps", "p50_ms", "p90_ms", "p99_ms",
             "mean_ms", "observed_mean_batch", "failures"]
        )
        for run in results:
            writer.writerow([
                run.mode, run.concurrency, round(run.throughput_rps, 2),
                round(run.latency_ms["p50"], 2), round(run.latency_ms["p90"], 2),
                round(run.latency_ms["p99"], 2), round(run.latency_ms["mean"], 2),
                round(run.observed_mean_batch, 2), run.failures,
            ])

    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def print_summary(results: list[RunResult]) -> None:
    print(f"\n{'mode':<9}{'conc':>6}{'rps':>10}{'p50 ms':>10}{'p90 ms':>10}"
          f"{'p99 ms':>10}{'batch':>8}{'fail':>6}")
    print("-" * 69)
    for run in results:
        print(f"{run.mode:<9}{run.concurrency:>6}{run.throughput_rps:>10.1f}"
              f"{run.latency_ms['p50']:>10.1f}{run.latency_ms['p90']:>10.1f}"
              f"{run.latency_ms['p99']:>10.1f}{run.observed_mean_batch:>8.1f}"
              f"{run.failures:>6}")

    # The headline comparison, printed only where both modes ran the same
    # concurrency so the speedup is apples-to-apples.
    by_key = {(r.mode, r.concurrency): r for r in results}
    shared = sorted({c for m, c in by_key if (("dynamic", c) in by_key and ("serial", c) in by_key)})
    if shared:
        print(f"\n{'conc':>6}{'speedup':>10}{'p99 change':>14}")
        print("-" * 30)
        for conc in shared:
            dyn, ser = by_key[("dynamic", conc)], by_key[("serial", conc)]
            speedup = dyn.throughput_rps / ser.throughput_rps if ser.throughput_rps else 0.0
            p99_delta = dyn.latency_ms["p99"] - ser.latency_ms["p99"]
            print(f"{conc:>6}{speedup:>9.2f}x{p99_delta:>13.1f}ms")


async def main_async(args: argparse.Namespace) -> None:
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    modes = ["dynamic", "serial"] if args.mode == "both" else [args.mode]
    model_path = os.environ.get("CUDA_DB_MODEL_PATH", "")
    require_onnx = not args.allow_stub

    results: list[RunResult] = []

    for mode in modes:
        if args.url:
            # External server: its batching config is whatever the operator set,
            # so only one mode can be measured per invocation.
            base_url = args.url
            await wait_until_ready(base_url)
            for concurrency in concurrencies:
                results.append(await measure(
                    base_url, mode, concurrency, args.num_requests, args.warmup,
                    args.input_elems, require_onnx))
        else:
            with ServerProcess(mode, args.port, model_path, args.input_elems,
                               args.output_elems, args.executor_workers) as base_url:
                await wait_until_ready(base_url)
                for concurrency in concurrencies:
                    print(f"running {mode} @ concurrency={concurrency} ...", flush=True)
                    results.append(await measure(
                        base_url, mode, concurrency, args.num_requests, args.warmup,
                        args.input_elems, require_onnx))

    print_summary(results)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name(),
        "model_path": model_path or "(stub engine)",
        "input_elems": args.input_elems,
        "output_elems": args.output_elems,
        "executor_workers": args.executor_workers,
        "warmup_requests": args.warmup,
        "num_requests": args.num_requests,
        "mode_config": MODE_CONFIG,
    }
    write_results(results, args.output, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["dynamic", "serial", "both"], default="both")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32",
                        help="comma-separated concurrency levels")
    parser.add_argument("--num-requests", type=int, default=300,
                        help="measured requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=50,
                        help="discarded requests before measuring; do not set to 0 "
                             "unless the server is already warm. Sized to cover the "
                             "range of batch shapes a run will produce, not just the "
                             "first one: ORT allocates workspace per input shape, and "
                             "a dynamic batcher emits several")
    parser.add_argument("--url", default="",
                        help="benchmark an already-running server instead of spawning one")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--input-elems", type=int, default=DEFAULT_INPUT_ELEMS)
    parser.add_argument("--output-elems", type=int, default=DEFAULT_OUTPUT_ELEMS)
    parser.add_argument("--executor-workers", type=int, default=16)
    parser.add_argument("--output", default="latest",
                        help="output file stem under benchmarks/results/")
    parser.add_argument("--allow-stub", action="store_true",
                        help="permit benchmarking the stub engine (harness self-test only; "
                             "the resulting numbers say nothing about inference)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
