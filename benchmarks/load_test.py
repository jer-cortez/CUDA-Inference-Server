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
* A sanity gate aborts if the stub engine is serving. Observed batch size one
  remains valid data under sparse traffic and is reported as measured.
* Requests use the binary /predict/raw endpoint. JSON costs ~90 ms per ResNet
  request in encode+decode, which is paid per request regardless of batching
  and would understate the batching win.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import (  # noqa: E402
    RequestResult, make_async_client, make_json_payload, make_payload,
    send_json_request, send_request,
)

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
    # The server's own measurement of time inside predict(), i.e. queueing plus
    # inference. Reported next to the client-observed figure because the gap
    # between them localizes a bottleneck: a large gap is HTTP/uvicorn/client
    # overhead, a small one means the time really is in the scheduler.
    server_latency_ms: dict[str, float] = field(default_factory=dict)
    server_stats: dict = field(default_factory=dict)
    workload: str = "closed_loop"
    target_rate_rps: float | None = None
    max_outstanding: int | None = None
    offered_requests: int = 0
    attempted_requests: int = 0
    successful_requests: int = 0
    completed_within_window: int = 0
    successful_within_window: int = 0
    missed_arrivals: int = 0
    unfinished_at_cutoff: int = 0
    late_completions: int = 0
    outcome_counts: dict[str, int] = field(default_factory=dict)
    http_status_counts: dict[str, int] = field(default_factory=dict)
    scheduling_delay_ms: dict[str, float] = field(default_factory=dict)
    scheduled_latency_ms: dict[str, float] = field(default_factory=dict)
    measured_started_at: str | None = None
    measured_ended_at: str | None = None
    successful_after_cutoff: int = 0
    failed_after_cutoff: int = 0

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
        index = min(len(ordered) - 1, max(0, math.ceil(p / 100.0 * len(ordered)) - 1))
        return ordered[index]

    return {
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "mean": statistics.fmean(ordered) if ordered else 0.0,
        "min": ordered[0] if ordered else 0.0,
        "max": ordered[-1] if ordered else 0.0,
    }


@dataclass(frozen=True)
class ScheduledResult:
    result: RequestResult
    scheduling_delay_s: float
    scheduled_latency_s: float
    completed_within_window: bool


@dataclass(frozen=True)
class WindowedResult:
    result: RequestResult
    completed_within_window: bool


def outcome_counts(results: list[RequestResult]) -> dict[str, int]:
    """Count response outcomes without collapsing distinct failure causes."""
    counts: dict[str, int] = {}
    for result in results:
        if result.ok:
            key = "success"
        else:
            kind = getattr(result, "failure_kind", None)
            detail = getattr(result, "failure_detail", None)
            key = detail if kind == "http_status" and detail else (kind or f"http_{result.status_code}")
        counts[key] = counts.get(key, 0) + 1
    return counts


def http_status_counts(results: list[RequestResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        if result.status_code:
            key = str(result.status_code)
            counts[key] = counts.get(key, 0) + 1
    return counts


async def send_payload(client, url: str, payload, endpoint: str, output_elems: int | None):
    sender = send_json_request if endpoint == "json" else send_request
    return await sender(client, url, payload, expected_output_elems=output_elems)


class Diagnostics:
    """Read private diagnostics through HTTPS or an operator-supplied argv."""

    def __init__(self, *, base_url: str = "", command: list[str] | None = None,
                 client_factory=None):
        if bool(base_url) == bool(command):
            raise ValueError("configure exactly one diagnostics URL or command")
        self.base_url = base_url.rstrip("/")
        self.command = command
        self.client_factory = client_factory or httpx.AsyncClient

    async def get(self, endpoint: str) -> dict:
        if self.command is not None:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                endpoint,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError(f"diagnostics command timed out for {endpoint}") from None
            if process.returncode != 0:
                raise RuntimeError(
                    f"diagnostics command failed for {endpoint} "
                    f"(exit {process.returncode})"
                )
            try:
                body = json.loads(stdout)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"diagnostics command returned invalid JSON for {endpoint}") from exc
            if not isinstance(body, dict):
                raise RuntimeError(f"diagnostics command returned non-object JSON for {endpoint}")
            return body

        async with self.client_factory(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise RuntimeError(f"diagnostics endpoint returned non-object JSON for {endpoint}")
            return body


def load_diagnostics_command(path: str) -> list[str]:
    command_path = Path(path)
    try:
        value = json.loads(command_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read diagnostics command file {command_path}: {exc}") from exc
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise SystemExit("diagnostics command file must contain a non-empty JSON array of strings")
    return value


def configured_client_factory(
    base_url: str, token_file: str | None, ca_file: str | None, *, credentials: bool = True
):
    """Bind trust/auth settings while allowing each phase to tune its pool."""
    def create(**kwargs):
        return make_async_client(
            base_url,
            token_file=token_file if credentials else None,
            ca_file=ca_file,
            environ=None if credentials else {
                "CUDA_DB_BENCHMARK_CA_FILE": os.environ.get("CUDA_DB_BENCHMARK_CA_FILE", "")
            },
            **kwargs,
        )
    return create


async def wait_until_ready(diagnostics: Diagnostics, timeout_s: float = 180.0) -> dict:
    """Poll private /readyz until the server answers.

    The timeout is generous because loading ResNet-50 and initializing the CUDA
    execution provider can take tens of seconds on a cold process.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = await diagnostics.get("/readyz")
            if response.get("status") == "ready":
                return response
        except (httpx.HTTPError, RuntimeError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError(f"server did not become ready within {timeout_s}s")


async def fetch_stats(diagnostics: Diagnostics) -> dict:
    stats = await diagnostics.get("/healthz")
    validate_stats(stats)
    return stats


def validate_stats(stats: dict) -> None:
    if not isinstance(stats.get("process_id"), str) or not stats["process_id"]:
        raise RuntimeError("diagnostics missing a non-empty process_id")
    if not isinstance(stats.get("effective_config"), dict) or not stats["effective_config"]:
        raise RuntimeError("diagnostics missing effective_config")
    if (stats.get("engine") == "onnx" and
            (not isinstance(stats.get("model_sha256"), str) or not stats["model_sha256"])):
        raise RuntimeError("ONNX diagnostics missing model_sha256")
    for key in ("total_batches", "total_requests"):
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise RuntimeError(f"diagnostics missing valid {key}")


def check_unique_request_ids(results: list[RequestResult]) -> None:
    ids = [result.request_id for result in results if result.ok]
    if any(request_id is None for request_id in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("successful responses contain missing or duplicate request IDs")


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
    payload,
    count: int,
    concurrency: int,
    *,
    expected_output_elems: int | None = None,
    client_factory=None,
    endpoint: str = "raw",
) -> list[RequestResult]:
    """Fire `count` requests holding `concurrency` of them in flight."""
    semaphore = asyncio.Semaphore(concurrency)
    # Client-side connection pool must not be the bottleneck: if it were
    # smaller than the target concurrency, requests would queue in the client
    # and never reach the scheduler together.
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)

    factory = client_factory or httpx.AsyncClient
    async with factory(timeout=300.0, limits=limits) as client:
        url = f"{base_url}/predict/{'raw' if endpoint == 'raw' else ''}".rstrip("/")

        payloads = payload if isinstance(payload, list) else [payload]

        async def one(index: int) -> RequestResult:
            async with semaphore:
                return await send_payload(
                    client, url, payloads[index % len(payloads)], endpoint,
                    expected_output_elems,
                )

        return await asyncio.gather(*(one(index) for index in range(count)))


async def run_rate_phase(
    base_url: str,
    payload,
    rate_rps: float,
    duration_s: float,
    max_outstanding: int,
    *,
    expected_output_elems: int | None = None,
    client_factory=None,
    endpoint: str = "raw",
) -> tuple[list[ScheduledResult], int, int]:
    """Offer traffic on a monotonic schedule independent of response completion.

    Arrivals that find ``max_outstanding`` requests still running are recorded
    as missed rather than queued in the client. Requests running at the end of
    the window are allowed to drain, but are marked so capacity calculations do
    not count work completed after the cutoff.
    """
    if rate_rps <= 0 or duration_s <= 0 or max_outstanding <= 0:
        raise ValueError("rate, duration, and max_outstanding must be positive")

    limits = httpx.Limits(
        max_connections=max_outstanding + 8,
        max_keepalive_connections=max_outstanding + 8,
    )
    factory = client_factory or httpx.AsyncClient
    client_kwargs = {"timeout": 300.0, "limits": limits}
    interval = 1.0 / rate_rps
    tasks: set[asyncio.Task[ScheduledResult]] = set()
    all_tasks: list[asyncio.Task[ScheduledResult]] = []
    missed = 0
    offered = 0
    url = f"{base_url}/predict/{'raw' if endpoint == 'raw' else ''}".rstrip("/")
    payloads = payload if isinstance(payload, list) else [payload]

    async with factory(**client_kwargs) as client:
        started = time.perf_counter()
        cutoff = started + duration_s

        async def one(scheduled_at: float, index: int) -> ScheduledResult:
            send_started = time.perf_counter()
            result = await send_payload(
                client, url, payloads[index % len(payloads)], endpoint,
                expected_output_elems,
            )
            completed_at = time.perf_counter()
            return ScheduledResult(
                result=result,
                scheduling_delay_s=max(0.0, send_started - scheduled_at),
                scheduled_latency_s=max(0.0, completed_at - scheduled_at),
                completed_within_window=completed_at <= cutoff,
            )

        index = 0
        while True:
            scheduled_at = started + index * interval
            if scheduled_at >= cutoff:
                break
            delay = scheduled_at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            offered += 1
            now = time.perf_counter()
            # Do not burst overdue arrivals to "catch up". A full interval of
            # client scheduling lag means that arrival was not actually offered.
            if now >= cutoff or now - scheduled_at >= interval:
                missed += 1
                index += 1
                continue
            tasks = {task for task in tasks if not task.done()}
            if len(tasks) >= max_outstanding:
                missed += 1
            else:
                task = asyncio.create_task(one(scheduled_at, index))
                tasks.add(task)
                all_tasks.append(task)
            index += 1

        completed = await asyncio.gather(*all_tasks)
    return completed, offered, missed


async def run_duration_phase(
    base_url: str,
    payload,
    duration_s: float,
    concurrency: int,
    *,
    expected_output_elems: int | None = None,
    client_factory=None,
    endpoint: str = "raw",
) -> list[WindowedResult]:
    """Hold fixed concurrency for a time window and mark draining requests."""
    if duration_s <= 0 or concurrency <= 0:
        raise ValueError("duration and concurrency must be positive")
    limits = httpx.Limits(
        max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8
    )
    factory = client_factory or httpx.AsyncClient
    payloads = payload if isinstance(payload, list) else [payload]
    url = f"{base_url}/predict/{'raw' if endpoint == 'raw' else ''}".rstrip("/")
    results: list[WindowedResult] = []
    next_payload = 0

    async with factory(timeout=300.0, limits=limits) as client:
        cutoff = time.perf_counter() + duration_s

        async def worker() -> None:
            nonlocal next_payload
            while time.perf_counter() < cutoff:
                index = next_payload
                next_payload += 1
                result = await send_payload(
                    client, url, payloads[index % len(payloads)], endpoint,
                    expected_output_elems,
                )
                results.append(WindowedResult(result, time.perf_counter() <= cutoff))

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


async def measure(
    base_url: str,
    diagnostics: Diagnostics,
    mode: str,
    concurrency: int,
    num_requests: int,
    warmup: int,
    input_elems: int,
    require_onnx: bool,
    output_elems: int = DEFAULT_OUTPUT_ELEMS,
    client_factory=None,
    endpoint: str = "raw",
) -> RunResult:
    maker = make_json_payload if endpoint == "json" else make_payload
    payload = [maker(input_elems, seed) for seed in range(8)]

    stats_before = await fetch_stats(diagnostics)
    check_engine(stats_before, require_onnx)

    if warmup:
        await run_phase(
            base_url, payload, warmup, concurrency,
            expected_output_elems=output_elems, client_factory=client_factory,
            endpoint=endpoint,
        )

    # Read counters after warmup so the reported batching stats describe the
    # measured phase only.
    baseline = await fetch_stats(diagnostics)

    measured_started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    results = await run_phase(
        base_url, payload, num_requests, concurrency,
        expected_output_elems=output_elems, client_factory=client_factory,
        endpoint=endpoint,
    )
    wall_clock_s = time.perf_counter() - started
    measured_ended_at = datetime.now(timezone.utc)
    check_unique_request_ids(results)

    stats_after = await fetch_stats(diagnostics)
    if baseline.get("process_id") != stats_after.get("process_id"):
        raise RuntimeError("server process changed during the measured phase")
    if baseline.get("effective_config") != stats_after.get("effective_config"):
        raise RuntimeError("effective server configuration changed during the measured phase")
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
    measured_stats["process_id"] = stats_after.get("process_id")
    measured_stats["model_sha256"] = stats_after.get("model_sha256")
    measured_stats["effective_config"] = stats_after.get("effective_config", {})

    ok = [r for r in results if r.ok]
    failures = len(results) - len(ok)

    return RunResult(
        mode=mode,
        concurrency=concurrency,
        num_requests=num_requests,
        failures=failures,
        wall_clock_s=wall_clock_s,
        throughput_rps=len(ok) / wall_clock_s if wall_clock_s > 0 else 0.0,
        latency_ms=percentiles([r.latency_s * 1000.0 for r in ok]),
        server_latency_ms=percentiles(
            [r.server_latency_ms for r in ok if r.server_latency_ms is not None]
        ),
        server_stats=measured_stats,
        offered_requests=num_requests,
        attempted_requests=len(results),
        successful_requests=len(ok),
        completed_within_window=len(results),
        successful_within_window=len(ok),
        outcome_counts=outcome_counts(results),
        http_status_counts=http_status_counts(results),
        measured_started_at=measured_started_at.isoformat(),
        measured_ended_at=measured_ended_at.isoformat(),
    )


async def measure_rate(
    base_url: str,
    diagnostics: Diagnostics,
    mode: str,
    rate_rps: float,
    duration_s: float,
    warmup_s: float,
    max_outstanding: int,
    input_elems: int,
    output_elems: int,
    require_onnx: bool,
    client_factory=None,
    endpoint: str = "raw",
) -> RunResult:
    maker = make_json_payload if endpoint == "json" else make_payload
    payload = [maker(input_elems, seed) for seed in range(8)]
    stats_before = await fetch_stats(diagnostics)
    check_engine(stats_before, require_onnx)

    if warmup_s:
        await run_rate_phase(
            base_url, payload, rate_rps, warmup_s, max_outstanding,
            expected_output_elems=output_elems, client_factory=client_factory,
            endpoint=endpoint,
        )
    baseline = await fetch_stats(diagnostics)

    measured_started_at = datetime.now(timezone.utc)
    scheduled, offered, missed = await run_rate_phase(
        base_url, payload, rate_rps, duration_s, max_outstanding,
        expected_output_elems=output_elems, client_factory=client_factory,
        endpoint=endpoint,
    )
    stats_after = await fetch_stats(diagnostics)
    if baseline.get("process_id") != stats_after.get("process_id"):
        raise RuntimeError("server process changed during the measured phase")
    if baseline.get("effective_config") != stats_after.get("effective_config"):
        raise RuntimeError("effective server configuration changed during the measured phase")

    results = [item.result for item in scheduled]
    check_unique_request_ids(results)
    ok = [item for item in scheduled if item.result.ok]
    within = [item for item in scheduled if item.completed_within_window]
    ok_within = [item for item in within if item.result.ok]
    unfinished = len(scheduled) - len(within)
    measured_stats = {
        key: stats_after.get(key, 0) - baseline.get(key, 0)
        for key in ("total_batches", "total_requests")
    }
    measured_stats.update({
        "max_batch_size_seen": stats_after.get("max_batch_size_seen", 0),
        "cumulative_avg_queue_wait_ms": stats_after.get("avg_queue_wait_ms", 0.0),
        "cumulative_avg_exec_ms": stats_after.get("avg_exec_ms", 0.0),
        "engine": stats_after.get("engine", "unknown"),
        "process_id": stats_after.get("process_id"),
        "model_sha256": stats_after.get("model_sha256"),
        "effective_config": stats_after.get("effective_config", {}),
        "counter_scope": "post_drain_includes_after_cutoff",
    })

    return RunResult(
        mode=mode,
        concurrency=max_outstanding,
        num_requests=offered,
        failures=len(results) - len(ok),
        wall_clock_s=duration_s,
        throughput_rps=len(ok_within) / duration_s,
        latency_ms=percentiles([item.result.latency_s * 1000.0 for item in ok]),
        server_latency_ms=percentiles([
            item.result.server_latency_ms for item in ok
            if item.result.server_latency_ms is not None
        ]),
        server_stats=measured_stats,
        workload="open_loop",
        target_rate_rps=rate_rps,
        max_outstanding=max_outstanding,
        offered_requests=offered,
        attempted_requests=len(results),
        successful_requests=len(ok),
        completed_within_window=len(within),
        successful_within_window=len(ok_within),
        missed_arrivals=missed,
        unfinished_at_cutoff=unfinished,
        late_completions=unfinished,
        successful_after_cutoff=sum(item.result.ok for item in scheduled if not item.completed_within_window),
        failed_after_cutoff=sum(not item.result.ok for item in scheduled if not item.completed_within_window),
        outcome_counts=outcome_counts(results),
        http_status_counts=http_status_counts(results),
        scheduling_delay_ms=percentiles([item.scheduling_delay_s * 1000.0 for item in scheduled]),
        scheduled_latency_ms=percentiles([item.scheduled_latency_s * 1000.0 for item in ok]),
        measured_started_at=measured_started_at.isoformat(),
        measured_ended_at=(measured_started_at + timedelta(seconds=duration_s)).isoformat(),
    )


async def measure_duration(
    base_url: str,
    diagnostics: Diagnostics,
    mode: str,
    concurrency: int,
    duration_s: float,
    warmup_s: float,
    input_elems: int,
    output_elems: int,
    require_onnx: bool,
    client_factory=None,
    endpoint: str = "raw",
) -> RunResult:
    maker = make_json_payload if endpoint == "json" else make_payload
    payload = [maker(input_elems, seed) for seed in range(8)]
    stats_before = await fetch_stats(diagnostics)
    check_engine(stats_before, require_onnx)
    if warmup_s:
        await run_duration_phase(
            base_url, payload, warmup_s, concurrency,
            expected_output_elems=output_elems, client_factory=client_factory,
            endpoint=endpoint,
        )
    baseline = await fetch_stats(diagnostics)
    measured_started_at = datetime.now(timezone.utc)
    windowed = await run_duration_phase(
        base_url, payload, duration_s, concurrency,
        expected_output_elems=output_elems, client_factory=client_factory,
        endpoint=endpoint,
    )
    stats_after = await fetch_stats(diagnostics)
    if baseline.get("process_id") != stats_after.get("process_id"):
        raise RuntimeError("server process changed during the measured phase")
    if baseline.get("effective_config") != stats_after.get("effective_config"):
        raise RuntimeError("effective server configuration changed during the measured phase")
    results = [item.result for item in windowed]
    check_unique_request_ids(results)
    ok = [item for item in windowed if item.result.ok]
    within = [item for item in windowed if item.completed_within_window]
    ok_within = [item for item in within if item.result.ok]
    measured_stats = {
        key: stats_after.get(key, 0) - baseline.get(key, 0)
        for key in ("total_batches", "total_requests")
    }
    measured_stats.update({
        "max_batch_size_seen": stats_after.get("max_batch_size_seen", 0),
        "cumulative_avg_queue_wait_ms": stats_after.get("avg_queue_wait_ms", 0.0),
        "cumulative_avg_exec_ms": stats_after.get("avg_exec_ms", 0.0),
        "engine": stats_after.get("engine", "unknown"),
        "process_id": stats_after.get("process_id"),
        "model_sha256": stats_after.get("model_sha256"),
        "effective_config": stats_after.get("effective_config", {}),
        "counter_scope": "post_drain_includes_after_cutoff",
    })
    unfinished = len(windowed) - len(within)
    return RunResult(
        mode=mode, concurrency=concurrency, num_requests=len(results),
        failures=len(results) - len(ok), wall_clock_s=duration_s,
        throughput_rps=len(ok_within) / duration_s,
        latency_ms=percentiles([item.result.latency_s * 1000 for item in ok]),
        server_latency_ms=percentiles([
            item.result.server_latency_ms for item in ok
            if item.result.server_latency_ms is not None
        ]),
        server_stats=measured_stats, workload="closed_loop_duration",
        max_outstanding=concurrency, offered_requests=len(results),
        attempted_requests=len(results), successful_requests=len(ok),
        completed_within_window=len(within),
        successful_within_window=len(ok_within),
        unfinished_at_cutoff=unfinished, late_completions=unfinished,
        successful_after_cutoff=sum(item.result.ok for item in windowed if not item.completed_within_window),
        failed_after_cutoff=sum(not item.result.ok for item in windowed if not item.completed_within_window),
        outcome_counts=outcome_counts(results),
        http_status_counts=http_status_counts(results),
        measured_started_at=measured_started_at.isoformat(),
        measured_ended_at=(measured_started_at + timedelta(seconds=duration_s)).isoformat(),
    )


class ServerProcess:
    """Runs uvicorn with a mode's batching config, since those are read at startup."""

    def __init__(self, mode: str, port: int, model_path: str, input_elems: int,
                 output_elems: int, executor_workers: int,
                 max_wait_ms: int | None = None, max_batch_size: int | None = None):
        self.mode = mode
        self.port = port
        env = os.environ.copy()
        env.update(MODE_CONFIG[mode])

        # Applied after MODE_CONFIG, which would otherwise silently win: the
        # mode defaults are a starting point, and the batching window is the
        # main knob worth sweeping. Only meaningful for dynamic mode -- forcing
        # a wait or a batch size onto "serial" would stop it being the
        # one-at-a-time baseline the comparison is against.
        if mode == "dynamic":
            if max_wait_ms is not None:
                env["CUDA_DB_MAX_WAIT_MS"] = str(max_wait_ms)
            if max_batch_size is not None:
                env["CUDA_DB_MAX_BATCH_SIZE"] = str(max_batch_size)
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
            {
                **asdict(run), "observed_mean_batch": round(run.observed_mean_batch, 2),
                "offered_rps": run.offered_requests / run.wall_clock_s,
                "attempted_rps": run.attempted_requests / run.wall_clock_s,
                "rejected_rps": run.http_status_counts.get("503", 0) / run.wall_clock_s,
                "rate_scope": "measured window; rejection count includes post-cutoff responses during drain",
            }
            for run in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = RESULTS_DIR / f"{output_stem}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["mode", "workload", "concurrency", "target_rate_rps", "throughput_rps",
             "p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms",
             "scheduled_p95_ms", "scheduling_delay_p95_ms", "observed_mean_batch",
             "offered", "attempted", "successful", "missed_arrivals",
             "unfinished_at_cutoff", "failures"]
        )
        for run in results:
            writer.writerow([
                run.mode, run.workload, run.concurrency, run.target_rate_rps,
                round(run.throughput_rps, 2),
                round(run.latency_ms["p50"], 2), round(run.latency_ms["p90"], 2),
                round(run.latency_ms["p95"], 2), round(run.latency_ms["p99"], 2),
                round(run.latency_ms["mean"], 2),
                round(run.scheduled_latency_ms.get("p95", 0.0), 2),
                round(run.scheduling_delay_ms.get("p95", 0.0), 2),
                round(run.observed_mean_batch, 2), run.offered_requests,
                run.attempted_requests, run.successful_requests, run.missed_arrivals,
                run.unfinished_at_cutoff, run.failures,
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


def git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: str) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_summary(results: list[RunResult]) -> None:
    # srv p50 is the server's own time inside predict(); "gap" is what the
    # client saw minus that, i.e. everything outside the scheduler -- HTTP,
    # uvicorn, the executor queue, and the client itself. A large gap means the
    # bottleneck is not where a scheduler-focused reading would assume.
    print(f"\n{'mode':<9}{'load':>10}{'rps':>10}{'p50 ms':>10}{'p99 ms':>10}"
          f"{'srv p50':>10}{'gap ms':>9}{'batch':>7}{'fail':>6}")
    print("-" * 81)
    for run in results:
        server_p50 = run.server_latency_ms.get("p50", 0.0)
        gap = run.latency_ms["p50"] - server_p50
        load = (f"{run.target_rate_rps:g}/s" if run.workload == "open_loop"
                else f"c={run.concurrency}")
        print(f"{run.mode:<9}{load:>10}{run.throughput_rps:>10.1f}"
              f"{run.latency_ms['p50']:>10.1f}{run.latency_ms['p99']:>10.1f}"
              f"{server_p50:>10.1f}{gap:>9.1f}{run.observed_mean_batch:>7.1f}"
              f"{run.failures:>6}")

    print(f"\n{'mode':<9}{'conc':>6}{'queue wait ms':>16}{'exec ms':>10}"
          "   (server counters, cumulative)")
    print("-" * 62)
    for run in results:
        print(f"{run.mode:<9}{run.concurrency:>6}"
              f"{run.server_stats.get('cumulative_avg_queue_wait_ms', 0.0):>16.1f}"
              f"{run.server_stats.get('cumulative_avg_exec_ms', 0.0):>10.1f}")

    # The headline comparison, printed only where both modes ran the same
    # concurrency so the speedup is apples-to-apples.
    closed = [run for run in results if run.workload == "closed_loop"]
    by_key = {(r.mode, r.concurrency): r for r in closed}
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
    rates = [float(rate) for rate in args.request_rate.split(",") if rate] if args.request_rate else []
    modes = ["dynamic", "serial"] if args.mode == "both" else [args.mode]
    model_path = os.environ.get("CUDA_DB_MODEL_PATH", "")
    require_onnx = not args.allow_stub
    started_at = datetime.now(timezone.utc)

    if args.url and len(modes) != 1:
        raise SystemExit("an external server has one effective configuration; select one --mode")
    if args.url and bool(args.diagnostics_url) == bool(args.diagnostics_command_file):
        raise SystemExit(
            "external benchmarks require exactly one of --diagnostics-url or "
            "--diagnostics-command-file"
        )
    if rates and (args.duration_s <= 0 or args.max_outstanding <= 0):
        raise SystemExit("--duration-s and --max-outstanding must be positive")
    if args.closed_loop_duration_s < 0 or args.closed_loop_warmup_s < 0:
        raise SystemExit("closed-loop durations must be nonnegative")

    results: list[RunResult] = []
    server_snapshots: list[dict] = []

    async def run_for_server(base_url: str, mode: str) -> None:
        inference_factory = configured_client_factory(base_url, args.token_file, args.ca_file)
        if args.diagnostics_command_file:
            diagnostics = Diagnostics(
                command=load_diagnostics_command(args.diagnostics_command_file)
            )
        else:
            diagnostics_url = args.diagnostics_url or base_url
            diagnostics = Diagnostics(
                base_url=diagnostics_url,
                client_factory=configured_client_factory(
                    diagnostics_url, None, args.ca_file, credentials=False
                ),
            )
        await wait_until_ready(diagnostics)
        snapshot = await fetch_stats(diagnostics)
        server_snapshots.append({
            "mode": mode,
            "process_id": snapshot.get("process_id"),
            "engine": snapshot.get("engine", "unknown"),
            "effective_config": snapshot.get("effective_config", {}),
            "model_sha256": snapshot.get("model_sha256"),
        })
        if rates:
            for rate in rates:
                print(f"running {mode} @ offered_rate={rate:g} rps ...", flush=True)
                results.append(await measure_rate(
                    base_url, diagnostics, mode, rate, args.duration_s, args.warmup_s,
                    args.max_outstanding, args.input_elems, args.output_elems,
                    require_onnx, inference_factory, args.endpoint,
                ))
        elif args.closed_loop_duration_s:
            for concurrency in concurrencies:
                print(f"running {mode} @ concurrency={concurrency} ...", flush=True)
                results.append(await measure_duration(
                    base_url, diagnostics, mode, concurrency,
                    args.closed_loop_duration_s, args.closed_loop_warmup_s,
                    args.input_elems, args.output_elems, require_onnx,
                    inference_factory, args.endpoint,
                ))
        else:
            for concurrency in concurrencies:
                print(f"running {mode} @ concurrency={concurrency} ...", flush=True)
                results.append(await measure(
                    base_url, diagnostics, mode, concurrency, args.num_requests,
                    args.warmup, args.input_elems, require_onnx, args.output_elems,
                    inference_factory, args.endpoint,
                ))

    for mode in modes:
        if args.url:
            await run_for_server(args.url.rstrip("/"), mode)
        else:
            with ServerProcess(mode, args.port, model_path, args.input_elems,
                               args.output_elems, args.executor_workers,
                               args.max_wait_ms, args.max_batch_size) as base_url:
                await run_for_server(base_url, mode)

    print_summary(results)

    metadata = {
        "schema_version": 2,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "client_gpu": gpu_name() if not args.url else None,
        "model_path": model_path if not args.url and model_path else None,
        "model_checksum": (args.model_checksum or file_sha256(model_path)
                           or next((item.get("model_sha256") for item in server_snapshots
                                    if item.get("model_sha256")), None)),
        "input_elems": args.input_elems,
        "output_elems": args.output_elems,
        "requested_executor_workers": args.executor_workers,
        "warmup_requests": args.warmup,
        "warmup_s": args.warmup_s,
        "num_requests": args.num_requests,
        "duration_s": args.duration_s,
        "closed_loop_duration_s": args.closed_loop_duration_s,
        "closed_loop_warmup_s": args.closed_loop_warmup_s,
        "max_outstanding": args.max_outstanding,
        "request_rates_rps": rates,
        "endpoint": args.endpoint,
        "requested_local_overrides": ({
            "max_wait_ms": args.max_wait_ms,
            "max_batch_size": args.max_batch_size,
        } if not args.url else None),
        "effective_server_snapshots": server_snapshots,
        "git_revision": git_revision(),
        "git_dirty": git_dirty(),
        "server_git_revision": args.server_git_revision or None,
        "driver_version": args.driver_version or None,
        "image_digest": args.image_digest or None,
        "instance_type": args.instance_type or None,
        "ami_id": args.ami_id or None,
        "region": args.region or None,
        "availability_zone": args.availability_zone or None,
        "client_location": args.client_location or None,
        "client_host": platform.node(),
    }
    write_results(results, args.output, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["dynamic", "serial", "both"], default="both")
    parser.add_argument("--endpoint", choices=["raw", "json"], default="raw",
                        help="raw is the capacity baseline; json quantifies serialization overhead")
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
    parser.add_argument("--request-rate", default="",
                        help="comma-separated scheduled request rates; enables open-loop mode")
    parser.add_argument("--duration-s", type=float, default=120.0,
                        help="measurement window per request rate")
    parser.add_argument("--warmup-s", type=float, default=30.0,
                        help="discarded warmup duration in open-loop mode")
    parser.add_argument("--max-outstanding", type=int, default=128,
                        help="client work bound; arrivals above it are recorded as missed")
    parser.add_argument("--closed-loop-duration-s", type=float, default=0.0,
                        help="time-bound fixed-concurrency measurement; 0 keeps request-count mode")
    parser.add_argument("--closed-loop-warmup-s", type=float, default=30.0,
                        help="discarded warmup duration for time-bound fixed concurrency")
    parser.add_argument("--url", default="",
                        help="benchmark an already-running server instead of spawning one")
    parser.add_argument("--diagnostics-url", default="",
                        help="private origin used only for /readyz and /healthz")
    parser.add_argument("--diagnostics-command-file", default="",
                        help="JSON argv file; the diagnostic endpoint is appended as one argument")
    parser.add_argument("--token-file", default=None,
                        help="protected bearer-token file (or CUDA_DB_BENCHMARK_TOKEN[_FILE])")
    parser.add_argument("--ca-file", default=None,
                        help="CA bundle for verified HTTPS (or CUDA_DB_BENCHMARK_CA_FILE)")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--input-elems", type=int, default=DEFAULT_INPUT_ELEMS)
    parser.add_argument("--output-elems", type=int, default=DEFAULT_OUTPUT_ELEMS)
    parser.add_argument("--executor-workers", type=int, default=16)
    parser.add_argument("--max-wait-ms", type=int, default=None,
                        help="override the dynamic-mode batching window (default 5). "
                             "The main tuning knob: a wider window fills batches closer "
                             "to max_batch_size, trading latency for throughput. Ignored "
                             "for serial mode, which must stay the one-at-a-time baseline")
    parser.add_argument("--max-batch-size", type=int, default=None,
                        help="override the dynamic-mode batch cap (default 8). The ONNX "
                             "engine pads every batch to this size, so raising it also "
                             "raises the cost of a partly-filled batch")
    parser.add_argument("--output", default="latest",
                        help="output file stem under benchmarks/results/")
    parser.add_argument("--image-digest", default="")
    parser.add_argument("--server-git-revision", default="")
    parser.add_argument("--driver-version", default="")
    parser.add_argument("--model-checksum", default="")
    parser.add_argument("--instance-type", default="")
    parser.add_argument("--ami-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--availability-zone", default="")
    parser.add_argument("--client-location", default="")
    parser.add_argument("--allow-stub", action="store_true",
                        help="permit benchmarking the stub engine (harness self-test only; "
                             "the resulting numbers say nothing about inference)")
    args = parser.parse_args()

    durations = (args.duration_s, args.warmup_s, args.closed_loop_duration_s, args.closed_loop_warmup_s)
    if not all(math.isfinite(value) and value >= 0 for value in durations) or args.duration_s == 0:
        parser.error("durations must be finite and nonnegative; --duration-s must be positive")
    try:
        concurrencies = [int(value) for value in args.concurrency.split(",")]
        rates = [float(value) for value in args.request_rate.split(",")] if args.request_rate else []
    except ValueError:
        parser.error("invalid concurrency or request-rate list")
    if not all(value > 0 for value in concurrencies) or not all(math.isfinite(value) and value > 0 for value in rates):
        parser.error("concurrency and request rates must be positive and finite")
    if min(args.num_requests, args.max_outstanding, args.input_elems, args.output_elems, args.executor_workers) < 1 or args.warmup < 0:
        parser.error("request counts, sizes and workers must be positive; warmup may be zero")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
