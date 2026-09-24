"""Sample host and NVIDIA GPU resources to newline-delimited JSON.

The sampler uses only the standard library and ``nvidia-smi``. Missing or
unsupported metrics are written as null with an explanatory availability/error
record; they are never represented as zero.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any, TextIO

GPU_FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
)
GPU_KEYS = (
    "index",
    "uuid",
    "utilization_gpu_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "temperature_c",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def optional_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported", "unknown"}:
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def parse_nvidia_csv(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(GPU_KEYS):
            raise ValueError(f"expected {len(GPU_KEYS)} NVIDIA fields, got {len(values)}")
        gpu: dict[str, Any] = {"index": optional_float(values[0]), "uuid": values[1] or None}
        gpu["index"] = int(gpu["index"]) if gpu["index"] is not None else None
        for key, value in zip(GPU_KEYS[2:], values[2:]):
            gpu[key] = optional_float(value)
        gpus.append(gpu)
    return gpus


def read_proc_cpu(path: Path = Path("/proc/stat")) -> tuple[int, int] | None:
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if not first or first[0] != "cpu":
        return None
    try:
        ticks = [int(value) for value in first[1:]]
    except ValueError:
        return None
    if len(ticks) < 4:
        return None
    idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
    # Linux guest/guest_nice are already included in user/nice. Restrict the
    # total to user..steal so virtualized guest time is not counted twice.
    return sum(ticks[:8]), idle


def cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0 or idle_delta < 0:
        return None
    return max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))


def read_proc_memory(path: Path = Path("/proc/meminfo")) -> dict[str, float | None]:
    try:
        values = {
            fields[0].rstrip(":"): int(fields[1])
            for line in path.read_text(encoding="utf-8").splitlines()
            if len(fields := line.split()) >= 2 and fields[1].isdigit()
        }
    except OSError:
        return {"memory_used_mib": None, "memory_total_mib": None}
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    return {
        "memory_used_mib": (total - available) / 1024 if total is not None and available is not None else None,
        "memory_total_mib": total / 1024 if total is not None else None,
    }


class ResourceSampler:
    def __init__(self, nvidia_smi: str | None = None) -> None:
        self.nvidia_smi = nvidia_smi if nvidia_smi is not None else shutil.which("nvidia-smi")
        self.previous_cpu = read_proc_cpu()

    def gpu_sample(self) -> tuple[list[dict[str, Any]] | None, str | None]:
        if self.nvidia_smi is None:
            return None, "nvidia-smi not found"
        command = [
            self.nvidia_smi,
            f"--query-gpu={','.join(GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"nvidia-smi failed: {type(exc).__name__}"
        if completed.returncode != 0:
            return None, f"nvidia-smi exited {completed.returncode}"
        try:
            return parse_nvidia_csv(completed.stdout), None
        except ValueError as exc:
            return None, str(exc)

    def sample(self) -> dict[str, Any]:
        current_cpu = read_proc_cpu()
        host = {"cpu_percent": cpu_percent(self.previous_cpu, current_cpu), **read_proc_memory()}
        self.previous_cpu = current_cpu
        try:
            host["load_average"] = list(os.getloadavg())
        except OSError:
            host["load_average"] = None
        gpus, error = self.gpu_sample()
        return {
            "type": "sample",
            "timestamp": utc_now(),
            "monotonic_s": time.monotonic(),
            "host": host,
            "gpus": gpus,
            "gpu_error": error,
        }


def run(interval_s: float, duration_s: float | None, output: TextIO) -> None:
    sampler = ResourceSampler()
    stopped = False

    def stop(_signum, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    header = {
        "type": "metadata",
        "schema_version": 1,
        "started_at": utc_now(),
        "interval_s": interval_s,
        "nvidia_smi_available": sampler.nvidia_smi is not None,
        "host_proc_metrics_available": sampler.previous_cpu is not None,
    }
    output.write(json.dumps(header, sort_keys=True) + "\n")
    output.flush()
    next_sample = started
    while not stopped and (duration_s is None or time.monotonic() - started < duration_s):
        if duration_s is not None and next_sample - started >= duration_s:
            break
        delay = next_sample - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        output.write(json.dumps(sampler.sample(), sort_keys=True) + "\n")
        output.flush()
        next_sample += interval_s
        now = time.monotonic()
        if next_sample <= now:
            next_sample = now + interval_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    parser.add_argument("--duration", type=float, help="stop after this many seconds; default runs until signaled")
    parser.add_argument("--output", type=Path, help="JSONL destination; default stdout")
    args = parser.parse_args()
    if args.interval <= 0 or args.duration is not None and args.duration <= 0:
        parser.error("interval and duration must be positive")
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            run(args.interval, args.duration, stream)
    else:
        import sys

        run(args.interval, args.duration, sys.stdout)


if __name__ == "__main__":
    main()
