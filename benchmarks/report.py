"""Combine benchmark JSON results with explicitly supplied prices and evidence.

No prices or validation outcomes are inferred. See docs/aws-benchmarking.md
for the inventory format. Historical results remain readable, but missing
evidence prevents an unconditional deployment recommendation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def resource_summary(path: Path) -> dict:
    """Summarize an explicitly scoped sampling file; never replace null by 0."""
    values = {}
    samples = 0
    unavailable_gpu_samples = 0
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if item.get("type") != "sample":
            continue
        samples += 1
        if item.get("gpus") is None:
            unavailable_gpu_samples += 1
        metrics = {"host." + key: value for key, value in item.get("host", {}).items()}
        for gpu in item.get("gpus") or []:
            identity = gpu.get("uuid") or str(gpu.get("index"))
            metrics.update({f"gpu.{identity}.{key}": value for key, value in gpu.items() if key not in {"index", "uuid"}})
        for key, value in metrics.items():
            values.setdefault(key, [])
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values[key].append(value)
    return {
        "file": str(path), "samples": samples,
        "unavailable_gpu_samples": unavailable_gpu_samples,
        "metrics": {key: {"available_samples": len(series), "maximum_sampled": max(series) if series else None, "mean": statistics.fmean(series) if series else None} for key, series in values.items()},
        "scope": "entire supplied file, not an individual run; periodic maxima can miss peaks",
    }


def build_report(inventory: dict, base: Path) -> dict:
    targets = inventory.get("targets", {})
    for key in ("p95_ms", "p99_ms", "max_error_fraction", "required_rps", "headroom_fraction"):
        if key in targets:
            nonnegative(targets[key], key)
    if targets.get("max_error_fraction", 0) > 1:
        raise ValueError("max_error_fraction must be <= 1")
    rows = []
    for entry in inventory["experiments"]:
        data = json.loads((base / entry["results"]).read_text())
        metadata = data.get("metadata", {})
        price = entry.get("instance_hourly_usd")
        if price is not None:
            nonnegative(price, "instance_hourly_usd")
        for index, run in enumerate(data["runs"]):
            success = run.get("successful_requests", run["num_requests"] - run["failures"])
            within_window = run.get("successful_within_window", success)
            offered = run.get("offered_requests", run["num_requests"])
            rate = nonnegative(run["throughput_rps"], "throughput_rps")
            error = (offered - success) / offered if offered else None
            window_shortfall = (offered - within_window) / offered if offered else None
            latency = run.get("latency_ms", {})
            gate_latency = run.get("scheduled_latency_ms", {}) if run.get("workload") == "open_loop" else latency
            reasons = []
            for gate in ("correctness", "recovery"):
                evidence = entry.get(gate, {})
                if evidence.get("status") != "passed" or not evidence.get("evidence"):
                    reasons.append(f"{gate} evidence missing or not passed")
                elif not (base / evidence["evidence"]).is_file():
                    reasons.append(f"{gate} evidence file missing")
            if not all(key in targets for key in ("required_rps", "max_error_fraction", "headroom_fraction")) or not any(key in targets for key in ("p95_ms", "p99_ms")):
                reasons.append("workload, latency, error and headroom targets not fully specified")
            for key in ("p95", "p99"):
                if key + "_ms" in targets and (gate_latency.get(key) is None or gate_latency[key] > targets[key + "_ms"]):
                    reasons.append(f"{key} latency target not met")
            if error is None or error > targets.get("max_error_fraction", 1):
                reasons.append("offered-request failure target not met")
            if window_shortfall is None or window_shortfall > targets.get("max_error_fraction", 1):
                reasons.append("within-window success shortfall exceeds error allowance (includes boundary completions)")
            required = targets.get("required_rps", 0) * (1 + targets.get("headroom_fraction", 0))
            if rate < required:
                reasons.append("throughput headroom not demonstrated")
            if price is None or not entry.get("price_checked_at") or not entry.get("price_source"):
                reasons.append("dated price/source missing")
            if run.get("unfinished_requests", 0):
                reasons.append("requests remained unfinished after drain")
            if success < 1000 and "p99_ms" in targets:
                reasons.append("p99 has fewer than 1000 successful observations; repeat/extend run")
            rows.append({
                "candidate": entry["candidate"], "results": entry["results"], "run_index": index,
                "mode": run["mode"], "concurrency": run["concurrency"],
                "workload": run.get("workload", "closed_loop"),
                "target_rate_rps": run.get("target_rate_rps"),
                "endpoint": metadata.get("endpoint", "unknown"),
                "image_digest": metadata.get("image_digest"),
                "model_sha256": run.get("server_stats", {}).get("model_sha256", metadata.get("model_checksum")),
                "region": metadata.get("region"),
                "client_location": metadata.get("client_location"),
                "effective_config": run.get("effective_config", run.get("server_stats", {}).get("effective_config", data.get("metadata", {}).get("effective_server_config"))),
                "successful_requests": success, "offered_requests": offered,
                "unfinished_at_cutoff": run.get("unfinished_at_cutoff", 0),
                "late_completions": run.get("late_completions", 0),
                "throughput_rps": rate, "latency_ms": latency,
                "scheduled_latency_ms": run.get("scheduled_latency_ms"),
                "error_fraction_of_offered": error,
                "within_window_shortfall_fraction": window_shortfall,
                "instance_hourly_usd": price,
                "compute_usd_per_1000_successes": price * 1000 / (rate * 3600) if price is not None and rate else None,
                "p99_support": "limited: fewer than 1000 successes" if success < 1000 else "at least 1000 successes; inspect repetition variation",
                "selection_blockers": reasons,
            })
    spend = []
    for item in inventory.get("spend", []):
        amount = nonnegative(item["quantity"], "quantity") * nonnegative(item["unit_price_usd"], "unit_price_usd")
        spend.append({**item, "usd": amount})
    groups = {}
    for row in rows:
        key = json.dumps([row[k] for k in ("candidate", "mode", "concurrency", "workload", "target_rate_rps", "effective_config", "endpoint", "image_digest", "model_sha256", "region", "client_location")], sort_keys=True)
        groups.setdefault(key, []).append(row)
    repetitions = [{
        "candidate": group[0]["candidate"], "mode": group[0]["mode"],
        "concurrency": group[0]["concurrency"], "target_rate_rps": group[0]["target_rate_rps"],
        "effective_config": group[0]["effective_config"], "count": len(group),
        "throughput_min": min(r["throughput_rps"] for r in group),
        "throughput_max": max(r["throughput_rps"] for r in group),
        "throughput_mean": statistics.fmean(r["throughput_rps"] for r in group),
    } for group in groups.values()]
    eligible = sorted([row for row in rows if not row["selection_blockers"]], key=lambda row: row["instance_hourly_usd"])
    return {
        "schema_version": 1, "rows": rows, "repetitions": repetitions,
        "resources": [resource_summary(base / path) for path in inventory.get("resources", [])],
        "targets": targets, "spend": spend,
        "recorded_benchmark_spend_usd": sum(item["usd"] for item in spend),
        "spend_complete": inventory.get("spend_complete", False),
        "provisional_lowest_hourly_cost_match": eligible[0] if eligible else None,
        "limitations": [
            "Selection is provisional: review repeated runs, soak, resource telemetry and lifecycle evidence before deployment.",
            "Evidence paths and pass statuses are operator supplied; this report does not independently certify their contents.",
            "Per-1000 cost includes instance compute only. Spend totals include only supplied ledger entries.",
            "Latency percentiles describe successful responses; inspect failures, missed arrivals and scheduled latency alongside them.",
            "Open-loop selection gates use scheduled-arrival latency and within-window success shortfall; boundary completions can make this conservative.",
        ],
    }


def markdown(report: dict) -> str:
    def fmt(value):
        return "unavailable" if value is None else f"{value:.4g}"
    lines = ["# AWS benchmark comparison", "", "| Candidate | Mode / load | Batch / wait ms | Success req/s | p95 ms | p99 ms | Scheduled p99 ms | Failure fraction of offered | Compute $/1000 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        candidate = str(row['candidate']).replace('|', '\\|').replace('\n', ' ')
        load = f"{row['target_rate_rps']} req/s" if row["workload"] == "open_loop" else f"concurrency {row['concurrency']}"
        config = row.get("effective_config") or {}
        batching = f"{config.get('max_batch_size', 'unknown')} / {config.get('max_wait_ms', 'unknown')}"
        scheduled = (row.get("scheduled_latency_ms") or {}).get("p99")
        lines.append(f"| {candidate} | {row['mode']} / {load} | {batching} | {fmt(row['throughput_rps'])} | {fmt(row['latency_ms'].get('p95'))} | {fmt(row['latency_ms'].get('p99'))} | {fmt(scheduled)} | {fmt(row['error_fraction_of_offered'])} | {fmt(row['compute_usd_per_1000_successes'])} |")
    lines += ["", f"Recorded benchmark spend: ${report['recorded_benchmark_spend_usd']:.4f}; ledger complete: {report['spend_complete']}.", ""]
    choice = report["provisional_lowest_hourly_cost_match"]
    lines.append(f"Provisional lowest hourly cost match: {choice['candidate']} ({choice['mode']}, concurrency {choice['concurrency']}). Review evidence and repetitions before selection." if choice else "No deployment recommendation: targets/evidence are incomplete or no measured run passes all gates. Use the measurements as a capacity envelope.")
    lines += ["", "## Repetition variation", ""]
    for group in report["repetitions"]:
        load = f"{group['target_rate_rps']} req/s" if group["target_rate_rps"] is not None else f"concurrency {group['concurrency']}"
        lines.append(f"- {group['candidate']}, {group['mode']}, {load}: n={group['count']}, throughput min/mean/max {fmt(group['throughput_min'])}/{fmt(group['throughput_mean'])}/{fmt(group['throughput_max'])} req/s.")
    lines += ["", "## Evidence and limitations", ""]
    for row in report["rows"]:
        if row["selection_blockers"]:
            lines.append(f"- {row['candidate']} run {row['run_index']}: " + "; ".join(row["selection_blockers"]) + ".")
    lines += ["", *[f"- {item}" for item in report["limitations"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Markdown output; companion JSON is also written")
    args = parser.parse_args()
    result = build_report(json.loads(args.inventory.read_text()), args.inventory.resolve().parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(result))
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
