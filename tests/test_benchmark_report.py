import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("benchmark_report", Path(__file__).resolve().parents[1] / "benchmarks/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def inventory(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"runs": [{
        "mode": "serial", "concurrency": 1, "num_requests": 3600,
        "failures": 0, "offered_requests": 3600, "successful_requests": 3600,
        "throughput_rps": 10, "latency_ms": {"p95": 20, "p99": 25},
    }]}))
    return {"experiments": [{"candidate": "test", "results": "results.json", "instance_hourly_usd": 0.36}]}


def test_missing_evidence_never_selects_and_cost_uses_success_rate(tmp_path):
    result = report.build_report(inventory(tmp_path), tmp_path)
    assert result["provisional_lowest_hourly_cost_match"] is None
    assert result["rows"][0]["compute_usd_per_1000_successes"] == pytest.approx(0.01)
    assert result["spend_complete"] is False


def test_missed_arrivals_count_against_error_target(tmp_path):
    config = inventory(tmp_path)
    config["targets"] = {"required_rps": 5, "headroom_fraction": 0.2, "p99_ms": 30, "max_error_fraction": 0.01}
    entry = config["experiments"][0]
    (tmp_path / "evidence.json").write_text("{}")
    for gate in ("correctness", "recovery"):
        entry[gate] = {"status": "passed", "evidence": "evidence.json"}
    entry.update(price_checked_at="2026-09-23", price_source="test fixture")
    assert report.build_report(config, tmp_path)["provisional_lowest_hourly_cost_match"]
    path = tmp_path / "results.json"
    data = json.loads(path.read_text())
    data["runs"][0]["offered_requests"] = 4000
    path.write_text(json.dumps(data))
    result = report.build_report(config, tmp_path)
    assert result["provisional_lowest_hourly_cost_match"] is None
    assert result["rows"][0]["error_fraction_of_offered"] == pytest.approx(0.1)


def test_spend_ledger_and_invalid_price(tmp_path):
    config = inventory(tmp_path)
    config["spend"] = [{"category": "gpu", "quantity": 2, "unit": "hours", "unit_price_usd": 0.36}, {"category": "storage", "quantity": 1, "unit": "GB-month", "unit_price_usd": 0.1}]
    assert report.build_report(config, tmp_path)["recorded_benchmark_spend_usd"] == pytest.approx(0.82)
    config["experiments"][0]["instance_hourly_usd"] = float("nan")
    with pytest.raises(ValueError):
        report.build_report(config, tmp_path)


def test_open_loop_selection_uses_scheduled_latency_and_window_shortfall(tmp_path):
    config = inventory(tmp_path)
    config["targets"] = {"p99_ms": 30, "max_error_fraction": 0.01}
    path = tmp_path / "results.json"
    data = json.loads(path.read_text())
    data["runs"][0].update(workload="open_loop", successful_within_window=3000,
                           scheduled_latency_ms={"p99": 200}, unfinished_at_cutoff=600)
    path.write_text(json.dumps(data))
    row = report.build_report(config, tmp_path)["rows"][0]
    assert row["error_fraction_of_offered"] == 0
    assert row["within_window_shortfall_fraction"] == pytest.approx(1 / 6)
    assert "p99 latency target not met" in row["selection_blockers"]


def test_resource_summary_preserves_unavailable_values(tmp_path):
    path = tmp_path / "resources.jsonl"
    path.write_text(json.dumps({"type": "sample", "host": {"cpu_percent": None}, "gpus": None}) + "\n")
    summary = report.resource_summary(path)
    assert summary["unavailable_gpu_samples"] == 1
    assert summary["metrics"]["host.cpu_percent"]["maximum_sampled"] is None
