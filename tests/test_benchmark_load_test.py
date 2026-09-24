import importlib.util
from pathlib import Path
import sys

import pytest
import httpx


BENCHMARKS = Path(__file__).parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location("benchmark_load_test", BENCHMARKS / "load_test.py")
load_test = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = load_test
SPEC.loader.exec_module(load_test)


def test_percentiles_use_nearest_rank_and_include_p95():
    values = list(range(1, 21))

    result = load_test.percentiles(values)

    assert result["p50"] == 10
    assert result["p90"] == 18
    assert result["p95"] == 19
    assert result["p99"] == 20


def test_outcomes_keep_http_statuses_and_failure_classes_separate():
    request = load_test.RequestResult
    results = [
        request(0.1, 200),
        request(0.1, 503, failure_kind="http_status", failure_detail="http_503"),
        request(0.1, 0, failure_kind="timeout", failure_detail="request_timed_out"),
        request(0.1, 200, failure_kind="malformed_response", failure_detail="invalid_json"),
    ]

    assert load_test.outcome_counts(results) == {
        "success": 1,
        "http_503": 1,
        "timeout": 1,
        "malformed_response": 1,
    }
    assert load_test.http_status_counts(results) == {"200": 2, "503": 1}


def test_diagnostics_schema_and_request_ids_are_required():
    valid = {
        "engine": "onnx", "process_id": "abc", "model_sha256": "deadbeef",
        "effective_config": {"max_batch_size": 8},
        "total_batches": 1, "total_requests": 2,
    }
    load_test.validate_stats(valid)
    with pytest.raises(RuntimeError, match="process_id"):
        load_test.validate_stats({**valid, "process_id": None})
    with pytest.raises(RuntimeError, match="model_sha256"):
        load_test.validate_stats({**valid, "model_sha256": None})
    with pytest.raises(RuntimeError, match="duplicate"):
        load_test.check_unique_request_ids([
            load_test.RequestResult(0.1, 200, request_id=1),
            load_test.RequestResult(0.1, 200, request_id=1),
        ])


def test_open_loop_bounds_work_and_records_missed_arrivals(monkeypatch):
    active = 0
    peak = 0

    async def fake_send_request(client, url, payload, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await load_test.asyncio.sleep(0.03)
        active -= 1
        return load_test.RequestResult(0.03, 200, server_latency_ms=20.0)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

    monkeypatch.setattr(load_test, "send_request", fake_send_request)

    completed, offered, missed = load_test.asyncio.run(
        load_test.run_rate_phase(
            "http://127.0.0.1:8000",
            b"input",
            rate_rps=100.0,
            duration_s=0.08,
            max_outstanding=1,
            client_factory=lambda **_kwargs: FakeClient(),
        )
    )

    assert offered == 8
    assert missed > 0
    assert len(completed) + missed == offered
    assert peak == 1
    assert all(item.scheduling_delay_s >= 0 for item in completed)


def test_diagnostics_command_appends_endpoint_and_parses_json(tmp_path):
    script = tmp_path / "diagnostic.py"
    script.write_text(
        "import json, sys\nprint(json.dumps({'endpoint': sys.argv[1]}))\n",
        encoding="utf-8",
    )
    diagnostics = load_test.Diagnostics(command=[sys.executable, str(script)])

    assert load_test.asyncio.run(diagnostics.get("/readyz")) == {"endpoint": "/readyz"}


def test_json_sender_uses_predict_shape_and_validates_response():
    async def scenario():
        async def handler(request):
            assert request.url.path == "/predict"
            assert request.read() == b'{"input":[1.0,2.0]}'
            return httpx.Response(
                200, json={"request_id": 4, "latency_ms": 1.0, "output": [3.0]}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await load_test.send_payload(
                client, "https://example.test/predict", [1.0, 2.0], "json", 1
            )

    assert load_test.asyncio.run(scenario()).ok


def test_time_bound_closed_loop_holds_requested_concurrency(monkeypatch):
    active = 0
    peak = 0

    async def fake_send(client, url, payload, endpoint, output_elems):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await load_test.asyncio.sleep(0.005)
        active -= 1
        return load_test.RequestResult(0.005, 200, server_latency_ms=4.0)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

    monkeypatch.setattr(load_test, "send_payload", fake_send)
    results = load_test.asyncio.run(load_test.run_duration_phase(
        "http://127.0.0.1:8000", [b"a", b"b"], 0.02, 2,
        client_factory=lambda **_kwargs: FakeClient(),
    ))

    assert results
    assert peak == 2
    assert sum(not item.completed_within_window for item in results) <= 2
