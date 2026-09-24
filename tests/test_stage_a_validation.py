"""Unit checks for Stage A smoke, lifecycle, and resource tooling."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import ssl
import subprocess
import sys
import threading

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "docker"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


smoke = load_module("stage_a_smoke", ROOT / "docker" / "smoke_test.py")
lifecycle = load_module("stage_a_lifecycle", ROOT / "docker" / "lifecycle_test.py")
resources = load_module("stage_a_resources", ROOT / "benchmarks" / "resource_sampler.py")


def test_token_file_must_be_private(tmp_path, monkeypatch):
    monkeypatch.delenv(smoke.TOKEN_ENV, raising=False)
    monkeypatch.delenv(smoke.TOKEN_FILE_ENV, raising=False)
    token_file = tmp_path / "token"
    token_file.write_text("pilot.secret\n", encoding="utf-8")
    token_file.chmod(0o600)
    assert smoke.load_token(token_file) == "pilot.secret"
    token_file.chmod(0o644)
    with pytest.raises(ValueError, match="group or other"):
        smoke.load_token(token_file)


def test_token_sources_are_mutually_exclusive(tmp_path, monkeypatch):
    token_file = tmp_path / "token"
    token_file.write_text("file.secret", encoding="utf-8")
    token_file.chmod(0o600)
    monkeypatch.setenv(smoke.TOKEN_ENV, "env.secret")
    with pytest.raises(ValueError, match="only one"):
        smoke.load_token(token_file)


def test_tls_context_verifies_certificates_and_hostname():
    context = smoke.tls_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_bearer_token_is_never_sent_over_http():
    with pytest.raises(ValueError, match="non-HTTPS"):
        smoke.request_json(
            "http://127.0.0.1:8000/predict/raw",
            context=smoke.tls_context(),
            token="pilot.secret",
            data=b"payload",
        )


def test_diagnostics_command_file_is_json_argv(tmp_path):
    path = tmp_path / "diagnostics.json"
    path.write_text(json.dumps(["ssh", "gpu-host", "python3", "private_diagnostics.py"]), encoding="utf-8")
    assert smoke.load_diagnostics_command(path)[0] == "ssh"
    path.write_text(json.dumps("ssh gpu-host"), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        smoke.load_diagnostics_command(path)


def model_fixture(tmp_path: Path) -> tuple[Path, np.ndarray]:
    model = tmp_path / "resnet50.onnx"
    model.write_bytes(b"model")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    inputs = np.arange(8 * 3 * 224 * 224, dtype=np.float32).reshape(8, 3, 224, 224)
    outputs = np.zeros((8, 1000), dtype=np.float32)
    for index in range(8):
        outputs[index, index] = 1.0
    reference = tmp_path / "resnet50.reference.npz"
    np.savez(reference, input=inputs, output=outputs)
    manifest = {
        "schema_version": 1,
        "model_version": "test-v1",
        "file": "resnet50.onnx",
        "sha256": model_sha,
        "input": {"name": "input", "dtype": "float32", "shape": ["batch", 3, 224, 224]},
        "output": {"name": "output", "dtype": "float32", "shape": ["batch", 1000]},
        "reference": {
            "file": reference.name,
            "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "resnet50.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, outputs


def test_smoke_requires_serving_model_identity_and_stable_process(tmp_path, monkeypatch):
    models, outputs = model_fixture(tmp_path)
    health = {
        "engine": "onnx",
        "model_sha256": hashlib.sha256((models / "resnet50.onnx").read_bytes()).hexdigest(),
        "process_id": "process-one",
    }
    calls = 0
    health_calls = 0
    lock = threading.Lock()

    def diagnostics(endpoint):
        nonlocal health_calls
        if endpoint == "/readyz":
            return {"status": "ready"}
        health_calls += 1
        value = dict(health)
        if health_calls == 4:
            value["process_id"] = "process-two"
        return value

    def fake_request(_url, **kwargs):
        nonlocal calls
        first = np.frombuffer(kwargs["data"], dtype="<f4", count=1)[0]
        index = int(first) // (3 * 224 * 224)
        with lock:
            calls += 1
            request_id = calls
        return {"request_id": request_id, "output": outputs[index].tolist(), "latency_ms": 1.0}

    monkeypatch.setattr(smoke, "request_json", fake_request)
    result = smoke.run_smoke(models, "https://example.test", diagnostics, token="secret", context=smoke.tls_context())
    assert result["request_count"] == 9
    with pytest.raises(ValueError, match="process changed"):
        smoke.run_smoke(models, "https://example.test", diagnostics, token="secret", context=smoke.tls_context())


def test_nvidia_csv_preserves_unavailable_as_null():
    parsed = resources.parse_nvidia_csv("0, GPU-abc, 91, 2048, 3072, [N/A], 62\n")
    assert parsed == [{
        "index": 0,
        "uuid": "GPU-abc",
        "utilization_gpu_percent": 91.0,
        "memory_used_mib": 2048.0,
        "memory_total_mib": 3072.0,
        "power_draw_w": None,
        "temperature_c": 62.0,
    }]
    assert resources.optional_float("nan") is None


def test_cpu_delta_and_lifecycle_outcomes():
    assert resources.cpu_percent((100, 50), (200, 75)) == 75.0
    counts = lifecycle.outcome_counts([
        {"outcome": "success"},
        {"outcome": "success"},
        {"outcome": "connection_error"},
        {"outcome": "timeout"},
    ])
    assert counts == {"success": 2, "connection_error": 1, "timeout": 1}


def passing_stop_record():
    return {
        "stop_command_exit_code": 0,
        "stop_command_error": None,
        "overlap_at_stop": True,
        "container_state": {
            "available": True,
            "exit_code": 0,
            "oom_killed": False,
            "possible_sigkill_or_oom": False,
        },
        "requests": [
            {"outcome": "success"},
            {"outcome": "http_error", "status": 503},
        ],
    }


def test_lifecycle_gate_accepts_only_clean_stop_and_accounted_rejection():
    assert lifecycle.stop_gate_failures(passing_stop_record()) == []
    cases = (
        ("overlap_at_stop", False, "no_admitted_unfinished_overlap_at_stop"),
        ("stop_command_exit_code", 1, "stop_command_failed"),
    )
    for field, value, expected in cases:
        record = passing_stop_record()
        record[field] = value
        assert expected in lifecycle.stop_gate_failures(record)
    record = passing_stop_record()
    record["container_state"]["oom_killed"] = True
    assert "container_oom_killed" in lifecycle.stop_gate_failures(record)
    record = passing_stop_record()
    record["requests"] = [{"outcome": "connection_error"}]
    assert "unaccounted_request_failure" in lifecycle.stop_gate_failures(record)


def test_recovery_requires_a_nonempty_new_process_identity():
    assert lifecycle.is_new_process_id("before", "after")
    assert not lifecycle.is_new_process_id("same", "same")
    assert not lifecycle.is_new_process_id(None, "after")
    assert not lifecycle.is_new_process_id("before", "")


def test_lifecycle_output_collision_prevents_execution(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    called = False

    def should_not_execute(_args, _evidence):
        nonlocal called
        called = True

    monkeypatch.setattr(lifecycle, "execute", should_not_execute)
    with pytest.raises(FileExistsError):
        lifecycle.run_and_write(Namespace(output=output))
    assert not called
    assert output.read_text(encoding="utf-8") == "preserve"


def test_lifecycle_failure_evidence_is_safe_and_persisted(tmp_path, monkeypatch):
    output = tmp_path / "evidence.json"

    def fail(_args, evidence):
        evidence["active_stage"] = "inflight_stop"
        raise lifecycle.subprocess.TimeoutExpired(["tool", "super-secret"], 30)

    monkeypatch.setattr(lifecycle, "execute", fail)
    assert not lifecycle.run_and_write(Namespace(output=output))
    raw = output.read_text(encoding="utf-8")
    evidence = json.loads(raw)
    assert evidence["result"] == "failed"
    assert evidence["active_stage"] == "inflight_stop"
    assert evidence["failure_classification"] == "operation_timeout"
    assert "super-secret" not in raw


@pytest.mark.parametrize(
    "body",
    [
        {"request_id": True, "output": [0.0] * 1000, "latency_ms": 1.0},
        {"request_id": 1, "output": [float("nan")] * 1000, "latency_ms": 1.0},
        {"request_id": 1, "output": [0.0] * 1000, "latency_ms": float("inf")},
    ],
)
def test_lifecycle_rejects_malformed_success_bodies(body, monkeypatch):
    monkeypatch.setattr(lifecycle, "request_json", lambda *args, **kwargs: body)
    outcome = lifecycle.request_outcome(
        "https://example.test", b"payload", "token", smoke.tls_context()
    )
    assert outcome["outcome"] == "malformed_response"


def test_stop_timeout_records_partial_state_and_attempts_recovery(tmp_path, monkeypatch):
    release_requests = threading.Event()

    class FakeCompose:
        def __init__(self, *_args):
            pass

        def run(self, *args, **_kwargs):
            if args[0] == "stop":
                release_requests.set()
                raise subprocess.TimeoutExpired(["docker", "secret-value"], 1)
            return subprocess.CompletedProcess(args, 0, "", "")

        def container_id(self):
            return "container-id"

        @staticmethod
        def inspect(_container_id):
            return {
                "available": True,
                "exit_code": 0,
                "oom_killed": False,
                "possible_sigkill_or_oom": False,
            }

    health_calls = 0

    def diagnostics(_command, endpoint):
        nonlocal health_calls
        if endpoint == "/readyz":
            return {"status": "ready"}
        health_calls += 1
        if health_calls == 1:
            return {"process_id": "p1"}
        if health_calls == 2:
            return {"process_id": "p2"}
        if health_calls == 3:
            return {"process_id": "p2", "admission_inflight": 1}
        return {"process_id": "p3"}

    def request(*_args):
        release_requests.wait(2)
        return {"outcome": "success", "latency_s": 0.1}

    monkeypatch.setattr(lifecycle, "Compose", FakeCompose)
    monkeypatch.setattr(lifecycle, "tls_context", lambda _path: object())
    monkeypatch.setattr(lifecycle, "load_token", lambda _path: "token")
    monkeypatch.setattr(lifecycle, "load_diagnostics_command", lambda _path: ["probe"])
    monkeypatch.setattr(lifecycle, "command_diagnostics", diagnostics)
    monkeypatch.setattr(
        lifecycle, "validate_reference",
        lambda _path: ({}, np.zeros((8, 3, 224, 224), dtype=np.float32), np.zeros((8, 1000))),
    )
    monkeypatch.setattr(lifecycle, "run_smoke", lambda *args, **kwargs: {"status": "pass"})
    monkeypatch.setattr(lifecycle, "request_outcome", request)
    args = Namespace(
        compose_file=tmp_path / "compose.yml",
        service="inference",
        ca_file=None,
        token_file=None,
        diagnostics_command_file=tmp_path / "command.json",
        models=tmp_path,
        readiness_timeout=1,
        inflight_requests=2,
        stop_delay=0,
        overlap_timeout=1,
        stop_timeout=1,
        url="https://example.test",
    )
    evidence = {}
    with pytest.raises(RuntimeError, match="stop gate failed"):
        lifecycle.execute(args, evidence)
    assert evidence["inflight_stop"]["stop_command_error"] == {
        "failure_type": "TimeoutExpired",
        "failure_classification": "operation_timeout",
    }
    assert evidence["inflight_stop"]["request_outcomes"] == {"success": 2}
    assert evidence["recovery"]["attempted"] is True
    assert evidence["recovery"]["result"] == "pass"
    assert evidence["recovery"]["process_changed"] is True
    assert "secret-value" not in json.dumps(evidence)
