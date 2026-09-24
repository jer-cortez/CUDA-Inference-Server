"""Record container restart and in-flight stop/recovery evidence as JSON.

This tool controls one local Docker Compose inference service. EC2 stop/start is
an operator runbook step because it changes cloud state and addressing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import ssl
import subprocess
import time
from typing import Any, Callable
import urllib.error

from smoke_test import (
    CA_FILE_ENV,
    command_diagnostics,
    load_diagnostics_command,
    load_token,
    request_json,
    run_smoke,
    tls_context,
    validate_reference,
)


class Compose:
    def __init__(self, compose_file: Path, service: str) -> None:
        self.prefix = ["docker", "compose", "-f", str(compose_file.resolve())]
        self.service = service

    def run(self, *args: str, timeout: float = 180) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.prefix, *args], check=False, capture_output=True, text=True, timeout=timeout
        )

    def require(self, *args: str, timeout: float = 180) -> subprocess.CompletedProcess[str]:
        completed = self.run(*args, timeout=timeout)
        if completed.returncode:
            raise RuntimeError(f"docker compose {' '.join(args)} exited {completed.returncode}")
        return completed

    def container_id(self) -> str:
        value = self.require("ps", "-q", self.service).stdout.strip()
        if not value:
            raise RuntimeError(f"compose service {self.service!r} has no container")
        return value

    @staticmethod
    def inspect(container_id: str) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                ["docker", "inspect", "--format", "{{json .State}}", container_id],
                check=False, capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            return {"available": False, "error": "docker inspect timed out"}
        except OSError:
            return {"available": False, "error": "docker inspect unavailable"}
        if completed.returncode:
            return {"available": False, "error": f"docker inspect exited {completed.returncode}"}
        try:
            state = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {"available": False, "error": "docker inspect returned invalid JSON"}
        return {
            "available": True,
            "status": state.get("Status"),
            "exit_code": state.get("ExitCode"),
            "oom_killed": state.get("OOMKilled"),
            "runtime_error_present": bool(state.get("Error")),
            "started_at": state.get("StartedAt"),
            "finished_at": state.get("FinishedAt"),
            # Exit 137 is consistent with SIGKILL, but Docker also uses it for
            # OOM termination. Preserve both raw facts without overclaiming.
            "possible_sigkill_or_oom": state.get("ExitCode") == 137,
        }


def wait_ready(diagnostics: Callable[[str], dict[str, Any]], timeout_s: float) -> tuple[float, dict[str, Any]]:
    started = time.monotonic()
    deadline = started + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            body = diagnostics("/readyz")
            if body.get("status") == "ready":
                return time.monotonic() - started, body
            last_error = f"unexpected response {body!r}"
        except Exception as exc:  # readiness errors are expected during startup
            last_error = type(exc).__name__
        time.sleep(0.5)
    raise RuntimeError(f"readiness timed out after {timeout_s}s (last error: {last_error})")


def request_outcome(url: str, payload: bytes, token: str | None, context: ssl.SSLContext) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = request_json(url.rstrip("/") + "/predict/raw", context=context, token=token, data=payload, timeout=90)
        output = result.get("output")
        latency = result.get("latency_ms")
        valid = (
            type(result.get("request_id")) is int
            and isinstance(output, list)
            and len(output) == 1000
            and all(type(value) in (int, float) and math.isfinite(value) for value in output)
            and type(latency) in (int, float)
            and math.isfinite(latency)
            and latency >= 0
        )
        return {"outcome": "success" if valid else "malformed_response", "latency_s": time.monotonic() - started}
    except urllib.error.HTTPError as exc:
        return {"outcome": "http_error", "status": exc.code, "latency_s": time.monotonic() - started}
    except urllib.error.URLError as exc:
        return {"outcome": "connection_error", "error_type": type(exc.reason).__name__, "latency_s": time.monotonic() - started}
    except TimeoutError:
        return {"outcome": "timeout", "latency_s": time.monotonic() - started}
    except Exception as exc:
        return {"outcome": "client_error", "error_type": type(exc).__name__, "latency_s": time.monotonic() - started}


def outcome_counts(outcomes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        name = str(outcome["outcome"])
        counts[name] = counts.get(name, 0) + 1
    return counts


def is_new_process_id(previous: Any, current: Any) -> bool:
    return (
        isinstance(previous, str)
        and bool(previous)
        and isinstance(current, str)
        and bool(current)
        and previous != current
    )


def safe_failure(exc: BaseException) -> dict[str, str]:
    """Classify a failure without serializing commands, URLs, or credentials."""
    if isinstance(exc, subprocess.TimeoutExpired):
        classification = "operation_timeout"
    elif isinstance(exc, FileNotFoundError):
        classification = "required_program_unavailable"
    elif isinstance(exc, PermissionError):
        classification = "permission_denied"
    elif isinstance(exc, ssl.SSLError):
        classification = "tls_failure"
    else:
        classification = "lifecycle_operation_failed"
    return {"failure_type": type(exc).__name__, "failure_classification": classification}


def stop_gate_failures(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if record.get("stop_command_exit_code") != 0:
        failures.append("stop_command_failed")
    if record.get("stop_command_error") is not None:
        failures.append("stop_command_error")
    if not record.get("overlap_at_stop"):
        failures.append("no_admitted_unfinished_overlap_at_stop")
    state = record.get("container_state", {})
    if not state.get("available"):
        failures.append("container_state_unavailable")
    else:
        if state.get("exit_code") != 0:
            failures.append("nonzero_container_exit")
        if state.get("oom_killed"):
            failures.append("container_oom_killed")
        if state.get("runtime_error_present"):
            failures.append("container_runtime_error")
        if state.get("possible_sigkill_or_oom"):
            failures.append("possible_forced_termination")
    for outcome in record.get("requests", []):
        # A 503 is an explicit, accounted admission/draining rejection. Other
        # HTTP failures and all transport/client failures invalidate recovery.
        if outcome.get("outcome") == "success":
            continue
        if outcome.get("outcome") == "http_error" and outcome.get("status") == 503:
            continue
        failures.append("unaccounted_request_failure")
        break
    return failures


def execute(args: argparse.Namespace, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    compose = Compose(args.compose_file, args.service)
    context = tls_context(args.ca_file)
    token = load_token(args.token_file)
    command = load_diagnostics_command(args.diagnostics_command_file)
    diagnostics = lambda endpoint: command_diagnostics(command, endpoint)
    _, inputs, _ = validate_reference(args.models)
    initial: dict[str, Any] = {
        "schema_version": 1,
        "started_at_unix_s": time.time(),
        "compose_file": str(args.compose_file.resolve()),
        "service": args.service,
        "idle_restart": {},
        "inflight_stop": {},
        "recovery": {},
    }
    if evidence is None:
        evidence = initial
    else:
        evidence.update(initial)

    evidence["active_stage"] = "idle_restart"
    before = diagnostics("/healthz")
    old_process_id = before.get("process_id")
    restart_started = time.monotonic()
    restart = compose.run("restart", args.service, timeout=args.readiness_timeout + 60)
    restart_command_duration = time.monotonic() - restart_started
    if restart.returncode:
        raise RuntimeError(f"idle restart exited {restart.returncode}")
    ready_s, _ = wait_ready(diagnostics, args.readiness_timeout)
    after = diagnostics("/healthz")
    process_changed = is_new_process_id(old_process_id, after.get("process_id"))
    evidence["idle_restart"] = {
        "restart_command_duration_s": restart_command_duration,
        "restart_to_ready_s": time.monotonic() - restart_started,
        "readiness_wait_s": ready_s,
        "old_process_id": old_process_id,
        "new_process_id": after.get("process_id"),
        "process_changed": process_changed,
    }
    if not process_changed:
        raise RuntimeError("idle restart did not produce a new server process_id")
    evidence["idle_restart"]["reference_validation"] = run_smoke(
        args.models, args.url, diagnostics, token=token, context=context
    )

    evidence["active_stage"] = "inflight_stop"
    container_id = compose.container_id()
    payloads = [inputs[index % len(inputs)].tobytes() for index in range(args.inflight_requests)]
    test_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.inflight_requests) as pool:
        futures = [pool.submit(request_outcome, args.url, payload, token, context) for payload in payloads]
        if args.stop_delay:
            time.sleep(args.stop_delay)
        pre_stop_inflight = None
        observation_error = None
        observation_deadline = time.monotonic() + args.overlap_timeout
        while time.monotonic() < observation_deadline and not all(future.done() for future in futures):
            try:
                pre_stop_inflight = diagnostics("/healthz").get("admission_inflight")
                if isinstance(pre_stop_inflight, int) and pre_stop_inflight > 0:
                    break
            except Exception as exc:
                observation_error = safe_failure(exc)
            time.sleep(0.01)
        unfinished_at_stop = sum(not future.done() for future in futures)
        overlap_at_stop = (
            isinstance(pre_stop_inflight, int)
            and pre_stop_inflight > 0
            and unfinished_at_stop > 0
        )
        stop_started = time.monotonic()
        stop_exit_code = None
        stop_error = None
        try:
            stop = compose.run(
                "stop", "-t", str(args.stop_timeout), args.service,
                timeout=args.stop_timeout + 30,
            )
            stop_exit_code = stop.returncode
        except Exception as exc:
            stop_error = safe_failure(exc)
        stop_duration = time.monotonic() - stop_started
        outcomes = [future.result() for future in futures]
    state = compose.inspect(container_id)
    stop_record = {
        "stop_command_exit_code": stop_exit_code,
        "stop_command_error": stop_error,
        "stop_command_duration_s": stop_duration,
        "test_window_s": time.monotonic() - test_started,
        "pre_stop_admission_inflight": pre_stop_inflight,
        "pre_stop_observation_error": observation_error,
        "unfinished_client_requests_at_stop": unfinished_at_stop,
        "overlap_at_stop": overlap_at_stop,
        "request_outcomes": outcome_counts(outcomes),
        "requests": outcomes,
        "container_state": state,
    }
    stop_record["gate_failures"] = stop_gate_failures(stop_record)
    stop_record["conclusion"] = "pass" if not stop_record["gate_failures"] else "failed"
    evidence["inflight_stop"] = stop_record

    evidence["active_stage"] = "recovery"
    recovery_started = time.monotonic()
    recovery: dict[str, Any] = {"attempted": True, "result": "failed"}
    try:
        start = compose.run("start", args.service, timeout=args.readiness_timeout + 30)
        recovery["start_command_exit_code"] = start.returncode
        if start.returncode:
            raise RuntimeError("container start failed")
        ready_s, _ = wait_ready(diagnostics, args.readiness_timeout)
        recovered = diagnostics("/healthz")
        recovered_process_id = recovered.get("process_id")
        process_changed = is_new_process_id(after.get("process_id"), recovered_process_id)
        recovery.update({
            "start_and_readiness_s": time.monotonic() - recovery_started,
            "readiness_wait_s": ready_s,
            "process_id": recovered_process_id,
            "process_changed": process_changed,
        })
        if not process_changed:
            raise RuntimeError("recovery did not produce a new server process_id")
        recovery["reference_validation"] = run_smoke(
            args.models, args.url, diagnostics, token=token, context=context
        )
        recovery["result"] = "pass"
    except Exception as exc:
        recovery.update(safe_failure(exc))
    evidence["recovery"] = recovery
    evidence["completed_at_unix_s"] = time.time()
    evidence["active_stage"] = "complete"
    if stop_record["gate_failures"]:
        raise RuntimeError("in-flight stop gate failed")
    if recovery.get("result") != "pass":
        raise RuntimeError("recovery gate failed")
    return evidence


def run_and_write(args: argparse.Namespace) -> bool:
    """Reserve the evidence path before Docker mutations and always finalize it."""
    # Opening with x is deliberately first: collision and permission failures
    # must happen before restart/stop can alter the service.
    with args.output.open("x", encoding="utf-8") as stream:
        evidence: dict[str, Any] = {}
        succeeded = False
        try:
            execute(args, evidence)
            evidence["result"] = "pass"
            succeeded = True
        except Exception as exc:
            evidence["result"] = "failed"
            evidence.update(safe_failure(exc))
        finally:
            evidence["evidence_written_at_unix_s"] = time.time()
            stream.seek(0)
            stream.truncate()
            json.dump(evidence, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return succeeded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", type=Path, default=Path(__file__).resolve().parent / "docker-compose.deploy.yml")
    parser.add_argument("--service", default="inference")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--url", required=True, help="public authenticated HTTPS base URL")
    parser.add_argument("--diagnostics-command-file", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--ca-file", type=Path, default=Path(os.environ[CA_FILE_ENV]) if os.environ.get(CA_FILE_ENV) else None)
    parser.add_argument("--inflight-requests", type=int, default=16)
    parser.add_argument("--stop-delay", type=float, default=0.0)
    parser.add_argument("--overlap-timeout", type=float, default=2.0)
    parser.add_argument("--stop-timeout", type=int, default=60)
    parser.add_argument("--readiness-timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.inflight_requests <= 0 or args.stop_delay < 0 or args.stop_timeout <= 0 or args.overlap_timeout <= 0:
        parser.error("request count and timeouts must be positive; stop delay may be zero")
    if not run_and_write(args):
        raise SystemExit(f"FAIL: lifecycle gate failed; partial evidence written to {args.output}")
    print(f"PASS: lifecycle evidence written to {args.output}")


if __name__ == "__main__":
    main()
