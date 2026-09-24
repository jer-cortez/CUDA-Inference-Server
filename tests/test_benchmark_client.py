import importlib.util
from pathlib import Path
import sys

import httpx
import pytest


SPEC = importlib.util.spec_from_file_location(
    "benchmark_client", Path(__file__).parents[1] / "benchmarks" / "client.py"
)
benchmark_client = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark_client
SPEC.loader.exec_module(benchmark_client)


def test_config_reads_protected_token_file_and_ca(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("pilot.secret\n", encoding="utf-8")
    token_file.chmod(0o600)
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("certificate", encoding="utf-8")

    config = benchmark_client.ClientTLSConfig.resolve(
        token_file=token_file, ca_file=ca_file, environ={}
    )

    assert config.token == "pilot.secret"
    assert config.ca_file == ca_file
    assert "pilot.secret" not in repr(config)


def test_config_rejects_ambiguous_or_exposed_token_file(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("secret", encoding="utf-8")
    token_file.chmod(0o644)

    with pytest.raises(ValueError, match="group or others"):
        benchmark_client.ClientTLSConfig.resolve(token_file=token_file, environ={})

    token_file.chmod(0o600)
    with pytest.raises(ValueError, match="only one"):
        benchmark_client.ClientTLSConfig.resolve(
            token_file=token_file,
            environ={"CUDA_DB_BENCHMARK_TOKEN": "inline-secret"},
        )


def test_client_requires_https_except_unauthenticated_loopback():
    client = benchmark_client.make_async_client("http://127.0.0.1:8000", environ={})
    assert client._transport._pool._ssl_context.verify_mode != 0

    with pytest.raises(ValueError, match="must use HTTPS"):
        benchmark_client.make_async_client(
            "http://example.test", environ={"CUDA_DB_BENCHMARK_TOKEN": "secret"}
        )


@pytest.mark.asyncio
async def test_send_request_validates_success_and_output_length():
    async def handler(request):
        assert request.headers["content-type"] == "application/octet-stream"
        return httpx.Response(
            200,
            json={"request_id": 1, "latency_ms": 2.5, "output": [1.0, 2.0]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await benchmark_client.send_request(
            client, "https://example.test/predict/raw", b"payload", expected_output_elems=2
        )
        malformed = await benchmark_client.send_request(
            client, "https://example.test/predict/raw", b"payload", expected_output_elems=3
        )

    assert result.ok
    assert result.server_latency_ms == 2.5
    assert not malformed.ok
    assert malformed.failure_kind == "malformed_response"
    assert malformed.failure_detail == "wrong_output_length"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "kind"),
    [
        (httpx.ReadTimeout("late"), "timeout"),
        (httpx.ConnectError("unreachable"), "transport"),
    ],
)
async def test_send_request_classifies_transport_failures(exception, kind):
    async def handler(request):
        raise exception

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await benchmark_client.send_request(
            client, "https://example.test/predict/raw", b"payload"
        )

    assert not result.ok
    assert result.status_code == 0
    assert result.failure_kind == kind
    assert "unreachable" not in (result.failure_detail or "")


@pytest.mark.asyncio
async def test_send_request_classifies_http_status_without_copying_body():
    secret = "must-not-appear"

    async def handler(request):
        return httpx.Response(401, text=secret)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await benchmark_client.send_request(
            client, "https://example.test/predict/raw", b"payload"
        )

    assert result.failure_kind == "http_status"
    assert result.failure_detail == "http_401"
    assert secret not in repr(result)
