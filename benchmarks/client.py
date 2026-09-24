"""Async request helper for the load test.

Kept separate from load_test.py so the timing logic stays small enough to audit:
everything that could inflate a latency measurement (payload construction, JSON
encoding, array allocation) has to happen outside the timed region, and that is
easier to verify when the timed region is four lines in one file.
"""

from __future__ import annotations

import math
import os
import stat
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

import httpx
import numpy as np

# Must match RAW_DTYPE in python/cuda_db/server/routes.py.
RAW_DTYPE = "<f4"


@dataclass(frozen=True)
class RequestResult:
    latency_s: float
    status_code: int
    # The server's own measurement of time spent inside predict(). Comparing it
    # against latency_s separates queueing/inference from HTTP and client
    # overhead -- if they diverge sharply, the bottleneck is not the GPU.
    server_latency_ms: float | None = None
    # Machine-readable and deliberately low-cardinality. In particular, never
    # put exception strings here: HTTP client exceptions may contain request
    # details and benchmark credentials must not reach result files or logs.
    failure_kind: str | None = None
    failure_detail: str | None = None
    request_id: int | None = None

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and self.failure_kind is None


@dataclass(frozen=True)
class ClientTLSConfig:
    """Credentials and trust roots for the remote benchmark client.

    The token is excluded from repr so accidentally logging this otherwise
    useful configuration object cannot disclose it.
    """

    token: str | None = dataclass_field(default=None, repr=False)
    ca_file: Path | None = None

    @classmethod
    def resolve(
        cls,
        *,
        token_file: str | os.PathLike[str] | None = None,
        ca_file: str | os.PathLike[str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "ClientTLSConfig":
        env = os.environ if environ is None else environ
        raw_inline_token = env.get("CUDA_DB_BENCHMARK_TOKEN", "")
        if "\n" in raw_inline_token or "\r" in raw_inline_token:
            raise ValueError("CUDA_DB_BENCHMARK_TOKEN must contain exactly one token")
        inline_token = raw_inline_token.strip()
        env_token_file = env.get("CUDA_DB_BENCHMARK_TOKEN_FILE", "").strip()
        selected_token_file = os.fspath(token_file) if token_file is not None else env_token_file

        if inline_token and selected_token_file:
            raise ValueError(
                "set only one of CUDA_DB_BENCHMARK_TOKEN and a benchmark token file"
            )

        token = inline_token or None
        if selected_token_file:
            token = _read_protected_secret(Path(selected_token_file))

        selected_ca = (
            os.fspath(ca_file)
            if ca_file is not None
            else env.get("CUDA_DB_BENCHMARK_CA_FILE", "").strip()
        )
        ca_path = Path(selected_ca) if selected_ca else None
        if ca_path is not None and not ca_path.is_file():
            raise ValueError(f"benchmark CA file is not a regular file: {ca_path}")

        return cls(token=token, ca_file=ca_path)


def _read_protected_secret(path: Path) -> str:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"benchmark token file is not a regular file: {path}")
    if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError("benchmark token file must not be accessible by group or others")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("benchmark token file is empty")
    if "\n" in token or "\r" in token:
        raise ValueError("benchmark token file must contain exactly one token")
    return token


def make_async_client(
    base_url: str,
    *,
    token_file: str | os.PathLike[str] | None = None,
    ca_file: str | os.PathLike[str] | None = None,
    timeout: float | httpx.Timeout = 300.0,
    limits: httpx.Limits | None = None,
    environ: Mapping[str, str] | None = None,
) -> httpx.AsyncClient:
    """Create an authenticated client whose TLS verification cannot be disabled."""
    config = ClientTLSConfig.resolve(
        token_file=token_file, ca_file=ca_file, environ=environ
    )
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("benchmark endpoint scheme must be HTTP or HTTPS")
    if not parsed.hostname:
        raise ValueError("benchmark endpoint must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("benchmark endpoint must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("benchmark endpoint must not contain a query or fragment")
    is_loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback and not config.token):
        raise ValueError(
            "benchmark endpoint must use HTTPS; only unauthenticated localhost HTTP is allowed"
        )

    headers = {"authorization": f"Bearer {config.token}"} if config.token else None
    verify: bool | str = str(config.ca_file) if config.ca_file is not None else True
    kwargs: dict = {"base_url": base_url, "timeout": timeout, "verify": verify}
    if headers is not None:
        kwargs["headers"] = headers
    if limits is not None:
        kwargs["limits"] = limits
    return httpx.AsyncClient(**kwargs)


def make_payload(input_elems: int, seed: int = 0) -> bytes:
    """Build one raw float32 request body.

    Called once and reused across requests: constructing this per request would
    put ~0.6 MB of array work inside the load loop and show up as latency that
    has nothing to do with the server.
    """
    rng = np.random.default_rng(seed)
    array = rng.standard_normal(input_elems, dtype=np.float32)
    return array.astype(RAW_DTYPE, copy=False).tobytes()


def make_json_payload(input_elems: int, seed: int = 0) -> list[float]:
    """Build the same deterministic input in the JSON endpoint's wire shape."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(input_elems, dtype=np.float32).tolist()


def _validate_success_body(
    response: httpx.Response, expected_output_elems: int | None
) -> tuple[float | None, int | None, str | None]:
    try:
        body = response.json()
    except (ValueError, TypeError):
        return None, None, "invalid_json"
    if not isinstance(body, dict):
        return None, None, "body_not_object"

    request_id = body.get("request_id")
    if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
        return None, None, "invalid_request_id"

    latency = body.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        return None, None, "invalid_latency_ms"
    server_latency_ms = float(latency)
    if not math.isfinite(server_latency_ms) or server_latency_ms < 0:
        return None, None, "invalid_latency_ms"

    output = body.get("output")
    if not isinstance(output, list):
        return None, None, "invalid_output"
    if expected_output_elems is not None and len(output) != expected_output_elems:
        return None, None, "wrong_output_length"
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in output
    ):
        return None, None, "invalid_output"
    return server_latency_ms, request_id, None


async def send_request(
    client,
    url: str,
    payload: bytes,
    *,
    expected_output_elems: int | None = None,
) -> RequestResult:
    """POST one payload and time the round trip.

    Timed with perf_counter around the await only. The response body is parsed
    after the clock stops, so JSON decoding of the 1000-float output is not
    charged to the server's latency.
    """
    started = time.perf_counter()
    try:
        response = await client.post(
            url, content=payload, headers={"content-type": "application/octet-stream"}
        )
    except httpx.TimeoutException:
        return RequestResult(
            latency_s=time.perf_counter() - started,
            status_code=0,
            failure_kind="timeout",
            failure_detail="request_timed_out",
        )
    except httpx.TransportError:
        return RequestResult(
            latency_s=time.perf_counter() - started,
            status_code=0,
            failure_kind="transport",
            failure_detail="request_transport_error",
        )
    latency_s = time.perf_counter() - started

    server_latency_ms: float | None = None
    failure_kind: str | None = None
    failure_detail: str | None = None
    request_id: int | None = None
    if response.status_code == 200:
        server_latency_ms, request_id, failure_detail = _validate_success_body(
            response, expected_output_elems
        )
        if failure_detail is not None:
            failure_kind = "malformed_response"
    else:
        failure_kind = "http_status"
        failure_detail = f"http_{response.status_code}"

    return RequestResult(
        latency_s=latency_s,
        status_code=response.status_code,
        server_latency_ms=server_latency_ms,
        failure_kind=failure_kind,
        failure_detail=failure_detail,
        request_id=request_id,
    )


async def send_json_request(
    client,
    url: str,
    payload: list[float],
    *,
    expected_output_elems: int | None = None,
) -> RequestResult:
    """POST the public JSON shape, including JSON encoding in client latency."""
    started = time.perf_counter()
    try:
        response = await client.post(url, json={"input": payload})
    except httpx.TimeoutException:
        return RequestResult(
            time.perf_counter() - started, 0,
            failure_kind="timeout", failure_detail="request_timed_out",
        )
    except httpx.TransportError:
        return RequestResult(
            time.perf_counter() - started, 0,
            failure_kind="transport", failure_detail="request_transport_error",
        )
    latency_s = time.perf_counter() - started
    if response.status_code != 200:
        return RequestResult(
            latency_s, response.status_code,
            failure_kind="http_status", failure_detail=f"http_{response.status_code}",
        )
    server_latency_ms, request_id, detail = _validate_success_body(response, expected_output_elems)
    return RequestResult(
        latency_s, response.status_code, server_latency_ms,
        failure_kind="malformed_response" if detail else None,
        failure_detail=detail,
        request_id=request_id,
    )
