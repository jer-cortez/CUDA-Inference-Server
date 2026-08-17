"""Async request helper for the load test.

Kept separate from load_test.py so the timing logic stays small enough to audit:
everything that could inflate a latency measurement (payload construction, JSON
encoding, array allocation) has to happen outside the timed region, and that is
easier to verify when the timed region is four lines in one file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

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

    @property
    def ok(self) -> bool:
        return self.status_code == 200


def make_payload(input_elems: int, seed: int = 0) -> bytes:
    """Build one raw float32 request body.

    Called once and reused across requests: constructing this per request would
    put ~0.6 MB of array work inside the load loop and show up as latency that
    has nothing to do with the server.
    """
    rng = np.random.default_rng(seed)
    array = rng.standard_normal(input_elems, dtype=np.float32)
    return array.astype(RAW_DTYPE, copy=False).tobytes()


async def send_request(client, url: str, payload: bytes) -> RequestResult:
    """POST one payload and time the round trip.

    Timed with perf_counter around the await only. The response body is parsed
    after the clock stops, so JSON decoding of the 1000-float output is not
    charged to the server's latency.
    """
    started = time.perf_counter()
    response = await client.post(
        url, content=payload, headers={"content-type": "application/octet-stream"}
    )
    latency_s = time.perf_counter() - started

    server_latency_ms: float | None = None
    if response.status_code == 200:
        try:
            server_latency_ms = float(response.json()["latency_ms"])
        except (ValueError, KeyError):
            # A malformed body should not abort a benchmark run; the status
            # code already tells us the request succeeded.
            server_latency_ms = None

    return RequestResult(
        latency_s=latency_s,
        status_code=response.status_code,
        server_latency_ms=server_latency_ms,
    )
