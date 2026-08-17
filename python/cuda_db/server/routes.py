"""HTTP surface: two prediction endpoints and one health endpoint.

/predict takes JSON and is the friendly default. /predict/raw takes the same
tensor as binary float32 and exists because JSON cannot carry a ResNet input
cheaply: 150,528 floats encode to ~3.1 MB of text costing ~90 ms to serialize
and parse, against single-digit-millisecond inference. Benchmarking through
JSON would measure Python's JSON parser and, because that cost is paid per
request regardless of batching, would understate the batching win. Both
endpoints share one execution path below so they cannot drift apart.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from ..schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter()

# Little-endian float32. Explicit rather than "f4" so the wire format is fixed
# by the protocol instead of by whatever the server's native byte order is.
RAW_DTYPE = "<f4"
RAW_ITEMSIZE = 4


async def _predict(array: np.ndarray, request: Request) -> PredictionResponse:
    """Submit one already-validated tensor and wait for its result.

    The native predict() blocks while the scheduler batches and runs, so it
    must not run on the event loop -- otherwise concurrent requests would
    serialize and never form a batch, which is the whole point of the system.
    """
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    request_id, output = await loop.run_in_executor(
        request.app.state.executor, request.app.state.runtime.predict, array
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    return PredictionResponse(
        request_id=request_id,
        output=output.tolist(),
        latency_ms=latency_ms,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
    settings = request.app.state.settings
    if len(body.input) != settings.input_elems:
        raise HTTPException(
            status_code=400,
            detail=f"expected {settings.input_elems} input elements, got {len(body.input)}",
        )

    return await _predict(np.asarray(body.input, dtype=np.float32), request)


@router.post("/predict/raw", response_model=PredictionResponse)
async def predict_raw(request: Request) -> PredictionResponse:
    """Binary tensor input: the body is input_elems little-endian float32 values.

    The response stays JSON -- 1000 output floats is ~12 KB, small enough not to
    matter, and it keeps the response readable.
    """
    settings = request.app.state.settings
    body = await request.body()

    expected_bytes = settings.input_elems * RAW_ITEMSIZE
    if len(body) != expected_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"expected {expected_bytes} bytes "
                f"({settings.input_elems} float32 values), got {len(body)}"
            ),
        )

    # A view over the request body, not a copy -- which is the entire point of
    # this endpoint existing.
    array = np.frombuffer(body, dtype=RAW_DTYPE)
    return await _predict(array, request)


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    return {"status": "ok", **request.app.state.runtime.stats()}
