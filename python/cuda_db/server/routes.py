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
import logging
import time
from concurrent.futures import Future

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from ..schemas.prediction import PredictionRequest, PredictionResponse
from .observability import log_message

router = APIRouter()
logger = logging.getLogger("cuda_db.inference")

# Little-endian float32. Explicit rather than "f4" so the wire format is fixed
# by the protocol instead of by whatever the server's native byte order is.
RAW_DTYPE = "<f4"
RAW_ITEMSIZE = 4


def _finish_native_future(future: asyncio.Future, request: Request) -> None:
    """Observe detached failures and make an unhealthy engine unready."""
    try:
        exception = future.exception()
    except (asyncio.CancelledError, Exception):
        return
    if isinstance(exception, request.app.state.inference_error_type):
        request.app.state.ready = False
        request.app.state.readiness_detail = f"inference engine failed: {exception}"
        request.app.state.admission.close()
        logger.error(
            log_message(
                "late_inference_engine_failure",
                failure_type=type(exception).__name__,
                request_id=request.scope.get("cuda_db.request_id", "-"),
            ),
            extra={
                "failure_type": type(exception).__name__,
                "request_id": request.scope.get("cuda_db.request_id", "-"),
            },
        )


async def _predict(array: np.ndarray, request: Request) -> PredictionResponse:
    """Submit one already-validated tensor and wait for its result.

    The native predict() blocks while the scheduler batches and runs, so it
    must not run on the event loop -- otherwise concurrent requests would
    serialize and never form a batch, which is the whole point of the system.
    """
    if not np.isfinite(array).all():
        raise HTTPException(status_code=400, detail="input values must all be finite")

    deadline = request.scope["cuda_db.deadline"]
    if time.monotonic() >= deadline:
        raise HTTPException(status_code=504, detail="inference deadline exceeded")

    started = time.perf_counter()
    work: Future = request.app.state.executor.submit(
        request.app.state.runtime.predict, array
    )
    request.scope["cuda_db.admission_lease"].attach_work(work)
    wrapped = asyncio.wrap_future(work)
    wrapped.add_done_callback(lambda future: _finish_native_future(future, request))
    request_id, output = await asyncio.shield(wrapped)
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
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="server is not ready")
    return {"status": "ok", **runtime.stats()}


@router.get("/readyz")
async def readyz(request: Request) -> dict:
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="server is not ready")
    return {"status": "ready"}
