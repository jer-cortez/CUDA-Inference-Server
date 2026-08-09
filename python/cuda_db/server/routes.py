"""HTTP surface: one prediction endpoint and one health endpoint."""

from __future__ import annotations

import asyncio
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from ..schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
    settings = request.app.state.settings
    if len(body.input) != settings.input_elems:
        raise HTTPException(
            status_code=400,
            detail=f"expected {settings.input_elems} input elements, got {len(body.input)}",
        )

    array = np.asarray(body.input, dtype=np.float32)

    # The native predict() blocks while the scheduler batches and runs, so it
    # must not run on the event loop -- otherwise concurrent requests would
    # serialize and never form a batch, which is the whole point of the system.
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


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    return {"status": "ok", **request.app.state.runtime.stats()}
