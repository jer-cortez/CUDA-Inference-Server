"""FastAPI application factory.

One ``InferenceRuntime`` (and therefore one scheduler thread) is created at
startup and shared by every request -- that shared queue is what lets
concurrent requests coalesce into a batch.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .._native import InferenceError, InferenceRuntime, RuntimeConfig
from ..config import RuntimeSettings
from .executor import make_executor
from .routes import router


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    settings = settings or RuntimeSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = InferenceRuntime(
            RuntimeConfig(
                max_batch_size=settings.max_batch_size,
                max_wait_ms=settings.max_wait_ms,
                input_elems=settings.input_elems,
                output_elems=settings.output_elems,
            )
        )
        executor = make_executor(settings.executor_workers)
        app.state.settings = settings
        app.state.runtime = runtime
        app.state.executor = executor
        try:
            yield
        finally:
            # Stop the scheduler before the pool: shutdown() drains queued
            # requests and resolves their futures, and the executor threads
            # still blocked in predict() need to return before they can be
            # joined.
            runtime.shutdown()
            executor.shutdown(wait=True)

    app = FastAPI(title="cuda-db", lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(InferenceError)
    async def _inference_error(_: Request, exc: InferenceError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
