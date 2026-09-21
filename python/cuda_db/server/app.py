"""FastAPI application factory.

One ``InferenceRuntime`` (and therefore one scheduler thread) is created at
startup and shared by every request -- that shared queue is what lets
concurrent requests coalesce into a batch.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from concurrent.futures import Future
from contextlib import suppress

import asyncio
import logging

import numpy as np

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .._native import InferenceError, InferenceRuntime, RuntimeConfig
from ..config import RuntimeSettings
from .admission import AdmissionController, PredictionAdmissionMiddleware
from .executor import make_executor
from .observability import RequestLoggingMiddleware, log_message
from .routes import router
from .security import ApiKeyAuthenticator, AuthenticationMiddleware

logger = logging.getLogger("cuda_db.server")


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    _configure_logging()
    settings = settings or RuntimeSettings.from_env()
    authenticator = ApiKeyAuthenticator(settings.auth_required)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.ready = False
        app.state.readiness_detail = "starting"
        runtime = None
        startup_probe: Future | None = None
        executor = None

        async def stop_runtime(detail: str) -> None:
            app.state.ready = False
            app.state.readiness_detail = detail
            app.state.admission.close()
            await app.state.admission.drain()
            # A cancelled startup task can leave its probe running outside the
            # admission controller. Join it before stopping the scheduler.
            if startup_probe is not None and not startup_probe.done():
                with suppress(BaseException):
                    await asyncio.to_thread(startup_probe.result)
            if runtime is not None:
                await asyncio.to_thread(runtime.shutdown)
            if executor is not None:
                await asyncio.to_thread(executor.shutdown, True)
            authenticator.clear()

        try:
            # Validate security configuration before allocating an executor or
            # constructing native state. A bad or missing store therefore
            # fails startup closed without briefly exposing inference.
            authenticator.load(settings.api_keys_file)
            executor = make_executor(settings.executor_workers)
            app.state.executor = executor
            runtime = InferenceRuntime(
                RuntimeConfig(
                    max_batch_size=settings.max_batch_size,
                    max_wait_ms=settings.max_wait_ms,
                    input_elems=settings.input_elems,
                    output_elems=settings.output_elems,
                    model_path=settings.model_path,
                )
            )
            app.state.runtime = runtime

            stats = runtime.stats()
            if settings.require_gpu and stats.get("engine") != "onnx":
                raise RuntimeError("require_gpu=true but the ONNX CUDA engine is not active")

            startup_probe = executor.submit(
                runtime.predict, np.zeros(settings.input_elems, dtype=np.float32)
            )
            wrapped_probe = asyncio.wrap_future(startup_probe)
            wrapped_probe.add_done_callback(_consume_future_exception)
            _, output = await asyncio.shield(wrapped_probe)
            output_array = np.asarray(output)
            if output_array.size != settings.output_elems:
                raise RuntimeError(
                    "startup probe returned "
                    f"{output_array.size} outputs; expected {settings.output_elems}"
                )
            if not np.isfinite(output_array).all():
                raise RuntimeError("startup probe returned non-finite output")
            runtime.reset_stats()
            app.state.ready = True
            app.state.readiness_detail = "ready"
        except BaseException as exc:
            app.state.ready = False
            app.state.readiness_detail = f"startup failed: {exc}"
            logger.error(
                log_message("startup_failed", failure_type=type(exc).__name__),
                extra={"failure_type": type(exc).__name__},
            )
            await stop_runtime(app.state.readiness_detail)
            raise
        try:
            yield
        finally:
            await stop_runtime("draining")

    app = FastAPI(
        title="cuda-db",
        lifespan=lifespan,
        docs_url=None if settings.deployment_mode else "/docs",
        redoc_url=None if settings.deployment_mode else "/redoc",
        openapi_url=None if settings.deployment_mode else "/openapi.json",
    )
    admission = AdmissionController(settings.max_inflight_requests)
    app.state.admission = admission
    app.state.ready = False
    app.state.readiness_detail = "not started"
    app.state.inference_error_type = InferenceError
    app.add_middleware(
        PredictionAdmissionMiddleware,
        controller=admission,
        max_request_bytes=settings.max_request_bytes,
        request_timeout_ms=settings.request_timeout_ms,
    )
    app.add_middleware(AuthenticationMiddleware, authenticator=authenticator)
    # Added last so Starlette makes it the outermost user middleware. It sees
    # authentication and admission rejections as well as routed responses.
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(router)

    @app.exception_handler(InferenceError)
    async def _inference_error(request: Request, exc: InferenceError) -> JSONResponse:
        request.app.state.ready = False
        request.app.state.readiness_detail = f"inference engine failed: {exc}"
        request.app.state.admission.close()
        logger.error(
            log_message(
                "inference_engine_failed",
                failure_type=type(exc).__name__,
                request_id=request.scope.get("cuda_db.request_id", "-"),
            ),
            extra={
                "failure_type": type(exc).__name__,
                "request_id": request.scope.get("cuda_db.request_id", "-"),
            },
        )
        return JSONResponse(status_code=500, content={"detail": "inference failed"})

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, __: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": "invalid request"})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        # Pydantic's normal error body includes the rejected input value.
        return JSONResponse(status_code=422, content={"detail": "invalid request"})

    return app


def _consume_future_exception(future: asyncio.Future) -> None:
    with suppress(asyncio.CancelledError, Exception):
        future.exception()


def _configure_logging() -> None:
    """Route application logs through Uvicorn when its config is active.

    Uvicorn's default configuration installs handlers only on ``uvicorn`` and
    not the root logger. Reusing those already-configured handlers keeps
    request audit records visible without adding duplicate handlers in tests
    or in embedding applications that configure their own root logger.
    """
    application_logger = logging.getLogger("cuda_db")
    application_logger.setLevel(logging.INFO)
    if application_logger.handlers:
        return
    uvicorn_logger = logging.getLogger("uvicorn")
    if uvicorn_logger.handlers:
        application_logger.handlers = list(uvicorn_logger.handlers)
        application_logger.propagate = False


app = create_app()
