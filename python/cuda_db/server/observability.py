"""Request identity, safe access logging, and a generic error boundary."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from .admission import ASGIApp, _json_response

request_logger = logging.getLogger("cuda_db.request")
request_logger.setLevel(logging.INFO)

_KNOWN_ENDPOINTS = {
    "/predict": "/predict",
    "/predict/": "/predict",
    "/predict/raw": "/predict/raw",
    "/predict/raw/": "/predict/raw",
    "/healthz": "/healthz",
    "/healthz/": "/healthz",
    "/readyz": "/readyz",
    "/readyz/": "/readyz",
    "/docs": "/docs",
    "/redoc": "/redoc",
    "/openapi.json": "/openapi.json",
}


def log_message(event: str, **fields: Any) -> str:
    """Render safe structured fields with even Uvicorn's default formatter."""
    return json.dumps(
        {"event": event, **fields}, separators=(",", ":"), sort_keys=True
    )


class RequestLoggingMiddleware:
    """Give every HTTP response a server-generated id and emit safe metadata."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope["cuda_db.request_id"] = request_id
        started = time.perf_counter()
        response_started = False
        response_interrupted = False
        status = 500

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started, status
            if message["type"] == "http.response.start":
                response_started = True
                status = int(message["status"])
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, tracked_send)
        except Exception as exc:
            response_interrupted = response_started
            # Deliberately omit exception text, traceback and locals here: an
            # arbitrary downstream failure can embed a body, token, or tensor.
            request_logger.error(
                log_message(
                    "unexpected_request_failure",
                    failure_type=type(exc).__name__,
                    request_id=request_id,
                ),
                extra={"failure_type": type(exc).__name__, "request_id": request_id},
            )
            if not response_started:
                status = 500
                await _json_response(
                    tracked_send, 500, {"detail": "internal server error"}
                )
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            client = scope.get("client")
            client_address = client[0] if isinstance(client, (tuple, list)) and client else "-"
            event = {
                "method": _safe_method(scope.get("method")),
                "endpoint": _KNOWN_ENDPOINTS.get(scope.get("path"), "unknown"),
                "status": status,
                "duration_ms": round(duration_ms, 3),
                "client": client_address,
                "key_id": scope.get("cuda_db.client_key_id", "-"),
                "request_id": request_id,
            }
            request_logger.info(
                json.dumps(event, separators=(",", ":"), sort_keys=True),
                extra=event,
            )
        if response_interrupted:
            # Raise outside the except block so the sanitized transport error
            # does not retain the secret-bearing original as __context__.
            raise RuntimeError("response interrupted") from None


def _safe_method(method: Any) -> str:
    if method in {"GET", "POST", "HEAD", "OPTIONS", "PUT", "PATCH", "DELETE"}:
        return method
    return "unknown"
