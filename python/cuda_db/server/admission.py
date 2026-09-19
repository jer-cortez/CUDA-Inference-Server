"""Bound prediction requests before reading or parsing their bodies.

The permit deliberately outlives the HTTP task when native inference has
already started. Cancelling an asyncio waiter cannot cancel a C++/CUDA call;
freeing its permit early would allow replacements to pile up behind work that
is still consuming the executor and GPU.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import Future
from contextlib import suppress
from typing import Any, Awaitable, Callable

Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], Receive, Send], Awaitable[None]]


class AdmissionController:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._inflight = 0
        self._closed = False
        self._condition = threading.Condition()

    def try_acquire(self) -> "AdmissionLease | None":
        with self._condition:
            if self._closed or self._inflight >= self.capacity:
                return None
            self._inflight += 1
        return AdmissionLease(self)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _release(self) -> None:
        with self._condition:
            self._inflight -= 1
            if self._inflight < 0:  # Defensive invariant; never recover silently.
                raise RuntimeError("admission permit released more than once")
            self._condition.notify_all()

    def _wait_until_drained(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._inflight == 0)

    async def drain(self) -> None:
        # A native call may hold the GIL only briefly on return, but waiting on
        # the condition in the event-loop thread would prevent HTTP cleanup and
        # future callbacks from finishing.
        await asyncio.to_thread(self._wait_until_drained)

    @property
    def inflight(self) -> int:
        with self._condition:
            return self._inflight


class AdmissionLease:
    """One slot, released after both HTTP handling and native work finish."""

    def __init__(self, controller: AdmissionController) -> None:
        self._controller = controller
        self._lock = threading.Lock()
        self._http_finished = False
        self._work: Future[Any] | None = None
        self._work_finished = False
        self._released = False

    def attach_work(self, future: Future[Any]) -> None:
        with self._lock:
            if self._work is not None:
                raise RuntimeError("an admission lease can own only one native submission")
            self._work = future
        future.add_done_callback(self._work_done)

    def _work_done(self, future: Future[Any]) -> None:
        # Observe detached failures after timeout/disconnect. concurrent.futures
        # does not log them, so explicit retrieval is part of ownership here.
        with suppress(BaseException):
            future.exception()
        release = False
        with self._lock:
            self._work_finished = True
            if self._http_finished and not self._released:
                self._released = True
                release = True
        if release:
            self._controller._release()

    def finish_http(self) -> None:
        release = False
        with self._lock:
            self._http_finished = True
            if (self._work is None or self._work_finished) and not self._released:
                self._released = True
                release = True
        if release:
            self._controller._release()


class PredictionAdmissionMiddleware:
    """Pure ASGI middleware for prediction admission, size and deadlines."""

    _PATHS = {"/predict", "/predict/", "/predict/raw", "/predict/raw/"}

    def __init__(
        self,
        app: ASGIApp,
        controller: AdmissionController,
        *,
        max_request_bytes: int,
        request_timeout_ms: int,
    ) -> None:
        self.app = app
        self.controller = controller
        self.max_request_bytes = max_request_bytes
        self.request_timeout_seconds = request_timeout_ms / 1000.0

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if not (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") in self._PATHS
        ):
            await self.app(scope, receive, send)
            return

        lease = self.controller.try_acquire()
        if lease is None:
            await _json_response(
                send,
                503,
                {"detail": "inference capacity is full or the server is draining"},
                headers=[(b"retry-after", b"1")],
            )
            return

        deadline = time.monotonic() + self.request_timeout_seconds
        scope["cuda_db.admission_lease"] = lease
        scope["cuda_db.deadline"] = deadline
        app_task: asyncio.Task[None] | None = None
        disconnect_task: asyncio.Task[None] | None = None
        try:
            content_length = _content_length(scope)
            if content_length is not None and content_length > self.max_request_bytes:
                await _json_response(send, 413, {"detail": "request body is too large"})
                return

            try:
                body, disconnected = await self._read_body(receive, deadline)
            except RequestBodyTooLarge:
                await _json_response(send, 413, {"detail": "request body is too large"})
                return
            if disconnected:
                return
            if body is None:  # Upload deadline.
                await _json_response(send, 408, {"detail": "request body deadline exceeded"})
                return

            replayed = False
            disconnected_event = asyncio.Event()

            async def replay_receive() -> dict[str, Any]:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                # Only the watcher below reads the real receive channel after
                # the terminal body message. Multiple concurrent receive()
                # calls violate ASGI and can lose the disconnect notification.
                await disconnected_event.wait()
                return {"type": "http.disconnect"}

            response_started = False

            async def tracked_send(message: dict[str, Any]) -> None:
                nonlocal response_started
                if message["type"] == "http.response.start":
                    response_started = True
                await send(message)

            app_task = asyncio.create_task(self.app(scope, replay_receive, tracked_send))
            disconnect_task = asyncio.create_task(
                _wait_for_disconnect(receive, disconnected_event)
            )
            remaining = max(0.0, deadline - time.monotonic())
            done, _ = await asyncio.wait(
                {app_task, disconnect_task},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if app_task in done:
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task
                await app_task
                return

            app_task.cancel()
            with suppress(asyncio.CancelledError):
                await app_task

            if disconnect_task in done:
                await disconnect_task
                return

            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
            if not response_started:
                await _json_response(send, 504, {"detail": "inference deadline exceeded"})
        finally:
            # This also runs when the server cancels the middleware task. Wait
            # for the downstream task to acknowledge cancellation before
            # marking HTTP finished; otherwise it could submit native work
            # after the lease had already released its permit.
            for task in (app_task, disconnect_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (app_task, disconnect_task):
                if task is not None:
                    with suppress(asyncio.CancelledError, Exception):
                        await task
            lease.finish_http()

    async def _read_body(
        self, receive: Any, deadline: float
    ) -> tuple[bytes | None, bool]:
        body = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, False
            try:
                message = await asyncio.wait_for(receive(), remaining)
            except asyncio.TimeoutError:
                return None, False
            if message["type"] == "http.disconnect":
                return b"", True
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_request_bytes:
                # Signal with a sentinel exception so the response is emitted
                # before FastAPI sees any of the oversized body.
                raise RequestBodyTooLarge
            body.extend(chunk)
            if not message.get("more_body", False):
                return bytes(body), False


class RequestBodyTooLarge(Exception):
    pass


async def _wait_for_disconnect(receive: Any, disconnected: asyncio.Event) -> None:
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            disconnected.set()
            return


def _content_length(scope: dict[str, Any]) -> int | None:
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                length = int(value)
            except ValueError:
                return None
            return max(0, length)
    return None


async def _json_response(
    send: Any,
    status: int,
    content: dict[str, Any],
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(content, separators=(",", ":")).encode("utf-8")
    response_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if headers:
        response_headers.extend(headers)
    await send({"type": "http.response.start", "status": status, "headers": response_headers})
    await send({"type": "http.response.body", "body": body})
