from __future__ import annotations

import asyncio
import importlib
import threading
from concurrent.futures import Future

import numpy as np
import pytest

from cuda_db._native import InferenceError
from cuda_db.config import RuntimeSettings
from cuda_db.server.admission import AdmissionController, PredictionAdmissionMiddleware

from .conftest import TEST_INPUT_ELEMS, TEST_OUTPUT_ELEMS


def _scope(path: str = "/predict", headers: list[tuple[bytes, bytes]] | None = None):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers or [],
        "client": ("test", 123),
        "server": ("test", 80),
    }


@pytest.mark.parametrize("path", ["/predict", "/predict/", "/predict/raw", "/predict/raw/"])
async def test_overload_rejects_before_reading_body(path):
    controller = AdmissionController(1)
    held = controller.try_acquire()
    assert held is not None
    receive_calls = 0
    sent = []

    async def downstream(scope, receive, send):  # pragma: no cover
        raise AssertionError("overloaded request reached application")

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b"large", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = PredictionAdmissionMiddleware(
        downstream, controller, max_request_bytes=16, request_timeout_ms=1_000
    )
    await middleware(_scope(path), receive, send)

    assert receive_calls == 0
    assert sent[0]["status"] == 503
    assert (b"retry-after", b"1") in sent[0]["headers"]
    held.finish_http()


async def test_streamed_body_limit_is_enforced_without_content_length():
    controller = AdmissionController(1)
    messages = iter(
        [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5", "more_body": False},
        ]
    )
    sent = []
    reached_app = False

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    async def downstream(scope, receive, send):
        nonlocal reached_app
        reached_app = True

    middleware = PredictionAdmissionMiddleware(
        downstream, controller, max_request_bytes=4, request_timeout_ms=1_000
    )
    await middleware(_scope(), receive, send)

    assert sent[0]["status"] == 413
    assert not reached_app
    assert controller.inflight == 0


async def test_disconnect_keeps_capacity_until_attached_work_finishes():
    controller = AdmissionController(1)
    native_work: Future[None] = Future()
    body_read = asyncio.Event()

    async def receive():
        if not body_read.is_set():
            body_read.set()
            return {"type": "http.request", "body": b"{}", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):  # pragma: no cover
        raise AssertionError(f"disconnected request sent {message}")

    async def downstream(scope, receive, send):
        scope["cuda_db.admission_lease"].attach_work(native_work)
        await asyncio.Event().wait()

    middleware = PredictionAdmissionMiddleware(
        downstream, controller, max_request_bytes=16, request_timeout_ms=1_000
    )
    await middleware(_scope(), receive, send)

    assert controller.inflight == 1
    assert controller.try_acquire() is None
    native_work.set_result(None)
    assert controller.inflight == 0


async def test_outer_cancellation_waits_for_downstream_ownership_handoff():
    controller = AdmissionController(1)
    native_work: Future[None] = Future()
    entered_app = asyncio.Event()
    never_disconnect = asyncio.Event()
    read_body = False

    async def receive():
        nonlocal read_body
        if not read_body:
            read_body = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await never_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    async def downstream(scope, receive, send):
        entered_app.set()
        try:
            await asyncio.Event().wait()
        finally:
            # This models cancellation landing just before executor submission.
            # Middleware must await this finally block before releasing HTTP
            # ownership of the permit.
            scope["cuda_db.admission_lease"].attach_work(native_work)

    middleware = PredictionAdmissionMiddleware(
        downstream, controller, max_request_bytes=16, request_timeout_ms=1_000
    )
    request_task = asyncio.create_task(middleware(_scope(), receive, send))
    await entered_app.wait()
    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert controller.inflight == 1
    native_work.set_result(None)
    assert controller.inflight == 0


async def test_upload_deadline_returns_408_and_releases_slot():
    controller = AdmissionController(1)
    sent = []

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    async def downstream(scope, receive, send):  # pragma: no cover
        raise AssertionError("timed-out upload reached application")

    middleware = PredictionAdmissionMiddleware(
        downstream, controller, max_request_bytes=16, request_timeout_ms=10
    )
    await middleware(_scope(), receive, send)

    assert sent[0]["status"] == 408
    assert controller.inflight == 0


async def test_invalid_request_releases_admission_slot(client_for):
    settings = RuntimeSettings(
        input_elems=TEST_INPUT_ELEMS,
        output_elems=TEST_OUTPUT_ELEMS,
        max_inflight_requests=1,
    )
    async with client_for(settings) as client:
        invalid = await client.post("/predict", json={"input": [1.0]})
        valid = await client.post(
            "/predict", json={"input": [1.0] * TEST_INPUT_ELEMS}
        )

    assert invalid.status_code == 400
    assert valid.status_code == 200


async def test_timeout_retains_capacity_until_native_work_finishes(client_for):
    settings = RuntimeSettings(
        input_elems=TEST_INPUT_ELEMS,
        output_elems=TEST_OUTPUT_ELEMS,
        max_inflight_requests=1,
        request_timeout_ms=50,
    )
    started = threading.Event()
    release = threading.Event()

    class BlockingRuntime:
        def predict(self, array):
            started.set()
            release.wait()
            return 123, np.zeros(TEST_OUTPUT_ELEMS, dtype=np.float32)

    try:
        async with client_for(settings) as client:
            app = client._transport.app
            app.state.runtime = BlockingRuntime()
            first_task = asyncio.create_task(
                client.post("/predict", json={"input": [1.0] * TEST_INPUT_ELEMS})
            )
            assert await asyncio.to_thread(started.wait, 1.0)

            first = await first_task
            assert first.status_code == 504

            overloaded = await client.post(
                "/predict", json={"input": [2.0] * TEST_INPUT_ELEMS}
            )
            assert overloaded.status_code == 503

            release.set()
            for _ in range(100):
                if app.state.admission.inflight == 0:
                    break
                await asyncio.sleep(0.01)
            assert app.state.admission.inflight == 0

            recovered = await client.post(
                "/predict", json={"input": [3.0] * TEST_INPUT_ELEMS}
            )
            assert recovered.status_code == 200
    finally:
        release.set()


async def test_nonfinite_input_is_client_error_and_server_stays_ready(client):
    response = await client.post(
        "/predict/raw",
        content=b"\x00\x00\xc0\x7f" * TEST_INPUT_ELEMS,
        headers={"content-type": "application/octet-stream"},
    )

    assert response.status_code == 400
    assert (await client.get("/readyz")).status_code == 200


async def test_readiness_requires_successful_startup_probe(client):
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


async def test_startup_probe_failure_aborts_lifespan(monkeypatch):
    app_module = importlib.import_module("cuda_db.server.app")

    class BrokenRuntime:
        def __init__(self, config):
            pass

        def stats(self):
            return {"engine": "stub"}

        def predict(self, array):
            raise RuntimeError("probe failed")

        def shutdown(self):
            pass

    monkeypatch.setattr(app_module, "InferenceRuntime", BrokenRuntime)
    app = app_module.create_app(
        RuntimeSettings(input_elems=TEST_INPUT_ELEMS, output_elems=TEST_OUTPUT_ELEMS)
    )

    with pytest.raises(RuntimeError, match="probe failed"):
        async with app.router.lifespan_context(app):
            pass


async def test_cancelled_startup_joins_probe_and_shuts_runtime_down(monkeypatch):
    app_module = importlib.import_module("cuda_db.server.app")
    started = threading.Event()
    release = threading.Event()
    shutdown_called = threading.Event()
    loop_errors = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()

    class BlockingProbeRuntime:
        def __init__(self, config):
            pass

        def stats(self):
            return {"engine": "stub"}

        def predict(self, array):
            started.set()
            release.wait()
            return 0, np.zeros(TEST_OUTPUT_ELEMS, dtype=np.float32)

        def shutdown(self):
            shutdown_called.set()

    monkeypatch.setattr(app_module, "InferenceRuntime", BlockingProbeRuntime)
    app = app_module.create_app(
        RuntimeSettings(input_elems=TEST_INPUT_ELEMS, output_elems=TEST_OUTPUT_ELEMS)
    )
    context = app.router.lifespan_context(app)
    loop.set_exception_handler(lambda _loop, detail: loop_errors.append(detail))
    startup = asyncio.create_task(context.__aenter__())
    try:
        assert await asyncio.to_thread(started.wait, 1.0)
        startup.cancel()
        await asyncio.sleep(0)
        assert not startup.done(), "startup returned before its native probe finished"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert shutdown_called.is_set()
        await asyncio.sleep(0)
        assert loop_errors == []
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)


def test_require_gpu_rejects_missing_model():
    with pytest.raises(ValueError, match="non-empty model_path"):
        RuntimeSettings(require_gpu=True)


async def test_require_gpu_rejects_stub_even_with_model_path(monkeypatch):
    app_module = importlib.import_module("cuda_db.server.app")
    shutdown_called = threading.Event()

    class StubRuntime:
        def __init__(self, config):
            pass

        def stats(self):
            return {"engine": "stub"}

        def shutdown(self):
            shutdown_called.set()

    monkeypatch.setattr(app_module, "InferenceRuntime", StubRuntime)
    app = app_module.create_app(
        RuntimeSettings(
            input_elems=TEST_INPUT_ELEMS,
            output_elems=TEST_OUTPUT_ELEMS,
            model_path="fake.onnx",
            require_gpu=True,
        )
    )

    with pytest.raises(RuntimeError, match="ONNX CUDA engine is not active"):
        async with app.router.lifespan_context(app):
            pass
    assert shutdown_called.is_set()


async def test_shutdown_drains_running_and_executor_queued_work(monkeypatch):
    from httpx import ASGITransport, AsyncClient

    app_module = importlib.import_module("cuda_db.server.app")
    running_started = threading.Event()
    release = threading.Event()
    completed = 0
    shutdown_after_completed = None
    lock = threading.Lock()

    class ControlledRuntime:
        def __init__(self, config):
            self.calls = 0

        def stats(self):
            return {"engine": "stub"}

        def predict(self, array):
            nonlocal completed
            with lock:
                self.calls += 1
                call = self.calls
            if call > 1:  # Call 1 is the startup probe.
                running_started.set()
                release.wait()
            with lock:
                completed += 1
            return call, np.zeros(TEST_OUTPUT_ELEMS, dtype=np.float32)

        def reset_stats(self):
            pass

        def shutdown(self):
            nonlocal shutdown_after_completed
            with lock:
                shutdown_after_completed = completed

    monkeypatch.setattr(app_module, "InferenceRuntime", ControlledRuntime)
    settings = RuntimeSettings(
        input_elems=TEST_INPUT_ELEMS,
        output_elems=TEST_OUTPUT_ELEMS,
        executor_workers=1,
        max_inflight_requests=2,
        request_timeout_ms=5_000,
    )
    app = app_module.create_app(settings)
    context = app.router.lifespan_context(app)
    await context.__aenter__()
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    shutdown = None
    exited = False
    try:
        first = asyncio.create_task(
            client.post("/predict", json={"input": [1.0] * TEST_INPUT_ELEMS})
        )
        assert await asyncio.to_thread(running_started.wait, 1.0)
        second = asyncio.create_task(
            client.post("/predict", json={"input": [2.0] * TEST_INPUT_ELEMS})
        )
        for _ in range(100):
            if app.state.admission.inflight == 2:
                break
            await asyncio.sleep(0.01)
        assert app.state.admission.inflight == 2

        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        await asyncio.sleep(0)
        assert not shutdown.done()
        assert (await client.get("/readyz")).status_code == 503
        rejected = await client.post(
            "/predict", json={"input": [3.0] * TEST_INPUT_ELEMS}
        )
        assert rejected.status_code == 503

        release.set()
        responses = await asyncio.gather(first, second)
        assert [response.status_code for response in responses] == [200, 200]
        await shutdown
        exited = True
        assert shutdown_after_completed == 3  # probe + both accepted requests
    finally:
        release.set()
        if shutdown is not None and not exited:
            await shutdown
            exited = True
        if not exited:
            await context.__aexit__(None, None, None)
        await client.aclose()


async def test_late_native_failure_marks_server_unready_without_loop_warning(client_for):
    settings = RuntimeSettings(
        input_elems=TEST_INPUT_ELEMS,
        output_elems=TEST_OUTPUT_ELEMS,
        max_inflight_requests=1,
        request_timeout_ms=50,
    )
    started = threading.Event()
    release = threading.Event()
    loop_errors = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()

    class FailingRuntime:
        def predict(self, array):
            started.set()
            release.wait()
            raise InferenceError("late engine failure")

    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    try:
        async with client_for(settings) as client:
            app = client._transport.app
            app.state.runtime = FailingRuntime()
            request_task = asyncio.create_task(
                client.post("/predict", json={"input": [1.0] * TEST_INPUT_ELEMS})
            )
            assert await asyncio.to_thread(started.wait, 1.0)
            response = await request_task
            assert response.status_code == 504

            release.set()
            for _ in range(100):
                if not app.state.ready:
                    break
                await asyncio.sleep(0.01)

            readiness = await client.get("/readyz")
            assert readiness.status_code == 503
            assert readiness.json() == {"detail": "server is not ready"}
            await asyncio.sleep(0)
            assert loop_errors == []
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)
