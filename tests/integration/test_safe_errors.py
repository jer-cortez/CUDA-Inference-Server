from __future__ import annotations

import io
import json
import logging
import uuid

import pytest

from cuda_db._native import InferenceError
from cuda_db.config import RuntimeSettings
from cuda_db.server.app import create_app
from cuda_db.server.observability import RequestLoggingMiddleware

from .conftest import TEST_INPUT_ELEMS


def test_application_logs_use_active_uvicorn_handler():
    """Mirror Uvicorn's default setup, which has no root logger handler."""
    stream = io.StringIO()
    uvicorn_logger = logging.getLogger("uvicorn")
    application_logger = logging.getLogger("cuda_db")
    old_uvicorn_handlers = list(uvicorn_logger.handlers)
    old_application_handlers = list(application_logger.handlers)
    old_propagate = application_logger.propagate
    try:
        uvicorn_logger.handlers = [logging.StreamHandler(stream)]
        application_logger.handlers = []
        application_logger.propagate = True
        create_app(RuntimeSettings())

        logging.getLogger("cuda_db.request").info('{"event":"logging_probe"}')

        assert stream.getvalue().strip() == '{"event":"logging_probe"}'
    finally:
        uvicorn_logger.handlers = old_uvicorn_handlers
        application_logger.handlers = old_application_handlers
        application_logger.propagate = old_propagate


async def test_error_after_response_start_propagates_only_sanitized_exception(caplog):
    secret = "started-response-secret-must-not-appear"
    sent = []

    async def failing_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError(secret)

    async def receive():  # pragma: no cover - the fake app does not read a body
        raise AssertionError

    async def send(message):
        sent.append(message)

    middleware = RequestLoggingMiddleware(failing_app)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/unknown",
        "client": ("test", 123),
    }
    caplog.set_level(logging.INFO, logger="cuda_db")

    with pytest.raises(RuntimeError, match="response interrupted") as raised:
        await middleware(scope, receive, send)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert secret not in str(raised.value)
    assert secret not in caplog.text
    assert sent[0]["headers"][0][0] == b"x-request-id"


async def test_validation_response_and_logs_do_not_echo_input(client, caplog):
    secret = "validation-secret-must-not-appear"
    caplog.set_level(logging.INFO, logger="cuda_db")

    response = await client.post("/predict", json={"input": [secret]})

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid request"}
    assert secret not in response.text
    assert secret not in caplog.text
    uuid.UUID(response.headers["x-request-id"])


async def test_native_failure_is_generic_and_secret_is_not_logged(client, caplog):
    secret = "native-secret-must-not-appear"

    class FailingRuntime:
        def predict(self, array):
            raise InferenceError(secret)

    app = client._transport.app
    app.state.runtime = FailingRuntime()
    caplog.set_level(logging.INFO, logger="cuda_db")

    response = await client.post(
        "/predict", json={"input": [1.0] * TEST_INPUT_ELEMS}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "inference failed"}
    assert secret not in response.text
    assert secret not in caplog.text
    request_id = response.headers["x-request-id"]
    uuid.UUID(request_id)
    native_records = [
        record
        for record in caplog.records
        if record.name in {"cuda_db.inference", "cuda_db.server"}
    ]
    assert native_records
    rendered_events = [json.loads(record.message) for record in native_records]
    assert any(
        event.get("failure_type") == "InferenceError"
        and event.get("request_id") == request_id
        for event in rendered_events
    )


async def test_unexpected_error_boundary_is_generic_and_path_is_normalized(
    client, caplog
):
    secret = "path-secret-must-not-appear"
    app = client._transport.app

    @app.get("/explode/{value}")
    async def explode(value: str):
        raise RuntimeError(value)

    caplog.set_level(logging.INFO, logger="cuda_db")
    response = await client.get(f"/explode/{secret}")

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert secret not in response.text
    assert secret not in caplog.text
    uuid.UUID(response.headers["x-request-id"])
    records = [r for r in caplog.records if r.name == "cuda_db.request"]
    assert records[-1].endpoint == "unknown"
