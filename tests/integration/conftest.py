"""Shared fixtures for the end-to-end tests.

Requests go through httpx's ASGI transport rather than a real socket, so the
tests exercise the full FastAPI -> executor -> pybind11 -> C++ scheduler path
without binding a port. The lifespan context is entered explicitly because
ASGITransport does not fire startup/shutdown events on its own -- and without
it there would be no InferenceRuntime on ``app.state``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from cuda_db.config import RuntimeSettings
from cuda_db.server.app import create_app

# Small tensors keep the JSON payloads readable; shape correctness is what
# these tests check, not model-sized throughput.
TEST_INPUT_ELEMS = 4
TEST_OUTPUT_ELEMS = 6


@asynccontextmanager
async def _client_for(settings: RuntimeSettings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


@pytest.fixture
def settings() -> RuntimeSettings:
    return RuntimeSettings(
        max_batch_size=8,
        max_wait_ms=20,
        input_elems=TEST_INPUT_ELEMS,
        output_elems=TEST_OUTPUT_ELEMS,
        executor_workers=8,
    )


@pytest.fixture
def client_for():
    """Factory so a test can spin up a server with its own batching config."""
    return _client_for


@pytest.fixture
async def client(settings: RuntimeSettings):
    async with _client_for(settings) as client:
        yield client
