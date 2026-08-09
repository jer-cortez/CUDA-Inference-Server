"""The milestone's round-trip check: an array in, a correctly shaped array out.

Passing means Python -> pybind11 -> C++ scheduler thread -> promise -> Python
all works, before any CUDA or ONNX Runtime code exists.
"""

from __future__ import annotations

import random

from .conftest import TEST_INPUT_ELEMS, TEST_OUTPUT_ELEMS


async def test_predict_returns_correctly_shaped_output(client):
    payload = [random.random() for _ in range(TEST_INPUT_ELEMS)]

    response = await client.post("/predict", json={"input": payload})

    assert response.status_code == 200
    body = response.json()
    assert len(body["output"]) == TEST_OUTPUT_ELEMS
    assert body["latency_ms"] > 0
    assert isinstance(body["request_id"], int)


async def test_predict_echoes_input_through_the_stub_engine(client):
    payload = [1.0, 2.0, 3.0, 4.0]

    body = (await client.post("/predict", json={"input": payload})).json()

    # The stub engine copies the input row and zero-pads to output_elems, so a
    # mismatch here means rows got crossed somewhere in the batch packing.
    assert body["output"] == payload + [0.0] * (TEST_OUTPUT_ELEMS - TEST_INPUT_ELEMS)


async def test_predict_rejects_wrong_length_input(client):
    response = await client.post("/predict", json={"input": [1.0, 2.0]})

    assert response.status_code == 400
    assert str(TEST_INPUT_ELEMS) in response.json()["detail"]


async def test_request_ids_are_unique(client):
    payload = [0.5] * TEST_INPUT_ELEMS

    ids = [
        (await client.post("/predict", json={"input": payload})).json()["request_id"]
        for _ in range(5)
    ]

    assert len(set(ids)) == len(ids)


async def test_healthz_reports_scheduler_stats(client):
    body = (await client.get("/healthz")).json()

    assert body["status"] == "ok"
    assert body["total_batches"] == 0
    assert body["total_requests"] == 0
