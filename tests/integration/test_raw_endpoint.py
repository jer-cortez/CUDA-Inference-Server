"""Tests for the binary /predict/raw endpoint.

The endpoint exists so the benchmark measures inference rather than JSON
parsing, which means its correctness matters twice: a wrong result here would
also silently invalidate every published number.
"""

from __future__ import annotations

import asyncio

import numpy as np

from .conftest import TEST_INPUT_ELEMS, TEST_OUTPUT_ELEMS

RAW_HEADERS = {"content-type": "application/octet-stream"}


def _payload(values: list[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


async def test_raw_returns_correctly_shaped_output(client):
    body = _payload([1.0, 2.0, 3.0, 4.0])

    response = await client.post("/predict/raw", content=body, headers=RAW_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["output"]) == TEST_OUTPUT_ELEMS
    assert payload["latency_ms"] >= 0


async def test_raw_and_json_agree(client):
    """The two endpoints must be interchangeable.

    If they diverged, the benchmark (raw) and the tests (JSON) would be
    exercising different behaviour, and neither would tell you about the other.
    """
    values = [1.5, -2.5, 3.25, 0.0]

    json_response = await client.post("/predict", json={"input": values})
    raw_response = await client.post(
        "/predict/raw", content=_payload(values), headers=RAW_HEADERS
    )

    assert json_response.status_code == 200
    assert raw_response.status_code == 200
    assert raw_response.json()["output"] == json_response.json()["output"]


async def test_raw_rejects_wrong_byte_count(client):
    # One float short: the count must be validated in bytes, since a truncated
    # buffer would otherwise be reinterpreted as a shorter tensor.
    short = _payload([1.0, 2.0, 3.0])

    response = await client.post("/predict/raw", content=short, headers=RAW_HEADERS)

    assert response.status_code == 400
    assert "bytes" in response.json()["detail"]


async def test_raw_rejects_non_multiple_of_four_bytes(client):
    response = await client.post(
        "/predict/raw", content=b"\x00\x01\x02", headers=RAW_HEADERS
    )

    assert response.status_code == 400


async def test_raw_preserves_per_request_values_under_concurrency(client):
    """Distinct concurrent payloads must come back distinct.

    The raw path hands the scheduler a read-only view over the request body
    rather than an owned copy, so this is the test that would catch that view
    being reused or outliving its request.
    """
    payloads = [[float(i)] * TEST_INPUT_ELEMS for i in range(8)]

    responses = await asyncio.gather(
        *(
            client.post("/predict/raw", content=_payload(values), headers=RAW_HEADERS)
            for values in payloads
        )
    )

    assert all(r.status_code == 200 for r in responses)

    outputs = [r.json()["output"] for r in responses]
    # The stub echoes input into output, so each response should lead with the
    # value that request sent.
    leading = sorted(output[0] for output in outputs)
    assert leading == [float(i) for i in range(8)]
