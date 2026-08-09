"""Proves requests actually coalesce instead of being served one at a time.

The shape assertions in test_predict_endpoint pass just as well with a
degenerate batch-size-1 queue, so this is the test that distinguishes "dynamic
batcher" from "serialized queue".
"""

from __future__ import annotations

import asyncio

from cuda_db.config import RuntimeSettings

from .conftest import TEST_INPUT_ELEMS, TEST_OUTPUT_ELEMS

CONCURRENCY = 16


async def test_concurrent_requests_form_multi_request_batches(client_for):
    settings = RuntimeSettings(
        max_batch_size=8,
        # Generous relative to the stub engine's near-zero execution time, so
        # the scheduler has a real window in which to accumulate arrivals.
        max_wait_ms=50,
        input_elems=TEST_INPUT_ELEMS,
        output_elems=TEST_OUTPUT_ELEMS,
        executor_workers=8,
    )

    async with client_for(settings) as client:
        responses = await asyncio.gather(
            *(
                client.post("/predict", json={"input": [float(i)] * TEST_INPUT_ELEMS})
                for i in range(CONCURRENCY)
            )
        )
        stats = (await client.get("/healthz")).json()

    assert all(response.status_code == 200 for response in responses)
    assert stats["total_requests"] == CONCURRENCY
    assert stats["max_batch_size_seen"] > 1, "requests never coalesced"
    assert stats["total_batches"] < CONCURRENCY, "one batch per request means no batching"


async def test_each_concurrent_request_gets_its_own_row(client_for):
    settings = RuntimeSettings(
        max_batch_size=8,
        max_wait_ms=50,
        input_elems=TEST_INPUT_ELEMS,
        output_elems=TEST_OUTPUT_ELEMS,
        executor_workers=8,
    )
    padding = [0.0] * (TEST_OUTPUT_ELEMS - TEST_INPUT_ELEMS)

    async with client_for(settings) as client:
        responses = await asyncio.gather(
            *(
                client.post("/predict", json={"input": [float(i)] * TEST_INPUT_ELEMS})
                for i in range(CONCURRENCY)
            )
        )

    # Each caller must get back its own input, not a batch-mate's -- this is
    # what catches a row/promise mix-up once several requests share a batch.
    for i, response in enumerate(responses):
        assert response.json()["output"] == [float(i)] * TEST_INPUT_ELEMS + padding


async def test_batch_size_one_config_still_serves_correctly(client_for):
    """The benchmark harness's baseline mode reuses this same code path."""
    settings = RuntimeSettings(
        max_batch_size=1,
        max_wait_ms=0,
        input_elems=TEST_INPUT_ELEMS,
        output_elems=TEST_OUTPUT_ELEMS,
        executor_workers=4,
    )

    async with client_for(settings) as client:
        responses = await asyncio.gather(
            *(
                client.post("/predict", json={"input": [1.0] * TEST_INPUT_ELEMS})
                for _ in range(8)
            )
        )
        stats = (await client.get("/healthz")).json()

    assert all(response.status_code == 200 for response in responses)
    assert stats["max_batch_size_seen"] == 1
    assert stats["total_batches"] == 8
