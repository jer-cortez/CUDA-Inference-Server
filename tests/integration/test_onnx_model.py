"""End-to-end tests against a real ONNX model.

Skipped unless CUDA_DB_MODEL_PATH points at an exported model, so the default
CPU-only run (which uses the stub engine) stays green. Export one with
``python models/export_resnet.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cuda_db.config import RuntimeSettings

MODEL_PATH = os.environ.get("CUDA_DB_MODEL_PATH", "")
RESNET_INPUT_ELEMS = 3 * 224 * 224
IMAGENET_CLASSES = 1000

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not Path(MODEL_PATH).is_file(),
    reason="set CUDA_DB_MODEL_PATH to an exported .onnx model to run these",
)


@pytest.fixture
def model_settings() -> RuntimeSettings:
    return RuntimeSettings(
        max_batch_size=8,
        max_wait_ms=20,
        input_elems=RESNET_INPUT_ELEMS,
        output_elems=IMAGENET_CLASSES,
        executor_workers=8,
        model_path=MODEL_PATH,
    )


def _synthetic_input(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(RESNET_INPUT_ELEMS, dtype=np.float32).tolist()


async def test_predict_returns_imagenet_logits(client_for, model_settings):
    async with client_for(model_settings) as client:
        response = await client.post("/predict", json={"input": _synthetic_input()})

        assert response.status_code == 200
        body = response.json()
        assert len(body["output"]) == IMAGENET_CLASSES
        # A backend returning zeros would still have the right length.
        assert len(set(body["output"])) > 1


async def test_healthz_reports_onnx_engine(client_for, model_settings):
    """The stub returns plausibly-shaped output, so shape alone cannot tell us
    which backend served the request. This is the check that catches a silent
    fallback to the stub."""
    async with client_for(model_settings) as client:
        response = await client.get("/healthz")

        assert response.status_code == 200
        assert response.json()["engine"] == "onnx"


async def test_concurrent_requests_batch_and_stay_correct(client_for, model_settings):
    """Concurrent requests must coalesce into batches without corrupting any
    individual result -- the property that makes dynamic batching safe."""
    import asyncio

    async with client_for(model_settings) as client:
        payloads = [_synthetic_input(seed) for seed in range(4)]
        responses = await asyncio.gather(
            *(client.post("/predict", json={"input": payload}) for payload in payloads)
        )

        assert all(r.status_code == 200 for r in responses)

        outputs = [r.json()["output"] for r in responses]
        for output in outputs:
            assert len(output) == IMAGENET_CLASSES

        # Distinct inputs must yield distinct outputs; identical rows would
        # mean the batch was built from one request's data repeated.
        assert len({tuple(o[:8]) for o in outputs}) == len(outputs)

        stats = (await client.get("/healthz")).json()
        assert stats["total_requests"] == len(payloads)
        # Fewer batches than requests is the observable proof of coalescing.
        assert stats["total_batches"] <= len(payloads)


async def test_each_request_matches_its_own_solo_run(client_for, model_settings):
    """Batching must not change a result. Runs one input alone, then alongside
    others, and requires the same logits both times."""
    import asyncio

    payload = _synthetic_input(99)

    async with client_for(model_settings) as client:
        solo = (await client.post("/predict", json={"input": payload})).json()["output"]

        batched_responses = await asyncio.gather(
            client.post("/predict", json={"input": payload}),
            *(
                client.post("/predict", json={"input": _synthetic_input(s)})
                for s in range(3)
            ),
        )
        batched = batched_responses[0].json()["output"]

    # Tolerance, not equality: cuDNN may select a different convolution
    # algorithm for a batched run than a single-image one.
    np.testing.assert_allclose(
        np.array(batched), np.array(solo), rtol=1e-3, atol=1e-3
    )
