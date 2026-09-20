"""Compare HTTP inference to export-time PyTorch logits (requires NumPy).

python docker/smoke_test.py --models models --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

import numpy as np

from verify_model import sha256, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    manifest = verify(args.models / "resnet50.onnx", args.models / "resnet50.manifest.json")
    reference = manifest["reference"]
    if Path(reference["file"]).name != reference["file"]:
        raise ValueError("reference must be a filename in the model directory")
    reference_path = args.models / reference["file"]
    if sha256(reference_path) != reference["sha256"]:
        raise ValueError("reference checksum mismatch")
    with np.load(reference_path, allow_pickle=False) as tensors:
        inputs = tensors["input"].astype("<f4")
        expected = tensors["output"]
    assert inputs.shape == (8, 3, 224, 224), "export eight distinct reference inputs"
    assert expected.shape == (8, 1000)
    assert len({row.tobytes() for row in inputs}) == 8
    with urllib.request.urlopen(args.url + "/readyz", timeout=5) as response:
        assert response.status == 200
    with urllib.request.urlopen(args.url + "/healthz", timeout=5) as response:
        before = json.load(response)
    assert before["engine"] == "onnx", before

    def predict(index: int) -> dict:
        request = urllib.request.Request(args.url + "/predict/raw", data=inputs[index].tobytes(), headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        actual = np.asarray(result["output"])
        assert actual.shape == (1000,)
        assert np.isfinite(actual).all()
        assert np.argmax(actual) == np.argmax(expected[index])
        error = float(np.max(np.abs(actual - expected[index])))
        assert error <= 0.01, f"max logit error {error} > 0.01"
        return result

    predict(0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(predict, range(8)))
    assert len({result["request_id"] for result in results}) == 8
    with urllib.request.urlopen(args.url + "/healthz", timeout=5) as response:
        after = json.load(response)
    print("PASS: readiness, ONNX engine, eight distinct reference logits, and concurrent response mapping")
    print("Batching counters (coalescing depends on timing):", json.dumps({"before": before, "after": after}))


if __name__ == "__main__":
    main()
