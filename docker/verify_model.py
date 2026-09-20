"""Validate a mounted ResNet artifact without importing CUDA or ONNX."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("model manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema")
    if not manifest.get("model_version"):
        raise ValueError("model_version is required")
    if manifest.get("file") != model_path.name:
        raise ValueError("manifest filename does not match model path")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("sha256", "")):
        raise ValueError("manifest requires a lowercase SHA-256 digest")
    if manifest.get("input") != {"name": "input", "dtype": "float32", "shape": ["batch", 3, 224, 224]}:
        raise ValueError("manifest input must describe dynamic NCHW float32 ResNet input")
    if manifest.get("output") != {"name": "output", "dtype": "float32", "shape": ["batch", 1000]}:
        raise ValueError("manifest output must describe dynamic float32 ImageNet logits")
    if sha256(model_path) != manifest["sha256"]:
        raise ValueError("model SHA-256 does not match manifest")
    return manifest


if __name__ == "__main__":
    try:
        result = verify(
            Path(os.environ["CUDA_DB_MODEL_PATH"]),
            Path(os.environ["CUDA_DB_MODEL_MANIFEST"]),
        )
    except (KeyError, OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"model validation failed: {exc}") from exc
    print(f"verified model {result['model_version']} sha256={result['sha256']}")
