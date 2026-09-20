"""CPU-only checks for the container artifact gate; no CUDA/ONNX required."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("container_verify", Path(__file__).parents[1] / "docker" / "verify_model.py")
verify_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_module)


@pytest.fixture
def artifact(tmp_path):
    model = tmp_path / "resnet50.onnx"
    model.write_bytes(b"test artifact")
    manifest = {
        "schema_version": 1,
        "model_version": "test-v1",
        "file": model.name,
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "input": {"name": "input", "dtype": "float32", "shape": ["batch", 3, 224, 224]},
        "output": {"name": "output", "dtype": "float32", "shape": ["batch", 1000]},
    }
    manifest_path = tmp_path / "resnet50.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return model, manifest_path, manifest


def test_accepts_matching_artifact(artifact):
    model, path, manifest = artifact
    assert verify_module.verify(model, path) == manifest


def test_rejects_modified_model(artifact):
    model, path, _ = artifact
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_module.verify(model, path)


@pytest.mark.parametrize("field,value", [("sha256", "bad"), ("file", "other.onnx"), ("schema_version", 2), ("model_version", ""), ("input", {"shape": [1, 3, 224, 224]}), ("output", {})])
def test_rejects_invalid_manifest(artifact, field, value):
    model, path, manifest = artifact
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_module.verify(model, path)


def test_rejects_missing_model(artifact):
    model, path, _ = artifact
    model.unlink()
    with pytest.raises(FileNotFoundError):
        verify_module.verify(model, path)
