"""Runtime settings, read from the environment.

Shared by the server and (later) the benchmark harness, so a batch-size-1
comparison run is just a different set of env vars against the same code path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Matches the eventual ResNet/MobileNet input (3x224x224) and ImageNet class
# count, so shapes don't change when the stub engine is swapped for ONNX
# Runtime in a later milestone.
DEFAULT_INPUT_ELEMS = 3 * 224 * 224
DEFAULT_OUTPUT_ELEMS = 1000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw


@dataclass(frozen=True)
class RuntimeSettings:
    max_batch_size: int = 8
    max_wait_ms: int = 5
    input_elems: int = DEFAULT_INPUT_ELEMS
    output_elems: int = DEFAULT_OUTPUT_ELEMS
    # Bounded on purpose: predict() releases the GIL, but every call still
    # funnels into one scheduler queue and one GPU, so more threads past this
    # buy queueing delay rather than throughput.
    executor_workers: int = 8
    # Path to the exported ONNX model. Empty selects the stub engine, which is
    # what keeps the server runnable (and the integration tests meaningful) on
    # a machine with no GPU or no exported model.
    model_path: str = ""

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            max_batch_size=_env_int("CUDA_DB_MAX_BATCH_SIZE", 8),
            max_wait_ms=_env_int("CUDA_DB_MAX_WAIT_MS", 5),
            input_elems=_env_int("CUDA_DB_INPUT_ELEMS", DEFAULT_INPUT_ELEMS),
            output_elems=_env_int("CUDA_DB_OUTPUT_ELEMS", DEFAULT_OUTPUT_ELEMS),
            executor_workers=_env_int("CUDA_DB_EXECUTOR_WORKERS", 8),
            model_path=_env_str("CUDA_DB_MODEL_PATH", ""),
        )
