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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off, got {raw!r}"
    )


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
    # Covers body upload, validation, executor queueing and native inference.
    request_timeout_ms: int = 30_000
    # Admission happens before the body is read, so this also bounds how many
    # large JSON bodies the process retains at once.
    max_inflight_requests: int = 16
    max_request_bytes: int = 4 * 1024 * 1024
    # Production guard: refuse inference unless a CUDA-backed ONNX runtime was
    # constructed and passed the startup probe. Local development keeps using
    # the stub by default.
    require_gpu: bool = False
    # Prediction endpoints can be protected with hashed bearer tokens.  The
    # deployment switch also forces authentication and removes interactive API
    # documentation; it intentionally does not imply require_gpu so production
    # security can be exercised on CPU-only hosts.
    require_auth: bool = False
    api_keys_file: str = ""
    deployment_mode: bool = False

    def __post_init__(self) -> None:
        positive = {
            "max_batch_size": self.max_batch_size,
            "input_elems": self.input_elems,
            "output_elems": self.output_elems,
            "executor_workers": self.executor_workers,
            "request_timeout_ms": self.request_timeout_ms,
            "max_inflight_requests": self.max_inflight_requests,
            "max_request_bytes": self.max_request_bytes,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.max_wait_ms < 0:
            raise ValueError(f"max_wait_ms must be >= 0, got {self.max_wait_ms}")
        if self.require_gpu and not self.model_path:
            raise ValueError("require_gpu=true requires a non-empty model_path")
        if (self.require_auth or self.deployment_mode) and not self.api_keys_file:
            raise ValueError(
                "authentication requires a non-empty api_keys_file"
            )

    @property
    def auth_required(self) -> bool:
        """Whether prediction routes must authenticate."""
        return self.require_auth or self.deployment_mode

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            max_batch_size=_env_int("CUDA_DB_MAX_BATCH_SIZE", 8),
            max_wait_ms=_env_int("CUDA_DB_MAX_WAIT_MS", 5),
            input_elems=_env_int("CUDA_DB_INPUT_ELEMS", DEFAULT_INPUT_ELEMS),
            output_elems=_env_int("CUDA_DB_OUTPUT_ELEMS", DEFAULT_OUTPUT_ELEMS),
            executor_workers=_env_int("CUDA_DB_EXECUTOR_WORKERS", 8),
            model_path=_env_str("CUDA_DB_MODEL_PATH", ""),
            request_timeout_ms=_env_int("CUDA_DB_REQUEST_TIMEOUT_MS", 30_000),
            max_inflight_requests=_env_int("CUDA_DB_MAX_INFLIGHT_REQUESTS", 16),
            max_request_bytes=_env_int("CUDA_DB_MAX_REQUEST_BYTES", 4 * 1024 * 1024),
            require_gpu=_env_bool("CUDA_DB_REQUIRE_GPU", False),
            require_auth=_env_bool("CUDA_DB_REQUIRE_AUTH", False),
            api_keys_file=_env_str("CUDA_DB_API_KEYS_FILE", ""),
            deployment_mode=_env_bool("CUDA_DB_DEPLOYMENT_MODE", False),
        )
