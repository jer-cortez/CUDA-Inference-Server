"""Validate deployed inference through public and private service surfaces."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import ssl
import stat
import subprocess
from typing import Any, Callable
import urllib.parse
import urllib.request

import numpy as np

from verify_model import sha256, verify

TOKEN_ENV = "CUDA_DB_BENCHMARK_TOKEN"
TOKEN_FILE_ENV = "CUDA_DB_BENCHMARK_TOKEN_FILE"
CA_FILE_ENV = "CUDA_DB_BENCHMARK_CA_FILE"
MAX_TOKEN_BYTES = 16 * 1024


def load_token(token_file: Path | None = None) -> str | None:
    """Load a bearer token without ever printing it."""
    env_token = os.environ.get(TOKEN_ENV)
    configured_file = token_file
    if configured_file is None and os.environ.get(TOKEN_FILE_ENV):
        configured_file = Path(os.environ[TOKEN_FILE_ENV])
    if env_token is not None and configured_file is not None:
        raise ValueError(f"set only one of {TOKEN_ENV} and {TOKEN_FILE_ENV}/--token-file")
    if configured_file is not None:
        file_stat = configured_file.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("token file must be a regular file")
        if os.name == "posix" and file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ValueError("token file must not be accessible by group or other users")
        if file_stat.st_size > MAX_TOKEN_BYTES:
            raise ValueError("token file is unexpectedly large")
        token = configured_file.read_text(encoding="utf-8").strip()
    else:
        token = env_token
    if token is not None and (not token or "\n" in token or "\r" in token):
        raise ValueError("benchmark token must be one non-empty line")
    return token


def tls_context(ca_file: Path | None = None) -> ssl.SSLContext:
    """Build a verifying TLS context, optionally trusting a private test CA."""
    return ssl.create_default_context(cafile=str(ca_file) if ca_file else None)


def request_json(
    url: str,
    *,
    context: ssl.SSLContext,
    token: str | None = None,
    data: bytes | None = None,
    timeout: float = 5,
) -> dict[str, Any]:
    if token is not None and urllib.parse.urlsplit(url).scheme.lower() != "https":
        raise ValueError("refusing to send a bearer token over a non-HTTPS URL")
    headers = {"Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = "application/octet-stream"
    request = urllib.request.Request(url, data=data, headers=headers)
    class NoRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(
        NoRedirects(), urllib.request.HTTPSHandler(context=context)
    )
    with opener.open(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"{url} returned a non-object JSON response")
    return value


def load_diagnostics_command(path: Path) -> list[str]:
    """Load a JSON argv prefix; the requested endpoint is appended at runtime."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError("diagnostics command file must contain a non-empty JSON array of strings")
    return value


def command_diagnostics(command: list[str], endpoint: str) -> dict[str, Any]:
    completed = subprocess.run(
        [*command, endpoint],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"diagnostics command failed for {endpoint} with exit code {completed.returncode}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"diagnostics command returned invalid JSON for {endpoint}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"diagnostics command returned non-object JSON for {endpoint}")
    return value


def validate_reference(models: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = verify(models / "resnet50.onnx", models / "resnet50.manifest.json")
    reference = manifest.get("reference")
    if not isinstance(reference, dict) or not isinstance(reference.get("file"), str):
        raise ValueError("manifest reference must contain a filename")
    if Path(reference["file"]).name != reference["file"]:
        raise ValueError("reference must be a filename in the model directory")
    if not isinstance(reference.get("sha256"), str):
        raise ValueError("manifest reference must contain a checksum")
    reference_path = models / reference["file"]
    if sha256(reference_path) != reference["sha256"]:
        raise ValueError("reference checksum mismatch")
    with np.load(reference_path, allow_pickle=False) as tensors:
        inputs = tensors["input"].astype("<f4")
        expected = tensors["output"]
    if inputs.shape != (8, 3, 224, 224):
        raise ValueError("reference must contain eight NCHW inputs")
    if expected.shape != (8, 1000):
        raise ValueError("reference must contain eight 1000-class outputs")
    if len({row.tobytes() for row in inputs}) != 8:
        raise ValueError("reference inputs must be distinct")
    if not np.isfinite(inputs).all() or not np.isfinite(expected).all():
        raise ValueError("reference tensors must contain only finite values")
    return manifest, inputs, expected


def run_smoke(
    models: Path,
    inference_url: str,
    diagnostics: Callable[[str], dict[str, Any]],
    *,
    token: str | None,
    context: ssl.SSLContext,
) -> dict[str, Any]:
    manifest, inputs, expected = validate_reference(models)
    ready = diagnostics("/readyz")
    if ready.get("status") != "ready":
        raise ValueError(f"unexpected readiness response: {ready!r}")
    before = diagnostics("/healthz")
    if before.get("engine") != "onnx":
        raise ValueError(f"ONNX engine is not active: {before!r}")
    reported_sha = before.get("model_sha256")
    if reported_sha != manifest["sha256"]:
        raise ValueError("serving model checksum differs from local manifest")
    process_id = before.get("process_id")
    if not isinstance(process_id, str) or not process_id:
        raise ValueError("diagnostics did not report a process_id")

    def predict(index: int) -> dict[str, Any]:
        result = request_json(
            inference_url.rstrip("/") + "/predict/raw",
            context=context,
            token=token,
            data=inputs[index].tobytes(),
            timeout=60,
        )
        if type(result.get("request_id")) is not int:
            raise ValueError("prediction response has no integer request_id")
        actual = np.asarray(result.get("output"))
        if actual.shape != (1000,):
            raise ValueError(f"prediction output has shape {actual.shape}, expected (1000,)")
        if not np.isfinite(actual).all():
            raise ValueError("prediction output contains non-finite values")
        if np.argmax(actual) != np.argmax(expected[index]):
            raise ValueError(f"reference {index} top-1 class does not match")
        error = float(np.max(np.abs(actual - expected[index])))
        if error > 0.01:
            raise ValueError(f"reference {index} max logit error {error} > 0.01")
        return result

    predict(0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(predict, range(8)))
    if len({result["request_id"] for result in results}) != 8:
        raise ValueError("concurrent predictions did not receive unique request IDs")
    after = diagnostics("/healthz")
    if after.get("process_id") != process_id:
        raise ValueError("server process changed during smoke validation")
    if after.get("model_sha256") != reported_sha:
        raise ValueError("serving model identity changed during smoke validation")
    return {
        "model_version": manifest["model_version"],
        "model_sha256": manifest["sha256"],
        "reference_sha256": manifest["reference"]["sha256"],
        "request_count": 9,
        "before": before,
        "after": after,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="public inference base URL")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--diagnostics-url", help="private diagnostics base URL; local default is --url")
    diagnostics.add_argument(
        "--diagnostics-command-file",
        type=Path,
        help="JSON argv prefix invoked with /readyz or /healthz appended",
    )
    parser.add_argument("--token-file", type=Path, help=f"protected token file; alternative to {TOKEN_ENV}")
    parser.add_argument("--ca-file", type=Path, help=f"private CA bundle (or {CA_FILE_ENV})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ca_file = args.ca_file or (Path(os.environ[CA_FILE_ENV]) if os.environ.get(CA_FILE_ENV) else None)
    context = tls_context(ca_file)
    if args.diagnostics_command_file:
        command = load_diagnostics_command(args.diagnostics_command_file)
        diagnostics = lambda path: command_diagnostics(command, path)
    else:
        diagnostics_url = (args.diagnostics_url or args.url).rstrip("/")
        diagnostics = lambda path: request_json(diagnostics_url + path, context=context)
    result = run_smoke(
        args.models,
        args.url,
        diagnostics,
        token=load_token(args.token_file),
        context=context,
    )
    print("PASS: ONNX readiness and nine reference inferences")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
