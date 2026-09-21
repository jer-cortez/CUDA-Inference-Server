#!/usr/bin/env python3
"""Exercise real Caddy TLS with a CPU stub; never contacts a public CA.

Run from an installed checkout with Docker Compose available. This creates an
isolated Compose project, trusts its CA only for this process, and removes its
containers, network, volumes, and temporary credentials on exit.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import ssl
import struct
import subprocess
import tempfile
import time
import uuid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    project = "cuda-db-security-" + uuid.uuid4().hex[:10]
    prefix = ["docker", "compose", "-p", project, "-f",
              str(root / "docker/docker-compose.https-test.yml")]
    with tempfile.TemporaryDirectory(prefix="cuda-db-https-") as temporary:
        work = Path(temporary)
        token = "smoke." + secrets.token_urlsafe(32)
        store = work / "api_keys.json"
        store.write_text(json.dumps({"schema_version": 1, "keys": [{
            "id": "smoke", "sha256": hashlib.sha256(token.encode()).hexdigest()
        }]}))
        # Digest only, in a private host directory; readable by container UID.
        store.chmod(0o444)
        environment = {**os.environ, "CUDA_DB_TEST_API_KEYS_FILE": str(store),
                       "CUDA_DB_TEST_HTTPS_PORT": str(args.port)}

        def compose(*arguments: str, timeout: int = 120) -> str:
            result = subprocess.run(prefix + list(arguments), env=environment,
                                    cwd=root, capture_output=True, text=True,
                                    timeout=timeout)
            if result.returncode:
                # Never include container logs or test credentials in errors.
                raise RuntimeError(f"Compose {arguments[0]} failed: " + result.stderr[-3000:])
            return result.stdout

        context = None

        def request(method: str, path: str, body: bytes | None = None,
                    headers: list[tuple[str, str]] | None = None):
            connection = http.client.HTTPSConnection("localhost", args.port,
                                                     context=context, timeout=8)
            try:
                connection.putrequest(method, path)
                for name, value in headers or []:
                    connection.putheader(name, value)
                if body is not None:
                    connection.putheader("Content-Length", str(len(body)))
                connection.endheaders(body)
                response = connection.getresponse()
                return response.status, {name.lower(): value for name, value in
                                         response.getheaders()}, response.read()
            finally:
                connection.close()

        def check(status: int, response, label: str) -> None:
            if response[0] != status:
                raise AssertionError(f"{label}: expected {status}, got {response[0]}")

        authorization = [("Authorization", "Bearer " + token)]
        json_headers = authorization + [("Content-Type", "application/json")]
        payload = json.dumps({"input": [1.0, 2.0, 3.0, 4.0]}).encode()
        try:
            if not args.skip_build:
                print("Building CPU-only security test image...", flush=True)
                compose("build", "inference", timeout=1200)
            compose("up", "-d", "--wait", "--wait-timeout", "120", timeout=180)
            for attempt in range(30):
                try:
                    compose("cp", "caddy:/data/caddy/pki/authorities/local/root.crt",
                            str(work / "root.crt"))
                    break
                except RuntimeError:
                    if attempt == 29:
                        raise
                    time.sleep(1)
            context = ssl.create_default_context(cafile=str(work / "root.crt"))
            for attempt in range(30):
                try:
                    check(401, request("POST", "/predict", payload), "TLS/auth readiness")
                    break
                except (OSError, AssertionError):
                    if attempt == 29:
                        raise
                    time.sleep(1)

            response = request("POST", "/predict", payload, json_headers)
            check(200, response, "authenticated JSON inference")
            uuid.UUID(response[1]["x-request-id"])
            check(200, request("POST", "/predict/raw", struct.pack("<4f", 1, 2, 3, 4),
                               authorization + [("Content-Type", "application/octet-stream")]),
                  "authenticated binary inference")
            check(401, request("POST", "/predict", payload,
                               [("Authorization", "Bearer invalid.secret")]), "invalid key")
            check(401, request("POST", "/predict", payload, authorization * 2),
                  "duplicate authorization")
            check(431, request("POST", "/predict", headers=[("X-Large", "x" * 40000)]),
                  "oversized headers")
            check(401, request("POST", "/predict?api_key=" + token, payload), "query key")
            # Send headers but no advertised body: auth must not wait for it.
            check(401, request("POST", "/predict", headers=[("Content-Length", "4000000")]),
                  "authentication without reading body")
            for path in ("/healthz", "/readyz", "/docs", "/redoc", "/openapi.json", "/unknown"):
                check(404, request("GET", path), "private route " + path)
                check(404, request("GET", path, headers=authorization),
                      "private route with valid credentials " + path)
            check(404, request("GET", "/predict"), "unsupported method")
            check(413, request("POST", "/predict/raw", b"0" * (4194304 + 1), authorization),
                  "oversized body")
            marker = "sensitive-marker-" + uuid.uuid4().hex
            check(422, request("POST", "/predict?secret=" + marker,
                               json.dumps({"input": [marker]}).encode(), json_headers),
                  "validation redaction")
            spoofed = request("POST", "/predict", payload, json_headers + [
                ("X-Forwarded-For", "203.0.113.99"), ("X-Request-ID", marker)])
            check(200, spoofed, "spoofed headers")
            assert marker.encode() not in spoofed[2]
            assert marker not in str(spoofed[1])

            app_id = compose("ps", "-q", "inference").strip()
            inspected = json.loads(subprocess.check_output(
                ["docker", "inspect", app_id], text=True))[0]
            assert not any(inspected["NetworkSettings"]["Ports"].values()), "App port published"
            original_ca = (work / "root.crt").read_bytes()
            compose("restart", "caddy")
            for attempt in range(30):
                try:
                    check(200, request("POST", "/predict", payload, json_headers), "proxy restart")
                    break
                except (OSError, AssertionError):
                    if attempt == 29:
                        raise
                    time.sleep(1)
            compose("cp", "caddy:/data/caddy/pki/authorities/local/root.crt", str(work / "after.crt"))
            assert original_ca == (work / "after.crt").read_bytes(), "CA changed on restart"
            compose("stop", "inference")
            error = request("POST", "/predict?secret=" + marker, payload, json_headers)
            check(502, error, "unavailable upstream")
            assert marker.encode() not in error[2]
            logs = compose("logs", "--no-color")
            for secret in (token, marker, "203.0.113.99"):
                assert secret not in logs, "Sensitive/spoofed value appeared in logs"
            assert '"key_id":"smoke"' in logs, "Structured audit logs missing"
            print("PASS: trusted TLS, JSON/raw inference, auth-before-body, route isolation, "
                  "body limit, header/log redaction, private app port, persisted CA, safe 502.")
        finally:
            # Scoped to the random project created by this invocation.
            compose("down", "--volumes", "--remove-orphans")


if __name__ == "__main__":
    main()
