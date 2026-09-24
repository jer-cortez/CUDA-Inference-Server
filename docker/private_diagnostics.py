"""Read private diagnostics through Docker exec without publishing a port.

Example command file on a remote load generator:
["ssh", "gpu-host", "python3", "/srv/cuda-db/docker/private_diagnostics.py"]
The client appends /readyz or /healthz. SSH credentials stay in the SSH agent.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", type=Path, default=Path(__file__).resolve().parent / "docker-compose.deploy.yml")
    parser.add_argument("endpoint", choices=("/readyz", "/healthz"))
    args = parser.parse_args()
    probe = (
        "import sys,urllib.request; "
        "r=urllib.request.urlopen('http://127.0.0.1:8000'+sys.argv[1],timeout=5); "
        "sys.stdout.buffer.write(r.read())"
    )
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(args.compose_file.resolve()), "exec", "-T", "inference", "python", "-c", probe, args.endpoint],
            capture_output=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise SystemExit("private diagnostics command unavailable or timed out")
    if result.returncode:
        # Compose stderr can include substituted environment values. Keep the
        # remote client's error generic; inspect host logs separately.
        raise SystemExit("private diagnostics failed")
    sys.stdout.buffer.write(result.stdout)


if __name__ == "__main__":
    main()
