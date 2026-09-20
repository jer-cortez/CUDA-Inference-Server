"""Check required native/ORT libraries during a GPU-less image build.

libcuda.so.1 is supplied by the host driver at launch and is the only missing
library allowed during a build without GPU access.
"""
from pathlib import Path
import subprocess

import cuda_db._native.cuda_db_native as native


paths = [
    Path(native.__file__),
    Path("/opt/onnxruntime/lib/libonnxruntime.so"),
    Path("/opt/onnxruntime/lib/libonnxruntime_providers_shared.so"),
    Path("/opt/onnxruntime/lib/libonnxruntime_providers_cuda.so"),
]
for path in paths:
    if not path.is_file():
        raise SystemExit(f"required runtime library missing: {path}")
    result = subprocess.run(["ldd", str(path)], capture_output=True, text=True, check=True)
    unresolved = [line.strip().split()[0] for line in result.stdout.splitlines() if "=> not found" in line]
    unexpected = [library for library in unresolved if library != "libcuda.so.1"]
    if unexpected:
        raise SystemExit(f"unresolved libraries for {path}: {unexpected}\n{result.stdout}")
    print(f"linkage verified: {path}; host-driver dependencies: {unresolved}")
