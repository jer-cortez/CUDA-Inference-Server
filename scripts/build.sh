#!/usr/bin/env bash
# Independent dev loops:
#   ./scripts/build.sh cpp     -- C++ library + gtest suite, no Python needed
#   ./scripts/build.sh python  -- editable install of the pybind11 extension
#                                  (stub engine; no CUDA or ONNX Runtime)
#   ./scripts/build.sh gpu-python
#                              -- editable install WITH CUDA + ONNX Runtime,
#                                  which is what the server needs to serve a
#                                  real model (requires ONNXRUNTIME_ROOT)
#   ./scripts/build.sh gpu     -- C++ library + gtest suite with CUDA enabled
#                                  (requires nvcc/CUDAToolkit; separate build
#                                  dir so the plain `cpp` loop above never
#                                  picks up CUDA flags)
#   ./scripts/build.sh all     -- cpp + python (default)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

build_cpp() {
    echo "==> Building C++ core + tests"
    cmake -S . -B build
    cmake --build build -j"$(getconf _NPROCESSORS_ONLN)"
    ctest --test-dir build --output-on-failure
}

build_gpu() {
    echo "==> Building C++ core + tests with CUDA enabled"
    cmake -S . -B build-gpu -DCUDA_DB_ENABLE_CUDA=ON
    cmake --build build-gpu -j"$(getconf _NPROCESSORS_ONLN)"
    ctest --test-dir build-gpu --output-on-failure
}

build_python() {
    echo "==> Installing the extension (editable, stub engine)"
    if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
    pip install -e ".[dev]"
}

build_gpu_python() {
    # The pip build is a SEPARATE CMake configuration from `build.sh gpu`:
    # pyproject.toml only defines CUDA_DB_BUILD_PYTHON, so without these the
    # extension is stub-only even on a box where the test binary has ONNX, and
    # the server fails at startup with "this build has no ONNX Runtime support".
    echo "==> Installing the extension (editable, CUDA + ONNX Runtime)"
    if [[ -z "${ONNXRUNTIME_ROOT:-}" ]]; then
        echo "ONNXRUNTIME_ROOT is not set; point it at an extracted" >&2
        echo "onnxruntime-linux-x64-gpu-<version> directory." >&2
        exit 1
    fi
    if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
    # Stale scikit-build-core config would otherwise keep the old defines.
    rm -rf build
    pip install -e ".[dev,bench]" \
        --config-settings=cmake.define.CUDA_DB_ENABLE_CUDA=ON \
        --config-settings=cmake.define.CUDA_DB_ENABLE_ONNX=ON \
        --config-settings=cmake.define.ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT}"
}

case "${1:-all}" in
    cpp) build_cpp ;;
    gpu) build_gpu ;;
    python) build_python ;;
    gpu-python) build_gpu_python ;;
    all) build_cpp && build_python ;;
    *) echo "usage: $0 [cpp|gpu|python|gpu-python|all]" >&2; exit 2 ;;
esac
