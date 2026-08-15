#!/usr/bin/env bash
# Independent dev loops:
#   ./scripts/build.sh cpp     -- C++ library + gtest suite, no Python needed
#   ./scripts/build.sh python  -- editable install of the pybind11 extension
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
    echo "==> Installing the extension (editable)"
    if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
    pip install -e ".[dev]"
}

case "${1:-all}" in
    cpp) build_cpp ;;
    gpu) build_gpu ;;
    python) build_python ;;
    all) build_cpp && build_python ;;
    *) echo "usage: $0 [cpp|gpu|python|all]" >&2; exit 2 ;;
esac
