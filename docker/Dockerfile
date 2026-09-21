# syntax=docker/dockerfile:1
# Linux AMD64 manifests resolved from NVIDIA's Docker registry.
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:59e0e4376a0f16d10b03d3a14344b80a866a1674cb4948cb318291387ac05010
FROM ${CUDA_DEVEL_IMAGE} AS builder
ARG TARGETARCH
ARG ORT_VERSION=1.23.2
# Official GitHub release asset digest, not a locally trusted first download.
ARG ORT_SHA256=2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53
ENV DEBIAN_FRONTEND=noninteractive ONNXRUNTIME_ROOT=/opt/onnxruntime
RUN test "$TARGETARCH" = amd64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3-dev python3-venv build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN test -n "$ORT_SHA256" \
    && curl --fail --location --retry 3 "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" -o /tmp/ort.tgz \
    && echo "$ORT_SHA256  /tmp/ort.tgz" | sha256sum --check --strict \
    && mkdir /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz --strip-components=1 -C /opt/onnxruntime \
    && rm /tmp/ort.tgz
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY docker/requirements-*.txt /tmp/
RUN python -m pip install --no-cache-dir pip==25.1.1 \
    && python -m pip install --no-cache-dir -r /tmp/requirements-build.txt \
    && python -m pip wheel --wheel-dir /wheels -r /tmp/requirements-runtime.txt
WORKDIR /src
COPY CMakeLists.txt pyproject.toml README.md ./
COPY cmake/ cmake/
COPY cpp/ cpp/
COPY python/ python/
RUN python -m pip wheel --verbose --no-build-isolation --no-deps --wheel-dir /wheels . \
    --config-settings=cmake.define.CUDA_DB_ENABLE_CUDA=ON \
    --config-settings=cmake.define.CUDA_DB_ENABLE_ONNX=ON \
    --config-settings=cmake.define.ONNXRUNTIME_ROOT=/opt/onnxruntime \
    '--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=75;89'

FROM ${CUDA_RUNTIME_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && groupadd --gid 10001 inference \
    && useradd --uid 10001 --gid inference --no-create-home --home-dir /tmp inference
COPY --from=builder /opt/onnxruntime/lib/ /opt/onnxruntime/lib/
COPY --from=builder /opt/onnxruntime/LICENSE /opt/onnxruntime/LICENSE
ENV PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CUDA_CACHE_PATH=/tmp/cuda-cache XDG_CACHE_HOME=/tmp/cache \
    CUDA_DB_REQUIRE_GPU=true CUDA_DB_MODEL_PATH=/models/resnet50.onnx \
    CUDA_DB_MODEL_MANIFEST=/models/resnet50.manifest.json \
    CUDA_DB_MAX_INFLIGHT_REQUESTS=16 CUDA_DB_MAX_REQUEST_BYTES=4194304 \
    CUDA_DB_REQUEST_TIMEOUT_MS=30000 NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    python -m pip install --no-cache-dir --no-index --find-links=/wheels cuda-db==0.1.0 \
    && python -m pip check \
    && python -c 'import cuda_db._native'
COPY docker/entrypoint.sh docker/verify_model.py docker/healthcheck.py docker/check_linkage.py docker/requirements-runtime.txt docker/uvicorn-logging.json /opt/cuda-db/
RUN python /opt/cuda-db/check_linkage.py && chmod 755 /opt/cuda-db/entrypoint.sh
WORKDIR /opt/cuda-db
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 CMD ["python", "/opt/cuda-db/healthcheck.py"]
ENTRYPOINT ["/opt/cuda-db/entrypoint.sh"]
