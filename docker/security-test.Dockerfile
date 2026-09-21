# Test-only CPU stub: never use this image for real model inference.
FROM python:3.11-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ git \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-build.txt /tmp/requirements-build.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-build.txt
WORKDIR /src
COPY CMakeLists.txt pyproject.toml README.md ./
COPY cmake/ cmake/
COPY cpp/ cpp/
COPY python/ python/
RUN python -m pip wheel --no-build-isolation --no-deps --wheel-dir /wheels . \
    --config-settings=cmake.define.CUDA_DB_ENABLE_CUDA=OFF \
    --config-settings=cmake.define.CUDA_DB_ENABLE_ONNX=OFF

FROM python:3.11-slim-bookworm
COPY docker/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt
COPY --from=builder /wheels/ /wheels/
RUN python -m pip install --no-cache-dir --no-deps /wheels/*.whl \
    && groupadd --gid 10001 inference \
    && useradd --uid 10001 --gid 10001 --no-create-home inference
COPY docker/healthcheck.py docker/uvicorn-logging.json /opt/cuda-db/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 10001:10001
HEALTHCHECK --interval=2s --timeout=2s --start-period=10s --retries=10 CMD ["python", "/opt/cuda-db/healthcheck.py"]
CMD ["python", "-m", "uvicorn", "cuda_db.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers", "--no-access-log", "--log-config", "/opt/cuda-db/uvicorn-logging.json"]
