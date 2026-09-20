#!/bin/sh
set -eu
# Validate before importing/initializing the native runtime. This production
# image deliberately refuses the CPU stub, even if its environment is changed.
case "${CUDA_DB_REQUIRE_GPU:-true}" in
    true|1|yes|on) ;;
    *) echo 'CUDA_DB_REQUIRE_GPU must be true in the GPU image' >&2; exit 1 ;;
esac
python /opt/cuda-db/verify_model.py
# exec gives Uvicorn PID 1 and direct receipt of Docker's SIGTERM. One worker
# owns one model/scheduler. Native work can outlive HTTP timeout; Docker's
# stop grace period is the final bound for a hung CUDA call.
exec python -m uvicorn cuda_db.server.app:app --host 0.0.0.0 --port 8000 \
    --workers 1 --timeout-graceful-shutdown 45 --no-proxy-headers
