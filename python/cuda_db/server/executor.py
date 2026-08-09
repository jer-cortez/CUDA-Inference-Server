"""The thread pool that blocking ``predict()`` calls run on.

Kept separate from FastAPI's default executor: that one also serves sync route
handlers and other library work, and we want the pool feeding the scheduler to
be independently sized and named for profiling.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def make_executor(workers: int) -> ThreadPoolExecutor:
    if workers < 1:
        raise ValueError(f"executor_workers must be >= 1, got {workers}")
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cuda-db-predict")
