import math
import threading
import time
from copy import deepcopy
from typing import Any

from fastapi import Request

_METRICS_LOCK = threading.Lock()

LATENCY_CACHE: dict[str, dict[str, Any]] = {
    "train": {
        "count": 0,
        "total_ms": 0.0,
        "avg_ms": 0.0,
        "p95_ms": 0.0,
        "latencies_ms": [],
    },
    "predict": {
        "count": 0,
        "total_ms": 0.0,
        "avg_ms": 0.0,
        "p95_ms": 0.0,
        "latencies_ms": [],
    },
}


def _target_from_path(path: str) -> str | None:
    """@brief Map an HTTP path to a latency cache bucket.

    @param path Request path (e.g., `/fit/series_a` or `/predict/series_a`).
    @return `train` for fit routes, `predict` for predict routes, otherwise `None`.
    """
    if path.startswith("/fit/"):
        return "train"
    if path.startswith("/predict/"):
        return "predict"
    return None


def _compute_p95(latencies_ms: list[float]) -> float:
    """@brief Compute P95 latency using the nearest-rank method.

    @param latencies_ms List of observed latencies in milliseconds.
    @return P95 latency in milliseconds. Returns `0.0` when the list is empty.
    """
    if not latencies_ms:
        return 0.0

    sorted_latencies = sorted(latencies_ms)
    rank = max(1, math.ceil(0.95 * len(sorted_latencies)))
    return sorted_latencies[rank - 1]


def _update_latency_cache(path: str, latency_ms: float) -> None:
    """@brief Update latency metrics for the route group derived from path.

    @param path Request path used to decide the target bucket.
    @param latency_ms Measured request latency in milliseconds.
    @return None.
    """
    target = _target_from_path(path)
    if target is None:
        return

    with _METRICS_LOCK:
        bucket = LATENCY_CACHE[target]
        bucket["latencies_ms"].append(latency_ms)
        bucket["count"] += 1
        bucket["total_ms"] += latency_ms
        bucket["avg_ms"] = bucket["total_ms"] / bucket["count"]
        bucket["p95_ms"] = _compute_p95(bucket["latencies_ms"])


async def track_request_latency(request: Request, call_next):
    """@brief FastAPI middleware that measures and stores request latency.

    @param request Incoming FastAPI request object.
    @param call_next FastAPI middleware callback used to continue request handling.
    @return Response produced by downstream handlers.

    @details
    Latency metrics are recorded only for successful (`2xx`) responses.
    Error responses and raised exceptions are intentionally excluded from
    cache aggregation.
    """
    start_time = time.perf_counter()
    response = await call_next(request)

    if 200 <= response.status_code < 300:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        _update_latency_cache(request.url.path, elapsed_ms)
        
    return response


def get_latency_cache() -> dict[str, dict[str, Any]]:
    """@brief Return a defensive copy of current latency metrics.

    @return Deep copy of the process-local latency cache.
    """
    with _METRICS_LOCK:
        return deepcopy(LATENCY_CACHE)


def reset_latency_cache() -> None:
    """@brief Clear all cached latency metrics.

    @return None.

    @details
    Primarily intended for tests and local diagnostics.
    """
    with _METRICS_LOCK:
        for bucket in LATENCY_CACHE.values():
            bucket["count"] = 0
            bucket["total_ms"] = 0.0
            bucket["avg_ms"] = 0.0
            bucket["p95_ms"] = 0.0
            bucket["latencies_ms"] = []
