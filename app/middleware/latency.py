import logging
import math
import time
from typing import Any

from fastapi import Request

from app.database.latency import LatencyRecord

_LOGGER = logging.getLogger(__name__)


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


def _metrics_from(latencies_ms: list[float]) -> dict[str, Any]:
    """@brief Compute summary metrics from raw latency values.

    @param latencies_ms Raw latency values in milliseconds.
    @return Dictionary with count, total, avg, p95 and raw list.
    """
    if not latencies_ms:
        return {
            "count": 0,
            "total_ms": 0.0,
            "avg_ms": 0.0,
            "p95_ms": 0.0,
            "latencies_ms": [],
        }

    total = float(sum(latencies_ms))
    count = len(latencies_ms)
    return {
        "count": count,
        "total_ms": total,
        "avg_ms": total / count,
        "p95_ms": _compute_p95(latencies_ms),
        "latencies_ms": latencies_ms,
    }


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

    target = _target_from_path(request.url.path)
    if target is not None and 200 <= response.status_code < 300:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        try:
            LatencyRecord().push_latency(target, elapsed_ms)
        except Exception as exc:
            _LOGGER.warning("Failed to store latency in Redis: %s", exc)
        
    return response


def get_latency_cache() -> dict[str, dict[str, Any]]:
    """@brief Return computed latency metrics sourced from Redis.

    @return Metrics dictionary grouped by `train` and `predict`.
    """
    try:
        record = LatencyRecord()
        train_latencies = record.get_latencies("train")
        predict_latencies = record.get_latencies("predict")
    except Exception:
        train_latencies = []
        predict_latencies = []

    return {
        "train": _metrics_from(train_latencies),
        "predict": _metrics_from(predict_latencies),
    }


def reset_latency_cache() -> None:
    """@brief Clear Redis-backed latency metrics.

    @return None.
    """
    try:
        LatencyRecord().clear()
    except Exception:
        return
