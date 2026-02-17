import os
from typing import Any


def get_latency_history_limit() -> int:
    """@brief Return max number of latency samples retained in Redis lists.

    @return Integer history limit from `LATENCY_HISTORY_LIMIT` (default `500`).
    """
    return int(os.getenv("LATENCY_HISTORY_LIMIT", "500"))


def get_redis_url() -> str:
    """@brief Return Redis connection URL used by application components.

    @return Redis URL from `REDIS_URL` (default `redis://redis:6379/0`).
    """
    return os.getenv("REDIS_URL", "redis://redis:6379/0")


def load_min_training_data_points() -> dict[str, Any]:
    """@brief Load training minimum-data configuration from environment.

    @return Dictionary containing minimum training data points parameter.
    """
    return {
        "min_training_data_points": os.getenv("MIN_TRAINING_DATA_POINTS", "3"),
    }
