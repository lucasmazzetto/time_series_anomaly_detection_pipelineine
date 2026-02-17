from math import isfinite
from typing import Literal

from redis import Redis

from app.utils.env import get_latency_history_limit, get_redis_url

LatencyTarget = Literal["train", "predict"]


class LatencyRecord:

    _KEYS: dict[LatencyTarget, str] = {
        "train": "train_latencies",
        "predict": "predict_latencies",
    }

    def __init__(
        self,
        redis_client: Redis | None = None,
        redis_url: str | None = None,
        history_limit: int | None = None,
    ) -> None:
        """@brief Initialize Redis connectivity and latency retention settings.

        @description Creates or receives a Redis client and configures the
        number of latency values retained per target list.

        @param redis_client Optional pre-configured Redis client.
        @param redis_url Optional Redis URL. Falls back to `get_redis_url()`.
        @param history_limit Optional max history size per list.
        Falls back to `get_latency_history_limit()`.
        @return None.
        @throws ValueError If `history_limit` is less than 1.
        """
        url = redis_url or get_redis_url()

        if history_limit is None:
            history_limit = get_latency_history_limit()

        if history_limit < 1:
            raise ValueError("LATENCY_HISTORY_LIMIT must be greater than or equal to 1.")

        self._redis = redis_client or Redis.from_url(url, decode_responses=True)
        self._history_limit = history_limit

    def push_latency(self, target: LatencyTarget, latency_ms: float) -> None:
        """@brief Append a latency value and trim the list to max size.

        @description Stores a latency sample into the Redis list mapped by
        `target`, then trims the list to keep only the newest N entries.

        @param target Latency bucket (`train` or `predict`).
        @param latency_ms Request latency in milliseconds.
        @return None.
        @throws ValueError If target is invalid or latency is not finite.
        """
        key = self._key_for(target)
        value = float(latency_ms)

        if not isfinite(value):
            raise ValueError("latency_ms must be a finite number.")

        pipeline = self._redis.pipeline()
        pipeline.rpush(key, value)
        pipeline.ltrim(key, -self._history_limit, -1)
        pipeline.execute()

    def get_latencies(self, target: LatencyTarget) -> list[float]:
        """@brief Read and sanitize all cached latencies for a target bucket.

        @description Fetches list values from Redis and converts them to
        floats, skipping invalid or non-finite entries.

        @param target Latency bucket (`train` or `predict`).
        @return List of finite latency values in milliseconds.
        @throws ValueError If target is invalid.
        """
        key = self._key_for(target)
        values = self._redis.lrange(key, 0, -1)

        latencies: list[float] = []
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue

            if isfinite(numeric):
                latencies.append(numeric)

        return latencies

    def clear(self) -> None:
        """@brief Remove all latency lists managed by this repository.

        @return None.
        """
        self._redis.delete(*self._KEYS.values())

    @classmethod
    def _key_for(cls, target: LatencyTarget) -> str:
        """@brief Resolve Redis key name for a latency bucket.

        @param target Latency bucket (`train` or `predict`).
        @return Redis key associated with the target list.
        @throws ValueError If target is not supported.
        """
        try:
            return cls._KEYS[target]
        except KeyError as exc:
            raise ValueError("target must be 'train' or 'predict'.") from exc
