import math

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.latency import LatencyRecord
from app.database.series_version import SeriesVersionRecord
from app.schemas.healthcheck import HealthCheckResponse, Metrics


class HealthCheckService:
    def __init__(self, session: Session, latency_record: LatencyRecord | None = None) -> None:
        """@brief Initialize healthcheck service dependencies.

        @param session Active database session.
        @param latency_record Optional latency repository implementation.
        """
        self._session = session
        self._latency_record = latency_record or LatencyRecord()

    @staticmethod
    def _compute_p95(latencies: list[float]) -> float:
        """@brief Compute P95 latency using nearest-rank method.

        @param latencies Latency samples in milliseconds.
        @return P95 latency. Returns 0.0 when list is empty.
        """
        if not latencies:
            return 0.0

        sorted_latencies = sorted(latencies)
        rank = max(1, math.ceil(0.95 * len(sorted_latencies)))
        return float(sorted_latencies[rank - 1])

    @classmethod
    def _metrics_from_latencies(cls, latencies: list[float]) -> Metrics:
        """@brief Build metrics object from raw latency values.

        @param latencies Latency samples in milliseconds.
        @return Metrics object for API response.
        """
        if not latencies:
            return Metrics(avg=0.0, p95=0.0)

        average = float(sum(latencies) / len(latencies))
        return Metrics(
            avg=average,
            p95=cls._compute_p95(latencies),
        )

    def healthcheck(self) -> HealthCheckResponse:
        """@brief Build healthcheck response from Redis and DB counters.

        @return HealthCheckResponse with latency metrics and training counters.
        """
        try:
            train_latencies = self._latency_record.get_latencies("train")
            predict_latencies = self._latency_record.get_latencies("predict")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Telemetry backend unavailable for healthcheck.",
            ) from exc

        series_count = SeriesVersionRecord.count_series(self._session)
        
        return HealthCheckResponse(
            series_trained=series_count,
            inference_latency_ms=self._metrics_from_latencies(predict_latencies),
            training_latency_ms=self._metrics_from_latencies(train_latencies),
        )
