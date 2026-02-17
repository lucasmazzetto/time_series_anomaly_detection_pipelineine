from sqlalchemy.orm import Session

from app.database.series_version_record import SeriesVersionRecord
from app.middleware.latency import get_latency_cache
from app.schemas.healthcheck import HealthCheckResponse, Metrics


class HealthCheckService:
    def __init__(self, session: Session) -> None:
        """@brief Initialize healthcheck service dependencies.

        @param session Active database session.
        """
        self._session = session

    @staticmethod
    def _metrics_from(bucket: dict[str, object]) -> Metrics:
        """@brief Convert cached latency bucket into response metrics.

        @param bucket Latency cache bucket with `avg_ms` and `p95_ms`.
        @return Metrics object for API response.
        """
        return Metrics(
            avg=float(bucket["avg_ms"]),
            p95=float(bucket["p95_ms"]),
        )

    def healthcheck(self) -> HealthCheckResponse:
        """@brief Build healthcheck response from cache and DB counters.

        @return HealthCheckResponse with latency metrics and training counters.
        """
        cache = get_latency_cache()
        train_metrics = cache["train"]
        predict_metrics = cache["predict"]
        series_count = SeriesVersionRecord.count_series(self._session)
        
        return HealthCheckResponse(
            series_trained=series_count,
            inference_latency_ms=self._metrics_from(predict_metrics),
            training_latency_ms=self._metrics_from(train_metrics),
        )
