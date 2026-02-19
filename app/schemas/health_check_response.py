from pydantic import BaseModel

from app.schemas.metrics import Metrics


class HealthCheckResponse(BaseModel):
    series_trained: int
    inference_latency_ms: Metrics
    training_latency_ms: Metrics
