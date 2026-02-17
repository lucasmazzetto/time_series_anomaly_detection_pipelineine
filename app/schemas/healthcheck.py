from pydantic import BaseModel


class Metrics(BaseModel):
    avg: float
    p95: float


class HealthCheckResponse(BaseModel):
    series_trained: int
    inference_latency_ms: Metrics
    training_latency_ms: Metrics
