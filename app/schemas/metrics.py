from pydantic import BaseModel


class Metrics(BaseModel):
    avg: float
    p95: float
